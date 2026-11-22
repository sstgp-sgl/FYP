"""Application service: scene editing, view confirmation invalidation,
storyboard persistence and generation runs. HTTP stays in routes.py.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from .importers import import_from_reconstruction, import_from_upload, import_phase1
from .models import (
    GenerationRequest,
    Storyboard,
    StoryboardValidation,
    TourRun,
    TourScene,
    now,
)
from .planner import plan_storyboard
from .providers import ChatCompletionsProvider
from .repository import (
    ConflictError,
    InvalidDataError,
    NotFoundError,
    TourRepository,
)
from .validation import validate_annotations, validate_storyboard


class TourService:
    def __init__(self, repository: TourRepository, config, reconstruction_output_dir=None):
        self.repository = repository
        self.config = config
        self.reconstruction_output_dir = reconstruction_output_dir
        self._active: dict[str, str] = {}          # scene_id -> run_id
        self._tasks: dict[str, asyncio.Task] = {}  # run_id -> task
        self._guard = asyncio.Lock()
        self._shutting_down = False

    # ------------------------------------------------------------- scenes
    def create_scene_upload(self, data: bytes, title: str) -> TourScene:
        return import_from_upload(self.repository, data, title, self.config.data)

    def create_scene_reconstruction(self, subject: str, title: str | None) -> TourScene:
        if not self.reconstruction_output_dir:
            raise InvalidDataError("reconstruction output directory is not configured")
        return import_from_reconstruction(
            self.repository, subject, self.reconstruction_output_dir, title, self.config.data
        )

    def create_scene_phase1(self, phase1_root: str, title: str | None) -> TourScene:
        return import_phase1(self.repository, phase1_root, title, self.config.data)

    def get_scene(self, scene_id: str, revision: int | None = None) -> TourScene:
        return self.repository.get_scene(scene_id, revision)

    def list_scenes(self) -> list[dict]:
        return self.repository.list_scenes(self.config.tour.get("max_scenes_per_page", 100))

    def update_annotations(self, scene_id: str, entities, views, expected_revision: int) -> TourScene:
        from pydantic import ValidationError

        from .models import Entity, ViewPreset

        try:
            parsed_entities = [Entity(**e) if isinstance(e, dict) else e for e in entities]
            parsed_views = [ViewPreset(**v) if isinstance(v, dict) else v for v in views]
        except ValidationError as exc:
            raise InvalidDataError(f"annotation payload failed validation: {exc}") from exc
        entities, views = parsed_entities, parsed_views
        with self.repository.locks.lock(f"scene:{scene_id}"):
            current = self.repository.get_scene(scene_id)
            if current.revision != expected_revision:
                raise ConflictError(f"scene revision moved to {current.revision}")
            # Human-confirmation invalidation rules:
            # - camera change clears that view's approvals
            # - entity bounds change removes the entity from every approval list
            old_views = {v.view_id: v for v in current.views}
            old_entities = {e.entity_id: e for e in current.entities}
            changed_entities = {
                e.entity_id
                for e in entities
                if e.entity_id in old_entities
                and old_entities[e.entity_id].bounds_asset != e.bounds_asset
            }
            for view in views:
                old = old_views.get(view.view_id)
                if old is not None and old.camera != view.camera:
                    view.approved_entity_ids = []
                    view.confirmation_source = "manual"
                    view.confirmed_at = None
                if changed_entities:
                    kept = [e for e in view.approved_entity_ids if e not in changed_entities]
                    if kept != view.approved_entity_ids:
                        view.approved_entity_ids = kept
                        view.confirmed_at = None
            scene_points = self.repository.assets.points(
                self.repository.assets.get(current.asset_id)
            )[0]
            validation = validate_annotations(entities, views, scene_points)
            if validation.status == "invalid":
                raise InvalidDataError("; ".join(validation.errors))
            updated = current.model_copy(
                update={
                    "revision": current.revision + 1,
                    "entities": entities,
                    "views": views,
                    "updated_at": now(),
                }
            )
            return self.repository.update_scene(updated, expected_revision)

    def preview_metadata(self, scene_id: str) -> dict:
        scene = self.repository.get_scene(scene_id)
        record = self.repository.assets.get(scene.asset_id)
        metadata = self.repository.ensure_preview(record, self.config.data)
        return {
            **metadata,
            "scene_id": scene.scene_id,
            "points_url": f"/api/tour/scenes/{scene.scene_id}/preview/points",
            "colors_url": f"/api/tour/scenes/{scene.scene_id}/preview/colors",
        }

    def preview_binary_path(self, scene_id: str, kind: str) -> Path:
        if kind not in ("points", "colors"):
            raise NotFoundError("unknown preview binary")
        scene = self.repository.get_scene(scene_id)
        record = self.repository.assets.get(scene.asset_id)
        metadata = self.repository.ensure_preview(record, self.config.data)
        path = self.repository.preview_dir(record.sha256) / f"{kind}.bin"
        if not path.is_file():
            raise NotFoundError(f"missing preview {kind}")
        return path

    # ------------------------------------------------------------- storyboards
    def save_storyboard_manual(self, payload: dict, expected_revision: int | None = None) -> Storyboard:
        """Create a new manual storyboard, or publish a new revision of an
        existing one when ``storyboard_id`` is present (with optimistic
        concurrency via ``expected_revision``)."""
        scene = self.repository.get_scene(payload["scene_id"], payload.get("scene_revision"))
        storyboard_id = payload.get("storyboard_id")
        if storyboard_id:
            current = self.repository.get_storyboard(storyboard_id)
            if expected_revision is None:
                raise ConflictError("expected_revision is required to update a storyboard")
            if current.revision != expected_revision:
                raise ConflictError(f"storyboard revision moved to {current.revision}")
            revision = current.revision + 1
            is_new = False
        else:
            storyboard_id = uuid.uuid4().hex
            revision = 1
            is_new = True
        candidate = Storyboard(
            storyboard_id=storyboard_id,
            revision=revision,
            scene_id=scene.scene_id,
            scene_revision=scene.revision,
            asset_sha256=scene.asset_sha256,
            title=payload["title"],
            language=payload["language"],
            source="manual",
            request_summary=payload.get("request_summary") or {},
            shots=payload["shots"],
            validation=StoryboardValidation(status="valid"),
            created_at=now(),
        )
        validation = validate_storyboard(
            scene,
            candidate,
            narration_limits=self.config.tour.get("narration", {}),
            duration_strict=False,
        )
        candidate.validation = validation
        if validation.status == "invalid":
            raise InvalidDataError("; ".join(validation.errors))
        if is_new:
            return self.repository.create_storyboard(candidate)
        return self.repository.update_storyboard(candidate, current.revision)

    def validate_candidate(self, payload: dict) -> dict:
        scene = self.repository.get_scene(payload["scene_id"], payload.get("scene_revision"))
        candidate = Storyboard(
            storyboard_id="candidate",
            revision=1,
            scene_id=scene.scene_id,
            scene_revision=scene.revision,
            asset_sha256=scene.asset_sha256,
            title=payload["title"],
            language=payload["language"],
            source="manual",
            request_summary=payload.get("request_summary") or {},
            shots=payload["shots"],
            validation=StoryboardValidation(status="valid"),
            created_at=now(),
        )
        validation = validate_storyboard(
            scene,
            candidate,
            narration_limits=self.config.tour.get("narration", {}),
            duration_strict=False,
        )
        return {
            "status": validation.status,
            "errors": validation.errors,
            "warnings": validation.warnings,
        }

    def get_storyboard(self, storyboard_id: str, revision: int | None = None) -> Storyboard:
        return self.repository.get_storyboard(storyboard_id, revision)

    def list_storyboards(self, scene_id: str | None = None) -> list[dict]:
        return self.repository.list_storyboards(scene_id)

    # ------------------------------------------------------------- generation
    def _build_provider(self):
        llm = self.config.llm
        if llm.get("provider") == "fixture":
            from .providers import FakeProvider

            return FakeProvider(llm.get("script", []), model=llm.get("model") or "fixture-model")
        key = self.config.llm_key()
        if not key or not llm.get("base_url") or not llm.get("model"):
            raise InvalidDataError("LLM is not configured (base_url/model/API key missing)")
        return ChatCompletionsProvider(
            base_url=llm["base_url"],
            model=llm["model"],
            api_key=key,
            response_mode=llm.get("response_mode", "text_json"),
            request_timeout_seconds=llm.get("request_timeout_seconds", 45),
            max_output_tokens=llm.get("max_output_tokens", 1500),
            temperature=llm.get("temperature", 0.2),
        )

    async def start_generation(self, request: GenerationRequest, request_id: str | None = None) -> TourRun:
        scene = self.repository.get_scene(request.scene_id, request.scene_revision)
        async with self._guard:
            if self._shutting_down:
                raise ConflictError("server is shutting down")
            existing = self._active.get(scene.scene_id)
            if existing:
                raise ConflictError(f"scene {scene.scene_id} already has active run {existing}")
            run_id = uuid.uuid4().hex
            run = TourRun(
                run_id=run_id,
                status="pending",
                scene_id=scene.scene_id,
                scene_revision=scene.revision,
                request=request,
                created_at=now(),
            )
            try:
                self.repository.create_run(run, request_id)
            except ConflictError as exc:
                if request_id:
                    previous = self.repository.run_for_request(request_id)
                    if previous is not None:
                        raise ConflictError(f"duplicate request_id, see run {previous.run_id}") from exc
                raise
            self._active[scene.scene_id] = run_id
            task = asyncio.create_task(self._execute_run(run_id))
            self._tasks[run_id] = task
        return run

    async def _execute_run(self, run_id: str):
        run = self.repository.get_run(run_id)
        try:
            scene = self.repository.get_scene(run.scene_id, run.scene_revision)
        except Exception as exc:  # noqa: BLE001
            run.status = "failed"
            run.error_code = "scene_missing"
            run.error_message = str(exc)
            run.started_at = now()
            run.finished_at = now()
            self.repository.update_run(run)
            await self._cleanup(run)
            return
        run.status = "running"
        run.started_at = now()
        self.repository.update_run(run)

        provider = None
        try:
            provider = self._build_provider()
        except InvalidDataError as exc:
            run.status = "failed"
            run.error_code = "llm_not_configured"
            run.error_message = str(exc)
            run.finished_at = now()
            self.repository.update_run(run)
            await self._cleanup(run)
            return

        outcome = await plan_storyboard(scene, run.request, provider, self.config.data)
        run.attempts = outcome.attempts
        if outcome.kind == "ok":
            storyboard = outcome.storyboard.model_copy(
                update={"storyboard_id": uuid.uuid4().hex}
            )
            self.repository.create_storyboard(storyboard)
            run.status = "succeeded"
            run.result_storyboard_id = storyboard.storyboard_id
        elif outcome.kind == "clarification":
            run.status = "needs_clarification"
            run.clarification_question = outcome.question
        else:
            run.status = "failed"
            run.error_code = outcome.error_code
            run.error_message = outcome.error_message
        run.finished_at = now()
        self.repository.update_run(run)
        await self._cleanup(run)

    async def _cleanup(self, run: TourRun):
        async with self._guard:
            if self._active.get(run.scene_id) == run.run_id:
                self._active.pop(run.scene_id, None)
            self._tasks.pop(run.run_id, None)

    def get_run(self, run_id: str) -> TourRun:
        return self.repository.get_run(run_id)

    async def shutdown(self):
        self._shutting_down = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._tasks.clear()
        self._active.clear()
