"""Tour persistence: versioned scene/storyboard snapshots, run records, and
the fixed preview cache. Reuses the Phase 1 ``AssetRepository`` for immutable
copied assets (``outputs/tours/assets/<id>/data.ply`` + ``asset.json``).

All writes go to a temporary file in the same directory and are published with
``os.replace``; indexes are updated only after the revision payload is on disk.
Single-process, per-key locks — no claim of multi-process transactions.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np

from src.scene.storage import AssetRepository, atomic_json, controlled, read_json

from .geometry import compute_display_transform, preview_seed, sample_preview
from .models import Storyboard, TourRun, TourScene


class TourError(Exception):
    """Base for typed tour errors mapped to HTTP status codes."""

    status_code = 500


class NotFoundError(TourError):
    status_code = 404


class ConflictError(TourError):
    status_code = 409


class RevisionConflictError(ConflictError):
    pass


class PayloadTooLargeError(TourError):
    status_code = 413


class InvalidDataError(TourError):
    status_code = 422


class LockRegistry:
    """Per-key re-entrant locks for the single worker process."""

    def __init__(self):
        self._guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}

    def lock(self, key: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(key, threading.RLock())


def _revision_dir(base: Path, revision: int) -> Path:
    return base / f"{revision:06d}"


class TourRepository:
    def __init__(self, root, lock_registry=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets = AssetRepository(self.root)
        self.locks = lock_registry or LockRegistry()

    # ------------------------------------------------------------- helpers
    def _scene_dir(self, scene_id) -> Path:
        return controlled(self.root, f"scenes/{scene_id}")

    def _storyboard_dir(self, storyboard_id) -> Path:
        return controlled(self.root, f"storyboards/{storyboard_id}")

    def _run_dir(self, run_id) -> Path:
        return controlled(self.root, f"runs/{run_id}")

    # ------------------------------------------------------------- scenes
    def create_scene(self, scene: TourScene) -> TourScene:
        directory = self._scene_dir(scene.scene_id)
        if directory.exists():
            raise ConflictError(f"scene {scene.scene_id} already exists")
        directory.mkdir(parents=True)
        try:
            self._write_scene_revision(scene, is_new=True)
        except BaseException:
            from shutil import rmtree

            rmtree(directory, ignore_errors=True)
            raise
        return scene

    def update_scene(self, scene: TourScene, expected_revision: int) -> TourScene:
        with self.locks.lock(f"scene:{scene.scene_id}"):
            current = self.get_scene(scene.scene_id)
            if current.revision != expected_revision:
                raise RevisionConflictError(
                    f"scene revision moved from {expected_revision} to {current.revision}"
                )
            self._write_scene_revision(scene, is_new=False)
        return scene

    def _write_scene_revision(self, scene: TourScene, is_new: bool):
        directory = self._scene_dir(scene.scene_id)
        revision_dir = _revision_dir(directory, scene.revision)
        revision_dir.mkdir(parents=True, exist_ok=False)
        payload = json.loads(scene.model_dump_json())
        atomic_json(revision_dir / "scene.json", payload)
        index = {
            "schema_version": 1,
            "scene_id": scene.scene_id,
            "latest_revision": scene.revision,
            "title": scene.title,
            "source_kind": scene.source_kind,
            "updated_at": scene.updated_at,
        }
        atomic_json(directory / "index.json", index)
        if is_new:
            seen = Path(self.root) / "scenes" / ".seen"
            seen.mkdir(parents=True, exist_ok=True)
            with open(seen / f"{scene.scene_id}", "w") as stream:
                stream.write(scene.created_at)

    def get_scene(self, scene_id: str, revision: int | None = None) -> TourScene:
        directory = self._scene_dir(scene_id)
        index_path = directory / "index.json"
        if not index_path.is_file():
            raise NotFoundError(f"scene {scene_id} does not exist")
        index = read_json(index_path)
        if revision is None:
            revision = index["latest_revision"]
        payload = read_json(_revision_dir(directory, revision) / "scene.json")
        scene = TourScene.model_validate(payload)
        if scene.scene_id != scene_id or scene.revision != revision:
            raise InvalidDataError("scene identity mismatch")
        self._verify_scene_asset(scene)
        return scene

    def list_scenes(self, limit: int = 100) -> list[dict]:
        scenes_root = Path(self.root) / "scenes"
        if not scenes_root.is_dir():
            return []
        summaries = []
        for directory in sorted(scenes_root.iterdir()):
            index_path = directory / "index.json"
            if not directory.is_dir() or not index_path.is_file():
                continue
            index = read_json(index_path)
            summaries.append(
                {
                    "scene_id": index["scene_id"],
                    "revision": index["latest_revision"],
                    "title": index.get("title"),
                    "source_kind": index.get("source_kind"),
                    "updated_at": index.get("updated_at"),
                }
            )
        summaries.sort(key=lambda s: s["updated_at"] or "", reverse=True)
        return summaries[:limit]

    def _verify_scene_asset(self, scene: TourScene):
        record = self.assets.get(scene.asset_id)
        if record.sha256 != scene.asset_sha256:
            raise InvalidDataError("scene asset hash mismatch")
        return record

    # ------------------------------------------------------------- previews
    def preview_dir(self, asset_sha256: str) -> Path:
        return controlled(self.root, f"previews/{asset_sha256}")

    def ensure_preview(self, record, config: dict) -> dict:
        """Generate (once) or load the fixed preview cache for one asset.

        ``config`` needs ``preview_points``. The cache is keyed by the asset
        hash, so a changed asset can never reuse stale preview bytes.
        """
        preview = self.preview_dir(record.sha256)
        metadata_path = preview / "metadata.json"
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            if (
                metadata.get("asset_sha256") == record.sha256
                and (preview / "points.bin").is_file()
                and (preview / "colors.bin").is_file()
            ):
                return metadata
        preview.mkdir(parents=True, exist_ok=True)
        points, colors = self.assets.points(record)
        transform = compute_display_transform(points)
        shown_points, shown_colors = sample_preview(
            points, colors, int(config["tour"]["preview_points"]), preview_seed(record.sha256)
        )
        display = (
            (np.asarray(shown_points, dtype=np.float64) - np.asarray(transform.center))
            * transform.scale
        )
        display_f32 = display.astype(np.float32)
        colors_u8 = np.asarray(shown_colors, dtype=np.uint8)
        metadata = {
            "schema_version": 1,
            "asset_sha256": record.sha256,
            "total_points": int(len(points)),
            "shown_points": int(len(shown_points)),
            "display_center": transform.center,
            "display_scale": transform.scale,
            "points_bytes": int(display_f32.nbytes),
            "colors_bytes": int(colors_u8.nbytes),
            "source_kind": record.source_kind,
        }
        temporary = preview / ".pending"
        temporary.mkdir(exist_ok=True)
        with open(preview / ".points.bin.tmp", "wb") as stream:
            stream.write(display_f32.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(preview / ".points.bin.tmp", preview / "points.bin")
        with open(preview / ".colors.bin.tmp", "wb") as stream:
            stream.write(colors_u8.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(preview / ".colors.bin.tmp", preview / "colors.bin")
        atomic_json(metadata_path, metadata)
        return metadata

    # ------------------------------------------------------------- storyboards
    def create_storyboard(self, storyboard: Storyboard) -> Storyboard:
        directory = self._storyboard_dir(storyboard.storyboard_id)
        if directory.exists():
            raise ConflictError(f"storyboard {storyboard.storyboard_id} already exists")
        directory.mkdir(parents=True)
        try:
            self._write_storyboard_revision(storyboard)
        except BaseException:
            from shutil import rmtree

            rmtree(directory, ignore_errors=True)
            raise
        return storyboard

    def update_storyboard(self, storyboard: Storyboard, expected_revision: int) -> Storyboard:
        with self.locks.lock(f"storyboard:{storyboard.storyboard_id}"):
            current = self.get_storyboard(storyboard.storyboard_id)
            if current.revision != expected_revision:
                raise RevisionConflictError(
                    f"storyboard revision moved from {expected_revision} to {current.revision}"
                )
            self._write_storyboard_revision(storyboard)
        return storyboard

    def _write_storyboard_revision(self, storyboard: Storyboard):
        directory = self._storyboard_dir(storyboard.storyboard_id)
        revision_dir = _revision_dir(directory, storyboard.revision)
        revision_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(
            revision_dir / "storyboard.json", json.loads(storyboard.model_dump_json())
        )
        atomic_json(
            directory / "index.json",
            {
                "schema_version": 1,
                "storyboard_id": storyboard.storyboard_id,
                "latest_revision": storyboard.revision,
                "scene_id": storyboard.scene_id,
                "scene_revision": storyboard.scene_revision,
                "title": storyboard.title,
                "source": storyboard.source,
                "created_at": storyboard.created_at,
            },
        )

    def get_storyboard(self, storyboard_id: str, revision: int | None = None) -> Storyboard:
        directory = self._storyboard_dir(storyboard_id)
        index_path = directory / "index.json"
        if not index_path.is_file():
            raise NotFoundError(f"storyboard {storyboard_id} does not exist")
        index = read_json(index_path)
        if revision is None:
            revision = index["latest_revision"]
        payload = read_json(_revision_dir(directory, revision) / "storyboard.json")
        storyboard = Storyboard.model_validate(payload)
        if storyboard.storyboard_id != storyboard_id or storyboard.revision != revision:
            raise InvalidDataError("storyboard identity mismatch")
        return storyboard

    def list_storyboards(self, scene_id: str | None = None) -> list[dict]:
        root = Path(self.root) / "storyboards"
        if not root.is_dir():
            return []
        summaries = []
        for directory in sorted(root.iterdir()):
            index_path = directory / "index.json"
            if not directory.is_dir() or not index_path.is_file():
                continue
            index = read_json(index_path)
            if scene_id is not None and index.get("scene_id") != scene_id:
                continue
            summaries.append(
                {
                    "storyboard_id": index["storyboard_id"],
                    "revision": index["latest_revision"],
                    "scene_id": index.get("scene_id"),
                    "scene_revision": index.get("scene_revision"),
                    "title": index.get("title"),
                    "source": index.get("source"),
                    "created_at": index.get("created_at"),
                }
            )
        summaries.sort(key=lambda s: s["created_at"] or "", reverse=True)
        return summaries

    # ------------------------------------------------------------- runs
    def create_run(self, run: TourRun, request_id: str | None = None) -> TourRun:
        directory = self._run_dir(run.run_id)
        if directory.exists():
            raise ConflictError(f"run {run.run_id} already exists")
        directory.mkdir(parents=True)
        if request_id:
            request_index = Path(self.root) / "runs" / "request_ids"
            request_index.mkdir(parents=True, exist_ok=True)
            marker = request_index / request_id
            if marker.exists():
                existing = read_json(marker)["run_id"]
                raise ConflictError(f"request already registered as run {existing}")
            atomic_json(marker, {"schema_version": 1, "run_id": run.run_id})
        self._write_run(run)
        return run

    def run_for_request(self, request_id: str) -> TourRun | None:
        marker = Path(self.root) / "runs" / "request_ids" / request_id
        if not marker.is_file():
            return None
        run_id = read_json(marker)["run_id"]
        return self.get_run(run_id)

    def _write_run(self, run: TourRun):
        directory = self._run_dir(run.run_id)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "run.json", json.loads(run.model_dump_json()))
        atomic_json(directory / "request.json", json.loads(run.request.model_dump_json()))

    def update_run(self, run: TourRun) -> TourRun:
        self._write_run(run)
        return run

    def get_run(self, run_id: str) -> TourRun:
        directory = self._run_dir(run_id)
        run_path = directory / "run.json"
        if not run_path.is_file():
            raise NotFoundError(f"run {run_id} does not exist")
        run = TourRun.model_validate(read_json(run_path))
        if run.run_id != run_id:
            raise InvalidDataError("run identity mismatch")
        return run

    def save_attempt(self, run_id: str, attempt_no: int, response_text: str, validation: dict):
        directory = self._run_dir(run_id)
        attempts = directory / "attempts" / f"{attempt_no:02d}"
        attempts.mkdir(parents=True, exist_ok=False)
        atomic_json(attempts / "response.json", {"schema_version": 1, "text": response_text})
        atomic_json(attempts / "validation.json", validation)

    def recover_interrupted(self):
        """Mark runs left pending/running by a previous process as interrupted."""
        root = Path(self.root) / "runs"
        if not root.is_dir():
            return 0
        recovered = 0
        for directory in root.iterdir():
            run_path = directory / "run.json"
            if not directory.is_dir() or not run_path.is_file():
                continue
            try:
                run = TourRun.model_validate(read_json(run_path))
            except Exception:  # noqa: BLE001
                continue
            if run.status in ("pending", "running"):
                run.status = "interrupted"
                run.finished_at = run.finished_at or None
                self._write_run(run)
                recovered += 1
        return recovered
