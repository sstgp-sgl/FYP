"""Thin HTTP routes for /api/tour. Business logic lives in TourService;
this module maps typed errors to status codes and shapes responses."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import GenerationRequest
from .repository import (
    InvalidDataError,
    PayloadTooLargeError,
)
from .service import TourService

MAX_UPLOAD_CHUNK = 1 << 20


class StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FromReconstructionIn(StrictPayload):
    subject: str
    title: Optional[str] = None


class FromPhase1In(StrictPayload):
    phase1_root: str
    title: Optional[str] = None


class CameraIn(StrictPayload):
    position_asset: list[float]
    target_asset: list[float]
    up: list[float]
    fov_y_deg: float = 60.0
    near_asset: float = 0.001
    far_asset: float = 100.0
    aspect_ratio: float = 16.0 / 9.0


class EntityIn(StrictPayload):
    entity_id: str
    label: str
    description: str = ""
    bounds_asset: list
    facts: list = Field(default_factory=list)
    status: str = "draft"


class ViewIn(StrictPayload):
    view_id: str
    name: str
    camera: CameraIn
    approved_entity_ids: list[str] = Field(default_factory=list)
    confirmation_source: str = "manual"
    confirmed_at: Optional[str] = None


class AnnotationsIn(StrictPayload):
    entities: list[EntityIn]
    views: list[ViewIn]
    expected_revision: int


class StoryboardSaveIn(StrictPayload):
    scene_id: str
    scene_revision: int
    title: str
    language: str
    shots: list[dict]
    storyboard_id: Optional[str] = None
    expected_revision: Optional[int] = None
    request_summary: dict = Field(default_factory=dict)


class ValidateIn(StrictPayload):
    scene_id: str
    scene_revision: int
    title: str
    language: str
    shots: list[dict]
    request_summary: dict = Field(default_factory=dict)


class GenerateIn(GenerationRequest):
    request_id: Optional[str] = None

    @field_validator("request_id")
    @classmethod
    def _rid(cls, value):
        if value is None:
            return None
        if not isinstance(value, str) or not value or len(value) > 128:
            raise ValueError("invalid request_id")
        if any(ch not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_" for ch in value):
            raise ValueError("invalid request_id characters")
        return value


async def _read_capped(upload: UploadFile, cap: int) -> bytes:
    buffer = bytearray()
    while True:
        chunk = await upload.read(MAX_UPLOAD_CHUNK)
        if not chunk:
            break
        if len(buffer) + len(chunk) > cap:
            raise PayloadTooLargeError(f"upload exceeds the {cap} byte limit")
        buffer.extend(chunk)
    return bytes(buffer)


def create_router(service: TourService) -> APIRouter:
    router = APIRouter(prefix="/api/tour", tags=["tour"])

    @router.get("/capabilities")
    def capabilities():
        return service.config.capabilities()

    @router.get("/scenes")
    def list_scenes():
        return {"scenes": service.list_scenes()}

    @router.post("/scenes/import", status_code=201)
    async def import_scene(title: str = Form(...), file: UploadFile = File(...)):
        if not (file.filename or "").lower().endswith(".ply"):
            raise InvalidDataError("uploaded file must be a .ply file")
        data = await _read_capped(file, service.config.tour["max_upload_bytes"])
        if not data:
            raise InvalidDataError("empty upload")
        scene = service.create_scene_upload(data, title.strip())
        return scene.model_dump()

    @router.post("/scenes/from-reconstruction", status_code=201)
    def import_reconstruction(body: FromReconstructionIn):
        scene = service.create_scene_reconstruction(body.subject, body.title)
        return scene.model_dump()

    @router.post("/scenes/from-phase1", status_code=201)
    def import_phase1(body: FromPhase1In):
        scene = service.create_scene_phase1(body.phase1_root, body.title)
        return scene.model_dump()

    @router.get("/scenes/{scene_id}")
    def get_scene(scene_id: str, revision: int | None = Query(default=None, ge=1)):
        return service.get_scene(scene_id, revision).model_dump()

    @router.put("/scenes/{scene_id}/annotations")
    def put_annotations(scene_id: str, body: AnnotationsIn):
        entities = []
        for item in body.entities:
            entities.append(
                {
                    "entity_id": item.entity_id,
                    "label": item.label,
                    "description": item.description,
                    "bounds_asset": item.bounds_asset,
                    "facts": [dict(fact) for fact in item.facts],
                    "status": item.status,
                }
            )
        views = []
        for item in body.views:
            views.append(
                {
                    "view_id": item.view_id,
                    "name": item.name,
                    "camera": item.camera.model_dump(),
                    "approved_entity_ids": item.approved_entity_ids,
                    "confirmation_source": item.confirmation_source,
                    "confirmed_at": item.confirmed_at,
                }
            )
        updated = service.update_annotations(scene_id, entities, views, body.expected_revision)
        return updated.model_dump()

    @router.get("/scenes/{scene_id}/preview")
    def preview(scene_id: str):
        return service.preview_metadata(scene_id)

    @router.get("/scenes/{scene_id}/preview/points")
    def preview_points(scene_id: str):
        path = service.preview_binary_path(scene_id, "points")
        return FileResponse(path, media_type="application/octet-stream")

    @router.get("/scenes/{scene_id}/preview/colors")
    def preview_colors(scene_id: str):
        path = service.preview_binary_path(scene_id, "colors")
        return FileResponse(path, media_type="application/octet-stream")

    @router.post("/storyboards/validate")
    def validate_storyboard(body: ValidateIn):
        return service.validate_candidate(body.model_dump())

    @router.post("/storyboards", status_code=201)
    def save_storyboard(body: StoryboardSaveIn):
        return service.save_storyboard_manual(body.model_dump(), body.expected_revision).model_dump()

    @router.get("/storyboards")
    def list_storyboards(scene_id: str | None = Query(default=None)):
        return {"storyboards": service.list_storyboards(scene_id)}

    @router.get("/storyboards/{storyboard_id}")
    def get_storyboard(storyboard_id: str, revision: int | None = Query(default=None, ge=1)):
        return service.get_storyboard(storyboard_id, revision).model_dump()

    @router.put("/storyboards/{storyboard_id}")
    def update_storyboard(storyboard_id: str, body: StoryboardSaveIn):
        if body.storyboard_id and body.storyboard_id != storyboard_id:
            raise InvalidDataError("path storyboard_id does not match payload")
        payload = body.model_dump()
        payload["storyboard_id"] = storyboard_id
        return service.save_storyboard_manual(payload, body.expected_revision).model_dump()

    @router.get("/storyboards/{storyboard_id}/download")
    def download_storyboard(storyboard_id: str,
                            revision: int | None = Query(default=None, ge=1)):
        storyboard = service.get_storyboard(storyboard_id, revision)
        content = json.dumps(
            storyboard.model_dump(), ensure_ascii=False, indent=2, allow_nan=False
        ).encode("utf-8")
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{storyboard_id}.storyboard.json"'
            },
        )

    @router.post("/generations", status_code=202)
    async def start_generation(body: GenerateIn):
        request = GenerationRequest(**body.model_dump(exclude={"request_id"}))
        run = await service.start_generation(request, body.request_id)
        return {"run_id": run.run_id, "status": run.status,
                "scene_id": run.scene_id, "scene_revision": run.scene_revision}

    @router.get("/runs/{run_id}")
    def get_run(run_id: str):
        return service.get_run(run_id).model_dump()

    return router
