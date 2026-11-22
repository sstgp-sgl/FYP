"""Tour v1 contracts (Pydantic v2). Coordinates are stored in the source asset
frame (``coordinate_frame="asset"``); no Y-up or ground assumption is made.

Everything is validated: identifiers, finite numbers, bounds order, camera
degeneracy, ID references. Extra fields are rejected.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VERSION = 1
TOUR_LANGUAGES = ("zh-CN", "en")
SOURCE_KINDS = ("real", "synthetic")
UNIT = "arbitrary"
COORDINATE_FRAME = "asset"

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def now():
    return datetime.now(timezone.utc).isoformat()


def check_identifier(value: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ValueError(f"invalid identifier: {value!r}")
    return value


def check_finite(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"expected a finite number, got {value!r}")
    return float(value)


Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Bounds = Annotated[list[Vec3], Field(min_length=2, max_length=2)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Camera(StrictModel):
    position_asset: Vec3
    target_asset: Vec3
    up: Vec3
    fov_y_deg: float = 60.0
    near_asset: float = 0.001
    far_asset: float = 100.0
    aspect_ratio: float = 16.0 / 9.0

    @field_validator("position_asset", "target_asset", "up", mode="before")
    @classmethod
    def _no_bool_vec(cls, value):
        if isinstance(value, (list, tuple)) and any(isinstance(x, bool) for x in value):
            raise ValueError("boolean values are not vector components")
        return value

    @field_validator("fov_y_deg", "near_asset", "far_asset", "aspect_ratio", mode="before")
    @classmethod
    def _no_bool_num(cls, value):
        if isinstance(value, bool):
            raise ValueError("boolean is not a number")
        return value

    @field_validator("position_asset", "target_asset", "up")
    @classmethod
    def _finite_vec(cls, value):
        return [check_finite(x) for x in value]

    @field_validator("fov_y_deg", "near_asset", "far_asset", "aspect_ratio")
    @classmethod
    def _finite_num(cls, value):
        return check_finite(value)

    @model_validator(mode="after")
    def _check_degenerate(self):
        if self.position_asset == self.target_asset:
            raise ValueError("camera position must differ from target")
        up_norm = math.sqrt(sum(x * x for x in self.up))
        if up_norm == 0:
            raise ValueError("camera up vector must be non-zero")
        view = [t - p for p, t in zip(self.position_asset, self.target_asset)]
        cross_norm = math.sqrt(
            sum(
                (self.up[(i + 1) % 3] * view[(i + 2) % 3] - self.up[(i + 2) % 3] * view[(i + 1) % 3]) ** 2
                for i in range(3)
            )
        )
        if cross_norm <= 1e-12 * (up_norm * math.sqrt(sum(v * v for v in view)) + 1e-300):
            raise ValueError("camera up vector must not be collinear with the view direction")
        if not 1.0 <= self.fov_y_deg < 180.0:
            raise ValueError("fov_y_deg must be in [1, 180)")
        if not 0 < self.near_asset < self.far_asset:
            raise ValueError("require 0 < near < far")
        if not 0.1 <= self.aspect_ratio <= 10:
            raise ValueError("aspect_ratio out of range")
        return self


class Fact(StrictModel):
    fact_id: str
    text: str
    source: Literal["user_confirmed"] = "user_confirmed"
    kind: Literal["descriptive", "spatial"] = "descriptive"
    scope_view_id: Optional[str] = None

    @field_validator("fact_id", "scope_view_id")
    @classmethod
    def _id(cls, value):
        if value is None:
            return None
        return check_identifier(value)

    @field_validator("text")
    @classmethod
    def _text(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("fact text must be non-empty")
        return value.strip()


class Entity(StrictModel):
    entity_id: str
    label: str
    description: str = ""
    bounds_asset: Bounds
    facts: list[Fact] = Field(default_factory=list)
    status: Literal["confirmed", "draft"] = "draft"

    @field_validator("entity_id")
    @classmethod
    def _id(cls, value):
        return check_identifier(value)

    @field_validator("label")
    @classmethod
    def _label(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("entity label must be non-empty")
        return value.strip()

    @field_validator("bounds_asset", mode="before")
    @classmethod
    def _no_bool_bounds(cls, value):
        if isinstance(value, (list, tuple)):
            for corner in value:
                if isinstance(corner, (list, tuple)) and any(isinstance(x, bool) for x in corner):
                    raise ValueError("boolean values are not bounds components")
        return value

    @field_validator("bounds_asset")
    @classmethod
    def _bounds(cls, value):
        lo = [check_finite(x) for x in value[0]]
        hi = [check_finite(x) for x in value[1]]
        if any(a > b for a, b in zip(lo, hi)):
            raise ValueError("inverted entity bounds (minimum exceeds maximum)")
        return [lo, hi]

    @field_validator("facts")
    @classmethod
    def _facts(cls, value):
        ids = [f.fact_id for f in value]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate fact_id inside entity")
        return value


class ViewPreset(StrictModel):
    view_id: str
    name: str
    camera: Camera
    approved_entity_ids: list[str] = Field(default_factory=list)
    confirmation_source: Literal["manual", "synthetic_fixture"] = "manual"
    confirmed_at: Optional[str] = None

    @field_validator("view_id")
    @classmethod
    def _id(cls, value):
        return check_identifier(value)

    @field_validator("name")
    @classmethod
    def _name(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("view name must be non-empty")
        return value.strip()

    @field_validator("approved_entity_ids")
    @classmethod
    def _approved(cls, value):
        ids = [check_identifier(v) for v in value]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate approved_entity_ids")
        return ids


class DisplayTransform(StrictModel):
    center: Vec3
    scale: float

    @field_validator("center")
    @classmethod
    def _center(cls, value):
        return [check_finite(x) for x in value]

    @field_validator("scale")
    @classmethod
    def _scale(cls, value):
        value = check_finite(value)
        if value <= 0:
            raise ValueError("display scale must be positive")
        return value


class TourScene(StrictModel):
    schema_version: Literal[1] = 1
    scene_id: str
    revision: int = Field(ge=1)
    title: str
    asset_id: str
    asset_sha256: str
    source_kind: Literal["real", "synthetic"]
    coordinate_frame: Literal["asset"] = "asset"
    unit: str = UNIT
    display_transform: DisplayTransform
    entities: list[Entity] = Field(default_factory=list)
    views: list[ViewPreset] = Field(default_factory=list)
    created_at: str
    updated_at: str

    @field_validator("scene_id", "asset_id")
    @classmethod
    def _id(cls, value):
        return check_identifier(value)

    @field_validator("asset_sha256")
    @classmethod
    def _sha(cls, value):
        if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
            raise ValueError("invalid sha256 digest")
        return value

    @field_validator("title")
    @classmethod
    def _title(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("scene title must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _unique_ids(self):
        entity_ids = [e.entity_id for e in self.entities]
        view_ids = [v.view_id for v in self.views]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("duplicate entity_id")
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("duplicate view_id")
        known = set(entity_ids)
        for view in self.views:
            for approved in view.approved_entity_ids:
                if approved not in known:
                    raise ValueError(f"view {view.view_id} approves unknown entity {approved}")
        for entity in self.entities:
            for fact in entity.facts:
                if fact.scope_view_id is not None and fact.scope_view_id not in set(view_ids):
                    raise ValueError(f"fact {fact.fact_id} references unknown view {fact.scope_view_id}")
        return self


class GenerationRequest(StrictModel):
    scene_id: str
    scene_revision: int = Field(ge=1)
    user_prompt: str
    language: Literal["zh-CN", "en"] = "zh-CN"
    target_duration_seconds: int = Field(default=30, ge=5, le=120)
    shot_count: int = Field(default=3, ge=1, le=6)
    required_entity_ids: list[str] = Field(default_factory=list)
    opening_view_id: Optional[str] = None

    @field_validator("scene_id", "required_entity_ids", "opening_view_id")
    @classmethod
    def _id(cls, value):
        if value is None:
            return None
        if isinstance(value, list):
            ids = [check_identifier(v) for v in value]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate required_entity_ids")
            return ids
        return check_identifier(value)

    @field_validator("user_prompt")
    @classmethod
    def _prompt(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("user_prompt must be non-empty")
        if len(value) > 4000:
            raise ValueError("user_prompt too long")
        return value.strip()


class Shot(StrictModel):
    shot_id: str
    view_id: str
    focus_entity_ids: list[str]
    narration: str
    fact_ids: list[str] = Field(default_factory=list)
    duration_seconds: int = Field(ge=2, le=60)
    transition: Literal["cut"] = "cut"

    @field_validator("shot_id", "view_id")
    @classmethod
    def _id(cls, value):
        return check_identifier(value)

    @field_validator("focus_entity_ids", "fact_ids")
    @classmethod
    def _ids(cls, value):
        ids = [check_identifier(v) for v in value]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ids in shot list")
        return ids

    @field_validator("narration")
    @classmethod
    def _narration(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("narration must be non-empty")
        return value.strip()


class StoryboardValidation(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["valid", "invalid"]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class Storyboard(StrictModel):
    schema_version: Literal[1] = 1
    storyboard_id: str
    revision: int = Field(ge=1)
    scene_id: str
    scene_revision: int = Field(ge=1)
    asset_sha256: str
    title: str
    language: Literal["zh-CN", "en"]
    source: Literal["manual", "template", "llm", "fixture"]
    request_summary: dict = Field(default_factory=dict)
    shots: list[Shot] = Field(min_length=1)
    validation: StoryboardValidation
    created_at: str

    @field_validator("storyboard_id", "scene_id")
    @classmethod
    def _id(cls, value):
        return check_identifier(value)

    @field_validator("asset_sha256")
    @classmethod
    def _sha(cls, value):
        if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
            raise ValueError("invalid sha256 digest")
        return value

    @field_validator("title")
    @classmethod
    def _title(cls, value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("storyboard title must be non-empty")
        return value.strip()

    @model_validator(mode="after")
    def _unique_shots(self):
        ids = [s.shot_id for s in self.shots]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate shot_id")
        return self


class LLMResponse(StrictModel):
    """Discriminated union over the two LLM terminal shapes.

    ``{status: ok, title, shots:[...]}`` or ``{status: needs_clarification, question}``.
    """

    status: Literal["ok", "needs_clarification"]
    title: Optional[str] = None
    shots: Optional[list[Shot]] = None
    question: Optional[str] = None

    @model_validator(mode="after")
    def _shape(self):
        if self.status == "ok":
            if not self.title or not self.shots or len(self.shots) == 0:
                raise ValueError("ok response requires title and non-empty shots")
        elif not self.question or not self.question.strip():
            raise ValueError("needs_clarification response requires a question")
        return self


class Attempt(StrictModel):
    attempt: int
    status: Literal["ok", "needs_clarification", "invalid_response", "provider_error", "validation_error"]
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_seconds: Optional[float] = None
    usage: Optional[dict] = None
    finish_reason: Optional[str] = None
    raw_sha256: Optional[str] = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("latency_seconds")
    @classmethod
    def _latency(cls, value):
        if value is None:
            return None
        value = check_finite(value)
        if value < 0:
            raise ValueError("negative latency")
        return value


class TourRun(StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    status: Literal["pending", "running", "succeeded", "failed", "interrupted", "needs_clarification"]
    scene_id: str
    scene_revision: int = Field(ge=1)
    request: GenerationRequest
    attempts: list[Attempt] = Field(default_factory=list)
    result_storyboard_id: Optional[str] = None
    clarification_question: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    @field_validator("run_id", "scene_id", "result_storyboard_id")
    @classmethod
    def _id(cls, value):
        if value is None:
            return None
        return check_identifier(value)
