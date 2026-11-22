"""Schema contracts: finite numbers, identifier rules, camera degeneracy,
bounds order, extra fields and version guards."""
import math

import pytest
from pydantic import ValidationError

from src.tour.models import Camera, Entity, Fact, GenerationRequest, Shot, Storyboard, TourScene


def base_scene(**overrides):
    data = {
        "scene_id": "scene_test",
        "revision": 1,
        "title": "测试场景",
        "asset_id": "asset_test",
        "asset_sha256": "a" * 64,
        "source_kind": "synthetic",
        "display_transform": {"center": [0.0, 0.0, 0.0], "scale": 1.0},
        "entities": [],
        "views": [],
        "created_at": "2026-11-08T00:00:00+00:00",
        "updated_at": "2026-11-08T00:00:00+00:00",
    }
    data.update(overrides)
    return TourScene(**data)


def test_nan_rejected_in_camera():
    with pytest.raises(ValidationError):
        Camera(position_asset=[0.0, float("nan"), 1.0], target_asset=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0])


def test_inf_rejected_in_bounds():
    with pytest.raises(ValidationError):
        Entity(entity_id="e1", label="x", bounds_asset=[[0.0, 0.0, 0.0], [1.0, math.inf, 1.0]])


def test_bool_rejected_as_number():
    with pytest.raises(ValidationError):
        Camera(position_asset=[0.0, 0.0, 1.0], target_asset=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
               fov_y_deg=True)


def test_inverted_bbox_rejected():
    with pytest.raises(ValidationError):
        Entity(entity_id="e1", label="x", bounds_asset=[[2.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def test_position_equals_target_rejected():
    with pytest.raises(ValidationError):
        Camera(position_asset=[0.0, 0.0, 1.0], target_asset=[0.0, 0.0, 1.0], up=[0.0, 1.0, 0.0])


def test_zero_up_rejected():
    with pytest.raises(ValidationError):
        Camera(position_asset=[0.0, 0.0, 1.0], target_asset=[0.0, 0.0, 0.0], up=[0.0, 0.0, 0.0])


def test_up_collinear_with_view_rejected():
    with pytest.raises(ValidationError):
        Camera(position_asset=[0.0, 0.0, 1.0], target_asset=[0.0, 0.0, 0.0], up=[0.0, 0.0, -1.0])


def test_wrong_schema_version_rejected():
    with pytest.raises(ValidationError):
        base_scene(schema_version=2)


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        base_scene(surprise="not allowed")


def test_duplicate_entity_ids_rejected():
    entity = {"entity_id": "e1", "label": "a", "bounds_asset": [[0, 0, 0], [1, 1, 1]], "facts": [], "status": "confirmed"}
    with pytest.raises(ValidationError):
        base_scene(entities=[entity, dict(entity)])


def test_view_approving_unknown_entity_rejected():
    view = {
        "view_id": "v1", "name": "v",
        "camera": {"position_asset": [0, 0, 1], "target_asset": [0, 0, 0], "up": [0, 1, 0]},
        "approved_entity_ids": ["ghost"],
    }
    with pytest.raises(ValidationError):
        base_scene(views=[view])


def test_invalid_identifier_rejected():
    with pytest.raises(ValidationError):
        base_scene(scene_id="bad id!")


def test_generation_request_limits():
    with pytest.raises(ValidationError):
        GenerationRequest(scene_id="s", scene_revision=1, user_prompt="x", shot_count=9)
    with pytest.raises(ValidationError):
        GenerationRequest(scene_id="s", scene_revision=1, user_prompt="x", target_duration_seconds=121)


def test_shot_duration_bounds():
    with pytest.raises(ValidationError):
        Shot(shot_id="s1", view_id="v1", focus_entity_ids=[], narration="x", duration_seconds=1)
    with pytest.raises(ValidationError):
        Shot(shot_id="s1", view_id="v1", focus_entity_ids=[], narration="x", duration_seconds=61)


def test_storyboard_duplicate_shot_ids_rejected():
    shot = Shot(shot_id="s1", view_id="v1", focus_entity_ids=[], narration="x", duration_seconds=10)
    with pytest.raises(ValidationError):
        Storyboard(
            storyboard_id="sb", revision=1, scene_id="scene_test", scene_revision=1,
            asset_sha256="a" * 64, title="t", language="zh-CN", source="manual",
            shots=[shot, shot.model_copy(deep=True)],
            validation={"schema_version": 1, "status": "valid", "errors": [], "warnings": []},
            created_at="2026-11-08T00:00:00+00:00",
        )


def test_fact_scope_unknown_view_rejected():
    entity = Entity(
        entity_id="e1", label="a", bounds_asset=[[0, 0, 0], [1, 1, 1]], status="confirmed",
        facts=[Fact(fact_id="f1", text="x", scope_view_id="missing_view")],
    )
    with pytest.raises(ValidationError):
        base_scene(entities=[entity])
