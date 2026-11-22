"""Validator: references, confirmed visibility, fact scoping, duration,
shot count, language and required-entity coverage."""
import pytest

from src.tour.importers import seed_demo
from src.tour.models import GenerationRequest, Shot, Storyboard, StoryboardValidation, now
from src.tour.repository import TourRepository
from src.tour.validation import generation_precheck, validate_annotations, validate_storyboard
from tests.tour.conftest import make_config


@pytest.fixture
def seeded(tour_root):
    repository = TourRepository(tour_root)
    result = seed_demo(repository, make_config())
    return repository, repository.get_scene(result["scene_id"])


def base_storyboard(scene, shots, language="zh-CN"):
    return Storyboard(
        storyboard_id="sb_test", revision=1, scene_id=scene.scene_id,
        scene_revision=scene.revision, asset_sha256=scene.asset_sha256,
        title="t", language=language, source="manual", shots=shots,
        validation=StoryboardValidation(status="valid"), created_at=now(),
    )


def valid_shot(scene, index=1):
    view = scene.views[0]
    focus = view.approved_entity_ids
    return Shot(shot_id=f"shot_{index:02d}", view_id=view.view_id,
                focus_entity_ids=focus, narration=f"第{index}段旁白",
                fact_ids=[], duration_seconds=10)


def request_for(scene, **overrides):
    data = dict(scene_id=scene.scene_id, scene_revision=scene.revision,
                user_prompt="test", language="zh-CN",
                target_duration_seconds=30, shot_count=3)
    data.update(overrides)
    return GenerationRequest(**data)


def test_valid_storyboard_passes(seeded):
    repo, scene = seeded
    shots = [valid_shot(scene, i) for i in range(1, 4)]
    validation = validate_storyboard(scene, base_storyboard(scene, shots),
                                     request_for(scene), duration_strict=True)
    assert validation.status == "valid", validation.errors


def test_unknown_view_rejected(seeded):
    repo, scene = seeded
    shots = [valid_shot(scene, 1)]
    shots.append(shots[0].model_copy(update={"shot_id": "shot_02", "view_id": "ghost"}))
    validation = validate_storyboard(scene, base_storyboard(scene, shots),
                                     request_for(scene, shot_count=2, target_duration_seconds=20))
    assert validation.status == "invalid"
    assert any("unknown view" in e for e in validation.errors)


def test_unknown_entity_rejected(seeded):
    repo, scene = seeded
    shots = [valid_shot(scene, 1)]
    shots.append(shots[0].model_copy(update={"shot_id": "shot_02", "focus_entity_ids": ["ghost"]}))
    validation = validate_storyboard(scene, base_storyboard(scene, shots))
    assert validation.status == "invalid"
    assert any("unknown entity" in e for e in validation.errors)


def test_invisible_focus_rejected(seeded):
    repo, scene = seeded
    building_view = next(v for v in scene.views if v.view_id == "view_building")
    tree_view = next(v for v in scene.views if v.view_id == "view_tree")
    shots = [
        Shot(shot_id="shot_01", view_id=building_view.view_id,
             focus_entity_ids=["entity_tree"],  # tree is NOT approved from this view
             narration="x", duration_seconds=10),
        Shot(shot_id="shot_02", view_id=tree_view.view_id,
             focus_entity_ids=["entity_tree"], narration="y", duration_seconds=10),
    ]
    validation = validate_storyboard(scene, base_storyboard(scene, shots))
    assert validation.status == "invalid"
    assert any("not human-confirmed visible" in e for e in validation.errors)


def test_fact_scope_enforced(seeded):
    repo, scene = seeded
    # fact_building_west is scoped to view_overview only
    shots = [
        Shot(shot_id="shot_01", view_id="view_building",
             focus_entity_ids=["entity_building"], narration="x",
             fact_ids=["fact_building_west"], duration_seconds=10),
    ]
    validation = validate_storyboard(scene, base_storyboard(scene, shots))
    assert validation.status == "invalid"
    assert any("view-dependent fact" in e for e in validation.errors)
    # same fact from its scope view is fine
    shots[0] = shots[0].model_copy(update={"view_id": "view_overview",
                                           "focus_entity_ids": ["entity_building"]})
    validation = validate_storyboard(scene, base_storyboard(scene, shots))
    assert validation.status == "valid", validation.errors


def test_duration_strictness(seeded):
    repo, scene = seeded
    shots = [valid_shot(scene, i) for i in range(1, 4)]
    shots[2] = shots[2].model_copy(update={"duration_seconds": 8})  # total 28 vs 30
    storyboard = base_storyboard(scene, shots)
    strict = validate_storyboard(scene, storyboard, request_for(scene), duration_strict=True)
    assert strict.status == "invalid"
    assert any("differs from the requested" in e for e in strict.errors)
    relaxed = validate_storyboard(scene, storyboard, request_for(scene), duration_strict=False)
    assert relaxed.status == "valid"
    assert any("differs from the requested" in w for w in relaxed.warnings)


def test_shot_count_mismatch(seeded):
    repo, scene = seeded
    shots = [valid_shot(scene, i) for i in range(1, 3)]
    validation = validate_storyboard(scene, base_storyboard(scene, shots),
                                     request_for(scene, shot_count=3, target_duration_seconds=20),
                                     duration_strict=True)
    assert validation.status == "invalid"
    assert any("expected 3 shots" in e for e in validation.errors)


def test_required_entity_coverage(seeded):
    repo, scene = seeded
    tree_view = next(v for v in scene.views if v.view_id == "view_tree")
    shots = [
        Shot(shot_id="shot_01", view_id=tree_view.view_id, focus_entity_ids=["entity_tree"],
             narration="x", duration_seconds=10),
    ]
    validation = validate_storyboard(
        scene, base_storyboard(scene, shots),
        request_for(scene, shot_count=1, target_duration_seconds=10,
                    required_entity_ids=["entity_tree", "entity_wall"]),
        duration_strict=True,
    )
    assert validation.status == "invalid"
    assert any("required entity entity_wall" in e for e in validation.errors)


def test_language_mismatch_strict(seeded):
    repo, scene = seeded
    shots = [valid_shot(scene, 1)]
    validation = validate_storyboard(
        scene, base_storyboard(scene, shots, language="en"),
        request_for(scene, shot_count=1, target_duration_seconds=10), duration_strict=True,
    )
    assert validation.status == "invalid"
    assert any("language" in e for e in validation.errors)


def test_opening_view_enforced(seeded):
    repo, scene = seeded
    shots = [valid_shot(scene, i) for i in range(1, 4)]
    shots[0] = shots[0].model_copy(update={"view_id": "view_tree"})
    validation = validate_storyboard(
        scene, base_storyboard(scene, shots),
        request_for(scene, opening_view_id="view_overview"), duration_strict=True,
    )
    assert validation.status == "invalid"
    assert any("opening view" in e for e in validation.errors)


def test_narration_length_warning(seeded):
    repo, scene = seeded
    shots = [valid_shot(scene, 1)]
    shots[0] = shots[0].model_copy(update={"narration": "很长的旁白" * 40, "duration_seconds": 2})
    validation = validate_storyboard(scene, base_storyboard(scene, shots))
    assert validation.status == "valid"
    assert any("may be too long" in w for w in validation.warnings)


def test_annotations_bounds_disjoint_rejected(seeded):
    repo, scene = seeded
    points, _ = repo.assets.points(repo.assets.get(scene.asset_id))
    entities = [
        dict(e.model_dump())
        for e in scene.entities
    ]
    entities[0]["bounds_asset"] = [[50, 50, 50], [51, 51, 51]]
    from src.tour.models import Entity, ViewPreset

    parsed_entities = [Entity(**e) for e in entities]
    parsed_views = [ViewPreset(**v.model_dump()) for v in scene.views]
    validation = validate_annotations(parsed_entities, parsed_views, points)
    assert validation.status == "invalid"
    assert any("do not intersect" in e for e in validation.errors)


def test_generation_precheck(seeded):
    repo, scene = seeded
    request = request_for(scene)
    assert generation_precheck(scene, request) == []
    # removing all views leaves nothing viewable
    scene2 = scene.model_copy(update={"views": []})
    assert any("no saved views" in e for e in generation_precheck(scene2, request))
