"""Planner: one-shot success, repair-once, double failure, clarification,
non-JSON responses and provider error mapping (httpx-free, fake provider)."""
import asyncio
import json

import pytest

from src.tour.importers import seed_demo
from src.tour.models import GenerationRequest
from src.tour.planner import build_context, plan_storyboard, parse_response
from src.tour.providers import FakeProvider, ProviderError
from src.tour.repository import TourRepository
from tests.tour.conftest import make_config


CONFIG = {
    "tour": {
        "max_upload_bytes": 104857600, "max_points": 2000000,
        "preview_points": 60000, "narration": {"zh_chars_per_second": 5.5},
    },
    "llm": {"max_output_tokens": 1500, "temperature": 0.2, "max_calls_per_run": 2},
}


def ok_response(scene, *, total=30, count=3, opening="view_overview", language="zh-CN"):
    views = {v.view_id: v for v in scene.views}
    order = ["view_overview", "view_building", "view_tree", "view_wall"]
    chosen = [views[v] for v in order[:count]]
    shots = []
    for index, view in enumerate(chosen, start=1):
        shots.append({
            "shot_id": f"shot_{index:02d}",
            "view_id": view.view_id,
            "focus_entity_ids": view.approved_entity_ids,
            "narration": f"第{index}段旁白。",
            "fact_ids": [],
            "duration_seconds": total // count,
            "transition": "cut",
        })
    return json.dumps({"status": "ok", "title": "测试导览", "shots": shots}, ensure_ascii=False)


@pytest.fixture
def seeded(tour_root):
    repository = TourRepository(tour_root)
    result = seed_demo(repository, make_config())
    return repository, repository.get_scene(result["scene_id"])


def request_for(scene, **overrides):
    data = dict(scene_id=scene.scene_id, scene_revision=scene.revision,
                user_prompt="先整体，再建筑，最后树丛", language="zh-CN",
                target_duration_seconds=30, shot_count=3,
                required_entity_ids=["entity_building", "entity_tree"],
                opening_view_id="view_overview")
    data.update(overrides)
    return GenerationRequest(**data)


def test_context_only_contains_allowed_material(seeded):
    repo, scene = seeded
    context = build_context(scene, request_for(scene))
    assert "entity_building" in context
    assert "view_overview" in context
    assert "fact_building_region" in context
    # no secrets or server paths
    assert "FYP_LLM_API_KEY" not in context
    assert "/root/" not in context


def test_parse_response_ok():
    response = parse_response(json.dumps({"status": "ok", "title": "t",
                                          "shots": [{"shot_id": "s1", "view_id": "v",
                                                     "focus_entity_ids": [], "narration": "x",
                                                     "duration_seconds": 10}]}))
    assert response.status == "ok"
    assert len(response.shots) == 1


def test_parse_response_code_fence():
    response = parse_response("```json\n" + json.dumps(
        {"status": "needs_clarification", "question": "哪个实体？"}) + "\n```")
    assert response.status == "needs_clarification"
    assert response.question == "哪个实体？"


def test_parse_response_non_json_raises():
    with pytest.raises(ProviderError):
        parse_response("not json at all")


def test_parse_response_broken_json_never_repaired():
    with pytest.raises(ProviderError):
        parse_response('{"status": "ok", "shots": [}')


def run(coro):
    return asyncio.run(coro)


def test_one_shot_success(seeded):
    repo, scene = seeded
    provider = FakeProvider([ok_response(scene)])
    outcome = run(plan_storyboard(scene, request_for(scene), provider, CONFIG))
    assert outcome.kind == "ok"
    assert outcome.storyboard.source == "llm"
    assert len(outcome.storyboard.shots) == 3
    assert sum(s.duration_seconds for s in outcome.storyboard.shots) == 30
    assert outcome.storyboard.shots[0].view_id == "view_overview"
    assert provider.calls == 1


def test_first_bad_then_repair(seeded):
    repo, scene = seeded
    bad = json.dumps({"status": "ok", "title": "t",
                      "shots": [{"shot_id": "s1", "view_id": "view_building",
                                 "focus_entity_ids": ["entity_tree"],  # invisible focus
                                 "narration": "x", "duration_seconds": 10},
                                {"shot_id": "s2", "view_id": "view_tree",
                                 "focus_entity_ids": ["entity_tree"], "narration": "y",
                                 "duration_seconds": 10},
                                {"shot_id": "s3", "view_id": "view_wall",
                                 "focus_entity_ids": ["entity_wall"], "narration": "z",
                                 "duration_seconds": 10}]}, ensure_ascii=False)
    good = ok_response(scene)
    provider = FakeProvider([bad, good])
    outcome = run(plan_storyboard(
        scene, request_for(scene, required_entity_ids=[]), provider, CONFIG))
    assert outcome.kind == "ok"
    assert provider.calls == 2
    assert len(outcome.attempts) == 2
    assert outcome.attempts[0].status == "validation_error"


def test_two_failures_fail(seeded):
    repo, scene = seeded
    bad = json.dumps({"status": "ok", "title": "t",
                      "shots": [{"shot_id": "s1", "view_id": "view_building",
                                 "focus_entity_ids": ["entity_tree"], "narration": "x",
                                 "duration_seconds": 10},
                                {"shot_id": "s2", "view_id": "view_tree",
                                 "focus_entity_ids": ["entity_tree"], "narration": "y",
                                 "duration_seconds": 10},
                                {"shot_id": "s3", "view_id": "view_wall",
                                 "focus_entity_ids": ["entity_wall"], "narration": "z",
                                 "duration_seconds": 10}]})
    provider = FakeProvider([bad, bad])
    outcome = run(plan_storyboard(
        scene, request_for(scene, required_entity_ids=[]), provider, CONFIG))
    assert outcome.kind == "failed"
    assert outcome.error_code == "validation_failed"
    assert provider.calls == 2


def test_clarification_terminal(seeded):
    repo, scene = seeded
    provider = FakeProvider([json.dumps({"status": "needs_clarification",
                                         "question": "先讲哪个实体？"}, ensure_ascii=False)])
    outcome = run(plan_storyboard(scene, request_for(scene), provider, CONFIG))
    assert outcome.kind == "clarification"
    assert outcome.question == "先讲哪个实体？"
    assert provider.calls == 1


def test_non_json_output_fails_with_parse_error(seeded):
    repo, scene = seeded
    provider = FakeProvider(["plain text"])
    outcome = run(plan_storyboard(scene, request_for(scene), provider, CONFIG))
    assert outcome.kind == "failed"
    assert outcome.error_code == "parse"


def test_provider_401_and_timeout(seeded):
    repo, scene = seeded
    provider = FakeProvider([ProviderError("http_401", "bad key", 401)])
    outcome = run(plan_storyboard(scene, request_for(scene), provider, CONFIG))
    assert outcome.kind == "failed"
    assert outcome.error_code == "http_401"
    provider = FakeProvider([ProviderError("timeout", "slow")])
    outcome = run(plan_storyboard(scene, request_for(scene), provider, CONFIG))
    assert outcome.kind == "failed"
    assert outcome.error_code == "timeout"


def test_precheck_blocks_generation(seeded):
    repo, scene = seeded
    stripped = scene.model_copy(update={"views": []})
    provider = FakeProvider([ok_response(scene)])
    outcome = run(plan_storyboard(stripped, request_for(stripped), provider, CONFIG))
    assert outcome.kind == "failed"
    assert outcome.error_code == "precheck_failed"
    assert provider.calls == 0
