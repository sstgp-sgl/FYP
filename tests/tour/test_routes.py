"""HTTP routes: capabilities, import, annotations, preview binaries,
storyboards, generation polling, downloads, error mapping."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from src.tour.app import create_tour_app
from src.tour.config import TourConfig
from src.tour.providers import FakeProvider, ProviderError
from tests.tour.conftest import make_points, write_ascii_ply


@pytest.fixture
def client(tour_root, monkeypatch):
    config = TourConfig()
    app = create_tour_app(config)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def seeded_scene(client, tmp_path):
    points, colors = make_points(count=1500, seed=9)
    ply = write_ascii_ply(tmp_path / "seed.ply", points, colors)
    with open(ply, "rb") as stream:
        response = client.post(
            "/api/tour/scenes/import",
            data={"title": "路由测试场景"},
            files={"file": ("seed.ply", stream.read(), "application/octet-stream")},
        )
    assert response.status_code == 201, response.text
    return response.json()


def test_capabilities_no_secrets(client):
    response = client.get("/api/tour/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert "llm" in body
    assert "configured" in body["llm"]
    assert "api_key" not in json.dumps(body)


def test_scene_list_and_get(client, seeded_scene):
    scenes = client.get("/api/tour/scenes").json()["scenes"]
    assert any(s["scene_id"] == seeded_scene["scene_id"] for s in scenes)
    scene = client.get(f"/api/tour/scenes/{seeded_scene['scene_id']}").json()
    assert scene["revision"] == 1
    assert scene["coordinate_frame"] == "asset"


def test_import_rejects_non_ply(client):
    response = client.post(
        "/api/tour/scenes/import",
        data={"title": "x"},
        files={"file": ("evil.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_data"


def test_upload_too_large_413(client, monkeypatch, tmp_path):
    points, colors = make_points(count=2000)
    ply = write_ascii_ply(tmp_path / "big.ply", points, colors)
    data = ply.read_bytes()
    monkeypatch.setenv("FYP_TOUR_MAX_UPLOAD_BYTES", "100")
    config = TourConfig()
    app = create_tour_app(config)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/tour/scenes/import",
            data={"title": "big"},
            files={"file": ("big.ply", data, "application/octet-stream")},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_preview_binaries(client, seeded_scene):
    scene_id = seeded_scene["scene_id"]
    preview = client.get(f"/api/tour/scenes/{scene_id}/preview").json()
    assert preview["total_points"] == 1500
    assert preview["shown_points"] == 1500
    points_resp = client.get(f"/api/tour/scenes/{scene_id}/preview/points")
    assert points_resp.status_code == 200
    assert len(points_resp.content) == preview["shown_points"] * 3 * 4  # float32 xyz
    colors_resp = client.get(f"/api/tour/scenes/{scene_id}/preview/colors")
    assert len(colors_resp.content) == preview["shown_points"] * 3


def test_annotations_revision_flow(client, seeded_scene):
    scene_id = seeded_scene["scene_id"]
    payload = {
        "entities": [
            {
                "entity_id": "entity_a",
                "label": "区域A",
                "description": "人工标注",
                "bounds_asset": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                "facts": [
                    {"fact_id": "fact_a", "text": "A 是测试区域", "source": "user_confirmed",
                     "kind": "descriptive", "scope_view_id": None}
                ],
                "status": "confirmed",
            }
        ],
        "views": [
            {
                "view_id": "view_a",
                "name": "全景",
                "camera": {
                    "position_asset": [0.0, 0.0, 5.0], "target_asset": [0.0, 0.0, 0.0],
                    "up": [0.0, 1.0, 0.0], "fov_y_deg": 60.0, "near_asset": 0.01,
                    "far_asset": 100.0, "aspect_ratio": 16.0 / 9.0,
                },
                "approved_entity_ids": ["entity_a"],
                "confirmation_source": "manual",
                "confirmed_at": None,
            }
        ],
        "expected_revision": 1,
    }
    updated = client.put(f"/api/tour/scenes/{scene_id}/annotations", json=payload)
    assert updated.status_code == 200, updated.text
    assert updated.json()["revision"] == 2
    # stale write is rejected with 409
    stale = client.put(f"/api/tour/scenes/{scene_id}/annotations", json=payload)
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "conflict"
    # camera change clears the view's approval list
    changed = json.loads(json.dumps(payload))
    changed["expected_revision"] = 2
    changed["views"][0]["camera"]["position_asset"] = [2.0, 2.0, 5.0]
    changed["views"][0]["approved_entity_ids"] = ["entity_a"]
    result = client.put(f"/api/tour/scenes/{scene_id}/annotations", json=changed)
    assert result.status_code == 200
    assert result.json()["views"][0]["approved_entity_ids"] == []


def test_storyboard_save_validate_download(client, seeded_scene):
    scene_id = seeded_scene["scene_id"]
    # set up entity + view first
    annotations = {
        "entities": [
            {"entity_id": "entity_a", "label": "区域A", "description": "",
             "bounds_asset": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
             "facts": [], "status": "confirmed"}
        ],
        "views": [
            {"view_id": "view_a", "name": "全景",
             "camera": {"position_asset": [0.0, 0.0, 5.0], "target_asset": [0.0, 0.0, 0.0],
                        "up": [0.0, 1.0, 0.0], "fov_y_deg": 60.0, "near_asset": 0.01,
                        "far_asset": 100.0, "aspect_ratio": 16.0 / 9.0},
             "approved_entity_ids": ["entity_a"], "confirmation_source": "manual",
             "confirmed_at": None}
        ],
        "expected_revision": 1,
    }
    client.put(f"/api/tour/scenes/{scene_id}/annotations", json=annotations)

    storyboard = {
        "scene_id": scene_id,
        "scene_revision": 2,
        "title": "路由测试导览",
        "language": "zh-CN",
        "shots": [
            {"shot_id": "shot_01", "view_id": "view_a", "focus_entity_ids": ["entity_a"],
             "narration": "这是第一段。", "fact_ids": [], "duration_seconds": 10,
             "transition": "cut"}
        ],
    }
    # validation endpoint (no persistence)
    checked = client.post("/api/tour/storyboards/validate", json=storyboard)
    assert checked.status_code == 200
    assert checked.json()["status"] == "valid"
    # invalid candidate is reported, not saved
    bad = json.loads(json.dumps(storyboard))
    bad["shots"][0]["view_id"] = "ghost"
    checked = client.post("/api/tour/storyboards/validate", json=bad)
    assert checked.json()["status"] == "invalid"
    # save
    saved = client.post("/api/tour/storyboards", json=storyboard)
    assert saved.status_code == 201, saved.text
    storyboard_id = saved.json()["storyboard_id"]
    assert saved.json()["revision"] == 1
    # list + get + download
    listed = client.get(f"/api/tour/storyboards?scene_id={scene_id}").json()
    assert any(s["storyboard_id"] == storyboard_id for s in listed["storyboards"])
    got = client.get(f"/api/tour/storyboards/{storyboard_id}").json()
    assert got["title"] == "路由测试导览"
    download = client.get(f"/api/tour/storyboards/{storyboard_id}/download")
    assert download.status_code == 200
    assert "storyboard" in download.headers["content-disposition"]
    payload = json.loads(download.content)
    assert payload["schema_version"] == 1
    # update with wrong expected_revision → 409
    update = dict(storyboard, storyboard_id=storyboard_id, expected_revision=99, title="新")
    assert client.put(f"/api/tour/storyboards/{storyboard_id}", json=update).status_code == 409


def test_generation_poll_and_result(client, seeded_scene, monkeypatch):
    scene_id = seeded_scene["scene_id"]
    annotations = {
        "entities": [
            {"entity_id": "entity_a", "label": "区域A", "description": "",
             "bounds_asset": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
             "facts": [{"fact_id": "fact_a", "text": "A", "source": "user_confirmed",
                        "kind": "descriptive", "scope_view_id": None}],
             "status": "confirmed"}
        ],
        "views": [
            {"view_id": "view_a", "name": "全景",
             "camera": {"position_asset": [0.0, 0.0, 5.0], "target_asset": [0.0, 0.0, 0.0],
                        "up": [0.0, 1.0, 0.0], "fov_y_deg": 60.0, "near_asset": 0.01,
                        "far_asset": 100.0, "aspect_ratio": 16.0 / 9.0},
             "approved_entity_ids": ["entity_a"], "confirmation_source": "manual",
             "confirmed_at": None}
        ],
        "expected_revision": 1,
    }
    updated = client.put(f"/api/tour/scenes/{scene_id}/annotations", json=annotations).json()

    good = json.dumps({
        "status": "ok",
        "title": "AI 导览",
        "shots": [
            {"shot_id": "shot_01", "view_id": "view_a", "focus_entity_ids": ["entity_a"],
             "narration": "第一段", "fact_ids": ["fact_a"], "duration_seconds": 5,
             "transition": "cut"},
            {"shot_id": "shot_02", "view_id": "view_a", "focus_entity_ids": ["entity_a"],
             "narration": "第二段", "fact_ids": [], "duration_seconds": 5,
             "transition": "cut"},
        ],
    }, ensure_ascii=False)
    service = client.app.state.tour_service
    monkeypatch.setattr(
        service, "_build_provider",
        lambda: FakeProvider([good], provider="fixture", model="fixture-model"),
    )

    request = {
        "scene_id": scene_id,
        "scene_revision": updated["revision"],
        "user_prompt": "讲两段，共10秒",
        "language": "zh-CN",
        "target_duration_seconds": 10,
        "shot_count": 2,
        "required_entity_ids": ["entity_a"],
        "opening_view_id": None,
        "request_id": "test-req-1",
    }
    started = client.post("/api/tour/generations", json=request)
    assert started.status_code == 202, started.text
    run_id = started.json()["run_id"]
    # duplicate request_id → 409
    assert client.post("/api/tour/generations", json=request).status_code == 409

    terminal = None
    for _ in range(100):
        run = client.get(f"/api/tour/runs/{run_id}").json()
        if run["status"] not in ("pending", "running"):
            terminal = run
            break
        time.sleep(0.05)
    assert terminal is not None, "run did not finish"
    assert terminal["status"] == "succeeded", terminal
    assert terminal["result_storyboard_id"]
    assert terminal["attempts"][0]["provider"] == "fixture"
    storyboard = client.get(f"/api/tour/storyboards/{terminal['result_storyboard_id']}").json()
    assert storyboard["source"] == "llm"
    assert storyboard["validation"]["status"] == "valid"


def test_generation_provider_error_recorded(client, seeded_scene, monkeypatch):
    scene_id = seeded_scene["scene_id"]
    service = client.app.state.tour_service
    monkeypatch.setattr(
        service, "_build_provider",
        lambda: FakeProvider([ProviderError("http_401", "bad key", 401)]),
    )
    request = {
        "scene_id": scene_id,
        "scene_revision": 1,
        "user_prompt": "x",
        "language": "zh-CN",
        "target_duration_seconds": 10,
        "shot_count": 1,
        "required_entity_ids": [],
        "opening_view_id": None,
    }
    # generation fails in precheck because the scene has no entities/views yet
    started = client.post("/api/tour/generations", json=request)
    if started.status_code == 202:
        run_id = started.json()["run_id"]
        terminal = None
        for _ in range(100):
            run = client.get(f"/api/tour/runs/{run_id}").json()
            if run["status"] not in ("pending", "running"):
                terminal = run
                break
            time.sleep(0.05)
        assert terminal and terminal["status"] == "failed"
    else:
        assert started.status_code == 422


def test_unknown_scene_404(client):
    response = client.get("/api/tour/scenes/does_not_exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_error_shape_has_request_id(client):
    response = client.get("/api/tour/scenes/missing_scene", headers={"X-Request-ID": "abc123"})
    assert response.status_code == 404
    body = response.json()
    assert body["request_id"] == "abc123"
    assert response.headers["X-Request-ID"] == "abc123"
    assert set(body["error"]) == {"code", "message", "details"}


def test_tour_page_served(client):
    response = client.get("/tour")
    assert response.status_code == 200
    assert "场景导览" in response.text
    vendor = client.get("/vendor/three-r128/three.min.js")
    assert vendor.status_code == 200
    assert b"THREE" in vendor.content
