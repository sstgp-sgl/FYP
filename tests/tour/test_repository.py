"""Repository: versioned writes, optimistic concurrency, asset hash guard,
path traversal, interrupted-run recovery."""
import json

import pytest

from src.tour.repository import (
    ConflictError,
    InvalidDataError,
    NotFoundError,
    RevisionConflictError,
    TourRepository,
)
from src.tour.importers import seed_demo
from src.tour.models import TourRun, GenerationRequest, now
from tests.tour.conftest import make_config


@pytest.fixture
def repo(tour_root):
    return TourRepository(tour_root)


@pytest.fixture
def seeded(repo):
    result = seed_demo(repo, make_config())
    return repo, repo.get_scene(result["scene_id"])


def test_scene_revisions_snapshot(seeded):
    repo, scene = seeded
    updated = scene.model_copy(update={"revision": 2, "title": "改名", "updated_at": now()})
    repo.update_scene(updated, expected_revision=1)
    old = repo.get_scene(scene.scene_id, revision=1)
    new = repo.get_scene(scene.scene_id)
    assert old.title != new.title
    assert new.revision == 2


def test_expected_revision_conflict(seeded):
    repo, scene = seeded
    updated = scene.model_copy(update={"revision": 2, "updated_at": now()})
    with pytest.raises(RevisionConflictError):
        repo.update_scene(updated, expected_revision=99)


def test_asset_hash_change_detected(seeded):
    repo, scene = seeded
    record = repo.assets.get(scene.asset_id)
    # corrupt the copied asset: hash check must fail on load
    data_path = repo.assets.verify(record)
    data_path.write_bytes(b"corrupted")
    with pytest.raises(Exception):
        repo.get_scene(scene.scene_id)


def test_unknown_scene_raises_not_found(repo):
    with pytest.raises(NotFoundError):
        repo.get_scene("missing")


def test_path_traversal_rejected(repo):
    with pytest.raises(ValueError):
        repo.get_scene("../escape")
    with pytest.raises(ValueError):
        repo.get_scene("a/../b")


def test_storyboard_revision_conflict(seeded):
    repo, scene = seeded
    storyboards = repo.list_storyboards(scene.scene_id)
    storyboard = repo.get_storyboard(storyboards[0]["storyboard_id"])
    updated = storyboard.model_copy(update={"revision": 2, "title": "新标题"})
    with pytest.raises(RevisionConflictError):
        repo.update_storyboard(updated, expected_revision=5)
    saved = repo.update_storyboard(updated, expected_revision=1)
    assert saved.revision == 2
    assert repo.get_storyboard(saved.storyboard_id, revision=1).title != "新标题"


def test_run_lifecycle_and_interrupted_recovery(repo, seeded):
    repo2, scene = seeded
    request = GenerationRequest(
        scene_id=scene.scene_id, scene_revision=1, user_prompt="test",
        language="zh-CN", target_duration_seconds=30, shot_count=3,
    )
    run = TourRun(run_id="run_test_01", status="pending", scene_id=scene.scene_id,
                  scene_revision=1, request=request, created_at=now())
    repo.create_run(run, request_id="req-1")
    # duplicate request_id
    with pytest.raises(ConflictError):
        repo.create_run(run.model_copy(update={"run_id": "run_test_02"}), request_id="req-1")
    # interrupted recovery
    recovered = repo.recover_interrupted()
    assert recovered == 1
    assert repo.get_run("run_test_01").status == "interrupted"


def test_run_request_id_lookup(repo, seeded):
    repo2, scene = seeded
    request = GenerationRequest(scene_id=scene.scene_id, scene_revision=1,
                                user_prompt="x", shot_count=2)
    run = TourRun(run_id="run_test_03", status="succeeded", scene_id=scene.scene_id,
                  scene_revision=1, request=request, created_at=now())
    repo.create_run(run, request_id="req-2")
    found = repo.run_for_request("req-2")
    assert found.run_id == "run_test_03"


def test_atomic_json_preserves_old_version_on_bad_write(repo):
    from src.tour.repository import atomic_json

    target = repo.root / "x" / "doc.json"
    atomic_json(target, {"v": 1})
    with pytest.raises(TypeError):
        atomic_json(target, {"v": {1, 2, 3}})  # sets are not JSON-serializable
    assert json.loads(target.read_text())["v"] == 1
