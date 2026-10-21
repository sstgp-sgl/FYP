from dataclasses import asdict
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest

from src.pointcloud_io import write_ply
from src.scene.models import (
    Anchor, AssetRecord, ArtifactRef, Instance, PlacementConstraint, RunRecord,
    SceneDocument, Transform, Y_UP,
)
from src.scene.storage import (
    AssetRepository, SceneRepository, atomic_json, controlled, read_json, read_vertices, sha256,
)
from src.scene.composition import place, export
from src.scene.demo import demo
from src.scene.validation import validate_export, EPSILON


@pytest.fixture
def scene_setup(tmp_path):
    root = tmp_path / "managed"
    demo(root)
    assets = AssetRepository(root)
    return assets, SceneRepository(assets).load()


@pytest.mark.parametrize("kwargs", [
    {"uniform_scale": 0}, {"uniform_scale": -1}, {"uniform_scale": float("nan")},
    {"translation_xyz": [0, float("inf"), 0]}, {"rotation_xyzw": [0, 0, 0, 0]},
    {"rotation_xyzw": [0, 0, 0, float("nan")]}, {"schema_version": 2},
])
def test_transform_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        Transform(**kwargs)


def test_transform_convention():
    t = Transform([1, 2, 3], [0, 0, -2, -2], 2)
    assert np.allclose(t.rotation_xyzw, [0, 0, 2**-.5, 2**-.5])
    assert np.allclose(t.matrix() @ [1, 0, 0, 1], [1, 4, 3, 1])


def test_schema_and_references(scene_setup):
    assets, scene = scene_setup
    for model in [scene, scene.instances[0], scene.anchors[0], scene.constraints[0], assets.get(scene.instances[0].asset_id)]:
        data = asdict(model)
        data["schema_version"] = 100
        with pytest.raises(ValueError):
            type(model)(**data)
    data = asdict(scene)
    data["constraints"][0]["anchor_id"] = "missing"
    with pytest.raises(ValueError, match="unknown"):
        SceneDocument(**data)
    with pytest.raises(ValueError):
        SceneDocument("scene", [], [], reference_frame={"right": [0, 0, 0]})
    with pytest.raises(ValueError):
        SceneDocument("scene", [], [], reference_frame={"right": [1, 1, 0]})
    with pytest.raises(ValueError):
        PlacementConstraint("subject", "box", "left_of", float("nan"), 2)
    with pytest.raises(ValueError):
        PlacementConstraint("subject", "box", "left_of", 1, 0)
    with pytest.raises(ValueError):
        RunRecord("test", "cpu", "test", {}, status="cached")
    with pytest.raises(ValueError):
        ArtifactRef("out.ply", "0"*64, "point_cloud", schema_version=2)
    persisted = asdict(scene)
    del persisted["instances"][0]["transform"]["schema_version"]
    SceneRepository(assets).save(scene)
    document = assets.root / "scene.json"
    atomic_json(document, persisted)
    with pytest.raises(ValueError, match="schema_version"):
        SceneRepository(assets).load()


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside", "assets/../../outside"])
def test_path_traversal(tmp_path, relative):
    with pytest.raises(ValueError):
        controlled(tmp_path, relative)


def test_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        controlled(root, "escape/data.ply")


def test_hash_missing_and_external_copy(scene_setup):
    assets, scene = scene_setup
    a = assets.get(scene.instances[0].asset_id)
    path = assets.verify(a)
    original = path.read_bytes()
    # External fixture is not the copied managed asset.
    external = assets.root / "fixture/background.ply"
    assert external.read_bytes() == original and external != path
    external.write_text("changed external input")
    assert assets.verify(a) == path
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="hash"):
        SceneRepository(assets).load()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        SceneRepository(assets).load()


def test_unknown_asset_and_instance_hash(scene_setup):
    assets, scene = scene_setup
    scene.instances[0].asset_id = "missing"
    with pytest.raises(FileNotFoundError):
        SceneRepository(assets).save(scene)
    scene = SceneRepository(assets).load()
    scene.instances[0].asset_hash = "0"*64
    with pytest.raises(ValueError, match="hash"):
        SceneRepository(assets).save(scene)


def test_atomic_save_preserves_previous(scene_setup, monkeypatch):
    assets, scene = scene_setup
    path = assets.root / "scene.json"
    before = path.read_bytes()
    scene.updated_at = "changed"
    def fail(*args):
        raise OSError("injected failure before replace")
    monkeypatch.setattr("src.scene.storage.os.replace", fail)
    with pytest.raises(OSError):
        SceneRepository(assets).save(scene)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".pending-*"))


@pytest.mark.parametrize("right", [[1, 0, 0], [0, 0, 1], [3, 0, 4], [-2, 0, 1]])
@pytest.mark.parametrize("relation", ["left_of", "right_of"])
def test_layout_analytic_roundtrip_background(scene_setup, right, relation):
    assets, scene = scene_setup
    data = asdict(scene)
    data["reference_frame"] = {"right": right}
    scene = SceneDocument(**data)
    before = asdict(scene)
    c = PlacementConstraint("subject", "box", relation, .75, 2.25)
    result, checks = place(scene, c, assets)
    assert asdict(scene) == before
    assert checks["height"] == pytest.approx(2.25)
    assert checks["bottom_y"] == pytest.approx(0)
    assert checks["gap"] == pytest.approx(.75)
    assert asdict(result.instances[0]) == before["instances"][0]
    SceneRepository(assets).save(result)
    assert asdict(SceneRepository(assets).load()) == asdict(result)
    export(result, assets)
    validation = validate_export(assets)
    assert validation["passed"]
    assert validation["epsilon"] == EPSILON == 1e-4
    p, colors = read_vertices(assets.root / "combined.ply", "point_cloud")
    original, rgb = assets.points(assets.get(scene.instances[0].asset_id))
    assert np.array_equal(p[:len(original)], original)
    assert np.array_equal(colors[:len(rgb)], rgb)


def test_no_accumulation_multiple_instances(scene_setup):
    assets, scene = scene_setup
    c = PlacementConstraint("subject", "box", "left_of", .5, 1.5)
    first, _ = place(scene, c, assets)
    right, _ = place(first, PlacementConstraint("subject", "box", "right_of", 2, 3), assets)
    again, _ = place(right, c, assets)
    assert asdict(first.instances[1].transform) == asdict(again.instances[1].transform)
    a = scene.instances[1]
    scene.instances.append(Instance("copy", a.asset_id, a.asset_hash))
    result, _ = place(scene, PlacementConstraint("copy", "box", "right_of", .5, 1), assets)
    assert asdict(result.instances[1].transform) == asdict(a.transform)


def test_export_tampering_is_detected(scene_setup):
    assets, scene = scene_setup
    path = assets.root / "combined.ply"
    points, colors = read_vertices(path, "point_cloud")
    points[-8:, 1] += .1
    write_ply(points, colors, path)
    with pytest.raises(ValueError, match="hash"):
        validate_export(assets)
    # Even a rehashed export cannot bypass the numerical verifier.
    ranges_path = path.with_suffix(".ranges.json")
    index = read_json(ranges_path)
    index["artifact"]["sha256"] = sha256(path)
    atomic_json(ranges_path, index)
    assert not validate_export(assets)["passed"]


def test_empty_zero_height_nonfinite_assets(tmp_path):
    assets = AssetRepository(tmp_path / "managed")
    path = tmp_path / "bad.ply"
    for points in [np.empty((0, 3)), np.array([[float("nan"), 1, 2]])]:
        write_ply(points, np.zeros((len(points), 3), dtype=np.uint8), path)
        with pytest.raises(ValueError):
            assets.import_asset(path, role="subject", representation="point_cloud", source_kind="synthetic", producer="test", coordinate_frame=Y_UP)
    write_ply(np.array([[0, 1, 0], [1, 1, 1]]), np.zeros((2, 3), dtype=np.uint8), path)
    a = assets.import_asset(path, role="subject", representation="point_cloud", source_kind="synthetic", producer="test", coordinate_frame=Y_UP)
    scene = SceneDocument("zero", [Instance("subject", a.asset_id, a.sha256)], [Anchor("a", [[0, 0, 0], [1, 1, 1]])])
    with pytest.raises(ValueError, match="zero-height"):
        place(scene, PlacementConstraint("subject", "a", "left_of", 1, 1), assets)


def test_cpu_dependency_isolation(tmp_path):
    root = str(Path(__file__).resolve().parents[1])
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'sharp', 'transformers', 'diffusers', 'vggt'}:
            raise AssertionError('GPU import: ' + fullname)
sys.meta_path.insert(0, Block())
import subprocess
subprocess.Popen = lambda *a, **k: (_ for _ in ()).throw(AssertionError('unexpected process'))
from src.scene.demo import demo
assert demo(sys.argv[1])['passed']
"""
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "isolated")], cwd=root, check=True)
