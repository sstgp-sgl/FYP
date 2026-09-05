"""Tests for the reconstruction task framework (dummy backend, no GPU/torch)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.reconstruct import Pipeline, TaskContext, available_tasks, get_task, load_config
from src.reconstruct.backends import get_backend
from src.pointcloud_io import read_ply, write_ply
from src.reconstruct.task import Task


def _write_images(dst: Path, n: int = 4, size: int = 32):
    dst.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = dst / f"cam{i}.png"
        arr = np.full((size, size, 3), 40 + i * 30, dtype=np.uint8)
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    return paths


def _cfg(tmp_path: Path, n: int = 4) -> dict:
    images = _write_images(tmp_path / "photos", n=n)
    return {
        "device": "cpu",
        "seed": 42,
        "backend": "dummy",
        "output_dir": str(tmp_path / "out"),
        "input": {"mode": "files", "files": images},
        "preprocess": {"enabled": True, "size": 32, "whiten_background": False},
        "vggt": {
            "model": "facebook/VGGT-1B",
            "conf_percentile": 40,
            "outlier_nb_neighbors": 5,
            "outlier_std_ratio": 2.0,
            "drop_fill_color": True,
            "fill_color_slack": 8,
        },
        "segment": {"background_gray": 255},
        "paths": {"vggt_repo": "/root/autodl-tmp/vggt"},
    }


def _ctx(tmp_path: Path, cfg: dict | None = None) -> TaskContext:
    cfg = cfg or _cfg(tmp_path)
    return TaskContext(cfg=cfg, workdir=Path(cfg["output_dir"]) / ".tasks")


# --- config ---

def test_load_config_layers_and_absolutizes():
    cfg = load_config()
    assert cfg["backend"] in ("vggt", "dummy")
    assert cfg["input"]["mode"] == "generated"
    assert Path(cfg["output_dir"]).is_absolute()
    assert Path(cfg["paths"]["views_dir"]).is_absolute()
    assert "visualize" in cfg and "mode" in cfg["visualize"]


# --- backends ---

def test_backend_registry():
    from src.reconstruct.backends.dummy import DummyBackend

    assert isinstance(get_backend("dummy", {}), DummyBackend)
    with pytest.raises(ValueError):
        get_backend("nope", {})


def test_dummy_backend_result_shapes(tmp_path: Path):
    cfg = _cfg(tmp_path)
    backend = get_backend("dummy", cfg)
    result = backend.reconstruct(cfg["input"]["files"], _ctx(tmp_path, cfg))
    assert result.points.ndim == 2 and result.points.shape[1] == 3
    assert result.colors.shape == result.points.shape
    assert result.conf.shape == (result.points.shape[0],)
    assert result.extrinsic.shape == (4, 3, 4)
    assert result.intrinsic.shape == (4, 3, 3)
    assert result.num_cameras == 4
    assert np.isfinite(result.points).all()


# --- pipeline ---

def test_pipeline_topological_order():
    pipeline = Pipeline([get_task(n) for n in
                         ("collect", "preprocess", "reconstruct", "filter", "export")])
    order = [t.name for t in pipeline.resolve()]
    assert order == ["collect", "preprocess", "reconstruct", "filter", "export"]


def test_pipeline_resolve_only_pulls_deps():
    pipeline = Pipeline([get_task(n) for n in
                         ("collect", "preprocess", "reconstruct", "filter", "export")])
    order = [t.name for t in pipeline.resolve(only=["filter"])]
    assert order == ["collect", "preprocess", "reconstruct", "filter"]


def test_pipeline_end_to_end_dummy(tmp_path: Path):
    cfg = _cfg(tmp_path)
    ctx = _ctx(tmp_path, cfg)
    pipeline = Pipeline([get_task(n) for n in
                         ("collect", "preprocess", "reconstruct", "filter", "export")])
    summary = pipeline.run(ctx)

    assert [r["status"] for r in summary["tasks"]] == ["done"] * 5
    out = Path(cfg["output_dir"])
    assert (out / "pointcloud.ply").exists()
    assert (out / "cameras.json").exists()
    assert (out / "stats.json").exists()

    points, colors = read_ply(out / "pointcloud.ply")
    assert points.ndim == 2 and points.shape[1] == 3
    assert colors.shape == points.shape
    assert len(points) > 0

    cams = json.loads((out / "cameras.json").read_text())
    assert len(cams["cameras"]) == 4


def test_pipeline_skips_cached_outputs(tmp_path: Path):
    cfg = _cfg(tmp_path)
    pipeline = Pipeline([get_task(n) for n in
                         ("collect", "preprocess", "reconstruct", "filter", "export")])

    pipeline.run(_ctx(tmp_path, cfg))
    second = pipeline.run(_ctx(tmp_path, cfg))
    assert all(r["status"] == "skipped" for r in second["tasks"])


def test_skip_load_populates_artifacts(tmp_path: Path):
    cfg = _cfg(tmp_path)
    pipeline = Pipeline([get_task(n) for n in
                         ("collect", "preprocess", "reconstruct", "filter", "export")])
    pipeline.run(_ctx(tmp_path, cfg))  # full run, populates disk outputs

    # Fresh ctx (no in-memory artifacts); re-run only export. Skipped deps must
    # restore their artifacts from disk so export can run without KeyError.
    ctx2 = _ctx(tmp_path, cfg)
    summary = pipeline.run(ctx2, only=["export"])
    assert all(r["status"] == "skipped" for r in summary["tasks"])
    assert ctx2.artifacts["preprocess"]["image_paths"]
    assert ctx2.artifacts["reconstruct"]["result"].num_points > 0
    assert ctx2.artifacts["export"]["pointcloud_ply"]


def test_pipeline_force_on_fresh_workdir(tmp_path: Path):
    """force=True skips is_complete(), so stage dirs must still be created."""
    cfg = _cfg(tmp_path)
    ctx = TaskContext(cfg=cfg, workdir=Path(cfg["output_dir"]) / ".tasks", force=True)
    pipeline = Pipeline([get_task(n) for n in
                         ("collect", "preprocess", "reconstruct", "filter", "export")])
    summary = pipeline.run(ctx)
    assert all(r["status"] == "done" for r in summary["tasks"])
    assert (Path(cfg["output_dir"]) / ".tasks" / "preprocess" / "view_00.png").exists()
    assert (Path(cfg["output_dir"]) / "pointcloud.ply").exists()


def test_pipeline_force_reruns(tmp_path: Path):
    cfg = _cfg(tmp_path)
    pipeline = Pipeline([get_task(n) for n in
                         ("collect", "preprocess", "reconstruct", "filter", "export")])
    pipeline.run(_ctx(tmp_path, cfg))
    ctx_force = TaskContext(cfg=cfg, workdir=Path(cfg["output_dir"]) / ".tasks", force=True)
    second = pipeline.run(ctx_force)
    assert all(r["status"] == "done" for r in second["tasks"])


# --- collect edge cases ---

def test_collect_requires_at_least_two_images(tmp_path: Path):
    cfg = _cfg(tmp_path, n=1)
    with pytest.raises(ValueError):
        get_task("collect").run(_ctx(tmp_path, cfg))


def test_collect_folder_mode(tmp_path: Path):
    cfg = _cfg(tmp_path, n=4)
    cfg["input"] = {"mode": "folder", "folder": str(tmp_path / "photos"), "glob": "*"}
    result = get_task("collect").run(_ctx(tmp_path, cfg))
    assert len(result["image_paths"]) == 4


# --- pointcloud_io round trip ---

def test_ply_round_trip(tmp_path: Path):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    p = tmp_path / "p.ply"
    write_ply(points, colors, p)
    p2, c2 = read_ply(p)
    assert np.allclose(p2, points)
    assert np.array_equal(c2, colors)


def test_all_expected_tasks_registered():
    names = set(available_tasks())
    assert {"collect", "preprocess", "reconstruct", "filter", "export", "visualize"} <= names


def test_pipeline_reruns_when_source_images_change(tmp_path: Path):
    """Regenerated views (same paths, newer pixels) must invalidate the cache."""
    cfg = _cfg(tmp_path)
    pipeline = Pipeline([get_task(n) for n in
                         ("collect", "preprocess", "reconstruct", "filter", "export")])
    pipeline.run(_ctx(tmp_path, cfg))

    src = Path(cfg["input"]["files"][0])
    Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8)).save(src)

    second = pipeline.run(_ctx(tmp_path, cfg))
    by_name = {r["task"]: r["status"] for r in second["tasks"]}
    assert by_name["collect"] == "done"
    assert by_name["preprocess"] == "done"
    assert by_name["reconstruct"] == "done"


def test_pipeline_reruns_when_a_seventh_view_appears(tmp_path: Path):
    cfg = _cfg(tmp_path, n=6)
    pipeline = Pipeline([get_task(n) for n in
                         ("collect", "preprocess", "reconstruct", "filter", "export")])
    pipeline.run(_ctx(tmp_path, cfg))

    extra = tmp_path / "photos" / "cam6.png"
    Image.fromarray(np.full((32, 32, 3), 220, dtype=np.uint8)).save(extra)
    cfg["input"]["files"].append(str(extra))

    second = pipeline.run(_ctx(tmp_path, cfg))
    by_name = {r["task"]: r["status"] for r in second["tasks"]}
    assert by_name["collect"] == "done"
    assert by_name["reconstruct"] == "done"
    cams = json.loads((Path(cfg["output_dir"]) / "cameras.json").read_text())
    assert len(cams["cameras"]) == 7
