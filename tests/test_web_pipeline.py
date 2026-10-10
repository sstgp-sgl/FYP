"""Regression coverage for Web entry points and six/seven-view reconstruction."""
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from src.config import load_config
from src.reconstruction import reconstruct_views
from src.server import create_app
from src.zero123 import generate_subject_views


@pytest.fixture
def cfg(tmp_path):
    config = load_config()
    config["device"] = "cpu"
    config["image_size"] = 32
    config["paths"]["subjects_dir"] = str(tmp_path / "subjects")
    config["paths"]["views_dir"] = str(tmp_path / "views")
    config["output_dir"] = str(tmp_path / "reconstruct")
    config["vggt"].update(conf_percentile=0, outlier_nb_neighbors=0)
    return config


class FakeZero123:
    def __call__(self, image, **kwargs):
        return SimpleNamespace(images=[Image.new("RGB", (640, 960), (80, 90, 100))])


def fake_vggt(paths, cfg):
    n = len(paths)
    return dict(points=np.ones((n, 3)), colors=np.full((n, 3), 80, dtype=np.uint8),
                conf=np.ones(n), extrinsic=np.zeros((n, 3, 4)),
                intrinsic=np.repeat(np.eye(3)[None], n, axis=0), image_paths=paths,
                use_point_map=True)


def test_switch_7_to_6_uses_exact_manifest_and_exports(cfg, tmp_path, monkeypatch):
    import src.vggt_reconstruct
    seen = []
    def infer(paths, config):
        seen.append([Path(p).name for p in paths])
        return fake_vggt(paths, config)
    monkeypatch.setattr(src.vggt_reconstruct, "run_vggt", infer)
    views = tmp_path / "views"
    for count in [7, 6]:
        cfg["views"]["include_original"] = count == 7
        generate_subject_views(Image.new("RGB", (40, 60)), FakeZero123(), cfg,
                               views, 42, "subject", {})
        # An unrelated image must never become an extra VGGT input.
        Image.new("RGB", (8, 8)).save(views / "view_99.png")
        stats = reconstruct_views(views, tmp_path / "out", cfg)
        assert len(seen[-1]) == count
        assert (views / "view_06.png").exists() == (count == 7)
        assert stats["num_views"] == count
        cameras = json.loads((tmp_path / "out/cameras.json").read_text())
        assert len(cameras["cameras"]) == count
        assert (tmp_path / "out/pointcloud.ply").exists()
        assert np.load(tmp_path / "out/predictions.npz")["image_paths"].shape == (count,)


def test_wrong_view_mode_fails_before_inference(cfg, tmp_path, monkeypatch):
    cfg["views"]["include_original"] = False
    generate_subject_views(Image.new("RGB", (32, 32)), FakeZero123(), cfg,
                           tmp_path / "views", 42, "subject", {})
    cfg["views"]["include_original"] = True
    with pytest.raises(ValueError, match="Expected 7"):
        reconstruct_views(tmp_path / "views", tmp_path / "out", cfg)


def prepare_subject(client, monkeypatch):
    import src.segment
    import src.server
    monkeypatch.setattr(src.segment, "build_sam2_predictor", lambda cfg: object())
    masks = np.zeros((3, 20, 20), dtype=bool)
    masks[:, 5:15, 5:15] = True
    monkeypatch.setattr(src.server, "segment_subject", lambda *a: (masks, np.array([.9, .8, .7])))
    buf = io.BytesIO()
    Image.new("RGB", (20, 20), (90, 80, 70)).save(buf, format="PNG")
    assert client.post("/api/upload", files={"file": ("sample.png", buf.getvalue(), "image/png")}).status_code == 200
    assert client.post("/api/segment", json={"points": [[10, 10]], "labels": [1]}).status_code == 200
    response = client.post("/api/export", json={"subjects": [{"name": "object", "points": [[10, 10]], "labels": [1]}]})
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("count", [6, 7])
def test_web_upload_sam_export_to_vggt_and_download(cfg, tmp_path, monkeypatch, count):
    import src.zero123
    import src.vggt_reconstruct
    def generate(doc, subjects_dir, views_root, config, progress):
        index = []
        for i, subject in enumerate(doc["subjects"]):
            folder = Path(views_root) / subject["name"]
            generate_subject_views(Image.open(Path(subjects_dir) / subject["subject_file"]),
                                   FakeZero123(), config, folder, 42, subject["name"], {})
            index.append({"name": subject["name"], "views_dir": str(folder)})
            progress(i, len(doc["subjects"]), "views", subject["name"], {"status": "done"})
        (Path(views_root) / "subjects_index.json").write_text(json.dumps({"subjects": index}))
    monkeypatch.setattr(src.zero123, "generate_views_for_subjects", generate)
    monkeypatch.setattr(src.vggt_reconstruct, "run_vggt", fake_vggt)
    with TestClient(create_app(cfg)) as client:
        assert client.get("/").status_code == 200
        assert "viewCount" in client.get("/").text
        assert client.get("/api/config").json()["view_count"] == 7
        assert client.post("/api/workflow/start").status_code == 400
        assert client.post("/api/workflow/start?view_count=8").status_code == 422
        prepare_subject(client, monkeypatch)
        result = client.post(f"/api/workflow/start?view_count={count}")
        assert result.json()["view_count"] == count
        for _ in range(100):
            state = client.get("/api/workflow/status").json()
            if state["status"] != "running":
                break
            time.sleep(.05)
        assert state["status"] == "done", state
        assert len(state["subjects"][0]["views"]) == count
        assert client.get("/api/pointcloud/object").json()["num_total"] == count
        assert client.get("/api/download/pointcloud/object").status_code == 200
        assert client.get("/outputs/views/object/view_00.png").status_code == 200
        assert client.get("/api/pointcloud/unknown").status_code == 404
        assert cfg["views"]["include_original"] is True  # request choice never mutates defaults


def test_model_failure_is_reported_as_error(cfg, monkeypatch):
    import src.zero123
    def fail(*args, **kwargs):
        raise RuntimeError("test model failure")
    monkeypatch.setattr(src.zero123, "generate_views_for_subjects", fail)
    with TestClient(create_app(cfg)) as client:
        prepare_subject(client, monkeypatch)
        client.post("/api/workflow/start?view_count=6")
        for _ in range(100):
            state = client.get("/api/workflow/status").json()
            if state["status"] != "running":
                break
            time.sleep(.05)
        assert state["status"] == "error"
        assert state["error"] == "test model failure"


def test_reconstruction_failure_propagates_to_web(cfg, monkeypatch):
    import src.reconstruction
    def fail(*args, **kwargs):
        raise RuntimeError("VGGT failed")
    monkeypatch.setattr(src.reconstruction, "reconstruct_views", fail)
    events = []
    with pytest.raises(RuntimeError, match="VGGT failed"):
        src.reconstruction.reconstruct_subjects({"subjects": [{"name": "object", "views_dir": "unused"}]},
                                         cfg["output_dir"], cfg,
                                         progress=lambda *args: events.append(args))
    assert events[-1][-1]["status"] == "error"
