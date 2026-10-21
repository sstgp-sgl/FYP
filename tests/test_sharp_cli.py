from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import pytest

from src.reconstruct.sharp_cli import SharpCliBackend, SharpConfig, inference_lock
from src.scene.composition import export
from src.scene.models import Instance, RunRecord, SceneDocument, Y_UP
from src.scene.storage import AssetRepository, GAUSSIAN_FIELDS, atomic_json, read_json, read_vertices


def gaussian_bytes():
    header = "ply\nformat ascii 1.0\nelement vertex 1\n"
    header += "".join(f"property float {name}\n" for name in GAUSSIAN_FIELDS)
    # x y z, SH0 RGB, opacity logit, log-scales, quaternion wxyz.
    return header + "end_header\n0 0 1 0 0 0 -2 -1 -1 -1 1 0 0 0\n"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    executable = tmp_path / "fake sharp"
    # Fake executable exists only in pytest's temporary folder.
    script = f"""#!{sys.executable}
import json, os, pathlib, subprocess, sys, time
if '--help' in sys.argv:
    print('predict -i --input-path -o --output-path -c --checkpoint-path --device')
    sys.exit(0)
argv = sys.argv[1:]
out = pathlib.Path(argv[argv.index('-o')+1])
inputs = pathlib.Path(argv[argv.index('-i')+1])
(out / 'argv.json').write_text(json.dumps(argv))
assert len(list(inputs.iterdir())) == 1
mode = os.environ.get('FYP_TEST_SHARP_MODE', 'success')
if mode == 'nonzero':
    print('test error', flush=True)
    sys.exit(7)
if mode == 'timeout':
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    (out / 'child.pid').write_text(str(child.pid))
    time.sleep(60)
if mode == 'missing':
    sys.exit(0)
if mode == 'wrong':
    (out / 'image.ply').write_text('ply\\nformat ascii 1.0\\nelement vertex 1\\nproperty float x\\nproperty float y\\nproperty float z\\nproperty uchar red\\nproperty uchar green\\nproperty uchar blue\\nend_header\\n0 0 0 1 2 3\\n')
else:
    (out / 'image.ply').write_text({gaussian_bytes()!r})
"""
    executable.write_text(script)
    executable.chmod(0o755)
    image = tmp_path / "image with spaces.png"
    Image.new("RGB", (8, 8), (120, 40, 30)).save(image)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake-checkpoint-not-a-model")
    config = SharpConfig(str(executable), str(checkpoint), output_root=str(tmp_path / "outputs"), timeout_seconds=3)
    backend = SharpCliBackend(config)
    monkeypatch.setattr(backend, "_run_source", "test")
    return backend, image


def test_fake_success_argv_and_gaussian_contract(fake):
    backend, image = fake
    diagnosis = backend.diagnose()
    assert diagnosis["available"] and not diagnosis["inference_run"]
    result = backend.reconstruct(image)
    assert result.status == "succeeded" and result.source == "test"
    root = Path(backend.config.output_root)
    saved = read_json(root / "runs" / result.run_id / "run.json")
    assert saved["status"] == "succeeded"
    assert saved["parameters"]["checkpoint_sha256"]
    assert saved["gpu_peak_memory_bytes"] is None
    argv = saved["parameters"]["argv"]
    assert "--render" not in argv and "-c" in argv
    assert Path(argv[argv.index("-i")+1]).parent == root / "runs" / result.run_id
    assets = AssetRepository(root)
    record = assets.get(saved["parameters"]["asset_id"])
    assert record.representation == "gaussian_splat"
    assert record.coordinate_frame == "opencv_camera"
    assert record.source_kind == "synthetic" and record.provenance["source"] == "test"
    assert record.bounds is None
    assert assets.verify(record).read_text() == gaussian_bytes()
    with pytest.raises(ValueError):
        read_vertices(assets.verify(record), "point_cloud")
    # Confirm unsupported representation explicitly, even if manually aligned.
    record.coordinate_frame = Y_UP
    atomic_json(root / "assets" / record.asset_id / "asset.json", asdict(record))
    scene = SceneDocument("g", [Instance("g", record.asset_id, record.sha256)], [], unit=record.unit)
    with pytest.raises(ValueError, match="unsupported_representation"):
        export(scene, assets)


@pytest.mark.parametrize("mode", ["nonzero", "missing", "wrong", "timeout"])
def test_fake_failures_old_output_isolation(fake, monkeypatch, mode):
    backend, image = fake
    first = backend.reconstruct(image)
    monkeypatch.setenv("FYP_TEST_SHARP_MODE", mode)
    backend.config.timeout_seconds = .7 if mode == "timeout" else 3
    with pytest.raises((RuntimeError, ValueError, subprocess.TimeoutExpired)):
        backend.reconstruct(image)
    root = Path(backend.config.output_root)
    runs = [read_json(p) for p in (root / "runs").glob("*/run.json")]
    failed = next(r for r in runs if r["run_id"] != first.run_id)
    assert failed["status"] == "failed" and not failed["artifacts"]
    assert failed["source"] == "test" and failed["error"]
    assert (root / failed["logs"][0]).exists()
    if mode == "nonzero":
        assert failed["exit_code"] == 7
    if mode == "timeout":
        pid = int((root / "runs" / failed["run_id"] / "output/child.pid").read_text())
        # A killed child can briefly remain a zombie until init reaps it.
        proc = Path(f"/proc/{pid}/stat")
        for _ in range(20):
            if not proc.exists() or proc.read_text().split()[2] == "Z":
                break
            time.sleep(.05)
        assert not proc.exists() or proc.read_text().split()[2] == "Z"


def test_lock_and_orphan_recovery(fake):
    backend, image = fake
    root = Path(backend.config.output_root)
    orphan = RunRecord("orphan", "official_sharp_cli", "test", {}, status="running")
    path = root / "runs/orphan/run.json"
    atomic_json(path, asdict(orphan))
    with inference_lock():
        with pytest.raises(RuntimeError, match="sharp_busy"):
            backend.recover()
        assert read_json(path)["status"] == "running"
    assert backend.recover() == ["orphan"]
    assert read_json(path)["status"] == "interrupted"
    assert backend.recover() == []


def test_input_policy_download_and_flag_validation(fake, monkeypatch):
    backend, image = fake
    backend.config.checkpoint_path = None
    with pytest.raises(ValueError, match="downloads disabled"):
        backend.reconstruct(image)
    Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(image)
    with pytest.raises(ValueError, match="RGB"):
        backend.reconstruct(image)
    with pytest.raises(ValueError):
        SharpConfig("sharp predict --render")
    backend.config.device = "cuda"
    executable = Path(backend.config.executable)
    executable.write_text(f"#!{sys.executable}\nprint('predict -i -o -c')\n")
    assert not backend.diagnose()["available"]


def test_gaussian_invalid_encoding(tmp_path):
    path = tmp_path / "g.ply"
    path.write_text(gaussian_bytes())
    read_vertices(path, "gaussian_splat")  # negative stored scale/opacity are valid
    for tail in [
        "0 0 1 0 0 0 -2 -1 -1 -1 0 0 0 0",
        "nan 0 1 0 0 0 -2 -1 -1 -1 1 0 0 0",
        "0 0 1 0 0 0 -2 1000 -1 -1 1 0 0 0",
    ]:
        path.write_text(gaussian_bytes().rsplit("\n", 2)[0] + "\n" + tail + "\n")
        with pytest.raises(ValueError):
            read_vertices(path, "gaussian_splat")
