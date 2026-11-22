"""CPU isolation: the tour package and the tour-only server must never import
GPU/model libraries (torch, fcntl, sharp_cli, SAM2, VGGT…), and the tour-only
server must serve seed/read/preview without them."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Note: fcntl is NOT banned here — the stdlib asyncio.unix_events imports it on
# Linux whenever FastAPI's loop starts. The real isolation targets are GPU /
# model libraries and the old reconstruction chain.
BANNED = ("torch", "diffusers", "transformers", "cv2", "viser", "safetensors")


def test_tour_package_does_not_import_gpu_libraries():
    script = (
        "import sys\n"
        "import src.tour.app, src.tour.service, src.tour.planner, src.tour.importers\n"
        "bad = [m for m in sorted(sys.modules) if m.split('.')[0] in %r]\n"
        "print('BANNED', bad)\n"
        "assert not bad, bad\n"
        "assert 'src.reconstruct' not in sys.modules\n"
        "assert 'src.server' not in sys.modules\n"
        "print('CLEAN')\n" % (BANNED,)
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(PROJECT_ROOT), env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CLEAN" in result.stdout


def _free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_tour_only_server_seed_read_play(tmp_path):
    output_root = tmp_path / "tours"
    port = _free_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["FYP_TOUR_OUTPUT_ROOT"] = str(output_root)

    # seed a demo scene via the CLI
    seed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "40_tour.py"), "seed-demo"],
        cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert seed.returncode == 0, seed.stdout + seed.stderr

    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "app.py"), "--tour-only",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(PROJECT_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        capabilities = None
        for _ in range(100):
            try:
                with httpx.Client(timeout=2.0) as client:
                    capabilities = client.get(f"{base}/api/tour/capabilities").json()
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        assert capabilities is not None, "tour-only server did not start"
        assert capabilities["tour"]["max_points"] > 0
        with httpx.Client(timeout=5.0) as client:
            scenes = client.get(f"{base}/api/tour/scenes").json()["scenes"]
            assert any(s["scene_id"] == "synthetic_courtyard" for s in scenes)
            scene = client.get(f"{base}/api/tour/scenes/synthetic_courtyard").json()
            assert scene["source_kind"] == "synthetic"
            preview = client.get(
                f"{base}/api/tour/scenes/synthetic_courtyard/preview").json()
            points = client.get(
                f"{base}/api/tour/scenes/synthetic_courtyard/preview/points")
            assert len(points.content) == preview["shown_points"] * 3 * 4
            storyboards = client.get(f"{base}/api/tour/storyboards").json()
            assert storyboards["storyboards"], "seed storyboard missing"
            # the old root page must NOT be served by the tour-only app
            root = client.get(f"{base}/")
            assert root.status_code == 404
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_old_entry_still_imports_server_on_demand():
    # app.py must not import src.server at module top level when --tour-only is
    # used; the import must sit inside the non-tour branch only.
    source = (PROJECT_ROOT / "app.py").read_text()
    assert "--tour-only" in source
    for line in source.splitlines():
        if "from src.server import main" in line:
            assert line.startswith((" ", "\t")), "src.server import must be lazy"
    for line in source.splitlines():
        if "from src.tour.app import tour_main" in line:
            assert line.startswith((" ", "\t")), "src.tour import must be lazy"
