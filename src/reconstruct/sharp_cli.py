"""Optional official SHARP CLI, isolated from the VGGT Python environment.

No model imports. The Linux advisory lock covers this adapter, not external GPU
jobs. The child inherits it, so an orphaned inference retains the lock.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid

from PIL import Image

from src.scene.models import ArtifactRef, RunRecord, now, number, version
from src.scene.storage import AssetRepository, atomic_json, controlled, read_json, read_vertices, require_versioned, sha256

ENCODING = "sharp_ply_log_scale_logit_opacity_quaternion_wxyz_v1"


@dataclass
class SharpConfig:
    executable: str = "sharp"
    checkpoint_path: str | None = None
    allow_download: bool = False
    timeout_seconds: float = 300.
    device: str | None = None
    output_root: str = "outputs/phase1"
    schema_version: int = 1

    def __post_init__(self):
        version(self.schema_version)
        if not isinstance(self.executable, str) or not self.executable:
            raise ValueError("missing executable")
        if not Path(self.executable).is_absolute() and not re.fullmatch(r"[A-Za-z0-9_.-]+", self.executable):
            raise ValueError("executable must be an absolute path or command name, not a shell string")
        if type(self.allow_download) is not bool or number(self.timeout_seconds) <= 0:
            raise ValueError("invalid timeout/download policy")
        if self.device not in {None, "cpu", "cuda", "mps"}:
            raise ValueError("unsupported device")


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=.5)
    except subprocess.TimeoutExpired:
        pass
    # Kill remaining descendants even when the direct parent exited promptly.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_logged(argv, log, timeout, lock_fd=None):
    with open(log, "wb") as stream:
        p = subprocess.Popen(
            argv, shell=False, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=() if lock_fd is None else (lock_fd,),
        )
        try:
            return p.wait(timeout=timeout)
        except BaseException:
            stop_group(p)
            raise


@contextmanager
def inference_lock():
    # Stable per-server-user location, independent of per-run output roots.
    path = Path(tempfile.gettempdir()) / f"fyp-sharp-{os.getuid()}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("sharp_busy: another adapter inference is active") from None
        yield fd
    finally:
        os.close(fd)


class SharpCliBackend:
    _run_source = "real"  # Tests monkeypatch this; not configurable by CLI.

    def __init__(self, config):
        self.config = SharpConfig(**asdict(config))

    def diagnose(self):
        executable = shutil.which(self.config.executable)
        result = {
            "schema_version": 1, "backend": "official_sharp_cli",
            "available": False, "executable": executable, "software_version": "unknown",
            "source_commit": "unknown", "encoding": ENCODING, "problems": [],
            "inference_run": False,
        }
        if executable is None:
            result["problems"].append("SHARP executable not found; use a separate SHARP environment")
            return result
        result["executable_sha256"] = sha256(executable)
        with tempfile.TemporaryDirectory(prefix="sharp-diagnose-") as directory:
            for name, args in (("help", ["--help"]), ("predict_help", ["predict", "--help"])):
                log = Path(directory) / name
                try:
                    code = run_logged([executable, *args], log, min(30., self.config.timeout_seconds))
                    with open(log, "rb") as stream:
                        text = stream.read(65536).decode(errors="replace")
                    result[name] = text
                    result[name + "_sha256"] = sha256(log)
                    if code != 0:
                        result["problems"].append(f"{name} exited {code}")
                except (OSError, subprocess.TimeoutExpired) as exc:
                    result["problems"].append(f"{name}: {type(exc).__name__}")
        help_text = result.get("predict_help", "")
        # Exact flag tokens, not substrings of unrelated words.
        flags = set(re.findall(r"(?<![\w-])--?[a-zA-Z][\w-]*", help_text))
        required = {"-i", "-o", "-c"}
        if not required <= flags:
            result["problems"].append("installed predict help does not expose -i/-o/-c")
        if self.config.device is not None and "--device" not in flags:
            result["problems"].append("installed CLI does not support --device")
        result["available"] = not result["problems"]
        result["checkpoint_ready"] = bool(self.config.checkpoint_path and Path(self.config.checkpoint_path).is_file())
        result["allow_download"] = self.config.allow_download
        return result

    def _recover(self, root):
        # Called only while holding the shared inference lock.
        recovered = []
        runs = controlled(root, "runs")
        if runs.exists():
            for path in sorted(runs.glob("*/run.json")):
                path = controlled(root, str(path.relative_to(root)))
                record_data = require_versioned(read_json(path))
                for artifact in record_data.get("artifacts", []):
                    require_versioned(artifact)
                record = RunRecord(**record_data)
                if record.status in {"pending", "running"}:
                    record.status = "interrupted"
                    record.finished_at = now()
                    record.error = {"type": "orphaned_run", "message": "previous adapter exited without final state"}
                    atomic_json(path, asdict(record))
                    recovered.append(record.run_id)
        return recovered

    def recover(self):
        root = Path(self.config.output_root).resolve()
        with inference_lock():
            return self._recover(root)

    def reconstruct(self, image, *, role="scene"):
        cfg = self.config
        image = Path(image).resolve(strict=True)
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("phase1 supports one RGB PNG/JPEG only")
        with Image.open(image) as im:
            im.load()
            if im.mode != "RGB" or "transparency" in im.info or getattr(im, "n_frames", 1) != 1:
                raise ValueError("one opaque RGB image required; no automatic segmentation/conversion")
        if role not in {"scene", "subject"}:
            raise ValueError("invalid asset role")
        checkpoint = Path(cfg.checkpoint_path).resolve(strict=True) if cfg.checkpoint_path else None
        if checkpoint is None and not cfg.allow_download:
            raise ValueError("checkpoint required: automatic downloads disabled")
        if checkpoint is not None and (not checkpoint.is_file() or checkpoint.stat().st_size == 0):
            raise ValueError("invalid checkpoint file")
        diagnosis = self.diagnose()
        if not diagnosis["available"]:
            raise RuntimeError("; ".join(diagnosis["problems"]))
        root = Path(cfg.output_root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        with inference_lock() as lock_fd:
            self._recover(root)
            run_id = uuid.uuid4().hex
            directory = controlled(root, f"runs/{run_id}")
            directory.mkdir(parents=True, exist_ok=False)
            inputs, outputs = directory / "input", directory / "output"
            inputs.mkdir()
            outputs.mkdir()
            target = inputs / ("image" + image.suffix.lower())
            shutil.copyfile(image, target)
            # Revalidate frozen bytes, preserving original image/EXIF without re-encoding.
            with Image.open(target) as im:
                im.load()
                if im.mode != "RGB" or "transparency" in im.info:
                    raise ValueError("image changed during import")
            checkpoint_hash = sha256(checkpoint) if checkpoint else None
            argv = [diagnosis["executable"], "predict", "-i", str(inputs), "-o", str(outputs)]
            if checkpoint:
                argv.extend(["-c", str(checkpoint)])
            if cfg.device:
                argv.extend(["--device", cfg.device])
            log = directory / "process.log"
            record_path = directory / "run.json"
            record = RunRecord(
                run_id, "official_sharp_cli", self._run_source,
                {"config": asdict(cfg), "argv": argv, "input_sha256": sha256(target),
                 "checkpoint_sha256": checkpoint_hash, "diagnosis": diagnosis,
                 "gpu_measurement": None, "role": role},
                logs=[str(log.relative_to(root))],
            )
            atomic_json(record_path, asdict(record))
            started = time.monotonic()
            try:
                record.status, record.started_at = "running", now()
                atomic_json(record_path, asdict(record))
                record.exit_code = run_logged(argv, log, cfg.timeout_seconds, lock_fd)
                if record.exit_code != 0:
                    raise RuntimeError(f"sharp_exit_code_{record.exit_code}")
                if checkpoint and sha256(checkpoint) != checkpoint_hash:
                    raise ValueError("checkpoint changed during inference")
                candidates = list(outputs.glob("*.ply"))
                if len(candidates) != 1:
                    raise ValueError("expected exactly one fresh Gaussian PLY in this run")
                candidate = controlled(outputs, candidates[0].name)
                if not candidate.is_file() or candidate.stat().st_size == 0:
                    raise ValueError("missing/empty output")
                points, _ = read_vertices(candidate, "gaussian_splat")
                assets = AssetRepository(root)
                asset = assets.import_asset(
                    candidate, role=role, representation="gaussian_splat",
                    source_kind="real" if self._run_source == "real" else "synthetic", producer="official_sharp_cli",
                    coordinate_frame="opencv_camera", unit="estimated_metric",
                    source_image_hash=record.parameters["input_sha256"],
                    provenance={"schema_version": 1, "run_id": run_id, "source": self._run_source, "encoding": ENCODING,
                                "checkpoint_sha256": checkpoint_hash, "software_version": diagnosis["software_version"],
                                "source_commit": diagnosis["source_commit"], "surface_geometry": False},
                )
                record.artifacts = [asdict(ArtifactRef(asset.relative_path, asset.sha256, "gaussian_splat"))]
                record.parameters["asset_id"] = asset.asset_id
                record.parameters["validated_gaussian_count"] = len(points)
                record.status = "succeeded"
            except BaseException as exc:
                record.status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
                record.error = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                record.finished_at = now()
                record.duration_seconds = time.monotonic()-started
                atomic_json(record_path, asdict(record))
            return record
