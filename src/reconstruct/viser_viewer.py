"""Viser-based point-cloud inspection, borrowed from ``vggt/run_viser_multi.py``.

Two modes:

* ``predictions`` — re-runs VGGT and opens the official ``demo_viser`` UI
  (per-frame point clouds, camera frustums, confidence slider). This mirrors
  VGGT's ``run_viser_multi.py`` / `scripts/20_reconstruct.py visualize --mode predictions`.
* ``ply`` — a lightweight scene that shows an exported ``pointcloud.ply`` plus
  camera frames from ``cameras.json``, with no re-inference (used by `scripts/21_view_ply.py`).

Viser and VGGT are imported lazily so the rest of the framework does not
require them.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np


def _require_viser():
    try:
        import viser  # noqa: F401

        return viser
    except ImportError as e:
        raise RuntimeError(
            "viser is not installed; run `pip install viser` to use visualization"
        ) from e


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z], dtype=np.float64)


def _load_colored_ply(ply_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (points, colors) from an ASCII or binary PLY.

    The stage-3 pipeline writes binary PLY via trimesh; the framework itself
    writes ASCII. Try the dependency-free ASCII reader first, then fall back to
    trimesh when it is installed.
    """
    from src.pointcloud_io import read_ply

    try:
        return read_ply(ply_path)
    except (UnicodeDecodeError, ValueError):
        pass

    import trimesh

    mesh = trimesh.load(str(ply_path), process=False)
    points = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
    return points, colors


def visualize_ply(
    ply_path: str | Path,
    cameras_json: str | Path | None = None,
    port: int = 8080,
) -> None:
    """Show an exported colored point cloud (+ camera frames) in Viser."""
    viser = _require_viser()

    points, colors = _load_colored_ply(ply_path)
    if len(points) == 0:
        raise ValueError(f"empty point cloud: {ply_path}")

    server = viser.ViserServer(port=port)
    # Normalize colors to float [0,1] if needed.
    cols = colors.astype(np.float32)
    if cols.max() > 1.0:
        cols /= 255.0
    server.scene.add_point_cloud(
        "reconstruction/points",
        points=points,
        colors=cols,
        point_size=0.004,
        point_shape="circle",
    )

    cams = None
    if cameras_json is not None and Path(cameras_json).exists():
        import json

        cams = json.loads(Path(cameras_json).read_text())["cameras"]
        for i, cam in enumerate(cams):
            E = np.asarray(cam["extrinsic"], dtype=np.float64)  # world→cam [R|t]
            R, t = E[:3, :3], E[:3, 3]
            center = -R.T @ t
            c2w_R = R.T
            server.scene.add_frame(
                f"cameras/{i:02d}",
                wxyz=_rotmat_to_quat_wxyz(c2w_R),
                position=center,
                axes_length=0.15,
                axes_radius=0.008,
            )

    print("=" * 60)
    print(f"   📡  Viser: http://localhost:{port}")
    print(f"   🧊  points: {len(points):,}  cameras: {len(cams) if cams else 0}")
    print("=" * 60)
    print("   Ctrl+C to exit\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopping Viser")


def visualize_predictions(
    image_paths: list[str],
    cfg: dict,
    port: int = 8080,
    init_conf_threshold: float = 25.0,
    use_point_map: bool = False,
    drop_white: bool = True,
) -> None:
    """Re-run VGGT on ``image_paths`` and open the official Viser UI.

    This is the ``run_viser_multi.py``-style path: the full per-frame prediction
    dict (world points, depth, per-frame confidence, images) is handed to
    ``demo_viser.viser_wrapper``.

    ``drop_white`` zeroes the confidence of near-white pixels (the Zero123++
    white background) so the official UI filters them out, mirroring what the
    framework's ``filter_pointcloud`` does for exported PLYs.
    """
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    vggt_repo = str(Path(cfg["paths"]["vggt_repo"]).resolve())
    if vggt_repo not in sys.path:
        sys.path.insert(0, vggt_repo)

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from demo_viser import viser_wrapper

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but unavailable; falling back to CPU")
        device = "cpu"

    print(f"loading VGGT ({cfg['vggt']['model']}) on {device} ...")
    model = VGGT.from_pretrained(cfg["vggt"]["model"])
    model = model.to(device).eval()

    images = load_and_preprocess_images(image_paths).to(device)
    print(f"preprocessed {len(image_paths)} images -> {tuple(images.shape)}")

    try:
        dtype = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
    except Exception:
        dtype = torch.float32
    if device == "cpu":
        dtype = torch.float32

    with torch.no_grad():
        if device == "cuda":
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    for key in predictions:
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    if drop_white:
        images_arr = predictions.get("images")  # (S, 3, H, W) in [0, 1]
        if images_arr is not None:
            white = (images_arr > 0.98).all(axis=1)  # (S, H, W)
            for conf_key in ("world_points_conf", "depth_conf"):
                conf_map = predictions.get(conf_key)
                if conf_map is not None and conf_map.shape == white.shape:
                    predictions[conf_key] = np.where(white, 0.0, conf_map)
            print(f"   🧹  white background masked: {int(white.sum()):,} pixels")

    print("=" * 60)
    print(f"   📡  Viser: http://localhost:{port}")
    print("     • Show Points from Frames 下拉框 → 选择某一帧或全部")
    print("     • 点击彩色相机锥体 → 跳转到该相机视角")
    print("     • Confidence Percent 滑块 → 实时调节过滤")
    print("=" * 60)

    viser_wrapper(
        predictions,
        port=port,
        init_conf_threshold=init_conf_threshold,
        use_point_map=use_point_map,
        background_mode=False,
        mask_sky=False,
        image_folder=None,
    )
