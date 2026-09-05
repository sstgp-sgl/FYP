"""VGGT multi-view reconstruction: inference, flatten, and point-cloud filtering."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import torch

# Filtering helpers live in a torch-free module so other consumers (the
# reconstruction task framework) can use them without importing torch.
from src.pointcloud_filter import fill_color_mask, filter_pointcloud  # noqa: F401


def _ensure_vggt_on_path(vggt_repo: str | Path) -> None:
    repo = str(Path(vggt_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def load_vggt_model(cfg: dict):
    """Load pretrained VGGT weights onto the configured device."""
    _ensure_vggt_on_path(cfg["paths"]["vggt_repo"])
    # HuggingFace hub is unreachable from this host; mirror works.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from vggt.models.vggt import VGGT

    device = cfg["device"]
    model = VGGT.from_pretrained(cfg["vggt"]["model"])
    model = model.to(device).eval()
    return model


def run_vggt(image_paths: list[str], cfg: dict, model=None, use_point_map: bool | None = None) -> dict:
    """Run VGGT on a list of image paths and return a flat point cloud + cameras.

    ``use_point_map=True`` (default) uses VGGT's native pointmap branch
    (``world_points`` / ``world_points_conf``) — the model's globally-consistent
    3D output, which is what the official multi-view demo uses for the best
    cross-view reconstruction. ``use_point_map=False`` falls back to unprojecting
    per-frame depth maps (noisier and cross-view inconsistent).

    When ``use_point_map`` is None it is read from ``cfg.vggt.use_point_map``
    (default True).

    Returns dict with keys:
        points (N,3), colors (N,3) uint8, conf (N,),
        extrinsic (S,3,4), intrinsic (S,3,3), image_paths
    """
    _ensure_vggt_on_path(cfg["paths"]["vggt_repo"])
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    if use_point_map is None:
        use_point_map = bool(cfg.get("vggt", {}).get("use_point_map", True))

    if model is None:
        model = load_vggt_model(cfg)

    device = cfg["device"]
    paths = [str(p) for p in image_paths]
    images = load_and_preprocess_images(paths).to(device)  # (S, 3, H, W)

    try:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
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
    # (1, S, ...) -> (S, ...)
    extrinsic = extrinsic.squeeze(0).detach().cpu().numpy()
    intrinsic = intrinsic.squeeze(0).detach().cpu().numpy()

    if use_point_map:
        # Native pointmap branch: globally-consistent across views.
        points_map = predictions["world_points"]  # (1, S, H, W, 3)
        if points_map.ndim == 5:
            points_map = points_map.squeeze(0)
        conf_map = predictions["world_points_conf"]  # (1, S, H, W)
        if conf_map.ndim == 4 and conf_map.shape[0] == 1:
            conf_map = conf_map.squeeze(0)
        points_np = points_map.detach().float().cpu().numpy()
        conf_np = conf_map.detach().float().cpu().numpy()
    else:
        # Depth branch: unproject each frame's depth (legacy / fallback).
        depth = predictions["depth"]
        if depth.ndim == 5:
            depth = depth.squeeze(0)
        depth_np = depth.detach().float().cpu().numpy()

        depth_conf = predictions["depth_conf"]
        if depth_conf.ndim == 4 and depth_conf.shape[0] == 1:
            depth_conf = depth_conf.squeeze(0)
        conf_np = depth_conf.detach().float().cpu().numpy()

        points_np = unproject_depth_map_to_point_map(depth_np, extrinsic, intrinsic)

    images_np = images.detach().float().cpu().numpy()  # (S, 3, H, W)
    colors = np.transpose(images_np, (0, 2, 3, 1))  # (S, H, W, 3)
    colors_u8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    points = points_np.reshape(-1, 3)
    colors_flat = colors_u8.reshape(-1, 3)
    conf_flat = conf_np.reshape(-1)

    return {
        "points": points.astype(np.float64),
        "colors": colors_flat,
        "conf": conf_flat.astype(np.float64),
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "image_paths": paths,
        "use_point_map": use_point_map,
    }


def export_pointcloud_ply(points, colors, path: str | Path) -> None:
    """Write a colored point cloud as PLY via trimesh."""
    import trimesh

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = trimesh.PointCloud(vertices=np.asarray(points), colors=np.asarray(colors))
    pcd.export(str(path))


def cameras_to_json(extrinsic, intrinsic, image_paths, views_meta=None) -> dict:
    """Serialize camera matrices for cameras.json (and COLMAP hook notes)."""
    cameras = []
    for i, img_path in enumerate(image_paths):
        entry = {
            "index": i,
            "file": Path(img_path).name,
            "extrinsic": np.asarray(extrinsic[i]).tolist(),
            "intrinsic": np.asarray(intrinsic[i]).tolist(),
        }
        if views_meta and i < len(views_meta):
            entry["azimuth"] = views_meta[i].get("azimuth")
            entry["elevation"] = views_meta[i].get("elevation")
        cameras.append(entry)
    return {
        "cameras": cameras,
        # Hook for stage 4 (gsplat): optionally place COLMAP sparse/ under vggt_dir.
        "colmap_export": {
            "enabled": False,
            "relative_dir": "sparse",
            "note": (
                "Set enabled=true and emit COLMAP cameras.bin/images.bin/points3D.bin "
                "under outputs/vggt/sparse/ for gsplat simple_trainer "
                "(see scripts/04_gaussian.py)."
            ),
        },
    }
