"""Export a COLMAP sparse model from stage-3 cameras + point cloud.

Writes text COLMAP under ``{vggt_dir}/sparse/`` and stages RGB images under
``{vggt_dir}/images/`` (resized to match VGGT intrinsics, typically 518²).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from src.io_utils import load_json, save_json


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
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


def _image_size_from_K(K: np.ndarray) -> tuple[int, int]:
    """Infer (width, height) from principal point (assumes centered pp)."""
    cx, cy = float(K[0, 2]), float(K[1, 2])
    width = max(1, int(round(cx * 2)))
    height = max(1, int(round(cy * 2)))
    return width, height


def stage_images_for_colmap(
    cameras: list[dict],
    source_dirs: list[Path],
    images_dir: Path,
    width: int,
    height: int,
) -> list[str]:
    """Copy/resize camera images into images_dir; return basenames in order."""
    images_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for cam in cameras:
        name = cam["file"]
        src = None
        for d in source_dirs:
            candidate = Path(d) / name
            if candidate.exists():
                src = candidate
                break
        if src is None:
            raise FileNotFoundError(f"Could not find image {name} in {source_dirs}")
        img = Image.open(src).convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        out_name = Path(name).stem + ".png"
        img.save(images_dir / out_name)
        names.append(out_name)
    return names


def export_colmap_sparse(
    vggt_dir: str | Path,
    max_points: int = 100_000,
    seed: int = 42,
) -> Path:
    """Write COLMAP text model + staged images from stage-3 artifacts.

    Returns path to the sparse/ directory.
    """
    vggt_dir = Path(vggt_dir)
    cameras_doc = load_json(vggt_dir / "cameras.json")
    cameras = cameras_doc["cameras"]
    if not cameras:
        raise ValueError("cameras.json has no cameras")

    K0 = np.asarray(cameras[0]["intrinsic"], dtype=np.float64)
    width, height = _image_size_from_K(K0)

    images_dir = vggt_dir / "images"
    source_dirs = [
        vggt_dir,
        vggt_dir.parent / "views",
        vggt_dir.parent / "foreground",
    ]
    image_names = stage_images_for_colmap(
        cameras, source_dirs, images_dir, width, height
    )

    # Load points from PLY (preferred) or predictions.npz
    ply_path = vggt_dir / "pointcloud.ply"
    if ply_path.exists():
        import trimesh

        pcd = trimesh.load(str(ply_path), process=False)
        points = np.asarray(pcd.vertices, dtype=np.float64)
        colors = np.asarray(pcd.colors[:, :3], dtype=np.uint8) if pcd.colors is not None \
            else np.full((len(points), 3), 128, dtype=np.uint8)
    else:
        data = np.load(vggt_dir / "predictions.npz")
        points = data["points"]
        colors = data["colors"]

    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    sparse_dir = vggt_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # cameras.txt — one PINHOLE camera per view (VGGT has per-frame K)
    cam_lines = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for i, cam in enumerate(cameras):
        K = np.asarray(cam["intrinsic"], dtype=np.float64)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        cam_lines.append(
            f"{i + 1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}"
        )
    (sparse_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n")

    # images.txt — two lines per image (pose line + empty points2D line)
    img_lines = [
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for i, cam in enumerate(cameras):
        E = np.asarray(cam["extrinsic"], dtype=np.float64)  # [R|t] world→cam
        R, t = E[:3, :3], E[:3, 3]
        q = rotmat_to_quat_wxyz(R)
        name = image_names[i]
        img_lines.append(
            f"{i + 1} {q[0]} {q[1]} {q[2]} {q[3]} "
            f"{t[0]} {t[1]} {t[2]} {i + 1} {name}"
        )
        img_lines.append("")  # no 2D observations
    (sparse_dir / "images.txt").write_text("\n".join(img_lines) + "\n")

    # points3D.txt
    pts_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"]
    for i, (p, c) in enumerate(zip(points, colors), start=1):
        pts_lines.append(
            f"{i} {p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])} 0"
        )
    (sparse_dir / "points3D.txt").write_text("\n".join(pts_lines) + "\n")

    # Also write sparse/0/ for gsplat Parser compatibility
    sparse0 = sparse_dir / "0"
    if sparse0.exists():
        shutil.rmtree(sparse0)
    sparse0.mkdir(parents=True)
    for name in ("cameras.txt", "images.txt", "points3D.txt"):
        shutil.copy2(sparse_dir / name, sparse0 / name)

    cameras_doc["colmap_export"] = {
        "enabled": True,
        "relative_dir": "sparse",
        "images_dir": "images",
        "num_cameras": len(cameras),
        "num_points": int(len(points)),
        "image_size": [width, height],
        "note": "Text COLMAP model for gsplat / stage 4.",
    }
    save_json(cameras_doc, vggt_dir / "cameras.json")
    print(f"COLMAP sparse written to {sparse_dir} "
          f"({len(cameras)} cams, {len(points)} points, {width}x{height})")
    return sparse_dir
