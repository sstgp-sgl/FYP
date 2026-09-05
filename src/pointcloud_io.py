"""Torch/trimesh-free point-cloud IO: ASCII PLY read/write + cameras JSON.

The reconstruction task framework writes/reads its own PLY so it does not
depend on trimesh (which is not installed on every host). The format is the
standard ASCII PLY with ``x/y/z`` float vertices and ``red/green/blue`` uchar
colors, which tools like MeshLab/CloudCompare/Viser read without issue.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply(points, colors, path: str | Path) -> Path:
    """Write a colored point cloud as ASCII PLY (numpy only)."""
    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"colors {colors.shape} must match points {points.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n")
    return path


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a colored ASCII PLY produced by :func:`write_ply`.

    Returns (points (N,3) float64, colors (N,3) uint8).
    """
    path = Path(path)
    text = path.read_text()
    header_done = False
    n_vertices = 0
    props: list[tuple[str, str]] = []  # (name, type)
    points: list[list[float]] = []
    colors: list[list[int]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("comment"):
            continue
        if not header_done:
            if line.startswith("format"):
                if "ascii" not in line:
                    raise ValueError(f"only ascii PLY supported, got: {line}")
            elif line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                # Scalar property: "property <type> <name>". Skip list properties.
                if len(parts) == 3:
                    _, ptype, pname = parts
                    props.append((pname, ptype))
            elif line == "end_header":
                header_done = True
            continue
        # body
        parts = line.split()
        if len(parts) != len(props):
            continue
        xyz = [0.0, 0.0, 0.0]
        rgb = [128, 128, 128]
        for (name, ptype), val in zip(props, parts):
            if name in ("x", "y", "z"):
                xyz["xyz".index(name)] = float(val)
            elif name in ("red", "green", "blue"):
                rgb[{"red": 0, "green": 1, "blue": 2}[name]] = int(float(val))
        points.append(xyz)
        colors.append(rgb)

    if n_vertices and len(points) != n_vertices:
        raise ValueError(f"PLY header promised {n_vertices} vertices, read {len(points)}")
    if not points:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(colors, dtype=np.uint8),
    )


def cameras_to_json(extrinsic, intrinsic, image_paths, views_meta=None) -> dict:
    """Serialize camera matrices into the stage-3 cameras.json shape."""
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
    return {"cameras": cameras}
