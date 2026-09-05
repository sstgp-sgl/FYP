"""Strip white background points from an existing stage-3 point cloud.

Stage 2 SAM2 only paints the 2D background white. VGGT still unprojects every
pixel, so those whites become a 3D sheet. This script does not re-run VGGT; it
rewrites the PLY (and optionally predictions.npz).

Usage:
    python scripts/22_drop_white.py
    python scripts/22_drop_white.py --slack 8
    python scripts/22_drop_white.py --in outputs/vggt/pointcloud.ply --out outputs/vggt/pointcloud_nowhite.ply
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import load_config
from src.vggt_reconstruct import export_pointcloud_ply, fill_color_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Drop white RGB points from a PLY")
    parser.add_argument("--in", dest="inp", default=None, help="Input PLY (default: outputs/vggt/pointcloud.ply)")
    parser.add_argument("--out", dest="out", default=None, help="Output PLY (default: overwrite input)")
    parser.add_argument(
        "--slack",
        type=int,
        default=8,
        help="Drop RGB within this distance of (255,255,255). 0 = exact 255 only.",
    )
    parser.add_argument(
        "--also-npz",
        action="store_true",
        help="Also rewrite outputs/vggt/predictions.npz colors/points",
    )
    return parser.parse_args()


def load_ply_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(path), process=False)
    points = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3]
    return points, colors


def main():
    args = parse_args()
    cfg = load_config()
    inp = Path(args.inp) if args.inp else Path(cfg["paths"]["vggt_dir"]) / "pointcloud.ply"
    out = Path(args.out) if args.out else inp
    if not inp.exists():
        raise FileNotFoundError(f"Missing point cloud: {inp}")

    points, colors = load_ply_rgb(inp)
    n0 = len(points)
    white = fill_color_mask(colors, gray=255, slack=args.slack)
    keep = ~white
    n_drop = int(white.sum())
    print(f"loaded {inp}")
    print(f"  {n0:,} points, dropping {n_drop:,} with RGB in "
          f"[{255 - args.slack}, 255]^3  ({100.0 * n_drop / max(n0, 1):.1f}%)")

    export_pointcloud_ply(points[keep], colors[keep], out)
    print(f"wrote {out}  ({int(keep.sum()):,} points)")

    if args.also_npz:
        npz_path = Path(cfg["paths"]["vggt_dir"]) / "predictions.npz"
        data = dict(np.load(npz_path))
        w = fill_color_mask(data["colors"], gray=255, slack=args.slack)
        k = ~w
        data["points"] = data["points"][k]
        data["colors"] = data["colors"][k]
        data["conf"] = data["conf"][k]
        np.savez_compressed(npz_path, **data)
        print(f"rewrote {npz_path}  ({int(k.sum()):,} points)")


if __name__ == "__main__":
    main()
