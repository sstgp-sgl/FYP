"""Stage 3: multi-view VGGT reconstruction from Zero123++ views.

Consumes (from stages 1–2):
    outputs/views/view_00.png..view_05.png
    outputs/views/views.json
    outputs/foreground/foreground.png  (optional 7th view, composited to RGB)

Outputs (in outputs/vggt/):
    pointcloud.ply   - filtered colored point cloud
    cameras.json     - per-view extrinsics/intrinsics + COLMAP hook metadata
    stats.json       - point counts and filter thresholds
    predictions.npz  - raw points/colors/conf before filtering (debug)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import (check_array, list_view_images, load_config, load_json,
                          require_files, save_json)
from src.vggt_reconstruct import (cameras_to_json, export_pointcloud_ply,
                                  filter_pointcloud, run_vggt)


def _optional_foreground_rgb(cfg) -> Path | None:
    """Composite RGBA foreground to a temp RGB path if present; else None."""
    from PIL import Image

    fg_path = Path(cfg["paths"]["foreground_dir"]) / "foreground.png"
    if not fg_path.exists():
        return None
    gray = int(cfg["segment"]["background_gray"])
    fg = Image.open(fg_path).convert("RGBA")
    bg = Image.new("RGBA", fg.size, (gray, gray, gray, 255))
    rgb = Image.alpha_composite(bg, fg).convert("RGB")
    out = Path(cfg["paths"]["vggt_dir"]) / "_cond_foreground_rgb.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out)
    return out


def main():
    cfg = load_config()
    torch.manual_seed(cfg["seed"])
    out_dir = Path(cfg["paths"]["vggt_dir"])
    views_dir = Path(cfg["paths"]["views_dir"])
    views_json = views_dir / "views.json"

    view_paths = list_view_images(views_dir)
    require_files(views_json, *view_paths)

    image_paths = [str(p) for p in view_paths]
    cond = _optional_foreground_rgb(cfg)
    if cond is not None:
        # Prepend original view as first frame (matches Zero123++ input pose).
        image_paths = [str(cond)] + image_paths

    print(f"running VGGT on {len(image_paths)} images")
    result = run_vggt(image_paths, cfg)
    check_array("points", result["points"], shape=(None, 3))
    check_array("colors", result["colors"], shape=(result["points"].shape[0], 3))

    # Save unfiltered arrays for debugging / COLMAP export later.
    np.savez_compressed(
        out_dir / "predictions.npz",
        points=result["points"],
        colors=result["colors"],
        conf=result["conf"],
        extrinsic=result["extrinsic"],
        intrinsic=result["intrinsic"],
    )
    print(f"saved {out_dir / 'predictions.npz'}")

    points, colors, stats = filter_pointcloud(
        result["points"], result["colors"], result["conf"], cfg
    )
    print(f"points: {stats['num_points_raw']:,} -> {stats['num_points_filtered']:,} "
          f"(conf p{stats['conf_percentile']}={stats['conf_threshold']:.4f}, "
          f"dropped white {stats['num_points_after_conf'] - stats['num_points_after_white']:,})")

    if stats["num_points_filtered"] < 10_000:
        raise ValueError(
            f"Suspicious point count {stats['num_points_filtered']}; "
            "expected >10k after filtering. Check views or conf_percentile."
        )
    if not np.isfinite(points).all():
        raise ValueError("Filtered point cloud contains NaN/Inf")

    ply_path = out_dir / "pointcloud.ply"
    export_pointcloud_ply(points, colors, ply_path)
    print(f"saved {ply_path}")

    views_meta = load_json(views_json).get("views", [])
    # If foreground was prepended, shift metadata: first camera has no az/el.
    if cond is not None:
        views_meta = [{"file": cond.name, "azimuth": 0, "elevation": 0}] + views_meta

    cameras = cameras_to_json(
        result["extrinsic"], result["intrinsic"], result["image_paths"], views_meta
    )
    save_json(cameras, out_dir / "cameras.json")
    save_json(stats, out_dir / "stats.json")
    print("stage 3 done")


if __name__ == "__main__":
    main()
