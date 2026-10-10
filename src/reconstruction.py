"""Direct VGGT inference and point-cloud export; no task registry or cache."""
from pathlib import Path
import numpy as np

from src.io_utils import list_view_images, save_json
from src.pointcloud_filter import filter_pointcloud
from src.pointcloud_io import cameras_to_json, write_ply


def reconstruct_views(views_dir, output_dir, cfg):
    # Read the manifest, never glob: stale images must not change view count.
    images = list_view_images(views_dir)
    expected = 7 if cfg["views"].get("include_original", True) else 6
    names = [p.name for p in images]
    if names != [f"view_{i:02d}.png" for i in range(expected)]:
        raise ValueError(f"Expected {expected} ordered views, got {names}; regenerate views")
    from src.vggt_reconstruct import run_vggt

    result = run_vggt([str(p) for p in images], cfg)
    points, colors, stats = filter_pointcloud(
        result["points"], result["colors"], result["conf"], cfg
    )
    if not len(points):
        raise ValueError("VGGT filtering produced an empty point cloud; check the views and confidence threshold")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats.update(num_views=len(images), include_original=(expected == 7))
    np.savez_compressed(output_dir / "predictions.npz", **result)
    write_ply(points, colors, output_dir / "pointcloud.ply")
    save_json(cameras_to_json(result["extrinsic"], result["intrinsic"], images),
              output_dir / "cameras.json")
    save_json(stats, output_dir / "stats.json")
    print(f"VGGT: {len(images)} views → {len(points):,} points → {output_dir}")
    return stats


def reconstruct_subjects(index_doc, base_out, cfg, progress=None):
    subjects = index_doc.get("subjects", [])
    if not subjects:
        raise ValueError("No subjects to reconstruct")
    for i, subject in enumerate(subjects):
        name = subject["name"]
        if progress:
            progress(i, len(subjects), "reconstruct", name, {"status": "running"})
        try:
            stats = reconstruct_views(subject["views_dir"], Path(base_out) / name, cfg)
        except Exception as exc:
            if progress:
                progress(i, len(subjects), "reconstruct", name,
                         {"status": "error", "error": str(exc)})
            raise
        if progress:
            progress(i, len(subjects), "reconstruct", name,
                     {"status": "done", "num_views": stats["num_views"]})
