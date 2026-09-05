"""Torch-free point-cloud filtering helpers.

Moved out of ``src/vggt_reconstruct.py`` so the reconstruction task framework
(and any other consumer) can filter a point cloud without importing PyTorch.
The logic is identical to the stage-3 filter; ``vggt_reconstruct`` re-exports
these names for backward compatibility.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def fill_color_mask(colors, gray: int, slack: int) -> np.ndarray:
    """True where RGB is within ``slack`` of the constant fill ``(gray, gray, gray)``.

    Stage 2 paints the background with ``segment.background_gray`` (usually 255).
    VGGT still unprojects those pixels. Resize 512→518 slightly bleeds the fill,
    so exact equality would miss the sheet.
    """
    colors = np.asarray(colors)
    lo = max(0, int(gray) - int(slack))
    hi = min(255, int(gray) + int(slack))
    return (colors >= lo).all(axis=-1) & (colors <= hi).all(axis=-1)


def filter_pointcloud(points, colors, conf, cfg) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply confidence percentile + statistical outlier filtering.

    Returns (filtered_points, filtered_colors, stats).
    """
    points = np.asarray(points)
    colors = np.asarray(colors)
    conf = np.asarray(conf).reshape(-1)
    vggt_cfg = cfg["vggt"]
    percentile = vggt_cfg["conf_percentile"]
    nb = int(vggt_cfg["outlier_nb_neighbors"])
    std_ratio = float(vggt_cfg["outlier_std_ratio"])

    num_raw = len(points)
    threshold = float(np.percentile(conf, percentile))
    mask = (conf >= threshold) & (conf > 1e-5) & np.isfinite(points).all(axis=1)

    if vggt_cfg.get("drop_fill_color", True):
        gray = int(cfg.get("segment", {}).get("background_gray", 255))
        slack = int(vggt_cfg.get("fill_color_slack", 8))
        mask = mask & ~fill_color_mask(colors, gray, slack)

    points = points[mask]
    colors = colors[mask]
    conf_kept = conf[mask]
    num_after_conf = len(points)

    # Stage-2 SAM2 composites the subject onto white; VGGT still unprojects
    # every pixel, so drop pure-white background points here.
    not_white = ~np.all(colors == 255, axis=1)
    points = points[not_white]
    colors = colors[not_white]
    conf_kept = conf_kept[not_white]
    num_after_white = len(points)

    if len(points) > nb and nb > 0:
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=nb + 1)  # include self
        mean_dist = dists[:, 1:].mean(axis=1)
        mu, sigma = mean_dist.mean(), mean_dist.std()
        inlier = mean_dist <= (mu + std_ratio * sigma)
        points = points[inlier]
        colors = colors[inlier]
        conf_kept = conf_kept[inlier]

    stats = {
        "num_points_raw": int(num_raw),
        "num_points_after_conf": int(num_after_conf),
        "num_points_after_white": int(num_after_white),
        "num_points_filtered": int(len(points)),
        "conf_percentile": percentile,
        "conf_threshold": threshold,
        "outlier_nb_neighbors": nb,
        "outlier_std_ratio": std_ratio,
    }
    return points, colors, stats
