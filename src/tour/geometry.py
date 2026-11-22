"""Fixed display transform, deterministic preview sampling and bounds checks.

The display transform ``p_display = scale * (p_asset - center)`` is computed
exactly once from the full finite point set and persisted in the scene; it is
never re-derived from a sample, so saved cameras and boxes stay consistent
across reloads.
"""
from __future__ import annotations

import numpy as np

from .models import DisplayTransform

DEFAULT_COLOR = (180, 180, 180)  # missing RGB falls back to a light gray

MIN_DEGENERATE_RADIUS = 1e-12


def compute_display_transform(points: np.ndarray) -> DisplayTransform:
    """Center on the AABB center of all finite points; scale by 1/max-distance.

    Rejects degenerate point clouds (zero radius) instead of inventing a size.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("empty point cloud")
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        raise ValueError("point cloud contains no finite points")
    pts = points[finite]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(pts - center, axis=1).max())
    if radius <= MIN_DEGENERATE_RADIUS:
        raise ValueError("degenerate point cloud: all points share one position")
    return DisplayTransform(center=center.tolist(), scale=1.0 / radius)


def to_display(points: np.ndarray, transform: DisplayTransform) -> np.ndarray:
    center = np.asarray(transform.center, dtype=np.float64)
    return (np.asarray(points, dtype=np.float64) - center) * transform.scale


def from_display(points: np.ndarray, transform: DisplayTransform) -> np.ndarray:
    center = np.asarray(transform.center, dtype=np.float64)
    return np.asarray(points, dtype=np.float64) / transform.scale + center


def preview_seed(asset_sha256: str) -> int:
    """Deterministic seed so the preview sample never changes for one asset."""
    return int(asset_sha256[:16], 16) & 0x7FFFFFFF


def sample_preview(points: np.ndarray, colors: np.ndarray, preview_points: int, seed: int):
    """Fixed-seed sampling; identical input always yields identical output."""
    n = len(points)
    if n <= preview_points:
        return np.asarray(points), np.asarray(colors)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, preview_points, replace=False)
    return np.asarray(points)[indices], np.asarray(colors)[indices]


def point_bounds(points: np.ndarray):
    points = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        raise ValueError("point cloud contains no finite points")
    pts = points[finite]
    return pts.min(axis=0), pts.max(axis=0)


def bounds_intersect_allow_margin(entity_lo, entity_hi, scene_lo, scene_hi, margin_ratio=0.05):
    """Entity box may stick out of the scene bounds a little, but a box that is
    completely disjoint from the scene (including margin) is rejected."""
    extent = scene_hi - scene_lo
    margin = margin_ratio * float(np.linalg.norm(extent))
    lo = scene_lo - margin
    hi = scene_hi + margin
    return bool(
        np.all(entity_hi >= lo) and np.all(entity_lo <= hi)
    )


def camera_frustum_hint(camera, entity_lo, entity_hi, transform: DisplayTransform):
    """Advisory check: does the AABB center fall inside the camera view cone?

    This is a hint only (no occlusion handling) and must never be presented as
    a proof of visibility.
    """
    import math

    import numpy as np

    center = np.asarray(transform.center)
    scale = transform.scale
    position = (np.asarray(camera.position_asset) - center) * scale
    target = (np.asarray(camera.target_asset) - center) * scale
    up = np.asarray(camera.up, dtype=np.float64)
    box_center = (np.asarray(entity_lo) + np.asarray(entity_hi)) / 2.0
    point = (box_center - center) * scale

    forward = target - position
    forward = forward / (np.linalg.norm(forward) + 1e-300)
    to_point = point - position
    distance = float(np.linalg.norm(to_point))
    if distance < 1e-12:
        return True, 0.0
    direction = to_point / distance
    angle = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(direction, forward))))))
    # Approximate the cone with the vertical fov and the aspect ratio.
    half_v = camera.fov_y_deg / 2.0
    half_h = math.degrees(math.atan(math.tan(math.radians(half_v)) * camera.aspect_ratio))
    return angle <= max(half_v, half_h) * 1.1, angle
