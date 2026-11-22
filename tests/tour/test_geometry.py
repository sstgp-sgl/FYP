"""Geometry: fixed display transform, roundtrip, preview sampling stability,
bounds helpers."""
import numpy as np
import pytest

from src.tour.geometry import (
    bounds_intersect_allow_margin,
    compute_display_transform,
    from_display,
    preview_seed,
    sample_preview,
    to_display,
)
from tests.tour.conftest import make_points


def test_transform_roundtrip_known_cloud():
    # Non-origin cloud with known extents: center must be the AABB center.
    points = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 4.0, 2.0], [2.0, 2.0, 4.0]])
    transform = compute_display_transform(points)
    assert np.allclose(transform.center, [2.0, 3.0, 2.5])
    display = to_display(points, transform)
    assert np.max(np.linalg.norm(display, axis=1)) == pytest.approx(1.0)
    back = from_display(display, transform)
    assert np.allclose(back, points)


def test_transform_handles_outliers_without_renormalization():
    rng = np.random.default_rng(3)
    points = rng.normal(size=(500, 3))
    transform = compute_display_transform(points)
    display = to_display(points, transform)
    assert np.max(np.linalg.norm(display, axis=1)) <= 1.0 + 1e-9


def test_degenerate_cloud_rejected():
    points = np.tile([1.0, 1.0, 1.0], (10, 1))
    with pytest.raises(ValueError):
        compute_display_transform(points)


def test_empty_cloud_rejected():
    with pytest.raises(ValueError):
        compute_display_transform(np.zeros((0, 3)))


def test_sampling_stable_and_capped():
    points, colors = make_points(count=5000, seed=1)
    seed = preview_seed("ab" * 32)
    first, first_colors = sample_preview(points, colors, 600, seed)
    second, second_colors = sample_preview(points, colors, 600, seed)
    assert len(first) == 600
    assert np.array_equal(first, second)
    assert np.array_equal(first_colors, second_colors)


def test_sampling_below_cap_returns_all():
    points, colors = make_points(count=50)
    sampled, sampled_colors = sample_preview(points, colors, 600, 7)
    assert len(sampled) == 50
    assert np.array_equal(sampled, points)


def test_bounds_intersect_allow_margin():
    scene_lo = np.array([0.0, 0.0, 0.0])
    scene_hi = np.array([2.0, 2.0, 2.0])
    # slightly outside but overlapping margin: allowed
    assert bounds_intersect_allow_margin(
        np.array([-0.02, 0.0, 0.0]), np.array([0.5, 1.0, 1.0]), scene_lo, scene_hi
    )
    # completely disjoint: rejected
    assert not bounds_intersect_allow_margin(
        np.array([10.0, 10.0, 10.0]), np.array([11.0, 11.0, 11.0]), scene_lo, scene_hi
    )
