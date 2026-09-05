"""Deterministic dummy backend (no model, no GPU) for testing and demos.

Generates a synthetic colored sphere and a ring of cameras looking at it, so
the whole pipeline can run end-to-end without PyTorch / VGGT / trimesh.
"""
from __future__ import annotations

import numpy as np

from src.reconstruct.backends.base import ReconstructionBackend, ReconstructionResult
from src.reconstruct.task import TaskContext


def _fibonacci_sphere(n: int, radius: float = 1.0, rng=None) -> np.ndarray:
    """Approximately uniform points on a sphere (Fibonacci lattice)."""
    rng = rng or np.random.default_rng(0)
    idx = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * idx / n)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    pts = np.stack([x, y, z], axis=1)
    jitter = rng.normal(0, radius * 0.004, size=pts.shape)
    return pts + jitter


def _look_at_w2c(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World→camera [R|t] for a camera at ``eye`` looking at ``center``."""
    forward = center - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up2 = np.cross(right, forward)
    R = np.stack([right, up2, -forward], axis=0)  # rows = camera axes
    t = -R @ eye
    return np.hstack([R, t[:, None]])


class DummyBackend(ReconstructionBackend):
    name = "dummy"

    def reconstruct(self, image_paths: list[str], ctx: TaskContext) -> ReconstructionResult:
        seed = int(ctx.cfg.get("seed", 42))
        rng = np.random.default_rng(seed)

        n = 6000
        radius = 1.0
        points = _fibonacci_sphere(n, radius, rng)

        # Color by latitude/longitude for a pleasant, deterministic gradient.
        phi = np.arccos(np.clip(points[:, 2] / radius, -1, 1))
        theta = np.arctan2(points[:, 1], points[:, 0])
        r = (np.sin(theta) + 1) / 2
        g = (np.sin(phi) + 1) / 2
        b = (np.cos(theta) + 1) / 2
        colors = (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)
        conf = np.ones(n, dtype=np.float64)

        # Cameras on a ring around the object.
        s = max(len(image_paths), 1)
        extrinsic = np.zeros((s, 3, 4), dtype=np.float64)
        intrinsic = np.zeros((s, 3, 3), dtype=np.float64)
        focal = 500.0
        for i in range(s):
            az = 2 * np.pi * i / s
            eye = np.array([2.5 * np.cos(az), 0.4, 2.5 * np.sin(az)])
            extrinsic[i] = _look_at_w2c(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))
            intrinsic[i] = np.array(
                [[focal, 0, 256], [0, focal, 256], [0, 0, 1]], dtype=np.float64
            )

        return ReconstructionResult(
            points=points,
            colors=colors,
            conf=conf,
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            image_paths=list(image_paths),
            meta={"synthetic": True, "seed": seed},
        )
