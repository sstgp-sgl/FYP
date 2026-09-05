"""Reconstruction backend abstraction and result container."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np

from src.reconstruct.task import TaskContext


@dataclass
class ReconstructionResult:
    """A fused multi-view point cloud plus per-camera poses.

    ``extrinsic`` is world→camera ``[R|t]`` (S,3,4); ``intrinsic`` is the
    per-camera 3x3 calibration. Points/colors/conf are flattened (N, ...).
    """

    points: np.ndarray
    colors: np.ndarray
    conf: np.ndarray
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    image_paths: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def num_points(self) -> int:
        return int(len(self.points))

    @property
    def num_cameras(self) -> int:
        return int(len(self.extrinsic))

    def save_npz(self, path) -> None:
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            points=self.points,
            colors=self.colors,
            conf=self.conf,
            extrinsic=self.extrinsic,
            intrinsic=self.intrinsic,
        )


class ReconstructionBackend(abc.ABC):
    """Interface for turning a list of images into a fused point cloud."""

    name: str = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abc.abstractmethod
    def reconstruct(self, image_paths: list[str], ctx: TaskContext) -> ReconstructionResult:
        """Reconstruct from ``image_paths`` (already preprocessed)."""
