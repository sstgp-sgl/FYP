"""Unit tests for VGGT reconstruction helpers (mocked model, no GPU weights)."""
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.vggt_reconstruct import filter_pointcloud, run_vggt


class _MockVGGT:
    """Returns deterministic depth/pose tensors matching VGGT output shapes."""

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, images: torch.Tensor):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        b, s, _, h, w = images.shape
        # Identity-ish pose: T=0, quat=identity (wxyz or xyzw — mat_to_quat uses w last typically)
        # absT_quaR_FoV: T(3) + quat(4) + fov_h + fov_w
        pose = torch.zeros(b, s, 9)
        pose[..., 3] = 1.0  # quaternion identity component (x,y,z,w) — check mat_to_quat
        # Use identity quaternion as (0,0,0,1) for xyzw
        pose[..., 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        pose[..., 7] = 0.8  # fov_h
        pose[..., 8] = 0.8  # fov_w
        # Vary translation per frame so cameras differ
        for i in range(s):
            pose[:, i, 0] = float(i) * 0.1

        depth = torch.ones(b, s, h, w, 1) * 2.0
        # Rising confidence so percentile filter is testable on real outputs too
        conf = torch.linspace(0, 1, h * w, dtype=torch.float32).reshape(1, 1, h, w)
        conf = conf.expand(b, s, h, w).clone()

        # Native pointmap (B,S,H,W,3): normalized pixel grid, x shifted per frame.
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        wp = torch.stack(
            [xx / max(w - 1, 1), yy / max(h - 1, 1), torch.zeros_like(xx)], dim=-1
        )  # (H, W, 3)
        world_points = wp[None, None].expand(b, s, h, w, 3).clone()
        for i in range(s):
            world_points[:, i, :, :, 0] += float(i) * 0.1
        world_points_conf = conf  # (B,S,H,W)

        return {
            "pose_enc": pose,
            "depth": depth,
            "depth_conf": conf,
            "world_points": world_points,
            "world_points_conf": world_points_conf,
            "images": images,
        }


def _write_dummy_views(tmp_path: Path, n: int = 2, size: int = 64):
    paths = []
    for i in range(n):
        p = tmp_path / f"view_{i:02d}.png"
        arr = np.full((size, size, 3), 40 + i * 20, dtype=np.uint8)
        Image.fromarray(arr).save(p)
        paths.append(str(p))
    return paths


def test_run_vggt_output_shapes(tmp_path):
    image_paths = _write_dummy_views(tmp_path, n=2, size=64)
    cfg = {
        "device": "cpu",
        "paths": {"vggt_repo": "/root/autodl-tmp/vggt"},
        "vggt": {"model": "facebook/VGGT-1B"},
    }
    result = run_vggt(image_paths, cfg, model=_MockVGGT())

    assert result["points"].ndim == 2 and result["points"].shape[1] == 3
    assert result["colors"].shape == result["points"].shape
    assert result["conf"].shape == (result["points"].shape[0],)
    assert result["extrinsic"].shape == (2, 3, 4)
    assert result["intrinsic"].shape == (2, 3, 3)
    # load_and_preprocess_images resizes square inputs to 518x518
    assert result["points"].shape[0] == 2 * 518 * 518  # S * H * W before filtering
    assert np.isfinite(result["points"]).all()
    # Default is the pointmap branch.
    assert result["use_point_map"] is True


def test_run_vggt_pointmap_and_depth_branches(tmp_path):
    image_paths = _write_dummy_views(tmp_path, n=2, size=64)
    cfg = {
        "device": "cpu",
        "paths": {"vggt_repo": "/root/autodl-tmp/vggt"},
        "vggt": {"model": "facebook/VGGT-1B"},
    }

    pm = run_vggt(image_paths, cfg, model=_MockVGGT(), use_point_map=True)
    dp = run_vggt(image_paths, cfg, model=_MockVGGT(), use_point_map=False)

    assert pm["use_point_map"] is True and dp["use_point_map"] is False
    assert pm["points"].shape == dp["points"].shape
    # Pointmap x coords come from the pixel grid (0..1 + per-frame offset);
    # depth branch unprojects constant depth=2 at identity-ish poses.
    assert pm["points"][:, 0].min() >= 0.0
    assert pm["points"][:, 0].max() <= 1.0 + 0.1 + 1e-6
    assert np.isfinite(dp["points"]).all()


def test_filter_removes_lowest_conf_percentile():
    n = 100
    points = np.zeros((n, 3), dtype=np.float64)
    points[:, 0] = np.arange(n)  # distinct coords
    colors = np.full((n, 3), 128, dtype=np.uint8)
    conf = np.arange(n, dtype=np.float64)  # 0..99
    cfg = {
        "vggt": {
            "conf_percentile": 40,
            "outlier_nb_neighbors": 5,
            "outlier_std_ratio": 100.0,  # effectively disable outlier removal
        }
    }
    f_pts, f_cols, stats = filter_pointcloud(points, colors, conf, cfg)
    # Keep conf >= percentile(40) of [0..99] → threshold ≈ 39.6, keep 40..99 → 60 points
    assert len(f_pts) == 60
    assert f_pts[:, 0].min() >= 40
    assert stats["conf_threshold"] == pytest.approx(np.percentile(conf, 40))
    assert stats["num_points_raw"] == 100
    assert stats["num_points_filtered"] == 60


def test_filter_drops_pure_white_points():
    n = 50
    points = np.zeros((n, 3), dtype=np.float64)
    points[:, 0] = np.arange(n)
    colors = np.full((n, 3), 128, dtype=np.uint8)
    colors[:10] = 255
    conf = np.full(n, 10.0)
    cfg = {
        "vggt": {
            "conf_percentile": 0,
            "outlier_nb_neighbors": 0,
            "outlier_std_ratio": 100.0,
        }
    }
    f_pts, f_cols, stats = filter_pointcloud(points, colors, conf, cfg)
    assert len(f_pts) == 40
    assert not np.all(f_cols == 255, axis=1).any()
    assert stats["num_points_after_white"] == 40


def test_filter_drops_pure_black_points():
    n = 10
    points = np.zeros((n, 3), dtype=np.float64)
    points[:, 0] = np.arange(n)
    colors = np.full((n, 3), 128, dtype=np.uint8)
    colors[:3] = 0
    conf = np.full(n, 1.0)
    cfg = {
        "vggt": {
            "conf_percentile": 0,
            "outlier_nb_neighbors": 0,
            "outlier_std_ratio": 100.0,
        }
    }
    f_pts, f_cols, stats = filter_pointcloud(points, colors, conf, cfg)
    assert len(f_pts) == 7
    assert not np.all(f_cols == 0, axis=1).any()
    assert stats["num_points_filtered"] == 7
