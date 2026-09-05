"""Minimal 3D Gaussian Splatting trainer using gsplat + stage-3 cameras/PLY."""
from __future__ import annotations

import math
import os
from pathlib import Path

# Must be set before gsplat JIT-compiles CUDA kernels (see scripts/04_gaussian.py).
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

from src.io_utils import load_json, save_json


def _load_scene(vggt_dir: Path, device: str, max_init_points: int = 80_000, seed: int = 42):
    """Load cameras, images, and init points for training."""
    import trimesh

    cameras = load_json(vggt_dir / "cameras.json")["cameras"]
    images_dir = vggt_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing {images_dir}; run COLMAP export first")

    views = []
    for cam in cameras:
        name = Path(cam["file"]).stem + ".png"
        path = images_dir / name
        if not path.exists():
            # fall back to original basename
            path = images_dir / cam["file"]
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        E = np.asarray(cam["extrinsic"], dtype=np.float64)  # w2c [R|t]
        K = np.asarray(cam["intrinsic"], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = E
        c2w = np.linalg.inv(w2c)
        views.append({
            "image": torch.from_numpy(arr).to(device),  # H,W,3
            "K": torch.from_numpy(K).float().to(device),
            "c2w": torch.from_numpy(c2w).float().to(device),
            "w2c": torch.from_numpy(w2c).float().to(device),
            "name": path.name,
        })

    pcd = trimesh.load(str(vggt_dir / "pointcloud.ply"), process=False)
    points = np.asarray(pcd.vertices, dtype=np.float32)
    colors = np.asarray(pcd.colors[:, :3], dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    if len(points) > max_init_points:
        idx = rng.choice(len(points), size=max_init_points, replace=False)
        points, colors = points[idx], colors[idx]

    return views, points, colors


def _init_splats(points: np.ndarray, colors: np.ndarray, device: str) -> torch.nn.ParameterDict:
    means = torch.nn.Parameter(torch.from_numpy(points).float().to(device))
    # isotropic scale from nearest-neighbor spacing
    with torch.no_grad():
        # rough scene scale
        extent = (means.max(0).values - means.min(0).values).mean().clamp(min=1e-3)
        init_scale = (extent / math.sqrt(len(means))).item()
    scales = torch.nn.Parameter(
        torch.full((len(means), 3), math.log(init_scale), device=device)
    )
    quats = torch.nn.Parameter(
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(len(means), 1)
    )
    opacities = torch.nn.Parameter(
        torch.logit(torch.full((len(means),), 0.1, device=device))
    )
    # SH degree 0 colors in [0,1] as DC logits via inverse sigmoid of rgb
    rgb = torch.from_numpy(colors).float().to(device).clamp(1e-4, 1 - 1e-4)
    colors_p = torch.nn.Parameter(torch.logit(rgb))
    return torch.nn.ParameterDict({
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors_p,
    })


def _make_optimizers(splats: torch.nn.ParameterDict, scene_scale: float):
    lr_means = 1.6e-4 * scene_scale
    optimizers = {
        "means": torch.optim.Adam([splats["means"]], lr=lr_means, eps=1e-15),
        "scales": torch.optim.Adam([splats["scales"]], lr=5e-3, eps=1e-15),
        "quats": torch.optim.Adam([splats["quats"]], lr=1e-3, eps=1e-15),
        "opacities": torch.optim.Adam([splats["opacities"]], lr=5e-2, eps=1e-15),
        "colors": torch.optim.Adam([splats["colors"]], lr=2.5e-3, eps=1e-15),
    }
    return optimizers


def _render(splats, view, width, height):
    means = splats["means"]
    quats = F.normalize(splats["quats"], dim=-1)
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.sigmoid(splats["colors"])
    viewmats = view["w2c"].unsqueeze(0)  # [1,4,4]
    Ks = view["K"].unsqueeze(0)
    renders, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=False,
        absgrad=True,
        render_mode="RGB",
    )
    return renders[0], alphas[0], info  # H,W,3


@torch.no_grad()
def _orbit_c2w(center: Tensor, radius: float, n: int, elev_deg: float = 15.0) -> list[Tensor]:
    """Simple orbit of camera-to-world matrices looking at center."""
    poses = []
    elev = math.radians(elev_deg)
    for i in range(n):
        az = 2 * math.pi * i / n
        x = center[0] + radius * math.cos(elev) * math.cos(az)
        y = center[1] + radius * math.sin(elev)
        z = center[2] + radius * math.cos(elev) * math.sin(az)
        eye = torch.tensor([x, y, z], device=center.device, dtype=center.dtype)
        forward = F.normalize(center - eye, dim=0)
        up = torch.tensor([0.0, 1.0, 0.0], device=center.device, dtype=center.dtype)
        right = F.normalize(torch.linalg.cross(forward, up), dim=0)
        up = F.normalize(torch.linalg.cross(right, forward), dim=0)
        c2w = torch.eye(4, device=center.device, dtype=center.dtype)
        c2w[:3, 0] = right
        c2w[:3, 1] = -up  # OpenCV: y down
        c2w[:3, 2] = forward
        c2w[:3, 3] = eye
        poses.append(c2w)
    return poses


def train_gaussians(cfg: dict) -> dict:
    """Train Gaussians from stage-3 outputs; write checkpoint + orbit frames."""
    device = cfg["device"]
    vggt_dir = Path(cfg["paths"]["vggt_dir"])
    out_dir = Path(cfg["paths"]["gaussian_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    gcfg = cfg["gaussian"]
    iterations = int(gcfg["iterations"])
    ssim_w = float(gcfg.get("ssim_weight", 0.2))
    orbit_frames = int(gcfg.get("orbit_frames", 60))

    views, points, colors = _load_scene(vggt_dir, device, seed=cfg["seed"])
    h, w = views[0]["image"].shape[:2]
    cam_locs = torch.stack([v["c2w"][:3, 3] for v in views])
    scene_scale = (cam_locs.max(0).values - cam_locs.min(0).values).norm().item()
    scene_scale = max(scene_scale, 0.1)

    splats = _init_splats(points, colors, device)
    optimizers = _make_optimizers(splats, scene_scale)
    strategy = DefaultStrategy(
        verbose=False,
        refine_start_iter=min(200, iterations // 4),
        refine_stop_iter=max(iterations - 100, iterations // 2),
        reset_every=max(iterations // 2, 500),
        absgrad=True,
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    use_ssim = False
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        use_ssim = True
    except Exception:
        ssim_fn = None

    print(f"training gaussians: {len(splats['means'])} init pts, "
          f"{len(views)} views, {iterations} iters, scene_scale={scene_scale:.3f}")

    for step in range(iterations):
        view = views[step % len(views)]
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)

        render, alpha, info = _render(splats, view, w, h)
        gt = view["image"]
        l1 = F.l1_loss(render, gt)
        loss = l1
        if use_ssim and ssim_w > 0:
            # torchmetrics expects NCHW
            ssim_val = ssim_fn(
                render.permute(2, 0, 1).unsqueeze(0),
                gt.permute(2, 0, 1).unsqueeze(0),
            )
            loss = (1 - ssim_w) * l1 + ssim_w * (1 - ssim_val)

        strategy.step_pre_backward(splats, optimizers, state, step, info)
        loss.backward()
        strategy.step_post_backward(splats, optimizers, state, step, info)
        for opt in optimizers.values():
            opt.step()

        if step % 200 == 0 or step == iterations - 1:
            print(f"  step {step:5d}/{iterations}  loss={loss.item():.4f}  "
                  f"n={len(splats['means'])}")

    # Save checkpoint
    ckpt = {k: v.detach().cpu() for k, v in splats.items()}
    ckpt_path = out_dir / "gaussians.pt"
    torch.save(ckpt, ckpt_path)
    print(f"saved {ckpt_path}")

    # Orbit renders
    orbit_dir = out_dir / "orbit"
    orbit_dir.mkdir(parents=True, exist_ok=True)
    center = splats["means"].detach().mean(0)
    radius = max(scene_scale * 1.2, 0.5)
    K = views[0]["K"]
    with torch.no_grad():
        for i, c2w in enumerate(_orbit_c2w(center, radius, orbit_frames)):
            w2c = torch.linalg.inv(c2w)
            fake_view = {"w2c": w2c, "K": K}
            render, _, _ = _render(splats, fake_view, w, h)
            arr = (render.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(orbit_dir / f"frame_{i:03d}.png")
    print(f"saved {orbit_frames} orbit frames to {orbit_dir}")

    # Train-view comparison (first view)
    with torch.no_grad():
        render, _, _ = _render(splats, views[0], w, h)
        pred = (render.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        gt = (views[0]["image"].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(pred).save(out_dir / "render_view0.png")
        Image.fromarray(gt).save(out_dir / "gt_view0.png")

    stats = {
        "iterations": iterations,
        "num_gaussians": int(len(splats["means"])),
        "num_views": len(views),
        "image_size": [w, h],
        "scene_scale": scene_scale,
        "orbit_frames": orbit_frames,
        "checkpoint": str(ckpt_path),
    }
    save_json(stats, out_dir / "stats.json")
    return stats
