"""Stage 4: COLMAP export + minimal 3D Gaussian Splatting training.

Consumes (from stage 3):
    outputs/vggt/pointcloud.ply
    outputs/vggt/cameras.json
    outputs/views/*.png (+ optional foreground composite)

Produces:
    outputs/vggt/images/          - staged RGB frames matching VGGT intrinsics
    outputs/vggt/sparse/          - COLMAP text model (also sparse/0/)
    outputs/gaussian/gaussians.pt - trained Gaussian parameters
    outputs/gaussian/orbit/       - novel-view orbit frames
    outputs/gaussian/stats.json
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Prefer conda CUDA 12.1 nvcc over system CUDA 11.3 (needed for sm_89 / gsplat).
_conda_prefix = os.environ.get("CONDA_PREFIX", "")
if _conda_prefix and (Path(_conda_prefix) / "bin" / "nvcc").exists():
    os.environ["CUDA_HOME"] = _conda_prefix
    os.environ["PATH"] = str(Path(_conda_prefix) / "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.colmap_export import export_colmap_sparse
from src.gaussian_train import train_gaussians
from src.io_utils import load_config, require_files, save_json


def main():
    cfg = load_config()
    torch.manual_seed(cfg["seed"])
    vggt_dir = Path(cfg["paths"]["vggt_dir"])
    require_files(vggt_dir / "pointcloud.ply", vggt_dir / "cameras.json")

    print("exporting COLMAP sparse model...")
    export_colmap_sparse(vggt_dir, seed=cfg["seed"])

    print("training gaussians...")
    stats = train_gaussians(cfg)
    print(f"stage 4 done — {stats['num_gaussians']} gaussians, "
          f"{stats['orbit_frames']} orbit frames")


if __name__ == "__main__":
    main()
