"""Stage 5: extract a surface mesh from the VGGT point cloud (Open3D Poisson).

Consumes:
    outputs/vggt/pointcloud.ply

Produces:
    outputs/mesh/mesh.ply
    outputs/mesh/stats.json

Note: This uses the stage-3 point cloud directly (reliable without a perfect
Gaussian fit). Orbit renders from stage 4 are available for visualization
but not required here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import load_config, require_files
from src.mesh_extract import run_mesh_stage


def main():
    cfg = load_config()
    require_files(Path(cfg["paths"]["vggt_dir"]) / "pointcloud.ply")
    stats = run_mesh_stage(cfg)
    print(f"stage 5 done — mesh at {stats['mesh_path']}")


if __name__ == "__main__":
    main()
