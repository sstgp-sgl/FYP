"""Extract a surface mesh from the VGGT point cloud via Open3D Poisson."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.io_utils import load_json, save_json


def camera_centers_from_json(cameras_json: str | Path) -> np.ndarray:
    """Return (N,3) camera centers in world coords from stage-3 cameras.json."""
    cams = load_json(cameras_json)["cameras"]
    centers = []
    for cam in cams:
        E = np.asarray(cam["extrinsic"], dtype=np.float64)  # world->cam [R|t]
        R, t = E[:3, :3], E[:3, 3]
        centers.append(-R.T @ t)
    return np.asarray(centers)


def extract_poisson_mesh(
    pointcloud_ply: str | Path,
    out_mesh: str | Path,
    depth: int = 9,
    density_quantile: float = 0.02,
    normal_radius_knn: int = 30,
    camera_centers: np.ndarray | None = None,
) -> dict:
    """Poisson reconstruct a mesh from a colored point cloud.

    Returns stats dict (vertex/triangle counts, paths).
    """
    import open3d as o3d

    pointcloud_ply = Path(pointcloud_ply)
    out_mesh = Path(out_mesh)
    out_mesh.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(pointcloud_ply))
    if len(pcd.points) == 0:
        raise ValueError(f"Empty point cloud: {pointcloud_ply}")

    # Downsample for tractable Poisson on large VGGT clouds.
    n = len(pcd.points)
    print(f"loaded point cloud: {n:,} points")
    target = 150_000
    if n > target:
        # Adaptive voxel size: grow until under target (cap iterations).
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = float(np.linalg.norm(bbox.get_extent()))
        voxel = max(extent / 200.0, 1e-4)
        for _ in range(12):
            down = pcd.voxel_down_sample(voxel_size=voxel)
            if len(down.points) <= target:
                pcd = down
                break
            voxel *= 1.4
        else:
            pcd = down
        print(f"voxel downsample: {n:,} -> {len(pcd.points):,} (voxel={voxel:.5f})")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_radius_knn)
    )
    # Orient normals toward the nearest camera. Much faster than the MST-based
    # orient_normals_consistent_tangent_plane, and correct here because every
    # point was observed from at least one camera.
    if camera_centers is not None and len(camera_centers):
        pts = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        diffs = pts[:, None, :] - camera_centers[None, :, :]  # (N, C, 3)
        nearest = np.argmin((diffs ** 2).sum(-1), axis=1)  # (N,)
        to_cam = camera_centers[nearest] - pts
        flip = (normals * to_cam).sum(-1) < 0
        normals[flip] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)
        print(f"oriented normals toward cameras ({flip.sum():,} flipped)")
    else:
        pcd.orient_normals_consistent_tangent_plane(k=normal_radius_knn)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    densities = np.asarray(densities)
    if len(densities) and density_quantile > 0:
        thresh = np.quantile(densities, density_quantile)
        vertices_to_remove = densities < thresh
        mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    # Transfer colors from nearest cloud points when available
    if pcd.has_colors():
        from scipy.spatial import cKDTree

        tree = cKDTree(np.asarray(pcd.points))
        _, idx = tree.query(np.asarray(mesh.vertices), k=1)
        cols = np.asarray(pcd.colors)[idx]
        mesh.vertex_colors = o3d.utility.Vector3dVector(cols)

    ok = o3d.io.write_triangle_mesh(str(out_mesh), mesh)
    if not ok:
        raise RuntimeError(f"Failed to write mesh to {out_mesh}")

    stats = {
        "source_ply": str(pointcloud_ply),
        "mesh_path": str(out_mesh),
        "num_vertices": int(len(mesh.vertices)),
        "num_triangles": int(len(mesh.triangles)),
        "poisson_depth": depth,
        "density_quantile": density_quantile,
    }
    print(f"mesh: {stats['num_vertices']:,} verts, "
          f"{stats['num_triangles']:,} tris → {out_mesh}")
    return stats


def run_mesh_stage(cfg: dict) -> dict:
    mesh_cfg = cfg.get("mesh", {})
    vggt_dir = Path(cfg["paths"]["vggt_dir"])
    mesh_dir = Path(cfg["paths"]["mesh_dir"])
    ply = vggt_dir / "pointcloud.ply"
    if not ply.exists():
        raise FileNotFoundError(f"Missing {ply}; run stage 3 first")

    cameras_json = vggt_dir / "cameras.json"
    centers = camera_centers_from_json(cameras_json) if cameras_json.exists() else None

    out = mesh_dir / "mesh.ply"
    stats = extract_poisson_mesh(
        ply,
        out,
        depth=int(mesh_cfg.get("poisson_depth", 9)),
        density_quantile=float(mesh_cfg.get("density_quantile", 0.02)),
        normal_radius_knn=int(mesh_cfg.get("normal_knn", 30)),
        camera_centers=centers,
    )
    save_json(stats, mesh_dir / "stats.json")
    return stats
