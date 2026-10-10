"""PLY and camera inspection launched by the Web Viser button."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _require_viser():
    try:
        import viser  # noqa: F401

        return viser
    except ImportError as e:
        raise RuntimeError(
            "viser is not installed; run `pip install viser` to use visualization"
        ) from e


def _rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z], dtype=np.float64)


def _load_colored_ply(ply_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (points, colors) from an ASCII or binary PLY.

    Historical results may use binary PLY; current results use ASCII. Try the dependency-free ASCII reader first, then fall back to
    trimesh when it is installed.
    """
    from src.pointcloud_io import read_ply

    try:
        return read_ply(ply_path)
    except (UnicodeDecodeError, ValueError):
        pass

    import trimesh

    mesh = trimesh.load(str(ply_path), process=False)
    points = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8)
    return points, colors


def visualize_ply(
    ply_path: str | Path,
    cameras_json: str | Path | None = None,
    port: int = 8080,
) -> None:
    """Show an exported colored point cloud (+ camera frames) in Viser."""
    viser = _require_viser()

    points, colors = _load_colored_ply(ply_path)
    if len(points) == 0:
        raise ValueError(f"empty point cloud: {ply_path}")

    server = viser.ViserServer(port=port)
    # Normalize colors to float [0,1] if needed.
    cols = colors.astype(np.float32)
    if cols.max() > 1.0:
        cols /= 255.0
    server.scene.add_point_cloud(
        "reconstruction/points",
        points=points,
        colors=cols,
        point_size=0.004,
        point_shape="circle",
    )

    cams = None
    if cameras_json is not None and Path(cameras_json).exists():
        import json

        cams = json.loads(Path(cameras_json).read_text())["cameras"]
        for i, cam in enumerate(cams):
            E = np.asarray(cam["extrinsic"], dtype=np.float64)  # world→cam [R|t]
            R, t = E[:3, :3], E[:3, 3]
            center = -R.T @ t
            c2w_R = R.T
            server.scene.add_frame(
                f"cameras/{i:02d}",
                wxyz=_rotmat_to_quat_wxyz(c2w_R),
                position=center,
                axes_length=0.15,
                axes_radius=0.008,
            )

    print("=" * 60)
    print(f"   📡  Viser: http://localhost:{port}")
    print(f"   🧊  points: {len(points):,}  cameras: {len(cams) if cams else 0}")
    print("=" * 60)
    print("   Ctrl+C to exit\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopping Viser")



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Web point-cloud viewer")
    parser.add_argument("ply")
    parser.add_argument("--cameras")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    visualize_ply(args.ply, args.cameras, port=args.port)


if __name__ == "__main__":
    main()
