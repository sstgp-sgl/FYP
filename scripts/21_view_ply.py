#!/usr/bin/env python3
"""Launch Viser for a point-cloud PLY (used by the web workflow's Viser button).

Usage:
    python scripts/21_view_ply.py <pointcloud.ply> [cameras.json|None] [port]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    ply = sys.argv[1]
    cams = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "None" else None
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080

    from src.reconstruct.viser_viewer import visualize_ply

    visualize_ply(ply, cams, port=port)


if __name__ == "__main__":
    main()
