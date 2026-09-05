"""Tests for 03b VGGT Viser diagnostic (views only, no foreground)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.io_utils import list_view_images


def test_list_view_images_ignores_extra_files(tmp_path: Path):
    views = []
    for i in range(6):
        name = f"view_{i:02d}.png"
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / name)
        views.append({"file": name, "azimuth": i * 60, "elevation": 0})
    (tmp_path / "views.json").write_text(json.dumps({"views": views}))
    # Extra files that must NOT be picked up by list_view_images
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "foreground.png")
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / "grid_raw.png")

    paths = list_view_images(tmp_path)
    assert len(paths) == 6
    assert all(p.name.startswith("view_") for p in paths)
    assert not any("foreground" in p.name for p in paths)
