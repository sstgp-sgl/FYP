"""Tests for the torch/cv2-free parts of src.subjects (multi-subject selection)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.subjects import Subject, composite_on_white, mask_to_png_bytes, sanitize_name


def test_sanitize_name():
    assert sanitize_name("a horse!") == "a_horse"
    assert sanitize_name("   ") == "subject"
    assert sanitize_name("白衣女子") == "白衣女子"
    assert sanitize_name(None) == "subject"


def test_composite_on_white():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[2:5, 3:6] = [10, 20, 30]
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 3:6] = True
    out = composite_on_white(img, mask)
    assert out.shape == (8, 8, 3)
    assert (out[2:5, 3:6] == [10, 20, 30]).all()
    assert (out[0, 0] == 255).all()
    assert (out[7, 7] == 255).all()


def test_mask_to_png_bytes():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True
    data = mask_to_png_bytes(mask)
    img = Image.open(io.BytesIO(data))
    arr = np.asarray(img)
    assert img.size == (16, 16)
    assert arr[6, 6] == 255
    assert arr[0, 0] == 0


def test_subject_dataclass_defaults():
    s = Subject(name="horse")
    assert s.boxes == [] and s.points == [] and s.labels == []
