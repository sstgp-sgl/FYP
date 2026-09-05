"""Tests for per-subject view framing / whitening (no GPU)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.workflow import frame_subject, whiten_uniform_background


def test_frame_subject_letterboxes_without_squashing():
    """A 2:1 subject must stay ~2:1 after framing, not be stretched to a square."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    arr = np.asarray(img).copy()
    arr[75:125, 50:150] = (200, 10, 10)  # 100x50 red rectangle
    img = Image.fromarray(arr)
    mask = Image.fromarray(((arr[:, :, 0] > 100).astype(np.uint8) * 255))

    out = frame_subject(img, mask, 64)
    assert out.size == (64, 64)
    a = np.asarray(out)
    ink = ~(a > 250).all(axis=-1)
    ys, xs = np.where(ink)
    assert len(xs) > 0
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    assert bw / bh > 1.5
    # letterbox: some rows stay white
    assert (a[0] > 250).all() or (a[:, 0] > 250).all()


def test_whiten_gray_corners():
    arr = np.full((64, 64, 3), 178, dtype=np.uint8)
    arr[20:44, 20:44] = 40
    out, ok = whiten_uniform_background(arr)
    assert ok
    assert (out[0, 0] == 255).all()
    assert (out[32, 32] == 40).all()


def test_whiten_falls_back_to_border_when_corners_occupied():
    rng = np.random.default_rng(0)
    arr = np.full((64, 64, 3), 178, dtype=np.uint8)
    noisy = rng.integers(20, 90, (8, 8, 3), dtype=np.uint8)
    arr[:8, :8] = noisy
    arr[:8, -8:] = noisy
    arr[-8:, :8] = noisy
    arr[-8:, -8:] = noisy
    out, ok = whiten_uniform_background(arr)
    assert ok
    assert (out[32, 32] == 255).all()
    assert not (out[2, 2] == 255).all()
