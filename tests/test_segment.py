"""Unit tests for shared segmentation helpers (no SAM2 weights)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.segment import clean_mask, composite_rgb_on_background


def test_clean_mask_keeps_largest_and_fills_hole():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True         # large blob
    mask[8:11, 8:11] = False         # hole inside
    mask[0:2, 0:2] = True            # tiny corner speck (isolated)
    cleaned = clean_mask(mask)
    assert not cleaned[0, 0]
    assert cleaned[9, 9]  # hole filled
    assert cleaned[6, 6]


def test_clean_mask_subject_touching_border_not_flooded():
    mask = np.zeros((20, 20), dtype=bool)
    mask[10:, 5:15] = True  # subject touches bottom edge
    cleaned = clean_mask(mask)
    assert cleaned[15, 10]
    assert not cleaned[0, 0]
    assert cleaned.mean() < 0.6


def test_select_multimask_prefers_plausible_coverage():
    from src.segment import select_multimask_index

    masks = np.zeros((3, 40, 40), dtype=np.float32)
    masks[0, 0:2, 0:2] = 1.0          # speck
    masks[1, 8:32, 8:32] = 1.0        # subject
    masks[2, :, :] = 1.0              # full frame
    scores = np.array([0.99, 0.50, 0.10], dtype=np.float32)
    assert select_multimask_index(masks, scores) == 1


def test_select_multimask_never_falls_back_to_empty_mask():
    """Regression: an empty candidate is nearer 0.4 than a 0.92 backdrop."""
    from src.segment import select_multimask_index

    masks = np.zeros((3, 40, 40), dtype=np.float32)
    # candidate 0 stays empty; 1 and 2 are backdrops that survive as near-full
    masks[1, 2:38, 2:38] = 1.0
    masks[2, 1:39, 1:39] = 1.0
    scores = np.array([0.02, 0.80, 0.92], dtype=np.float32)
    assert select_multimask_index(masks, scores) != 0


def test_border_coverage_ignores_thin_edge_ring():
    """Generated tiles leave a ragged ring the backdrop mask does not cover."""
    from src.segment import border_coverage

    mask = np.zeros((512, 512), dtype=bool)
    mask[12:500, 12:500] = True   # backdrop stopping short of the frame edge
    assert border_coverage(mask) > 0.5
    assert mask[0, :].mean() == 0.0  # invisible to a naive edge-row test


def test_orient_mask_inverts_background_hugging_border():
    from src.segment import orient_mask_to_subject

    wrong = np.ones((8, 8), dtype=bool)   # backdrop selected
    wrong[2:6, 2:6] = False               # subject hole
    fixed = orient_mask_to_subject(wrong)
    assert fixed[4, 4]
    assert not fixed[0, 0]


def test_orient_mask_keeps_centred_subject():
    from src.segment import orient_mask_to_subject

    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    kept = orient_mask_to_subject(mask)
    assert kept[8, 8]
    assert not kept[0, 0]


def test_orient_mask_uses_content_rect_to_ignore_padding():
    """Regression: a backdrop that stops at the letterbox padding is background.

    Without the content rect this mask touches only 2 of 4 outer edges, so it
    looks like a subject and the polarity is left inverted.
    """
    from src.segment import border_coverage, orient_mask_to_subject

    mask = np.zeros((20, 20), dtype=bool)
    mask[:, 5:15] = True        # backdrop spanning the un-padded content column
    mask[6:14, 7:13] = False    # subject cut out of it
    content = (5, 0, 15, 20)

    assert border_coverage(mask) < 0.5          # misleading on the full frame
    assert border_coverage(mask, content) > 0.5  # decisive on the content area

    assert not orient_mask_to_subject(mask)[10, 10]
    assert orient_mask_to_subject(mask, content)[10, 10]


def test_restrict_to_content_drops_padding_pixels():
    from src.segment import restrict_to_content

    mask = np.ones((10, 10), dtype=bool)
    out = restrict_to_content(mask, (2, 0, 8, 10))
    assert out[5, 5]
    assert not out[5, 0]
    assert not out[5, 9]


def test_content_rect_from_pad_info():
    from src.segment import content_rect_from_pad_info

    assert content_rect_from_pad_info(
        {"offset_x": 82, "offset_y": 0, "resized_size": [348, 512]}
    ) == (82, 0, 430, 512)


def test_composite_rgb_on_background_whitens_outside_mask():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[:, :] = (10, 20, 30)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    out = np.asarray(composite_rgb_on_background(rgb, mask, gray=255))
    assert (out[0, 0] == 255).all()
    assert (out[1, 1] == (10, 20, 30)).all()
