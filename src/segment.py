"""SAM2 subject segmentation helpers shared by stage 1 and stage 2."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

# Prefer masks whose subject coverage looks plausible after cleanup.
_MIN_COVERAGE = 0.15
_MAX_COVERAGE = 0.85
# A region covering more than this share of the frame border is the background.
_BORDER_IS_BACKGROUND = 0.5
# Generated tiles carry a thin ragged ring at the frame edge that masks exclude,
# so the border is sampled slightly inside the frame.
_BORDER_INSET_FRAC = 0.03


def build_sam2_predictor(cfg):
    """Load SAM2 image predictor onto the configured device (once per stage)."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(
        cfg["segment"]["sam2_config"],
        cfg["segment"]["sam2_checkpoint"],
        device=cfg["device"],
    )
    return SAM2ImagePredictor(model)


def restrict_to_content(mask: np.ndarray, content_rect=None) -> np.ndarray:
    """Drop mask pixels lying in the letterbox padding added by resize_pad_square."""
    mask = mask.astype(bool)
    if content_rect is None:
        return mask
    x0, y0, x1, y1 = content_rect
    inside = np.zeros_like(mask)
    inside[y0:y1, x0:x1] = True
    return mask & inside


def border_coverage(mask: np.ndarray, content_rect=None) -> float:
    """Fraction of the content-area border covered by the mask.

    ``content_rect`` excludes the padding bars, without which a background that
    stops at the padding looks like it barely touches the frame.
    """
    m = mask.astype(bool)
    if content_rect is not None:
        x0, y0, x1, y1 = content_rect
        m = m[y0:y1, x0:x1]
    inset = int(round(_BORDER_INSET_FRAC * min(m.shape))) if m.size else 0
    if inset and min(m.shape) > 2 * inset + 2:
        m = m[inset:-inset, inset:-inset]
    if m.size == 0:
        return 0.0
    border = np.concatenate([m[0, :], m[-1, :], m[:, 0], m[:, -1]])
    return float(border.mean())


def orient_mask_to_subject(mask: np.ndarray, content_rect=None) -> np.ndarray:
    """Ensure True marks the subject, inverting when SAM2 selected the backdrop.

    Subjects sit inside the frame; backgrounds hug its border. Border contact is
    reliable where color statistics are not — a lightly textured wall and a
    smoothly painted animal can have near-identical variance.
    """
    mask = mask.astype(bool)
    if not mask.any() or mask.all():
        return mask
    if border_coverage(mask, content_rect) > _BORDER_IS_BACKGROUND:
        return ~mask
    return mask


def select_multimask_index(
    masks: np.ndarray,
    scores: np.ndarray,
    content_rect=None,
) -> int:
    """Pick multimask index using score × post-clean coverage (plausible only)."""
    coverages = []
    for m in masks:
        m_bool = orient_mask_to_subject(m.astype(bool), content_rect)
        m_bool = clean_mask(restrict_to_content(m_bool, content_rect))
        coverages.append(float(m_bool.mean()))
    coverages = np.asarray(coverages)

    plausible = [
        i for i, c in enumerate(coverages)
        if _MIN_COVERAGE <= c <= _MAX_COVERAGE
    ]
    if plausible:
        best = max(plausible, key=lambda i: float(scores[i]) * coverages[i])
    else:
        # An empty or full mask is never a usable subject, even as a fallback.
        usable = [i for i, c in enumerate(coverages) if 0.0 < c < 1.0]
        pool = usable or list(range(len(coverages)))
        best = min(pool, key=lambda i: abs(coverages[i] - 0.4))
    print(
        f"mask scores: {np.round(scores.astype(float), 3).tolist()}, "
        f"clean_cov: {np.round(coverages, 3).tolist()}, "
        f"using mask {best}"
    )
    return best


def predict_box_mask(predictor, image_rgb, cfg, content_rect=None) -> np.ndarray:
    """Run SAM2 with the configured fractional box prompt; return bool [H,W]."""
    arr = np.asarray(image_rgb)  # [H,W,3] uint8
    h, w = arr.shape[:2]
    fx0, fy0, fx1, fy1 = cfg["segment"]["box"]
    box = np.array([fx0 * w, fy0 * h, fx1 * w, fy1 * h])
    with torch.no_grad(), torch.autocast(cfg["device"], dtype=torch.float16):
        predictor.set_image(arr)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )
    best = select_multimask_index(masks, scores, content_rect)
    oriented = orient_mask_to_subject(masks[best].astype(bool), content_rect)
    return restrict_to_content(oriented, content_rect)


def predict_mask_sam2(
    predictor,
    image_rgb,
    points=None,
    point_labels=None,
    boxes=None,
    multimask_output=True,
):
    """Run SAM2 with arbitrary prompts (points and/or a box).

    Args:
        points: (N,2) pixel coords.
        point_labels: (N,) int, 1=foreground, 0=background (default all 1).
        boxes: (1,4) xyxy box in pixels (or (K,4)).
        multimask_output: return 3 candidate masks instead of 1.

    Returns:
        (masks (K,H,W) bool, scores (K,) float32) — caller picks a candidate.
    """
    import contextlib

    if (points is None or len(points) == 0) and (boxes is None or len(boxes) == 0):
        raise ValueError("predict_mask_sam2 needs at least one point or box prompt")

    kwargs = {"multimask_output": bool(multimask_output)}
    if points is not None and len(points):
        kwargs["point_coords"] = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if point_labels is None:
            point_labels = [1] * len(points)
        kwargs["point_labels"] = np.asarray(point_labels, dtype=np.int32).reshape(-1)
    if boxes is not None and len(boxes):
        kwargs["box"] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (
        torch.autocast(device, dtype=torch.float16)
        if device == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), ctx:
        masks, scores, _ = predictor.predict(**kwargs)
    return masks.astype(bool), np.asarray(scores, dtype=np.float32)


def content_rect_from_pad_info(pad_info) -> tuple[int, int, int, int]:
    """Content box (x0, y0, x1, y1) of a resize_pad_square canvas."""
    ox, oy = pad_info["offset_x"], pad_info["offset_y"]
    rw, rh = pad_info["resized_size"]
    return int(ox), int(oy), int(ox + rw), int(oy + rh)


def segment_center_subject(image_rgb, cfg, content_rect=None) -> np.ndarray:
    """Convenience: build predictor, run one box-prompt segmentation."""
    predictor = build_sam2_predictor(cfg)
    return predict_box_mask(predictor, image_rgb, cfg, content_rect)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component and fill interior holes.

    Hole fill is border-aware so subjects that touch the frame edge (common in
    Zero123++ tiles) are not flooded into a near-full mask. Border pixels are
    treated as background only while discovering exterior components — the
    returned mask keeps real subject pixels on the frame edge.
    """
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask_u8 = (labels == largest).astype(np.uint8)

    # Temporary: break frame-edge mask rings so exterior bg is reachable.
    border = 2
    work = mask_u8.copy()
    work[:border, :] = 0
    work[-border:, :] = 0
    work[:, :border] = 0
    work[:, -border:] = 0

    inv = (1 - work).astype(np.uint8)
    n_bg, bg_labels, _, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    border_labels = set(
        np.unique(
            np.concatenate(
                [
                    bg_labels[0, :],
                    bg_labels[-1, :],
                    bg_labels[:, 0],
                    bg_labels[:, -1],
                ]
            )
        ).tolist()
    )
    filled = mask_u8.copy()
    for lab in range(1, n_bg):
        if lab not in border_labels:
            # Map hole pixels from the work-space labels onto the real mask.
            # Only fill where work said background (interior pockets).
            hole = (bg_labels == lab) & (work == 0)
            filled[hole] = 1
    return filled.astype(bool)


def composite_rgb_on_background(
    image_rgb: Image.Image | np.ndarray,
    mask: np.ndarray,
    gray: int,
) -> Image.Image:
    """Keep masked subject pixels; fill the rest with a constant gray/white RGB."""
    arr = np.asarray(image_rgb).copy()
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {arr.shape}")
    if mask.shape != arr.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != image {arr.shape[:2]}")
    out = np.full_like(arr, int(gray))
    out[mask] = arr[mask]
    return Image.fromarray(out)
