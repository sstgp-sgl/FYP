"""Stage 1: normalize the input painting and segment the subject with SAM2.

Outputs (in outputs/foreground/):
    normalized.png  - 512x512 RGB, aspect-preserving resize + white padding
    pad_info.json   - scale/offset info to map coordinates back to the original
    mask.png        - grayscale feathered mask (0..255)
    foreground.png  - RGBA, normalized image with the feathered mask as alpha
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import (check_array, load_config, load_image_rgb,
                          resize_pad_square, save_image, save_json)
from src.segment import (clean_mask, content_rect_from_pad_info,
                         segment_center_subject)


def main():
    cfg = load_config()
    torch.manual_seed(cfg["seed"])
    size = cfg["image_size"]
    out_dir = Path(cfg["paths"]["foreground_dir"])

    img = load_image_rgb(cfg["paths"]["input_image"])
    normalized, pad_info = resize_pad_square(img, size)
    save_image(normalized, out_dir / "normalized.png")
    save_json(pad_info, out_dir / "pad_info.json")

    # The padding bars are background by construction; excluding them keeps the
    # border test from mistaking a backdrop that stops at the padding for a subject.
    content_rect = content_rect_from_pad_info(pad_info)
    mask = segment_center_subject(normalized, cfg, content_rect)
    check_array("mask", mask, shape=(size, size))
    mask = clean_mask(mask)

    # Gate 1 sanity check: the subject should occupy a plausible share of the frame.
    coverage = mask.mean()
    print(f"mask coverage: {coverage:.1%} of frame")
    if not 0.05 <= coverage <= 0.90:
        raise ValueError(f"Suspicious mask coverage {coverage:.1%}; "
                         "expected 5-90%. Check the prompt or input image.")

    # Feather the hard edge so compositing doesn't leave a jagged halo.
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    feathered = mask_img.filter(ImageFilter.GaussianBlur(cfg["segment"]["feather_px"]))
    save_image(feathered, out_dir / "mask.png")

    foreground = normalized.convert("RGBA")
    foreground.putalpha(feathered)
    save_image(foreground, out_dir / "foreground.png")
    print("stage 1 done")


if __name__ == "__main__":
    main()
