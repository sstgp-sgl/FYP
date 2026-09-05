"""Stage 2: generate 6 novel views with Zero123++ v1.2, then whiten background.

The model outputs one 640x960 image containing a 3x2 grid of views (each tile
320x320) at fixed cameras: azimuth 30/90/150/210/270/330 deg relative to the
input view, elevation alternating +20/-10 deg.

Each tile is upscaled to 512x512 and its uniform gray background is replaced
with pure white (255,255,255) so downstream stages (VGGT) see a consistent
white background like stage 1. No SAM2 segmentation is performed.

Outputs (in outputs/views/):
    grid_raw.png              - raw model output (still has gray bg), for debug
    view_00.png..view_05.png  - whitened views (subject on white bg), 512x512
    views.json                - per-view azimuth/elevation + generation settings
"""
import os
import sys
from pathlib import Path

# Must be set before diffusers/huggingface imports; huggingface.co is
# unreachable from this host, hf-mirror.com works.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import (check_array, load_config, require_files,
                          save_image, save_json)


def composite_on_background(rgba_path, gray):
    """Flatten the RGBA foreground onto a plain background."""
    fg = Image.open(rgba_path).convert("RGBA")
    bg = Image.new("RGBA", fg.size, (gray, gray, gray, 255))
    return Image.alpha_composite(bg, fg).convert("RGB")


def whiten_uniform_background(
    arr: np.ndarray,
    max_std: int = 10,
    per_channel_diff: int = 20,
) -> tuple[np.ndarray, bool]:
    """Replace a uniform gray background with pure white (255,255,255).

    Zero123++ tiles have a uniform gray background (≈178). Subject matter
    can bleed into one or more corners (e.g. the ear/shoulder in back views),
    so this function first identifies which corners are still uniform gray,
    then uses only those to estimate the background color.

    Args:
        arr: HxWx3 uint8 RGB array (the upscaled tile).
        max_std:  Per-channel std threshold for a corner to count as "uniform".
        per_channel_diff:  Max per-channel distance from the sampled bg mean
                  to still count as background.

    Returns:
        (result_array, was_modified)
    """
    h, w = arr.shape[:2]
    cs = min(10, h // 4, w // 4)  # corner sample size

    # Test each corner for uniformity
    corner_regions = [
        arr[0:cs, 0:cs],         # top-left
        arr[0:cs, -cs:],         # top-right
        arr[-cs:, 0:cs],         # bottom-left
        arr[-cs:, -cs:],         # bottom-right
    ]
    uniform_means = []
    for region in corner_regions:
        if float(region.std(axis=(0, 1)).max()) <= max_std:
            uniform_means.append(region.mean(axis=(0, 1)))

    if len(uniform_means) < 2:
        return arr.copy(), False

    bg_mean = np.mean(uniform_means, axis=0).astype(np.float32)

    diff = np.abs(arr.astype(np.float32) - bg_mean)
    is_bg = diff.max(axis=-1) < per_channel_diff

    result = arr.copy()
    result[is_bg] = 255
    return result, True


def main():
    cfg = load_config()
    views_cfg = cfg["views"]
    out_dir = Path(cfg["paths"]["views_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fg_path = Path(cfg["paths"]["foreground_dir"]) / "foreground.png"
    require_files(fg_path)

    bg_gray = cfg["segment"]["background_gray"]
    cond = composite_on_background(fg_path, bg_gray)

    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        views_cfg["model"],
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    ).to(cfg["device"])

    generator = torch.Generator(device=cfg["device"]).manual_seed(cfg["seed"])
    with torch.no_grad():
        result = pipe(
            cond,
            num_inference_steps=views_cfg["num_inference_steps"],
            guidance_scale=views_cfg["guidance_scale"],
            generator=generator,
        )
    grid = result.images[0]  # PIL image, 640 wide x 960 tall (2 cols x 3 rows)
    save_image(grid, out_dir / "grid_raw.png")
    check_array("grid", np.asarray(grid), shape=(960, 640, 3))

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tile = 320
    size = cfg["image_size"]
    azimuths = views_cfg["azimuths"]
    elevations = views_cfg["elevations"]
    meta = {"seed": cfg["seed"],
            "num_inference_steps": views_cfg["num_inference_steps"],
            "guidance_scale": views_cfg["guidance_scale"],
            "segmented": False,
            "background_gray": bg_gray,
            "views": []}
    for i in range(6):
        row, col = divmod(i, 2)
        box = (col * tile, row * tile, (col + 1) * tile, (row + 1) * tile)
        view = grid.crop(box).resize((size, size), Image.LANCZOS)

        # Whiten uniform gray background (no SAM2 segmentation needed)
        view_arr = np.asarray(view)
        whitened, was_modified = whiten_uniform_background(view_arr)
        if was_modified:
            bg_pct = (whitened == 255).all(axis=-1).mean()
            print(f"view_{i:02d}: background whitened ({bg_pct:.1%} uniform gray → white)")
            view = Image.fromarray(whitened)
        else:
            print(f"view_{i:02d}: background non-uniform, skipping whitening")

        name = f"view_{i:02d}.png"
        save_image(view, out_dir / name)
        meta["views"].append({"file": name,
                              "azimuth": azimuths[i],
                              "elevation": elevations[i]})
    save_json(meta, out_dir / "views.json")
    print("stage 2 done")


if __name__ == "__main__":
    main()
