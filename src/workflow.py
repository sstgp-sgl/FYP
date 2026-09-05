"""Shared per-subject pipeline logic: views (Zero123++) + reconstruction (VGGT).

Used by:
  - scripts/11_generate_subject_views.py   (CLI)
  - scripts/12_reconstruct_subjects.py     (CLI)
  - scripts/10_subjects_web.py             (web workflow, progress callbacks)

``progress`` callbacks receive (idx, total, stage, name, info) where stage is
"views" or "reconstruct" and info carries {"status": running|done|error, ...}.
"""
from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

# HuggingFace 下载镜像必须在任何 HF 库导入前设置（本机直连 huggingface.co 不通）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _border_ring(arr: np.ndarray, width: int) -> np.ndarray:
    """Flattened pixels of the ``width``-px frame around the image edge."""
    h, w = arr.shape[:2]
    width = max(1, min(int(width), h // 2, w // 2))
    if h < 2 or w < 2:
        return arr.reshape(-1, 3)
    parts = [arr[:width].reshape(-1, 3), arr[-width:].reshape(-1, 3)]
    inner = h - 2 * width
    if inner > 0:
        parts.append(arr[width:-width, :width].reshape(-1, 3))
        parts.append(arr[width:-width, -width:].reshape(-1, 3))
    return np.concatenate(parts)


def whiten_uniform_background(
    arr: np.ndarray,
    max_std: int = 10,
    per_channel_diff: int = 20,
) -> tuple[np.ndarray, bool]:
    """Replace Zero123++'s uniform gray tile background with pure white.

    Corners are sampled first. When the subject bleeds into three or more
    corners the estimate falls back to the median of the whole border ring,
    which only counts as background if most of the ring agrees with it.
    """
    h, w = arr.shape[:2]
    cs = min(10, h // 4, w // 4)
    corner_regions = [
        arr[0:cs, 0:cs], arr[0:cs, -cs:],
        arr[-cs:, 0:cs], arr[-cs:, -cs:],
    ]
    uniform_means = [
        r.mean(axis=(0, 1))
        for r in corner_regions
        if float(r.std(axis=(0, 1)).max()) <= max_std
    ]
    if len(uniform_means) >= 2:
        bg_mean = np.mean(uniform_means, axis=0).astype(np.float32)
    else:
        ring = _border_ring(arr, max(1, min(h, w) // 32)).astype(np.float32)
        bg_mean = np.median(ring, axis=0)
        agree = (np.abs(ring - bg_mean).max(axis=-1) < per_channel_diff).mean()
        if agree < 0.6:
            return arr.copy(), False

    is_bg = np.abs(arr.astype(np.float32) - bg_mean).max(axis=-1) < per_channel_diff
    result = arr.copy()
    result[is_bg] = 255
    return result, True


def _save_image(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def frame_subject(img: Image.Image, mask, size: int, pad_frac: float = 0.06) -> Image.Image:
    """Crop to the subject's mask bbox and letterbox onto a square white canvas.

    Both Zero123++ and the identity-pose anchor view consume this result, so the
    subject occupies the same scale and position in all views VGGT receives.
    Padding (rather than a plain resize to ``size x size``) keeps the aspect
    ratio; squashing it would make the anchor view geometrically inconsistent
    with the generated ones.
    """
    from src.io_utils import resize_pad_square

    img = img.convert("RGB")
    if mask is not None and np.asarray(mask).shape[:2] == (img.size[1], img.size[0]):
        m = np.asarray(mask) > 127  # mask.png is 0/255
        ys, xs = np.where(m)
        if len(xs) > 0:
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            pad = max(1, int(pad_frac * max(x1 - x0, y1 - y0)))
            x0 = max(0, x0 - pad)
            x1 = min(img.size[0], x1 + pad)
            y0 = max(0, y0 - pad)
            y1 = min(img.size[1], y1 + pad)
            img = img.crop((x0, y0, x1, y1))
    framed, _ = resize_pad_square(img, size, fill=(255, 255, 255))
    return framed


def generate_subject_views(
    cond_image: Image.Image,
    pipe,
    cfg: dict,
    out_dir: Path,
    seed: int,
    name: str,
    prompts: dict,
    mask=None,
) -> Path:
    """Run Zero123++ on one subject; write views + views.json; return out_dir.

    ``cond_image`` is first framed (bbox crop + square letterbox) so the subject
    fills the frame consistently. That same framed image is what Zero123++ sees
    and, when ``cfg.views.include_original`` is true, what gets written as
    ``view_06.png`` — the identity-pose (azimuth 0°, elevation 0°) anchor view.
    Feeding Zero123++ the uncropped cut-out while cropping only the anchor made
    the two disagree on scale by up to 10x, which VGGT cannot register.
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    size = cfg["image_size"]
    cond_framed = frame_subject(cond_image, mask, size)

    generator = torch.Generator(device=cfg["device"]).manual_seed(seed)
    with torch.no_grad():
        result = pipe(
            cond_framed,
            num_inference_steps=cfg["views"]["num_inference_steps"],
            guidance_scale=cfg["views"]["guidance_scale"],
            generator=generator,
        )
    grid = result.images[0]  # 640 x 960 (2 cols x 3 rows)
    _save_image(grid, out_dir / "grid_raw.png")

    tile = 320
    azimuths = cfg["views"]["azimuths"]
    elevations = cfg["views"]["elevations"]
    meta = {
        "subject": name,
        "prompts": prompts,
        "seed": seed,
        "num_inference_steps": cfg["views"]["num_inference_steps"],
        "guidance_scale": cfg["views"]["guidance_scale"],
        "background_gray": cfg["segment"]["background_gray"],
        "views": [],
    }
    not_whitened = []
    for i in range(6):
        row, col = divmod(i, 2)
        box = (col * tile, row * tile, (col + 1) * tile, (row + 1) * tile)
        view = grid.crop(box).resize((size, size), Image.LANCZOS)

        view_arr = np.asarray(view)
        whitened, was_modified = whiten_uniform_background(view_arr)
        if was_modified:
            view = Image.fromarray(whitened)
        else:
            not_whitened.append(i)

        fname = f"view_{i:02d}.png"
        _save_image(view, out_dir / fname)
        meta["views"].append(
            {"file": fname, "azimuth": azimuths[i], "elevation": elevations[i],
             "whitened": was_modified}
        )
    if not_whitened:
        # A non-white tile background is unprojected as a full-frame plane and
        # pollutes the point cloud, so make it loud rather than silent.
        print(f"[subject] {name}: WARNING background not whitened for views "
              f"{not_whitened} — those frames will add a background plane; "
              f"consider re-running with a different seed")
    meta["views_not_whitened"] = not_whitened

    # 原图作为第 7 张基准视角（真实纹理锚点），与条件图完全同构图
    if cfg["views"].get("include_original", True):
        _save_image(cond_framed, out_dir / "view_06.png")
        meta["views"].append(
            {"file": "view_06.png", "azimuth": 0, "elevation": 0, "source": "original"}
        )
        print(f"[subject] {name}: original input added as view_06 (azimuth 0, elevation 0)")
    from src.io_utils import save_json

    save_json(meta, out_dir / "views.json")
    print(f"[subject] {name}: {len(meta['views'])} views → {out_dir}")
    return out_dir


def generate_views_for_subjects(
    subjects_doc: dict,
    subjects_dir: str | Path,
    views_root: str | Path,
    cfg: dict,
    progress=None,
    stop_on_error: bool = True,
) -> list[dict]:
    """Load Zero123++ once and loop all subjects. Returns the index list."""
    import torch
    from diffusers import DiffusionPipeline

    from src.io_utils import save_json

    subjects = subjects_doc.get("subjects", [])
    if not subjects:
        raise ValueError("subjects.json has no subjects")

    pipe = DiffusionPipeline.from_pretrained(
        cfg["views"]["model"],
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    ).to(cfg["device"])
    print(f"[zero123++] loaded {cfg['views']['model']} on {cfg['device']}")

    subjects_dir = Path(subjects_dir)
    views_root = Path(views_root)
    index: list[dict] = []
    try:
        for i, subj in enumerate(subjects):
            name = subj["name"]
            if progress:
                progress(i, len(subjects), "views", name, {"status": "running"})
            try:
                subject_png = subjects_dir / subj["subject_file"]
                if not subject_png.exists():
                    raise FileNotFoundError(f"missing subject cut-out: {subject_png}")
                mask = None
                mask_path = subjects_dir / subj["mask_file"]
                if mask_path.exists():
                    mask = Image.open(mask_path).convert("L")
                cond = Image.open(subject_png).convert("RGB")
                out_dir = generate_subject_views(
                    cond, pipe, cfg, views_root / name,
                    seed=cfg["seed"] + i, name=name, prompts=subj.get("prompts", {}),
                    mask=mask,
                )
                index.append(
                    {"name": name, "views_dir": str(out_dir), "prompts": subj.get("prompts", {})}
                )
                if progress:
                    progress(i, len(subjects), "views", name, {"status": "done"})
            except Exception as e:  # noqa: BLE001
                if progress:
                    progress(i, len(subjects), "views", name, {"status": "error", "error": str(e)})
                if stop_on_error:
                    raise
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json({"subjects": index}, views_root / "subjects_index.json")
    return index


def reconstruct_subjects(
    index_doc: dict,
    base_out: str | Path,
    cfg: dict,
    progress=None,
    force: bool = False,
    stop_on_error: bool = True,
) -> None:
    """Loop subjects through the reconstruction framework (pointmap)."""
    from src.reconstruct import Pipeline, TaskContext, get_task

    pipeline = Pipeline(
        [get_task(n) for n in ("collect", "preprocess", "reconstruct", "filter", "export")]
    )
    subjects = index_doc.get("subjects", [])
    base_out = Path(base_out)

    for i, s in enumerate(subjects):
        name = s["name"]
        if progress:
            progress(i, len(subjects), "reconstruct", name, {"status": "running"})
        buf = io.StringIO()
        sub_cfg = dict(cfg)
        sub_cfg["output_dir"] = str(base_out / name)
        sub_cfg["input"] = {
            "mode": "folder",
            "folder": s["views_dir"],
            "glob": "view_*.png",
        }
        ctx = TaskContext(
            cfg=sub_cfg,
            workdir=Path(sub_cfg["output_dir"]) / ".tasks",
            force=force,
        )
        try:
            with redirect_stdout(buf):
                pipeline.run(ctx)
            if progress:
                progress(
                    i, len(subjects), "reconstruct", name,
                    {"status": "done", "log": buf.getvalue()},
                )
        except Exception as e:  # noqa: BLE001
            if progress:
                progress(
                    i, len(subjects), "reconstruct", name,
                    {"status": "error", "error": str(e), "log": buf.getvalue()},
                )
            if stop_on_error:
                raise
