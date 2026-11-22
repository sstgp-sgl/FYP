#!/usr/bin/env python3
"""Headless SAM2 -> Zero123++ -> VGGT pipeline for a single image.

Same modules the web app uses (``src.segment`` / ``src.subjects`` /
``src.zero123`` / ``src.reconstruction``), but driven from the command line so it
can run on a compute node under Slurm. No HTTP server, no browser, no Viser.

Typical use::

    python scripts/vggt_headless.py \
        --image data/inbox/dog.jpg \
        --config configs/hpc.yaml \
        --out runs/vggt/dog \
        --prompt center

Stages can be rerun on their own::

    --skip-views          reuse <out>/views/{views.json,subjects_index.json}
    --skip-reconstruct    stop after view generation (inspect the 6/7 views)

Output layout (all paths are under --out)::

    subjects/<name>/mask.png, subject.png, subjects.json
    views/<name>/view_00.png ... grid_raw.png, views.json
    reconstruct/<name>/pointcloud.ply, cameras.json, stats.json
    run.json                       # resolved arguments + config, for the record
    summary.json                   # timings and final artifact paths
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- helpers
def parse_numbers(value: str, count: int, flag: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != count:
        raise SystemExit(f"{flag} expects {count} comma-separated numbers, got {value!r}")
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise SystemExit(f"{flag} expects numbers, got {value!r}") from exc


def build_prompts(args, size):
    """Return (boxes, points, labels) in pixel coords for the SAM2 call."""
    width, height = size
    mode = args.prompt
    if mode == "center":
        # Central 60% box plus a centre point: a decent default for single-subject
        # photos. SAM2 still returns 3 candidates; select_multimask_index picks.
        box = [0.2 * width, 0.2 * height, 0.8 * width, 0.8 * height]
        points, labels = [[0.5 * width, 0.5 * height]], [1]
    elif mode == "full":
        box = [0.0, 0.0, float(width - 1), float(height - 1)]
        points, labels = [], []
    elif mode == "box":
        if not args.box:
            raise SystemExit("--prompt box requires --box x0,y0,x1,y1")
        box = parse_numbers(args.box, 4, "--box")
        points, labels = [], []
    elif mode == "point":
        if not args.point:
            raise SystemExit("--prompt point requires --point x,y")
        box = []
        points, labels = [parse_numbers(args.point, 2, "--point")], [1]
    else:  # pragma: no cover - argparse choices prevent this
        raise SystemExit(f"unknown prompt mode {mode!r}")
    return ([box] if box else []), points, labels


def vram_report(tag: str) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 2**30
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"[vram] {tag}: allocated={used:.2f} GiB peak={peak:.2f} GiB", flush=True)
    except Exception:  # noqa: BLE001 - diagnostics must never break a run
        pass


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="single RGB image (JPEG/PNG)")
    parser.add_argument("--out", required=True, help="output root for this image")
    parser.add_argument("--config", default="configs/default.yaml", help="base YAML config")
    parser.add_argument("--prompt", choices=["center", "full", "box", "point"], default="center")
    parser.add_argument("--box", help="x0,y0,x1,y1 pixel box (with --prompt box)")
    parser.add_argument("--point", help="x,y pixel point (with --prompt point)")
    parser.add_argument("--name", help="subject name; defaults to the image stem")
    parser.add_argument("--views", type=int, choices=[6, 7], default=6, help="6 generated or 6+original")
    parser.add_argument("--image-size", type=int, default=None, help="override cfg image_size (default 512)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-views", action="store_true", help="reuse existing <out>/views")
    parser.add_argument("--skip-reconstruct", action="store_true", help="stop after views")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, load no model")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    import os

    os.chdir(REPO_ROOT)  # config paths are resolved against the repo root

    from src.config import load_config
    from src.io_utils import load_json

    image_path = Path(args.image).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    cfg["device"] = args.device
    if args.image_size:
        cfg["image_size"] = int(args.image_size)
    cfg["views"]["include_original"] = args.views == 7
    cfg["paths"]["subjects_dir"] = str(out_root / "subjects")
    cfg["paths"]["views_dir"] = str(out_root / "views")
    cfg["output_dir"] = str(out_root / "reconstruct")

    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")

    subject_name = args.name or image_path.stem
    # Mirror src.subjects.sanitize_name so this matches the directory
    # save_subjects() creates; otherwise the PLY path would not line up.
    subject_name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", subject_name).strip("_") or "subject"
    plan = {
        "image": str(image_path),
        "out": str(out_root),
        "config": str(args.config),
        "prompt": args.prompt,
        "views": args.views,
        "image_size": cfg["image_size"],
        "device": cfg["device"],
        "vggt_repo": cfg["paths"]["vggt_repo"],
        "sam2_checkpoint": cfg["segment"]["sam2_checkpoint"],
        "stages": ["views", "reconstruct"],
    }
    write_json(out_root / "run.json", plan)
    print(f"[plan] image={image_path}")
    print(f"[plan] out={out_root} prompt={args.prompt} views={args.views} size={cfg['image_size']}")
    print(f"[plan] vggt_repo={cfg['paths']['vggt_repo']}")
    print(f"[plan] sam2_ckpt={cfg['segment']['sam2_checkpoint']}")

    if args.dry_run:
        print("[plan] dry run: no model loaded")
        return 0

    import torch

    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in this environment; run on a GPU node or pass --device cpu")

    summary = {"image": str(image_path), "out": str(out_root), "stages": {}}
    started = time.monotonic()

    # ------------------------------------------------------------- stage 1
    if not args.skip_views:
        from PIL import Image
        import numpy as np

        t0 = time.monotonic()
        image = Image.open(image_path).convert("RGB")
        image_rgb = np.asarray(image)
        print(f"[stage1] image {image.size[0]}x{image.size[1]}")

        from src.segment import build_sam2_predictor, select_multimask_index
        from src.subjects import Subject, save_subjects, segment_subject

        print("[stage1] loading SAM2 ...")
        predictor = build_sam2_predictor(cfg)
        boxes, points, labels = build_prompts(args, image.size)
        subject = Subject(name=subject_name, boxes=boxes, points=points, labels=labels)
        masks, scores = segment_subject(predictor, image_rgb, subject)
        index = select_multimask_index(masks, scores, None)
        print(f"[stage1] SAM2 returned {len(masks)} masks, using #{index}")

        doc = save_subjects(
            cfg["paths"]["subjects_dir"],
            str(image_path),
            image_rgb,
            [{"name": subject_name, "mask": masks[index], "prompts": {"boxes": boxes, "points": points, "labels": labels}}],
        )
        vram_report("after SAM2")

        # Release SAM2 before Zero123++ loads; both would not fit comfortably.
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from src.zero123 import generate_views_for_subjects

        print("[stage1] generating views (Zero123++) ...")
        generate_views_for_subjects(doc, cfg["paths"]["subjects_dir"], cfg["paths"]["views_dir"], cfg)
        vram_report("after Zero123++")
        summary["stages"]["views"] = {"seconds": round(time.monotonic() - t0, 2)}

    if args.skip_reconstruct:
        print("[stage2] skipped (--skip-reconstruct)")
        write_json(out_root / "summary.json", summary)
        return 0

    # ------------------------------------------------------------- stage 2
    from src.reconstruction import reconstruct_subjects

    index_path = Path(cfg["paths"]["views_dir"]) / "subjects_index.json"
    if not index_path.is_file():
        raise SystemExit(f"missing {index_path}; run without --skip-views first")
    index_doc = load_json(index_path)

    t1 = time.monotonic()
    print("[stage2] running VGGT ...")
    reconstruct_subjects(index_doc, Path(cfg["output_dir"]), cfg)
    summary["stages"]["reconstruct"] = {"seconds": round(time.monotonic() - t1, 2)}
    vram_report("after VGGT")

    ply = Path(cfg["output_dir"]) / subject_name / "pointcloud.ply"
    summary["pointcloud"] = str(ply)
    summary["exists"] = ply.is_file()
    summary["total_seconds"] = round(time.monotonic() - started, 2)
    write_json(out_root / "summary.json", summary)

    print(f"[done] {ply} ({ply.stat().st_size / 2**20:.1f} MiB)" if ply.is_file() else f"[done] missing {ply}")
    print(f"[done] views: {Path(cfg['paths']['views_dir']) / subject_name}")
    print(f"[done] summary: {out_root / 'summary.json'}")
    return 0 if ply.is_file() else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
