"""Optional normalization: square resize+pad, and (for generated views) whiten bg."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.io_utils import resize_pad_square
from src.reconstruct.task import Task, TaskContext, register


def _whiten_uniform_background(arr: np.ndarray, max_std: int = 10,
                               per_channel_diff: int = 20) -> np.ndarray:
    """Replace a uniform background with white (mirrors stage 2)."""
    h, w = arr.shape[:2]
    cs = min(10, h // 4, w // 4)
    regions = [
        arr[0:cs, 0:cs], arr[0:cs, -cs:],
        arr[-cs:, 0:cs], arr[-cs:, -cs:],
    ]
    means = [r.mean(axis=(0, 1)) for r in regions
             if float(r.std(axis=(0, 1)).max()) <= max_std]
    if len(means) < 2:
        return arr
    bg_mean = np.mean(means, axis=0).astype(np.float32)
    is_bg = np.abs(arr.astype(np.float32) - bg_mean).max(axis=-1) < per_channel_diff
    out = arr.copy()
    out[is_bg] = 255
    return out


@register
class PreprocessTask(Task):
    name = "preprocess"
    deps = ("collect",)

    def outputs(self, ctx: TaskContext) -> list[Path]:
        outs = [ctx.path("preprocess", "manifest.json")]
        man = ctx.workdir / "preprocess" / "manifest.json"
        if man.exists():
            try:
                m = json.loads(man.read_text())
                if m.get("enabled", True):
                    outs.extend(Path(img["preprocessed"]) for img in m.get("images", []))
            except Exception:  # noqa: BLE001
                pass
        return outs

    def inputs(self, ctx: TaskContext) -> list[Path]:
        # Source photos, not just the collect manifest: collect may rerun without
        # rewriting the manifest (same path list, newer pixels).
        paths = [ctx.path("collect", "manifest.json")]
        art = ctx.artifacts.get("collect")
        if art:
            paths.extend(Path(p) for p in art.get("image_paths", []))
        else:
            man = ctx.path("collect", "manifest.json")
            if man.exists():
                try:
                    paths.extend(Path(p) for p in json.loads(man.read_text()).get("images", []))
                except Exception:  # noqa: BLE001
                    pass
        return paths

    def load(self, ctx: TaskContext) -> dict | None:
        p = ctx.path("preprocess", "manifest.json")
        if not p.exists():
            return None
        m = json.loads(p.read_text())
        if m.get("enabled", True):
            image_paths = [img["preprocessed"] for img in m["images"]]
        else:
            image_paths = list(m["images"])
        return {"image_paths": image_paths, "manifest_path": str(p)}

    def run(self, ctx: TaskContext) -> dict:
        pp = ctx.cfg.get("preprocess", {})
        image_paths = ctx.artifacts["collect"]["image_paths"]
        out_dir = ctx.path("preprocess")
        out_dir.mkdir(parents=True, exist_ok=True)

        if not pp.get("enabled", True):
            manifest = {"enabled": False, "images": image_paths}
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"preprocess: disabled, passing through {len(image_paths)} images")
            return {"image_paths": image_paths}

        size = int(pp.get("size", 512))
        fill = tuple(int(c) for c in pp.get("fill", [255, 255, 255]))
        whiten = bool(pp.get("whiten_background", False))

        new_paths: list[str] = []
        mapping: list[dict] = []
        for i, p in enumerate(image_paths):
            img = Image.open(p).convert("RGB")
            img, info = resize_pad_square(img, size, fill=fill)
            if whiten:
                img = Image.fromarray(_whiten_uniform_background(np.asarray(img)))
            out = out_dir / f"view_{i:02d}.png"
            img.save(out)
            new_paths.append(str(out))
            mapping.append({"source": p, "preprocessed": str(out), **info})

        manifest = {"enabled": True, "size": size, "whiten_background": whiten,
                    "images": mapping}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"preprocess: {len(new_paths)} images -> {size}x{size}")
        return {"image_paths": new_paths, "manifest_path": str(out_dir / "manifest.json")}
