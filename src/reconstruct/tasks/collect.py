"""Collect input images (generated views, a folder, or an explicit list)."""
from __future__ import annotations

import json
from pathlib import Path

from src.reconstruct.task import Task, TaskContext, register, write_if_changed

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _glob_images(folder: Path, pattern: str) -> list[Path]:
    files = [p for p in folder.glob(pattern) if p.suffix.lower() in _IMAGE_EXTS]
    return sorted(files, key=lambda p: p.name)


@register
class CollectTask(Task):
    name = "collect"
    deps = ()

    def outputs(self, ctx: TaskContext) -> list[Path]:
        return [ctx.path("collect", "manifest.json")]

    def inputs(self, ctx: TaskContext) -> list[Path]:
        """The source images, so regenerated views invalidate a cached run."""
        try:
            return self._collect(ctx, ctx.cfg.get("input", {}).get("mode", "generated"))
        except Exception:  # noqa: BLE001 - a broken source is run()'s problem
            return []

    def is_complete(self, ctx: TaskContext) -> bool:
        """Also reject a cached manifest that lists a different set of images."""
        if not super().is_complete(ctx):
            return False
        try:
            cached = json.loads(ctx.path("collect", "manifest.json").read_text())["images"]
        except Exception:  # noqa: BLE001
            return False
        return cached == [str(p) for p in self.inputs(ctx)]

    def load(self, ctx: TaskContext) -> dict | None:
        p = ctx.path("collect", "manifest.json")
        if not p.exists():
            return None
        m = json.loads(p.read_text())
        return {"image_paths": m["images"], "manifest_path": str(p), "mode": m.get("mode")}

    def run(self, ctx: TaskContext) -> dict:
        mode = ctx.cfg.get("input", {}).get("mode", "generated")
        paths = self._collect(ctx, mode)
        if len(paths) < 2:
            raise ValueError(
                f"multi-view reconstruction needs >=2 images, got {len(paths)} "
                f"(input.mode={mode!r}); add more photos or switch input.mode"
            )
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(f"collected image missing: {p}")

        manifest_path = ctx.path("collect", "manifest.json")
        manifest = {"mode": mode, "num_images": len(paths), "images": [str(p) for p in paths]}
        write_if_changed(manifest_path, json.dumps(manifest, indent=2))
        print(f"collect: {len(paths)} images (mode={mode})")
        return {
            "image_paths": [str(p) for p in paths],
            "manifest_path": str(manifest_path),
            "mode": mode,
        }

    def _collect(self, ctx: TaskContext, mode: str) -> list[Path]:
        cfg = ctx.cfg
        if mode == "generated":
            from src.io_utils import list_view_images

            return list_view_images(cfg["paths"]["views_dir"])
        if mode == "folder":
            folder = Path(cfg["input"]["folder"])
            pattern = cfg["input"].get("glob", "*")
            if not folder.is_dir():
                raise FileNotFoundError(f"input.folder is not a directory: {folder}")
            return _glob_images(folder, pattern)
        if mode == "files":
            return [Path(p) for p in cfg["input"].get("files", [])]
        raise ValueError(
            f"unknown input.mode {mode!r}; use 'generated' | 'folder' | 'files'"
        )
