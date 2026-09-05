"""Run the selected reconstruction backend and persist raw predictions."""
from __future__ import annotations

import json
from pathlib import Path

from src.reconstruct.task import Task, TaskContext, register


@register
class ReconstructTask(Task):
    name = "reconstruct"
    deps = ("preprocess",)

    def outputs(self, ctx: TaskContext) -> list[Path]:
        return [ctx.path("reconstruct", "predictions.npz")]

    def inputs(self, ctx: TaskContext) -> list[Path]:
        paths = [ctx.path("preprocess", "manifest.json")]
        art = ctx.artifacts.get("preprocess")
        if art:
            paths.extend(Path(p) for p in art.get("image_paths", []))
        return paths

    def load(self, ctx: TaskContext) -> dict | None:
        npz_path = ctx.path("reconstruct", "predictions.npz")
        images_path = ctx.path("reconstruct", "images.json")
        if not npz_path.exists():
            return None
        import numpy as np

        from src.reconstruct.backends.base import ReconstructionResult

        d = np.load(npz_path)
        images = json.loads(images_path.read_text()) if images_path.exists() else []
        result = ReconstructionResult(
            points=d["points"],
            colors=d["colors"],
            conf=d["conf"],
            extrinsic=d["extrinsic"],
            intrinsic=d["intrinsic"],
            image_paths=images,
        )
        return {"result": result, "backend": ctx.cfg.get("backend", "vggt")}

    def run(self, ctx: TaskContext) -> dict:
        from src.reconstruct.backends import get_backend

        image_paths = ctx.artifacts["preprocess"]["image_paths"]
        backend_name = ctx.cfg.get("backend", "vggt")
        backend = get_backend(backend_name, ctx.cfg)
        result = backend.reconstruct(image_paths, ctx)

        out_dir = ctx.path("reconstruct")
        out_dir.mkdir(parents=True, exist_ok=True)
        result.save_npz(out_dir / "predictions.npz")
        (out_dir / "images.json").write_text(json.dumps(result.image_paths, indent=2))

        print(f"reconstruct: {result.num_points:,} points, "
              f"{result.num_cameras} cameras (backend={backend_name})")
        return {"result": result, "backend": backend_name}
