"""VGGT-1B reconstruction backend (GPU)."""
from __future__ import annotations

import os

from src.reconstruct.backends.base import ReconstructionBackend, ReconstructionResult
from src.reconstruct.task import TaskContext


class VGGTBackend(ReconstructionBackend):
    name = "vggt"

    def reconstruct(self, image_paths: list[str], ctx: TaskContext) -> ReconstructionResult:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        # Imported lazily: pulls torch + a local VGGT checkout.
        from src.vggt_reconstruct import run_vggt

        use_point_map = bool(ctx.cfg.get("vggt", {}).get("use_point_map", True))
        out = run_vggt(image_paths, ctx.cfg, model=None, use_point_map=use_point_map)
        return ReconstructionResult(
            points=out["points"],
            colors=out["colors"],
            conf=out["conf"],
            extrinsic=out["extrinsic"],
            intrinsic=out["intrinsic"],
            image_paths=out["image_paths"],
            meta={
                "backend": self.name,
                "model": ctx.cfg.get("vggt", {}).get("model"),
                "use_point_map": use_point_map,
            },
        )
