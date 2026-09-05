"""Interactive Viser inspection of the reconstructed point cloud."""
from __future__ import annotations

from pathlib import Path

from src.reconstruct.task import Task, TaskContext, register


@register
class VisualizeTask(Task):
    name = "visualize"
    deps = ("export",)
    cacheable = False  # interactive; never skipped

    def run(self, ctx: TaskContext) -> dict:
        vcfg = dict(ctx.cfg.get("visualize", {}))
        mode = vcfg.get("mode", "predictions")
        port = int(vcfg.get("port", 8080))

        if mode == "ply":
            from src.reconstruct.viser_viewer import visualize_ply

            out = Path(ctx.cfg["output_dir"])
            ply = out / "pointcloud.ply"
            cams = out / "cameras.json"
            if not ply.exists():
                raise FileNotFoundError(f"missing {ply}; run the pipeline first")
            visualize_ply(ply, cams if cams.exists() else None, port=port)
            return {"mode": "ply"}

        if mode == "predictions":
            from src.reconstruct.viser_viewer import visualize_predictions

            image_paths = ctx.artifacts["preprocess"]["image_paths"]
            visualize_predictions(
                image_paths,
                ctx.cfg,
                port=port,
                init_conf_threshold=float(vcfg.get("init_conf_threshold", 25.0)),
                use_point_map=bool(vcfg.get("use_point_map", False)),
                drop_white=bool(vcfg.get("drop_white", True)),
            )
            return {"mode": "predictions"}

        raise ValueError(f"unknown visualize.mode {mode!r}; use 'ply' | 'predictions'")
