"""Confidence + statistical-outlier + background filtering."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.reconstruct.task import Task, TaskContext, register


@register
class FilterTask(Task):
    name = "filter"
    deps = ("reconstruct",)

    _PARAMS = ("conf_percentile", "outlier_nb_neighbors", "outlier_std_ratio")

    def outputs(self, ctx: TaskContext) -> list[Path]:
        return [ctx.path("filter", "stats.json")]

    def inputs(self, ctx: TaskContext) -> list[Path]:
        return [ctx.path("reconstruct", "predictions.npz")]

    def is_complete(self, ctx: TaskContext) -> bool:
        """Also reject a cached run whose filter parameters have since changed."""
        if not super().is_complete(ctx):
            return False
        try:
            stats = json.loads(ctx.path("filter", "stats.json").read_text())
        except Exception:  # noqa: BLE001
            return False
        vggt_cfg = ctx.cfg["vggt"]
        return all(stats.get(k) == vggt_cfg.get(k) for k in self._PARAMS)

    def load(self, ctx: TaskContext) -> dict | None:
        npz_path = ctx.path("filter", "filtered.npz")
        stats_path = ctx.path("filter", "stats.json")
        if not npz_path.exists():
            return None
        d = np.load(npz_path)
        stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        return {"points": d["points"], "colors": d["colors"], "stats": stats}

    def run(self, ctx: TaskContext) -> dict:
        from src.pointcloud_filter import filter_pointcloud

        result = ctx.artifacts["reconstruct"]["result"]
        points, colors, stats = filter_pointcloud(
            result.points, result.colors, result.conf, ctx.cfg
        )

        out_dir = ctx.path("filter")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        np.savez_compressed(out_dir / "filtered.npz", points=points, colors=colors)

        print(f"filter: {stats['num_points_raw']:,} -> "
              f"{stats['num_points_filtered']:,} points")
        return {"points": points, "colors": colors, "stats": stats}
