"""Write the final PLY point cloud, cameras.json, and stats.json."""
from __future__ import annotations

import json
from pathlib import Path

from src.reconstruct.task import Task, TaskContext, register


@register
class ExportTask(Task):
    name = "export"
    deps = ("filter",)

    def outputs(self, ctx: TaskContext) -> list[Path]:
        out = Path(ctx.cfg["output_dir"])
        return [out / "pointcloud.ply", out / "cameras.json", out / "stats.json"]

    def inputs(self, ctx: TaskContext) -> list[Path]:
        return [ctx.path("filter", "stats.json"), ctx.path("filter", "filtered.npz")]

    def load(self, ctx: TaskContext) -> dict | None:
        out = Path(ctx.cfg["output_dir"])
        ply = out / "pointcloud.ply"
        if not ply.exists():
            return None
        return {
            "pointcloud_ply": str(ply),
            "cameras_json": str(out / "cameras.json"),
            "stats_json": str(out / "stats.json"),
        }

    def run(self, ctx: TaskContext) -> dict:
        from src.pointcloud_io import cameras_to_json, write_ply

        points = ctx.artifacts["filter"]["points"]
        colors = ctx.artifacts["filter"]["colors"]
        stats = ctx.artifacts["filter"]["stats"]
        result = ctx.artifacts["reconstruct"]["result"]

        out = Path(ctx.cfg["output_dir"])
        out.mkdir(parents=True, exist_ok=True)

        ply = write_ply(points, colors, out / "pointcloud.ply")
        cameras = cameras_to_json(result.extrinsic, result.intrinsic, result.image_paths)
        (out / "cameras.json").write_text(json.dumps(cameras, indent=2))
        (out / "stats.json").write_text(json.dumps(stats, indent=2))

        print(f"export: pointcloud.ply ({len(points):,} pts), cameras.json, stats.json -> {out}")
        return {
            "pointcloud_ply": str(ply),
            "cameras_json": str(out / "cameras.json"),
            "stats_json": str(out / "stats.json"),
        }
