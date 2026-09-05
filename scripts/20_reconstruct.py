#!/usr/bin/env python3
"""Task-framework CLI: multi-view 3D point-cloud reconstruction (no rendering).

Usage:
    python scripts/20_reconstruct.py list
    python scripts/20_reconstruct.py run [--only collect,reconstruct] [--force]
    python scripts/20_reconstruct.py visualize [--mode ply|predictions] [--port 8080]

The pipeline is collect → preprocess → reconstruct → filter → export. Each task
is cached on disk, so re-running skips already-produced artifacts unless
``--force`` is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reconstruct import Pipeline, TaskContext, available_tasks, get_task, load_config

PIPELINE_ORDER = ("collect", "preprocess", "reconstruct", "filter", "export")


def build_pipeline() -> Pipeline:
    return Pipeline([get_task(n) for n in PIPELINE_ORDER])


def make_context(cfg: dict, force: bool = False) -> TaskContext:
    workdir = Path(cfg["output_dir"]) / ".tasks"
    return TaskContext(cfg=cfg, workdir=workdir, force=force)


def cmd_list(args: argparse.Namespace) -> None:
    for name, task in available_tasks().items():
        print(f"{name:12s} deps={list(task.deps)} cacheable={task.cacheable}")


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ctx = make_context(cfg, force=args.force)
    only = [n.strip() for n in args.only.split(",")] if args.only else None
    summary = build_pipeline().run(ctx, only=only)
    print("\n=== pipeline summary ===")
    for row in summary["tasks"]:
        print(f"  {row['task']:12s} {row['status']}")


def cmd_visualize(args: argparse.Namespace) -> None:
    from src.reconstruct.tasks.visualize import VisualizeTask

    cfg = load_config(args.config)
    vcfg = dict(cfg.get("visualize", {}))
    if args.mode:
        vcfg["mode"] = args.mode
    if args.port:
        vcfg["port"] = args.port
    cfg["visualize"] = vcfg
    mode = vcfg["mode"]

    ctx = make_context(cfg, force=args.force)
    # Ensure upstream artifacts exist; caching skips anything already done.
    if mode == "ply":
        build_pipeline().run(ctx, only=["export"])
    else:  # predictions re-runs VGGT from the preprocessed views
        build_pipeline().run(ctx, only=["preprocess"])

    VisualizeTask().run(ctx)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="reconstruct",
        description="Multi-view 3D point-cloud reconstruction task framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available tasks")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="run the reconstruction pipeline")
    p_run.add_argument("--config", default=None, help="path to a config YAML")
    p_run.add_argument("--force", action="store_true", help="ignore cached outputs")
    p_run.add_argument("--only", default=None, help="comma-separated task names")
    p_run.set_defaults(func=cmd_run)

    p_vis = sub.add_parser("visualize", help="open Viser on the point cloud")
    p_vis.add_argument("--config", default=None, help="path to a config YAML")
    p_vis.add_argument("--mode", choices=["ply", "predictions"], default=None)
    p_vis.add_argument("--port", type=int, default=None)
    p_vis.add_argument("--force", action="store_true")
    p_vis.set_defaults(func=cmd_visualize)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
