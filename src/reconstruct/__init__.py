"""Task framework for multi-view 3D point-cloud reconstruction (no rendering).

Pipeline: collect → preprocess → reconstruct → filter → export, plus an
optional interactive ``visualize`` task backed by Viser.

    from src.reconstruct import Pipeline, TaskContext, load_config
    from src.reconstruct.task import get_task

    cfg = load_config()
    ctx = TaskContext(cfg, workdir=Path(cfg["output_dir"]) / ".tasks")
    pipeline = Pipeline([get_task(n) for n in
                         ("collect", "preprocess", "reconstruct", "filter", "export")])
    pipeline.run(ctx)
"""
from __future__ import annotations

from src.reconstruct.config import load_config
from src.reconstruct.pipeline import Pipeline, PipelineError
from src.reconstruct.task import Task, TaskContext, available_tasks, get_task

from src.reconstruct import tasks  # noqa: F401  (populates the registry)

__all__ = [
    "Pipeline",
    "PipelineError",
    "Task",
    "TaskContext",
    "available_tasks",
    "get_task",
    "load_config",
]
