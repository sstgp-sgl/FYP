"""Task-framework primitives: Task, TaskContext, and a name registry.

A ``Task`` is a small, independently-runnable stage of the reconstruction
pipeline. Tasks declare ``deps`` (names of tasks that must run first) and may
declare ``outputs`` (files whose existence lets the pipeline skip re-running
them). The registry lets the CLI and pipeline resolve tasks by name.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskContext:
    """Shared state threaded through every task in a pipeline run."""

    cfg: dict
    workdir: Path
    force: bool = False
    artifacts: dict[str, Any] = field(default_factory=dict)

    def path(self, *parts: str) -> Path:
        """Return a path under the task working dir, creating missing dirs.

        ``path("preprocess")`` creates that directory (a stage folder).
        ``path("preprocess", "manifest.json")`` creates the parent folder.
        Previously only the parent of the joined path was created, so
        ``path("preprocess") / "view_00.png"`` failed on a fresh subject
        when ``force=True`` skipped ``is_complete()`` (which used to mkdir
        as a side-effect of ``outputs()``).
        """
        p = self.workdir.joinpath(*parts)
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
        return p


class Task(abc.ABC):
    """Base class for a pipeline stage."""

    name: str = ""
    deps: tuple[str, ...] = ()
    cacheable: bool = True

    @abc.abstractmethod
    def run(self, ctx: TaskContext) -> dict:
        """Execute the task; return an artifact dict stored at ``ctx.artifacts[name]``."""

    def load(self, ctx: TaskContext) -> dict | None:
        """Reconstruct this task's artifact dict from its persisted outputs.

        Called by the pipeline when a task is skipped (cache hit) so downstream
        tasks still find ``ctx.artifacts[name]``. Return None when the artifact
        cannot be restored (the pipeline then leaves it absent).
        """
        return None

    def outputs(self, ctx: TaskContext) -> list[Path]:
        """Files that, when all present (and not ``ctx.force``), let this task be skipped."""
        return []

    def inputs(self, ctx: TaskContext) -> list[Path]:
        """Files this task reads. Outputs older than any of them are stale."""
        return []

    def is_complete(self, ctx: TaskContext) -> bool:
        outs = [Path(p) for p in self.outputs(ctx)]
        if not outs or not all(p.exists() for p in outs):
            return False
        ins = [Path(p) for p in self.inputs(ctx)]
        ins = [p for p in ins if p.exists()]
        if not ins:
            return True
        newest_in = max(p.stat().st_mtime for p in ins)
        oldest_out = min(p.stat().st_mtime for p in outs)
        return newest_in <= oldest_out


def write_if_changed(path: Path, text: str) -> bool:
    """Write ``text`` only when it differs from what is on disk.

    Keeps manifest mtimes stable across no-op reruns so the staleness check in
    ``Task.is_complete`` does not cascade an expensive rebuild for nothing.
    """
    path = Path(path)
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return True


_REGISTRY: dict[str, Task] = {}


def register(cls):
    """Class decorator registering a Task instance under ``cls.name``."""
    inst = cls()
    if not inst.name:
        raise ValueError(f"Task {cls.__name__} must define a non-empty ``name``")
    _REGISTRY[inst.name] = inst
    return cls


def get_task(name: str) -> Task:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown task {name!r}; available: {sorted(_REGISTRY)}"
        ) from None


def available_tasks() -> dict[str, Task]:
    return dict(_REGISTRY)
