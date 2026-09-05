"""Dependency-resolving pipeline runner for reconstruction tasks."""
from __future__ import annotations

from typing import Iterable

from src.reconstruct.task import Task, TaskContext


class PipelineError(RuntimeError):
    """Raised for malformed pipelines (cycles, unknown deps)."""


class Pipeline:
    """Run an ordered set of tasks, honoring dependencies and caching."""

    def __init__(self, tasks: Iterable[Task]):
        self.tasks = list(tasks)
        self._by_name = {t.name: t for t in self.tasks}
        if len(self._by_name) != len(self.tasks):
            raise PipelineError("duplicate task names in pipeline")

    def resolve(self, only: list[str] | None = None) -> list[Task]:
        """Return tasks in dependency order (requested + transitive deps)."""
        requested = [self._by_name[n] for n in only] if only else self.tasks
        for name in only or []:
            if name not in self._by_name:
                raise PipelineError(f"unknown task {name!r}")

        ordered: list[Task] = []
        seen: set[str] = set()
        visiting: set[str] = set()

        def visit(t: Task) -> None:
            if t.name in seen:
                return
            if t.name in visiting:
                raise PipelineError(f"dependency cycle detected at {t.name!r}")
            visiting.add(t.name)
            for dep in t.deps:
                if dep not in self._by_name:
                    raise PipelineError(f"task {t.name!r} depends on unknown {dep!r}")
                visit(self._by_name[dep])
            visiting.discard(t.name)
            seen.add(t.name)
            ordered.append(t)

        for t in requested:
            visit(t)
        return ordered

    def run(self, ctx: TaskContext, only: list[str] | None = None) -> dict:
        """Run tasks in order; return a summary. Raises on the first failure."""
        summary: list[dict] = []
        for t in self.resolve(only):
            if t.cacheable and not ctx.force and t.is_complete(ctx):
                print(f"[skip] {t.name} (outputs exist)")
                loaded = t.load(ctx)
                if loaded is not None:
                    ctx.artifacts[t.name] = loaded
                summary.append({"task": t.name, "status": "skipped", "reason": "outputs exist"})
                continue
            print(f"[run ] {t.name}")
            result = t.run(ctx) or {}
            ctx.artifacts[t.name] = result
            summary.append({"task": t.name, "status": "done"})
        return {"tasks": summary, "artifacts": sorted(ctx.artifacts.keys())}
