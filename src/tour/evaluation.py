"""Offline, no-cost evaluation over a fixed task set.

For each task the candidate storyboard (file with title/language/shots, no
scene binding) is bound to the live scene and pushed through the production
validator in strict mode. Results distinguish structural validity, reference
correctness, confirmed visibility, fact scoping, duration/shot-count/language
compliance and required-entity coverage. ``human_check`` is filled in by a
human reviewer, not by this script. No LLM calls are made.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.scene.storage import atomic_json

from .models import GenerationRequest, Storyboard, StoryboardValidation, now
from .validation import validate_storyboard

REFERENCE_ERROR_MARKERS = ("unknown view", "unknown entity", "unknown fact", "non-focused entity")
VISIBILITY_MARKERS = ("not human-confirmed visible", "draft entity")
SCOPE_MARKERS = ("view-dependent fact",)
DURATION_MARKERS = ("total duration", "differs from the requested")
COUNT_MARKERS = ("expected", "shots, got")
LANGUAGE_MARKERS = ("storyboard language",)
REQUIRED_MARKERS = ("required entity",)


def _has(errors, markers):
    return any(any(m in error for m in markers) for error in errors)


def _score_task(task: dict, scene, repository) -> dict:
    binding = json.loads(Path(task["candidate"]).read_text(encoding="utf-8"))
    request = GenerationRequest(
        **{**task["request"], "scene_id": scene.scene_id, "scene_revision": scene.revision}
    )
    storyboard = Storyboard(
        storyboard_id=f"eval_{task['task_id']}",
        revision=1,
        scene_id=scene.scene_id,
        scene_revision=scene.revision,
        asset_sha256=scene.asset_sha256,
        title=binding["title"],
        language=binding["language"],
        source="fixture",
        request_summary=request.model_dump(),
        shots=binding["shots"],
        validation=StoryboardValidation(status="valid"),
        created_at=now(),
    )
    validation = validate_storyboard(
        scene, storyboard, request, duration_strict=True
    )
    errors = validation.errors
    metrics = {
        "task_id": task["task_id"],
        "request": request.model_dump(),
        "schema_valid": storyboard is not None,
        "references_correct": not _has(errors, REFERENCE_ERROR_MARKERS),
        "visibility_correct": not _has(errors, VISIBILITY_MARKERS),
        "fact_scope_correct": not _has(errors, SCOPE_MARKERS),
        "duration_match": not _has(errors, DURATION_MARKERS),
        "shot_count_match": not _has(errors, COUNT_MARKERS),
        "language_match": not _has(errors, LANGUAGE_MARKERS),
        "required_coverage": not _has(errors, REQUIRED_MARKERS),
        "validation_status": validation.status,
        "errors": errors,
        "warnings": validation.warnings,
        "expected_valid": bool(task.get("expected", {}).get("valid", True)),
        "human_check": task.get("human_check") or {"fact_consistency": None},
        "provider": None,
        "latency_seconds": None,
        "model_calls": None,
    }
    metrics["agrees_with_expectation"] = (
        metrics["validation_status"] == "valid"
    ) == metrics["expected_valid"]
    return metrics


def run_evaluation(repository, tasks_path: Path, evaluation_id: str | None = None) -> dict:
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    evaluation_id = evaluation_id or uuid.uuid4().hex
    base = Path(tasks_path).resolve().parent
    results = []
    for task in tasks["tasks"]:
        try:
            scene = repository.get_scene(task["scene_id"])
            task["candidate"] = str((base / task["candidate"]).resolve())
            results.append(_score_task(task, scene, repository))
        except Exception as exc:  # noqa: BLE001
            results.append({
                "task_id": task.get("task_id"),
                "error": f"{type(exc).__name__}: {exc}",
                "expected_valid": bool(task.get("expected", {}).get("valid", True)),
                "agrees_with_expectation": False,
            })
    passed = [r for r in results if r.get("agrees_with_expectation")]
    report = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "created_at": now(),
        "tasks_path": str(tasks_path),
        "no_llm_calls": True,
        "summary": {
            "total": len(results),
            "passed": len(passed),
            "failed": len(results) - len(passed),
        },
        "tasks": results,
    }
    out = Path(repository.root) / "evaluation" / evaluation_id
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "report.json", report)
    report["report_path"] = str(out / "report.json")
    return report
