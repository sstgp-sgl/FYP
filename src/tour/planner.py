"""Grounded storyboard generation: build context → call provider → parse →
validate → repair at most once. Independent of HTTP and storage.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import (
    Attempt,
    GenerationRequest,
    LLMResponse,
    Shot,
    Storyboard,
    StoryboardValidation,
    TourScene,
    now,
)
from .providers import ProviderError, parse_strict_json
from .validation import generation_precheck, validate_storyboard

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

OUTPUT_SCHEMA = {
    "description": "storyboard response",
    "type": "object",
}


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def build_context(scene: TourScene, request: GenerationRequest) -> str:
    """Serialize only what the model may use: confirmed entities/facts and
    approved views. No point data, image bytes, server paths or secrets."""
    lines = []
    lines.append(f"scene_id: {scene.scene_id}  revision: {scene.revision}")
    lines.append(f"语言: {request.language}  目标时长: {request.target_duration_seconds}s  镜头数: {request.shot_count}")
    if request.required_entity_ids:
        lines.append("必须覆盖的实体: " + ", ".join(request.required_entity_ids))
    if request.opening_view_id:
        lines.append(f"开场镜头: {request.opening_view_id}")
    lines.append("")
    lines.append("## 实体（仅 confirmed 提供）")
    for entity in scene.entities:
        if entity.status != "confirmed":
            continue
        lines.append(f"- {entity.entity_id} | label: {entity.label}")
        if entity.description:
            lines.append(f"  description: {entity.description}")
        lines.append(f"  bounds_asset: {entity.bounds_asset}")
        for fact in entity.facts:
            scope = f" [scope_view_id={fact.scope_view_id}]" if fact.scope_view_id else ""
            lines.append(f"  fact {fact.fact_id} ({fact.kind}): {fact.text}{scope}")
    lines.append("")
    lines.append("## 镜头（人工确认）")
    for view in scene.views:
        lines.append(
            f"- {view.view_id} | name: {view.name} | "
            f"approved_entity_ids: {view.approved_entity_ids}"
        )
    return "\n".join(lines)


def build_messages(scene: TourScene, request: GenerationRequest) -> list[dict]:
    system = load_prompt("tour_system.txt").replace("{context}", build_context(scene, request))
    system = system.replace("{request}", json.dumps(request.model_dump(), ensure_ascii=False, indent=2))
    return [{"role": "system", "content": system}]


def build_repair_messages(scene: TourScene, request: GenerationRequest, errors: list[str], previous: str) -> list[dict]:
    system = load_prompt("tour_system.txt").replace("{context}", build_context(scene, request))
    system = system.replace("{request}", json.dumps(request.model_dump(), ensure_ascii=False, indent=2))
    repair = (
        load_prompt("tour_repair.txt")
        .replace("{errors}", "\n".join(f"- {e}" for e in errors))
        .replace("{previous_response}", previous)
        .replace("{system_prompt}", system)
    )
    return [{"role": "system", "content": repair}]


def parse_response(content: str) -> LLMResponse:
    data = parse_strict_json(content)
    try:
        response = LLMResponse.model_validate(data)
    except Exception as exc:  # pydantic ValidationError
        raise ProviderError("invalid_response", f"model JSON failed schema validation: {exc}") from exc
    return response


def _normalize_shot_ids(shots: list[Shot]) -> list[Shot]:
    """Server owns shot ordering IDs. Model-provided IDs are kept only when
    valid and unique; otherwise deterministic ids are assigned."""
    raw_ids = [shot.shot_id for shot in shots]
    if len(raw_ids) == len(set(raw_ids)):
        return shots
    normalized = []
    for index, shot in enumerate(shots, start=1):
        normalized.append(
            shot.model_copy(update={"shot_id": f"shot_{index:02d}"})
        )
    return normalized


@dataclass
class PlanOutcome:
    kind: str  # ok | clarification | failed
    storyboard: Storyboard | None = None
    question: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    errors: list[str] = field(default_factory=list)


def _attempt_record(index: int, status: str, result=None, errors=None, warnings=None) -> Attempt:
    if result is not None:
        return Attempt(
            attempt=index,
            status=status,
            provider=result.provider,
            model=result.model,
            latency_seconds=result.latency_seconds,
            usage=result.usage,
            finish_reason=result.finish_reason,
            raw_sha256=hashlib.sha256(result.content.encode("utf-8")).hexdigest(),
            errors=errors or [],
            warnings=warnings or [],
        )
    return Attempt(attempt=index, status=status, errors=errors or [])


async def plan_storyboard(
    scene: TourScene,
    request: GenerationRequest,
    provider,
    config: dict,
    on_attempt=None,
) -> PlanOutcome:
    """Generate → validate → repair at most once. Never more than
    ``llm.max_calls_per_run`` provider calls; no hidden network retries."""
    precheck = generation_precheck(scene, request)
    if precheck:
        return PlanOutcome(kind="failed", error_code="precheck_failed",
                           error_message="; ".join(precheck), errors=list(precheck))

    limits = {
        "max_output_tokens": int(config.get("llm", {}).get("max_output_tokens", 1500)),
        "temperature": float(config.get("llm", {}).get("temperature", 0.2)),
    }
    max_calls = int(config.get("llm", {}).get("max_calls_per_run", 2))
    narration_limits = config.get("tour", {}).get("narration", {})
    attempts: list[Attempt] = []
    previous_content: str | None = None
    previous_errors: list[str] = []

    for call in range(1, max_calls + 1):
        messages = (
            build_repair_messages(scene, request, previous_errors, previous_content)
            if call > 1
            else build_messages(scene, request)
        )
        try:
            result = await provider.generate(messages, OUTPUT_SCHEMA, limits)
        except ProviderError as exc:
            attempts.append(_attempt_record(call, "provider_error", errors=[str(exc)]))
            return PlanOutcome(
                kind="failed", attempts=attempts, error_code=exc.kind,
                error_message=str(exc), errors=[str(exc)],
            )
        if on_attempt is not None:
            on_attempt(call, result)
        try:
            response = parse_response(result.content)
        except ProviderError as exc:
            attempts.append(_attempt_record(call, "invalid_response", result, errors=[str(exc)]))
            previous_content, previous_errors = result.content, [str(exc)]
            continue

        if response.status == "needs_clarification":
            attempts.append(_attempt_record(call, "needs_clarification", result))
            return PlanOutcome(kind="clarification", question=response.question,
                               attempts=attempts)

        shots = _normalize_shot_ids(response.shots)
        candidate = Storyboard(
            storyboard_id="pending",
            revision=1,
            scene_id=scene.scene_id,
            scene_revision=scene.revision,
            asset_sha256=scene.asset_sha256,
            title=response.title,
            language=request.language,
            source="llm",
            request_summary=request.model_dump(),
            shots=shots,
            validation=StoryboardValidation(status="valid"),
            created_at=now(),
        )
        validation = validate_storyboard(
            scene, candidate, request, narration_limits=narration_limits, duration_strict=True
        )
        candidate.validation = validation
        if validation.status == "valid":
            attempts.append(_attempt_record(call, "ok", result,
                                            warnings=validation.warnings))
            return PlanOutcome(kind="ok", storyboard=candidate, attempts=attempts)
        attempts.append(_attempt_record(call, "validation_error", result,
                                        errors=validation.errors,
                                        warnings=validation.warnings))
        previous_content, previous_errors = result.content, list(validation.errors)

    last_status = attempts[-1].status if attempts else "unknown"
    if last_status == "invalid_response":
        error_code, error_message = "parse", previous_errors[0] if previous_errors else "invalid response"
    else:
        error_code = "validation_failed"
        error_message = "; ".join(previous_errors)
    return PlanOutcome(
        kind="failed",
        attempts=attempts,
        error_code=error_code,
        error_message=error_message,
        errors=list(previous_errors),
    )
