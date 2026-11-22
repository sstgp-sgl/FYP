"""Pure validation for tour scenes and storyboards. No I/O, no LLM.

Validation errors block saving / playback; warnings are advisory. Visibility
checks rely on human-confirmed ``approved_entity_ids`` — they are never
claimed to be automatic occlusion analysis.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import bounds_intersect_allow_margin, point_bounds
from .models import Storyboard, StoryboardValidation, TourScene


def _narration_warning(language: str, text: str, duration: float, limits: dict) -> list[str]:
    warnings = []
    if language == "zh-CN":
        threshold = limits.get("zh_chars_per_second", 5.5)
        cps = len(text) / duration
        if cps > threshold:
            warnings.append(
                f"narration may be too long for {duration}s "
                f"({len(text)} chars, {cps:.1f} chars/s vs heuristic {threshold}/s)"
            )
    else:
        threshold = limits.get("en_words_per_second", 2.5)
        words = len(text.split())
        wps = words / duration
        if wps > threshold:
            warnings.append(
                f"narration may be too long for {duration}s "
                f"({words} words, {wps:.1f} words/s vs heuristic {threshold}/s)"
            )
    return warnings


def validate_annotations(
    entities: list, views: list, scene_points: np.ndarray, limits: dict | None = None
) -> StoryboardValidation:
    """Validate a proposed annotations update (entities + views) before saving."""
    limits = limits or {}
    errors: list[str] = []
    warnings: list[str] = []
    try:
        scene_lo, scene_hi = point_bounds(scene_points)
    except ValueError as exc:
        return StoryboardValidation(status="invalid", errors=[str(exc)])

    entity_ids = [e.entity_id for e in entities]
    if len(entity_ids) != len(set(entity_ids)):
        errors.append("duplicate entity_id")
    for entity in entities:
        if entity.status == "confirmed" and len(entity.facts) == 0:
            warnings.append(f"entity {entity.entity_id} is confirmed but has no facts")
        lo, hi = np.asarray(entity.bounds_asset[0]), np.asarray(entity.bounds_asset[1])
        if not bounds_intersect_allow_margin(lo, hi, scene_lo, scene_hi):
            errors.append(f"entity {entity.entity_id} bounds do not intersect the point cloud")
        for fact in entity.facts:
            if fact.scope_view_id is not None and fact.scope_view_id not in [v.view_id for v in views]:
                errors.append(f"fact {fact.fact_id} references unknown view {fact.scope_view_id}")
    view_ids = [v.view_id for v in views]
    if len(view_ids) != len(set(view_ids)):
        errors.append("duplicate view_id")
    for view in views:
        for approved in view.approved_entity_ids:
            entity = next((e for e in entities if e.entity_id == approved), None)
            if entity is None:
                errors.append(f"view {view.view_id} approves unknown entity {approved}")
            elif entity.status != "confirmed":
                warnings.append(
                    f"view {view.view_id} approves draft entity {approved}; "
                    "only confirmed entities are offered to the LLM"
                )
        if not view.approved_entity_ids:
            warnings.append(f"view {view.view_id} has no confirmed visible entities")
    status = "valid" if not errors else "invalid"
    return StoryboardValidation(status=status, errors=errors, warnings=warnings)


def validate_storyboard(
    scene: TourScene,
    storyboard: Storyboard,
    request=None,
    narration_limits: dict | None = None,
    duration_strict: bool = False,
) -> StoryboardValidation:
    """Validate a storyboard against a frozen scene revision.

    ``duration_strict`` controls whether shot count / total duration
    mismatches with the request are blocking errors (LLM output repair) or
    warnings (manual editing keeps the user in control).
    """
    narration_limits = narration_limits or {}
    errors: list[str] = []
    warnings: list[str] = []

    if storyboard.scene_id != scene.scene_id:
        errors.append("storyboard scene_id does not match the frozen scene")
    if storyboard.scene_revision != scene.revision:
        errors.append("storyboard was built for a different scene revision")
    if storyboard.asset_sha256 != scene.asset_sha256:
        errors.append("storyboard asset hash does not match the scene asset")

    entities = {e.entity_id: e for e in scene.entities}
    facts = {
        fact.fact_id: (entity.entity_id, fact)
        for entity in scene.entities
        for fact in entity.facts
    }
    views = {v.view_id: v for v in scene.views}

    total = 0
    for index, shot in enumerate(storyboard.shots):
        prefix = f"shot {shot.shot_id}"
        if shot.view_id not in views:
            errors.append(f"{prefix} references unknown view {shot.view_id}")
            continue
        view = views[shot.view_id]
        for focus in shot.focus_entity_ids:
            if focus not in entities:
                errors.append(f"{prefix} focuses unknown entity {focus}")
                continue
            entity = entities[focus]
            if entity.status != "confirmed":
                errors.append(f"{prefix} focuses draft entity {focus}")
            if focus not in view.approved_entity_ids:
                errors.append(
                    f"{prefix} focuses entity {focus} that is not human-confirmed "
                    f"visible from view {shot.view_id}"
                )
        for fact_id in shot.fact_ids:
            if fact_id not in facts:
                errors.append(f"{prefix} cites unknown fact {fact_id}")
                continue
            owner_id, fact = facts[fact_id]
            if owner_id not in shot.focus_entity_ids:
                errors.append(f"{prefix} cites fact {fact_id} of a non-focused entity")
            if fact.scope_view_id is not None and fact.scope_view_id != shot.view_id:
                errors.append(
                    f"{prefix} cites view-dependent fact {fact_id} "
                    f"(scoped to {fact.scope_view_id})"
                )
        total += shot.duration_seconds
        warnings.extend(
            _narration_warning(storyboard.language, shot.narration, shot.duration_seconds, narration_limits)
        )
        if index == 0 and request is not None and request.opening_view_id:
            if shot.view_id != request.opening_view_id:
                msg = "first shot does not use the requested opening view"
                if duration_strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)

    if total > 120:
        errors.append(f"total duration {total}s exceeds the 120s limit")

    if request is not None:
        if len(storyboard.shots) != request.shot_count:
            msg = f"expected {request.shot_count} shots, got {len(storyboard.shots)}"
            if duration_strict:
                errors.append(msg)
            else:
                warnings.append(msg)
        if abs(total - request.target_duration_seconds) > 1:
            msg = (
                f"total duration {total}s differs from the requested "
                f"{request.target_duration_seconds}s by more than 1s"
            )
            if duration_strict:
                errors.append(msg)
            else:
                warnings.append(msg)
        covered = {
            entity_id
            for shot in storyboard.shots
            for entity_id in shot.focus_entity_ids
        }
        for required in request.required_entity_ids:
            if required not in entities:
                errors.append(f"required entity {required} does not exist")
            elif required not in covered:
                msg = f"required entity {required} is not covered by any shot"
                if duration_strict:
                    errors.append(msg)
                else:
                    warnings.append(msg)
        if storyboard.language != request.language:
            msg = f"storyboard language {storyboard.language} != requested {request.language}"
            if duration_strict:
                errors.append(msg)
            else:
                warnings.append(msg)

    status = "valid" if not errors else "invalid"
    return StoryboardValidation(status=status, errors=errors, warnings=warnings)


def generation_precheck(scene: TourScene, request) -> list[str]:
    """Blocking preconditions before an LLM run starts."""
    errors: list[str] = []
    confirmed = {e.entity_id for e in scene.entities if e.status == "confirmed"}
    if not scene.views:
        errors.append("no saved views; record at least one view first")
    for required in request.required_entity_ids:
        if required not in confirmed:
            errors.append(f"required entity {required} is not confirmed")
    if request.opening_view_id and request.opening_view_id not in {v.view_id for v in scene.views}:
        errors.append(f"opening view {request.opening_view_id} does not exist")
    # An entity is usable by the LLM only if some approved view shows it.
    viewable = set().union(*[set(v.approved_entity_ids) for v in scene.views])
    for entity in scene.entities:
        if entity.status == "confirmed" and entity.entity_id not in viewable:
            errors.append(
                f"entity {entity.entity_id} is confirmed but not visible in any approved view"
            )
    return errors
