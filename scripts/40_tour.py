#!/usr/bin/env python3
"""Scene tour CLI: seed-demo / import-ply / validate / evaluate / list.

CPU-only. No torch, no model downloads. Run with the tour virtualenv:
    python scripts/40_tour.py seed-demo --output-root outputs/tours
"""
import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROJECT_ROOT  # noqa: E402
from src.tour.config import TourConfig  # noqa: E402
from src.tour.repository import TourRepository  # noqa: E402
from src.tour.validation import validate_storyboard  # noqa: E402


def _repository(args) -> TourRepository:
    root = _root(args)
    return TourRepository(root)


def _root(args) -> Path:
    """output_root resolution: explicit --output-root wins; otherwise the
    tour config (config file + env override) decides. Relative paths resolve
    against the project root, never the process cwd."""
    if args.output_root != "outputs/tours":
        path = Path(args.output_root)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return _config(args).output_root


def cmd_seed_demo(args):
    from src.tour.importers import seed_demo

    result = seed_demo(_repository(args), _config(args).data)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_import_ply(args):
    from src.tour.importers import import_from_upload

    path = Path(args.path)
    if not path.is_file():
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    scene = import_from_upload(
        _repository(args), path.read_bytes(), args.title or path.stem, _config(args).data
    )
    print(json.dumps({"scene_id": scene.scene_id, "revision": scene.revision,
                      "asset_sha256": scene.asset_sha256, "title": scene.title},
                     indent=2, ensure_ascii=False))
    return 0


def cmd_list(args):
    repository = _repository(args)
    print("scenes:")
    for summary in repository.list_scenes():
        print(f"  {summary['scene_id']}  rev={summary['revision']}  "
              f"{summary['source_kind']}  {summary['title']}")
    print("storyboards:")
    for summary in repository.list_storyboards():
        print(f"  {summary['storyboard_id']}  rev={summary['revision']}  "
              f"scene={summary['scene_id']}  {summary['title']}")
    return 0


def cmd_validate(args):
    repository = _repository(args)
    if args.storyboard_id:
        from src.tour.models import Storyboard

        storyboard = repository.get_storyboard(args.storyboard_id, args.storyboard_revision)
        scene = repository.get_scene(storyboard.scene_id, storyboard.scene_revision)
        validation = validate_storyboard(scene, storyboard, duration_strict=False)
        print(json.dumps(validation.model_dump(), indent=2, ensure_ascii=False))
        return 0 if validation.status == "valid" else 1
    # validate every stored storyboard against its frozen scene
    failed = 0
    for summary in repository.list_storyboards():
        storyboard = repository.get_storyboard(summary["storyboard_id"])
        scene = repository.get_scene(storyboard.scene_id, storyboard.scene_revision)
        validation = validate_storyboard(scene, storyboard, duration_strict=False)
        print(f"{summary['storyboard_id']}: {validation.status}")
        for error in validation.errors:
            print(f"  ERROR {error}")
            failed += 1
        for warning in validation.warnings:
            print(f"  WARN  {warning}")
    return 1 if failed else 0


def cmd_evaluate(args):
    from src.tour.evaluation import run_evaluation

    report = run_evaluation(_repository(args), Path(args.tasks), args.evaluation_id)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["summary"]["failed"] == 0 else 1


def _config(args):
    if args.tour_config:
        return TourConfig(args.tour_config)
    return TourConfig()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)

    for name in ("seed-demo", "import-ply", "list", "validate", "evaluate"):
        sub = subs.add_parser(name)
        sub.add_argument("--output-root", default="outputs/tours")
        sub.add_argument("--tour-config", default=None)
    seed = subs.choices["seed-demo"]
    imp = subs.choices["import-ply"]
    imp.add_argument("--path", required=True)
    imp.add_argument("--title", default=None)
    val = subs.choices["validate"]
    val.add_argument("--storyboard-id", default=None)
    val.add_argument("--storyboard-revision", type=int, default=None)
    ev = subs.choices["evaluate"]
    ev.add_argument("--tasks", default="evaluation/tasks-v1.json")
    ev.add_argument("--evaluation-id", default=None)

    args = parser.parse_args()
    handlers = {
        "seed-demo": cmd_seed_demo,
        "import-ply": cmd_import_ply,
        "list": cmd_list,
        "validate": cmd_validate,
        "evaluate": cmd_evaluate,
    }
    try:
        return handlers[args.command](args)
    except Exception as exc:  # noqa: BLE001
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
