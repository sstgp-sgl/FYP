#!/usr/bin/env python3
"""Phase 1 CLI. No model downloads unless reconstruct explicitly opts in."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    demo = subs.add_parser("demo-compose")
    demo.add_argument("--root", required=True, help="Fresh managed output directory")
    demo.add_argument("--relation", choices=["left_of", "right_of"], default="left_of")
    demo.add_argument("--right", nargs=3, type=float, default=[1., 0., 0.])
    validate = subs.add_parser("validate-scene")
    validate.add_argument("--root", required=True)
    validate.add_argument("--scene", default="scene.json")
    validate.add_argument("--ply", default="combined.ply")
    for command in ("diagnose-sharp", "reconstruct-sharp", "recover-sharp"):
        sub = subs.add_parser(command)
        sub.add_argument("--config", required=True)
        if command == "reconstruct-sharp":
            sub.add_argument("--image", required=True)
            sub.add_argument("--role", choices=["scene", "subject"], default="scene")
    args = parser.parse_args()
    if args.command == "demo-compose":
        from src.scene.demo import demo
        result = demo(args.root, args.relation, args.right)
    elif args.command == "validate-scene":
        from src.scene.storage import AssetRepository
        from src.scene.validation import validate_export
        result = validate_export(AssetRepository(args.root), args.scene, args.ply)
    else:
        from src.reconstruct.sharp_cli import SharpConfig, SharpCliBackend
        from src.scene.storage import read_json
        config_path = Path(args.config).resolve()
        data = read_json(config_path)
        # Relative configuration paths are relative to the config file.
        for key in ("checkpoint_path", "output_root"):
            if data.get(key) and not Path(data[key]).is_absolute():
                data[key] = str(config_path.parent / data[key])
        backend = SharpCliBackend(SharpConfig(**data))
        if args.command == "diagnose-sharp":
            result = backend.diagnose()
        elif args.command == "recover-sharp":
            result = {"schema_version": 1, "interrupted_run_ids": backend.recover()}
        else:
            result = asdict(backend.reconstruct(args.image, role=args.role))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if result.get("passed") is False or result.get("available") is False:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
