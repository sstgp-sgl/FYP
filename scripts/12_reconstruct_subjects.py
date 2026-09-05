#!/usr/bin/env python3
"""Stage 2c: per-subject VGGT point-cloud reconstruction (CLI).

Thin wrapper around src.workflow.reconstruct_subjects (the web app runs the
same code path).

Usage:
    python scripts/12_reconstruct_subjects.py
    python scripts/12_reconstruct_subjects.py --index outputs/views/subjects_index.json
    python scripts/12_reconstruct_subjects.py --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-subject point-cloud reconstruction")
    parser.add_argument("--index", default=None)
    parser.add_argument("--force", action="store_true", help="redo cached subjects")
    args = parser.parse_args()

    from src.io_utils import load_json
    from src.reconstruct import load_config

    cfg = load_config()
    index_path = (
        Path(args.index)
        if args.index
        else Path(cfg["paths"]["views_dir"]) / "subjects_index.json"
    )
    if not index_path.exists():
        raise FileNotFoundError(
            f"missing {index_path}; run scripts/11_generate_subject_views.py first"
        )

    from src.workflow import reconstruct_subjects

    index_doc = load_json(index_path)
    reconstruct_subjects(index_doc, Path(cfg["output_dir"]), cfg, force=args.force)
    print(f"stage 2c done — {len(index_doc.get('subjects', []))} subjects under {cfg['output_dir']}")


if __name__ == "__main__":
    main()
