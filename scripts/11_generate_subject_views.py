#!/usr/bin/env python3
"""Stage 2b: per-subject Zero123++ multi-view generation (CLI).

Thin wrapper around src.workflow.generate_views_for_subjects (the web app
runs the same code path).

Usage:
    python scripts/11_generate_subject_views.py
    python scripts/11_generate_subject_views.py --subjects-json outputs/subjects/subjects.json
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io_utils import load_config, load_json


def main():
    parser = argparse.ArgumentParser(description="Per-subject Zero123++ view generation")
    parser.add_argument("--subjects-json", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config()
    subjects_json = (
        Path(args.subjects_json)
        if args.subjects_json
        else Path(cfg["paths"]["subjects_dir"]) / "subjects.json"
    )
    if not subjects_json.exists():
        raise FileNotFoundError(
            f"missing {subjects_json}; run scripts/10_subjects_web.py and save subjects first"
        )
    views_root = Path(args.out) if args.out else Path(cfg["paths"]["views_dir"])

    from src.workflow import generate_views_for_subjects

    doc = load_json(subjects_json)
    index = generate_views_for_subjects(doc, subjects_json.parent, views_root, cfg)
    print(f"stage 2b done — {len(index)} subjects in {views_root}")


if __name__ == "__main__":
    main()
