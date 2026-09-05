"""Config loading for the reconstruction task framework.

Layers ``configs/reconstruct.yaml`` over ``configs/default.yaml`` (which holds
the shared ``paths``, ``vggt`` and ``segment`` blocks), then absolutizes every
path-like value against the project root so tasks can be run from any cwd.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "reconstruct.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _absolutize(cfg: dict) -> dict:
    for key, value in cfg.get("paths", {}).items():
        if isinstance(value, str):
            p = Path(value)
            if not p.is_absolute():
                cfg["paths"][key] = str(PROJECT_ROOT / p)

    output_dir = cfg.get("output_dir")
    if isinstance(output_dir, str):
        p = Path(output_dir)
        if not p.is_absolute():
            cfg["output_dir"] = str(PROJECT_ROOT / p)

    inp = cfg.get("input", {})
    folder = inp.get("folder")
    if isinstance(folder, str) and folder:
        p = Path(folder)
        if not p.is_absolute():
            cfg["input"]["folder"] = str(PROJECT_ROOT / p)
    files = inp.get("files")
    if isinstance(files, list):
        cfg["input"]["files"] = [
            str(p if (p := Path(f)).is_absolute() else PROJECT_ROOT / p) for f in files
        ]
    return cfg


def load_config(config_path: str | Path | None = None) -> dict:
    """Load the merged reconstruction config with absolutized paths."""
    # Base: default.yaml, already absolutized by io_utils.load_config.
    from src.io_utils import load_config as load_default

    cfg = load_default()

    override_path = Path(config_path) if config_path else DEFAULT_CONFIG
    if config_path is not None and not override_path.exists():
        raise FileNotFoundError(f"config not found: {override_path}")
    if override_path.exists():
        with open(override_path) as f:
            override = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, override)

    return _absolutize(cfg)
