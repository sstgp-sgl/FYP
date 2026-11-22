"""Shared fixtures for tour tests. Point clouds are generated in code; no
external fixture files are required."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tour_root(tmp_path, monkeypatch):
    root = tmp_path / "tours"
    monkeypatch.setenv("FYP_TOUR_OUTPUT_ROOT", str(root))
    return root


@pytest.fixture
def config_data():
    return {
        "tour": {
            "output_root": "outputs/tours",
            "max_upload_bytes": 104857600,
            "max_points": 2000000,
            "preview_points": 60000,
            "aspect_ratio": 16.0 / 9.0,
            "language": "zh-CN",
            "max_scenes_per_page": 100,
            "narration": {"zh_chars_per_second": 5.5, "en_words_per_second": 2.5},
        },
        "llm": {
            "enabled": False,
            "provider": "chat_completions_compatible",
            "base_url": "",
            "model": "",
            "api_key_env": "FYP_LLM_API_KEY",
            "response_mode": "text_json",
            "request_timeout_seconds": 45,
            "run_timeout_seconds": 100,
            "max_calls_per_run": 2,
            "max_output_tokens": 1500,
            "temperature": 0.2,
        },
    }


def make_config(overrides=None):
    config = {
        "tour": {
            "max_upload_bytes": 104857600,
            "max_points": 2000000,
            "preview_points": 60000,
        },
        "llm": {"max_output_tokens": 1500, "temperature": 0.2, "max_calls_per_run": 2},
    }
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def make_points(count=400, seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-3.0, 3.0, size=(count, 3))
    colors = rng.integers(0, 255, size=(count, 3)).astype(np.uint8)
    return points, colors


def write_ascii_ply(path, points, colors=None, extra_props=None, as_binary=False):
    """Write a small PLY with optional extra properties for import tests."""
    from plyfile import PlyData, PlyElement

    names = ["x", "y", "z"]
    dtypes = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    data = [points[:, 0], points[:, 1], points[:, 2]]
    if colors is not None:
        names += ["red", "green", "blue"]
        dtypes += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        data += [colors[:, 0], colors[:, 1], colors[:, 2]]
    for prop_name, prop_dtype, values in extra_props or []:
        names.append(prop_name)
        dtypes.append((prop_name, prop_dtype))
        data.append(values)
    vertex = np.empty(len(points), dtype=dtypes)
    for name, values in zip(names, data):
        vertex[name] = values
    element = PlyElement.describe(vertex, "vertex")
    PlyData([element], text=not as_binary).write(str(path))
    return path
