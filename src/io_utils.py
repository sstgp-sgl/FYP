"""Shared helpers: config loading, path resolution, image IO, stage validation."""
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path=None):
    path = Path(config_path) if config_path else PROJECT_ROOT / "configs" / "default.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths against the project root.
    for key, value in cfg["paths"].items():
        p = Path(value)
        if not p.is_absolute():
            cfg["paths"][key] = str(PROJECT_ROOT / p)
    return cfg


def load_image_rgb(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing input image: {p}")
    return Image.open(p).convert("RGB")


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"saved {path}")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {path}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def list_view_images(views_dir):
    """Return ordered view image paths from views.json under views_dir.

    Expects views.json with a "views" list of {"file": "view_XX.png", ...}.
    """
    views_dir = Path(views_dir)
    meta = load_json(views_dir / "views.json")
    paths = []
    for entry in meta["views"]:
        p = views_dir / entry["file"]
        if not p.exists():
            raise FileNotFoundError(f"Missing view image listed in views.json: {p}")
        paths.append(p)
    return paths


def require_files(*paths):
    """Stage-boundary check: fail fast if a previous stage's output is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifacts from a previous stage: {missing}")


def check_array(name, arr, shape=None, min_val=None, max_val=None):
    """Validate tensor/array shape and value range at stage boundaries."""
    arr = np.asarray(arr)
    if shape is not None:
        if len(shape) != arr.ndim or any(s is not None and s != a for s, a in zip(shape, arr.shape)):
            raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
    if min_val is not None and arr.min() < min_val:
        raise ValueError(f"{name}: min {arr.min()} < {min_val}")
    if max_val is not None and arr.max() > max_val:
        raise ValueError(f"{name}: max {arr.max()} > {max_val}")
    return arr


def resize_pad_square(img, size, fill=(255, 255, 255)):
    """Resize keeping aspect ratio, then pad to size x size.

    Returns (padded_image, info) where info records the scale and offsets so
    coordinates can be mapped back to the original image if needed.
    """
    w, h = img.size
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    off_x, off_y = (size - new_w) // 2, (size - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    info = {"scale": scale, "offset_x": off_x, "offset_y": off_y,
            "orig_size": [w, h], "resized_size": [new_w, new_h]}
    return canvas, info
