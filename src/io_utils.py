"""Shared helpers: JSON, view manifests and aspect-preserving image resizing."""
import json
from pathlib import Path

from PIL import Image

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
