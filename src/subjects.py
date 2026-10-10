"""Named SAM2 subjects, masks and white-background cut-outs."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image



@dataclass
class Subject:
    """One selectable subject and its SAM2 prompts (pixel coords)."""

    name: str
    boxes: list = field(default_factory=list)   # [[x0, y0, x1, y1], ...]
    points: list = field(default_factory=list)  # [[x, y], ...]
    labels: list = field(default_factory=list)  # 1=fg, 0=bg per point


def sanitize_name(name: str) -> str:
    """Make a filesystem-safe subject name."""
    name = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", (name or "subject").strip()).strip("_")
    return name or "subject"


# ---------------------------------------------------------------- SAM2 + export
def segment_subject(predictor, image_rgb, subject: Subject) -> tuple[list[np.ndarray], np.ndarray]:
    """Run SAM2 for one subject; returns (mask list (K,H,W) bool, scores)."""
    from src.segment import predict_mask_sam2

    predictor.set_image(np.asarray(image_rgb))
    return predict_mask_sam2(
        predictor,
        image_rgb,
        points=subject.points or None,
        point_labels=subject.labels or None,
        boxes=subject.boxes or None,
    )


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Encode a bool mask as PNG bytes (for the web preview)."""
    import io

    buf = io.BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255).save(buf, format="PNG")
    return buf.getvalue()


def composite_on_white(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subject cut-out composited onto white (ready for Zero123++)."""
    out = np.full_like(image_rgb, 255)
    out[mask] = image_rgb[mask]
    return out


def save_subjects(
    subjects_dir: str | Path,
    image_path: str,
    image_rgb: np.ndarray,
    results: list[dict],
) -> dict:
    """Persist subjects: per-subject mask.png + subject.png + subjects.json.

    ``results`` entries: {"name", "mask": bool (H,W), "prompts": {...}}.
    Returns the subjects.json document.
    """
    subjects_dir = Path(subjects_dir)
    subjects_dir.mkdir(parents=True, exist_ok=True)

    from src.segment import clean_mask  # lazy: pulls cv2/torch

    doc = {"image": str(image_path), "subjects": []}
    used: set[str] = set()
    for r in results:
        base = sanitize_name(r["name"])
        # 重名主体自动加后缀，避免互相覆盖文件
        name = base
        k = 2
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        sub_dir = subjects_dir / name
        sub_dir.mkdir(parents=True, exist_ok=True)

        mask = clean_mask(np.asarray(r["mask"], dtype=bool))
        Image.fromarray(mask.astype(np.uint8) * 255).save(sub_dir / "mask.png")
        comp = composite_on_white(image_rgb, mask)
        Image.fromarray(comp).save(sub_dir / "subject.png")

        entry = {
            "name": name,
            "mask_file": f"{name}/mask.png",
            "subject_file": f"{name}/subject.png",
            "coverage": round(float(mask.mean()), 4),
            "prompts": r.get("prompts", {}),
        }
        doc["subjects"].append(entry)
        print(f"[subjects] saved {name}: coverage {entry['coverage']:.1%}")

    (subjects_dir / "subjects.json").write_text(json.dumps(doc, indent=2))
    print(f"[subjects] wrote {subjects_dir / 'subjects.json'}")
    return doc
