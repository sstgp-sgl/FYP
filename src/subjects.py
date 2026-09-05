"""Interactive multi-subject selection for oil paintings.

Text proposal (Florence-2 open-vocabulary detection) + SAM2 prompt refinement
(points/boxes) → named subject masks → per-subject cut-outs ready for the
per-subject reconstruction pipeline (Zero123++ / 3DGS).

Florence-2 and SAM2 are imported lazily so importing this module stays cheap.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

_TASK = "<OPEN_VOCABULARY_DETECTION>"


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


# ---------------------------------------------------------------- Florence-2
def load_florence(model_id: str = "microsoft/Florence-2-base", device: str = "cuda"):
    """Load Florence-2 (lazy, ~0.7GB on first download). Returns (model, processor)."""
    import os

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"[florence] loading {model_id} ...")
    # attn_implementation="eager" avoids needing real flash-attn (a stub package
    # satisfies transformers' static import check; eager never calls it).
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, attn_implementation="eager"
    )
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def detect_boxes(florence, texts: list[str], image: Image.Image) -> list[dict]:
    """Open-vocabulary detection: texts → [{"label", "box": [x0,y0,x1,y1]}].

    Categories are appended to the task prompt (comma-separated). Returns [] if
    nothing is detected (an empty result raises for retry-free UX upstream).
    """
    import torch

    model, processor = florence
    prompt = _TASK + ", ".join(t for t in texts if t)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    device = next(model.parameters()).device
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"].to(device),
            pixel_values=inputs["pixel_values"].to(device),
            max_new_tokens=1024,
            num_beams=3,
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        text, task=_TASK, image_size=image.size
    )
    det = parsed.get(_TASK, {})
    out = []
    for box, label in zip(det.get("bboxes", []), det.get("bboxes_labels", [])):
        out.append({"label": str(label), "box": [float(v) for v in box]})
    print(f"[florence] detected {len(out)} boxes for {texts}")
    return out


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
