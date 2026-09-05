#!/usr/bin/env python3
"""多主体选择 Web 服务：文字检测(Florence-2) + 点击/框选(SAM2) → subjects.json。

每个主体：文字自动提案检测框 → 鼠标点选/框选微调 → SAM2 掩码预览 →
命名保存。导出 outputs/subjects/<name>/{mask.png, subject.png} + subjects.json，
subject.png 是白底抠图，可直接喂给下游 Zero123++ / 3DGS 逐主体重建。

用法:
    python scripts/10_subjects_web.py [--config configs/default.yaml] [--port 7861]
浏览器打开 http://127.0.0.1:7861
"""
import argparse
import base64
import io
import os
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 必须在任何 HuggingFace 库导入前设置（服务器直连 huggingface.co 不通）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.io_utils import load_json
from src.reconstruct.config import load_config  # default.yaml + reconstruct.yaml 合并
from src.subjects import (
    Subject,
    detect_boxes,
    load_florence,
    mask_to_png_bytes,
    save_subjects,
    segment_subject,
)


def create_app(cfg: dict) -> FastAPI:
    app = FastAPI(title="Subjects Selector (Florence-2 + SAM2)")

    # 静态挂载 outputs/，网页可直接看中间产物（views 缩略图、mask、pointcloud 文件）
    app.mount("/outputs", StaticFiles(directory=str(PROJECT_ROOT / "outputs")), name="outputs")

    state: dict = {
        "cfg": cfg,
        "image": None,       # PIL RGB
        "image_rgb": None,   # np (H,W,3) uint8
        "image_name": "uploaded.png",
        "predictor": None,   # SAM2 predictor (lazy)
        "florence": None,    # (model, processor) lazy
        "workflow": {        # 02b+02c 工作流状态（后台线程更新，前端轮询）
            "status": "idle",      # idle | running | done | error
            "subjects": [],
            "log": [],
            "error": None,
        },
    }

    def get_predictor():
        if state["predictor"] is None:
            from src.segment import build_sam2_predictor

            print("[sam2] loading predictor ...")
            state["predictor"] = build_sam2_predictor(cfg)
            print("[sam2] predictor ready")
        return state["predictor"]

    def get_florence():
        if state["florence"] is None:
            state["florence"] = load_florence(
                cfg["subjects"]["florence_model"], cfg.get("device", "cuda")
            )
        return state["florence"]

    @app.get("/")
    def index():
        return FileResponse(PROJECT_ROOT / "web" / "subjects" / "index.html")

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        state["image"] = img
        state["image_rgb"] = np.asarray(img)
        state["image_name"] = file.filename or "uploaded.png"
        print(f"[upload] {state['image_name']} {img.size}")
        return {"size": [img.size[0], img.size[1]], "name": state["image_name"]}

    @app.get("/api/image")
    def image():
        if state["image"] is None:
            return Response(status_code=404)
        buf = io.BytesIO()
        state["image"].save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    class DetectIn(BaseModel):
        texts: list[str] = []

    @app.post("/api/detect")
    def detect(body: DetectIn):
        if state["image"] is None:
            return JSONResponse({"error": "no image uploaded"}, status_code=400)
        if not body.texts:
            return JSONResponse({"error": "empty texts"}, status_code=400)
        try:
            boxes = detect_boxes(get_florence(), body.texts, state["image"])
        except Exception as e:  # noqa: BLE001
            print(f"[detect] failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
        return {"boxes": boxes}

    class SegmentIn(BaseModel):
        boxes: list = []
        points: list = []
        labels: list = []

    @app.post("/api/segment")
    def segment(body: SegmentIn):
        if state["image"] is None:
            return JSONResponse({"error": "no image uploaded"}, status_code=400)
        try:
            subj = Subject(name="_", boxes=body.boxes, points=body.points, labels=body.labels)
            masks, scores = segment_subject(get_predictor(), state["image_rgb"], subj)
        except Exception as e:  # noqa: BLE001
            print(f"[segment] failed: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
        payload = [base64.b64encode(mask_to_png_bytes(m)).decode() for m in masks]
        return {"masks": payload, "scores": [float(s) for s in scores]}

    class ExportIn(BaseModel):
        subjects: list = []

    @app.post("/api/export")
    def export(body: ExportIn):
        if state["image"] is None:
            return JSONResponse({"error": "no image uploaded"}, status_code=400)
        pred = get_predictor()
        results = []
        for s in body.subjects:
            subj = Subject(
                name=s.get("name", "subject"),
                boxes=s.get("boxes", []),
                points=s.get("points", []),
                labels=s.get("labels", []),
            )
            masks, scores = segment_subject(pred, state["image_rgb"], subj)
            idx = int(s.get("mask_index", -1))
            if idx < 0 or idx >= len(masks):
                idx = int(np.argmax(scores))
            results.append(
                {
                    "name": subj.name,
                    "mask": masks[idx],
                    "prompts": {
                        "boxes": subj.boxes,
                        "points": subj.points,
                        "labels": subj.labels,
                    },
                }
            )
        doc = save_subjects(
            cfg["paths"]["subjects_dir"], state["image_name"], state["image_rgb"], results
        )
        return doc

    # ------------------------------------------------------- workflow (02b+02c)
    def _run_workflow(force: bool = False) -> None:
        st = state["workflow"]
        st["status"] = "running"
        st["error"] = None
        st["log"] = []
        subjects_dir = Path(cfg["paths"]["subjects_dir"])
        views_root = Path(cfg["paths"]["views_dir"])
        base_out = Path(cfg["output_dir"])

        try:
            subjects_json = subjects_dir / "subjects.json"
            if not subjects_json.exists():
                raise FileNotFoundError("还没有保存主体（subjects.json 不存在）")
            doc = load_json(subjects_json)
            names = [s["name"] for s in doc.get("subjects", [])]
            st["subjects"] = [
                {"name": n, "stage": "pending", "status": "pending", "views": []} for n in names
            ]

            def progress(i, total, stage, name, info):
                entry = next((x for x in st["subjects"] if x["name"] == name), None)
                if entry is not None:
                    entry["stage"] = stage
                    entry["status"] = info.get("status", "running")
                    if info.get("error"):
                        entry["error"] = info["error"]
                    if info.get("log"):
                        entry["log"] = info["log"]
                    if stage == "views" and info.get("status") == "done":
                        vdir = views_root / name
                        entry["views"] = (
                            sorted(f"views/{name}/{p.name}" for p in vdir.glob("view_*.png"))
                            if vdir.exists()
                            else []
                        )
                st["log"] = (st["log"] + [f"[{stage}] {name}: {info.get('status')}"])[-60:]

            from src.workflow import generate_views_for_subjects, reconstruct_subjects

            generate_views_for_subjects(doc, subjects_dir, views_root, cfg, progress=progress)
            index_doc = load_json(views_root / "subjects_index.json")
            reconstruct_subjects(
                index_doc, base_out, cfg, progress=progress, force=force, stop_on_error=False
            )
            st["status"] = "done"
        except Exception as e:  # noqa: BLE001
            st["status"] = "error"
            st["error"] = str(e)
            print(f"[workflow] failed: {e}")

    @app.post("/api/workflow/start")
    def workflow_start(force: bool = False):
        if state["workflow"]["status"] == "running":
            return JSONResponse({"error": "workflow already running"}, status_code=409)
        threading.Thread(target=_run_workflow, kwargs={"force": force}, daemon=True).start()
        return {"status": "started"}

    @app.get("/api/workflow/status")
    def workflow_status():
        return state["workflow"]

    # ------------------------------------------------------- point cloud preview
    @app.get("/api/pointcloud/{subject}")
    def pointcloud(subject: str):
        from src.pointcloud_io import read_ply

        ply = Path(cfg["output_dir"]) / subject / "pointcloud.ply"
        if not ply.exists():
            return JSONResponse({"error": f"no point cloud for {subject}"}, status_code=404)
        pts, cols = read_ply(ply)
        n = len(pts)
        if n > 60000:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, 60000, replace=False)
            pts, cols = pts[idx], cols[idx]
        center = pts.mean(axis=0)
        radius = float(np.linalg.norm(pts - center, axis=1).max()) or 1.0
        pts = ((pts - center) / radius).astype(np.float32)
        return {
            "num_total": n,
            "num_shown": len(pts),
            "points": base64.b64encode(pts.tobytes()).decode(),
            "colors": base64.b64encode(cols.astype(np.uint8).tobytes()).decode(),
        }

    # ------------------------------------------------------- download PLY / Viser
    @app.get("/api/download/pointcloud/{subject}")
    def download_pointcloud(subject: str):
        ply = Path(cfg["output_dir"]) / subject / "pointcloud.ply"
        if not ply.exists():
            return JSONResponse({"error": f"no point cloud for {subject}"}, status_code=404)
        return FileResponse(
            str(ply),
            filename=f"{subject}_pointcloud.ply",
            media_type="application/octet-stream",
        )

    @app.post("/api/viser/open/{subject}")
    def viser_open(subject: str):
        import subprocess

        ply = Path(cfg["output_dir"]) / subject / "pointcloud.ply"
        if not ply.exists():
            return JSONResponse({"error": f"no point cloud for {subject}"}, status_code=404)
        cams = Path(cfg["output_dir"]) / subject / "cameras.json"
        cams_arg = str(cams) if cams.exists() else "None"

        port = 8080
        # 先杀掉占用 8080 的旧 Viser（本项目两个启动器命名）
        subprocess.run(["pkill", "-f", "view_any"], capture_output=True)
        subprocess.run(["pkill", "-f", "21_view_ply"], capture_output=True)

        log = PROJECT_ROOT / "outputs" / "reconstruct" / f"viser_{subject}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            proc = subprocess.Popen(
                [
                    sys.executable,  # 与 web 服务同环境（vggt conda env）
                    "-u",
                    str(PROJECT_ROOT / "scripts" / "21_view_ply.py"),
                    str(ply),
                    cams_arg,
                    str(port),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return {"port": port, "pid": proc.pid, "url": f"http://localhost:{port}", "log": str(log)}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive multi-subject selector")
    parser.add_argument("--config", default=None, help="config yaml (default: configs/default.yaml)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    host = args.host or cfg.get("subjects", {}).get("server", {}).get("host", "127.0.0.1")
    port = args.port or cfg.get("subjects", {}).get("server", {}).get("port", 7861)

    import uvicorn

    app = create_app(cfg)
    print(f"Subjects selector: http://{host}:{port}")
    print(f"  florence: {cfg['subjects']['florence_model']}")
    print(f"  export:   {cfg['paths']['subjects_dir']}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
