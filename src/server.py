"""Web UI for SAM2 selection, Zero123++ views and VGGT reconstruction."""
import argparse
import base64
import copy
import io
import os
import sys
import threading
from pathlib import Path
from contextlib import asynccontextmanager

from src.config import PROJECT_ROOT, load_config

# 必须在任何 HuggingFace 库导入前设置（服务器直连 huggingface.co 不通）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from src.io_utils import load_json
from src.subjects import (
    Subject,
    sanitize_name,
    mask_to_png_bytes,
    save_subjects,
    segment_subject,
)


def create_app(cfg: dict) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        yield
        stop_viewer()

    app = FastAPI(title="Photo to Mesh · SAM2 / Zero123++ / VGGT", lifespan=lifespan)

    for directory in [cfg["paths"]["subjects_dir"], cfg["paths"]["views_dir"], cfg["output_dir"]]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    app.mount("/outputs/views", StaticFiles(directory=cfg["paths"]["views_dir"]), name="views")
    workflow_lock = threading.Lock()

    state: dict = {
        "cfg": cfg,
        "image": None,       # PIL RGB
        "image_rgb": None,   # np (H,W,3) uint8
        "image_name": "uploaded.png",
        "predictor": None,   # SAM2 predictor (lazy)
        "saved": False,
        "viewer": None,
        "workflow": {
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

    @app.get("/")
    def index():
        return FileResponse(PROJECT_ROOT / "web" / "index.html")

    @app.get("/api/config")
    def public_config():
        return {"view_count": 7 if cfg["views"].get("include_original", True) else 6}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        if state["workflow"]["status"] == "running":
            return JSONResponse({"error": "工作流运行中，请完成后再上传图片"}, status_code=409)
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        state["image"] = img
        state["image_rgb"] = np.asarray(img)
        state["image_name"] = file.filename or "uploaded.png"
        state["saved"] = False
        print(f"[upload] {state['image_name']} {img.size}")
        return {"size": [img.size[0], img.size[1]], "name": state["image_name"]}

    @app.get("/api/image")
    def image():
        if state["image"] is None:
            return Response(status_code=404)
        buf = io.BytesIO()
        state["image"].save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    class SegmentIn(BaseModel):
        boxes: list = []
        points: list = []
        labels: list = []

    @app.post("/api/segment")
    def segment(body: SegmentIn):
        if state["workflow"]["status"] == "running":
            return JSONResponse({"error": "工作流运行中"}, status_code=409)
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
        if state["workflow"]["status"] == "running":
            return JSONResponse({"error": "工作流运行中"}, status_code=409)
        if not body.subjects:
            return JSONResponse({"error": "请至少添加并分割一个主体"}, status_code=400)
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
        state["saved"] = True
        return doc

    # ------------------------------------------------------- workflow
    def _run_workflow(run_cfg: dict) -> None:
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
                            [f"views/{name}/{v['file']}" for v in load_json(vdir / "views.json")["views"]]
                            if vdir.exists()
                            else []
                        )
                st["log"] = (st["log"] + [f"[{stage}] {name}: {info.get('status')}"])[-60:]

            from src.zero123 import generate_views_for_subjects
            from src.reconstruction import reconstruct_subjects
            # SAM2 can be loaded again for the next edit; release VRAM for reconstruction.
            state["predictor"] = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            generate_views_for_subjects(doc, subjects_dir, views_root, run_cfg, progress=progress)
            index_doc = load_json(views_root / "subjects_index.json")
            reconstruct_subjects(
                index_doc, base_out, run_cfg, progress=progress
            )
            st["status"] = "done"
        except Exception as e:  # noqa: BLE001
            st["status"] = "error"
            st["error"] = str(e)
            print(f"[workflow] failed: {e}")

    @app.post("/api/workflow/start")
    def workflow_start(view_count: int | None = Query(default=None, ge=6, le=7)):
        with workflow_lock:
            if state["workflow"]["status"] == "running":
                return JSONResponse({"error": "workflow already running"}, status_code=409)
            if not state["saved"]:
                return JSONResponse({"error": "请先上传图片、选择主体并保存"}, status_code=400)
            run_cfg = copy.deepcopy(cfg)
            if view_count is not None:
                run_cfg["views"]["include_original"] = (view_count == 7)
            count = 7 if run_cfg["views"].get("include_original", True) else 6
            state["workflow"].update(status="running", subjects=[], log=[], error=None, view_count=count)
            threading.Thread(target=_run_workflow, args=(run_cfg,), daemon=True).start()
        return {"status": "started", "view_count": count}

    @app.get("/api/workflow/status")
    def workflow_status():
        return state["workflow"]

    def subject_output(subject: str) -> Path:
        if sanitize_name(subject) != subject:
            raise HTTPException(status_code=400, detail="Invalid subject name")
        return Path(cfg["output_dir"]) / subject

    # ------------------------------------------------------- point cloud preview
    @app.get("/api/pointcloud/{subject}")
    def pointcloud(subject: str):
        from src.pointcloud_io import read_ply

        ply = subject_output(subject) / "pointcloud.ply"
        if not ply.exists():
            return JSONResponse({"error": f"no point cloud for {subject}"}, status_code=404)
        pts, cols = read_ply(ply)
        n = len(pts)
        if not n:
            return JSONResponse({"error": "点云为空，请重新运行重建"}, status_code=422)
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
        ply = subject_output(subject) / "pointcloud.ply"
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

        ply = subject_output(subject) / "pointcloud.ply"
        if not ply.exists():
            return JSONResponse({"error": f"no point cloud for {subject}"}, status_code=404)
        cams = subject_output(subject) / "cameras.json"

        port = 8080
        # Stop only the viewer owned by this Web app.
        previous = state["viewer"]
        if previous is not None and previous.poll() is None:
            previous.terminate()
            try:
                previous.wait(timeout=5)
            except subprocess.TimeoutExpired:
                previous.kill()
                previous.wait()

        log = Path(cfg["output_dir"]) / f"viser_{subject}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            proc = subprocess.Popen(
                [
                    sys.executable,  # 与 web 服务同环境（vggt conda env）
                    "-u",
                    "-m", "src.viewer", str(ply),
                    *(["--cameras", str(cams)] if cams.exists() else []),
                    "--port", str(port),
                ],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(PROJECT_ROOT),
            )
        state["viewer"] = proc
        return {"port": port, "pid": proc.pid, "url": f"http://localhost:{port}", "log": str(log)}

    def stop_viewer():
        proc = state["viewer"]
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM2 → Zero123++ → VGGT Web application")
    parser.add_argument("--config", default=None, help="config yaml (default: configs/default.yaml)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    host = args.host or cfg.get("server", {}).get("host", "127.0.0.1")
    port = args.port or cfg.get("server", {}).get("port", 7861)

    import uvicorn

    app = create_app(cfg)
    print(f"Subjects selector: http://{host}:{port}")
    print(f"  export:   {cfg['paths']['subjects_dir']}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
