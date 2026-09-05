"""Day 3–4 · 把流水线包成 agent 工具（JSON Schema + 实现）。

对应 src/workflow.py 里的两个核心函数:
  - generate_views_for_subjects(...)  → 02b：每个主体生成 6 视角（Zero123++）
  - reconstruct_subjects(...)         → 02c：每个主体重建点云（VGGT pointmap）

本机没有 GPU / torch 也能开发: 设 AGENT_MOCK=1（不设则自动判断——
本地没有 torch 就自动 mock）。mock 模式下视角是纯 PIL 假图、
重建走框架自带的 dummy 后端；到服务器上 AGENT_MOCK=0 就是真的
Zero123++ + VGGT，代码一行不用改。

每个工具 = TOOL_DEFS 里一份 JSON Schema + IMPLEMENTATIONS 里一个函数。
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- mock 判断
def mock_enabled() -> bool:
    env = os.environ.get("AGENT_MOCK")
    if env is not None:
        return env not in ("0", "false", "False", "")
    # 本地没有 torch → 自动切 mock（torch 是 Zero123++/VGGT 的硬依赖）
    return importlib.util.find_spec("torch") is None


# ---------------------------------------------------------------- 配置加载
def load_cfg() -> dict:
    from src.reconstruct.config import load_config

    cfg = load_config()
    if mock_enabled():
        cfg["backend"] = "dummy"        # 框架自带的无模型后端（本地可跑）
    return cfg


def _progress(i, total, stage, name, info):
    """workflow 的 progress 回调 → 打印成步骤日志（Day 7 会接进 SSE）。"""
    status = info.get("status")
    msg = f"[tool:{stage}] {name}: {status}"
    if info.get("error"):
        msg += f" ({info['error']})"
    print(msg)


# ---------------------------------------------------------------- 工具 1
def list_subjects(_args: dict) -> dict:
    """读 outputs/subjects/subjects.json，返回所有主体。"""
    cfg = load_cfg()
    p = Path(cfg["paths"]["subjects_dir"]) / "subjects.json"
    if not p.exists():
        return {
            "error": f"没有 subjects.json: {p}。"
                     "先运行 scripts/10_subjects_web.py 保存主体，"
                     "或本地开发用 python agent/run_agent_workflow.py --make-sample"
        }
    doc = json.loads(p.read_text())
    subs = doc.get("subjects", [])
    return {
        "subjects": [
            {
                "name": s["name"],
                "subject_file": s.get("subject_file"),
                "prompts": s.get("prompts", {}),
            }
            for s in subs
        ],
        "count": len(subs),
    }


# ---------------------------------------------------------------- 工具 2
def generate_subject_views(args: dict) -> dict:
    """02b：对指定主体跑 Zero123++，每主体写 6 张 view_XX.png + views.json。"""
    cfg = load_cfg()
    subjects_dir = Path(cfg["paths"]["subjects_dir"])
    views_root = Path(cfg["paths"]["views_dir"])
    doc_path = subjects_dir / "subjects.json"
    if not doc_path.exists():
        return {"error": f"没有 subjects.json: {doc_path}"}
    doc = json.loads(doc_path.read_text())

    names = args.get("subject_names") or [s["name"] for s in doc.get("subjects", [])]
    doc["subjects"] = [s for s in doc.get("subjects", []) if s["name"] in names]
    if not doc["subjects"]:
        return {"error": f"主体不存在: {names}"}

    if mock_enabled():
        index = _mock_generate_views(doc, views_root)
    else:
        from src.workflow import generate_views_for_subjects

        index = generate_views_for_subjects(
            doc, subjects_dir, views_root, cfg,
            progress=_progress, stop_on_error=False,
        )
    return {
        "done": [
            {
                "name": s["name"],
                "views_dir": s["views_dir"],
                "views": 6,
            }
            for s in index
        ],
        "failed": [
            {"name": s["name"], "error": s.get("error")}
            for s in index
            if s.get("error")
        ],
    }


def _mock_generate_views(doc: dict, views_root: Path) -> list[dict]:
    """无 torch 的假 02b：写 6 张纯色假视角 + views.json + subjects_index.json。"""
    from PIL import Image
    import numpy as np

    from src.io_utils import save_json

    index: list[dict] = []
    for i, s in enumerate(doc["subjects"]):
        name = s["name"]
        out_dir = views_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        base = 70 + 50 * i
        for j in range(6):
            rgb = np.full((512, 512, 3), base + 10 * j, dtype=np.uint8)
            Image.fromarray(rgb).save(out_dir / f"view_{j:02d}.png")
        save_json(
            {"subject": name, "views": [{"file": f"view_{j:02d}.png"} for j in range(6)]},
            out_dir / "views.json",
        )
        index.append({"name": name, "views_dir": str(out_dir), "prompts": s.get("prompts", {})})
        print(f"[mock views] {name}: 6 views → {out_dir}")
    save_json({"subjects": index}, views_root / "subjects_index.json")
    return index


# ---------------------------------------------------------------- 工具 3
def reconstruct_subjects(args: dict) -> dict:
    """02c：对指定主体重建点云。可传 conf_percentile 调置信度过滤（Day 5 重跑用）。"""
    cfg = load_cfg()
    views_root = Path(cfg["paths"]["views_dir"])
    index_path = views_root / "subjects_index.json"
    if not index_path.exists():
        return {"error": f"没有 subjects_index.json: {index_path}（先调 generate_subject_views）"}
    doc = json.loads(index_path.read_text())

    names = args.get("subject_names") or [s["name"] for s in doc.get("subjects", [])]
    doc["subjects"] = [s for s in doc.get("subjects", []) if s["name"] in names]
    if not doc["subjects"]:
        return {"error": f"主体不存在: {names}"}

    percentile = args.get("conf_percentile")
    if percentile is not None:
        cfg["vggt"]["conf_percentile"] = int(percentile)
    # 改了过滤参数就必须重跑（框架按产物缓存，不 force 会跳过）
    force = bool(percentile is not None or args.get("force"))

    from src.workflow import reconstruct_subjects as run_recon

    run_recon(
        doc, Path(cfg["output_dir"]), cfg,
        progress=_progress, force=force, stop_on_error=False,
    )

    results = []
    for s in doc["subjects"]:
        out = Path(cfg["output_dir"]) / s["name"]
        stats_p = out / "stats.json"
        if stats_p.exists():
            st = json.loads(stats_p.read_text())
            results.append(
                {
                    "name": s["name"],
                    "pointcloud_ply": str(out / "pointcloud.ply"),
                    "num_points": st.get("num_points_filtered"),
                    "conf_percentile": st.get("conf_percentile"),
                    "conf_threshold": st.get("conf_threshold"),
                }
            )
        else:
            results.append({"name": s["name"], "error": "重建失败（没有 stats.json）"})
    return {"subjects": results}


# ---------------------------------------------------------------- 注册表
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "list_subjects",
            "description": "列出当前已保存的主体（outputs/subjects/subjects.json），"
                           "返回每个主体的名字和抠图文件。开始干活前先调用它。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_subject_views",
            "description": "对主体生成 6 个视角（Zero123++）。"
                           "参数 subject_names 不传则处理全部主体。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要处理的主体名列表，例如 [\"horse\", \"lady\"]",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reconstruct_subjects",
            "description": "对已有视角的主体重建点云（VGGT pointmap）。"
                           "conf_percentile 是置信度过滤百分位，默认 65；"
                           "点云太脏可调大（70/80），点云太少可调小。"
                           "改了 conf_percentile 会强制重跑该主体。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要重建的主体名列表；不传则重建全部",
                    },
                    "conf_percentile": {
                        "type": "number",
                        "description": "置信度过滤百分位（0-100），可选",
                    },
                },
                "required": [],
            },
        },
    },
]

IMPLEMENTATIONS = {
    "list_subjects": list_subjects,
    "generate_subject_views": generate_subject_views,
    "reconstruct_subjects": reconstruct_subjects,
}


def execute_tool(name: str, args: dict) -> dict:
    """执行工具并兜底异常：错误也作为结果回传，让 LLM 自己决定怎么补救。"""
    fn = IMPLEMENTATIONS.get(name)
    if fn is None:
        return {"error": f"未知工具: {name}"}
    try:
        return fn(args)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def get_tools() -> list[dict]:
    return list(TOOL_DEFS)


def add_tools(defs: list[dict], impls: dict) -> None:
    """Day 5–6 扩展用：把质检工具注册进来。"""
    TOOL_DEFS.extend(defs)
    IMPLEMENTATIONS.update(impls)
