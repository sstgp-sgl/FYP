"""Day 5–6 · 质检工具 + agent 自动决定是否换 conf_percentile 重跑。

新增两个工具:
  - inspect_pointcloud(subject):  读 stats.json + 点云几何统计，给出 verdict
  - rerun_reconstruction(subject, conf_percentile): 换过滤参数强制重跑

agent 策略（写进 system prompt）:
    重建后必须 inspect_pointcloud 检查；verdict 不是 ok 时，
    按 reasons 的提示换 conf_percentile 重跑，最多 2 次。

跑法（本地 mock 也能跑通）:
    .venv/bin/python -m agent.day05_qc            # 命令行演示：先重建再质检
    .venv/bin/python agent/run_agent_workflow.py --task "$(cat <<'EOF'
请重建所有主体；每个主体重建后调用 inspect_pointcloud 质检，
如果点云过小就用 rerun_reconstruction 调低 conf_percentile 重跑一次，
然后总结最终结果。
EOF
)"

真实的"渲染对比"质检（需要 GPU 上跑 3DGS 渲染新视角再和原图比 SSIM）
属于加分项，见 docs/AGENT_LEARNING.md Day 5–6 的说明。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from agent.tools import add_tools, load_cfg, reconstruct_subjects
from agent.loop import Agent

# ---------------------------------------------------------------- 启发式质检
MIN_POINTS = 3000          # 点云小于这个数 → 视为失败/过小
MIN_KEPT_FRACTION = 0.15   # 过滤后点数 / 原始点数 小于这个 → 过滤太狠


def evaluate_stats(stats: dict) -> dict:
    """纯规则质检。返回 {"verdict": ok|suspect|fail, "reasons": [...]}。"""
    reasons = []
    n = stats.get("num_points_filtered", 0)
    raw = stats.get("num_points_raw", 0)
    if n < MIN_POINTS:
        reasons.append(f"点云过小: 只有 {n} 点（阈值 {MIN_POINTS}），"
                       f"建议把 conf_percentile 调低（如 {max(0, stats.get('conf_percentile', 65) - 15)}）")
    if raw > 0 and n / raw < MIN_KEPT_FRACTION:
        reasons.append(f"过滤太狠: 只剩 {n/raw:.1%} 的点（阈值 {MIN_KEPT_FRACTION:.0%}），"
                       f"建议调低 conf_percentile")
    if not reasons:
        return {"verdict": "ok", "reasons": []}
    return {"verdict": "fail", "reasons": reasons}


# ---------------------------------------------------------------- 工具 4
def inspect_pointcloud(args: dict) -> dict:
    """读某主体 outputs/reconstruct/<subject>/stats.json + PLY 几何，给质检结论。"""
    subject = args.get("subject")
    if not subject:
        return {"error": "需要参数 subject"}
    cfg = load_cfg()
    out = Path(cfg["output_dir"]) / subject
    stats_p = out / "stats.json"
    if not stats_p.exists():
        return {"error": f"没有 stats.json: {stats_p}（先跑 reconstruct_subjects）"}

    stats = json.loads(stats_p.read_text())
    qc = evaluate_stats(stats)

    # 点云几何：包围盒尺寸 + 密度（不需要 torch，直接读 PLY）
    from src.pointcloud_io import read_ply

    ply = out / "pointcloud.ply"
    geometry = {}
    if ply.exists():
        pts, _ = read_ply(ply)
        if len(pts):
            bbox = pts.max(axis=0) - pts.min(axis=0)
            diag = float(np.linalg.norm(bbox))
            geometry = {
                "n_points": int(len(pts)),
                "bbox_xyz": [round(float(v), 4) for v in bbox],
                "bbox_diagonal": round(diag, 4),
            }
            if diag > 1e-6:
                geometry["density"] = round(float(len(pts) / (diag ** 3)), 1)

    return {"subject": subject, "stats": stats, "geometry": geometry, "qc": qc}


# ---------------------------------------------------------------- 工具 5
def rerun_reconstruction(args: dict) -> dict:
    """换 conf_percentile 强制重跑某主体的重建。"""
    subject = args.get("subject")
    percentile = args.get("conf_percentile")
    if not subject:
        return {"error": "需要参数 subject"}
    if percentile is None:
        return {"error": "需要参数 conf_percentile"}
    return reconstruct_subjects(
        {"subject_names": [subject], "conf_percentile": int(percentile)}
    )


QC_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_pointcloud",
            "description": "质检一个主体的点云：读取 stats.json 和 PLY 几何，"
                           "返回 qc.verdict（ok/fail）和 reasons。"
                           "重建完成后必须调用它检查质量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "主体名，如 horse"}
                },
                "required": ["subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rerun_reconstruction",
            "description": "用新的 conf_percentile 强制重跑一个主体的重建。"
                           "点云过小→调低百分位；点云太脏/太多白点→调高。",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "主体名"},
                    "conf_percentile": {
                        "type": "number",
                        "description": "新的置信度过滤百分位（0-100）",
                    },
                },
                "required": ["subject", "conf_percentile"],
            },
        },
    },
]

QC_IMPLS = {
    "inspect_pointcloud": inspect_pointcloud,
    "rerun_reconstruction": rerun_reconstruction,
}

QC_SYSTEM_PROMPT = (
    "你是一个照片转 3D 流水线助手。流程：\n"
    "1. list_subjects 查看主体；\n"
    "2. generate_subject_views 生成视角；\n"
    "3. reconstruct_subjects 重建点云；\n"
    "4. 每个主体重建后必须调用 inspect_pointcloud 质检；\n"
    "5. 若 qc.verdict 不是 ok，按 reasons 的提示用 rerun_reconstruction "
    "换 conf_percentile 重跑（同一主体最多 2 次）；\n"
    "6. 全部完成后用简洁中文总结每个主体的点云路径、点数和最终质检结论。\n"
)


def main() -> None:
    """命令行演示：注册质检工具 → 跑一个完整任务。"""
    add_tools(QC_TOOL_DEFS, QC_IMPLS)
    task = (
        "请查看当前有哪些主体，为每个主体生成视角并重建点云；"
        "每个主体重建后调用 inspect_pointcloud 质检，"
        "如果点云过小就 rerun_reconstruction 调低 conf_percentile 重跑一次。"
        "最后总结每个主体的点云路径、点数和质检结论。"
    )
    print(Agent().run(task, system_prompt=QC_SYSTEM_PROMPT))


if __name__ == "__main__":
    main()
