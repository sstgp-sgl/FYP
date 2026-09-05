#!/usr/bin/env python3
"""Day 3–4 demo · 让 agent 替你跑 02b → 02c。

本地（无 GPU，自动 mock）:
    .venv/bin/python agent/run_agent_workflow.py --make-sample   # 造 2 个假主体
    .venv/bin/python agent/run_agent_workflow.py                 # 视角=假图, 重建=dummy

服务器（真 Zero123++ + VGGT）:
    export DEEPSEEK_API_KEY=sk-xxx
    export AGENT_MOCK=0
    python agent/run_agent_workflow.py                           # 同一段代码
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from agent.loop import Agent

SAMPLE_TASK = (
    "请查看当前有哪些主体，然后为每个主体生成视角并重建点云，"
    "最后总结每个主体的点云路径和点数。"
)


def make_sample(subjects_dir: Path) -> None:
    """造一个假的 subjects.json（2 个主体，各一张假抠图），供本地 mock 测试。"""
    subjects_dir.mkdir(parents=True, exist_ok=True)
    subs = []
    for i, name in enumerate(["horse", "lady"]):
        img = np.full((512, 512, 3), 90 + 60 * i, dtype=np.uint8)
        f = subjects_dir / name / "subject.png"
        f.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(f)
        subs.append({"name": name, "subject_file": f"{name}/subject.png", "prompts": {}})
    (subjects_dir / "subjects.json").write_text(
        json.dumps({"subjects": subs}, indent=2)
    )
    print(f"sample subjects → {subjects_dir / 'subjects.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-sample", action="store_true", help="造假主体（本地 mock 用）")
    ap.add_argument("--task", default=SAMPLE_TASK)
    args = ap.parse_args()

    if args.make_sample:
        make_sample(Path("outputs/subjects"))
        return

    answer = Agent().run(args.task)
    print("\n" + "=" * 60)
    print("最终回答:")
    print(answer)


if __name__ == "__main__":
    main()
