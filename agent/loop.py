"""Day 3–4 · 手写 agent loop（不依赖任何 agent 框架）。

核心只有一件事，反复执行：
    把完整对话历史发给 LLM
      → 如果它想调工具（tool_calls）: 我们在本地执行，把结果以 role="tool"
        回传，继续下一轮
      → 如果它直接回答: 结束，这就是最终答案

这就是 ReAct 循环（Thought → Action → Observation → Answer）的代码形态。
你 Day 1–2 手写过的那个小循环，只是把"add/multiply"换成了你的流水线工具。
"""
from __future__ import annotations

import json
import time
from typing import Callable

from agent.llm import get_llm
from agent.tools import execute_tool, get_tools

SYSTEM_PROMPT = (
    "你是一个照片转 3D 流水线助手，负责指挥多主体重建流程。\n"
    "规则：\n"
    "1. 先调用 list_subjects 查看有哪些主体；\n"
    "2. 用 generate_subject_views 生成视角，再用 reconstruct_subjects 重建点云；\n"
    "3. 工具返回 error 时，读懂错误并尝试修复后重试（同一问题最多 2 次）；\n"
    "4. 全部完成后，用简洁中文总结每个主体的点云路径和点数。\n"
)

# on_event(step, kind, payload)：kind ∈ tool_call | tool_result | answer
EventCallback = Callable[[int, str, dict], None]


class Agent:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        max_steps: int = 15,
        verbose: bool = True,
    ):
        self.client, self.model = get_llm(provider, model)
        self.max_steps = max_steps
        self.verbose = verbose

    # ------------------------------------------------------------ 主循环
    def run(self, user_task: str, system_prompt: str = SYSTEM_PROMPT,
            on_event: EventCallback | None = None) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_task},
        ]
        for step in range(1, self.max_steps + 1):
            resp = self._chat(messages)
            msg = resp.choices[0].message
            messages.append(msg)                       # 模型这句话进历史

            if not msg.tool_calls:
                answer = msg.content or ""
                self._emit(on_event, step, "answer", {"content": answer})
                return answer

            for tc in msg.tool_calls:                  # 模型想调一个或多个工具
                fn = tc.function
                try:
                    args = json.loads(fn.arguments) or {}
                except json.JSONDecodeError:
                    args = {}
                self._emit(on_event, step, "tool_call",
                           {"name": fn.name, "arguments": args})

                result = execute_tool(fn.name, args)   # ★ 执行永远在你这边
                self._emit(on_event, step, "tool_result",
                           {"name": fn.name, "result": result})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,             # 必须回传这个 id
                    "content": json.dumps(result, ensure_ascii=False),
                })
        raise RuntimeError(f"超过 {self.max_steps} 步仍未完成")

    # ------------------------------------------------------------ 工具方法
    def _chat(self, messages, tries: int = 3):
        """带退避重试的 API 调用（网络抖动/限流时自动重试）。"""
        last_err = None
        for i in range(tries):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=get_tools(),
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                if i < tries - 1:
                    time.sleep(1.5 * (i + 1))
        raise last_err

    def _emit(self, on_event, step, kind, payload):
        if on_event:
            on_event(step, kind, payload)
        if self.verbose:
            if kind == "tool_call":
                print(f"  step {step} [model → tool] {payload['name']}"
                      f"({json.dumps(payload['arguments'], ensure_ascii=False)})")
            elif kind == "tool_result":
                print(f"  step {step} [tool → model] "
                      f"{json.dumps(payload['result'], ensure_ascii=False)[:300]}")
            else:
                print(f"  step {step} [answer] {payload['content']}")


if __name__ == "__main__":
    # 命令行直接试跑（等价于 run_agent_workflow.py）
    agent = Agent()
    task = (
        "请查看当前有哪些主体，然后为每个主体生成视角并重建点云，"
        "最后总结每个主体的点云路径和点数。"
    )
    print(agent.run(task))
