#!/usr/bin/env python3
"""Day 1–2 · 最小可跑的 function calling 示例（openai SDK 调 DeepSeek/Qwen）。

先装 SDK（本仓库已装）:
    .venv/bin/pip install openai        # 或 pip install openai

跑 DeepSeek:
    export DEEPSEEK_API_KEY=sk-xxxx     # https://platform.deepseek.com/ 创建
    .venv/bin/python agent/day01_function_calling.py

换 Qwen（阿里云百炼）:
    export LLM_PROVIDER=qwen
    export QWEN_API_KEY=sk-xxxx         # https://bailian.console.aliyun.com/ 创建
    .venv/bin/python agent/day01_function_calling.py

它会让模型"想"调用 add / multiply 工具 → 代码这边执行 → 把结果回传 →
模型给出最终答案。这就是 function calling，也就是 ReAct 循环
（Thought → Action → Observation → Answer）里"Action + Observation"的部分。

这个文件刻意自包含、不带任何 import agent.*，是为了让你把每一行都看懂。
Day 3–4 你会把它升级成调用你真实流水线的工具。
"""
import json
import os

from openai import OpenAI

# ---------------------------------------------------------------- 配置
PROVIDERS = {
    # 两个都是 OpenAI 兼容接口，所以只需换 base_url + model。
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",       # V3：支持 function calling
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",           # 百炼通用对话模型
        "api_key_env": "QWEN_API_KEY",
    },
}


def get_client():
    provider = os.environ.get("LLM_PROVIDER", "deepseek")
    if provider not in PROVIDERS:
        raise SystemExit(f"未知 provider: {provider}（可选: {list(PROVIDERS)}）")
    p = PROVIDERS[provider]
    api_key = os.environ.get(p["api_key_env"])
    if not api_key:
        raise SystemExit(
            f"缺少环境变量 {p['api_key_env']}。\n"
            f"DeepSeek: https://platform.deepseek.com/ 创建 key\n"
            f"Qwen:     https://bailian.console.aliyun.com/ 创建 key"
        )
    return OpenAI(base_url=p["base_url"], api_key=api_key), p["model"]


# ---------------------------------------------------------------- 工具定义
# 每个工具 = 一份 JSON Schema（告诉模型"有什么工具、参数长什么样"）
#           + 一个 Python 函数（真正干活的是你）。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "把两个数相加，返回整数结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个加数"},
                    "b": {"type": "number", "description": "第二个加数"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "把两个数相乘，返回整数结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个乘数"},
                    "b": {"type": "number", "description": "第二个乘数"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


def execute_tool(name: str, args: dict):
    """你（代码）这一侧执行工具。返回值要能 json.dumps，尽量简洁。"""
    if name == "add":
        return {"result": args["a"] + args["b"]}
    if name == "multiply":
        return {"result": args["a"] * args["b"]}
    return {"error": f"未知工具: {name}"}


# ---------------------------------------------------------------- 主循环
def main():
    client, model = get_client()

    # 模型看不到工具实现，只看到上面的 JSON Schema；执行永远发生在你这边。
    messages = [
        {"role": "system", "content": "你是计算助手。需要计算时调用工具，不要心算。"},
        {"role": "user", "content": "请帮我计算 (1234 * 5678) + 42 的结果。"},
    ]

    print(f"== provider/model: {model} ==")
    for step in range(1, 8):                      # 步数上限，防止死循环
        print(f"\n--- step {step} ---")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,                          # 把工具清单发给模型
        )
        msg = resp.choices[0].message
        messages.append(msg)                      # 模型的话（可能含 tool_calls）进历史

        if not msg.tool_calls:
            print(f"[answer] {msg.content}")
            return

        for tc in msg.tool_calls:                 # 模型想调一个或多个工具
            fn = tc.function
            args = json.loads(fn.arguments)
            print(f"[model → tool] {fn.name}({json.dumps(args)})")
            result = execute_tool(fn.name, args)
            print(f"[tool → model] {json.dumps(result, ensure_ascii=False)}")
            messages.append({                     # 工具结果以 role="tool" 回传
                "role": "tool",
                "tool_call_id": tc.id,            # 必须带上这个 id，模型才能对上号
                "content": json.dumps(result, ensure_ascii=False),
            })
    print("[timeout] 步数用完了")


if __name__ == "__main__":
    main()
