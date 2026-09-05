"""LLM 客户端配置：DeepSeek / Qwen（都是 OpenAI 兼容接口）。

用法（二选一）:
    export DEEPSEEK_API_KEY=sk-...            # https://platform.deepseek.com/
    # 或
    export QWEN_API_KEY=sk-...                # https://bailian.console.aliyun.com/
    export LLM_PROVIDER=qwen                  # 默认 deepseek

函数调用注意事项:
    - DeepSeek 用 deepseek-chat（V3）做 function calling 最稳；
      deepseek-reasoner（R1）目前不建议用于工具调用。
    - Qwen 用 qwen-plus / qwen-max；qwen-turbo 也能用。
"""
from __future__ import annotations

import os

from openai import OpenAI

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "api_key_env": "QWEN_API_KEY",
    },
}


def get_llm(provider: str | None = None, model: str | None = None):
    """返回 (OpenAI client, model_name)。provider 和 model 均可被环境变量覆盖。"""
    name = provider or os.environ.get("LLM_PROVIDER", "deepseek")
    if name not in PROVIDERS:
        raise SystemExit(f"未知 provider: {name}（可选: {list(PROVIDERS)}）")
    p = PROVIDERS[name]
    api_key = os.environ.get(p["api_key_env"])
    if not api_key:
        raise SystemExit(
            f"缺少环境变量 {p['api_key_env']}。\n"
            f"  DeepSeek key: https://platform.deepseek.com/\n"
            f"  Qwen key:     https://bailian.console.aliyun.com/\n"
            f"然后: export {p['api_key_env']}=sk-xxx"
        )
    return OpenAI(base_url=p["base_url"], api_key=api_key), model or p["model"]
