# Agent 学习路径 · 手把手教程（照片 → 3D 流水线的 authoring 层）

> 目标：1–2 周内，从"会调 LLM API"到"写出一个能自己指挥 02b→02c 流程、
> 自己做质检重跑、网页上实时看步骤日志的 Agent"，最后接 Qwen-VL 做叙事生成。
> 本教程全部对着**你自己仓库里的真实函数**讲，参考实现都在 [`agent/`](../agent/) 目录。

---

## 0. 最终长什么样（先记住目标）

```
浏览器(Agent 工作台)
   │  POST /api/agent/run  {"task":"重建所有主体"}
   ▼
FastAPI ──→ 后台线程跑 Agent loop（agent/loop.py）
              │ 每步: LLM 想调工具 → 执行 → 结果回传
              ▼
          工具层（agent/tools.py）
              ├─ list_subjects         读 outputs/subjects/subjects.json
              ├─ generate_subject_views → src.workflow.generate_views_for_subjects（02b）
              ├─ reconstruct_subjects   → src.workflow.reconstruct_subjects（02c）
              ├─ inspect_pointcloud     质检（Day 5–6）
              └─ rerun_reconstruction   换 conf_percentile 重跑（Day 5–6）
              ▼
          src/workflow.py（你已有的代码，一行不改）
              ▼
          Zero123++ 生成视角 → VGGT pointmap 重建点云
   │
   ▼  SSE 实时推送: step 1 调用 list_subjects → 返回 {...} → step 2 …
网页逐行显示步骤日志
```

**核心心智模型（ReAct）**：LLM 不直接执行任何事，它只"想"——
每一步它输出"我要调哪个工具、参数是什么"，**执行永远发生在你的代码里**，
执行结果（Observation）回传给它，它再决定下一步（Thought），直到给出最终答案（Answer）。
这就是 `Thought → Action → Observation → … → Answer`。

---

## 1. 环境准备（10 分钟）

### 1.1 本地（本机，开发用）

```bash
cd /Users/apple/project/photo-to-mesh
.venv/bin/pip install openai        # 已装好（v3.x，老教程的 v1 写法仍然兼容）
.venv/bin/python -c "import openai; print(openai.__version__)"
```

- 本地没有 GPU / torch 没关系：`agent/` 的代码带 **mock 模式**（`AGENT_MOCK=1`，
  或本地检测不到 torch 时自动开启）——视角用纯 PIL 假图、重建走框架自带的
  `dummy` 后端（无模型、确定性出点云），**整个 agent 逻辑在本机就能跑通**。

### 1.2 服务器（AutoDL，跑真实模型）

```bash
ssh autodl6     # 或完整地址 connect.cqa1.seetacloud.com:20781
cd /root/autodl-tmp/photo-to-mesh
export PY=/root/miniconda3/envs/vggt/bin/python
$PY -m pip install openai            # 装 SDK
```

把 `agent/` 和 `docs/AGENT_LEARNING.md` 同步到服务器（git 或 scp 都行）。
服务器上跑真实流水线时：`export AGENT_MOCK=0`。

### 1.3 API Key（两个二选一，都是 OpenAI 兼容）

| 厂商 | 拿 key 的地方 | base_url | 模型 |
|------|--------------|----------|------|
| DeepSeek | <https://platform.deepseek.com/> | `https://api.deepseek.com` | `deepseek-chat`（V3，支持 function calling） |
| Qwen（阿里云百炼） | <https://bailian.console.aliyun.com/> | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` / `qwen-max` |

```bash
export DEEPSEEK_API_KEY=sk-xxxx      # 或 export QWEN_API_KEY=sk-xxxx
```

> ⚠️ DeepSeek 的 `deepseek-reasoner`（R1）目前不建议用于 function calling，
> 工具调用用 `deepseek-chat` 最稳（后续如有变化看官方文档）。
> 配置统一写在 [`agent/llm.py`](../agent/llm.py)，换厂商只需改环境变量。

---

## 2. Day 1–2：用 openai SDK 调通 function calling，理解 ReAct

**今天不碰你的流水线**，先用两个玩具工具（`add` / `multiply`）把机制跑通。

### 2.1 先读（10 分钟，可中英对照）

- [OpenAI Function Calling 官方指南](https://platform.openai.com/docs/guides/function-calling)
- [DeepSeek 官方文档](https://api-docs.deepseek.com/) · [DeepSeek Function Calling 示例](https://api-docs.deepseek.com/guides/function_calling)
- [Qwen 百炼文档](https://help.aliyun.com/zh/model-studio/) · [OpenAI 兼容模式说明](https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope)
- [ReAct 论文（Thought/Action/Observation）](https://arxiv.org/abs/2210.03629)

### 2.2 跑通最小示例（15 分钟）

```bash
export DEEPSEEK_API_KEY=sk-xxxx
.venv/bin/python agent/day01_function_calling.py
```

你会看到类似输出，注意每行代表 ReAct 的一环：

```
--- step 1 ---
[model → tool] multiply({"a": 1234, "b": 5678})   # Action：模型想调工具
[tool → model] {"result": 7006652}                # Observation：你执行完回传
--- step 2 ---
[model → tool] add({"a": 7006652, "b": 42})
[tool → model] {"result": 7006694}
--- step 3 ---
[answer] (1234 * 5678) + 42 = 7006694             # Answer：模型给出最终答案
```

**把 [`agent/day01_function_calling.py`](../agent/day01_function_calling.py) 每一行看懂**，
它只有 4 块：

1. `TOOLS`：每个工具一份 **JSON Schema**（`name` / `description` / `parameters`）——
   模型只看这个，不知道实现；
2. `execute_tool()`：**你**这一侧真正干活的地方；
3. 循环：把完整对话历史发给模型 → 有 `tool_calls` 就逐个执行、以
   `role="tool"` + `tool_call_id` 回传 → 没有就输出最终答案；
4. 步数上限：防止模型陷入死循环（这是手写 loop 必须有的护栏）。

### 2.3 练习（今天必须亲手做，别跳）

1. **加第三个工具** `divide(a, b)`（schema + 实现），让模型算 `(1234*5678+42)/7`。
   → 你会发现：模型会根据 description 自动选工具、组合工具。
2. **换 Qwen 跑一遍**：`export LLM_PROVIDER=qwen` + `QWEN_API_KEY`。
   → 你会验证"OpenAI 兼容"不是吹的：代码一行不用改。
3. **让工具返回 dict 而不是数字**，比如返回 `{"result": x, "unit": "个"}`。
   → 为 Day 3 铺垫：你的工具要返回"LLM 能读懂的中文摘要"，不是原始数据。

**验收标准**：不看参考代码，能默写出上面 4 块结构 + 说出
"为什么工具结果必须带 `tool_call_id` 回传"。

---

## 3. Day 3–4：把 workflow 包成 3 个工具，手写 agent loop 替你跑 02b→02c

### 3.1 先做设计（对着你的真实代码）

打开 [`src/workflow.py`](../src/workflow.py)，你已经有：

| 函数 | 对应流水线 | 输入 | 输出 |
|------|-----------|------|------|
| `generate_views_for_subjects(subjects_doc, subjects_dir, views_root, cfg, ...)` | 02b | `outputs/subjects/subjects.json` | `outputs/views/<name>/view_*.png` + `subjects_index.json` |
| `reconstruct_subjects(index_doc, base_out, cfg, ...)` | 02c | `subjects_index.json` | `outputs/reconstruct/<name>/pointcloud.ply` + `stats.json` |

**练习：先别看下面的参考，自己写 3 份 JSON Schema**，想清楚每个参数：

1. `list_subjects` —— 无参数。返回所有主体名字。
2. `generate_subject_views` —— `subject_names`（可选数组）。
3. `reconstruct_subjects` —— `subject_names`（可选数组）、`conf_percentile`（可选数字）。

写 schema 的要点（LLM 读 description 选工具，描述要像"给实习生写任务说明"）：

```json
{
  "type": "function",
  "function": {
    "name": "generate_subject_views",
    "description": "对主体生成 6 个视角（Zero123++）。subject_names 不传则处理全部。",
    "parameters": {
      "type": "object",
      "properties": {
        "subject_names": {
          "type": "array",
          "items": {"type": "string"},
          "description": "要处理的主体名列表，例如 [\"horse\", \"lady\"]"
        }
      },
      "required": []
    }
  }
}
```

### 3.2 参考实现

- [`agent/tools.py`](../agent/tools.py)：3 个工具的 schema + 实现 + 注册表
  （`TOOL_DEFS` / `IMPLEMENTATIONS` / `execute_tool` / `get_tools`）。
  工具返回值刻意做成**简短的摘要 dict**（如 `{"name", "views_dir", "views": 6}`），
  而不是大段日志——LLM 的上下文窗口有限，喂给它该知道的就够了。
- [`agent/loop.py`](../agent/loop.py)：手写 agent loop（`Agent.run`）。
  对比 Day 1 的循环，多了两样东西：**工具执行失败也作为 `{"error": ...}` 回传**
  （让模型自己读错补救，这就是"带失败重试"的雏形），和 `on_event` 回调
  （Day 7 接 SSE 就用它）。
- [`agent/run_agent_workflow.py`](../agent/run_agent_workflow.py)：入口。

### 3.3 本地跑通（mock，10 分钟）

```bash
.venv/bin/python agent/run_agent_workflow.py --make-sample   # 造 2 个假主体 horse/lady
.venv/bin/python agent/run_agent_workflow.py                 # LLM 自动指挥 02b→02c
```

观察输出：模型会自己决定"先 list → 再 generate → 再 reconstruct"的顺序，
哪怕你的任务只写了"帮我重建"。这就是 agent 替你跑流程的意义。

### 3.4 服务器跑真的（AGENT_MOCK=0）

```bash
ssh autodl6
cd /root/autodl-tmp/photo-to-mesh
export DEEPSEEK_API_KEY=sk-xxxx
export AGENT_MOCK=0
$PY agent/run_agent_workflow.py          # 真的 Zero123++ + VGGT，同一段代码
```

**练习**：给 `list_subjects` 加一个 `names` 过滤参数，让模型只处理指定的主体，
重跑一遍看模型会不会正确使用。

**验收标准**：你能说清楚"为什么工具返回 error 也要回传给 LLM 而不是直接抛异常"。

---

## 4. Day 5–6：质检工具 + agent 自动决定要不要重跑

### 4.1 质检依据（你已有的产物）

每次重建，`outputs/reconstruct/<name>/stats.json` 里已经有过滤统计：

```json
{
  "num_points_raw": 6000,
  "num_points_after_conf": 6000,
  "num_points_filtered": 5857,
  "conf_percentile": 65,
  "conf_threshold": 1.0
}
```

再加两个几何量（直接读 PLY，不需要 torch）：点数、包围盒对角线、密度。
见 [`agent/day05_qc.py`](../agent/day05_qc.py) 的 `evaluate_stats()`：

- 点数 < 3000 → "点云过小" → 建议 **调低** conf_percentile 重跑；
- 过滤后/原始 < 15% → "过滤太狠" → 建议调低；
- 点数很多但 bbox 巨大/破碎 → 建议 **调高** conf_percentile（去噪）。

### 4.2 两个新工具

1. `inspect_pointcloud(subject)` → 返回 `{"stats": {...}, "geometry": {...}, "qc": {"verdict": "ok"|"fail", "reasons": [...]}}`；
2. `rerun_reconstruction(subject, conf_percentile)` → 换百分位强制重跑
   （注意 `reconstruct_subjects` 里：**改了参数必须 `force=True`**，否则框架按产物缓存直接跳过——这是你 README 里踩过的坑）。

### 4.3 让 agent 自动决策

关键不在代码，在 **system prompt 把"质检—决策—重跑"写清楚**
（[`agent/day05_qc.py`](../agent/day05_qc.py) 的 `QC_SYSTEM_PROMPT`）：

```
4. 每个主体重建后必须调用 inspect_pointcloud 质检；
5. 若 qc.verdict 不是 ok，按 reasons 的提示用 rerun_reconstruction
   换 conf_percentile 重跑（同一主体最多 2 次）；
```

跑法：

```bash
.venv/bin/python -m agent.day05_qc          # 命令行演示（本地 mock 即可）
```

**练习**：把 `MIN_POINTS` 调成 100000（假主体只有 ~6000 点），重跑，
观察 agent 会不会真的按规则去 `rerun_reconstruction` 调低百分位、再质检一次。

> 加分项——"渲染对比"质检：服务器上重建后跑 `scripts/04_gaussian.py`
> 训一个轻量 3DGS，渲染新视角和 Zero123++ 原视角比 SSIM/PSNR，低于阈值就重跑。
> 这一步不阻塞主线，先做"点云统计"版。

---

## 5. Day 7–10：网页 Agent 工作台（FastAPI + SSE）

### 5.1 原理：把 07 的"轮询"换成 SSE

你的 [`scripts/10_subjects_web.py`](../scripts/10_subjects_web.py) 现在的工作流是
**前端每 500ms 轮询** `/api/workflow/status` 拿整个 state。SSE（Server-Sent Events）
改成**服务器主动推**：后台线程每走一步，就向一个事件流推一条
`{"type": "tool_call", "step": 1, "name": "list_subjects", ...}`，前端
`EventSource` 收到就显示，零轮询、近实时。

参考实现 [`agent/day07_server.py`](../agent/day07_server.py)，三个端点：

| 端点 | 作用 |
|------|------|
| `POST /api/agent/run` | 收 `{"task": "..."}`，起后台线程跑 `Agent.run`（`on_event` 把每步推进一个线程安全的 `queue.Queue`），返回 `run_id` |
| `GET /api/agent/run/{id}/events` | `text/event-stream`：从 queue 里取事件逐条 `data: {...}\n\n` 推给浏览器，`done`/`error` 后关闭 |
| `GET /` | 一个带 `textarea` + `EventSource` 的最小页面 |

跑法（本机就能试，mock 模式）：

```bash
.venv/bin/python -m uvicorn agent.day07_server:app --port 7862
# 浏览器打开 http://127.0.0.1:7862
```

输入任务点运行，你会看到步骤日志**一行一行地**冒出来：
`step 1 调用工具 list_subjects({})` → `step 1 工具返回 {...}` → …

### 5.2 练习（按难度递增）

1. 在页面加一个"停止"按钮：`POST /api/agent/run/{id}/stop` 置一个 flag，
   `Agent.run` 每步检查 flag 就抛 `InterruptedError`（参考实现留了接口位）。
2. 把 SSE 端点**并进 10_subjects_web.py**：同一个 FastAPI app 挂上
   `/api/agent/run`，前端把轮询改成 `EventSource`，让"选主体 → 存主体 → Agent 跑 02b→02c"在同一个页面完成。
3. 事件里带上时间戳，前端按时间线渲染；`tool_result` 太大时后端截断（已做）。

**验收标准**：关闭浏览器标签页后，服务端 SSE 生成器要优雅退出（`GeneratorExit`
分支），不泄漏线程。

### 5.3 参考资料

- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [MDN：Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [MDN：EventSource](https://developer.mozilla.org/en-US/docs/Web/API/EventSource)
- FastAPI 自定义响应（StreamingResponse）: <https://fastapi.tiangolo.com/advanced/custom-response/>

---

## 6. 后期：Qwen-VL 叙事生成 → compose_scene（FYP authoring 层）

Agent 已经会"执行"，下一步是让 Agent 会"创作"：

1. 加一个工具 `describe_painting(subject)`：调 Qwen-VL（百炼多模态模型，如
   `qwen-vl-plus` / `qwen-vl-max`）看图生成一段叙事文案（"画面中一位女士与一匹马…"）；
2. 加一个工具 `compose_scene(narrative, subject_names)`：LLM 把叙事 + 各主体点云
   组装成一条场景合成指令（JSON），下游渲染器（3DGS/网格）按指令摆放主体、打光、出图；
3. agent 的 system prompt 变成"叙事导演"：先看画 → 决定主体 → 指挥重建 → 写叙事 → 出场景指令。

这就是 FYP 的 **authoring 层**：用户只上传一张画，得到"3D 场景 + 叙事"。

---

## 7. 网址汇总

### API / 模型
| 用途 | 网址 |
|------|------|
| DeepSeek 控制台（拿 API key） | <https://platform.deepseek.com/> |
| DeepSeek API 文档 | <https://api-docs.deepseek.com/> |
| DeepSeek Function Calling 示例 | <https://api-docs.deepseek.com/guides/function_calling> |
| 阿里云百炼控制台（拿 Qwen key） | <https://bailian.console.aliyun.com/> |
| 百炼（Qwen）文档 | <https://help.aliyun.com/zh/model-studio/> |
| 百炼 OpenAI 兼容模式 | <https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope> |
| OpenAI Function Calling 指南 | <https://platform.openai.com/docs/guides/function-calling> |
| openai-python SDK | <https://github.com/openai/openai-python> |
| ReAct 论文 | <https://arxiv.org/abs/2210.03629> |

### 网页 / 后端
| 用途 | 网址 |
|------|------|
| FastAPI 文档 | <https://fastapi.tiangolo.com/> |
| FastAPI 自定义响应（SSE 用） | <https://fastapi.tiangolo.com/advanced/custom-response/> |
| MDN Server-Sent Events | <https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events> |
| MDN EventSource | <https://developer.mozilla.org/en-US/docs/Web/API/EventSource> |

### 你本项目的入口
| 用途 | 地址 |
|------|------|
| Agent 工作台（Day 7，本机 mock） | <http://127.0.0.1:7862> |
| 主体选择页（07，服务器） | `ssh -N -L 7861:127.0.0.1:7861 autodl6` 后开 <http://127.0.0.1:7861> |
| Viser 点云查看（服务器） | `ssh -N -L 8080:127.0.0.1:8080 autodl6` 后开 <http://127.0.0.1:8080> |

---

## 8. 常见坑（先记住，省得踩）

| 坑 | 解决 |
|----|------|
| 报 `缺少环境变量 DEEPSEEK_API_KEY` | 先 `export`，或把 key 写进服务器 `~/.bashrc` |
| 模型一直不调工具、直接瞎答 | description 写清楚；system prompt 明说"必须调用工具" |
| 工具结果太大把上下文塞爆 | 工具只返回摘要；`tool_result` 截断（loop.py 已截 300 字符） |
| 改了 conf_percentile 重跑没变化 | 框架按产物缓存，必须 `force=True`（tools.py 已处理） |
| SSE 只收到几条就断 | 检查 Nginx 缓冲：`X-Accel-Buffering: no`（day07_server.py 已加） |
| 本地 mock 与服务器结果不一致 | mock 只是逻辑验证；真实效果以服务器 `AGENT_MOCK=0` 为准 |
| DeepSeek 用量/限流 | 便宜但非无限；loop.py 自带退避重试，别把 max_steps 调太大 |
