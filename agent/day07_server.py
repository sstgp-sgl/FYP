"""Day 7–10 · Agent 工作台：FastAPI + SSE 流式输出步骤日志。

跑法（本机，mock 模式也能用）:
    .venv/bin/python -m uvicorn agent.day07_server:app --port 7862
    浏览器打开 http://127.0.0.1:7862

API:
    POST /api/agent/run                {"task": "..."} → {"run_id": "..."}
    GET  /api/agent/run/{id}/events    SSE: 每条 data 是一步日志（JSON）
    POST /api/agent/run/{id}/stop      请求停止（示例保留位，见 docs）

设计要点（对比 scripts/10_subjects_web.py 现在的"轮询"）:
    - 07 里前端每 500ms 轮询 /api/workflow/status 拿整个 state；
    - 这里改成 SSE：服务器主动往一个事件流里推"第几步调了哪个工具、
      返回了什么"，前端 EventSource 一行行实时显示。
    下一步可以把这套端点并进 10_subjects_web.py 的 FastAPI app。
"""
from __future__ import annotations

import json
import queue
import threading
import uuid

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from agent.day05_qc import QC_IMPLS, QC_TOOL_DEFS
from agent.loop import Agent
from agent.tools import add_tools

# 默认把质检工具也注册进来（Day 5–6 的成果直接复用）
add_tools(QC_TOOL_DEFS, QC_IMPLS)

app = FastAPI(title="Agent Workbench")

# run_id -> {"queue": Queue, "status": running|done|error, "answer": str|None}
RUNS: dict[str, dict] = {}


@app.post("/api/agent/run")
def run_agent(body: dict):
    task = (body.get("task") or "").strip()
    if not task:
        return JSONResponse({"error": "task 不能为空"}, status_code=400)

    run_id = uuid.uuid4().hex[:8]
    q: queue.Queue = queue.Queue()
    RUNS[run_id] = {"queue": q, "status": "running", "answer": None}

    def worker():
        try:
            def on_event(step, kind, payload):
                q.put({"type": kind, "step": step, **payload})

            answer = Agent().run(task, on_event=on_event)
            RUNS[run_id]["status"] = "done"
            RUNS[run_id]["answer"] = answer
            q.put({"type": "done", "answer": answer})
        except Exception as e:  # noqa: BLE001
            RUNS[run_id]["status"] = "error"
            q.put({"type": "error", "error": str(e)})

    threading.Thread(target=worker, daemon=True).start()
    return {"run_id": run_id}


@app.get("/api/agent/run/{run_id}/events")
def events(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        return JSONResponse({"error": "no such run"}, status_code=404)
    q: queue.Queue = run["queue"]

    def gen():
        try:
            while True:
                try:
                    ev = q.get(timeout=30)
                except queue.Empty:
                    yield ": keepalive\n\n"      # 30s 无事件时保活
                    continue
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if ev["type"] in ("done", "error"):
                    break
        except GeneratorExit:
            pass

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/agent/run/{run_id}")
def run_status(run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        return JSONResponse({"error": "no such run"}, status_code=404)
    return {"run_id": run_id, "status": run["status"], "answer": run["answer"]}


HTML_PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Agent 工作台</title>
<style>
  body { font-family: ui-monospace, Menlo, monospace; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  textarea { width: 100%; height: 5rem; font: inherit; }
  button { font: inherit; padding: .5rem 1.2rem; cursor: pointer; }
  #log { background: #111; color: #ddd; padding: 1rem; border-radius: 8px;
         height: 55vh; overflow-y: auto; white-space: pre-wrap; margin-top: 1rem; }
  .tool { color: #7dd3fc; } .result { color: #86efac; } .answer { color: #fde68a; } .err { color: #fca5a5; }
</style>
</head>
<body>
<h2>Agent 工作台（SSE 实时步骤日志）</h2>
<p>任务示例：请查看有哪些主体，为每个主体生成视角并重建点云，然后质检每个主体的点云，
    如果点云过小就用 rerun_reconstruction 调低 conf_percentile 重跑一次，最后总结结果。</p>
<textarea id="task">请查看当前有哪些主体，为每个主体生成视角并重建点云，然后质检每个主体的点云，最后总结每个主体的点云路径和点数。</textarea>
<button onclick="start()">▶ 运行</button>
<button onclick="document.getElementById('log').textContent=''">清空</button>
<div id="log">等待运行…</div>
<script>
let es = null;
function append(cls, text) {
  const div = document.getElementById('log');
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = text;
  div.appendChild(line);
  div.scrollTop = div.scrollHeight;
}
async function start() {
  if (es) { es.close(); es = null; }
  const task = document.getElementById('task').value;
  const r = await fetch('/api/agent/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({task}),
  });
  const {run_id, error} = await r.json();
  if (error) { append('err', '[error] ' + error); return; }
  append('', `[run] ${run_id} 开始`);
  es = new EventSource(`/api/agent/run/${run_id}/events`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'tool_call')
      append('tool', `step ${ev.step} 调用工具 ${ev.name}(${JSON.stringify(ev.arguments)})`);
    else if (ev.type === 'tool_result')
      append('result', `step ${ev.step} 工具返回 ${JSON.stringify(ev.result).slice(0, 400)}`);
    else if (ev.type === 'answer')
      append('answer', `step ${ev.step} 最终回答：\\n${ev.content}`);
    else if (ev.type === 'done') { append('answer', '== 完成 =='); es.close(); es = null; }
    else if (ev.type === 'error') { append('err', '[error] ' + ev.error); es.close(); es = null; }
  };
  es.onerror = () => { append('err', '[sse] 连接断开'); if (es) es.close(); es = null; };
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE
