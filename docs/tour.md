# Scene Tour（语言驱动的三维点云场景导览）

在 FYP 仓库中增量实现的 `/tour` 页面与 `/api/tour` 后端：导入点云 → 人工标注实体 →
人工记录镜头 → 手动或 LLM 生成分镜 → 校验 → 播放字幕/目标框/硬切镜头 → 保存并重新打开。

首个闭环：一个点云文件 + 人工标注的 3 个区域 + 4 个镜头 + 3 段约 30 秒的中文文字旁白。

## 1. 快速开始（CPU，不需要 GPU / torch / 模型）

环境：Python 3.11（Linux 与 Windows 命令等价）。不安装 torch、diffusers、transformers、
SAM2、VGGT、SHARP、Open3D；重建环境保持独立。

```bash
# Linux
python3.11 -m venv .venv-tour
.venv-tour/bin/python -m pip install -r requirements-tour-dev.txt
.venv-tour/bin/python scripts/40_tour.py seed-demo          # 合成庭院演示 + 手工分镜
.venv-tour/bin/python app.py --tour-only --host 127.0.0.1 --port 7861
# 浏览器打开 http://127.0.0.1:7861/tour
```

```powershell
# Windows PowerShell（无需激活环境）
py -3.11 -m venv .venv-tour
.\.venv-tour\Scripts\python.exe -m pip install -r requirements-tour-dev.txt
.\.venv-tour\Scripts\python.exe scripts/40_tour.py seed-demo
.\.venv-tour\Scripts\python.exe app.py --tour-only --host 127.0.0.1 --port 7861
# 浏览器打开 http://127.0.0.1:7861/tour
```

`app.py` 默认入口仍进入原重建工作流；`--tour-only` 分支在导入旧 `src.server` 之前分流，
导览独立模式不导入 torch/fcntl/sharp_cli，不创建/读取模型路径。原主页保留，另加
“场景导览”入口；重建成功的行有“用于导览”按钮（后端按 subject 名复制 pointcloud.ply
快照，浏览器不能传任意服务器路径）。

启动参数：`--tour-config <yaml>`（默认 `configs/tour.example.yaml`）、
`--reconstruction-root <dir>`（默认读 `configs/default.yaml` 的 `output_dir`）。
`output_root` 相对项目根解析，不随进程 cwd 漂移；可用环境变量 `FYP_TOUR_OUTPUT_ROOT` 覆盖。

## 2. 依赖与已验证版本（2026-11-08，Linux x86_64 实测）

requirements-tour.txt 是兼容范围，不是锁文件；以下为实际解析结果（写入
`docs/tour.md` 而非通用锁文件，操作系统专属依赖不锁入仓库）：

| 包 | 实测版本 |
|---|---|
| Python | 3.11.18（conda `py311` 环境创建的 venv） |
| fastapi | 0.162.2 |
| uvicorn | 0.54.0 |
| pydantic | 2.14.3 |
| numpy | 2.4.6 |
| plyfile | 1.1.5 |
| Pillow | 12.3.0 |
| httpx | 0.28.1 |
| python-multipart | 0.0.33 |
| PyYAML | 6.0.3 |
| pytest | 8.4.2 |
| @playwright/test | 1.54.1（仅测试用；网页运行不需要 Node） |
| three.js | 0.128.0（vendored，见 `web/vendor/three-r128/VENDOR.md`） |

前端不依赖 CDN：three r128 与 OrbitControls 位于 `web/vendor/three-r128/`，由
`/vendor/three-r128/` 静态路径提供（来源与 SHA256 记录在 VENDOR.md）。

## 3. 架构与文件

依赖方向：页面 → HTTP API（`src/tour/routes.py`）→ 服务（`src/tour/service.py`）→
存储（`src/tour/repository.py`）/ LLM（`src/tour/providers.py` +
`src/tour/planner.py`）/ 纯校验（`src/tour/validation.py`）。播放器只用已取回的
scene/storyboard，播放期间不调用 LLM。

```text
src/tour/
  __init__.py     # 导入时不产生 I/O、不加载模型
  app.py          # create_tour_app / mount_tour / tour_main（--tour-only）
  config.py       # configs/tour.example.yaml + 环境变量覆盖（不泄露密钥）
  models.py       # TourScene/Entity/ViewPreset/Storyboard/Shot/GenerationRequest/TourRun…
  repository.py   # 版本化快照、索引、原子写、expected_revision、预览二进制缓存
  importers.py    # 上传 PLY / 重建结果复制 / Phase 1 combined.ply / seed-demo
  geometry.py     # 固定显示变换（p_display = s·(p_asset − c)）、稳定采样、bbox
  service.py      # 场景编辑、确认失效、分镜保存、后台生成任务
  routes.py       # /api/tour/* 薄路由与错误映射
  planner.py      # 生成→校验→最多修正一次，独立于 HTTP
  providers.py    # Chat-Completions 兼容 HTTP adapter + 测试 fixture
  validation.py   # ID/镜头覆盖/事实归属/时长/语言/必需实体检查
  evaluation.py   # 离线固定任务集评测（不调用 LLM）
  prompts/        # tour_system.txt（grounded 叙事）、tour_repair.txt
scripts/40_tour.py        # seed-demo / import-ply / validate / evaluate / list
web/tour/*                # index.html + ES modules（state/api/viewer/annotations/storyboard/player/main）
web/vendor/three-r128/    # 固定 three.js r128 + LICENSE + VENDOR.md
tests/tour/*              # 后端测试（8 个模块）
tests/e2e/tour.spec.js    # Playwright 浏览器验收（8 个场景）
evaluation/tasks-v1.json  # 固定任务集 + 4 个候选分镜
docs/tour.md              # 本文档
```

对旧代码的复用：`src/scene/storage.py` 的 `AssetRepository`（资产复制、哈希校验、
受控路径、原子 JSON）直接作为导览资产层；`src/scene/models.py` 的 identifier/number/
digest 校验风格由 Pydantic v2 模型重新实现；`src/pointcloud_io.py` 仅被 seed-demo 用于
写 fixture。旧的 `SceneDocument`（Y-up、地面 y=0 布局模型）不被导览复用——`TourScene`
使用源资产坐标（`coordinate_frame="asset"`），不假设地面。

## 4. 坐标与数据契约

- 实体 AABB、相机 position/target/up 一律存源资产坐标（asset 系）。
- 预览使用**只计算一次**并写入 `TourScene.display_transform` 的固定变换：
  `p_display = s * (p_asset - c)`，c = 全量有限点 AABB 中心，s = 1/r（r = 到 c 的最大距离）。
  退化点云（零半径）拒绝导入。相机 near/far 以 s 同步缩放；up 不缩放（方向向量）。
- 浏览器保存视角时按该变换逆变换回 asset 坐标；重新加载不重新抽样推算中心。
- 相机校验：position≠target、up 非零、up 不与视线共线、1≤fov<180、0<near<far。
  首版无额外 roll；全部镜头 aspect_ratio 固定 16:9（窗口窄时留边，不改变构图假设）。
- 实体框与场景点云共享同一变换；每个实体不单独归一化。
- 人工确认语义：`approved_entity_ids` 表示“人工核对过该镜头看得清”，不是遮挡算法结果。
  用户改镜头 camera 后清空该镜头确认；实体 bounds 改变使所有涉及该实体的确认失效
  （后端强制）。只有 confirmed 实体及其事实提供给 LLM。
- 版本：场景/分镜每次保存 revision+1；分镜绑定冻结的 scene_revision + asset_sha256，
  不匹配时拒绝播放（旧导览需在新版本上重新校验保存后才可播放）。
  保存用 `expected_revision` 乐观并发，过期写入返回 409，不静默覆盖其他标签页。

## 5. 存储布局（outputs/tours）

```text
outputs/tours/
  assets/<asset_id>/data.ply, asset.json          # 复用 AssetRepository 布局
  scenes/<scene_id>/index.json, revisions/000001/scene.json
  previews/<asset_hash>/metadata.json, points.bin, colors.bin
      # points.bin = float32 小端 Nx3（display 坐标）；colors.bin = uint8 Nx3
      # 抽样固定 seed（由 asset hash 派生），刷新不重抽
  storyboards/<storyboard_id>/index.json, revisions/000001/storyboard.json
  runs/<run_id>/run.json, request.json, attempts/NN/{response.json,validation.json}
  evaluation/<evaluation_id>/report.json
```

所有写操作：同目录临时文件写完 `os.replace`，先落 revision 内容再更新 index。
单进程、按 key 加锁；不做多进程/多机事务承诺。运行中留下的 pending/running 记录在
下次启动时标记为 interrupted，不自动重试计费调用。

## 6. HTTP API（前缀 /api/tour，Pydantic v2，禁止额外字段）

| 方法与路径 | 用途 |
|---|---|
| GET /capabilities | 能力与 LLM 就绪摘要（无密钥） |
| GET /scenes | 场景列表（≤100/页） |
| POST /scenes/import | multipart PLY + title（流式限量，超出 413） |
| POST /scenes/from-reconstruction | 已验证 subject 名 → 复制旧重建结果（不接受浏览器路径） |
| POST /scenes/from-phase1 | 导入 Phase 1 输出的 combined.ply 快照 |
| GET /scenes/{id}?revision=n | 指定/最新快照 |
| PUT /scenes/{id}/annotations | entities/views + expected_revision（含确认失效规则） |
| GET /scenes/{id}/preview · /preview/points · /preview/colors | 预览元数据与二进制 |
| POST /storyboards | 新建手工分镜（校验通过才落盘） |
| GET /storyboards · GET /storyboards/{id}?revision=n | 列表/读取版本 |
| PUT /storyboards/{id} | 修改 + expected_revision（过期 409） |
| POST /storyboards/validate | 候选分镜校验（不落盘）；注意先于动态路由注册 |
| POST /generations | GenerationRequest + request_id → 202 run_id |
| GET /runs/{id} | 轮询 pending/running/succeeded/failed/interrupted/needs_clarification |
| GET /storyboards/{id}/download | JSON 下载（含资产引用，不含密钥/资产本身） |

统一错误：`{"error":{"code","message","details"},"request_id"}`；400/404/409/413/422，
供应商错误映射为 502/504 语义的错误码（http_401/http_429/http_5xx/timeout/parse）。
生成用单个后台任务：同场景最多一个活跃 run，同 request_id 幂等（重复返回 409 并给出
已有 run），应用关闭时取消任务并关闭 HTTP client；页面每 1 秒轮询（非 WebSocket），
页面关闭不取消服务端任务。

## 7. LLM 接入

- Provider：`src/tour/providers.py` 的 Chat-Completions 兼容 HTTP adapter（httpx 共享超时
  配置）。`response_mode=text_json`（纯文本+严格 JSON 解析）或 `json_object`
  （显式发送 response_format，仅当服务商支持时）。其他协议另写 provider，不混字段。
- 只发送：用户要求、scene id/revision、已确认实体与事实、人工确认镜头及其可见实体、
  语言/时长/段数、输出 schema、规则。不发送点云数据、图片、服务器路径、环境变量、
  重建日志。labels/descriptions/facts 是数据，不作为系统指令。
- 输出为判别联合：`{status:"ok",title,shots:[...]}` 或
  `{status:"needs_clarification",question}`。shot_id 由服务端归一化；storyboard id/
  revision/来源/时间全部由服务端补全。模型不产生相机坐标、代码、URL、路径。
- 流程：预检 → 记录 run(pending→running) → 调用 → 严格解析（至多剥一层完整
  markdown fence，绝不 eval/启发式拼接）→ 校验（引用、人工可见性、事实 scope、
  段数、时长±1s、必需实体、开场镜头、语言）→ 可修复则携带具体错误重试**一次**
  （总调用 ≤ `llm.max_calls_per_run`，无隐藏网络重试）→ 成功保存 validated 分镜
  （不自动播放）；仍失败则 failed 保留错误详情；needs_clarification 是独立终态。
- 配置（`configs/tour.example.yaml` 的 `llm` 段 + 环境变量）：
  `FYP_LLM_API_KEY`（必填）、`FYP_LLM_BASE_URL`、`FYP_LLM_MODEL`、`FYP_LLM_ENABLED`。
  密钥只在后端环境变量读取，不返回网页、不记录 Authorization、不写 .env。
  base_url 由本机配置控制，浏览器不能指定。配置未就绪时生成返回
  `llm_not_configured`，手动分镜始终可用，不做模板静默冒充 LLM。

## 8. 播放状态机

idle/playing/paused/finished。硬切（transition 一律 cut）；时长即段的完整显示时间。
elapsed 用单调时钟（performance.now）+ RAF 计算，不用定时器计数；页面隐藏自动暂停，
恢复后由用户继续。暂停保持当前镜头且 controls 禁用（避免“继续”时跳镜头）；finished
保留末段画面并恢复 controls；重播从零开始。播放时不做窗口自适应重新取景（16:9 留边）。
切场景/载入其他分镜前停止播放器，保证只有一个 RAF 控制相机。字幕与模型输出使用
textContent；实体高亮只是 AABB 框+标签，不冒充点级分割或遮挡后的真实可见区域。

## 9. 测试与验证记录

### 后端（84 项，远程 Linux Python 3.11 venv 实测通过）

```bash
.venv-tour/bin/python -m pytest tests/tour/ -q        # 84 passed
.venv-tour/bin/python -m pytest tests/test_scene_phase1.py -q   # 28 passed（回归）
```

覆盖：schema（NaN/Inf/bool 冒充数值/倒置 bbox/零相机方向/错误版本/额外字段）、
导入（ASCII/二进制 XYZRGB、缺 RGB 默认灰、空点云、Gaussian 字段拒绝、超点/超字节）、
坐标（非原点云正逆变换一致、抽样不改变持久相机）、预览（float32 小端/长度/稳定抽样）、
路径（traversal、未知 ID、浏览器任意路径拒绝）、repository（原子写失败保留旧版本、
expected_revision 冲突、hash 变更检测、run 中断恢复）、validation（未知引用、重复 shot、
不可见焦点、scope 错误、时长/段数/语言/必需实体/开场镜头）、planner（一次成功、一次
修正、两次失败、clarification、非 JSON、401/timeout）、provider、routes（含假 provider
全流程生成与轮询、重复 request_id 409、错误 shape 带 request_id）、CPU 隔离（subprocess
启动 tour-only、禁用 GPU 包导入、seed/read/play）、并发（双击不重复、快照不受编辑影响）。

### 旧环境回归（vggt conda env，FastAPI 0.141.1 / pydantic 2.x / numpy 2.2.6）

```bash
~/miniconda3/envs/vggt/bin/python -m pytest tests/test_web_pipeline.py tests/tour/ -q  # 90 passed
```

（vggt 环境补装了两个纯 CPU 包：`httpx`、`plyfile`。）完整模式实测：
`python app.py`（旧入口）下 `/`、`/tour`、`/api/tour/capabilities`、`/api/config` 均 200。

### 浏览器（Playwright 1.54.1 / Chromium，8/8 通过）

```bash
npm install && npx playwright install chromium
# 先起服务并 seed（见 §1），或针对 e2e 专用根：
rm -rf outputs/tours-e2e && FYP_TOUR_OUTPUT_ROOT=outputs/tours-e2e \
  .venv-tour/bin/python scripts/40_tour.py seed-demo
FYP_TOUR_OUTPUT_ROOT=outputs/tours-e2e .venv-tour/bin/python app.py \
  --tour-only --host 127.0.0.1 --port 7861
npx playwright test   # tests/e2e/tour.spec.js，8 个场景
```

场景：页面加载无 console error/WebGL 可用/几何可见且零外部网络请求（截图
`tests/e2e/artifacts/tour-demo-loaded.png`）；上传夹具→标注→记镜头→确认→保存→刷新恢复；
三段播放（字幕/镜头序号/时间在正确时刻切换）；暂停冻结/恢复不重启/上下段/结束/重播；
改实体 bounds 使确认失效、view-scoped 事实违规的分镜校验失败且不可播放、旧版本分镜
重新校验保存后重绑播放；无 LLM 配置时生成干净失败且场景版本不变；多次切场景单一渲染
循环且旧对象 dispose；16:9 信箱在多种窗口尺寸下保持。

测试假定“刚 seed 的干净状态”（会修改场景/分镜 revision）。`E2E_DEBUG_REQUESTS=1` 可
打印非 GET 请求体用于调试。

### 评测（M7，离线，无 LLM 调用）

```bash
.venv-tour/bin/python scripts/40_tour.py evaluate --tasks evaluation/tasks-v1.json
# 输出 outputs/tours/evaluation/<id>/report.json；固定任务集 4/4 通过
```

指标：结构有效、引用正确、人工确认可见性覆盖、事实 scope 正确、时长/段数/语言符合、
必需实体覆盖、与期望一致性；`human_check.fact_consistency` 留给人审阅（旁白语义忠实度
自动校验不作 100% 承诺）。真实模型批量评测按预算执行，未获预算时只做本离线集。

## 10. 功能边界（如实声明）

- 旁白是**文字字幕**，不是语音；TTS 未实现。
- 实体是**人工标注的空间区域**（AABB + 标签），不是自动分割；高亮不改变源点颜色。
- 镜头来自**人工记录**的候选；LLM 只做叙事组织与候选镜头选择，不预测相机坐标、不生成代码。
- 投影视锥检查只是提示；不解决遮挡。approved 只表示人工核对过。
- 不实现：自动实体识别、VLM、自动镜头生成、遮挡求解、点云分割、主体插入、真实地面
  约束、动态动作、高斯联合渲染、TTS、视频导出、多人协作、登录、云部署。
- JSON 下载含元数据与资产引用，**不含资产本身**；缺资产时给出缺失提示。
- Windows 实测未做（本次在 Linux 验证；命令按计划给出，Windows 标为未测）。
- 真实 LLM 请求未做（本机未配置 key）；L3 由假 provider 契约测试覆盖，
  真实调用需配置 §7 环境变量后在同流程生成。

## 11. 演示数据

`seed-demo` 生成明确标记 `synthetic` 的“合成庭院演示”场景（地面 + 红/绿/蓝三个区域、
3 个 confirmed 实体 + 4 个镜头 + 3 段 30 秒中文分镜），用于离线闭环演示，不冒充重建成果。
真实资产走 `/tour` 上传或旧重建结果“用于导览”按钮。
