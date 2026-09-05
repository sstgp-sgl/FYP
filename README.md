# photo-to-mesh

从**单张照片**或**多视角照片**出发，重建物体/人像的 **3D 点云**，并可进一步做网格、高斯泼溅渲染。

- 单图流水线：`01_segment → 02_generate_views → 03_vggt → 04_gaussian → 05_mesh`
- 多主体流水线（FYP 主线）：`10_subjects_web → 11_generate_subject_views → 12_reconstruct_subjects`
- 通用点云重建框架：`scripts/20_reconstruct.py`（只到点云，不含渲染）
- Agent 学习路径（1–2 周，含可运行参考代码）：**`docs/AGENT_LEARNING.md`** + [`agent/`](agent/)

---

## 1. 环境（服务器）

| 项 | 值 |
|----|----|
| 服务器 | `connect.cqa1.seetacloud.com:20781`（AutoDL 别名 `autodl6`/`ulip`） |
| 用户 | `root` |
| GPU | NVIDIA RTX 4090 D 24GB |
| 项目目录 | `/root/autodl-tmp/photo-to-mesh` |
| VGGT 仓库 | `/root/autodl-tmp/vggt` |
| SAM2 仓库 | `/root/autodl-tmp/sam2` |
| Conda 环境 | `/root/miniconda3/envs/vggt`（Python 3.10，torch 2.3.1+cu121） |

进入服务器后，推荐固定一个变量方便下面所有命令：

```bash
cd /root/autodl-tmp/photo-to-mesh
export PY=/root/miniconda3/envs/vggt/bin/python
```

依赖检查（应全部 `yes`）：

```bash
$PY -c "import torch,viser,trimesh,open3d,gsplat,diffusers,transformers,numpy,scipy,PIL,cv2; print('deps ok')"
```

> 提示：HuggingFace 下载自动走镜像 `https://hf-mirror.com`（脚本里已 `setdefault`）。

---

## 2. 目录结构

```
photo-to-mesh/
├── configs/
│   ├── default.yaml        # 全流水线主配置（输入图、路径、vggt/segment 参数）
│   └── reconstruct.yaml    # 新重建框架配置（叠加在 default.yaml 之上）
├── scripts/
│   ├── 01_segment.py               # 单图：SAM2 分割主体
│   ├── 02_generate_views.py        # 单图：Zero123++ 生成 6 视角
│   ├── 03_vggt.py                  # 单图：VGGT 多视图重建（旧，深度分支）
│   ├── 04_gaussian.py              # 单图：COLMAP 导出 + 3DGS 训练
│   ├── 05_mesh.py                  # 单图：Poisson 网格提取
│   ├── 10_subjects_web.py          # 多主体：选择 + 工作流 Web
│   ├── 11_generate_subject_views.py# 多主体：逐主体 Zero123++ 视角
│   ├── 12_reconstruct_subjects.py  # 多主体：逐主体 VGGT 点云
│   ├── 20_reconstruct.py           # ★ 通用点云重建任务框架 CLI
│   ├── 21_view_ply.py              # Viser 点云查看器
│   └── 22_drop_white.py            # 去白点工具
├── agent/                   # Agent 学习参考实现（loop/tools/llm/...）
├── src/
│   ├── reconstruct/        # ★ 任务框架（task/pipeline/backends/tasks/viser）
│   ├── workflow.py         # 多主体管线共享逻辑（11/12 与 Web 共用）
│   ├── subjects.py         # Florence-2 + SAM2 多主体选择 + 导出
│   ├── vggt_reconstruct.py # VGGT 推理（pointmap/depth 分支）+ 导出
│   ├── pointcloud_filter.py# 点云过滤（置信度/离群点/背景色）
│   ├── pointcloud_io.py    # ASCII PLY 读写 + cameras.json
│   ├── segment.py          # SAM2 分割
│   ├── colmap_export.py    # COLMAP 导出
│   ├── gaussian_train.py   # 3DGS 训练
│   └── mesh_extract.py     # 网格提取
├── data/input/             # 输入照片
├── outputs/                # 各阶段产物
└── tests/                  # 单元测试
```

---

## 3. 完整流水线（单张照片 → 点云 → 网格）

### 阶段 1：分割主体（SAM2）

```bash
$PY scripts/01_segment.py
```

- 输入：`configs/default.yaml` 的 `paths.input_image`
- 输出：`outputs/foreground/foreground.png`（带 alpha 的主体）、`mask.png`、`normalized.png`

### 阶段 2：生成 6 视角（Zero123++）

```bash
$PY scripts/02_generate_views.py
```

- 输入：`outputs/foreground/foreground.png`
- 输出：`outputs/views/view_00.png` … `view_05.png`、`views.json`、`grid_raw.png`

### 阶段 3：VGGT 多视图重建（旧版，深度分支）

```bash
$PY scripts/03_vggt.py
```

- 输入：`outputs/views/`
- 输出：`outputs/vggt/pointcloud.ply`、`cameras.json`、`stats.json`、`predictions.npz`

> ⚠️ 旧版 `03_vggt.py` 用「深度反投影」方式，跨视角一致性差（人脸会杂乱）。
> **推荐用第 4 节的新框架（pointmap 分支）替代。**

### 阶段 4：3D 高斯泼溅（渲染，可选）

```bash
$PY scripts/04_gaussian.py
```

- 输入：`outputs/vggt/pointcloud.ply` + `cameras.json`
- 输出：`outputs/gaussian/gaussians.pt`、`orbit/` 新视角帧

### 阶段 5：表面网格（可选）

```bash
$PY scripts/05_mesh.py
```

- 输入：`outputs/vggt/pointcloud.ply`
- 输出：`outputs/mesh/mesh.ply`

---

## 4. 新点云重建任务框架（推荐，只到点云）

```bash
# 查看所有任务
$PY scripts/20_reconstruct.py list

# 完整重建（默认：6 张生成视图 + VGGT pointmap 分支）
$PY scripts/20_reconstruct.py run

# 交互查看点云
$PY scripts/20_reconstruct.py visualize --mode ply          # 轻量：直接显示已导出的 PLY
$PY scripts/20_reconstruct.py visualize --mode predictions  # 官方 UI：重新推理 + 置信度滑杆/逐帧/相机锥体
```

流水线：`collect → preprocess → reconstruct → filter → export`

- 输出目录：`outputs/reconstruct/`
  - `pointcloud.ply` — 最终点云
  - `cameras.json` — 每个视角的外参/内参
  - `stats.json` — 过滤统计
  - `.tasks/reconstruct/predictions.npz` — 未过滤的原始点/色/置信度（调参用，无需重跑推理）

### 常用参数（`configs/reconstruct.yaml`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `backend` | `vggt` | `vggt`（真实）/ `dummy`（无 GPU 冒烟测试） |
| `vggt.use_point_map` | `true` | **核心**：`true`=pointmap 分支（全局一致，效果好）；`false`=深度反投影（旧、差） |
| `vggt.conf_percentile` | `65` | 置信度过滤百分位；pointmap 下 <55 基本无效 |
| `vggt.outlier_std_ratio` | `2.0` | 统计离群点强度 |
| `visualize.mode` | `predictions` | `predictions` / `ply` |
| `input.mode` | `generated` | `generated` / `folder` / `files` |

> 改完参数重跑要加 `--force`（框架按输出文件是否存在做缓存）。

---

## 5. 换一张照片

### 5.1 单张照片 → 自动生成 6 视角

```bash
# 1) 把新照片放到 data/input/
# 2) 改 configs/default.yaml
paths:
  input_image: data/input/你的照片.png

# 3) 重新分割 + 生成视角（必须都跑）
$PY scripts/01_segment.py
$PY scripts/02_generate_views.py

# 4) 重建点云
$PY scripts/20_reconstruct.py run --force
```

### 5.2 已有多张同物体不同角度的照片

```bash
# 1) 照片放进一个目录，如 data/photos/
# 2) 改 configs/reconstruct.yaml
input:
  mode: folder
  folder: data/photos
  glob: "*"

# 3) 直接重建
$PY scripts/20_reconstruct.py run --force
```

---

## 6. 可视化（Viser）

框架内置两种：

```bash
# 轻量：显示 pointcloud.ply + 相机坐标系
$PY scripts/20_reconstruct.py visualize --mode ply --port 8080

# 官方 UI（run_viser_multi 同款）：逐帧点云选择 + 置信度滑杆 + 相机锥体跳转
$PY scripts/20_reconstruct.py visualize --mode predictions --port 8080
```

服务器上运行后，有两种方式在浏览器打开：

1. **本机 SSH 隧道**（端口转发）：
   ```bash
   ssh -N -L 8080:127.0.0.1:8080 autodl6
   # 浏览器打开 http://localhost:8080
   ```
2. **AutoDL 自定义服务**：控制台 → 自定义服务 → 内部端口 `8080` → 用生成的公网 URL 打开。

Viser 旧版官方脚本（直接跑 VGGT 仓库里的）：

```bash
cd /root/autodl-tmp/vggt
python run_viser_multi.py --image-folder <多图文件夹> --use-point-map --port 8080
```

---

## 7. 调参 / 调试

### 只调过滤、不重跑推理

直接读未过滤的 `predictions.npz`，扫不同 `conf_percentile`：

```bash
$PY - <<'PY'
import numpy as np, sys; sys.path.insert(0, '/root/autodl-tmp/photo-to-mesh')
from src.pointcloud_filter import filter_pointcloud
d = np.load('outputs/reconstruct/.tasks/reconstruct/predictions.npz')
for p in [55, 65, 70, 80]:
    cfg = {"vggt": {"conf_percentile": p, "outlier_nb_neighbors": 20,
                    "outlier_std_ratio": 2.0, "drop_fill_color": True,
                    "fill_color_slack": 8}, "segment": {"background_gray": 255}}
    _, _, st = filter_pointcloud(d['points'], d['colors'], d['conf'], cfg)
    print(p, st['num_points_filtered'])
PY
```

### 看置信度分布

```bash
$PY - <<'PY'
import numpy as np
c = np.load('outputs/reconstruct/.tasks/reconstruct/predictions.npz')['conf']
print('min/max:', c.min(), c.max(), ' frac==1.0:', float((c <= 1.0001).mean()))
for p in [50, 60, 70, 80, 90, 95, 99]:
    print(p, round(float(np.percentile(c, p)), 4))
PY
```

### 去白点（旧流水线产物）

```bash
$PY scripts/22_drop_white.py --slack 8
```

---

## 8. 交互式多主体选择（Florence-2 + SAM2，新）

在浏览器里**用文字检测物体、再用鼠标点选/框选微调**，导出每个主体的掩码和白底抠图：

```bash
$PY scripts/10_subjects_web.py --port 7861
# 浏览器打开 http://127.0.0.1:7861
```

- 上传油画 → 输入如 `a horse, a woman` → Florence-2 自动出框
- 画布上左键=正点、右键=负点、拖拽=框选，SAM2 实时出掩码（3 个候选可选）
- 每个主体可命名、增删；「保存全部」写出：
  - `outputs/subjects/<name>/mask.png`（掩码）
  - `outputs/subjects/<name>/subject.png`（白底抠图，可直接喂 Zero123++）
  - `outputs/subjects/subjects.json`（prompts + coverage 记录）

> 首次文字检测会下载 Florence-2（~0.7GB，走 hf-mirror）；服务器已装好 timm，
> 并用 flash-attn stub + eager attention 免去编译 flash-attn。

### 8.1 下游：每主体独立生成视角 + 点云重建（CLI 或网页一体化）

**方式一：网页一体化（推荐）** —— 网页第 4 步点「▶ 运行工作流」：

- 网页实时显示每个主体的进度：生成视角(11) → 点云重建(12) → 完成/失败
- 中间产物（7 张视角缩略图：6 生成 + 1 原图锚点）直接在网页展示
- 完成后点「🔍 查看点云」→ 网页内嵌 Three.js 查看器（拖拽旋转/滚轮缩放）

**方式二：CLI**

```bash
$PY scripts/11_generate_subject_views.py   # 每主体 7 视角（6 生成 + 1 原图）→ outputs/views/<subject>/
$PY scripts/12_reconstruct_subjects.py     # 每主体点云 → outputs/reconstruct/<subject>/
```

> 注意：逐主体流程**不用 `03_vggt.py`**（那是旧的整图深度分支管线）。
> 两套方式共享同一套代码（`src/workflow.py`），进度/结果一致。

后续每主体 3DGS：把该主体的 views + pointcloud 交给
`scripts/04_gaussian.py`（即 FYP 的「每主体独立 Gaussian 模型」）。

---

## 9. 测试

```bash
$PY -m pytest tests/test_reconstruct_framework.py tests/test_vggt_reconstruct.py -v
$PY -m pytest tests/test_subjects.py -v
```

- `test_reconstruct_framework.py`：任务框架（dummy 后端，无需 GPU/torch）
- `test_vggt_reconstruct.py`：VGGT 推理 mock（pointmap/depth 分支形状）
- `test_subjects.py`：多主体选择纯函数（命名/白底合成/掩码编码）

---

## 10. 常见问题

| 问题 | 解决 |
|------|------|
| 换照片后点云没变 | 没重跑 `02_generate_views.py`，或没加 `--force` |
| `list_view_images` 报缺文件 | `outputs/views/view_*.png` 不完整，重跑阶段 2 |
| 点云脸部杂乱 | 确认 `vggt.use_point_map: true`；调高 `conf_percentile`（65→70/80） |
| 有大片白板 | 白底被反投影，跑 `22_drop_white.py` 或确认 `drop_fill_color` 生效 |
| Viser 打不开 | 服务器上 `curl http://127.0.0.1:8080` 应返回 200；再检查隧道/自定义服务 |
| 显存不足 | 减少视图数，或换更小分辨率（`preprocess.size`） |
