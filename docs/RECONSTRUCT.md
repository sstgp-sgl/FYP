# 多视图点云重建任务框架 (Reconstruction Task Framework)

从多张照片 / 多视角生成图出发，重建一个**彩色 3D 点云**并可在 Viser 中交互查看。
框架**只做到点云为止**——不包含高斯泼溅渲染 (stage 4) 或网格提取 (stage 5)。

> 参考了 VGGT 官方的 `run_viser_multi.py`（本项目里对应 `scripts/21_view_ply.py（原 03b 诊断脚本已合并为 Viser 查看器）`
> 以及 `demo_viser.viser_wrapper`），把「推理 → 可视化」做成可复用的任务框架。

## 流水线

```
collect ──► preprocess ──► reconstruct ──► filter ──► export
  (取图)      (标准化)       (后端重建)      (过滤)      (PLY/相机/统计)
                                                    │
                                                    └──► visualize (可选, Viser)
```

| 任务 | 作用 | 输出 |
|------|------|------|
| `collect` | 按 `input.mode` 收集多视图图片（≥2 张） | `collect/manifest.json` |
| `preprocess` | 方形 resize+pad、可选白底化 | `preprocess/view_*.png` |
| `reconstruct` | 调用后端生成融合点云 + 相机位姿 | `reconstruct/predictions.npz` |
| `filter` | 置信度分位 + 统计离群 + 背景色过滤 | `filter/filtered.npz` |
| `export` | 写出最终点云与相机 | `pointcloud.ply`, `cameras.json`, `stats.json` |
| `visualize` | Viser 交互查看（见下） | — |

每个任务都有磁盘缓存：输出已存在时跳过，`--force` 强制重跑。

## 快速开始

```bash
cd photo-to-mesh

# 1) 查看可用任务
python scripts/20_reconstruct.py list

# 2) 完整重建（默认用 Zero123++ 生成视图 + VGGT 后端）
python scripts/20_reconstruct.py run

# 3) 交互查看点云
#    predictions: 重新跑 VGGT，打开官方 Viser（逐帧点云/相机锥体/置信度滑杆，run_viser_multi 风格）
#    ply:         直接显示已导出的 pointcloud.ply + 相机坐标系（不重新推理，快）
python scripts/20_reconstruct.py visualize --mode predictions
python scripts/20_reconstruct.py visualize --mode ply
```

## 输入：多摄像头照片 / 生成图

`configs/reconstruct.yaml` 的 `input.mode` 决定图片来源：

| mode | 说明 |
|------|------|
| `generated` (默认) | 读取 Zero123++ 生成视图：`outputs/views/views.json` 里列出的 `view_*.png` |
| `folder` | 从某个目录 glob 图片（多相机阵列 / 转台拍摄的同一物体多视角照片） |
| `files` | 显式列出图片路径 |

```yaml
input:
  mode: folder
  folder: data/photos          # 同一物体/场景的多视角照片
  glob: "*"
```

> `folder` / `files` 模式请使用**同一物体/场景**的多视角照片（不同机位拍摄同一目标），
> 否则 VGGT 会输出散乱、无意义的点云。

## 后端

`configs/reconstruct.yaml` 的 `backend`：

- `vggt` (默认)：`facebook/VGGT-1B`，多视图联合位姿+深度+点云，需要 GPU 与本地
  VGGT checkout（`paths.vggt_repo`，见 `configs/default.yaml`）。自动使用
  `HF_ENDPOINT=https://hf-mirror.com`。
- `dummy`：确定性合成球体，**不需要模型 / GPU / torch**，用于测试、排错、跑通流程。

## 配置

`configs/reconstruct.yaml` 叠加在 `configs/default.yaml` 之上（共享 `paths`、`vggt`、
`segment` 块）。关键项：

| 键 | 默认 | 说明 |
|----|------|------|
| `output_dir` | `outputs/reconstruct` | 最终产物目录 |
| `backend` | `vggt` | `vggt` 或 `dummy` |
| `input.mode` | `generated` | `generated` / `folder` / `files` |
| `preprocess.size` | `512` | 标准化后的方形边长 |
| `preprocess.whiten_background` | `false` | 是否白底化（仅生成视图有意义） |
| `visualize.mode` | `predictions` | `predictions` 或 `ply` |
| `visualize.port` | `8080` | Viser 端口 |

过滤参数沿用 `default.yaml` 的 `vggt.conf_percentile`（默认 40）、
`vggt.outlier_nb_neighbors`、`vggt.outlier_std_ratio`，以及
`segment.background_gray`（默认 255）。

## Python API

```python
from pathlib import Path
from src.reconstruct import Pipeline, TaskContext, load_config, get_task

cfg = load_config()                                   # 或 load_config("my.yaml")
ctx = TaskContext(cfg, workdir=Path(cfg["output_dir"]) / ".tasks")

pipeline = Pipeline([get_task(n) for n in
                     ("collect", "preprocess", "reconstruct", "filter", "export")])
summary = pipeline.run(ctx)                           # 或 run(ctx, only=["filter"])
```

自定义任务：继承 `src.reconstruct.task.Task`，用 `@register` 装饰并声明 `name`、
`deps`、`outputs()` 与 `run(ctx)`，然后导入该模块即可被 CLI/pipeline 识别。

## 说明 / 边界

- 本框架**不做渲染**：不训练高斯泼溅、不提取网格。需要渲染时再用既有的
  `scripts/04_gaussian.py` / `scripts/05_mesh.py`（它们消费 `outputs/vggt/` 的产物）。
- `visualize --mode predictions` 会**重新运行一次 VGGT 推理**（与 `run_viser_multi.py`
  一致），因为官方 Viser 需要逐帧的原始深度/置信度/图像；只想快速看点云就用
  `--mode ply`。
- 框架核心不依赖 torch/trimesh/viser：`dummy` 后端 + `export` 在纯 numpy 环境即可跑；
  VGGT 后端、`predictions` 可视化和 `ply` 可视化在用到时才惰性 import。
