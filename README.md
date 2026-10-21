# photo-to-mesh

通过一个 Web 页面完成：上传图片 → SAM2 分割主体 → Zero123++ 生成视角 → VGGT 重建点云。
支持多个主体逐个重建，以及 **6 视图 / 7 视图** 两种 VGGT 输入模式。保留网页点云预览、PLY 下载和 Viser 查看。

## Phase 1：资产登记、CPU 场景布局与 SHARP CLI

Phase 1 为后续“场景 + 主体 → 空间约束 → 场景布局”提供了可测试的数据契约和最小 CPU 演示。它与现有 Zero123++ 生成视图和 VGGT 点云流程并行；目前不提供自然语言解析、自动分割、高斯场景合成或新查看器。

### 快速运行：无需 GPU 或模型

在服务器项目目录 `/root/autodl-tmp/FYP` 运行：

```bash
export PY=/root/miniconda3/envs/vggt/bin/python
$PY scripts/30_scene.py demo-compose --root outputs/phase1/demo-left
$PY scripts/30_scene.py validate-scene --root outputs/phase1/demo-left
$PY -m pytest -q tests/test_scene_phase1.py tests/test_sharp_cli.py
```

演示使用固定的合成立方体和地面点云，并将主体放到锚点左侧。每次选一个新的 `--root`，避免覆盖已有演示产物。输出包括：

```text
outputs/phase1/demo-left/
├── assets/<asset-id>/asset.json     # 资产元数据与 SHA-256
├── assets/<asset-id>/data.ply       # 不可变源点云
├── scene.json                       # 实例变换与空间约束
├── combined.ply                     # 合并后的 XYZ/RGB 点云
├── combined.ranges.json             # 导出中每个实例的点范围
├── validation.json                  # 从保存产物独立复算的验收结果
└── run.json                          # synthetic 运行记录
```

坐标约定为右手系、Y 轴向上、地面 `y=0`。布局仅接受已确认 Y-up 的 XYZ/RGB 点云；从原始资产计算统一缩放和底部中心位置，不累乘实例上一次的变换。`left_of` / `right_of` 的 gap 是沿场景固定水平参考方向的边界投影距离。场景单位默认为 `arbitrary`，演示结果不能解读为现实米制或视觉质量证明。

持久化对象使用 `schema_version: 1`。场景保存会校验资产存在且哈希一致；外部资产导入时复制到资产目录。PLY 类型通过顶点字段校验。Gaussian PLY 会保留为原始高斯资产，不计算表面 bounds，也不会被伪装成表面点云参与布局或合并导出。

### SHARP（可选，真实模型尚未在本环境运行）

诊断不会运行推理或下载模型：

```bash
$PY scripts/30_scene.py diagnose-sharp --config configs/sharp.example.json
```

当前配置默认关闭自动权重下载。`sharp.example.json` 可配置 SHARP 可执行文件、权重、设备、超时和输出路径。真实单图推理命令如下；图片须是单张不透明 RGB PNG/JPEG，前景需要先由使用者准备：

```bash
$PY scripts/30_scene.py reconstruct-sharp \
  --config configs/sharp.example.json \
  --image /path/to/rgb-photo.png \
  --role subject
```

适配器通过安装环境中的 `sharp --help` 和 `sharp predict --help` 核实参数，再以独立输入/输出目录运行官方 CLI。它要求显式本地 checkpoint（除非配置明确开启官方下载），记录图片与权重哈希，并校验本次运行生成的 Gaussian PLY。官方 SHARP 的输出使用 OpenCV 相机轴约定（x 右、y 下、z 前）；它不会自动对齐到场景地面。官方 CLI 若不存在，先按 [Apple ML-SHARP 官方安装说明](https://github.com/apple-aiml-research/ml-sharp#using-the-cli) 建立独立 Python 环境并安装，避免改动现有 VGGT 环境；适配器本身不安装依赖或下载权重。运行模型前分别阅读官方[代码许可](https://github.com/apple-aiml-research/ml-sharp/blob/main/LICENSE)和[模型许可](https://github.com/apple-aiml-research/ml-sharp/blob/main/LICENSE_MODEL)。

SHARP 的 Gaussian 适合附近视角渲染，不代表恢复了完整表面或物体背面。本轮没有实现 Gaussian 查看器、Gaussian 之间的联合渲染或高斯到准确表面点云的转换。适配器用临时假 CLI 做了子进程契约测试；在未安装官方 CLI、checkpoint 且未实际推理之前，不应称作 SHARP 端到端验证通过。

### 本轮边界与状态

- 已实现：版本化资产/场景/变换/约束/运行记录；受控路径和哈希检查；原子 JSON 保存；CPU 点云左右布局、落地、按目标高度缩放、PLY 合并导出和独立数值验收。
- 已实现：隔离的官方 SHARP CLI 适配器、环境诊断、超时/子进程组清理、运行日志、Gaussian PLY 字段检查、孤儿运行恢复。
- 仅定义后续方向：`ConstraintParser`、`SubjectExtractor`、`StoryPlanner`、`ViewerAdapter`。当前没有注册可用的空实现。
- 本仓库已有视图配置包含 6 张 Zero123+ 生成图，并默认 `include_original: true` 将原图也交给 VGGT；这次没有改变该配置。
- `requirements-phase1.txt` 只列出 Phase 1 演示/契约测试依赖，未自动安装。SHARP 依赖与模型许可按官方仓库分别确认。


## 启动

当前项目在 `ulip:/root/autodl-tmp/photo-to-mesh`，使用现有 Conda 环境，无需先重新安装依赖。
在服务器执行：

```bash
cd /root/autodl-tmp/photo-to-mesh
/root/miniconda3/envs/vggt/bin/python app.py --host 127.0.0.1 --port 7861
```

本机另开终端建立转发：

```bash
ssh -N -L 7861:127.0.0.1:7861 -L 8080:127.0.0.1:8080 ulip
```

浏览器打开 `http://localhost:7861`。网页中的 Viser 按钮启动独立点云查看器，访问 `http://localhost:8080`。
如果通过 AutoDL 自定义服务访问，可使用 `--host 0.0.0.0`，并在控制台映射对应端口。

可覆盖配置文件或监听地址：

```bash
/root/miniconda3/envs/vggt/bin/python app.py --config configs/default.yaml --port 7861
```

该页面按单人操作、一次一个工作流设计。前台运行时按 Ctrl+C 停止服务。

## 网页操作

1. 上传图片，点击「添加主体」。
2. 在画布上用左键标记前景、右键标记背景，或切换为框选；SAM2 返回候选掩码。可添加多个主体并分别命名。
3. 选择合适的掩码候选，点击「保存全部」。修改主体或提示点后需重新保存。
4. 选择 6 或 7 视图，点击「生成视角并重建」。页面显示每个主体的生成和重建进度。
5. 完成后可在网页查看点云、下载 PLY，或打开 Viser 查看点云和相机坐标系。

每次运行都会重新生成视角并执行 VGGT。同名主体的视角和重建结果会覆盖，比较两种模式时请先保存需要保留的结果。
切换视图数量后直接重新运行即可；重新上传图片后必须重新选择并保存主体。

## 6 视图与 7 视图

| 模式 | VGGT 输入 | 配置默认值 |
| --- | --- | --- |
| 6 视图 | Zero123++ 生成的 `view_00.png` 到 `view_05.png` | `views.include_original: false` |
| 7 视图 | 上述 6 张 + `view_06.png` 原始主体图 | `views.include_original: true` |

Zero123++ 始终生成 6 张图。第 7 张来自 SAM2 白底主体图，经过主体包围盒裁剪和等比例补边，与 Zero123++ 的条件图保持相同构图；它不是额外生成的视角。

网页选择覆盖本次运行的配置，不修改 YAML 文件。VGGT 严格按 `views.json` 读取 6 或 7 张图；切回 6 视图会移除当前主体目录里旧的 `view_06.png`，其他残留图片不会加入推理。

## 项目结构

```text
photo-to-mesh/
├── app.py                     # 唯一 Web 启动入口
├── configs/default.yaml       # 唯一运行配置
├── requirements.txt           # 当前服务器的运行依赖版本
├── requirements-dev.txt       # 运行依赖 + 测试依赖
├── web/index.html             # SAM 交互、模式选择、进度和点云预览
├── src/
│   ├── config.py              # 配置加载；相对路径以项目根目录为基准
│   ├── server.py              # Web API、后台线程和 Viser 进程管理
│   ├── segment.py             # SAM2 加载、提示分割和掩码处理
│   ├── subjects.py            # 主体命名、掩码与白底图导出
│   ├── zero123.py             # Zero123++、6/7 视图与视图清单
│   ├── reconstruction.py      # 逐主体调用 VGGT、过滤和导出
│   ├── vggt_reconstruct.py     # VGGT 模型加载与推理
│   ├── pointcloud_filter.py   # 置信度、白色背景和离群点过滤
│   ├── pointcloud_io.py       # PLY 与相机参数读写
│   ├── viewer.py              # 网页按需启动的 Viser 模块
│   └── io_utils.py            # JSON、视图清单和图像缩放
├── tests/                     # SAM、视图切换、VGGT 与 Web 回归测试
├── data/                      # 已有输入资料（不纳入 Git）
└── outputs/                   # 生成结果（不纳入 Git）
```

调用关系：`app.py → server.py → subjects/segment → zero123 → reconstruction → vggt_reconstruct`。
点云过滤和导出由普通函数完成，不再使用任务注册表、可替换后端、DAG 或分阶段缓存。

## 配置与模型

所有配置集中在 `configs/default.yaml`，保留原 Web 的有效 VGGT 设置：pointmap 分支，置信度过滤百分位为 65。

| 配置 | 用途 |
| --- | --- |
| `device` | 默认 `cuda` |
| `image_size` | 主体和生成视角尺寸，默认 512 |
| `seed` | Zero123++ 随机种子；多主体依次递增 |
| `segment.sam2_checkpoint` | 本地 SAM2 权重 |
| `segment.sam2_config` | SAM2 包内 Hydra 配置名，不是文件路径 |
| `views.model` | `sudo-ai/zero123plus-v1.2` |
| `views.num_inference_steps` | 默认 75 |
| `views.include_original` | 网页初始模式，默认 7 视图 |
| `vggt.model` | `facebook/VGGT-1B` |
| `vggt.use_point_map` | 默认 `true`；仍可设为 `false` 使用深度分支 |
| `vggt.conf_percentile` | 置信度过滤百分位，默认 65 |
| `vggt.outlier_nb_neighbors` / `outlier_std_ratio` | 离群点过滤参数 |
| `paths.subjects_dir` / `paths.views_dir` / `output_dir` | 输出位置 |
| `server.host` / `server.port` | Web 默认监听地址和端口 |

当前服务器使用：

- Python：`/root/miniconda3/envs/vggt/bin/python`
- SAM2 仓库：`/root/autodl-tmp/sam2`，已安装到现有环境。
- SAM2 权重：`/root/autodl-tmp/sam2/checkpoints/sam2.1_hiera_large.pt`
- VGGT 仓库：`/root/autodl-tmp/vggt`，由 `paths.vggt_repo` 指定。

`requirements.txt` 记录现有运行环境的直接依赖版本；它不包含模型权重和本地模型仓库。
迁移机器时需先准备适配 CUDA 的 PyTorch、SAM2 安装与权重、VGGT 仓库，再安装其余依赖并更新 YAML 中的绝对路径。
Zero123++ 与 VGGT 首次使用时可能下载模型；默认设置 `HF_ENDPOINT=https://hf-mirror.com`，可通过环境变量覆盖。
网页内 Three.js 点云预览使用 jsDelivr CDN；无法访问 CDN 时可下载 PLY 或使用 Viser。

## 输出

```text
outputs/
├── subjects/
│   ├── subjects.json
│   └── <主体名>/mask.png, subject.png
├── views/
│   ├── subjects_index.json
│   └── <主体名>/views.json, grid_raw.png, view_00.png …
└── reconstruct/
    ├── <主体名>/pointcloud.ply, cameras.json, stats.json, predictions.npz
    └── viser_<主体名>.log
```

`stats.json` 记录输入视图数量、是否包含原图及点云过滤统计。`predictions.npz` 保存原始推理输出。
旧 `outputs/` 中的结果与 `.tasks` 缓存保留在磁盘上；新程序不再读取旧任务缓存。

## 验证

```bash
cd /root/autodl-tmp/photo-to-mesh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /root/miniconda3/envs/vggt/bin/python -m pytest -q
```

测试包括 SAM 掩码处理、生成视图构图、7→6 视图切换、视图清单校验、VGGT 输出与过滤，以及 Web 上传/分割/保存/重建/下载接口。
测试使用模型替身，不下载权重、不执行真实模型推理；不能替代真实图片的 GPU 重建质量检查。

本次整理验证了 30 项测试、前端 JavaScript 语法，以及真实 Web 进程的首页和配置接口（HTTP 200）。检查时 `torch.cuda.is_available()` 返回 `False`；运行默认 CUDA 模式的模型推理前，需要先恢复该实例的 GPU 可用性。

## 本次精简

已移除旧编号脚本、通用任务框架、Agent 学习示例、Florence 文字检测、COLMAP、Gaussian 和网格生成代码。
旧启动命令改为 `app.py`；相关配置和文档统一到本文件。原有输入图片、模型仓库、权重和历史输出保留。

改动前源码备份：
`/root/autodl-tmp/project-backups/photo-to-mesh-before-web-cleanup-20261010-153030.tar.gz`。
Git 历史也保留了旧实现。
