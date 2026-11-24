# FYP 代码执行说明书

更新日期：2026 年 11 月 23 日。

本说明书按“先看效果，再执行完整流程”的顺序，介绍项目在哪里、命令在哪台机器执行、如何启动网页、选择主体和重建、生成与修改故事，以及下载和检查结果。当前主要功能为 P0–P7，P8 已完成本轮探索性比较；详细范围与结果见 [P8 报告](progress/P8.md)。独立来源正式评估和人工评价仍待补充，P9 尚未开始。

## 1. 先分清本地和服务器

你的主要模型、照片、点云和运行结果在 HPC，Mac 上保存代码副本与下载的成果。两边有相似目录，但不是同一个文件系统。

- **Mac 工作目录**：`/Users/apple/Documents/ChatGPT/FYP`。
- **Mac 代码仓库**：`/Users/apple/Documents/ChatGPT/FYP/code/FYP`。执行项目脚本时应进入这一层，不能只停在上一级。
- **HPC 代码仓库**：`/gpfs/work/int/feijiayao22/work/code/FYP`，即登录后的 `~/work/code/FYP`。
- **HPC 模型与第三方代码**：`~/work/deps/`。
- **HPC 输入图片**：`~/work/data/inbox/`。
- **HPC 环境**：`~/work/envs/`，以及已经建立的测试/服务环境。
- **HPC 结果、作业脚本及日志**：`~/work/runs/`。
- **Mac 已下载的成果**：`/Users/apple/Documents/ChatGPT/FYP/outputs/`。

下面每段命令都会标出执行位置。看到“Mac 终端”就在本地执行；看到“HPC 登录终端”先 `ssh xjtlu`；看到“Slurm 脚本”则由 `sbatch` 提交到计算节点执行。登录节点用于文件操作和提交作业，测试、渲染、API 流程和模型计算都放在 Slurm 作业中。

`~/work` 中的 `~` 在 HPC 指服务器家目录；在 Mac 指 Mac 家目录。请不要把服务器路径直接交给 Mac 上的 Python。

## 2. 选择适合你的运行方式

第一次接触代码，先执行第 3 节的 CPU 演示；它只验证安装和故事板代码，不需要 GPU、模型或 API。

想查看已有“人物＋熊”的真实效果，使用第 5 节的 **P7 CPU 网页**。它支持已有资产生成故事和局部编辑，不执行新的分割或重建。

想上传新照片、点击主体、选择重建方法，使用第 5 节的 **P6 完整 GPU 网页**。它需要 GPU 排队，以及已有 SAM2 和重建模型。

想在终端分别测试三种重建方法，使用第 7 节。想研究故事理解和编译代码，使用第 8 节。想查看 P8 方法比较，使用第 10 节。

## 3. 在 Mac 运行最小 CPU 演示

### 3.1 进入正确的代码目录

**Mac 终端：**

```bash
cd /Users/apple/Documents/ChatGPT/FYP/code/FYP
pwd
ls scripts/storyboard.py requirements-storyboard.txt
```

`pwd` 应显示末尾为 `code/FYP` 的路径。找不到这两个文件，说明没有进入代码仓库。

### 3.2 为 CPU 层建立独立环境

建议 Python 3.11，与现有 HPC 环境保持一致。下面的 `python3` 应指向你准备使用的 Python；如安装了 `python3.11`，可将第一条替换为 `python3.11 -m venv .venv`。

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements-storyboard.txt
```

后续重新打开终端，只需重新进入代码目录并执行 `source .venv/bin/activate`。不需要每次重新安装。

`requirements-storyboard.txt` 是 CPU 故事板与网页依赖，**不包含** SAM2、CUDA、Zero123++、VGGT、TripoSR 或 Hunyuan 的完整模型环境。Mac 的 CPU 演示不执行这些 GPU 模型。

### 3.3 生成并验证演示

```bash
python scripts/storyboard.py demo --out outputs/manual-demo-01
python scripts/storyboard.py validate --run outputs/manual-demo-01
python -m http.server 8000 --bind 127.0.0.1
```

保持终端运行，在 Mac 浏览器打开：

```text
http://127.0.0.1:8000/outputs/manual-demo-01/storyboard.html
```

预期：生成故事板页面、点云/场景文件和验证 JSON；验证输出包含 `passed: true`。这是合成几何演示，不能代表照片重建效果。

退出静态服务时在此终端按 `Ctrl+C`。如果再次生成，换成 `outputs/manual-demo-02`；输出目录已存在时程序会拒绝覆盖。

## 4. 连接 HPC 并检查工作区

### 4.1 确认 VPN 与 SSH

先登录 Mac 上的 aTrust VPN，然后在 **Mac 终端**执行：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlPath=none xjtlu \
  'hostname; cd ~/work/code/FYP && pwd -P && ls scripts/storyboard.py'
```

预期：打印登录节点名称、服务器项目绝对路径及脚本名。`BatchMode=yes` 是检查现有免密认证，不会自动创建 SSH 密钥。

需要进入交互终端时：

```bash
ssh -o ControlPath=none xjtlu
cd ~/work/code/FYP
```

如果出现 `No route to host` 或连接超时，先检查 VPN 和网络。如果出现认证失败，检查 SSH 配置与既有密钥。保留主机身份校验，不用关闭 `StrictHostKeyChecking` 来掩盖连接问题。

### 4.2 查看作业

**HPC 登录终端：**

```bash
squeue -u "$USER" -o '%.18i %.10T %.10M %.20N %.35R'
```

- `PENDING`：正在排队，还没有运行服务，浏览器暂时不能连接。
- `RUNNING`：计算资源已分配；仍需检查日志是否已启动网页。
- 查询不到：作业可能已经结束，用 `sacct` 看历史。

```bash
sacct -j 你的作业ID --format=JobID,State,Elapsed,ExitCode
```

`COMPLETED` 表示作业正常退出，`TIMEOUT` 表示达到时限，`FAILED` 表示程序错误。网页作业结束后，旧隧道不会让服务继续存在。

## 5. 启动真实资产网页

### 5.1 已有资产和编辑：使用 P7 CPU 服务

**HPC 登录终端：**

```bash
sbatch --output="$HOME/work/runs/p7/20261122-edit-01/server-%j.log" \
  "$HOME/work/runs/p7/20261122-edit-01/serve.sbatch"
```

终端会返回 `Submitted batch job 数字`。记下这个数字，之后查看：

```bash
squeue -j 你的作业ID -o '%.18i %.10T %.20N %.35R'
tail -n 50 ~/work/runs/p7/20261122-edit-01/server-你的作业ID.log
```

此脚本使用 2 个 CPU 核、无 GPU，请求限时在脚本的 `#SBATCH -t` 中。编写说明书时为 25 分钟；后续以实际脚本为准。它使用 P7 独立状态目录，适合已有资产的故事生成与版本编辑。

### 5.2 新图片分割和重建：使用 P6 GPU 服务

需要完整上传流程时，改为提交：

```bash
sbatch --output="$HOME/work/runs/p6/20261122-web-01/server-%j.log" \
  "$HOME/work/runs/p6/20261122-web-01/serve.sbatch"
```

```bash
squeue -j 你的作业ID -o '%.18i %.10T %.20N %.35R'
tail -n 50 ~/work/runs/p6/20261122-web-01/server-你的作业ID.log
```

此脚本请求一张 GTX3090、4 核和 32GB 内存，限时一小时，启动前先跑测试。GPU 忙时可能排队。日志中出现 Uvicorn 启动信息后再连接网页。

**同一份状态目录同时只运行一个服务。** 已有服务在运行时复用它，不要反复提交。需要停止自己启动的服务时，确认作业 ID 后执行 `scancel 你的服务作业ID`；不要取消 P8 评估或其他不相关作业。

### 5.3 在 Mac 建立隧道

从 `squeue` 或服务日志获取**本次实际分配的计算节点**。在 **Mac 新开终端**，将下面的 `实际节点` 替换为它：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ControlPath=none \
  -L 17862:实际节点:17862 xjtlu
```

这个终端正常情况下没有输出，保持开启。Mac 浏览器打开：

```text
http://127.0.0.1:17862/
```

如果 Mac 的 17862 已占用，可仅改本地端口：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ControlPath=none \
  -L 17866:实际节点:17862 xjtlu
```

此时打开 `http://127.0.0.1:17866/`。不要把旧作业的节点名直接复制给新作业。

### 5.4 为什么“拒绝连接”

按以下顺序排查：

1. VPN 是否已连接，`ssh xjtlu` 是否成功。
2. 服务作业是否为 `RUNNING`，是否达到时限后退出。
3. 日志是否已经出现服务监听信息，还是停在测试/加载模型或报错。
4. SSH 隧道是否仍在运行，转发目标是否为本次实际节点。
5. 浏览器端口是否与 `-L` 左侧端口相同。

Mac 可检查本地监听：

```bash
lsof -nP -iTCP:17862 -sTCP:LISTEN
curl --noproxy '*' -I http://127.0.0.1:17862/
```

静态成果页与交互服务是两回事：静态页只读已生成结果；服务页面才能提交新任务。P7/P6 服务有 Slurm 时限，不是永久网站。

## 6. 在网页执行完整流程

### 6.1 从照片选择主体

使用 P6 GPU 服务上传照片，在图片上给目标主体添加前景点；背景或相邻物体可以添加背景点。查看 SAM2 掩码，确认它覆盖完整主体且没有大量背景，再确认名称。

一张照片有多个主体时，对每个主体分别选择、分别保存掩码和名称、分别重建。SAM2 负责“选哪个主体”，重建后端负责“如何生成三维形状”；切换 TripoSR/Hunyuan 不会取消主体选择步骤。

掩码本身不理想时，先调整选点或使用更清晰的照片。掩码已经混入其他物体，再换重建模型通常也不能保证形状正确。

### 6.2 选择重建方法

- `vggt`：SAM2 前景 → Zero123++ 多视图 → VGGT，结果可能受生成视图不一致影响。
- `triposr`：同一前景直接生成几何，当前默认 256 提取分辨率。
- `hunyuan`：Hunyuan3D-2mini 纯几何，当前默认 30 步、256 提取分辨率。

TripoSR 现在支持预测 RGB：使用 `--color` 或该方法配置 `gray: false`，可输出彩色点云、彩色三角网格及 WebGL 查看器；不做 UV 纹理烘焙。Hunyuan 当前仍为灰色几何，把 `gray` 设为 false 不会自动给它上色。旧灰色资产需新建重建任务，不能只刷新页面恢复颜色。名称确认不等于朝向确认；资产仍标为 proposed 时应人工检查上下、前后。

### 6.3 使用已有资产生成故事

勾选至少两个名称不同的已有资产，例如“人物”“熊”。先测试清晰、可执行的三幕要求：

```text
第一幕人物在熊的左侧附近。第二幕人物和熊保持原位，镜头关注熊。
第三幕两者继续保持原位，使用全景镜头。
```

系统调用模型形成结构化计划，再布局、渲染、验证。查看三张图、旁白和验证详情；`succeeded` 不能替代你对形状、朝向与旁白的人工判断。

当前支持静态刚体的左右/远近、靠近/退远、保持、出现/消失、overview/focus。距离为任意场景单位 U，不是米。骨骼动作、真实物理接触、连续动画和任意复杂动作不在当前能力内。

### 6.4 修改故事板

在成功故事下方选择指定幕，修改旁白、镜头或已有关系/距离，也可以输入一句局部修改要求。其他幕默认锁定。

先测试“只改第二幕旁白”：三张图片应保持。再测试“第二幕镜头关注熊”：物体应不动，只重渲染第二幕。修改后可以切换版本并查看差异。

如果修改与第三幕既有 approach/hold 等事件冲突，系统会拒绝提案并保留旧版本。需要明确解除相关幕锁定并指定兼容修改；若要改变事件本身，返回原故事重新生成。当前编辑接口不直接开放事件、任意坐标或矩阵编辑。

## 7. 在终端切换重建方法

以下是可复用的命令模板。**GPU 推理应放入 Slurm GPU 作业，不能在 HPC 登录节点直接执行。** `/path/to/...` 是占位路径，必须替换。

### 7.1 准备配置

在代码目录中复制示例：

```bash
cp configs/reconstruction.example.yaml configs/reconstruction.local.yaml
```

填写选用后端的 `repo`、`weights`，或 VGGT 的 `pipeline_config`。配置文件的顶层只允许 `common` 和 `methods`；示例文件本身不含模型权重。机器本地配置已在 Git 忽略规则中。

共同选项包括 `points: 120000`、`seed: 42`、`gray: true`、`normalize: true`。`--points` 主要改变最终导出的采样点数，不能据此认为模型推理会按比例变快。轻量比较先保持默认 256 分辨率，不同时装载多个后端。

### 7.2 使用同一张图和同一掩码分别重建

**以下命令在已分配 GPU 且配好相应环境的计算节点执行：**

```bash
python scripts/reconstruct_asset.py --list-methods

python scripts/reconstruct_asset.py --method triposr --color \
  --image /path/to/photo.jpg --mask /path/to/mask.png \
  --config configs/reconstruction.local.yaml --out /path/to/new-triposr-run

python scripts/reconstruct_asset.py --method hunyuan \
  --image /path/to/photo.jpg --mask /path/to/mask.png \
  --config configs/reconstruction.local.yaml --out /path/to/new-hunyuan-run

python scripts/reconstruct_asset.py --method vggt \
  --image /path/to/photo.jpg --mask /path/to/mask.png \
  --config configs/reconstruction.local.yaml --out /path/to/new-vggt-run
```

每种方法使用不同的新输出目录。`mask.png` 必须与 `--image` 的尺寸一致；裁过图就要配套裁掩码。普通白底图片不是透明前景，不能因为背景白就省略掩码。

输出目录主要包含：

```text
foreground.png       实际送入重建的 RGBA 前景
pointcloud.ply       统一 XYZ/RGB 点云
run.json             方法、参数、输入哈希、计时和成功/失败原因
```

TripoSR 还输出 `mesh.ply`（顶点 RGB 与三角面）、`mesh_native.ply`（未归一化中间网格）和 `viewer.html`。把输出目录作为静态网站打开后可旋转模型、开关光照、与灰色对照，并导出当前视角 PNG；双击 HTML 也可用文件选择器载入 `mesh.ply`。下载的网格和点云保持原始 TripoSR Z-up，查看器只在显示时转换为 Y-up。隐藏部位的颜色仍由模型推测。

网页中新 TripoSR 任务完成后，可在“3 · 当前任务”直接点击“打开三维网格（旋转、光照与 PNG 导出）”；无需先下载 ZIP。旧任务没有网格时该入口不会显示。

后端还可能保存额外中间产物。默认归一化把最长边缩放到 2，保留各后端原始轴向；同点数不代表相同采样过程。比较时检查 `foreground.png`，再看侧面/背面、碎片、缺面和四肢连接。

当前 HPC GPU 环境与第三方代码有依赖适配。复用已验证网页脚本最方便；自行建新机器时按 [重建模块说明](reconstruction-methods.md) 准备，不直接升级现有 `vggt` 环境或把所有模型依赖混装。

## 8. 用 CLI 生成故事和编译计划

### 8.1 现有 HPC 的 Python 环境

以下环境变量可放入自己的 Slurm CPU 作业脚本：

```bash
STORY_PY="$HOME/work/runs/p0/20261119-regression-01/test-env/bin/python"
export PYTHONPATH="$HOME/work/envs/bear-compare/lib/python3.11/site-packages:$HOME/work/code/FYP:$HOME/work/deps/vggt"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$HOME/work/code/FYP"
```

`STORY_PY` 保存现有解释器的路径，调用时使用 `"$STORY_PY" scripts/脚本名.py`。已有资产的 CPU 故事和编辑不需要 GPU；新的图像分割/重建仍需要 GPU。

### 8.2 使用现有冻结资产执行一次新故事

下面示例复用 P8 已冻结的“人物＋熊”初态和已确认名称。它是一次新的演示，会产生真实 API 调用，不属于冻结主比较；输出写到独立 `manual/` 目录。

先在 **HPC 登录终端**准备故事文件：

```bash
mkdir -p ~/work/runs/manual
cat > ~/work/runs/manual/story.txt <<'EOF'
第一幕人物在熊的左侧附近。第二幕两者保持原位，镜头关注熊。
第三幕两者继续保持原位，使用全景镜头。
EOF
```

将下面内容保存为 `~/work/runs/manual/story.sbatch`。账户、分区与 QOS 使用当前已验证的 CPU 提交方式；队列规则变化时以集群实际配置为准。

```bash
#!/bin/bash
#SBATCH -J fyp-manual-story
#SBATCH -A yushili_phd
#SBATCH -p gpua800
#SBATCH --qos=2a800
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
set -euo pipefail
STORY_PY="$HOME/work/runs/p0/20261119-regression-01/test-env/bin/python"
export PYTHONPATH="$HOME/work/envs/bear-compare/lib/python3.11/site-packages:$HOME/work/code/FYP:$HOME/work/deps/vggt"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export FYP_LLM_API_KEY="$(cat "$HOME/work/configs/.p5-deepseek-20261120.key")"
cd "$HOME/work/code/FYP"
STORY_GROUP="$HOME/work/runs/p8/20261123-main-01/frozen/groups/person_bear"
"$STORY_PY" scripts/storyboard.py understand \
  --story-file "$HOME/work/runs/manual/story.txt" \
  --catalog "$STORY_GROUP/catalog.json" \
  --asset-root "$STORY_GROUP" --scene "$STORY_GROUP/initial_scene.json" \
  --api-model deepseek-v4-pro --max-calls 3 --allow-provisional \
  --out "$HOME/work/runs/manual/story-$SLURM_JOB_ID"
unset FYP_LLM_API_KEY
```

密钥从仓库外权限为 600 的文件读取，不在命令行写密钥内容，不开启 `set -x`。`storyboard.py understand` **没有 `--key-file` 参数**，因此此处用环境变量；网页脚本和评估脚本支持 `--key-file`，不要混用参数。

提交并查看：

```bash
sbatch --output="$HOME/work/runs/manual/story-%j.log" ~/work/runs/manual/story.sbatch
```

成功的理解结果通常在 `attempt_00/compiled/`，修正后可能是 `attempt_01/compiled/` 或 `attempt_02/compiled/`；以该次 `run.json` 的 `accepted_attempt` 为准。解析失败、澄清请求或编译失败没有可用接受版本时，先看原始响应与错误，不自行假定最后一次已经成功。

### 8.3 已有 StoryPlan 时跳过模型

`compile` 用手写或已保存的结构化计划编译，不调用语言模型：

```bash
python scripts/storyboard.py compile \
  --plan /path/to/story_plan.json \
  --asset-root /path/to/asset-root --scene /path/to/initial_scene.json \
  --allow-provisional --out /path/to/new-compiled-run
```

`--allow-provisional` 允许候选朝向，不代表朝向已确认，也不允许使用未确认的名称。故事计划、资产库与初态必须来自兼容的同一批资产 ID，不能把另一批 ID 直接拼进去。

验证普通演示、布局、P4 编译或 P5 理解结果时：

```bash
python scripts/storyboard.py validate --run /path/to/run-root
```

命令返回码为 0 表示相应验证通过。P8 基线及消融应使用其专用 `--phase report` 评分，不以此命令替代 P8 的用户要求审计。

## 9. 用 CLI 修改已成功故事

初学时优先使用网页。CLI 的 `--story-root` 指 **网页原始故事任务目录**，例如 `state/jobs/<story-job-id>`；不是任意一个 `compiled/` 目录。系统根据该任务的 `pipeline/run.json` 和 `versions.json` 找到当前有效版本。

先读任务的 `versions.json` 确认 `current_version`。没有版本索引的原始成功任务，首次有效编辑从版本 1 开始。

下面是只改第二幕旁白的请求结构，`base_version` 两处须与当前版本一致，幕 ID 也须存在：

```json
{
  "base_version": 1,
  "patch": {
    "schema_version": 1,
    "base_version": 1,
    "operations": [
      {"kind": "narration", "scene_id": "scene_2", "text": "人物与熊保持原位，镜头静静观察它们。"}
    ],
    "locked_scene_ids": ["scene_1", "scene_3"]
  }
}
```

保存到 `request.json`，在配置好的 Slurm CPU 作业中运行：

```bash
python scripts/edit_storyboard.py \
  --story-root /path/to/state/jobs/STORY_JOB_ID \
  --request /path/to/request.json \
  --out /path/to/state/jobs/NEW_EDIT_JOB_ID/compiled
```

输出要在同一 `state/jobs/` 树的新任务中，便于版本索引保存相对路径。结构化 `patch` 不调用模型；语言模式改用顶层 `text` 与 `locked_scene_ids`，并传入仓库外 `--key-file`，会进行一次 API 调用。两种模式不能同时填写。

在 CPU 环境检查十项既定修改规则：

```bash
python scripts/evaluate_edits.py --out /path/to/new-p7-check
```

加 `--language --key-file /outside/repository/private.key` 才会额外执行六条真实语言修改请求。这些验收用合成资产，验证编辑机制，不评估真实形状。

## 10. P8 方法比较如何执行和查看

P8 比较 B0 直接坐标、B1 静态关系与 Ours 关系/事件/状态/修正。当前是用户确认的探索性范围，不是十组独立留出数据的正式泛化研究。

- 先导：5 个任务 × 3 方法 × 1 次，共 15 次。
- 主比较：6 个资产组/任务组合 × 3 方法 × 3 次，共 54 次。
- 预算对照：相同 6 组合 × 2 种基线，共 12 次，调用上限也为 3。
- 消融：相同 6 组合 × 3 种机制，共 18 次，重放已有响应，不产生新模型调用。

现有主运行根目录：

```text
/gpfs/work/int/feijiayao22/work/runs/p8/20261123-main-01
```

其中 `code_snapshot/` 是冻结执行代码，`frozen/` 是冻结协议、资产、初态和独立评分答案；不要编辑这两个目录。`main/`、`sensitivity/`、`ablation/` 保存每项运行；对应 `*-report/` 为浏览页面，`*-analysis/` 为原始引用与用户目标的独立审计。

本轮全部阶段完成后，在 **Slurm CPU 作业**中可用以下命令重新评分并生成全部表格，无新 API 调用：

```bash
python scripts/evaluate_p8.py \
  --run-root "$HOME/work/runs/p8/20261123-main-01"
python scripts/summarize_p8.py \
  --run-root "$HOME/work/runs/p8/20261123-main-01"
```

`evaluate_p8.py` 会检查预登记数量，未完成时拒绝当成完整研究；还会核对冻结输入与原始调用日志哈希。它需要 HPC 的完整产物，本地轻量下载副本未包含所有点云和数组，不能直接替代完整复现数据。

新实验的调用模板如下，**会产生新费用**，不要为了查看现有结果而重新执行：

```bash
python scripts/benchmark_storyboard.py --phase pilot \
  --frozen /path/to/FROZEN --out /path/to/NEW_PILOT --key-file /outside/repo/private.key
python scripts/benchmark_storyboard.py --phase main \
  --frozen /path/to/FROZEN --out /path/to/NEW_MAIN --key-file /outside/repo/private.key
python scripts/benchmark_storyboard.py --phase sensitivity \
  --frozen /path/to/FROZEN --out /path/to/NEW_BUDGET --key-file /outside/repo/private.key
python scripts/benchmark_storyboard.py --phase ablation \
  --frozen /path/to/FROZEN --out /path/to/NEW_ABLATION --source-batch /path/to/NEW_MAIN
```

新批次还需先用 `scripts/freeze_p8.py --sources SOURCE_JSON --out NEW_FROZEN` 冻结来源和代码；具体协议见 [P8 评估说明](p8-evaluation.md)。不要直接修改现有主比较协议后继续写进同一输出目录。恢复中断时保存原批次和冻结输入，已完成项不重复请求，不确定的在途调用作为中断记录保留。

报告中的“结构化条件全部满足”不等于旁白完整语义通过。三个原图来源相互重叠，主比较两组共享鸭图；本轮不能据此给出独立组置信区间或泛化结论。人工评价页已准备，未填写前不计为有人类评分。

## 11. 把结果下载到 Mac

### 11.1 下载一份点云

**Mac 终端：**

```bash
mkdir -p /Users/apple/Documents/ChatGPT/FYP/outputs/downloads
scp -o ControlPath=none \
  xjtlu:work/runs/reconstruction-methods/20261122-01/triposr/pointcloud.ply \
  /Users/apple/Documents/ChatGPT/FYP/outputs/downloads/bear-triposr.ply
```

这是已存在的熊 TripoSR 结果路径。其他结果将远端路径替换为对应运行的 `pointcloud.ply`；先 `ssh xjtlu` 后用 `ls` 确认文件存在。下载仅复制，不重新推理。

可用支持 PLY 的三维查看器打开；若安装了 CloudCompare 或 MeshLab，直接载入文件并旋转查看。当前主要是点云，不保证带有三角面；以网格模式看不到面，不代表文件为空。

### 11.2 下载一个完整运行目录

```bash
rsync -az --progress -e 'ssh -o ControlPath=none' \
  xjtlu:work/runs/manual/实际运行目录/ \
  /Users/apple/Documents/ChatGPT/FYP/outputs/downloads/实际运行目录/
```

源路径末尾的 `/` 表示复制目录内容。完整结果可能包含较大的点云、深度和 ID 数组。只想看三幕图片时，可从网页下载 ZIP，或下载 `renders/`、`narration/` 与页面所需文件。

P8 轻量报告下载可以排除冻结资产与数组：

```bash
mkdir -p /Users/apple/Documents/ChatGPT/FYP/outputs/p8/main
rsync -az --progress -e 'ssh -o ControlPath=none' \
  --exclude code_snapshot --exclude frozen --exclude '*.tar.gz' \
  --exclude '*.ply' --exclude '*.npy' --exclude '*.npz' \
  xjtlu:work/runs/p8/20261123-main-01/ \
  /Users/apple/Documents/ChatGPT/FYP/outputs/p8/main/
```

此副本适合浏览报告，不是完整离线复现包。报告正在生成时可能尚未有全部 HTML，完成后再次同步即可。

### 11.3 在本地打开静态报告

在 **Mac** 启动：

```bash
python3 -m http.server 8001 --bind 127.0.0.1 \
  --directory /Users/apple/Documents/ChatGPT/FYP/outputs/p8
```

先导报告：`http://127.0.0.1:8001/pilot-v2/pilot-report/index.html`。

主比较全部收尾后，汇总入口：`http://127.0.0.1:8001/main/index.html`；单独主比较：`http://127.0.0.1:8001/main/main-report/index.html`。汇总入口由 `summarize_p8.py` 生成，尚未生成时不会自动出现。

当前助手启动的本地预览使用 17865 端口，它是临时进程；以后失效时，用上面的命令自行启动即可。下载后看静态报告不依赖 VPN、API 或 HPC 服务仍在运行。

## 12. 常见错误与排查位置

- **`No such file` / 找不到 scripts**：先执行 `pwd`；Mac 应在 `.../code/FYP`，服务器应在 `~/work/code/FYP`。
- **`ModuleNotFoundError`**：确认解释器和环境。CPU 演示激活自己的 `.venv`；HPC 复用服务脚本中的解释器与 `PYTHONPATH`，不要随意升级共享环境。
- **`CUDA required`**：重建命令没有在已分配 GPU 的节点运行，或解释器没有对应 CUDA PyTorch。CPU 演示和已有资产故事不需要加载重建模型。
- **输出目录已存在**：换新目录；不要删除原始结果或覆盖历史来“继续”。P8 专用脚本可以读取已有结果续跑，其他单次脚本不一定支持。
- **掩码尺寸不符/空掩码**：原图、裁剪和 mask 必须配套；先查看 `foreground.png`。
- **模型权重不存在**：检查本地配置的路径，GitHub 不提供权重、照片或资产库。clone 成功不等于 GPU 模型已经部署。
- **点云是块状、碎片或缺面**：先检查掩码，再检查生成视图/后端结果。查看 `run.json` 的实际方法和参数；导出成功不能证明重建质量好。
- **API 认证或余额错误**：看保存的提供方错误及当前服务密钥文件配置；不把密钥粘贴进日志或 Git。
- **故事没有 accepted_attempt**：查看该运行 `run.json`、各 `attempt_*/result.json` 和原始响应。失败或澄清不能当成成功故事。
- **修改返回版本冲突**：刷新当前版本并更新 `base_version`，不要强制覆盖旧版。
- **修改破坏第三幕事件**：看冲突报告，明确调整依赖或重新生成原故事；系统保留旧版本是预期行为。
- **SLURM `TIMEOUT`**：计算达到申请时限。网页重新提交并更新节点隧道；评估按同一冻结批次续跑，避免重复收费调用。

## 13. 代码入口与测试

常用入口：

- `scripts/storyboard.py`：CPU demo、验证、资产导入/组合、空间布局、StoryPlan 编译、模型故事理解。
- `scripts/serve_storyboard.py`：当前 P6/P7 网页服务。
- `scripts/reconstruct_asset.py`：统一三种重建方法。
- `scripts/edit_storyboard.py`：结构化或一次语言局部编辑。
- `scripts/evaluate_edits.py`：P7 修改验收。
- `scripts/benchmark_storyboard.py`：P8 批次与离线评分。
- `scripts/evaluate_p8.py`、`scripts/summarize_p8.py`：完整 P8 重算和汇总入口。
- `app.py`、`web/index.html`：保留的旧版 VGGT 网页；它们不是当前三种重建选择与故事编辑的入口。

在合适的 CPU 环境执行：

```bash
python -m pytest -q tests/storyboard
python -m pytest -q tests/test_reconstruction_methods.py
```

需要完整项目检查时用 `python -m pytest -q tests`。HPC 测试也放入 Slurm CPU 作业。本轮冻结代码的全量记录为 359 passed、1 warning；这证明相应测试通过，不替代真实重建和人类叙事评价。

想先走完一遍，建议顺序为：CPU 演示 → P7 已有资产网页 → P6 新图 SAM2/重建 → 下载 PLY → CLI 单次故事 → P8 已保存结果。先掌握每一步的输入和输出，再改模型或增加复杂度。
