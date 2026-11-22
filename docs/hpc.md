# XJTLU HPC 使用指南（本机实测版）

> 本文所有参数均在 `feijiayao22` 账户上实机验证（2026-11-16）。
> 官方文档 `https://hpc-xjtlu-edu-cn-s.xjtlu.edu.cn/quickstart/` 位于 aTrust 网关之后，
> 需登录 aTrust 才能访问；参考资料另见
> [WuResearchGroup HPC 教程](https://wuresearchgroup.github.io/wiki/wiki/cluster_usage/cluster_usage/)。

## 1. 登录

本机已配置好免密登录（见 `~/.ssh/config` 的 `Host xjtlu`）：

```bash
ssh xjtlu                    # 等价于 ssh feijiayao22@login.hpc.xjtlu.edu.cn
```

| 项目 | 值 |
| --- | --- |
| 集群名 | `xipuhpc` |
| 登录节点 | `xpszlogin1` / `xpszcm2`（`10.7.88.95`） |
| 账户 | `feijiayao22`，账户组 `yushili_phd` |
| 家目录 | `/gpfs/work/int/feijiayao22`（直接落在 work 上） |
| 系统 | Rocky Linux 8.6，48 核 / 125 GB（**共享登录节点**） |

## 2. 你的账号能提交什么（最关键）

提交作业必须指定 **QoS**，QoS 决定你能用哪个分区、几张卡、跑多久。你的 QoS 只有三个：

```bash
sacctmgr show assoc user=$USER format=User,Account,Cluster,QOS
# feijiayao22 | yushili_phd | xipuhpc | 2a800,cpudebug,gpudebug
```

| QoS | 对应分区 | 上限 | 单作业时长 | 并发作业数 |
| --- | --- | --- | --- | --- |
| `2a800` | `gpua800`（A800） | **2 张 GPU** | 7 天 | 2 |
| `gpudebug` | `gpudebug`（RTX 3090） | **1 张 GPU** | 1 小时 | 1 |
| `cpudebug` | `cpudebug` | **4 核** | 1 小时 | 1 |

> **注意**：`gpu4090`、`gpu3090`、`cpu8358`、`sat3090`、`aiaca40` 等分区你**没有对应 QoS，
> 提交会被拒**。想用别的卡需要找管理员（`yushili_phd` 账户的 PI）加 QoS。
> `normal` 这个 QoS 存在，但没有分配给该账户。

实际用法就是两条路：

- 调试 / 快速验证 → `gpudebug`（3090，实测**秒开**）
- 正式训练 / 大显存 → `2a800`（A800 ×2，可能排队）

## 3. 分区与硬件（`scontrol show partition` 实测）

| 分区 | 节点数 | CPU | 内存 | GPU |
| --- | --- | --- | --- | --- |
| `gpua800` | 5 | 288 | 2576 GB | 32 × `a800`（n6 另有 4 × `a800g2`） |
| `gpu4090` | 7 | 384 | 2575 GB | 55 × `rtx4090` |
| `gpu3090` | 5 | 160 | 959 GB | 40 × `gtx3090` |
| `sat3090` | 3 | 96 | 575 GB | 24 × `gtx3090` |
| `aiaca40` | 2 | 96 | 1030 GB | 16 × `a40` |
| `aiaca800` | 1 | 48 | 515 GB | 4 × `a800` |
| `cpu8358` | 14 | 896 | 7214 GB | — |
| `cpu6348` | 4 | 224 | 2061 GB | — |
| `cpuhuge1t` | 2 | 192 | 2062 GB | — |
| `gpudebug` / `cpudebug` | 各 1 | 48 / 64 | 257 / 515 GB | 8 × `gtx3090` / — |

`gpudebug` 实测节点：`gpu3090n6`，**RTX 3090 24 GB**，驱动 565.77。
A800 节点当时全被占满，未能实测显存规格（按型号通常为 80 GB）。

## 4. 存储

```bash
df -h $HOME                          # 文件系统总量（不是你的配额）
/usr/lpp/mmfs/bin/mmlsquota -u $USER # 真正的用户配额
```

实测结果：

| 项目 | 值 |
| --- | --- |
| 配额上限 | **300 GB**（当前 224.5 GB，剩余约 **75 GB**） |
| 已用文件数 | 1,095,789 个 |
| 文件系统 | GPFS，648 TB（全局，剩 58 TB） |

- `/gpfs/home/` 个人目录按官方说明仅 **10 GB**，你的 `$HOME` 已经在 `/gpfs/work` 上，没这个烦恼。
- `du -sh $HOME` **不要**在登录节点跑（GPFS 上极慢，会拖垮登录节点），用 `mmlsquota`。

## 5. 提交作业

### 5.1 快速验证模板（`gpudebug`，秒开）

`probe.sbatch`：

```bash
#!/bin/bash
#SBATCH --job-name=fyp-probe
#SBATCH --qos=gpudebug
#SBATCH --partition=gpudebug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=probe_%j.out
#SBATCH --error=probe_%j.err

hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

### 5.2 正式训练模板（`gpua800`，双卡）

```bash
#!/bin/bash
#SBATCH --job-name=fyp-train
#SBATCH --account=yushili_phd
#SBATCH --qos=2a800
#SBATCH --partition=gpua800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd $SLURM_SUBMIT_DIR

# 软件环境
module load cuda/12.1.0-gcc-8.5.0-uz47zxn

# 自己的 conda 环境（推荐把 env 建在 /gpfs/work 下）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /gpfs/work/int/feijiayao22/envs/fyp

python -u train.py
```

提交前先 `mkdir -p logs`，否则 `--output=logs/...` 会直接报错。

```bash
sbatch train.sbatch            # 提交
sbatch --parsable train.sbatch # 只回 JobID，方便脚本捕获
```

### 5.3 监控与取消

```bash
squeue -u $USER                                        # 自己的作业
squeue -u $USER -o "%.8i %.9P %.20j %.2t %.10M %R"     # 带原因
scontrol show job <JOBID>                              # 详情/为什么排队
scancel <JOBID>                                        # 取消
scancel -u $USER                                       # 取消全部
```

## 6. 排队诊断（实测踩坑）

`sinfo` 的 `mixed` **不代表还有空卡**，它只看 CPU/内存：

```
gpua800n1  mixed  cpu=48 mem=515372 gpu:a800:4
```

节点显示 `mixed`，但 `scontrol show node gpua800n1 | grep AllocTRES` 显示
`gres/gpu=4`——4 张卡全被占了。当时 5 个 A800 节点共 32 张卡**一张不剩**，
我的作业 `Reason=Priority` 排了两分钟没动静。

看空卡要看 GRES 而不是 STATE：

```bash
scontrol show node gpua800n1 | tr ' ' '\n' | grep -E '^(AllocTRES|CfgTRES)='
```

Reason 含义速查：

| Reason | 含义 | 怎么办 |
| --- | --- | --- |
| `Priority` | 资源排在你前面（GPU 被占满） | 等，或降配/换分区 |
| `Resources` | 请求量超过空闲量 | 减卡/减内存 |
| `QOSMaxGRESPerUser` | 超过 QoS 的卡数上限 | 你现在最多 2 张 A800 |
| `QOSMaxJobsPerUser` | 并发数超限 | debug 只允许 1 个 |
| `PartitionTimeLimit` | `--time` 超过分区上限 | 缩短时长 |
| `ReqNodeNotAvail` | 节点被预留/维护 | 换节点或等 |

## 7. 软件环境

Lmod 8.7.20，**3205 个模块**，全部来自 Spack：

```bash
module av                    # 全部（很长）
module av cuda               # 按名字过滤
module load cuda/12.1.0-gcc-8.5.0-uz47zxn
module list
module purge
```

**必须用完整名字**（带 gcc 版本和 hash），`module load cuda/12.1.0` 是不存在的。

⚠️ 实测大坑：**`module load` 不要放进管道**，否则模块只加载到子 shell 里，白费：

```bash
module load cuda/12.1.0-gcc-8.5.0-uz47zxn | head    # ❌ 环境变量丢失
module load cuda/12.1.0-gcc-8.5.0-uz47zxn           # ✅
```

实测可加载并生效的版本：`cuda/12.1.0`（连带 libiconv/xz/zlib/libxml2 共 5 个依赖），
加载后 `nvcc` → release 12.1，`CUDA_HOME` 正确指向 spack 前缀。
可选的 CUDA：10.2 / 11.1.1 / 11.3.1 / 11.4.0 / 11.5.2 / 11.6.2 / 11.8.0 / 12.1.0 / 12.2.0 / 12.4.1 / 12.6.0（12.6.0 为默认）。
GCC：7.3 / 8.5 / 9.5 / 10.4 / 11.4 / 12.3 / 13.1。OpenMPI：4.1.5（多套 gcc 组合）。

### Python / Conda

- 系统 `python3` = **3.9.13**，`gcc` = 8.5.0
- 系统自带 `torch 1.12.1` 但 **`cuda.is_available() == False`**（CPU 版），**别直接用它**
- 共享 conda 在 spack anaconda3（2022.10）下，已有一批公共环境：
  `base, Benchmark, BindCraft, BindCraft2, FEanalysis, PEPPAN, PopCOGenT, SE3nv,
  ViWrap, alchemlyb, alphafold3, alphagenome, antismash, anvio-8 …`
- **自己的环境建议建在 `/gpfs/work` 下**，别放 `/gpfs/home`：

```bash
conda create -p /gpfs/work/int/feijiayao22/envs/fyp python=3.10 -y
conda activate /gpfs/work/int/feijiayao22/envs/fyp
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

在作业脚本里用绝对路径 `-p` 更稳，不依赖 `conda activate <name>` 的 envs 搜索路径。

## 8. 铁律

1. **绝不在登录节点跑计算**。`xpszlogin1` 是 48 核共享节点，跑重活会被管理员警告/封号。
   装包、`git`、`module`、`sbatch` 可以；训练、推理、`du`、大文件解压不行。
2. 作业结束时间到了会被直接杀掉，注意 `--time` 留余量，中间存 checkpoint。
3. 配额 300 GB，已用 75%，注意清理 `checkpoints/`、`outputs/`。
4. `--output`/`--error` 指向的目录必须**先存在**。
5. 账上 2 张 A800 上限，多机/多卡并行要申请新 QoS。

## 9. 从本机传数据

实测 `rsync` 与 `scp` 到 HPC 均正常（`rsync` 对 AutoDL 不可用是 macOS openrsync 协议
不兼容问题，对 HPC 无此问题）：

```bash
rsync -avz --exclude 'outputs/' ./ xjtlu:~/fyp/
scp file xjtlu:~/
```

端口转发（跑 Jupyter / Viser / TensorBoard 时）：

```bash
ssh -N -L 7861:127.0.0.1:7861 xjtlu     # 然后本机开 http://127.0.0.1:7861
```
