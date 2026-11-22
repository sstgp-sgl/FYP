# P6 网页运行说明

README 提供现有 HPC 的直接启动步骤。本文件说明从其他环境部署需要准备什么。

## CPU 演示和真实网页的区别

`python scripts/storyboard.py demo --out outputs/demo-01` 只使用合成资产和 CPU 渲染，结果页面为 `outputs/demo-01/storyboard.html`。只需 `requirements-storyboard.txt`，不用模型或 API 密钥。

真实网页使用 `python scripts/serve_storyboard.py --config /absolute/path/web.local.json --host 127.0.0.1 --port 17862`。配置路径由服务器管理员指定，浏览器不能提交任意磁盘路径。相对路径按进程工作目录解释，建议配置中使用绝对路径。POSIX 系统（Linux/macOS）用于进程锁与子进程超时；HPC 推荐 Python 3.11。

## 准备依赖

1. 新建独立环境，安装 `requirements-storyboard.txt`。这是 CPU/API 层依赖，不包含 GPU 推理环境。
2. 按 SAM2 官方说明安装模型代码与权重；按 [重建模块说明](reconstruction-methods.md) 安装需要的重建后端。GPU 需匹配 CUDA/PyTorch，不要直接将旧 `requirements.txt` 当作新机器的完整环境安装方案。
3. 准备至少两个已登记资产，或先使用网页上传重建。资产登记目录及 asset_id 可由 `scripts/storyboard.py import-asset --help` 查看；名称需要确认，朝向不明确时保留 proposed。
4. 配置 DeepSeek 官方模型和密钥。密钥只通过 `FYP_LLM_API_KEY` 环境变量或 `--key-file /outside/repository/private.key` 读取；后者文件应为 600 权限。不要把密钥写进 JSON、README 或命令参数。

复制 `configs/storyboard-web.example.json` 到仓库外的本地 JSON，替换所有 `/absolute/path` 和资产 ID。`sam` 对象对应 `configs/default.yaml` 的分割配置；VGGT 另用完整 pipeline_config，示例只启用 Hunyuan。`reconstruction.methods` 中只填准备好的后端，网页将按配置展示。每个资产的 library 指向包含 `assets/<asset_id>/asset.json` 的资产库根目录，key 为网页选择标识，instance_id 为故事中的唯一实例标识。

```bash
python scripts/serve_storyboard.py \
  --config /absolute/path/storyboard-web.local.json \
  --key-file /absolute/path/private.key --host 127.0.0.1 --port 17862
```

打开 http://127.0.0.1:17862/。已有资产生成故事不加载重建模型；上传链路需要 SAM2 和所选后端。首次启动空资产库可以先上传并重建至少两个命名不同的主体，再生成故事。模型代码和权重不在仓库内，示例 JSON 不是可直接运行的真实配置。

## 当前 HPC 的实际配置

- 项目：`~/work/code/FYP`
- 配置：`~/work/runs/p6/20261122-web-01/settings.json`
- 启动脚本：`~/work/runs/p6/20261122-web-01/serve.sbatch`
- 状态及冻结资产：`~/work/runs/p6/20261122-web-01/state`
- 现有 Python：`~/work/runs/p0/20261119-regression-01/test-env/bin/python`
- 已有依赖覆盖：`~/work/envs/bear-compare/lib/python3.11/site-packages`，具体 PYTHONPATH 见上述启动脚本。

这些是现有服务器路径，并非 clone 后会生成的目录。当前设置使用原有环境覆盖运行，没有重新安装或升级 vggt。只有 Slurm 计算节点执行测试/模型与服务，登录节点用于文件操作和提交。服务作业有一小时限时，结束后需要重新提交，并按新分配节点重建 SSH 隧道。运行中的服务不会自动重载新代码；先结束旧服务再启动新服务。

## 产物与恢复

每个任务保存到 `state/jobs/<job_id>/`，故事通常在 `pipeline/attempt_00/compiled/` 保存三幕图片、场景 JSON 与验证记录；修复重试可能产生更多 attempt，以 run.json 的 accepted_attempt 为准。上传和新资产分别保存在状态目录 uploads/ 与资产库中。

网页轮询状态，刷新可恢复历史。请求 ID 相同且输入相同时返回同一任务；输入不同返回 409。重启后未结束任务标为 interrupted，不自动重复收费调用；部分失败保留已有图。ZIP 只打包登记的产物和任务元数据，冻结资产、深度/ID 数组仍在服务器，不是完整离线复现包。

这是 SSH 隧道内的单用户原型，当前不提供公网多用户认证。页面展示逐幕验证结果，验证通过不代表重建质量、朝向或开放故事语义已通过独立人类评审。
