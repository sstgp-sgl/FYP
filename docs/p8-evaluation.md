# P8 探索性评估

协议在 `evaluation/p8_protocol.json`。用户已确认探索性范围和五项任务预期。数据来源不足以构成建议的十组独立留出测试：同源多种重建不能算新照片，同一照片的多个主体也归为同一来源。结论仅适用于当前小样本。

## 方法

- B0：模型直接输出各幕全部实例的绝对平移和可见性。冻结尺度/旋转，只做格式、引用和正常导出/渲染，不用布局器修正坐标。
- B1：模型逐幕输出静态关系、显式可见性、镜头和旁白，events 必须为空。允许普通状态继承与 keep_pose，执行静态关系求解；可见性布尔值直接转交既有出现/消失操作，不提供 approach/retreat/hold 事件求解。主比较一次调用，不进行模型修正。
- Ours：使用当前关系、事件、状态继承与独立检查，最多三次调用。反馈只含执行错误，不含隐藏评分答案。

三种方法共享名称/ID、初态、真实资产尺寸、坐标与距离定义、相机/渲染器、旁白长度提示、API 模型及生成参数。所有输出采用同一套独立任务要求评分；缺失要求、解析/渲染失败均保留在分母。真实图片和点云不上传 GitHub。

## 先五项试验，再确定规模

先运行五项试验 × 三方法 × 一次。根据时间、调用数、token 决定是否执行协议中两资产组各三任务 × 三方法 × 三次的探索性主比较（54 次运行）；不把试验结果当独立测试。

预算敏感性：B0/B1 也允许最多三次调用，仍不得使用隐藏答案反馈。B0 的额外调用仅根据格式、引用、物理基础/渲染检查重试，内部有效但漏掉用户要求时不会被隐藏评分器“救回”。记录实际用量，不声称调用数完全一致。

消融：去掉反馈修正、去掉继承、统一 overview 镜头。采用对应 Ours 已存响应，逐次只改变一个机制，重放不算新 API 调用或独立模型样本。无反馈使用第一份响应；另两种使用已接受响应，若主方法无成功响应则使用最后响应。失败和源响应缺失需保留记录。

## 命令

先由 `scripts/freeze_p8.py` 生成一次冻结输入，记录 commit、代码/配置/资产哈希及独立答案。现有机器路径保存在运行目录，不写入公共示例。CPU 执行均通过 Slurm，不使用登录节点推理。

```bash
python scripts/benchmark_storyboard.py --phase pilot --frozen FROZEN --out NEW_PILOT --key-file PRIVATE_KEY
python scripts/benchmark_storyboard.py --phase main --frozen FROZEN --out NEW_MAIN --key-file PRIVATE_KEY
python scripts/benchmark_storyboard.py --phase sensitivity --frozen FROZEN --out NEW_BUDGET --key-file PRIVATE_KEY
python scripts/benchmark_storyboard.py --phase ablation --frozen FROZEN --out NEW_ABLATION --source-batch NEW_MAIN
```

已经完成的 result 不会重新发出模型调用。不确定的中断请求不会自动重发，保存为中断失败并继续后续任务，避免重置调用预算。

从保存的场景、图片和用户要求重新评分并生成结果页（不调用模型）：

```bash
python scripts/benchmark_storyboard.py --phase report --frozen FROZEN --out EXISTING_BATCH
python scripts/report_p8.py --batch EXISTING_BATCH --out REPORT_DIRECTORY
```

## 指标和人工评价

按任务内先汇总重复运行，再按共享来源关系处理资产组，不将三个画面视为三条独立样本。共享原图的组会合并为来源连通簇；不足两个独立簇时不生成置信区间。报告故事成功、空间/事件要求、引用、目标可见性、锁定字段变化与覆盖率、耗时及调用量。

几何正确不等于照片重建质量，也不等于叙事质量。旁白完整事实支持、自然性、读图理解和连贯性需要人类阅读，自动结果中保持未评状态。报告工具产生匿名配对页面与独立映射文件，不虚构评审者。建议后续 8–12 人参加、部分样本两人重叠并记录一致性；本轮不自动联系任何人。

基础评分中的引用列检查已保存场景的实例绑定；原始模型引用和用户预定目标另由离线工具审计，避免把模型自行选中的目标当成用户目标。该审计不参与模型修正，不产生新 API 调用，并记录分析源码与输入哈希：

```bash
python scripts/analyze_p8.py --frozen FROZEN --batch EXISTING_BATCH --out SECONDARY_REPORT
```

主比较的预登记分配为：人物/鸭使用 T1、T2、T4；熊/鸭使用 T3、T4、T5。规模保持 54 次，同时覆盖全部五类已确认任务，两个资产组都包含 focus，便于检查镜头消融。此分配在任何主比较结果产生前冻结，不根据主比较结果筛选案例。

完整批次结束后，可用一条离线命令重新计算全部三部分的结果表、CSV、图片索引与独立审计，不调用模型：

```bash
python scripts/evaluate_p8.py --run-root /absolute/path/to/complete/p8-run
```

该命令要求主比较、预算对照和消融的所有预登记运行均有记录；中断或未完成批次不会被当成完整研究。

后续重建代码升级后，该入口优先使用运行目录的不可变 `code_snapshot/` 重新评分旧批次，仍然检查其原始源码哈希，不用新版本重建适配器替换 P8 历史执行代码。

`python scripts/summarize_p8.py --run-root /absolute/path/to/complete/p8-run` 可将三部分结果汇总成同目录下的 `index.html`。它只读取表格和独立审计，展示全部案例入口及数据局限；不会发出模型调用。

本轮六份资产变体只作来源清单，主比较实际使用人物 VGGT、熊 Hunyuan 和鸭 VGGT；并未分别测试全部重建变体。新增鸭 Hunyuan 作为形状较差的资产检查样本保留，太阳镜因掩码和形状失败排除出故事布局。两者均保留检查记录，不用于声称资产质量通过。
