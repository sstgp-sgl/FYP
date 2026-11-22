#!/usr/bin/env python3
"""Build a compact, offline study cover from complete P8 batch summaries."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root
    sections = []
    totals = dict(runs=0, calls=0, reported_tokens=0)
    names = {'main': '主比较', 'sensitivity': '最多三次调用的预算对照',
             'ablation': '已保存响应的机制消融（不产生新模型调用）'}
    all_summaries = {}
    for phase, title in names.items():
        summary = json.loads((root / phase / 'summary.json').read_text())
        batch = json.loads((root / phase / 'batch.json').read_text())
        if sum(s['runs'] for s in summary.values()) != batch['planned_runs']:
            raise ValueError(f'{phase} incomplete')
        secondary = json.loads((root / (phase + '-analysis') / 'secondary-summary.json').read_text())
        all_summaries[phase] = summary
        rows = []
        for method, s in summary.items():
            audit = secondary[method]
            percent = lambda value: '未评' if value is None else f'{value:.1%}'
            cells = [method, str(s['runs']), percent(s['story_success_task_mean']),
                     percent(s['spatial_requirement_rate']), percent(s['event_requirement_rate']),
                     percent(audit['user_target_visible_frame_rate']), str(s['calls']),
                     f"{s['median_seconds']:.1f}"]
            rows.append('<tr>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>')
            totals['runs'] += s['runs']
            totals['calls'] += s['calls']
            totals['reported_tokens'] += s['reported_tokens']
        sections.append(f'<section><h2>{title}</h2><p><a href="{phase}-report/index.html">查看全部实际画面、旁白和失败项</a> · '
                        f'<a href="{phase}-report/summary.csv">汇总 CSV</a> · '
                        f'<a href="{phase}-analysis/secondary-summary.csv">原始引用/用户目标独立审计</a></p>'
                        '<div class="scroll"><table><thead><tr><th>方法</th><th>运行数</th><th>结构化条件全部满足</th>'
                        '<th>空间要求</th><th>事件要求</th><th>用户目标可见</th><th>新 API 调用</th><th>中位秒数</th>'
                        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section>')
    totals['scope'] = 'Main + budget sensitivity + response replays; excludes pilot/debug/asset generation'
    (root / 'study-totals.json').write_text(json.dumps(totals, ensure_ascii=False, indent=2))
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>P8 探索性评估结果</title><style>body{font:16px/1.75 system-ui;background:#f3f5f8;color:#243445;max-width:1400px;margin:35px auto;padding:20px}section,header{background:white;padding:24px;margin-bottom:24px;border-radius:12px}h1{margin-top:0}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{padding:12px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow:auto}a{color:#17616c}.note{border-left:4px solid #cd9a32;padding-left:16px}</style>
<header><h1>P8 · 探索性评估结果</h1><p>对比直接坐标 B0、静态关系 B1 和关系＋事件＋状态＋修正 Ours。全部失败保留在分母；三个画面不作为三个独立样本。</p>
<p class="note">“结构化条件全部满足”只表示预先确认的空间、时序、镜头与导出条件通过，不代表旁白完整语义或重建形状正确。</p></header>
SECTIONS
<section><h2>如何理解这些结果</h2><p>主比较为六个资产组/任务组合，每种方法重复三次，共 54 次。预算对照为 12 次，消融重放为 18 次。先按任务汇总重复，再汇总资产组。Ours 最多三次调用，B0/B1 主比较一次；预算对照检查额外调用的影响，不声称实际调用次数相同。</p>
<p>资产清单保留六份真实重建变体，来自三张照片、三类主体；主比较实际使用人物 VGGT、熊 Hunyuan 与鸭 VGGT，未把全部变体都当成独立测试资产。熊与泰迪熊原图相同，人物与太阳镜同源；两个主比较资产组共用鸭图，合并后只有一个独立来源连通簇。因此不提供独立组置信区间，不据此声称泛化或显著优势。</p>
<p>橡皮鸭新重建形状较差，作为适用性较差的资产检查样本保留，未进入本轮故事布局；太阳镜掩码混入皮肤和头发，结果近似块状，已记录并排除出故事比较。名称已确认，朝向仍为候选方向。重建质量问题没有由故事规划指标解决。</p>
<p>五项预期由用户在运行前确认，具体数值由程序编码；不是独立盲标数据。完整旁白事实支持、自然性和可读性尚未人工评分，当前评审人数为零。<a href="main-report/review/index.html">匿名配对评价表</a>已准备，映射文件单独保存。</p></section>
<section><h2>复现与运行记录</h2><p>主比较执行代码冻结于 76aef3c1c4c1d2d5a1f24d93b84fc1db3219888b，通过不可变代码快照与输入哈希检查；后续报告脚本只读取已保存结果，不调用模型。</p>
<p>先导 v2：15 次运行、17 次 API 调用、77,924 reported tokens，B0/B1/Ours 完全满足结构化条件分别为 0/5、3/5、4/5；它用于估计预算，不计入主比较。更早调试批次因旁白 Schema 与执行长度不一致而停止，已保留失败和中断记录；v2 同步所有方法的约束后重新冻结。</p>
<p>原始响应、解析、尝试、场景、图片和错误保存在 HPC 运行目录。<a href="environment.json">环境记录</a> · <a href="pip-freeze.txt">依赖版本</a> · <a href="study-totals.json">本轮调用统计</a>。离线重建全部表格：<code>python scripts/evaluate_p8.py --run-root /absolute/path/to/complete/p8-run</code>。P9 尚未开始。</p></section></html>'''.replace('SECTIONS', ''.join(sections))
    (root / 'index.html').write_text(page)
    print(json.dumps(totals, ensure_ascii=False))


if __name__ == '__main__':
    main()
