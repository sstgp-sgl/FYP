#!/usr/bin/env python3
"""Export P8 CSV, a self-contained result index and anonymous human-review pairs."""
import argparse,csv,json,html,random,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=json.loads((a.batch/'results.json').read_text());summary=json.loads((a.batch/'summary.json').read_text())
 with (a.out/'summary.csv').open('w') as f:
  columns=['method']+list(next(iter(summary.values())).keys());writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
  for method,data in summary.items():writer.writerow(dict(method=method,**data))
 flat=[];cards=[]
 for row in rows:
  root=a.batch/'runs'/f"{row['group_id']}-{row['task_id']}-{row['method']}-r{row['repeat']}";relative=root.relative_to(a.batch);run=json.loads((root/'run.json').read_text());candidate=root/run['accepted_attempt']/'compiled' if run.get('accepted_attempt') else next(iter(sorted(root.glob('attempt_*/compiled'),reverse=True)),root/'missing')
  flat.append(dict(group=row['group_id'],task=row['task_id'],method=row['method'],repeat=row['repeat'],success=row['score']['story_success'],requirements_passed=row['score']['requirements']['passed'],requirements_total=row['score']['requirements']['total'],calls=row['calls'],tokens=row['tokens'],seconds=row['seconds']))
  destination=a.out/'cases'/relative.name;destination.mkdir(parents=True,exist_ok=True);frames=[]
  plan=json.loads((candidate/'story_plan.json').read_text()) if (candidate/'story_plan.json').exists() else {};narrations={s['scene_id']:s['narration']['text'] for s in plan.get('scenes',[])}
  for n in range(1,4):
   sid='scene_'+str(n);image=candidate/f'renders/{sid}.png'
   if image.exists():shutil.copy2(image,destination/(sid+'.png'))
   frames.append(dict(scene_id=sid,image=f'cases/{relative.name}/{sid}.png' if image.exists() else None,narration=narrations.get(sid,'')))
  messages=json.loads((root/'attempt_00/messages.json').read_text()) if (root/'attempt_00/messages.json').exists() else []
  story=json.loads(messages[1]['content'].split('\nSHARED_ASSET_INFORMATION=')[0]).get('user_story','') if len(messages)>1 else ''
  cards.append(dict(story=story,key=relative.name,group=row['group_id'],task=row['task_id'],method=row['method'],repeat=row['repeat'],success=row['score']['story_success'],requirements=row['score']['requirements'],calls=row['calls'],seconds=row['seconds'],frames=frames,failures=[f for f in row['score']['facts'] if not f['passed']]))
 with (a.out/'runs.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
 data=json.dumps(dict(summary=summary,cases=cards),ensure_ascii=False).replace('<','\\u003c')
 page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P8 探索性比较</title><style>body{font:15px/1.7 system-ui;max-width:1450px;margin:30px auto;padding:18px;color:#243445;background:#f3f5f8}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.frames{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}img{width:100%}article{background:white;margin:22px 0;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}select,button{font:inherit;padding:6px}a{color:#17616c}@media(max-width:750px){.frames{grid-template-columns:1fr}}</style><h1>P8 · 探索性方法比较</h1><p>共享照片来源的小样本比较，不是独立冻结测试集的泛化结论。分母来自预先确认的用户要求，失败保留。旁白完整语义尚待人工评价。</p><p><a href="summary.csv">汇总 CSV</a> · <a href="runs.csv">全部运行 CSV</a> · <a href="review/index.html">匿名人工评价表</a></p><table><thead><tr><th>方法</th><th>运行数</th><th>故事完全成功（任务内先平均）</th><th>空间要求</th><th>事件要求</th><th>调用数</th><th>耗时中位数（秒）</th></tr></thead><tbody id="summary"></tbody></table><p>共同原始照片使资产组相关；独立来源簇不足时不生成置信区间。三幕不作为三个独立样本。</p><label>方法 <select id="method"><option value="all">全部</option></select></label> <label>结果 <select id="status"><option value="all">全部</option><option value="pass">成功</option><option value="fail">失败</option></select></label><div id="cards"></div><script>const data=DATA;const $=id=>document.getElementById(id);const percent=v=>v==null?'未适用':(v*100).toFixed(1)+'%';for(const [method,s]of Object.entries(data.summary)){const tr=document.createElement('tr');for(const v of [method,s.runs,percent(s.story_success_task_mean),percent(s.spatial_requirement_rate),percent(s.event_requirement_rate),s.calls,s.median_seconds.toFixed(2)]){const td=document.createElement('td');td.textContent=v;tr.append(td);}$('summary').append(tr);const op=document.createElement('option');op.value=method;op.textContent=method;$('method').append(op);}function draw(){$('cards').replaceChildren();for(const row of data.cases){if($('method').value!=='all'&&$('method').value!==row.method)continue;if($('status').value!=='all'&&(row.success?'pass':'fail')!==$('status').value)continue;const card=document.createElement('article'),h=document.createElement('h2'),p=document.createElement('p'),frames=document.createElement('div');h.textContent=row.group+' / '+row.task+' / '+row.method+' / r'+row.repeat;p.textContent=(row.success?'完整成功':'未完整成功')+' · 要求 '+row.requirements.passed+'/'+row.requirements.total+' · 调用 '+row.calls+' · '+row.seconds.toFixed(2)+' 秒';frames.className='frames';for(const frame of row.frames){const section=document.createElement('section'),title=document.createElement('h3'),caption=document.createElement('p');title.textContent=frame.scene_id;section.append(title);if(frame.image){const img=document.createElement('img');img.src=frame.image;img.alt='实际点云分镜';section.append(img);}else{const absent=document.createElement('p');absent.textContent='本次运行未生成该幕图片';section.append(absent);}caption.textContent=frame.narration;section.append(caption);frames.append(section);}card.append(h,p,frames);if(row.failures.length){const detail=document.createElement('details'),title=document.createElement('summary'),pre=document.createElement('pre');title.textContent='未满足的预标注要求';pre.textContent=JSON.stringify(row.failures,null,2);detail.append(title,pre);card.append(detail);}$('cards').append(card);}}$('method').onchange=draw;$('status').onchange=draw;draw();</script>'''.replace('DATA',data)
 (a.out/'index.html').write_text(page)
 # Blinding copies remove method names from image paths; mapping stays outside review page.
 review=a.out/'review';review.mkdir(exist_ok=True);pairs=[];key=[];rng=random.Random(20261123)
 groups={}
 for card in cards:
  if card['repeat']==0:groups.setdefault((card['group'],card['task']),{})[card['method']]=card
 for (group,task),methods in groups.items():
  for baseline in ['b0','b1']:
   if baseline not in methods or 'ours' not in methods:continue
   candidates=[methods[baseline],methods['ours']];rng.shuffle(candidates);code='P'+str(len(pairs)+1).zfill(3);items=[]
   for label,card in zip(['A','B'],candidates):
    frame_data=[]
    for n,frame in enumerate(card['frames'],1):
     rel=f'{code}-{label}-{n}.png'
     if frame['image']:shutil.copy2(a.out/frame['image'],review/rel)
     frame_data.append(dict(image=rel if frame['image'] else None,narration=frame['narration']))
    items.append(dict(label=label,frames=frame_data));key.append(dict(pair=code,label=label,run=card['key'],method=card['method']))
   pairs.append(dict(pair=code,group=group,task=task,story=methods['ours']['story'],items=items))
 (a.out/'blind-key.json').write_text(json.dumps(key,ensure_ascii=False,indent=2))
 (review/'pairs.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2))
 # Ratings are optional until the user actually completes them; no invented participants.
 human='''<!doctype html><meta charset="utf-8"><title>P8 人工评价</title><style>body{font:16px/1.7 system-ui;max-width:1350px;margin:30px auto;padding:20px}.pair{border-bottom:1px solid #ddd;padding:20px 0}.frames{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}img{width:100%}textarea{width:100%;min-height:90px}select,input,button{font:inherit;padding:6px}</style><h1>P8 匿名人工评价表</h1><p>方法名已隐藏。请独立看完整旁白与图片，逐句记录可检验事实是否有依据；隐喻单独评价。缺图是本次方法失败，不能自行补想象。建议 8–12 位评审，部分样本两人独立评分并比较一致性；当前未产生任何人类评分。</p><label>匿名评审编号 <input id="reviewer"></label><p>每组分别记录：理解是否正确、连贯性、无依据/遗漏的原子事实，以及理由。不要用模型的 fact_claims 替代阅读。</p><div id="pairs"></div><button id="export">导出已填写评价 JSON</button><script>const data=PAIRS;for(const pair of data){const section=document.createElement('section');section.className='pair';const h=document.createElement('h2');h.textContent=pair.pair+' / '+pair.group+' / '+pair.task;section.append(h);const story=document.createElement('p');story.textContent='原始要求：'+pair.story;section.append(story);for(const item of pair.items){const title=document.createElement('h3');title.textContent='方案 '+item.label;const frames=document.createElement('div');frames.className='frames';for(const f of item.frames){const div=document.createElement('div');if(f.image){const im=document.createElement('img');im.src=f.image;div.append(im);}else div.textContent='未生成图片';const p=document.createElement('p');p.textContent=f.narration;div.append(p);frames.append(div);}section.append(title,frames);}const label=document.createElement('label');label.textContent='评价与逐句事实记录：';const text=document.createElement('textarea');text.dataset.pair=pair.pair;label.append(text);section.append(label);document.getElementById('pairs').append(section);}document.getElementById('export').onclick=()=>{const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){alert('请填写匿名评审编号');return;}const ratings=[...document.querySelectorAll('textarea')].filter(t=>t.value.trim()).map(t=>({pair:t.dataset.pair,notes:t.value}));if(!ratings.length){alert('尚未填写评价');return;}const blob=new Blob([JSON.stringify({reviewer,ratings},null,2)],{type:'application/json'}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='p8-review-'+reviewer+'.json';link.click();};</script>'''.replace('PAIRS',json.dumps(pairs,ensure_ascii=False).replace('<','\\u003c'))
 (review/'index.html').write_text(human)
 print(json.dumps(dict(cases=len(cards),blind_pairs=len(pairs),human_ratings=0,report=str(a.out/'index.html'))))
if __name__=='__main__':main()
