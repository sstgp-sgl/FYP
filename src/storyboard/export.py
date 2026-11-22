"""Escaped, portable three-frame storyboard index; retain readable partial failures."""
from pathlib import Path
import html,json


def write_storyboard(root,plan,report,run):
    root=Path(root);cards=[]
    for intent in plan.scenes if plan else []:
        image=root/'renders'/(intent.scene_id+'.png')
        record=next((s for s in report.get('scenes',[]) if s['scene_id']==intent.scene_id),{})
        status='PASS' if record.get('passed') else 'FAILED / UNCHECKED'
        visual='<img src="renders/'+intent.scene_id+'.png" alt="'+html.escape(intent.scene_id,quote=True)+' actual point-cloud render">' if image.is_file() else ''
        links=[]
        for path,label in [('scenes/'+intent.scene_id+'.json','场景'),('shots/shot_'+intent.scene_id+'.json','相机'),('narration/'+intent.scene_id+'.txt','旁白')]:
            if (root/path).is_file():links.append('<a href="'+path+'">'+label+'</a>')
        cards.append('<article><h2>'+html.escape(intent.scene_id)+' · '+status+'</h2>'+visual+'<p class="caption">'+html.escape(intent.narration['text'])+'</p><p>镜头：'+html.escape(intent.shot['template'])+'</p><p>'+' · '.join(links)+'</p><details><summary>空间、事件与镜头检查</summary><pre>'+html.escape(json.dumps(record,ensure_ascii=False,indent=2))+'</pre></details></article>')
    error=html.escape(json.dumps(run.get('error',{}),ensure_ascii=False,indent=2))
    title=html.escape(plan.story_id if plan else 'P4 failed story');premise=html.escape(plan.premise if plan else '')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'''+title+'''</title><style>body{font:16px/1.7 system-ui;background:#f4f6f9;color:#253247;margin:0}main{max-width:1450px;margin:30px auto;padding:24px}h1{font-size:28px}h2{font-size:19px}a{color:#235f9f}.frames{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}article{background:white;padding:14px}img{width:100%}.caption{white-space:pre-wrap}.status{padding:14px;background:#e5edf6}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}@media(max-width:850px){.frames{grid-template-columns:1fr}}</style><main><p>FYP / P4 / HANDWRITTEN STORYPLAN / NO LLM</p><h1>'''+title+'</h1><p>'+premise+'</p><p class="status">'+html.escape(report.get('status','failed').upper())+' · '+html.escape(run.get('human_status','unchecked'))+'</p><div class="frames">'+''.join(cards)+'</div><pre>'+error+'</pre><p><a href="validation.json">整部故事验证</a> · <a href="run.json">运行记录</a></p><p>静态三维故事板；不是连续动作动画。旁白为手写。focus 只改变相机，实例大小保持。候选真实资产方向沿用原确认状态。</p></main></html>'
    (root/'storyboard.html').write_text(page)
