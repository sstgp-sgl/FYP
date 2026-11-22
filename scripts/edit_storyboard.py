#!/usr/bin/env python3
"""Apply a typed patch or one bounded language edit to a completed story."""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--story-root',required=True,type=Path)
    p.add_argument('--request',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--api-model',default='deepseek-v4-pro')
    p.add_argument('--key-file',type=Path)
    a=p.parse_args()
    if a.key_file:os.environ['FYP_LLM_API_KEY']=a.key_file.read_text().strip()
    from src.storyboard.edits import EditStore
    from src.storyboard.llm_provider import DeepSeekProvider
    from src.scene.storage import read_json,atomic_json
    request=read_json(a.request);evidence=a.out.parent/'edit-evidence'
    evidence.mkdir(parents=True,exist_ok=True)
    atomic_json(evidence/'request.json',request)
    try:
        if set(request)-{'base_version','patch','text','locked_scene_ids'}:raise ValueError('unknown edit request fields')
        if ('patch' in request)==('text' in request):raise ValueError('supply exactly one patch or text')
        report=EditStore(a.story_root).apply(a.out,request['base_version'],patch=request.get('patch'),request=request.get('text'),locked_scene_ids=request.get('locked_scene_ids'),provider=DeepSeekProvider(a.api_model),evidence=evidence)
    except Exception as exc:
        report=dict(status='invalid_patch',request_satisfied=False,error=dict(code=getattr(exc,'code','invalid_patch'),message=str(exc)))
    report['input_mode']='language' if 'text' in request else 'typed'
    if (evidence/'provider.json').exists():report['provider']=read_json(evidence/'provider.json')
    atomic_json(a.out.parent/'edit-result.json',report)
    print(json.dumps(report,ensure_ascii=False))
    return 0 if report['status']=='succeeded' else 1

if __name__=='__main__':raise SystemExit(main())
