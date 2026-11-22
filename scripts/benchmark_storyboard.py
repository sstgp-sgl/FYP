#!/usr/bin/env python3
"""Run or resume P8 without repeating completed model calls; export reports."""
import argparse,json,sys,os,random,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.scene.storage import read_json,atomic_json,sha256
from src.storyboard.benchmark import run_case,METHODS,PROTOCOL
from src.storyboard.benchmark_metrics import aggregate,score
from src.storyboard.llm_provider import DeepSeekProvider,ReplayProvider

def manifest_ok(root):
 d=read_json(root/'freeze.json')
 for path,expected in d['files'].items():
  if sha256(root/path)!=expected:raise ValueError('frozen input changed: '+path)
 for path,expected in d.get('code_files',{}).items():
  if sha256(Path(path))!=expected:raise ValueError('frozen code changed: '+path)
 return d

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frozen',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--phase',choices=['pilot','main','sensitivity','ablation','report'],required=True);p.add_argument('--key-file',type=Path);p.add_argument('--source-batch',type=Path);a=p.parse_args()
 if a.key_file:os.environ['FYP_LLM_API_KEY']=a.key_file.read_text().strip()
 freeze=manifest_ok(a.frozen);protocol=read_json(a.frozen/'protocol.json');all_tasks=read_json(a.frozen/'tasks.json');a.out.mkdir(parents=True,exist_ok=True)
 if a.phase=='report':
  rows=[]
  for path in sorted(a.out.glob('runs/*/result.json')):
   row=read_json(path);run=read_json(path.parent/'run.json')
   for artifact in run.get('artifacts',[]):
    if sha256(path.parent/artifact['path'])!=artifact['sha256']:raise ValueError('output log changed: '+artifact['path'])
   task=next(t for t in all_tasks if t['id']==row['task_id'] and t['group_id']==row['group_id'])
   candidates=sorted(path.parent.glob('attempt_*/compiled'));compiled=path.parent/run['accepted_attempt']/'compiled' if run.get('accepted_attempt') else (candidates[-1] if candidates else path.parent/'missing')
   row['score']=score(compiled,task,run.get('accepted_attempt') is not None);rows.append(row)
  atomic_json(a.out/'results.json',rows);atomic_json(a.out/'summary.json',aggregate(rows));print(json.dumps(aggregate(rows),ensure_ascii=False));return
 spec=protocol['pilot'] if a.phase=='pilot' else protocol['planned_main']
 methods=protocol['methods'];repeats=spec['repeats']
 if a.phase=='sensitivity':methods=['b0_budget3','b1_budget3'];repeats=1
 if a.phase=='ablation':methods=['no_feedback','no_state','fixed_camera'];repeats=1
 schedule=[(task,method,repeat) for task in all_tasks if task['group_id'] in spec['group_ids'] and task['id'] in spec['task_ids'] for method in methods for repeat in range(repeats)]
 random.Random(protocol['seed']).shuffle(schedule)
 existing=read_json(a.out/'batch.json') if (a.out/'batch.json').exists() else None
 batch=dict(protocol=PROTOCOL,phase=a.phase,frozen_sha256=sha256(a.frozen/'freeze.json'),code_commit=freeze['code_commit'],planned_runs=len(schedule))
 if existing and existing!=batch:raise ValueError('batch configuration changed; choose a new output directory')
 atomic_json(a.out/'batch.json',batch);rows=[]
 for task,method,repeat in schedule:
  key=f"{task['group_id']}-{task['id']}-{method}-r{repeat}";root=a.out/'runs'/key
  if (root/'result.json').exists():
   result=read_json(root/'result.json');rows.append(result);continue
  if root.exists():
   # Never repeat an uncertain request. Interrupted executions count as failures.
   run=read_json(root/'run.json') if (root/'run.json').exists() else {};candidates=sorted(root.glob('attempt_*/compiled'));compiled=root/run['accepted_attempt']/'compiled' if run.get('accepted_attempt') else (candidates[-1] if candidates else root/'missing')
   result=dict(task_id=task['id'],group_id=task['group_id'],source_groups=task.get('source_groups',[]),method=method,repeat=repeat,status='interrupted' if run.get('status')=='running' else run.get('status','interrupted'),calls=run.get('calls',0),tokens=run.get('tokens',0),seconds=run.get('seconds',0),score=score(compiled,task,run.get('accepted_attempt') is not None and run.get('status')!='running'),run_directory=str(root),accepted_attempt=run.get('accepted_attempt'),mode=run.get('mode','unknown'))
   atomic_json(root/'interruption-recovery.json',dict(policy='No uncertain call repeated',result=result));atomic_json(root/'result.json',result);rows.append(result);continue
  provider=DeepSeekProvider(protocol['shared']['model']);replay=None
  if a.phase=='ablation':
   if not a.source_batch:raise ValueError('--source-batch required for ablation')
   original=a.source_batch/'runs'/f"{task['group_id']}-{task['id']}-ours-r0";run=read_json(original/'run.json')
   choice='attempt_00' if method=='no_feedback' else (run.get('accepted_attempt') or sorted(p.name for p in original.glob('attempt_*'))[-1]);replay=original/choice/'raw_response.txt';provider=ReplayProvider(str(replay.parent))
  result=run_case(task,method,a.frozen/'groups'/task['group_id'],root,provider,max_calls=1 if replay else None,replay_from=str(replay) if replay else None)
  result.update(repeat=repeat,source_groups=task.get('source_groups',[]));atomic_json(root/'result.json',result);rows.append(result)
  atomic_json(a.out/'results.json',rows);atomic_json(a.out/'summary.json',aggregate(rows));print(key,result['score']['story_success'],result['calls'],round(result['seconds'],2),flush=True)
 atomic_json(a.out/'results.json',rows);atomic_json(a.out/'summary.json',aggregate(rows))
if __name__=='__main__':main()
