#!/usr/bin/env python3
"""Offline P8 secondary metrics, failure taxonomy and paired task differences.
Never sends these measurements or hidden answers back to a model.
"""
import argparse,json,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.scene.storage import read_json,atomic_json,sha256
from src.scene.models import SceneDocument
from src.storyboard.spatial_service import LayoutAssets
from src.storyboard.spatial_validation import measured_bounds
from src.storyboard.camera import check_shot

def references(plan,instance_ids,asset_ids):
 values=[]
 def walk(obj):
  if isinstance(obj,list):
   for v in obj:walk(v)
  elif isinstance(obj,dict):
   for key,value in obj.items():
    if key in {'instance_id','subject_instance_id','object_instance_id'} and value is not None:
     valid=value in instance_ids
     if key=='object_instance_id' and value==obj.get('subject_instance_id') and (obj.get('action') in {'approach','retreat'} or obj.get('relation') in {'left_of','right_of','near','far','non_overlap'}):valid=False
     values.append(valid)
    elif key in {'entity_refs','target_ids'} and isinstance(value,list):values.extend(v in instance_ids for v in value)
    elif key=='asset_ids' and isinstance(value,list):values.extend(v in asset_ids for v in value)
    elif key=='visibility' and isinstance(value,dict):values.extend(v in instance_ids for v in value)
    else:walk(value)
   if obj.get('action') in {'approach','retreat'} and not obj.get('object_instance_id'):values.append(False)
 walk(plan)
 return dict(correct=sum(values),total=len(values),rate=sum(values)/len(values) if values else None)

def secondary(row,batch,frozen,task):
 root=batch/'runs'/f"{row['group_id']}-{row['task_id']}-{row['method']}-r{row['repeat']}";run=read_json(root/'run.json')
 attempts=sorted(root.glob('attempt_*'));choice=root/run['accepted_attempt'] if run.get('accepted_attempt') else (attempts[-1] if attempts else root/'missing')
 group=frozen/'groups'/row['group_id'];initial=read_json(group/'initial_scene.json');ref=dict(correct=0,total=0,rate=None)
 try:
  raw=read_json(choice/'parsed.json');ref=references(raw.get('plan'),{i['instance_id'] for i in initial['instances']},{i['asset_id'] for i in initial['instances']})
 except (OSError,ValueError,TypeError):pass
 visible=[];compiled=choice/'compiled'
 for n in range(1,4):
  try:
   expected=next(f for f in task['requirements'] if f['kind']=='shot' and f['scene']==n)
   targets=expected['targets'] if expected['template']=='focus' else [f['subject'] for f in task['requirements'] if f['kind']=='visibility' and f['scene']==n and f['visible']]
   scene=SceneDocument(**read_json(compiled/f'scenes/scene_{n}.json'));assets=LayoutAssets(compiled);bounds=measured_bounds(scene,assets)
   shot=read_json(compiled/f'shots/shot_scene_{n}.json');shot.update(target_ids=targets,complete_display=True)
   identity=np.load(compiled/f'renders/scene_{n}.ids.npy',allow_pickle=False);config=read_json(compiled/'config.json');saved=read_json(compiled/'run.json')
   visible.append(check_shot(shot,bounds,identity,saved['instance_ids'],config['camera'])['passed'])
  except Exception:visible.append(False)
 failure='none'
 if not row['score']['story_success']:
  if run.get('accepted_attempt'):failure='user_requirements_unmet'
  else:
   last=run.get('attempts',[{}])[-1];error=last.get('error',{})
   if last.get('status')=='needs_clarification':failure='unexpected_clarification'
   elif 'narration_length' in error.get('message',''):failure='narration_format'
   else:
    saved=read_json(compiled/'run.json') if (compiled/'run.json').exists() else {}
    failure=error.get('code') or saved.get('error',{}).get('code') or 'executor_validation_or_interruption'
 return dict(group=row['group_id'],task=row['task_id'],method=row['method'],repeat=row['repeat'],raw_references=ref,user_target_visible_frames=sum(visible),user_target_expected_frames=3,failure_type=failure,structured_success=row['score']['story_success'])

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frozen',required=True,type=Path);p.add_argument('--batch',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=read_json(a.batch/'results.json');tasks=read_json(a.frozen/'tasks.json');values=[]
 atomic_json(a.out/'analysis-provenance.json',dict(analysis_code_sha256=sha256(Path(__file__)),input_results_sha256=sha256(a.batch/'results.json'),frozen_manifest_sha256=sha256(a.frozen/'freeze.json'),task_sha256=sha256(a.frozen/'tasks.json'),policy='Offline only; raw references and expected user targets, never fed back to planners'))
 for r in rows:values.append(secondary(r,a.batch,a.frozen,next(t for t in tasks if t['id']==r['task_id'] and t['group_id']==r['group_id'])))
 summaries={}
 for method in sorted({v['method'] for v in values}):
  selected=[v for v in values if v['method']==method];grouped={}
  for v in selected:grouped.setdefault((v['group'],v['task']),[]).append(v)
  def balanced(fn):
   means=[]
   for rr in grouped.values():
    obs=[fn(v) for v in rr];obs=[v for v in obs if v is not None]
    if obs:means.append(float(np.mean(obs)))
   return float(np.mean(means)) if means else None
  counts={}
  for v in selected:counts[v['failure_type']]=counts.get(v['failure_type'],0)+1
  summaries[method]=dict(raw_reference_correct_rate=balanced(lambda v:v['raw_references']['rate']),readable_reference_runs=sum(v['raw_references']['total']>0 for v in selected),runs=len(selected),user_target_visible_frame_rate=balanced(lambda v:v['user_target_visible_frames']/3),failure_types=counts)
 paired={}
 for baseline in ['b0','b1']:
  deltas=[]
  for key in sorted({(v['group'],v['task']) for v in values}):
   ours=[v['structured_success'] for v in values if (v['group'],v['task'])==key and v['method']=='ours'];other=[v['structured_success'] for v in values if (v['group'],v['task'])==key and v['method']==baseline]
   if ours and other:deltas.append(dict(group=key[0],task=key[1],difference=float(np.mean(ours)-np.mean(other))))
  if deltas:paired['ours-minus-'+baseline]=dict(task_differences=deltas,mean_difference=float(np.mean([v['difference'] for v in deltas])),confidence_interval=None,note='Shared photo sources form one connected cluster; no independent-group confidence interval.')
 atomic_json(a.out/'secondary-results.json',values);atomic_json(a.out/'secondary-summary.json',summaries);atomic_json(a.out/'paired-differences.json',paired)
 with (a.out/'secondary-summary.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['method']+list(next(iter(summaries.values()))));writer.writeheader()
  for method,value in summaries.items():writer.writerow(dict(method=method,**value))
 print(json.dumps(summaries,ensure_ascii=False))
if __name__=='__main__':main()
