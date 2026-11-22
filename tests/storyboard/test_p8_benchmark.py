"""Independent P8 requirements and fair-method bookkeeping, no model downloads."""
import copy,json,shutil
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pytest
from src.scene.storage import atomic_json,read_json,sha256
from src.storyboard.assets import freeze_assets
from src.storyboard.benchmark import run_case,execute_response,prompt,context,render_direct
from src.storyboard.benchmark_metrics import score,aggregate
from src.storyboard.benchmark_tasks import tasks
from src.storyboard.llm_provider import FixtureProvider,ProviderError,ProviderResponse
from src.storyboard.story_service import camera_settings
from src.storyboard.spatial_contracts import DEFAULTS
from tests.storyboard.test_p4_story import source

@pytest.fixture
def frozen(source,tmp_path):
 repo,scene=source;root=tmp_path/'frozen';root.mkdir();freeze_assets(repo.root,[i.asset_id for i in scene.instances],root)
 atomic_json(root/'initial_scene.json',asdict(scene));atomic_json(root/'catalog.json',[dict(instance_id=i.instance_id,asset_id=i.asset_id,display_name=['人物','熊'][n],name_confirmed=True,confirmation_source='synthetic/test') for n,i in enumerate(scene.instances)])
 atomic_json(root/'config.json',dict(schema_version=1,spatial=DEFAULTS,camera=camera_settings({'camera':dict(width=160,height=120,max_points=1000)}),events={'min_distance_change':.2}))
 return root

def direct(frozen,focus=True):
 initial=read_json(frozen/'initial_scene.json');frames=[]
 for n in range(1,4):
  frames.append(dict(scene_id='scene_'+str(n),instances=[dict(instance_id=i['instance_id'],translation_xyz=i['transform']['translation_xyz'],visible=True) for i in initial['instances']],narration=dict(text='人物与熊保持原位，镜头观察它们。',entity_refs=['a','b'],fact_claims=[],creative_text=''),shot=dict(template='focus' if n==2 and focus else 'overview',target_ids=['b'] if n==2 and focus else [],complete_display=True)))
 return dict(status='ready',plan=dict(schema_version=1,story_id='direct_test',asset_ids=[i['asset_id'] for i in initial['instances']],scenes=frames))

def relations(frozen,events=None):
 d=direct(frozen);plan=d['plan']
 for i,row in enumerate(plan['scenes']):
  row.pop('instances');row.update(constraints=[],events=events if i==1 and events else [])
 return d

def test_missing_user_requirement_fails_even_if_executor_valid(frozen,tmp_path):
 task=tasks('g',['人物','熊'])[3];task['hidden_marker']='NOT_FOR_MODEL'
 result=run_case(task,'b0',frozen,tmp_path/'run',FixtureProvider([json.dumps(direct(frozen,False),ensure_ascii=False)]))
 assert result['accepted_attempt']=='attempt_00' and not result['score']['story_success']
 assert result['score']['categories']['shot']['total']==3 and result['score']['categories']['shot']['passed']==2
 assert 'NOT_FOR_MODEL' not in (tmp_path/'run/attempt_00/messages.json').read_text()

def test_good_direct_satisfies_actual_pose_without_self_reported_ir(frozen,tmp_path):
 task=tasks('g',['人物','熊'])[3];result=run_case(task,'b0',frozen,tmp_path/'run',FixtureProvider([json.dumps(direct(frozen),ensure_ascii=False)]))
 assert result['score']['story_success'],result['score']
 assert result['score']['narration_fact_support'] is None
 assert result['score']['locked_fields']==dict(expected=30,observed=30,changed=0)

@pytest.mark.parametrize('method',['b1','ours'])
def test_relation_and_event_methods_use_same_input_and_task_audit(frozen,tmp_path,method):
 task=tasks('g',['人物','熊'])[3];result=run_case(task,method,frozen,tmp_path/method,FixtureProvider([json.dumps(relations(frozen),ensure_ascii=False)]),max_calls=1)
 assert result['score']['story_success'],result
 assert result['mode']=='synthetic/test' and result['calls']==0 and result['tokens']==0


def test_b1_cannot_silently_use_temporal_event_solver(frozen,tmp_path):
 d=relations(frozen,[dict(event_id='hold2',action='hold',subject_instance_id='a')])
 with pytest.raises(ValueError,match='relation-only'):execute_response(d,frozen,tmp_path/'b1','b1')

def test_b1_explicit_static_visibility_can_represent_disappear_and_reappear(frozen,tmp_path):
 d=relations(frozen)
 for n,s in enumerate(d['plan']['scenes'],1):s['visibility']={'a':n!=2,'b':True};s['shot']=dict(template='overview',target_ids=[],complete_display=True)
 assert execute_response(d,frozen,tmp_path/'b1','b1')
 assert score(tmp_path/'b1',tasks('g',['人物','熊'])[2],True)['story_success']


def test_missing_frame_and_tampered_artifact_do_not_disappear_from_denominator(frozen,tmp_path):
 out=tmp_path/'direct';render_direct(direct(frozen),frozen,out);task=tasks('g',['人物','熊'])[3]
 (out/'renders/scene_3.png').unlink();report=score(out,task,True)
 assert not report['story_success'] and len(report['facts'])==len(task['requirements'])
 assert report['references']['total']==6 and report['artifact_errors']
 (out/'scenes/scene_2.json').write_text('{}');report=score(out,task,True)
 assert report['references']['total']==6 and report['references']['passed']==4


def test_all_methods_receive_identical_bounds_camera_and_initial_geometry(frozen):
 _,initial,catalog,config,info=context(frozen)
 messages=[prompt('同一个故事',method,initial,catalog,config,info)[0] for method in ['b0','b1','ours']]
 suffix=[m[1]['content'].split('SHARED_ASSET_INFORMATION=')[1] for m in messages]
 assert suffix[0]==suffix[1]==suffix[2]
 for m in messages:
  assert initial.instances[0].asset_id in m[1]['content'] and 'canonical_bounds' in m[1]['content']


def test_failed_live_request_counts_as_attempt_and_keeps_full_requirements(frozen,tmp_path):
 class Failed:
  def metadata(self):return dict(mode='live_model',provider='test_transport')
  def generate(self,*args):raise ProviderError('http_500','sanitized failure')
 task=tasks('g',['人物','熊'])[3];result=run_case(task,'ours',frozen,tmp_path/'failed',Failed())
 assert result['calls']==1 and result['score']['requirements']['total']==len(task['requirements'])
 assert result['score']['requirements']['passed']==0 and not result['score']['story_success']


def test_aggregation_uses_tasks_and_connected_source_groups_not_frames():
 def row(group,task,success,sources):
  return dict(group_id=group,task_id=task,method='ours',source_groups=sources,calls=1,tokens=10,seconds=2,score=dict(story_success=success,categories={},references={'rate':1},frames=[{'visible':True}]*3,locked_fields=dict(changed=0,observed=5,expected=5)))
 rows=[row('g1','t1',x,['bear','duck']) for x in [True,False,False]]+[row('g1','t2',True,['bear','duck']),row('g2','t1',True,['person','duck'])]
 summary=aggregate(rows)['ours']
 assert summary['tasks']==3 and summary['runs']==5
 assert summary['story_success_task_mean']==pytest.approx((1/3+1+1)/3)
 assert summary['independent_source_components']==1 and summary['group_bootstrap_95_ci'] is None


def test_fixed_camera_ablation_changes_camera_only(frozen,tmp_path):
 d=relations(frozen);assert execute_response(d,frozen,tmp_path/'normal','ours')
 assert execute_response(d,frozen,tmp_path/'fixed','fixed_camera')
 for n in range(1,4):assert read_json(tmp_path/f'normal/scenes/scene_{n}.json')['instances']==read_json(tmp_path/f'fixed/scenes/scene_{n}.json')['instances']
 task=tasks('g',['人物','熊'])[3];assert not score(tmp_path/'fixed',task,True)['story_success']


def test_direct_rejects_unknown_instances_and_nonfinite_coordinates(frozen,tmp_path):
 d=direct(frozen);d['plan']['scenes'][0]['instances'][0]['instance_id']='unknown'
 with pytest.raises(ValueError):render_direct(d,frozen,tmp_path/'bad')
 d=direct(frozen);d['plan']['scenes'][0]['instances'][0]['translation_xyz'][0]=float('nan')
 with pytest.raises(ValueError):render_direct(d,frozen,tmp_path/'nan')

def test_report_recomputes_scores_from_outputs_and_refuses_changed_model_logs(frozen,tmp_path):
 import subprocess,sys
 batch=tmp_path/'batch';task=tasks('g',['人物','熊'])[3]
 for method,response in [('b0',direct(frozen)),('ours',relations(frozen))]:
  root=batch/'runs'/f'g-T4-{method}-r0';row=run_case(task,method,frozen,root,FixtureProvider([json.dumps(response,ensure_ascii=False)]),max_calls=1)
  row.update(repeat=0,source_groups=['fixture']);row['score']['story_success']=False;atomic_json(root/'result.json',row)
 atomic_json(frozen/'tasks.json',[task]);atomic_json(frozen/'protocol.json',{});atomic_json(frozen/'freeze.json',dict(files={'tasks.json':sha256(frozen/'tasks.json'),'protocol.json':sha256(frozen/'protocol.json')}))
 script=Path(__file__).resolve().parents[2]/'scripts/benchmark_storyboard.py'
 args=[sys.executable,str(script),'--phase','report','--frozen',str(frozen),'--out',str(batch)]
 assert subprocess.run(args,capture_output=True).returncode==0
 assert all(r['score']['story_success'] for r in read_json(batch/'results.json'))
 report=script.with_name('report_p8.py');completed=subprocess.run([sys.executable,str(report),'--batch',str(batch),'--out',str(tmp_path/'report')],capture_output=True)
 assert completed.returncode==0,completed.stderr
 assert (tmp_path/'report/review/index.html').is_file() and len(read_json(tmp_path/'report/blind-key.json'))==2
 (batch/'runs/g-T4-b0-r0/attempt_00/raw_response.txt').write_text('{}')
 assert subprocess.run(args,capture_output=True).returncode!=0
