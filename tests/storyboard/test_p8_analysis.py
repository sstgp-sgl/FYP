"""Offline metrics must check user targets, not the model's chosen targets."""
import json
from pathlib import Path
import importlib.util
from src.scene.storage import atomic_json
from src.storyboard.benchmark import run_case
from src.storyboard.benchmark_tasks import tasks
from src.storyboard.llm_provider import FixtureProvider
from tests.storyboard.test_p8_benchmark import frozen,direct
from tests.storyboard.test_p4_story import source
spec=importlib.util.spec_from_file_location('p8_analysis',Path(__file__).resolve().parents[2]/'scripts/analyze_p8.py');analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)

def test_raw_reference_denominator_includes_invalid_and_missing_binary_targets():
 plan=dict(asset_ids=['one','unknown'],scenes=[dict(constraints=[dict(subject_instance_id='a',object_instance_id='unknown')],narration={'entity_refs':['a','b']},shot={'target_ids':['b']},events=[dict(action='approach',subject_instance_id='a')])])
 result=analysis.references(plan,{'a','b'},{'one','two'})
 assert result['total']==9 and result['correct']==6

def test_expected_target_visibility_is_not_model_self_reported_visibility(frozen,tmp_path):
 task=tasks('g',['人物','熊'])[3];response=direct(frozen);response['plan']['scenes'][1]['shot']['target_ids']=['a']
 batch=tmp_path/'batch';root=batch/'runs/g-T4-b0-r0';row=run_case(task,'b0',frozen,root,FixtureProvider([json.dumps(response,ensure_ascii=False)]));row['repeat']=0
 group=frozen/'groups/g';group.mkdir(parents=True);atomic_json(group/'initial_scene.json',json.loads((frozen/'initial_scene.json').read_text()))
 result=analysis.secondary(row,batch,frozen,task)
 assert result['raw_references']['rate']==1
 assert result['user_target_visible_frames']==2
 assert not result['structured_success']

def test_frozen_main_plan_has_all_five_task_types_and_focus_in_both_groups():
 from scripts.benchmark_storyboard import eligible_tasks
 from src.scene.storage import read_json
 protocol=read_json(Path(__file__).resolve().parents[2]/'evaluation/p8_protocol.json')
 data=tasks('person_duck',['人物','橡皮鸭'])+tasks('bear_duck',['熊','橡皮鸭'])
 chosen=eligible_tasks(data,protocol['planned_main'])
 assert len(chosen)==6 and {t['id'] for t in chosen}=={'T1','T2','T3','T4','T5'}
 assert sum(t['id']=='T4' for t in chosen)==2
