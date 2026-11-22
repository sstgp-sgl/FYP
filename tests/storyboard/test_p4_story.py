from dataclasses import asdict
from itertools import product
from pathlib import Path
import copy,json,subprocess,sys
import numpy as np
from PIL import Image
import pytest
from src.pointcloud_io import write_ply
from src.scene.models import Instance,SceneDocument,Transform,Y_UP
from src.scene.storage import AssetRepository,SceneRepository,read_json,atomic_json,sha256
from src.scene.composition import export
from src.storyboard.contracts import StoryPlan,json_schema
from src.storyboard.story_service import compile_story,validate
from src.storyboard.story_validation import event_check
from src.storyboard.camera import make_shot,check_shot,DEFAULT_CAMERA
from src.storyboard.spatial_contracts import DEFAULTS

ROOT=Path(__file__).resolve().parents[2]


def c(cid,rel,subject='a',obj='b',**params):return dict(constraint_id=cid,relation=rel,subject_instance_id=subject,object_instance_id=obj,parameters=params)
def ev(eid,action,subject='a',obj='b',change=.2):
    e=dict(event_id=eid,action=action,subject_instance_id=subject)
    if action in ('approach','retreat'):e.update(object_instance_id=obj,min_change=change)
    return e

def narration(text,claims=None):return dict(text=text,entity_refs=['a','b'],fact_claims=claims or [],creative_text='')

def plan_for(scene,kind='approach'):
    first='far' if kind=='approach' else 'near';second='near' if kind=='approach' else 'far'
    rows=[dict(scene_id='s1',constraints=[c('left1','left_of'),c('distance1',first)],events=[],narration=narration('第一幕：两个主体相隔。',['distance1'])),
          dict(scene_id='s2',constraints=[c('left2','left_of'),c('distance2',second)],events=[ev('motion2',kind)],narration=narration('第二幕：距离发生变化。',['motion2'])),
          dict(scene_id='s3',constraints=[c('distance3',second)],events=[ev('hold3','hold')],narration=narration('第三幕：保持位置。',['hold3']),shot=dict(template='focus',target_ids=['a']))]
    if kind=='visibility':
        rows=[dict(scene_id='s1',narration=narration('两个主体在场。')),
              dict(scene_id='s2',events=[ev('hide2','disappear')],narration=narration('红色主体暂时消失。',['hide2'])),
              dict(scene_id='s3',events=[ev('show3','appear')],narration=narration('红色主体重新出现。',['show3']))]
    return dict(schema_version=1,story_id='story_'+kind,asset_ids=list(dict.fromkeys(i.asset_id for i in scene.instances)),premise='手写三幕。',scenes=rows)


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'source';assets=AssetRepository(root);instances=[]
    for name,color,x in [('a',[225,60,45],-3),('b',[40,100,220],0)]:
        p=np.array(list(product([-.5,.5],[0.,1.],[-.5,.5])));path=tmp_path/(name+'.ply');write_ply(p,np.tile(color,(len(p),1)),path)
        r=assets.import_asset(path,role='subject',representation='point_cloud',source_kind='synthetic',producer='P4_unit_fixture',coordinate_frame=Y_UP)
        instances.append(Instance(name,r.asset_id,r.sha256,Transform(translation_xyz=[x,0,0])))
    scene=SceneDocument('input_scene',instances,[]);SceneRepository(assets).save(scene)
    return assets,scene


def run_plan(tmp_path,source,plan,name='run'):
    repo,scene=source;path=tmp_path/(name+'.json');atomic_json(path,plan);out=tmp_path/name
    report=compile_story(path,repo.root,repo.root/'scene.json',out)
    return out,report


@pytest.mark.parametrize('kind',['approach','retreat','visibility'])
def test_three_complete_handwritten_stories(source,tmp_path,kind):
    out,report=run_plan(tmp_path,source,plan_for(source[1],kind))
    assert report['passed'],report
    assert report['status']=='succeeded' and len(report['scenes'])==3
    assert validate(out)['passed']
    assert len(list((out/'renders').glob('*.png')))==3
    assert read_json(out/'run.json')['provider'] is None
    before=asdict(source[1].instances[1].transform)
    scenes=[read_json(out/'scenes'/('s'+str(i)+'.json')) for i in range(1,4)]
    for scene in scenes:
        assert [i['instance_id'] for i in scene['instances']]==['a','b']
        assert scene['instances'][1]['transform']==before
    if kind=='visibility':
        assert [s['instances'][0]['visible'] for s in scenes]==[True,False,True]
        assert set(np.unique(np.load(out/'renders/s2.ids.npy')))<={0,2}
    else:
        checks=report['scenes'][1]['event_checks'][0]
        assert checks['passed'] and checks['measured']['distance_change']>=.2-1e-4
        assert scenes[2]['instances'][0]['transform']==scenes[1]['instances'][0]['transform']
        shot1=read_json(out/'shots/shot_s1.json');shot2=read_json(out/'shots/shot_s2.json');shot3=read_json(out/'shots/shot_s3.json')
        assert shot1['T_world_from_camera']==shot2['T_world_from_camera']
        assert shot3['T_world_from_camera']!=shot2['T_world_from_camera']
        assert shot1['K']==shot3['K']
        assert scenes[1]['instances'][0]['transform']['uniform_scale']==scenes[2]['instances'][0]['transform']['uniform_scale']
    assert sha256(source[0].verify(source[0].get(source[1].instances[0].asset_id)))==source[1].instances[0].asset_hash


def test_approach_can_still_be_far(source,tmp_path):
    p=plan_for(source[1]);p['scenes'][1]['constraints']=[c('left2','left_of'),c('distance2','far')];p['scenes'][2]['constraints']=[c('distance3','far')]
    out,report=run_plan(tmp_path,source,p)
    assert report['passed'],report
    motion=report['scenes'][1]['event_checks'][0]
    assert motion['measured']['distance_change']>=.2-1e-4
    far=next(c for c in report['scenes'][1]['checks'] if c['constraint_id']=='distance2')
    assert far['measured']>=1.5-1e-4


@pytest.mark.parametrize('mutation,code',[
 (lambda p:p['scenes'][2]['events'][0].update(action='pick_up'),'unsupported_action'),
 (lambda p:p['asset_ids'].append('missing_asset'),'unknown_asset'),
 (lambda p:p['scenes'][1]['events'][0].update(object_instance_id='missing'),'unknown_reference'),
 (lambda p:p['scenes'][0].update(base_scene_id='s3'),'invalid_inheritance'),
 (lambda p:p['scenes'][1].update(scene_id='s1'),'invalid_story_plan'),
 (lambda p:p['scenes'][0]['narration'].update(entity_refs=['missing']),'unknown_reference'),
 (lambda p:p['scenes'][1]['events'].append(ev('another2','hold')),'conflicting_events'),
])
def test_invalid_plan_fails_without_success_fallback(source,tmp_path,mutation,code):
    p=plan_for(source[1]);mutation(p);out,report=run_plan(tmp_path,source,p)
    assert not report['passed'] and report['error']['code']==code,report
    assert read_json(out/'run.json')['status']=='failed'
    assert not list((out/'renders').glob('*.png'))


def test_runtime_third_scene_failure_is_partial(source,tmp_path):
    p=plan_for(source[1]);p['scenes'][2]['events']=[ev('impossible3','approach',change=100)];p['scenes'][2]['narration']['fact_claims']=[]
    out,report=run_plan(tmp_path,source,p)
    assert not report['passed'] and report['status']=='partial_failed',report
    assert [s['passed'] for s in report['scenes']]==[True,True,False]
    assert (out/'renders/s1.png').is_file() and (out/'renders/s2.png').is_file() and not (out/'renders/s3.png').exists()
    assert read_json(out/'run.json')['status']=='partial_failed'
    assert not validate(out)['passed']


def test_text_is_escaped_and_independent_of_images(source,tmp_path):
    p=plan_for(source[1]);p['premise']='<script id="evil">alert(1)</script>';p['scenes'][0]['narration']['text']='<img src=x onerror="alert(2)">'
    out,report=run_plan(tmp_path,source,p)
    assert report['passed'],report
    page=(out/'storyboard.html').read_text()
    assert '<script id="evil">' not in page and '<img src=x onerror=' not in page
    assert '&lt;script' in page and '&lt;img' in page
    assert (out/'narration/s1.txt').read_text().strip()==p['scenes'][0]['narration']['text']


def test_occlusion_and_projected_crop_are_checked():
    b={'a':[[-.5,0,-.5],[.5,1,.5]]}
    shot=make_shot('s1',dict(template='focus',target_ids=['a'],complete_display=True),b,b,DEFAULT_CAMERA)
    hidden=np.zeros((640,960),np.int32)
    report=check_shot(shot,b,hidden,{'a':1},DEFAULT_CAMERA)
    assert not report['passed'] and not report['checks'][0]['passed']
    visible=hidden.copy();visible[300,400]=1
    assert check_shot(shot,b,visible,{'a':1},DEFAULT_CAMERA)['passed']
    moved={'a':[[99,0,-.5],[100,1,.5]]}
    assert not check_shot(shot,moved,visible,{'a':1},DEFAULT_CAMERA)['passed']


def test_inherited_unrequested_object_tampering_fails_after_rehash(source,tmp_path,monkeypatch):
    out,report=run_plan(tmp_path,source,plan_for(source[1]));assert report['passed']
    assets=AssetRepository(out);s=SceneRepository(assets).load('scenes/s2.json');s.instances[1].transform.translation_xyz[2]+=1
    SceneRepository(assets).save(s,'scenes/s2.json');export(s,assets,'scenes/s2.ply')
    run=read_json(out/'run.json')
    for a in run['artifacts']:a['sha256']=sha256(out/a['path'])
    atomic_json(out/'run.json',run)
    monkeypatch.setattr('src.storyboard.layout.solve',lambda *a,**k:pytest.fail('validator called solver'))
    bad=validate(out);assert not bad['passed']
    assert any('unrequested position change: b' in s.get('inheritance_errors',[]) for s in bad['scenes'])


def test_ground_near_is_not_approach_by_itself(source):
    assets,scene=source;bounds={'a':[[-3.5,0,-.5],[-2.5,1,.5]],'b':[[-.5,0,-.5],[.5,1,.5]]}
    same=event_check(ev('e','approach'),scene,scene,bounds,bounds,DEFAULTS)
    assert not same['passed'] and same['measured']['distance_change']==0


def test_frozen_plan_references_and_source_scale_dont_drift(source,tmp_path):
    out,report=run_plan(tmp_path,source,plan_for(source[1]));assert report['passed']
    p=read_json(out/'story_plan.json');p['scenes'][1]['narration']['text']='changed output'
    atomic_json(out/'story_plan.json',p)
    run=read_json(out/'run.json')
    for a in run['artifacts']:a['sha256']=sha256(out/a['path'])
    atomic_json(out/'run.json',run)
    bad=validate(out);assert not bad['passed']
    assert 'normalized plan differs from frozen original input' in bad['artifact_errors']


def test_cli_compile_and_validate_exit_codes(source,tmp_path):
    p=tmp_path/'plan.json';atomic_json(p,plan_for(source[1]));out=tmp_path/'cli';cli=[sys.executable,str(ROOT/'scripts/storyboard.py')]
    args=['compile','--plan',str(p),'--asset-root',str(source[0].root),'--scene',str(source[0].root/'scene.json'),'--out',str(out)]
    proc=subprocess.run(cli+args,cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode==0,proc.stdout+proc.stderr
    assert subprocess.run(cli+['validate','--run',str(out)],cwd=ROOT,capture_output=True).returncode==0
    (out/'renders/s3.png').unlink()
    assert subprocess.run(cli+['validate','--run',str(out)],cwd=ROOT,capture_output=True).returncode==1
    before=sha256(out/'run.json')
    assert subprocess.run(cli+args,cwd=ROOT,capture_output=True).returncode==1
    assert sha256(out/'run.json')==before


def test_schema_is_exportable_and_references_are_not_guessed(source):
    schema=json_schema();assert schema['properties']['schema_version']['const']==1
    assert set(schema['properties']['scenes']['items']['properties']['events']['items']['properties']['action']['enum'])=={'hold','approach','retreat','appear','disappear'}
    parsed=StoryPlan.parse(plan_for(source[1]),source[1],source[0])
    assert [s.base_scene_id for s in parsed.scenes]==['initial','s1','s2']


def test_bad_visibility_transition_is_not_silent(source,tmp_path):
    p=plan_for(source[1],'visibility');p['scenes'][1]['events'][0]['action']='appear'
    out,report=run_plan(tmp_path,source,p)
    assert not report['passed'] and read_json(out/'run.json')['error']['code']=='invalid_visibility_transition'
    assert (out/'renders/s1.png').is_file()


def test_event_uses_chronological_previous_not_an_older_inheritance_base(source,tmp_path):
    p=plan_for(source[1]);p['scenes'][2]['base_scene_id']='s1'
    # Restoring the older far state does not satisfy "hold" relative to the preceding near frame.
    out,report=run_plan(tmp_path,source,p)
    assert report['status']=='partial_failed' and not report['passed']
    assert [s['passed'] for s in report['scenes']]==[True,True,False]


def test_hand_computed_event_distance():
    from types import SimpleNamespace
    def state():return SceneDocument('frame',[Instance('a','asset','0'*64),Instance('b','asset','0'*64)],[])
    before={'a':[[-3.5,0,-.5],[-2.5,1,.5]],'b':[[-.5,0,-.5],[.5,1,.5]]}
    after={'a':[[-3.3,0,-.5],[-2.3,1,.5]],'b':before['b']}
    result=event_check(ev('event','approach'),state(),state(),after,before,DEFAULTS)
    assert result['passed'] and result['measured']['before_center_distance']==3
    assert result['measured']['after_center_distance']==pytest.approx(2.8)
    assert result['measured']['distance_change']==pytest.approx(.2)
