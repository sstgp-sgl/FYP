"""P7 regression cases follow the user's ten confirmed modification rules."""
from pathlib import Path
import copy,json
import pytest
from pydantic import ValidationError
from src.scene.storage import read_json,atomic_json,sha256
from src.storyboard.edits import EditPatch,edit_run,EditStore,frame_files,file_hashes,language_patch
from src.storyboard.story_validation import validate_story
from src.storyboard.spatial_service import LayoutAssets
from src.storyboard.llm_provider import FixtureProvider
from tests.storyboard.test_p4_story import source,run_plan,plan_for


@pytest.fixture
def independent(source,tmp_path):
    plan=plan_for(source[1])
    for n,row in enumerate(plan['scenes'],1):
        row.update(base_scene_id='initial',events=[],constraints=[dict(constraint_id='distance'+str(n),relation='near',subject_instance_id='a',object_instance_id='b',parameters={'near_max':.5})])
        row['narration'].update(text='两个主体在场。',fact_claims=[])
    plan['scenes'][1]['shot']=dict(template='focus',target_ids=['a'])
    out,report=run_plan(tmp_path,source,plan)
    assert report['passed'],report
    return out

def patch(*operations,version=1,locks=None):
    return EditPatch.model_validate(dict(base_version=version,operations=list(operations),locked_scene_ids=locks))

def narration():return dict(kind='narration',scene_id='s2',text='小熊静静等候。')
def shot(template='focus',target='b'):return dict(kind='shot',scene_id='s2',template=template,target_ids=[target] if template=='focus' else [])
def relation(rel='near',value=.3):return dict(kind='relation',scene_id='s2',constraint_id='distance2',subject_instance_id='a',object_instance_id='b',relation=rel,parameters={('near_max' if rel=='near' else 'far_min'):value})

@pytest.mark.parametrize('operations,rendered,recompiled',[
    ([narration()],[],[]),([shot()],['s2'],[]),([shot('overview')],['s2'],[]),
    ([relation()],['s2'],['s2']),([relation('far',1.5)],['s2'],['s2']),
    ([narration(),shot('focus','b')],['s2'],[]),
])
def test_six_targeted_edits_preserve_unrelated_frames(independent,tmp_path,operations,rendered,recompiled):
    before=file_hashes(independent)
    out=tmp_path/'edit';report=edit_run(independent,out,patch(*operations))
    assert report['status']=='succeeded',report
    assert read_json(out/'run.json')['duration_seconds']>0
    assert report['request_satisfied'] and not report['unexpected_changes']
    assert report['rendered_scenes']==rendered and report['recompiled_scenes']==recompiled
    assert frame_files(out,'s1')==frame_files(independent,'s1')
    assert frame_files(out,'s3')==frame_files(independent,'s3')
    assert file_hashes(independent)==before
    assert validate_story(out,LayoutAssets(out))['passed']
    if all(op['kind']!='relation' for op in operations):assert sha256(out/'scenes/s2.json')==sha256(independent/'scenes/s2.json')
    if all(op['kind']=='narration' for op in operations):
        for i in range(1,4):assert sha256(out/f'renders/s{i}.png')==sha256(independent/f'renders/s{i}.png')

def story_store(tmp_path,compiled):
    root=tmp_path/'jobs'/'original';root.mkdir(parents=True)
    pipeline=root/'pipeline';pipeline.mkdir()
    atomic_json(pipeline/'run.json',dict(status='succeeded',accepted_attempt='attempt_00'))
    import shutil
    shutil.copytree(compiled,pipeline/'attempt_00/compiled')
    atomic_json(root/'catalog.json',[])
    return EditStore(root)

def test_version_conflict_and_history_remain_usable(independent,tmp_path):
    store=story_store(tmp_path,independent);out=store.jobs/'edit1/pipeline/attempt_00/compiled';out.parent.mkdir(parents=True)
    assert store.apply(out,1,patch=patch(narration()).model_dump())['status']=='succeeded'
    assert store.state()['current_version']==2
    stale=store.jobs/'stale/pipeline/attempt_00/compiled'
    result=store.apply(stale,1,patch=patch(narration()).model_dump())
    assert result['status']=='version_conflict' and not stale.exists()
    for v in [1,2]:assert validate_story(store.root(store.state(),v),LayoutAssets(store.root(store.state(),v)))['passed']

def test_locked_scene_rejected_and_failed_version_not_published(independent,tmp_path):
    store=story_store(tmp_path,independent);out=store.jobs/'bad/pipeline/attempt_00/compiled';out.parent.mkdir(parents=True)
    report=store.apply(out,1,patch=patch(narration(),locks=['s1','s2','s3']).model_dump())
    assert report['status']=='conflict' and 'locked' in report['error']['message']
    assert store.state()['current_version']==1

def test_third_scene_approach_dependency_is_a_conflict(source,tmp_path):
    plan=plan_for(source[1]);plan['scenes'][1].update(events=[],constraints=[dict(constraint_id='distance2',relation='far',subject_instance_id='a',object_instance_id='b',parameters={'far_min':3.0})])
    plan['scenes'][1]['narration']['fact_claims']=[]
    plan['scenes'][2].update(events=[dict(event_id='approach3',action='approach',subject_instance_id='a',object_instance_id='b',min_change=.2)])
    plan['scenes'][2]['narration']['fact_claims']=['approach3']
    base,report=run_plan(tmp_path,source,plan)
    assert report['passed'],report
    edited=edit_run(base,tmp_path/'conflict',patch(relation()))
    assert edited['status']=='conflict'
    third=next(s for s in edited['validation']['scenes'] if s['scene_id']=='s3')
    assert not third['event_checks'][0]['passed']
    assert frame_files(base,'s3')==frame_files(tmp_path/'conflict','s3')
    relaxed=patch(relation(),{**relation(value=0.),'scene_id':'s3','constraint_id':'distance3'},locks=['s1'])
    repaired=edit_run(base,tmp_path/'explicit-relaxation',relaxed)
    assert repaired['status']=='succeeded',repaired
    assert frame_files(base,'s1')==frame_files(tmp_path/'explicit-relaxation','s1')

@pytest.mark.parametrize('operation',[dict(kind='event',scene_id='s2',action='fly'),dict(kind='shot',scene_id='s2',template='free_matrix'),{**narration(),'arbitrary_path':'/tmp/x'}])
def test_untyped_or_unsupported_edits_rejected(operation):
    with pytest.raises(ValidationError):patch(operation)

def test_language_patch_preserves_server_version_and_locks(independent,tmp_path):
    value=patch(narration(),locks=['s1','s3']).model_dump()
    provider=FixtureProvider([json.dumps(value,ensure_ascii=False)])
    parsed=language_patch('只改第二幕旁白',1,['s1','s3'],independent,[],provider,tmp_path/'evidence')
    assert parsed.model_dump()==value
    assert read_json(tmp_path/'evidence/provider.json')['mode']=='synthetic/test'
    value['base_version']=999
    with pytest.raises(ValueError,match='version'):
        language_patch('不能改版本',1,['s1','s3'],independent,[],FixtureProvider([json.dumps(value)]),tmp_path/'bad-evidence')

def test_tampered_base_rejected(independent,tmp_path):
    store=story_store(tmp_path,independent);base=store.root(store.state(),1)
    (base/'narration/s1.txt').write_text('tampered')
    with pytest.raises(ValueError,match='integrity'):
        store.apply(store.jobs/'bad',1,patch=patch(narration()).model_dump())

def test_web_edit_cli_versions_download_and_dedup(independent,tmp_path):
    import uuid,time,shutil
    from fastapi.testclient import TestClient
    from src.storyboard.web import create_app
    root=tmp_path/'web';parent_id=uuid.uuid4().hex;parent=root/'jobs'/parent_id
    (parent/'pipeline').mkdir(parents=True)
    shutil.copytree(independent,parent/'pipeline/attempt_00/compiled')
    atomic_json(parent/'pipeline/run.json',dict(status='succeeded',accepted_attempt='attempt_00'))
    atomic_json(parent/'job.json',dict(id=parent_id,kind='story',status='succeeded',stage='complete',created_at=0,request={}))
    atomic_json(parent/'catalog.json',[dict(instance_id='a',display_name='人物'),dict(instance_id='b',display_name='熊')])
    app=create_app(dict(root=str(root),assets=[]))
    with TestClient(app) as client:
        assert client.get(f'/api/p7/stories/{parent_id}/versions').json()['current_version']==1
        assert client.post(f'/api/p7/stories/{parent_id}/edits',json=dict(request_id=str(uuid.uuid4()),base_version=1,text='改旁白')).status_code==400
        payload=dict(request_id=str(uuid.uuid4()),base_version=1,patch=patch(narration()).model_dump())
        response=client.post(f'/api/p7/stories/{parent_id}/edits',json=payload)
        assert response.status_code==200,response.text
        jid=response.json()['id']
        for _ in range(500):
            job=client.get('/api/p6/jobs/'+jid).json()
            if job['status']!='running' and job['status']!='queued':break
            time.sleep(.02)
        assert job['status']=='succeeded',job
        assert len(job['frames'])==3
        assert client.post(f'/api/p7/stories/{parent_id}/edits',json=payload).json()['id']==jid
        assert client.get(f'/api/p7/stories/{parent_id}/versions').json()['current_version']==2
        assert client.post(f'/api/p7/stories/{parent_id}/edits',json={**payload,'request_id':str(uuid.uuid4())}).status_code==409
        download=client.get('/api/p6/jobs/'+jid+'/download');assert download.status_code==200
        assert len(download.content)>0
