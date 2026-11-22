"""P6 HTTP/queue contracts; fake executors are explicitly tests, not model evidence."""
from dataclasses import asdict
from itertools import product
from io import BytesIO
from pathlib import Path
import json,time,uuid,zipfile,threading
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.scene.storage import AssetRepository,atomic_json,read_json,sha256
from src.scene.models import Y_UP
from src.pointcloud_io import write_ply
from src.storyboard.web import create_app

@pytest.fixture
def settings(tmp_path):
    library=tmp_path/'source';repo=AssetRepository(library);assets=[]
    for n,color in enumerate(([255,0,0],[0,0,255])):
        p=tmp_path/f'{n}.ply';points=np.array(list(product([-.5,.5],[0,1],[-.5,.5])))
        write_ply(points,np.tile(color,(8,1)),p)
        r=repo.import_asset(p,role='subject',representation='point_cloud',source_kind='synthetic',producer='p6_test',coordinate_frame=Y_UP)
        assets.append(dict(key=str(n),name=['人物','熊'][n],asset_id=r.asset_id,instance_id=['portrait_01','bear_01'][n],library=str(library),orientation='proposed'))
    config=tmp_path/'story.yaml';config.write_text('schema_version: 1\n')
    return dict(root=str(tmp_path/'web'),assets=assets,story_config=str(config),reconstruction=dict(common={},methods={'hunyuan':{}}),sam={})

def body(settings,request_id=None,story='三幕保持'):
    return dict(request_id=request_id or str(uuid.uuid4()),asset_keys=[a['key'] for a in settings['assets']],story=story)

def wait(client,job_id):
    for _ in range(100):
        row=client.get('/api/p6/jobs/'+job_id).json()
        if row['status'] in ('succeeded','failed','needs_clarification','interrupted'):return row
        time.sleep(.01)
    raise AssertionError('job did not finish')

def fake_story(store,job_id,partial=False):
    root=store.path(job_id);compiled=root/'pipeline/attempt_00/compiled';compiled.mkdir(parents=True)
    (compiled/'renders').mkdir();scenes=[];validation=[]
    color=(int(job_id[:2],16),50,100)
    for i in range(1,4):
        sid='scene_'+str(i);scenes.append(dict(scene_id=sid,narration=dict(text='<script>narration</script> '+sid)))
        if i<3 or not partial:Image.new('RGB',(16,16),color).save(compiled/'renders'/(sid+'.png'))
        validation.append(dict(scene_id=sid,passed=not partial or i<3))
    atomic_json(compiled/'story_plan.json',dict(scenes=scenes));atomic_json(compiled/'validation.json',dict(scenes=validation))
    atomic_json(compiled/'run.json',dict(stage='render',artifacts=[]));atomic_json(root/'pipeline/run.json',dict(attempts=[dict(directory='attempt_00')],artifacts=[]))
    if partial:raise RuntimeError('scene_3 deliberate failure')


def test_deduplication_freeze_and_conflicting_request_id(settings):
    app=create_app(settings,start=False)
    with TestClient(app) as client:
        request=body(settings);first=client.post('/api/p6/stories',json=request);assert first.status_code==200
        job_id=first.json()['id'];root=app.state.jobs.path(job_id)
        assert client.post('/api/p6/stories',json=request).json()['deduplicated']
        assert app.state.jobs.pending.qsize()==1
        assert client.post('/api/p6/stories',json={**request,'story':'不同请求'}).status_code==409
        scene=read_json(root/'frozen/scene.json');assert len(scene['instances'])==2
        asset=settings['assets'][0];source=Path(asset['library'])/'assets'/asset['asset_id']/'data.ply'
        saved=sha256(root/'frozen/assets'/asset['asset_id']/'data.ply');source.write_text('changed')
        assert sha256(root/'frozen/assets'/asset['asset_id']/'data.ply')==saved
        settings['assets'][0]['name']='changed name';assert read_json(root/'catalog.json')[0]['display_name']=='人物'


def test_two_runs_refresh_registered_download_and_no_cross_images(settings):
    app=create_app(settings,executors={'story':fake_story})
    with TestClient(app) as client:
        ids=[client.post('/api/p6/stories',json=body(settings)).json()['id'] for _ in range(2)]
        rows=[wait(client,j) for j in ids];assert all(r['status']=='succeeded' for r in rows)
        assert len(client.get('/api/p6/jobs').json())==2
        for job_id,row in zip(ids,rows):
            assert len(row['frames'])==3 and all(f['image'] for f in row['frames'])
            refreshed=client.get('/api/p6/jobs/'+job_id).json();assert refreshed['frames']==row['frames']
            png=client.get(f"/api/p6/jobs/{job_id}/artifacts/{row['frames'][0]['image']}")
            assert Image.open(BytesIO(png.content)).getpixel((0,0))[0]==int(job_id[:2],16)
            archive=zipfile.ZipFile(BytesIO(client.get(f'/api/p6/jobs/{job_id}/download').content))
            assert len([n for n in archive.namelist() if n.endswith('.png')])==3
            assert all('assets/' not in n for n in archive.namelist())
        # Same relative frame key is scoped to its job, never a global image URL.
        assert rows[0]['frames'][0]['image']==rows[1]['frames'][0]['image']


def test_partial_third_scene_failure_and_narration_are_data(settings):
    app=create_app(settings,executors={'story':lambda s,j:fake_story(s,j,True)})
    with TestClient(app) as client:
        job_id=client.post('/api/p6/stories',json=body(settings,story='<img src=x onerror=alert(1)>')).json()['id'];row=wait(client,job_id)
        assert row['status']=='failed' and row['frames'][2]['image'] is None
        assert len(row['artifacts'])==2
        assert '<script>' in row['frames'][0]['narration']
        page=client.get('/').text;assert 'caption.textContent=frame.narration' in page and 'innerHTML' not in page


def test_missing_and_tampered_artifact_fail_instead_of_success(settings):
    app=create_app(settings,executors={'story':fake_story})
    with TestClient(app) as client:
        job_id=client.post('/api/p6/stories',json=body(settings)).json()['id'];row=wait(client,job_id);key=row['frames'][0]['image']
        app.state.jobs.artifact_path(job_id,key).unlink()
        assert client.get(f'/api/p6/jobs/{job_id}/artifacts/{key}').status_code==404
        assert client.get('/api/p6/jobs/'+job_id).json()['status']=='failed'


def test_path_escape_and_unregistered_file_are_rejected(settings,tmp_path):
    app=create_app(settings,executors={'story':fake_story})
    with TestClient(app) as client:
        job_id=client.post('/api/p6/stories',json=body(settings)).json()['id'];wait(client,job_id)
        root=app.state.jobs.path(job_id);(root/'private.txt').write_text('not registered')
        for path in ['private.txt','..%2F..%2Fetc%2Fpasswd','0'*24]:assert client.get(f'/api/p6/jobs/{job_id}/artifacts/{path}').status_code>=400
        key=app.state.jobs.register(job_id,'private.txt');outside=tmp_path/'outside.txt';outside.write_text('outside')
        (root/'private.txt').unlink();(root/'private.txt').symlink_to(outside)
        assert client.get(f'/api/p6/jobs/{job_id}/artifacts/{key}').status_code>=400


def test_restart_interrupts_old_jobs_and_duplicate_does_not_rerun(settings):
    request=body(settings);app=create_app(settings,start=False)
    with TestClient(app) as client:job_id=client.post('/api/p6/stories',json=request).json()['id'];app.state.jobs.update(job_id,status='running')
    restarted=create_app(settings,start=False)
    with TestClient(restarted) as client:
        assert client.get('/api/p6/jobs/'+job_id).json()['status']=='interrupted'
        assert client.post('/api/p6/stories',json=request).json()['deduplicated']
        assert restarted.state.jobs.pending.qsize()==0


def test_upload_selection_validation_and_safe_filename(settings):
    app=create_app(settings,start=False)
    with TestClient(app) as client:
        image=BytesIO();Image.new('RGB',(20,20),'white').save(image,format='PNG')
        upload=client.post('/api/p6/uploads',files={'file':('../../secret.png',image.getvalue(),'image/png')}).json()
        assert client.get('/api/p6/uploads/'+upload['id']+'/image').status_code==200
        invalid=dict(request_id=str(uuid.uuid4()),upload_id=upload['id'],points=[[30,30,1]])
        assert client.post('/api/p6/segments',json=invalid).status_code==400
        valid={**invalid,'request_id':str(uuid.uuid4()),'points':[[10,10,1]]}
        job_id=client.post('/api/p6/segments',json=valid).json()['id']
        assert (app.state.jobs.path(job_id)/'image.png').is_file()
        assert client.post('/api/p6/stories',json={**body(settings),'asset_root':'/etc'}).status_code==422


def test_single_worker_serial_execution(settings):
    active=0;peak=0;lock=threading.Lock()
    def execute(store,job):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        time.sleep(.03)
        with lock:active-=1
    app=create_app(settings,executors={'story':execute})
    with TestClient(app) as client:
        ids=[client.post('/api/p6/stories',json=body(settings)).json()['id'] for _ in range(3)]
        for j in ids:assert wait(client,j)['status']=='succeeded'
        assert peak==1


def test_real_p5_cli_boundary_uses_main_thread_and_labels_fixture(settings):
    from src.storyboard.llm_provider import FixtureProvider
    response=json.dumps(dict(status='needs_clarification',question='需要选择支持的动作',reason_code='unsupported_action'))
    app=create_app(settings,provider_factory=lambda:FixtureProvider([response]))
    with TestClient(app) as client:
        job_id=client.post('/api/p6/stories',json=body(settings)).json()['id']
        for _ in range(500):
            row=client.get('/api/p6/jobs/'+job_id).json()
            if row['status'] in ('succeeded','failed','needs_clarification'):break
            time.sleep(.01)
        assert row['status']=='needs_clarification',row
        assert row['provider']['mode']=='synthetic/test'
        saved=read_json(app.state.jobs.path(job_id)/'pipeline/run.json')
        assert len(saved['attempts'])==1


def test_confirmation_required_and_duplicate_asset_names_rejected(settings):
    app=create_app(settings,start=False)
    with TestClient(app) as client:
        bad=client.post('/api/p6/reconstructions',json=dict(request_id=str(uuid.uuid4()),segment_id=str(uuid.uuid4()),name='x',name_confirmed=False,method='hunyuan'))
        assert bad.status_code>=400
        assert client.post('/api/p6/stories',json={**body(settings),'asset_keys':['0','0']}).status_code==400
        app.state.jobs.assets_path.write_text(json.dumps([{**a,'name':'相同名称'} for a in settings['assets']]))
        assert client.post('/api/p6/stories',json=body(settings)).status_code==400


def test_poll_during_incomplete_png_write_is_not_a_job_failure(settings):
    ready=threading.Event();release=threading.Event()
    def executor(store,job_id):
        path=store.path(job_id)/'foreground.png';path.write_bytes(b'')
        ready.set();assert release.wait(3)
        Image.new('RGBA',(10,10),(200,100,0,255)).save(path)
    app=create_app(settings,executors={'story':executor})
    with TestClient(app) as client:
        job_id=client.post('/api/p6/stories',json=body(settings)).json()['id'];assert ready.wait(2)
        response=client.get('/api/p6/jobs/'+job_id)
        assert response.status_code==200 and response.json()['status']=='running' and not response.json()['artifacts']
        release.set();assert wait(client,job_id)['status']=='succeeded'
