"""Real FFmpeg CPU fixtures; model executors below are contract tests only."""
import io,json,time,uuid,shutil
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from src.storyboard.video import command,probe
from src.storyboard.web import create_app
from src.scene.storage import read_json,atomic_json

@pytest.fixture
def clip(tmp_path):
    path=tmp_path/'cuts.mp4'
    command(['-f','lavfi','-i','color=red:s=160x120:r=12:d=2','-f','lavfi','-i','color=blue:s=160x120:r=12:d=2',
        '-filter_complex','[0:v][1:v]concat=n=2:v=1:a=0[out]','-map','[out]','-c:v','libx264','-pix_fmt','yuv420p','-y',path])
    return path

@pytest.fixture
def settings(tmp_path):
    return dict(root=str(tmp_path/'state'),assets=[],reconstruction=dict(common={'gray':True},methods={'triposr':{}}))

def finish(client,pid):
    for _ in range(400):
        row=client.get('/api/video/projects/'+pid).json()
        if row['status'] in {'ready','failed','interrupted'}:return row
        time.sleep(.02)
    raise AssertionError('video did not finish')

def upload(client,clip):
    with clip.open('rb') as file:r=client.post('/api/video/projects',files={'file':('cut.mp4',file,'video/mp4')})
    assert r.status_code==200,r.text
    return r.json()['id']

def analyze(client,pid,start=0,end=4):
    return client.post(f'/api/video/projects/{pid}/analyze',json=dict(request_id=str(uuid.uuid4()),start=start,end=end))

def test_real_cuts_times_edit_restore_and_media(settings,clip):
    assert probe(clip)['duration']==4
    app=create_app(settings)
    with TestClient(app) as client:
        assert client.get('/video').status_code==200
        pid=upload(client,clip);assert analyze(client,pid).status_code==200
        row=finish(client,pid);assert row['status']=='ready',row
        assert len(row['shots'])==2 and row['shots'][1]['start']==pytest.approx(2,abs=.1)
        assert len(row['frames'])==2
        assert client.get(f'/api/video/projects/{pid}/media/preview.mp4').headers['content-type']=='video/mp4'
        assert client.get(f'/api/video/projects/{pid}/media/source.video').status_code==404
        first=row['frames'][0]
        image=client.get(f"/api/video/projects/{pid}/media/{first['id']}.png")
        assert Image.open(io.BytesIO(image.content)).getpixel((20,20))[0]>200
        url=f'/api/video/projects/{pid}/frames'
        added=client.post(url,json=dict(action='add',time=3.6,label='<script>标签</script>',revision=row['revision'])).json()
        assert len(added['frames'])==3
        assert client.post(url,json=dict(action='delete',frame_id=first['id'],revision=row['revision'])).status_code==409
        replacement=client.post(url,json=dict(action='replace',frame_id=first['id'],time=2.7,revision=added['revision'])).json()
        assert first['id'] not in [f['id'] for f in replacement['frames']]
        assert replacement['archived_frames'][0]['image_sha256']==first['image_sha256']
        assert client.post(url,json=dict(action='add',time=4,revision=replacement['revision'])).status_code==400
        active=replacement['frames'][0]
        renamed=client.post(url,json=dict(action='label',frame_id=active['id'],label='演员',revision=replacement['revision'])).json()
        removed=client.post(url,json=dict(action='delete',frame_id=active['id'],revision=renamed['revision'])).json()
        assert len(removed['frames'])==2
    app=create_app(settings)
    with TestClient(app) as client:
        assert client.get('/api/video/projects/'+pid).json()['frames']==removed['frames']


def test_invalid_upload_limits_range_and_single_shot(settings,clip,tmp_path):
    settings['video_max_bytes']=100
    with TestClient(create_app(settings)) as client:
        assert client.post('/api/video/projects',files={'file':('x',b'not a video')}).status_code==400
        assert client.post('/api/video/projects',files={'file':('x',b'x'*101)}).status_code==400
        assert not list((Path(settings['root'])/'videos').glob('tmp*'))
    settings['video_max_bytes']=1000000
    with TestClient(create_app(settings)) as client:
        pid=upload(client,clip)
        assert analyze(client,pid,-1,2).status_code==422
        assert analyze(client,pid,0,5).status_code==400
        assert analyze(client,pid,1,2).status_code==200
        row=finish(client,pid);assert row['status']=='ready' and len(row['shots'])==1
        assert row['shots'][0]['start']==1 and all(1<=f['time']<2 for f in row['frames'])


def test_frame_job_binding_and_color_override(settings,clip):
    def segment(store,jid):
        root=store.path(jid);Image.new('L',(160,120),255).save(root/'mask.png');Image.new('RGBA',(160,120),'red').save(root/'foreground.png')
    app=create_app(settings,executors={'segment':segment,'asset':lambda s,j:None})
    with TestClient(app) as client:
        pid=upload(client,clip);analyze(client,pid);row=finish(client,pid);fid=row['frames'][0]['id']
        base=f'/api/video/projects/{pid}/frames/{fid}'
        activated=client.post(base+'/activate').json()
        assert client.post(base+'/activate').json()['upload_id']==activated['upload_id']
        body=dict(request_id=str(uuid.uuid4()),upload_id=activated['upload_id'],points=[[20,20,1]])
        sid=client.post(base+'/segments',json=body).json()['id']
        for _ in range(100):
            if client.get('/api/p6/jobs/'+sid).json()['status']=='succeeded':break
            time.sleep(.02)
        payload=dict(request_id=str(uuid.uuid4()),segment_id=sid,name='测试演员',name_confirmed=True,method='triposr')
        result=client.post(base+'/reconstructions',json=payload);assert result.status_code==200,result.text
        aid=result.json()['id'];job=app.state.jobs.job(aid)
        assert job['reconstruction_options']['gray'] is False
        assert job['request']['video_source']['frame_id']==fid
        assert client.post(base+'/reconstructions',json=payload).json()['deduplicated'] is True
        other=row['frames'][1]['id']
        assert client.post(f'/api/video/projects/{pid}/frames/{other}/reconstructions',json={**payload,'request_id':str(uuid.uuid4())}).status_code==400
        unconfirmed={**payload,'request_id':str(uuid.uuid4()),'name_confirmed':False}
        assert client.post(base+'/reconstructions',json=unconfirmed).status_code==400
        preview=client.post(base+'/reconstructions',json={**unconfirmed,'request_id':str(uuid.uuid4()),'register_asset':False})
        assert preview.status_code==200,preview.text
        assert app.state.jobs.job(preview.json()['id'])['request']['register_asset'] is False
        linked=client.get('/api/video/projects/'+pid).json()['frames'][0]['links']
        assert [x['kind'] for x in linked]==['segment','asset','asset']
        assert all('status' in x for x in linked)


def test_interrupted_job_can_retry_and_changed_source_fails(settings,clip):
    app=create_app(settings,start=False)
    with TestClient(app) as client:
        pid=upload(client,clip);assert analyze(client,pid).status_code==200
    app=create_app(settings)
    with TestClient(app) as client:
        assert client.get('/api/video/projects/'+pid).json()['status']=='interrupted'
        source=app.state.jobs.videos.path(pid)/'source.video'
        with source.open('ab') as file:file.write(b'changed')
        assert analyze(client,pid).status_code==200
        row=finish(client,pid)
        assert row['status']=='failed' and '原始视频已改变' in row['error']
        assert row['frames']==[]
