"""Video HTTP routes. Only UUID project media and existing job APIs are exposed."""
from pathlib import Path
import tempfile, uuid
from uuid import UUID
from fastapi import File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from src.scene.storage import sha256

class Analyze(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id:UUID
    start:float=Field(ge=0)
    end:float=Field(gt=0)
    threshold:float=.28
class FrameEdit(BaseModel):
    model_config=ConfigDict(extra='forbid')
    revision:int
    action:str
    frame_id:UUID|None=None
    time:float|None=None
    label:str|None=Field(default=None,max_length=120)
class Attribution(BaseModel):
    text:str=Field(max_length=2000)


def install_video_routes(app,store,error,settings):
    videos=store.videos
    def available():
        if not settings.get('uploads_enabled',True):raise HTTPException(409,'当前服务未启用上传与重建，请启动视频/GPU服务')
    @app.get('/video')
    def page():return FileResponse(Path(__file__).resolve().parents[2]/'web/storyboard/video.html')
    @app.get('/api/video/projects')
    def projects():return videos.list()
    @app.post('/api/video/projects')
    async def upload_video(file:UploadFile=File(...)):
        available();temporary=None
        try:
            limit=settings.get('video_max_bytes',1024*1024*1024)
            with tempfile.NamedTemporaryFile(dir=videos.root,delete=False) as output:
                temporary=Path(output.name);size=0
                while chunk:=await file.read(1024*1024):
                    size+=len(chunk)
                    if size>limit:raise ValueError('视频超过上传上限（默认1GB）')
                    output.write(chunk)
            return await run_in_threadpool(videos.ingest,temporary,file.filename)
        except Exception as exc:raise error(exc)
        finally:
            if temporary:temporary.unlink(missing_ok=True)
            await file.close()
    @app.get('/api/video/projects/{project_id}')
    def project(project_id:UUID):
        try:
            row=videos.get(project_id)
            # Job status remains live; files and timestamps remain persistent.
            for f in row['frames']:
                for link in f['links']:
                    job=store.job(link['job_id']);link.update(status=job['status'],error=job.get('error'))
            return row
        except Exception as exc:raise error(exc)
    @app.post('/api/video/projects/{project_id}/analyze')
    def analyze(project_id:UUID,body:Analyze):
        available()
        try:return videos.schedule(project_id,body.start,body.end,body.threshold,str(body.request_id))
        except Exception as exc:raise error(exc)
    @app.put('/api/video/projects/{project_id}/attribution')
    def attribution(project_id:UUID,body:Attribution):
        try:
            with videos.lock:
                row=videos.get(project_id);row['source_attribution']=body.text;videos.save(row)
            return row
        except Exception as exc:raise error(exc)
    @app.post('/api/video/projects/{project_id}/frames')
    def edit(project_id:UUID,body:FrameEdit):
        available()
        try:return videos.edit(project_id,body.action,str(body.frame_id) if body.frame_id else None,body.time,body.label,body.revision)
        except Exception as exc:raise error(exc)
    @app.get('/api/video/projects/{project_id}/media/{name}')
    def media(project_id:UUID,name:str):
        try:
            row=videos.get(project_id);root=videos.path(project_id)
            if name=='preview.mp4':
                if row['status']!='ready':raise FileNotFoundError()
                path=root/name;kind='video/mp4'
            elif name.endswith('.png'):
                frame=videos.frame(project_id,name[:-4]);path=root/(frame['id']+'.png');kind='image/png'
                if sha256(path)!=frame['image_sha256']:raise ValueError('关键帧已改变')
            else:raise FileNotFoundError()
            return FileResponse(path,media_type=kind)
        except Exception as exc:raise error(exc)
    @app.post('/api/video/projects/{project_id}/frames/{frame_id}/activate')
    def activate(project_id:UUID,frame_id:UUID):
        available()
        try:return videos.activate(project_id,frame_id)
        except Exception as exc:raise error(exc)
    from .web import SegmentRequest,AssetRequest
    @app.post('/api/video/projects/{project_id}/frames/{frame_id}/segments')
    def segment(project_id:UUID,frame_id:UUID,body:SegmentRequest):
        available()
        if settings.get('video_only'):raise HTTPException(409,'本地视频演示不执行模型，请打开 HPC GPU 网页')
        try:
            data=body.model_dump(mode='json');request=data.pop('request_id')
            return videos.link(project_id,frame_id,'segment',request,data)
        except Exception as exc:raise error(exc)
    @app.post('/api/video/projects/{project_id}/frames/{frame_id}/reconstructions')
    def reconstruct(project_id:UUID,frame_id:UUID,body:AssetRequest):
        available()
        if settings.get('video_only'):raise HTTPException(409,'本地视频演示不执行模型，请打开 HPC GPU 网页')
        try:
            if body.method!='triposr':raise ValueError('视频第一版使用 TripoSR')
            data=body.model_dump(mode='json');request=data.pop('request_id')
            return videos.link(project_id,frame_id,'asset',request,data)
        except Exception as exc:raise error(exc)
