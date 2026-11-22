"""P6 HTTP boundary: persistent jobs and registered artifacts only."""
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID
import json
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,ConfigDict,Field
from starlette.background import BackgroundTask
from .web_jobs import JobStore

class Request(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_id:UUID
class StoryRequest(Request):
    asset_keys:list[str]=Field(min_length=2,max_length=3)
    story:str=Field(min_length=1,max_length=6000)
class SegmentRequest(Request):
    upload_id:UUID
    points:list[tuple[float,float,int]]=Field(min_length=1,max_length=32)
class AssetRequest(Request):
    segment_id:UUID
    name:str=Field(min_length=1,max_length=50)
    name_confirmed:bool
    method:str


def create_app(settings,provider_factory=None,executors=None,start=True):
    store=JobStore(settings['root'],settings,provider_factory,executors,start)
    @asynccontextmanager
    async def lifespan(app):
        yield
        store.close()
    app=FastAPI(title='FYP Storyboard P6',lifespan=lifespan);app.state.jobs=store
    def error(exc):
        if isinstance(exc,FileExistsError):return HTTPException(409,str(exc))
        if isinstance(exc,(FileNotFoundError,KeyError)):return HTTPException(404,'记录不存在或产物丢失')
        return HTTPException(400,str(exc))
    @app.get('/')
    def index():return FileResponse(Path(__file__).resolve().parents[2]/'web/storyboard/index.html')
    @app.get('/api/p6/config')
    def config():return dict(methods=list(settings.get('reconstruction',{}).get('methods',{})),provisional=True)
    @app.get('/api/p6/assets')
    def assets():return [dict(key=a['key'],name=a['name'],asset_id=a['asset_id'],instance_id=a['instance_id'],orientation=a.get('orientation','proposed'),source_job=a.get('source_job')) for a in store.assets()]
    @app.post('/api/p6/uploads')
    async def upload(file:UploadFile=File(...)):
        try:return store.upload(await file.read(20*1024*1024+1))
        except Exception as exc:raise error(exc)
    @app.get('/api/p6/uploads/{upload_id}/image')
    def upload_image(upload_id:UUID):
        path=store.root/'uploads'/upload_id.hex/'image.png'
        if not path.is_file():raise HTTPException(404)
        from src.scene.storage import controlled,read_json,sha256
        try:
            safe=controlled(store.root/'uploads'/upload_id.hex,'image.png')
            if path.is_symlink() or sha256(safe)!=read_json(path.parent/'upload.json')['sha256']:raise ValueError('上传图片已改变')
        except (OSError,ValueError):raise HTTPException(410,'上传图片丢失或改变')
        return FileResponse(safe,media_type='image/png')
    def submit(kind,body):
        data=body.model_dump(mode='json');request_id=data.pop('request_id')
        try:
            job,created=store.submit(kind,request_id,data)
            return dict(id=job['id'],status=job['status'],deduplicated=not created)
        except Exception as exc:raise error(exc)
    @app.post('/api/p6/stories')
    def story(body:StoryRequest):return submit('story',body)
    @app.post('/api/p6/segments')
    def segment(body:SegmentRequest):return submit('segment',body)
    @app.post('/api/p6/reconstructions')
    def asset(body:AssetRequest):return submit('asset',body)
    @app.get('/api/p6/jobs')
    def history():
        rows=[]
        for path in (store.root/'jobs').glob('*/job.json'):
            job=json.loads(path.read_text());rows.append(dict(id=job['id'],kind=job['kind'],status=job['status'],created_at=job['created_at']))
        return sorted(rows,key=lambda j:j['created_at'],reverse=True)
    @app.get('/api/p6/jobs/{job_id}')
    def job(job_id:UUID):
        try:return store.collect(job_id.hex)
        except Exception as exc:raise error(exc)
    @app.get('/api/p6/jobs/{job_id}/artifacts/{key}')
    def artifact(job_id:UUID,key:str,download:bool=False):
        try:
            path=store.artifact_path(job_id.hex,key)
            return FileResponse(path,filename=path.name if download else None,media_type='image/png' if path.suffix=='.png' else 'application/octet-stream')
        except Exception as exc:raise error(exc)
    @app.get('/api/p6/jobs/{job_id}/download')
    def download(job_id:UUID):
        try:
            path=store.archive(job_id.hex)
            return FileResponse(path,filename='storyboard-'+job_id.hex+'.zip',background=BackgroundTask(path.unlink,missing_ok=True))
        except Exception as exc:raise error(exc)
    return app
