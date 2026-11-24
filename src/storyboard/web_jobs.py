"""P6 persisted single-worker queue. No model imports at module load."""
from pathlib import Path
from dataclasses import asdict
import fcntl,hashlib,json,queue,re,shutil,threading,time,uuid,zipfile,subprocess,sys
import numpy as np
from PIL import Image
from src.scene.models import SceneDocument,Instance,Transform
from src.scene.storage import atomic_json,read_json,sha256,SceneRepository,controlled
from .assets import freeze_assets
from .spatial_service import LayoutAssets

TERMINAL={'succeeded','failed','needs_clarification','interrupted','conflict','version_conflict','invalid_patch'}

def uid(value):return uuid.UUID(str(value)).hex

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def atomic_png(image,path):
    path=Path(path);temporary=path.with_name('.pending-'+uuid.uuid4().hex+'.png')
    try:image.save(temporary,format='PNG');temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


class JobStore:
    def __init__(self,root,settings,provider_factory=None,executors=None,start=True):
        self.root=Path(root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        for d in ['jobs','uploads','library','downloads']:(self.root/d).mkdir(exist_ok=True)
        self.settings=settings;self.provider_factory=provider_factory;self.executors=executors or {}
        self.lock=threading.RLock();self.pending=queue.Queue();self.stopping=threading.Event();self.child=None
        self.process_lock=(self.root/'.worker.lock').open('a');fcntl.flock(self.process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for p in (self.root/'jobs').glob('*/job.json'):
            job=read_json(p)
            if job['status'] not in TERMINAL:
                job.update(status='interrupted',stage='interrupted',error='服务重启中断了任务；请用新请求重试。');atomic_json(p,job)
        self.assets_path=self.root/'assets.json'
        if not self.assets_path.exists():atomic_json(self.assets_path,settings.get('assets',[]))
        from .video import VideoProjects
        self.videos=VideoProjects(self)
        self.thread=threading.Thread(target=self._worker,daemon=True)
        if start:self.thread.start()

    def close(self):
        self.stopping.set();self.pending.put(None)
        if self.child is not None and self.child.poll() is None:self.child.terminate()
        if self.thread.is_alive():self.thread.join(timeout=1)
        if not self.thread.is_alive():self.process_lock.close()

    def path(self,job_id):return self.root/'jobs'/uid(job_id)
    def job(self,job_id):return read_json(self.path(job_id)/'job.json')
    def update(self,job_id,**values):
        with self.lock:
            job=self.job(job_id);job.update(values);atomic_json(self.path(job_id)/'job.json',job)
        return job

    def assets(self):return read_json(self.assets_path)

    def upload(self,data):
        from io import BytesIO
        if len(data)>20*1024*1024:raise ValueError('图片最大20MB')
        with Image.open(BytesIO(data)) as source:
            if source.width*source.height>20000000:raise ValueError('图片像素过多')
            image=source.convert('RGB');image.load()
        key=uuid.uuid4().hex;path=self.root/'uploads'/key;path.mkdir()
        image.save(path/'image.png')
        atomic_json(path/'upload.json',dict(id=key,sha256=sha256(path/'image.png'),size=list(image.size)))
        return read_json(path/'upload.json')

    def submit(self,kind,request_id,payload):
        if kind not in {'story','segment','asset','edit','video'}:raise ValueError('unknown job kind')
        job_id=uid(request_id);digest=fingerprint(dict(kind=kind,payload=payload));root=self.path(job_id)
        with self.lock:
            if root.exists():
                job=self.job(job_id)
                if job['request_hash']!=digest:raise FileExistsError('request_id 已用于不同输入')
                return job,False
            root.mkdir();job=dict(id=job_id,kind=kind,status='freezing',stage='freezing',request_hash=digest,request=payload,created_at=time.time())
            atomic_json(root/'job.json',job)
            try:
                atomic_json(root/'settings.json',self.settings)
                if kind=='story':self.freeze_story(root,payload)
                elif kind=='edit':
                    parent=self.job(payload['story_job_id'])
                    if parent['kind']!='story' or parent['status']!='succeeded':raise ValueError('只能修改已完成的故事')
                    from .edits import EditStore
                    state=EditStore(self.path(parent['id'])).state()
                    if payload['base_version']!=state['current_version']:raise FileExistsError('版本已过期，请刷新后重试')
                    edit={k:v for k,v in payload.items() if k!='story_job_id' and v is not None}
                    atomic_json(root/'edit-request.json',edit)
                elif kind=='video':
                    self.videos.get(payload['project_id'])
                elif kind=='segment':
                    upload=self.root/'uploads'/uid(payload['upload_id']);meta=read_json(upload/'upload.json')
                    if sha256(upload/'image.png')!=meta['sha256']:raise ValueError('上传图片已改变')
                    points=payload['points']
                    if not points or len(points)>32 or not any(p[2]==1 for p in points):raise ValueError('请至少点击一个主体正点')
                    for x,y,label in points:
                        if not np.isfinite([x,y]).all() or not 0<=x<meta['size'][0] or not 0<=y<meta['size'][1] or label not in (0,1):raise ValueError('无效选点')
                    shutil.copyfile(upload/'image.png',root/'image.png')
                else:
                    previous=self.job(payload['segment_id']);prior=self.path(payload['segment_id'])
                    if previous['kind']!='segment' or previous['status']!='succeeded':raise ValueError('请先完成并确认分割')
                    if not payload['name'].strip() or len(payload['name'])>50:raise ValueError('请填写主体名称')
                    if payload.get('register_asset',True) and payload.get('name_confirmed') is not True:raise ValueError('请确认资产名称')
                    if payload['method'] not in self.settings['reconstruction']['methods']:raise ValueError('重建方法未配置')
                    for name in ['image.png','mask.png']:shutil.copyfile(self.artifact_path(previous['id'],self.register(previous['id'],name)),root/name)
                    job['reconstruction_options']={**self.settings['reconstruction'].get('common',{}),**self.settings['reconstruction']['methods'][payload['method']]}
                    if payload.get('video_color'):job['reconstruction_options']['gray']=False
                job.update(status='queued',stage='queued');atomic_json(root/'job.json',job);self.pending.put(job_id)
            except Exception as exc:
                job.update(status='failed',stage='input',error=str(exc));atomic_json(root/'job.json',job);raise
            return job,True

    def freeze_story(self,root,payload):
        keys=payload['asset_keys']
        if not 2<=len(keys)<=3 or len(set(keys))!=len(keys):raise ValueError('请选择2–3个不同资产')
        if not payload['story'].strip() or len(payload['story'])>6000:raise ValueError('故事应为1–6000字')
        mapping={a['key']:a for a in self.assets()};selected=[mapping[k] for k in keys]
        if len({a['name'] for a in selected})!=len(selected):raise ValueError('选择的资产名称需不同')
        if len({a['asset_id'] for a in selected})!=len(selected):raise ValueError('请选择不同的资产记录')
        frozen=root/'frozen';frozen.mkdir();catalog=[];instances=[]
        for a in selected:freeze_assets(a['library'],[a['asset_id']],frozen)
        repo=LayoutAssets(frozen);records=[repo.get(a['asset_id']) for a in selected]
        positions={};cursor=None
        for n in reversed(range(len(selected))):
            bounds=np.array(records[n].bounds)
            x=0. if cursor is None else cursor-.4-float(bounds[1,0])
            positions[n]=[x,-float(bounds[0,1]),0.];cursor=x+float(bounds[0,0])
        for n,(a,record) in enumerate(zip(selected,records)):
            instances.append(Instance(a['instance_id'],a['asset_id'],record.sha256,Transform(translation_xyz=positions[n])))
            catalog.append(dict(instance_id=a['instance_id'],asset_id=a['asset_id'],display_name=a['name'],name_confirmed=True,confirmation_source='user'))
        scene=SceneDocument('initial',instances,[]);SceneRepository(repo).save(scene)
        atomic_json(root/'catalog.json',catalog);atomic_json(root/'selected-assets.json',selected)
        source=Path(self.settings['story_config']);shutil.copyfile(source,root/'story_config.yaml')

    def _worker(self):
        while not self.stopping.is_set():
            job_id=self.pending.get()
            if job_id is None:return
            try:
                self.update(job_id,status='running',stage='starting',started_at=time.time())
                kind=self.job(job_id)['kind']
                if kind in self.executors:self.executors[kind](self,job_id)
                else:getattr(self,'execute_'+kind)(job_id)
                if self.job(job_id)['status']=='running':self.update(job_id,status='succeeded',stage='complete',finished_at=time.time())
            except Exception as exc:self.update(job_id,status='failed',stage='failed',error=type(exc).__name__+': '+str(exc),finished_at=time.time())
            finally:
                try:self.collect(job_id)
                except Exception:pass

    def execute_video(self,job_id):
        self.videos.analyze(job_id)

    def execute_segment(self,job_id):
        import gc,torch
        from src.segment import build_sam2_predictor,select_multimask_index
        from src.subjects import Subject,segment_subject
        root=self.path(job_id);job=self.job(job_id);settings=read_json(root/'settings.json')
        self.update(job_id,stage='segmentation');image=Image.open(root/'image.png').convert('RGB')
        points=job['request']['points'];predictor=build_sam2_predictor(settings['sam'])
        masks,scores=segment_subject(predictor,np.array(image),Subject('selected',points=[[p[0],p[1]] for p in points],labels=[p[2] for p in points]))
        mask=masks[select_multimask_index(masks,scores,None)]
        if not mask.any() or mask.all():raise ValueError('SAM2没有分出有效主体')
        atomic_png(Image.fromarray(mask.astype(np.uint8)*255),root/'mask.png');image.putalpha(Image.fromarray(mask.astype(np.uint8)*255));atomic_png(image,root/'foreground.png')
        del predictor;gc.collect();torch.cuda.empty_cache()

    def execute_asset(self,job_id):
        import gc,torch
        from src.reconstruction_methods import reconstruct_asset
        from .assets import import_asset
        root=self.path(job_id);job=self.job(job_id);p=job['request'];self.update(job_id,stage='reconstruction')
        run=reconstruct_asset(p['method'],root/'image.png',root/'mask.png',root/'reconstruction',job['reconstruction_options'])
        if not p.get('register_asset',True):
            self.update(job_id,registered_asset=False);gc.collect();torch.cuda.empty_cache();return
        self.update(job_id,stage='asset_registration')
        # Axis proposals are explicitly provisional; names alone are user-confirmed.
        axes=dict(source_up=[0,0,1],source_forward=[0,-1,0]) if p['method']=='triposr' else dict(source_up=[0,1,0],source_forward=[0,0,1])
        side=import_asset(self.root/'library',root/'reconstruction/pointcloud.ply',dict(display_name=p['name'],display_name_confirmed=True,source_image=str(root/'image.png'),producer=p['method'],target_height=1.6,orientation_basis='backend_axis_proposal_not_user_confirmed',reconstruction=run,**axes))
        self.update(job_id,stage='asset_preview')
        from .assets import CanonicalAssetRepository
        from .real_scene import save_render
        asset_repo=CanonicalAssetRepository(self.root/'library');pcloud,colors=asset_repo.points(asset_repo.get(side['asset_id']))
        save_render(root,'asset_preview',pcloud,colors,np.ones(len(pcloud),dtype=np.int32))
        item=dict(key=uuid.uuid4().hex,asset_id=side['asset_id'],instance_id='object_'+side['asset_id'][:12],name=p['name'],library=str(self.root/'library'),orientation='proposed',source_job=job_id)
        with self.lock:
            entries=self.assets();entries.append(item);atomic_json(self.assets_path,entries)
        self.update(job_id,asset_key=item['key']);gc.collect();torch.cuda.empty_cache()

    def execute_story(self,job_id):
        root=self.path(job_id);job=self.job(job_id);settings=read_json(root/'settings.json')
        repository=Path(__file__).resolve().parents[2];(root/'story.txt').write_text(job['request']['story'])
        command=[sys.executable,str(repository/'scripts/storyboard.py'),'understand','--story-file',str(root/'story.txt'),'--catalog',str(root/'catalog.json'),'--asset-root',str(root/'frozen'),'--scene',str(root/'frozen/scene.json'),'--out',str(root/'pipeline'),'--config',str(root/'story_config.yaml'),'--call-timeout','300','--allow-provisional']
        if self.provider_factory:
            provider=self.provider_factory()
            if provider.metadata().get('mode')!='synthetic/test':raise ValueError('custom web test providers must be synthetic fixtures')
            atomic_json(root/'fixture-responses.json',provider.responses);command+=['--fixture-responses',str(root/'fixture-responses.json')]
        else:command+=['--api-model',settings.get('api_model','deepseek-v4-pro')]
        self.update(job_id,stage='understanding')
        # P5's SIGALRM deadline must run in a process main thread, not the HTTP worker thread.
        with (root/'pipeline.log').open('w') as log:
            self.child=subprocess.Popen(command,cwd=repository,stdout=log,stderr=subprocess.STDOUT)
            try:code=self.child.wait(timeout=settings.get('story_timeout_seconds',1800))
            except subprocess.TimeoutExpired:
                self.child.terminate()
                try:self.child.wait(timeout=5)
                except subprocess.TimeoutExpired:self.child.kill();self.child.wait()
                raise RuntimeError('故事任务超过总时限，已有产物已保留')
            finally:self.child=None
        if code not in (0,1,2) or not (root/'pipeline/run.json').is_file():raise RuntimeError('故事执行器中断，请查看任务日志')
        run=read_json(root/'pipeline/run.json')
        if run['status']=='succeeded':
            self.update(job_id,stage='validation')
            with (root/'validation.log').open('w') as log:
                result=subprocess.run([sys.executable,str(repository/'scripts/storyboard.py'),'validate','--run',str(root/'pipeline')],cwd=repository,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            if result.returncode:raise ValueError('独立验收未通过')
        self.update(job_id,status=run['status'],stage='complete',provider=run.get('provider'),question=run.get('question'),error=run.get('error'),finished_at=time.time())

    def execute_edit(self,job_id):
        root=self.path(job_id);job=self.job(job_id);settings=read_json(root/'settings.json')
        repository=Path(__file__).resolve().parents[2];pipeline=root/'pipeline';pipeline.mkdir()
        compiled=pipeline/'attempt_00/compiled';compiled.parent.mkdir()
        self.update(job_id,stage='editing')
        command=[sys.executable,str(repository/'scripts/edit_storyboard.py'),'--story-root',str(self.path(job['request']['story_job_id'])),
            '--request',str(root/'edit-request.json'),'--out',str(compiled),'--api-model',settings.get('api_model','deepseek-v4-pro')]
        with (root/'pipeline.log').open('w') as log:
            self.child=subprocess.Popen(command,cwd=repository,stdout=log,stderr=subprocess.STDOUT)
            try:self.child.wait(timeout=600)
            except subprocess.TimeoutExpired:
                self.child.terminate();self.child.wait(timeout=5);raise RuntimeError('修改超时，旧版本保持可用')
            finally:self.child=None
        result=read_json(compiled.parent/'edit-result.json')
        artifacts=[dict(path=str(p.relative_to(pipeline)),sha256=sha256(p)) for p in compiled.parent.rglob('*')
            if p.is_file() and not p.is_relative_to(compiled) and p.suffix in {'.json','.txt'}]
        atomic_json(pipeline/'run.json',dict(pipeline='p7_edit_job',status=result['status'],accepted_attempt='attempt_00',attempts=[dict(directory='attempt_00')],artifacts=artifacts))
        self.update(job_id,status=result['status'],stage='complete',edit_result=result,
            error=result.get('conflict_summary') or result.get('error') or ('修改与锁定幕或跨幕事件冲突；旧版本未改变。' if result['status']!='succeeded' else None),finished_at=time.time())

    def register(self,job_id,relative):
        root=self.path(job_id);path=controlled(root,relative)
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p!=self.root and p.is_relative_to(root)):raise ValueError('symlink artifact')
        if not path.is_file():raise FileNotFoundError(relative)
        if path.suffix=='.png':
            with Image.open(path) as im:im.verify()
        key=hashlib.sha256(relative.encode()).hexdigest()[:24]
        with self.lock:
            index=read_json(root/'artifacts.json') if (root/'artifacts.json').exists() else {}
            if key not in index:
                stat=path.stat();index[key]=dict(path=relative,sha256=sha256(path),name=path.name,size=stat.st_size,mtime_ns=stat.st_mtime_ns);atomic_json(root/'artifacts.json',index)
        return key

    def artifact_path(self,job_id,key):
        if not re.fullmatch('[0-9a-f]{24}',key):raise ValueError('unknown artifact')
        root=self.path(job_id);entry=read_json(root/'artifacts.json')[key];raw=root/entry['path']
        if raw.is_symlink() or any(p.is_symlink() for p in raw.parents if p.is_relative_to(root)):raise ValueError('symlink artifact')
        path=controlled(root,entry['path'])
        if not path.is_file() or sha256(path)!=entry['sha256']:raise FileNotFoundError('产物丢失或已改变')
        return path

    def collect(self,job_id):
        root=self.path(job_id);job=self.job(job_id)
        reconstruction=root/'reconstruction/run.json'
        ready=reconstruction.is_file() and read_json(reconstruction).get('status')=='succeeded'
        for name in ['image.png','mask.png','foreground.png','reconstruction/pointcloud.ply','reconstruction/mesh.ply','reconstruction/viewer.html','reconstruction/run.json','renders/asset_preview.png']:
            if name.startswith('reconstruction/') and not ready:continue
            if (root/name).is_file():
                try:self.register(job_id,name)
                except (OSError,ValueError):
                    if job['status'] in TERMINAL:self.update(job_id,status='failed',error='预览产物无效或不完整');job=self.job(job_id)
        frames=[];pipeline=root/'pipeline';saved=read_json(pipeline/'run.json') if (pipeline/'run.json').exists() else {}
        attempts=saved.get('attempts',[])
        accepted=saved.get('accepted_attempt') or next((a['directory'] for a in reversed(attempts) if re.fullmatch('attempt_[0-9]{2}',a['directory']) and (pipeline/a['directory']/'compiled/story_plan.json').is_file()),None)
        if accepted and re.fullmatch('attempt_[0-9]{2}',accepted):
            compiled=pipeline/accepted/'compiled';plan=read_json(compiled/'story_plan.json') if (compiled/'story_plan.json').exists() else {}
            validation=read_json(compiled/'validation.json') if (compiled/'validation.json').exists() else {}
            reports={v['scene_id']:v for v in validation.get('scenes',[])}
            for scene in plan.get('scenes',[]):
                sid=scene['scene_id']
                if not re.fullmatch('[A-Za-z0-9_-]+',sid):continue
                relative=str((compiled/'renders'/(sid+'.png')).relative_to(root));key=None
                if (root/relative).is_file():
                    try:key=self.register(job_id,relative)
                    except (OSError,ValueError):pass
                frames.append(dict(scene_id=sid,narration=scene['narration']['text'],image=key,validation=reports.get(sid,{})))
            if (compiled/'run.json').exists() and job['status']=='running' and job['stage']!='validation':job['stage']='repair' if attempts and attempts[-1]['directory']!=accepted else read_json(compiled/'run.json').get('stage','understanding')
        if job['status'] in TERMINAL:
            # Only explicit output manifests are eligible, never arbitrary paths/globs.
            manifests=[pipeline/'run.json']+list(pipeline.glob('attempt_[0-9][0-9]/compiled/run.json'))
            for manifest in manifests:
                if not manifest.is_file():continue
                for item in read_json(manifest).get('artifacts',[]):
                    relative=str((manifest.parent/item['path']).relative_to(root))
                    if Path(relative).suffix not in {'.json','.png','.txt','.ply'} or '/assets/' in '/'+relative:continue
                    candidate=controlled(root,relative)
                    if candidate.is_file():self.register(job_id,relative)
            if (pipeline/'revalidation.json').exists():self.register(job_id,'pipeline/revalidation.json')
        index=read_json(root/'artifacts.json') if (root/'artifacts.json').exists() else {}
        missing=[]
        for key in index:
            try:
                entry=index[key];raw=root/entry['path'];path=controlled(root,entry['path']);stat=path.stat()
                if raw.is_symlink() or any(p.is_symlink() for p in raw.parents if p.is_relative_to(root)):raise ValueError('symlink')
                if stat.st_size!=entry.get('size',stat.st_size) or stat.st_mtime_ns!=entry.get('mtime_ns',stat.st_mtime_ns):raise ValueError('artifact changed')
            except (OSError,ValueError,KeyError):missing.append(key)
        if missing:
            self.update(job_id,status='failed',error='已登记产物丢失或改变',integrity_errors=missing);job=self.job(job_id)
        job.update(frames=frames,artifacts=[dict(id=k,name=v['name']) for k,v in index.items()],integrity_errors=missing)
        return job

    def archive(self,job_id):
        self.collect(job_id);root=self.path(job_id);index=read_json(root/'artifacts.json');path=self.root/'downloads'/(uid(job_id)+'-'+uuid.uuid4().hex+'.zip')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for key,entry in index.items():z.write(self.artifact_path(job_id,key),entry['path'])
            z.writestr('web-job.json',json.dumps(self.job(job_id),ensure_ascii=False,indent=2))
        return path
