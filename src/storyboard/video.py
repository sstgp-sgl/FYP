"""Small video projects: original timestamps, immutable frame images and job links."""
from pathlib import Path
import json, math, os, re, shutil, subprocess, threading, time, uuid
import numpy as np
from PIL import Image, ImageFilter
from src.scene.storage import atomic_json, read_json, sha256
from .web_jobs import uid


def ffmpeg():
    executable=os.environ.get('FYP_FFMPEG') or shutil.which('ffmpeg')
    if not executable:
        try:
            import imageio_ffmpeg
            executable=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as exc:
            raise ValueError('视频功能需要 FFmpeg：安装 requirements-video.txt 或设置 FYP_FFMPEG') from exc
    return executable


def command(args, timeout=300, store=None):
    process=subprocess.Popen([ffmpeg(),'-hide_banner','-nostdin',*map(str,args)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if store:store.child=process
    try:
        out,err=process.communicate(timeout=timeout)
        if process.returncode:raise ValueError('视频解码失败：'+err.decode(errors='replace')[-1400:])
        return out,err.decode(errors='replace')
    except subprocess.TimeoutExpired:
        process.kill();process.communicate();raise ValueError('视频处理超时，请缩短测试片段')
    finally:
        if store:store.child=None


def probe(path):
    # ffmpeg's header inspection exits nonzero because no output is requested.
    result=subprocess.run([ffmpeg(),'-hide_banner','-nostdin','-i',str(path)],capture_output=True,timeout=30)
    text=result.stderr.decode(errors='replace')
    duration=re.search(r'Duration: (\d+):(\d+):([\d.]+)',text)
    stream=next((line for line in text.splitlines() if 'Video:' in line),'')
    size=re.search(r'\b(\d{2,5})x(\d{2,5})\b',stream)
    fps=re.search(r'([\d.]+) fps',stream)
    if not duration or not size:raise ValueError('无法读取视频时长或画面尺寸，请使用有效的 WebM / MP4 / MOV 视频')
    h,m,s=duration.groups();seconds=int(h)*3600+int(m)*60+float(s)
    width,height=map(int,size.groups())
    if not 0<seconds<=7200 or width*height>16777216:raise ValueError('视频应在两小时以内，画面不超过1600万像素')
    return dict(duration=seconds,width=width,height=height,fps=float(fps.group(1)) if fps else None,stream=stream.strip())


class VideoProjects:
    def __init__(self,store):
        self.store=store;self.root=store.root/'videos';self.root.mkdir(exist_ok=True);self.lock=threading.RLock()

    def path(self,project_id):return self.root/uid(project_id)
    def get(self,project_id):
        row=read_json(self.path(project_id)/'project.json')
        if row.get('job_id'):
            job=self.store.job(row['job_id'])
            if job['status'] in {'failed','interrupted'} and row['status'] in {'queued','processing'}:
                row.update(status=job['status'],error=job.get('error'))
        return row
    def save(self,row):atomic_json(self.path(row['id'])/'project.json',row)
    def list(self):return sorted([self.get(p.parent.name) for p in self.root.glob('*/project.json')],key=lambda r:r['created_at'],reverse=True)
    def ingest(self,path,filename):
        info=probe(path);key=uuid.uuid4().hex;root=self.path(key);root.mkdir()
        shutil.move(str(path),root/'source.video')
        row=dict(id=key,title=Path(filename or '视频').name[:120],created_at=time.time(),status='uploaded',metadata=info,
            source_sha256=sha256(root/'source.video'),source_filename=filename,source_attribution='',shots=[],frames=[],revision=0)
        self.save(row);return row
    def schedule(self,project_id,start,end,threshold,request_id):
        with self.lock:
            row=self.get(project_id)
            if row['status'] in {'queued','processing'}:raise FileExistsError('该视频正在处理')
            if row['frames']:raise ValueError('已有关键帧；请新建视频项目选择其他范围，以保留已有关联')
            if not all(math.isfinite(v) for v in [start,end,threshold]) or not 0<=start<end<=row['metadata']['duration']+.02 or end-start>180 or not .05<=threshold<=.8:
                raise ValueError('请选择有效的起止时间，单次最长180秒，阈值0.05–0.8')
            job,_=self.store.submit('video',request_id,dict(project_id=row['id'],start=start,end=end,threshold=threshold))
            row.update(status='queued',job_id=job['id'],error=None);self.save(row);return row
    def extract(self,project_id,timestamp):
        root=self.path(project_id);row=self.get(project_id)
        if not math.isfinite(timestamp) or not row['range']['start']<=timestamp<row['range']['end']:raise ValueError('关键帧时间必须在所选片段内')
        key=uuid.uuid4().hex;path=root/(key+'.png')
        command(['-ss',f'{timestamp:.6f}','-i',root/'source.video','-frames:v','1','-y',path],timeout=60)
        if not path.is_file():raise ValueError('该时刻没有可解码画面')
        with Image.open(path) as image:
            # PNG is decoded from source, never from the compressed preview.
            if image.width*image.height>20000000:raise ValueError('关键帧尺寸过大')
        return dict(id=key,time=timestamp,label='',image_sha256=sha256(path),links=[])
    def frame(self,project_id,frame_id):
        row=self.get(project_id)
        frame=next((f for f in row['frames'] if f['id']==uid(frame_id)),None)
        if frame is None:raise FileNotFoundError('关键帧不存在')
        return frame
    def edit(self,project_id,action,frame_id=None,timestamp=None,label=None,revision=None):
        with self.lock:
            row=self.get(project_id)
            if row['status']!='ready':raise ValueError('请先完成视频处理')
            if revision!=row['revision']:raise FileExistsError('项目已更新，请刷新后重试')
            old=self.frame(project_id,frame_id) if frame_id else None
            if action in {'add','replace'}:
                frame=self.extract(project_id,timestamp);frame['label']=str(label if label is not None else (old or {}).get('label',''))[:120]
                if old:row.setdefault('archived_frames',[]).append(old);row['frames'].remove(old)
                row['frames'].append(frame)
            elif action=='delete':row.setdefault('archived_frames',[]).append(old);row['frames'].remove(old)
            elif action=='label':old['label']=str(label or '')[:120]
            else:raise ValueError('未知关键帧操作')
            row['frames'].sort(key=lambda f:f['time']);row['revision']+=1;self.save(row);return row
    def activate(self,project_id,frame_id):
        with self.lock:
            row=self.get(project_id);frame=self.frame(project_id,frame_id)
            path=self.path(project_id)/(frame['id']+'.png')
            if sha256(path)!=frame['image_sha256']:raise ValueError('关键帧图片已改变')
            if not frame.get('upload_id'):
                upload=self.store.upload(path.read_bytes());frame['upload_id']=upload['id']
                row['frames']=[frame if f['id']==frame['id'] else f for f in row['frames']];self.save(row)
            return dict(upload_id=frame['upload_id'],frame=frame)
    def link(self,project_id,frame_id,kind,request_id,payload):
        with self.lock:
            self.activate(project_id,frame_id);row=self.get(project_id);frame=self.frame(project_id,frame_id)
            payload['video_source']=dict(project_id=row['id'],frame_id=frame['id'],time=frame['time'],image_sha256=frame['image_sha256'],source_sha256=row['source_sha256'],attribution=row['source_attribution'])
            if kind=='segment':payload['upload_id']=frame['upload_id']
            else:
                segment=self.store.job(payload['segment_id'])
                if segment['request'].get('upload_id')!=frame['upload_id']:raise ValueError('分割不属于当前关键帧，请重新选择演员')
                payload['method']='triposr';payload['video_color']=True
            job,created=self.store.submit(kind,request_id,payload)
            if not any(x['job_id']==job['id'] for x in frame['links']):frame['links'].append(dict(kind=kind,job_id=job['id']))
            row['frames']=[frame if f['id']==frame['id'] else f for f in row['frames']];self.save(row)
            return dict(id=job['id'],status=job['status'],deduplicated=not created)
    def analyze(self,job_id):
        job=self.store.job(job_id);p=job['request'];pid=p['project_id'];root=self.path(pid)
        with self.lock:
            row=self.get(pid);row.update(status='processing',range=dict(start=p['start'],end=p['end']),shots=[],frames=[]);self.save(row)
        if sha256(root/'source.video')!=row['source_sha256']:raise ValueError('原始视频已改变')
        duration=p['end']-p['start'];self.store.update(job_id,stage='video_preview')
        command(['-ss',p['start'],'-i',root/'source.video','-t',duration,'-map','0:v:0','-map','0:a:0?',
            '-vf','scale=w=min(960\\,iw):h=-2','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart','-y',root/'preview.mp4'],store=self.store)
        self.store.update(job_id,stage='shot_detection')
        _,log=command(['-ss',p['start'],'-i',root/'source.video','-t',duration,'-an','-vf',
            f"setpts=PTS-STARTPTS,scale=320:-2,select='gt(scene,{p['threshold']})',showinfo",'-f','null','-'],store=self.store)
        cuts=[0.]
        for value in re.findall(r'pts_time:([\d.e+-]+)',log):
            value=float(value)
            if value-cuts[-1]>=.6 and duration-value>=.6:cuts.append(value)
        if len(cuts)>100:raise ValueError('检测到过多镜头，请提高阈值或缩短片段')
        cuts.append(duration);shots=[];frames=[]
        for n,(a,b) in enumerate(zip(cuts,cuts[1:])):
            self.store.update(job_id,stage=f'keyframes_{n+1}_{len(cuts)-1}')
            candidates=[]
            for ratio in [.3,.5,.7]:
                f=self.extract(pid,p['start']+a+(b-a)*ratio)
                with Image.open(root/(f['id']+'.png')) as img:
                    gray=img.convert('L');gray.thumbnail((320,240));pixels=np.array(gray)
                    score=float(np.var(np.array(gray.filter(ImageFilter.FIND_EDGES)))) if pixels.mean()>12 else -1
                candidates.append((score,f))
            _,best=max(candidates,key=lambda c:c[0]);best.update(label=f'镜头 {n+1}',shot_id=n+1);frames.append(best)
            for _,f in candidates:
                if f is not best:(root/(f['id']+'.png')).unlink(missing_ok=True)
            shots.append(dict(id=n+1,start=p['start']+a,end=p['start']+b,representative=best['id']))
        with self.lock:
            row=self.get(pid);row.update(status='ready',shots=shots,frames=frames,revision=row['revision']+1,
                preview_sha256=sha256(root/'preview.mp4'),detection=dict(method='ffmpeg_scene_score',threshold=p['threshold'],minimum_shot_seconds=.6,representative='sharpest_of_30_50_70_percent'),error=None)
            self.save(row)
