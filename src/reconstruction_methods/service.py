"""One masked input and PLY contract for all reconstruction methods."""
import importlib,json,time
from pathlib import Path
import numpy as np
from .common import prepare_foreground,sha256,write_ply

METHODS=('vggt','triposr','hunyuan')
DEFAULTS=dict(points=120000,seed=42,resolution=256,chunk_size=8192,steps=30,threshold=25.,gray=True,normalize=True)
BACKEND_KEYS={'vggt':{'pipeline_config','views_dir'},'triposr':{'repo','weights','dino_config'},'hunyuan':{'repo','weights'}}


def resolve_options(method,options):
    if method not in METHODS:raise ValueError('unknown method: '+method)
    extra=set(options)-set(DEFAULTS)-BACKEND_KEYS[method]
    if extra:raise ValueError('unknown options: '+', '.join(sorted(extra)))
    cfg={**DEFAULTS,**options}
    for key,lo,hi in [('points',1,1000000),('seed',0,2**32-1),('resolution',32,256),('chunk_size',1,65536),('steps',1,100)]:
        if type(cfg[key]) is not int or not lo<=cfg[key]<=hi:raise ValueError('invalid '+key)
    if type(cfg['gray']) is not bool or type(cfg['normalize']) is not bool:raise ValueError('gray/normalize must be bool')
    if not isinstance(cfg['threshold'],(int,float)) or not np.isfinite(cfg['threshold']) or cfg['threshold']<=0:raise ValueError('invalid threshold')
    return cfg


def save_json(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def reconstruct_asset(method,image,mask,out,options=None):
    cfg=resolve_options(method,options or {})
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();run=dict(method=method,status='running',options=cfg,artifacts=[])
    save_json(root/'run.json',run)
    try:
        image=Path(image).resolve();mask=Path(mask).resolve() if mask else None
        foreground=prepare_foreground(image,mask);foreground.save(root/'foreground.png')
        run['inputs']=dict(image_sha256=sha256(image),mask_sha256=sha256(mask) if mask else None,foreground_sha256=sha256(root/'foreground.png'))
        save_json(root/'run.json',run)
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; use a Slurm GPU allocation on HPC')
        torch.manual_seed(cfg['seed'])
        backend=importlib.import_module('.'+method,__package__)
        points,colors,details=backend.reconstruct(foreground,root,cfg)
        points=np.asarray(points,dtype=float)
        if points.shape!=(cfg['points'],3) or not np.isfinite(points).all():raise ValueError('backend must return exactly requested finite XYZ points')
        if cfg['normalize']:
            lo,hi=points.min(0),points.max(0);extent=float((hi-lo).max())
            if extent<=0:raise ValueError('zero point cloud extent')
            center=(lo+hi)/2;scale=2/extent;points=(points-center)*scale
            run['normalization']=dict(center=center.tolist(),uniform_scale=scale,rotation='none; native backend axes retained')
        if cfg['gray'] or colors is None:colors=np.full(points.shape,180,dtype=np.uint8)
        run['artifacts']=[write_ply(root/'pointcloud.ply',points,colors)]
        run.update(status='succeeded',backend=details,color='gray' if cfg['gray'] or method!='vggt' else 'source_rgb')
    except Exception as exc:
        run.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        run['seconds']=time.monotonic()-start
        run['code_sha256']={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
        save_json(root/'run.json',run)
    return run
