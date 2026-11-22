#!/usr/bin/env python3
"""Small bear experiment: existing PLY, six-view VGGT, single-image TripoSR.
Each group writes ordinary RGB PLY only; no web app, training or texture baking.
"""
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_cloud(path,points,colors):
    from plyfile import PlyData,PlyElement
    p=np.asarray(points,dtype=np.float32);c=np.asarray(colors)
    if p.ndim!=2 or p.shape[1]!=3 or c.shape!=p.shape or not len(p) or not np.isfinite(p).all():raise ValueError('invalid/empty point cloud')
    if c.dtype!=np.uint8:raise ValueError('RGB must be uint8')
    v=np.empty(len(p),dtype=[(k,'f4') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue')])
    for n,k in enumerate(('x','y','z')):v[k]=p[:,n]
    for n,k in enumerate(('red','green','blue')):v[k]=c[:,n]
    PlyData([PlyElement.describe(v,'vertex')],text=False,byte_order='<').write(str(path))
    check=PlyData.read(str(path))['vertex'].data
    if len(check)!=len(p):raise ValueError('PLY readback count mismatch')
    if not np.allclose(np.column_stack([check[k] for k in ('x','y','z')]),p,rtol=0,atol=0):raise ValueError('PLY coordinates changed')
    if not np.array_equal(np.column_stack([check[k] for k in ('red','green','blue')]),c):raise ValueError('PLY RGB changed')
    return dict(file=path.name,points=len(p),sha256=digest(path))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--group',required=True,choices=['baseline','six','triposr'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--work',type=Path,default=Path.home()/'work')
    ap.add_argument('--points',type=int,default=120000)
    ap.add_argument('--resolution',type=int,default=128)
    args=ap.parse_args()
    if args.points<1 or not 32<=args.resolution<=256:ap.error('invalid sample/grid budget')
    base=args.work/'runs/vggt/bear';out=args.out/args.group
    out.mkdir(parents=True,exist_ok=False)
    record=dict(group=args.group,status='running',seed=42,files=[],source_image_sha256=digest(args.work/'data/inbox/bear.jpg'))
    started=time.monotonic()
    try:
        if args.group=='baseline':
            from src.scene.storage import read_vertices
            source=base/'reconstruct/bear/pointcloud.ply'
            p,c=read_vertices(source,'point_cloud')
            shutil.copyfile(source,out/'01_full.ply')
            record['files'].append(dict(file='01_full.ply',points=len(p),sha256=digest(out/'01_full.ply')))
            index=np.linspace(0,len(p)-1,min(args.points,len(p)),dtype=int)
            record['files'].append(write_cloud(out/'02_sampled.ply',p[index],c[index]))
            record.update(source_ply_sha256=digest(source),sample_rule='deterministic linspace indices; identical to one-instance preview sampling')
        elif args.group=='six':
            import yaml,torch
            if not torch.cuda.is_available():raise RuntimeError('Slurm CUDA allocation required')
            from src.vggt_reconstruct import run_vggt
            from src.pointcloud_filter import filter_pointcloud
            cfg=yaml.safe_load((REPO/'configs/hpc-local.yaml').read_text())
            cfg['views']['include_original']=False
            paths=[base/'views/bear'/f'view_{i:02d}.png' for i in range(6)]
            result=run_vggt([str(p) for p in paths],cfg)
            points,colors,stats=filter_pointcloud(result['points'],result['colors'],result['conf'],cfg)
            record['files'].append(write_cloud(out/'03_six_generated_views.ply',points,colors))
            for p in paths:shutil.copyfile(p,out/p.name)
            record.update(stats=stats,view_sha256={p.name:digest(p) for p in paths},note='Same six historical generated views; excludes original view_06. This is an input ablation, not proof that the views are consistent.',config=cfg)
        else:
            import torch
            from PIL import Image
            from skimage.measure import marching_cubes
            if not torch.cuda.is_available():raise RuntimeError('Slurm CUDA allocation required')
            sys.path.insert(0,str(args.work/'deps/TripoSR'))
            from tsr.system import TSR
            from tsr.utils import resize_foreground
            torch.manual_seed(42)
            weights=args.work/'deps/TripoSR-weights'
            model=TSR.from_pretrained(str(weights),config_name='config.yaml',weight_name='model.ckpt').to('cuda').eval()
            model.renderer.set_chunk_size(8192)
            foreground=base/'subjects/bear/subject.png'
            image=resize_foreground(Image.open(foreground).convert('RGBA'),.85)
            x=np.asarray(image,dtype=np.float32)/255
            image=Image.fromarray(np.uint8((x[:,:,:3]*x[:,:,3:4]+.5*(1-x[:,:,3:4]))*255))
            image.save(out/'input.png')
            with torch.inference_mode():
                code=model([image],device='cuda')[0]
                r=float(model.renderer.cfg.radius);n=args.resolution
                axis=torch.linspace(-r,r,n)
                grid=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
                density=[]
                for chunk in grid.split(8192):density.append(model.renderer.query_triplane(model.decoder,chunk.to('cuda'),code)['density_act'].detach().float().cpu().numpy().reshape(-1))
                field=np.concatenate(density).reshape(n,n,n)
                vertices,_,_,_=marching_cubes(field,level=25.,spacing=(2*r/(n-1),)*3)
                vertices=np.ascontiguousarray(vertices-r,dtype=np.float32)
                colors=[]
                for chunk in torch.from_numpy(vertices).split(8192):colors.append(model.renderer.query_triplane(model.decoder,chunk.to('cuda'),code)['color'].detach().float().cpu().numpy())
            colors=np.uint8(np.clip(np.concatenate(colors),0,1)*255)
            record['files'].append(write_cloud(out/'04_triposr_surface.ply',vertices,colors))
            record.update(resolution=n,density_threshold=25,foreground_sha256=digest(foreground),weights_sha256=digest(weights/'model.ckpt'),note='Single-image TripoSR; CPU marching cubes extracts surface vertices and model RGB. No UV texture baking, mesh file, or claim of TRELLIS/Hunyuan quality. Coordinates differ from VGGT; evaluate shape independently.')
        record['status']='succeeded'
    except Exception as exc:
        record.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        record['seconds']=time.monotonic()-started
        (out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
