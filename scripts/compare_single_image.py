#!/usr/bin/env python3
"""Two single-image geometry baselines; equal-area sampling, equal gray PLY."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
from compare_bear import digest,write_cloud

def sample_surface(vertices,faces,count,seed=42):
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces,dtype=np.int64)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.min()<0 or f.max()>=len(v):raise ValueError('invalid mesh')
    triangles=v[f];area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)/2
    if not np.isfinite(area).all() or area.sum()<=0:raise ValueError('no positive mesh area')
    rng=np.random.default_rng(seed);tri=triangles[rng.choice(len(f),size=count,p=area/area.sum())]
    u=np.sqrt(rng.random((count,1)));w=rng.random((count,1))
    return (1-u)*tri[:,0]+u*(1-w)*tri[:,1]+u*w*tri[:,2]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--method',choices=['triposr','hunyuan'],required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--points',type=int,default=120000)
    a=ap.parse_args()
    if not 1<=a.points<=1000000:ap.error('invalid point budget')
    out=a.out/a.method;out.mkdir(parents=True,exist_ok=False)
    work=Path.home()/'work';foreground=work/'runs/vggt/bear/subjects/bear/subject.png'
    record=dict(method=a.method,status='running',seed=42,foreground_sha256=digest(foreground),source_image_sha256=digest(work/'data/inbox/bear.jpg'),files=[]);start=time.monotonic()
    try:
        import torch
        from PIL import Image
        if not torch.cuda.is_available():raise RuntimeError('Slurm GPU allocation required')
        torch.manual_seed(42)
        foreground_image=Image.open(foreground).convert('RGBA')
        mask_path=foreground.with_name('mask.png')
        mask=Image.open(mask_path).convert('L')
        if mask.size!=foreground_image.size:raise ValueError('foreground/mask size mismatch')
        alpha=np.asarray(mask)
        if alpha.min()==alpha.max():raise ValueError('foreground mask is constant')
        foreground_image.putalpha(mask)
        foreground_image.save(out/'foreground_rgba.png')
        record.update(mask_sha256=digest(mask_path),actual_rgba_sha256=digest(out/'foreground_rgba.png'))
        if a.method=='triposr':
            sys.path.insert(0,str(work/'deps/TripoSR'));os.environ['TRIPOSR_DINO_CONFIG']=str(work/'deps/TripoSR-weights/dino-config.json')
            from tsr.system import TSR
            from tsr.utils import resize_foreground
            from skimage.measure import marching_cubes
            model=TSR.from_pretrained(str(work/'deps/TripoSR-weights'),config_name='config.yaml',weight_name='model.ckpt').to('cuda').eval();model.renderer.set_chunk_size(8192)
            rgba=np.array(resize_foreground(foreground_image,.85),dtype=np.float32)/255
            image=Image.fromarray(np.uint8((rgba[:,:,:3]*rgba[:,:,3:4]+.5*(1-rgba[:,:,3:4]))*255));image.save(out/'input.png')
            with torch.inference_mode():
                code=model([image],device='cuda')[0];r=float(model.renderer.cfg.radius);n=256
                axis=torch.linspace(-r,r,n);grid=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3);density=[]
                for chunk in grid.split(8192):density.append(model.renderer.query_triplane(model.decoder,chunk.to('cuda'),code)['density_act'].detach().float().cpu().numpy().reshape(-1))
                field=np.concatenate(density).reshape(n,n,n)
                vertices,faces,_,_=marching_cubes(field,level=25,spacing=(2*r/(n-1),)*3);vertices=vertices-r
            record.update(resolution=256,density_threshold=25,model_revision='5b521936b01fbe1890f6f9baed0254ab6351c04a')
        else:
            sys.path.insert(0,str(work/'deps/Hunyuan3D-2-lite'))
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
            pipeline=Hunyuan3DDiTFlowMatchingPipeline.from_single_file(ckpt_path=str(work/'deps/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini/model.fp16.safetensors'),config_path=str(work/'deps/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini/config.yaml'),use_safetensors=True,device='cuda')
            image=foreground_image;image.save(out/'input.png')
            with torch.inference_mode():mesh=pipeline(image=image,num_inference_steps=30,octree_resolution=256,num_chunks=8192,generator=torch.Generator(device='cuda').manual_seed(42),output_type='trimesh')[0]
            vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
            record.update(resolution=256,num_inference_steps=30,model_revision=json.loads((work/'deps/Hunyuan3D-2mini/model-info.json').read_text())['sha'])
        points=sample_surface(vertices,faces,a.points)
        center=(np.min(vertices,axis=0)+np.max(vertices,axis=0))/2;extent=float(np.ptp(vertices,axis=0).max())
        if extent<=0:raise ValueError('zero extent')
        points=(points-center)*(2/extent)
        record['files']=[write_cloud(out/(a.method+'_120k_gray.ply'),points,np.full((len(points),3),180,dtype=np.uint8))]
        record.update(status='succeeded',mesh_vertices=len(vertices),mesh_faces=len(faces),normalization=dict(center=center.tolist(),uniform_scale=2/extent,rotation='none; native model axis conventions may differ'),sampling='area-weighted triangle surface, seed42',color='uniform RGB 180; geometry only')
    except Exception as exc:record.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:
        record['seconds']=time.monotonic()-start;(out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
