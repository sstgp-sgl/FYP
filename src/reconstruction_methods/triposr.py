"""TripoSR geometry and predicted surface RGB; no UV/texture baking."""
import os,sys
from pathlib import Path
import numpy as np
from PIL import Image
from .common import sample_surface,sha256,write_mesh_ply


def query_colors(model,code,positions,chunk_size):
    import torch
    positions=np.asarray(positions,dtype=np.float32);colors=[]
    if positions.ndim!=2 or positions.shape[1]!=3 or not len(positions) or not np.isfinite(positions).all():raise ValueError('invalid RGB query positions')
    with torch.inference_mode():
        for start in range(0,len(positions),chunk_size):
            points=torch.from_numpy(positions[start:start+chunk_size]).to(code.device)
            rgb=model.renderer.query_triplane(model.decoder,points,code)['color'].detach().float().cpu().numpy()
            if rgb.shape!=points.shape or not np.isfinite(rgb).all() or rgb.min()<-1e-6 or rgb.max()>1+1e-6:raise ValueError('TripoSR returned invalid surface RGB')
            colors.append(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8))
    return np.concatenate(colors)


def reconstruct(image,out,cfg):
    import torch
    from skimage.measure import marching_cubes
    sys.path.insert(0,str(Path(cfg['repo']).expanduser().resolve()))
    if cfg.get('dino_config'):os.environ['TRIPOSR_DINO_CONFIG']=str(Path(cfg['dino_config']).expanduser().resolve())
    from tsr.system import TSR
    from tsr.utils import resize_foreground
    weights=Path(cfg['weights']).expanduser().resolve()
    model=TSR.from_pretrained(str(weights),config_name='config.yaml',weight_name='model.ckpt').to('cuda').eval()
    model.renderer.set_chunk_size(cfg['chunk_size'])
    rgba=np.asarray(resize_foreground(image,.85),dtype=np.float32)/255
    model_image=Image.fromarray(np.uint8((rgba[:,:,:3]*rgba[:,:,3:4]+.5*(1-rgba[:,:,3:4]))*255))
    model_image.save(out/'model_input.png')
    with torch.inference_mode():
        code=model([model_image],device='cuda')[0];r=float(model.renderer.cfg.radius);n=cfg['resolution']
        axis=torch.linspace(-r,r,n);grid=torch.stack(torch.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3);density=[]
        for chunk in grid.split(cfg['chunk_size']):density.append(model.renderer.query_triplane(model.decoder,chunk.to('cuda'),code)['density_act'].detach().float().cpu().numpy().reshape(-1))
        field=np.concatenate(density).reshape(n,n,n)
        vertices,faces,_,_=marching_cubes(field,level=cfg['threshold'],spacing=(2*r/(n-1),)*3);vertices=vertices-r
    points=sample_surface(vertices,faces,cfg['points'],cfg['seed'])
    colors=None;vertex_colors=np.full(vertices.shape,180,dtype=np.uint8)
    if not cfg['gray']:
        colors=query_colors(model,code,points,cfg['chunk_size'])
        vertex_colors=query_colors(model,code,vertices,cfg['chunk_size'])
    write_mesh_ply(out/'mesh_native.ply',vertices,faces,vertex_colors)
    return points,colors,dict(sampling='triangle_area',vertices=len(vertices),faces=len(faces),
        color_source='predicted_rgb' if colors is not None else 'gray',
        rgb_query_coordinates='native model coordinates before output normalization',
        native_up=[0,0,1],texture_baked=False,weight_sha256=sha256(weights/'model.ckpt'))
