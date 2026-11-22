"""Single-image TripoSR geometry; no UV/texture baking."""
import os,sys
from pathlib import Path
import numpy as np
from PIL import Image
from .common import sample_surface,sha256


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
    return points,None,dict(sampling='triangle_area',vertices=len(vertices),faces=len(faces),weight_sha256=sha256(weights/'model.ckpt'))
