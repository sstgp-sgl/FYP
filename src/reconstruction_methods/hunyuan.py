"""Single-image Hunyuan3D-2mini shape only."""
import sys
from pathlib import Path
import numpy as np
from .common import sample_surface,sha256


def reconstruct(image,out,cfg):
    import torch
    sys.path.insert(0,str(Path(cfg['repo']).expanduser().resolve()))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    weights=Path(cfg['weights']).expanduser().resolve()
    pipeline=Hunyuan3DDiTFlowMatchingPipeline.from_single_file(ckpt_path=str(weights/'model.fp16.safetensors'),config_path=str(weights/'config.yaml'),use_safetensors=True,device='cuda')
    with torch.inference_mode():
        mesh=pipeline(image=image,num_inference_steps=cfg['steps'],octree_resolution=cfg['resolution'],num_chunks=cfg['chunk_size'],generator=torch.Generator(device='cuda').manual_seed(cfg['seed']),output_type='trimesh')[0]
    if mesh is None:raise ValueError('Hunyuan produced no surface')
    points=sample_surface(mesh.vertices,mesh.faces,cfg['points'],cfg['seed'])
    return points,None,dict(sampling='triangle_area',vertices=len(mesh.vertices),faces=len(mesh.faces),weight_sha256=sha256(weights/'model.fp16.safetensors'))
