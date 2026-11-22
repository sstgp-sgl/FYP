"""Existing Zero123++ -> VGGT, with optional reuse of generated views."""
import gc
from pathlib import Path
import numpy as np
from PIL import Image


def reconstruct(image,out,cfg):
    import torch
    from src.config import load_config
    from src.reconstruction import reconstruct_views
    from src.pointcloud_io import read_ply
    settings=load_config(cfg['pipeline_config']);settings['seed']=cfg['seed'];settings['device']='cuda'
    if cfg.get('views_dir'):
        views=Path(cfg['views_dir']).expanduser().resolve()
    else:
        from src.zero123 import generate_views_for_subjects
        subject=out/'subjects'/'subject';subject.mkdir(parents=True)
        mask=image.getchannel('A');mask.save(subject/'mask.png')
        white=Image.new('RGB',image.size,'white');white.paste(image,mask=mask);white.save(subject/'subject.png')
        doc=dict(subjects=[dict(name='subject',subject_file='subject/subject.png',mask_file='subject/mask.png')])
        generate_views_for_subjects(doc,out/'subjects',out/'views',settings)
        gc.collect();torch.cuda.empty_cache();views=out/'views/subject'
    reconstruct_views(views,out/'vggt',settings)
    points,colors=read_ply(out/'vggt/pointcloud.ply')
    raw=len(points);rng=np.random.default_rng(cfg['seed'])
    # This is sampling existing observations, not mesh surface sampling.
    indices=rng.choice(raw,size=cfg['points'],replace=raw<cfg['points'])
    return points[indices],colors[indices],dict(sampling='observed_points',raw_points=raw,repeated_points=raw<cfg['points'],views=str(views))
