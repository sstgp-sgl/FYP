"""Overview/focus templates alter camera only; saved poses are not touched."""
from itertools import product
import numpy as np
from .geometry import camera_from_look_at,project
from .render_cpu import render

DEFAULT_CAMERA=dict(width=960,height=640,fov_degrees=45.,margin_fraction=.05,point_size=3,min_visible_pixels=1,min_visible_ratio=1e-6,max_points=120000)


def corners(bounds):
    return np.array(list(product(*zip(bounds[0],bounds[1]))),float)


def make_shot(scene_id,spec,bounds,overview_bounds,settings):
    visible=set(bounds);targets=spec['target_ids'] or list(bounds)
    if any(t not in visible for t in targets):raise ValueError('focus/shot target is hidden')
    fit=overview_bounds if spec['template']=='overview' else {t:bounds[t] for t in targets}
    p=np.concatenate([corners(v) for v in fit.values()]);center=(p.min(0)+p.max(0))/2
    radius=max(float(np.linalg.norm(p-center,axis=1).max()),.01)
    direction=np.array([0.,.18,1.]);direction/=np.linalg.norm(direction)
    margin=settings['margin_fraction']
    distance=radius/(np.sin(np.deg2rad(settings['fov_degrees'])/2)*(1-2*margin)) * 1.1
    shot=camera_from_look_at((center+direction*distance).tolist(),center.tolist(),settings['width'],settings['height'],settings['fov_degrees'],max(radius*.005,.0001),distance+radius*4)
    shot.update(shot_id='shot_'+scene_id,scene_id=scene_id,template=spec['template'],target_ids=targets,
                complete_display=spec['complete_display'],margin_fraction=margin,point_size=settings['point_size'])
    return shot


def sample_by_instance(points,colors,ids,max_points):
    # Each visible object gets a budget, so a small asset cannot vanish in global sampling.
    labels=sorted(np.unique(ids).tolist());selected=[];budget=max(1,max_points//len(labels))
    for label in labels:
        indices=np.flatnonzero(ids==label)
        if len(indices)>budget:indices=indices[np.linspace(0,len(indices)-1,budget,dtype=int)]
        selected.extend(indices.tolist())
    selected=np.array(selected,int)
    return points[selected],colors[selected],ids[selected]


def render_shot(points,colors,ids,shot,settings):
    p,c,i=sample_by_instance(points,colors,ids,settings['max_points'])
    rgb,identity,depth=render(p,c,i,shot,settings['point_size'],(242,244,248))
    return rgb,identity,depth,len(p)


def check_shot(shot,bounds,idbuffer,id_map,settings):
    checks=[]
    for target in shot['target_ids']:
        if target not in bounds:
            checks.append(dict(name='target_visible:'+target,passed=False,measured='hidden'));continue
        uv,_,front=project(corners(bounds[target]),shot)
        margin=shot['margin_fraction'];w,h=shot['width'],shot['height'];radius=settings['point_size']//2
        complete=bool(front.all() and (uv[:,0]>=margin*w+radius).all() and (uv[:,0]<=(1-margin)*w-1-radius).all() and (uv[:,1]>=margin*h+radius).all() and (uv[:,1]<=(1-margin)*h-1-radius).all())
        pixels=int(np.count_nonzero(idbuffer==id_map[target]));ratio=pixels/(w*h)
        checks.append(dict(name='target_visible:'+target,passed=pixels>=settings['min_visible_pixels'] and ratio>=settings['min_visible_ratio'],measured=dict(pixels=pixels,ratio=ratio),expected=dict(min_pixels=settings['min_visible_pixels'],min_ratio=settings['min_visible_ratio'])))
        if shot['complete_display']:checks.append(dict(name='target_uncropped:'+target,passed=complete,measured=dict(projected_bounds=[uv.min(0).tolist(),uv.max(0).tolist()]),expected=dict(margin_fraction=margin)))
    return dict(passed=all(c['passed'] for c in checks),checks=checks)
