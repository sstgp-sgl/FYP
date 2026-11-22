"""Independent spatial, temporal, inheritance and shot acceptance from persisted outputs."""
from dataclasses import asdict
from pathlib import Path
from math import hypot
import numpy as np
from PIL import Image
from src.scene.models import SceneDocument
from src.scene.storage import read_json,controlled,sha256,SceneRepository
from src.scene.validation import validate_export
from .contracts import StoryPlan
from .spatial_contracts import SpatialConstraint,DEFAULTS
from .spatial_validation import measured_bounds,check_constraint
from .camera import check_shot


def event_check(event,current,base,bounds,base_bounds,settings):
    ids={i.instance_id:i for i in current.instances};old={i.instance_id:i for i in base.instances}
    s=event['subject_instance_id'];a=event['action'];before=old[s];after=ids[s];measured={};passed=False
    if a in ('approach','retreat'):
        t=event['object_instance_id']
        if not all(k in bounds and k in base_bounds for k in (s,t)):
            measured={'error':'motion requires both objects visible before and after'}
        else:
            def distance(bb):
                p=np.mean(np.asarray(bb[s]),axis=0);q=np.mean(np.asarray(bb[t]),axis=0)
                return hypot(float(p[0]-q[0]),float(p[2]-q[2]))
            d0,d1=distance(base_bounds),distance(bounds);change=d0-d1 if a=='approach' else d1-d0
            measured=dict(before_center_distance=d0,after_center_distance=d1,distance_change=change)
            passed=change>=event['min_change']-settings['epsilon']
    elif a=='hold':
        translation=max(abs(x-y) for x,y in zip(before.transform.translation_xyz,after.transform.translation_xyz))
        scale=abs(before.transform.uniform_scale-after.transform.uniform_scale)
        dot=abs(float(np.dot(before.transform.rotation_xyzw,after.transform.rotation_xyzw)))
        angle=2*np.arccos(min(1.,max(0.,dot)))
        measured=dict(translation_max_abs=translation,scale_abs=scale,rotation_radians=float(angle))
        passed=translation<=settings['translation_tolerance'] and scale<=settings['scale_tolerance'] and angle<=settings['rotation_tolerance'] and before.visible==after.visible
    elif a=='appear':measured=dict(before_visible=before.visible,after_visible=after.visible);passed=not before.visible and after.visible
    elif a=='disappear':measured=dict(before_visible=before.visible,after_visible=after.visible);passed=before.visible and not after.visible
    return dict(event_id=event['event_id'],action=a,passed=bool(passed),measured=measured,
                expected=dict(min_change=event.get('min_change')) if a in ('approach','retreat') else {},
                evidence_paths=[f'scenes/{base.scene_id}.json' if base.scene_id!='initial' else 'initial_scene.json',f'scenes/{current.scene_id}.json'])


def frame_geometry(intent,current,base,assets,settings,temporal_base=None):
    bounds=measured_bounds(current,assets);previous=measured_bounds(base,assets)
    poses={i.instance_id:i for i in current.instances};old={i.instance_id:i for i in base.instances}
    checks=[];errors=[]
    for c in intent.constraints:
        constraint=SpatialConstraint(**c)
        if constraint.subject_instance_id not in bounds or (constraint.object_instance_id is not None and constraint.object_instance_id not in bounds):
            errors.append('constraint references hidden object: '+constraint.constraint_id);continue
        checks.append(check_constraint(constraint,bounds,poses,old,settings))
    active=list(bounds)
    for n,name in enumerate(active):checks.append(check_constraint(SpatialConstraint('system_ground_'+name,name,'on_ground',origin='system'),bounds,poses,old,settings))
    for n,name in enumerate(active):
        for other in active[n+1:]:checks.append(check_constraint(SpatialConstraint('system_non_overlap_'+name+'_'+other,name,'non_overlap',other,origin='system'),bounds,poses,old,settings))
    temporal=temporal_base if temporal_base is not None else base
    temporal_bounds=measured_bounds(temporal,assets)
    events=[event_check(e,current,temporal,bounds,temporal_bounds,settings) for e in intent.events]
    movable={c['subject_instance_id'] for c in intent.constraints if c['relation']!='keep_pose'}|{e['subject_instance_id'] for e in intent.events if e['action'] in ('approach','retreat')}
    holds={e['subject_instance_id'] for e in intent.events if e['action']=='hold'}
    movable-=holds
    visibility_changes={e['subject_instance_id'] for e in intent.events if e['action'] in ('appear','disappear')}
    if set(poses)!=set(old):errors.append('instance ID drift')
    for name,pose in poses.items():
        if name not in old:continue
        before=old[name]
        if (pose.asset_id,pose.asset_hash)!=(before.asset_id,before.asset_hash):errors.append('asset reference drift: '+name)
        if abs(pose.transform.uniform_scale-before.transform.uniform_scale)>settings['scale_tolerance']:errors.append('scale changed: '+name)
        if abs(abs(np.dot(pose.transform.rotation_xyzw,before.transform.rotation_xyzw))-1)>1e-12:errors.append('rotation changed: '+name)
        if name not in movable and pose.transform.translation_xyz!=before.transform.translation_xyz:errors.append('unrequested position change: '+name)
        if name not in visibility_changes and pose.visible!=before.visible:errors.append('unrequested visibility change: '+name)
    return dict(passed=not errors and all(c['status']=='passed' for c in checks) and all(e['passed'] for e in events),
                checks=checks,event_checks=events,inheritance_errors=errors,bounds=bounds)


def validate_story(root,assets):
    root=Path(root);run=read_json(root/'run.json');errors=[];frames=[];states={}
    try:
        initial=SceneDocument(**read_json(root/'initial_scene.json'));initial.scene_id='initial';states['initial']=initial
        config=read_json(root/'config.json');settings={**DEFAULTS,**config.get('spatial',{})}
        plan=StoryPlan.parse(read_json(root/'input_snapshot.json')['plan'],initial,assets,settings,config.get('events',{}).get('min_distance_change',.2))
        if plan.to_dict()!=read_json(root/'story_plan.json'):errors.append('normalized plan differs from frozen original input')
        previous_id='initial'
        for intent in plan.scenes:
            item=dict(scene_id=intent.scene_id,passed=False,errors=[])
            try:
                scene=SceneRepository(assets).load('scenes/'+intent.scene_id+'.json')
                base=states[intent.base_scene_id]
                temporal=states.get(previous_id)
                if temporal is None and intent.events:raise ValueError('missing chronological predecessor for event')
                item.update(frame_geometry(intent,scene,base,assets,settings,temporal))
                states[intent.scene_id]=scene
                exported=validate_export(assets,'scenes/'+intent.scene_id+'.json','scenes/'+intent.scene_id+'.ply')
                if not exported['passed']:item['errors'].append('export coordinates/colors mismatch')
                shot=read_json(root/'shots'/('shot_'+intent.scene_id+'.json'))
                with Image.open(root/'renders'/(intent.scene_id+'.png')) as im:
                    im.load();size=(im.width,im.height)
                ids=np.load(root/'renders'/(intent.scene_id+'.ids.npy'),allow_pickle=False)
                depth=np.load(root/'renders'/(intent.scene_id+'.depth.npy'),allow_pickle=False)
                if ids.shape!=(size[1],size[0]) or depth.shape!=ids.shape or size!=(shot['width'],shot['height']):item['errors'].append('image/ID/depth shape mismatch')
                if shot['template']!=intent.shot['template'] or shot['scene_id']!=intent.scene_id:item['errors'].append('shot identity mismatch')
                expected_focal=shot['height']/(2*np.tan(np.deg2rad(config['camera']['fov_degrees'])/2))
                if shot['width']!=config['camera']['width'] or shot['height']!=config['camera']['height'] or not np.isclose(shot['K'][1][1],expected_focal):item['errors'].append('shot frame/FOV changed')
                targets=intent.shot['target_ids'] or list(item['bounds'])
                if shot['target_ids']!=targets or shot['complete_display']!=intent.shot['complete_display']:item['errors'].append('shot targets changed')
                item['shot_checks']=check_shot(shot,item['bounds'],ids,run['instance_ids'],config['camera'])
                if not item['shot_checks']['passed']:item['errors'].append('shot visibility/cropping failed')
                fg=ids>0
                if not np.isfinite(depth[fg]).all() or not (depth[fg]>=shot['near']).all() or not (depth[fg]<=shot['far']).all():item['errors'].append('invalid depth buffer')
                expected=intent.narration['text']+'\n'
                if (root/'narration'/(intent.scene_id+'.txt')).read_text()!=expected:item['errors'].append('narration changed')
                item['passed']=bool(item['passed']) and not item['errors']
            except Exception as exc:item['errors'].append(type(exc).__name__+': '+str(exc));item['passed']=False
            frames.append(item)
            previous_id=intent.scene_id
    except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
    if not run.get('artifacts'):errors.append('missing artifact manifest')
    for artifact in run.get('artifacts',[]):
        try:
            if sha256(controlled(root,artifact['path']))!=artifact['sha256']:errors.append('hash mismatch: '+artifact['path'])
        except Exception as exc:errors.append(str(exc))
    count=sum(bool(f['passed']) for f in frames);passed=len(frames)==3 and count==3 and not errors
    return dict(schema_version=1,passed=passed,status='succeeded' if passed else ('partial_failed' if count else 'failed'),
                scenes=frames,artifact_errors=errors,scope='Spatial/temporal/reference/inheritance/shot facts; complete natural-language meaning remains manual review')
