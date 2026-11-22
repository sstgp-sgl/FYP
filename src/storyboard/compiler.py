"""StoryPlan -> inherited static SceneDocuments. Events become hard search predicates."""
from dataclasses import asdict
import math
import numpy as np
from src.scene.models import SceneDocument,now
from .spatial_contracts import SpatialRequest,SpatialError,DEFAULTS
from .spatial_validation import measured_bounds
from .layout import solve
from .story_validation import frame_geometry


def compile_scene(intent,base,assets,settings,temporal_base=None):
    current=SceneDocument(**asdict(base));current.scene_id=intent.scene_id;current.updated_at=now();current.constraints=[]
    temporal_base=temporal_base if temporal_base is not None else base
    poses={i.instance_id:i for i in current.instances};before={i.instance_id:i for i in temporal_base.instances}
    for event in intent.events:
        action=event['action'];subject=poses[event['subject_instance_id']]
        if action=='appear':
            if before[event['subject_instance_id']].visible:raise SpatialError('invalid_visibility_transition','appear requires a hidden base instance')
            subject.visible=True
        elif action=='disappear':
            if not before[event['subject_instance_id']].visible:raise SpatialError('invalid_visibility_transition','disappear requires a visible base instance')
            subject.visible=False
    visible=[i.instance_id for i in current.instances if i.visible]
    if not visible:raise SpatialError('empty_scene','P4 requires at least one visible instance')
    for c in intent.constraints:
        if c['subject_instance_id'] not in visible or (c['object_instance_id'] is not None and c['object_instance_id'] not in visible):raise SpatialError('hidden_reference','spatial relation refers to hidden instance')
    motions=[e for e in intent.events if e['action'] in ('approach','retreat')]
    held={e['subject_instance_id'] for e in intent.events if e['action']=='hold'}
    movable=({c['subject_instance_id'] for c in intent.constraints if c['relation']!='keep_pose'}|{e['subject_instance_id'] for e in motions})-held
    fixed=[name for name in visible if name not in movable]
    baseline=measured_bounds(temporal_base,assets);current_bounds=measured_bounds(current,assets);extra={}
    for event in motions:
        name,target=event['subject_instance_id'],event['object_instance_id']
        if name not in baseline or target not in baseline or name not in visible or target not in visible:raise SpatialError('event_visibility_required','motion needs visible objects before and after')
        a=np.mean(baseline[name],axis=0)[[0,2]];b=np.mean(baseline[target],axis=0)[[0,2]]
        old=float(np.linalg.norm(a-b));delta=event['min_change'];radius=old-delta if event['action']=='approach' else old+delta
        if radius<0:raise SpatialError('layout_not_found','approach bound is below zero center distance')
        current_a=np.mean(current_bounds[name],axis=0)[[0,2]];current_b=np.mean(current_bounds[target],axis=0)[[0,2]]
        theta=math.atan2(a[1]-b[1],a[0]-b[0]);angles=[theta]+[i*math.pi/8 for i in range(16)]
        radii=[radius,radius*.8,radius*.5] if event['action']=='approach' else [radius,radius+.5,radius+1.5]
        for r in radii:
            for angle in angles:
                target_center=current_b+r*np.array([math.cos(angle),math.sin(angle)])
                t=np.array(poses[name].transform.translation_xyz)[[0,2]]+target_center-current_a
                extra.setdefault(name,[]).append(t.tolist())
    def temporal_filter(bounds,chosen):
        for event in motions:
            a=np.mean(bounds[event['subject_instance_id']],axis=0)[[0,2]];b=np.mean(bounds[event['object_instance_id']],axis=0)[[0,2]]
            olda=np.mean(baseline[event['subject_instance_id']],axis=0)[[0,2]];oldb=np.mean(baseline[event['object_instance_id']],axis=0)[[0,2]]
            d0,d1=float(np.linalg.norm(olda-oldb)),float(np.linalg.norm(a-b))
            change=d0-d1 if event['action']=='approach' else d1-d0
            if change<event['min_change']-settings['epsilon']:return False
        return True
    diagnostic=dict(status='inherited',movable_instance_ids=sorted(movable),fixed_instance_ids=fixed)
    if movable and len(visible)>=2:
        cs=intent.constraints or [dict(constraint_id='event_ground',subject_instance_id=visible[0],relation='on_ground')]
        request=SpatialRequest.parse(dict(schema_version=1,constraints=cs,fixed_instance_ids=fixed,settings=settings),visible)
        diagnostic,candidate=solve(current,request,assets,candidate_filter=temporal_filter,extra_positions=extra)
        if candidate is None:raise SpatialError(diagnostic['status'],'event/spatial layout not found',diagnostic)
        current=candidate
    elif movable:
        for name in movable:
            if name in visible:poses[name].transform.translation_xyz[1]-=measured_bounds(current,assets)[name][0][1]
    if any(target not in visible for target in intent.shot['target_ids']):raise SpatialError('hidden_shot_target','shot target is hidden')
    temporal_rules=[]
    for event in motions:
        a=np.mean(baseline[event['subject_instance_id']],axis=0)[[0,2]];b=np.mean(baseline[event['object_instance_id']],axis=0)[[0,2]]
        old=float(np.linalg.norm(a-b))
        temporal_rules.append(dict(event_id=event['event_id'],relation='center_distance_max' if event['action']=='approach' else 'center_distance_min',
                                  before_distance=old,bound=old-event['min_change'] if event['action']=='approach' else old+event['min_change'],
                                  temporal_base_scene_id=temporal_base.scene_id))
    diagnostic.update(temporal_constraints=temporal_rules,inheritance_base_scene_id=base.scene_id,temporal_base_scene_id=temporal_base.scene_id)
    report=frame_geometry(intent,current,base,assets,settings,temporal_base)
    if not report['passed']:raise SpatialError('independent_validation_failed','compiled frame violates original intent',report)
    return current,diagnostic
