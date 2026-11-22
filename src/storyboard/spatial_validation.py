"""Independent P3 measurements: reads persisted transforms; never imports/executes layout."""
from itertools import product
from math import hypot,acos
from pathlib import Path
import numpy as np
from src.scene.storage import read_json,sha256,controlled,SceneRepository
from src.scene.models import SceneDocument
from src.scene.validation import validate_export
from .spatial_contracts import SpatialRequest


def measured_bounds(scene,assets):
    result={}
    for instance in scene.instances:
        if not instance.visible:continue
        p,_=assets.points(assets.get(instance.asset_id))
        lo,hi=p.min(0),p.max(0)
        # Enumerate eight source/canonical AABB corners, not min/max endpoints only.
        corners=np.array(list(product(*[(lo[k],hi[k]) for k in range(3)])))
        x,y,z,w=instance.transform.rotation_xyzw;v=np.array([x,y,z])
        q=(corners+2*np.cross(v,np.cross(v,corners)+w*corners))*instance.transform.uniform_scale
        q+=instance.transform.translation_xyz
        result[instance.instance_id]=[q.min(0).tolist(),q.max(0).tolist()]
    return result


def check_constraint(c,bounds,poses,base_poses,settings):
    eps=settings['epsilon'];a=bounds[c.subject_instance_id];b=bounds.get(c.object_instance_id)
    relation=c.relation;expected={};tolerance=eps
    if relation in ('left_of','right_of'):
        value=b[0][0]-a[1][0] if relation=='left_of' else a[0][0]-b[1][0]
        threshold=c.parameters['gap'];passed=value>=threshold-eps;expected={'minimum_gap':threshold}
    elif relation in ('near','far'):
        dx=max(0.,a[0][0]-b[1][0],b[0][0]-a[1][0]);dz=max(0.,a[0][2]-b[1][2],b[0][2]-a[1][2])
        value=hypot(dx,dz)
        if relation=='near':threshold=c.parameters['near_max'];passed=value<=threshold+eps;expected={'maximum_footprint_distance':threshold}
        else:threshold=c.parameters['far_min'];passed=value>=threshold-eps;expected={'minimum_footprint_distance':threshold}
    elif relation=='on_ground':value=a[0][1];passed=abs(value)<=eps;expected={'min_y':0}
    elif relation=='non_overlap':
        width=max(0.,min(a[1][0],b[1][0])-max(a[0][0],b[0][0]));depth=max(0.,min(a[1][2],b[1][2])-max(a[0][2],b[0][2]))
        value=width*depth;passed=value<=eps**2;tolerance=eps**2;expected={'maximum_footprint_overlap_area':eps**2}
    elif relation=='keep_pose':
        current=poses[c.subject_instance_id].transform;before=base_poses[c.subject_instance_id].transform
        displacement=max(abs(a-b) for a,b in zip(current.translation_xyz,before.translation_xyz))
        scale=abs(current.uniform_scale-before.uniform_scale)
        dot=abs(sum(a*b for a,b in zip(current.rotation_xyzw,before.rotation_xyzw)))
        angle=2*acos(min(1.,max(0.,dot)))
        value=dict(translation_max_abs=displacement,scale_abs=scale,rotation_radians=angle)
        passed=displacement<=settings['translation_tolerance'] and scale<=settings['scale_tolerance'] and angle<=settings['rotation_tolerance']
        expected={k:settings[k] for k in ('translation_tolerance','scale_tolerance','rotation_tolerance')};tolerance=None
    else:raise ValueError('unsupported relation')
    return dict(constraint_id=c.constraint_id,relation=relation,subject_instance_id=c.subject_instance_id,
                object_instance_id=c.object_instance_id,origin=c.origin,status='passed' if passed else 'failed',
                measured=value,expected=expected,tolerance=tolerance,evidence_paths=['scene.json','combined.ply','base_scene.json'])


def validate_geometry(scene,base,request,assets):
    bounds=measured_bounds(scene,assets);poses={i.instance_id:i for i in scene.instances};before={i.instance_id:i for i in base.instances}
    checks=[check_constraint(c,bounds,poses,before,request.settings) for c in request.constraints]
    invariant_errors=[]
    if set(poses)!=set(before):invariant_errors.append('instance IDs changed')
    for inst_id,i in poses.items():
        old=before.get(inst_id)
        if old is None:continue
        if (i.asset_id,i.asset_hash,i.visible)!=(old.asset_id,old.asset_hash,old.visible):invariant_errors.append('source/reference/visibility changed: '+inst_id)
        if abs(i.transform.uniform_scale-old.transform.uniform_scale)>request.settings['scale_tolerance']:invariant_errors.append('scale changed: '+inst_id)
        if abs(abs(np.dot(i.transform.rotation_xyzw,old.transform.rotation_xyzw))-1)>1e-12:invariant_errors.append('fixed rotation changed: '+inst_id)
    for inst_id in request.fixed_instance_ids:
        if poses[inst_id].transform.translation_xyz!=before[inst_id].transform.translation_xyz:invariant_errors.append('anchor translation changed: '+inst_id)
    return dict(schema_version=1,passed=not invariant_errors and all(c['status']=='passed' for c in checks),
                checks=checks,bounds=bounds,invariant_errors=invariant_errors,scope='World-axis conservative AABBs and footprint distance; not exact physical collision')


def validate_run(root,assets):
    root=Path(root);errors=[]
    try:
        scene=SceneRepository(assets).load();base=SceneDocument(**read_json(root/'base_scene.json'))
        # Re-parse original request, not a solver-predicted set of satisfied relations.
        request=SpatialRequest.parse(read_json(root/'spatial_request.json'),[i.instance_id for i in base.instances if i.visible])
        result=validate_geometry(scene,base,request,assets)
        exported=validate_export(assets)
        if not exported['passed']:errors.append('export coordinates/colors inconsistent')
        run=read_json(root/'run.json')
        from PIL import Image
        with Image.open(root/'renders/layout_front.png') as image:
            image.load();shape=(image.height,image.width)
        ids=np.load(root/'renders/layout_front.ids.npy',allow_pickle=False)
        depth=np.load(root/'renders/layout_front.depth.npy',allow_pickle=False)
        shot=read_json(root/'renders/layout_front.shot.json')
        if ids.shape!=shape or depth.shape!=shape:errors.append('render buffer shape mismatch')
        else:
            for name,identity in run.get('instance_ids',{}).items():
                if not np.any(ids==identity):errors.append('invisible target: '+name)
            front=ids>0
            if not np.all(np.isfinite(depth[front])) or not np.all(depth[front]>=shot['near']) or not np.all(depth[front]<=shot['far']):errors.append('invalid rendered depth')
        if not run.get('instance_ids'):errors.append('missing instance ID map')
        if not run.get('artifacts'):errors.append('missing artifact manifest')
        for artifact in run.get('artifacts',[]):
            p=controlled(root,artifact['path'])
            if not p.is_file() or sha256(p)!=artifact['sha256']:errors.append('hash mismatch: '+artifact['path'])
    except Exception as exc:
        result=dict(schema_version=1,passed=False,checks=[]);errors.append(type(exc).__name__+': '+str(exc))
    result.update(artifact_errors=errors,passed=bool(result['passed']) and not errors)
    return result
