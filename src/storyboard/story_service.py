"""Shared P4 compile/validate workflow. Frozen assets, three static frames, finite spatial search."""
from dataclasses import asdict
from pathlib import Path
import json,subprocess,time,shutil
import numpy as np
from PIL import Image
import yaml
from src.scene.models import SceneDocument,now
from src.scene.storage import SceneRepository,read_json,atomic_json,read_vertices,sha256
from src.scene.composition import export
from .assets import freeze_assets
from .spatial_service import LayoutAssets
from .spatial_contracts import SpatialError,SpatialRequest,DEFAULTS
from .spatial_validation import measured_bounds
from .contracts import StoryPlan,json_schema
from .compiler import compile_scene
from .camera import DEFAULT_CAMERA,make_shot,render_shot,check_shot
from .service import atomic_array
from .story_validation import validate_story
from .export import write_storyboard


def camera_settings(config):
    values={**DEFAULT_CAMERA,**config.get('camera',{})}
    if set(values)-set(DEFAULT_CAMERA):raise SpatialError('invalid_camera_config','unknown camera settings')
    for k in ('width','height','point_size','min_visible_pixels','max_points'):
        if type(values[k]) is not int or values[k]<1:raise SpatialError('invalid_camera_config','invalid '+k)
    if values['width']>4096 or values['height']>4096 or values['point_size']>15 or values['point_size']%2!=1 or values['max_points']<3:raise SpatialError('invalid_camera_config','invalid render budget')
    for k in ('fov_degrees','margin_fraction','min_visible_ratio'):
        if type(values[k]) not in (int,float) or not np.isfinite(values[k]):raise SpatialError('invalid_camera_config','invalid '+k)
    if not 1<values['fov_degrees']<179 or not 0<=values['margin_fraction']<.4 or not 0<=values['min_visible_ratio']<=1:raise SpatialError('invalid_camera_config','invalid FOV/margin/visibility')
    return values


def compile_story(plan_path,asset_root,scene_path,out,config_path=None,allow_provisional=False):
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    code_root=Path(__file__).resolve().parents[2]
    run=dict(schema_version=1,pipeline='storyboard_p4',algorithm_version='story_compiler_v1',status='running',stage='input',started_at=now(),scenes=[],artifacts=[],provider=None)
    plan=None;assets=None;compiled=[];records={};states={}
    atomic_json(root/'run.json',run)
    try:
        if not Path(asset_root).is_dir():raise SpatialError('missing_asset_root','source asset root does not exist')
        cfg_path=Path(config_path) if config_path else code_root/'configs/storyboard.yaml'
        cfg_path=cfg_path if cfg_path.is_absolute() else code_root/cfg_path
        config=yaml.safe_load(cfg_path.read_text())
        if not isinstance(config,dict) or config.get('schema_version')!=1:raise SpatialError('invalid_config','invalid configuration schema')
        config['camera']=camera_settings(config)
        settings={**DEFAULTS,**config.get('spatial',{})}
        source_assets=LayoutAssets(asset_root)
        initial=SceneRepository(source_assets).validate(SceneDocument(**read_json(scene_path)));initial.scene_id='initial'
        if not np.allclose(initial.reference_frame['right'],[1,0,0],atol=1e-12):raise SpatialError('unsupported_frame','story world right must be +X')
        names=[i.instance_id for i in initial.instances]
        settings=SpatialRequest.parse(dict(schema_version=1,constraints=[dict(constraint_id='config_ground',subject_instance_id=names[0],relation='on_ground')],settings=config.get('spatial',{})),names).settings
        config['spatial']=settings
        raw=read_json(plan_path)
        atomic_json(root/'input_snapshot.json',dict(plan=raw,source_scene_sha256=sha256(scene_path)))
        atomic_json(root/'config.json',config);atomic_json(root/'initial_scene.json',asdict(initial))
        plan=StoryPlan.parse(raw,initial,source_assets,settings,config.get('events',{}).get('min_distance_change',.2))
        atomic_json(root/'story_plan.json',plan.to_dict());atomic_json(root/'story_plan.schema.json',json_schema())
        freeze_assets(asset_root,[i.asset_id for i in initial.instances],root);assets=LayoutAssets(root)
        kinds={assets.get(i.asset_id).source_kind for i in initial.instances};run['source']=next(iter(kinds)) if len(kinds)==1 else 'mixed'
        confirmation=[]
        for aid in plan.asset_ids:
            p=root/'assets'/aid/'semantics.json'
            if p.is_file():
                sem=read_json(p);confirmation.append(sem['orientation_status']=='confirmed' and sem['display_name_confirmed'])
        run['human_status']='needs_confirmation' if confirmation and not all(confirmation) else ('confirmed' if confirmation else 'synthetic')
        if run['human_status']=='needs_confirmation' and not allow_provisional:raise SpatialError('needs_confirmation','explicit --allow-provisional required')
        run['instance_ids']={name:i+1 for i,name in enumerate(sorted(i.instance_id for i in initial.instances))}
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=code_root,capture_output=True,text=True)
        run['code_commit']=git.stdout.strip() if git.returncode==0 else None
        run['code_sha256']={str(p.relative_to(code_root)):sha256(p) for p in sorted((code_root/'src/storyboard').glob('*.py'))}
        run['input_hashes']={p:sha256(root/p) for p in ('input_snapshot.json','config.json','initial_scene.json')}
        for folder in ('scenes','shots','renders','narration','logs'):(root/folder).mkdir()
        states={'initial':initial};previous=initial;run['stage']='compile'
        for intent in plan.scenes:
            record=dict(scene_id=intent.scene_id,status='pending');run['scenes'].append(record);records[intent.scene_id]=record
            try:
                base=states[intent.base_scene_id]
                scene,diagnostic=compile_scene(intent,base,assets,settings,previous)
                SceneRepository(assets).save(scene,'scenes/'+intent.scene_id+'.json')
                export(scene,assets,'scenes/'+intent.scene_id+'.ply')
                atomic_json(root/'logs'/(intent.scene_id+'.solver.json'),diagnostic)
                states[intent.scene_id]=scene;previous=scene;compiled.append((intent,scene));record['status']='compiled'
            except Exception as exc:
                record.update(status='failed',error=dict(code=getattr(exc,'code','compile_failed'),message=str(exc)))
                run['error']=dict(stage='compile',scene_id=intent.scene_id,**record['error']);break
            finally:atomic_json(root/'run.json',run)
        # A single common overview camera uses the union of compiled states; focus fits only its targets.
        all_bounds={}
        frame_bounds={intent.scene_id:measured_bounds(scene,assets) for intent,scene in compiled}
        for bounds in frame_bounds.values():
            for name,bb in bounds.items():
                if name not in all_bounds:all_bounds[name]=bb
                else:all_bounds[name]=[np.minimum(all_bounds[name][0],bb[0]).tolist(),np.maximum(all_bounds[name][1],bb[1]).tolist()]
        run['stage']='render'
        for intent,scene in compiled:
            try:
                sid=intent.scene_id;shot=make_shot(sid,intent.shot,frame_bounds[sid],all_bounds,config['camera'])
                points,colors=read_vertices(root/'scenes'/(sid+'.ply'),'point_cloud')
                ranges=read_json(root/'scenes'/(sid+'.ranges.json'))['ranges'];ids=np.empty(len(points),np.int32)
                for r in ranges:ids[r['start']:r['end']]=run['instance_ids'][r['instance_id']]
                rgb,identity,depth,shown=render_shot(points,colors,ids,shot,config['camera']);shot['rendered_points']=shown;shot['source_point_count']=len(points)
                atomic_json(root/'shots'/('shot_'+sid+'.json'),shot)
                Image.fromarray(rgb).save(root/'renders'/(sid+'.png'))
                atomic_array(root/'renders'/(sid+'.ids.npy'),identity);atomic_array(root/'renders'/(sid+'.depth.npy'),depth)
                (root/'narration'/(sid+'.txt')).write_text(intent.narration['text']+'\n')
                if not check_shot(shot,frame_bounds[sid],identity,run['instance_ids'],config['camera'])['passed']:raise SpatialError('shot_validation_failed','target cropped/occluded in actual render')
                records[sid]['status']='rendered'
            except Exception as exc:
                records[intent.scene_id].update(status='failed',error=dict(code=getattr(exc,'code','render_failed'),message=str(exc)))
                run['error']=dict(stage='render',scene_id=intent.scene_id,**records[intent.scene_id]['error'])
            finally:atomic_json(root/'run.json',run)
        run['stage']='validation'
        paths=[p for p in root.rglob('*') if p.is_file() and p.name not in ('run.json','validation.json','revalidation.json','storyboard.html')]
        run['artifacts']=[dict(path=str(p.relative_to(root)),sha256=sha256(p)) for p in sorted(paths)]
        atomic_json(root/'run.json',run)
        report=validate_story(root,LayoutAssets(root))
        for item in report['scenes']:
            if item['scene_id'] in records:records[item['scene_id']]['status']='succeeded' if item['passed'] else 'failed'
        run.update(status=report['status'],stage='completed' if report['passed'] else 'failed',exit_code=0 if report['passed'] else 1,finished_at=now(),duration_seconds=time.monotonic()-start)
        atomic_json(root/'validation.json',report);atomic_json(root/'run.json',run);write_storyboard(root,plan,report,run)
        return report
    except Exception as exc:
        run.update(status='failed',stage='failed',exit_code=1,finished_at=now(),duration_seconds=time.monotonic()-start,error=dict(code=getattr(exc,'code','invalid_input'),message=str(exc)))
        report=dict(schema_version=1,passed=False,status='failed',scenes=[],error=run['error'])
        atomic_json(root/'run.json',run);atomic_json(root/'validation.json',report);write_storyboard(root,plan,report,run)
        return report


def validate(root):
    root=Path(root).resolve()
    if not root.is_dir():raise FileNotFoundError('missing story run')
    report=validate_story(root,LayoutAssets(root));atomic_json(root/'revalidation.json',report)
    if not report['passed']:
        run=read_json(root/'run.json');run.update(status=report['status'],exit_code=1);atomic_json(root/'run.json',run)
    return report
