"""P8 scoring from frozen user requirements, never model-declared fact_claims."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
from PIL import Image
from src.scene.models import SceneDocument
from src.scene.storage import read_json,sha256,controlled
from src.scene.validation import validate_export
from .spatial_service import LayoutAssets
from .spatial_contracts import DEFAULTS,SpatialConstraint
from .spatial_validation import measured_bounds,check_constraint
from .story_validation import event_check
from .camera import check_shot


def score(compiled,task,executed):
    """Every frozen requirement remains in its denominator on any failure."""
    root=Path(compiled);facts=[];states={};bounds={};frames=[];references=[];locked_observed=0;locked_changed=0
    try:
        assets=LayoutAssets(root);initial=SceneDocument(**read_json(root/'initial_scene.json'))
        states['initial']=initial;bounds['initial']=measured_bounds(initial,assets)
        config=read_json(root/'config.json');settings={**DEFAULTS,**config['spatial']}
        run=read_json(root/'run.json');registered={i.instance_id:(i.asset_id,i.asset_hash) for i in initial.instances}
    except Exception:
        registered={n:None for n in task.get('instance_ids',['a','b'])};config={};settings=DEFAULTS
    for n in range(1,4):
        sid='scene_'+str(n);item=dict(scene_id=sid,passed=False);reference_recorded=False
        try:
            scene=SceneDocument(**read_json(root/f'scenes/{sid}.json'));states[sid]=scene
            present=set()
            for inst in scene.instances:
                present.add(inst.instance_id);references.append(registered.get(inst.instance_id)==(inst.asset_id,inst.asset_hash))
            references.extend(False for name in registered if name not in present)
            reference_recorded=True;bounds[sid]=measured_bounds(scene,assets)
            poses={i.instance_id:i for i in scene.instances};base={i.instance_id:i for i in initial.instances}
            invariants=[]
            for name in bounds[sid]:invariants.append(check_constraint(SpatialConstraint('ground',name,'on_ground'),bounds[sid],poses,base,settings)['status']=='passed')
            for i,name in enumerate(bounds[sid]):
                for other in list(bounds[sid])[i+1:]:invariants.append(check_constraint(SpatialConstraint('overlap',name,'non_overlap',other),bounds[sid],poses,base,settings)['status']=='passed')
            for name,pose in poses.items():
                if name not in base:invariants.append(False);continue
                invariants.append(pose.transform.uniform_scale==base[name].transform.uniform_scale and pose.transform.rotation_xyzw==base[name].transform.rotation_xyzw)
            exported=validate_export(assets,f'scenes/{sid}.json',f'scenes/{sid}.ply')['passed']
            shot=read_json(root/f'shots/shot_{sid}.json')
            with Image.open(root/f'renders/{sid}.png') as im:im.load();size=im.size
            identity=np.load(root/f'renders/{sid}.ids.npy',allow_pickle=False)
            depth=np.load(root/f'renders/{sid}.depth.npy',allow_pickle=False)
            foreground=identity>0
            valid_buffers=identity.shape==(size[1],size[0]) and depth.shape==identity.shape and np.isfinite(depth[foreground]).all() and (depth[foreground]>=shot['near']).all() and (depth[foreground]<=shot['far']).all()
            visible=check_shot(shot,bounds[sid],identity,run['instance_ids'],config['camera'])['passed']
            item.update(passed=bool(all(invariants) and exported and valid_buffers and visible),invariants=invariants,visible=visible)
        except Exception as exc:
            if not reference_recorded:references.extend(False for _ in registered)
            item['error']=type(exc).__name__+': '+str(exc)
        frames.append(item)
    for index,fact in enumerate(task['requirements']):
        item=dict(requirement=index,kind=fact['kind'],scene=fact['scene'],passed=False)
        sid='scene_'+str(fact['scene']);previous='initial' if fact['scene']==1 else 'scene_'+str(fact['scene']-1)
        try:
            scene=states[sid];poses={i.instance_id:i for i in scene.instances};old={i.instance_id:i for i in states[previous].instances}
            if fact['kind']=='relation':
                c=SpatialConstraint('oracle_'+str(index),fact['subject'],fact['relation'],fact.get('object'),fact.get('parameters',{}))
                result=check_constraint(c,bounds[sid],poses,old,settings);item.update(passed=result['status']=='passed',measured=result['measured'],expected=result['expected'])
            elif fact['kind']=='event':
                event=dict(event_id='oracle_'+str(index),action=fact['action'],subject_instance_id=fact['subject'])
                if fact['action'] in ('approach','retreat'):event.update(object_instance_id=fact['object'],min_change=fact.get('min_change',.2))
                result=event_check(event,scene,states[previous],bounds[sid],bounds[previous],settings);item.update(passed=result['passed'],measured=result['measured'])
            elif fact['kind']=='visibility':item['passed']=poses[fact['subject']].visible is fact['visible']
            elif fact['kind']=='pose':
                reference=states[fact.get('reference','initial')];before=next(i for i in reference.instances if i.instance_id==fact['subject'])
                current=poses[fact['subject']].transform;prior=before.transform
                locked_observed+=5
                locked_changed+=int(np.count_nonzero(np.abs(np.array(current.translation_xyz)-prior.translation_xyz)>settings['translation_tolerance']))+int(abs(current.uniform_scale-prior.uniform_scale)>settings['scale_tolerance'])+int(abs(abs(np.dot(current.rotation_xyzw,prior.rotation_xyzw))-1)>1e-12)
                item['passed']=bool(np.allclose(current.translation_xyz,prior.translation_xyz,atol=settings['translation_tolerance'],rtol=0) and abs(current.uniform_scale-prior.uniform_scale)<=settings['scale_tolerance'] and abs(abs(np.dot(current.rotation_xyzw,prior.rotation_xyzw))-1)<=1e-12)
            elif fact['kind']=='shot':
                shot=read_json(root/f'shots/shot_{sid}.json')
                item['passed']=shot['template']==fact['template'] and (fact['template']!='focus' or shot['target_ids']==fact['targets'])
                item['passed']=item['passed'] and frames[fact['scene']-1].get('visible',False)
            else:raise ValueError('unknown frozen requirement kind')
        except Exception as exc:item['error']=type(exc).__name__+': '+str(exc)
        facts.append(item)
    artifact_errors=[]
    if (root/'run.json').exists():
        if not read_json(root/'run.json').get('artifacts'):artifact_errors.append('missing_manifest')
        for a in read_json(root/'run.json').get('artifacts',[]):
            try:
                if sha256(controlled(root,a['path']))!=a['sha256']:artifact_errors.append(a['path'])
            except Exception:artifact_errors.append(a['path'])
    def ratio(rows):return dict(passed=sum(bool(x) for x in rows),total=len(rows),rate=sum(bool(x) for x in rows)/len(rows) if rows else None)
    categories={kind:ratio([f['passed'] for f in facts if f['kind']==kind]) for kind in sorted({f['kind'] for f in facts})}
    return dict(schema_version=1,story_success=bool(executed and all(f['passed'] for f in facts) and all(f['passed'] for f in frames) and all(references) and not artifact_errors),
        requirements=ratio([f['passed'] for f in facts]),categories=categories,references=ratio(references),frames=frames,facts=facts,artifact_errors=artifact_errors,
        locked_fields=dict(expected=5*sum(f['kind']=='pose' for f in task['requirements']),observed=locked_observed,changed=locked_changed),
        narration_fact_support=None,narration_note='Requires human reading of full captions and images; fact_claims is never a correctness score.')


def aggregate(rows,seed=20261123):
    """Aggregate repeats within task, then groups. Bootstrap groups, not frames."""
    result={};methods=sorted({r['method'] for r in rows})
    for method in methods:
        selected=[r for r in rows if r['method']==method];tasks={}
        for row in selected:tasks.setdefault((row['group_id'],row['task_id']),[]).append(row)
        values={k:float(np.mean([r['score']['story_success'] for r in v])) for k,v in tasks.items()}
        groups={}
        for (group,task),value in values.items():groups.setdefault(group,[]).append(value)
        rates=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(seed)
        # Merge source-pair groups sharing any underlying photo into one cluster.
        components=[set([g]) for g in groups];sources={g:set().union(*(set(r.get('source_groups') or [g]) for r in selected if r['group_id']==g)) for g in groups}
        changed=True
        while changed:
            changed=False
            for i in range(len(components)):
                for j in range(i+1,len(components)):
                    if set().union(*(sources[g] for g in components[i])) & set().union(*(sources[g] for g in components[j])):
                        components[i]|=components.pop(j);changed=True;break
                if changed:break
        independent=np.array([np.mean([np.mean(groups[g]) for g in c]) for c in components])
        # One shared source group cannot support a meaningful group-level CI.
        ci=np.quantile([rng.choice(independent,len(independent),replace=True).mean() for _ in range(1000)],[.025,.975]).tolist() if len(independent)>1 else None
        def balanced(metric):
            means=[]
            for values_for_task in tasks.values():
                vals=[metric(r['score']) for r in values_for_task];vals=[v for v in vals if v is not None]
                if vals:means.append(float(np.mean(vals)))
            return float(np.mean(means)) if means else None
        result[method]=dict(runs=len(selected),tasks=len(tasks),asset_groups=len(groups),story_success_task_mean=float(np.mean(list(values.values()))),group_mean=float(rates.mean()),group_bootstrap_95_ci=ci,
            independent_source_components=len(components),spatial_requirement_rate=balanced(lambda s:s['categories'].get('relation',{}).get('rate')),event_requirement_rate=balanced(lambda s:s['categories'].get('event',{}).get('rate')),
            reference_correct_rate=balanced(lambda s:s['references']['rate']),target_visible_frame_rate=balanced(lambda s:sum(bool(f.get('visible')) for f in s['frames'])/3),
            locked_field_change_rate=balanced(lambda s:s['locked_fields']['changed']/s['locked_fields']['observed'] if s['locked_fields']['observed'] else None),locked_field_coverage=balanced(lambda s:s['locked_fields']['observed']/s['locked_fields']['expected'] if s['locked_fields']['expected'] else None),narration_fact_support=None,
            calls=sum(r['calls'] for r in selected),reported_tokens=sum(r.get('tokens',0) for r in selected),
            median_seconds=float(np.median([r['seconds'] for r in selected])),failed_runs=sum(not r['score']['story_success'] for r in selected),
            scope='Exploratory, shared asset sources; asset groups may not be independent. CI descriptive only.')
    return result
