"""P7 typed, copy-on-write edits. Publish a version only after full validation."""
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal, Union
import copy, fcntl, json, os, shutil, time
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from src.scene.models import SceneDocument,now
from src.scene.storage import atomic_json, controlled, read_json, read_vertices, sha256, SceneRepository
from src.scene.composition import export
from .contracts import StoryPlan
from .compiler import compile_scene
from .spatial_service import LayoutAssets
from .spatial_validation import measured_bounds
from .camera import make_shot, render_shot
from .service import atomic_array
from .story_validation import validate_story
from .export import write_storyboard


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)

class RelationEdit(Strict):
    kind: Literal['relation']
    scene_id: str
    constraint_id: str
    subject_instance_id: str
    object_instance_id: str
    relation: Literal['left_of', 'right_of', 'near', 'far']
    parameters: dict[str, float] = Field(default_factory=dict)

class ShotEdit(Strict):
    kind: Literal['shot']
    scene_id: str
    template: Literal['overview', 'focus']
    target_ids: list[str] = Field(default_factory=list)
    complete_display: bool = True

class NarrationEdit(Strict):
    kind: Literal['narration']
    scene_id: str
    text: str = Field(min_length=1, max_length=2000)

Operation = Annotated[Union[RelationEdit, ShotEdit, NarrationEdit], Field(discriminator='kind')]

class EditPatch(Strict):
    schema_version: Literal[1] = 1
    base_version: int = Field(ge=1)
    operations: list[Operation] = Field(min_length=1, max_length=9)
    locked_scene_ids: list[str] | None = None


def language_patch(request, base_version, locked_scene_ids, base, catalog, provider, evidence):
    """One bounded call, no regex/fixed-answer fallback; CLI main thread only."""
    from .llm_provider import deadline
    evidence = Path(evidence); evidence.mkdir(parents=True, exist_ok=True)
    schema = EditPatch.model_json_schema()
    messages = [dict(role='system', content=(
        'Return one JSON EditPatch conforming exactly to the provided schema. '
        'Edit only explicitly requested scenes/fields. Keep all other scenes locked. '
        'Allowed kinds: narration, shot, relation. No event or pose/matrix editing. '
        'Replace relation by existing constraint_id. Use exact registered instance IDs. '
        'Do not invent narration when only a camera/relation change was requested. '
        'Do not remove locks. Copy base_version and locked_scene_ids exactly. '
        'No markdown. Schema: '+json.dumps(schema))),
        dict(role='user', content=json.dumps(dict(request=request, base_version=base_version,
            locked_scene_ids=locked_scene_ids, catalog=catalog,
            plan=read_json(Path(base)/'story_plan.json')), ensure_ascii=False))]
    atomic_json(evidence/'messages.json', messages); atomic_json(evidence/'schema.json', schema)
    limits=dict(call_timeout_seconds=90, max_output_tokens=1800)
    with deadline(95):response=provider.generate(messages,schema,limits)
    (evidence/'raw_response.txt').write_text(response.content)
    atomic_json(evidence/'provider.json',{**provider.metadata(), **response.metadata})
    patch=EditPatch.model_validate(json.loads(response.content))
    if patch.base_version != base_version or patch.locked_scene_ids != locked_scene_ids:
        raise ValueError('model changed base_version or locks')
    atomic_json(evidence/'patch.json',patch.model_dump())
    return patch


def file_hashes(root):
    return {str(p.relative_to(root)):sha256(p) for p in Path(root).rglob('*') if p.is_file()}

def frame_files(root,sid):
    keys={f'scenes/{sid}.json',f'scenes/{sid}.ply',f'scenes/{sid}.ranges.json',
              f'shots/shot_{sid}.json',f'narration/{sid}.txt',
              f'renders/{sid}.png',f'renders/{sid}.ids.npy',f'renders/{sid}.depth.npy'}
    return {k:sha256(Path(root)/k) for k in sorted(keys) if (Path(root)/k).is_file()}

def _clone(source,destination):
    # Reused files are immutable hard links. Every later write detaches first.
    def link(src,dst):
        if Path(src).is_symlink():raise ValueError('symlink in source version')
        try:os.link(src,dst)
        except OSError:shutil.copyfile(src,dst)
        return dst
    shutil.copytree(source,destination,copy_function=link)

def _detach(path):
    Path(path).unlink(missing_ok=True)

def edit_run(base,out,patch):
    """Recompile targeted geometry only; validate every persisted scene/event."""
    base=Path(base).resolve();out=Path(out).resolve();started=time.monotonic();started_at=now()
    if out.exists():raise FileExistsError('edit output already exists')
    original_hashes=file_hashes(base)
    original=read_json(base/'story_plan.json');raw=copy.deepcopy(original)
    ids=[s['scene_id'] for s in raw['scenes']];targets={op.scene_id for op in patch.operations}
    locks=set(patch.locked_scene_ids if patch.locked_scene_ids is not None else set(ids)-targets)
    report=dict(schema_version=1,status='conflict',base_version=patch.base_version,
        version=patch.base_version+1,expected_changes=[],unexpected_changes=[],
        locked_scene_ids=sorted(locks),recompiled_scenes=[],rendered_scenes=[],
        source_assets_reconstructed=False,request_satisfied=False)
    _clone(base,out)
    # Never retain a parent's P7 diff/evidence as the new version's own result.
    for n in ['run.json','validation.json','revalidation.json','storyboard.html','edit-report.json']:_detach(out/n)
    try:
        if not targets <= set(ids) or not locks <= set(ids):raise ValueError('unknown scene')
        if targets & locks:raise ValueError('cannot edit locked scene: '+','.join(sorted(targets&locks)))
        seen=set();geometry=set();shots=set();narration=set()
        for op in patch.operations:
            row=next(s for s in raw['scenes'] if s['scene_id']==op.scene_id)
            key=(op.scene_id,op.kind,getattr(op,'constraint_id',''))
            if key in seen:raise ValueError('duplicate edit field')
            seen.add(key);report['expected_changes'].append(dict(scene_id=op.scene_id,field=op.kind))
            if op.kind=='narration':
                if not op.text.strip():raise ValueError('empty narration')
                row['narration']=dict(text=op.text,entity_refs=[],fact_claims=[],creative_text='')
                narration.add(op.scene_id)
            elif op.kind=='shot':
                updated=dict(template=op.template,target_ids=op.target_ids,complete_display=op.complete_display)
                if row['shot']!=updated:shots.add(op.scene_id)
                row['shot']=updated
            else:
                old=next((c for c in row['constraints'] if c['constraint_id']==op.constraint_id),None)
                if old is None:raise ValueError('relation edits require an existing constraint_id')
                old.clear();old.update(constraint_id=op.constraint_id,subject_instance_id=op.subject_instance_id,
                    object_instance_id=op.object_instance_id,relation=op.relation,parameters=op.parameters,frame='reference_ground',schema_version=1)
                geometry.add(op.scene_id)
        config=read_json(out/'config.json');assets=LayoutAssets(out)
        initial=SceneDocument(**read_json(out/'initial_scene.json'));initial.scene_id='initial'
        plan=StoryPlan.parse(raw,initial,assets,config['spatial'],config.get('events',{}).get('min_distance_change',.2))
        atomic_json(out/'story_plan.json',plan.to_dict())
        atomic_json(out/'input_snapshot.json',dict(plan=plan.to_dict(),source_scene_sha256=sha256(out/'initial_scene.json')))
        states={'initial':initial};changed_geometry=set();previous=initial
        for intent in plan.scenes:
            sid=intent.scene_id
            old=SceneDocument(**read_json(base/f'scenes/{sid}.json'));scene=old
            if sid in geometry:
                scene,diagnostic=compile_scene(intent,states[intent.base_scene_id],assets,config['spatial'],previous)
                report['recompiled_scenes'].append(sid)
                atomic_json(out/f'logs/{sid}.solver.json',diagnostic)
                if asdict(scene)['instances'] != asdict(old)['instances']:
                    changed_geometry.add(sid)
                    _detach(out/f'scenes/{sid}.json');_detach(out/f'scenes/{sid}.ply');_detach(out/f'scenes/{sid}.ranges.json')
                    SceneRepository(assets).save(scene,f'scenes/{sid}.json');export(scene,assets,f'scenes/{sid}.ply')
                else:scene=old
            states[sid]=scene;previous=scene
        all_bounds={};bounds={sid:measured_bounds(states[sid],assets) for sid in ids}
        for values in bounds.values():
            for name,bb in values.items():
                if name not in all_bounds:all_bounds[name]=bb
                else:all_bounds[name]=[np.minimum(all_bounds[name][0],bb[0]).tolist(),np.maximum(all_bounds[name][1],bb[1]).tolist()]
        run=read_json(base/'run.json');run.update(pipeline='storyboard_edit_p7',status='running',stage='validation',artifacts=[],edit_base_version=patch.base_version,source_run_sha256=sha256(base/'run.json'),started_at=started_at)
        run['input_hashes']={p:sha256(out/p) for p in ['input_snapshot.json','config.json','initial_scene.json']}
        code_root=Path(__file__).resolve().parents[2]
        run['code_sha256']={str(p.relative_to(code_root)):sha256(p) for p in sorted((code_root/'src/storyboard').glob('*.py'))}
        for intent in plan.scenes:
            sid=intent.scene_id
            if sid in changed_geometry|shots:
                shot=make_shot(sid,intent.shot,bounds[sid],all_bounds,config['camera'])
                p,c=read_vertices(out/f'scenes/{sid}.ply','point_cloud');ranges=read_json(out/f'scenes/{sid}.ranges.json')['ranges'];identity=np.empty(len(p),np.int32)
                for v in ranges:identity[v['start']:v['end']]=run['instance_ids'][v['instance_id']]
                rgb,identity,depth,shown=render_shot(p,c,identity,shot,config['camera']);shot.update(rendered_points=shown,source_point_count=len(p))
                for rel in [f'shots/shot_{sid}.json',f'renders/{sid}.png',f'renders/{sid}.ids.npy',f'renders/{sid}.depth.npy']:_detach(out/rel)
                atomic_json(out/f'shots/shot_{sid}.json',shot);Image.fromarray(rgb).save(out/f'renders/{sid}.png')
                atomic_array(out/f'renders/{sid}.ids.npy',identity);atomic_array(out/f'renders/{sid}.depth.npy',depth)
                report['rendered_scenes'].append(sid)
            if sid in narration:
                _detach(out/f'narration/{sid}.txt');(out/f'narration/{sid}.txt').write_text(intent.narration['text']+'\n')
        run['artifacts']=[dict(path=str(p.relative_to(out)),sha256=sha256(p)) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name not in {'run.json','validation.json','revalidation.json','storyboard.html','edit-report.json'}]
        atomic_json(out/'run.json',run)
        validation=validate_story(out,assets);atomic_json(out/'validation.json',validation)
        # Include all unchanged scenes, not merely explicitly locked ones.
        for sid in set(ids)-targets:
            if frame_files(base,sid)!=frame_files(out,sid):report['unexpected_changes'].append(sid+': untouched frame changed')
        for sid in shots-geometry:
            if sha256(base/f'scenes/{sid}.json')!=sha256(out/f'scenes/{sid}.json'):report['unexpected_changes'].append(sid+': camera changed pose')
        report.update(validation=validation,locked_scene_hashes={sid:frame_files(out,sid) for sid in locks},
            unchanged_scene_count=len(set(ids)-targets),narration_semantics='manual_review_required' if narration or geometry else 'unchanged_text_not_reinterpreted')
        report['changes']=[];report['artifact_changes']=[]
        allowed={(op.scene_id,{'relation':'constraints','shot':'shot','narration':'narration'}[op.kind]) for op in patch.operations}
        for old,new in zip(original['scenes'],plan.to_dict()['scenes']):
            sid=old['scene_id']
            for field in old:
                if old[field]!=new[field]:
                    report['changes'].append(dict(scene_id=sid,field=field,before=old[field],after=new[field]))
                    if (sid,field) not in allowed:report['unexpected_changes'].append(sid+': unrequested '+field)
            a,b=frame_files(base,sid),frame_files(out,sid)
            for path in sorted(set(a)|set(b)):
                if a.get(path)!=b.get(path):report['artifact_changes'].append(dict(path=path,before=a.get(path),after=b.get(path)))
        total=sum(len(frame_files(base,sid)) for sid in locks)
        kept=sum(sum(frame_files(base,sid).get(k)==v for k,v in frame_files(out,sid).items()) for sid in locks)
        report['locked_artifact_preservation']=dict(unchanged=kept,total=total,rate=kept/total if total else None)
        report['request_satisfied']=validation['passed'] and not report['unexpected_changes']
        report['status']='succeeded' if report['request_satisfied'] else 'conflict'
        if not validation['passed']:report['conflicts']=[dict(scene_id=s['scene_id'],errors=s['errors'],events=s.get('event_checks',[]),inheritance=s.get('inheritance_errors',[]),checks=s.get('checks',[])) for s in validation['scenes'] if not s['passed']]
        run.update(status=report['status'],stage='completed',exit_code=0 if report['request_satisfied'] else 1,finished_at=now(),duration_seconds=time.monotonic()-started)
        atomic_json(out/'run.json',run);write_storyboard(out,plan,validation,run)
        # The shared P4 exporter labels handwritten input; distinguish edited output.
        page=out/'storyboard.html';page.write_text(page.read_text().replace('P4 / HANDWRITTEN STORYPLAN / NO LLM','P7 / VERSIONED EDIT / SEE EDIT REPORT').replace('旁白为手写。','旁白可能经用户或模型修改，语义待人工核对。'))
    except Exception as exc:
        evidence=getattr(exc,'evidence',{})
        failed=[c['constraint_id']+' ('+c['relation']+')' for c in evidence.get('checks',[]) if c.get('status')=='failed']
        failed += [e['event_id']+' ('+e['action']+')' for e in evidence.get('event_checks',[]) if not e.get('passed')]
        scene_id=getattr(locals().get('intent'),'scene_id',None)
        report.update(status='conflict',error=dict(code=getattr(exc,'code','edit_conflict'),message=str(exc),
            scene_id=scene_id,evidence=evidence))
        report['conflict_summary']=('幕 '+str(scene_id)+' 未满足：'+', '.join(failed)+'；现有跨幕事件和锁定幕均未被自动放宽。') if failed else str(exc)
    report['source_version_unchanged']=file_hashes(base)==original_hashes
    if not report['source_version_unchanged']:raise RuntimeError('source version changed')
    report['seconds']=time.monotonic()-started
    atomic_json(out/'edit-report.json',report)
    return report


class EditStore:
    """Optimistic version check plus one atomic pointer; failed proposals stay separate."""
    def __init__(self,story_root):
        self.story_root=Path(story_root).resolve();self.jobs=self.story_root.parent
        self.index=self.story_root/'versions.json'

    def state(self):
        if self.index.exists():return read_json(self.index)
        pipeline=self.story_root/'pipeline';run=read_json(pipeline/'run.json')
        if run['status']!='succeeded':raise ValueError('only completed stories can be edited')
        compiled=pipeline/run['accepted_attempt']/'compiled'
        return dict(schema_version=1,current_version=1,versions={'1':str(compiled.relative_to(self.jobs))})

    def root(self,state,version):return controlled(self.jobs,state['versions'][str(version)])

    def apply(self,out,base_version,patch=None,request=None,locked_scene_ids=None,provider=None,evidence=None):
        with (self.story_root/'.edit.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            state=self.state()
            if base_version!=state['current_version']:
                return dict(status='version_conflict',base_version=base_version,current_version=state['current_version'],request_satisfied=False)
            base=self.root(state,base_version)
            if not validate_story(base,LayoutAssets(base))['passed']:
                raise ValueError('base version failed integrity/geometry validation')
            if request is not None:
                patch=language_patch(request,base_version,locked_scene_ids,base,read_json(self.story_root/'catalog.json'),provider,evidence)
            else:patch=EditPatch.model_validate(patch)
            if patch.base_version!=base_version:raise ValueError('patch base_version mismatch')
            report=edit_run(base,out,patch)
            if report['status']=='succeeded':
                version=base_version+1;state['versions'][str(version)]=str(Path(out).resolve().relative_to(self.jobs));state['current_version']=version
                atomic_json(self.index,state)
            report['current_version']=state['current_version']
            atomic_json(Path(out)/'edit-report.json',report)
            return report
