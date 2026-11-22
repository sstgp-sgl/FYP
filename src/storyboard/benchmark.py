"""P8 explicit baselines, shared rendering and bounded/resumable model calls."""
from dataclasses import asdict
from pathlib import Path
from typing import Literal
import copy,json,time,subprocess
import numpy as np
from PIL import Image
import yaml
from pydantic import Field
from src.scene.models import SceneDocument,now
from src.scene.storage import read_json,atomic_json,SceneRepository,read_vertices,sha256
from src.scene.composition import export
from .llm_contracts import StrictModel,Narration,Shot,ModelResponse,Intent,Plan,Event,response_schema
from .llm_prompt import messages_for
from .llm_service import strict_json,parse_response,check_catalog,check_event_references,check_fact_references
from .llm_provider import deadline,ProviderError
from .assets import freeze_assets
from .spatial_service import LayoutAssets
from .contracts import StoryPlan
from .story_service import compile_story,camera_settings
from .camera import make_shot,render_shot
from .spatial_validation import measured_bounds
from .service import atomic_array
from .export import write_storyboard
from .benchmark_metrics import score,aggregate

PROTOCOL='p8-exploratory-v1'
METHODS={'b0':('direct',1),'b1':('relations',1),'ours':('events',3),
         'b0_budget3':('direct',3),'b1_budget3':('relations',3),
         'no_feedback':('events',1),'no_state':('events',3),'fixed_camera':('events',3)}

class DirectInstance(StrictModel):
    instance_id:str
    translation_xyz:list[float]=Field(min_length=3,max_length=3)
    visible:bool=True
class DirectFrame(StrictModel):
    scene_id:str
    instances:list[DirectInstance]=Field(min_length=2,max_length=3)
    narration:Narration
    shot:Shot=Field(default_factory=Shot)
class DirectPlan(StrictModel):
    schema_version:Literal[1]
    story_id:str
    asset_ids:list[str]
    scenes:list[DirectFrame]=Field(min_length=3,max_length=3)
class DirectResponse(StrictModel):
    status:Literal['ready','needs_clarification']
    plan:DirectPlan|None=None
    question:str|None=None
    reason_code:str|None=None

class RelationIntent(Intent):
    events:list[Event]=Field(default_factory=list,max_length=0)
    visibility:dict[str,bool]=Field(default_factory=dict)
class RelationPlan(Plan):
    scenes:list[RelationIntent]=Field(min_length=3,max_length=3)
class RelationResponse(ModelResponse):
    plan:RelationPlan|None=None


def context(frozen):
    root=Path(frozen);assets=LayoutAssets(root);initial=SceneDocument(**read_json(root/'initial_scene.json'))
    catalog=read_json(root/'catalog.json');check_catalog(catalog,initial,assets)
    config=read_json(root/'config.json');info=[]
    for inst in initial.instances:
        rec=assets.get(inst.asset_id);p,_=assets.points(rec)
        info.append(dict(instance_id=inst.instance_id,asset_id=inst.asset_id,canonical_bounds=[p.min(0).tolist(),p.max(0).tolist()],point_count=len(p),coordinate_frame='right_handed_y_up',unit='arbitrary'))
    return assets,initial,catalog,config,info

def prompt(story,method,initial,catalog,config,info):
    mode,_=METHODS[method]
    settings={**config['spatial'],'event_min_distance_change':config.get('events',{}).get('min_distance_change',.2)}
    schema=DirectResponse.model_json_schema() if mode=='direct' else (RelationResponse.model_json_schema() if mode=='relations' else response_schema())
    if mode=='direct':
        system='''你是三幕静态三维故事规划器。只输出符合 Schema 的 JSON，不要 Markdown。根据用户每幕要求直接给出每个登记实例的绝对 translation_xyz 和 visible，恰好 scene_1、scene_2、scene_3。不能改变 asset_id、尺度或旋转；只使用给定实例。坐标 Y-up，+X为右。所有可见物体落地且地面 AABB 不重叠。near/far 是地面 AABB 间距，默认 near<=0.5U，far>=1.5U；approach/retreat 是相对上一幕的中心距离减少/增加至少0.2U。没有要求改变的对象保持原位，hold 保持时间上一幕位置。以给定真实尺寸和初始姿态计算，不能假设单位立方体。每幕简短中文旁白15–60字，entity_refs仅实例ID，fact_claims=[]，不虚构新对象或动作。shot为overview或focus，focus填写目标实例，complete_display=true；focus只改变镜头。存在歧义、不支持的动作或新对象时 needs_clarification，否则 ready 并包含 plan。只给基础格式/引用保护，程序不会替你修正坐标。'''
        messages=[dict(role='system',content=system+'\nSchema:'+json.dumps(schema,ensure_ascii=False)),dict(role='user',content=json.dumps(dict(user_story=story,confirmed_catalog=catalog,initial_state=asdict(initial),runtime_settings=settings),ensure_ascii=False))]
    else:
        messages=messages_for(story,catalog,asdict(initial),settings)
        if mode=='relations':
            messages[0]['content']=messages[0]['content'].split('\n输出语法 Schema：')[0]+'\n本次方法是 Relations-only + explicit visibility。逐幕输出静态空间关系、镜头和旁白，events 必须始终为空数组，不提供 approach/retreat/hold 事件求解。每幕可用 visibility 字典填写登记实例最终是否可见（布尔值），未写时继承；这是直接静态可见性设置。需要运动时用逐幕关系与距离表达，保持时可用 keep_pose；仍可继承前一幕。隐藏实例不能出现在空间约束或镜头目标。fact_claims只引用本幕constraint_id；可用[]，不要捏造可见性ID。不要输出坐标。此规则覆盖前文要求输出事件的部分。\n输出语法 Schema：'+json.dumps(schema,ensure_ascii=False)
    # Identical geometry, budgets and narration instructions for every method.
    messages[1]['content']+='\nSHARED_ASSET_INFORMATION='+json.dumps(dict(assets=info,camera=config['camera'],narration_characters=[15,60]),ensure_ascii=False)
    messages[0]['content']+='\n本次统一硬性格式：每幕 narration.text 去除首尾空白后必须为15–60个Unicode字符（包括标点）；超出范围会被同一执行器拒绝。'
    return messages,schema

def render_direct(raw,frozen,out):
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    assets,initial,catalog,config,info=context(frozen);freeze_assets(frozen,[i.asset_id for i in initial.instances],root);assets=LayoutAssets(root)
    for name in ['scenes','shots','renders','narration']:(root/name).mkdir()
    parsed=DirectResponse.model_validate(raw)
    if parsed.status!='ready' or parsed.plan is None or parsed.question is not None or parsed.reason_code is not None:raise ValueError('direct plan not ready or inconsistent envelope')
    plan=parsed.plan;ids=[i.instance_id for i in initial.instances]
    if set(plan.asset_ids)!={i.asset_id for i in initial.instances} or len(plan.asset_ids)!=len(set(plan.asset_ids)):raise ValueError('direct asset mismatch')
    intents=[];states={}
    for index,row in enumerate(plan.scenes,1):
        sid='scene_'+str(index)
        if row.scene_id!=sid or len(row.instances)!=len(ids) or {i.instance_id for i in row.instances}!=set(ids):raise ValueError('direct frame/reference mismatch')
        if any(v not in ids for v in row.narration.entity_refs) or row.narration.fact_claims:raise ValueError('invalid direct narration references')
        scene=SceneDocument(**asdict(initial));scene.scene_id=sid
        mapping={i.instance_id:i for i in row.instances}
        for inst in scene.instances:
            inst.transform.translation_xyz=mapping[inst.instance_id].translation_xyz;inst.visible=mapping[inst.instance_id].visible
        SceneRepository(assets).save(scene,f'scenes/{sid}.json');export(scene,assets,f'scenes/{sid}.ply');states[sid]=scene
        intents.append(dict(scene_id=sid,base_scene_id='initial',constraints=[],events=[],narration=row.narration.model_dump(),shot=row.shot.model_dump()))
    envelope=dict(schema_version=1,story_id=plan.story_id,asset_ids=plan.asset_ids,premise='B0 direct coordinates; no deterministic semantic correction.',scenes=intents)
    display=StoryPlan.parse(envelope,initial,assets,config['spatial']);atomic_json(root/'story_plan.json',display.to_dict())
    atomic_json(root/'initial_scene.json',asdict(initial));atomic_json(root/'config.json',config);atomic_json(root/'input_snapshot.json',dict(plan=envelope))
    bounds={sid:measured_bounds(s,assets) for sid,s in states.items()};union={}
    for bb in bounds.values():
        for name,v in bb.items():union[name]=v if name not in union else [np.minimum(union[name][0],v[0]).tolist(),np.maximum(union[name][1],v[1]).tolist()]
    run=dict(pipeline='p8_direct',status='executed',instance_ids={name:i+1 for i,name in enumerate(sorted(ids))},artifacts=[])
    for intent in display.scenes:
        sid=intent.scene_id;shot=make_shot(sid,intent.shot,bounds[sid],union,config['camera'])
        p,c=read_vertices(root/f'scenes/{sid}.ply','point_cloud');identity=np.empty(len(p),np.int32)
        for entry in read_json(root/f'scenes/{sid}.ranges.json')['ranges']:identity[entry['start']:entry['end']]=run['instance_ids'][entry['instance_id']]
        rgb,identity,depth,shown=render_shot(p,c,identity,shot,config['camera']);shot.update(rendered_points=shown,source_point_count=len(p))
        atomic_json(root/f'shots/shot_{sid}.json',shot);Image.fromarray(rgb).save(root/f'renders/{sid}.png');atomic_array(root/f'renders/{sid}.ids.npy',identity);atomic_array(root/f'renders/{sid}.depth.npy',depth)
        (root/f'narration/{sid}.txt').write_text(intent.narration['text']+'\n')
    run['artifacts']=[dict(path=str(p.relative_to(root)),sha256=sha256(p)) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in {'run.json','storyboard.html'}];atomic_json(root/'run.json',run)
    write_storyboard(root,display,dict(status='executed, awaiting independent task audit',scenes=[]),run)
    page=root/'storyboard.html';page.write_text(page.read_text().replace('P4 / HANDWRITTEN STORYPLAN / NO LLM','P8 / B0 DIRECT / SEE TASK AUDIT').replace('旁白为手写。','旁白来自真实模型或明确标注的测试响应。'))
    return True

def execute_response(raw,frozen,out,method):
    for row in (raw.get('plan') or {}).get('scenes',[]):
        if not 15<=len(row.get('narration',{}).get('text','').strip())<=60:raise ValueError('narration length must be 15..60 Unicode characters for every method')
    mode,_=METHODS[method]
    if mode=='direct':return render_direct(raw,frozen,out)
    if mode=='relations':
        if any(s.get('events') for s in (raw.get('plan') or {}).get('scenes',[])):raise ValueError('B1 relation-only response may not use event solver')
        parsed=RelationResponse.model_validate(raw).model_dump(exclude_none=True,exclude_unset=True)
    else:parsed=parse_response(json.dumps(raw,ensure_ascii=False))
    if parsed['status']!='ready':raise ValueError('needs_clarification')
    if parsed.get('question') is not None or parsed.get('reason_code') is not None:raise ValueError('ready plan may not include clarification')
    assets,initial,catalog,config,info=context(frozen);plan=copy.deepcopy(parsed['plan'])
    if mode=='relations':
        visibility={i.instance_id:i.visible for i in initial.instances}
        for n,s in enumerate(plan['scenes']):
            requested=s.pop('visibility',{});s['events']=[]
            for name,value in requested.items():
                if name not in visibility:raise ValueError('unknown visibility instance')
                if visibility[name]!=value:s['events'].append(dict(event_id=f'b1_visibility_{n}_{name}',action='appear' if value else 'disappear',subject_instance_id=name))
                visibility[name]=value
    check_event_references(plan,[i.instance_id for i in initial.instances]);check_fact_references(plan)
    if [s['scene_id'] for s in plan['scenes']]!=['scene_1','scene_2','scene_3']:raise ValueError('scene IDs must be scene_1..scene_3')
    if method=='no_state':
        for s in plan['scenes']:s['base_scene_id']='initial'
    if method=='fixed_camera':
        for s in plan['scenes']:s['shot']=dict(template='overview',target_ids=[],complete_display=True)
    out=Path(out);atomic_json(out.parent/'execution-plan.json',plan)
    cfg=out.parent/'config.yaml';cfg.write_text(yaml.safe_dump(config,allow_unicode=True))
    result=compile_story(out.parent/'execution-plan.json',frozen,Path(frozen)/'initial_scene.json',out,cfg,True)
    return result['passed']

def run_case(task,method,frozen,out,provider,max_calls=None,replay_from=None):
    root=Path(out);root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    assets,initial,catalog,config,info=context(frozen)
    messages,schema=prompt(task['story'],method,initial,catalog,config,info)
    budget=max_calls or METHODS[method][1];calls=0;tokens=0;attempts=[];accepted=None;latest=None
    if type(budget) is not int or not 1<=budget<=3:raise ValueError('invalid call budget')
    run=dict(protocol=PROTOCOL,method=method,task_id=task['id'],status='running',started_at=now(),max_calls=budget,attempts=attempts,mode=provider.metadata()['mode'],replay_from=replay_from)
    atomic_json(root/'run.json',run)
    for index in range(budget):
        attempt=root/f'attempt_{index:02d}';attempt.mkdir();atomic_json(attempt/'messages.json',messages);atomic_json(attempt/'schema.json',schema)
        entry=dict(directory=attempt.name,status='calling');attempts.append(entry);atomic_json(root/'run.json',run)
        limits=dict(max_output_tokens=3072,call_timeout_seconds=120)
        calls+=int(provider.metadata()['mode']=='live_model')
        run['calls']=calls
        atomic_json(attempt/'provider.json',dict(**provider.metadata(),status='calling'))
        try:
            with deadline(125):response=provider.generate(messages,schema,limits)
            metadata={**provider.metadata(),**response.metadata};tokens+=int(metadata.get('usage',{}).get('total_tokens',0)) if provider.metadata()['mode']=='live_model' else 0
            (attempt/'raw_response.txt').write_text(response.content);atomic_json(attempt/'provider.json',metadata)
            raw=strict_json(response.content);atomic_json(attempt/'parsed.json',raw)
            if raw.get('status')=='needs_clarification':entry.update(status='needs_clarification');break
            latest=attempt/'compiled';executed=execute_response(raw,frozen,latest,method)
            if executed and method=='b0_budget3':
                generic=score(latest,dict(requirements=[],instance_ids=[i.instance_id for i in initial.instances]),True)
                executed=all(f['passed'] for f in generic['frames']) and not generic['artifact_errors']
            entry.update(status='executed' if executed else 'execution_failed')
            if executed:accepted=attempt.name;break
            failure=read_json(latest/'validation.json') if (latest/'validation.json').exists() else dict(frames=generic['frames'],artifact_errors=generic['artifact_errors'])
            entry['error']=dict(message='executor validation failed',details=failure)
        except Exception as exc:
            entry.update(status='failed',error=dict(code=getattr(exc,'code','invalid_response'),message=str(exc)))
            if isinstance(exc,ProviderError):break
        finally:
            atomic_json(attempt/'result.json',entry);atomic_json(root/'run.json',run)
        if index+1<budget:
            if (attempt/'raw_response.txt').exists():messages.append(dict(role='assistant',content=(attempt/'raw_response.txt').read_text()))
            messages.append(dict(role='user',content=json.dumps(dict(repair_request=True,execution_errors=entry.get('error'),original_user_story=task['story'],instruction='修正格式/引用/执行错误并检查原故事，不要发明新要求。只输出完整 JSON。'),ensure_ascii=False)))
    chosen=root/accepted/'compiled' if accepted else (latest or root/'missing')
    audited=score(chosen,task,accepted is not None);atomic_json(root/'audit.json',audited)
    run.update(status='succeeded' if audited['story_success'] else 'failed',accepted_attempt=accepted,seconds=time.monotonic()-started,calls=calls,tokens=tokens,finished_at=now())
    run['artifacts']=[dict(path=str(p.relative_to(root)),sha256=sha256(p)) for p in root.rglob('*') if p.is_file() and 'compiled' not in p.relative_to(root).parts and p.name not in {'run.json','result.json'}]
    atomic_json(root/'run.json',run)
    return dict(task_id=task['id'],method=method,group_id=task['group_id'],calls=calls,tokens=tokens,seconds=run['seconds'],score=audited,run_directory=str(root),accepted_attempt=accepted,mode=run['mode'])
