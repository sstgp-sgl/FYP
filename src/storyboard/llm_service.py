"""P5 bounded understanding -> strict syntax -> P4 compile/render -> independent checks."""
from pathlib import Path
from dataclasses import asdict
import json,time,html
import yaml
from pydantic import ValidationError
from src.scene.storage import read_json,atomic_json,sha256,SceneRepository
from src.scene.models import SceneDocument,now
from .spatial_service import LayoutAssets
from .spatial_contracts import SpatialError,DEFAULTS
from .contracts import StoryPlan
from .story_service import compile_story,validate as validate_compiled
from .llm_contracts import ModelResponse,response_schema,RESPONSE_CONTRACT_VERSION
from .llm_prompt import messages_for,PROMPT_VERSION
from .llm_provider import ProviderError,deadline

LIMITS=dict(max_calls=3,call_timeout_seconds=180,max_output_tokens=3072,seed=42)

def strict_json(raw):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate JSON field')
            result[k]=v
        return result
    def reject(_):raise ValueError('non-finite JSON number')
    value=json.loads(raw,parse_constant=reject,object_pairs_hook=unique)
    if not isinstance(value,dict):raise ValueError('response must be an object')
    return value

def parse_response(raw):
    parsed=ModelResponse.model_validate(strict_json(raw))
    if parsed.status=='ready':
        if parsed.plan is None or parsed.question is not None or parsed.reason_code is not None:raise ValueError('ready requires plan only')
    elif parsed.plan is not None or not parsed.question or not parsed.reason_code:
        raise ValueError('clarification requires question/reason_code and no plan')
    return parsed.model_dump(exclude_none=True,exclude_unset=True)

def check_event_references(plan,instance_ids):
    failures=[];registered=set(instance_ids)
    for n,scene in enumerate(plan['scenes']):
        for j,event in enumerate(scene.get('events',[])):
            subject=event['subject_instance_id']
            if subject not in registered:
                failures.append(dict(path=['plan','scenes',n,'events',j,'subject_instance_id'],provided=subject,allowed_ids=sorted(registered)))
            if event['action'] in ('approach','retreat') and (event.get('object_instance_id') not in registered or event.get('object_instance_id')==subject):
                failures.append(dict(path=['plan','scenes',n,'events',j,'object_instance_id'],action=event['action'],provided=event.get('object_instance_id'),allowed_ids=sorted(registered-{subject}),requirement='Both approach and retreat require a different registered target instance.'))
    if failures:raise SpatialError('unknown_reference','event reference invalid; fill the exact indicated field using a permitted instance ID',dict(items=failures))


def check_fact_references(plan):
    failures=[]
    for n,scene in enumerate(plan['scenes']):
        allowed={c['constraint_id'] for c in scene.get('constraints',[])}|{e['event_id'] for e in scene.get('events',[])}
        invalid=[ref for ref in scene['narration'].get('fact_claims',[]) if ref not in allowed]
        if invalid:failures.append(dict(path=['plan','scenes',n,'narration','fact_claims'],invalid_references=invalid,allowed_ids=sorted(allowed)))
    if failures:raise SpatialError('unknown_fact_reference','fact_claims accepts current scene constraint/event IDs, never prose or instance IDs',dict(items=failures))


def check_catalog(catalog,scene,assets):
    if not isinstance(catalog,list) or len(catalog)!=len(scene.instances):raise SpatialError('unconfirmed_names','catalog must cover every input instance')
    expected={i.instance_id:i.asset_id for i in scene.instances};seen=set()
    for row in catalog:
        if not isinstance(row,dict) or set(row)!={'instance_id','asset_id','display_name','name_confirmed','confirmation_source'}:raise SpatialError('unconfirmed_names','invalid name confirmation record')
        iid=row['instance_id']
        if iid in seen or expected.get(iid)!=row['asset_id']:raise SpatialError('unknown_reference','catalog does not match scene instances')
        seen.add(iid)
        if row['name_confirmed'] is not True or not isinstance(row['display_name'],str) or not row['display_name'].strip():raise SpatialError('unconfirmed_names','user-confirmed names required')
        source=assets.get(row['asset_id']).source_kind
        if row['confirmation_source']!='user' and not (source=='synthetic' and row['confirmation_source']=='synthetic/test'):
            raise SpatialError('unconfirmed_names','real asset names require user confirmation')
    return catalog

def _failure(exc,stage):
    if isinstance(exc,ValidationError):return dict(stage=stage,code='schema_error',items=exc.errors(include_input=False,include_url=False))
    return dict(stage=stage,code=getattr(exc,'code','invalid_json' if stage=='json' else 'invalid_response'),message=str(exc) if isinstance(exc,(ValueError,SpatialError,ProviderError)) else type(exc).__name__,evidence=getattr(exc,'evidence',{}))

def _index(root,run):
    links=[]
    for a in run['attempts']:
        d=a['directory'];available=[]
        for path,label in [('messages.json','请求'),('raw_response.txt','原始响应'),('errors.json','错误'),('compiled/storyboard.html','故事板')]:
            if (root/d/path).is_file():available.append('<a href="'+d+'/'+path+'">'+label+'</a>')
        links.append('<li>'+html.escape(d+' · '+a['status'])+' · '+' · '.join(available)+'</li>')
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P5 model run</title><style>body{font:16px/1.8 system-ui;max-width:1100px;margin:35px auto;padding:20px}pre{white-space:pre-wrap}</style><h1>P5 '+html.escape(run['status'])+'</h1><p>'+html.escape(run.get('mode','unknown'))+' · '+html.escape(run['request'])+'</p><ol>'+''.join(links)+'</ol><pre>'+html.escape(json.dumps(run.get('error',{}),ensure_ascii=False,indent=2))+'</pre><p>所有原始调用均保留；未使用模板成功回退。旁白事实的完整语义仍需人工查看。</p>')

def understand(story,catalog,asset_root,scene_path,out,provider,limits=None,config_path=None,allow_provisional=False):
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False)
    limits={**LIMITS,**(limits or {})}
    if set(limits)!=set(LIMITS) or type(limits['max_calls']) is not int or not 1<=limits['max_calls']<=3 or type(limits['max_output_tokens']) is not int or not 64<=limits['max_output_tokens']<=4096 or type(limits['seed']) is not int or not 0<float(limits['call_timeout_seconds'])<=600:
        raise ValueError('invalid generation budgets')
    started=time.monotonic();run=dict(schema_version=1,pipeline='story_understanding_p5',response_contract_version=RESPONSE_CONTRACT_VERSION,status='running',started_at=now(),prompt_version=PROMPT_VERSION,request=story,limits=limits,attempts=[],provider=provider.metadata(),mode=provider.metadata()['mode'])
    atomic_json(root/'run.json',run)
    try:
        if not isinstance(story,str) or not story.strip() or len(story)>6000:raise SpatialError('invalid_input','story must contain 1–6000 characters')
        if not Path(asset_root).is_dir():raise SpatialError('missing_asset_root','source asset root missing')
        assets=LayoutAssets(asset_root);initial=SceneRepository(assets).validate(SceneDocument(**read_json(scene_path)));initial.scene_id='initial'
        check_catalog(catalog,initial,assets)
        atomic_json(root/'request.json',dict(story=story,catalog=catalog,initial_scene=asdict(initial),source_scene_sha256=sha256(scene_path)))
        atomic_json(root/'response.schema.json',response_schema())
        cfg=Path(config_path) if config_path else Path(__file__).resolve().parents[2]/'configs/storyboard.yaml'
        if not cfg.is_absolute():cfg=Path(__file__).resolve().parents[2]/cfg
        config=yaml.safe_load(cfg.read_text())
        settings={**DEFAULTS,**config.get('spatial',{})}
        messages=messages_for(story,catalog,asdict(initial),{**settings,'event_min_distance_change':config.get('events',{}).get('min_distance_change',.2)})
        for index in range(limits['max_calls']):
            attempt=root/f'attempt_{index:02d}';attempt.mkdir();entry=dict(directory=attempt.name,status='running',call_index=index,seconds=0);run['attempts'].append(entry)
            atomic_json(attempt/'messages.json',messages);atomic_json(root/'run.json',run);call_start=time.monotonic();errors=[];stage='provider'
            atomic_json(attempt/'provider.json',dict(**provider.metadata(),generation=limits,status='calling'))
            response=None
            try:
                with deadline(limits['call_timeout_seconds']):response=provider.generate(messages,response_schema(),limits)
                entry['provider_seconds']=time.monotonic()-call_start
                (attempt/'raw_response.txt').write_text(response.content)
                atomic_json(attempt/'provider.json',dict(**provider.metadata(),generation=limits,response=response.metadata,seconds=entry['provider_seconds']))
                stage='json';value=strict_json(response.content);atomic_json(attempt/'json_response.json',value)
                stage='schema';parsed=parse_response(response.content);atomic_json(attempt/'parsed_response.json',parsed)
                if parsed['status']=='needs_clarification':
                    run.update(status='needs_clarification',question=parsed['question'],reason_code=parsed['reason_code']);entry['status']='needs_clarification';break
                stage='references'
                check_event_references(parsed['plan'],[i.instance_id for i in initial.instances])
                check_fact_references(parsed['plan'])
                normalized=StoryPlan.parse(parsed['plan'],initial,assets,settings,config.get('events',{}).get('min_distance_change',.2)).to_dict()
                atomic_json(attempt/'plan.json',parsed['plan']);atomic_json(attempt/'normalized_plan.json',normalized)
                stage='compile_render'
                report=compile_story(attempt/'plan.json',asset_root,scene_path,attempt/'compiled',config_path,allow_provisional)
                if not report['passed']:
                    saved=read_json(attempt/'compiled/run.json')
                    errors=[dict(stage=stage,code=saved.get('error',{}).get('code','validation_failed'),details=saved.get('error',{})),dict(stage='validation',scenes=[dict(scene_id=s['scene_id'],errors=s.get('errors',[]),event_checks=s.get('event_checks',[])) for s in report.get('scenes',[]) if not s.get('passed')])]
                else:
                    # compile_story already rereads persisted scenes/assets/images through its independent validator.
                    # Explicit P5 validate performs a fresh provenance + geometric revalidation when requested.
                    entry['status']='succeeded';run.update(status='succeeded',accepted_attempt=attempt.name);break
            except Exception as exc:errors=[_failure(exc,stage)]
            finally:
                entry['seconds']=time.monotonic()-call_start;atomic_json(attempt/'provider.json',dict(**provider.metadata(),generation=limits,response=response.metadata if response else None,status='responded' if response else 'failed',seconds=entry.get('provider_seconds',entry['seconds'])));atomic_json(attempt/'errors.json',errors);atomic_json(root/'run.json',run)
            entry['status']='failed';entry['errors']=errors;atomic_json(root/'run.json',run)
            # Transport/unavailable/timeouts are visible terminal failures, not content repair requests.
            if stage=='provider':run['error']=errors[0];break
            if index+1<limits['max_calls']:
                if response is not None:messages.append(dict(role='assistant',content=response.content))
                messages.append(dict(role='user',content=json.dumps(dict(repair_request=True,failures=errors,registered_instances=catalog,original_user_story=story,instruction='修正这些失败并逐幕重新核对原始故事的全部要求，保留已正确的关系与事件，返回完整 JSON。格式或引用错误应使用所列合法 ID 自行修复，不能向用户询问技术字段。仅原始故事的歧义、不支持或冲突才允许 needs_clarification。'),ensure_ascii=False)))
        if run['status']=='running':run.update(status='failed',error=run.get('error',dict(code='repair_budget_exhausted',message='no accepted story within configured call budget')))
    except Exception as exc:run.update(status='failed',error=_failure(exc,'input'))
    run.update(finished_at=now(),seconds=time.monotonic()-started,provider=provider.metadata(),exit_code=0 if run['status']=='succeeded' else (2 if run['status']=='needs_clarification' else 1))
    # The P4 compiled run carries its own full asset/render manifest. Freeze P5 request and call logs separately.
    files=[p for p in root.rglob('*') if p.is_file() and 'compiled' not in p.relative_to(root).parts and p.name not in ('run.json','index.html','revalidation.json')]
    run['artifacts']=[dict(path=str(p.relative_to(root)),sha256=sha256(p)) for p in sorted(files)]
    atomic_json(root/'run.json',run);_index(root,run);return run

def validate(root):
    root=Path(root).resolve();run=read_json(root/'run.json');errors=[];compiled=None
    if run.get('response_contract_version')!=RESPONSE_CONTRACT_VERSION:
        report=dict(passed=False,status='failed',errors=['unsupported_response_contract_version'],saved_version=run.get('response_contract_version'),validator_version=RESPONSE_CONTRACT_VERSION)
        atomic_json(root/'revalidation.json',report);return report
    for row in run.get('artifacts',[]):
        p=root/row['path']
        if not p.is_file() or sha256(p)!=row['sha256']:errors.append('artifact_mismatch:'+row['path'])
    if run['status']=='succeeded':
        try:
            attempt=root/run['accepted_attempt'];original=parse_response((attempt/'raw_response.txt').read_text())
            if original['plan']!=read_json(attempt/'plan.json'):errors.append('raw_plan_mismatch')
            initial=SceneDocument(**read_json(attempt/'compiled/initial_scene.json'));assets=LayoutAssets(attempt/'compiled')
            request=read_json(root/'request.json');check_catalog(request['catalog'],initial,assets)
            config=read_json(attempt/'compiled/config.json')
            normalized=StoryPlan.parse(original['plan'],initial,assets,config['spatial'],config.get('events',{}).get('min_distance_change',.2)).to_dict()
            if normalized!=read_json(attempt/'compiled/story_plan.json'):errors.append('compiled_plan_mismatch')
            if request['story']!=run['request']:errors.append('original_request_mismatch')
            compiled=validate_compiled(attempt/'compiled')
            if not compiled['passed']:errors.append('compiled_validation_failed')
        except Exception as exc:errors.append(type(exc).__name__+':'+str(exc))
    else:errors.append('no_accepted_story:'+run['status'])
    report=dict(passed=not errors,status='succeeded' if not errors else 'failed',errors=errors,compiled=compiled,scope='Syntax/references/geometry/images/provenance; semantic fidelity needs human labels')
    atomic_json(root/'revalidation.json',report);return report
