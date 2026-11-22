"""StoryPlan v1 and SceneIntent: handwritten boundary, independent of legacy scenes."""
from dataclasses import dataclass,asdict
from src.scene.models import identifier
from .spatial_contracts import SpatialError,SpatialRequest,DEFAULTS,finite

ACTIONS={'approach','retreat','hold','appear','disappear'}


@dataclass
class SceneIntent:
    scene_id:str
    base_scene_id:str
    constraints:list
    events:list
    narration:dict
    shot:dict


@dataclass
class StoryPlan:
    story_id:str
    asset_ids:list
    premise:str
    scenes:list
    schema_version:int=1

    def to_dict(self):return asdict(self)

    @classmethod
    def parse(cls,data,initial,assets,spatial_settings=None,default_min_change=.2):
        def bad(message,code='invalid_story_plan'):raise SpatialError(code,message)
        if not isinstance(data,dict) or type(data.get('schema_version')) is not int or data['schema_version']!=1:bad('unsupported StoryPlan schema_version')
        if set(data)-{'schema_version','story_id','asset_ids','premise','scenes'}:bad('unknown StoryPlan fields')
        try:identifier(data.get('story_id'))
        except ValueError:bad('invalid story_id')
        declared=data.get('asset_ids');actual={i.asset_id for i in initial.instances}
        if not isinstance(declared,list) or len(set(declared))!=len(declared) or set(declared)!=actual:bad('asset_ids must exactly match registered input scene assets','unknown_asset')
        for aid in declared:assets.get(aid)
        premise=data.get('premise','')
        if not isinstance(premise,str) or len(premise)>4000:bad('invalid premise')
        rows=data.get('scenes')
        if not isinstance(rows,list) or len(rows)!=3:bad('P4 requires exactly three scenes')
        ids=[i.instance_id for i in initial.instances]
        if not 2<=len(ids)<=3:bad('P4 supports two or three registered instances')
        seen={'initial'};scenes=[];previous='initial';event_ids=set()
        for row in rows:
            if not isinstance(row,dict) or set(row)-{'scene_id','base_scene_id','constraints','events','narration','shot'}:bad('unknown SceneIntent fields')
            sid=row.get('scene_id')
            try:identifier(sid)
            except ValueError:bad('invalid scene_id')
            if sid in seen:bad('duplicate/reserved scene_id')
            base=row.get('base_scene_id',previous)
            if base not in seen:bad('base_scene_id must reference initial or an earlier scene; forward/cyclic references rejected','invalid_inheritance')
            constraints=row.get('constraints',[])
            if not isinstance(constraints,list):bad('constraints must be a list')
            if constraints:
                parsed=SpatialRequest.parse(dict(schema_version=1,constraints=constraints,settings=spatial_settings or {}),ids)
                constraints=[{k:v for k,v in asdict(c).items() if k!='origin'} for c in parsed.constraints if c.origin=='user']
            events=row.get('events',[])
            if not isinstance(events,list):bad('events must be a list')
            clean=[];subject_actions={}
            for event in events:
                if not isinstance(event,dict) or set(event)-{'event_id','action','subject_instance_id','object_instance_id','min_change'}:bad('unknown event fields')
                eid=event.get('event_id');action=event.get('action');subject=event.get('subject_instance_id');obj=event.get('object_instance_id')
                try:identifier(eid)
                except ValueError:bad('invalid event_id')
                if eid in event_ids:bad('duplicate event_id')
                event_ids.add(eid)
                if action not in ACTIONS:bad('unsupported action: '+str(action),'unsupported_action')
                if subject not in ids or (action in ('approach','retreat') and (obj not in ids or obj==subject)):bad('unknown event instance','unknown_reference')
                if action not in ('approach','retreat') and (obj is not None or 'min_change' in event):bad('unary event cannot have object/min_change')
                if subject in subject_actions:bad('one event per subject per scene; conflicting actions rejected','conflicting_events')
                subject_actions[subject]=action
                item=dict(event_id=eid,action=action,subject_instance_id=subject)
                if action in ('approach','retreat'):
                    change=finite(event.get('min_change',default_min_change),'min_change')
                    if change<=0:bad('min_change must be positive')
                    item.update(object_instance_id=obj,min_change=change)
                clean.append(item)
            narration=row.get('narration')
            if not isinstance(narration,dict) or set(narration)-{'text','entity_refs','fact_claims','creative_text'} or not isinstance(narration.get('text'),str) or not narration['text'].strip() or len(narration['text'])>2000:bad('invalid narration')
            refs=narration.get('entity_refs',[])
            if not isinstance(refs,list) or any(ref not in ids for ref in refs):bad('unknown narration entity reference','unknown_reference')
            claims=narration.get('fact_claims',[])
            if not isinstance(claims,list) or any(not isinstance(c,str) or c not in {v['constraint_id'] for v in constraints}|{v['event_id'] for v in clean} for c in claims):bad('fact_claims must reference current constraint/event IDs')
            if not isinstance(narration.get('creative_text',''),str):bad('creative_text must be text')
            shot=row.get('shot',{})
            if not isinstance(shot,dict) or set(shot)-{'template','target_ids','complete_display'}:bad('invalid shot fields')
            template=shot.get('template','overview');targets=shot.get('target_ids',[] if template=='overview' else ids)
            if template not in ('overview','focus') or not isinstance(targets,list) or (not targets and template=='focus') or len(set(targets))!=len(targets) or any(t not in ids for t in targets):bad('invalid camera template/target references','unknown_reference')
            if type(shot.get('complete_display',True)) is not bool:bad('complete_display must be bool')
            scenes.append(SceneIntent(sid,base,constraints,clean,narration,dict(template=template,target_ids=targets,complete_display=shot.get('complete_display',True))))
            previous=sid;seen.add(sid)
        return cls(data['story_id'],declared,premise,scenes)


def json_schema():
    """Portable schema for the public envelope; reference/geometric rules remain independent checks."""
    return {'$schema':'https://json-schema.org/draft/2020-12/schema','title':'StoryPlan v1','type':'object',
      'required':['schema_version','story_id','asset_ids','scenes'],'additionalProperties':False,
      'properties':{'schema_version':{'const':1},'story_id':{'type':'string'},'asset_ids':{'type':'array','items':{'type':'string'},'uniqueItems':True},
       'premise':{'type':'string'},'scenes':{'type':'array','minItems':3,'maxItems':3,'items':{'type':'object',
        'required':['scene_id','narration'],'additionalProperties':False,'properties':{'scene_id':{'type':'string'},'base_scene_id':{'type':'string'},
         'constraints':{'type':'array'},'events':{'type':'array','items':{'type':'object','required':['event_id','action','subject_instance_id'],
          'properties':{'action':{'enum':sorted(ACTIONS)},'event_id':{'type':'string'},'subject_instance_id':{'type':'string'},'object_instance_id':{'type':'string'},'min_change':{'type':'number','exclusiveMinimum':0}}}},
         'narration':{'type':'object','required':['text'],'properties':{'text':{'type':'string','minLength':1},'entity_refs':{'type':'array'},'fact_claims':{'type':'array'},'creative_text':{'type':'string'}}},
         'shot':{'type':'object','properties':{'template':{'enum':['overview','focus']},'target_ids':{'type':'array'},'complete_display':{'type':'boolean'}}}}}}}}
