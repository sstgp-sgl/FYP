"""P5 model-output syntax; P4 remains the reference and geometric authority."""
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,StrictStr,StrictBool

RESPONSE_CONTRACT_VERSION=2

class StrictModel(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
class Constraint(StrictModel):
    constraint_id:StrictStr
    subject_instance_id:StrictStr
    relation:StrictStr
    object_instance_id:StrictStr|None=None
    parameters:dict[str,float]=Field(default_factory=dict)
class Event(StrictModel):
    event_id:StrictStr
    action:StrictStr
    subject_instance_id:StrictStr
    object_instance_id:StrictStr|None=Field(default=None,description='REQUIRED non-null target instance for BOTH approach AND retreat; a different registered instance. Omit/null only for hold, appear, disappear.')
    min_change:float|None=None
class Narration(StrictModel):
    text:StrictStr=Field(min_length=1,max_length=2000)
    entity_refs:list[StrictStr]=Field(default_factory=list,description='Registered instance IDs mentioned in this narration; never display names or asset IDs.')
    fact_claims:list[StrictStr]=Field(default_factory=list,description='IDs of CURRENT SCENE constraints/events only. Example: if constraint_id is rel_left_1 then use ["rel_left_1"]. NEVER natural-language claims, instance IDs, or IDs from earlier scenes.')
    creative_text:StrictStr=''
class Shot(StrictModel):
    template:Literal['overview','focus']='overview'
    target_ids:list[StrictStr]=Field(default_factory=list)
    complete_display:Literal[True]=True
class Intent(StrictModel):
    scene_id:StrictStr
    base_scene_id:StrictStr|None=None
    constraints:list[Constraint]=Field(default_factory=list)
    events:list[Event]=Field(default_factory=list)
    narration:Narration
    shot:Shot=Field(default_factory=Shot)
class Plan(StrictModel):
    schema_version:Literal[1]
    story_id:StrictStr
    asset_ids:list[StrictStr]
    premise:StrictStr=''
    scenes:list[Intent]=Field(min_length=3,max_length=3)
class ModelResponse(StrictModel):
    status:Literal['ready','needs_clarification']
    plan:Plan|None=None
    question:StrictStr|None=None
    reason_code:Literal['unsupported_action','unknown_asset','ambiguous_reference','contradictory_constraints','layout_not_found']|None=None

def response_schema():
    schema=ModelResponse.model_json_schema();schema['$id']='urn:fyp:story-response:v2'
    schema['$defs']['Event']['allOf']=[{'if':{'required':['action'],'properties':{'action':{'enum':['approach','retreat']}}},'then':{'required':['object_instance_id'],'properties':{'object_instance_id':{'type':'string'}}}}]
    return schema
