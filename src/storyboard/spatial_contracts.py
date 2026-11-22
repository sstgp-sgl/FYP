"""P3 structured spatial boundary; separate from the legacy PlacementConstraint v1."""
from dataclasses import dataclass, field, asdict
import math
from src.scene.models import identifier

RELATIONS={'left_of','right_of','near','far','on_ground','non_overlap','keep_pose'}
DEFAULTS=dict(gap=.2,near_max=.5,far_min=1.5,epsilon=1e-4,translation_tolerance=1e-6,
              scale_tolerance=1e-6,rotation_tolerance=1e-5,max_evaluations=5000,
              search_bounds=[-6.,6.,-6.,6.])


class SpatialError(ValueError):
    def __init__(self,code,message,evidence=None):
        super().__init__(message);self.code=code;self.evidence=evidence or {}


def finite(value,name,minimum=0):
    if type(value) not in (int,float) or not math.isfinite(value) or value<minimum:
        raise SpatialError('invalid_parameter','invalid '+name,dict(value=repr(value)))
    return float(value)


@dataclass
class SpatialConstraint:
    constraint_id: str
    subject_instance_id: str
    relation: str
    object_instance_id: str | None = None
    parameters: dict = field(default_factory=dict)
    frame: str = 'reference_ground'
    origin: str = 'user'
    schema_version: int = 1


@dataclass
class SpatialRequest:
    constraints: list
    fixed_instance_ids: list
    settings: dict
    schema_version: int = 1

    @classmethod
    def parse(cls,data,instance_ids):
        def bad(msg):raise SpatialError('invalid_request',msg)
        if not isinstance(data,dict) or (type(data.get('schema_version')) is not int or data.get('schema_version')!=1):bad('unsupported spatial schema_version')
        if set(data)-{'schema_version','constraints','fixed_instance_ids','settings'}:bad('unknown request fields')
        if not 2<=len(instance_ids)<=3 or len(set(instance_ids))!=len(instance_ids):bad('P3 supports 2 or 3 distinct visible instances')
        settings=dict(DEFAULTS)
        raw_settings=data.get('settings',{})
        if not isinstance(raw_settings,dict) or set(raw_settings)-set(DEFAULTS):bad('unknown settings')
        settings.update(raw_settings)
        for name in ('gap','near_max','far_min','epsilon','translation_tolerance','scale_tolerance','rotation_tolerance'):
            settings[name]=finite(settings[name],name)
        if settings['epsilon']>1e-3:bad('epsilon must not exceed 1e-3')
        budget=settings['max_evaluations']
        if type(budget) is not int or not 1<=budget<=50000:bad('invalid bounded search budget')
        domain=settings['search_bounds']
        if not isinstance(domain,list) or len(domain)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in domain) or domain[0]>=domain[1] or domain[2]>=domain[3]:bad('invalid XZ search bounds')
        fixed=data.get('fixed_instance_ids',[])
        if not isinstance(fixed,list) or len(set(fixed))!=len(fixed) or any(i not in instance_ids for i in fixed):bad('unknown/duplicate fixed instance')
        values=data.get('constraints')
        if not isinstance(values,list) or not values:bad('constraints must be a nonempty list')
        constraints=[];seen=set()
        fields={'constraint_id','subject_instance_id','relation','object_instance_id','parameters','frame','schema_version'}
        for row in values:
            if not isinstance(row,dict) or set(row)-fields:bad('unknown constraint fields')
            if type(row.get('schema_version',1)) is not int or row.get('schema_version',1)!=1:bad('unsupported constraint version')
            cid=row.get('constraint_id');subject=row.get('subject_instance_id');obj=row.get('object_instance_id');rel=row.get('relation')
            try:identifier(cid)
            except ValueError:bad('invalid constraint ID')
            if cid in seen or cid.startswith('sys_'):bad('duplicate/reserved constraint ID')
            seen.add(cid)
            if rel not in RELATIONS:raise SpatialError('unsupported_relation','unsupported relation: '+str(rel))
            if subject not in instance_ids or (rel not in ('on_ground','keep_pose') and obj not in instance_ids):raise SpatialError('unknown_reference','unknown instance reference')
            if rel in ('on_ground','keep_pose') and obj is not None:bad('unary relation cannot have object')
            if obj==subject:bad('self spatial relation unsupported')
            if row.get('frame','reference_ground')!='reference_ground':raise SpatialError('unsupported_frame','only world reference_ground is supported')
            params=row.get('parameters',{})
            key={'left_of':'gap','right_of':'gap','near':'near_max','far':'far_min'}.get(rel)
            if not isinstance(params,dict) or set(params)-({key} if key else set()):bad('unknown relation parameters')
            params={key:finite(params.get(key,settings[key]),key)} if key else {}
            constraints.append(SpatialConstraint(cid,subject,rel,obj,params))
        # Ground support and non-overlap are hard system constraints, not model self-reports.
        for n,inst in enumerate(instance_ids):constraints.append(SpatialConstraint(f'sys_ground_{n}',inst,'on_ground',origin='system'))
        for n,a in enumerate(instance_ids):
            for m,b in enumerate(instance_ids[n+1:]):constraints.append(SpatialConstraint(f'sys_non_overlap_{n}_{m}',a,'non_overlap',b,origin='system'))
        if not fixed:fixed=[instance_ids[-1]]
        return cls(constraints,list(fixed),settings)

    def to_dict(self):return asdict(self)
