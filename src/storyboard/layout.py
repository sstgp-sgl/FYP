"""P3 deterministic bounded XZ candidate search. No model imports or continuous-solver success shortcuts."""
from dataclasses import asdict
from itertools import product,permutations
import math
import numpy as np
from src.scene.models import SceneDocument


def offsets(scene,assets):
    result={}
    for i in scene.instances:
        if not i.visible:continue
        record=assets.get(i.asset_id)
        if record.representation!='point_cloud':raise ValueError('unsupported_representation')
        p,_=assets.points(record);lo,hi=p.min(0),p.max(0)
        corners=np.asarray(list(product(*zip(lo,hi))))
        linear=i.transform.matrix()[:3,:3]
        q=corners@linear.T
        result[i.instance_id]=np.array([q.min(0),q.max(0)])
    return result


def footprint_distance(a,b):
    d=np.maximum(0,np.maximum(a[0,[0,2]]-b[1,[0,2]],b[0,[0,2]]-a[1,[0,2]]))
    return float(np.linalg.norm(d))


def violation(c,bounds,eps):
    a=bounds[c.subject_instance_id];b=bounds.get(c.object_instance_id);r=c.relation
    if r=='left_of':return max(0.,a[1,0]+c.parameters['gap']-b[0,0]-eps)
    if r=='right_of':return max(0.,b[1,0]+c.parameters['gap']-a[0,0]-eps)
    if r=='near':return max(0.,footprint_distance(a,b)-c.parameters['near_max']-eps)
    if r=='far':return max(0.,c.parameters['far_min']-footprint_distance(a,b)-eps)
    if r=='on_ground':return max(0.,abs(a[0,1])-eps)
    if r=='non_overlap':
        extent=np.maximum(0,np.minimum(a[1,[0,2]],b[1,[0,2]])-np.maximum(a[0,[0,2]],b[0,[0,2]]))
        return max(0.,float(np.prod(extent))-eps**2)
    if r=='keep_pose':return 0. # These subjects are removed from search and kept exactly at base pose.
    raise ValueError('unsupported relation')


def contradiction(request,local,positions):
    eps=request.settings['epsilon'];edges=[]
    for c in request.constraints:
        if c.relation not in ('left_of','right_of'):continue
        a,b=c.subject_instance_id,c.object_instance_id
        if c.relation=='right_of':a,b=b,a
        edges.append((a,b,float(local[a][1,0]-local[b][0,0]+c.parameters['gap']-eps),c.constraint_id))
    # A positive cycle proves impossible difference inequalities, independent of search budget.
    distances={i:0. for i in local}
    for n in range(len(local)):
        changed=False
        for a,b,weight,cid in edges:
            if distances[b]<distances[a]+weight-1e-12:
                distances[b]=distances[a]+weight;changed=True
        if not changed:break
        if n==len(local)-1:return dict(proof='positive_left_right_cycle',constraint_ids=[e[3] for e in edges])
    pairs={}
    for c in request.constraints:
        if c.relation in ('near','far'):
            key=tuple(sorted([c.subject_instance_id,c.object_instance_id]));pairs.setdefault(key,[]).append(c)
    for pair,cs in pairs.items():
        near=[c for c in cs if c.relation=='near'];far=[c for c in cs if c.relation=='far']
        if near and far and max(c.parameters['far_min'] for c in far)>min(c.parameters['near_max'] for c in near)+2*eps:
            return dict(proof='disjoint_near_far_intervals',pair=list(pair),constraint_ids=[c.constraint_id for c in cs])
        for c in request.constraints:
            if c.relation in ('left_of','right_of') and tuple(sorted([c.subject_instance_id,c.object_instance_id]))==pair and near:
                if c.parameters['gap']>min(n.parameters['near_max'] for n in near)+2*eps:
                    return dict(proof='left_right_gap_exceeds_near',constraint_ids=[c.constraint_id]+[n.constraint_id for n in near])
    return None


def candidate_positions(name,local,placed,initial,request):
    o=local[name];center=o.mean(0);settings=request.settings;domain=settings['search_bounds']
    distances={0.,settings['gap'],settings['near_max'],settings['far_min'],settings['near_max']/2}
    for c in request.constraints:
        distances.update(c.parameters.values())
        if c.relation=='near':distances.add(c.parameters['near_max']/2)
    candidates=[(initial[name][0],initial[name][2]),(domain[0],domain[2]),(domain[1],domain[3])]
    xvalues={initial[name][0]};zvalues={initial[name][2]}
    for other,b in placed.items():
        for d in sorted(distances):
            left=b[0,0]-o[1,0]-d;right=b[1,0]-o[0,0]+d
            back=b[0,2]-o[1,2]-d;front=b[1,2]-o[0,2]+d
            x=b[:,0].mean()-center[0];z=b[:,2].mean()-center[2]
            candidates.extend([(left,z),(right,z),(x,back),(x,front),
                               (left,initial[name][2]),(right,initial[name][2]),
                               (initial[name][0],back),(initial[name][0],front)])
            delta=d/math.sqrt(2)
            for signx,signz in product((-1,1),repeat=2):
                cx=(b[0,0]-o[1,0]-delta) if signx<0 else (b[1,0]-o[0,0]+delta)
                cz=(b[0,2]-o[1,2]-delta) if signz<0 else (b[1,2]-o[0,2]+delta)
                candidates.append((cx,cz))
            xvalues.update((left,right,x));zvalues.update((back,front,z))
    # Intersections of independently derived X and Z boundary values, plus base position.
    candidates.extend(product(sorted(xvalues),sorted(zvalues)))
    unique={(float(x),float(z)) for x,z in candidates if domain[0]<=x<=domain[1] and domain[2]<=z<=domain[3]}
    return sorted(unique,key=lambda p:((p[0]-initial[name][0])**2+(p[1]-initial[name][2])**2,p[0],p[1]))


def solve(scene,request,assets,candidate_filter=None,extra_positions=None):
    base=SceneDocument(**asdict(scene));local=offsets(base,assets)
    initial={i.instance_id:np.array(i.transform.translation_xyz,dtype=float) for i in base.instances if i.visible}
    proof=contradiction(request,local,initial)
    if proof:return dict(status='contradictory_constraints',passed=False,proof=proof,evaluations=0),None
    fixed=set(request.fixed_instance_ids)|{c.subject_instance_id for c in request.constraints if c.relation=='keep_pose'}
    placed={name:local[name]+initial[name] for name in sorted(fixed)}
    eps=request.settings['epsilon'];budget=request.settings['max_evaluations']
    def applicable(bounds):return [c for c in request.constraints if c.subject_instance_id in bounds and (c.object_instance_id is None or c.object_instance_id in bounds)]
    bad=[c.constraint_id for c in applicable(placed) if violation(c,placed,eps)>0]
    if bad:return dict(status='contradictory_constraints',passed=False,proof=dict(proof='constraints_violate_locked_pose',constraint_ids=bad),evaluations=0),None
    free=sorted(set(local)-fixed);best=None;best_cost=float('inf');evaluations=0;best_residual=float('inf')
    chosen={k:initial[k].copy() for k in fixed}
    all_orders=list(permutations(free)) or [()]
    def search(order,depth,bounds,cost,limit):
        nonlocal best,best_cost,evaluations,best_residual
        if depth==len(order):
            residual=sum(violation(c,bounds,eps)**2 for c in request.constraints)
            best_residual=min(best_residual,residual)
            temporal_ok=candidate_filter(bounds,chosen) if candidate_filter is not None else True
            if residual==0 and temporal_ok and cost<best_cost:
                best_cost=cost;best={k:v.copy() for k,v in chosen.items()}
            return
        if evaluations>=limit:return
        name=order[depth]
        options=candidate_positions(name,local,bounds,initial,request)
        if extra_positions and name in extra_positions:
            domain=request.settings['search_bounds']
            options=list(set(options)|{(float(x),float(z)) for x,z in extra_positions[name] if domain[0]<=x<=domain[1] and domain[2]<=z<=domain[3]})
            options.sort(key=lambda p:((p[0]-initial[name][0])**2+(p[1]-initial[name][2])**2,p[0],p[1]))
        for x,z in options:
            if evaluations>=limit or best_cost==0:break
            evaluations+=1
            pose=np.array([x,-local[name][0,1],z]);next_cost=cost+float(np.sum((pose-initial[name])**2))
            if next_cost>=best_cost:continue
            next_bounds={**bounds,name:local[name]+pose}
            residual=sum(violation(c,next_bounds,eps)**2 for c in applicable(next_bounds))
            best_residual=min(best_residual,residual)
            if residual>0:continue
            chosen[name]=pose
            search(order,depth+1,next_bounds,next_cost,limit)
            chosen.pop(name,None)
    for n,order in enumerate(all_orders):
        limit=budget if n==len(all_orders)-1 else max(evaluations,budget*(n+1)//len(all_orders))
        search(order,0,placed,0.,limit)
        if best_cost==0:break
    diagnostics=dict(evaluations=evaluations,max_evaluations=budget,budget_exhausted=evaluations>=budget,
                     orders=[list(o) for o in all_orders],fixed_instance_ids=sorted(fixed),
                     search_bounds=request.settings['search_bounds'],best_checked_partial_residual=None if not math.isfinite(best_residual) else best_residual,
                     algorithm='bounded_analytic_candidates_v1',global_optimum_claimed=False)
    if best is None:return dict(status='layout_not_found',passed=False,proof=None,**diagnostics),None
    result=SceneDocument(**asdict(base))
    for i in result.instances:
        if i.instance_id in best:i.transform.translation_xyz=best[i.instance_id].tolist()
    # New relationships live in spatial_request.json, never in legacy PlacementConstraint fields.
    result.constraints=[]
    # A solver result is only a candidate. Independent service validation must accept it.
    return dict(status='candidate',passed=False,movement_cost=best_cost,**diagnostics),result
