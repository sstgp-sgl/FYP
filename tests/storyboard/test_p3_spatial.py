from dataclasses import asdict
from itertools import product
from pathlib import Path
import copy,json,subprocess,sys
import numpy as np
import pytest
from src.pointcloud_io import write_ply
from src.scene.models import SceneDocument,Instance,Transform,Y_UP
from src.scene.storage import AssetRepository,SceneRepository,atomic_json,read_json,sha256
from src.scene.composition import export
from src.storyboard.spatial_contracts import SpatialRequest,SpatialConstraint,SpatialError,DEFAULTS
from src.storyboard.spatial_validation import check_constraint,validate_geometry,measured_bounds
from src.storyboard.layout import solve
from src.storyboard.spatial_service import compile_layout,validate

ROOT=Path(__file__).resolve().parents[2]
TRUTH=json.loads((ROOT/'tests/fixtures/storyboard/spatial_truth.json').read_text())


def constraint(relation,subject='a',obj='b',**parameters):
    return dict(constraint_id='user_'+relation+'_'+subject+'_'+str(obj),subject_instance_id=subject,relation=relation,
                object_instance_id=obj if relation not in ('on_ground','keep_pose') else None,parameters=parameters)


def req(cs,fixed=None,**settings):
    data=dict(schema_version=1,constraints=cs)
    if fixed is not None:data['fixed_instance_ids']=fixed
    if settings:data['settings']=settings
    return data


@pytest.fixture
def setup(tmp_path):
    def build(n=2,positions=None,widths=None):
        lib=tmp_path/('library'+str(len(list(tmp_path.iterdir()))));repo=AssetRepository(lib);instances=[]
        for j,name in enumerate(['a','b','c'][:n]):
            width=(widths or {}).get(name,1.)
            p=np.array(list(product([-width/2,width/2],[0.,1.],[-.5,.5])))
            path=tmp_path/(name+str(j)+'.ply');write_ply(p,np.tile([[220,60,40],[40,90,220],[40,175,80]][j],(len(p),1)),path)
            record=repo.import_asset(path,role='subject',representation='point_cloud',source_kind='synthetic',producer='P3_hand_cuboid_fixture',coordinate_frame=Y_UP)
            t=(positions or {}).get(name,[0.,0.,0.])
            instances.append(Instance(name,record.asset_id,record.sha256,Transform(translation_xyz=t)))
        scene=SceneDocument('p3_fixture',instances,[]);SceneRepository(repo).save(scene)
        return repo,scene
    return build


@pytest.mark.parametrize('case',TRUTH['cases'],ids=[c['id'] for c in TRUTH['cases']])
def test_hand_written_measurements(case):
    c=SpatialConstraint(case['id'],'a',case['relation'],None if case['relation']=='on_ground' else 'b',case['parameters'])
    report=check_constraint(c,{'a':case['a'],'b':TRUTH['b']},{},{},DEFAULTS)
    assert (report['status']=='passed')==case['expected']
    assert report['measured']==pytest.approx(case['measured'],abs=1e-10)


@pytest.mark.parametrize('cs',[
 [constraint('left_of')], [constraint('right_of')], [constraint('near')], [constraint('far')],
 [constraint('left_of'),constraint('near')], [constraint('left_of'),constraint('far')],
 [constraint('right_of'),constraint('near')], [constraint('right_of'),constraint('far')],
 [constraint('left_of',gap=0),constraint('near',near_max=0)],
 [constraint('left_of',gap=.7),constraint('far',far_min=2)],
])
def test_two_object_solutions_are_independently_accepted(setup,cs):
    assets,scene=setup();before=asdict(scene)
    request=SpatialRequest.parse(req(cs),['a','b']);diagnostic,result=solve(scene,request,assets)
    assert result is not None,diagnostic
    assert validate_geometry(result,scene,request,assets)['passed']
    assert asdict(scene)==before
    assert result.instances[1].transform.translation_xyz==[0,0,0]
    assert diagnostic['evaluations']<=request.settings['max_evaluations']


@pytest.mark.parametrize('cs',[
 [constraint('left_of','a','b'),constraint('right_of','c','b')],
 [constraint('left_of','a','b'),constraint('left_of','c','a')],
 [constraint('near','a','b'),constraint('near','c','b')],
 [constraint('far','a','b'),constraint('far','c','b'),constraint('near','a','c')],
 [constraint('right_of','a','b'),constraint('right_of','c','a'),constraint('far','c','b')],
])
def test_three_object_feasible_cases(setup,cs):
    assets,scene=setup(3)
    request=SpatialRequest.parse(req(cs,fixed=['b']),['a','b','c'])
    diagnostic,result=solve(scene,request,assets)
    assert result is not None,diagnostic
    report=validate_geometry(result,scene,request,assets)
    assert report['passed'],report
    assert diagnostic['evaluations']<=5000


@pytest.mark.parametrize('cs',[
 [constraint('left_of'),constraint('right_of')],
 [constraint('near',near_max=.5),constraint('far',far_min=1.5)],
 [constraint('left_of',gap=1),constraint('near',near_max=.5)],
 [constraint('left_of','a','b'),constraint('left_of','b','c'),constraint('left_of','c','a')],
])
def test_proven_conflicts_not_silent_repairs(setup,cs):
    n=3 if any(c['subject_instance_id']=='c' or c['object_instance_id']=='c' for c in cs) else 2
    assets,scene=setup(n);request=SpatialRequest.parse(req(cs),[i.instance_id for i in scene.instances])
    report,result=solve(scene,request,assets)
    assert result is None and report['status']=='contradictory_constraints' and report['proof']


def test_bounded_search_failure_is_not_infeasibility_proof(setup):
    assets,scene=setup();request=SpatialRequest.parse(req([constraint('far',far_min=10)],search_bounds=[-.1,.1,-.1,.1],max_evaluations=8),['a','b'])
    diagnostic,result=solve(scene,request,assets)
    assert result is None and diagnostic['status']=='layout_not_found' and diagnostic['proof'] is None
    assert diagnostic['evaluations']<=8


@pytest.mark.parametrize('mutation,code',[
 (lambda d:d['constraints'][0].update(relation='pick_up'),'unsupported_relation'),
 (lambda d:d['constraints'][0]['parameters'].update(gap=-.1),'invalid_parameter'),
 (lambda d:d['constraints'][0]['parameters'].update(gap=float('nan')),'invalid_parameter'),
 (lambda d:d['constraints'][0].update(object_instance_id='missing'),'unknown_reference'),
 (lambda d:d['constraints'][0].update(frame='camera'),'unsupported_frame'),
 (lambda d:d.update(schema_version=True),'invalid_request'),
])
def test_invalid_structured_requests(mutation,code):
    data=req([constraint('left_of')]);mutation(data)
    with pytest.raises(SpatialError) as exc:SpatialRequest.parse(data,['a','b'])
    assert exc.value.code==code


def test_scale_yaw_fixed_and_eight_corners(setup):
    assets,scene=setup(widths={'a':2.})
    scene.instances[0].transform=Transform(rotation_xyzw=[0,np.sin(np.pi/8),0,np.cos(np.pi/8)],uniform_scale=2)
    b=measured_bounds(scene,assets)['a']
    assert b[1][0]==pytest.approx(3/np.sqrt(2))
    request=SpatialRequest.parse(req([constraint('left_of'),constraint('near')]),['a','b'])
    _,result=solve(scene,request,assets)
    assert result is not None and result.instances[0].transform.uniform_scale==2
    np.testing.assert_array_equal(result.instances[0].transform.rotation_xyzw,scene.instances[0].transform.rotation_xyzw)
    assert validate_geometry(result,scene,request,assets)['passed']


def test_keep_pose_and_ground_conflict(setup):
    assets,scene=setup(positions={'a':[-1.2,0,0]})
    request=SpatialRequest.parse(req([constraint('keep_pose','a',None),constraint('left_of')]),['a','b'])
    _,result=solve(scene,request,assets)
    assert result is not None
    assert asdict(result.instances[0].transform)==asdict(scene.instances[0].transform)
    scene.instances[0].transform.translation_xyz[1]=.3
    request=SpatialRequest.parse(req([constraint('keep_pose','a',None)]),['a','b'])
    report,result=solve(scene,request,assets)
    assert result is None and report['proof']['proof']=='constraints_violate_locked_pose'


def test_near_is_not_permission_to_overlap(setup):
    assets,scene=setup();request=SpatialRequest.parse(req([constraint('near')]),['a','b'])
    report=validate_geometry(scene,scene,request,assets)
    assert not report['passed']
    assert any(c['relation']=='near' and c['status']=='passed' for c in report['checks'])
    assert any(c['relation']=='non_overlap' and c['status']=='failed' for c in report['checks'])


def test_deterministic_solver_and_unchanged_hashes(setup):
    assets,scene=setup(3);request=SpatialRequest.parse(req([constraint('left_of','a','b'),constraint('near','c','b')],fixed=['b']),['a','b','c'])
    hashes=[assets.get(i.asset_id).sha256 for i in scene.instances]
    d1,s1=solve(scene,request,assets);d2,s2=solve(scene,request,assets)
    assert asdict(s1)==asdict(s2) and d1==d2
    assert [assets.get(i.asset_id).sha256 for i in scene.instances]==hashes


def test_persisted_pipeline_and_independent_tamper_detection(setup,tmp_path,monkeypatch):
    assets,scene=setup();request=req([constraint('left_of'),constraint('near')])
    path=tmp_path/'request.json';atomic_json(path,request)
    out=tmp_path/'result';report=compile_layout(assets.root,assets.root/'scene.json',path,out)
    assert report['passed'],report
    assert read_json(out/'run.json')['status']=='succeeded'
    monkeypatch.setattr('src.storyboard.layout.solve',lambda *a:pytest.fail('validation called solver'))
    assert validate(out)['passed']
    # Deliberately move subject and refresh export + byte manifest: semantic validator still fails.
    repo=AssetRepository(out);saved=SceneRepository(repo).load();saved.instances[0].transform.translation_xyz[0]+=4
    SceneRepository(repo).save(saved);export(saved,repo)
    run=read_json(out/'run.json')
    for a in run['artifacts']:a['sha256']=sha256(out/a['path'])
    atomic_json(out/'run.json',run)
    assert not validate(out)['passed']
    assert read_json(out/'run.json')['status']=='failed'


def test_cli_conflict_exit_and_no_success_image(setup,tmp_path):
    assets,scene=setup();p=tmp_path/'conflict.json';atomic_json(p,req([constraint('left_of'),constraint('right_of')]))
    out=tmp_path/'failure';cli=[sys.executable,str(ROOT/'scripts/storyboard.py')]
    proc=subprocess.run(cli+['layout','--asset-root',str(assets.root),'--scene',str(assets.root/'scene.json'),'--request',str(p),'--out',str(out)],cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode==1,proc.stdout+proc.stderr
    assert read_json(out/'run.json')['error']['code']=='contradictory_constraints'
    assert not (out/'renders/layout_front.png').exists()
    assert read_json(out/'run.json')['status']=='failed'


def test_non_world_reference_is_rejected_explicitly(setup,tmp_path):
    assets,scene=setup();scene.reference_frame={'right':[0.,0.,1.]};SceneRepository(assets).save(scene)
    path=tmp_path/'request.json';atomic_json(path,req([constraint('left_of')]))
    result=compile_layout(assets.root,assets.root/'scene.json',path,tmp_path/'bad-frame')
    assert not result['passed'] and result['error']['code']=='unsupported_frame'


def test_missing_read_sources_are_not_created(tmp_path):
    root=tmp_path/'missing-assets';scene=tmp_path/'scene.json';request=tmp_path/'request.json'
    result=compile_layout(root,scene,request,tmp_path/'failure')
    assert not result['passed'] and not root.exists()
    with pytest.raises(FileNotFoundError):validate(tmp_path/'missing-run')
    assert not (tmp_path/'missing-run').exists()


def test_budget_exhaustion_is_explicit(setup):
    assets,scene=setup();request=SpatialRequest.parse(req([constraint('far',far_min=100)],max_evaluations=1),['a','b'])
    diagnostic,result=solve(scene,request,assets)
    assert result is None and diagnostic['status']=='layout_not_found'
    assert diagnostic['evaluations']==1 and diagnostic['budget_exhausted'] is True
    assert diagnostic['proof'] is None


def test_three_object_bounded_domain_not_found(setup):
    assets,scene=setup(3)
    request=SpatialRequest.parse(req([constraint('near','a','b'),constraint('near','c','b')],fixed=['b'],search_bounds=[-.1,.1,-.1,.1]),['a','b','c'])
    diagnostic,result=solve(scene,request,assets)
    assert result is None and diagnostic['status']=='layout_not_found'
    assert diagnostic['proof'] is None


def test_frozen_config_defaults_are_used(setup,tmp_path):
    import yaml
    assets,scene=setup();request=tmp_path/'request.json';atomic_json(request,req([constraint('left_of'),constraint('near')]))
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(dict(schema_version=1,spatial=dict(gap=.8,near_max=.5))))
    out=tmp_path/'configured'
    report=compile_layout(assets.root,assets.root/'scene.json',request,out,config_path=config)
    assert not report['passed'] and report['error']['code']=='contradictory_constraints'
    assert read_json(out/'spatial_request.json')['settings']['gap']==.8
    assert read_json(out/'config.json')['spatial']['gap']==.8
