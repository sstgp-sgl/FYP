"""P3 persisted layout workflow. Existing raw assets and P1/P2 scenes are never overwritten."""
from dataclasses import asdict
from pathlib import Path
import html,json,subprocess,time
import numpy as np
from src.scene.models import SceneDocument,Y_UP,now
from src.scene.storage import AssetRepository,SceneRepository,read_json,atomic_json,sha256,read_vertices
from src.scene.composition import export
from .assets import CanonicalAssetRepository,freeze_assets
from .spatial_contracts import SpatialRequest,SpatialError,DEFAULTS
from .layout import solve
from .spatial_validation import validate_geometry,validate_run
from .real_scene import save_render


class LayoutAssets(CanonicalAssetRepository):
    """Mixed scenes: canonicalize raw reconstructed assets, preserve existing Y-up assets."""
    def get(self,asset_id):
        raw=AssetRepository.get(self,asset_id)
        return raw if raw.coordinate_frame==Y_UP else super().get(asset_id)

    def points(self,record):
        raw=AssetRepository.get(self,record.asset_id)
        return AssetRepository.points(self,raw) if raw.coordinate_frame==Y_UP else super().points(record)


def index(root,report):
    status='PASS' if report.get('passed') else 'FAIL'
    image='<img src="renders/layout_front.png" alt="P3 实际三维点云布局">' if (root/'renders/layout_front.png').is_file() else ''
    failed=''.join('<li>'+html.escape(c['constraint_id'])+'：'+html.escape(str(c['measured']))+'</li>' for c in report.get('checks',[]) if c['status']=='failed')
    links=' · '.join('<a href="'+name+'">'+label+'</a>' for name,label in [('spatial_request.json','冻结的要求'),('request_input.raw.json','原始输入'),('validation.json','独立验证'),('solver.json','求解诊断'),('run.json','运行记录')] if (root/name).is_file())
    template="""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P3 空间约束布局</title><style>body{font:16px/1.7 system-ui;background:#f4f6f9;color:#263044;margin:0}main{max-width:1100px;margin:32px auto;padding:24px}img{max-width:100%;width:960px}a{color:#205ca0}pre{white-space:pre-wrap}</style><main><p>FYP / P3 / STRUCTURED CONSTRAINTS</p><h1>空间约束布局：__STATUS__</h1>__IMAGE__<p>仅使用结构化关系；无 LLM。世界 +X 为右，Y 向上；AABB 检查为保守近似。</p><p>__LINKS__</p><ul>__FAILURES__</ul><pre>__ERROR__</pre><p>真实资产的人工名称、上/前方确认状态沿用来源，未自动提升。</p></main></html>"""
    page=template.replace('__STATUS__',status).replace('__IMAGE__',image).replace('__LINKS__',links).replace('__FAILURES__',failed).replace('__ERROR__',html.escape(json.dumps(report.get('error',{}),ensure_ascii=False,indent=2)))
    (root/'storyboard.html').write_text(page)


def compile_layout(asset_root,scene_path,request_path,out,allow_provisional=False,config_path=None):
    root=Path(out).resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    run=dict(schema_version=1,pipeline='spatial_layout_p3',status='running',started_at=now(),stage='input',source='unknown',artifacts=[])
    atomic_json(root/'run.json',run)
    try:
        if not Path(asset_root).is_dir():raise SpatialError('missing_asset_root','source asset root does not exist')
        source_assets=LayoutAssets(asset_root)
        base=SceneRepository(source_assets).validate(SceneDocument(**read_json(scene_path)))
        if not np.allclose(base.reference_frame['right'],[1,0,0],atol=1e-12):raise SpatialError('unsupported_frame','P3 world reference right must be +X')
        visible=[i.instance_id for i in base.instances if i.visible]
        import shutil
        shutil.copyfile(request_path,root/'request_input.raw.json')
        data=read_json(request_path)
        atomic_json(root/'request_input.json',data)
        import yaml
        code_root=Path(__file__).resolve().parents[2]
        config_path=Path(config_path) if config_path is not None else code_root/'configs/storyboard.yaml'
        config_path=config_path if config_path.is_absolute() else code_root/config_path
        config=yaml.safe_load(config_path.read_text())
        if not isinstance(config,dict) or config.get('schema_version')!=1:raise SpatialError('invalid_config','unsupported config schema')
        atomic_json(root/'config.json',config)
        configured={**data,'settings':{**config.get('spatial',{}),**data.get('settings',{})}}
        request=SpatialRequest.parse(configured,visible)
        # Persist user requirements and resolved defaults; system checks are regenerated independently.
        raw_constraints=[{k:v for k,v in asdict(c).items() if k!='origin'} for c in request.constraints if c.origin=='user']
        frozen=dict(schema_version=1,constraints=raw_constraints,fixed_instance_ids=request.fixed_instance_ids,settings=request.settings)
        atomic_json(root/'spatial_request.json',frozen)
        atomic_json(root/'resolved_constraints.json',request.to_dict())
        atomic_json(root/'base_scene.json',asdict(base))
        kinds={source_assets.get(i.asset_id).source_kind for i in base.instances}
        run['source']=next(iter(kinds)) if len(kinds)==1 else 'mixed'
        ids=[i.asset_id for i in base.instances]
        freeze_assets(asset_root,ids,root)
        assets=LayoutAssets(root);human=[]
        for aid in dict.fromkeys(ids):
            sem_path=root/'assets'/aid/'semantics.json'
            if sem_path.is_file():
                sem=read_json(sem_path)
                human.append(sem['orientation_status']=='confirmed' and sem['display_name_confirmed'])
        run['human_status']='needs_confirmation' if human and not all(human) else ('confirmed' if human else 'synthetic')
        if run['human_status']=='needs_confirmation' and not allow_provisional:raise SpatialError('needs_confirmation','explicit --allow-provisional required for inspection of unconfirmed real assets')
        code_root=Path(__file__).resolve().parents[2]
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=code_root,capture_output=True,text=True)
        run['code_commit']=git.stdout.strip() if git.returncode==0 else None
        run['algorithm_version']='bounded_analytic_candidates_v1'
        run['code_sha256']={str(p.relative_to(code_root)):sha256(p) for p in sorted((code_root/'src/storyboard').glob('*.py'))}
        run['input_hashes']={name:sha256(root/name) for name in ['base_scene.json','spatial_request.json','config.json']}
        run['stage']='layout';atomic_json(root/'run.json',run)
        diagnostic,scene=solve(base,request,assets)
        atomic_json(root/'solver.json',diagnostic)
        if scene is None:raise SpatialError(diagnostic['status'],'no accepted layout',diagnostic)
        SceneRepository(assets).save(scene)
        # Independent geometry acceptance, before any render or success marker.
        report=validate_geometry(SceneRepository(assets).load(),base,request,assets)
        if not report['passed']:raise SpatialError('independent_validation_failed','candidate violates persisted geometry',report)
        export(SceneRepository(assets).load(),assets)
        run['stage']='render';atomic_json(root/'run.json',run)
        points,colors=read_vertices(root/'combined.ply','point_cloud')
        ranges=read_json(root/'combined.ranges.json')['ranges'];mapping={n:i+1 for i,n in enumerate(visible)}
        point_ids=np.zeros(len(points),np.int32)
        for r in ranges:point_ids[r['start']:r['end']]=mapping[r['instance_id']]
        preview=save_render(root,'layout_front',points,colors,point_ids)
        ids_image=np.load(root/'renders/layout_front.ids.npy',allow_pickle=False)
        if any(not np.any(ids_image==v) for v in mapping.values()):raise SpatialError('invisible_target','not all instances are visible')
        run['instance_ids']=mapping;run['preview']=preview;run['stage']='validation'
        paths=[p for p in root.rglob('*') if p.is_file() and p.name not in ('run.json','validation.json')]
        run['artifacts']=[dict(path=str(p.relative_to(root)),sha256=sha256(p)) for p in sorted(paths)]
        atomic_json(root/'run.json',run)
        report=validate_run(root,LayoutAssets(root));report['human_status']=run['human_status']
        atomic_json(root/'validation.json',report)
        if not report['passed']:raise SpatialError('independent_validation_failed','persisted artifacts failed',report)
        run.update(status='succeeded',stage='completed',finished_at=now(),duration_seconds=time.monotonic()-start,exit_code=0)
        atomic_json(root/'run.json',run);index(root,report)
        return report
    except Exception as exc:
        code=getattr(exc,'code','invalid_input' if run['stage']=='input' else 'stage_failed')
        error=dict(code=code,stage=run['stage'],type=type(exc).__name__,message=str(exc),evidence=getattr(exc,'evidence',{}))
        run.update(status='failed',error=error,finished_at=now(),duration_seconds=time.monotonic()-start,exit_code=1)
        atomic_json(root/'run.json',run)
        report=dict(schema_version=1,passed=False,status='failed',checks=[],error=error)
        atomic_json(root/'validation.json',report);index(root,report)
        return report


def validate(root):
    root=Path(root).resolve()
    if not root.is_dir():raise FileNotFoundError('missing spatial run directory')
    report=validate_run(root,LayoutAssets(root))
    atomic_json(root/'revalidation.json',report)
    run=read_json(root/'run.json')
    if not report['passed']:
        run.update(status='failed',exit_code=1,error=dict(code='independent_validation_failed',stage='validation'))
        atomic_json(root/'run.json',run)
    return report
