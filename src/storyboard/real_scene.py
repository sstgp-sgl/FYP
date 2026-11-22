"""P2 asset snapshots and explicit/provisional coordinate previews, using the P1 renderer."""
from dataclasses import asdict
from pathlib import Path
import html
import json
import time
import subprocess
import numpy as np
from PIL import Image
from src.scene.models import Anchor, Instance, PlacementConstraint, SceneDocument, Transform
from src.scene.storage import atomic_json, read_json, read_vertices, sha256, SceneRepository
from src.scene.composition import place, export, transformed
from src.scene.validation import validate_export
from .assets import freeze_assets, CanonicalAssetRepository
from .geometry import camera_from_look_at
from .render_cpu import render
from .service import atomic_array


def fit_shot(points, width=960, height=640, side=0):
    center = (points.min(0)+points.max(0))/2
    radius = max(float(np.linalg.norm(points-center, axis=1).max()), .01)
    direction = np.array([np.sin(side), .18, np.cos(side)])
    direction /= np.linalg.norm(direction)
    eye = center + direction*radius*3.2
    return camera_from_look_at(eye.tolist(), center.tolist(), width, height, 45, radius*.01, radius*10)


def sample(points, colors, ids, budget=120000):
    # Stable source order sampling, no filtering by color; full combined.ply is retained.
    if len(points) <= budget:return points, colors, ids
    selected = np.linspace(0, len(points)-1, budget, dtype=int)
    return points[selected], colors[selected], ids[selected]


def save_render(root, name, points, colors, ids, side=0):
    shot = fit_shot(points, side=side)
    full_count = len(points)
    p, c, i = sample(points, colors, ids)
    rgb, identity, depth = render(p, c, i, shot, 3, (242, 244, 248))
    (root/'renders').mkdir(exist_ok=True)
    Image.fromarray(rgb).save(root/'renders'/(name+'.png'))
    atomic_array(root/'renders'/(name+'.ids.npy'), identity)
    atomic_array(root/'renders'/(name+'.depth.npy'), depth)
    shot.update(rendered_points=len(p), source_point_count=full_count, sampling='stable_even_indices_no_color_filter', point_size=3)
    atomic_json(root/'renders'/(name+'.shot.json'), shot)
    return dict(name=name, rendered_points=len(p), source_point_count=full_count,
                visible_pixels={str(k):int(np.count_nonzero(identity==k)) for k in np.unique(i)})


def compose(library, entries, out, allow_provisional=False, gap=.4):
    root = Path(out).resolve();root.mkdir(parents=True, exist_ok=False)
    started=time.monotonic()
    run=dict(schema_version=1,source='real',status='running',stage='snapshot',artifacts=[],human_status='pending')
    code_root=Path(__file__).resolve().parents[2]
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=code_root,capture_output=True,text=True)
    run.update(algorithm_version='real_assets_p2_v1',code_commit=git.stdout.strip() if git.returncode==0 else None,
               code_sha256={str(p.relative_to(code_root)):sha256(p) for p in sorted((code_root/'src/storyboard').glob('*.py'))})
    atomic_json(root/'run.json',run)
    try:
        if len(entries)!=2:raise ValueError('P2 composition requires exactly two real asset instances')
        repo=freeze_assets(library,[e['asset_id'] for e in entries],root)
        semantics=[repo.semantics(e['asset_id']) for e in entries]
        if not all(s['orientation_status']=='confirmed' and s['display_name_confirmed'] for s in semantics) and not allow_provisional:
            raise ValueError('needs_confirmation: use explicit --allow-provisional for inspection only')
        run['human_status']='confirmed' if all(s['orientation_status']=='confirmed' and s['display_name_confirmed'] for s in semantics) else 'needs_confirmation'
        atomic_json(root/'input_snapshot.json',dict(entries=entries,semantics=semantics,allow_provisional=allow_provisional,gap=gap))
        records=[repo.get(e['asset_id']) for e in entries]
        heights=[e.get('target_height',s['target_height']) for e,s in zip(entries,semantics)]
        if not np.isfinite(heights).all() or min(heights)<=0:raise ValueError('invalid instance target_height')
        scales=[h/s['target_height'] for h,s in zip(heights,semantics)]
        instances=[Instance(e['instance_id'],r.asset_id,r.sha256,Transform(uniform_scale=scale)) for e,r,scale in zip(entries,records,scales)]
        right_bounds=(np.asarray(records[1].bounds)*scales[1]).tolist()
        scene=SceneDocument('p2_real_assets',instances,[Anchor('right_anchor',right_bounds,instance_id=instances[1].instance_id)])
        scene,_=place(scene,PlacementConstraint(instances[0].instance_id,'right_anchor','left_of',gap,heights[0]),repo)
        SceneRepository(repo).save(scene)
        scene=SceneRepository(repo).load()
        export(scene,repo)
        run['stage']='render'
        atomic_json(root/'run.json',run)
        points,colors=read_vertices(root/'combined.ply','point_cloud')
        ranges=read_json(root/'combined.ranges.json')['ranges'];ids=np.zeros(len(points),np.int32)
        labels={e['instance_id']:i+1 for i,e in enumerate(entries)}
        for r in ranges:ids[r['start']:r['end']]=labels[r['instance_id']]
        previews=[save_render(root,'scene_front',points,colors,ids)]
        previews.append(save_render(root,'scene_side',points,colors,ids,side=.75))
        for entry in entries:
            p,c=repo.points(repo.get(entry['asset_id']));i=np.ones(len(p),np.int32)
            previews.append(save_render(root,entry['instance_id']+'_front',p,c,i))
            previews.append(save_render(root,entry['instance_id']+'_side',p,c,i,side=np.pi/2))
        # Recompute from persisted source/canonical/instance transforms through legacy validator.
        report=validate_export(CanonicalAssetRepository(root))
        run['stage']='validation'
        for render_info in previews:
            if any(v<=0 for v in render_info['visible_pixels'].values()):raise ValueError('invisible target in '+render_info['name'])
        report.update(human_status=run['human_status'],source='real',previews=previews,
                      scope='Real registered bytes, canonical/instance transforms, geometry and target visibility; perceptual/orientation confirmation pending')
        atomic_json(root/'validation.json',report)
        if not report['passed']:raise ValueError('geometry validation failed')
        paths=[p for p in root.rglob('*') if p.is_file() and p.name!='run.json']
        run['artifacts']=[dict(path=str(p.relative_to(root)),sha256=sha256(p)) for p in sorted(paths)]
        run.update(status='succeeded',stage='completed',duration_seconds=time.monotonic()-started,previews=previews)
        atomic_json(root/'run.json',run)
        cards=[]
        for entry,sem in zip(entries,semantics):
            name=html.escape(sem['display_name']);folder='assets/'+entry['asset_id']+'/'
            cards.append('<section><h2>'+name+' — 待确认名称与朝向</h2><img class="source" src="'+folder+html.escape(sem['source_image'],quote=True)+'" alt="来源图片"><img src="renders/'+html.escape(entry['instance_id'],quote=True)+'_front.png" alt="规范化点云正视"><img src="renders/'+html.escape(entry['instance_id'],quote=True)+'_side.png" alt="规范化点云侧视"><p><a href="'+folder+'semantics.json">坐标、名称与来源记录</a></p></section>')
        page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P2 真实资产预览</title><style>body{font:16px/1.7 system-ui;background:#f3f5f8;color:#243044;margin:0}main{max-width:1120px;margin:32px auto;padding:24px}img{max-width:100%;width:480px;vertical-align:top}.source{width:220px}a{color:#205caa}.status{padding:16px;background:#fff1ce}section{margin:32px 0}</style><main><p>FYP / P2 / REAL RECONSTRUCTED POINT CLOUDS</p><h1>两份真实重建资产同场景</h1><p class="status">CPU 渲染与几何检查通过；名称、上方、前方与可辨认性仍需用户确认。当前朝向为相机元数据提出的候选，不代表人工真值。</p><img src="renders/scene_front.png" alt="两份真实点云同场景正视"><img src="renders/scene_side.png" alt="同场景侧视"><p>保留全部源点和 RGB。预览使用稳定下采样，完整组合点云单独保存。左侧为第一份资产，右侧为第二份资产。</p>'''+''.join(cards)+'''<p><a href="scene.json">场景</a> · <a href="combined.ply">完整点云</a> · <a href="validation.json">验证报告</a> · <a href="run.json">运行记录</a></p><p>不报告 Chamfer/F-score；没有三维真值。单张图片重建含生成视图，不代表真实不可见面几何。</p></main></html>'''
        (root/'storyboard.html').write_text(page)
        return report
    except Exception as exc:
        run.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc)),duration_seconds=time.monotonic()-started)
        atomic_json(root/'run.json',run)
        raise


def validate(root):
    """Standalone independent validation of frozen real-asset artifacts; never lays out."""
    root=Path(root).resolve()
    run=read_json(root/'run.json')
    errors=[]
    try:
        report=validate_export(CanonicalAssetRepository(root))
    except Exception as exc:
        report=dict(schema_version=1,source='real',passed=False,error=str(exc))
    from src.scene.storage import controlled
    if not run.get('artifacts'):errors.append('missing artifact manifest')
    for a in run.get('artifacts',[]):
        try:
            p=controlled(root,a['path'])
            if sha256(p)!=a['sha256']:errors.append('hash mismatch: '+a['path'])
        except Exception as exc:errors.append(str(exc))
    for preview in run.get('previews',[]):
        try:
            name=preview['name']
            # IDs cannot be used as arbitrary filesystem names.
            from src.scene.models import identifier
            identifier(name)
            ids=np.load(root/'renders'/(name+'.ids.npy'),allow_pickle=False)
            with Image.open(root/'renders'/(name+'.png')) as image:
                image.load()
                if ids.shape!=(image.height,image.width):errors.append('image/ID shape mismatch: '+name)
            for target_id in preview['visible_pixels']:
                if not np.any(ids==int(target_id)):errors.append('invisible target: '+name+'/'+target_id)
        except Exception as exc:errors.append(str(exc))
    report.update(source='real',human_status=run.get('human_status','needs_confirmation'),artifact_errors=errors,
                  passed=bool(report.get('passed')) and not errors)
    # Keep the original accepted validation.json immutable as a hashed artifact.
    atomic_json(root/'revalidation.json',report)
    if not report['passed']:
        run.update(status='failed',error=dict(stage='validation',message='persisted real-asset artifacts failed validation'))
        atomic_json(root/'run.json',run)
    return report
