"""Shared P1 CPU workflow used by CLI. Writes only to a fresh, isolated run directory."""
from dataclasses import asdict
from itertools import product
from pathlib import Path
import html
import json
import os
import subprocess
import tempfile
import time
import numpy as np
from PIL import Image
import yaml

from src.pointcloud_io import write_ply
from src.scene.composition import place, export
from src.scene.models import Anchor, Instance, PlacementConstraint, SceneDocument, Y_UP, now
from src.scene.storage import AssetRepository, SceneRepository, atomic_json, sha256
from .geometry import camera_from_look_at
from .render_cpu import render
from .validation import validate_run


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    with open(path) as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or cfg.get('schema_version') != 1:
        raise ValueError('unsupported config schema_version')
    r, f = cfg['render'], cfg['fixture']
    for key in ('gap', 'target_height', 'epsilon'):
        if type(f[key]) not in (int, float) or not np.isfinite(f[key]) or f[key] <= 0:
            raise ValueError('invalid fixture ' + key)
    if f['epsilon'] > 1e-3 or type(f['surface_steps']) is not int or not 5 <= f['surface_steps'] <= 101:
        raise ValueError('invalid fixture resolution/tolerance')
    if type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**32:
        raise ValueError('invalid seed')
    if type(r['point_size']) is not int or not 1 <= r['point_size'] <= 15 or r['point_size'] % 2 != 1:
        raise ValueError('invalid point_size')
    if not 0 <= r['margin_fraction'] <= .25 or type(r['min_visible_pixels']) is not int or r['min_visible_pixels'] < 1:
        raise ValueError('invalid visibility settings')
    if not 1 <= r['fov_degrees'] < 179:
        raise ValueError('invalid fov')
    camera_from_look_at(r['eye'], r['target'], r['width'], r['height'], r['fov_degrees'], r['near'], r['far'])
    return cfg


def cuboid(lo, hi, steps):
    axes = [np.linspace(a, b, steps) for a, b in zip(lo, hi)]
    faces = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for end in (lo[axis], hi[axis]):
            face = np.empty((steps**2, 3))
            face[:, axis] = end
            face[:, other] = list(product(axes[other[0]], axes[other[1]]))
            faces.append(face)
    return np.unique(np.concatenate(faces), axis=0)


def atomic_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.pending-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def create_assets(root, cfg, red_ply=None):
    assets = AssetRepository(root)
    n = cfg['fixture']['surface_steps']
    sources = {}
    for name, lo, hi, color in (
        ('red', [2, 1, -2], [4, 4, 0], [226, 60, 48]),
        ('blue', [-.65, 0, -.5], [.65, 1.7, .5], [42, 111, 214]),
    ):
        p = cuboid(lo, hi, n)
        c = np.tile(np.array(color, np.uint8), (len(p), 1))
        # A small light patch on the upper front-left corner makes orientation visible.
        patch = (p[:, 0] < lo[0]+(hi[0]-lo[0])*.25) & (p[:, 1] > hi[1]-(hi[1]-lo[1])*.18) & (p[:, 2] == hi[2])
        c[patch] = [255, 216, 100]
        path = root / 'fixture' / (name + '.ply')
        write_ply(p, c, path)
        sources[name] = path
    xz = np.array(list(product(np.linspace(-3.6, 2.8, 101), np.linspace(-2.6, 2.6, 81))))
    ground = np.column_stack([xz[:, 0], np.zeros(len(xz)), xz[:, 1]])
    colors = np.tile(np.array([198, 208, 220], np.uint8), (len(ground), 1))
    sources['ground'] = root / 'fixture/ground.ply'
    write_ply(ground, colors, sources['ground'])
    if red_ply is not None:
        sources['red'] = Path(red_ply)
    records = {}
    for name, path in sources.items():
        records[name] = assets.import_asset(path, role='subject' if name == 'red' else 'scene',
            representation='point_cloud', source_kind='synthetic', producer='storyboard_p1_cuboids_v1',
            coordinate_frame=Y_UP, provenance=dict(fixture_name=name, seed=cfg['seed'], synthetic=True))
    return assets, records


def write_index(root, report):
    status = html.escape(report['status'].upper())
    failed = ''.join('<li>'+html.escape(c['name'])+': '+html.escape(json.dumps(c.get('measured'), ensure_ascii=False))+'</li>' for c in report['checks'] if c['status'] == 'failed')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P1 CPU 故事板</title><style>body{font:16px/1.7 system-ui;background:#f4f5f7;color:#243044;margin:0}main{max-width:960px;margin:36px auto;padding:24px}img{width:100%;max-width:640px;background:white}a{color:#205cba}code{overflow-wrap:anywhere}.status{font-weight:700;padding:12px;background:#e3eaf4}</style><main><p>FYP / P1 / SYNTHETIC</p><h1>合成资产 → 静态三维场景 → CPU 图片</h1><p class="status">'''+status+'''</p><p>红色长方体位于蓝色锚点左侧；两个主体落地、互不重叠。黄色角标用于辨认朝向。全部对象为合成测试资产。</p><img src="renders/scene_01.png" alt="CPU 透视投影的红蓝长方体与地面"><p>实际点云透视投影，逐像素深度缓冲；无 GPU、无模型推理。</p><p><a href="scene.json">场景</a> · <a href="shot.json">相机</a> · <a href="validation.json">独立验证</a> · <a href="run.json">运行记录</a> · <a href="combined.ply">点云</a> · <a href="renders/scene_01.ids.npy">实例 ID 图</a> · <a href="renders/scene_01.depth.npy">深度图</a> · <a href="config.json">配置快照</a></p><ul>'''+failed+'''</ul><p>这是单场景合成验收，尚未接真实重建资产或 LLM 三幕故事。</p></main></html>'''
    (root / 'storyboard.html').write_text(page)


def demo(config_path, out, red_ply=None):
    root = Path(out).resolve()
    # Atomic exclusive creation: never overwrite an old success or partial failure.
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    run = dict(schema_version=1, algorithm_version='storyboard_p1_v1', run_id=root.name,
               source='synthetic', status='running', started_at=now(), stages=[], artifacts=[])
    stage = 'config'

    def record():
        atomic_json(root / 'run.json', run)

    def begin(name):
        nonlocal stage
        stage = name
        run['stages'].append(dict(name=name, status='running', started_at=now()))
        record()

    def finish():
        run['stages'][-1].update(status='succeeded', finished_at=now())
        record()

    record()
    try:
        begin('config')
        cfg = load_config(config_path)
        atomic_json(root / 'config.json', cfg)
        atomic_json(root / 'input_snapshot.json', dict(source='synthetic', seed=cfg['seed'],
            relation='left_of', subject='red', object='blue', gap=cfg['fixture']['gap'],
            target_height=cfg['fixture']['target_height'], epsilon=cfg['fixture']['epsilon']))
        code_files = sorted((PROJECT_ROOT / 'src/storyboard').glob('*.py')) + [PROJECT_ROOT / 'scripts/storyboard.py']
        run['code_sha256'] = {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in code_files}
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, capture_output=True, text=True)
        run['code_commit'] = git.stdout.strip() if git.returncode == 0 else None
        run['config_hash'] = sha256(root / 'config.json')
        run['python'] = __import__('sys').version
        run['provider'] = None
        finish()
        begin('assets')
        assets, records = create_assets(root, cfg, red_ply)
        run['input_hashes'] = {name: r.sha256 for name, r in records.items()}
        finish()
        begin('layout')
        scene = SceneDocument('p1_synthetic', [Instance(name, r.asset_id, r.sha256) for name, r in records.items()],
            [Anchor('blue_anchor', records['blue'].bounds, source='synthetic', instance_id='blue')])
        f = cfg['fixture']
        scene, _ = place(scene, PlacementConstraint('red', 'blue_anchor', 'left_of', f['gap'], f['target_height']), assets)
        SceneRepository(assets).save(scene)
        # Reload before export; source coordinates are never normalized twice.
        export(SceneRepository(assets).load(), assets)
        finish()
        begin('render')
        from src.scene.storage import read_vertices, read_json
        points, colors = read_vertices(root / 'combined.ply', 'point_cloud')
        ranges = read_json(root / 'combined.ranges.json')['ranges']
        mapping = {'red': 1, 'blue': 2, 'ground': 3}
        ids = np.empty(len(points), np.int32)
        for r in ranges:
            ids[r['start']:r['end']] = mapping[r['instance_id']]
        r = cfg['render']
        shot = camera_from_look_at(r['eye'], r['target'], r['width'], r['height'], r['fov_degrees'], r['near'], r['far'])
        shot.update(scene_id=scene.scene_id, shot_id='shot_01', instance_ids=mapping,
                    rasterization=dict(point_size=r['point_size'], depth_test='strict_less', tie_policy='first_point', background_id=0))
        atomic_json(root / 'shot.json', shot)
        rgb, ids, depth = render(points, colors, ids, shot, r['point_size'], tuple(r['background']))
        (root / 'renders').mkdir()
        Image.fromarray(rgb).save(root / 'renders/scene_01.png')
        atomic_array(root / 'renders/scene_01.ids.npy', ids)
        atomic_array(root / 'renders/scene_01.depth.npy', depth)
        finish()
        # Store the exact bytes being accepted; empty/cached manifests are rejected.
        paths = ['config.json', 'input_snapshot.json', 'scene.json', 'shot.json', 'combined.ply', 'combined.ranges.json',
                 'renders/scene_01.png', 'renders/scene_01.ids.npy', 'renders/scene_01.depth.npy']
        paths += [str(p.relative_to(root)) for p in sorted((root / 'assets').rglob('*')) if p.is_file()]
        run['artifacts'] = [dict(path=p, sha256=sha256(root/p)) for p in paths]
        begin('validation')
        report = validate_run(root)
        atomic_json(root / 'validation.json', report)
        write_index(root, report)
        if not report['passed']:
            raise ValueError('independent validation failed: ' + ', '.join(c['name'] for c in report['checks'] if c['status'] == 'failed'))
        finish()
        run.update(status='succeeded', finished_at=now(), duration_seconds=time.monotonic()-started, exit_code=0)
        record()
        return report
    except Exception as exc:
        if run['stages']:
            run['stages'][-1].update(status='failed', finished_at=now())
        run.update(status='failed', finished_at=now(), exit_code=1, duration_seconds=time.monotonic()-started,
                   error=dict(stage=stage, type=type(exc).__name__, message=str(exc)))
        record()
        (root / 'error.log').write_text(stage + ': ' + type(exc).__name__ + ': ' + str(exc) + '\n')
        raise RuntimeError(stage + ': ' + str(exc)) from exc


def validate(root):
    root = Path(root).resolve()
    report = validate_run(root)
    atomic_json(root / 'validation.json', report)
    from src.scene.storage import read_json
    run = read_json(root / 'run.json')
    # Standalone validation never calls layout, exports points, or repairs assets.
    if not report['passed']:
        run.update(status='failed', exit_code=1, error=dict(stage='validation', message='persisted artifacts failed validation'))
        atomic_json(root / 'run.json', run)
    write_index(root, report)
    return report
