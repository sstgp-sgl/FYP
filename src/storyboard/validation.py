"""Re-read persisted artifacts and measure actual geometry. Never imports the layout solver."""
from pathlib import Path
import numpy as np
from PIL import Image
from src.scene.storage import AssetRepository, controlled, read_json, read_vertices, sha256
from src.scene.validation import validate_export
from .geometry import project


def validate_run(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError('missing run directory: ' + str(root))
    checks = []

    def check(name, passed, measured=None, expected=None, evidence=None):
        checks.append(dict(name=name, status='passed' if bool(passed) else 'failed',
                           measured=measured, expected=expected, evidence_paths=evidence or []))

    def guarded(name, fn):
        try:
            fn()
        except Exception as exc:
            check(name, False, dict(error_type=type(exc).__name__, message=str(exc)))

    def integrity():
        run = read_json(root / 'run.json')
        if not run.get('artifacts'):
            raise ValueError('run has no complete artifact manifest')
        for item in run['artifacts']:
            path = controlled(root, item['path'])
            check('sha256:' + item['path'], path.is_file() and sha256(path) == item['sha256'],
                  expected=item['sha256'], evidence=[item['path']])
    guarded('artifact_integrity', integrity)

    def geometry():
        assets = AssetRepository(root)
        old = validate_export(assets)
        check('legacy_export_coordinates_colors', old['passed'], old, evidence=['scene.json', 'combined.ply'])
        p, _ = read_vertices(root / 'combined.ply', 'point_cloud')
        index = read_json(root / 'combined.ranges.json')
        chunks = {r['instance_id']: p[r['start']:r['end']] for r in index['ranges']}
        spec = read_json(root / 'input_snapshot.json')
        check('required_instances', set(chunks) == {'red', 'blue', 'ground'}, sorted(chunks), ['blue', 'ground', 'red'])
        a, b = chunks['red'], chunks['blue']
        eps = spec['epsilon']
        if not 0 < eps <= 1e-3:
            raise ValueError('invalid acceptance tolerance')
        amin, amax, bmin, bmax = a.min(0), a.max(0), b.min(0), b.max(0)
        gap = float(bmin[0] - amax[0])
        check('requested_left_of', gap >= spec['gap']-eps, gap, dict(min_gap=spec['gap'], tolerance=eps), ['combined.ply'])
        check('fixed_gap', abs(gap-spec['gap']) <= eps, gap, spec['gap'])
        for name in ('red', 'blue', 'ground'):
            bottom = float(chunks[name][:, 1].min())
            check('on_ground:' + name, abs(bottom) <= eps, bottom, dict(value=0, tolerance=eps))
        check('subject_height', abs(float(np.ptp(a[:, 1]))-spec['target_height']) <= eps,
              float(np.ptp(a[:, 1])), spec['target_height'])
        overlap = np.maximum(0, np.minimum(amax[[0, 2]], bmax[[0, 2]]) - np.maximum(amin[[0, 2]], bmin[[0, 2]]))
        area = float(np.prod(overlap))
        check('non_overlap_red_blue', area <= eps**2, area, dict(max_area=eps**2))
    guarded('persisted_geometry', geometry)

    def picture():
        shot = read_json(root / 'shot.json')
        cfg = read_json(root / 'config.json')
        points, _ = read_vertices(root / 'combined.ply', 'point_cloud')
        index = read_json(root / 'combined.ranges.json')
        ids = np.load(root / 'renders/scene_01.ids.npy', allow_pickle=False)
        depth = np.load(root / 'renders/scene_01.depth.npy', allow_pickle=False)
        with Image.open(root / 'renders/scene_01.png') as im:
            im.load()
            check('png_resolution', im.size == (shot['width'], shot['height']) and im.mode == 'RGB',
                  dict(size=list(im.size), mode=im.mode), [shot['width'], shot['height']])
        shape = (shot['height'], shot['width'])
        check('buffer_shape', ids.shape == shape and depth.shape == shape and ids.dtype == np.int32,
              dict(ids=list(ids.shape), depth=list(depth.shape)))
        if ids.shape != shape or depth.shape != shape:
            raise ValueError('buffer dimensions mismatch')
        check('buffer_ids', set(np.unique(ids).tolist()) <= {0, 1, 2, 3}, np.unique(ids).tolist())
        foreground = ids > 0
        check('depth_clipping', np.all(np.isfinite(depth[foreground])) and np.all(depth[foreground] >= shot['near']) and np.all(depth[foreground] <= shot['far']) and np.isinf(depth[~foreground]).all())
        centroids = {}
        for r in index['ranges']:
            name = r['instance_id']
            instance_id = shot['instance_ids'][name]
            y, x = np.where(ids == instance_id)
            check('visible_pixels:' + name, len(x) >= cfg['render']['min_visible_pixels'], int(len(x)),
                  dict(min=cfg['render']['min_visible_pixels']), ['renders/scene_01.ids.npy'])
            if len(x):
                centroids[name] = float(x.mean())
            if name in ('red', 'blue'):
                uv, _, front = project(points[r['start']:r['end']], shot)
                margin = cfg['render']['margin_fraction']
                inside = front.all() and np.all(uv[:, 0] >= margin*shot['width']) and np.all(uv[:, 0] <= (1-margin)*shot['width']-1) and np.all(uv[:, 1] >= margin*shot['height']) and np.all(uv[:, 1] <= (1-margin)*shot['height']-1)
                check('uncropped:' + name, inside, dict(bounds=[uv.min(0).tolist(), uv.max(0).tolist()]), dict(margin_fraction=margin), ['shot.json'])
        check('image_left_right', centroids.get('red', np.inf) < centroids.get('blue', -np.inf), centroids)
    guarded('render_artifacts', picture)
    return dict(schema_version=1, source='synthetic', passed=all(c['status'] == 'passed' for c in checks),
                status='passed' if all(c['status'] == 'passed' for c in checks) else 'failed', checks=checks,
                scope='Persisted asset hashes, exported geometry, clipping, image integrity and visibility; no perceptual or real-model quality claim')
