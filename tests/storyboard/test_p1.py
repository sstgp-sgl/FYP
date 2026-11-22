"""Behavior/numerical acceptance for P1, including deliberately corrupted saved outputs."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
import yaml

from src.pointcloud_io import write_ply
from src.scene.composition import export
from src.scene.models import Transform
from src.scene.storage import AssetRepository, SceneRepository, atomic_json, read_json, read_vertices, sha256
from src.storyboard import service
from src.storyboard.geometry import project
from src.storyboard.render_cpu import render
from src.storyboard.validation import validate_run

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/storyboard.yaml'


@pytest.fixture
def demo(tmp_path):
    root = tmp_path / 'run'
    report = service.demo(CONFIG, root)
    assert report['passed'], report
    return root


def camera():
    # Hand-computable camera: world points at z>0 are in front; +Y goes DOWN.
    return dict(schema_version=1, projection='perspective', convention='opencv_x_right_y_down_z_forward',
                width=11, height=11, near=.1, far=10, K=[[10, 0, 5], [0, 10, 5], [0, 0, 1]],
                T_world_from_camera=np.eye(4).tolist())


def test_saved_demo_and_visibility(demo):
    report = validate_run(demo)
    checks = {c['name']: c for c in report['checks']}
    assert report['passed']
    assert checks['fixed_gap']['measured'] == pytest.approx(.2, abs=1e-6)
    assert checks['on_ground:red']['measured'] == pytest.approx(0)
    assert checks['non_overlap_red_blue']['measured'] == 0
    ids = np.load(demo/'renders/scene_01.ids.npy')
    assert set(np.unique(ids)) == {0, 1, 2, 3}
    assert all(checks['visible_pixels:'+i]['measured'] >= 100 for i in ['red', 'blue'])
    with Image.open(demo/'renders/scene_01.png') as im:
        assert im.size == (640, 480)
        rgb = np.asarray(im)
    red, blue = rgb[ids == 1], rgb[ids == 2]
    assert (red[:, 0] > red[:, 2]).mean() > .9
    assert (blue[:, 2] > blue[:, 0]).mean() > .9
    assert read_json(demo/'run.json')['status'] == 'succeeded'
    assert 'scene_01.png' in (demo/'storyboard.html').read_text()


def test_moved_to_right_fails_even_when_export_rehashed(demo, monkeypatch):
    assets = AssetRepository(demo)
    scene = SceneRepository(assets).load()
    red = next(i for i in scene.instances if i.instance_id == 'red')
    red.transform.translation_xyz[0] += 4
    SceneRepository(assets).save(scene)
    export(scene, assets)
    run = read_json(demo/'run.json')
    for a in run['artifacts']:
        a['sha256'] = sha256(demo/a['path'])
    atomic_json(demo/'run.json', run)
    # Independence: no solver can silently restore the requested scene.
    monkeypatch.setattr(service, 'place', lambda *a: pytest.fail('validator called layout'))
    report = service.validate(demo)
    assert not report['passed']
    assert any(c['name'] == 'requested_left_of' and c['status'] == 'failed' for c in report['checks'])
    assert read_json(demo/'run.json')['status'] == 'failed'
    assert next(i for i in SceneRepository(assets).load().instances if i.instance_id == 'red').transform.translation_xyz[0] == red.transform.translation_xyz[0]


def test_source_hash_tampering_rejected(demo):
    source = next((demo/'assets').rglob('data.ply'))
    source.write_bytes(source.read_bytes()+b'\n')
    report = validate_run(demo)
    assert not report['passed']
    with pytest.raises(ValueError, match='hash'):
        SceneRepository(AssetRepository(demo)).load()


def test_roundtrip_no_drift_or_source_mutation(demo):
    repo = SceneRepository(AssetRepository(demo))
    scene = repo.load()
    before = asdict(scene)
    paths = list((demo/'assets').rglob('data.ply'))
    hashes = [sha256(p) for p in paths]
    p0, c0 = read_vertices(demo/'combined.ply', 'point_cloud')
    for _ in range(3):
        repo.save(repo.load())
    export(repo.load(), repo.assets, 'roundtrip.ply')
    p1, c1 = read_vertices(demo/'roundtrip.ply', 'point_cloud')
    assert asdict(repo.load()) == before
    np.testing.assert_array_equal(p0, p1)
    np.testing.assert_array_equal(c0, c1)
    assert [sha256(p) for p in paths] == hashes


def test_projection_hand_computed_and_xyzw_rotation():
    shot = camera()
    uv, depth, front = project([[0, 0, 2], [.4, .2, 2], [0, 0, -1]], shot)
    np.testing.assert_allclose(uv[:2], [[5, 5], [7, 6]])
    assert front.tolist() == [True, True, False]
    q = Transform(rotation_xyzw=[0, 0, 2**-.5, 2**-.5])
    np.testing.assert_allclose(q.matrix() @ [1, 0, 2, 1], [0, 1, 2, 1], atol=1e-12)
    uv, _, _ = project([[0, 1, 2]], shot)
    np.testing.assert_allclose(uv, [[5, 10]])


@pytest.mark.parametrize('reverse', [False, True])
def test_zbuffer_nearer_point_wins_in_both_input_orders(reverse):
    p = np.array([[0, 0, 4], [0, 0, 2]])
    c = np.array([[0, 0, 255], [255, 0, 0]], np.uint8)
    ids = np.array([2, 1], np.int32)
    if reverse:
        p, c, ids = p[::-1], c[::-1], ids[::-1]
    rgb, out, z = render(p, c, ids, camera(), point_size=3)
    np.testing.assert_array_equal(rgb[4:7, 4:7], np.tile([255, 0, 0], (3, 3, 1)))
    assert (out[4:7, 4:7] == 1).all()
    assert (z[4:7, 4:7] == 2).all()
    assert np.count_nonzero(out) == 9


@pytest.mark.parametrize('point', [[0, 0, -1], [0, 0, 0], [0, 0, .05], [0, 0, 11], [100, 0, 1]])
def test_clipping_no_foreground(point):
    rgb, ids, z = render([point], np.array([[255, 0, 0]], np.uint8), np.array([1]), camera())
    assert not ids.any()
    assert np.isinf(z).all()
    assert (rgb == np.array([239, 242, 247])).all()


def test_equal_depth_tie_and_reject_empty():
    rgb, ids, _ = render([[0, 0, 2], [0, 0, 2]], np.array([[255, 0, 0], [0, 0, 255]], np.uint8), np.array([1, 2]), camera(), 1)
    assert ids[5, 5] == 1 and rgb[5, 5].tolist() == [255, 0, 0]
    with pytest.raises(ValueError, match='empty'):
        render(np.empty((0, 3)), np.empty((0, 3), np.uint8), np.array([], int), camera())


def test_deterministic_runs(demo, tmp_path):
    other = tmp_path/'second'
    assert service.demo(CONFIG, other)['passed']
    for p in ['combined.ply', 'renders/scene_01.png', 'renders/scene_01.ids.npy', 'renders/scene_01.depth.npy', 'shot.json']:
        assert sha256(demo/p) == sha256(other/p)
    a, b = read_json(demo/'scene.json'), read_json(other/'scene.json')
    assert [i['transform'] for i in a['instances']] == [i['transform'] for i in b['instances']]
    # Registry UUIDs and timestamps intentionally identify distinct runs.


def test_overwrite_protection(demo):
    before = sha256(demo/'run.json')
    with pytest.raises(FileExistsError):
        service.demo(CONFIG, demo)
    assert sha256(demo/'run.json') == before


@pytest.mark.parametrize('kind', ['missing', 'empty'])
def test_bad_input_creates_failed_run(tmp_path, kind):
    source = tmp_path/'input.ply'
    if kind == 'empty':
        write_ply(np.empty((0, 3)), np.empty((0, 3), np.uint8), source)
    root = tmp_path/'bad-run'
    with pytest.raises(RuntimeError, match='assets'):
        service.demo(CONFIG, root, source)
    run = read_json(root/'run.json')
    assert run['status'] == 'failed' and run['error']['stage'] == 'assets'
    assert not (root/'renders/scene_01.png').exists()


@pytest.mark.parametrize('path', ['renders/scene_01.png', 'renders/scene_01.ids.npy', 'combined.ply'])
def test_missing_output_is_not_a_cache_success(demo, path):
    (demo/path).unlink()
    assert not validate_run(demo)['passed']


def test_cli_validation_exit_codes(demo, tmp_path):
    cli = [sys.executable, str(ROOT/'scripts/storyboard.py')]
    assert subprocess.run(cli+['validate', '--run', str(demo)], cwd=ROOT, capture_output=True).returncode == 0
    (demo/'renders/scene_01.png').write_bytes(b'broken')
    assert subprocess.run(cli+['validate', '--run', str(demo)], cwd=ROOT, capture_output=True).returncode == 1
    missing = tmp_path/'no-such-run'
    assert subprocess.run(cli+['validate', '--run', str(missing)], cwd=ROOT, capture_output=True).returncode == 1
    assert not missing.exists()
    bad = tmp_path/'bad-config'
    proc = subprocess.run(cli+['demo', '--config', str(tmp_path/'absent.yaml'), '--out', str(bad)], cwd=ROOT, capture_output=True)
    assert proc.returncode == 1 and read_json(bad/'run.json')['status'] == 'failed'


def test_cpu_pipeline_no_model_imports(tmp_path):
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','transformers','diffusers','vggt','sharp','cv2','scipy'}:
   raise AssertionError('unexpected model import: '+fullname)
sys.meta_path.insert(0, Block())
from src.storyboard.service import demo
assert demo(sys.argv[1],sys.argv[2])['passed']
'''
    proc = subprocess.run([sys.executable, '-c', code, str(CONFIG), str(tmp_path/'isolated')], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout+proc.stderr
