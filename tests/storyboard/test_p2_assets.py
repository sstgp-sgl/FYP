from dataclasses import asdict
from itertools import product
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
from src.pointcloud_io import write_ply
from src.scene.models import Transform
from src.scene.composition import export
from src.scene.storage import AssetRepository, SceneRepository, sha256, read_json
from src.storyboard.assets import import_asset, CanonicalAssetRepository, canonical_transform, freeze_assets
from src.storyboard.real_scene import compose

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def source(tmp_path):
    p=np.array(list(product([0.,1.],[0.,2.],[0.,3.])))
    c=np.full(p.shape,255,np.uint8) # light/white points must never be dropped at import
    ply=tmp_path/'input.ply';write_ply(p,c,ply)
    image=tmp_path/'source.png';Image.new('RGB',(32,32),'white').save(image)
    m=dict(display_name='fixture-real-label-test-only',source_image=str(image),producer='test_fixture',
           source_up=[0,0,1],source_forward=[0,-1,0],target_height=1.5,
           confirmed_by_user=False,display_name_confirmed=False)
    return ply,m


def test_raw_bytes_rgb_and_canonical_once(source,tmp_path):
    ply,m=source;before=sha256(ply);lib=tmp_path/'lib'
    info=import_asset(lib,ply,m)
    raw=AssetRepository(lib).get(info['asset_id'])
    assert sha256(ply)==before==raw.sha256
    assert raw.coordinate_frame=='source_reconstruction_world'
    assert info['orientation_status']=='proposed' and not info['confirmed_by_user']
    repo=CanonicalAssetRepository(lib);r=repo.get(raw.asset_id)
    p,c=repo.points(r)
    np.testing.assert_allclose([p.min(0),p.max(0)],[[-.25,0,-.5],[.25,1.5,.5]],atol=1e-12)
    assert len(p)==8 and (c==255).all()
    for _ in range(3):np.testing.assert_array_equal(repo.points(repo.get(raw.asset_id))[0],p)
    assert sha256(lib/raw.relative_path)==before


def test_unknown_orientation_is_not_confirmed(source,tmp_path):
    ply,m=source;m.pop('source_up');m.pop('source_forward')
    info=import_asset(tmp_path/'lib',ply,m)
    assert info['orientation_status']=='needs_orientation'
    assert info['source_to_canonical'] is None
    with pytest.raises(ValueError,match='needs_orientation'):
        CanonicalAssetRepository(tmp_path/'lib').get(info['asset_id'])
    m.update(confirmed_by_user=True,confirmation_actor='user')
    with pytest.raises(ValueError,match='confirmation'):
        import_asset(tmp_path/'lib',ply,m)


def test_explicit_user_confirmation_required(source,tmp_path):
    ply,m=source;m['confirmed_by_user']=True
    with pytest.raises(ValueError,match='confirmation'):
        import_asset(tmp_path/'lib',ply,m)
    m['confirmation_actor']='user';m['display_name_confirmed']=True
    info=import_asset(tmp_path/'lib',ply,m)
    assert info['orientation_status']=='confirmed'


def test_two_instances_share_raw_source_and_remain_independent(source,tmp_path):
    ply,m=source;lib=tmp_path/'lib';s=import_asset(lib,ply,m)
    entries=[dict(instance_id='left',asset_id=s['asset_id'],target_height=1.0),dict(instance_id='right',asset_id=s['asset_id'],target_height=2.0)]
    out=tmp_path/'preview';report=compose(lib,entries,out,allow_provisional=True)
    assert report['passed'] and report['human_status']=='needs_confirmation'
    repo=CanonicalAssetRepository(out);scene=SceneRepository(repo).load();before=asdict(scene)
    assert scene.instances[0].asset_id==scene.instances[1].asset_id
    assert scene.instances[0].transform.translation_xyz!=scene.instances[1].transform.translation_xyz
    assert scene.instances[0].transform.uniform_scale==pytest.approx(2/3)
    assert scene.instances[1].transform.uniform_scale==pytest.approx(4/3)
    rawhash=sha256(out/'assets'/s['asset_id']/'data.ply')
    export(scene,repo,'again.ply')
    assert sha256(out/'combined.ply')==sha256(out/'again.ply')
    SceneRepository(repo).save(SceneRepository(repo).load())
    assert asdict(SceneRepository(repo).load())==before
    assert sha256(out/'assets'/s['asset_id']/'data.ply')==rawhash
    assert len(list((out/'assets').iterdir()))==1
    assert (out/'renders/scene_front.png').is_file()
    from src.storyboard.real_scene import validate
    assert validate(out)['passed']
    proc=subprocess.run([sys.executable,str(ROOT/'scripts/storyboard.py'),'validate','--run',str(out)],cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode==0,proc.stdout+proc.stderr
    (out/'renders/scene_front.png').write_bytes(b'broken image')
    assert not validate(out)['passed']
    assert read_json(out/'run.json')['status']=='failed'


def test_proposal_needs_explicit_preview_flag(source,tmp_path):
    ply,m=source;s=import_asset(tmp_path/'lib',ply,m)
    entries=[dict(instance_id='a',asset_id=s['asset_id']),dict(instance_id='b',asset_id=s['asset_id'])]
    with pytest.raises(ValueError,match='needs_confirmation'):
        compose(tmp_path/'lib',entries,tmp_path/'bad')
    assert read_json(tmp_path/'bad/run.json')['status']=='failed'


def test_frozen_snapshot_survives_upstream_overwrite(source,tmp_path):
    ply,m=source;lib=tmp_path/'lib';s=import_asset(lib,ply,m)
    snap=tmp_path/'snapshot';snap.mkdir();repo=freeze_assets(lib,[s['asset_id']],snap)
    p,c=repo.points(repo.get(s['asset_id']));before=sha256(snap/'assets'/s['asset_id']/'data.ply')
    ply.write_text('overwritten by old pipeline')
    (lib/'assets'/s['asset_id']/'data.ply').write_text('changed upstream library')
    np.testing.assert_array_equal(repo.points(repo.get(s['asset_id']))[0],p)
    assert sha256(snap/'assets'/s['asset_id']/'data.ply')==before


@pytest.mark.parametrize('kind',['empty','nan','gaussian','bad_rgb'])
def test_reject_invalid_ply(source,tmp_path,kind):
    from plyfile import PlyData,PlyElement
    ply,m=source
    if kind=='empty':write_ply(np.empty((0,3)),np.empty((0,3),np.uint8),ply)
    elif kind=='nan':write_ply(np.array([[0,np.nan,1]]),np.array([[255,255,255]],np.uint8),ply)
    else:
        dtype=[('x','f4'),('y','f4'),('z','f4'),('red','f4' if kind=='bad_rgb' else 'u1'),('green','u1'),('blue','u1')]
        if kind=='gaussian':dtype+=[('f_dc_0','f4')]
        v=np.zeros(4,dtype=dtype);v['x']=range(4);v['y']=range(4);v['z']=range(4)
        if kind=='bad_rgb':v['red']=999
        PlyData([PlyElement.describe(v,'vertex')]).write(ply)
    with pytest.raises(ValueError):import_asset(tmp_path/'lib',ply,m)


def test_yaw_is_explicit_and_known_rotation():
    p=np.array(list(product([0.,1.],[0.,2.],[0.,3.])))
    t=canonical_transform(p,[0,1,0],[0,0,1],2,90)
    q=t.matrix()[:3,:3]@np.array([0,0,1.])
    np.testing.assert_allclose(q,[1,0,0],atol=1e-12)
    with pytest.raises(ValueError,match='parallel'):
        canonical_transform(p,[0,1,0],[0,2,0],2)


@pytest.mark.parametrize('views',[None,7])
def test_new_generation_defaults_six_but_explicit_seven_works(source,tmp_path,views):
    _,m=source
    args=[sys.executable,str(ROOT/'scripts/vggt_headless.py'),'--image',m['source_image'],
          '--out',str(tmp_path/'dry'),'--dry-run']
    if views:args+=['--views',str(views)]
    proc=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
    assert proc.returncode==0,proc.stdout+proc.stderr
    assert read_json(tmp_path/'dry/run.json')['views']==(views or 6)


def test_real_multiview_manifest_not_globbed_or_truncated(tmp_path):
    from src.io_utils import list_view_images,save_json
    for n in range(9):Image.new('RGB',(4,4)).save(tmp_path/f'actual_{n}.png')
    Image.new('RGB',(4,4)).save(tmp_path/'grid_raw.png')
    save_json(dict(views=[dict(file=f'actual_{n}.png') for n in range(9)]),tmp_path/'views.json')
    assert [p.name for p in list_view_images(tmp_path)]==[f'actual_{n}.png' for n in range(9)]
