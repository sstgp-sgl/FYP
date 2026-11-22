import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from src.reconstruction_methods import service
from src.reconstruction_methods.common import prepare_foreground,sample_surface,write_ply


def inputs(tmp_path):
    image=tmp_path/'image.png';mask=tmp_path/'mask.png'
    Image.new('RGB',(8,8),'white').save(image)
    alpha=np.zeros((8,8),np.uint8);alpha[2:6,2:6]=255;Image.fromarray(alpha).save(mask)
    return image,mask


def test_white_background_requires_mask(tmp_path):
    image,mask=inputs(tmp_path)
    with pytest.raises(ValueError,match='opaque'):prepare_foreground(image)
    rgba=np.array(prepare_foreground(image,mask))
    assert rgba[0,0,3]==0 and rgba[3,3,3]==255
    assert (rgba[:,:,:3]==255).all() # White subject pixels must not be removed by RGB filtering.


def test_transparent_input_and_invalid_masks(tmp_path):
    image,mask=inputs(tmp_path);rgba=prepare_foreground(image,mask);rgba.save(tmp_path/'rgba.png')
    assert prepare_foreground(tmp_path/'rgba.png').getchannel('A').getextrema()==(0,255)
    Image.new('L',(8,8),0).save(mask)
    with pytest.raises(ValueError,match='empty'):prepare_foreground(image,mask)
    Image.new('L',(2,2),255).save(mask)
    with pytest.raises(ValueError,match='sizes'):prepare_foreground(image,mask)


def test_area_sampling_is_weighted_and_deterministic():
    v=np.array([[0,0,0],[1,0,0],[0,1,0],[10,0,0],[12,0,0],[10,2,0]])
    f=np.array([[0,1,2],[3,4,5]])
    p=sample_surface(v,f,10000,42)
    assert .78<(p[:,0]>5).mean()<.82
    assert np.array_equal(p,sample_surface(v,f,10000,42))
    with pytest.raises(ValueError):sample_surface(v,np.array([[0,0,0]]),10,42)


def test_ply_roundtrip_and_nan_rejection(tmp_path):
    from plyfile import PlyData
    p=np.array([[1,2,3],[-1,0,.5]]);c=np.array([[255,255,255],[0,4,255]],np.uint8)
    result=write_ply(tmp_path/'a.ply',p,c)
    assert result['points']==2
    assert list(PlyData.read(str(tmp_path/'a.ply'))['vertex']['red'])==[255,0]
    with pytest.raises(ValueError):write_ply(tmp_path/'bad.ply',np.array([[np.nan,0,0]]),c[:1])


def test_method_listing_loads_no_model():
    code="import sys;from src.reconstruction_methods import METHODS;assert len(METHODS)==3;assert not any(m in sys.modules for m in ['torch','diffusers','tsr','hy3dgen'])"
    subprocess.run([sys.executable,'-c',code],check=True)


@pytest.mark.parametrize('options',[{'points':0},{'resolution':512},{'seed':-1},{'gray':'yes'},{'unexpected':1}])
def test_bad_options(options):
    with pytest.raises(ValueError):service.resolve_options('triposr',options)


def fake_backend(monkeypatch,reconstruct):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True),manual_seed=lambda seed:None))
    original=importlib.import_module
    monkeypatch.setattr(service.importlib,'import_module',lambda name,package=None:SimpleNamespace(reconstruct=reconstruct) if name=='.triposr' else original(name,package))


def test_dispatch_normalization_manifest_and_no_overwrite(tmp_path,monkeypatch):
    image,mask=inputs(tmp_path)
    fake_backend(monkeypatch,lambda *args:(np.array([[0,0,0],[4,2,0]]),None,dict(test_fixture=True)))
    out=tmp_path/'run';r=service.reconstruct_asset('triposr',image,mask,out,{'points':2})
    assert r['status']=='succeeded' and r['artifacts'][0]['points']==2
    assert r['normalization']['uniform_scale']==.5
    with pytest.raises(FileExistsError):service.reconstruct_asset('triposr',image,mask,out,{'points':2})
    assert json.loads((out/'run.json').read_text())['status']=='succeeded'


def test_backend_failure_is_saved(tmp_path,monkeypatch):
    image,mask=inputs(tmp_path)
    def fail(*args):raise RuntimeError('test provider unavailable')
    fake_backend(monkeypatch,fail)
    with pytest.raises(RuntimeError):service.reconstruct_asset('triposr',image,mask,tmp_path/'failure')
    r=json.loads((tmp_path/'failure/run.json').read_text())
    assert r['status']=='failed' and not r['artifacts']
