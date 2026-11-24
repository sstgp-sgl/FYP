"""Shared masked input, area sampling, and ordinary RGB PLY output."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def prepare_foreground(image_path,mask_path=None):
    image=Image.open(image_path).convert('RGBA')
    if mask_path is not None:
        mask=Image.open(mask_path).convert('L')
        if mask.size!=image.size:raise ValueError('image and mask sizes differ')
        image.putalpha(mask)
    alpha=np.asarray(image.getchannel('A'))
    if not np.any(alpha):raise ValueError('empty foreground mask')
    if np.all(alpha==255):raise ValueError('opaque input: provide a subject mask or transparent RGBA image')
    return image


def sample_surface(vertices,faces,count,seed):
    v=np.asarray(vertices,dtype=float);f=np.asarray(faces,dtype=np.int64)
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3 or not len(f) or f.min()<0 or f.max()>=len(v):raise ValueError('invalid mesh')
    tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if not np.isfinite(area).all() or area.sum()<=0:raise ValueError('mesh has no positive surface area')
    rng=np.random.default_rng(seed);tri=tri[rng.choice(len(f),size=count,p=area/area.sum())]
    u=np.sqrt(rng.random((count,1)));w=rng.random((count,1))
    return (1-u)*tri[:,0]+u*(1-w)*tri[:,1]+u*w*tri[:,2]


def write_ply(path,points,colors):
    from plyfile import PlyData,PlyElement
    p=np.asarray(points,dtype=np.float32);c=np.asarray(colors)
    if p.ndim!=2 or p.shape[1]!=3 or c.shape!=p.shape or not len(p) or not np.isfinite(p).all():raise ValueError('invalid point cloud')
    if c.dtype!=np.uint8:raise ValueError('RGB must be uint8')
    data=np.empty(len(p),dtype=[(k,'f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')])
    for i,k in enumerate('xyz'):data[k]=p[:,i]
    for i,k in enumerate(('red','green','blue')):data[k]=c[:,i]
    PlyData([PlyElement.describe(data,'vertex')],text=False,byte_order='<').write(str(path))
    saved=PlyData.read(str(path))['vertex'].data
    if not np.array_equal(np.column_stack([saved[k] for k in 'xyz']),p) or not np.array_equal(np.column_stack([saved[k] for k in ('red','green','blue')]),c):raise ValueError('PLY readback differs')
    return dict(file=Path(path).name,points=len(p),sha256=sha256(path))


def read_mesh_ply(path):
    from plyfile import PlyData
    data=PlyData.read(str(path));v=data['vertex'].data
    points=np.column_stack([v[k] for k in 'xyz'])
    colors=np.column_stack([v[k] for k in ('red','green','blue')])
    faces=np.asarray(list(data['face'].data['vertex_indices']),dtype=np.int32)
    return points,faces,colors


def write_mesh_ply(path,vertices,faces,colors):
    from plyfile import PlyData,PlyElement
    v=np.asarray(vertices,dtype=np.float32);f=np.asarray(faces);c=np.asarray(colors)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or not np.isfinite(v).all():raise ValueError('invalid mesh vertices')
    if f.ndim!=2 or f.shape[1]!=3 or not len(f) or not np.issubdtype(f.dtype,np.integer) or f.min()<0 or f.max()>=len(v):raise ValueError('invalid mesh faces')
    if c.shape!=v.shape or c.dtype!=np.uint8:raise ValueError('mesh RGB must be matching uint8')
    data=np.empty(len(v),dtype=[(k,'f4') for k in 'xyz']+[(k,'u1') for k in ('red','green','blue')])
    for i,k in enumerate('xyz'):data[k]=v[:,i]
    for i,k in enumerate(('red','green','blue')):data[k]=c[:,i]
    triangles=np.empty(len(f),dtype=[('vertex_indices','i4',(3,))]);triangles['vertex_indices']=f
    PlyData([PlyElement.describe(data,'vertex'),PlyElement.describe(triangles,'face')],text=False,byte_order='<').write(str(path))
    actual,indices,rgb=read_mesh_ply(path)
    if not np.array_equal(actual,v) or not np.array_equal(indices,f) or not np.array_equal(rgb,c):raise ValueError('mesh PLY readback differs')
    return dict(file=Path(path).name,vertices=len(v),faces=len(f),sha256=sha256(path))
