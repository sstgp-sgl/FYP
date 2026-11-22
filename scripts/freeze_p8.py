#!/usr/bin/env python3
"""Freeze sampled real assets, groups, protocol and independent P8 answers."""
import argparse,json,sys,subprocess,shutil
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,yaml
from src.scene.storage import AssetRepository,SceneRepository,atomic_json,read_json,sha256
from src.scene.models import Instance,Transform,SceneDocument,Y_UP
from src.pointcloud_io import write_ply
from src.storyboard.spatial_service import LayoutAssets
from src.storyboard.assets import import_asset,freeze_assets
from src.storyboard.story_service import camera_settings
from src.storyboard.spatial_contracts import SpatialRequest
from src.storyboard.benchmark_tasks import tasks

def main():
 p=argparse.ArgumentParser();p.add_argument('--sources',required=True,type=Path);p.add_argument('--protocol',default='evaluation/p8_protocol.json',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 protocol=read_json(a.protocol);sources=read_json(a.sources);atomic_json(a.out/'protocol.json',protocol);atomic_json(a.out/'sources.json',sources)
 for path,value in sources.get('code_files',{}).items():
  if sha256(Path(path))!=value:raise ValueError('deployment differs from frozen commit: '+path)
 git=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True);commit=git.stdout.strip() if git.returncode==0 else sources.get('code_commit');commit=sources.get('code_commit',commit)
 registry=AssetRepository(a.out/'library');items={};inventory=[]
 for spec in sources['assets']:
  if 'asset_id' not in spec:
   side=import_asset(a.out/'source-imports',spec['ply'],dict(display_name=spec['name'],display_name_confirmed=True,source_image=spec['image'],producer=spec['producer'],source_up=spec['source_up'],source_forward=spec['source_forward'],target_height=1.6,orientation_basis='backend_axis_proposal',quality_flags=spec.get('quality_flags',[])))
   spec={**spec, 'library':str(a.out/'source-imports'),'asset_id':side['asset_id']}
  src=LayoutAssets(spec['library']);rec=src.get(spec['asset_id']);points,colors=src.points(rec);count=min(len(points),protocol['shared']['point_count_per_asset'])
  if rec.source_kind!='real':raise ValueError('P8 real-asset inventory cannot contain synthetic sources')
  extremes=np.unique(np.r_[np.argmin(points,axis=0),np.argmax(points,axis=0)]);remaining=np.setdiff1d(np.arange(len(points)),extremes);rng=np.random.default_rng(protocol['seed'])
  chosen=np.r_[extremes,rng.choice(remaining,count-len(extremes),replace=False)] if count>=len(extremes) else extremes[:count]
  path=a.out/(spec['key']+'.ply');write_ply(points[chosen],colors[chosen],path)
  record=registry.import_asset(path,role='subject',representation='point_cloud',source_kind='real',producer='P8 frozen canonical display sample',coordinate_frame=Y_UP,source_to_canonical=asdict(Transform()),source_image_hash=rec.source_image_hash,provenance=dict(source_asset_id=rec.asset_id,source_sha256=rec.sha256,source_group=spec['source_group'],original_point_count=len(points),sampling_seed=protocol['seed'],extrema_preserved=True,orientation='proposed',source_library=str(src.root)))
  atomic_json(registry.root/f'assets/{record.asset_id}/semantics.json',dict(display_name=spec['name'],display_name_confirmed=True,orientation_status='proposed',source_group=spec['source_group'],quality_flags=spec.get('quality_flags',[])))
  items[spec['key']]=dict(asset_id=record.asset_id,name=spec['name'],source_group=spec['source_group']);inventory.append(dict(**spec,frozen_asset_id=record.asset_id,frozen_sha256=record.sha256,point_count=count,source_point_count=len(points)))
 atomic_json(a.out/'inventory.json',inventory)
 all_tasks=[]
 for group in sources['groups']:
  root=a.out/'groups'/group['id'];root.mkdir(parents=True);selected=[items[group['a']],items[group['b']]];freeze_assets(registry.root,[s['asset_id'] for s in selected],root);repo=LayoutAssets(root);instances=[];catalog=[]
  for n,(iid,item) in enumerate(zip(['a','b'],selected)):
   rec=repo.get(item['asset_id']);pnts,_=repo.points(rec);lo,hi=pnts.min(0),pnts.max(0);translation=[0.,-float(lo[1]),0.]
   if n==0:
    other=repo.get(selected[1]['asset_id']);other_points,_=repo.points(other);translation[0]=float(other_points[:,0].min()-hi[0]-.8)
   instances.append(Instance(iid,rec.asset_id,rec.sha256,Transform(translation_xyz=translation)));catalog.append(dict(instance_id=iid,asset_id=rec.asset_id,display_name=item['name'],name_confirmed=True,confirmation_source='user'))
  SceneRepository(repo).save(SceneDocument('initial',instances,[]),'initial_scene.json');atomic_json(root/'catalog.json',catalog)
  config=yaml.safe_load(Path('configs/storyboard.yaml').read_text());config['camera'].update(width=protocol['shared']['camera_width'],height=protocol['shared']['camera_height'],max_points=protocol['shared']['render_max_points']);config['camera']=camera_settings(config)
  config['spatial']=SpatialRequest.parse(dict(schema_version=1,constraints=[dict(constraint_id='freeze_ground',subject_instance_id='a',relation='on_ground')],settings=config['spatial']),['a','b']).settings;atomic_json(root/'config.json',config)
  generated=tasks(group['id'],[s['name'] for s in selected]);source_groups=sorted({s['source_group'] for s in selected})
  for task in generated:task['source_groups']=source_groups
  all_tasks.extend(generated);atomic_json(root/'tasks.json',generated);atomic_json(root/'group.json',dict(**group,source_groups=source_groups))
 atomic_json(a.out/'tasks.json',all_tasks)
 files={str(p.relative_to(a.out)):sha256(p) for p in a.out.rglob('*') if p.is_file()}
 atomic_json(a.out/'freeze.json',dict(schema_version=1,code_commit=commit,code_files=sources.get('code_files',{}),protocol=protocol['protocol'],files=files,scope=protocol['scope'],source_groups=sorted({v['source_group'] for v in items.values()})))
 print(json.dumps(dict(root=str(a.out),assets=len(inventory),source_groups=len({v['source_group'] for v in items.values()}),groups=len(sources['groups']),commit=commit)))
if __name__=='__main__':main()
