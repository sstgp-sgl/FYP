#!/usr/bin/env python3
"""Two real P8 candidate assets using existing models; all paths configured."""
import argparse,json,sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from src.scene.storage import atomic_json,sha256
from src.storyboard.assets import import_asset,CanonicalAssetRepository
from src.storyboard.real_scene import save_render
from src.reconstruction_methods import reconstruct_asset

def main():
 p=argparse.ArgumentParser();p.add_argument('--settings',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 import torch
 if not torch.cuda.is_available():raise RuntimeError('Slurm GPU allocation required')
 settings=json.loads(a.settings.read_text());work=Path.home()/'work';rows=[]
 options={**settings['reconstruction']['common'],**settings['reconstruction']['methods']['hunyuan']}
 candidates=[('duck_hunyuan','橡皮鸭',work/'data/inbox/rubber_duck.jpg'),('glasses_hunyuan','太阳镜',work/'data/inbox/sunglasses.jpg')]
 for key,name,source in candidates:
  root=a.out/key;root.mkdir();record=dict(key=key,name=name,source=str(source),source_sha256=sha256(source),status='running',review='assistant_review_pending',name_confirmation='user_2026-11-23',orientation='proposed');rows.append(record);atomic_json(a.out/'candidates.json',rows)
  try:
   if key=='duck_hunyuan':
    prior=work/'runs/p2/20261119-reconstruct-01/reconstruction/subjects/rubber_duck';image=prior/'subject.png';mask=prior/'mask.png';record['selection']='Reused P2 selected subject and its exact SAM2 mask; no new independent image.'
   else:
    from src.segment import build_sam2_predictor,select_multimask_index
    from src.subjects import Subject,segment_subject
    original=Image.open(source).convert('RGB');w,h=original.size;roi=[int(w*.36),int(h*.28),int(w*.73),int(h*.47)];crop=original.crop(roi);image=root/'crop.png';crop.save(image)
    # Fixed input-space prompts selected before model inference, not tuned to results.
    points=[[(w*x-roi[0]),(h*y-roi[1])] for x,y in [(.44,.36),(.56,.36),(.66,.35),(.50,.455),(.60,.29)]]
    predictor=build_sam2_predictor(settings['sam']);masks,scores=segment_subject(predictor,np.array(crop),Subject('sunglasses',boxes=[[0,0,crop.width,crop.height]],points=points,labels=[1,1,1,0,0]))
    chosen=select_multimask_index(masks,scores,None);mask=root/'mask.png';Image.fromarray(masks[chosen].astype(np.uint8)*255).save(mask);record.update(roi_xyxy=roi,mask_coverage=float(masks[chosen].mean()),sam_scores=[float(s) for s in scores],sam_choice=chosen)
    for index,m in enumerate(masks):Image.fromarray(m.astype(np.uint8)*255).save(root/f'mask_candidate_{index}.png')
    del predictor;gc.collect();torch.cuda.empty_cache()
   run=reconstruct_asset('hunyuan',image,mask,root/'reconstruction',options)
   side=import_asset(a.out/'library',root/'reconstruction/pointcloud.ply',dict(display_name=name,display_name_confirmed=True,source_image=str(source),producer='Hunyuan3D-2mini shape-only P8 candidate',source_up=[0,1,0],source_forward=[0,0,1],target_height=1.6,orientation_basis='backend_axis_proposal_not_user_confirmed',reconstruction=run,quality_flags=['single_view_unseen_geometry','review_pending']))
   repo=CanonicalAssetRepository(a.out/'library');pts,colors=repo.points(repo.get(side['asset_id']));save_render(root,'preview',pts,colors,np.ones(len(pts),np.int32));record.update(status='succeeded',asset_id=side['asset_id'],library=str(a.out/'library'))
  except Exception as exc:record.update(status='failed',error=type(exc).__name__+': '+str(exc))
  finally:atomic_json(a.out/'candidates.json',rows);gc.collect();torch.cuda.empty_cache()
 print(json.dumps(rows,ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
