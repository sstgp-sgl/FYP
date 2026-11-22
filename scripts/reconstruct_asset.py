#!/usr/bin/env python3
"""Choose a reconstruction backend; reuse any SAM2-selected subject mask."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.reconstruction_methods import METHODS,reconstruct_asset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS);p.add_argument('--list-methods',action='store_true')
    p.add_argument('--image',type=Path);p.add_argument('--mask',type=Path)
    p.add_argument('--out',type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--points',type=int);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.list_methods:print('\n'.join(METHODS));return 0
    if not a.method or not a.image or not a.out:p.error('--method, --image and --out are required')
    try:
        cfg={}
        if a.config:
            import yaml
            document=yaml.safe_load(a.config.read_text())
            if not isinstance(document,dict) or set(document)-{'common','methods'}:raise ValueError('config supports only common and methods sections')
            cfg={**document.get('common',{}),**document.get('methods',{}).get(a.method,{})}
        for key in ('points','seed'):
            if getattr(a,key) is not None:cfg[key]=getattr(a,key)
        run=reconstruct_asset(a.method,a.image,a.mask,a.out,cfg)
        print(json.dumps(run,ensure_ascii=False));return 0
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
