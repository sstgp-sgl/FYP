#!/usr/bin/env python3
"""Run the P6 web app inside a Slurm allocation; credentials stay env-only."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=17862);p.add_argument('--key-file',type=Path)
    a=p.parse_args()
    if a.key_file:os.environ['FYP_LLM_API_KEY']=a.key_file.read_text().strip()
    from src.storyboard.web import create_app
    import uvicorn
    uvicorn.run(create_app(json.loads(a.config.read_text())),host=a.host,port=a.port)
if __name__=='__main__':main()
