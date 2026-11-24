#!/usr/bin/env python3
"""Inspect a local source or import it into a stopped web service's state directory."""
import argparse,json,sys,shutil,time,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.storyboard.video import probe,command
from src.storyboard.web_jobs import JobStore

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inspect','import']);parser.add_argument('--video',required=True,type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--config',type=Path);parser.add_argument('--start',type=float,default=30);parser.add_argument('--end',type=float,default=210);parser.add_argument('--threshold',type=float,default=.28);parser.add_argument('--attribution',default='')
    args=parser.parse_args()
    if args.action=='inspect':
        if not args.out:parser.error('inspect needs --out')
        from PIL import Image,ImageDraw
        info=probe(args.video);args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'source-info.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
        sheet=Image.new('RGB',(1000,880),'#eef2f4');draw=ImageDraw.Draw(sheet)
        for n in range(16):
            t=(info['duration']-.1)*n/16;path=args.out/f'overview-{n:02}.jpg';command(['-ss',t,'-i',args.video,'-frames:v','1','-vf','scale=240:-2','-y',path],60)
            sheet.paste(Image.open(path),(n%4*250,n//4*220));draw.text((n%4*250+5,n//4*220+185),f'{int(t//60):02}:{t%60:05.2f}',fill='black')
        sheet.save(args.out/'overview.jpg');print(json.dumps(info,ensure_ascii=False));return
    if not args.config:parser.error('import needs --config')
    settings=json.loads(args.config.read_text());store=JobStore(settings['root'],settings)
    try:
        with tempfile.NamedTemporaryFile(dir=store.root,delete=False) as tmp:path=Path(tmp.name)
        try:
            shutil.copyfile(args.video,path);row=store.videos.ingest(path,args.video.name)
        finally:path.unlink(missing_ok=True)
        row['source_attribution']=args.attribution;store.videos.save(row)
        import uuid
        row=store.videos.schedule(row['id'],args.start,args.end,args.threshold,uuid.uuid4().hex)
        while store.job(row['job_id'])['status'] not in {'succeeded','failed','interrupted'}:time.sleep(.5)
        row=store.videos.get(row['id']);print(json.dumps(row,ensure_ascii=False,indent=2))
        if row['status']!='ready':raise SystemExit(1)
    finally:store.close()
if __name__=='__main__':main()
