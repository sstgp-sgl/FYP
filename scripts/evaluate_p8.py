#!/usr/bin/env python3
"""One offline command to rebuild complete P8 tables and review pages on HPC."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True,type=Path);a=p.parse_args();code=Path(__file__).resolve().parents[1];root=a.run_root.resolve();frozen=root/'frozen';summary={}
 for phase in ['main','sensitivity','ablation']:
  batch=root/phase;config=json.loads((batch/'batch.json').read_text());count=len(list((batch/'runs').glob('*/result.json')))
  if count!=config['planned_runs']:raise RuntimeError(f'{phase} incomplete: {count}/{config["planned_runs"]}; no complete-study claim')
  subprocess.run([sys.executable,str(code/'scripts/benchmark_storyboard.py'),'--phase','report','--frozen',str(frozen),'--out',str(batch)],cwd=code,check=True,stdout=subprocess.DEVNULL)
  subprocess.run([sys.executable,str(code/'scripts/report_p8.py'),'--batch',str(batch),'--out',str(root/(phase+'-report'))],cwd=code,check=True)
  subprocess.run([sys.executable,str(code/'scripts/analyze_p8.py'),'--frozen',str(frozen),'--batch',str(batch),'--out',str(root/(phase+'-analysis'))],cwd=code,check=True,stdout=subprocess.DEVNULL)
  summary[phase]=json.loads((batch/'summary.json').read_text())
 (root/'all-tables.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print('All P8 tables rebuilt without model calls:',root/'all-tables.json')
if __name__=='__main__':main()
