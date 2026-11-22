#!/usr/bin/env python3
"""Run the ten predeclared P7 tasks; optionally six real language calls."""
import argparse,json,sys,os,shutil,copy
from pathlib import Path
from itertools import product
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.scene.models import Instance,Transform,SceneDocument,Y_UP
from src.scene.storage import AssetRepository,SceneRepository,atomic_json,read_json,sha256
from src.pointcloud_io import write_ply
from src.storyboard.story_service import compile_story
from src.storyboard.edits import EditStore,frame_files,file_hashes
from src.storyboard.llm_provider import DeepSeekProvider

def fixture(root,dependency=False):
    repo=AssetRepository(root/'source');instances=[]
    for name,color,x in [('a',[225,60,45],-3.),('b',[40,100,220],0.)]:
        p=np.array(list(product([-.5,.5],[0.,1.],[-.5,.5])));path=root/(name+'.ply');write_ply(p,np.tile(color,(len(p),1)),path)
        r=repo.import_asset(path,role='subject',representation='point_cloud',source_kind='synthetic',producer='p7_predeclared_fixture',coordinate_frame=Y_UP)
        instances.append(Instance(name,r.asset_id,r.sha256,Transform(translation_xyz=[x,0.,0.])))
    scene=SceneDocument('initial',instances,[]);SceneRepository(repo).save(scene)
    rows=[]
    for n in range(1,4):
        rows.append(dict(scene_id='s'+str(n),base_scene_id='initial',constraints=[dict(constraint_id='distance'+str(n),subject_instance_id='a',object_instance_id='b',relation='near',parameters={'near_max':.5})],events=[],narration=dict(text='两个主体在场。',entity_refs=['a','b'],fact_claims=[],creative_text=''),shot=dict(template='focus' if n==2 else 'overview',target_ids=['a'] if n==2 else [],complete_display=True)))
    if dependency:
        rows[0]['constraints'][0].update(relation='far',parameters={'far_min':1.5})
        rows[1]['constraints'][0].update(relation='far',parameters={'far_min':3.0})
        rows[2]['events']=[dict(event_id='approach3',action='approach',subject_instance_id='a',object_instance_id='b',min_change=.2)]
    plan=dict(schema_version=1,story_id='p7_fixture',asset_ids=[i.asset_id for i in instances],premise='Synthetic independent fixture, not reconstruction evidence.',scenes=rows)
    atomic_json(root/'plan.json',plan);compiled=root/'compiled'
    result=compile_story(root/'plan.json',repo.root,repo.root/'scene.json',compiled)
    if not result['passed']:raise RuntimeError(result)
    return compiled

def case_store(root,compiled):
    story=root/'jobs/original';(story/'pipeline').mkdir(parents=True)
    # Source snapshots remain immutable; candidate editing uses copy-on-write.
    shutil.copytree(compiled,story/'pipeline/attempt_00/compiled',copy_function=os.link)
    atomic_json(story/'pipeline/run.json',dict(status='succeeded',accepted_attempt='attempt_00'))
    atomic_json(story/'catalog.json',[dict(instance_id='a',display_name='人物',name_confirmed=True),dict(instance_id='b',display_name='熊',name_confirmed=True)])
    return EditStore(story)

def evaluate_one(case,store,mode):
    state=store.state();base=store.root(state,1);before=file_hashes(base)
    patch=dict(schema_version=1,base_version=1,operations=case['operations'],locked_scene_ids=case.get('locks',['s1','s3']))
    if case.get('stale'):
        store.apply(store.jobs/'seed/compiled',1,patch=patch)
    out=store.jobs/'proposal/compiled'
    try:
        kwargs=dict(request=case['request'],locked_scene_ids=['s1','s3'],provider=DeepSeekProvider(),evidence=store.jobs/'proposal/evidence') if mode=='language' else dict(patch=patch)
        report=store.apply(out,1,**kwargs)
    except Exception as exc:report=dict(status='invalid_patch',request_satisfied=False,error=dict(message=str(exc)))
    unchanged=file_hashes(base)==before
    match=report['status']==case['expected_status'];unrelated=[];requirements=[]
    if report['status']=='succeeded':
        old=read_json(base/'story_plan.json');new=read_json(out/'story_plan.json')
        row=next(s for s in new['scenes'] if s['scene_id']=='s2')
        for op in case['operations']:
            if op['kind']=='narration':requirements.append(row['narration']['text']==op['text'])
            elif op['kind']=='shot':requirements.append(all(row['shot'][k]==op[k] for k in ['template','target_ids','complete_display']))
            else:
                c=next(c for c in row['constraints'] if c['constraint_id']==op['constraint_id'])
                requirements.append(all(c[k]==op[k] for k in ['relation','subject_instance_id','object_instance_id','parameters']))
        for a,b in zip(old['scenes'],new['scenes']):
            for k in a:
                if a[k]!=b[k] and (a['scene_id']!='s2' or k not in case.get('allowed_fields',[])):unrelated.append(a['scene_id']+'.'+k)
        changed_ids={op['constraint_id'] for op in case['operations'] if op['kind']=='relation'}
        original_row=next(s for s in old['scenes'] if s['scene_id']=='s2')
        if [c for c in row['constraints'] if c['constraint_id'] not in changed_ids]!=[c for c in original_row['constraints'] if c['constraint_id'] not in changed_ids]:unrelated.append('s2.unrequested_constraints')
        for sid in ['s1','s3']:
            if frame_files(base,sid)!=frame_files(out,sid):unrelated.append(sid+'.files')
        if 'constraints' not in case.get('allowed_fields',[]) and sha256(base/'scenes/s2.json')!=sha256(out/'scenes/s2.json'):unrelated.append('s2.pose')
        if case['allowed_fields']==['narration']:
            for sid in ['s1','s2','s3']:
                if sha256(base/f'renders/{sid}.png')!=sha256(out/f'renders/{sid}.png'):unrelated.append(sid+'.png')
    result=dict(case_id=case['id'],mode=mode,expected_status=case['expected_status'],actual_status=report['status'],
        expected_outcome_matched=match,requirements_met=all(requirements) if requirements else None,
        unexpected_changes=unrelated,source_version_unchanged=unchanged,
        passed=match and unchanged and not unrelated and all(requirements),report=report)
    atomic_json(store.story_root.parent/'case-result.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--cases',default='evaluation/edits_p7.json',type=Path);p.add_argument('--language',action='store_true');p.add_argument('--key-file',type=Path);a=p.parse_args()
    if a.key_file:os.environ['FYP_LLM_API_KEY']=a.key_file.read_text().strip()
    a.out.mkdir(parents=True,exist_ok=False);data=read_json(a.cases);atomic_json(a.out/'tasks.json',data)
    code_root=Path(__file__).resolve().parents[1]
    atomic_json(a.out/'source.json',{str(p.relative_to(code_root)):sha256(p) for p in [Path(__file__).resolve(),code_root/'src/storyboard/edits.py',a.cases.resolve()]})
    frozen=sha256(a.out/'tasks.json');bases={}
    for dep in [False,True]:
        root=a.out/('dependency' if dep else 'independent');root.mkdir();bases[dep]=fixture(root,dep)
    results=[]
    for mode in ['typed']+(['language'] if a.language else []):
        for case in data['cases']:
            if mode=='language' and case['expected_status']!='succeeded':continue
            root=a.out/(mode+'-'+case['id']);root.mkdir();store=case_store(root,bases[case.get('fixture')=='third_approach'])
            result=evaluate_one(case,store,mode);results.append(result)
            atomic_json(a.out/'results.json',dict(task_sha256=frozen,annotation=data['annotation'],cases=results))
            print(mode,case['id'],result['actual_status'],'PASS' if result['passed'] else 'FAIL',flush=True)
    summary={mode:dict(tasks=len(rows),passed=sum(r['passed'] for r in rows),unexpected_change_tasks=sum(bool(r['unexpected_changes']) for r in rows)) for mode in ['typed','language'] if (rows:=[r for r in results if r['mode']==mode])}
    atomic_json(a.out/'summary.json',summary);print(json.dumps(summary));return 0 if all(r['passed'] for r in results) else 1
if __name__=='__main__':raise SystemExit(main())
