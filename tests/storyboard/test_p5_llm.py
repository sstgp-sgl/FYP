"""CPU-only provider protocol/failure tests: no model import or weight download."""
import copy,json,time
from pathlib import Path
import pytest
from test_p4_story import source,plan_for
from src.scene.storage import read_json,atomic_json,sha256
from src.storyboard.llm_service import understand,validate,parse_response
from src.storyboard.llm_provider import FixtureProvider,ReplayProvider,ProviderError,ProviderResponse
from src.storyboard.llm_prompt import messages_for


def catalog_for(scene):return [dict(instance_id=i.instance_id,asset_id=i.asset_id,display_name={'a':'红色方块','b':'蓝色方块'}[i.instance_id],name_confirmed=True,confirmation_source='synthetic/test') for i in scene.instances]
def raw(source):return json.dumps(dict(status='ready',plan=plan_for(source[1])),ensure_ascii=False)
def run(tmp_path,source,responses,**kw):
 a,s=source
 return understand('红色先远离蓝色，然后靠近，最后保持。',catalog_for(s),a.root,a.root/'scene.json',tmp_path/'model-run',FixtureProvider(responses),**kw)

def test_success_full_compile_and_replay_are_not_live_model(source,tmp_path):
 result=run(tmp_path,source,[raw(source)])
 assert result['status']=='succeeded' and result['mode']=='synthetic/test'
 root=tmp_path/'model-run';assert validate(root)['passed']
 assert len(list((root/'attempt_00/compiled/renders').glob('*.png')))==3
 replay=ReplayProvider(root/'attempt_00')
 out=understand('同一输入',catalog_for(source[1]),source[0].root,source[0].root/'scene.json',tmp_path/'replay',replay)
 assert out['status']=='succeeded' and out['mode']=='replay'
 assert read_json(root/'attempt_00/normalized_plan.json')==read_json(tmp_path/'replay/attempt_00/normalized_plan.json')

@pytest.mark.parametrize('invalid,code',[('not json','invalid_json'),('{"status":"ready"}','invalid_response'),('{"status":"bogus"}','schema_error'),('{"status":"ready","plan":null,"extra":2}','schema_error')])
def test_invalid_response_repairs_at_most_twice(source,tmp_path,invalid,code):
 result=run(tmp_path,source,[invalid,raw(source)])
 assert result['status']=='succeeded' and len(result['attempts'])==2
 root=tmp_path/'model-run';assert read_json(root/'attempt_00/errors.json')[0]['code']==code
 messages=read_json(root/'attempt_01/messages.json')
 assert json.loads(messages[-1]['content'])['failures'][0]['code']==code
 assert (root/'attempt_00/raw_response.txt').read_text()==invalid

@pytest.mark.parametrize('mutation,code',[
 (lambda p:p['asset_ids'].append('unknown_asset'),'unknown_asset'),
 (lambda p:p['scenes'][1]['events'][0].update(object_instance_id='missing'),'unknown_reference'),
 (lambda p:p['scenes'][1]['events'][0].update(action='pick_up'),'unsupported_action'),
])
def test_semantic_errors_are_returned_without_replacement(source,tmp_path,mutation,code):
 p=plan_for(source[1]);mutation(p)
 result=run(tmp_path,source,[json.dumps(dict(status='ready',plan=p))]*3)
 assert result['status']=='failed' and len(result['attempts'])==3
 root=tmp_path/'model-run'
 assert all(read_json(root/f'attempt_{n:02d}/errors.json')[0]['code']==code for n in range(3))
 assert not list(root.rglob('*.png'))
 assert result['error']['code']=='repair_budget_exhausted'

def test_compilation_failure_is_repair_feedback_not_template(source,tmp_path):
 p=plan_for(source[1]);p['scenes'][1]['events'][0]['min_change']=100
 result=run(tmp_path,source,[json.dumps(dict(status='ready',plan=p)),raw(source)])
 assert result['status']=='succeeded'
 assert read_json(tmp_path/'model-run/attempt_00/compiled/run.json')['status']=='partial_failed'
 errors=read_json(tmp_path/'model-run/attempt_00/errors.json')
 assert errors[0]['code']=='layout_not_found'

@pytest.mark.parametrize('error',[ProviderError('provider_unavailable','offline'),ProviderError('timeout','timeout')])
def test_provider_failure_is_terminal_and_not_model_success(source,tmp_path,error):
 result=run(tmp_path,source,[error,raw(source)])
 assert result['status']=='failed' and result['error']['code']==error.code
 assert len(result['attempts'])==1 and not list((tmp_path/'model-run').rglob('*.png'))

def test_wall_clock_deadline_is_enforced(source,tmp_path):
 class Slow(FixtureProvider):
  def generate(self,*args):time.sleep(.2);return ProviderResponse(raw(source))
 a,s=source
 result=understand('测试',catalog_for(s),a.root,a.root/'scene.json',tmp_path/'slow',Slow([]),limits=dict(call_timeout_seconds=.02))
 assert result['error']['code']=='timeout' and result['seconds']<.2

def test_clarification_is_legal_but_not_a_completed_story(source,tmp_path):
 result=run(tmp_path,source,[json.dumps(dict(status='needs_clarification',question='拿起动作目前不支持，要换成什么？',reason_code='unsupported_action'))])
 assert result['status']=='needs_clarification' and result['exit_code']==2
 assert not validate(tmp_path/'model-run')['passed']
 assert not list((tmp_path/'model-run').rglob('*.png'))

def test_unconfirmed_names_rejected_before_provider(source,tmp_path):
 a,s=source;catalog=catalog_for(s);catalog[0]['name_confirmed']=False;p=FixtureProvider([raw(source)])
 result=understand('故事',catalog,a.root,a.root/'scene.json',tmp_path/'bad',p)
 assert result['error']['code']=='unconfirmed_names' and p.calls==0

def test_real_confirmation_cannot_be_faked_by_fixture_label(source,tmp_path,monkeypatch):
 a,s=source;catalog=catalog_for(s)
 from src.storyboard.llm_service import check_catalog
 original=a.get
 def real(aid):
  r=copy.copy(original(aid));r.source_kind='real';return r
 monkeypatch.setattr(a,'get',real)
 with pytest.raises(Exception,match='user confirmation'):check_catalog(catalog,s,a)

def test_response_versions_and_duplicate_keys_are_rejected(source,tmp_path):
 p=plan_for(source[1]);p['schema_version']=2
 result=run(tmp_path,source,[json.dumps(dict(status='ready',plan=p))]*3)
 assert result['status']=='failed'
 with pytest.raises(ValueError):parse_response('{"status":"ready","status":"needs_clarification"}')
 with pytest.raises(ValueError):parse_response('{"status":"ready","plan":NaN}')

def test_raw_and_compiled_provenance_tampering_fails(source,tmp_path):
 result=run(tmp_path,source,[raw(source)]);assert result['status']=='succeeded'
 root=tmp_path/'model-run';p=root/'attempt_00/raw_response.txt';text=p.read_text();p.write_text(text.replace('第一幕','已篡改'))
 report=validate(root)
 assert not report['passed'] and any('artifact_mismatch' in e for e in report['errors'])

def test_prompt_has_registered_names_but_no_development_answers(source):
 a,s=source;m=messages_for('故事',catalog_for(s),{})
 user=json.loads(m[-1]['content']);assert user['confirmed_catalog'][0]['display_name']=='红色方块'
 assert '2.853660' not in json.dumps(m) and 'approach' in m[0]['content']

@pytest.mark.parametrize('calls',[0,4])
def test_cannot_increase_repair_budget(source,tmp_path,calls):
 with pytest.raises(ValueError,match='budgets'):run(tmp_path,source,[raw(source)],limits=dict(max_calls=calls))


def test_prose_fact_claims_get_scoped_ids_not_development_answers(source,tmp_path):
 p=plan_for(source[1]);p['scenes'][0]['narration']['fact_claims']=['两个主体距离较远']
 result=run(tmp_path,source,[json.dumps(dict(status='ready',plan=p)),raw(source)])
 assert result['status']=='succeeded' and len(result['attempts'])==2
 errors=read_json(tmp_path/'model-run/attempt_00/errors.json')
 assert errors[0]['code']=='unknown_fact_reference'
 item=errors[0]['evidence']['items'][0]
 assert item['path']==['plan','scenes',0,'narration','fact_claims']
 assert item['allowed_ids']==['distance1','left1']
 assert 'expected_plan' not in json.dumps(errors)


def test_model_cannot_turn_off_complete_target_check(source,tmp_path):
 p=plan_for(source[1]);p['scenes'][2]['shot']['complete_display']=False
 result=run(tmp_path,source,[json.dumps(dict(status='ready',plan=p)),raw(source)])
 assert result['status']=='succeeded'
 assert read_json(tmp_path/'model-run/attempt_00/errors.json')[0]['code']=='schema_error'


def test_prompt_uses_current_settings_and_omits_legacy_constraints(source):
 from dataclasses import asdict
 initial=asdict(source[1]);initial['constraints']=[{'old':'not a current user instruction'}]
 messages=messages_for('最新故事',catalog_for(source[1]),initial,{'near_max':.25,'far_min':2.})
 payload=json.loads(messages[-1]['content'])
 assert payload['runtime_settings']=={'near_max':.25,'far_min':2.}
 assert 'constraints' not in payload['initial_state']
 assert payload['user_story']=='最新故事'


def test_contract_versions_and_output_directories_are_isolated(source,tmp_path):
 result=run(tmp_path,source,[raw(source)]);root=tmp_path/'model-run'
 assert result['response_contract_version']==2
 original=sha256(root/'attempt_00/raw_response.txt')
 with pytest.raises(FileExistsError):run(tmp_path,source,['replacement'])
 assert sha256(root/'attempt_00/raw_response.txt')==original
 changed=read_json(root/'run.json');changed['response_contract_version']=99;atomic_json(root/'run.json',changed)
 report=validate(root)
 assert not report['passed'] and report['errors']==['unsupported_response_contract_version']
 assert sha256(root/'attempt_00/raw_response.txt')==original


def test_technical_field_question_is_not_a_legal_clarification(source,tmp_path):
 invalid=json.dumps(dict(status='needs_clarification',question='如何填写 fact_claims？',reason_code='fact_claims_must_reference_current_ids'))
 result=run(tmp_path,source,[invalid,raw(source)])
 assert result['status']=='succeeded' and len(result['attempts'])==2
 assert read_json(tmp_path/'model-run/attempt_00/errors.json')[0]['code']=='schema_error'


def test_missing_motion_target_gets_precise_path_and_original_story(source,tmp_path):
 p=plan_for(source[1],'retreat');p['scenes'][1]['events'][0].pop('object_instance_id')
 a,s=source;story='红色先在蓝色左侧附近，随后相对蓝色退远并达到远处，最后保持。'
 result=understand(story,catalog_for(s),a.root,a.root/'scene.json',tmp_path/'motion',FixtureProvider([json.dumps(dict(status='ready',plan=p)),json.dumps(dict(status='ready',plan=plan_for(s,'retreat')))]))
 assert result['status']=='succeeded'
 error=read_json(tmp_path/'motion/attempt_00/errors.json')[0]
 assert error['code']=='unknown_reference'
 item=error['evidence']['items'][0]
 assert item['path']==['plan','scenes',1,'events',0,'object_instance_id'] and item['allowed_ids']==['b']
 repair=json.loads(read_json(tmp_path/'motion/attempt_01/messages.json')[-1]['content'])
 assert repair['original_user_story']==story and 'expected_plan' not in repair


def test_deepseek_success_records_transport_and_never_the_key(monkeypatch):
 from src.storyboard.llm_provider import DeepSeekProvider
 class Response:
  headers={'x-request-id':'unit-request'}
  def __enter__(self):return self
  def __exit__(self,*_):pass
  def read(self):return json.dumps(dict(id='unit-response',model='deepseek-v4-pro',system_fingerprint='unit-revision',usage={'completion_tokens':10},choices=[dict(message=dict(content='{"status":"needs_clarification"}'),finish_reason='stop')])).encode()
 class Opener:
  def open(self,request,timeout):
   assert request.full_url=='https://api.deepseek.com/chat/completions'
   assert request.get_header('Authorization')=='Bearer unit-secret'
   payload=json.loads(request.data);assert payload['response_format']=={'type':'json_object'} and payload['thinking']=={'type':'disabled'}
   assert 'seed' not in payload and timeout==15
   return Response()
 monkeypatch.setenv('FYP_LLM_API_KEY','unit-secret');p=DeepSeekProvider(opener=Opener())
 r=p.generate([dict(role='user',content='json')],{},dict(max_output_tokens=100,call_timeout_seconds=15))
 assert r.metadata['request_id']=='unit-request' and p.metadata()['revision']=='unit-revision'
 assert 'unit-secret' not in json.dumps(dict(metadata=p.metadata(),response=r.metadata))

@pytest.mark.parametrize('status',[301,401,403,429,500])
def test_deepseek_error_bodies_are_not_logged(monkeypatch,status):
 import io,urllib.error
 from src.storyboard.llm_provider import DeepSeekProvider
 class Opener:
  def open(self,*_,**kw):raise urllib.error.HTTPError('https://api.deepseek.com',status,'unit-secret',{},io.BytesIO(b'unit-secret'))
 monkeypatch.setenv('FYP_LLM_API_KEY','unit-secret')
 with pytest.raises(ProviderError) as e:DeepSeekProvider(opener=Opener()).generate([],{},dict(max_output_tokens=100,call_timeout_seconds=15))
 assert e.value.code=='http_'+str(status) and 'unit-secret' not in str(e.value)


def test_deepseek_missing_credential_fails_without_network(monkeypatch):
 from src.storyboard.llm_provider import DeepSeekProvider
 monkeypatch.delenv('FYP_LLM_API_KEY',raising=False)
 with pytest.raises(ProviderError,match='environment variable'):DeepSeekProvider().generate([],{},dict(max_output_tokens=100,call_timeout_seconds=15))

@pytest.mark.parametrize('kind',['timeout','network','invalid','empty','credential_echo'])
def test_deepseek_protocol_failures_are_safe(monkeypatch,kind):
 import urllib.error
 from src.storyboard.llm_provider import DeepSeekProvider
 class Response:
  headers={}
  def __enter__(self):return self
  def __exit__(self,*_):pass
  def read(self):
   if kind=='invalid':return b'not json'
   if kind=='credential_echo':return b'unit-secret'
   return b'{"choices":[{"message":{"content":" "}}]}'
 class Opener:
  def open(self,*_,**kw):
   if kind=='timeout':raise TimeoutError('unit-secret')
   if kind=='network':raise urllib.error.URLError('unit-secret')
   return Response()
 monkeypatch.setenv('FYP_LLM_API_KEY','unit-secret')
 with pytest.raises(ProviderError) as e:DeepSeekProvider(opener=Opener()).generate([],{},dict(max_output_tokens=100,call_timeout_seconds=15))
 assert 'unit-secret' not in str(e.value)
