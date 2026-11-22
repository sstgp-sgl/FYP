"""Replaceable synchronous providers; fixtures and replays are explicitly labelled."""
from dataclasses import dataclass,field
from typing import Protocol
import json,time,signal
from contextlib import contextmanager
from pathlib import Path

class ProviderError(Exception):
    def __init__(self,code,message):super().__init__(message);self.code=code
@dataclass
class ProviderResponse:
    content:str
    metadata:dict=field(default_factory=dict)
class Provider(Protocol):
    def metadata(self)->dict:...
    def generate(self,messages:list[dict],schema:dict,limits:dict)->ProviderResponse:...

@contextmanager
def deadline(seconds):
    """Hard wall-clock provider budget on the POSIX CLI main thread, restored after use."""
    def expired(*_):raise ProviderError('timeout','model call exceeded configured time budget')
    old=signal.getsignal(signal.SIGALRM);previous=signal.setitimer(signal.ITIMER_REAL,0)
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if previous[0]:signal.setitimer(signal.ITIMER_REAL,*previous)

class FixtureProvider:
    def __init__(self,responses):self.responses=list(responses);self.calls=0
    def metadata(self):return dict(provider='fixture',model='scripted-test',revision='fixture-v1',mode='synthetic/test')
    def generate(self,messages,schema,limits):
        if self.calls>=len(self.responses):raise ProviderError('provider_unavailable','fixture exhausted')
        r=self.responses[self.calls];self.calls+=1
        if isinstance(r,Exception):raise r
        return ProviderResponse(r,dict(finish_reason='fixture'))
class ReplayProvider(FixtureProvider):
    def __init__(self,attempt):
        p=Path(attempt);super().__init__([(p/'raw_response.txt').read_text()]);self.source=str(p.resolve())
    def metadata(self):return dict(provider='replay',model='saved-response',revision='replay-v1',mode='replay',source=self.source)
class TransformersProvider:
    def __init__(self,path):
        self.path=Path(path).resolve();self.model=None;self.tokenizer=None
        manifest=self.path/'download-manifest.json'
        if not manifest.is_file():raise ProviderError('provider_unavailable','pinned local model manifest missing')
        self.manifest=json.loads(manifest.read_text());self.load_seconds=None
    def metadata(self):
        return dict(provider='transformers',mode='live_model',model=self.manifest['repo'],revision=self.manifest['revision'],local_path=str(self.path),load_seconds=self.load_seconds,dtype='float16',attention_implementation='sdpa',low_cpu_mem_usage=True,generation_settings=dict(do_sample=False,temperature=1.0,top_p=1.0,top_k=50))
    def generate(self,messages,schema,limits):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
        if not torch.cuda.is_available():raise ProviderError('provider_unavailable','CUDA allocation required; CPU fallback disabled')
        if self.model is None:
            started=time.monotonic()
            self.tokenizer=AutoTokenizer.from_pretrained(str(self.path),local_files_only=True,trust_remote_code=False)
            self.model=AutoModelForCausalLM.from_pretrained(str(self.path),local_files_only=True,trust_remote_code=False,torch_dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda').eval()
            self.load_seconds=time.monotonic()-started
        set_seed(limits['seed'])
        encoded=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt').to(self.model.device)
        if encoded.shape[1]+limits['max_output_tokens']>16384:raise ProviderError('input_too_long','model context budget exceeded')
        with torch.inference_mode():
            result=self.model.generate(encoded,attention_mask=torch.ones_like(encoded),max_new_tokens=limits['max_output_tokens'],do_sample=False,temperature=1.0,top_p=1.0,top_k=50,pad_token_id=self.tokenizer.eos_token_id)
        tokens=result[0,encoded.shape[1]:];content=self.tokenizer.decode(tokens,skip_special_tokens=True)
        return ProviderResponse(content,dict(input_tokens=encoded.shape[1],output_tokens=len(tokens),finish_reason='length' if len(tokens)>=limits['max_output_tokens'] else 'eos',torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0)))

class DeepSeekProvider:
    """Official HTTPS Chat Completions adapter. Secrets are env-only and never serialized."""
    def __init__(self,model='deepseek-v4-pro',api_key_env='FYP_LLM_API_KEY',opener=None):
        self.model=model;self.api_key_env=api_key_env;self.base_url='https://api.deepseek.com'
        self.revision='unreported';self._opener=opener
    def metadata(self):
        return dict(provider='deepseek_chat_completions',mode='live_model',model=self.model,revision=self.revision,revision_source='API model_version/system_fingerprint when reported; cloud weights cannot be pinned',base_url=self.base_url,api_key_env=self.api_key_env,generation_settings=dict(thinking='disabled',temperature=0,response_format='json_object',stream=False,seed='not_supported'))
    def generate(self,messages,schema,limits):
        import os,socket,urllib.request,urllib.error
        key=os.environ.get(self.api_key_env,'')
        if not key:raise ProviderError('provider_unavailable','API credential environment variable is not configured')
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*args,**kwargs):return None
        payload=dict(model=self.model,messages=messages,max_tokens=limits['max_output_tokens'],temperature=0,thinking=dict(type='disabled'),response_format=dict(type='json_object'),stream=False)
        request=urllib.request.Request(self.base_url+'/chat/completions',data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
        opener=self._opener or urllib.request.build_opener(NoRedirect())
        started=time.monotonic()
        try:
            with opener.open(request,timeout=limits['call_timeout_seconds']) as response:
                body=response.read().decode('utf-8')
                request_id=response.headers.get('x-request-id')
        except urllib.error.HTTPError as exc:
            # Error bodies/headers are never logged: servers may echo request credentials.
            status=exc.code;exc.close()
            raise ProviderError('http_'+str(status),'DeepSeek request failed with HTTP '+str(status)) from None
        except (TimeoutError,socket.timeout):raise ProviderError('timeout','DeepSeek request exceeded the configured time budget') from None
        except urllib.error.URLError:raise ProviderError('provider_unavailable','DeepSeek network request failed') from None
        if key in body:raise ProviderError('credential_exposure','API response unexpectedly contained credential; response discarded')
        try:
            data=json.loads(body);choice=data['choices'][0];content=choice['message']['content']
            if not isinstance(content,str) or not content.strip():raise ValueError('empty content')
        except (ValueError,KeyError,IndexError,TypeError):raise ProviderError('invalid_api_response','DeepSeek returned no usable completion') from None
        self.revision=data.get('model_version') or data.get('system_fingerprint') or 'unreported'
        return ProviderResponse(content,dict(seconds=time.monotonic()-started,request_id=request_id,response_id=data.get('id'),returned_model=data.get('model'),system_fingerprint=data.get('system_fingerprint'),usage=data.get('usage'),finish_reason=choice.get('finish_reason'),generation={k:v for k,v in payload.items() if k!='messages'},raw_api_response=data))
