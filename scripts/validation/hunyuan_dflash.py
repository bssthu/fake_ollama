"""Compare corrected Hunyuan AR and official DFlash on the same local images."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from fake_ollama.ocr_worker import Runtime

async def main():
    config=json.loads(Path('config/local/ocr.json').read_text(encoding='utf-8'))
    root=Path('.tmp/ocr-research')
    scope={};exec((root/'tasks.py').read_text(encoding='utf-8'),scope)
    results=[]
    for name,path,task in [('short',root/'samples/mixed.png','structured_parse'),('document',root/'hunyuan-official-page.png','doc_parse')]:
        image=Image.open(path).convert('RGB')
        with tempfile.TemporaryDirectory() as work:
            runtime=Runtime(config,image,work)
            for mode in ['ar','dflash']:
                spec=copy.deepcopy(config['engines']['hunyuan-ocr-current'])
                if mode=='dflash':spec['extra_args']+=['--spec-draft-model','K:/Projects/LLM_Models/HunyuanOCR-1.5-b11960/dflash-bf16.gguf','--spec-type','draft-dflash','--spec-draft-n-max','15']
                try:
                    async with runtime.engine(spec) as client:
                        for iteration in range(2):
                            started=time.perf_counter()
                            response=await client.post('/v1/chat/completions',json={
                                'model':'ocr-local','messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':runtime.image_url}},{'type':'text','text':scope['TASK_PROMPTS'][task]}]}],
                                'max_tokens':4096,'temperature':0,'top_p':1,'top_k':-1,'repeat_penalty':1.08,'stream':False,'cache_prompt':False})
                            response.raise_for_status();data=response.json()
                            results.append({'sample':name,'mode':mode,'iteration':iteration,'seconds':round(time.perf_counter()-started,3),'response':data})
                            print(name,mode,iteration,results[-1]['seconds'],data['choices'][0]['finish_reason'],data['usage']['completion_tokens'],flush=True)
                except Exception as exc:
                    results.append({'sample':name,'mode':mode,'error':str(exc)});print(name,mode,str(exc),flush=True)
                (root/'hunyuan-dflash.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':asyncio.run(main())
