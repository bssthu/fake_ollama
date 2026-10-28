"""Controlled Hunyuan request/EOS comparison, keeping the original GGUF unchanged."""
import asyncio
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fake_ollama.ocr_worker import Runtime

async def main():
    config=json.loads(Path('config/local/ocr.json').read_text(encoding='utf-8'))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='GGUF under investigation; never inferred from an external app')
    parser.add_argument('--mmproj', required=True)
    args=parser.parse_args()
    spec={'model':args.model,'mmproj':args.mmproj,'context':8192}
    root=Path('.tmp/ocr-research')
    image=Image.open(root/'samples/mixed.png').convert('RGB')
    tasks={};exec((root/'tasks.py').read_text(encoding='utf-8'),tasks)
    custom='请提取图片中的全部文字内容，保持原有阅读顺序。只输出识别结果，不要解释。'
    rows=[]
    with tempfile.TemporaryDirectory() as work:
        runtime=Runtime(config,image,work)
        for variant,args in [('original',[]),('eos120007',['--override-kv','tokenizer.ggml.eos_token_id=int:120007']),('jinja',['--jinja']),('eos_jinja',['--override-kv','tokenizer.ggml.eos_token_id=int:120007','--jinja'])]:
            model={**spec,'extra_args':args}
            async with runtime.engine(model) as client:
                for task,prompt,empty_system in [('custom',custom,True),('official_text',tasks['TASK_PROMPTS']['structured_parse'],False),('official_doc',tasks['TASK_PROMPTS']['doc_parse'],False)]:
                    messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':runtime.image_url}},{'type':'text','text':prompt}]}]
                    if empty_system:messages.insert(0,{'role':'system','content':''})
                    t=time.perf_counter()
                    response=await client.post('/v1/chat/completions',json={'model':'ocr-local','messages':messages,'max_tokens':512,'temperature':0,'top_p':1,'top_k':-1,'repeat_penalty':1.08,'stream':False})
                    data=response.json()
                    row={'variant':variant,'task':task,'http_status':response.status_code,'seconds':round(time.perf_counter()-t,3),'response':data}
                    rows.append(row)
                    (root/'hunyuan-ab.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
                    print(variant,task,response.status_code,data.get('choices',[{}])[0].get('finish_reason'),data.get('usage',{}).get('completion_tokens'),flush=True)

if __name__=='__main__':asyncio.run(main())
