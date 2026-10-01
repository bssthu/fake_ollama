"""Local-only Jev-compatible service, with fail-closed token budgets and graceful stop."""
import asyncio, json, os, secrets, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from laya import Agent
from laya.common import build_sequence, render_options

ROOT=Path(__file__).resolve().parent
PORT=21461
MODEL='laya-multilingual'
torch.set_num_threads(6)
agent=Agent(str(ROOT/'runtime/model'),device='cpu')
agent.model_id=MODEL
app=FastAPI(title='fake_ollama local Laya companion')
pool=ThreadPoolExecutor(max_workers=1)
gate=asyncio.Lock()
stop_token=(ROOT/'runtime/stop-token').read_text().strip()
server=None

def predict(body):
    state=body['state']; questions=body['questions']
    if not isinstance(questions,dict) or not 1<=len(questions)<=10: raise ValueError('question_count')
    for qid,qdef in questions.items():
        agent._check_question(qid,qdef)
        internal=agent._to_internal(qdef)
        if any(len(agent.tok(' '+option,add_special_tokens=False)['input_ids'])>48 for option in render_options(internal)):
            raise OverflowError('option_token_limit')
        # Build without truncation first, compare actual sequence against both trained limits.
        full,_=build_sequence(agent.tok,state,internal,1000000,1000000)
        head,_=build_sequence(agent.tok,'',internal,1000000,1000000)
        if len(full)>agent.cfg['max_len'] or len(head)>agent.cfg['head_max_len']:
            raise OverflowError('model_context_limit')
    result=agent.predict(state,questions)
    result['model']=MODEL
    return result

@app.get('/health')
def health(): return {'status':'ok','model':MODEL,'device':'cpu','max_tokens':agent.cfg['max_len'],'head_max_tokens':agent.cfg['head_max_len'],'calibrated_for_chat_reply':False}

@app.get('/v1/models')
def models(): return {'data':[{'id':MODEL,'object':'model'}]}

@app.post('/v1/systemone')
async def systemone(request:Request):
    data=bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data)>256*1024: raise HTTPException(413,'body_limit')
    try: body=json.loads(data)
    except (ValueError,UnicodeError): raise HTTPException(400,'invalid_json')
    if not isinstance(body,dict): raise HTTPException(422,'body_must_be_object')
    if body.get('model')!=MODEL: raise HTTPException(422,'unknown_model')
    if not {'state','questions'}<=body.keys(): raise HTTPException(422,'missing_fields')
    async with gate:
        try: return await asyncio.get_running_loop().run_in_executor(pool,predict,body)
        except OverflowError: raise HTTPException(413,'model_context_limit; shorten context or question; nothing was truncated')
        except (ValueError,KeyError,TypeError): raise HTTPException(422,'invalid_question')
        except Exception: raise HTTPException(500,'inference_failed')

@app.post('/admin/shutdown')
async def shutdown(request:Request):
    if not secrets.compare_digest(request.headers.get('Authorization',''),'Bearer '+stop_token): raise HTTPException(403)
    server.should_exit=True
    return {'stopping':True}

if __name__=='__main__':
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=PORT,access_log=False,log_level='warning'))
    try: server.run()
    finally: pool.shutdown(wait=True); agent.__exit__(None,None,None)
