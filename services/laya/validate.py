"""Real CPU inference and protocol limits, synthetic text only."""
import json, urllib.request, urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def post(data,path='/v1/systemone'):
    request=urllib.request.Request('http://127.0.0.1:21461'+path,data=json.dumps(data,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    try:
        with opener.open(request,timeout=30) as response:return response.status,json.load(response)
    except urllib.error.HTTPError as error:return error.code,json.load(error)
body={'model':'laya-multilingual','state':{'note':'会议定在周五。'},'questions':{
    'day':{'type':'choice','instructions':'On which day is the meeting?','criteria':{'fri':'Friday','mon':'Monday'}},
    'unsupported':{'type':'noul','instructions':'Does the state say the meeting is on Monday?','criteria':{'true':'Monday is stated','false':'Monday is not stated'}},
    'match':{'type':'score','instructions':'How well does Friday match the stated meeting day?','criteria':['Wrong','Partly wrong','Mostly right','Exactly right']},
}}
code,response=post(body);assert code==200,response
assert response['answers']['day']['choice']=='fri'
assert abs(sum(response['answers']['day']['probabilities'].values())-1)<.001
assert 0<=response['answers']['unsupported']['noul']<=1
assert 0<=response['answers']['match']['score']<=3
checks={
    'real_inference':code,
    'unknown_model':post(dict(body,model='not-installed'))[0],
    'oversize_context':post(dict(body,state='长文本'*10000))[0],
    'malformed_body':post([])[0],
    'unauthorized_shutdown':post({},'/admin/shutdown')[0],
}
assert checks==dict(real_inference=200,unknown_model=422,oversize_context=413,malformed_body=422,unauthorized_shutdown=403),checks
out=ROOT/'validation';out.mkdir(exist_ok=True)
(out/'synthetic-contract.json').write_text(json.dumps(dict(checks=checks,response=response),ensure_ascii=False,indent=2),encoding='utf-8')
print('LAYA_CONTRACT_PASS',checks)
