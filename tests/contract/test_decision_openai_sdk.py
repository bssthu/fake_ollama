"""Optional real OpenAI SDK contract check; all transports are local mocks.

Validated with openai 3.34.0/httpx2; neither is a runtime dependency.
"""
import httpx
import pytest

openai = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")
AsyncOpenAI = openai.AsyncOpenAI
from tests.contract.test_decision_api import harness
usage={'input_tokens':12,'output_tokens':3,'total_tokens':15,'input_tokens_details':{'cached_tokens':0,'cache_write_tokens':0},'output_tokens_details':{'reasoning_tokens':0}}
def responder(dialect):
 def f(r,p):
  answers={} if dialect=='systemone' else []
  qs=p['questions'].items() if dialect=='systemone' else enumerate(p['questions'])
  for key,q in qs:
   typ=q['type'];a={'type':typ}
   if dialect=='systemone':
    if typ=='noul':a['noul']=.8
    elif typ=='choice':
     keys=list(q['criteria']);a.update(choice=keys[0],confidence=.6,probabilities={keys[0]:.75,keys[1]:.25})
    else:a.update(score=.5,confidence=.4,probabilities={'0':.75,'1':0.,'2':.25},legend={str(i):v for i,v in enumerate(q['criteria'])})
    answers[key]=a
   else:
    a['name']=q.get('name')
    if typ=='predicate':a['probability']=.8
    elif typ=='choice':a.update(choice=q['choices'][0]['value'],confidence=.6,probabilities=[{'value':x['value'],'probability':v} for x,v in zip(q['choices'],[.75,.25])])
    else:a.update(score=.5,confidence=.4,probabilities=[{'value':i,'label':x['label'],'probability':v} for i,(x,v) in enumerate(zip(q['levels'],[.75,0.,.25]))])
    answers.append(a)
  return httpx.Response(200,json={'model':'wire-model','answers':answers,'usage':usage})
 return f
@pytest.mark.asyncio
@pytest.mark.parametrize("dialect", ["systemone", "openai_decisions"])
async def test_openai_sdk_typed_decisions_strict_response(dialect):
    test_client, seen, app = harness(dialect=dialect, responder=responder(dialect))
    native = app.state.decision_clients["decision"]
    try:
        async with AsyncOpenAI(api_key="offline-fixture-key", base_url="http://127.0.0.1:21435/v1",
            http_client=httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app)),
            _strict_response_validation=True) as sdk:
            result = await sdk.decisions.create(model="judge", input="test", questions=[
                {"type": "predicate", "name": "same", "instructions": "True?"},
                {"type": "choice", "name": "same", "instructions": "Pick", "choices": [{"value": True}, {"value": "true"}]},
                {"type": "score", "instructions": "Rate", "levels": [{"label": "low"}, {"label": "mid"}, {"label": "high"}]},
            ])
            assert result.model == "judge"
            assert result.answers[0].probability == .8
            assert result.answers[1].choice is True
            assert result.answers[1].probabilities[1].value == "true"
            assert result.answers[1].confidence == .6
            assert result.answers[2].score == .5
            assert [answer.name for answer in result.answers] == ["same", "same", None]
            assert result.usage.total_tokens == 15
            assert seen[0][1]["model"] == "wire-model"
    finally:
        test_client.close()
        await native.aclose()
        await native._client.aclose()
