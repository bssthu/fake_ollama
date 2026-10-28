import pytest
from pydantic import ValidationError

from fake_ollama.ocr_pipeline import OCRRequest, run_pipeline


@pytest.mark.asyncio
async def test_auto_stops_after_success_and_retries_only_once():
    calls = []
    async def recognize(name, category, max_tokens):
        calls.append(name)
        return {'text': name, 'truncated': False}
    async def score(texts, category):
        return [{'score': .95 if t == 'fallback' else .2, 'verdict': 'scored'} for t in texts]
    req = OCRRequest(image='abc', mode='auto', engines=['primary', 'fallback'], verifier='jev')
    result = await run_pipeline(req, recognize, score)
    assert calls == ['primary', 'fallback']
    assert result['decision'] == 'accept' and result['best_index'] == 1
    assert result['retry_count'] == 1
    calls.clear()
    result = await run_pipeline(req.model_copy(update={'engines': ['fallback', 'primary']}), recognize, score)
    assert calls == ['fallback'] and result['retry_count'] == 0


@pytest.mark.asyncio
async def test_failed_verification_is_not_low_quality_and_never_accepts():
    calls = []
    async def recognize(name, *args):
        calls.append(name)
        return {'text': 'valid candidate'}
    async def score(*args):
        return [{'score': 1.0, 'verdict': 'render_error', 'error': 'renderer unavailable'}]
    result = await run_pipeline(OCRRequest(image='x', mode='auto', engines=['a', 'b'], verifier='jev'), recognize, score)
    assert calls == ['a'] and result['best_index'] is None
    assert result['decision'] == 'review'
    assert result['results'][0]['score'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 1.1, True])
async def test_invalid_probability_is_never_accepted(value):
    async def score(*args):
        return [{'score': value, 'verdict': 'scored'}]
    result = await run_pipeline(OCRRequest(image='x', mode='score', candidates=['a'], verifier='jev'), None, score)
    assert result['best_index'] is None and result['decision'] == 'review'


@pytest.mark.asyncio
async def test_truncated_output_not_scored_and_failures_preserved():
    async def recognize(engine, *args):
        if engine == 'failed':
            raise RuntimeError('backend failed')
        return {'text': 'incomplete', 'truncated': True}
    async def score(*args):
        pytest.fail('truncated candidates must not be accepted')
    result = await run_pipeline(OCRRequest(image='x', mode='compare', engines=['truncated', 'failed'], verifier='jev'), recognize, score)
    assert result['decision'] == 'review' and len(result['results']) == 2
    assert result['results'][1]['error'] == 'backend failed'


def test_request_validation():
    for changes in [{'mode':'auto'}, {'engines':['a','a']}, {'mode':'score'}, {'threshold':float('nan')}, {'candidates':['']}, {'max_tokens':10000},
                    {'mode':'auto','engines':['a','b']}, {'mode':'score','candidates':['a']}, {'engines':[]}, {'verifier':''}]:
        with pytest.raises(ValidationError):
            OCRRequest(image='x', **changes)


@pytest.mark.asyncio
async def test_optional_verifier_is_explicit_and_no_verifier_never_scores():
    async def recognize(*args):
        return {'text': 'recognized'}
    async def score(*args):
        pytest.fail('No verifier selected: must not load a scorer')
    result = await run_pipeline(OCRRequest(image='x'), recognize, score)
    assert result['decision'] == 'unverified'
    assert result['verifier'] is None and result['best_index'] is None
    assert result['results'][0]['score'] is None
