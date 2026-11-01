import asyncio
import sys
import time

import pytest

from fake_ollama.speech_config import SpeechRequest
from fake_ollama.speech_runtime import SpeechSession
from tests.unit.test_speech import frame


def make_session(interval=.5):
    config = {'python': sys.executable, 'asr': {'a': {'kind': 'voxtral'}}, 'translators': {'t': {}}}
    session = SpeechSession(config, SpeechRequest(asr='a', translator='t', source='browser',
                            partial_translation_interval_seconds=interval))
    source = {'text': 'first words', 'final': 'final sentence', 'frames': 0}
    async def asr(payload, pcm=None):
        if payload['op'] == 'stream_push': source['frames'] += 1
        return {'text': source['final'] if payload['op'] == 'stream_finish' else source['text'], 'seconds': .01}
    session.asr.call = asr
    return session, source


async def until(predicate):
    async def poll():
        while not predicate(): await asyncio.sleep(.01)
    await asyncio.wait_for(poll(), 3)


@pytest.mark.parametrize('value', [0, .1, 31, float('nan'), float('inf')])
def test_partial_translation_interval_is_bounded(value):
    with pytest.raises(ValueError):
        SpeechRequest(asr='a', translator='t', partial_translation_interval_seconds=value)


async def test_continuous_tokens_do_not_postpone_translation_and_slow_work_coalesces():
    session, source = make_session()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def translate(payload, pcm=None):
        calls.append(payload)
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return {'translation': 'translated ' + payload['text'], 'seconds': .01}
    session.translator.call = translate
    try:
        await session.native_frame(frame(), .1, None)
        started = time.monotonic()
        # Keep changing the source right through the deadline.
        while not entered.is_set():
            assert time.monotonic() - started < 3
            source['text'] += ' next'
            await session.native_frame(frame(), .2, None)
            await asyncio.sleep(.02)
        active = session.partial_task
        assert session.stream_active and session.count == 0
        for index in range(25):
            source['text'] = f'newer source {index}'
            await session.native_frame(frame(), .3, None)
        assert session.partial_task is active and len(calls) == 1
        assert session.stream_row['text'] == 'newer source 24'
        # ASR progressed without waiting for the blocked translator.
        assert source['frames'] >= 26
        await asyncio.sleep(.55)
        release.set()
        await until(lambda: session.stream_row.get('translation_source') == 'newer source 24')
        assert [call['text'] for call in calls[1:]] == ['newer source 24']
        previews = [e['segment'] for e in session.events if e['segment'].get('translation')]
        assert previews[0]['text'] == 'newer source 24'
        assert previews[0]['translation_source'] == calls[0]['text']
        assert previews[0]['translation_pending'] and previews[0]['translation_is_partial']
        assert not session.context
        await session.finish_native(1)
        final = session.events[-1]['segment']
        assert final['state'] == 'complete' and final['translation'] == 'translated final sentence'
        assert not final['translation_is_partial'] and not final['translation_pending']
        assert len(session.context) == 1 and session.context[0]['source'] == 'final sentence'
        assert session.partial_task is None
    finally:
        release.set()
        await session.cleanup()


async def test_finalization_bypasses_a_pending_long_interval():
    session, source = make_session(interval=30)
    calls = []
    async def translate(payload, pcm=None):
        calls.append(payload['text'])
        return {'translation': 'final translation', 'seconds': .01}
    session.translator.call = translate
    try:
        await session.native_frame(frame(), .1, None)
        await asyncio.sleep(0)
        assert not calls
        await asyncio.wait_for(session.finish_native(.2), 1)
        assert calls == ['final sentence']
        assert session.events[-1]['segment']['state'] == 'complete'
    finally:
        await session.cleanup()


async def test_final_revision_waits_for_inflight_preview_and_cannot_be_overwritten():
    session, source = make_session()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def translate(payload, pcm=None):
        calls.append(payload['text'])
        if len(calls) == 1:
            entered.set()
            await release.wait()
        return {'translation': 'translated ' + payload['text'], 'seconds': .01}
    session.translator.call = translate
    finishing = None
    try:
        await session.native_frame(frame(), .1, None)
        await asyncio.wait_for(entered.wait(), 3)
        finishing = asyncio.create_task(session.finish_native(.2))
        await asyncio.sleep(.02)
        assert not finishing.done()
        assert session.stream_row['text'] == 'final sentence'
        assert session.stream_row['state'] == 'translating'
        release.set()
        await asyncio.wait_for(finishing, 1)
        assert calls == ['first words', 'final sentence']
        assert session.events[-1]['segment']['translation'] == 'translated final sentence'
        assert session.events[-1]['segment']['state'] == 'complete'
    finally:
        release.set()
        if finishing: await finishing
        await session.cleanup()


async def test_matching_preview_is_reused_and_next_utterance_starts_a_new_timer():
    session, source = make_session()
    source['final'] = source['text']
    calls = []
    async def translate(payload, pcm=None):
        calls.append(payload)
        return {'translation': 'translation', 'seconds': .01}
    session.translator.call = translate
    try:
        for index in range(2):
            await session.native_frame(frame(), index + .1, None)
            await until(lambda: bool(session.stream_row.get('translation')))
            assert len(session.context) == index
            await session.finish_native(index + .2)
            assert len(calls) == index + 1
            assert session.events[-1]['segment']['id'] == index + 1
        assert [len(call['context']) for call in calls] == [0, 1]
        assert len(session.context) == 2
    finally:
        await session.cleanup()


async def test_cancel_joins_active_preview_before_releasing_workers():
    session, source = make_session()
    entered, cancelled = asyncio.Event(), asyncio.Event()
    async def translate(payload, pcm=None):
        entered.set()
        try: await asyncio.Event().wait()
        finally: cancelled.set()
    session.translator.call = translate
    await session.native_frame(frame(), .1, None)
    await asyncio.wait_for(entered.wait(), 3)
    task = session.partial_task
    await session.cancel()
    assert cancelled.is_set() and task.done()
    assert session.cleaned and session.partial_task is None and session.stream_row is None


async def test_preview_error_is_retried_on_final_text_without_polluting_context():
    session, source = make_session()
    source['final'] = source['text']
    calls = []
    async def translate(payload, pcm=None):
        calls.append(payload)
        if len(calls) == 1: raise RuntimeError('public fixture translation failure')
        return {'translation': 'recovered', 'seconds': .01}
    session.translator.call = translate
    try:
        await session.native_frame(frame(), .1, None)
        await until(lambda: session.stream_row.get('error'))
        assert not session.context
        await session.finish_native(.2)
        row = session.events[-1]['segment']
        assert len(calls) == 2 and row['state'] == 'complete' and row['translation'] == 'recovered'
        assert 'error' not in row
    finally:
        await session.cleanup()


async def test_empty_final_recognition_discards_provisional_caption():
    session, source = make_session()
    source['final'] = ''
    async def translate(payload, pcm=None):
        return {'translation': 'provisional', 'seconds': .01}
    session.translator.call = translate
    try:
        await session.native_frame(frame(), .1, None)
        await until(lambda: bool(session.stream_row.get('translation')))
        await session.finish_native(.2)
        assert session.events[-1]['segment']['state'] == 'discarded'
        assert session.empty_segments == 1 and session.count == 0 and not session.context
    finally:
        await session.cleanup()
