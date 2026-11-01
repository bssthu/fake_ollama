import asyncio
import struct
import sys
from pathlib import Path
import pytest
from fake_ollama.memory_worker import MemoryWorker, packet, validate_lengths
from fake_ollama.privacy import PRIVACY_MODE_ENV
from fake_ollama.speech_audio import Segmenter, FRAME_BYTES
from fake_ollama.speech_runtime import SpeechSession
from fake_ollama.speech_config import SpeechRequest
from tests.unit.test_speech import frame


def test_short_utterance_is_emitted_after_silence_without_manual_finish():
    segmenter = Segmenter()
    for _ in range(4): assert segmenter.push(frame()) is None
    pieces = [segmenter.push(frame(0)) for _ in range(8)]
    pieces = [p for p in pieces if p]
    assert len(pieces) == 1 and pieces[0].end < 2
    assert segmenter.finish() is None


def test_packetless_idle_flushes_once_and_next_utterance_has_new_time():
    segmenter = Segmenter(min_seconds=8)
    segmenter.push(frame())
    first = segmenter.idle()
    assert first and segmenter.idle() is None
    segmenter.push(frame())
    second = segmenter.idle()
    assert second.start == first.end


@pytest.mark.parametrize('sizes', [(0, 0), (49 * 1024**2, 0), (10, 5 * 1024**2), (10, -1)])
def test_worker_rejects_unbounded_frames(sizes):
    with pytest.raises(ValueError): validate_lengths(*sizes)


async def test_owned_memory_channel_roundtrip_reverse_read_and_private_write_guard(monkeypatch, tmp_path):
    monkeypatch.setenv(PRIVACY_MODE_ENV, '1')
    worker = MemoryWorker(sys.executable, 'tests.fixtures.memory_echo')
    target = tmp_path / 'must-not-exist.txt'
    try:
        result = await worker.call({}, b'x' * 65537)
        assert result == {'size': 65537, 'privacy': True}
        async def event(meta, data):
            assert meta == {'kind': 'read', 'length': 4} and data == b''
            return {}, b'test'
        assert await worker.call({'reverse': True}, on_event=event) == {'data': 'test'}
        assert await worker.call({'write': str(target)}) == {'denied': True}
        assert not target.exists()
    finally:
        await worker.stop()
    assert worker.process is None and worker.writer is None and worker.server is None


async def test_cleanup_attempts_every_worker_and_can_retry_after_failure():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    config = {'python': sys.executable, 'asr': {'a': {'kind': 'whisper'}}, 'translators': {'t': {}}}
    session = SpeechSession(config, SpeechRequest(asr='a', translator='t', target_language='zh'))
    workers = [SimpleNamespace(stop=AsyncMock()) for _ in range(3)]
    session.capture, session.asr, session.translator = workers
    workers[0].stop.side_effect = RuntimeError('cleanup failure')
    session.upload_buffer.extend(b'private content')
    with pytest.raises(RuntimeError, match='cleanup failure'):
        await session.cleanup()
    assert not session.cleaned and not session.upload_buffer
    for worker in workers: worker.stop.assert_awaited_once()
    workers[0].stop.side_effect = None
    await session.cleanup()
    assert session.cleaned


async def test_empty_recognition_is_visible_without_inventing_a_transcript():
    config = {'python': sys.executable, 'asr': {'a': {'kind': 'voxtral'}}, 'translators': {'t': {}}}
    session = SpeechSession(config, SpeechRequest(asr='a', translator='t'))
    await session.translate_row({'text': ''}, 0, 30)
    assert session.snapshot()['empty_segments'] == 1
    assert session.snapshot()['segments'] == 0 and not session.events
