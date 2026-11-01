"""Full native session coverage diagnostics; no transcript or media output files."""
import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fake_ollama.privacy import configure_privacy_mode
from fake_ollama.speech_runtime import SpeechSession
from fake_ollama.speech_config import SpeechRequest


async def main():
    configure_privacy_mode(True)
    config = json.loads(Path('config/local/speech.json').read_text('utf-8'))
    path = Path('R:/测试视频.mkv')
    session = SpeechSession(config, SpeechRequest(source='file', asr='voxtral-mini-4b-realtime-2602',
                            translator='qwen3.5-2b', file_size=path.stat().st_size))
    finishes = []
    call = session.asr.call
    async def observed_call(payload, pcm=None):
        result = await call(payload, pcm)
        if payload['op'] == 'stream_finish':
            item = {'start': round(session.stream_start, 2), 'text_chars': len(result.get('text', ''))}
            finishes.append(item)
            print(json.dumps(item), flush=True)
        return result
    session.asr.call = observed_call
    with path.open('rb') as source:
        async def read_range(offset, length):
            source.seek(offset)
            return source.read(length)
        session.read_range = read_range
        try:
            await session.start()
            await session.task
            snapshot = session.snapshot()
            assert snapshot['status'] == 'finished', snapshot['error']
            assert snapshot['resources_released'] and snapshot['processed_seconds'] > 659
            rows = {event['segment']['id']: event['segment'] for event in snapshot['events']}
            assert all(row['state'] == 'complete' for row in rows.values())
            print(json.dumps({k: v for k, v in snapshot.items() if k not in ('id', 'events')}), flush=True)
            print(json.dumps({'native_contexts': len(finishes), 'empty_contexts': sum(x['text_chars'] == 0 for x in finishes)}), flush=True)
        finally:
            await session.cancel()


if __name__ == '__main__':
    asyncio.run(main())
