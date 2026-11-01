"""Count complete decoded PCM and random-access reads without saving media content."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fake_ollama.memory_worker import MemoryWorker
from fake_ollama.privacy import configure_privacy_mode


async def decode(path, config):
    worker = MemoryWorker(config['python'], 'fake_ollama.speech_source')
    counts = {'pcm_bytes': 0, 'reads': 0, 'backward_reads': 0, 'max_read': 0}
    previous = 0
    with path.open('rb') as source:
        async def event(meta, data):
            nonlocal previous
            if meta['kind'] == 'read':
                offset, length = meta['offset'], meta['length']
                counts['reads'] += 1
                counts['backward_reads'] += offset < previous
                counts['max_read'] = max(counts['max_read'], length)
                previous = offset
                source.seek(offset)
                return {}, source.read(length)
            counts['pcm_bytes'] += len(data)
            return {}
        try:
            await worker.call({'op': 'capture', 'config': {'source': 'file', 'file_size': path.stat().st_size}}, on_event=event, timeout=120)
        finally:
            await worker.stop()
    assert counts['max_read'] <= 1024**2 and counts['pcm_bytes'] > 0
    print(json.dumps({'file_kind': path.suffix, 'audio_seconds': counts['pcm_bytes'] / 32000, **counts, 'released': worker.process is None}), flush=True)
    return counts['pcm_bytes'] / 32000


async def main():
    config = json.loads(Path('config/local/speech.json').read_text('utf-8'))
    fixture = Path('.tmp/speech-live/public-seek.mp4')
    # A public generated tone; MP4 moov stays at the end to exercise seeking.
    subprocess.run([config['ffmpeg'], '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'sine=frequency=440:duration=2', '-c:a', 'aac', str(fixture)], check=True)
    configure_privacy_mode(True)
    assert abs(await decode(fixture, config) - 2) < .1
    await decode(Path('R:/测试视频.mkv'), config)


if __name__ == '__main__':
    asyncio.run(main())
