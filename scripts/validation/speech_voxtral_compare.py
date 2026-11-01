"""Compare offline/native output coverage on the same clip; metadata only."""
import asyncio
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fake_ollama.privacy import configure_privacy_mode
from fake_ollama.speech_runtime import ModelWorker


async def main():
    configure_privacy_mode(True)
    config = json.loads(Path('config/local/speech.json').read_text('utf-8'))
    worker = ModelWorker(None, config['asr']['voxtral-mini-4b-realtime-2602'], config['python'])
    try:
        await worker.call({'op': 'load'})
        for offset in (60.2, 180.6):
            pcm = subprocess.run([config['ffmpeg'], '-v', 'error', '-ss', str(offset), '-i', 'R:/测试视频.mkv',
                                  '-t', '30', '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
            batch = await worker.call({'op': 'transcribe'}, pcm)
            for i in range(0, len(pcm), 3200):
                await worker.call({'op': 'stream_push', 'delay_ms': 480}, pcm[i:i+3200])
            native = await worker.call({'op': 'stream_finish'})
            print(json.dumps({'offset': offset, 'offline_chars': len(batch['text']), 'native_chars': len(native['text']),
                              'equal': batch['text'] == native['text']}), flush=True)
    finally:
        await worker.stop()
        print(json.dumps({'released': worker.process is None}), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
