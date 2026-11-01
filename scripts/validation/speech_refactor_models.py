"""GPU checks for memory IPC and Voxtral incremental input. No input copies on disk."""
import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fake_ollama.speech_runtime import ModelWorker
from fake_ollama.privacy import configure_privacy_mode


async def main(args):
    configure_privacy_mode(args.privacy)
    config = json.loads(Path('config/local/speech.json').read_text('utf-8'))
    pcm = subprocess.run([config['ffmpeg'], '-v', 'error', '-ss', str(args.offset), '-i', args.file,
                          '-t', str(args.seconds), '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    for name in args.models:
        spec = config['asr'].get(name) or config['translators'][name]
        worker = ModelWorker(None, spec, spec.get('python', config['python']))
        try:
            started = time.monotonic()
            await worker.call({'op': 'load'})
            print(json.dumps({'model': name, 'load_seconds': time.monotonic()-started}), flush=True)
            for repeat in range(args.repeat):
                if name in config['translators']:
                    result = await worker.call({'op': 'translate', 'text': 'Hello. This is a public test. Bonjour.', 'target': 'zh', 'context': []})
                    assert result.get('translation')
                elif name.startswith('voxtral') and args.streaming:
                    count, first, first_wall = 0, None, None
                    started = time.monotonic()
                    for index in range(0, len(pcm), 3200):
                        if args.realtime:
                            await asyncio.sleep(max(0, started + (index + 3200) / 32000 - time.monotonic()))
                        result = await worker.call({'op': 'stream_push', 'delay_ms': args.delay}, pcm[index:index + 3200])
                        if result.get('text'):
                            count += 1
                            if first is None:
                                first = index/32000
                                first_wall = time.monotonic()-started
                    result = await worker.call({'op': 'stream_finish'})
                    result.update(updates=count, first_audio_seconds=first, first_wall_seconds=first_wall, elapsed=time.monotonic()-started)
                else:
                    result = await worker.call({'op': 'transcribe'}, pcm)
                if args.privacy:
                    if 'text' in result: result['text_chars'] = len(result['text'])
                    result = {k: v for k, v in result.items() if k not in ('text', 'translation', 'diagnostics')}
                print(json.dumps({'model': name, 'repeat': repeat, **result}, ensure_ascii=True), flush=True)
                print(json.dumps({'stats': await worker.call({'op': 'stats'})}), flush=True)
        finally:
            await worker.stop()
            print(json.dumps({'released': name}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='R:/测试视频.mkv')
    parser.add_argument('--offset', type=float, default=30)
    parser.add_argument('--seconds', type=float, default=12)
    parser.add_argument('--models', nargs='+', default=['whisper-large-v3-turbo'])
    parser.add_argument('--streaming', action='store_true')
    parser.add_argument('--delay', type=int, default=480)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--privacy', action='store_true')
    parser.add_argument('--realtime', action='store_true')
    asyncio.run(main(parser.parse_args()))
