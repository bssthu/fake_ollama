"""Compare actual short windows, longer context, and VAD on the user's material.

Run in the selected ASR environment. Frame and PCM decoding stays in memory.
Normal diagnostic mode prints samples; --summary-only never prints transcripts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fake_ollama.speech_worker import Engine
import numpy as np


def main(args):
    config = json.loads(Path('config/local/speech.json').read_text('utf-8'))
    spec = dict(config['asr'][args.model])
    if args.vad: spec['vad_filter'] = True
    engine = Engine(spec)
    pcm = subprocess.run([config['ffmpeg'], '-v', 'error', '-i', args.file, '-t', str(args.seconds),
                          '-vn', '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    for window in args.windows:
        for start in range(0, int(args.seconds), window):
            block = pcm[start * 32000:(start + window) * 32000]
            if not block: continue
            t = time.monotonic()
            result = engine.transcribe(block)
            record = {'start': start, 'window': window, 'model': args.model, 'vad': args.vad,
                      'language': result.get('language'), 'characters': len(result['text']), 'seconds': round(time.monotonic()-t, 3)}
            if not args.summary_only: record['text'] = result['text']
            if args.model.startswith('qwen') and (not result['text'] or result.get('language') != 'Japanese'):
                audio = np.frombuffer(block, '<i2').astype(np.float32) / 32768
                forced = engine.model.transcribe(audio=(audio, 16000), language='Japanese')[0]
                record['diagnostic_forced_characters'] = len(forced.text)
                if not args.summary_only: record['diagnostic_forced_text'] = forced.text
            print(json.dumps(record, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='R:/测试视频.mkv')
    parser.add_argument('--model', default='qwen3-asr-1.7b')
    parser.add_argument('--seconds', type=float, default=96)
    parser.add_argument('--windows', nargs='+', type=int, default=[8, 16])
    parser.add_argument('--vad', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    main(parser.parse_args())
