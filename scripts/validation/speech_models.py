"""Explicit GPU smoke matrix using the production owned-worker protocol.

Run from the gateway environment with PYTHONPATH pointing at the repository.
Input is raw mono PCM16LE / 16 kHz. Reports are local validation artifacts.
"""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import time

from fake_ollama.speech_runtime import ModelWorker

ROOT = Path(__file__).resolve().parents[2]
TEXTS = [
    ('short_japanese', '平和だなあ、暇だなあ。', 'zh'),
    ('mixed', '今天的 meeting 改到下午三点。Please bring the report, 予算は五万円です。', 'zh'),
    ('negation', 'The train does not depart at 15:30. It leaves at 16:45 from platform seven.', 'ja'),
]


async def main(args):
    config = json.loads((ROOT / 'config/local/speech.json').read_text('utf-8'))
    report = {'source': str(args.pcm), 'source_seconds': args.pcm.stat().st_size / 32000,
              'started': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'asr': {}, 'pairs': {}}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pcm = args.pcm.read_bytes()
    for aid, spec in config['asr'].items():
        if args.asr and aid not in args.asr: continue
        with tempfile.TemporaryDirectory(prefix='speech-model-smoke-') as temp:
            work = Path(temp)
            asr = ModelWorker(work / 'asr', spec, spec.get('python', config['python']))
            try:
                print('ASR_LOAD ' + aid, flush=True)
                loaded = await asr.call({'op':'load'})
                result = await asr.call({'op':'transcribe'}, pcm=pcm)
                report['asr'][aid] = {'load': loaded, 'result': result}
                print('ASR_RESULT ' + aid + ' ' + json.dumps(result, ensure_ascii=False), flush=True)
                report['asr'][aid]['corpus'] = {}
                for extra in args.extra_pcm:
                    value = await asr.call({'op':'transcribe'}, pcm=extra.read_bytes())
                    report['asr'][aid]['corpus'][extra.stem] = value
                    print('CORPUS ' + aid + ' ' + extra.stem + ' ' + json.dumps(value, ensure_ascii=False), flush=True)
                save()
                if args.asr_only: continue
                for tid, tspec in config['translators'].items():
                    if args.translator and tid not in args.translator: continue
                    pair = aid + '/' + tid
                    translator = ModelWorker(work / tid, tspec, tspec.get('python', config['python']))
                    report['pairs'][pair] = {}
                    try:
                        print('TRANSLATOR_LOAD ' + pair, flush=True)
                        report['pairs'][pair]['load'] = await translator.call({'op':'load'})
                        cases = [('audio', result['text'], 'zh'), *TEXTS]
                        for name, text, target in cases:
                            value = await translator.call({'op':'translate', 'text':text, 'target':target})
                            report['pairs'][pair][name] = value
                            print('TRANSLATION ' + pair + ' ' + name + ' ' + json.dumps(value, ensure_ascii=False), flush=True)
                        # ASR must still work while both models occupy the GPU.
                        report['pairs'][pair]['co_resident_asr'] = await asr.call({'op':'transcribe'}, pcm=pcm)
                    except Exception as exc:
                        report['pairs'][pair]['error'] = str(exc)
                        print('PAIR_ERROR ' + pair + ' ' + str(exc), flush=True)
                    finally:
                        await translator.stop()
                        save()
            except Exception as exc:
                report['asr'][aid] = {'error': str(exc)}
                print('ASR_ERROR ' + aid + ' ' + str(exc), flush=True)
            finally:
                await asr.stop()
                save()
    report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True, help='Authorize loading models on the GPU')
    parser.add_argument('--pcm', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / '.tmp/speech-live/models.json')
    parser.add_argument('--asr', nargs='+')
    parser.add_argument('--translator', nargs='+')
    parser.add_argument('--asr-only', action='store_true')
    parser.add_argument('--extra-pcm', type=Path, nargs='*', default=[])
    asyncio.run(main(parser.parse_args()))
