"""Synthetic CPU audio/decode checks and device enumeration; no model imports or weights."""
import asyncio
from array import array
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import wave

ROOT = Path(__file__).resolve().parents[2]


def silent_stop():
    """A silent WASAPI source must never enter an indefinitely blocking read."""
    from fake_ollama.speech_capture import capture
    with tempfile.TemporaryDirectory(prefix='speech-silent-', dir=ROOT / '.tmp') as temp:
        work = Path(temp)
        calls = []
        class Stream:
            def get_read_available(self): calls.append('available'); return 0
            def is_active(self): return True
            def read(self, *args, **kwargs): raise AssertionError('Must not read an empty device buffer')
            def stop_stream(self): calls.append('stop')
            def close(self): calls.append('close')
        class Audio:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def get_default_wasapi_loopback(self):
                return {'index':0, 'isLoopbackDevice':True, 'defaultSampleRate':48000, 'maxInputChannels':2}
            def open(self, **kwargs):
                timer = threading.Timer(.2, lambda: (work / 'finish').touch())
                timer.start()
                return Stream()
        sys.modules['pyaudiowpatch'] = types.SimpleNamespace(PyAudio=Audio, paFloat32=1)
        capture(work, {'source':'loopback','device':'default','segment_seconds':3,'silence_threshold':.003})
        assert 'available' in calls and calls[-2:] == ['stop','close']
        assert json.loads((work / 'done.json').read_text())['segments'] == 0
    print('SILENT_WASAPI_STOP_PASS; synthetic device, no recording or model')


async def main():
    from fake_ollama.process_utils import create_managed_subprocess_exec, terminate_process_tree
    python = str(ROOT / 'services/speech/.venv/Scripts/python.exe')
    config = json.loads((ROOT / 'config/local/speech.json').read_text('utf-8'))
    report = {}
    with tempfile.TemporaryDirectory(prefix='speech-cpu-', dir=ROOT / '.tmp') as temp:
        work = Path(temp)
        capture = {'source':'file','device':'default','segment_seconds':3,'silence_threshold':.003,'ffmpeg':config['ffmpeg']}
        (work / 'capture.json').write_text(json.dumps(capture))
        samples = array('h')
        # Stereo 48 kHz WAV requires real FFmpeg downmix/resampling, then trailing
        # silence and a short final voiced tail exercise pause/EOF flushing.
        for i in range(48000 * 7):
            t = i / 48000
            value = int(6000 * math.sin(2*math.pi*440*t)) if t < 4.2 or t > 6.5 else 0
            samples.extend([value, value])
        if sys.byteorder != 'little': samples.byteswap()
        with wave.open(str(work / 'input.media'), 'wb') as wav:
            wav.setparams((2,2,48000,0,'NONE','not compressed'))
            wav.writeframes(samples.tobytes())
        async def run(extra=()):
            process = await create_managed_subprocess_exec([python,'-m','fake_ollama.speech_capture','--work',str(work),*extra], cwd=str(ROOT),
                                                          env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))
            try:
                await asyncio.wait_for(process.wait(), 25)
                if (work / 'error.json').exists():
                    raise RuntimeError((work / 'error.json').read_text('utf-8'))
            finally:
                assert await terminate_process_tree(process)
        await run()
        done = json.loads((work / 'done.json').read_text())
        segments = [json.loads(p.read_text()) for p in sorted(work.glob('[0-9]*.json'))]
        assert done['segments'] == len(segments) >= 2
        assert segments[-1]['end'] == 7
        for meta in sorted(work.glob('[0-9]*.json')):
            data = json.loads(meta.read_text())
            assert meta.with_suffix('.pcm').stat().st_size == round((data['end']-data['start']) * 32000)
        report['decoded_segments'] = len(segments)
        report['source_seconds'] = segments[-1]['end']
        await run(('--devices',))
        report['loopback_devices'] = len(json.loads((work / 'devices.json').read_text('utf-8'))['devices'])
        # Invalid media must produce an explicit decoder error instead of success.
        (work / 'input.media').write_bytes(b'not audio')
        process = await create_managed_subprocess_exec([python,'-m','fake_ollama.speech_capture','--work',str(work)], cwd=str(ROOT))
        try:
            await asyncio.wait_for(process.wait(), 20)
            assert (work / 'error.json').exists()
            assert process.returncode == 1
        finally:
            assert await terminate_process_tree(process)
        report['invalid_media_rejected'] = True
        # A long file must pause at four queued segments and resume without
        # dropping any when the consumer catches up.
        bounded = work / 'bounded'
        bounded.mkdir()
        (bounded / 'capture.json').write_text(json.dumps(capture))
        with wave.open(str(bounded / 'input.media'), 'wb') as wav:
            wav.setparams((2,2,48000,0,'NONE','not compressed'))
            for _ in range(8): wav.writeframes(samples.tobytes())
        process = await create_managed_subprocess_exec([python,'-m','fake_ollama.speech_capture','--work',str(bounded)], cwd=str(ROOT))
        try:
            for _ in range(100):
                if len(list(bounded.glob('*.pcm'))) == 4: break
                await asyncio.sleep(.05)
            await asyncio.sleep(.2)
            assert len(list(bounded.glob('*.pcm'))) == 4 and process.returncode is None
            consumed = 0
            final_end = 0
            for _ in range(400):
                assert len(list(bounded.glob('*.pcm'))) <= 4
                for meta in sorted(bounded.glob('[0-9]*.json')):
                    data = json.loads(meta.read_text())
                    final_end = data['end']
                    meta.with_suffix('.pcm').unlink()
                    meta.unlink()
                    consumed += 1
                if (bounded / 'done.json').exists() and not list(bounded.glob('*.pcm')): break
                await asyncio.sleep(.025)
            assert consumed == json.loads((bounded / 'done.json').read_text())['segments']
            assert final_end == 56
            report['file_backpressure_segments'] = consumed
            report['max_queued_segments'] = 4
        finally:
            assert await terminate_process_tree(process)
        graceful = work / 'graceful'
        graceful.mkdir()
        (graceful / 'capture.json').write_text(json.dumps(capture))
        with wave.open(str(graceful / 'input.media'), 'wb') as wav:
            wav.setparams((2,2,48000,0,'NONE','not compressed'))
            for _ in range(8): wav.writeframes(samples.tobytes())
        process = await create_managed_subprocess_exec([python,'-m','fake_ollama.speech_capture','--work',str(graceful)], cwd=str(ROOT))
        try:
            for _ in range(100):
                if len(list(graceful.glob('*.pcm'))) == 4: break
                await asyncio.sleep(.05)
            await asyncio.sleep(.2)  # the fifth captured segment is waiting for queue space
            (graceful / 'finish').touch()
            consumed = 0
            for _ in range(400):
                for meta in sorted(graceful.glob('[0-9]*.json')):
                    meta.with_suffix('.pcm').unlink()
                    meta.unlink()
                    consumed += 1
                if (graceful / 'done.json').exists() and not list(graceful.glob('*.pcm')): break
                await asyncio.sleep(.025)
            assert consumed >= 5, 'Graceful stop dropped the segment waiting behind a full queue'
            assert consumed == json.loads((graceful / 'done.json').read_text())['segments']
            report['graceful_stop_drains_full_queue'] = consumed
        finally:
            assert await terminate_process_tree(process)
    report['temporary_audio_removed'] = not Path(temp).exists()
    assert report['temporary_audio_removed']
    process = await create_managed_subprocess_exec([python, str(Path(__file__).resolve()), '--silent-stop'], cwd=str(ROOT),
                                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONPATH=str(ROOT)))
    try:
        await asyncio.wait_for(process.wait(), 15)
        assert process.returncode == 0, 'Silent device stop regression'
    finally:
        assert await terminate_process_tree(process)
    report['silent_device_stop'] = True
    print('SPEECH_CPU_PASS ' + json.dumps(report))


if __name__ == '__main__':
    if '--silent-stop' in sys.argv:
        silent_stop()
    else:
        asyncio.run(main())
