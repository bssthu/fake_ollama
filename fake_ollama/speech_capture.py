"""CPU-only file decoding / Windows WASAPI loopback into a bounded PCM spool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from .speech_io import atomic_json as write_json, retry_io


def devices():
    import pyaudiowpatch as pa
    with pa.PyAudio() as audio:
        default = audio.get_default_wasapi_loopback()
        return [{'id': str(d['index']), 'label': d['name'],
                 'default': d['index'] == default['index']}
                for d in audio.get_loopback_device_info_generator()]


def capture(work, config):
    import numpy as np
    from .speech_audio import Segmenter, FRAME_BYTES
    segmenter = Segmenter(config['segment_seconds'], threshold=config['silence_threshold'])
    pending = bytearray()
    sequence = 0
    audio_end_at = None
    audio_end_position = 0.
    clock_written_at = 0.

    def emit(segment):
        nonlocal sequence
        if segment is None:
            return
        while len(list(work.glob('*.pcm'))) >= 4:
            if config['source'] != 'file':
                raise RuntimeError('识别处理落后，音频队列已满；已停止采集，未静默丢弃片段。请换用更快模型。')
            # Graceful finish still drains the already captured segment. Immediate
            # cancellation is handled by terminating this owned process tree.
            time.sleep(.1)  # file decoding obeys backpressure instead of growing RAM/disk
        stem = work / f'{sequence:08d}'
        stem.with_suffix('.partial').write_bytes(segment.pcm)
        retry_io(lambda: stem.with_suffix('.partial').replace(stem.with_suffix('.pcm')))
        metadata = {'start': segment.start, 'end': segment.end, 'overlap': segment.overlap}
        if config['source'] == 'loopback' and audio_end_at is not None:
            # Position of the final captured frame on this host's monotonic clock.
            # Includes queue/inference time, but not the sound device's own latency.
            metadata['captured_end_at'] = audio_end_at - (audio_end_position - segment.end)
        write_json(stem.with_suffix('.json'), metadata)
        sequence += 1

    def feed(data):
        nonlocal audio_end_at, audio_end_position, clock_written_at
        if data:
            audio_end_at = time.monotonic()
            audio_end_position = (segmenter.clock + (len(pending) + len(data)) / 2) / 16000
        pending.extend(data)
        while len(pending) >= FRAME_BYTES:
            frame = bytes(pending[:FRAME_BYTES])
            del pending[:FRAME_BYTES]
            emit(segmenter.push(frame))
        now = time.monotonic()
        if now - clock_written_at >= .1:
            write_json(work / 'clock.json', {'seconds': segmenter.clock / 16000})
            clock_written_at = now

    if config['source'] == 'file':
        command = [config['ffmpeg'], '-nostdin', '-v', 'error', '-protocol_whitelist', 'file,pipe',
                   '-format_whitelist', 'aac,ac3,aiff,amr,asf,avi,flac,flv,matroska,webm,mov,mp3,mpeg,mpegts,ogg,wav',
                   '-i', str(work / 'input.media'), '-map', '0:a:0', '-vn', '-sn', '-dn',
                   '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1']
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        # The decoder is a child of the gateway-owned capture process / Windows job.
        with (work / 'decode-error.txt').open('wb') as error:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error, creationflags=flags)
            try:
                while not (work / 'finish').exists():
                    data = process.stdout.read(FRAME_BYTES)
                    if not data:
                        break
                    feed(data)
                if (work / 'finish').exists():
                    process.terminate()
                code = process.wait(timeout=10)
                if code and not (work / 'finish').exists():
                    raise RuntimeError('无法解码音频：文件损坏、不含音轨或格式不受 FFmpeg 支持。')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
    else:
        import pyaudiowpatch as pa
        import soxr
        with pa.PyAudio() as audio:
            device = audio.get_default_wasapi_loopback() if config['device'] == 'default' else audio.get_device_info_by_index(int(config['device']))
            if not device.get('isLoopbackDevice'):
                raise ValueError('所选设备不是系统回放采集设备')
            rate, channels = int(device['defaultSampleRate']), int(device['maxInputChannels'])
            resampler = soxr.ResampleStream(rate, 16000, 1, dtype='float32')
            stream = audio.open(format=pa.paFloat32, channels=channels, rate=rate, input=True,
                                input_device_index=device['index'], frames_per_buffer=rate//10)
            try:
                while not (work / 'finish').exists():
                    # WASAPI can stop delivering packets when nothing is playing.
                    # Read only available frames so graceful stop also works in silence.
                    available = min(rate//10, stream.get_read_available())
                    if not available:
                        if not stream.is_active():
                            raise RuntimeError('系统播放设备已断开或停止响应')
                        time.sleep(.02)
                        continue
                    raw = stream.read(available, exception_on_overflow=True)
                    mono = np.frombuffer(raw, np.float32).reshape(-1, channels).mean(axis=1)
                    resampled = resampler.resample_chunk(mono)
                    feed((np.clip(resampled, -1, 1) * 32767).astype('<i2').tobytes())
                tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                feed((np.clip(tail, -1, 1) * 32767).astype('<i2').tobytes())
            finally:
                stream.stop_stream()
                stream.close()
    if pending:
        emit(segmenter.push(bytes(pending)))
    emit(segmenter.finish())
    write_json(work / 'clock.json', {'seconds': segmenter.clock / 16000})
    write_json(work / 'done.json', {'segments': sequence})


def main():
    from .privacy import privacy_mode_enabled
    if privacy_mode_enabled():
        raise RuntimeError('隐私模式不允许旧的磁盘音频缓冲入口')
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--devices', action='store_true')
    args = parser.parse_args()
    try:
        if args.devices:
            write_json(args.work / 'devices.json', {'devices': devices()})
        else:
            capture(args.work, json.loads((args.work / 'capture.json').read_text('utf-8')))
    except Exception as exc:
        write_json(args.work / 'error.json', {'error': str(exc)})
        raise SystemExit(1)


if __name__ == '__main__':
    main()
