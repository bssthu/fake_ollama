"""Memory-only PCM sources. Random access reads are supplied by the owner/browser."""
from __future__ import annotations

import io
import time
from .memory_worker import Channel


class RemoteFile(io.RawIOBase):
    def __init__(self, channel, size):
        self.channel, self.size, self.position = channel, size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError('Invalid media seek')
        self.position = position
        return position

    def read(self, size=-1):
        count = min(1024 * 1024, self.size - self.position, size if size >= 0 else 1024 * 1024)
        if count <= 0:
            return b''
        reply, data = self.channel.event({'kind': 'read', 'offset': self.position, 'length': count})
        if reply.get('stop'):
            raise InterruptedError('Media reading stopped')
        if len(data) != count:
            raise ValueError('Incomplete media range')
        self.position += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def file_audio(channel, config):
    import av
    source = RemoteFile(channel, config['file_size'])
    with av.open(source, mode='r', options={'protocol_whitelist': 'pipe', 'threads': '1'}) as container:
        if not container.streams.audio:
            raise ValueError('文件不含音轨')
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                pcm = converted.to_ndarray().tobytes()
                # Decode frames can be larger than a queue item; keep 100 ms packets.
                for start in range(0, len(pcm), 3200):
                    reply, _ = channel.event({'kind': 'pcm'}, pcm[start:start + 3200])
                    if reply.get('stop'):
                        return
        for converted in resampler.resample(None):
            channel.event({'kind': 'pcm'}, converted.to_ndarray().tobytes())


def loopback_audio(channel, config):
    import numpy as np
    import pyaudiowpatch as pa
    import soxr
    with pa.PyAudio() as audio:
        device = audio.get_default_wasapi_loopback() if config['device'] == 'default' else audio.get_device_info_by_index(int(config['device']))
        if not device.get('isLoopbackDevice'):
            raise ValueError('所选设备不是系统回放采集设备')
        rate, channels = int(device['defaultSampleRate']), int(device['maxInputChannels'])
        resampler = soxr.ResampleStream(rate, 16000, 1, dtype='float32')
        stream = audio.open(format=pa.paFloat32, channels=channels, rate=rate, input=True,
                            input_device_index=device['index'], frames_per_buffer=rate // 10)
        last_packet = time.monotonic()
        idle_sent = False
        try:
            while True:
                available = min(rate // 10, stream.get_read_available())
                if available:
                    raw = stream.read(available, exception_on_overflow=True)
                    mono = np.frombuffer(raw, np.float32).reshape(-1, channels).mean(axis=1)
                    converted = resampler.resample_chunk(mono)
                    pcm = (np.clip(converted, -1, 1) * 32767).astype('<i2').tobytes()
                    reply, _ = channel.event({'kind': 'pcm', 'captured_at': time.monotonic()}, pcm)
                    last_packet, idle_sent = time.monotonic(), False
                else:
                    if not stream.is_active():
                        raise RuntimeError('系统播放设备已断开或停止响应')
                    idle = time.monotonic() - last_packet
                    if idle >= .8 and not idle_sent:
                        tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                        if tail.size:
                            channel.event({'kind': 'pcm'}, (np.clip(tail, -1, 1) * 32767).astype('<i2').tobytes())
                        resampler = soxr.ResampleStream(rate, 16000, 1, dtype='float32')
                        reply, _ = channel.event({'kind': 'idle'})
                        idle_sent = True
                    else:
                        reply, _ = channel.event({'kind': 'heartbeat'})
                    time.sleep(.05)
                if reply.get('stop'):
                    tail = resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
                    if tail.size:
                        channel.event({'kind': 'pcm'}, (np.clip(tail, -1, 1) * 32767).astype('<i2').tobytes())
                    return
        finally:
            stream.stop_stream()
            stream.close()


def main():
    channel = Channel.connect()
    try:
        payload, _ = channel.receive()
        if payload['op'] == 'devices':
            from .speech_capture import devices
            result = {'devices': devices()}
        else:
            config = payload['config']
            (file_audio if config['source'] == 'file' else loopback_audio)(channel, config)
            result = {'finished': True}
        channel.send({'result': result})
    except Exception as exc:
        channel.send({'error': str(exc)[:2000]})
    finally:
        channel.close()


if __name__ == '__main__':
    main()
