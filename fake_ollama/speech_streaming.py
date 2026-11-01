"""Voxtral native incremental encoder/decoder, bounded to one utterance.

Audio is consumed once with encoder/decoder KV caches, not re-transcribed on
each update. An utterance ends on silence or a hard context limit; its thread,
audio ring and generation caches are released before another is started.
"""
from __future__ import annotations

import threading
import time


class VoxtralStream:
    def __init__(self, engine, delay_ms=480):
        self.engine = engine
        engine.processor.mistral_common_audio_config.transcription_delay_ms = delay_ms
        self.condition = threading.Condition()
        self.audio = bytearray()
        self.base = 0
        self.closed = False
        self.text = ''
        self.error = None
        self.peak_buffer_bytes = 0
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self._run, name='voxtral-stream', daemon=True)
        self.thread.start()

    def _take(self, start, count):
        import numpy as np
        with self.condition:
            deadline = time.monotonic() + 60
            while self.base + len(self.audio) // 2 < start + count:
                if self.closed:
                    return None
                if time.monotonic() >= deadline:
                    raise TimeoutError('流式音频输入超时')
                self.condition.wait(.1)
            offset = (start - self.base) * 2
            if offset < 0:
                raise RuntimeError('Streaming audio window was already consumed')
            result = np.frombuffer(bytes(self.audio[offset:offset + count * 2]), '<i2').astype('float32') / 32768
            # Leave the small STFT overlap needed by the next chunk.
            discard = max(0, offset + count * 2 - 1280)
            del self.audio[:discard]
            self.base += discard // 2
            self.condition.notify_all()
            return result

    def push(self, pcm):
        with self.condition:
            deadline = time.monotonic() + 15
            while len(self.audio) + len(pcm) > 4 * 32000:
                if self.error or not self.thread.is_alive():
                    raise RuntimeError(self.error or '流式模型提前结束')
                if time.monotonic() >= deadline:
                    raise TimeoutError('流式模型处理落后，音频缓冲已满')
                self.condition.wait(.05)
            if self.closed:
                raise RuntimeError('Stream is already closed')
            self.audio.extend(pcm)
            self.peak_buffer_bytes = max(self.peak_buffer_bytes, len(self.audio))
            self.condition.notify_all()
        return self.snapshot()

    def snapshot(self):
        if self.error:
            raise RuntimeError(self.error)
        return {'text': self.text.strip(), 'language': 'unknown', 'partial': self.thread.is_alive(),
                'stream_buffer_bytes': len(self.audio), 'stream_peak_buffer_bytes': self.peak_buffer_bytes}

    def finish(self):
        processor = self.engine.processor
        self.push(bytes(processor.num_right_pad_tokens * processor.raw_audio_length_per_tok * 2))
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(90)
        if self.thread.is_alive():
            raise TimeoutError('流式推理未能结束；将释放整个模型进程')
        self.audio.clear()
        result = self.snapshot()
        result['partial'] = False
        return result

    def _run(self):
        engine, owner = self.engine, self
        processor = engine.processor
        class TextSink:
            def __init__(self):
                self.prompt = True
                self.tokens = []
            def put(self, value):
                if self.prompt:
                    self.prompt = False
                    return
                self.tokens.extend(value.reshape(-1).tolist())
                if len(self.tokens) > 4096:
                    raise RuntimeError('流式文本超出单段上限')
                owner.text = processor.decode(self.tokens, skip_special_tokens=True)
            def end(self): pass
        try:
            first = self._take(0, processor.num_samples_first_audio_chunk)
            if first is None:
                return
            initial = processor(first, is_streaming=True, is_first_audio_chunk=True, return_tensors='pt')
            initial.to(engine.model.device, dtype=engine.model.dtype)
            def features():
                yield initial.input_features
                mel = processor.num_mel_frames_first_audio_chunk
                while True:
                    start = mel * processor.feature_extractor.hop_length - processor.feature_extractor.win_length // 2
                    chunk = self._take(start, processor.num_samples_per_audio_chunk)
                    if chunk is None:
                        return
                    item = processor(chunk, is_streaming=True, is_first_audio_chunk=False, return_tensors='pt')
                    item.to(engine.model.device, dtype=engine.model.dtype)
                    yield item.input_features
                    mel += processor.audio_length_per_tok
            with engine.torch.inference_mode():
                engine.model.generate(input_ids=initial.input_ids, input_features=features(),
                                      num_delay_tokens=initial.num_delay_tokens,
                                      streamer=TextSink(), do_sample=False)
        except Exception as exc:
            self.error = str(exc)
        finally:
            engine.model._stream_exhausted = False
            with self.condition:
                self.condition.notify_all()
