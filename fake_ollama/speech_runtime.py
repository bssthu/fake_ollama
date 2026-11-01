"""Owned, bounded speech sessions with memory-only content transport."""
from __future__ import annotations
import asyncio
from array import array
from collections import deque
from contextlib import suppress
import math
import secrets
import time
from fastapi import HTTPException
from .lifecycle import finish_cleanup
from .memory_worker import MemoryWorker
from .privacy import privacy_status
from .speech_audio import Segmenter, FRAME_BYTES, remove_boundary_overlap
from .process_utils import create_managed_subprocess_exec  # extension compatibility

TERMINAL = {'finished', 'cancelled', 'error'}


class ModelWorker:
    def __init__(self, work, spec, python):
        self.work, self.spec = work, spec
        self.worker = MemoryWorker(python, 'fake_ollama.speech_worker')
    @property
    def process(self): return self.worker.process
    async def call(self, payload, pcm=None):
        if payload['op'] == 'load': payload = {**payload, 'spec': self.spec}
        return await self.worker.call(payload, pcm or b'', timeout=600 if payload['op'] == 'load' else 180)
    async def stop(self): await self.worker.stop()


class SpeechSession:
    def __init__(self, config, payload, owner=''):
        self.id, self.owner = secrets.token_urlsafe(24), owner
        self.config, self.payload = config, payload
        self.asr = ModelWorker(None, {**config['asr'][payload.asr], 'speech_detection': payload.speech_detection}, config['asr'][payload.asr].get('python', config['python']))
        self.translator = ModelWorker(None, config['translators'][payload.translator], config['translators'][payload.translator].get('python', config['python']))
        self.capture = MemoryWorker(config['python'], 'fake_ollama.speech_source')
        self.task = self.watchdog = self.source_task = None
        self.websocket = None
        self.send_lock = asyncio.Lock()
        self.read_requests = {}
        self.frames = asyncio.Queue(maxsize=40)
        self.status, self.error = 'created', ''
        self.last_seen = time.monotonic()
        self.uploaded = 0
        self.file_size = payload.file_size
        self.upload_buffer = bytearray()
        self.input_seconds = self.processed_seconds = 0.
        self.events = deque(maxlen=2000)
        self.sequence = self.count = 0
        self.empty_segments = 0
        self.context = deque(maxlen=4)
        self.cleaned = self.stopping = False
        self.cleanup_lock, self.cancel_lock, self.io_lock = asyncio.Lock(), asyncio.Lock(), asyncio.Lock()
        self.source_done = False
        self.max_queue_frames = 0
        self.native = config['asr'][payload.asr].get('kind') == 'voxtral' and payload.asr_mode != 'segments'
        self.stream_active = False
        self.stream_start = self.stream_silence = 0
        self.stream_preroll = bytearray()
        self.stream_text = ''
        self.stream_row = None
        self.partial_task = None
        self.partial_due = None
        self.partial_wakeup = asyncio.Event()
        self.stream_finalizing = False
        self.preview_attempted = ''
        self.previous = ''

    @property
    def capture_process(self): return self.capture.process

    def snapshot(self, after=0):
        return {'id': self.id, 'status': self.status, 'error': self.error,
                'uploaded_bytes': self.uploaded, 'input_seconds': round(self.input_seconds, 2),
                'processed_seconds': round(self.processed_seconds, 2),
                'backlog_seconds': round(max(0., self.input_seconds - self.processed_seconds), 2),
                'segments': self.count, 'cursor': self.sequence,
                'empty_segments': self.empty_segments,
                'events': [e for e in self.events if e['sequence'] > after],
                'history_truncated': bool(self.events and after < self.events[0]['sequence'] - 1),
                'resources_released': self.cleaned, 'privacy': privacy_status(),
                'asr_mode': 'native_streaming' if self.native else 'segments',
                'queued_frames': self.frames.qsize(), 'max_queue_frames': self.max_queue_frames}

    def publish(self, row):
        self.sequence += 1
        self.events.append({'sequence': self.sequence, 'segment': dict(row)})

    async def monitor(self):
        try:
            while not self.cleaned:
                await asyncio.sleep(2)
                if time.monotonic() - self.last_seen > self.config.get('client_timeout_seconds', 45):
                    await self.cancel('页面断开或停止响应，已停止采集并释放模型')
                    return
        except asyncio.CancelledError: pass

    async def send(self, value):
        if self.websocket is None: raise RuntimeError('浏览器文件连接已断开')
        async with self.send_lock: await self.websocket.send_json(value)

    async def read_range(self, offset, length):
        if offset < 0 or length > 1024**2 or offset + length > self.file_size:
            raise ValueError('Invalid media range')
        if self.upload_buffer: return bytes(self.upload_buffer[offset:offset + length])
        request_id = secrets.token_hex(8)
        future = asyncio.get_running_loop().create_future()
        self.read_requests[request_id] = (future, length)
        try:
            await self.send({'type': 'read', 'id': request_id, 'offset': offset, 'length': length})
            return await asyncio.wait_for(future, 30)
        finally: self.read_requests.pop(request_id, None)

    def range_reply(self, request_id, data):
        item = self.read_requests.get(request_id)
        if item is None: raise ValueError('Unknown media range')
        future, length = item
        if len(data) != length or future.done(): raise ValueError('Invalid media range response')
        future.set_result(data)

    async def feed(self, pcm=b'', kind='pcm', captured_at=None):
        if self.source_done or self.cleaned: return
        if kind == 'pcm':
            if len(pcm) % 2 or len(pcm) > 64000: raise ValueError('Invalid PCM16 frame')
            self.input_seconds += len(pcm) / 32000
        value = (kind, pcm, captured_at)
        if self.payload.source == 'file': await self.frames.put(value)
        else:
            try: await asyncio.wait_for(self.frames.put(value), 2)
            except asyncio.TimeoutError:
                raise RuntimeError('识别处理落后，内存音频队列已满；已停止，未丢弃音频或转存磁盘')
        self.max_queue_frames = max(self.max_queue_frames, self.frames.qsize())

    async def start(self):
        if self.status != 'created': raise HTTPException(409, '会话已经开始或结束')
        if self.payload.source == 'file' and not (self.file_size or self.uploaded):
            raise HTTPException(422, '请先选择或上传音视频文件')
        self.file_size = self.file_size or self.uploaded
        self.status = 'loading_asr'
        self.task = asyncio.create_task(self.run())

    async def capture_audio(self):
        async def event(meta, data):
            kind = meta['kind']
            if kind == 'read':
                if self.stopping: return {'stop': True}, b''
                return {}, await self.read_range(meta['offset'], meta['length'])
            if kind in ('pcm', 'idle'): await self.feed(data, kind, meta.get('captured_at'))
            return {'stop': self.stopping}
        try:
            await self.capture.call({'op': 'capture', 'config': {**self.payload.model_dump(), 'file_size': self.file_size}},
                                    on_event=event, timeout=24 * 3600)
            await self.frames.put(('eof', b'', None))
        except asyncio.CancelledError: raise
        except Exception as exc:
            await self.frames.put(('eof', b'', None) if self.stopping else ('error', str(exc), None))

    async def translate_row(self, result, start, end, overlap=False, captured_at=None, row=None):
        text = remove_boundary_overlap(self.previous, result.get('text', ''), overlap)
        self.previous = result.get('text', '')
        if not text:
            self.empty_segments += 1
            if row is not None:
                self.publish({**row, 'state': 'discarded', 'text': '', 'translation': ''})
            return
        self.count += 1
        if row is None:
            row = {'translation': '', 'translation_source': ''}
        reuse = bool(row['translation'] and row.get('translation_source') == text)
        row.update(id=self.count, start=start, end=end, text=text, state='translating',
                   language=result.get('language', 'unknown'), asr_seconds=result.get('seconds', 0),
                   translation_is_partial=bool(row['translation']), translation_pending=not reuse,
                   translation_in_progress=not reuse)
        row.pop('error', None)
        started = time.monotonic()
        if captured_at: row['source_latency_seconds'] = round(max(0., started - captured_at), 3)
        self.publish(row)
        self.status = 'translating'
        try:
            if not reuse:
                translated = await self.translator.call({'op': 'translate', 'text': text,
                            'target': self.payload.target_language, 'context': list(self.context)})
                row.update(translated, translation_source=text, translation_seconds=translated['seconds'])
            row.update(state='complete', translation_is_partial=False, translation_pending=False)
            self.context.append({'source': text[-1200:], 'translation': row['translation'][-1200:]})
        except (TimeoutError, EOFError, OSError):
            # A transport failure leaves the reply stream unusable. End the
            # session and release workers instead of consuming a stale reply.
            raise
        except Exception as exc:
            row.update(state='translation_error', error=str(exc))
        row['translation_in_progress'] = False
        row['processing_seconds'] = round(time.monotonic() - started + result.get('seconds', 0), 3)
        row['segment_duration'] = round(end - start, 3)
        if captured_at: row['translation_latency_seconds'] = round(max(0., time.monotonic() - captured_at), 3)
        self.publish(row)

    def update_native_text(self, text, clock):
        self.stream_text = text
        if self.stream_row is None:
            self.stream_row = {'id': self.count + 1, 'start': self.stream_start,
                               'translation': '', 'translation_source': '', 'language': 'unknown',
                               'translation_is_partial': True, 'translation_in_progress': False}
        row = self.stream_row
        row.update(text=text, end=clock, state='recognizing',
                   translation_pending=row['translation_source'] != text)
        self.publish(row)
        # First dirty update starts a fixed wall-clock window. Later tokens
        # replace its payload, not its deadline, so continuous speech cannot
        # starve translation. There is only one active translation task.
        if self.partial_due is None and text != self.preview_attempted:
            self.partial_due = time.monotonic() + self.payload.partial_translation_interval_seconds
        if self.partial_task is None or self.partial_task.done():
            if self.partial_task is not None: self.partial_task.result()
            self.partial_task = asyncio.create_task(self.translate_partial(row))

    async def translate_partial(self, row):
        while row is self.stream_row and not self.stream_finalizing and self.partial_due is not None:
            delay = max(0., self.partial_due - time.monotonic())
            if delay:
                try: await asyncio.wait_for(self.partial_wakeup.wait(), delay)
                except asyncio.TimeoutError: pass
            if self.stream_finalizing: return
            text = row['text']
            self.partial_due = None
            self.preview_attempted = text
            if row['translation_source'] == text and row['translation']: continue
            row['translation_in_progress'] = True
            row.pop('error', None)
            self.publish(row)
            try:
                translated = await self.translator.call({'op': 'translate', 'text': text,
                            'target': self.payload.target_language, 'context': list(self.context)})
            except (TimeoutError, EOFError, OSError):
                raise
            except Exception as exc:
                row['error'] = str(exc)
            else:
                row.update(translated, translation_source=text, translation_seconds=translated['seconds'])
            # Recognition may have advanced while translation was in flight.
            # Keep its newer source text and identify the translated snapshot.
            row.update(translation_in_progress=False, translation_is_partial=True,
                       translation_pending=row['translation_source'] != row['text'])
            self.publish(row)

    async def native_frame(self, pcm, clock, captured_at):
        if self.partial_task is not None and self.partial_task.done(): self.partial_task.result()
        samples = array('h', pcm)
        rms = math.sqrt(sum(x*x for x in samples) / max(1, len(samples))) / 32768
        voiced = rms >= max(self.payload.silence_threshold, .00001)
        if not self.stream_active:
            self.stream_preroll.extend(pcm)
            if not voiced:
                self.stream_preroll = self.stream_preroll[-9600:]
                return
            pcm = bytes(self.stream_preroll)
            self.stream_preroll.clear()
            self.stream_start = clock - len(pcm) / 32000
            self.stream_active, self.stream_silence = True, 0
            self.stream_finalizing = False
            self.partial_wakeup.clear()
            self.preview_attempted = ''
        self.stream_silence = 0 if voiced else self.stream_silence + len(samples)
        result = await self.asr.call({'op': 'stream_push', 'delay_ms': self.payload.transcription_delay_ms}, pcm)
        if self.stream_row is not None: self.stream_row['end'] = clock
        if result.get('text') and result['text'] != self.stream_text:
            self.update_native_text(result['text'], clock)
        if self.stream_silence >= 16000 * .8 or clock - self.stream_start >= self.payload.stream_context_seconds:
            await self.finish_native(clock, captured_at)

    async def finish_native(self, clock, captured_at=None):
        if self.stream_active:
            self.stream_finalizing = True
            self.partial_wakeup.set()
            result = await self.asr.call({'op': 'stream_finish'})
            if self.stream_row is not None and result.get('text'):
                self.stream_row.update(text=result['text'], end=clock, state='translating',
                                       translation_pending=self.stream_row['translation_source'] != result['text'])
                self.publish(self.stream_row)
            # Wake a pending timer immediately. Join an in-flight inference
            # without cancelling its IPC; then translate the final revision.
            if self.partial_task is not None: await self.partial_task
            await self.translate_row(result, self.stream_start, clock, captured_at=captured_at, row=self.stream_row)
            self.stream_active = False
            self.stream_silence = 0
            self.stream_text = ''
            self.stream_row = self.partial_task = self.partial_due = None

    async def run(self):
        pending = bytearray()
        qwen = self.config['asr'][self.payload.asr].get('kind') == 'qwen_asr'
        segmenter = Segmenter(self.payload.segment_seconds, threshold=self.payload.silence_threshold,
                              min_seconds=8 if qwen else 0)
        clock, captured_at = 0., None
        try:
            await self.asr.call({'op': 'load'})
            self.status = 'loading_translation'
            await self.translator.call({'op': 'load'})
            self.status = 'listening'
            if self.payload.source != 'browser': self.source_task = asyncio.create_task(self.capture_audio())
            while True:
                kind, content, stamp = await self.frames.get()
                if stamp: captured_at = stamp
                if kind == 'error': raise RuntimeError(content)
                if kind == 'pcm': pending.extend(content)
                while len(pending) >= FRAME_BYTES or (kind in ('idle', 'eof') and pending):
                    frame = bytes(pending[:FRAME_BYTES])
                    del pending[:len(frame)]
                    clock += len(frame) / 32000
                    if self.native: await self.native_frame(frame, clock, captured_at)
                    else:
                        segment = segmenter.push(frame)
                        if segment:
                            self.status = 'transcribing'
                            result = await self.asr.call({'op': 'transcribe'}, segment.pcm)
                            await self.translate_row(result, segment.start, segment.end, segment.overlap, captured_at)
                            self.processed_seconds = segment.end
                if kind in ('idle', 'eof'):
                    if self.native: await self.finish_native(clock, captured_at)
                    elif segment := segmenter.idle():
                        result = await self.asr.call({'op': 'transcribe'}, segment.pcm)
                        await self.translate_row(result, segment.start, segment.end, segment.overlap, captured_at)
                    self.processed_seconds = clock
                if kind == 'eof':
                    self.status = 'finished'
                    break
                if self.native: self.processed_seconds = clock
                self.status = 'draining' if self.stopping else 'listening'
        except asyncio.CancelledError:
            self.status = 'cancelled'
            raise
        except Exception as exc: self.error, self.status = str(exc)[:2000], 'error'
        finally:
            try: await finish_cleanup(self.cleanup())
            except Exception as exc: self.error, self.status = str(exc), 'error'

    async def finish(self):
        if self.status in TERMINAL or self.stopping: return
        if self.status in ('created', 'loading_asr', 'loading_translation'):
            await self.cancel()
            return
        self.stopping = True
        if self.payload.source == 'browser':
            await self.frames.put(('eof', b'', None))
            self.source_done = True
        self.status = 'draining'

    async def cleanup(self):
        async with self.cleanup_lock:
            if self.cleaned: return
            self.source_done = True
            errors = []
            if self.partial_task is not None:
                self.partial_task.cancel()
                with suppress(asyncio.CancelledError, Exception): await self.partial_task
                self.partial_task = None
            self.stream_row = self.partial_due = None
            self.stream_text = ''
            if self.source_task and not self.source_task.done():
                self.source_task.cancel()
                try: await self.source_task
                except asyncio.CancelledError: pass
                except Exception as exc: errors.append(str(exc))
            for future, _ in self.read_requests.values():
                if not future.done(): future.cancel()
            for worker in (self.capture, self.asr, self.translator):
                try: await worker.stop()
                except Exception as exc: errors.append(str(exc))
            while not self.frames.empty(): self.frames.get_nowait()
            self.upload_buffer.clear()
            self.stream_preroll.clear()
            self.context.clear()
            if errors: raise RuntimeError('资源回收失败：' + '; '.join(errors))
            self.cleaned = True

    async def cancel(self, reason=''):
        async with self.cancel_lock:
            if reason: self.error = reason
            if self.task is not None and not self.task.done():
                self.task.cancel()
                with suppress(asyncio.CancelledError): await self.task
            if self.status not in TERMINAL: self.status = 'cancelled'
            await finish_cleanup(self.cleanup())


class SpeechManager:
    def __init__(self):
        self.session = None
        self.shutting_down = False
        self.lock = asyncio.Lock()
        self.probe = None
    def begin_shutdown(self): self.shutting_down = True
    def get(self, session_id, owner=None):
        if self.session is None or not secrets.compare_digest(self.session.id, session_id):
            raise HTTPException(404, '会话不存在或已过期')
        if owner is not None and not secrets.compare_digest(self.session.owner, owner):
            raise HTTPException(404, '会话不存在或已过期')
        self.session.last_seen = time.monotonic()
        return self.session
    async def create(self, config, payload, owner=''):
        async with self.lock:
            if self.shutting_down: raise HTTPException(503, '语音服务正在关闭')
            if self.session and not self.session.cleaned:
                raise HTTPException(409, '已有语音任务正在使用模型，请等待其结束')
            if self.session and self.session.watchdog:
                self.session.watchdog.cancel()
                with suppress(asyncio.CancelledError): await self.session.watchdog
            self.session = SpeechSession(config, payload, owner)
            self.session.watchdog = asyncio.create_task(self.session.monitor())
            return self.session
    async def stop_if_owned(self):
        async with self.lock:
            if self.session:
                if self.session.watchdog:
                    self.session.watchdog.cancel()
                    with suppress(asyncio.CancelledError): await self.session.watchdog
                await self.session.cancel()
            if self.probe: await self.probe.stop()
    async def devices(self, config):
        async with self.lock:
            self.probe = MemoryWorker(config['python'], 'fake_ollama.speech_source')
            try: return await self.probe.call({'op': 'devices'}, timeout=30)
            finally:
                await self.probe.stop()
                self.probe = None
