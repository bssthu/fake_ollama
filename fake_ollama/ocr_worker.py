"""Optional isolated OCR runtime. Invoked by the authenticated playground only.

Models can be retained by the gateway-owned worker. Inference is serialized;
external model servers are never stopped.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
import io
import json
import math
import os
import re
from pathlib import Path
import socket
import sys
import tempfile
import time

import httpx
from PIL import Image

from .ocr_pipeline import OCRRequest, run_pipeline
from .process_utils import create_managed_subprocess_exec, terminate_process_tree


class InvalidOCRImage(ValueError):
    pass


def decode_image(value):
    try:
        return _decode_image(value)
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidOCRImage(f'Invalid OCR image: {exc}') from exc


def _decode_image(value):
    if value.startswith('data:'):
        header, value = value.split(',', 1)
        if header not in ('data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64'):
            raise ValueError('Only PNG, JPEG and WebP images are accepted')
    raw = base64.b64decode(value, validate=True)
    with Image.open(io.BytesIO(raw)) as img:
        if img.width * img.height > 16_000_000 or max(img.size) > 8192:
            raise ValueError('Image exceeds 16 megapixels or 8192 pixels per side; crop a region first')
        return img.convert('RGB')


def data_url(raw):
    return 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')


class Runtime:
    def __init__(self, config, image, work, verifier=None, cache=None):
        self.config, self.image, self.work = config, image, Path(work)
        self.cache = cache
        self.verifier = config.get('verifiers', {}).get(verifier) if verifier else None
        if verifier and (self.verifier is None or self.verifier.get('kind') != 'minicpm_jev'):
            raise ValueError('Unknown or unsupported OCR verifier')
        self.original = self.work / 'original.png'
        image.save(self.original)
        self.image_url = data_url(self.original.read_bytes())
        self.events = []
        self.renderer_metadata = None

    @asynccontextmanager
    async def engine(self, spec):
        if self.cache is None:
            async with self.fresh_engine(self.config, spec, self.events) as client:
                yield client
        else:
            async with self.cache.engine(self, spec) as client:
                yield client

    @staticmethod
    @asynccontextmanager
    async def fresh_engine(config, spec, events):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        command = [config['llama_server'], '-m', spec['model'], '--mmproj', spec['mmproj'],
                   '--host', '127.0.0.1', '--port', str(port), '--ctx-size', str(spec.get('context', 8192)),
                   '-np', '1', '-ngl', str(spec.get('gpu_layers', 99)), '--no-webui',
                   '--alias', 'ocr-local']
        command.extend(spec.get('extra_args', []))
        started = time.perf_counter()
        process = await create_managed_subprocess_exec(command, stderr=config.get('engine_log', 'logs/ocr-engine.err.log'))
        try:
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}', timeout=300, trust_env=False) as client:
                for _ in range(480):
                    if process.returncode is not None:
                        raise RuntimeError(f'llama-server exited ({process.returncode}); see logs/ocr-engine.err.log')
                    try:
                        response = await client.get('/health', timeout=1)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.25)
                else:
                    raise TimeoutError('llama-server did not become ready')
                events.append({'model': spec.get('label', Path(spec['model']).name),
                                    'load_seconds': round(time.perf_counter()-started, 4), 'reused': False})
                yield client
        finally:
            if not await terminate_process_tree(process):
                raise RuntimeError('Could not release owned OCR engine')

    async def recognize(self, engine, category, max_tokens):
        spec = self.config['engines'][engine]
        if spec['kind'] == 'rapidocr':
            if category != 'text':
                raise ValueError('PP-OCR returns plain text; select MiniCPM/Hunyuan for formula or table structure')
            return await asyncio.to_thread(self.rapidocr, spec)
        prompts = {'text': 'Text Recognition:', 'formula': 'Formula Recognition:', 'table': 'Table Recognition:'}
        prompt = prompts[category] if engine == 'minicpm-ocr' else {
            'text': '请提取图片中的全部文字内容，保持原有阅读顺序。只输出识别结果，不要解释。',
            'formula': '识别图片中的公式，输出 LaTeX 格式。',
            'table': '识别图片中的表格，输出 HTML 格式。',
        }[category]
        if spec.get('prompt_style') == 'hunyuan_official':
            prompt = {'text': '提取图中的文字。', 'formula': '识别图片中的公式，用LaTeX格式表示。',
                      'table': '把图中的表格解析为HTML。'}[category]
        async with self.engine(spec) as client:
            started = time.perf_counter()
            response = await client.post('/v1/chat/completions', json={
                'model': 'ocr-local', 'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': self.image_url}}, {'type': 'text', 'text': prompt},
                ]}], 'max_tokens': max_tokens, 'temperature': 0, 'top_p': 1, 'stream': False,
                'repeat_penalty': 1.08 if engine.startswith('hunyuan-ocr') else 1.0,
                'top_k': -1, 'cache_prompt': False,
            })
            response.raise_for_status()
            result = response.json()
            choice = result['choices'][0]
            return dict(text=choice['message'].get('content') or '', usage=result.get('usage'),
                        inference_seconds=round(time.perf_counter()-started, 3),
                        load_seconds=self.events[-1]['load_seconds'], model_reused=self.events[-1]['reused'],
                        truncated=choice.get('finish_reason') != 'stop', finish_reason=choice.get('finish_reason'),
                        model_file=Path(spec['model']).name)

    @staticmethod
    def load_rapidocr(spec):
        from rapidocr import RapidOCR, OCRVersion, ModelType
        return RapidOCR(params={
            'Global.use_cls': False, 'Global.log_level': 'error',
            'Det.model_path': spec['det'], 'Rec.model_path': spec['rec'],
            'Rec.rec_keys_path': spec['keys'], 'Cls.model_path': spec['cls'],
            'Det.ocr_version': OCRVersion.PPOCRV6, 'Rec.ocr_version': OCRVersion.PPOCRV6,
            'Det.model_type': ModelType(spec.get('size', 'small')), 'Rec.model_type': ModelType(spec.get('size', 'small')),
            'EngineConfig.onnxruntime.intra_op_num_threads': 4,
            'EngineConfig.onnxruntime.inter_op_num_threads': 1,
        })

    def rapidocr(self, spec):
        import cv2
        import numpy as np
        key = json.dumps(spec, sort_keys=True)
        reused = self.cache is not None and key in self.cache.cpu
        load_started = time.perf_counter()
        if reused:
            engine = self.cache.cpu[key][0]
        else:
            engine = self.load_rapidocr(spec)
            if self.cache is not None:
                self.cache.cpu[key] = (engine, spec.get('label', 'PP-OCR'))
        load_seconds = 0 if reused else round(time.perf_counter() - load_started, 4)
        self.events.append({'model': spec.get('label', 'PP-OCR'), 'load_seconds': load_seconds, 'reused': reused})
        started = time.perf_counter()
        output = engine(cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR), use_cls=False)
        return dict(text='\n'.join(output.txts or []), inference_seconds=round(time.perf_counter()-started, 3),
                    load_seconds=load_seconds, model_reused=reused,
                    boxes=output.boxes.tolist() if output.boxes is not None else [],
                    recognition_scores=list(map(float, output.scores)) if output.scores is not None else [],
                    runtime='RapidOCR 3.10.0 / ONNX CPU', truncated=False)

    async def score(self, texts, category):
        if self.verifier is None:
            raise ValueError('No OCR verifier selected')
        # Import official pinned renderer instead of inventing different pixels/prompts.
        sys.path.insert(0, self.verifier['renderer_source'])
        os.environ['PLAYWRIGHT_BROWSERS_PATH'] = self.config['browsers']
        from rm_render import Candidate
        from serve.render_pool import RenderPool
        # Preserve ordinary OCR HTML tables, but never execute candidate-supplied
        # browser code. Reject rather than silently rewriting the model input.
        if any(re.search(r'<\s*(?:script|iframe|object|embed|meta|base|link|style)\b|\bon\w+\s*=|javascript\s*:', text, re.I) for text in texts):
            raise ValueError('Candidate contains active HTML; provide plain OCR text or a static HTML table')
        pool = RenderPool(browsers=1, pages_per_browser=1, precheck=True)
        try:
            await pool.start()
            # Candidate HTML must not fetch external resources or execute scripts.
            # KaTeX is injected explicitly by the official renderer, using local assets.
            for slot in pool._slots:
                for context in slot.contexts.values():
                    await context.route('**/*', lambda route: route.abort() if route.request.url.startswith(('http:', 'https:', 'file:')) else route.continue_())
            with tempfile.TemporaryDirectory(dir=self.work) as render_dir:
                rendered = await pool.render([Candidate(str(i), str(self.original), text, category) for i, text in enumerate(texts)], Path(render_dir))
                self.renderer_metadata = pool.describe()
                results = [{'score': None, 'verdict': item.reason or 'render_error', 'error': item.detail} for item in rendered]
                if any(item.ok for item in rendered):
                    async with self.engine(self.verifier) as client:
                        for i, item in enumerate(rendered):
                            if not item.ok:
                                continue
                            started = time.perf_counter()
                            response = await client.post('/v1/systemone', json={
                                'state': texts[i], 'files': [self.image_url, data_url(Path(item.render_image_path).read_bytes())],
                                'questions': {'consistency': {'type': 'choice', 'criteria': {'0': None, '1': None}}},
                            })
                            response.raise_for_status()
                            probs = response.json()['answers']['consistency']['probabilities']
                            if (set(probs) != {'0', '1'}
                                    or not all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in probs.values())
                                    or abs(sum(probs.values()) - 1) > 0.001):
                                raise ValueError('Jev did not return two normalized classification probabilities')
                            results[i] = dict(score=probs['1'], verdict='scored', score_seconds=round(time.perf_counter()-started, 3))
                return results
        finally:
            await pool.close()


class ModelCache:
    """Instances contain only loaded models; per-request images stay in temp dirs."""
    def __init__(self):
        self.gpu = {}
        self.cpu = {}

    def loaded(self):
        return [entry[2] for entry in self.gpu.values()] + [entry[1] for entry in self.cpu.values()]

    @asynccontextmanager
    async def engine(self, runtime, spec):
        key = json.dumps(spec, sort_keys=True)
        entry = self.gpu.get(key)
        if entry is not None:
            try:
                (await entry[1].get('/health', timeout=2)).raise_for_status()
            except Exception:
                self.gpu.pop(key)
                await entry[0].__aexit__(None, None, None)
                entry = None
        if entry is None:
            context = runtime.fresh_engine(runtime.config, spec, runtime.events)
            client = await context.__aenter__()
            entry = (context, client, spec.get('label', Path(spec['model']).name))
            self.gpu[key] = entry
        else:
            runtime.events.append({'model': entry[2], 'load_seconds': 0, 'reused': True})
        try:
            yield entry[1]
        except BaseException:
            self.gpu.pop(key, None)
            await entry[0].__aexit__(None, None, None)
            raise

    async def close(self):
        try:
            for key in list(self.gpu):
                await self.gpu.pop(key)[0].__aexit__(None, None, None)
        finally:
            self.cpu.clear()


async def run(config, request, cache=None):
    image = decode_image(request.image)
    with tempfile.TemporaryDirectory(prefix='fake-ocr-images-') as work:
        runtime = Runtime(config, image, work, request.verifier, cache)
        result = await run_pipeline(request, runtime.recognize, runtime.score)
        result.update(runtime_events=runtime.events, renderer=runtime.renderer_metadata,
                      verifier_label=runtime.verifier.get('label', request.verifier) if runtime.verifier else None,
                      renderer_environment='Windows SDK; absolute scores are not calibrated against Docker',
                      image_size=list(image.size))
        result.update(keep_loaded=cache is not None, loaded_models=cache.loaded() if cache else [])
        return result


async def serve(config, directory):
    work = Path(directory)
    source, output = work / 'request.json', work / 'result.json'
    cache = ModelCache()
    try:
        while True:
            if not source.is_file():
                await asyncio.sleep(0.01)
                continue
            try:
                req = OCRRequest.model_validate_json(source.read_text(encoding='utf-8'))
                source.unlink()
                result = await run(config, req, cache)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                source.unlink(missing_ok=True)
                result = {'fatal_error': str(exc), 'error_code': 'invalid_image' if isinstance(exc, InvalidOCRImage) else 'runtime_error'}
            partial = work / 'result.partial'
            partial.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
            partial.replace(output)
            req = result = None
    finally:
        await cache.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--serve')
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.serve:
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        asyncio.run(serve(config, args.serve))
        return
    if not args.input or not args.output:
        parser.error('--input and --output are required without --serve')
    try:
        config = json.loads(Path(args.config).read_text(encoding='utf-8'))
        req = OCRRequest.model_validate_json(Path(args.input).read_text(encoding='utf-8'))
        result = asyncio.run(run(config, req))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        result = {'fatal_error': str(exc), 'error_code': 'invalid_image' if isinstance(exc, InvalidOCRImage) else 'runtime_error'}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
