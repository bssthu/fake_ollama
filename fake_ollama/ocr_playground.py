"""Authenticated playground routes for an isolated, request-owned OCR worker."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from .ocr_pipeline import OCRRequest
from .process_utils import create_managed_subprocess_exec, terminate_process_tree
from .ocr_resident import ResidentOCR

def read_config(settings):
    if not settings.playground_ocr_config:
        raise HTTPException(503, 'OCR 尚未配置：请设置 playground_ocr_config')
    path = Path(settings.playground_ocr_config).resolve()
    try:
        return path, json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise HTTPException(503, '无法读取 OCR 本机配置') from exc


def ocr_catalog(settings):
    _, config = read_config(settings)
    return {'engines': [{'id': name, 'label': spec.get('label', name),
                         'available': all(Path(p).is_file() for p in spec.get('required_files', []))}
                        for name, spec in config['engines'].items()],
            'verifiers': [{'id': name, 'label': spec.get('label', name), 'kind': spec.get('kind'),
                           'available': verifier_available(spec)}
                          for name, spec in config.get('verifiers', {}).items()],
            'scope': 'single_region'}


def verifier_available(spec):
    return (spec.get('kind') == 'minicpm_jev'
            and all(Path(spec.get(key, '')).is_file() for key in ('model', 'mmproj'))
            and bool(spec.get('renderer_source')) and Path(spec['renderer_source']).is_dir())


def playground_entries(catalog):
    def operation(mode, ready):
        return {'id': 'ocr_' + mode, 'ocr_mode': mode, 'endpoint': '/playground/api/ocr/run',
                'accepts_images': True, 'requires_images': True, 'multiple_images': False,
                'history_mode': 'single_turn', 'configured': ready,
                'limits': {'max_reference_images': 1}}

    entries = []
    verifier_ready = any(v['available'] for v in catalog['verifiers'])
    for engine in catalog['engines']:
        entries.append({'id': 'ocr:' + engine['id'], 'label': engine['label'],
                        'ocr_engine': engine['id'], 'capabilities': ['ocr'],
                        'operations': [operation('recognize', engine['available']),
                                       operation('auto', engine['available'] and verifier_ready)]})
    ready = any(e['available'] for e in catalog['engines'])
    entries.extend([
        {'id': 'ocr:compare', 'label': 'OCR 多模型对比', 'capabilities': ['ocr'],
         'operations': [operation('compare', ready)]},
        {'id': 'ocr:verify', 'label': 'OCR 候选复核', 'capabilities': ['ocr'],
         'operations': [operation('score', verifier_ready)]},
    ])
    return entries


def register_ocr_routes(app: FastAPI):
    gate = asyncio.Lock()
    resident = ResidentOCR()
    app.state.ocr_resident = resident

    @app.get('/playground/ocr', include_in_schema=False)
    async def page():
        return RedirectResponse('/playground/', headers={'Cache-Control': 'no-store'})

    @app.get('/playground/api/ocr/models', include_in_schema=False)
    async def models(request: Request):
        return {**ocr_catalog(request.app.state.settings), 'busy': gate.locked(), **resident.status()}

    @app.post('/playground/api/ocr/release', include_in_schema=False)
    async def release():
        if gate.locked():
            raise HTTPException(409, 'OCR 正忙，请先停止当前任务')
        async with gate:
            await resident.stop_if_owned()
            return {'released': True, **resident.status()}

    @app.post('/playground/api/ocr/run', include_in_schema=False)
    async def run(payload: OCRRequest, request: Request):
        config_path, config = read_config(request.app.state.settings)
        if payload.mode != 'score' and any(name not in config['engines'] for name in payload.engines):
            raise HTTPException(422, 'Unknown OCR engine')
        if payload.verifier is not None:
            spec = config.get('verifiers', {}).get(payload.verifier)
            if spec is None:
                raise HTTPException(422, 'Unknown OCR verifier')
            if not verifier_available(spec):
                raise HTTPException(503, 'Selected OCR verifier is unavailable')
        if gate.locked():
            raise HTTPException(409, 'OCR 正忙，请等待当前任务完成')
        async with gate:
            if resident.shutting_down:
                raise HTTPException(503, 'OCR is shutting down')
            if payload.keep_loaded:
                return await resident.run(payload, config, request)
            # Switching back to release mode also clears previously retained models.
            await resident.stop_if_owned()
            started = time.perf_counter()
            # Temporary inputs/results are deleted after completion, cancellation or error.
            with tempfile.TemporaryDirectory(prefix='fake-ocr-') as directory:
                work = Path(directory)
                source, output = work / 'request.json', work / 'result.json'
                source.write_text(payload.model_dump_json(), encoding='utf-8')
                process = await create_managed_subprocess_exec([
                    config['python'], '-m', 'fake_ollama.ocr_worker', '--config', str(config_path),
                    '--input', str(source), '--output', str(output),
                ], cwd=str(Path(__file__).resolve().parent.parent), stderr='logs/ocr-worker.err.log')
                try:
                    deadline = time.monotonic() + 900
                    while process.returncode is None:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('OCR worker timed out')
                        if await request.is_disconnected():
                            raise HTTPException(499, 'OCR request cancelled')
                        await asyncio.sleep(0.2)
                    if not output.is_file():
                        raise HTTPException(502, 'OCR worker failed; see logs/ocr-worker.err.log')
                    result = json.loads(output.read_text(encoding='utf-8'))
                    if 'fatal_error' in result:
                        raise HTTPException(422 if result.get('error_code') == 'invalid_image' else 502, result['fatal_error'])
                    result['request_seconds'] = round(time.perf_counter() - started, 4)
                    return result
                except TimeoutError as exc:
                    raise HTTPException(504, 'OCR timed out') from exc
                finally:
                    if not await terminate_process_tree(process):
                        raise HTTPException(500, 'OCR process cleanup failed')
