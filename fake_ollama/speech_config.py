"""Speech discovery and validation without importing inference dependencies."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from .privacy import privacy_status


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    asr: str = Field(max_length=80)
    translator: str = Field(max_length=80)
    target_language: str = Field(default='zh', pattern=r'^[a-z]{2,3}([_-][A-Za-z]{2,4})?$')
    source: Literal['loopback', 'file', 'browser'] = 'loopback'
    device: str = Field(default='default', max_length=100)
    segment_seconds: int = Field(default=8, ge=3, le=30)
    silence_threshold: float = Field(default=.003, ge=0, le=.1)
    file_size: int = Field(default=0, ge=0, le=2**41)
    asr_mode: Literal['auto', 'segments', 'native_streaming'] = 'auto'
    transcription_delay_ms: int = 480
    stream_context_seconds: int = Field(default=30, ge=5, le=120)
    speech_detection: bool = True

    @field_validator('transcription_delay_ms')
    @classmethod
    def valid_delay(cls, value):
        if value != 2400 and (value < 80 or value > 1200 or value % 80):
            raise ValueError('Delay must be a multiple of 80 ms up to 1200 ms, or 2400 ms')
        return value


def read_config(settings):
    value = settings.playground_speech_config
    if not value:
        raise HTTPException(503, '语音翻译尚未配置：playground_speech_config')
    try:
        config = json.loads(Path(value).read_text('utf-8'))
        if not all(k in config for k in ('python', 'asr', 'translators')):
            raise ValueError('missing speech configuration')
        return config
    except (OSError, ValueError) as exc:
        raise HTTPException(503, '无法读取语音翻译本机配置') from exc


def files_ready(spec):
    root = Path(spec.get('path', ''))
    required = spec.get('required_files', [])
    if not required or not root.is_dir() or not all((root / name).is_file() for name in required):
        return False
    index = root / 'model.safetensors.index.json'
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text('utf-8'))['weight_map'].values())
            if not all((root / shard).is_file() for shard in shards):
                return False
        except (OSError, ValueError, KeyError):
            return False
    return True


def catalog(config):
    def choices(group):
        return [{'id': name, 'label': spec.get('label', name), 'available': files_ready(spec),
                 'runtime_ready': Path(spec.get('python', config['python'])).is_file(),
                 'note': spec.get('note', ''), 'kind': spec.get('kind'),
                 'recommended_segment_seconds': 16 if spec.get('kind') == 'qwen_asr' else 8,
                 'modes': ['native_streaming', 'segments'] if spec.get('kind') == 'voxtral' else ['segments'],
                 'mode': 'native_streaming' if spec.get('kind') == 'voxtral' else 'bounded_segments',
                 'streaming_note': ('原生增量编码与解码；停顿或达到上下文上限后提交并释放缓存。复杂声线可尝试 2400 ms 延迟，低延迟更容易漏识别。' if spec.get('kind') == 'voxtral' else
                    '官方流式接口需要 vLLM；当前 Windows Transformers 环境使用分段识别。' if spec.get('kind') == 'qwen_asr' else
                    'Whisper 使用分段识别，不支持原生流式输入。')}
                for name, spec in config[group].items()]
    return {'asr': choices('asr'), 'translators': choices('translators'),
            'privacy': privacy_status(),
            'inference_enabled': config.get('inference_enabled', False),
            'loopback_supported': os.name == 'nt', 'source_language': 'auto',
            'max_upload_bytes': config.get('max_upload_bytes', 2 * 1024**3),
            'default_asr': config.get('default_asr', 'whisper-large-v3-turbo'),
            'default_translator': config.get('default_translator', 'qwen3.5-2b'),
            'validation': config.get('validation', '请根据本机实际音频确认速度与准确率')}
