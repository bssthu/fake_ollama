"""Bounded OCR selection policy. Scores are valid only after successful verification."""
from __future__ import annotations

import math
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OCRRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    image: str = Field(min_length=1, max_length=16 * 1024 * 1024)
    category: Literal['text', 'formula', 'table'] = 'text'
    mode: Literal['recognize', 'score', 'compare', 'auto'] = 'recognize'
    engines: list[str] = Field(default_factory=lambda: ['minicpm-ocr'], max_length=6)
    candidates: list[str] = Field(default_factory=list, max_length=8)
    verifier: str | None = Field(default=None, min_length=1, max_length=100)
    threshold: float = Field(default=0.9, ge=0, le=1, allow_inf_nan=False)
    max_tokens: int = Field(default=4096, ge=64, le=8192)
    keep_loaded: bool = False

    @model_validator(mode='after')
    def check(self):
        if len(set(self.engines)) != len(self.engines):
            raise ValueError('duplicate engines')
        if self.mode == 'score' and not self.candidates:
            raise ValueError('score requires candidates')
        if self.mode != 'score' and not self.engines:
            raise ValueError('recognition requires an engine')
        if self.mode in ('auto', 'score') and not self.verifier:
            raise ValueError('auto and score require an explicit verifier')
        if any(not text.strip() or len(text) > 24000 for text in self.candidates):
            raise ValueError('candidate must contain 1 to 24000 characters')
        if self.mode == 'auto' and len(self.engines) != 2:
            raise ValueError('auto requires exactly two different engines, primary then fallback')
        return self


async def run_pipeline(req, recognize, score):
    started = time.perf_counter()
    results = []
    needs_score = req.verifier is not None

    async def check_rows(rows):
        eligible = [r for r in rows if r.get('text', '').strip() and not r.get('error') and not r.get('truncated')]
        if not eligible:
            return
        try:
            scores = await score([r['text'] for r in eligible], req.category)
            if len(scores) != len(eligible):
                raise ValueError('Jev returned a different number of candidates')
            for row, value in zip(eligible, scores):
                p = value.get('score')
                if value.get('verdict') == 'scored' and type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1:
                    row.update(value)
                else:
                    row.update(score=None, verdict=value.get('verdict', 'verification_failed'),
                               verification_error=value.get('error', 'Jev did not produce a valid score'))
        except Exception as exc:
            for row in eligible:
                row.update(score=None, verdict='verification_failed', verification_error=str(exc))

    async def parse(engine):
        t = time.perf_counter()
        try:
            row = {'engine': engine, **await recognize(engine, req.category, req.max_tokens)}
            row.update(score=None, verdict='unverified')
        except Exception as exc:
            row = {'engine': engine, 'text': '', 'error': str(exc), 'score': None, 'verdict': 'recognition_failed'}
        row['elapsed_seconds'] = round(time.perf_counter() - t, 3)
        results.append(row)
        return row

    if req.mode == 'score':
        results = [dict(engine=f'candidate-{i+1}', text=text, score=None, verdict='unverified') for i, text in enumerate(req.candidates)]
        await check_rows(results)
    elif req.mode == 'auto':
        row = await parse(req.engines[0])
        await check_rows([row])
        # A renderer/backend failure is not evidence of low OCR quality.
        # Retry recognition failures/truncation or a valid score below threshold.
        if row.get('error') or row.get('truncated') or not row.get('text', '').strip() or (row['score'] is not None and row['score'] < req.threshold):
            row = await parse(req.engines[1])
            await check_rows([row])
    else:
        for engine in (req.engines if req.mode == 'compare' else req.engines[:1]):
            await parse(engine)
        if needs_score:
            await check_rows(results)

    valid = [(i, row) for i, row in enumerate(results) if row.get('verdict') == 'scored' and row.get('score') is not None]
    best = max(valid, key=lambda item: item[1]['score'])[0] if valid else None
    accepted = best is not None and results[best]['score'] >= req.threshold
    return dict(results=results, best_index=best, threshold=req.threshold, verifier=req.verifier,
                decision=('accept' if accepted else 'review') if needs_score else 'unverified',
                retry_count=max(0, len(results)-1) if req.mode == 'auto' else 0,
                elapsed_seconds=round(time.perf_counter()-started, 3),
                scope='single_region', score_semantics='Local Jev consistency score; threshold is not calibrated accuracy.')
