"""FastAPI surface and serialized inference coordination for Mage-VL."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
import uuid
from typing import Any, AsyncIterator, Iterator, Mapping

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .engine import MageEngine
from .request import prepare_request
from .settings import AdapterSettings


LOGGER = logging.getLogger("mage_vl_adapter")


def _openai_chunk(
    request_id: str,
    model: str,
    *,
    content: str = "",
    role: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if role is not None:
        delta["role"] = role
    if content:
        delta["content"] = content
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


def _sse(data: Mapping[str, Any] | str) -> str:
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def _run_blocking(function: Any, *args: Any) -> Any:
    """Finish an in-flight GPU call before allowing another request after cancel."""

    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            await task
        except BaseException:
            pass
        raise


_ITERATOR_END = object()


async def _next_or_end(iterator: Iterator[str]) -> tuple[bool, str]:
    value = await _run_blocking(next, iterator, _ITERATOR_END)
    if value is _ITERATOR_END:
        return True, ""
    return False, str(value)


def create_app(
    settings: AdapterSettings | None = None, engine: MageEngine | None = None
) -> FastAPI:
    adapter_settings = settings or AdapterSettings.from_env()
    mage = engine or MageEngine(adapter_settings)
    app = FastAPI(title="Mage-VL Local Adapter", version="1.0")
    app.state.settings = adapter_settings
    app.state.engine = mage
    app.state.inference_lock = asyncio.Lock()
    app.state.uvicorn_server = None

    @app.get("/health")
    async def health() -> JSONResponse:
        problems = mage.validate_runtime()
        body = {
            "status": "ok" if not problems else "error",
            "model": adapter_settings.model_id,
            "model_status": mage.status,
            "model_dir": str(adapter_settings.model_dir),
            "ffmpeg": str(adapter_settings.ffmpeg_path),
        }
        if problems:
            body["problems"] = problems
            return JSONResponse(body, status_code=503)
        if mage.last_error:
            body["last_error"] = mage.last_error
        return JSONResponse(body)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": adapter_settings.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "microsoft",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="request body must be an object")
        requested_model = str(payload.get("model") or adapter_settings.model_id)
        if requested_model != adapter_settings.model_id:
            raise HTTPException(status_code=404, detail=f"unknown model: {requested_model}")
        prepared = prepare_request(payload, adapter_settings)
        request_id = "chatcmpl-mage-" + uuid.uuid4().hex

        if bool(payload.get("stream", False)):
            async def stream_body() -> AsyncIterator[str]:
                try:
                    yield _sse(
                        _openai_chunk(
                            request_id,
                            adapter_settings.model_id,
                            role="assistant",
                        )
                    )
                    async with app.state.inference_lock:
                        iterator = iter(mage.analyze(prepared))
                        while True:
                            ended, content = await _next_or_end(iterator)
                            if ended:
                                break
                            yield _sse(
                                _openai_chunk(
                                    request_id,
                                    adapter_settings.model_id,
                                    content=content,
                                )
                            )
                    yield _sse(
                        _openai_chunk(
                            request_id,
                            adapter_settings.model_id,
                            finish_reason="stop",
                        )
                    )
                    yield _sse("[DONE]")
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    LOGGER.exception("video analysis failed")
                    yield _sse(
                        {
                            "error": {
                                "message": f"{type(exc).__name__}: {exc}",
                                "type": "mage_vl_error",
                            }
                        }
                    )
                    yield _sse("[DONE]")
                finally:
                    shutil.rmtree(prepared.request_dir, ignore_errors=True)

            return StreamingResponse(
                stream_body(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        try:
            async with app.state.inference_lock:
                parts = await _run_blocking(lambda: list(mage.analyze(prepared)))
        except BaseException as exc:
            LOGGER.exception("video analysis failed")
            raise HTTPException(
                status_code=500, detail=f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            shutil.rmtree(prepared.request_dir, ignore_errors=True)
        content = "".join(parts)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": adapter_settings.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    @app.post("/shutdown")
    async def shutdown(request: Request) -> dict[str, Any]:
        client_host = request.client.host if request.client else ""
        if client_host not in {"127.0.0.1", "::1", "testclient"}:
            raise HTTPException(status_code=403, detail="shutdown is loopback-only")
        server = app.state.uvicorn_server
        if server is None:
            return {"status": "accepted", "detail": "test server has no runtime owner"}
        asyncio.get_running_loop().call_later(
            0.2, lambda: setattr(server, "should_exit", True)
        )
        return {"status": "accepted", "detail": "graceful shutdown requested"}

    return app
