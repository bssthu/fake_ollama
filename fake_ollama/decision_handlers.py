"""Async decision dispatch through the existing listener/exposure boundary."""
import asyncio
from dataclasses import replace
import json

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.requests import ClientDisconnect

from .decisions import CODECS, DecisionValidationError
from .lifecycle import finish_cleanup
from .proxy_errors import _log_upstream_error, _upstream_error
from .routing import _authorise_interface, _backend_client, _dispatch


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DecisionValidationError("Duplicate JSON object keys are not supported")
        value[key] = item
    return value


async def _decide_until_disconnect(request: Request, client, decision, profile):
    """Cancel waiting work; submitted work retains its lease until it drains."""
    async def disconnected():
        # handle_decision has already consumed the complete request body.
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return
    inference = asyncio.create_task(client.decide(
        decision, estimated_vram_gb=profile.estimated_vram_gb,
        estimated_memory_gb=profile.estimated_memory_gb))
    disconnect = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait({inference, disconnect}, return_when=asyncio.FIRST_COMPLETED)
        if disconnect in done:
            disconnect.result()
            raise ClientDisconnect()
        return inference.result()
    finally:
        async def cleanup():
            for task in (inference, disconnect):
                if not task.done():
                    task.cancel()
            await asyncio.gather(inference, disconnect, return_exceptions=True)
        await finish_cleanup(cleanup())


async def handle_decision(request: Request, dialect: str):
    # Authenticate even malformed requests before inspecting their content.
    _authorise_interface(request)
    encoding = request.headers.get("content-encoding", "identity").lower()
    if encoding != "identity":
        raise HTTPException(415, "Decision requests currently require uncompressed JSON")
    codec = CODECS[dialect]
    try:
        payload = json.loads((await request.body()).decode("utf-8"), object_pairs_hook=_unique_object)
        parsed = codec.parse_request(payload)
    except (ValueError, UnicodeError, RecursionError) as exc:
        message = str(exc) if isinstance(exc, DecisionValidationError) else "Invalid decision request JSON"
        raise HTTPException(400, message) from exc
    settings = request.app.state.settings
    backend, real_model = _dispatch(request, settings, parsed.model)
    if backend.protocol != "decision":
        raise HTTPException(400, "This model is not configured for decisions; select a decision target")
    if parsed.options.get("assets") and request.headers.get("typesafe-danger-zone-images", "").lower() != "true":
        raise HTTPException(400, "SystemOne images require TypeSafe-Danger-Zone-Images: true")
    client = _backend_client(request.app, backend)
    profile = settings.profile_for(f"{real_model}@{backend.name}")
    wire_request = replace(parsed, model=backend.source.resolve_model(real_model))
    try:
        result = await _decide_until_disconnect(request, client, wire_request, profile)
        return JSONResponse(codec.encode_response(result, parsed))
    except DecisionValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    except httpx.HTTPError as exc:
        _log_upstream_error(request, exc, {"model": parsed.model})
        return _upstream_error(exc)
