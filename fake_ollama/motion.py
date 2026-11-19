"""Dispatch pose-conditioned skeletal animation through a managed adapter."""

import json

import httpx
from fastapi import HTTPException, Request

from .config import GenericOpenAITarget
from .proxy_errors import _log_upstream_error, _upstream_error
from .routing import _authorise_interface, _backend_client, _dispatch


async def handle_motion(request: Request):
    _authorise_interface(request)
    raw = await request.body()
    if len(raw) > 512 * 1024:
        raise HTTPException(413, "Motion requests must fit within 512 KiB")
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise HTTPException(400, "Invalid motion JSON") from exc
    if not isinstance(body, dict) or not isinstance(body.get("model"), str) or not body["model"].strip():
        raise HTTPException(400, "Expected a motion request with a model ID")
    settings = request.app.state.settings
    backend, display = _dispatch(request, settings, body["model"])
    profile = settings.profile_for(f"{display}@{backend.name}")
    if not isinstance(backend.source, GenericOpenAITarget) or "motion_generation" not in profile.capabilities:
        raise HTTPException(400, "This model does not support motion generation")
    walk_example = set(body) == {"model", "example"} and body["example"] == "walk"
    custom_poses = (set(body) == {"model", "context_poses", "target_poses"}
                    and isinstance(body["context_poses"], list) and len(body["context_poses"]) == 4
                    and isinstance(body["target_poses"], list) and 1 <= len(body["target_poses"]) <= 4)
    if not (walk_example or custom_poses):
        raise HTTPException(400, "Supply four context_poses and one to four target_poses, or example=walk")
    try:
        # The standard JSON client cannot forward NaN or infinities.
        json.dumps(body, allow_nan=False)
    except (ValueError, RecursionError) as exc:
        raise HTTPException(400, "Pose values must be finite numbers") from exc
    payload = {**body, "model": backend.source.resolve_model(display)}
    try:
        result = await _backend_client(request.app, backend).generate_motion(
            payload, estimated_vram_gb=profile.estimated_vram_gb,
            estimated_memory_gb=profile.estimated_memory_gb,
        )
    except httpx.HTTPError as exc:
        _log_upstream_error(request, exc, {"model": body["model"]})
        return _upstream_error(exc)
    return {**result, "model": body["model"]}
