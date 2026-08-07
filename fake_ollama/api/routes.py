"""HTTP route registration separated from proxy implementation details."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse



def register_routes(app: FastAPI) -> None:
    from .. import server as core
    @app.get("/")
    async def root() -> str:
        return "Ollama is running"

    @app.head("/")
    async def root_head() -> str:
        return ""

    @app.get("/api/version")
    async def version() -> Dict[str, str]:
        settings: Settings = app.state.settings
        return {"version": settings.advertised_version}

    @app.get("/api/tags")
    async def tags(request: Request) -> Dict[str, Any]:
        settings: Settings = app.state.settings
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        models = []
        iface = core._interface_for(request)
        public_ids = iface.public_ids() if iface is not None else []
        for name in public_ids:
            models.append(
                {
                    "name": name,
                    "model": name,
                    "modified_at": now,
                    "size": 0,
                    "digest": "sha256:" + "0" * 64,
                    "details": {
                        "parent_model": "",
                        "format": "anthropic-proxy",
                        "family": "claude",
                        "families": ["claude"],
                        "parameter_size": "unknown",
                        "quantization_level": "none",
                    },
                }
            )
        return {"models": models}

    @app.get("/api/ps")
    async def ps() -> Dict[str, Any]:
        return {"models": []}

    @app.post("/api/show")
    async def show(payload: Dict[str, Any]) -> Dict[str, Any]:
        settings: Settings = app.state.settings
        name = payload.get("name") or payload.get("model") or ""
        profile = settings.profile_for(name)
        # Some clients (e.g. the GitHub Copilot VS Code extension's Ollama
        # integration) silently drop models whose /api/show response does not
        # advertise the capabilities they need ("completion" for chat,
        # "tools" for tool-calling, "vision" for image input). We surface
        # the per-model profile here so users can configure exactly what each
        # model claims to support.
        capabilities = [
            capability
            for capability in profile.capabilities
            if capability in {"completion", "tools", "vision"}
        ]
        ctx_len = profile.context_length
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return {
            "license": "proprietary (proxied)",
            "modelfile": f"# Proxied via fake-ollama -> {name}",
            "parameters": f"num_ctx                        {ctx_len}",
            "template": "{{ .Prompt }}",
            "details": {
                "parent_model": "",
                "format": "anthropic-proxy",
                "family": "claude",
                "families": ["claude"],
                "parameter_size": "unknown",
                "quantization_level": "none",
            },
            "model_info": {
                "general.architecture": "claude",
                "general.basename": name,
                "general.parameter_count": 0,
                "general.context_length": ctx_len,
                "claude.context_length": ctx_len,
            },
            "capabilities": capabilities,
            "modified_at": now,
        }

    @app.post("/api/chat")
    async def chat(request: Request) -> Any:
        return await core._handle(request, mode="chat")

    @app.post("/api/generate")
    async def generate(request: Request) -> Any:
        return await core._handle(request, mode="generate")

    @app.post("/api/embeddings")
    @app.post("/api/embed")
    async def embeddings(payload: Dict[str, Any]) -> JSONResponse:
        # Anthropic does not provide embeddings; expose a clear error.
        raise HTTPException(
            status_code=501,
            detail="Embeddings are not supported by the upstream Anthropic API.",
        )

    # ---- Provider-native model discovery + OpenAI-compatible endpoints --

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        settings, iface = core._model_interface_for_request(request)
        names = core._public_model_ids(iface)
        headers = {
            "Cache-Control": "no-store",
            "Vary": "anthropic-version",
        }
        if request.headers.get("anthropic-version"):
            entries = [
                core._anthropic_model_entry(settings, iface, name) for name in names
            ]
            return JSONResponse(
                {
                    "data": entries,
                    "first_id": names[0] if names else None,
                    "has_more": False,
                    "last_id": names[-1] if names else None,
                },
                headers=headers,
            )

        now = int(datetime.now(timezone.utc).timestamp())
        return JSONResponse(
            {
                "object": "list",
                "data": [core._openai_model_entry(name, now) for name in names],
            },
            headers=headers,
        )

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, request: Request) -> JSONResponse:
        settings, iface = core._model_interface_for_request(request)
        if model_id not in core._public_model_ids(iface):
            raise HTTPException(status_code=404, detail="model not found")
        if request.headers.get("anthropic-version"):
            payload = core._anthropic_model_entry(settings, iface, model_id)
        else:
            now = int(datetime.now(timezone.utc).timestamp())
            payload = core._openai_model_entry(model_id, now)
        return JSONResponse(
            payload,
            headers={
                "Cache-Control": "no-store",
                "Vary": "anthropic-version",
            },
        )

    @app.get("/playground/api/models", include_in_schema=False)
    async def playground_models(request: Request) -> JSONResponse:
        settings, iface = core._model_interface_for_request(request)
        models = [
            core._playground_model_entry(settings, iface, name)
            for name in core._public_model_ids(iface)
        ]
        models.extend(
            core._playground_context_ir_entry(settings, iface, profile)
            for profile in settings.h3_context_ir_profiles
            if profile.enabled and profile.playground_visible
        )
        return JSONResponse(
            {
                "schema_version": 1,
                "models": models,
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/playground/api/external-models", include_in_schema=False)
    async def playground_external_models(request: Request) -> JSONResponse:
        settings, _iface = core._model_interface_for_request(request)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        requested_profile = str(payload.get("profile") or "").strip()
        profile = settings.h3_context_ir_profile_by_name(requested_profile)
        if profile is None or not profile.enabled:
            raise HTTPException(
                status_code=404,
                detail=f"H3 Context-IR profile {requested_profile!r} not found",
            )
        if not profile.allow_external_api:
            raise HTTPException(
                status_code=403,
                detail="this H3 Context-IR profile does not allow external APIs",
            )
        protocol = core._external_planner_protocol(payload.get("protocol"))
        base_url = core._normalize_external_planner_base_url(payload.get("base_url"))
        token = core._external_planner_token(request)
        try:
            async with core._external_planner_http_client(request.app) as client:
                response = await client.get(
                    f"{base_url}/v1/models",
                    headers=core._external_planner_headers(protocol, token),
                    timeout=min(30.0, settings.timeout_seconds),
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    "external model detection failed: upstream returned "
                    f"HTTP {exc.response.status_code}"
                ),
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail=f"external model detection failed: {exc}",
            ) from exc
        if not isinstance(data, dict):
            raise HTTPException(
                status_code=502,
                detail="external model endpoint returned a non-object response",
            )
        raw_models = data.get("data") or data.get("models") or []
        names: List[str] = []
        seen: set[str] = set()
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            name = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)
            if len(names) >= 1000:
                break
        return JSONResponse(
            {
                "protocol": protocol,
                "base_url": base_url,
                "models": names,
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Any:
        return await core._handle_openai_chat(request)

    @app.post("/v1/images/generations")
    async def openai_image_generations(request: Request) -> Any:
        return await core._handle_openai_image_generation(request)

    @app.post("/v1/images/edits")
    async def openai_image_edits(request: Request) -> Any:
        return await core._handle_openai_image_edit(request)

    @app.post("/v1/videos/generations")
    async def openai_video_generations(request: Request) -> Any:
        return await core._handle_openai_video_generation(request)

    @app.post("/v1/videos/context-ir")
    async def h3_context_ir(request: Request) -> Any:
        return await core._handle_h3_context_ir(request)

    @app.post("/v1/embeddings")
    async def openai_embeddings(payload: Dict[str, Any]) -> JSONResponse:
        raise HTTPException(
            status_code=501,
            detail="Embeddings are not supported by the upstream Anthropic API.",
        )

    # ---- Reverse proxy: Anthropic Messages API -> local Ollama ----------

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Any:
        return await core._handle_anthropic_messages(request)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request) -> Any:
        return await core._handle_anthropic_count_tokens(request)

    # ---- Web admin UI ---------------------------------------------------

    from .. import admin as _admin
    _admin.register_admin_routes(app)
    from .. import dashboard as _dashboard
    _dashboard.register_dashboard_routes(app)
    from .. import playground as _playground
    _playground.register_playground_routes(app)
