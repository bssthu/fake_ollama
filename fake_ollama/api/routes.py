"""HTTP route registration separated from proxy implementation details."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

def register_routes(app: FastAPI) -> None:
    from .. import routing, protocol_handlers, media_handlers, model_catalog

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
        settings, iface = routing._model_interface_for_request(request)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        models = []
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
    async def show(request: Request, payload: Dict[str, Any]) -> Dict[str, Any]:
        settings: Settings = app.state.settings
        name = payload.get("name") or payload.get("model") or ""
        backend, display_model = routing._dispatch(request, settings, name)
        profile = settings.profile_for(f"{display_model}@{backend.name}")
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
        return await protocol_handlers._handle(request, mode="chat")

    @app.post("/api/generate")
    async def generate(request: Request) -> Any:
        return await protocol_handlers._handle(request, mode="generate")

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
        settings, iface = routing._model_interface_for_request(request)
        names = routing._public_model_ids(iface)
        headers = {
            "Cache-Control": "no-store",
            "Vary": "anthropic-version",
        }
        if request.headers.get("anthropic-version"):
            entries = [
                model_catalog._anthropic_model_entry(settings, iface, name) for name in names
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
                "data": [model_catalog._openai_model_entry(name, now) for name in names],
            },
            headers=headers,
        )

    @app.get("/v1/models/{model_id}")
    async def retrieve_model(model_id: str, request: Request) -> JSONResponse:
        settings, iface = routing._model_interface_for_request(request)
        if model_id not in routing._public_model_ids(iface):
            raise HTTPException(status_code=404, detail="model not found")
        if request.headers.get("anthropic-version"):
            payload = model_catalog._anthropic_model_entry(settings, iface, model_id)
        else:
            now = int(datetime.now(timezone.utc).timestamp())
            payload = model_catalog._openai_model_entry(model_id, now)
        return JSONResponse(
            payload,
            headers={
                "Cache-Control": "no-store",
                "Vary": "anthropic-version",
            },
        )

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Any:
        return await protocol_handlers._handle_openai_chat(request)

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> Any:
        from ..decision_handlers import handle_decision
        return await handle_decision(request, "systemone")

    @app.post("/v1/decisions")
    async def decisions(request: Request) -> Any:
        from ..decision_handlers import handle_decision
        return await handle_decision(request, "openai_decisions")

    @app.post("/v1/images/generations")
    async def openai_image_generations(request: Request) -> Any:
        return await media_handlers._handle_openai_image_generation(request)

    @app.post("/v1/images/edits")
    async def openai_image_edits(request: Request) -> Any:
        return await media_handlers._handle_openai_image_edit(request)

    @app.post("/v1/videos/generations")
    async def openai_video_generations(request: Request) -> Any:
        return await media_handlers._handle_openai_video_generation(request)

    @app.post("/v1/videos/context-ir")
    async def h3_context_ir(request: Request) -> Any:
        return await media_handlers._handle_h3_context_ir(request)

    @app.post("/v1/embeddings")
    async def openai_embeddings(payload: Dict[str, Any]) -> JSONResponse:
        raise HTTPException(
            status_code=501,
            detail="Embeddings are not supported by the upstream Anthropic API.",
        )

    # ---- Reverse proxy: Anthropic Messages API -> local Ollama ----------

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> Any:
        return await protocol_handlers._handle_anthropic_messages(request)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request) -> Any:
        return await protocol_handlers._handle_anthropic_count_tokens(request)

    # ---- Web admin UI ---------------------------------------------------

    from .. import admin as _admin
    _admin.register_admin_routes(app)
    from .. import dashboard as _dashboard
    _dashboard.register_dashboard_routes(app)
    from .. import playground as _playground
    _playground.register_playground_routes(app)
