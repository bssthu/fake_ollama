"""FastAPI composition root and backwards-compatible helper exports."""

from __future__ import annotations

import secrets
from fastapi import FastAPI, Request
from . import __version__
from .config import Settings, get_settings
from .protocol_tools import UpstreamProtocolError
from .runtime_state import RuntimeState, RuntimeResources, RuntimeSnapshotMiddleware
from .security import RequestBodyLimitMiddleware
from .request_data_log import RequestDataLogMiddleware
from .vram import MemoryCoordinator, VramCoordinator
from .dashboard import DashboardState, RequestMetrics
from .app_lifecycle import (
    _local_target_idle_monitor,
    _sync_local_target_idle_monitor,
    _sync_runtime_monitor,
    lifespan,
    request_shutdown,
)
from .media_handlers import (
    _anthropic_context_ir_messages,
    _default_image_model,
    _dispatch_image_backend,
    _handle_h3_context_ir,
    _handle_openai_image_edit,
    _handle_openai_image_generation,
    _handle_openai_video_generation,
    _image_media_type,
    _image_response,
    _invoke_context_ir_provider,
    _neutral_context_ir_messages,
    _ollama_context_ir_messages,
    _openai_context_ir_messages,
    _release_context_ir_provider_for_video,
    _run_h3_context_ir,
    _select_context_ir_provider,
    _text_from_anthropic_response,
    _text_from_openai_response,
)
from .model_catalog import (
    _anthropic_model_capabilities,
    _anthropic_model_entry,
    _openai_model_entry,
)
from .playground_support import (
    EXTERNAL_PLANNER_TOKEN_HEADER,
    ExternalPlannerConnection,
    _EXTERNAL_PLANNER_PROTOCOLS,
    _H3_BASE_MODE_DESCRIPTION,
    _authorise_external_planner_url,
    _compatible_context_ir_provider_choices,
    _context_ir_provider_choice,
    _context_ir_provider_is_exposed,
    _external_context_ir_provider_choice,
    _external_planner_connection,
    _external_planner_headers,
    _external_planner_http_client,
    _external_planner_protocol,
    _external_planner_token,
    _h3_base_mode_choices,
    _normalize_external_planner_base_url,
    _playground_context_ir_entry,
    _playground_model_entry,
)
from .protocol_handlers import (
    _handle,
    _handle_anthropic_count_tokens,
    _handle_anthropic_messages,
    _handle_ollama_via_openai_upstream,
    _handle_openai_chat,
    _openai_upstream_messages,
    _openai_upstream_stream_messages,
)
from .proxy_errors import (
    _anthropic_error_response,
    _json_error,
    _log_upstream_error,
    _read_error_text,
    _upstream_error,
    _upstream_error_status,
)
from .request_validation import (
    _COUNT_TOKENS_BODY_KEYS,
    _apply_llama_cpp_thinking_config,
    _apply_ollama_thinking_config,
    _apply_reverse_output_limits,
    _apply_thinking_config,
    _coerce_bool_payload,
    _coerce_float_payload,
    _coerce_int_payload,
    _context_ir_duration,
    _count_tokens_payload,
    _decode_base64_image,
    _enforce_limits,
    _enforce_reference_image_limit,
    _image_payload_from_request,
    _image_request_params,
    _parse_image_size,
    _resolve_image_seed,
    _video_request_params,
)
from .routing import (
    ForwardedCycleMiddleware,
    _ADMIN_ONLY_PATH_PREFIXES,
    _API_ONLY_PATH_PREFIXES,
    _DASHBOARD_ONLY_PATH_PREFIXES,
    _OLLAMA_ONLY_PATH_PREFIXES,
    _PLAYGROUND_ONLY_PATH_PREFIXES,
    _SHARED_V1_PATH_PREFIXES,
    _authorise_interface,
    _backend_client,
    _bearer_or_api_key,
    _dispatch,
    _has_path_prefix,
    _install_port_router,
    _interface_for,
    _is_external_request,
    _listener_name,
    _local_port,
    _log_request_access,
    _log_request_exception,
    _model_interface_for_request,
    _openai_client_for,
    _playground_interface_for,
    _profile_for_public_id,
    _public_model_ids,
    _request_log_context,
    _request_surface,
)

def create_app(settings: Settings | None = None, *, test_listener_fallback: bool = False) -> FastAPI:
    settings = settings or get_settings()


    app = FastAPI(title="fake-ollama", version=__version__, lifespan=lifespan)
    app.state = RuntimeState()
    app.state.runtime_resources = RuntimeResources(app)
    app.state.actual_listener_settings = settings.model_copy(deep=True)
    app.state.desired_settings = settings
    app.state.settings = settings
    app.state.test_listener_fallback = test_listener_fallback
    app.state.thinking_cache_namespace = secrets.token_hex(16)
    @app.exception_handler(UpstreamProtocolError)
    async def protocol_error(request: Request, exc: UpstreamProtocolError):
        if request.url.path.startswith("/v1/messages"):
            return _anthropic_error_response(exc)
        return _upstream_error(exc)
    app.state.shutdown_requested = False
    app.state.vram_coordinator = VramCoordinator()
    app.state.memory_coordinator = MemoryCoordinator()
    app.state.dashboard_state = DashboardState()
    app.state.request_metrics = RequestMetrics()
    app.state.ensure_local_target_idle_monitor = _sync_local_target_idle_monitor
    app.state.ensure_runtime_monitor = _sync_runtime_monitor
    app.add_middleware(RequestDataLogMiddleware)
    app.add_middleware(RequestBodyLimitMiddleware)
    _install_port_router(app)
    app.add_middleware(ForwardedCycleMiddleware)
    app.add_middleware(RuntimeSnapshotMiddleware)
    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:
    from .api.routes import register_routes

    register_routes(app)
