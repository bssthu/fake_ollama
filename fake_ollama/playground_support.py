"""Playground capabilities and explicitly configured external planners."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit
import httpx
from fastapi import FastAPI, HTTPException, Request
from .config import ComfyUITarget, DecisionTarget, H3ContextIRProfile, H3ContextIRProvider, Settings, outbound_cycle_headers
from .media_operations import describe_comfyui_operation
from .routing import (
    _listener_name,
    _profile_for_public_id,
)

EXTERNAL_PLANNER_TOKEN_HEADER = "x-playground-upstream-key"


_EXTERNAL_PLANNER_PROTOCOLS = ("openai", "anthropic")


def _h3_base_mode_choices() -> List[Dict[str, str]]:
    """Human-readable H3 Base modes shared by both Playground operations."""

    return [
        {
            "value": "auto",
            "label": "自动选择 · 0 张文本 / 1 张首帧 / 2 张首尾帧",
        },
        {
            "value": "t2va",
            "label": "纯文本生成视频 · T2VA（0 张参考图）",
        },
        {
            "value": "i2va",
            "label": "首帧引导视频 · I2VA（1 张参考图）",
        },
        {
            "value": "fl2va",
            "label": "首尾帧约束视频 · FL2VA（2 张参考图）",
        },
        {
            "value": "l2va",
            "label": "末帧约束视频 · L2VA（1 张参考图，需手动选择）",
        },
    ]


_H3_BASE_MODE_DESCRIPTION = (
    "自动模式按参考图数量选择：0 张用 T2VA，1 张用首帧 I2VA，"
    "2 张用首尾帧 FL2VA。L2VA 同样只需要 1 张图，但该图作为末帧，"
    "因此必须手动选择。"
)


@dataclass(frozen=True)
class ExternalPlannerConnection:
    """Validated, request-scoped credentials for a Playground Planner."""

    protocol: str
    base_url: str
    auth_token: str
    model: str
    modalities: tuple[str, ...]

    @property
    def accepts_images(self) -> bool:
        return "image" in self.modalities


def _playground_model_entry(
    settings: Settings, iface: Any, public_id: str, decision_clients: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Describe one model for fake-ollama's versioned Playground API."""

    profile = _profile_for_public_id(settings, iface, public_id)
    exposure = iface.exposure_for_public_id(public_id) if iface is not None else None
    source = settings.source_by_name(exposure.target) if exposure is not None else None
    capabilities = ["decision"] if isinstance(source, DecisionTarget) else list(profile.capabilities)
    capability_set = set(capabilities)
    operations: List[Dict[str, Any]] = []
    if isinstance(source, DecisionTarget):
        # Public wire formats are distinct codecs, never aliases. Images remain
        # native-dialect only until an explicit multimodal conversion exists.
        metadata = source.capability_metadata()
        client = (decision_clients or {}).get(source.name)
        if client is not None:
            # Discovery is read-only and never starts/probes a provider. Observed
            # restrictions can narrow declared features, but cannot expand them.
            observed = client.capability_snapshot()
            for field in ("question_types", "modalities", "extensions", "think_modes"):
                if field in observed:
                    metadata[field] = [item for item in metadata.get(field, []) if item in observed[field]]
            for field in ("version", "endpoint_verified", "status", "warnings", "errors", "runtime_metadata"):
                if field in observed:
                    metadata[field] = observed[field]
        operations.append({
            **metadata,
            "id": "decision",
            "endpoint": "/v1/systemone" if source.dialect == "systemone" else "/v1/decisions",
            "dialects": [
                {"id": "systemone", "label": "TypeSafe SystemOne", "endpoint": "/v1/systemone"},
                {"id": "openai_decisions", "label": "OpenAI Decisions", "endpoint": "/v1/decisions"},
            ],
            "stream": False,
            "history_mode": "single_turn",
            "accepts_images": "image" in metadata.get("modalities", []),
            "multiple_images": True,
            "image_dialect": source.dialect,
            "parameters": [],
        })
    if capability_set.intersection({"completion", "tools", "vision"}):
        operations.append(
            {
                "id": "chat",
                "endpoint": "/v1/chat/completions",
                "stream": True,
                "history_mode": "conversation",
                "accepts_images": "vision" in capability_set,
                "tool_calling": "tools" in capability_set,
            }
        )
    if "image_generation" in capability_set:
        operations.append(
            {
                "id": "image_generation",
                "endpoint": "/v1/images/generations",
                "stream": False,
                "history_mode": "single_turn",
                "accepts_images": False,
            }
        )
    if "image_edit" in capability_set:
        operations.append(
            {
                "id": "image_edit",
                "endpoint": "/v1/images/edits",
                "stream": False,
                "history_mode": "single_turn",
                "accepts_images": True,
                "requires_images": True,
                "multiple_images": True,
            }
        )
    if "video_generation" in capability_set:
        operations.append(
            {
                "id": "video_generation",
                "endpoint": "/v1/videos/generations",
                "stream": False,
                "history_mode": "single_turn",
                "accepts_images": True,
                "requires_images": False,
                "multiple_images": True,
            }
        )
    if "video_understanding" in capability_set:
        operations.append(
            {
                "id": "video_analysis",
                "endpoint": "/v1/chat/completions",
                "stream": True,
                "history_mode": "single_turn",
                "accepts_videos": True,
                "requires_videos": True,
                "multiple_videos": False,
                "limits": {
                    "max_videos": 1,
                    "max_video_bytes": 64 * 1024 * 1024,
                },
                "live_camera": {
                    "supported": True,
                    "capture_mode": "windowed_media_recorder",
                    "max_pending_segments": 1,
                },
                "parameters": [
                    {
                        "name": "segment_seconds",
                        "label": "时间窗口（秒）",
                        "type": "number",
                        "default": 8,
                        "min": 2,
                        "max": 60,
                        "step": 1,
                        "description": "逐段分析的视频窗口长度。",
                    },
                    {
                        "name": "frames_per_segment",
                        "label": "每段抽帧数",
                        "type": "integer",
                        "default": 8,
                        "min": 2,
                        "max": 32,
                        "step": 1,
                        "description": "FFmpeg 在每个时间窗口内均匀抽取的帧数。",
                    },
                    {
                        "name": "max_segments",
                        "label": "最多分析段数",
                        "type": "integer",
                        "default": 12,
                        "min": 1,
                        "max": 120,
                        "step": 1,
                        "description": "长视频超过上限时沿完整时间轴均匀选段；增加段数只增加串行耗时。",
                    },
                    {
                        "name": "include_summary",
                        "label": "生成整体总结",
                        "type": "boolean",
                        "default": False,
                        "description": "逐段分析后再调用一次模型汇总，耗时会增加。",
                    },
                ],
            }
        )
    if isinstance(source, ComfyUITarget):
        for operation in operations:
            if operation["id"] not in {
                "image_generation", "image_edit", "video_generation"
            }:
                continue
            operation.update(describe_comfyui_operation(source, operation["id"]))
            if operation["id"] == "video_generation" and source.context_ir_profile:
                ir_profile = settings.h3_context_ir_profile_by_name(
                    source.context_ir_profile
                )
                if ir_profile is None or not ir_profile.enabled:
                    operation["configured"] = False
                    operation["context_ir_profile"] = source.context_ir_profile
                    continue
                provider_choices = [
                    {
                        "value": "auto",
                        "label": "Auto by input modality",
                        "group": "推荐 Planner",
                        "selection_kind": "auto",
                    }
                ]
                for provider in ir_profile.providers:
                    provider_choices.append(
                        _context_ir_provider_choice(
                            settings,
                            provider,
                            value=provider.name,
                            label=(
                                f"{provider.name} "
                                f"({'/'.join(provider.modalities)}; "
                                f"{provider.model}@{provider.target})"
                            ),
                            group="推荐 Planner",
                            selection_kind="recommended",
                        )
                    )
                provider_choices.extend(
                    _compatible_context_ir_provider_choices(
                        settings, iface, ir_profile
                    )
                )
                external_choice = _external_context_ir_provider_choice(ir_profile)
                if external_choice is not None:
                    provider_choices.append(external_choice)
                operation["context_ir_profile"] = ir_profile.name
                operation["planner_defaults"] = {
                    "text": ir_profile.default_text_provider,
                    "image": (
                        ir_profile.default_multimodal_provider
                        or ir_profile.default_text_provider
                    ),
                }
                if ir_profile.allow_external_api:
                    operation["external_planner_api"] = {
                        "models_endpoint": "/playground/api/external-models",
                        "protocols": list(_EXTERNAL_PLANNER_PROTOCOLS),
                    }
                operation.setdefault("parameters", []).extend(
                    [
                        {
                            "name": "prompt_mode",
                            "label": "Prompt 处理方式",
                            "type": "select",
                            "default": source.context_ir_prompt_mode,
                            "choices": [
                                {
                                    "value": "auto",
                                    "label": "自动判断（结构化 Prompt 直接使用）",
                                },
                                {
                                    "value": "raw",
                                    "label": "直接使用输入（跳过结构化生成）",
                                },
                                {
                                    "value": "enhance",
                                    "label": "始终自动结构化",
                                },
                            ],
                            "description": (
                                "自动判断会让 H3 三段式 Prompt 直接进入 MiniMax H3，"
                                "仅增强普通描述；直接使用输入则完全跳过 Planner。"
                            ),
                        },
                        {
                            "name": "context_ir_provider",
                            "label": "Context-IR provider",
                            "type": "select",
                            "default": "auto",
                            "choices": provider_choices,
                            "advanced": True,
                        },
                        {
                            "name": "context_ir_mode",
                            "label": "H3 Base 模式",
                            "type": "select",
                            "default": "auto",
                            "choices": _h3_base_mode_choices(),
                            "description": _H3_BASE_MODE_DESCRIPTION,
                            "advanced": True,
                            "wide": True,
                        },
                    ]
                )
            if source.preset == "minimax_h3" and operation["id"] == "video_generation":
                operation["parameters"] = [parameter for parameter in operation["parameters"]
                                           if parameter["name"] != "context_ir_mode"]
    return {
        "id": public_id,
        "context_length": profile.context_length,
        "max_output_tokens": profile.max_output_tokens,
        "estimated_vram_gb": profile.estimated_vram_gb,
        "estimated_memory_gb": profile.estimated_memory_gb,
        "request_vram_headroom_gb": profile.request_vram_headroom_gb,
        "min_free_vram_gb": profile.min_free_vram_gb,
        "vram_cleanup_policy": profile.vram_cleanup_policy,
        "exclusive_gpu": profile.exclusive_gpu,
        "capabilities": capabilities,
        "operations": operations,
    }


def _context_ir_provider_choice(
    settings: Settings,
    provider: H3ContextIRProvider,
    *,
    value: str,
    label: str,
    group: str,
    selection_kind: str,
) -> Dict[str, Any]:
    model_profile = settings.profile_for(
        f"{provider.model}@{provider.target}"
    )
    backend = settings.backend_by_name(provider.target)
    return {
        "value": value,
        "label": label,
        "group": group,
        "selection_kind": selection_kind,
        "model": provider.model,
        "target": provider.target,
        "modalities": list(provider.modalities),
        "backend_kind": backend.kind if backend is not None else None,
        "protocol": backend.protocol if backend is not None else None,
        "context_length": model_profile.context_length,
        "max_output_tokens": model_profile.max_output_tokens,
        "estimated_vram_gb": model_profile.estimated_vram_gb,
        "estimated_memory_gb": model_profile.estimated_memory_gb,
    }


def _external_context_ir_provider_choice(
    profile: H3ContextIRProfile,
) -> Optional[Dict[str, Any]]:
    if not profile.allow_external_api:
        return None
    return {
        "value": "external",
        "label": "临时第三方 API（URL + Token）",
        "group": "第三方 API",
        "selection_kind": "external",
        "model": None,
        "target": "request-scoped",
        "modalities": ["text"],
        "backend_kind": "remote",
        "protocol": None,
        "context_length": None,
        "max_output_tokens": profile.max_output_tokens,
        "estimated_vram_gb": None,
        "estimated_memory_gb": None,
    }


def _external_planner_protocol(value: Any) -> str:
    protocol = str(value or "openai").strip().lower()
    if protocol not in _EXTERNAL_PLANNER_PROTOCOLS:
        raise HTTPException(
            status_code=400,
            detail="external API protocol must be openai|anthropic",
        )
    return protocol


def _normalize_external_planner_base_url(value: Any) -> str:
    raw = str(value or "").strip().rstrip("/")
    if not raw:
        raise HTTPException(status_code=400, detail="external API base_url is required")
    if len(raw) > 2048:
        raise HTTPException(status_code=400, detail="external API base_url is too long")
    parsed = urlsplit(raw)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise HTTPException(
            status_code=400,
            detail="external API base_url must be an absolute http(s) URL",
        )
    if parsed.username or parsed.password:
        raise HTTPException(
            status_code=400,
            detail="external API credentials must use the token field, not URL userinfo",
        )
    if parsed.query or parsed.fragment:
        raise HTTPException(
            status_code=400,
            detail="external API base_url must not contain a query or fragment",
        )
    path = parsed.path.rstrip("/")
    lowered = path.lower()
    for suffix in ("/chat/completions", "/messages", "/models"):
        if lowered.endswith(suffix):
            path = path[: -len(suffix)].rstrip("/")
            lowered = path.lower()
            break
    if lowered.endswith("/v1"):
        path = path[:-3].rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, path, "", ""))


def _external_planner_token(request: Request) -> str:
    token = request.headers.get(EXTERNAL_PLANNER_TOKEN_HEADER, "").strip()
    if not token:
        raise HTTPException(
            status_code=400,
            detail="external API token is required in x-playground-upstream-key",
        )
    if len(token) > 16384:
        raise HTTPException(status_code=400, detail="external API token is too long")
    return token


def _authorise_external_planner_url(profile: H3ContextIRProfile, value: Any) -> str:
    base_url = _normalize_external_planner_base_url(value)
    allowed = {_normalize_external_planner_base_url(url) for url in profile.external_api_allowed_base_urls}
    if base_url not in allowed:
        raise HTTPException(403, "external API URL is not in this profile's allowed base URLs")
    return base_url


def _external_planner_connection(
    request: Request,
    profile: H3ContextIRProfile,
    payload: Dict[str, Any],
    requested_provider: Any,
) -> Optional[ExternalPlannerConnection]:
    if str(requested_provider or "").strip() != "external":
        return None
    if not profile.allow_external_api:
        raise HTTPException(
            status_code=400,
            detail="this H3 Context-IR profile does not allow external APIs",
        )
    if _listener_name(request) != "playground":
        raise HTTPException(
            status_code=400,
            detail="request-scoped external APIs are only available on the Playground listener",
        )
    protocol = _external_planner_protocol(payload.get("external_api_protocol"))
    base_url = _authorise_external_planner_url(
        profile, payload.get("external_api_base_url")
    )
    model = str(payload.get("external_api_model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="external API model is required")
    if len(model) > 512:
        raise HTTPException(status_code=400, detail="external API model is too long")
    raw_modalities = payload.get("external_api_modalities") or "text"
    if isinstance(raw_modalities, str):
        values = raw_modalities.replace(";", ",").split(",")
    elif isinstance(raw_modalities, list):
        values = raw_modalities
    else:
        values = [raw_modalities]
    modalities = ["text"]
    if any(str(value).strip().lower() == "image" for value in values):
        modalities.append("image")
    return ExternalPlannerConnection(
        protocol=protocol,
        base_url=base_url,
        auth_token=_external_planner_token(request),
        model=model,
        modalities=tuple(modalities),
    )


def _external_planner_headers(protocol: str, token: str) -> Dict[str, str]:
    headers = {"accept": "application/json"}
    if protocol == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    headers["authorization"] = f"Bearer {token}"
    headers["x-api-key"] = token
    headers.update(outbound_cycle_headers())
    return headers


@asynccontextmanager
async def _external_planner_http_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    injected = getattr(app.state, "external_planner_http_client", None)
    if injected is not None:
        yield injected
        return
    settings: Settings = app.state.settings
    async with httpx.AsyncClient(
        timeout=settings.timeout_seconds,
        trust_env=settings.use_system_proxy,
        follow_redirects=False,
    ) as client:
        yield client


def _compatible_context_ir_provider_choices(
    settings: Settings,
    iface: Any,
    profile: H3ContextIRProfile,
) -> List[Dict[str, Any]]:
    """Return compatible chat models exposed to the authenticated interface."""

    if not profile.allow_compatible_models:
        return []
    recommended = {
        (provider.model, provider.target) for provider in profile.providers
    }
    choices: List[Dict[str, Any]] = []
    for exposure in iface.exposed_models:
        try:
            backend, display_model = settings.resolve_request(
                exposure.public_id, interface_name=iface.name
            )
        except ValueError:
            continue
        if backend.protocol not in ("anthropic", "openai", "ollama"):
            continue
        if (display_model, backend.name) in recommended:
            continue
        model_profile = settings.profile_for(
            f"{display_model}@{backend.name}"
        )
        capabilities = set(model_profile.capabilities)
        if "completion" not in capabilities:
            continue
        modalities = ["text"]
        if "vision" in capabilities:
            modalities.append("image")
        provider = H3ContextIRProvider(
            name=f"model:{exposure.public_id}",
            model=display_model,
            target=backend.name,
            modalities=modalities,
            # Plain JSON parsing is the portable choice for arbitrary
            # OpenAI-compatible APIs; recommended providers may opt in to
            # response_format=json_object explicitly in config.
            json_mode=False,
        )
        modality_label = "/".join(modalities)
        choices.append(
            _context_ir_provider_choice(
                settings,
                provider,
                value=provider.name,
                label=f"{exposure.public_id} ({modality_label})",
                group="自选兼容模型",
                selection_kind="compatible",
            )
        )
    return choices


def _playground_context_ir_entry(
    settings: Settings, iface: Any, profile: H3ContextIRProfile
) -> Dict[str, Any]:
    """Expose one orchestration profile as a Playground-only virtual model."""

    default_provider = profile.provider_by_name(
        profile.default_text_provider or profile.providers[0].name
    ) or profile.providers[0]
    model_profile = settings.profile_for(
        f"{default_provider.model}@{default_provider.target}"
    )
    provider_choices = [
        {
            "value": "auto",
            "label": "Auto by input modality",
            "group": "推荐 Planner",
            "selection_kind": "auto",
        }
    ]
    for provider in profile.providers:
        if not _context_ir_provider_is_exposed(settings, iface.name, provider):
            continue
        modalities = "/".join(provider.modalities)
        provider_choices.append(
            _context_ir_provider_choice(
                settings,
                provider,
                value=provider.name,
                label=(
                    f"{provider.name} ({modalities}; "
                    f"{provider.model}@{provider.target})"
                ),
                group="推荐 Planner",
                selection_kind="recommended",
            )
        )
    provider_choices.extend(
        _compatible_context_ir_provider_choices(settings, iface, profile)
    )
    external_choice = _external_context_ir_provider_choice(profile)
    if external_choice is not None:
        provider_choices.append(external_choice)
    operation = {
        "id": "h3_context_ir",
        "endpoint": "/v1/videos/context-ir",
        "stream": False,
        "history_mode": "single_turn",
        "accepts_images": True,
        "requires_images": False,
        "multiple_images": True,
        "limits": {"max_reference_images": 2},
        "planner_defaults": {
            "text": profile.default_text_provider,
            "image": (
                profile.default_multimodal_provider
                or profile.default_text_provider
            ),
        },
        "parameters": [
            {
                "name": "provider",
                "label": "Planner provider",
                "type": "select",
                "default": "auto",
                "choices": provider_choices,
                "description": (
                    "Auto uses the text default without images and the multimodal "
                    "default when reference images are attached. Recommended "
                    "providers are followed by compatible chat models exposed "
                    "on the selected interface."
                ),
            },
            {
                "name": "mode",
                "label": "H3 Base 模式",
                "type": "select",
                "default": "auto",
                "choices": _h3_base_mode_choices(),
                "description": _H3_BASE_MODE_DESCRIPTION,
                "wide": True,
            },
            {
                "name": "duration_seconds",
                "label": "Duration (seconds)",
                "type": "number",
                "default": profile.default_duration_seconds,
                "min": 4,
                "max": 15,
                "step": 0.1,
            },
        ],
    }
    if profile.allow_external_api:
        operation["external_planner_api"] = {
            "models_endpoint": "/playground/api/external-models",
            "protocols": list(_EXTERNAL_PLANNER_PROTOCOLS),
        }
    return {
        "id": profile.public_model_id,
        "context_length": model_profile.context_length,
        "max_output_tokens": profile.max_output_tokens,
        "estimated_vram_gb": model_profile.estimated_vram_gb,
        "estimated_memory_gb": model_profile.estimated_memory_gb,
        "capabilities": ["h3_context_ir"],
        "operations": [operation],
    }


def _context_ir_provider_is_exposed(settings: Settings, interface_name: str, provider: H3ContextIRProvider) -> bool:
    iface = settings.interface_by_name(interface_name)
    backend = settings.backend_by_name(provider.target)
    if iface is None or backend is None:
        return False
    wanted = backend.source.resolve_model(provider.model)
    for exposure in iface.exposed_models:
        if exposure.target == provider.target and backend.source.resolve_model(exposure.model) == wanted:
            return True
    return False
