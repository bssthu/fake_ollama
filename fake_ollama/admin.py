"""Web admin UI for editing config.json.

Architecture
------------
Three small endpoints back the UI:

  GET  /admin/             -> HTML shell + bundled vanilla-JS form renderer
  GET  /admin/schema       -> machine-readable form schema (see ``CONFIG_SCHEMA``)
  GET  /admin/config       -> current effective config (with ``_path`` metadata)
  PUT  /admin/config       -> validate (Pydantic) + persist + hot-reload

The HTML page reads ``schema`` and ``config`` and renders one labelled
input per field (with description and a "use this field" checkbox so the
user can drop a field back to its default). Object lists
(``upstreams`` / ``ollama_targets`` / ``llama_cpp_targets``) and the ``model_profiles`` map are
rendered as repeatable groups with add/remove buttons. A "raw JSON"
toggle is kept as an escape hatch.

Security: there is **no** authentication. Keep the admin listener on
localhost, or disable via ``"admin_enabled": false`` if the service is
reachable from anything untrusted.
"""

from __future__ import annotations

import json
import secrets
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .anthropic_client import AnthropicClient
from .comfyui_client import ComfyUIClient
from .config import (
    ApiInterface,
    AnthropicUpstream,
    ComfyUITarget,
    ExposureEntry,
    GenericOpenAITarget,
    H3ContextIRProfile,
    H3ContextIRProvider,
    LlamaCppDefaults,
    LlamaCppTarget,
    ModelEntry,
    OllamaInterface,
    OllamaTarget,
    OpenAIUpstream,
    Settings,
)
from .llama_cpp_client import LlamaCppClient
from .generic_openai_client import GenericOpenAIClient
from .ollama_client import OllamaClient


def generate_access_token() -> str:
    """Generate the same 24-byte access token exposed by the admin UI."""

    return "tk-" + secrets.token_hex(24)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
#
# Field types understood by the front-end:
#   string, int, float, bool, string_list, string_map, json,
#   object, object_list, object_map
#
# - ``required`` fields cannot be omitted (no delete checkbox).
# - ``secret`` fields render as <input type=password>.
# - ``object_list`` / ``object_map`` carry an ``item_schema`` for inner fields.

MODEL_ENTRY_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "name", "type": "string", "default": "", "required": True,
     "autocomplete": "model_names",
     "description": "上游侧的真实模型名/ID。一个 source 内允许同名出现多次（需为每条指定不同的 alias）"},
    {"key": "alias", "type": "string", "default": None,
     "description": "可选：对外公开的别名。填了则接口层以 alias 为 public id；未填则以 name 为 public id。不能含 '@'"},
]

UPSTREAM_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "name", "type": "string", "default": "", "required": True,
     "description": "唯一名字（routing key）。不能含 '@'"},
    {"key": "base_url", "type": "string", "default": "", "required": True,
     "description": "Anthropic 兼容上游 base URL，例如 https://api.anthropic.com"},
    {"key": "auth_token", "type": "string", "default": "", "secret": True,
     "description": "上游 API token。建议放 .env 而非提交到仓库"},
    {"key": "models", "type": "object_list", "default": [],
     "item_schema": MODEL_ENTRY_ITEM_SCHEMA,
     "nav_label_keys": ["alias", "name"],
     "description": "该 upstream 提供的模型列表。是否被外露取决于哪个 interface 的 exposed_models 包含它。复合 source id = alias_or_name@source_name"},
]

OPENAI_UPSTREAM_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "name", "type": "string", "default": "", "required": True,
     "description": "唯一名字（routing key）。不可与其它 source 同名，且不能含 '@'"},
    {"key": "base_url", "type": "string", "default": "", "required": True,
     "description": "OpenAI 兼容上游 base URL，例如 https://api.openai.com 或 https://api.deepseek.com"},
    {"key": "auth_token", "type": "string", "default": "", "secret": True,
     "description": "上游 API token；同时以 Authorization: Bearer 与 x-api-key 发出，兼容大多数网关"},
    {"key": "models", "type": "object_list", "default": [],
     "item_schema": MODEL_ENTRY_ITEM_SCHEMA,
     "nav_label_keys": ["alias", "name"],
     "description": "该 OpenAI upstream 提供的模型列表。复合 source id = alias_or_name@source_name"},
]

LOCAL_OPENAI_TARGET_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "name", "type": "string", "default": "", "required": True,
     "description": "Unique source name for this generic OpenAI-compatible target."},
    {"key": "base_url", "type": "string", "default": "http://127.0.0.1:8000",
     "description": "Generic OpenAI-compatible server URL, for example vLLM, SGLang, TGI, or an adapter."},
    {"key": "auth_token", "type": "string", "default": "", "secret": True,
     "description": "Optional token sent as Bearer and x-api-key when calling this target."},
    {"key": "models", "type": "object_list", "default": [], "required": True,
     "item_schema": MODEL_ENTRY_ITEM_SCHEMA,
     "nav_label_keys": ["alias", "name"],
     "description": "Models served by this generic OpenAI-compatible target."},
    {"key": "auto_start", "type": "bool", "default": False,
     "description": "Run start_command when a request arrives and health check is not ready."},
    {"key": "start_command", "type": "string", "default": None,
     "description": "Command that starts the generic OpenAI-compatible server."},
    {"key": "stop_command", "type": "string", "default": None,
     "description": "Optional command used to stop the server."},
    {"key": "idle_timeout_seconds", "type": "float", "default": None,
     "description": "Stop the server after this many idle seconds."},
    {"key": "startup_timeout_seconds", "type": "float", "default": 120.0,
     "description": "Maximum seconds to wait for health after auto-start."},
    {"key": "health_path", "type": "string", "default": "/health",
     "description": "Health-check path on the generic OpenAI-compatible server."},
    {"key": "cwd", "type": "string", "default": None,
     "description": "Working directory for start_command and stop_command."},
    {"key": "max_concurrent_requests", "type": "int", "default": None,
     "description": "Optional fake_ollama-side concurrency cap for this target."},
    {"key": "request_read_timeout_seconds", "type": "float", "default": None,
     "description": "Optional read-timeout override; <=0 disables read timeout."},
]

OLLAMA_TARGET_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "name", "type": "string", "default": "", "required": True,
     "description": "唯一名字。不能含 '@'"},
    {"key": "base_url", "type": "string", "default": "http://127.0.0.1:11434",
     "description": "本机 Ollama 服务 URL"},
    {"key": "models", "type": "object_list", "default": [],
     "item_schema": MODEL_ENTRY_ITEM_SCHEMA,
     "nav_label_keys": ["alias", "name"],
     "description": "该本机 Ollama target 可服务的模型列表。复合 source id = alias_or_name@source_name"},
    {"key": "auto_start", "type": "bool", "default": False,
     "description": "请求到来且 health 检查失败时，是否执行 start_command 启动 Ollama daemon"},
    {"key": "start_command", "type": "string", "default": None,
     "description": "可选：启动 Ollama daemon 的命令，例如完整路径的 ollama.exe serve"},
    {"key": "stop_command", "type": "string", "default": None,
     "description": "可选：停止命令。未配置时，仅会回收 fake-ollama 自己启动的进程"},
    {"key": "idle_timeout_seconds", "type": "float", "default": None,
     "description": "可选：空闲超过该秒数后停止 Ollama daemon。通常留空，让 Ollama 自己管理模型卸载"},
    {"key": "startup_timeout_seconds", "type": "float", "default": 60.0,
     "description": "auto_start 后等待 health 变为可用的最长秒数"},
    {"key": "health_path", "type": "string", "default": "/api/version",
     "description": "健康检查路径；Ollama 默认为 /api/version"},
    {"key": "cwd", "type": "string", "default": None,
     "description": "可选：执行 start_command / stop_command 时的工作目录"},
]

LLAMA_CPP_TARGET_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "name", "type": "string", "default": "",
     "description": "可选：target 标识；不填时自动使用 model。用于内部 client 字典、日志和去重，不是全局配置"},
    {"key": "base_url", "type": "string", "default": "http://127.0.0.1:8080",
      "description": "该模型专属 llama.cpp server URL，例如 http://127.0.0.1:21436。每个模型应使用独立端口"},
    {"key": "auth_token", "type": "string", "default": "", "secret": True,
     "description": "可选：llama.cpp --api-key；fake-ollama 调用该 target 时会带上 Bearer / x-api-key"},
    {"key": "model", "type": "string", "default": "", "required": True,
     "autocomplete": "model_names",
      "description": "该 llama.cpp target 服务的唯一模型显示名；一个 target = 一个模型进程。不能含 '@'"},
    {"key": "alias", "type": "string", "default": None,
     "description": "可选：对外公开的别名；填了以 alias 作为 public id。不能含 '@'"},
    {"key": "upstream_id", "type": "string", "default": None,
     "description": "可选：发送给 llama.cpp 的真实 OpenAI model / --alias。不填则使用 model 字段"},
    {"key": "auto_start", "type": "bool", "default": None,
     "description": "可选覆盖全局 llama_cpp_defaults.auto_start；请求到来且 health 检查失败时，是否执行 start_command 启动 llama.cpp"},
    {"key": "start_command", "type": "string", "default": None,
      "description": "可选：启动该模型专属 llama.cpp server 的命令；长驻进程会由 fake-ollama 持有。如果留空，会用下面的 binary_path / model_path 等字段自动拼装命令"},
    {"key": "stop_command", "type": "string", "default": None,
     "description": "可选：停止命令。未配置时，仅会回收 fake-ollama 自己启动的进程"},
    {"key": "binary_path", "type": "string", "default": None,
     "description": "可选：llama-server 可执行文件路径（包含路径或仅命令名）。仅在没填 start_command 时用于自动拼装；不填则继承 llama_cpp_defaults.binary_path 或回退到 'llama-server'"},
    {"key": "runtime_root", "type": "string", "default": None,
     "description": "可选：CUDA runtime 目录（含 cudart64_*.dll），启动时会与 binary 所在目录一起 prepend 到 PATH。Windows 下 llama.cpp CUDA release 的 cudart-llama-bin-win-cuda-XX.X-x64 文件夹"},
    {"key": "model_path", "type": "string", "default": None,
     "description": "可选：传给 --model 的 GGUF 路径；填了它且 start_command 留空时，fake-ollama 会自动拼装启动命令"},
    {"key": "mmproj_path", "type": "string", "default": None,
     "description": "可选：多模态 mmproj 文件路径（--mmproj），仅自动拼装时使用"},
    {"key": "gpu_layers", "type": "int", "default": None,
     "description": "可选：传给 -ngl 的层数（999 = 全部上 GPU）。仅自动拼装时使用；不填继承 llama_cpp_defaults.gpu_layers"},
    {"key": "ctx_size", "type": "int", "default": None,
     "description": "可选：传给 --ctx-size 的 KV cache 总长度（所有 slot 共享）。llama.cpp 会按 parallel 平均切分，每个 slot 实际可用 ctx ≈ ctx_size / parallel——单条请求的 prompt+生成不能超过这个 per-slot 值，否则会返回 400。仅自动拼装时使用；不填继承 llama_cpp_defaults.ctx_size"},
    {"key": "parallel", "type": "int", "default": None,
     "description": "可选：传给 --parallel 的并发槽数。仅自动拼装时使用；不填继承 llama_cpp_defaults.parallel"},
    {"key": "batch_size", "type": "int", "default": None,
     "description": "可选：传给 -b / --batch-size 的逻辑 batch（llama.cpp 默认 2048）。仅自动拼装时使用；不填继承 llama_cpp_defaults.batch_size"},
    {"key": "ubatch_size", "type": "int", "default": None,
     "description": "可选：传给 -ub / --ubatch-size 的物理 micro-batch（llama.cpp 默认 512）。仅自动拼装时使用；不填继承 llama_cpp_defaults.ubatch_size"},
    {"key": "flash_attn", "type": "bool", "default": None,
     "description": "可选：勾选后 true=启用 -fa on / --flash-attn on；false 与不勾选同义（不传 -fa）。不勾选继承 llama_cpp_defaults.flash_attn"},
    {"key": "cache_type_k", "type": "string", "default": None,
     "description": "可选：传给 -ctk 的 KV cache K 类型（如 f16/q8_0/q5_1/q5_0/q4_1/q4_0/iq4_nl）。默认 f16；自动拼装时使用，不填继承 llama_cpp_defaults.cache_type_k"},
    {"key": "cache_type_v", "type": "string", "default": None,
     "description": "可选：传给 -ctv 的 KV cache V 类型（同上）。默认 f16；自动拼装时使用，不填继承 llama_cpp_defaults.cache_type_v"},
    {"key": "extra_args", "type": "string", "default": None,
     "description": "可选：额外原样追加到自动拼装命令末尾的参数串（例如 --jinja --slots）"},
    {"key": "idle_timeout_seconds", "type": "float", "default": None,
     "description": "可选覆盖全局 llama_cpp_defaults.idle_timeout_seconds；空闲超过该秒数后停止 llama.cpp"},
    {"key": "startup_timeout_seconds", "type": "float", "default": None,
     "description": "可选覆盖全局 llama_cpp_defaults.startup_timeout_seconds；auto_start 后等待 health 变为可用的最长秒数"},
    {"key": "health_path", "type": "string", "default": None,
     "description": "可选覆盖全局 llama_cpp_defaults.health_path；健康检查路径"},
    {"key": "cwd", "type": "string", "default": None,
     "description": "可选覆盖全局 llama_cpp_defaults.cwd；执行 start_command / stop_command 时的工作目录"},
    {"key": "max_concurrent_requests", "type": "int", "default": None,
     "description": "可选：fake_ollama 内部限制同时打上游 llama.cpp 的请求数，超出的在内存里 FIFO 排队。留空时若配了 parallel 则默认跟 parallel 一致；0 = 明确不限制"},
    {"key": "request_read_timeout_seconds", "type": "float", "default": None,
     "description": "可选：单独调整 fake_ollama -> llama.cpp 的 read timeout（秒）。留空沿用全局 timeout_seconds；<=0 表示不超时（适合长排队 / 长生成，避免 502 ReadTimeout）"},
]

LLAMA_CPP_DEFAULTS_SCHEMA: List[Dict[str, Any]] = [
    {"key": "auto_start", "type": "bool", "default": False,
     "description": "默认是否在 health 检查失败时执行 target.start_command；target 可覆盖"},
    {"key": "idle_timeout_seconds", "type": "float", "default": None,
     "description": "默认空闲回收秒数；target 可覆盖。留空表示不做 idle 回收"},
    {"key": "startup_timeout_seconds", "type": "float", "default": 120.0,
     "description": "默认启动等待 health 变为可用的最长秒数；target 可覆盖"},
    {"key": "health_path", "type": "string", "default": "/health",
     "description": "默认健康检查路径；target 可覆盖"},
    {"key": "cwd", "type": "string", "default": None,
     "description": "默认执行 start_command / stop_command 时的工作目录；target 可覆盖"},
    {"key": "binary_path", "type": "string", "default": None,
     "description": "默认 llama-server 可执行文件路径；target 留空 binary_path 时会沿用此字段，再回退到 'llama-server'"},
    {"key": "runtime_root", "type": "string", "default": None,
     "description": "默认 CUDA runtime 目录（含 cudart64_*.dll）；target 留空 runtime_root 时沿用此字段。启动时会和 binary 所在目录一起 prepend 到 PATH"},
    {"key": "gpu_layers", "type": "int", "default": None,
     "description": "默认 -ngl 层数；自动拼装命令时使用，target 可覆盖"},
    {"key": "ctx_size", "type": "int", "default": None,
     "description": "默认 --ctx-size 的 KV cache 总长度（所有 slot 共享，llama.cpp 会按 parallel 平均切分）。每个 slot 实际可用 ctx ≈ ctx_size / parallel。自动拼装命令时使用，target 可覆盖"},
    {"key": "parallel", "type": "int", "default": None,
     "description": "默认 --parallel 并发槽数；自动拼装命令时使用，target 可覆盖"},
    {"key": "batch_size", "type": "int", "default": None,
     "description": "默认 -b / --batch-size 逻辑 batch（llama.cpp 默认 2048）；target 可覆盖。不勾选 = 不传该参数、由 llama.cpp 自己决定"},
    {"key": "ubatch_size", "type": "int", "default": None,
     "description": "默认 -ub / --ubatch-size 物理 micro-batch（llama.cpp 默认 512）；target 可覆盖。不勾选 = 不传该参数"},
    {"key": "flash_attn", "type": "bool", "default": None,
     "description": "默认是否启用 -fa on / --flash-attn on。不勾选 = 不传该参数；target 可覆盖"},
    {"key": "cache_type_k", "type": "string", "default": None,
     "description": "默认 -ctk KV cache K 类型（f16/q8_0/q5_1/q5_0/q4_1/q4_0/iq4_nl 等）。留空 = llama.cpp 默认 f16；target 可覆盖"},
    {"key": "cache_type_v", "type": "string", "default": None,
     "description": "默认 -ctv KV cache V 类型（同上）。留空 = llama.cpp 默认 f16；target 可覆盖"},
    {"key": "extra_args", "type": "string", "default": None,
     "description": "默认追加到自动拼装命令末尾的参数串；target 可覆盖"},
    {"key": "max_concurrent_requests", "type": "int", "default": None,
     "description": "默认 fake_ollama 内部限制同时打上游 llama.cpp 的请求数；target 可覆盖。留空且配了 parallel 时会自动取 parallel；0 = 不限制"},
    {"key": "request_read_timeout_seconds", "type": "float", "default": None,
     "description": "默认 fake_ollama -> llama.cpp 的 read timeout 覆盖；target 可覆盖。留空沿用全局 timeout_seconds；<=0 = 不超时"},
]

COMFYUI_TARGET_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "name", "type": "string", "default": "",
     "description": "可选：唯一 target 名；不填时默认使用 model / alias。用于内部路由、日志和去重"},
    {"key": "base_url", "type": "string", "default": "http://127.0.0.1:8188",
     "description": "ComfyUI server 的 base URL，例如 http://127.0.0.1:21480"},
    {"key": "auth_token", "type": "string", "default": "", "secret": True,
     "description": "可选：调用 ComfyUI 时同时以 Authorization: Bearer 和 x-api-key 发送的 token"},
    {"key": "vram_runtime_group", "type": "string", "default": None,
     "description": "可选：显存运行时分组；留空时按 base_url 自动归组。共享同一 ComfyUI 进程的 target 必须处于同一组"},
    {"key": "gpu_device", "type": "string", "default": "0",
     "description": "该 ComfyUI 运行时使用的 GPU 设备标识；参与显存运行时分组"},
    {"key": "model", "type": "string", "default": "z-image-turbo", "required": True,
     "autocomplete": "model_names",
     "description": "该 ComfyUI workflow target 对外提供的图片模型显示名"},
    {"key": "alias", "type": "string", "default": None,
     "description": "可选：source 级别别名；填了会作为默认公开名。不能包含 '@'"},
    {"key": "upstream_id", "type": "string", "default": None,
     "description": "可选：内部模型 ID，用于显存占用记录；不填则使用 model"},
    {"key": "auto_start", "type": "bool", "default": False,
     "description": "health 检查失败时，是否执行 start_command 启动 ComfyUI"},
    {"key": "start_command", "type": "string", "default": None,
     "description": "启动 ComfyUI 的命令"},
    {"key": "stop_command", "type": "string", "default": None,
     "description": "可选：停止 ComfyUI 的命令"},
    {"key": "idle_timeout_seconds", "type": "float", "default": None,
     "description": "由 fake-ollama 启动的 ComfyUI 空闲超过该秒数后自动停止；留空表示不回收"},
    {"key": "startup_timeout_seconds", "type": "float", "default": 120.0,
     "description": "auto_start 后等待 ComfyUI health 变为可用的最长秒数"},
    {"key": "health_path", "type": "string", "default": "/system_stats",
     "description": "ComfyUI 健康检查路径；默认 /system_stats"},
    {"key": "cwd", "type": "string", "default": None,
     "description": "执行 start_command / stop_command 时使用的工作目录"},
    {"key": "preset", "type": "string", "default": "z_image_turbo",
     "description": "Declarative workflow preset：z_image_turbo、qwen_image_edit_aio、sensenova_u1、joyai_echo、minimax_h3；提供自定义 workflow path + bindings 时使用 custom/comfyui_api。"},
    {"key": "bindings", "type": "json", "default": None,
     "description": "Optional JSON mapping of request fields to workflow node inputs, keyed by mode: t2i, i2i, video, i2v, fl2va, l2va."},
    {"key": "static_inputs", "type": "json", "default": None,
     "description": "Optional JSON mapping of fixed workflow node inputs, keyed by mode: t2i, i2i, video, i2v, fl2va, l2va."},
    {"key": "text_to_image_workflow_path", "type": "string", "default": None,
     "description": "可选：/v1/images/generations 使用的 ComfyUI API-format workflow JSON；留空使用内置 Z-Image-Turbo text-to-image workflow"},
    {"key": "image_to_image_workflow_path", "type": "string", "default": None,
     "description": "可选：/v1/images/edits 使用的 ComfyUI API-format workflow JSON；留空使用内置 Z-Image-Turbo image-to-image workflow"},
    {"key": "video_workflow_path", "type": "string", "default": None,
     "description": "Optional ComfyUI API-format workflow JSON for /v1/videos/generations text-to-video."},
    {"key": "image_to_video_workflow_path", "type": "string", "default": None,
     "description": "Optional ComfyUI API-format workflow JSON for /v1/videos/generations image-to-video."},
    {"key": "first_last_to_video_workflow_path", "type": "string", "default": None,
     "description": "Optional ComfyUI API-format workflow JSON for first+last-frame video generation (FL2VA)."},
    {"key": "last_to_video_workflow_path", "type": "string", "default": None,
     "description": "Optional ComfyUI API-format workflow JSON for last-frame video generation (L2VA)."},
    {"key": "context_ir_profile", "type": "string", "default": None,
     "description": "可选：生成视频前使用的 h3_context_ir_profiles 名称；留空表示不做 Prompt 自动增强"},
    {"key": "context_ir_prompt_mode", "type": "string", "default": "auto",
     "description": "raw=原样传递；auto=普通 Prompt 自动增强、已结构化 Prompt 直通；enhance=始终重写"},
    {"key": "diffusion_model", "type": "string", "default": "z-image-turbo-fp8-e4m3fn.safetensors",
     "description": "ComfyUI models/diffusion_models 下的 diffusion / UNet 模型文件名"},
    {"key": "diffusion_weight_dtype", "type": "string", "default": "default",
     "description": "传给 UNETLoader 的 weight_dtype；通常保持 default"},
    {"key": "text_encoder_model", "type": "string", "default": "qwen_3_4b_fp4_mixed.safetensors",
     "description": "ComfyUI models/text_encoders 下的文本编码器文件名"},
    {"key": "text_encoder_type", "type": "string", "default": "lumina2",
     "description": "Z-Image Qwen3 文本编码器的 CLIPLoader type；默认 lumina2"},
    {"key": "text_encoder_device", "type": "string", "default": "default",
     "description": "CLIPLoader device；default 表示按 ComfyUI 默认策略，cpu 表示放 CPU"},
    {"key": "vae_model", "type": "string", "default": "ae.safetensors",
     "description": "ComfyUI models/vae 下的 VAE 文件名"},
    {"key": "min_width", "type": "int", "default": 8,
     "description": "workflow 接受的最小宽度"},
    {"key": "default_width", "type": "int", "default": 1024,
     "description": "默认图片宽度；客户端可用 size 或 width 覆盖"},
    {"key": "max_width", "type": "int", "default": None,
     "description": "workflow 接受的最大宽度；留空表示不额外限制"},
    {"key": "width_modulo", "type": "int", "default": 8,
     "description": "宽度必须能被该值整除"},
    {"key": "min_height", "type": "int", "default": 8,
     "description": "workflow 接受的最小高度"},
    {"key": "default_height", "type": "int", "default": 1024,
     "description": "默认图片高度；客户端可用 size 或 height 覆盖"},
    {"key": "max_height", "type": "int", "default": None,
     "description": "workflow 接受的最大高度；留空表示不额外限制"},
    {"key": "height_modulo", "type": "int", "default": 8,
     "description": "高度必须能被该值整除"},
    {"key": "default_steps", "type": "int", "default": 8,
     "description": "默认 Z-Image-Turbo 采样步数"},
    {"key": "default_cfg", "type": "float", "default": 1.0,
     "description": "默认 CFG；Turbo 模型通常保持 1"},
    {"key": "default_sampler_name", "type": "string", "default": "res_multistep",
     "description": "默认 KSampler sampler_name"},
    {"key": "default_scheduler", "type": "string", "default": "simple",
     "description": "默认 KSampler scheduler"},
    {"key": "default_denoise", "type": "float", "default": 1.0,
     "description": "默认 text-to-image denoise；一般为 1.0"},
    {"key": "default_edit_denoise", "type": "float", "default": 0.25,
     "description": "默认 image-to-image / image edit denoise；越高越偏向重绘"},
    {"key": "default_shift", "type": "float", "default": 3.0,
     "description": "ModelSamplingAuraFlow shift 参数"},
    {"key": "min_num_frames", "type": "int", "default": 1,
     "description": "视频 workflow 允许的最小帧数"},
    {"key": "default_num_frames", "type": "int", "default": 121,
     "description": "视频生成默认帧数；Playground 参数 schema 与 API 缺省值都从这里读取"},
    {"key": "max_num_frames", "type": "int", "default": 241,
     "description": "视频 workflow 允许的最大帧数"},
    {"key": "num_frames_offset", "type": "int", "default": 0,
     "description": "帧数合法性偏移；与 num_frames_modulo 共同约束 (frames-offset)%modulo=0"},
    {"key": "num_frames_modulo", "type": "int", "default": 1,
     "description": "帧数合法性步进/模数"},
    {"key": "min_frame_rate", "type": "float", "default": 1.0,
     "description": "视频 workflow 允许的最小 FPS"},
    {"key": "default_frame_rate", "type": "float", "default": 24.0,
     "description": "视频生成默认 FPS"},
    {"key": "max_frame_rate", "type": "float", "default": 120.0,
     "description": "视频 workflow 允许的最大 FPS"},
    {"key": "default_prefetch_count", "type": "int", "default": 1,
     "description": "视频节点默认层预取数量"},
    {"key": "max_prefetch_count", "type": "int", "default": 48,
     "description": "视频节点允许的最大层预取数量"},
    {"key": "default_enable_tile", "type": "bool", "default": False,
     "description": "视频 workflow 默认是否启用分块 VAE 解码；可由请求参数 enable_tile 覆盖"},
    {"key": "default_enable_streaming", "type": "bool", "default": False,
     "description": "视频 workflow 默认是否启用模型内部流式加载；不是 HTTP streaming"},
    {"key": "seed_mode", "type": "string", "default": "random",
     "description": "随机种子模式：random=每次随机（默认）；fixed=固定用下面的 seed；increment=从 seed 开始每次递增（按本次出图张数 n 递增）。请求体显式传 seed 时一律以请求为准"},
    {"key": "seed", "type": "int", "default": 0,
     "description": "seed_mode=fixed/increment 时的基准种子；random 模式下忽略"},
    {"key": "max_batch_size", "type": "int", "default": 4,
     "description": "允许的 OpenAI Images 参数 n 最大值"},
    {"key": "max_reference_images", "type": "int", "default": None,
     "description": "可选：单次 ComfyUI 请求允许上传的参考图最大张数；适用于 bindings 中用 images 绑定 IMAGE batch 输入的工作流"},
    {"key": "output_prefix", "type": "string", "default": "fake_ollama/z-image-turbo",
     "description": "ComfyUI SaveImage 节点使用的 filename_prefix"},
    {"key": "image_upscale_method", "type": "string", "default": "lanczos",
     "description": "image edit workflow 中 ImageScale 使用的缩放方法"},
    {"key": "image_crop", "type": "string", "default": "center",
     "description": "image edit workflow 中 ImageScale 使用的裁剪模式"},
    {"key": "prompt_timeout_seconds", "type": "float", "default": 600.0,
     "description": "等待 ComfyUI 队列中 prompt 完成的最长秒数"},
    {"key": "poll_interval_seconds", "type": "float", "default": 0.5,
     "description": "轮询 ComfyUI history 的间隔秒数"},
]

H3_CONTEXT_IR_PROVIDER_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "name", "type": "string", "default": "", "required": True,
     "description": "Profile 内的 provider 名称；API 和 Playground 用它切换模型"},
    {"key": "model", "type": "string", "default": "", "required": True,
     "autocomplete": "model_names",
     "description": "现有 source 对外声明的模型名（alias 或 name）"},
    {"key": "target", "type": "string", "default": "", "required": True,
     "autocomplete": "source_names",
     "description": "现有 Ollama / llama.cpp / Generic OpenAI / OpenAI / Anthropic target 名"},
    {"key": "modalities", "type": "string_list", "default": ["text"],
     "description": "支持的输入：text；视觉模型填写 text,image"},
    {"key": "json_mode", "type": "bool", "default": False,
     "description": "是否向 OpenAI-compatible provider 发送 response_format=json_object"},
]

H3_CONTEXT_IR_PROFILE_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "name", "type": "string", "default": "", "required": True,
     "description": "唯一 profile 名；默认 Playground 模型 ID 为 h3-context-ir-fake@<name>"},
    {"key": "alias", "type": "string", "default": None,
     "description": "可选：Playground 中显示的虚拟模型 ID"},
    {"key": "enabled", "type": "bool", "default": True,
     "description": "关闭后 API 与视频生成都不会调用该 profile"},
    {"key": "playground_visible", "type": "bool", "default": True,
     "description": "是否在 /playground/api/models 中显示为可单独测试的虚拟模型"},
    {"key": "allow_compatible_models", "type": "bool", "default": True,
     "description": "允许从当前接口 exposed_models 中自选其它 completion 模型；带图兼容性按 vision capability 判断"},
    {"key": "allow_external_api", "type": "bool", "default": False,
     "description": "允许 Playground 临时输入第三方 OpenAI/Anthropic-compatible URL 与 token；该能力可访问任意网络地址，建议只在受信任的本机 Playground 开启"},
    {"key": "providers", "type": "object_list", "default": [],
     "item_schema": H3_CONTEXT_IR_PROVIDER_ITEM_SCHEMA,
     "nav_label_keys": ["name", "model"],
     "description": "可切换的规划模型；provider 复用已有 target 的鉴权、生命周期与显存协调"},
    {"key": "default_text_provider", "type": "string", "default": None,
     "description": "纯文字请求默认 provider；留空取第一个"},
    {"key": "default_multimodal_provider", "type": "string", "default": None,
     "description": "附图请求默认 provider；留空取第一个声明 image 的 provider"},
    {"key": "temperature", "type": "float", "default": 0.2,
     "description": "规划模型温度，建议保持较低以提高 JSON 稳定性"},
    {"key": "max_output_tokens", "type": "int", "default": 4096,
     "description": "单次规划最大输出 token"},
    {"key": "max_attempts", "type": "int", "default": 2,
     "description": "JSON/时间线校验失败后的总尝试次数，范围 1-3"},
    {"key": "failure_mode", "type": "string", "default": "fallback",
     "description": "fallback=使用保留原意的单镜头结构继续；error=返回错误"},
    {"key": "default_duration_seconds", "type": "float", "default": 5.0,
     "description": "独立调用 Context-IR 未指定时长时的默认值，范围 4-15 秒"},
]

MODEL_PROFILE_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "model", "type": "string", "default": "", "required": True,
     "autocomplete": "model_names",
     "description": "模型名（裸名应用于所有 target；与 target 一起拼出最终 key 'model@target' 仅覆盖该 target）"},
    {"key": "target", "type": "string", "default": "",
     "autocomplete": "source_names",
     "description": "可选：source 名字（anthropic_upstreams / openai_upstreams / ollama_targets / llama_cpp_targets / comfyui_targets 中的某个 name）。留空 = 对该 model 名所有 target 生效"},
    {"key": "capabilities", "type": "string_list",
     "default": ["completion", "tools", "vision"],
     "description": "模型对外声明的 capability 标签，用于 /api/show、Anthropic 模型能力映射和 /playground/api/models 功能发现，不会自动给模型增加能力。/api/show 只报告 Ollama 标准的 completion/tools/vision，媒体标签只在 Playground 专用发现接口中返回。常用值：completion=文本/聊天补全，tools=支持 tool calling，vision=聊天接口可接收图片输入，video_understanding=/v1/chat/completions 接收视频分析，image_generation=/v1/images/generations 图片生成，image_edit=/v1/images/edits 图片编辑，video_generation=/v1/videos/generations 视频或图生视频。聊天模型通常至少含 completion；纯媒体 workflow 不需要 completion"},
    {"key": "context_length", "type": "int", "default": 200000,
     "description": "上下文 token 上限（输入 + 输出）。① 通过 /api/show.model_info、Anthropic /v1/models 和 /playground/api/models 报告为模型的 context window；② fake-ollama 在转发前预检：估算 input token + max_tokens 超过此值时返回 400（可用环境变量 FAKE_OLLAMA_ENFORCE_CONTEXT_LIMIT=false 关闭，OpenAI 透传路径不强制）。配置文件中亦可写作 num_ctx。对于 llama.cpp / Ollama 反向代理，建议设为 ≤ 上游 per-slot 实际容量（ctx_size / parallel），否则预检通过但上游会拒"},
    {"key": "max_output_tokens", "type": "int", "default": None,
     "description": "可选：单条响应的输出 token 数；同时作为 max_tokens 的下限与上限——配了之后，无论客户端传什么 max_tokens / num_predict，最终都会被强制设为该值（防止 VS Code Copilot 等客户端的小默认值导致 finish_reason=length 被整段拒）。受 context_length 预检约束"},
    {"key": "estimated_vram_gb", "type": "float", "default": None,
     "description": "可选：该模型加载后预计占用的 GPU 显存（GB）。本地 Ollama / llama.cpp / ComfyUI target 会使用它做启动前预检、空闲模型回收和 Dashboard 展示"},
    {"key": "request_vram_headroom_gb", "type": "float", "default": 0.0,
     "description": "模型已驻留时，单次推理仍需保留的瞬时显存余量（GB）；媒体工作流会按分辨率、帧数和原生 batch 放大"},
    {"key": "min_free_vram_gb", "type": "float", "default": 0.0,
     "description": "请求准入及推理观测的空闲显存下限（GB）；运行中越线会记录并在 prompt 完成后清理，不会抢先中断 ComfyUI DynamicVRAM；真实 OOM 仍按 workflow 失败返回"},
    {"key": "vram_cleanup_policy", "type": "string", "default": "keep",
     "description": "keep=始终复用；adaptive=余量不足时仅卸载 GPU 权重；unload=每次请求前卸载 GPU 权重"},
    {"key": "exclusive_gpu", "type": "bool", "default": False,
     "description": "是否在完整推理期间独占 GPU 执行租约；适合高波动视频工作流"},
    {"key": "estimated_memory_gb", "type": "float", "default": None,
     "description": "可选：该模型加载后预计占用的系统内存 / RAM（GB）。部分模型（如 SenseNova、JoyAI 这类会把一部分计算 offload 到内存的 workflow）除显存外还需占用大量主机内存。本地 Ollama / llama.cpp / ComfyUI target 会使用它做启动前预检、空闲模型回收和 Dashboard 展示，逻辑与 estimated_vram_gb 一致"},
    {"key": "thinking_mode", "type": "string", "default": "auto",
     "description": "auto / enabled / disabled；控制是否注入 thinking 字段"},
    {"key": "thinking_budget_tokens", "type": "int", "default": 1024,
     "description": "thinking=enabled 时的预算（DeepSeek 会忽略）"},
    {"key": "show_thinking", "type": "bool", "default": True,
     "description": "是否把上游 thinking 透传给客户端（<think> 块）"},
]

EXPOSURE_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "model", "type": "string", "default": "", "required": True,
     "autocomplete": "model_names",
     "description": "source 中某个模型的公开名（alias 或 name）"},
    {"key": "target", "type": "string", "default": "", "required": True,
     "autocomplete": "source_names",
     "description": "提供该模型的 source 名字（anthropic_upstreams / openai_upstreams / ollama_targets / llama_cpp_targets / comfyui_targets 中的某个 name）"},
    {"key": "alias", "type": "string", "default": None,
     "description": "可选：该接口上对外公开的 public id。不填则默认为 'model@target'。同一接口内 alias 不可重复"},
]

OLLAMA_INTERFACE_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "name", "type": "string", "default": "", "required": True,
     "description": "interface 唯一名字"},
    {"key": "host", "type": "string", "default": "127.0.0.1",
     "description": "监听地址。建议保持 127.0.0.1；需要外部访问请配合 Nginx 等反代"},
    {"key": "port", "type": "int", "default": 21434, "required": True,
     "description": "监听端口；所有 interface/admin/dashboard 不可冲突"},
    {"key": "access_tokens", "type": "string_list", "default": [],
     "secret_each": True, "generate_each": True,
     "description": "该 interface 的访问 token 池。留空 = 不需要鉴权"},
    {"key": "exposed_models", "type": "object_list", "default": [],
     "item_schema": EXPOSURE_ITEM_SCHEMA,
     "nav_label_keys": ["alias", "model"],
     "description": "该 interface 外露的模型。需明确填写 model + target；可选 alias 改变公开 id"},
]

# Keep the two schemas independent: their actual default ports differ.
API_INTERFACE_ITEM_SCHEMA: List[Dict[str, Any]] = deepcopy(
    OLLAMA_INTERFACE_ITEM_SCHEMA
)

CONFIG_SCHEMA: List[Dict[str, Any]] = [
  # ---- Model sources (远端 + 本机) -----------------------------------------
  {"key": "anthropic_upstreams", "type": "object_list", "default": [], "group": "model_sources_remote",
   "item_schema": UPSTREAM_ITEM_SCHEMA,
   "detect_models": "anthropic",
   "nav_label_keys": ["name"],
   "description": "远端 Anthropic 兼容上游"},

  {"key": "openai_upstreams", "type": "object_list", "default": [], "group": "model_sources_remote",
   "item_schema": OPENAI_UPSTREAM_ITEM_SCHEMA,
   "detect_models": "openai",
   "nav_label_keys": ["name"],
   "description": "OpenAI 兼容远端上游（OpenAI / DeepSeek / Together / Groq 等）"},

  {"key": "generic_openai_targets", "type": "object_list", "default": [], "group": "model_sources_generic_openai",
   "item_schema": LOCAL_OPENAI_TARGET_ITEM_SCHEMA,
   "detect_models": "openai",
   "nav_label_keys": ["name"],
   "description": "Lifecycle-managed generic OpenAI-compatible servers such as vLLM, SGLang, TGI, or adapters."},

  {"key": "ollama_targets", "type": "object_list", "default": [], "group": "model_sources_ollama",
   "item_schema": OLLAMA_TARGET_ITEM_SCHEMA,
   "detect_models": "ollama",
   "nav_label_keys": ["name"],
   "description": "本机或远端 Ollama 服务"},
  {"key": "llama_cpp_defaults", "type": "object", "default": {}, "group": "model_sources_llama_cpp",
   "item_schema": LLAMA_CPP_DEFAULTS_SCHEMA,
   "nav_label": "Defaults",
   "description": "llama.cpp targets 的全局默认值；每个 target 勾选同名字段后可覆盖"},
  {"key": "llama_cpp_targets", "type": "object_list", "default": [], "group": "model_sources_llama_cpp",
   "item_schema": LLAMA_CPP_TARGET_ITEM_SCHEMA,
   "detect_models": "llama_cpp",
   "nav_label_keys": ["name", "model"],
    "description": "llama.cpp server（OpenAI 兼容）；一个 target = 一个模型 / 进程 / 端口"},

  {"key": "comfyui_targets", "type": "object_list", "default": [], "group": "model_sources_comfyui",
   "item_schema": COMFYUI_TARGET_ITEM_SCHEMA,
   "nav_label_keys": ["name", "model"],
   "description": "由 ComfyUI workflow 承载的图片模型来源，用于 /v1/images/generations 和 /v1/images/edits"},

  {"key": "h3_context_ir_profiles", "type": "object_list", "default": [], "group": "h3_context_ir",
   "item_schema": H3_CONTEXT_IR_PROFILE_ITEM_SCHEMA,
   "nav_label_keys": ["alias", "name"],
   "description": "H3-Context-IR-fake 编排：引用现有本地或第三方模型，产出并校验官方三字段 Prompt"},

  # ---- Interfaces ---------------------------------------------------------
  {"key": "advertised_version", "type": "string", "default": "0.6.4", "group": "interface_ollama",
   "description": "仅用于 Ollama 接口的 GET /api/version 返回值"},
  {"key": "ollama_interfaces", "type": "object_list",
   "default": [{"name": "ollama", "host": "127.0.0.1", "port": 21434, "access_tokens": [], "exposed_models": []}],
   "group": "interface_ollama",
   "item_schema": OLLAMA_INTERFACE_ITEM_SCHEMA,
   "nav_label_keys": ["name"],
   "description": "Ollama 兼容接口数组。每个 entry 独立的 host/port/access_tokens/exposed_models。服务 /api/* 与 /v1/chat/completions"},

  {"key": "api_interfaces", "type": "object_list", "default": [], "group": "interface_api",
   "item_schema": API_INTERFACE_ITEM_SCHEMA,
   "nav_label_keys": ["name"],
   "description": "API 接口数组（Anthropic /v1/messages + OpenAI /v1/chat/completions + /v1/images/* + /v1/models）。每个 entry 独立的 host/port/access_tokens/exposed_models"},

  # ---- Runtime & profiles --------------------------------------------------
  {"key": "default_max_tokens", "type": "int", "default": 4096, "group": "runtime",
   "description": "缺省的 max_tokens / num_predict"},
  {"key": "timeout_seconds", "type": "float", "default": 300.0, "group": "runtime",
   "description": "所有出站 HTTP 调用的超时"},
  {"key": "use_system_proxy", "type": "bool", "default": False, "group": "runtime",
   "description": "所有出站 HTTP 调用是否走系统代理"},
  {"key": "enforce_context_limit", "type": "bool", "default": True, "group": "runtime",
   "description": "在带 context_length 的请求上，估算输入+max_tokens 超限时直接 400"},

  {"key": "model_profiles", "type": "object_list", "default": [], "group": "runtime",
   "item_schema": MODEL_PROFILE_ITEM_SCHEMA,
   "nav_label_keys": ["model", "target"],
   "nav_label_join": "@",
   "description": "模型 capabilities / 上下文 / 思维链设置。每项写 model（必填）和可选 target，两者拼起来作为最终 key：填 target 时为 'model@target' 仅覆盖该 target；不填 target 时为裸 'model' 适用于所有 target"},

  {"key": "dashboard_enabled", "type": "bool", "default": True, "group": "dashboard",
   "description": "是否启用 Dashboard；启用后会在独立 listener 上挂载 /dashboard"},
  {"key": "dashboard_host", "type": "string", "default": "127.0.0.1", "group": "dashboard",
   "description": "Dashboard listener bind address。除非前面有 trusted proxy，否则建议保持 127.0.0.1"},
  {"key": "dashboard_port", "type": "int", "default": 21432, "group": "dashboard",
   "description": "Dashboard listener port；必须与 internal、external 和 admin ports 不同"},
  {"key": "dashboard_sample_interval_seconds", "type": "float", "default": 10.0, "group": "dashboard",
   "description": "Runtime metrics 采样间隔秒数"},
  {"key": "dashboard_retention_seconds", "type": "float", "default": 604800.0, "group": "dashboard",
   "description": "Dashboard history 保留秒数"},
  {"key": "dashboard_data_path", "type": "string", "default": "logs/dashboard_history.json", "group": "dashboard",
   "description": "持久化 Dashboard history 的 JSON 文件路径；留空表示不写入文件"},
  {"key": "dashboard_model_reclaim_enabled", "type": "bool", "default": False, "group": "dashboard",
   "description": "是否允许在 Dashboard 的 Current Models 表中手动释放符合条件的 idle local models"},
  {"key": "dashboard_reclaim_idle_seconds", "type": "float", "default": 20.0, "group": "dashboard",
   "description": "用户在 Dashboard 点击关闭按钮时所需的最小 idle 秒数。和自动 LRU 回收的 60s 阈值独立；用户判断更宽松，默认 20s。"},
  {"key": "vram_low_free_reclaim_enabled", "type": "bool", "default": True, "group": "dashboard",
   "description": "是否启用 periodic low-free-VRAM check，并释放符合条件的 idle local models"},
  {"key": "vram_low_free_threshold_mib", "type": "float", "default": 200.0, "group": "dashboard",
   "description": "当 free GPU VRAM 低于该 MiB threshold 时，可释放符合条件的 idle models"},
  {"key": "memory_low_free_reclaim_enabled", "type": "bool", "default": True, "group": "dashboard",
   "description": "是否启用 periodic low-free-RAM check，并释放声明了 estimated_memory_gb 的 idle local models"},
  {"key": "memory_low_free_threshold_mib", "type": "float", "default": 2048.0, "group": "dashboard",
   "description": "当可用系统内存低于该 MiB threshold 时，可释放符合条件的 idle models（仅释放声明了 estimated_memory_gb 的模型）"},

  {"key": "playground_enabled", "type": "bool", "default": False, "group": "playground",
   "description": "模型调试页的实际启用开关。修改后请点击顶部 Save & Reload，再重启 fake-ollama"},
  {"key": "playground_host", "type": "string", "default": "127.0.0.1", "group": "playground",
   "description": "模型调试页监听地址。保持 127.0.0.1 仅限本机；手机或其他局域网设备访问时可设为 0.0.0.0。API key 会由浏览器发送到该地址，请勿直接暴露到互联网"},
  {"key": "playground_port", "type": "int", "default": 21431, "group": "playground",
   "description": "模型调试页独立监听端口；必须与所有 interface、admin 和 dashboard 端口不同。启用后访问 /playground/"},

  {"key": "admin_enabled", "type": "bool", "default": True, "group": "admin",
   "description": "是否启用本 /admin 编辑器（关闭后需手动改 config.json）"},
  {"key": "admin_host", "type": "string", "default": "127.0.0.1", "group": "admin",
   "description": "Admin UI 监听地址。强烈建议保持 127.0.0.1；/admin 没有内置鉴权"},
  {"key": "admin_port", "type": "int", "default": 21433, "group": "admin",
   "description": "Admin UI 独立监听端口。设为 null 才会把 /admin 挂回 internal 端口（旧行为，不推荐）"},
]


def _sync_ui_schema_defaults() -> None:
    """Use the validated config models as the single source of defaults.

    The form schema still owns labels, descriptions and widget types, but it
    must not maintain a second hand-written set of values that can drift from
    Pydantic. Required model fields have no runtime default and keep their UI
    placeholder (usually an empty string).
    """

    item_schemas = (
        (MODEL_ENTRY_ITEM_SCHEMA, ModelEntry),
        (UPSTREAM_ITEM_SCHEMA, AnthropicUpstream),
        (OPENAI_UPSTREAM_ITEM_SCHEMA, OpenAIUpstream),
        (LOCAL_OPENAI_TARGET_ITEM_SCHEMA, GenericOpenAITarget),
        (OLLAMA_TARGET_ITEM_SCHEMA, OllamaTarget),
        (LLAMA_CPP_TARGET_ITEM_SCHEMA, LlamaCppTarget),
        (LLAMA_CPP_DEFAULTS_SCHEMA, LlamaCppDefaults),
        (COMFYUI_TARGET_ITEM_SCHEMA, ComfyUITarget),
        (H3_CONTEXT_IR_PROVIDER_ITEM_SCHEMA, H3ContextIRProvider),
        (H3_CONTEXT_IR_PROFILE_ITEM_SCHEMA, H3ContextIRProfile),
        (EXPOSURE_ITEM_SCHEMA, ExposureEntry),
        (OLLAMA_INTERFACE_ITEM_SCHEMA, OllamaInterface),
        (API_INTERFACE_ITEM_SCHEMA, ApiInterface),
    )
    for schema, model_type in item_schemas:
        for spec in schema:
            model_field = model_type.model_fields.get(spec["key"])
            if model_field is None or model_field.is_required():
                continue
            spec["default"] = deepcopy(
                model_field.get_default(call_default_factory=True)
            )

    settings_defaults = Settings().model_dump()
    for spec in CONFIG_SCHEMA:
        if spec["key"] in settings_defaults:
            spec["default"] = deepcopy(settings_defaults[spec["key"]])


_sync_ui_schema_defaults()

GROUP_LABELS: List[Dict[str, str]] = [
  {"key": "model_sources_remote", "label": "Remote Sources", "hint": "远端模型来源：Anthropic / OpenAI 兼容上游", "section": "model_sources", "section_label": "Model Sources", "section_hint": "配置可被路由的模型来源（source / target）"},
  {"key": "model_sources_generic_openai", "label": "Generic OpenAI Sources", "hint": "Lifecycle-managed generic OpenAI-compatible servers such as vLLM, SGLang, TGI, or adapters", "section": "model_sources", "section_label": "Model Sources", "section_hint": "配置可被路由的模型来源（source / target）"},
  {"key": "model_sources_ollama", "label": "Ollama Sources", "hint": "本机或内网 Ollama 服务", "section": "model_sources", "section_label": "Model Sources", "section_hint": "配置可被路由的模型来源（source / target）"},
  {"key": "model_sources_llama_cpp", "label": "llama.cpp Sources", "hint": "本机 llama.cpp server 进程", "section": "model_sources", "section_label": "Model Sources", "section_hint": "配置可被路由的模型来源（source / target）"},
  {"key": "model_sources_comfyui", "label": "ComfyUI Sources", "hint": "本地 ComfyUI image workflow target", "section": "model_sources", "section_label": "Model Sources", "section_hint": "配置可被路由的模型来源（source / target）"},
  {"key": "h3_context_ir", "label": "H3 Context-IR-fake", "hint": "可切换本地文本/VL模型或第三方 API 的 H3 Prompt 规划与校验流水线", "section": "h3_context_ir", "section_label": "H3 Context-IR-fake", "section_hint": "Playground 可独立预览；ComfyUI 视频 target 可在生成前串行调用"},
  {"key": "interface_ollama", "label": "Ollama Interface", "hint": "Ollama 兼容接口（/api/* 与 /v1/chat/completions）。每个 entry 各自选择暴露哪些模型", "section": "interfaces", "section_label": "Interfaces", "section_hint": "对用户暴露的接口；每个 entry 都是独立的 host/port/access_tokens/exposed_models"},
  {"key": "interface_api", "label": "API Interface", "hint": "Anthropic /v1/messages + OpenAI /v1/chat/completions + /v1/images/* + /v1/models。每个 entry 各自选择暴露哪些模型", "section": "interfaces", "section_label": "Interfaces", "section_hint": "对用户暴露的接口；每个 entry 都是独立的 host/port/access_tokens/exposed_models"},
  {"key": "runtime", "label": "Runtime & Profiles", "hint": "运行时缺省值、出站网络与每模型 capability / thinking profile", "section": "runtime", "section_label": "Runtime", "section_hint": "跨所有 target / 接口共享的运行时设置"},
  {"key": "dashboard", "label": "Dashboard", "hint": "Runtime graphs and the low-VRAM safety monitor", "section": "dashboard", "section_label": "Dashboard", "section_hint": "Memory, VRAM, and loaded local model telemetry"},
  {"key": "playground", "label": "Model Playground", "hint": "轻量流式模型调试页（不保存历史记录或会话）", "section": "playground", "section_label": "Model Playground", "section_hint": "修改监听开关或端口后，请保存配置并重启进程"},
  {"key": "admin", "label": "Admin UI", "hint": "配置页面自身的开关与监听地址（无内置鉴权）", "section": "admin", "section_label": "Admin UI", "section_hint": "仅影响 /admin 配置页面本身"},
]


# ---------------------------------------------------------------------------
# HTML / JS bundle
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_HTML = (_STATIC_DIR / "admin.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings_to_dict(s: Settings) -> Dict[str, Any]:
    data = s.model_dump()
    data.pop("config_path", None)
    defaults = data.get("llama_cpp_defaults")
    if isinstance(defaults, dict):
        for key in list(defaults.keys()):
            if defaults[key] is None:
                defaults.pop(key)
    for target in data.get("generic_openai_targets", []):
        if not isinstance(target, dict):
            continue
        for key in [
            "start_command",
            "stop_command",
            "idle_timeout_seconds",
            "cwd",
            "max_concurrent_requests",
            "request_read_timeout_seconds",
        ]:
            if target.get(key) is None:
                target.pop(key, None)
    for target in data.get("llama_cpp_targets", []):
        if not isinstance(target, dict):
            continue
        for key in [
            "upstream_id",
            "alias",
            "auto_start",
            "start_command",
            "stop_command",
            "idle_timeout_seconds",
            "startup_timeout_seconds",
            "health_path",
            "cwd",
            "binary_path",
            "runtime_root",
            "model_path",
            "mmproj_path",
            "gpu_layers",
            "ctx_size",
            "parallel",
            "batch_size",
            "ubatch_size",
            "flash_attn",
            "cache_type_k",
            "cache_type_v",
            "extra_args",
        ]:
            if target.get(key) is None:
                target.pop(key, None)
        if target.get("name") == target.get("model"):
            target.pop("name", None)
    for target in data.get("comfyui_targets", []):
        if not isinstance(target, dict):
            continue
        for key in [
            "upstream_id",
            "alias",
            "start_command",
            "stop_command",
            "idle_timeout_seconds",
            "cwd",
            "text_to_image_workflow_path",
            "image_to_image_workflow_path",
            "video_workflow_path",
            "image_to_video_workflow_path",
            "first_last_to_video_workflow_path",
            "last_to_video_workflow_path",
            "vram_runtime_group",
        ]:
            if target.get(key) is None:
                target.pop(key, None)
        if target.get("name") == target.get("model"):
            target.pop("name", None)
    data["_path"] = s.config_path or ""
    return data


def _resolve_save_path(s: Settings) -> Path:
    if s.config_path:
        return Path(s.config_path)
    return Path("config.json")


def _ollama_client_matches(
    client: OllamaClient,
    *,
    settings: Settings,
    target: Any,
) -> bool:
    return (
        getattr(client, "_base", None) == target.base_url.rstrip("/")
        and getattr(client, "_timeout", None) == settings.timeout_seconds
        and getattr(client, "_trust_env", None) == settings.use_system_proxy
        and getattr(client, "_auto_start", None) == target.auto_start
        and getattr(client, "_start_command", None) == target.start_command
        and getattr(client, "_stop_command", None) == target.stop_command
        and getattr(client, "_idle_timeout", None) == target.idle_timeout_seconds
        and getattr(client, "_startup_timeout", None) == target.startup_timeout_seconds
        and getattr(client, "_health_path", None) == target.health_path
        and getattr(client, "_cwd", None) == target.cwd
    )


def _llama_cpp_client_matches(
    client: LlamaCppClient,
    *,
    settings: Settings,
    target: Any,
) -> bool:
    expected_argv = target.synthesize_start_argv()
    return (
        getattr(client, "_base", None) == target.base_url.rstrip("/")
        and getattr(client, "_auth_token", None) == target.auth_token
        and getattr(client, "_timeout", None) == settings.timeout_seconds
        and getattr(client, "_trust_env", None) == settings.use_system_proxy
        and getattr(client, "_auto_start", None) == target.auto_start
        and getattr(client, "_start_command", None) == target.start_command
        and getattr(client, "_start_argv", None) == (
            list(expected_argv) if expected_argv is not None else None
        )
        and getattr(client, "_stop_command", None) == target.stop_command
        and getattr(client, "_idle_timeout", None) == target.idle_timeout_seconds
        and getattr(client, "_startup_timeout", None) == target.startup_timeout_seconds
        and getattr(client, "_health_path", None) == target.health_path
        and getattr(client, "_cwd", None) == target.cwd
        and getattr(client, "_launch_env", None) == target.effective_env()
        and getattr(client, "_max_concurrent_requests", None)
            == target.effective_max_concurrent_requests
        and getattr(client, "_request_read_timeout_seconds", None)
            == target.request_read_timeout_seconds
    )


def _generic_openai_client_matches(
    client: GenericOpenAIClient,
    *,
    settings: Settings,
    target: Any,
) -> bool:
    return (
        getattr(client, "_base", None) == target.base_url.rstrip("/")
        and getattr(client, "_auth_token", None) == target.auth_token
        and getattr(client, "_timeout", None) == settings.timeout_seconds
        and getattr(client, "_trust_env", None) == settings.use_system_proxy
        and getattr(client, "_auto_start", None) == target.auto_start
        and getattr(client, "_start_command", None) == target.start_command
        and getattr(client, "_start_argv", None) is None
        and getattr(client, "_stop_command", None) == target.stop_command
        and getattr(client, "_idle_timeout", None) == target.idle_timeout_seconds
        and getattr(client, "_startup_timeout", None) == target.startup_timeout_seconds
        and getattr(client, "_health_path", None) == target.health_path
        and getattr(client, "_cwd", None) == target.cwd
        and getattr(client, "_launch_env", None) is None
        and getattr(client, "_max_concurrent_requests", None)
            == target.max_concurrent_requests
        and getattr(client, "_request_read_timeout_seconds", None)
            == target.request_read_timeout_seconds
    )


def _comfyui_client_matches(
    client: ComfyUIClient,
    *,
    settings: Settings,
    target: Any,
) -> bool:
    group_name = target.vram_runtime_group or f"comfyui:{target.base_url.rstrip('/')}"
    expected_runtime_group = f"gpu:{target.gpu_device}|{group_name}"
    return (
        getattr(client, "_base", None) == target.base_url.rstrip("/")
        and getattr(client, "_auth_token", None) == target.auth_token
        and getattr(client, "_timeout", None) == settings.timeout_seconds
        and getattr(client, "_trust_env", None) == settings.use_system_proxy
        and getattr(client, "_auto_start", None) == target.auto_start
        and getattr(client, "_start_command", None) == target.start_command
        and getattr(client, "_stop_command", None) == target.stop_command
        and getattr(client, "_idle_timeout", None) == target.idle_timeout_seconds
        and getattr(client, "_startup_timeout", None) == target.startup_timeout_seconds
        and getattr(client, "_health_path", None) == target.health_path
        and getattr(client, "_cwd", None) == target.cwd
        and getattr(client, "_gpu_device", None) == target.gpu_device
        and getattr(client, "vram_runtime_group", None)
            == expected_runtime_group
        and getattr(client, "_workflow_config", None) == target.workflow_config()
    )


async def _swap_settings(app: FastAPI, new_settings: Settings) -> None:
    """Atomically replace app.state.settings and rebuild client pools."""
    old_clients: Dict[str, AnthropicClient] = dict(getattr(app.state, "clients", {}))
    old_ollama: Dict[str, OllamaClient] = dict(getattr(app.state, "ollama_clients", {}))
    old_llama_cpp: Dict[str, LlamaCppClient] = dict(getattr(app.state, "llama_cpp_clients", {}))
    old_generic_openai: Dict[str, GenericOpenAIClient] = dict(
        getattr(app.state, "generic_openai_clients", {})
    )
    old_comfyui: Dict[str, ComfyUIClient] = dict(getattr(app.state, "comfyui_clients", {}))
    vram_coordinator = getattr(app.state, "vram_coordinator", None)
    memory_coordinator = getattr(app.state, "memory_coordinator", None)

    new_clients: Dict[str, AnthropicClient] = {}
    for up in new_settings.anthropic_upstreams:
        new_clients[up.name] = AnthropicClient(
            up.base_url,
            up.auth_token,
            timeout=new_settings.timeout_seconds,
            trust_env=new_settings.use_system_proxy,
        )
    new_ollama: Dict[str, OllamaClient] = {}
    remaining_ollama = dict(old_ollama)
    for tgt in new_settings.ollama_targets:
        existing = remaining_ollama.pop(tgt.name, None)
        if existing is not None and _ollama_client_matches(
            existing, settings=new_settings, target=tgt
        ):
            new_ollama[tgt.name] = existing
            continue
        if existing is not None:
            try:
                await existing.aclose()
            except Exception:  # pragma: no cover
                pass
        new_ollama[tgt.name] = OllamaClient(
            tgt.base_url,
            timeout=new_settings.timeout_seconds,
            trust_env=new_settings.use_system_proxy,
            auto_start=tgt.auto_start,
            start_command=tgt.start_command,
            stop_command=tgt.stop_command,
            idle_timeout_seconds=tgt.idle_timeout_seconds,
            startup_timeout_seconds=tgt.startup_timeout_seconds,
            health_path=tgt.health_path,
            cwd=tgt.cwd,
            target_name=tgt.name,
            vram_coordinator=vram_coordinator,
            memory_coordinator=memory_coordinator,
        )
    new_llama_cpp: Dict[str, LlamaCppClient] = {}
    remaining_llama_cpp = dict(old_llama_cpp)
    for raw_tgt in new_settings.llama_cpp_targets:
        tgt = new_settings.effective_llama_cpp_target(raw_tgt)
        existing = remaining_llama_cpp.pop(tgt.name, None)
        if existing is not None and _llama_cpp_client_matches(
            existing, settings=new_settings, target=tgt
        ):
            new_llama_cpp[tgt.name] = existing
            continue
        if existing is not None:
            try:
                await existing.aclose()
            except Exception:  # pragma: no cover
                pass
        new_llama_cpp[tgt.name] = LlamaCppClient(
            tgt.base_url,
            auth_token=tgt.auth_token,
            timeout=new_settings.timeout_seconds,
            trust_env=new_settings.use_system_proxy,
            auto_start=tgt.auto_start,
            start_command=tgt.start_command,
            start_argv=tgt.synthesize_start_argv(),
            stop_command=tgt.stop_command,
            idle_timeout_seconds=tgt.idle_timeout_seconds,
            startup_timeout_seconds=tgt.startup_timeout_seconds,
            health_path=tgt.health_path,
            cwd=tgt.cwd,
            launch_env=tgt.effective_env(),
            target_name=tgt.name,
            vram_coordinator=vram_coordinator,
            memory_coordinator=memory_coordinator,
            max_concurrent_requests=tgt.effective_max_concurrent_requests,
            request_read_timeout_seconds=tgt.request_read_timeout_seconds,
        )
    new_generic_openai: Dict[str, GenericOpenAIClient] = {}
    remaining_generic_openai = dict(old_generic_openai)
    for tgt in new_settings.generic_openai_targets:
        existing = remaining_generic_openai.pop(tgt.name, None)
        if existing is not None and _generic_openai_client_matches(
            existing, settings=new_settings, target=tgt
        ):
            new_generic_openai[tgt.name] = existing
            continue
        if existing is not None:
            try:
                await existing.aclose()
            except Exception:  # pragma: no cover
                pass
        new_generic_openai[tgt.name] = GenericOpenAIClient(
            tgt.base_url,
            auth_token=tgt.auth_token,
            timeout=new_settings.timeout_seconds,
            trust_env=new_settings.use_system_proxy,
            auto_start=tgt.auto_start,
            start_command=tgt.start_command,
            stop_command=tgt.stop_command,
            idle_timeout_seconds=tgt.idle_timeout_seconds,
            startup_timeout_seconds=tgt.startup_timeout_seconds,
            health_path=tgt.health_path,
            cwd=tgt.cwd,
            target_name=tgt.name,
            vram_coordinator=vram_coordinator,
            memory_coordinator=memory_coordinator,
            max_concurrent_requests=tgt.max_concurrent_requests,
            request_read_timeout_seconds=tgt.request_read_timeout_seconds,
        )
    new_comfyui: Dict[str, ComfyUIClient] = {}
    remaining_comfyui = dict(old_comfyui)
    for tgt in new_settings.comfyui_targets:
        existing = remaining_comfyui.pop(tgt.name, None)
        if existing is not None and _comfyui_client_matches(
            existing, settings=new_settings, target=tgt
        ):
            new_comfyui[tgt.name] = existing
            continue
        if existing is not None:
            try:
                await existing.aclose()
            except Exception:  # pragma: no cover
                pass
        new_comfyui[tgt.name] = ComfyUIClient(
            tgt.base_url,
            auth_token=tgt.auth_token,
            timeout=new_settings.timeout_seconds,
            trust_env=new_settings.use_system_proxy,
            auto_start=tgt.auto_start,
            start_command=tgt.start_command,
            stop_command=tgt.stop_command,
            idle_timeout_seconds=tgt.idle_timeout_seconds,
            startup_timeout_seconds=tgt.startup_timeout_seconds,
            health_path=tgt.health_path,
            cwd=tgt.cwd,
            target_name=tgt.name,
            runtime_group=tgt.vram_runtime_group,
            gpu_device=tgt.gpu_device,
            workflow_config=tgt.workflow_config(),
            vram_coordinator=vram_coordinator,
            memory_coordinator=memory_coordinator,
        )

    app.state.settings = new_settings
    app.state.clients = new_clients
    app.state.ollama_clients = new_ollama
    app.state.llama_cpp_clients = new_llama_cpp
    app.state.generic_openai_clients = new_generic_openai
    app.state.comfyui_clients = new_comfyui
    ensure_idle_monitor = getattr(app.state, "ensure_local_target_idle_monitor", None)
    if ensure_idle_monitor is not None:
        ensure_idle_monitor(app)
    ensure_runtime_monitor = getattr(app.state, "ensure_runtime_monitor", None)
    if ensure_runtime_monitor is not None:
        ensure_runtime_monitor(app)

    for c in old_clients.values():
        try:
            await c.aclose()
        except Exception:  # pragma: no cover
            pass
    for c in old_ollama.values():
        if c in new_ollama.values():
            continue
        if c not in remaining_ollama.values():
            continue
        try:
            await c.aclose()
        except Exception:  # pragma: no cover
            pass
    for c in old_llama_cpp.values():
        if c in new_llama_cpp.values():
            continue
        if c not in remaining_llama_cpp.values():
            continue
        try:
            await c.aclose()
        except Exception:  # pragma: no cover
            pass
    for c in old_generic_openai.values():
        if c in new_generic_openai.values():
            continue
        if c not in remaining_generic_openai.values():
            continue
        try:
            await c.aclose()
        except Exception:  # pragma: no cover
            pass
    for c in old_comfyui.values():
        if c in new_comfyui.values():
            continue
        if c not in remaining_comfyui.values():
            continue
        try:
            await c.aclose()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_admin_routes(app: FastAPI) -> None:
    """Mount the admin UI. No-op when admin is disabled in settings."""
    settings: Settings = app.state.settings
    if not settings.admin_enabled:
        return

    @app.get("/admin", include_in_schema=False)
    @app.get("/admin/", include_in_schema=False)
    async def admin_index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/admin/schema", include_in_schema=False)
    async def admin_schema() -> JSONResponse:
        return JSONResponse({"fields": CONFIG_SCHEMA, "groups": GROUP_LABELS})

    @app.post("/admin/generate-token", include_in_schema=False)
    async def admin_generate_token() -> JSONResponse:
        return JSONResponse({"token": generate_access_token()})

    @app.get("/admin/config", include_in_schema=False)
    async def admin_get_config(request: Request) -> JSONResponse:
        return JSONResponse(_settings_to_dict(request.app.state.settings))

    @app.put("/admin/config", include_in_schema=False)
    async def admin_put_config(request: Request) -> PlainTextResponse:
        try:
            data = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="config must be an object")
        data.pop("_path", None)
        data.pop("_inactive_ollama_targets", None)  # legacy front-end key
        data.pop("config_path", None)
        try:
            new_settings = Settings(**data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid config: {exc}") from exc

        save_path = _resolve_save_path(request.app.state.settings)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        on_disk = new_settings.model_dump()
        on_disk.pop("config_path", None)
        save_path.write_text(
            json.dumps(on_disk, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        new_settings = new_settings.model_copy(update={"config_path": str(save_path)})

        await _swap_settings(request.app, new_settings)
        return PlainTextResponse(f"saved to {save_path}")

    @app.post("/admin/probe-models", include_in_schema=False)
    async def admin_probe_models(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}") from exc
        kind = (body.get("kind") or "").lower()
        base_url = (body.get("base_url") or "").rstrip("/")
        token = body.get("auth_token") or ""
        if not base_url:
            raise HTTPException(status_code=400, detail="base_url is required")

        cur: Settings = request.app.state.settings
        timeout = float(body.get("timeout") or min(30.0, cur.timeout_seconds))
        try:
            async with httpx.AsyncClient(
                timeout=timeout, trust_env=cur.use_system_proxy
            ) as cli:
                if kind == "ollama":
                    resp = await cli.get(base_url + "/api/tags")
                    resp.raise_for_status()
                    data = resp.json()
                    names = [
                        m.get("name")
                        for m in (data.get("models") or [])
                        if isinstance(m, dict) and m.get("name")
                    ]
                elif kind == "anthropic":
                    headers = {"anthropic-version": "2023-06-01"}
                    if token:
                        headers["x-api-key"] = token
                        headers["authorization"] = f"Bearer {token}"
                    resp = await cli.get(base_url + "/v1/models", headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    # Anthropic-style: {"data": [{"id": ...}, ...]}.
                    # OpenAI-style is identical for this field.
                    names = [
                        m.get("id")
                        for m in (data.get("data") or [])
                        if isinstance(m, dict) and m.get("id")
                    ]
                elif kind == "llama_cpp":
                    headers = {}
                    if token:
                        headers["x-api-key"] = token
                        headers["authorization"] = f"Bearer {token}"
                    resp = await cli.get(base_url + "/v1/models", headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    names = [
                        m.get("id")
                        for m in (data.get("data") or [])
                        if isinstance(m, dict) and m.get("id")
                    ]
                    if not names:
                        names = [
                            m.get("name") or m.get("model")
                            for m in (data.get("models") or [])
                            if isinstance(m, dict) and (m.get("name") or m.get("model"))
                        ]
                elif kind == "openai":
                    headers = {}
                    if token:
                        headers["authorization"] = f"Bearer {token}"
                        headers["x-api-key"] = token
                    resp = await cli.get(base_url + "/v1/models", headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    names = [
                        m.get("id")
                        for m in (data.get("data") or [])
                        if isinstance(m, dict) and m.get("id")
                    ]
                else:
                    raise HTTPException(
                        status_code=400, detail=f"unknown kind: {kind!r}"
                    )
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"upstream {exc.response.status_code}: {exc.response.text[:300]}",
            ) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"probe failed: {exc}") from exc
        return JSONResponse({"models": names})
