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
from pathlib import Path
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from .anthropic_client import AnthropicClient
from .config import Settings
from .llama_cpp_client import LlamaCppClient
from .ollama_client import OllamaClient


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
#
# Field types understood by the front-end:
#   string, int, float, bool, string_list, string_map,
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
     "description": "可选：传给 --ctx-size 的上下文长度。仅自动拼装时使用；不填继承 llama_cpp_defaults.ctx_size"},
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
     "description": "默认 --ctx-size；自动拼装命令时使用，target 可覆盖"},
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

MODEL_PROFILE_ITEM_SCHEMA: List[Dict[str, Any]] = [
    {"key": "model", "type": "string", "default": "", "required": True,
     "autocomplete": "model_names",
     "description": "模型名（裸名应用于所有 target；与 target 一起拼出最终 key 'model@target' 仅覆盖该 target）"},
    {"key": "target", "type": "string", "default": "",
     "autocomplete": "source_names",
     "description": "可选：source 名字（anthropic_upstreams / openai_upstreams / ollama_targets / llama_cpp_targets 中的某个 name）。留空 = 对该 model 名所有 target 生效"},
    {"key": "capabilities", "type": "string_list",
     "default": ["completion", "tools", "vision"],
     "description": "子集自 completion / tools / vision；至少包含 completion"},
    {"key": "context_length", "type": "int", "default": 200000,
     "description": "总上下文 token 上限（输入 + 输出）"},
    {"key": "max_output_tokens", "type": "int", "default": None,
     "description": "可选：覆盖默认 num_predict；同时是 max_tokens 的上限"},
    {"key": "estimated_vram_gb", "type": "float", "default": None,
     "description": "可选：该模型加载后预计占用的 GPU 显存（GB）。仅本地 Ollama / llama.cpp 反向代理会使用，用于启动前预检和空闲模型回收"},
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
     "description": "提供该模型的 source 名字（anthropic_upstreams / openai_upstreams / ollama_targets / llama_cpp_targets 中的某个 name）"},
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

API_INTERFACE_ITEM_SCHEMA: List[Dict[str, Any]] = OLLAMA_INTERFACE_ITEM_SCHEMA

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
   "description": "API 接口数组（Anthropic /v1/messages + OpenAI /v1/chat/completions + /v1/models）。每个 entry 独立的 host/port/access_tokens/exposed_models"},

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
   "description": "Enable the runtime dashboard mounted at /dashboard on its own listener."},
  {"key": "dashboard_host", "type": "string", "default": "127.0.0.1", "group": "dashboard",
   "description": "Dashboard listener bind address. Keep 127.0.0.1 unless fronted by a trusted proxy."},
  {"key": "dashboard_port", "type": "int", "default": 21432, "group": "dashboard",
   "description": "Dashboard listener port. It must be distinct from internal, external, and admin ports."},
  {"key": "dashboard_sample_interval_seconds", "type": "float", "default": 10.0, "group": "dashboard",
   "description": "Runtime metric sampling interval in seconds."},
  {"key": "dashboard_retention_seconds", "type": "float", "default": 604800.0, "group": "dashboard",
   "description": "Dashboard history retention in seconds."},
  {"key": "dashboard_data_path", "type": "string", "default": "logs/dashboard_history.json", "group": "dashboard",
   "description": "JSON file used to persist dashboard history. Leave empty to disable file persistence."},
  {"key": "dashboard_model_reclaim_enabled", "type": "bool", "default": False, "group": "dashboard",
   "description": "Allow the dashboard Current Models table to request release of eligible idle local models."},
  {"key": "dashboard_reclaim_idle_seconds", "type": "float", "default": 20.0, "group": "dashboard",
   "description": "用户在 dashboard 点击关闭按钮时所需的最小空闲秒数。和自动 LRU 回收的 60s 阈值独立；用户判断更宽松，默认 20s。"},
  {"key": "vram_low_free_reclaim_enabled", "type": "bool", "default": True, "group": "dashboard",
   "description": "Enable periodic low-free-VRAM checks that release eligible idle local models."},
  {"key": "vram_low_free_threshold_mib", "type": "float", "default": 200.0, "group": "dashboard",
   "description": "When free GPU VRAM falls below this MiB threshold, eligible idle models may be released."},

  {"key": "admin_enabled", "type": "bool", "default": True, "group": "admin",
   "description": "是否启用本 /admin 编辑器（关闭后需手动改 config.json）"},
  {"key": "admin_host", "type": "string", "default": "127.0.0.1", "group": "admin",
   "description": "Admin UI 监听地址。强烈建议保持 127.0.0.1；/admin 没有内置鉴权"},
  {"key": "admin_port", "type": "int", "default": 21433, "group": "admin",
   "description": "Admin UI 独立监听端口。设为 null 才会把 /admin 挂回 internal 端口（旧行为，不推荐）"},
]

GROUP_LABELS: List[Dict[str, str]] = [
  {"key": "model_sources_remote", "label": "Remote Sources", "hint": "远端模型来源：Anthropic / OpenAI 兼容上游", "section": "model_sources", "section_label": "Model Sources", "section_hint": "配置可被路由的所有模型来源（target / API）"},
  {"key": "model_sources_ollama", "label": "Ollama Sources", "hint": "本机或远端 Ollama 服务", "section": "model_sources", "section_label": "Model Sources", "section_hint": "配置可被路由的所有模型来源（target / API）"},
  {"key": "model_sources_llama_cpp", "label": "llama.cpp Sources", "hint": "本机 llama.cpp server 进程", "section": "model_sources", "section_label": "Model Sources", "section_hint": "配置可被路由的所有模型来源（target / API）"},
  {"key": "interface_ollama", "label": "Ollama Interface", "hint": "Ollama 兼容接口（/api/* 与 /v1/chat/completions）。每个 entry 各自选择暴露哪些模型", "section": "interfaces", "section_label": "Interfaces", "section_hint": "对用户暴露的接口；每个 entry 都是独立的 host/port/access_tokens/exposed_models"},
  {"key": "interface_api", "label": "API Interface", "hint": "Anthropic /v1/messages + OpenAI /v1/chat/completions + /v1/models。每个 entry 各自选择暴露哪些模型", "section": "interfaces", "section_label": "Interfaces", "section_hint": "对用户暴露的接口；每个 entry 都是独立的 host/port/access_tokens/exposed_models"},
  {"key": "runtime", "label": "Runtime & Profiles", "hint": "运行时缺省值、出站网络与每模型 capability/thinking profile", "section": "runtime", "section_label": "Runtime", "section_hint": "跨所有 target / 接口共享的运行时设置"},
  {"key": "dashboard", "label": "Dashboard", "hint": "Runtime graphs and the low-VRAM safety monitor", "section": "dashboard", "section_label": "Dashboard", "section_hint": "Memory, VRAM, and loaded local model telemetry"},
  {"key": "admin", "label": "Admin UI", "hint": "配置页面自身的开关与监听地址（无内置鉴权）", "section": "admin", "section_label": "Admin UI", "section_hint": "仅影响 /admin 配置页面本身"},
]


# ---------------------------------------------------------------------------
# HTML / JS bundle
# ---------------------------------------------------------------------------

_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>fake-ollama config editor</title>
<style>
 :root { color-scheme: light dark; --gap: 0.55rem; --border: rgba(127,127,127,0.3); --accent: #3a7bd5; }
 body { font-family: ui-sans-serif, system-ui, sans-serif; max-width: 1280px; margin: 0 auto; padding: 0 1rem 2rem; }
 .topbar { position: sticky; top: 0; z-index: 50; background: Canvas; padding: 0.8rem 0 0.6rem; border-bottom: 1px solid var(--border); margin-bottom: 0.8rem; }
 .topbar h1 { margin: 0 0 0.4rem 0; font-size: 1.4rem; }
 .sub { color: #888; margin-bottom: 0.5rem; font-size: 0.9rem; }
 .warn { background: rgba(255, 200, 0, 0.15); padding: 0.5rem 0.8rem; border-left: 3px solid orange; border-radius: 3px; margin-bottom: 1rem; font-size: 0.88rem; }
 .layout { display: grid; grid-template-columns: 220px 1fr; gap: 1.2rem; align-items: start; }
 @media (max-width: 760px) { .layout { grid-template-columns: 1fr; } .sidenav { position: static !important; max-height: none !important; } }
 .sidenav { position: sticky; top: 5.5rem; max-height: calc(100vh - 6rem); overflow-y: auto;
   border: 1px solid var(--border); border-radius: 6px; padding: 0.4rem 0; background: rgba(127,127,127,0.04); }
 .sidenav h3 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #888;
   margin: 0.2rem 0.8rem 0.3rem; font-weight: 600; }
 .sidenav .nav-section { padding: 0.55rem 0.8rem 0.15rem; font-size: 0.74rem; text-transform: uppercase;
   letter-spacing: 0.05em; color: #666; font-weight: 700; }
 .sidenav .nav-section small { display: block; margin-top: 0.12rem; color: #888; font-size: 0.74rem;
   font-weight: 400; text-transform: none; letter-spacing: normal; line-height: 1.3; }
 .sidenav a { display: block; padding: 0.4rem 0.8rem 0.4rem 1.1rem; text-decoration: none; color: inherit;
   border-left: 3px solid transparent; font-size: 0.92rem; line-height: 1.25; }
 .sidenav a:hover { background: rgba(127,127,127,0.1); }
 .sidenav a.active { border-left-color: var(--accent); background: rgba(58,123,213,0.1); font-weight: 600; }
 .sidenav a small { display: block; font-size: 0.75rem; color: #888; font-weight: 400; margin-top: 0.1rem; }
 .sidenav .subnav { display: flex; flex-direction: column; }
 .sidenav .subnav a { padding: 0.22rem 0.6rem 0.22rem 2rem; font-size: 0.85rem; border-left: 3px solid transparent; color: #555; }
 .sidenav .subnav a:hover { background: rgba(127,127,127,0.08); }
 .sidenav .subnav a.active { border-left-color: var(--accent); background: rgba(58,123,213,0.07); color: inherit; font-weight: 600; }
 .sidenav .subnav a .idx { display: inline-block; min-width: 1.6em; color: #888; font-family: ui-monospace, Consolas, monospace; font-size: 0.75rem; }
 .sidenav .subnav .empty { padding: 0.2rem 0.6rem 0.2rem 2rem; font-size: 0.78rem; color: #888; font-style: italic; }
 .sidenav .nav-filter { padding: 0.4rem 0.6rem 0.5rem; border-bottom: 1px solid var(--border); }
 .sidenav .nav-filter input { width: 100%; box-sizing: border-box; padding: 0.25rem 0.4rem; font-size: 0.85rem; font-family: inherit; }
 .sidenav .nav-empty { padding: 0.4rem 0.8rem; font-size: 0.8rem; color: #888; font-style: italic; }
 .section-super { margin: 0 0 0.9rem 0; padding: 0.65rem 0.8rem; border: 1px solid var(--border);
   border-radius: 6px; background: rgba(58,123,213,0.05); }
 .section-super .section-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em;
   color: #2f5f98; font-weight: 700; }
 .section-super .section-hint { margin-top: 0.2rem; color: #666; font-size: 0.86rem; }
 section.group-section { margin-bottom: 1.6rem; scroll-margin-top: 5.5rem; }
 section.group-section > h2 { margin: 0 0 0.2rem 0; font-size: 1.1rem; padding-bottom: 0.3rem; border-bottom: 2px solid var(--accent); }
 section.group-section > .group-hint { color: #888; font-size: 0.85rem; margin-bottom: 0.6rem; }
 .field { border: 1px solid var(--border); border-radius: 5px; padding: 0.5rem 0.7rem; margin-bottom: var(--gap); background: Canvas; }
 .group .item { scroll-margin-top: 5.5rem; }
 .group .item.nav-flash { box-shadow: 0 0 0 2px var(--accent); transition: box-shadow 1.2s ease-out; }
 .field > label { display: flex; align-items: baseline; gap: 0.4rem; font-weight: 600; }
 .field .desc { color: #888; font-size: 0.82rem; margin: 0.15rem 0 0.4rem 0; }
 .field input[type=text], .field input[type=password], .field input[type=number],
 .field textarea, .field select {
   width: 100%; box-sizing: border-box; padding: 0.3rem 0.4rem;
   font-family: inherit; font-size: 0.92rem;
 }
 .field textarea { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; min-height: 4.5rem; }
 .field.disabled > .body { opacity: 0.45; pointer-events: none; }
 .bool-row { display: flex; align-items: center; gap: 0.5rem; }
 .bool-row .bool-label { font-family: ui-monospace, Consolas, monospace; font-weight: 600; }
 .bool-row .bool-label.t { color: #2a8a2a; }
 .bool-row .bool-label.f { color: #c0392b; }
 .entries { display: flex; flex-direction: column; gap: 0.3rem; }
 .entries .entry { display: flex; gap: 0.4rem; align-items: center; }
 .entries .entry input { flex: 1; }
 .entries .empty { color: #888; font-size: 0.85rem; font-style: italic; }
 .modal-back { position: fixed; inset: 0; background: rgba(0,0,0,0.45); z-index: 100;
   display: flex; align-items: center; justify-content: center; }
 .modal { background: Canvas; color: CanvasText; border: 1px solid var(--border);
   border-radius: 6px; padding: 1rem 1.2rem; max-width: 560px; width: 90%;
   max-height: 80vh; display: flex; flex-direction: column; gap: 0.6rem; }
 .modal h2 { margin: 0; font-size: 1.1rem; }
 .modal .modal-body { overflow-y: auto; max-height: 55vh; border: 1px solid var(--border);
   padding: 0.5rem 0.7rem; border-radius: 4px; }
 .modal .modal-body label { display: flex; align-items: center; gap: 0.5rem;
   padding: 0.15rem 0; font-family: ui-monospace, Consolas, monospace; font-size: 0.9rem; }
 .modal .modal-body label.exists { color: #888; }
 .modal .modal-body .tag { font-size: 0.7rem; padding: 0 0.3rem; border-radius: 2px;
   background: rgba(127,127,127,0.2); }
 .modal .modal-foot { display: flex; justify-content: flex-end; gap: 0.5rem; }
 .modal .modal-tools { display: flex; gap: 0.5rem; align-items: center; font-size: 0.85rem; }
 .group { border: 1px dashed var(--border); border-radius: 5px; padding: 0.5rem 0.7rem; margin-bottom: var(--gap); background: rgba(127,127,127,0.04); }
 .group .item { border-left: 3px solid var(--border); padding: 0.4rem 0.6rem; margin-bottom: 0.5rem; background: rgba(127,127,127,0.04); border-radius: 3px; }
 .map-key { display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.4rem; }
 .map-key input { flex: 1; }
 button { padding: 0.35rem 0.8rem; font-size: 0.9rem; cursor: pointer; }
 button.primary { font-weight: 600; }
 button.danger { color: #c0392b; }
 button.detect { font-size: 0.8rem; padding: 0.2rem 0.5rem; }
 .row { display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem; }
 #status { margin-left: 0.5rem; font-size: 0.9rem; min-height: 1.2em; }
 #status.ok { color: #2a8a2a; }
 #status.err { color: #c0392b; white-space: pre-wrap; }
 details > summary { cursor: pointer; }
 details pre { overflow: auto; max-height: 50vh; background: rgba(127,127,127,0.08); padding: 0.5rem; border-radius: 4px; }
 .field-head { display: flex; justify-content: space-between; align-items: baseline; gap: 0.5rem; }
</style>
</head>
<body>

<div class="topbar">
  <h1>fake-ollama config editor</h1>
  <div class="row">
    <button class="primary" id="save">Save & Reload</button>
    <button id="reload">Discard & Reload from disk</button>
    <button id="rawmode">Toggle raw JSON</button>
    <span id="status"></span>
  </div>
</div>

<div class="sub">file: <code id="path">(loading...)</code></div>
<div class="warn">
  <b>Warning:</b> 本编辑器没有鉴权。请保持 <code>admin_host=127.0.0.1</code>，或在 <code>config.json</code> 里设置
  <code>"admin_enabled": false</code>。保存会原子重建上游连接池。
</div>

<div class="layout">
  <aside class="sidenav" id="sidenav">
    <div class="nav-filter"><input type="search" id="navFilter" placeholder="过滤 sections / targets / models" autocomplete="off"></div>
    <h3>Sections</h3>
  </aside>
  <main class="content">
    <div id="form"></div>
<datalist id="dl-model-names"></datalist>
<datalist id="dl-source-names"></datalist>
<textarea id="raw" hidden></textarea>
    <details style="margin-top: 1rem;">
      <summary>当前 schema（只读）</summary>
      <pre id="schemaDump"></pre>
    </details>
  </main>
</div>

<script>
// Resolve admin base whether the user landed on /admin or /admin/.
const ADMIN_BASE = (() => {
  const m = location.pathname.match(/^(.*?\/admin)(\/.*)?$/);
  return m ? m[1] : '/admin';
})();

const $form = document.getElementById('form');
const $raw = document.getElementById('raw');
const $status = document.getElementById('status');
const $path = document.getElementById('path');
const $schemaDump = document.getElementById('schemaDump');
const $sidenav = document.getElementById('sidenav');

let SCHEMA = [];
let GROUPS = [];
let RAW_MODE = false;
const PROBED_MODEL_NAMES = new Set();

function setStatus(text, kind) {
  $status.textContent = text;
  $status.className = kind || '';
}

function el(tag, props, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (k === 'class') e.className = v;
    else if (k === 'style') e.setAttribute('style', v);
    else if (k.startsWith('on')) e[k] = v;
    else if (v === true) e.setAttribute(k, '');
    else if (v === false || v == null) { /* skip */ }
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.append(c.nodeType ? c : document.createTextNode(c));
  }
  return e;
}

// ---- Renderers (each returns {node, read}) -------------------------

function makeScalar(field, value) {
  let input;
  if (field.type === 'bool') {
    input = el('input', {type: 'checkbox'});
    input.checked = !!value;
    const label = el('span', {class: 'bool-label ' + (input.checked ? 't' : 'f')},
                     input.checked ? 'true' : 'false');
    input.addEventListener('change', () => {
      label.textContent = input.checked ? 'true' : 'false';
      label.className = 'bool-label ' + (input.checked ? 't' : 'f');
    });
    const wrap = el('div', {class: 'bool-row'}, input, label);
    return {
      node: wrap,
      read: () => input.checked,
      get: () => input.checked,
      set: (v) => {
        input.checked = !!v;
        input.dispatchEvent(new Event('change'));
      },
    };
  }
  if (field.type === 'int' || field.type === 'float') {
    input = el('input', {type: 'number', step: field.type === 'float' ? 'any' : '1'});
    if (value != null) input.value = value;
  } else if (field.secret) {
    input = el('input', {type: 'password'});
    input.value = value == null ? '' : String(value);
  } else {
    input = el('input', {type: 'text'});
    input.value = value == null ? '' : String(value);
  }
  // Wire datalist-based autocomplete on text inputs (skips bool / number / secret).
  if ((field.type === 'string' || !field.type) && !field.secret && field.autocomplete) {
    if (field.autocomplete === 'model_names') {
      input.setAttribute('list', 'dl-model-names');
      input.addEventListener('focus', refreshModelDatalist);
      input.addEventListener('input', refreshModelDatalist);
    } else if (field.autocomplete === 'source_names') {
      input.setAttribute('list', 'dl-source-names');
      input.addEventListener('focus', refreshSourceDatalist);
      input.addEventListener('input', refreshSourceDatalist);
    }
  }
  // Wrap secret/generate fields with toggle + generate buttons.
  if (field.secret || field.generate) {
    const row = el('div', {class: 'row', style: 'gap:0.4rem;'});
    input.style.flex = '1';
    row.append(input);
    if (field.secret) {
      const eye = el('button', {type: 'button', class: 'detect',
        title: 'Show / hide', onclick: () => {
          input.type = input.type === 'password' ? 'text' : 'password';
          eye.textContent = input.type === 'password' ? 'Show' : 'Hide';
        }}, 'Show');
      row.append(eye);
    }
    if (field.generate) {
      const gen = el('button', {type: 'button', class: 'detect',
        title: 'Generate a random token', onclick: () => {
          input.value = randomToken();
          if (input.type === 'password') {
            // Briefly show so user can copy it.
            input.type = 'text';
          }
        }}, 'Generate');
      row.append(gen);
    }
    return {
      node: row,
      read() { return input.value; },
      get() { return input.value; },
      set(v) { input.value = v == null ? '' : String(v); },
    };
  }
  return {
    node: input,
    read() {
      if (field.type === 'int') return input.value === '' ? null : parseInt(input.value, 10);
      if (field.type === 'float') return input.value === '' ? null : parseFloat(input.value);
      return input.value;
    },
    get() { return input.value; },
    set(v) { input.value = v == null ? '' : String(v); },
  };
}

function randomToken(bytes) {
  const arr = new Uint8Array(bytes || 24);
  crypto.getRandomValues(arr);
  return 'tk-' + Array.from(arr, b => b.toString(16).padStart(2, '0')).join('');
}

function makeStringList(field, value) {
  const wrap = el('div', {class: 'entries'});
  const rows = [];
  const empty = el('div', {class: 'empty'}, '(empty — click “+ add” to insert)');
  function refreshEmpty() {
    if (rows.length === 0) {
      if (!wrap.contains(empty)) wrap.insertBefore(empty, addBtn);
    } else if (wrap.contains(empty)) {
      wrap.removeChild(empty);
    }
  }
  function addRow(text) {
    const isSecret = !!field.secret_each;
    const input = el('input', {
      type: isSecret ? 'password' : 'text',
      placeholder: field.key,
    });
    if (field.autocomplete === 'model_names') {
      input.setAttribute('list', 'dl-model-names');
      input.addEventListener('focus', refreshModelDatalist);
      input.addEventListener('input', refreshModelDatalist);
    }
    input.value = text == null ? '' : String(text);
    const entry = {input};
    const extras = [];
    if (isSecret) {
      const eye = el('button', {type: 'button', class: 'detect',
        title: 'Show / hide', onclick: () => {
          input.type = input.type === 'password' ? 'text' : 'password';
          eye.textContent = input.type === 'password' ? 'Show' : 'Hide';
        }}, 'Show');
      extras.push(eye);
    }
    if (field.generate_each) {
      const gen = el('button', {type: 'button', class: 'detect',
        title: 'Generate a random token', onclick: () => {
          input.value = randomToken();
          if (input.type === 'password') input.type = 'text';
        }}, 'Generate');
      extras.push(gen);
    }
    const rm = el('button', {class: 'danger', type: 'button', onclick: () => {
      wrap.removeChild(row);
      rows.splice(rows.indexOf(entry), 1);
      refreshEmpty();
    }}, '×');
    const row = el('div', {class: 'entry'}, input, ...extras, rm);
    rows.push(entry);
    wrap.insertBefore(row, addBtn);
    return input;
  }
  const addLabel = field.generate_each ? '+ add (blank)' : '+ add';
  const addBtn = el('button', {type: 'button', onclick: () => {
    const i = addRow('');
    refreshEmpty();
    i.focus();
  }}, addLabel);
  wrap.append(addBtn);
  if (field.generate_each) {
    const genBtn = el('button', {type: 'button',
      style: 'margin-left:0.4rem;', onclick: () => {
        const i = addRow(randomToken());
        if (i.type === 'password') i.type = 'text';
        refreshEmpty();
      }}, '+ generate token');
    wrap.append(genBtn);
  }
  if (Array.isArray(value)) for (const v of value) addRow(v);
  refreshEmpty();
  return {
    node: wrap,
    read: () => rows.map(r => r.input.value.trim()).filter(Boolean),
    get: () => rows.map(r => r.input.value.trim()).filter(Boolean),
    set: (arr) => {
      // Replace all rows with the given list.
      for (const r of rows.slice()) {
        const node = r.input.parentNode;
        if (node && node.parentNode === wrap) wrap.removeChild(node);
      }
      rows.length = 0;
      if (Array.isArray(arr)) for (const v of arr) addRow(v);
      refreshEmpty();
    },
    add: (arr) => {
      if (!Array.isArray(arr)) return;
      for (const v of arr) addRow(v);
      refreshEmpty();
    },
  };
}

// Modal dialog: show a list of candidate models with checkboxes; pre-checks
// items already present in `existing`. Resolves to {selected: string[]}
// (only the items the user wants to keep) or null on cancel. The caller
// decides whether to merge or replace.
function pickModelsDialog({title, candidates, existing}) {
  return new Promise((resolve) => {
    const existingSet = new Set(existing || []);
    const back = el('div', {class: 'modal-back'});
    const body = el('div', {class: 'modal-body'});
    const checkboxes = [];
    const fresh = [];
    for (const name of candidates) {
      const cb = el('input', {type: 'checkbox'});
      const isExisting = existingSet.has(name);
      cb.checked = !isExisting;  // pre-check only NEW models
      const tag = isExisting
        ? el('span', {class: 'tag'}, 'already added')
        : el('span', {class: 'tag', style: 'background:rgba(42,138,42,0.25);'}, 'new');
      const lab = el('label', {class: isExisting ? 'exists' : ''}, cb, name, tag);
      body.append(lab);
      checkboxes.push({cb, name});
      if (!isExisting) fresh.push(cb);
    }
    if (candidates.length === 0) {
      body.append(el('div', {class: 'empty'}, '(上游未返回任何模型)'));
    }

    const checkAll = el('button', {type: 'button', onclick: () => {
      for (const {cb} of checkboxes) cb.checked = true;
    }}, 'Check all');
    const checkNone = el('button', {type: 'button', onclick: () => {
      for (const {cb} of checkboxes) cb.checked = false;
    }}, 'Check none');
    const checkNew = el('button', {type: 'button', onclick: () => {
      for (const {cb, name} of checkboxes) cb.checked = !existingSet.has(name);
    }}, 'Only new');

    const cancel = el('button', {type: 'button', onclick: () => close(null)}, 'Cancel');
    const merge = el('button', {class: 'primary', type: 'button', onclick: () => {
      const picked = checkboxes.filter(x => x.cb.checked).map(x => x.name);
      // Merge: keep existing + add picked (de-dup, preserve order: existing first, then new picks).
      const seen = new Set(existing || []);
      const out = [...(existing || [])];
      for (const n of picked) if (!seen.has(n)) { out.push(n); seen.add(n); }
      close(out);
    }}, 'Merge into list');
    const replace = el('button', {type: 'button', onclick: () => {
      const picked = checkboxes.filter(x => x.cb.checked).map(x => x.name);
      close(picked);
    }}, 'Replace list');

    const dialog = el('div', {class: 'modal'},
      el('h2', {}, title || 'Detected models'),
      el('div', {class: 'modal-tools'}, checkAll, checkNone, checkNew,
        el('span', {style: 'margin-left:auto;color:#888;'},
          candidates.length + ' total, ' + fresh.length + ' new')),
      body,
      el('div', {class: 'modal-foot'}, cancel, replace, merge),
    );
    back.append(dialog);
    function onKey(e) { if (e.key === 'Escape') close(null); }
    function close(result) {
      document.removeEventListener('keydown', onKey);
      if (back.parentNode) back.parentNode.removeChild(back);
      resolve(result);
    }
    back.addEventListener('click', (e) => { if (e.target === back) close(null); });
    document.addEventListener('keydown', onKey);
    document.body.append(back);
  });
}

function makeStringMap(field, value) {
  const wrap = el('div');
  const rows = [];
  function addRow(k, v) {
    const ki = el('input', {type: 'text', placeholder: 'key'});
    ki.value = k || '';
    const vi = el('input', {type: 'text', placeholder: 'value'});
    vi.value = v == null ? '' : String(v);
    const entry = {ki, vi};
    const rm = el('button', {class: 'danger', type: 'button', onclick: () => {
      wrap.removeChild(row);
      rows.splice(rows.indexOf(entry), 1);
    }}, '×');
    const row = el('div', {class: 'map-key'}, ki, vi, rm);
    rows.push(entry);
    wrap.insertBefore(row, addBtn);
  }
  const addBtn = el('button', {type: 'button', onclick: () => addRow()}, '+ add entry');
  wrap.append(addBtn);
  if (value && typeof value === 'object') {
    for (const [k, v] of Object.entries(value)) addRow(k, v);
  }
  return {
    node: wrap,
    read() {
      const out = {};
      for (const {ki, vi} of rows) {
        const k = ki.value.trim();
        if (k) out[k] = vi.value;
      }
      return out;
    },
  };
}

function makeObjectGroup(itemSchema, value) {
  const wrap = el('div');
  const renderers = [];
  const byKey = {};
  const ctx = { getSibling: (k) => byKey[k] };
  for (const sub of itemSchema) {
    const f = renderField(sub, value ? value[sub.key] : undefined, ctx);
    renderers.push({sub, f});
    byKey[sub.key] = f;
    wrap.append(f.node);
  }
  return {
    node: wrap,
    read() {
      const out = {};
      for (const {sub, f} of renderers) {
        if (!f.isPresent()) continue;
        out[sub.key] = f.read();
      }
      return out;
    },
    getRenderer: (key) => byKey[key],
  };
}

function makeObject(field, value) {
  const wrap = el('div', {class: 'group'});
  const renderer = makeObjectGroup(field.item_schema, value || {});
  wrap.append(renderer.node);
  return {
    node: wrap,
    read: () => renderer.read(),
  };
}

function makeObjectList(field, value) {
  const wrap = el('div', {class: 'group'});
  const items = [];
  const changeListeners = [];
  let _idSeq = 0;
  function fireChange() { for (const cb of changeListeners) { try { cb(); } catch (e) { console.error(e); } } }
  function addItem(initial) {
    const renderer = makeObjectGroup(field.item_schema, initial || {});
    const entry = {id: ++_idSeq, renderer};
    const remove = el('button', {class: 'danger', type: 'button', onclick: () => {
      wrap.removeChild(itemBox);
      items.splice(items.indexOf(entry), 1);
      fireChange();
    }}, 'Remove');
    const headerChildren = [];
    if (field.detect_models) {
      const detectBtn = el('button', {class: 'detect', type: 'button', onclick: async () => {
        const baseUrl = renderer.getRenderer('base_url')?.read();
        const tokenR = renderer.getRenderer('auth_token');
        const token = tokenR ? tokenR.read() : '';
        if (!baseUrl) { setStatus('detect: base_url is empty', 'err'); return; }
        detectBtn.disabled = true;
        const prev = detectBtn.textContent;
        detectBtn.textContent = 'detecting...';
        try {
          const r = await fetch(ADMIN_BASE + '/probe-models', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({kind: field.detect_models, base_url: baseUrl, auth_token: token}),
          });
          if (!r.ok) throw new Error(await r.text());
          const j = await r.json();
          const candidates = j.models || [];
          for (const n of candidates) if (n) PROBED_MODEL_NAMES.add(n);
          refreshModelDatalist();
          setStatus('detected ' + candidates.length + ' models, choose...', 'ok');
          const modelKey = field.detect_models === 'llama_cpp' ? 'model' : 'models';
          const modelsR = renderer.getRenderer(modelKey);
          const current = modelsR && modelsR.get ? modelsR.get() : [];
          const existing = Array.isArray(current) ? current : (current ? [current] : []);
          const result = await pickModelsDialog({
            title: 'Detected models from ' + baseUrl,
            candidates,
            existing,
          });
          if (result == null) { setStatus('detect: cancelled'); return; }
          if (modelsR && modelsR.set) {
            if (field.detect_models === 'llama_cpp') modelsR.set(result[0] || '');
            else modelsR.set(result);
          }
          setStatus(field.detect_models === 'llama_cpp'
            ? 'model updated' : 'models updated (' + result.length + ' total)', 'ok');
        } catch (e) {
          setStatus('detect failed: ' + e.message, 'err');
        } finally {
          detectBtn.disabled = false;
          detectBtn.textContent = prev;
        }
      }}, 'Detect models');
      headerChildren.push(detectBtn);
    }
    headerChildren.push(remove);
    const itemBox = el('div', {class: 'item'},
      el('div', {class: 'row', style: 'justify-content: flex-end;'}, ...headerChildren),
      renderer.node,
    );
    entry.itemBox = itemBox;
    items.push(entry);
    wrap.insertBefore(itemBox, addBtn);
    fireChange();
  }
  const addBtn = el('button', {type: 'button', onclick: () => addItem({})},
                       '+ add ' + field.key.replace(/s$/, ''));
  wrap.append(addBtn);
  if (Array.isArray(value)) for (const v of value) addItem(v);
  return {
    node: wrap,
    read: () => items.map(i => i.renderer.read()),
    _items: items,
    _onChange: (cb) => changeListeners.push(cb),
  };
}

function makeObjectMap(field, value) {
  const wrap = el('div', {class: 'group'});
  const items = [];
  const changeListeners = [];
  let _idSeq = 0;
  function fireChange() { for (const cb of changeListeners) { try { cb(); } catch (e) { console.error(e); } } }
  function addItem(key, initial) {
    const ki = el('input', {type: 'text', placeholder: 'model name'});
    if (field.key_autocomplete === 'model_names') {
      ki.setAttribute('list', 'dl-model-names');
      ki.addEventListener('focus', refreshModelDatalist);
    }
    ki.value = key || '';
    const renderer = makeObjectGroup(field.item_schema, initial || {});
    const entry = {id: ++_idSeq, ki, renderer};
    const remove = el('button', {class: 'danger', type: 'button', onclick: () => {
      wrap.removeChild(itemBox);
      items.splice(items.indexOf(entry), 1);
      fireChange();
    }}, 'Remove');
    const head = el('div', {class: 'map-key'}, el('strong', {}, 'key:'), ki, remove);
    const itemBox = el('div', {class: 'item'}, head, renderer.node);
    entry.itemBox = itemBox;
    items.push(entry);
    wrap.insertBefore(itemBox, addBtn);
    fireChange();
  }
  const addBtn = el('button', {type: 'button', onclick: () => {
    addItem('', {});
    if (field.key_autocomplete === 'model_names') refreshModelDatalist();
  }}, '+ add entry');
  wrap.append(addBtn);
  if (value && typeof value === 'object') {
    for (const [k, v] of Object.entries(value)) addItem(k, v);
  }
  return {
    node: wrap,
    read() {
      const out = {};
      for (const {ki, renderer} of items) {
        const k = ki.value.trim();
        if (k) out[k] = renderer.read();
      }
      return out;
    },
    _items: items,
    _onChange: (cb) => changeListeners.push(cb),
  };
}

// ---- Subset-of-sibling list (checkbox UI driven by sibling field) ----
function makeStringListSubsetOf(field, value, ctx) {
  const wrap = el('div');
  const list = el('div', {style: 'display:flex;flex-direction:column;gap:0.1rem;margin-top:0.3rem;'});
  const status = el('div', {class: 'desc', style: 'margin:0.2rem 0 0;'});
  const checked = new Set(Array.isArray(value) ? value : []);
  // Names known to the renderer at any point: union of sibling values + saved.
  let knownNames = Array.from(checked);

  function rerender() {
    list.innerHTML = '';
    if (!knownNames.length) {
      list.append(el('div', {class: 'empty'},
        '(上方 ' + field.subset_of + ' 为空。填好后点“Refresh”生成复选框)'));
      return;
    }
    for (const name of knownNames) {
      const cb = el('input', {type: 'checkbox'});
      cb.checked = checked.has(name);
      cb.onchange = () => { if (cb.checked) checked.add(name); else checked.delete(name); };
      const lab = el('label',
        {style: 'display:flex;align-items:center;gap:0.4rem;padding:0.1rem 0;font-family:ui-monospace,Consolas,monospace;font-size:0.9rem;'},
        cb, name);
      list.append(lab);
    }
  }

  function doRefresh() {
    const sib = ctx && ctx.getSibling ? ctx.getSibling(field.subset_of) : null;
    const siblingValues = sib && sib.read ? sib.read() : [];
    const seen = new Set();
    const merged = [];
    for (const n of siblingValues) if (!seen.has(n)) { seen.add(n); merged.push(n); }
    // Keep stale checks visible so the user can decide to drop them.
    const stale = [];
    for (const n of checked) if (!seen.has(n)) stale.push(n);
    for (const n of stale) merged.push(n);
    knownNames = merged;
    rerender();
    status.textContent = 'refreshed: ' + siblingValues.length + ' from ' + field.subset_of
      + ', ' + checked.size + ' checked'
      + (stale.length ? ', ' + stale.length + ' stale (no longer in ' + field.subset_of + ')' : '');
  }

  const refreshBtn = el('button', {type: 'button', class: 'detect',
    onclick: doRefresh}, 'Refresh from ' + field.subset_of);
  const allBtn = el('button', {type: 'button', class: 'detect', style: 'margin-left:0.3rem;',
    onclick: () => { for (const n of knownNames) checked.add(n); rerender(); }}, 'Check all');
  const noneBtn = el('button', {type: 'button', class: 'detect', style: 'margin-left:0.3rem;',
    onclick: () => { checked.clear(); rerender(); }}, 'Uncheck all');
  wrap.append(
    el('div', {class: 'row'}, refreshBtn, allBtn, noneBtn),
    status,
    list,
  );
  rerender();

  return {
    node: wrap,
    read: () => Array.from(checked),
    get: () => Array.from(checked),
    set: (arr) => {
      checked.clear();
      if (Array.isArray(arr)) for (const v of arr) checked.add(v);
      knownNames = Array.from(checked);
      rerender();
    },
  };
}

// ---- Global model-name autocomplete (datalist for model_profiles keys) ----
function collectKnownModelNames() {
  const out = new Set();
  for (const n of PROBED_MODEL_NAMES) if (n) out.add(n);
  function walkList(renderer) {
    if (!renderer || typeof renderer.read !== 'function') return;
    const arr = renderer.read();
    if (!Array.isArray(arr)) return;
    for (const item of arr) {
      if (item && Array.isArray(item.models)) {
        for (const m of item.models) {
          if (typeof m === 'string') { if (m) out.add(m); }
          else if (m && typeof m === 'object') {
            if (m.name) out.add(m.name);
            if (m.alias) out.add(m.alias);
          }
        }
      }
      if (item && item.model) out.add(item.model);
      if (item && item.alias) out.add(item.alias);
    }
  }
  for (const {field, r} of topRenderers) {
    if (field.key === 'anthropic_upstreams' || field.key === 'openai_upstreams' || field.key === 'ollama_targets' || field.key === 'llama_cpp_targets') walkList(r);
    if (field.key === 'model_profiles' && r && typeof r.read === 'function') {
      const m = r.read();
      if (Array.isArray(m)) {
        for (const entry of m) {
          if (entry && entry.model) out.add(String(entry.model));
        }
      } else if (m && typeof m === 'object') {
        // legacy dict shape
        for (const k of Object.keys(m)) if (k) out.add(k);
      }
    }
  }
  return Array.from(out);
}
function refreshModelDatalist() {
  const dl = document.getElementById('dl-model-names');
  if (!dl) return;
  const names = collectKnownModelNames();
  dl.innerHTML = '';
  for (const n of names) {
    const o = document.createElement('option');
    o.value = n;
    dl.append(o);
  }
}

// ---- Source-name autocomplete (datalist for exposed_models.target & model_profiles.target) ----
function collectKnownSourceNames() {
  const out = new Set();
  function walkList(renderer, fallbackKey) {
    if (!renderer || typeof renderer.read !== 'function') return;
    const arr = renderer.read();
    if (!Array.isArray(arr)) return;
    for (const item of arr) {
      if (!item) continue;
      // llama.cpp targets fall back to the `model` field when `name` is empty.
      const n = item.name || (fallbackKey ? item[fallbackKey] : '');
      if (n) out.add(String(n));
    }
  }
  for (const {field, r} of topRenderers) {
    if (field.key === 'anthropic_upstreams' || field.key === 'openai_upstreams'
        || field.key === 'ollama_targets') {
      walkList(r);
    } else if (field.key === 'llama_cpp_targets') {
      walkList(r, 'model');
    }
  }
  return Array.from(out);
}
function refreshSourceDatalist() {
  const dl = document.getElementById('dl-source-names');
  if (!dl) return;
  const names = collectKnownSourceNames();
  dl.innerHTML = '';
  for (const n of names) {
    const o = document.createElement('option');
    o.value = n;
    dl.append(o);
  }
}

// renderField returns {node, read, isPresent}
function renderField(field, value, ctx) {
  const present = value !== undefined;
  let inner;
  switch (field.type) {
    case 'string':
    case 'int':
    case 'float':
    case 'bool':
      inner = makeScalar(field, present ? value : field.default); break;
    case 'string_list':
      inner = makeStringList(field, present ? value : field.default); break;
    case 'string_list_subset_of':
      inner = makeStringListSubsetOf(field, present ? value : field.default, ctx); break;
    case 'string_map':
      inner = makeStringMap(field, present ? value : field.default); break;
    case 'object':
      inner = makeObject(field, present ? value : field.default); break;
    case 'object_list':
      inner = makeObjectList(field, present ? value : field.default); break;
    case 'object_map':
      inner = makeObjectMap(field, present ? value : field.default); break;
    default:
      inner = makeScalar({type: 'string'}, JSON.stringify(value));
  }

  const fieldBox = el('div', {class: 'field'});
  fieldBox.id = 'field-' + slug(field.key);
  const labelChildren = [];
  let presentBox = null;
  if (!field.required) {
    presentBox = el('input', {type: 'checkbox'});
    presentBox.checked = present;
    presentBox.title = '勾选以包含此字段；取消则使用默认值';
    presentBox.onchange = () => fieldBox.classList.toggle('disabled', !presentBox.checked);
    labelChildren.push(presentBox);
  }
  labelChildren.push(field.key);
  if (field.required) labelChildren.push(el('span', {style: 'color:#c0392b;font-weight:400;'}, ' *required'));

  const defaultStr = (field.default === undefined || field.default === null)
    ? 'null'
    : (typeof field.default === 'object' ? JSON.stringify(field.default) : String(field.default));
  const showDefault = field.type !== 'object' && field.type !== 'object_list' && field.type !== 'object_map';
  const desc = el('div', {class: 'desc'},
    field.description || '',
    showDefault ? ` (default: ${defaultStr})` : '',
  );
  fieldBox.append(
    el('label', {}, ...labelChildren),
    desc,
    el('div', {class: 'body'}, inner.node),
  );
  if (presentBox && !presentBox.checked) fieldBox.classList.add('disabled');

  function setPresent(on) {
    if (!presentBox) return;
    presentBox.checked = !!on;
    fieldBox.classList.toggle('disabled', !presentBox.checked);
  }

  return {
    node: fieldBox,
    isPresent: () => field.required || (presentBox && presentBox.checked),
    read: () => inner.read(),
    get: inner.get ? (...a) => inner.get(...a) : undefined,
    set: inner.set ? (...a) => { setPresent(true); return inner.set(...a); } : undefined,
    add: inner.add ? (...a) => { setPresent(true); return inner.add(...a); } : undefined,
    // Pass through list/map internals so the sidebar sub-nav can hook them.
    _items: inner._items,
    _onChange: inner._onChange ? (cb) => inner._onChange(cb) : undefined,
  };
}

// ---- Top-level form -----------------------------------------------

let topRenderers = [];

function slug(s) { return String(s).replace(/[^a-z0-9]+/gi, '-').toLowerCase(); }

// Registered sub-nav builders, populated during renderForm. Each entry
// rebuilds the nested <a> list under its parent section link.
let subNavBuilders = [];

function scrollToItemBox(itemBox) {
  if (!itemBox) return;
  itemBox.scrollIntoView({behavior: 'smooth', block: 'start'});
  itemBox.classList.add('nav-flash');
  setTimeout(() => itemBox.classList.remove('nav-flash'), 1200);
}

function attachSubNav({sectionId, parentLinkEl, sources, changeNotifiers}) {
  // Single .subnav container per section, fed by multiple sources.
  // Each source is a function returning [{itemBox, label, idx?, kind?}].
  const subWrap = el('div', {class: 'subnav'});
  subWrap.dataset.parent = sectionId;
  parentLinkEl.after(subWrap);
  let _anchorSeq = 0;

  function rebuild() {
    subWrap.innerHTML = '';
    let total = 0;
    for (const src of sources) {
      let entries;
      try { entries = src() || []; } catch (e) { console.error(e); entries = []; }
      for (const entry of entries) {
        if (!entry || !entry.itemBox) continue;
        if (!entry.itemBox.id) {
          entry.itemBox.id = sectionId + '-anchor-' + (++_anchorSeq);
        }
        const itemId = entry.itemBox.id;
        const label = entry.label || '(unnamed)';
        const a = el('a', {href: '#' + itemId, onclick: (ev) => {
          ev.preventDefault();
          scrollToItemBox(entry.itemBox);
          history.replaceState(null, '', '#' + itemId);
        }});
        if (entry.idx != null) a.append(el('span', {class: 'idx'}, String(entry.idx)));
        a.append(label);
        if (entry.kind === 'static') a.classList.add('static');
        a.dataset.itemTarget = itemId;
        a.dataset.searchText = label.toLowerCase();
        subWrap.append(a);
        total += 1;
      }
    }
    if (total === 0) subWrap.append(el('div', {class: 'empty'}, '(empty)'));
  }

  for (const n of (changeNotifiers || [])) {
    try { n(rebuild); } catch (e) { console.error(e); }
  }
  rebuild();
  return {rebuild};
}

function renderForm(config) {
  $form.innerHTML = '';
  // Preserve the filter input + heading; drop only previously-built links.
  const navFilterEl = $sidenav.querySelector('.nav-filter');
  $sidenav.innerHTML = '';
  if (navFilterEl) $sidenav.append(navFilterEl);
  $sidenav.append(el('h3', {}, 'Sections'));
  topRenderers = [];
  subNavBuilders = [];

  // Group fields by their declared group (preserving GROUPS order; unknown groups go last).
  const groupOrder = GROUPS.length
    ? GROUPS.map(g => g.key)
    : [...new Set(SCHEMA.map(f => f.group || 'misc'))];
  const groupMeta = {};
  for (const g of GROUPS) groupMeta[g.key] = g;
  const buckets = {};
  for (const k of groupOrder) buckets[k] = [];
  for (const field of SCHEMA) {
    const k = field.group || 'misc';
    if (!buckets[k]) { buckets[k] = []; groupOrder.push(k); }
    buckets[k].push(field);
  }

  const sectionEls = [];
  const renderedSectionBlocks = new Set();
  for (const k of groupOrder) {
    const fields = buckets[k];
    if (!fields || !fields.length) continue;
    const meta = groupMeta[k] || {label: k, hint: ''};
    const parentSection = meta.section || 'misc';
    if (!renderedSectionBlocks.has(parentSection)) {
      renderedSectionBlocks.add(parentSection);
      const parentLabel = meta.section_label || parentSection;
      const parentHint = meta.section_hint || '';
      $form.append(el('div', {class: 'section-super'},
        el('div', {class: 'section-title'}, parentLabel),
        parentHint ? el('div', {class: 'section-hint'}, parentHint) : null,
      ));
      const navSection = el('div', {class: 'nav-section'}, parentLabel);
      if (parentHint) navSection.append(el('small', {}, parentHint));
      $sidenav.append(navSection);
    }
    const sectionId = 'grp-' + slug(k);
    const section = el('section', {class: 'group-section', id: sectionId},
      el('h2', {}, meta.label || k),
    );
    if (meta.hint) section.append(el('div', {class: 'group-hint'}, meta.hint));
    // Track top renderers in this section so sub-nav builders can hook them.
    const sectionRenderers = [];
    for (const field of fields) {
      const r = renderField(field, config[field.key]);
      topRenderers.push({field, r});
      sectionRenderers.push({field, r});
      section.append(r.node);
    }
    $form.append(section);
    sectionEls.push({id: sectionId, section, label: meta.label || k, hint: meta.hint || ''});

    const link = el('a', {href: '#' + sectionId,
      onclick: (e) => { e.preventDefault();
        document.getElementById(sectionId).scrollIntoView({behavior: 'smooth', block: 'start'});
        history.replaceState(null, '', '#' + sectionId);
      }},
      meta.label || k,
    );
    if (meta.hint) link.append(el('small', {}, meta.hint));
    link.dataset.target = sectionId;
    link.dataset.searchText = ((meta.label || k) + ' ' + (meta.hint || '')).toLowerCase();
    $sidenav.append(link);

    // Build sub-nav for this section. Multiple fields can contribute:
    //   - object fields with `nav_label` get a single static anchor
    //   - object_list fields with `nav_label_keys` expand into one
    //     anchor per item, labelled by the first non-empty key value
    //   - object_map fields expand into one anchor per map key
    const sources = [];
    const changeNotifiers = [];
    for (const {field, r} of sectionRenderers) {
      if (!r) continue;
      if (field.nav_label && r.node) {
        const label = field.nav_label;
        // r.node is the .field wrapper; it already has id="field-<key>".
        sources.push(() => [{itemBox: r.node, label, kind: 'static'}]);
        continue;
      }
      if (!r._items) continue;
      if (field.type === 'object_map') {
        sources.push(() => r._items.map((entry, idx) => ({
          itemBox: entry.itemBox,
          label: ((entry.ki && entry.ki.value) || '').trim() || '(unnamed #' + (idx + 1) + ')',
          idx: idx + 1,
        })));
      } else if (field.type === 'object_list' && field.nav_label_keys && field.nav_label_keys.length) {
        const labelKeys = field.nav_label_keys;
        const joinSep = (typeof field.nav_label_join === 'string') ? field.nav_label_join : '';
        sources.push(() => r._items.map((entry, idx) => {
          let label = '';
          if (joinSep) {
            // Concatenate every non-empty key value with the given separator.
            const parts = [];
            for (const key of labelKeys) {
              const sub = entry.renderer && entry.renderer.getRenderer
                ? entry.renderer.getRenderer(key) : null;
              const v = sub && sub.get ? sub.get() : (sub && sub.read ? sub.read() : '');
              if (v && typeof v === 'string' && v.trim()) parts.push(v.trim());
            }
            label = parts.join(joinSep);
          } else {
            for (const key of labelKeys) {
              const sub = entry.renderer && entry.renderer.getRenderer
                ? entry.renderer.getRenderer(key) : null;
              const v = sub && sub.get ? sub.get() : (sub && sub.read ? sub.read() : '');
              if (v && typeof v === 'string' && v.trim()) { label = v.trim(); break; }
            }
          }
          return {
            itemBox: entry.itemBox,
            label: label || '(unnamed #' + (idx + 1) + ')',
            idx: idx + 1,
          };
        }));
      } else {
        continue;
      }
      if (r._onChange) changeNotifiers.push(r._onChange);
    }
    if (sources.length) {
      const builder = attachSubNav({sectionId, parentLinkEl: link, sources, changeNotifiers});
      subNavBuilders.push(builder);
    }
  }

  // Hook input events to live-refresh sub-nav labels whenever any nav
  // label field (the primary name input, or an object_map key input)
  // changes. Cheap enough on a config form.
  $form.addEventListener('input', (e) => {
    const t = e.target;
    if (!t || t.tagName !== 'INPUT') return;
    for (const b of subNavBuilders) {
      try { b.rebuild(); } catch (err) { console.error(err); }
    }
  });

  // Active-section highlighting via IntersectionObserver. Highlights both
  // the top-level section link and (if scrolled into a specific item) the
  // matching sub-nav entry.
  if (window._navObserver) window._navObserver.disconnect();
  if (window._navItemObserver) window._navItemObserver.disconnect();
  const links = $sidenav.querySelectorAll('a[data-target]');
  const visible = new Set();
  window._navObserver = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) visible.add(e.target.id);
      else visible.delete(e.target.id);
    }
    let active = null;
    for (const {id} of sectionEls) if (visible.has(id)) { active = id; break; }
    if (!active && sectionEls.length) active = sectionEls[0].id;
    links.forEach(a => a.classList.toggle('active', a.dataset.target === active));
  }, {rootMargin: '-30% 0px -55% 0px', threshold: 0});
  for (const {section} of sectionEls) window._navObserver.observe(section);

  // Sub-nav item highlighting.
  const itemVisible = new Set();
  window._navItemObserver = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) itemVisible.add(e.target.id);
      else itemVisible.delete(e.target.id);
    }
    const itemLinks = $sidenav.querySelectorAll('a[data-item-target]');
    itemLinks.forEach(a => a.classList.toggle('active', itemVisible.has(a.dataset.itemTarget)));
  }, {rootMargin: '-30% 0px -55% 0px', threshold: 0});
  // Observe currently-known item boxes; new ones get observed on rebuild.
  function observeItemBoxes() {
    for (const {field, r} of topRenderers) {
      if (!r || !r._items) continue;
      for (const entry of r._items) {
        if (entry.itemBox && entry.itemBox.id) window._navItemObserver.observe(entry.itemBox);
      }
    }
  }
  observeItemBoxes();
  // Re-observe after any list/map mutation.
  for (const b of subNavBuilders) {
    const orig = b.rebuild;
    b.rebuild = function () { orig(); observeItemBoxes(); };
  }

  // Filter wiring.
  const filterInput = document.getElementById('navFilter');
  if (filterInput && !filterInput._wired) {
    filterInput._wired = true;
    filterInput.addEventListener('input', applyNavFilter);
  }
  applyNavFilter();

  // Populate autocomplete datalists now that all fields exist.
  refreshModelDatalist();
  refreshSourceDatalist();
}

function applyNavFilter() {
  const input = document.getElementById('navFilter');
  const q = (input && input.value || '').trim().toLowerCase();
  const sectionLinks = $sidenav.querySelectorAll('a[data-target]');
  let anyVisible = false;
  sectionLinks.forEach(a => {
    const subWrap = a.nextElementSibling && a.nextElementSibling.classList &&
      a.nextElementSibling.classList.contains('subnav') ? a.nextElementSibling : null;
    let matchedSelf = !q || (a.dataset.searchText || '').includes(q);
    let matchedItems = 0;
    if (subWrap) {
      const itemLinks = subWrap.querySelectorAll('a[data-item-target]');
      itemLinks.forEach(il => {
        const m = !q || (il.dataset.searchText || '').includes(q);
        il.style.display = m ? '' : 'none';
        if (m) matchedItems += 1;
      });
      // Hide "(empty)" placeholder when filtering.
      const empty = subWrap.querySelector('.empty');
      if (empty) empty.style.display = q ? 'none' : '';
      subWrap.style.display = (matchedSelf || matchedItems) ? '' : 'none';
    }
    const showLink = matchedSelf || matchedItems > 0;
    a.style.display = showLink ? '' : 'none';
    if (showLink) anyVisible = true;
  });
  // Hide section headings whose links are all filtered out.
  const headings = $sidenav.querySelectorAll('.nav-section');
  headings.forEach(h => {
    let n = h.nextElementSibling;
    let visible = false;
    while (n && !(n.classList && n.classList.contains('nav-section'))) {
      if (n.tagName === 'A' && n.style.display !== 'none') { visible = true; break; }
      n = n.nextElementSibling;
    }
    h.style.display = visible ? '' : 'none';
  });
  // Friendly empty-state.
  let emptyMsg = $sidenav.querySelector('.nav-empty');
  if (!anyVisible && q) {
    if (!emptyMsg) {
      emptyMsg = el('div', {class: 'nav-empty'}, 'No matches');
      $sidenav.append(emptyMsg);
    }
    emptyMsg.style.display = '';
  } else if (emptyMsg) {
    emptyMsg.style.display = 'none';
  }
}

function readForm() {
  const out = {};
  for (const {field, r} of topRenderers) {
    if (!r.isPresent()) continue;
    out[field.key] = r.read();
  }
  return out;
}

// ---- Network -------------------------------------------------------

async function load() {
  setStatus('loading...');
  try {
    const [schemaResp, cfgResp] = await Promise.all([
      fetch(ADMIN_BASE + '/schema'),
      fetch(ADMIN_BASE + '/config'),
    ]);
    if (!schemaResp.ok) throw new Error('schema: ' + await schemaResp.text());
    if (!cfgResp.ok) throw new Error('config: ' + await cfgResp.text());
    const sch = await schemaResp.json();
    // Tolerate the legacy bare-array shape just in case.
    if (Array.isArray(sch)) { SCHEMA = sch; GROUPS = []; }
    else { SCHEMA = sch.fields || []; GROUPS = sch.groups || []; }
    const cfg = await cfgResp.json();
    $path.textContent = cfg._path || '(none)';
    delete cfg._path;
    delete cfg._inactive_ollama_targets;  // legacy field, ignore
    $schemaDump.textContent = JSON.stringify({fields: SCHEMA, groups: GROUPS}, null, 2);
    if (RAW_MODE) {
      $raw.value = JSON.stringify(cfg, null, 2);
    } else {
      renderForm(cfg);
    }
    setStatus('loaded', 'ok');
  } catch (e) {
    setStatus('load failed: ' + e.message, 'err');
  }
}

async function save() {
  setStatus('saving...');
  let payload;
  if (RAW_MODE) {
    try { payload = JSON.parse($raw.value); }
    catch (e) { return setStatus('invalid JSON: ' + e.message, 'err'); }
  } else {
    payload = readForm();
  }
  try {
    const r = await fetch(ADMIN_BASE + '/config', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(await r.text());
    setStatus('saved & reloaded', 'ok');
    load();
  } catch (e) {
    setStatus('save failed: ' + e.message, 'err');
  }
}

document.getElementById('save').onclick = save;
document.getElementById('reload').onclick = load;
document.getElementById('rawmode').onclick = () => {
  RAW_MODE = !RAW_MODE;
  $form.hidden = RAW_MODE;
  $raw.hidden = !RAW_MODE;
  if (RAW_MODE) {
    $raw.style.minHeight = '60vh';
    $raw.style.fontFamily = 'ui-monospace, Consolas, monospace';
    $raw.style.fontSize = '13px';
  }
  load();
};

load();
</script>
</body>
</html>
"""


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


async def _swap_settings(app: FastAPI, new_settings: Settings) -> None:
    """Atomically replace app.state.settings and rebuild client pools."""
    old_clients: Dict[str, AnthropicClient] = dict(getattr(app.state, "clients", {}))
    old_ollama: Dict[str, OllamaClient] = dict(getattr(app.state, "ollama_clients", {}))
    old_llama_cpp: Dict[str, LlamaCppClient] = dict(getattr(app.state, "llama_cpp_clients", {}))
    vram_coordinator = getattr(app.state, "vram_coordinator", None)

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
            max_concurrent_requests=tgt.effective_max_concurrent_requests,
            request_read_timeout_seconds=tgt.request_read_timeout_seconds,
        )

    app.state.settings = new_settings
    app.state.clients = new_clients
    app.state.ollama_clients = new_ollama
    app.state.llama_cpp_clients = new_llama_cpp
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
