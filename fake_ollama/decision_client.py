"""Native typed-decision transport with protocol-neutral runtime ownership.

No chat completion method is used here. Wire codecs are supplied by decisions;
all process, request, queue and resource lifetimes go through ManagedHTTPRuntime.
Optional StartLux code is loaded only when that runtime is configured.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Protocol, TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from .config import DecisionTarget
from .managed_http_runtime import ManagedHTTPRuntime
from .vram import LocalTargetResourceError

if TYPE_CHECKING:
    from .decisions import DecisionRequest, DecisionResult

logger = logging.getLogger("fake_ollama")


class DecisionClientProtocol(Protocol):
    async def decide(self, request: DecisionRequest, *, estimated_vram_gb: float | None = None,
                     estimated_memory_gb: float | None = None) -> DecisionResult: ...
    async def aclose(self) -> None: ...
    def capability_snapshot(self) -> dict[str, Any]: ...


def _local_model_path(target: DecisionTarget) -> Path:
    if not target.model_path:
        raise LocalTargetResourceError("A local model_path is required; download weights separately before starting decisions",
                                       error_type="decision_model_missing")
    path = Path(target.model_path).expanduser()
    if not path.is_absolute():
        path = Path(target.cwd or os.getcwd()) / path
    if not path.exists():
        raise LocalTargetResourceError(f"Decision model_path does not exist: {path}. Download weights separately; automatic downloads are disabled",
                                       error_type="decision_model_missing")
    if target.runtime == "llama_cpp" and not path.is_file():
        raise LocalTargetResourceError("llama_cpp decision model_path must be a local GGUF file", error_type="decision_model_missing")
    if target.runtime in {"vllm", "startlux"} and not path.is_dir():
        raise LocalTargetResourceError("This managed decision runtime requires a complete local model directory", error_type="decision_model_missing")
    return path.resolve()


def build_native_argv(target: DecisionTarget, *, validate: bool = False) -> list[str] | None:
    """No shell expansion or remote model identifiers in built-in launchers."""
    if target.runtime == "external" or not target.auto_start:
        return None
    if target.runtime == "startlux":
        from .decision_launch import build_startlux_argv
        return build_startlux_argv(target)
    path = _local_model_path(target) if validate else Path(target.model_path or "")
    host = urlparse(target.base_url).hostname or "127.0.0.1"
    port = urlparse(target.base_url).port or 80
    if target.runtime == "llama_cpp":
        return [str(target.binary_path), "--model", str(path), "--host", host,
                "--port", str(port), "--alias", target.models[0].wire_id, *target.start_args]
    return [target.python_path or sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(path), "--host", host, "--port", str(port),
            "--served-model-name", *dict.fromkeys(entry.wire_id for entry in target.models), *target.start_args]


def _version(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("version", "server_version", "build_version"):
        if isinstance(payload.get(key), str) and payload[key].strip():
            return payload[key].strip()
    for key in ("runtime", "build_info"):
        value = _version(payload.get(key))
        if value:
            return value
    return None


class DecisionClient(ManagedHTTPRuntime):
    def __init__(self, target: DecisionTarget, *, timeout: float = 300.0,
                 trust_env: bool = False, vram_coordinator=None, memory_coordinator=None,
                 ownership_active: bool = True, client: httpx.AsyncClient | None = None):
        self.target = target.model_copy(deep=True)
        self._config_signature = target.model_dump(mode="json")
        self._capabilities = target.capability_metadata()
        self._probe_lock = asyncio.Lock()
        self._probe_at: float | None = None
        self._probe = {"version": None, "endpoint_verified": False, "route_advertised": False, "status": "not_probed", "warnings": [], "errors": []}
        # Validation of local weights and optional packages happens on use, not
        # import/config parsing or reload preparation. Prepared clients never spawn.
        argv = build_native_argv(target, validate=False) if target.runtime != "startlux" else None
        env = None
        if target.runtime != "external":
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        super().__init__(target.base_url, auth_token=target.auth_token, timeout=timeout,
            trust_env=trust_env, auto_start=target.auto_start, start_command=target.start_command,
            start_argv=argv, stop_command=target.stop_command,
            manage_external_runtime=target.manage_external_runtime,
            stop_timeout_seconds=target.stop_timeout_seconds, ownership_active=ownership_active,
            idle_timeout_seconds=target.idle_timeout_seconds,
            startup_timeout_seconds=target.startup_timeout_seconds, health_path=target.health_path,
            cwd=target.cwd, launch_env=env, target_name=target.name, target_prefix="decision",
            target_log_label="decision", vram_coordinator=vram_coordinator,
            memory_coordinator=memory_coordinator, client=client,
            max_concurrent_requests=target.max_concurrent_requests,
            request_read_timeout_seconds=target.request_read_timeout_seconds)
        # StartLux argv includes a packaged launcher and is prepared lazily.
        if target.runtime == "startlux" and target.auto_start:
            self._start_argv = ["startlux (prepared on use)"]

    async def prepare_start(self) -> None:
        try:
            if self.target.runtime != "external":
                _local_model_path(self.target)
                if self.target.runtime == "startlux":
                    from .decision_launch import validate_startlux_setup
                    validate_startlux_setup(self.target)
                self._start_argv = build_native_argv(self.target, validate=True)
        except ValueError as exc:
            raise LocalTargetResourceError(str(exc), error_type="decision_setup_invalid") from exc
        self._probe_at = None
        self._capabilities = self.target.capability_metadata()
        self._probe = {"version": None, "endpoint_verified": False, "route_advertised": False, "status": "not_probed", "warnings": [], "errors": []}

    def matches(self, target: DecisionTarget, *, timeout: float, trust_env: bool) -> bool:
        return self._config_signature == target.model_dump(mode="json") and self._timeout == timeout and self._trust_env == trust_env

    def capability_snapshot(self) -> dict[str, Any]:
        return {**self.target.capability_metadata(), **self._capabilities, **self._probe}

    async def probe_capabilities(self, *, force: bool = False) -> dict[str, Any]:
        """Probe read-only schema/version routes; unknown versions stay advisory.

        A schema POST path only advertises a route. Only a successfully decoded
        actual response verifies the protocol for the requested model/questions.
        """
        if not force and self._probe_at is not None and time.monotonic() - self._probe_at < self.target.probe_interval_seconds:
            return self.capability_snapshot()
        async with self._probe_lock:
            if not force and self._probe_at is not None and time.monotonic() - self._probe_at < self.target.probe_interval_seconds:
                return self.capability_snapshot()
            advertised = False
            verified = self._probe.get("endpoint_verified", False)
            version = None
            schema_seen = False
            warnings = []
            paths = ["/openapi.json", "/version", "/props"]
            if self.target.runtime == "startlux":
                paths.append(self.target.health_path)
            runtime_metadata = {}
            for path in paths:
                try:
                    response = await self._client.get(self._base + path, headers=self._headers(), timeout=min(3.0, self._timeout))
                    if response.status_code != 200:
                        continue
                    payload = response.json()
                    if path != "/openapi.json":
                        version = version or _version(payload)
                    if path == "/openapi.json" and isinstance(payload, dict) and isinstance(payload.get("paths"), dict):
                        schema_seen = True
                        methods = payload["paths"].get(self.target.effective_endpoint_path)
                        advertised = isinstance(methods, dict) and "post" in methods
                    if self.target.runtime == "startlux" and path == self.target.health_path:
                        from .decision_launch import derive_startlux_capabilities
                        restrictions = derive_startlux_capabilities(payload)
                        # Replace only fields that this health response actually
                        # establishes. Missing/redacted/failed probes retain the
                        # previous restriction rather than inventing support.
                        if isinstance(payload, dict) and payload.get("status") == "ok":
                            declared = self.target.capability_metadata()
                            if payload.get("kind") in {"instant", "auto_think", "v1"} or payload.get("prompt_attention") in {"causal", "bidirectional"}:
                                self._capabilities["think_modes"] = declared["think_modes"]
                            if isinstance(payload.get("images"), bool):
                                self._capabilities["modalities"] = declared["modalities"]
                        for field, allowed in restrictions.items():
                            self._capabilities[field] = [item for item in self.target.decision_capabilities.get(field, []) if item in allowed]
                        if isinstance(payload, dict):
                            runtime_metadata = {key: payload[key] for key in ("kind", "prompt_attention", "images", "context_length") if key in payload}
                except (httpx.HTTPError, ValueError, TypeError):
                    continue
            from .decision_launch import verified_runtime_support
            if version is not None and self._probe.get("version") is not None and version != self._probe["version"]:
                verified = False
            support = verified_runtime_support(self.target.runtime, version, self.target.dialect,
                                               endpoint_verified=verified)
            # Release evidence cannot disprove a vendor/custom backport. Keep
            # it actionable but verify with the real typed request below.
            errors = []
            warnings.extend(support.get("warnings", []))
            warnings.extend(support.get("errors", []))
            if schema_seen and not advertised:
                # Some gateways publish incomplete schemas: an actual native
                # POST may still succeed. Do not downgrade to chat silently.
                warnings.append(f"Upstream schema does not advertise POST {self.target.effective_endpoint_path}; the native request will verify support")
            if not verified:
                warnings.append("Decision endpoint is not yet verified; health/version alone do not establish decision support")
            self._probe = {**support, "version": version, "endpoint_verified": verified,
                           "route_advertised": advertised,
                           "runtime_metadata": runtime_metadata,
                           "status": "supported" if verified else "unverified",
                           "warnings": list(dict.fromkeys(warnings)), "errors": errors}
            self._probe_at = time.monotonic()
            for warning in self._probe["warnings"]:
                logger.warning("[%s] %s", self.target_id, warning)
            return self.capability_snapshot()

    def _validate_capabilities(self, request: DecisionRequest, *, declared_only: bool = False) -> None:
        from .decisions import DecisionValidationError, request_has_images
        capabilities = self.target.capability_metadata() if declared_only else self._capabilities
        if capabilities["image_format"] == "startlux_images" and request.options.get("assets"):
            raise DecisionValidationError("Official StartLux uses top-level images, not TypeSafe assets; supply its native images format to avoid losing visual input")
        unsupported = {question.type for question in request.questions} - set(capabilities["question_types"])
        if unsupported:
            raise DecisionValidationError("This decision target does not declare question types: " + ", ".join(sorted(unsupported)))
        if request_has_images(request) and "image" not in capabilities["modalities"]:
            raise DecisionValidationError("This decision target does not declare image support")
        if "images" in request.options:
            if capabilities["image_format"] != "startlux_images":
                raise DecisionValidationError("Top-level images require explicit image_format=startlux_images")
            images = request.options["images"]
            if not isinstance(images, list) or any(not isinstance(item, str) for item in images):
                raise DecisionValidationError("Native images must be a list of image strings")
        if "think_mode" in request.options:
            if "think_mode" not in capabilities["extensions"] or request.options["think_mode"] not in capabilities["think_modes"]:
                raise DecisionValidationError("This target does not declare the requested think_mode; Instant and AutoThink capabilities differ")
        if request.options.get("extensions"):
            extensions = request.options["extensions"]
            if not isinstance(extensions, dict) or set(extensions) - set(capabilities["extensions"]):
                raise DecisionValidationError("This target does not declare the requested decision extensions")

    async def decide(self, request: DecisionRequest, *, estimated_vram_gb=None,
                     estimated_memory_gb=None) -> DecisionResult:
        from .decisions import NativeSystemOneCodec, OpenAIDecisionsCodec
        self._validate_capabilities(request, declared_only=True)
        codec = NativeSystemOneCodec() if self.target.dialect == "systemone" else OpenAIDecisionsCodec()
        payload = codec.encode_request(request, model=request.model)
        headers = self._headers()
        if request.options.get("assets"):
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        async with self.request_lease(request.model, estimated_vram_gb=estimated_vram_gb,
                                      estimated_memory_gb=estimated_memory_gb):
            await self.probe_capabilities()
            self._validate_capabilities(request)
            response = await self.post_until_complete(self.target.effective_endpoint_path, payload, headers=headers)
            if response.status_code in {404, 405, 501}:
                self._probe.update(endpoint_verified=False, status="unsupported")
                raise LocalTargetResourceError(
                    f"{self.target_id} does not support POST {self.target.effective_endpoint_path} "
                    f"({response.status_code}). Configure a native {self.target.dialect} decision endpoint; "
                    "llama.cpp needs a compatible decision build (v0.6.0+), native vLLM needs a post-merge build "
                    "(stable 0.31.0 predates native support). An official StartLux wrapper is a separate runtime. "
                    "Chat completion endpoints are not interchangeable with decision endpoints.",
                    error_type="decision_endpoint_unsupported", status_code=502)
            response.raise_for_status()
            try:
                result = codec.decode_response(response.json(), request)
            except (ValueError, TypeError, KeyError) as exc:
                raise LocalTargetResourceError(f"Invalid {self.target.dialect} decision response: {exc}",
                                               error_type="decision_response_invalid", status_code=502) from exc
            self._probe.update(endpoint_verified=True, status="supported", errors=[])
            if self.target.runtime == "startlux":
                result.source = "official_startlux"
            result.metadata.update({"runtime": self.target.runtime, "dialect": self.target.dialect,
                                    "version": self._probe.get("version"),
                                    "endpoint_verified": True,
                                    "source_url": self.target.source_url, "license": self.target.license})
            return result
