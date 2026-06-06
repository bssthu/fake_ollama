"""Async client for ComfyUI workflow-backed image generation.

The client owns optional lifecycle management for a ComfyUI server and
submits API-format workflows through ComfyUI's ``/prompt`` endpoint.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import logging
import mimetypes
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from .config import outbound_cycle_headers
from .process_utils import create_managed_subprocess_shell, terminate_process_tree
from .request_data_log import (
    body_from_bytes,
    body_from_json,
    headers_from_mapping,
    log_data_event,
    request_data_logging_enabled,
)
from .vram import VRAM_IDLE_RECLAIM_SECONDS, VramCoordinator, VramReleaseCandidate

logger = logging.getLogger("fake_ollama")


_WORKFLOW_DIR = Path(__file__).resolve().parent / "workflows"
_DEFAULT_T2I_WORKFLOW = _WORKFLOW_DIR / "z_image_turbo_t2i.json"
_DEFAULT_I2I_WORKFLOW = _WORKFLOW_DIR / "z_image_turbo_i2i.json"


@dataclass
class _LoadedModel:
    model: str
    estimated_vram_gb: float
    last_used_monotonic: float


@dataclass
class ComfyUIImage:
    data: bytes
    filename: str
    subfolder: str
    image_type: str
    mime_type: str

    @property
    def b64_json(self) -> str:
        return base64.b64encode(self.data).decode("ascii")


class ComfyUIClient:
    """Small lifecycle-aware wrapper around ComfyUI's HTTP API."""

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: str = "",
        timeout: float = 300.0,
        trust_env: bool = False,
        auto_start: bool = False,
        start_command: Optional[str] = None,
        stop_command: Optional[str] = None,
        idle_timeout_seconds: Optional[float] = None,
        startup_timeout_seconds: float = 120.0,
        health_path: str = "/system_stats",
        cwd: Optional[str] = None,
        target_name: str = "comfyui",
        workflow_config: Optional[Dict[str, Any]] = None,
        vram_coordinator: Optional[VramCoordinator] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._auth_token = auth_token
        self._timeout = timeout
        self._trust_env = trust_env
        self._auto_start = auto_start
        self._start_command = start_command
        self._stop_command = stop_command
        self._idle_timeout = idle_timeout_seconds
        self._startup_timeout = startup_timeout_seconds
        self._health_path = health_path if health_path.startswith("/") else "/" + health_path
        self._cwd = cwd
        self.target_id = f"comfyui:{target_name}"
        self._workflow_config = dict(workflow_config or {})
        self._vram_coordinator = vram_coordinator
        self._client_id = f"fake-ollama-{uuid.uuid4().hex}"
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(timeout=timeout, trust_env=trust_env)
            self._owns_client = True
        self._process: Optional[asyncio.subprocess.Process] = None
        self._started_by_us = False
        self._start_lock = asyncio.Lock()
        self._active = 0
        self._request_refs = 0
        self._last_used = time.monotonic()
        self._loaded_model: Optional[_LoadedModel] = None
        self._shutdown_requested = False
        if self._vram_coordinator is not None:
            self._vram_coordinator.register(self)

    @property
    def idle_timeout_seconds(self) -> Optional[float]:
        return self._idle_timeout

    @property
    def active_requests(self) -> int:
        return self._active

    @property
    def last_used_monotonic(self) -> float:
        return self._last_used

    def begin_shutdown(self) -> None:
        self._shutdown_requested = True

    async def aclose(self) -> None:
        if self._vram_coordinator is not None:
            self._vram_coordinator.unregister(self)
        if self._owns_client:
            await self._client.aclose()
        await self.stop_if_owned()

    def has_vram_reservation(self, model: str) -> bool:
        return self._loaded_model is not None

    def _begin_request_lifecycle(self) -> None:
        self._request_refs += 1

    def _end_request_lifecycle(self) -> None:
        self._request_refs -= 1
        self._last_used = time.monotonic()

    async def _ensure_vram(
        self, model: Optional[str], estimated_vram_gb: Optional[float]
    ) -> None:
        if self._vram_coordinator is None or estimated_vram_gb is None:
            return
        await self._vram_coordinator.ensure_available(
            self,
            model=model or "",
            estimated_vram_gb=estimated_vram_gb,
        )

    def _mark_vram_reserved(
        self, model: Optional[str], estimated_vram_gb: Optional[float]
    ) -> None:
        if not model or estimated_vram_gb is None:
            return
        self._loaded_model = _LoadedModel(
            model=model,
            estimated_vram_gb=estimated_vram_gb,
            last_used_monotonic=time.monotonic(),
        )
        if self._vram_coordinator is not None:
            self._vram_coordinator.confirm_loaded(self.target_id, model)

    def _touch_vram_reservation(self, model: Optional[str]) -> None:
        if self._loaded_model is not None:
            self._loaded_model.last_used_monotonic = time.monotonic()

    def _discard_vram_pending(self, model: Optional[str]) -> None:
        if not model or self._vram_coordinator is None:
            return
        self._vram_coordinator.discard_pending(self.target_id, model)

    def _clear_all_vram_state(self) -> None:
        loaded = self._loaded_model
        self._loaded_model = None
        if loaded is not None and self._vram_coordinator is not None:
            self._vram_coordinator.discard_pending(self.target_id, loaded.model)

    def vram_release_candidates(
        self, *, now: float, idle_seconds: float
    ) -> list[VramReleaseCandidate]:
        if self._active or self._request_refs or self._loaded_model is None:
            return []
        loaded = self._loaded_model
        if now - loaded.last_used_monotonic < idle_seconds:
            return []
        return [
            VramReleaseCandidate(
                owner_id=self.target_id,
                model=loaded.model,
                estimated_vram_gb=loaded.estimated_vram_gb,
                last_used_monotonic=loaded.last_used_monotonic,
                release=self._release_for_vram,
            )
        ]

    def vram_force_release_candidates(
        self, *, now: float
    ) -> list[VramReleaseCandidate]:
        if self._loaded_model is None:
            return []
        loaded = self._loaded_model
        return [
            VramReleaseCandidate(
                owner_id=self.target_id,
                model=loaded.model,
                estimated_vram_gb=loaded.estimated_vram_gb,
                last_used_monotonic=loaded.last_used_monotonic,
                release=lambda: self._release_for_vram(force=True),
            )
        ]

    def loaded_model_snapshots(
        self,
        *,
        now: Optional[float] = None,
        idle_reclaim_seconds: float = VRAM_IDLE_RECLAIM_SECONDS,
    ) -> list[dict[str, object]]:
        if self._loaded_model is None:
            return []
        now = time.monotonic() if now is None else now
        loaded = self._loaded_model
        in_flight = bool(self._active or self._request_refs)
        idle_seconds = 0.0 if in_flight else max(0.0, now - loaded.last_used_monotonic)
        return [
            {
                "backend": "comfyui",
                "target_id": self.target_id,
                "model": loaded.model,
                "estimated_vram_gb": loaded.estimated_vram_gb,
                "estimated_vram_mib": loaded.estimated_vram_gb * 1024.0,
                "active_requests": self._active,
                "request_refs": self._request_refs,
                "idle_seconds": idle_seconds,
                "reclaimable": (
                    not in_flight and idle_seconds >= idle_reclaim_seconds
                ),
            }
        ]

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"content-type": "application/json"}
        if self._auth_token:
            headers["authorization"] = f"Bearer {self._auth_token}"
            headers["x-api-key"] = self._auth_token
        headers.update(outbound_cycle_headers())
        return headers

    async def _healthy(self) -> bool:
        try:
            resp = await self._client.get(
                self._base + self._health_path,
                timeout=min(5.0, self._timeout),
                headers=self._headers(),
            )
            return 200 <= resp.status_code < 300
        except httpx.HTTPError:
            return False

    def _stderr_log_path(self) -> Optional[Path]:
        parsed = urlparse(self._base)
        port = parsed.port
        host = (parsed.hostname or "").strip()
        safe_target = re.sub(r"[^A-Za-z0-9._-]+", "_", self.target_id).strip("_")
        if port:
            stem = f"comfyui-{port}"
        elif safe_target:
            stem = f"comfyui-{safe_target}"
        else:
            return None
        if host and host not in {"127.0.0.1", "localhost", "::1"}:
            stem = f"{stem}-{re.sub(r'[^A-Za-z0-9._-]+', '_', host)}"
        return Path("logs") / f"{stem}.err.log"

    async def _ensure_ready(self) -> None:
        if await self._healthy():
            return
        self._clear_all_vram_state()
        if self._shutdown_requested:
            raise httpx.ConnectError(
                "ComfyUI target is unavailable and fake-ollama is shutting down; "
                "refusing auto-start"
            )
        if not (self._auto_start and self._start_command):
            return

        async with self._start_lock:
            if await self._healthy():
                return
            if self._shutdown_requested:
                raise httpx.ConnectError(
                    "ComfyUI target is unavailable and fake-ollama is shutting down; "
                    "refusing auto-start"
            )
            if self._process is None or self._process.returncode is not None:
                logger.info("starting ComfyUI target: %s", self._start_command)
                stderr_log = self._stderr_log_path()
                if stderr_log is not None:
                    logger.info("[%s] capturing ComfyUI stderr to %s", self.target_id, stderr_log)
                self._process = await create_managed_subprocess_shell(
                    self._start_command,
                    cwd=self._cwd,
                    stderr=stderr_log,
                )
                self._started_by_us = True

            deadline = time.monotonic() + self._startup_timeout
            while time.monotonic() < deadline:
                if await self._healthy():
                    logger.info("ComfyUI target ready at %s", self._base)
                    return
                await asyncio.sleep(1.0)
            raise httpx.ConnectError(
                f"ComfyUI target did not become healthy within {self._startup_timeout}s"
            )

    async def generate_image(
        self,
        *,
        model: str,
        prompt: str,
        width: int,
        height: int,
        n: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        estimated_vram_gb: Optional[float] = None,
    ) -> List[ComfyUIImage]:
        workflow = self._build_workflow(
            "text_to_image",
            prompt=prompt,
            width=width,
            height=height,
            batch_size=n,
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
        )
        return await self._run_workflow(
            workflow,
            model=model,
            estimated_vram_gb=estimated_vram_gb,
            operation="generate_image",
        )

    async def edit_image(
        self,
        *,
        model: str,
        prompt: str,
        image_bytes: bytes,
        filename: str,
        width: int,
        height: int,
        n: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        estimated_vram_gb: Optional[float] = None,
    ) -> List[ComfyUIImage]:
        self._begin_request_lifecycle()
        try:
            await self._ensure_vram(model, estimated_vram_gb)
            try:
                await self._ensure_ready()
            except BaseException:
                self._discard_vram_pending(model)
                raise
            self._active += 1
            try:
                upload_name = await self._upload_image(image_bytes, filename)
                images: List[ComfyUIImage] = []
                self._mark_vram_reserved(model, estimated_vram_gb)
                for idx in range(max(1, n)):
                    workflow = self._build_workflow(
                        "image_to_image",
                        prompt=prompt,
                        width=width,
                        height=height,
                        batch_size=1,
                        seed=seed + idx,
                        steps=steps,
                        cfg=cfg,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        denoise=denoise,
                        image_name=upload_name,
                    )
                    images.extend(
                        await self._submit_and_collect(
                            workflow,
                            operation="edit_image",
                        )
                    )
                return images[: max(1, n)]
            finally:
                self._active -= 1
        finally:
            self._touch_vram_reservation(model)
            self._end_request_lifecycle()

    async def _run_workflow(
        self,
        workflow: Dict[str, Any],
        *,
        model: str,
        estimated_vram_gb: Optional[float],
        operation: str,
    ) -> List[ComfyUIImage]:
        self._begin_request_lifecycle()
        try:
            await self._ensure_vram(model, estimated_vram_gb)
            try:
                await self._ensure_ready()
            except BaseException:
                self._discard_vram_pending(model)
                raise
            self._active += 1
            try:
                self._mark_vram_reserved(model, estimated_vram_gb)
                return await self._submit_and_collect(workflow, operation=operation)
            finally:
                self._active -= 1
        finally:
            self._touch_vram_reservation(model)
            self._end_request_lifecycle()

    async def _upload_image(self, image_bytes: bytes, filename: str) -> str:
        safe_name = Path(filename or "input.png").name or "input.png"
        content_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
        url = f"{self._base}/upload/image"
        headers = dict(outbound_cycle_headers())
        if self._auth_token:
            headers["authorization"] = f"Bearer {self._auth_token}"
            headers["x-api-key"] = self._auth_token
        if request_data_logging_enabled():
            log_data_event(
                "backend_request",
                backend="comfyui",
                target_id=self.target_id,
                operation="upload_image",
                method="POST",
                url=url,
                headers=headers_from_mapping(headers),
                body={"bytes": len(image_bytes), "filename": safe_name},
            )
        try:
            resp = await self._client.post(
                url,
                data={"type": "input", "overwrite": "true"},
                files={"image": (safe_name, image_bytes, content_type)},
                headers=headers,
            )
        except BaseException as exc:
            log_data_event(
                "backend_error",
                backend="comfyui",
                target_id=self.target_id,
                operation="upload_image",
                method="POST",
                url=url,
                error=f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}",
            )
            raise
        if request_data_logging_enabled():
            log_data_event(
                "backend_response_start",
                backend="comfyui",
                target_id=self.target_id,
                operation="upload_image",
                method="POST",
                url=url,
                status=resp.status_code,
                headers=headers_from_mapping(dict(resp.headers)),
            )
            log_data_event(
                "backend_response_body",
                backend="comfyui",
                target_id=self.target_id,
                operation="upload_image",
                method="POST",
                url=url,
                body=body_from_bytes(resp.content),
            )
            log_data_event(
                "backend_response_end",
                backend="comfyui",
                target_id=self.target_id,
                operation="upload_image",
                method="POST",
                url=url,
                status=resp.status_code,
                response_bytes=len(resp.content),
            )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("name") or safe_name)

    async def _submit_and_collect(
        self,
        workflow: Dict[str, Any],
        *,
        operation: str,
    ) -> List[ComfyUIImage]:
        prompt_id = await self._queue_prompt(workflow, operation=operation)
        history = await self._wait_for_history(prompt_id, operation=operation)
        return await self._collect_images(history, operation=operation)

    async def _queue_prompt(self, workflow: Dict[str, Any], *, operation: str) -> str:
        url = f"{self._base}/prompt"
        body = {"prompt": workflow, "client_id": self._client_id}
        headers = self._headers()
        if request_data_logging_enabled():
            log_data_event(
                "backend_request",
                backend="comfyui",
                target_id=self.target_id,
                operation=operation,
                method="POST",
                url=url,
                headers=headers_from_mapping(headers),
                body=body_from_json(body),
            )
        try:
            resp = await self._client.post(url, json=body, headers=headers)
        except BaseException as exc:
            log_data_event(
                "backend_error",
                backend="comfyui",
                target_id=self.target_id,
                operation=operation,
                method="POST",
                url=url,
                error=f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}",
            )
            raise
        if request_data_logging_enabled():
            log_data_event(
                "backend_response_start",
                backend="comfyui",
                target_id=self.target_id,
                operation=operation,
                method="POST",
                url=url,
                status=resp.status_code,
                headers=headers_from_mapping(dict(resp.headers)),
            )
            log_data_event(
                "backend_response_body",
                backend="comfyui",
                target_id=self.target_id,
                operation=operation,
                method="POST",
                url=url,
                body=body_from_bytes(resp.content),
            )
            log_data_event(
                "backend_response_end",
                backend="comfyui",
                target_id=self.target_id,
                operation=operation,
                method="POST",
                url=url,
                status=resp.status_code,
                response_bytes=len(resp.content),
            )
        resp.raise_for_status()
        data = resp.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise httpx.ProtocolError("ComfyUI /prompt response did not include prompt_id")
        return str(prompt_id)

    async def _wait_for_history(
        self, prompt_id: str, *, operation: str
    ) -> Dict[str, Any]:
        prompt_timeout = float(self._workflow_config.get("prompt_timeout_seconds", 600.0))
        poll_interval = float(self._workflow_config.get("poll_interval_seconds", 0.5))
        deadline = time.monotonic() + prompt_timeout
        url = f"{self._base}/history/{prompt_id}"
        headers = self._headers()
        while time.monotonic() < deadline:
            resp = await self._client.get(url, headers=headers)
            if resp.status_code >= 400:
                resp.raise_for_status()
            data = resp.json()
            entry = data.get(prompt_id)
            if isinstance(entry, dict):
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise httpx.ProtocolError(
                        f"ComfyUI workflow failed: {json.dumps(status, ensure_ascii=False)}"
                    )
                outputs = entry.get("outputs")
                if isinstance(outputs, dict) and outputs:
                    return entry
            await asyncio.sleep(max(0.05, poll_interval))
        log_data_event(
            "backend_error",
            backend="comfyui",
            target_id=self.target_id,
            operation=operation,
            method="GET",
            url=url,
            error=f"history timeout after {prompt_timeout}s for prompt_id={prompt_id}",
        )
        raise httpx.TimeoutException(
            f"ComfyUI prompt {prompt_id} did not finish within {prompt_timeout}s"
        )

    async def _collect_images(
        self, history_entry: Dict[str, Any], *, operation: str
    ) -> List[ComfyUIImage]:
        outputs = history_entry.get("outputs") or {}
        save_node_id = str(self._workflow_config.get("save_image_node_id", "9"))
        ordered_nodes: List[Any] = []
        if save_node_id in outputs:
            ordered_nodes.append(outputs[save_node_id])
        ordered_nodes.extend(v for k, v in outputs.items() if str(k) != save_node_id)

        images: List[ComfyUIImage] = []
        for node in ordered_nodes:
            if not isinstance(node, dict):
                continue
            for item in node.get("images") or []:
                if not isinstance(item, dict):
                    continue
                filename = str(item.get("filename") or "")
                if not filename:
                    continue
                subfolder = str(item.get("subfolder") or "")
                image_type = str(item.get("type") or "output")
                data = await self._view_image(
                    filename=filename,
                    subfolder=subfolder,
                    image_type=image_type,
                    operation=operation,
                )
                mime_type = mimetypes.guess_type(filename)[0] or "image/png"
                images.append(
                    ComfyUIImage(
                        data=data,
                        filename=filename,
                        subfolder=subfolder,
                        image_type=image_type,
                        mime_type=mime_type,
                    )
                )
        if not images:
            raise httpx.ProtocolError("ComfyUI history did not contain output images")
        return images

    async def _view_image(
        self,
        *,
        filename: str,
        subfolder: str,
        image_type: str,
        operation: str,
    ) -> bytes:
        url = f"{self._base}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": image_type}
        headers = dict(outbound_cycle_headers())
        if self._auth_token:
            headers["authorization"] = f"Bearer {self._auth_token}"
            headers["x-api-key"] = self._auth_token
        resp = await self._client.get(url, params=params, headers=headers)
        if resp.status_code >= 400:
            resp.raise_for_status()
        log_data_event(
            "backend_response_body",
            backend="comfyui",
            target_id=self.target_id,
            operation=operation,
            method="GET",
            url=url,
            body={"bytes": len(resp.content), "filename": filename},
        )
        return bytes(resp.content)

    def _build_workflow(self, mode: str, **params: Any) -> Dict[str, Any]:
        path_key = (
            "text_to_image_workflow_path"
            if mode == "text_to_image"
            else "image_to_image_workflow_path"
        )
        default_path = _DEFAULT_T2I_WORKFLOW if mode == "text_to_image" else _DEFAULT_I2I_WORKFLOW
        workflow = self._load_workflow(self._workflow_config.get(path_key), default_path)

        self._set(workflow, "unet_node_id", "28", "unet_name", self._workflow_config.get("diffusion_model"))
        self._set(workflow, "unet_node_id", "28", "weight_dtype", self._workflow_config.get("diffusion_weight_dtype"))
        self._set(workflow, "clip_node_id", "30", "clip_name", self._workflow_config.get("text_encoder_model"))
        self._set(workflow, "clip_node_id", "30", "type", self._workflow_config.get("text_encoder_type"))
        self._set(workflow, "clip_node_id", "30", "device", self._workflow_config.get("text_encoder_device"))
        self._set(workflow, "vae_node_id", "29", "vae_name", self._workflow_config.get("vae_model"))
        self._set(workflow, "prompt_node_id", "27", "text", params["prompt"])
        self._set(workflow, "sampling_node_id", "11", "shift", params.get("shift", self._workflow_config.get("default_shift", 3.0)))
        self._set(workflow, "ksampler_node_id", "3", "seed", int(params["seed"]))
        self._set(workflow, "ksampler_node_id", "3", "steps", int(params["steps"]))
        self._set(workflow, "ksampler_node_id", "3", "cfg", float(params["cfg"]))
        self._set(workflow, "ksampler_node_id", "3", "sampler_name", params["sampler_name"])
        self._set(workflow, "ksampler_node_id", "3", "scheduler", params["scheduler"])
        self._set(workflow, "ksampler_node_id", "3", "denoise", float(params["denoise"]))
        self._set(workflow, "save_image_node_id", "9", "filename_prefix", self._workflow_config.get("output_prefix"))

        if mode == "text_to_image":
            self._set(workflow, "latent_node_id", "13", "width", int(params["width"]))
            self._set(workflow, "latent_node_id", "13", "height", int(params["height"]))
            self._set(workflow, "latent_node_id", "13", "batch_size", int(params["batch_size"]))
        else:
            self._set(workflow, "load_image_node_id", "12", "image", params["image_name"])
            self._set(workflow, "image_scale_node_id", "14", "width", int(params["width"]))
            self._set(workflow, "image_scale_node_id", "14", "height", int(params["height"]))
            self._set(workflow, "image_scale_node_id", "14", "upscale_method", self._workflow_config.get("image_upscale_method"))
            self._set(workflow, "image_scale_node_id", "14", "crop", self._workflow_config.get("image_crop"))
        return workflow

    @staticmethod
    def _load_workflow(path: Any, default_path: Path) -> Dict[str, Any]:
        src = Path(str(path)) if path else default_path
        raw = src.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"ComfyUI workflow {src} must be an API prompt object")
        return copy.deepcopy(data)

    def _set(
        self,
        workflow: Dict[str, Any],
        node_key: str,
        default_node_id: str,
        input_name: str,
        value: Any,
    ) -> None:
        if value is None:
            return
        node_id = str(self._workflow_config.get(node_key) or default_node_id)
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            raise ValueError(f"ComfyUI workflow is missing node {node_id!r}")
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            raise ValueError(f"ComfyUI workflow node {node_id!r} has invalid inputs")
        inputs[input_name] = value

    async def stop_if_idle(self) -> None:
        if not self._idle_timeout or self._active or self._request_refs:
            return
        idle_for = time.monotonic() - self._last_used
        if idle_for < self._idle_timeout:
            return
        if not (self._started_by_us or self._stop_command):
            return
        logger.info(
            "stopping idle ComfyUI target %s after %.1fs idle",
            self.target_id,
            idle_for,
        )
        await self.stop_if_owned()

    async def release_for_vram(self) -> bool:
        return await self._release_for_vram()

    async def _release_for_vram(self, *, force: bool = False) -> bool:
        if not force and (self._active or self._request_refs):
            return False
        if await self._free_memory():
            self._clear_all_vram_state()
            return True
        if self._started_by_us or self._stop_command:
            return await self.stop_if_owned()
        return False

    async def _free_memory(self) -> bool:
        try:
            resp = await self._client.post(
                f"{self._base}/free",
                json={"unload_models": True, "free_memory": True},
                headers=self._headers(),
                timeout=min(30.0, self._timeout),
            )
            await resp.aread()
            if 200 <= resp.status_code < 300:
                logger.info("requested ComfyUI model unload/free_memory at %s", self._base)
                return True
            logger.warning(
                "ComfyUI /free returned status=%s body=%s",
                resp.status_code,
                resp.text,
            )
        except httpx.HTTPError as exc:
            logger.warning("failed to call ComfyUI /free at %s: %s", self._base, exc)
        return False

    async def stop_if_owned(self) -> bool:
        if self._stop_command:
            logger.info("stopping ComfyUI target with stop_command")
            proc = await asyncio.create_subprocess_shell(
                self._stop_command,
                cwd=self._cwd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            returncode = await proc.wait()
            if returncode == 0:
                self._started_by_us = False
                self._clear_all_vram_state()
                return True
            logger.warning("ComfyUI target stop_command exited with status %s", returncode)
            return False

        if self._process is not None and self._started_by_us:
            if self._process.returncode is None:
                logger.info("terminating owned ComfyUI target process tree")
                if not await terminate_process_tree(self._process, timeout=10.0):
                    logger.warning("failed to terminate owned ComfyUI target process tree")
                    return False
                self._started_by_us = False
                self._clear_all_vram_state()
                return True
            logger.warning(
                "owned ComfyUI target launcher process already exited; cannot stop "
                "remaining detached server process without stop_command"
            )
            return False
        return False
