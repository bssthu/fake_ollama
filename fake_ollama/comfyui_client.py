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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .comfyui_presets import WorkflowSpec, nearest_ratio, resolve_workflows
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

# Request parameters that must be coerced to a concrete numeric type before
# they are dropped into a workflow node (everything else is passed through as
# the string/value the caller supplied).
_PARAM_COERCE = {
    "seed": int,
    "steps": int,
    "width": int,
    "height": int,
    "batch_size": int,
    "num_frames": int,
    "prefetch_count": int,
    "image_count": int,
    "cfg": float,
    "denoise": float,
    "frame_rate": float,
}


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
        # Resolve the declarative {t2i, i2i} workflow specs once. Reading the
        # JSON files themselves is deferred to build time so a missing custom
        # workflow only fails the request that needs it, not construction.
        self._workflows: Dict[str, Optional[WorkflowSpec]] = resolve_workflows(
            self._workflow_config
        )
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
        return await self._run(
            mode="t2i",
            model=model,
            prompt=prompt,
            n=n,
            seed=seed,
            params={
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
            },
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
        image_inputs: Optional[List[Tuple[bytes, str]]] = None,
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
        return await self._run(
            mode="i2i",
            model=model,
            prompt=prompt,
            n=n,
            seed=seed,
            params={
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
            },
            estimated_vram_gb=estimated_vram_gb,
            operation="edit_image",
            image_bytes=image_bytes,
            filename=filename,
            image_inputs=image_inputs,
        )

    async def generate_video(
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
        num_frames: int,
        frame_rate: float,
        prefetch_count: int,
        estimated_vram_gb: Optional[float] = None,
        image_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        image_inputs: Optional[List[Tuple[bytes, str]]] = None,
    ) -> List[ComfyUIImage]:
        has_refs = bool(image_inputs) or image_bytes is not None
        mode = (
            "i2v"
            if has_refs and self._workflows.get("i2v") is not None
            else "video"
        )
        return await self._run(
            mode=mode,
            model=model,
            prompt=prompt,
            n=n,
            seed=seed,
            params={
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "prefetch_count": prefetch_count,
            },
            estimated_vram_gb=estimated_vram_gb,
            operation="generate_video",
            image_bytes=image_bytes,
            filename=filename,
            image_inputs=image_inputs,
        )

    async def _run(
        self,
        *,
        mode: str,
        model: str,
        prompt: str,
        n: int,
        seed: int,
        params: Dict[str, Any],
        estimated_vram_gb: Optional[float],
        operation: str,
        image_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        image_inputs: Optional[List[Tuple[bytes, str]]] = None,
    ) -> List[ComfyUIImage]:
        spec = self._workflows.get(mode)
        if spec is None:
            verb = {
                "t2i": "text-to-image",
                "i2i": "image-to-image",
                "video": "text-to-video",
                "i2v": "image-to-video",
            }.get(mode, mode)
            raise httpx.ProtocolError(
                f"ComfyUI target {self.target_id} has no {verb} workflow configured"
            )
        n = max(1, int(n))
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
                base: Dict[str, Any] = {**params, "prompt": prompt}
                uploads: List[str] = []
                refs = list(image_inputs or [])
                if not refs and image_bytes is not None:
                    refs = [(image_bytes, filename or "input.png")]
                ref_limit = self._image_ref_limit(spec)
                if ref_limit is not None:
                    refs = refs[:ref_limit]
                for idx, (ref_bytes, ref_name) in enumerate(refs, start=1):
                    uploads.append(
                        await self._upload_image(
                            ref_bytes, ref_name or f"input-{idx}.png"
                        )
                    )
                if uploads:
                    base["image"] = uploads[0]
                    base["images"] = uploads
                    base["image_count"] = len(uploads)
                    for idx, uploaded in enumerate(uploads, start=1):
                        base[f"image_{idx}"] = uploaded
                images: List[ComfyUIImage] = []
                if spec.binds("batch_size"):
                    # The workflow batches natively: one submission yields n.
                    workflow = self._build_workflow(
                        spec, {**base, "batch_size": n, "seed": int(seed)}
                    )
                    images.extend(
                        await self._submit_and_collect(workflow, operation=operation)
                    )
                else:
                    # No batch slot (e.g. single-image edit graphs): loop with
                    # an advancing seed so the n outputs differ.
                    for idx in range(n):
                        workflow = self._build_workflow(
                            spec, {**base, "batch_size": 1, "seed": int(seed) + idx}
                        )
                        images.extend(
                            await self._submit_and_collect(
                                workflow, operation=operation
                            )
                        )
                return images[:n]
            finally:
                self._active -= 1
        finally:
            self._touch_vram_reservation(model)
            self._end_request_lifecycle()

    @staticmethod
    def _image_ref_limit(spec: WorkflowSpec) -> Optional[int]:
        if spec.binds("images"):
            return None
        limit = 0
        if spec.binds("image"):
            limit = 1
        for idx in range(1, 5):
            if spec.binds(f"image_{idx}"):
                limit = max(limit, idx)
        return limit

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
        return await self._collect_outputs(history, operation=operation)

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

    async def _collect_outputs(
        self, history_entry: Dict[str, Any], *, operation: str
    ) -> List[ComfyUIImage]:
        outputs = history_entry.get("outputs") or {}
        save_key = "save_video_node_id" if "video" in operation else "save_image_node_id"
        save_node_id = str(self._workflow_config.get(save_key) or "9")
        ordered_nodes: List[Any] = []
        if save_node_id in outputs:
            ordered_nodes.append(outputs[save_node_id])
        ordered_nodes.extend(v for k, v in outputs.items() if str(k) != save_node_id)

        images: List[ComfyUIImage] = []
        for node in ordered_nodes:
            if not isinstance(node, dict):
                continue
            for field in ("images", "gifs", "videos"):
                for item in node.get(field) or []:
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
                    mime_type = self._mime_type(
                        filename, str(item.get("format") or "")
                    )
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
            raise httpx.ProtocolError("ComfyUI history did not contain output media")
        return images

    @staticmethod
    def _mime_type(filename: str, format_hint: str = "") -> str:
        hint = (format_hint or "").strip().lower()
        if hint:
            if hint == "video/h264-mp4" or (hint.startswith("video/") and "mp4" in hint):
                return "video/mp4"
            if hint == "video/webm" or (hint.startswith("video/") and "webm" in hint):
                return "video/webm"
            if hint.startswith("image/") or hint.startswith("audio/") or hint.startswith("video/"):
                return hint
        return mimetypes.guess_type(filename)[0] or "application/octet-stream"

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

    def _build_workflow(
        self, spec: WorkflowSpec, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Materialise an API-format workflow from a spec + request params.

        ``static_inputs`` are fixed per target (model files, sampler knobs that
        are not request-driven); ``bindings`` map request params onto node
        inputs. Params with no binding — or a ``None`` value — are skipped, so
        a workflow only receives the knobs it actually exposes.
        """
        workflow = self._load_workflow(spec.path)
        for node_id, inputs in spec.static_inputs.items():
            for input_name, value in inputs.items():
                if value is not None:
                    self._set_input(workflow, str(node_id), str(input_name), value)
        for param, places in spec.bindings.items():
            if not places:
                continue
            if param == "size_ratio":
                value: Any = nearest_ratio(
                    int(params.get("width") or 0),
                    int(params.get("height") or 0),
                    spec.size_ratio_options,
                )
                if not value:
                    continue
            else:
                value = params.get(param)
            if value is None:
                continue
            coerce = _PARAM_COERCE.get(param)
            if coerce is not None:
                value = coerce(value)
            for node_id, input_name in places:
                self._set_input(workflow, node_id, input_name, value)
        return workflow

    @staticmethod
    def _load_workflow(path: Path) -> Dict[str, Any]:
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"ComfyUI workflow {path} must be an API prompt object")
        return copy.deepcopy(data)

    @staticmethod
    def _set_input(
        workflow: Dict[str, Any], node_id: str, input_name: str, value: Any
    ) -> None:
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            raise ValueError(
                f"ComfyUI workflow is missing node {node_id!r} bound by this target"
            )
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
