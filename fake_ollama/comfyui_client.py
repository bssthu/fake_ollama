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
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from .comfyui_presets import WorkflowSpec, nearest_ratio, resolve_workflows
from . import runtime_ownership
from .process_utils import run_stop_command, process_tree_running
from .config import outbound_cycle_headers
from .h3_media import add_keyframes, add_reference_inputs
from .comfyui_jobs import PromptRecord, runtime_for, recover
from .process_utils import create_managed_subprocess_shell, terminate_process_tree
from .request_data_log import (
    body_from_bytes,
    body_from_json,
    headers_from_mapping,
    log_data_event,
    request_data_logging_enabled,
)
from .vram import (
    VRAM_IDLE_RECLAIM_SECONDS,
    LocalTargetResourceError,
    MemoryCoordinator,
    MemoryReleaseCandidate,
    VramCoordinator,
    VramExecutionLease,
    VramReleaseCandidate,
)

logger = logging.getLogger("fake_ollama")


def _workflow_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
    "enable_tile": _workflow_bool,
    "enable_streaming": _workflow_bool,
}


@dataclass
class _LoadedModel:
    model: str
    estimated_vram_gb: float
    last_used_monotonic: float
    estimated_memory_gb: float = 0.0


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
        manage_external_runtime: bool = False,
        stop_timeout_seconds: float = 30.0,
        ownership_active: bool = True,
        idle_timeout_seconds: Optional[float] = None,
        startup_timeout_seconds: float = 120.0,
        health_path: str = "/system_stats",
        cwd: Optional[str] = None,
        target_name: str = "comfyui",
        runtime_group: Optional[str] = None,
        gpu_device: str = "0",
        workflow_config: Optional[Dict[str, Any]] = None,
        vram_coordinator: Optional[VramCoordinator] = None,
        memory_coordinator: Optional[MemoryCoordinator] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._auth_token = auth_token
        self._timeout = timeout
        self._trust_env = trust_env
        self._auto_start = auto_start
        self._start_command = start_command
        self._stop_command = stop_command
        self._manage_external_runtime = manage_external_runtime
        self._stop_timeout = stop_timeout_seconds
        self._used_since_last_stop = False
        self._idle_timeout = idle_timeout_seconds
        self._startup_timeout = startup_timeout_seconds
        self._health_path = health_path if health_path.startswith("/") else "/" + health_path
        self._cwd = cwd
        self.target_id = f"comfyui:{target_name}"
        self._gpu_device = str(gpu_device or "0")
        group_name = str(runtime_group or f"comfyui:{self._base}").strip()
        self.vram_runtime_group = f"gpu:{self._gpu_device}|{group_name}"
        self._workflow_config = dict(workflow_config or {})
        # Resolve the declarative {t2i, i2i} workflow specs once. Reading the
        # JSON files themselves is deferred to build time so a missing custom
        # workflow only fails the request that needs it, not construction.
        self._workflows: Dict[str, Optional[WorkflowSpec]] = resolve_workflows(
            self._workflow_config
        )
        self._vram_coordinator = vram_coordinator
        self._memory_coordinator = memory_coordinator
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
        self._active_prompt_ids: set[str] = set()
        self._runtime = runtime_for(self._base)
        self._recovery_timeout_seconds = 30.0
        self._recovery_poll_seconds = 0.5
        self._shutdown_requested = False
        self._ownership = runtime_ownership.register(self, group=getattr(self, "vram_runtime_group", None), active=ownership_active)
        self._start_lock = self._ownership.lock
        if self._vram_coordinator is not None:
            self._vram_coordinator.register(self)
        if self._memory_coordinator is not None:
            self._memory_coordinator.register(self)

    @property
    def idle_timeout_seconds(self) -> Optional[float]:
        return self._idle_timeout

    @property
    def active_requests(self) -> int:
        return self._active

    @property
    def request_refs(self) -> int:
        return max(self._active, self._request_refs)

    @property
    def last_used_monotonic(self) -> float:
        return self._last_used

    def begin_shutdown(self) -> None:
        self._shutdown_requested = True

    async def aclose(self) -> None:
        if not await self.recover_pending_prompts():
            raise LocalTargetResourceError("ComfyUI cleanup is unconfirmed; client retained for recovery", error_type="runtime_quarantined")
        await runtime_ownership.close_client(self, terminate=terminate_process_tree, run_command=run_stop_command)

    def has_vram_reservation(self, model: str) -> bool:
        loaded = self._loaded_model
        return bool(
            loaded is not None
            and loaded.model == model
            and loaded.estimated_vram_gb > 0
        )

    def has_memory_reservation(self, model: str) -> bool:
        loaded = self._loaded_model
        return bool(
            loaded is not None
            and loaded.model == model
            and loaded.estimated_memory_gb > 0
        )

    def invalidate_vram_reservation(self) -> None:
        if self._loaded_model is not None:
            self._loaded_model.estimated_vram_gb = 0.0

    def invalidate_memory_reservation(self) -> None:
        if self._loaded_model is not None:
            self._loaded_model.estimated_memory_gb = 0.0
            if self._loaded_model.estimated_vram_gb <= 0:
                self._loaded_model = None

    def _begin_request_lifecycle(self) -> None:
        runtime_ownership.begin_request(self)
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

    async def _ensure_memory(
        self, model: Optional[str], estimated_memory_gb: Optional[float]
    ) -> None:
        if self._memory_coordinator is None or estimated_memory_gb is None:
            return
        await self._memory_coordinator.ensure_available(
            self,
            model=model or "",
            estimated_memory_gb=estimated_memory_gb,
        )

    def _mark_vram_reserved(
        self, model: Optional[str], estimated_vram_gb: Optional[float]
    ) -> None:
        if not model or estimated_vram_gb is None:
            return
        self._used_since_last_stop = True
        existing = self._loaded_model
        self._loaded_model = _LoadedModel(
            model=model,
            estimated_vram_gb=estimated_vram_gb,
            last_used_monotonic=time.monotonic(),
            estimated_memory_gb=(
                existing.estimated_memory_gb
                if existing is not None and existing.model == model
                else 0.0
            ),
        )
        if self._vram_coordinator is not None:
            self._vram_coordinator.confirm_loaded(self.target_id, model)

    def _mark_memory_reserved(
        self, model: Optional[str], estimated_memory_gb: Optional[float]
    ) -> None:
        if not model or estimated_memory_gb is None:
            return
        loaded = self._loaded_model
        if loaded is None or loaded.model != model:
            loaded = _LoadedModel(
                model=model,
                estimated_vram_gb=0.0,
                last_used_monotonic=time.monotonic(),
            )
            self._loaded_model = loaded
        loaded.estimated_memory_gb = estimated_memory_gb
        if self._memory_coordinator is not None:
            self._memory_coordinator.confirm_loaded(self.target_id, model)

    def _touch_vram_reservation(self, model: Optional[str]) -> None:
        if self._loaded_model is not None:
            self._loaded_model.last_used_monotonic = time.monotonic()

    def _discard_vram_pending(self, model: Optional[str]) -> None:
        if not model or self._vram_coordinator is None:
            return
        self._vram_coordinator.discard_pending(self.target_id, model)

    def _discard_memory_pending(self, model: Optional[str]) -> None:
        if not model or self._memory_coordinator is None:
            return
        self._memory_coordinator.discard_pending(self.target_id, model)

    def _clear_all_vram_state(self) -> None:
        loaded = self._loaded_model
        self._loaded_model = None
        if loaded is not None:
            if self._vram_coordinator is not None:
                self._vram_coordinator.discard_pending(self.target_id, loaded.model)
            if self._memory_coordinator is not None:
                self._memory_coordinator.discard_pending(self.target_id, loaded.model)

    def vram_release_candidates(
        self, *, now: float, idle_seconds: float
    ) -> list[VramReleaseCandidate]:
        if (
            self._runtime_in_use(allow_admission=True)
            or self._loaded_model is None
            or self._loaded_model.estimated_vram_gb <= 0
        ):
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
        if (
            self._loaded_model is None
            or self._loaded_model.estimated_vram_gb <= 0
        ):
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

    def memory_release_candidates(
        self, *, now: float, idle_seconds: float
    ) -> list[MemoryReleaseCandidate]:
        if self._runtime_in_use(allow_admission=True) or self._loaded_model is None:
            return []
        loaded = self._loaded_model
        if loaded.estimated_memory_gb <= 0:
            return []
        if now - loaded.last_used_monotonic < idle_seconds:
            return []
        return [
            MemoryReleaseCandidate(
                owner_id=self.target_id,
                model=loaded.model,
                estimated_memory_gb=loaded.estimated_memory_gb,
                last_used_monotonic=loaded.last_used_monotonic,
                release=self._release_for_memory,
            )
        ]

    def memory_force_release_candidates(
        self, *, now: float
    ) -> list[MemoryReleaseCandidate]:
        if self._loaded_model is None or self._loaded_model.estimated_memory_gb <= 0:
            return []
        loaded = self._loaded_model
        return [
            MemoryReleaseCandidate(
                owner_id=self.target_id,
                model=loaded.model,
                estimated_memory_gb=loaded.estimated_memory_gb,
                last_used_monotonic=loaded.last_used_monotonic,
                release=lambda: self._release_for_memory(force=True),
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
                "estimated_memory_gb": loaded.estimated_memory_gb,
                "estimated_memory_mib": loaded.estimated_memory_gb * 1024.0,
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
        self._invalidate_runtime_resources(full=True)
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
            await runtime_ownership.discard_finished_process(self, inspect=process_tree_running, terminate=terminate_process_tree)
            if self._ownership.process is None and not self._started_by_us:
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
                self._ownership.capture(self)

            deadline = time.monotonic() + self._startup_timeout
            while time.monotonic() < deadline:
                if self._shutdown_requested:
                    raise httpx.ConnectError("Shutdown interrupted backend startup")
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
        negative_prompt: str = "",
        estimated_vram_gb: Optional[float] = None,
        estimated_memory_gb: Optional[float] = None,
        request_vram_headroom_gb: float = 0.0,
        min_free_vram_gb: float = 0.0,
        vram_cleanup_policy: str = "keep",
        exclusive_gpu: bool = False,
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
                "negative_prompt": negative_prompt,
            },
            estimated_vram_gb=estimated_vram_gb,
            estimated_memory_gb=estimated_memory_gb,
            request_vram_headroom_gb=request_vram_headroom_gb,
            min_free_vram_gb=min_free_vram_gb,
            vram_cleanup_policy=vram_cleanup_policy,
            exclusive_gpu=exclusive_gpu,
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
        negative_prompt: str = "",
        estimated_vram_gb: Optional[float] = None,
        estimated_memory_gb: Optional[float] = None,
        request_vram_headroom_gb: float = 0.0,
        min_free_vram_gb: float = 0.0,
        vram_cleanup_policy: str = "keep",
        exclusive_gpu: bool = False,
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
                "negative_prompt": negative_prompt,
            },
            estimated_vram_gb=estimated_vram_gb,
            estimated_memory_gb=estimated_memory_gb,
            request_vram_headroom_gb=request_vram_headroom_gb,
            min_free_vram_gb=min_free_vram_gb,
            vram_cleanup_policy=vram_cleanup_policy,
            exclusive_gpu=exclusive_gpu,
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
        negative_prompt: str = "",
        enable_tile: bool = False,
        enable_streaming: bool = False,
        estimated_vram_gb: Optional[float] = None,
        estimated_memory_gb: Optional[float] = None,
        request_vram_headroom_gb: float = 0.0,
        min_free_vram_gb: float = 0.0,
        vram_cleanup_policy: str = "keep",
        exclusive_gpu: bool = False,
        image_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        image_inputs: Optional[List[Tuple[bytes, str]]] = None,
        video_mode: str = "auto",
        video_inputs: Optional[List[Tuple[bytes, str]]] = None,
        keyframe_inputs: Optional[List[Tuple[int, bytes, str]]] = None,
    ) -> List[ComfyUIImage]:
        ref_count = len(image_inputs) if image_inputs else int(image_bytes is not None)
        mode = self._select_video_workflow_mode(video_mode, ref_count)
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
                "negative_prompt": negative_prompt,
                "prefetch_count": prefetch_count,
                "enable_tile": enable_tile,
                "enable_streaming": enable_streaming,
            },
            estimated_vram_gb=estimated_vram_gb,
            estimated_memory_gb=estimated_memory_gb,
            request_vram_headroom_gb=request_vram_headroom_gb,
            min_free_vram_gb=min_free_vram_gb,
            vram_cleanup_policy=vram_cleanup_policy,
            exclusive_gpu=exclusive_gpu,
            operation="generate_video",
            image_bytes=image_bytes,
            filename=filename,
            image_inputs=image_inputs,
            video_inputs=video_inputs,
            keyframe_inputs=keyframe_inputs,
        )

    def _select_video_workflow_mode(self, requested: str, ref_count: int) -> str:
        """Map public H3 base modes onto configured workflow slots.

        ``auto`` preserves existing video targets: no image selects ``video``;
        one or more images select ``i2v`` unless an explicit two-keyframe
        ``fl2va`` workflow exists.  L2VA is necessarily explicit because it
        has the same image count as first-frame I2V.
        """

        value = (requested or "auto").strip().lower()
        aliases = {
            "t2v": "video",
            "t2va": "video",
            "video": "video",
            "i2va": "i2v",
            "i2v": "i2v",
            "fl2v": "fl2va",
            "fl2va": "fl2va",
            "l2v": "l2va",
            "l2va": "l2va",
            "ref2va": "ref2va",
        }
        if value == "auto":
            if ref_count >= 2 and self._workflows.get("fl2va") is not None:
                return "fl2va"
            if ref_count and self._workflows.get("i2v") is not None:
                return "i2v"
            return "video"
        if value not in aliases:
            raise ValueError(
                "video_mode must be auto|t2va|i2va|fl2va|l2va|ref2va"
            )
        return aliases[value]

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
        estimated_memory_gb: Optional[float] = None,
        request_vram_headroom_gb: float = 0.0,
        min_free_vram_gb: float = 0.0,
        vram_cleanup_policy: str = "keep",
        exclusive_gpu: bool = False,
        operation: str,
        image_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        image_inputs: Optional[List[Tuple[bytes, str]]] = None,
        video_inputs: Optional[List[Tuple[bytes, str]]] = None,
        keyframe_inputs: Optional[List[Tuple[int, bytes, str]]] = None,
    ) -> List[ComfyUIImage]:
        spec = self._workflows.get(mode)
        self._runtime.check()
        if spec is None:
            verb = {
                "t2i": "text-to-image",
                "i2i": "image-to-image",
                "video": "text-to-video",
                "i2v": "image-to-video",
                "fl2va": "first-last-frame-to-video",
                "l2va": "last-frame-to-video",
            }.get(mode, mode)
            raise httpx.ProtocolError(
                f"ComfyUI target {self.target_id} has no {verb} workflow configured"
            )
        n = max(1, int(n))
        refs = list(image_inputs or [])
        if not refs and image_bytes is not None:
            refs = [(image_bytes, filename or "input.png")]
        ref_limit = self._image_ref_limit(spec)
        if ref_limit is not None and len(refs) > ref_limit:
            if mode == "ref2va" or self._workflow_config.get("preset") == "minimax_h3":
                raise ValueError(f"{mode} accepts at most {ref_limit} reference image(s)")
            refs = refs[:ref_limit]
        if video_inputs and not spec.binds("reference_videos"):
            raise ValueError("reference videos require a Ref2VA workflow")
        if len(video_inputs or []) > 3:
            raise ValueError("Ref2VA accepts at most 3 reference videos")
        if mode == "ref2va" and not (refs or video_inputs):
            raise ValueError("Ref2VA requires at least one reference image or video")
        self._begin_request_lifecycle()
        try:
            scaled_headroom_gb, workload_key = self._vram_workload(
                spec,
                mode=mode,
                model=model,
                n=n,
                params=params,
                base_headroom_gb=request_vram_headroom_gb,
            )
            admission_vram_gb = estimated_vram_gb
            if admission_vram_gb is not None:
                admission_vram_gb += max(
                    0.0,
                    scaled_headroom_gb
                    - max(0.0, float(request_vram_headroom_gb)),
                )
            async with self._execution_guard(
                model=model,
                reload_vram_gb=float(admission_vram_gb or 0.0),
                workload_key=workload_key,
                request_headroom_gb=scaled_headroom_gb,
                min_free_vram_gb=min_free_vram_gb,
                cleanup_policy=vram_cleanup_policy,
                exclusive=exclusive_gpu,
            ) as lease:
                try:
                    with lease.admission() if lease is not None else nullcontext():
                        await self._ensure_vram(model, admission_vram_gb)
                        await self._ensure_memory(model, estimated_memory_gb)
                        # Releasing the shared RAM/object cache also invalidates
                        # the GPU reservation acquired above. Re-admit it before
                        # submitting a prompt (normally this is an idempotent call).
                        await self._ensure_vram(model, admission_vram_gb)
                    await self._ensure_ready()
                except BaseException:
                    self._discard_vram_pending(model)
                    self._discard_memory_pending(model)
                    raise
                self._active += 1
                try:
                    self._mark_vram_reserved(model, estimated_vram_gb)
                    self._mark_memory_reserved(model, estimated_memory_gb)
                    base: Dict[str, Any] = {**params, "prompt": prompt}
                    uploads: List[str] = []
                    if keyframe_inputs:
                        info = await self._client.get(
                            f"{self._base}/object_info/MiniMaxH3AddGuide", headers=self._headers()
                        )
                        info.raise_for_status()
                        if "MiniMaxH3AddGuide" not in info.json():
                            raise httpx.ProtocolError(
                                "ComfyUI is missing MiniMaxH3AddGuide; install H3 guide support and restart ComfyUI"
                            )
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
                        base["reference_images"] = uploads
                        for idx, uploaded in enumerate(uploads, start=1):
                            base[f"image_{idx}"] = uploaded
                    if video_inputs:
                        base["reference_videos"] = [
                            await self._upload_image(data, name) for data, name in video_inputs
                        ]
                    if keyframe_inputs:
                        base["keyframes"] = [
                            (index, await self._upload_image(data, name))
                            for index, data, name in keyframe_inputs
                        ]
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

    def _vram_workload(
        self,
        spec: WorkflowSpec,
        *,
        mode: str,
        model: str,
        n: int,
        params: Dict[str, Any],
        base_headroom_gb: float,
    ) -> tuple[float, str]:
        width = max(1, int(params.get("width") or 1))
        height = max(1, int(params.get("height") or 1))
        default_width = max(1, int(self._workflow_config.get("default_width") or width))
        default_height = max(1, int(self._workflow_config.get("default_height") or height))
        scale = max(1.0, (width * height) / (default_width * default_height))
        frames = max(1, int(params.get("num_frames") or 1))
        if mode in {"video", "i2v", "fl2va", "l2va", "ref2va"}:
            default_frames = max(
                1, int(self._workflow_config.get("default_num_frames") or frames)
            )
            scale *= max(1.0, frames / default_frames)
        native_batch = n if spec.binds("batch_size") else 1
        scale *= max(1, native_batch)
        headroom_gb = max(0.0, float(base_headroom_gb)) * scale
        workload_key = (
            f"{self.vram_runtime_group}|{model}|{mode}|{width}x{height}"
            f"|frames={frames}|batch={native_batch}"
        )
        return headroom_gb, workload_key

    @asynccontextmanager
    async def _execution_guard(
        self,
        *,
        model: str,
        workload_key: str,
        reload_vram_gb: float,
        request_headroom_gb: float,
        min_free_vram_gb: float,
        cleanup_policy: str,
        exclusive: bool,
    ) -> AsyncIterator[Optional[VramExecutionLease]]:
        if self._vram_coordinator is None:
            yield None
            return
        lease = await self._vram_coordinator.acquire_execution(
            self,
            model=model,
            workload_key=workload_key,
            reload_vram_gb=reload_vram_gb,
            request_headroom_gb=request_headroom_gb,
            min_free_vram_gb=min_free_vram_gb,
            cleanup_policy=cleanup_policy,
            exclusive=exclusive,
            cleanup=self._adaptive_unload_gpu,
        )
        async with lease:
            yield lease

    @staticmethod
    def _image_ref_limit(spec: WorkflowSpec) -> Optional[int]:
        if spec.max_image_refs is not None:
            return max(0, int(spec.max_image_refs))
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
        safe_name = Path((filename or "input.png").replace("\\", "/")).name or "input.png"
        # Concurrent requests can use the same client filename. Keep each upload
        # immutable until its queued prompt has consumed it (videos use this endpoint too).
        safe_name = f"{uuid.uuid4().hex}-{safe_name}"
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
        name = str(data.get("name") or safe_name)
        subfolder = str(data.get("subfolder") or "").strip("/\\")
        return f"{subfolder}/{name}" if subfolder else name

    async def _submit_and_collect(
        self,
        workflow: Dict[str, Any],
        *,
        operation: str,
    ) -> List[ComfyUIImage]:
        self._runtime.check()
        record = PromptRecord()
        record.tracking = self._active_prompt_ids
        key = record.prompt_id
        self._active_prompt_ids.add(key)
        try:
            prompt_id = await self._queue_prompt(workflow, operation=operation, record=record)
            self._active_prompt_ids.discard(key)
            key = prompt_id
            self._active_prompt_ids.add(key)
            record.prompt_id = prompt_id
            record.uncertain = False
            record.status = "accepted"
            history = await self._wait_for_history(prompt_id, operation=operation)
            record.status = "finished"
            return await self._collect_outputs(history, operation=operation)
        except BaseException as exc:
            if record.status != "finished":
                # A rejected request is known not to be queued; other transport
                # failures (including lost acknowledgement) remain uncertain.
                rejected = record.status == "submitting" and isinstance(exc, httpx.HTTPStatusError) and 400 <= exc.response.status_code < 500
                if not rejected:
                    self._runtime.pending[key] = record
                    if self._vram_coordinator is not None:
                        async def confirmed_idle():
                            return not self._runtime.pending
                        self._vram_coordinator.quarantine_runtime(self.vram_runtime_group, "unconfirmed ComfyUI prompt", verify=confirmed_idle)
                    record.recovery_task = asyncio.create_task(recover(self, key, record), name=f"comfy-recover-{record.prompt_id}")
                    raise
            self._active_prompt_ids.discard(key)
            raise
        finally:
            if key not in self._runtime.pending:
                self._active_prompt_ids.discard(key)

    async def recover_pending_prompts(self) -> bool:
        """Bounded retry; unresolved records and admission protection persist."""
        for key, record in list(self._runtime.pending.items()):
            if record.recovery_task is None or record.recovery_task.done():
                record.recovery_task = asyncio.create_task(recover(self, key, record))
            await asyncio.shield(record.recovery_task)
        return not self._runtime.pending

    def prompt_snapshots(self):
        return [{"prompt_id": r.prompt_id, "status": r.status} for r in self._runtime.pending.values()]

    async def _queue_prompt(self, workflow: Dict[str, Any], *, operation: str, record: PromptRecord | None = None) -> str:
        self._runtime.check()
        record = record or PromptRecord()
        url = f"{self._base}/prompt"
        body = {"prompt": workflow, "client_id": record.client_id, "prompt_id": record.prompt_id}
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
                if param in {"reference_images", "reference_videos"}:
                    add_reference_inputs(workflow, node_id, input_name, value,
                                         video=param == "reference_videos")
                elif param == "images" and isinstance(value, list):
                    self._set_image_batch_input(
                        workflow, node_id, input_name, value
                    )
                else:
                    self._set_input(workflow, node_id, input_name, value)
        add_keyframes(workflow, params.get("keyframes") or [])
        return workflow

    def _set_image_batch_input(
        self,
        workflow: Dict[str, Any],
        node_id: str,
        input_name: str,
        image_names: List[Any],
    ) -> None:
        """Bind multiple uploaded images to one ComfyUI IMAGE input.

        The target input must already be connected to a LoadImage node in the
        workflow. That node is used as the first reference and as the template
        for clones; extra references are combined with core ImageBatch nodes.
        If the input is not a LoadImage link, fall back to assigning the raw
        filename list so custom workflows that expect a list-valued widget keep
        working.
        """
        names = [str(name) for name in image_names if str(name or "").strip()]
        if not names:
            return

        target = workflow.get(str(node_id))
        if not isinstance(target, dict):
            raise ValueError(
                f"ComfyUI workflow is missing node {node_id!r} bound by this target"
            )
        target_inputs = target.setdefault("inputs", {})
        if not isinstance(target_inputs, dict):
            raise ValueError(f"ComfyUI workflow node {node_id!r} has invalid inputs")

        current = target_inputs.get(str(input_name))
        link = self._comfy_link(current)
        if link is None:
            target_inputs[str(input_name)] = names
            return

        load_node_id, load_output_idx = link
        load_node = workflow.get(load_node_id)
        if not self._is_load_image_node(load_node):
            target_inputs[str(input_name)] = names
            return

        self._set_input(workflow, load_node_id, "image", names[0])
        if len(names) == 1:
            target_inputs[str(input_name)] = [load_node_id, load_output_idx]
            return

        next_id = self._next_workflow_node_id(workflow)
        last_link: List[Any] = [load_node_id, load_output_idx]
        for ref_name in names[1:]:
            clone_id = str(next_id)
            next_id += 1
            clone = copy.deepcopy(load_node)
            clone_inputs = clone.setdefault("inputs", {})
            if not isinstance(clone_inputs, dict):
                clone_inputs = {}
                clone["inputs"] = clone_inputs
            clone_inputs["image"] = ref_name
            workflow[clone_id] = clone

            batch_id = str(next_id)
            next_id += 1
            workflow[batch_id] = {
                "class_type": "ImageBatch",
                "inputs": {
                    "image1": list(last_link),
                    "image2": [clone_id, load_output_idx],
                },
            }
            last_link = [batch_id, 0]

        target_inputs[str(input_name)] = last_link

    @staticmethod
    def _comfy_link(value: Any) -> Optional[Tuple[str, int]]:
        if not isinstance(value, list) or len(value) != 2:
            return None
        node_id, output_idx = value
        if not isinstance(node_id, (str, int)) or not isinstance(output_idx, int):
            return None
        return str(node_id), int(output_idx)

    @staticmethod
    def _is_load_image_node(node: Any) -> bool:
        if not isinstance(node, dict):
            return False
        if str(node.get("class_type") or "") != "LoadImage":
            return False
        inputs = node.get("inputs")
        return isinstance(inputs, dict) and "image" in inputs

    @staticmethod
    def _next_workflow_node_id(workflow: Dict[str, Any]) -> int:
        next_id = 1
        for key in workflow:
            try:
                next_id = max(next_id, int(str(key)) + 1)
            except ValueError:
                continue
        while str(next_id) in workflow:
            next_id += 1
        return next_id

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

    def _runtime_in_use(self, *, allow_admission: bool = False) -> bool:
        if any(member is not self and member._active for member in self._ownership.members):
            return True
        if self._runtime.pending:
            return True
        busy = bool(
            self.request_refs
            or (
                self._vram_coordinator is not None
                and self._vram_coordinator.runtime_group_request_refs(
                    self.vram_runtime_group
                )
            )
        )
        if busy and allow_admission and self._vram_coordinator is not None:
            return not self._vram_coordinator.runtime_group_admitting_here(
                self.vram_runtime_group
            )
        return busy

    async def stop_if_idle(self) -> None:
        if not self._idle_timeout or self._runtime_in_use():
            return
        last_used = self._last_used
        if self._vram_coordinator is not None:
            last_used = max(
                last_used,
                self._vram_coordinator.runtime_group_last_used(self.vram_runtime_group),
            )
        idle_for = time.monotonic() - max(last_used, *(m._last_used for m in self._ownership.members))
        if idle_for < self._idle_timeout:
            return
        if not (
            self._ownership.can_manage()
        ):
            return
        logger.info(
            "stopping idle ComfyUI target %s after %.1fs idle",
            self.target_id,
            idle_for,
        )
        if await self.stop_if_owned():
            self._last_used = time.monotonic()

    async def release_for_vram(self) -> bool:
        return await self._release_for_vram()

    async def _release_for_vram(self, *, force: bool = False) -> bool:
        if self._runtime.pending:
            return False
        if not force and self._runtime_in_use(allow_admission=True):
            return False
        if self._loaded_model is None or self._loaded_model.estimated_vram_gb <= 0:
            return True
        if await self._free_memory():
            return True
        if self._ownership.can_manage():
            return await self.stop_if_owned()
        return False

    async def _release_for_memory(self, *, force: bool = False) -> bool:
        if self._runtime.pending:
            return False
        if not force and self._runtime_in_use(allow_admission=True):
            return False
        if self._loaded_model is None or self._loaded_model.estimated_memory_gb <= 0:
            return True
        if await self._free_memory():
            return True
        if self._ownership.can_manage():
            return await self.stop_if_owned()
        return False

    async def _free_memory(self) -> bool:
        if self._runtime.pending:
            return False
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
                self._invalidate_runtime_resources(full=True)
                return True
            logger.warning(
                "ComfyUI /free returned status=%s body=%s",
                resp.status_code,
                resp.text,
            )
        except httpx.HTTPError as exc:
            logger.warning("failed to call ComfyUI /free at %s: %s", self._base, exc)
        return False

    async def _adaptive_unload_gpu(self) -> bool:
        """Unload runtime GPU weights without resetting ComfyUI's object cache."""

        if self._active or self._runtime.pending:
            return False
        if (
            self._vram_coordinator is not None
            and self._vram_coordinator.runtime_group_active_requests(
                self.vram_runtime_group
            )
        ):
            return False
        try:
            resp = await self._client.post(
                f"{self._base}/free",
                json={"unload_models": True, "free_memory": False},
                headers=self._headers(),
                timeout=min(30.0, self._timeout),
            )
            await resp.aread()
            if 200 <= resp.status_code < 300:
                logger.info(
                    "requested adaptive ComfyUI GPU model unload at %s without cache reset",
                    self._base,
                )
                self._invalidate_runtime_resources(full=False)
                return True
            logger.warning(
                "ComfyUI adaptive /free returned status=%s body=%s",
                resp.status_code,
                resp.text,
            )
        except httpx.HTTPError as exc:
            logger.warning("failed adaptive ComfyUI /free at %s: %s", self._base, exc)
        return False

    def _invalidate_runtime_resources(self, *, full: bool) -> None:
        if self._vram_coordinator is not None:
            self._vram_coordinator.invalidate_runtime_group(self.vram_runtime_group)
        self.invalidate_vram_reservation()
        if full:
            invalidate_memory = getattr(
                self._memory_coordinator, "invalidate_runtime_group", None
            )
            if callable(invalidate_memory):
                invalidate_memory(self.vram_runtime_group)
            self.invalidate_memory_reservation()

    async def stop_if_owned(self) -> bool:
        return await runtime_ownership.stop_owned(self, terminate=terminate_process_tree, run_command=run_stop_command)
