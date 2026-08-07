"""Unit tests for VRAM reservation queueing and races."""

from __future__ import annotations

import asyncio
import time
import shlex
from dataclasses import dataclass, field
from typing import Optional

import pytest

from fake_ollama import vram as vram_mod
from fake_ollama.vram import (
    LocalTargetResourceError,
    PENDING_LOADED_GRACE_SECONDS,
    VramCoordinator,
    VramReleaseCandidate,
)


@dataclass
class _FakeParticipant:
    target_id: str
    _reservations: set[str] = field(default_factory=set)
    active_requests: int = 0  # type: ignore[assignment]

    def has_vram_reservation(self, model: str) -> bool:
        return model in self._reservations

    def vram_release_candidates(self, *, now: float, idle_seconds: float):
        return []


def _free_provider(mib: float):
    async def _f() -> Optional[float]:
        return mib
    return _f


def _mutable_free_provider(state: dict[str, float]):
    async def _f() -> Optional[float]:
        return state["mib"]
    return _f


def _total_provider(mib: float):
    async def _f() -> Optional[float]:
        return mib
    return _f


# -- pending bookkeeping --------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_different_models_respect_capacity():
    """Two concurrent admissions against the same physical free pool should
    not both pass when together they exceed it."""
    coord = VramCoordinator(provider=_free_provider(16 * 1024.0))
    p_a = _FakeParticipant("target-a")
    p_b = _FakeParticipant("target-b")
    coord.register(p_a)
    coord.register(p_b)

    # First request reserves 10 GiB.
    await coord.ensure_available(p_a, model="a", estimated_vram_gb=10)
    assert coord.has_pending("target-a", "a")

    # Second request for a different model wants 10 GiB too; must fail
    # because a's pending already consumes 10 of the 16 GiB.
    with pytest.raises(LocalTargetResourceError, match="Insufficient GPU VRAM"):
        await coord.ensure_available(p_b, model="b", estimated_vram_gb=10)
    assert not coord.has_pending("target-b", "b")


@pytest.mark.asyncio
async def test_same_model_pending_is_idempotent():
    """A second ensure_available for the same (target,model) must not
    double-book the budget."""
    coord = VramCoordinator(provider=_free_provider(12 * 1024.0))
    p = _FakeParticipant("t")
    coord.register(p)

    await coord.ensure_available(p, model="m", estimated_vram_gb=8)
    # Without idempotency the second call would treat 8 GiB as already
    # subtracted and see only 4 GiB free, raising. It should succeed.
    await coord.ensure_available(p, model="m", estimated_vram_gb=8)
    assert coord.has_pending("t", "m")


@pytest.mark.asyncio
async def test_failed_startup_releases_pending():
    coord = VramCoordinator(provider=_free_provider(16 * 1024.0))
    p = _FakeParticipant("t")
    coord.register(p)

    await coord.ensure_available(p, model="m", estimated_vram_gb=10)
    coord.discard_pending("t", "m")
    assert not coord.has_pending("t", "m")

    # Subsequent admission for a different model should succeed.
    p2 = _FakeParticipant("t2")
    coord.register(p2)
    await coord.ensure_available(p2, model="m2", estimated_vram_gb=10)


@pytest.mark.asyncio
async def test_confirm_loaded_then_grace_expiry_drops_pending(monkeypatch: pytest.MonkeyPatch):
    """After confirm_loaded + grace window, pending entry stops counting."""
    free_mib = 16 * 1024.0
    coord = VramCoordinator(provider=_free_provider(free_mib))
    p = _FakeParticipant("t")
    coord.register(p)

    await coord.ensure_available(p, model="m", estimated_vram_gb=10)
    coord.confirm_loaded("t", "m")
    # has_vram_reservation stays False on the fake — coordinator still uses
    # the pending entry to subtract 10 GiB, so a 10 GiB request for another
    # model would fail right now.
    p2 = _FakeParticipant("t2")
    coord.register(p2)
    with pytest.raises(LocalTargetResourceError):
        await coord.ensure_available(p2, model="other", estimated_vram_gb=10)

    # Advance monotonic clock past the grace window.
    real_monotonic = time.monotonic
    base = real_monotonic()
    monkeypatch.setattr(
        vram_mod.time, "monotonic", lambda: base + PENDING_LOADED_GRACE_SECONDS + 1.0
    )

    # Now the pending entry should drop on the next admission, and the
    # second target should fit (assuming nvidia-smi reflects the load —
    # here our fake free_provider still reports 16 GiB).
    await coord.ensure_available(p2, model="other", estimated_vram_gb=10)


@pytest.mark.asyncio
async def test_reclaim_if_below_releases_idle_candidates():
    free_mib = {"mib": 100.0}
    coord = VramCoordinator(provider=_mutable_free_provider(free_mib))

    class _IdleParticipant:
        target_id = "target"
        active_requests = 0

        def __init__(self) -> None:
            self.released = False
            self.last_used = time.monotonic() - 120.0

        def has_vram_reservation(self, model: str) -> bool:
            return model == "old"

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            if now - self.last_used < idle_seconds:
                return []
            return [
                VramReleaseCandidate(
                    owner_id=self.target_id,
                    model="old",
                    estimated_vram_gb=1.0,
                    last_used_monotonic=self.last_used,
                    release=self.release,
                )
            ]

        async def release(self) -> bool:
            self.released = True
            free_mib["mib"] = 512.0
            return True

    participant = _IdleParticipant()
    coord.register(participant)

    result = await coord.reclaim_if_below(threshold_mib=200.0, idle_seconds=60.0)

    assert participant.released is True
    assert result["available_mib"] == 512.0
    assert result["released"] == [
        {"owner_id": "target", "model": "old", "estimated_vram_gb": 1.0}
    ]


@pytest.mark.asyncio
async def test_reclaim_if_below_does_not_release_active_participants():
    free_mib = {"mib": 100.0}
    coord = VramCoordinator(provider=_mutable_free_provider(free_mib))

    class _ActiveParticipant:
        target_id = "target"
        active_requests = 1

        def has_vram_reservation(self, model: str) -> bool:
            return model == "old"

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            raise AssertionError("active participant should be skipped")

    coord.register(_ActiveParticipant())

    result = await coord.reclaim_if_below(threshold_mib=200.0, idle_seconds=60.0)

    assert result["available_mib"] == 100.0
    assert result["released"] == []


@pytest.mark.asyncio
async def test_reclaim_model_releases_matching_idle_candidate():
    coord = VramCoordinator(provider=_free_provider(100.0))

    class _IdleParticipant:
        target_id = "target"
        active_requests = 0

        def __init__(self) -> None:
            self.released = False
            self.last_used = time.monotonic() - 120.0

        def has_vram_reservation(self, model: str) -> bool:
            return model == "old"

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            if now - self.last_used < idle_seconds:
                return []
            return [
                VramReleaseCandidate(
                    owner_id=self.target_id,
                    model="old",
                    estimated_vram_gb=1.0,
                    last_used_monotonic=self.last_used,
                    release=self.release,
                )
            ]

        async def release(self) -> bool:
            self.released = True
            return True

    participant = _IdleParticipant()
    coord.register(participant)

    result = await coord.reclaim_model(
        target_id="target", model="old", idle_seconds=60.0
    )

    assert participant.released is True
    assert result == {
        "target_id": "target",
        "model": "old",
        "estimated_vram_gb": 1.0,
        "released": True,
        "reason": None,
    }


@pytest.mark.asyncio
async def test_reclaim_model_requires_idle_eligible_candidate():
    coord = VramCoordinator(provider=_free_provider(100.0))

    class _BusyParticipant:
        target_id = "target"
        active_requests = 1

        def has_vram_reservation(self, model: str) -> bool:
            return model == "old"

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            raise AssertionError("active participant should be skipped")

    coord.register(_BusyParticipant())

    result = await coord.reclaim_model(
        target_id="target", model="old", idle_seconds=60.0
    )

    assert result == {
        "target_id": "target",
        "model": "old",
        "released": False,
        "reason": "not_eligible",
    }


@pytest.mark.asyncio
async def test_reclaim_model_force_uses_force_candidate_for_active_model():
    coord = VramCoordinator(provider=_free_provider(100.0))

    class _ActiveParticipant:
        target_id = "target"
        active_requests = 1

        def __init__(self) -> None:
            self.released = False
            self.last_used = time.monotonic()

        def has_vram_reservation(self, model: str) -> bool:
            return model == "old"

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            raise AssertionError("force reclaim should not use idle candidates")

        def vram_force_release_candidates(self, *, now: float):
            return [
                VramReleaseCandidate(
                    owner_id=self.target_id,
                    model="old",
                    estimated_vram_gb=1.0,
                    last_used_monotonic=self.last_used,
                    release=self.release,
                )
            ]

        async def release(self) -> bool:
            self.released = True
            return True

    participant = _ActiveParticipant()
    coord.register(participant)

    result = await coord.reclaim_model(
        target_id="target", model="old", idle_seconds=60.0, force=True
    )

    assert participant.released is True
    assert result == {
        "target_id": "target",
        "model": "old",
        "estimated_vram_gb": 1.0,
        "released": True,
        "reason": None,
        "forced": True,
    }


@pytest.mark.asyncio
async def test_total_capacity_allows_best_effort_reclaim_when_estimates_are_short():
    free_mib = {"mib": 512.0}
    coord = VramCoordinator(
        provider=_mutable_free_provider(free_mib),
        total_provider=_total_provider(8 * 1024.0),
    )
    requester = _FakeParticipant("new")
    coord.register(requester)

    class _IdleParticipant:
        target_id = "old"
        active_requests = 0

        def __init__(self) -> None:
            self.released = False
            self.last_used = time.monotonic() - 120.0

        def has_vram_reservation(self, model: str) -> bool:
            return model == "old-model"

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            if now - self.last_used < idle_seconds:
                return []
            return [
                VramReleaseCandidate(
                    owner_id=self.target_id,
                    model="old-model",
                    estimated_vram_gb=1.0,
                    last_used_monotonic=self.last_used,
                    release=self.release,
                )
            ]

        async def release(self) -> bool:
            self.released = True
            # The estimate was too low; the real nvidia-smi refresh now shows
            # enough free VRAM for the new model.
            free_mib["mib"] = 5 * 1024.0
            return True

    old = _IdleParticipant()
    coord.register(old)

    await coord.ensure_available(requester, model="new-model", estimated_vram_gb=4)

    assert old.released is True
    assert coord.has_pending("new", "new-model")


@pytest.mark.asyncio
async def test_best_effort_reclaim_is_skipped_when_total_capacity_is_too_small():
    free_mib = {"mib": 512.0}
    coord = VramCoordinator(
        provider=_mutable_free_provider(free_mib),
        total_provider=_total_provider(2 * 1024.0),
    )
    requester = _FakeParticipant("new")
    coord.register(requester)

    class _IdleParticipant:
        target_id = "old"
        active_requests = 0

        def __init__(self) -> None:
            self.released = False
            self.last_used = time.monotonic() - 120.0

        def has_vram_reservation(self, model: str) -> bool:
            return model == "old-model"

        def vram_release_candidates(self, *, now: float, idle_seconds: float):
            return [
                VramReleaseCandidate(
                    owner_id=self.target_id,
                    model="old-model",
                    estimated_vram_gb=1.0,
                    last_used_monotonic=self.last_used,
                    release=self.release,
                )
            ]

        async def release(self) -> bool:
            self.released = True
            free_mib["mib"] = 5 * 1024.0
            return True

    old = _IdleParticipant()
    coord.register(old)

    with pytest.raises(LocalTargetResourceError, match="Insufficient GPU VRAM"):
        await coord.ensure_available(requester, model="new-model", estimated_vram_gb=4)

    assert old.released is False
    assert not coord.has_pending("new", "new-model")


# -- per-client startup queueing -----------------------------------------


@pytest.mark.asyncio
async def test_concurrent_same_model_requests_share_startup(
    monkeypatch: pytest.MonkeyPatch,
):
    """Two concurrent chat() calls for the same model should serialise on
    the per-client startup lock and only invoke start_command once."""
    import json
    import httpx

    from fake_ollama.ollama_client import OllamaClient

    state = {"start_calls": 0, "started": False}
    start_gate = asyncio.Event()

    class _SlowProc:
        returncode = None

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = 1

        async def wait(self) -> int:
            return 0

    async def fake_create_subprocess_shell(command: str, **kwargs):
        state["start_calls"] += 1
        # Simulate a slow boot — flip to "started" only after the gate.
        async def _later():
            await start_gate.wait()
            state["started"] = True
        asyncio.create_task(_later())
        return _SlowProc()

    monkeypatch.setattr(
        "fake_ollama.process_utils.asyncio.create_subprocess_shell",
        fake_create_subprocess_shell,
    )

    async def _async_true(*_a, **_kw) -> bool:
        return True

    monkeypatch.setattr(
        "fake_ollama.ollama_client.terminate_process_tree", _async_true
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            if not state["started"]:
                raise httpx.ConnectError("daemon down", request=request)
            return httpx.Response(200, json={"version": "0.0.0-test"})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "message": {"role": "assistant", "content": "ok"},
                    "done": True,
                },
            )
        raise AssertionError(f"unexpected path: {request.url.path}")

    coord = VramCoordinator(provider=_free_provider(24 * 1024.0))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OllamaClient(
        "http://127.0.0.1:11434",
        target_name="ollama",
        auto_start=True,
        start_command="ollama serve",
        startup_timeout_seconds=5,
        vram_coordinator=coord,
        client=async_client,
    )

    async def _do():
        return await client.chat(
            {"model": "m", "messages": [], "stream": False},
            estimated_vram_gb=8,
        )

    # Fire both concurrently.
    task_a = asyncio.create_task(_do())
    task_b = asyncio.create_task(_do())
    # Give them a moment to both reach the startup wait, then release.
    await asyncio.sleep(0.05)
    start_gate.set()

    try:
        results = await asyncio.gather(task_a, task_b)
    finally:
        await client.aclose()

    assert state["start_calls"] == 1
    for r in results:
        assert r["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_concurrent_different_model_clients_queue_via_coordinator(
    monkeypatch: pytest.MonkeyPatch,
):
    """Two clients (different targets, different models) hitting a shared
    coordinator with limited VRAM: the second must observe pending and
    fail rather than racing past admission."""
    import httpx

    from fake_ollama.ollama_client import OllamaClient

    # Each client has its own auto-start gate.
    state = {"started_a": True, "started_b": True}

    async def fake_create_subprocess_shell(command: str, **kwargs):  # pragma: no cover
        raise AssertionError("auto_start should not be triggered in this test")

    monkeypatch.setattr(
        "fake_ollama.process_utils.asyncio.create_subprocess_shell",
        fake_create_subprocess_shell,
    )

    chat_gate = asyncio.Event()

    def make_handler(name: str):
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.0.0-test"})
            if request.url.path == "/api/chat":
                return httpx.Response(
                    200,
                    json={
                        "message": {"role": "assistant", "content": name},
                        "done": True,
                    },
                )
            raise AssertionError(f"unexpected path: {request.url.path}")
        return handler

    coord = VramCoordinator(provider=_free_provider(16 * 1024.0))
    client_a = OllamaClient(
        "http://127.0.0.1:11434",
        target_name="a",
        auto_start=False,
        vram_coordinator=coord,
        client=httpx.AsyncClient(transport=httpx.MockTransport(make_handler("a"))),
    )
    client_b = OllamaClient(
        "http://127.0.0.1:11435",
        target_name="b",
        auto_start=False,
        vram_coordinator=coord,
        client=httpx.AsyncClient(transport=httpx.MockTransport(make_handler("b"))),
    )

    # Slow A's request body so B's admission happens while A still pending.
    original_post = client_a._client.post  # type: ignore[attr-defined]

    async def slow_post(*args, **kwargs):
        await chat_gate.wait()
        return await original_post(*args, **kwargs)

    client_a._client.post = slow_post  # type: ignore[attr-defined]

    async def _do(c: OllamaClient, model: str):
        return await c.chat({"model": model, "messages": []}, estimated_vram_gb=10)

    task_a = asyncio.create_task(_do(client_a, "ma"))
    # Let A pass admission first.
    await asyncio.sleep(0.05)
    task_b = asyncio.create_task(_do(client_b, "mb"))

    try:
        # B should fail fast on admission.
        with pytest.raises(LocalTargetResourceError):
            await task_b
        # Now release A.
        chat_gate.set()
        result_a = await task_a
    finally:
        await client_a.aclose()
        await client_b.aclose()

    assert result_a["message"]["content"] == "a"


# -- synthesize_start_command --------------------------------------------


def test_synthesize_start_command_passthrough_when_explicit():
    from fake_ollama.config import LlamaCppTarget

    tgt = LlamaCppTarget(
        base_url="http://127.0.0.1:21500",
        model="qa",
        start_command="custom --serve",
        model_path="C:/m.gguf",  # ignored when start_command set
    )
    assert tgt.synthesize_start_command() == "custom --serve"


def test_synthesize_start_command_returns_none_without_inputs():
    from fake_ollama.config import LlamaCppTarget

    tgt = LlamaCppTarget(base_url="http://127.0.0.1:21500", model="qa")
    assert tgt.synthesize_start_command() is None


def test_synthesize_start_command_builds_full_command():
    from fake_ollama.config import LlamaCppTarget

    tgt = LlamaCppTarget(
        base_url="http://127.0.0.1:21500",
        model="qa",
        binary_path="C:/llama/llama-server.exe",
        model_path="C:/models/qwen.gguf",
        mmproj_path="C:/models/mm.gguf",
        gpu_layers=99,
        ctx_size=8192,
        parallel=4,
        upstream_id="qa",
        auth_token="secret",
        extra_args="--jinja",
    )
    cmd = tgt.synthesize_start_command()
    assert cmd is not None
    # Components must all be present; quoting via shlex should keep paths intact.
    assert "llama-server.exe" in cmd
    assert "--host" in cmd and "127.0.0.1" in cmd
    assert "--port" in cmd and "21500" in cmd
    assert "--model" in cmd and "qwen.gguf" in cmd
    assert "--mmproj" in cmd and "mm.gguf" in cmd
    assert "-ngl" in cmd and "99" in cmd
    assert "--ctx-size" in cmd and "8192" in cmd
    assert "--parallel" in cmd and "4" in cmd
    assert "--alias" in cmd and "qa" in cmd
    assert "--api-key" in cmd and "secret" in cmd
    assert cmd.rstrip().endswith("--jinja")


def test_synthesize_start_command_defaults_binary_to_llama_server():
    from fake_ollama.config import LlamaCppTarget

    tgt = LlamaCppTarget(
        base_url="http://127.0.0.1:21500",
        model="qa",
        model_path="C:/m.gguf",
    )
    cmd = tgt.synthesize_start_command()
    assert cmd is not None
    assert cmd.split()[0].strip("\"'") == "llama-server"


def test_synthesize_start_command_resolves_binary_directory(tmp_path):
    """When binary_path points to an extracted llama.cpp release folder,
    auto-append the actual llama-server[.exe] inside it instead of feeding
    a directory to cmd.exe (which silently fails)."""
    from fake_ollama.config import LlamaCppTarget

    exe_name = "llama-server.exe"
    (tmp_path / exe_name).write_bytes(b"")
    tgt = LlamaCppTarget(
        base_url="http://127.0.0.1:21500",
        model="qa",
        binary_path=str(tmp_path),
        model_path="C:/m.gguf",
    )
    cmd = tgt.synthesize_start_command()
    assert cmd is not None
    expected = str(tmp_path / exe_name)
    # Must include the resolved .exe path (possibly shell-quoted).
    assert expected in cmd or shlex.quote(expected) in cmd


def test_effective_env_prepends_runtime_root_and_binary_dir(tmp_path, monkeypatch):
    """runtime_root (cudart folder) and the binary's directory must be
    prepended to PATH so llama-server can locate cudart64_*.dll and
    actually use CUDA. Mirrors the original PowerShell launcher."""
    import os as _os
    from fake_ollama.config import LlamaCppTarget

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-server.exe").write_bytes(b"")
    runtime_dir = tmp_path / "cudart"
    runtime_dir.mkdir()

    # Pick a sentinel PATH entry that doesn't itself contain os.pathsep
    # (':' on POSIX, ';' on Windows), so PATH.split(os.pathsep) keeps it
    # as a single component on every platform.
    existing_entry = str(tmp_path / "existing")
    monkeypatch.setenv("PATH", existing_entry)
    tgt = LlamaCppTarget(
        base_url="http://127.0.0.1:21500",
        model="qa",
        binary_path=str(bin_dir / "llama-server.exe"),
        runtime_root=str(runtime_dir),
        model_path="C:/m.gguf",
    )
    env = tgt.effective_env()
    assert env is not None
    new_path = env["PATH"]
    parts = new_path.split(_os.pathsep)
    # Both prepended, runtime_root or bin_dir before existing.
    assert str(bin_dir.resolve()) in parts
    assert str(runtime_dir.resolve()) in parts
    assert parts.index(str(bin_dir.resolve())) < parts.index(existing_entry)
    assert parts.index(str(runtime_dir.resolve())) < parts.index(existing_entry)


def test_effective_env_returns_none_without_runtime_or_binary():
    from fake_ollama.config import LlamaCppTarget

    tgt = LlamaCppTarget(base_url="http://127.0.0.1:21500", model="qa")
    assert tgt.effective_env() is None


def test_with_defaults_inherits_runtime_root():
    from fake_ollama.config import LlamaCppDefaults, LlamaCppTarget

    defaults = LlamaCppDefaults(runtime_root="C:/cudart")
    tgt = LlamaCppTarget(base_url="http://127.0.0.1:21500", model="qa")
    merged = tgt.with_defaults(defaults)
    assert merged.runtime_root == "C:/cudart"


def test_synthesize_start_argv_returns_unquoted_argv():
    """argv must be raw strings (no shell quoting). Spaces in paths stay
    as a single argv element so create_subprocess_exec passes them
    verbatim to llama-server."""
    from fake_ollama.config import LlamaCppTarget

    tgt = LlamaCppTarget(
        base_url="http://127.0.0.1:21500",
        model="qa",
        binary_path="C:/path with space/llama-server.exe",
        model_path="C:/m models/m.gguf",
        gpu_layers=99,
    )
    argv = tgt.synthesize_start_argv()
    assert argv is not None
    assert argv[0] == "C:/path with space/llama-server.exe"
    assert "C:/m models/m.gguf" in argv
    assert "-ngl" in argv and "99" in argv


def test_synthesize_start_argv_passes_flash_attn_value():
    from fake_ollama.config import LlamaCppTarget

    tgt = LlamaCppTarget(
        base_url="http://127.0.0.1:21500",
        model="qa",
        model_path="C:/m.gguf",
        flash_attn=True,
    )
    argv = tgt.synthesize_start_argv()
    assert argv is not None
    idx = argv.index("-fa")
    assert argv[idx + 1] == "on"


def test_synthesize_start_argv_returns_none_when_user_shell_command_set():
    from fake_ollama.config import LlamaCppTarget

    tgt = LlamaCppTarget(
        base_url="http://127.0.0.1:21500",
        model="qa",
        start_command="custom --serve",
        model_path="C:/m.gguf",
    )
    assert tgt.synthesize_start_argv() is None


@pytest.mark.asyncio
async def test_create_managed_subprocess_shell_normalizes_blank_cwd(
    monkeypatch: pytest.MonkeyPatch,
):
    """A blank cwd ('' or '   ') from config must be coerced to None;
    otherwise Windows raises WinError 123 ('invalid filename/directory')
    when the shell is spawned."""
    from fake_ollama import process_utils

    captured: dict = {}

    async def fake_create(command: str, **kwargs):
        captured.update(kwargs)

        class _P:
            returncode = 0

            def terminate(self):
                pass

            def kill(self):
                pass

            async def wait(self):
                return 0

        return _P()

    monkeypatch.setattr(
        "fake_ollama.process_utils.asyncio.create_subprocess_shell", fake_create
    )

    await process_utils.create_managed_subprocess_shell("echo hi", cwd="")
    assert captured["cwd"] is None

    captured.clear()
    await process_utils.create_managed_subprocess_shell("echo hi", cwd="   ")
    assert captured["cwd"] is None

    captured.clear()
    await process_utils.create_managed_subprocess_shell("echo hi", cwd=None)
    assert captured["cwd"] is None
