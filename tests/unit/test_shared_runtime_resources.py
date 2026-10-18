"""One physical model must not be reserved twice by two protocol clients."""
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest

from fake_ollama.config import DecisionTarget
from fake_ollama.decision_client import DecisionClient
from fake_ollama.generic_openai_client import GenericOpenAIClient
from fake_ollama.ollama_client import OllamaClient
from fake_ollama.vram import (LocalTargetResourceError, MemoryCoordinator,
                             MemoryReleaseCandidate, VramCoordinator,
                             VramReleaseCandidate)


@asynccontextmanager
async def clients(resource, free, chat_kind="generic"):
    async def provider():
        return free["mib"]
    coordinator = (VramCoordinator if resource == "vram" else MemoryCoordinator)(
        provider=provider, total_provider=provider)
    transport = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={})))
    kwargs = {resource + "_coordinator": coordinator, "client": transport}
    chat_type = GenericOpenAIClient if chat_kind == "generic" else OllamaClient
    chat = chat_type("http://127.0.0.1:28911", target_name="chat", **kwargs)
    decision = DecisionClient(DecisionTarget(name="decision", models=["m"],
        base_url="http://127.0.0.1:28911"), **kwargs)
    try:
        assert chat._ownership is decision._ownership
        yield coordinator, chat, decision
    finally:
        await chat.aclose()
        await decision.aclose()
        await transport.aclose()


@pytest.mark.parametrize("resource", ["vram", "memory"])
@pytest.mark.parametrize("chat_kind", ["generic", "ollama"])
async def test_protocol_siblings_share_pending_and_cancel_independently(resource, chat_kind):
    free = {"mib": 6 * 1024.0}
    async with clients(resource, free, chat_kind) as (coordinator, chat, decision):
        await getattr(chat, "_ensure_" + resource)("m", 4)
        await getattr(decision, "_ensure_" + resource)("m", 4)
        assert coordinator.has_pending(chat.target_id, "m")
        assert coordinator.has_pending(decision.target_id, "m")
        assert coordinator._pending_used_mib() == 4 * 1024
        coordinator.discard_pending(chat.target_id, "m")
        assert coordinator.has_pending(decision.target_id, "m")
        assert coordinator._pending_used_mib() == 4 * 1024
        coordinator.discard_pending(decision.target_id, "m")
        assert coordinator._pending_used_mib() == 0


@pytest.mark.parametrize("resource", ["vram", "memory"])
@pytest.mark.parametrize("first", ["chat", "decision"])
async def test_loaded_sibling_reuses_residency_with_low_free_memory(resource, first):
    free = {"mib": 6 * 1024.0}
    async with clients(resource, free) as (coordinator, chat, decision):
        one, two = (chat, decision) if first == "chat" else (decision, chat)
        await getattr(one, "_ensure_" + resource)("m", 4)
        getattr(one, "_mark_" + resource + "_reserved")("m", 4)
        # Discard the temporary measurement grace to test persistent residency.
        coordinator.discard_pending(one.target_id, "m")
        free["mib"] = 1
        await getattr(two, "_ensure_" + resource)("m", 4)
        assert not coordinator.has_pending(two.target_id, "m")
        with pytest.raises(LocalTargetResourceError):
            await getattr(two, "_ensure_" + resource)("different-model", 4)


@pytest.mark.parametrize("resource", ["vram", "memory"])
async def test_shared_pending_uses_larger_estimate_without_double_count(resource):
    free = {"mib": 6 * 1024.0}
    async with clients(resource, free) as (coordinator, chat, decision):
        await getattr(chat, "_ensure_" + resource)("m", 3)
        await getattr(decision, "_ensure_" + resource)("m", 5)
        assert coordinator._pending_used_mib() == 5 * 1024
        coordinator.confirm_loaded(decision.target_id, "m")
        assert coordinator._pending[(chat.target_id, "m")].confirmed_loaded_at is not None


@pytest.mark.parametrize("resource", ["vram", "memory"])
async def test_distinct_models_on_shared_runtime_reserve_separately(resource):
    free = {"mib": 6 * 1024.0}
    async with clients(resource, free) as (coordinator, chat, decision):
        await getattr(chat, "_ensure_" + resource)("m", 4)
        with pytest.raises(LocalTargetResourceError):
            await getattr(decision, "_ensure_" + resource)("different-model", 4)
        assert coordinator._pending_used_mib() == 4 * 1024


@pytest.mark.parametrize("resource", ["vram", "memory"])
def test_multi_model_sibling_uses_exact_dictionary_residency(resource):
    coordinator = (VramCoordinator if resource == "vram" else MemoryCoordinator)()
    participant = SimpleNamespace(_loaded_models={"a": object(), "b": object()},
                                  _loaded_model=SimpleNamespace(model="b"))
    setattr(participant, "has_" + resource + "_reservation", lambda model: model in participant._loaded_models)
    assert coordinator._sibling_has_model(participant, "a")
    assert coordinator._sibling_has_model(participant, "b")
    assert not coordinator._sibling_has_model(participant, "c")


def test_reclaim_estimates_deduplicate_physical_model_and_check_sibling_busy():
    coordinator = VramCoordinator()
    class Owner:
        is_busy = False
        def busy(self):
            return self.is_busy
    owner = Owner()
    async def release():
        return True
    for target, amount in (("chat", 4), ("decision", 5)):
        participant = SimpleNamespace(target_id=target, _ownership=owner, active_requests=0)
        participant.vram_release_candidates = lambda *, now, idle_seconds, target=target, amount=amount: [
            VramReleaseCandidate(target, "m", amount, 0, release)]
        coordinator.register(participant)
    candidates = coordinator._eligible_candidates("other")
    assert len(candidates) == 1
    assert candidates[0].estimated_vram_gb == 5
    owner.is_busy = True
    assert coordinator._eligible_candidates("other") == []


@pytest.mark.parametrize("resource", ["vram", "memory"])
@pytest.mark.parametrize("target", ["chat", "decision"])
@pytest.mark.parametrize("force", [False, True])
async def test_explicit_reclaim_accepts_either_shared_alias(resource, target, force):
    coordinator = (VramCoordinator if resource == "vram" else MemoryCoordinator)()
    # RuntimeOwner is hashable, as physical identities must be.
    owner = type("Owner", (), {"busy": lambda self: False})()
    releases = []
    async def release():
        releases.append("physical model")
        return True
    candidate_type = VramReleaseCandidate if resource == "vram" else MemoryReleaseCandidate
    for name, amount in (("chat", 4), ("decision", 5)):
        participant = SimpleNamespace(target_id=name, _ownership=owner, active_requests=0)
        getter = lambda *, now, idle_seconds=0, name=name, amount=amount: [
            candidate_type(name, "m", amount, 0, release)]
        setattr(participant, resource + "_release_candidates", getter)
        setattr(participant, resource + "_force_release_candidates", getter)
        coordinator.register(participant)
    result = await coordinator.reclaim_model(target_id=target, model="m", force=force)
    assert result["released"] is True
    assert result["target_id"] == target
    assert result["estimated_" + resource + "_gb"] == 5
    assert releases == ["physical model"]


@pytest.mark.parametrize("resource", ["vram", "memory"])
async def test_automatic_reclaim_calls_shared_physical_model_only_once(resource, monkeypatch):
    monkeypatch.setattr("fake_ollama.vram._POST_RELEASE_REFRESH_DELAYS_SECONDS", (0.0,))
    async def provider():
        return 0.0
    coordinator = (VramCoordinator if resource == "vram" else MemoryCoordinator)(provider=provider)
    owner = type("Owner", (), {"busy": lambda self: False})()
    releases = []
    async def release():
        releases.append("physical model")
        return True
    candidate_type = VramReleaseCandidate if resource == "vram" else MemoryReleaseCandidate
    for name, amount in (("chat", 4), ("decision", 5)):
        participant = SimpleNamespace(target_id=name, _ownership=owner, active_requests=0)
        setattr(participant, resource + "_release_candidates", lambda *, now, idle_seconds, name=name, amount=amount: [
            candidate_type(name, "m", amount, 0, release)])
        coordinator.register(participant)
    result = await coordinator.reclaim_if_below(threshold_mib=1024)
    assert releases == ["physical model"]
    assert len(result["released"]) == 1
