import asyncio

import pytest
from fastapi import HTTPException

from fake_ollama import thinking_cache as cache
from fake_ollama.converters import (
    anthropic_to_openai_chat, openai_chat_to_anthropic,
    ollama_messages_to_anthropic,
)


@pytest.fixture(autouse=True)
def isolated_cache():
    cache.invalidate()
    token = cache._SCOPE.set(None)
    yield
    cache.reset_scope(token)
    cache.invalidate()


def bind(**overrides):
    values = dict(principal="synthetic-A", backend={"url": "https://a.test", "credential": "synthetic-upstream"},
                  model="model-A", configuration="generation-A", allow_unsigned=True)
    values.update(overrides)
    return cache.bind_scope(**values)


def seed(history=None, *, text="好的", tool=False):
    history = history or [{"role": "user", "content": "session-A"}]
    cache.set_history(history)
    blocks = [{"type": "thinking", "thinking": "private-A", "signature": "signature-A"},
              {"type": "text", "text": text}]
    if tool:
        blocks.append({"type": "tool_use", "id": "repeated-id", "name": "calc", "input": {"x": 1}})
    return anthropic_to_openai_chat({"content": blocks}, openai_model="m", show_thinking=False)["choices"][0]["message"]


def convert(history, protocol="openai"):
    if protocol == "ollama":
        return ollama_messages_to_anthropic(history)[1]
    return openai_chat_to_anthropic({"messages": history}, upstream_model="m", default_max_tokens=10)["messages"]


def thoughts(message):
    content = message["content"]
    return [b for b in content if b.get("type") == "thinking"] if isinstance(content, list) else []


@pytest.mark.parametrize("protocol", ["openai", "ollama"])
@pytest.mark.parametrize("change", ["principal", "backend", "model", "configuration", "history", "text"])
def test_exact_history_and_scope_required(protocol, change):
    bind()
    reply = seed()
    history = [{"role": "user", "content": "session-A"}, reply]
    if change in ("history", "text"):
        history[0 if change == "history" else 1]["content"] += " changed"
    else:
        bind(**{change: "different"})
    assert not thoughts(convert(history, protocol)[1])


@pytest.mark.parametrize("protocol", ["openai", "ollama"])
def test_each_assistant_turn_recovers_its_own_thinking(protocol):
    bind()
    first = seed()
    history = [{"role": "user", "content": "session-A"}, first, {"role": "user", "content": "next"}]
    cache.set_history(history)
    cache.remember_thinking([{"type": "thinking", "thinking": "private-B", "signature": "signature-B"}], text="好的")
    history.append({"role": "assistant", "content": "好的"})
    result = convert(history, protocol)
    assert thoughts(result[1])[0]["signature"] == "signature-A"
    assert thoughts(result[3])[0]["signature"] == "signature-B"


@pytest.mark.parametrize("protocol", ["openai", "ollama"])
@pytest.mark.parametrize("explicit", ["reasoning_content", "inline"])
def test_explicit_reasoning_wins_even_on_exact_hit(protocol, explicit):
    bind()
    reply = seed()
    if explicit == "inline":
        reply["content"] = "<think>current</think>好的"
    else:
        reply[explicit] = "current"
    result = convert([{"role": "user", "content": "session-A"}, reply], protocol)
    assert thoughts(result[1]) == [{"type": "thinking", "thinking": "current"}]


@pytest.mark.parametrize("change", ["missing", "arguments", "name", "history"])
def test_tool_id_alone_is_not_authority(change):
    bind()
    reply = seed(tool=True)
    history = [{"role": "user", "content": "session-A"}, reply]
    if change == "missing":
        reply["tool_calls"][0].pop("id")
    elif change in ("arguments", "name"):
        reply["tool_calls"][0]["function"][change] = "{}" if change == "arguments" else "other"
    else:
        history[0]["content"] = "session-B"
    assert not thoughts(convert(history)[1])


def test_unsigned_policy_and_missing_error():
    bind(allow_unsigned=False)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "answer", "reasoning_content": "current"}]
    assert not thoughts(convert(history)[1])
    bind(allow_unsigned=False, missing_policy="error")
    with pytest.raises(HTTPException) as error:
        convert(history)
    assert error.value.status_code == 400
    assert "current" not in error.value.detail
    bind(missing_policy="error")
    assert thoughts(convert(history)[1])[0]["thinking"] == "current"


def test_ttl_capacity_copy_and_reload_invalidate_inflight_writer(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cache, "_MAX_ENTRIES", 2)
    bind()
    reply = seed()
    history = [{"role": "user", "content": "session-A"}]
    got = cache.recall_thinking(history=history, message=reply)
    got[0]["thinking"] = "mutated"
    assert cache.recall_thinking(history=history, message=reply)[0]["thinking"] == "private-A"
    now[0] += cache._TTL_SECONDS
    assert cache.recall_thinking(history=history, message=reply) == []
    for i in range(3):
        seed(text=str(i))
    assert len(cache._CACHE) == 2
    assert cache.recall_thinking(history=history, message={"role": "assistant", "content": "0"}) == []
    cache.invalidate()
    seed()  # An old response finishes AFTER reload: it must not repopulate.
    assert not cache._CACHE
    bind(configuration="generation-B")
    assert cache.recall_thinking(history=history, message=reply) == []


async def test_concurrent_sessions_credentials_models_and_repeated_tool_ids():
    ready = asyncio.Event()
    count = 0

    async def session(i):
        nonlocal count
        bind(principal=f"credential-{i % 2}", model=f"model-{i // 2}")
        history = [{"role": "user", "content": f"session-{i}"}]
        cache.set_history(history)
        blocks = [{"type": "thinking", "thinking": f"private-{i}", "signature": f"signature-{i}"},
                  {"type": "tool_use", "id": "repeated-id", "name": "calc", "input": {}}]
        reply = anthropic_to_openai_chat({"content": blocks}, openai_model="m", show_thinking=False)["choices"][0]["message"]
        count += 1
        if count == 4:
            ready.set()
        await ready.wait()
        result = convert(history + [reply])
        assert thoughts(result[1])[0]["thinking"] == f"private-{i}"
    await asyncio.gather(*(session(i) for i in range(4)))


def test_cache_contains_no_credential_or_plain_history():
    bind()
    seed()
    snapshot = repr(cache._CACHE)
    assert "synthetic-A" not in snapshot
    assert "synthetic-upstream" not in snapshot
    assert "session-A" not in snapshot


def test_unscoped_converters_cannot_use_cache():
    bind()
    reply = seed()
    cache._SCOPE.set(None)
    assert not thoughts(convert([{"role": "user", "content": "session-A"}, reply])[1])
