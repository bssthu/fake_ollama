import json

import pytest
from fastapi import HTTPException

from fake_ollama.converters import (
    AnthropicStreamTranslator, OpenAIChatStreamTranslator,
    anthropic_to_ollama_chat, ollama_messages_to_anthropic,
)
from fake_ollama.reverse_converters import (
    ollama_chat_to_anthropic, ollama_stream_to_anthropic_events,
    openai_stream_to_anthropic_events,
)


def tool_events(*, malformed=False):
    yield "message_start", {"message": {"id": "m"}}
    yield "content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}
    yield "content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "checking"}}
    yield "content_block_stop", {"index": 0}
    for i in range(2):
        yield "content_block_start", {"index": i + 1, "content_block": {"type": "tool_use", "id": f"tool-{i}", "name": "weather", "input": {}}}
        yield "content_block_delta", {"index": i + 1, "delta": {"type": "input_json_delta", "partial_json": '{"city":'}}
        if not malformed:
            yield "content_block_delta", {"index": i + 1, "delta": {"type": "input_json_delta", "partial_json": json.dumps(str(i)) + "}"}}
        yield "content_block_stop", {"index": i + 1}
    yield "message_delta", {"delta": {"stop_reason": "tool_use"}}
    yield "message_stop", {}


def test_anthropic_stream_tools_have_full_arguments_ids_indices_once():
    translator = AnthropicStreamTranslator("m")
    chunks = [chunk for kind, data in tool_events() for chunk in translator.feed_event(kind, data)]
    calls = [tc for chunk in chunks for tc in chunk.get("message", {}).get("tool_calls", [])]
    assert calls == [{"id": f"tool-{i}", "function": {"index": i, "name": "weather", "arguments": {"city": str(i)}}} for i in range(2)]
    assert "".join(c.get("message", {}).get("content", "") for c in chunks) == "checking"
    assert chunks[-1]["done_reason"] == "tool_calls"


def test_anthropic_nonstream_tools_have_ids_and_indices():
    response = anthropic_to_ollama_chat({"content": [{"type": "tool_use", "id": "tool-0", "name": "weather", "input": {"city": "0"}}], "stop_reason": "tool_use"}, ollama_model="m")
    assert response["message"]["tool_calls"] == [{"id": "tool-0", "function": {"index": 0, "name": "weather", "arguments": {"city": "0"}}}]
    assert response["done_reason"] == "tool_calls"


@pytest.mark.parametrize("translator_type", [AnthropicStreamTranslator, OpenAIChatStreamTranslator])
def test_malformed_tool_stream_never_finishes_successfully(translator_type):
    translator = translator_type("m")
    frames = [frame for kind, data in tool_events(malformed=True) for frame in translator.feed_event(kind, data)]
    assert frames[-1].get("error")
    assert not any(frame.get("done_reason") or any(c.get("finish_reason") for c in frame.get("choices", [])) for frame in frames)


def test_openai_translator_upstream_error_is_error_envelope():
    result = OpenAIChatStreamTranslator("m").feed_event("error", {"error": {"type": "overloaded_error", "message": "unavailable"}})
    assert result[0]["error"]["type"] == "overloaded_error"
    assert "choices" not in result[0]


def test_ollama_same_name_results_pair_in_order_across_rounds():
    history = []
    for _ in range(2):
        history.extend([
            {"role": "assistant", "tool_calls": [{"function": {"name": "weather", "arguments": {"city": city}}} for city in ("A", "B")]},
            {"role": "tool", "tool_name": "weather", "content": "1"},
            {"role": "tool", "tool_name": "weather", "content": "2"},
        ])
    _, messages = ollama_messages_to_anthropic(history)
    ids = [b["id"] for m in messages if m["role"] == "assistant" for b in m["content"] if b["type"] == "tool_use"]
    results = [b for m in messages if m["role"] == "user" for b in m["content"]]
    assert len(set(ids)) == 4
    assert [r["tool_use_id"] for r in results] == ids
    assert [r["content"] for r in results] == ["1", "2", "1", "2"]


@pytest.mark.parametrize("fault", ["missing", "extra", "wrong_name", "wrong_id"])
def test_ollama_invalid_tool_results_are_400(fault):
    messages = [{"role": "assistant", "tool_calls": [{"id": "id", "function": {"name": "weather", "arguments": {}}}]},
                {"role": "tool", "tool_name": "weather", "content": "1"}]
    if fault == "missing":
        messages.pop()
    elif fault == "extra":
        messages.append(dict(messages[-1]))
    elif fault == "wrong_name":
        messages[-1]["tool_name"] = "other"
    else:
        messages[-1]["tool_call_id"] = "unknown"
    with pytest.raises(HTTPException) as error:
        ollama_messages_to_anthropic(messages)
    assert error.value.status_code == 400


async def lines(values):
    for value in values:
        yield json.dumps(value)


async def test_ollama_to_anthropic_indices_thinking_and_tool_stop():
    events = [item async for item in ollama_stream_to_anthropic_events(lines([
        {"message": {"thinking": "plan"}}, {"message": {"content": "checking"}},
        {"message": {"tool_calls": [{"id": "t", "function": {"name": "weather", "arguments": {}}}]}},
        {"message": {"content": ""}, "done": True, "done_reason": "stop"},
    ]), anthropic_model="m")]
    starts = [data for kind, data in events if kind == "content_block_start"]
    assert [b["content_block"]["type"] for b in starts] == ["thinking", "text", "tool_use"]
    assert [b["index"] for b in starts] == [0, 1, 2]
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"


def test_ollama_nonstream_preserves_thinking_and_tool_stop():
    result = ollama_chat_to_anthropic({"message": {"thinking": "plan", "content": "checking", "tool_calls": [{"function": {"name": "weather", "arguments": {}}}]}, "done_reason": "stop"}, anthropic_model="m")
    assert result["content"][0] == {"type": "thinking", "thinking": "plan"}
    assert result["stop_reason"] == "tool_use"


@pytest.mark.parametrize("protocol", ["openai", "ollama"])
async def test_reverse_eof_is_error(protocol):
    async def source():
        if protocol == "openai":
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}'
        else:
            yield '{"message":{"content":"partial"}}'
    converter = openai_stream_to_anthropic_events if protocol == "openai" else ollama_stream_to_anthropic_events
    events = [item async for item in converter(source(), anthropic_model="m")]
    assert events[-1][0] == "error"
    assert not any(kind == "message_stop" for kind, data in events)


@pytest.mark.parametrize("translator_type", [AnthropicStreamTranslator, OpenAIChatStreamTranslator])
def test_translator_eof_error_is_terminal_once(translator_type):
    translator = translator_type("m")
    translator.feed_event("message_start", {"message": {}})
    assert translator.finish()[0].get("error")
    assert translator.finish() == []
    assert translator.feed_event("message_stop", {}) == []


@pytest.mark.parametrize("translator_type", [AnthropicStreamTranslator, OpenAIChatStreamTranslator])
def test_initial_tool_input_and_fragments_combine(translator_type):
    translator = translator_type("m")
    chunks = translator.feed_event("content_block_start", {"index": 3, "content_block": {
        "type": "tool_use", "id": "t", "name": "weather", "input": {"city": "A"},
    }})
    chunks += translator.feed_event("content_block_delta", {"index": 3, "delta": {"type": "input_json_delta", "partial_json": '{"units":"C"}'}})
    chunks += translator.feed_event("content_block_stop", {"index": 3})
    chunks += translator.feed_event("message_stop", {})
    if translator_type is AnthropicStreamTranslator:
        arguments = chunks[0]["message"]["tool_calls"][0]["function"]["arguments"]
    else:
        arguments = json.loads("".join(tc["function"].get("arguments", "") for c in chunks for tc in c["choices"][0]["delta"].get("tool_calls", [])))
    assert arguments == {"city": "A", "units": "C"}


@pytest.mark.parametrize("protocol", ["openai", "ollama"])
async def test_reverse_stream_cancellation_propagates(protocol):
    import asyncio
    entered = asyncio.Event()
    closed = asyncio.Event()
    async def source():
        try:
            entered.set()
            await asyncio.Event().wait()
            yield ""
        finally:
            closed.set()
    converter = openai_stream_to_anthropic_events if protocol == "openai" else ollama_stream_to_anthropic_events
    async def consume():
        return [item async for item in converter(source(), anthropic_model="m")]
    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
