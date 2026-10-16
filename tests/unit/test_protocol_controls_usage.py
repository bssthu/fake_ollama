import json

import pytest
from fastapi import HTTPException

from fake_ollama.converters import (openai_chat_to_anthropic, anthropic_to_openai_chat,
    anthropic_to_ollama_chat, AnthropicStreamTranslator, OpenAIChatStreamTranslator)
from fake_ollama.reverse_converters import (anthropic_to_openai_chat as request_to_openai,
    openai_chat_to_anthropic as response_to_anthropic, ollama_chat_to_anthropic,
    openai_stream_to_anthropic_events)

OPENAI_TOOLS = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
ANTHROPIC_TOOLS = [{"name": "weather", "input_schema": {"type": "object"}}]


@pytest.mark.parametrize("choice,expected", [
    ("none", {"type": "none"}), ("auto", {"type": "auto"}), ("required", {"type": "any"}),
    ({"type": "function", "function": {"name": "weather"}}, {"type": "tool", "name": "weather"}),
])
@pytest.mark.parametrize("parallel", [True, False])
def test_openai_controls_reach_anthropic(choice, expected, parallel):
    result = openai_chat_to_anthropic({"messages": [], "tools": OPENAI_TOOLS, "tool_choice": choice,
        "parallel_tool_calls": parallel}, upstream_model="m", default_max_tokens=100)
    expected = dict(expected)
    if expected["type"] != "none":
        expected["disable_parallel_tool_use"] = not parallel
    assert result["tool_choice"] == expected


@pytest.mark.parametrize("choice,expected", [
    ({"type": "none"}, "none"), ({"type": "auto", "disable_parallel_tool_use": True}, "auto"),
    ({"type": "any", "disable_parallel_tool_use": False}, "required"),
    ({"type": "tool", "name": "weather", "disable_parallel_tool_use": True}, {"type": "function", "function": {"name": "weather"}}),
])
def test_anthropic_controls_reach_openai(choice, expected):
    result = request_to_openai({"messages": [], "tools": ANTHROPIC_TOOLS, "tool_choice": choice}, target_model="m")
    assert result["tool_choice"] == expected
    if "disable_parallel_tool_use" in choice:
        assert result["parallel_tool_calls"] is not choice["disable_parallel_tool_use"]


@pytest.mark.parametrize("fields", [
    {"tool_choice": "surprise"}, {"parallel_tool_calls": "false"},
    {"tool_choice": {"type": "function", "function": {"name": "missing"}}},
    {"tool_choice": {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": []}}},
    {"tools": [], "tool_choice": "required"},
    {"stream": True, "stream_options": {"include_usage": "true"}},
])
def test_unrepresentable_or_invalid_controls_are_400(fields):
    with pytest.raises(HTTPException) as error:
        openai_chat_to_anthropic({"messages": [], "tools": OPENAI_TOOLS, **fields}, upstream_model="m", default_max_tokens=100)
    assert error.value.status_code == 400


def test_anthropic_stream_requests_openai_usage():
    result = request_to_openai({"messages": [], "stream": True}, target_model="m")
    assert result["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("kind", ["openai", "ollama", "anthropic_from_openai", "anthropic_from_ollama"])
def test_nonstream_unknown_usage_is_not_zero(kind):
    if kind == "openai":
        usage = anthropic_to_openai_chat({"content": []}, openai_model="m").get("usage") or {}
    elif kind == "ollama":
        response = anthropic_to_ollama_chat({"content": []}, ollama_model="m")
        usage = {key: response[key] for key in ("eval_count", "prompt_eval_count") if key in response}
    elif kind == "anthropic_from_openai":
        usage = response_to_anthropic({"choices": []}, anthropic_model="m")["usage"]
    else:
        usage = ollama_chat_to_anthropic({"message": {}}, anthropic_model="m")["usage"]
    assert not usage


def test_stream_unknown_ollama_usage_is_not_zero():
    translator = AnthropicStreamTranslator("m")
    translator.feed_event("message_start", {"message": {}})
    final = translator.feed_event("message_stop", {})[-1]
    assert "eval_count" not in final and "prompt_eval_count" not in final


async def test_openai_usage_only_chunk_preserves_known_zero_and_missing_counts():
    async def lines():
        for value in [
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 0}},
        ]:
            yield "data: " + json.dumps(value)
        yield "data: [DONE]"
    events = [event async for event in openai_stream_to_anthropic_events(lines(), anthropic_model="m")]
    assert events[0][1]["message"]["usage"] == {}
    assert events[-2][1]["usage"] == {"input_tokens": 0}


def test_openai_translator_emits_final_usage_only_chunk():
    translator = OpenAIChatStreamTranslator("m", include_usage=True)
    first = translator.feed_event("message_start", {"message": {"usage": {"input_tokens": 7, "output_tokens": 0}}})
    translator.feed_event("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}})
    frames = first + translator.feed_event("message_stop", {})
    assert frames[-1]["choices"] == []
    assert frames[-1]["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert all(frame["usage"] is None for frame in frames[:-1])


@pytest.mark.parametrize("convert,keyword", [(anthropic_to_openai_chat, "openai_model"), (anthropic_to_ollama_chat, "ollama_model")])
@pytest.mark.parametrize("ids", [[None], ["same", "same"]])
def test_nonstream_tools_reject_missing_or_duplicate_upstream_identity(convert, keyword, ids):
    from fake_ollama.protocol_tools import UpstreamProtocolError
    blocks = [{"type": "tool_use", "id": value, "name": "weather", "input": {}} for value in ids]
    with pytest.raises(UpstreamProtocolError):
        convert({"content": blocks}, **{keyword: "m"})


def test_invalid_openai_history_tool_json_returns_400():
    with pytest.raises(HTTPException) as error:
        openai_chat_to_anthropic({"messages": [{"role": "assistant", "tool_calls": [
            {"id": "call", "function": {"name": "weather", "arguments": '{"city":'}}]}]},
            upstream_model="m", default_max_tokens=100)
    assert error.value.status_code == 400


def test_native_ollama_missing_ids_get_response_unique_associations():
    response = {"message": {"tool_calls": [{"function": {"name": "weather", "arguments": {}}}]}}
    one = ollama_chat_to_anthropic(response, anthropic_model="m")["content"][0]["id"]
    two = ollama_chat_to_anthropic(response, anthropic_model="m")["content"][0]["id"]
    assert one and two and one != two
