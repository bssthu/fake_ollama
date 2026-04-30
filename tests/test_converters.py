"""Unit tests for the format converters."""

from __future__ import annotations

from fake_ollama.converters import (
    AnthropicStreamTranslator,
    anthropic_to_ollama_chat,
    anthropic_to_ollama_generate,
    ollama_chat_to_anthropic,
    ollama_generate_to_anthropic,
    ollama_messages_to_anthropic,
    openai_chat_to_anthropic,
)


def test_messages_split_system_and_roles():
    system, msgs = ollama_messages_to_anthropic(
        [
            {"role": "system", "content": "be brief"},
            {"role": "system", "content": "and helpful"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you?"},
        ]
    )
    assert system == "be brief\n\nand helpful"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "hi"


def test_chat_to_anthropic_maps_options():
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 64, "top_p": 0.9, "stop": ["END"]},
    }
    body = ollama_chat_to_anthropic(
        payload, upstream_model="claude-3-5-sonnet-20241022", default_max_tokens=2048
    )
    assert body["model"] == "claude-3-5-sonnet-20241022"
    assert body["max_tokens"] == 64  # num_predict beats default
    assert body["temperature"] == 0.2
    assert body["top_p"] == 0.9
    assert body["stop_sequences"] == ["END"]
    assert body["stream"] is False


def test_chat_to_anthropic_uses_default_max_tokens():
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    body = ollama_chat_to_anthropic(
        payload, upstream_model="m", default_max_tokens=777
    )
    assert body["max_tokens"] == 777


def test_chat_to_anthropic_image_blocks():
    payload = {
        "messages": [
            {"role": "user", "content": "what is this?", "images": ["YmFzZTY0ZGF0YQ=="]}
        ]
    }
    body = ollama_chat_to_anthropic(payload, upstream_model="m", default_max_tokens=10)
    content = body["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"
    assert content[0]["source"]["data"] == "YmFzZTY0ZGF0YQ=="
    assert content[1] == {"type": "text", "text": "what is this?"}


def test_generate_to_anthropic_includes_system_and_prompt():
    payload = {
        "prompt": "tell me a joke",
        "system": "you are funny",
        "options": {"temperature": 0.5},
    }
    body = ollama_generate_to_anthropic(
        payload, upstream_model="m", default_max_tokens=100
    )
    assert body["system"] == "you are funny"
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "tell me a joke"
    assert body["temperature"] == 0.5


def test_anthropic_to_ollama_chat_basic():
    upstream = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello there"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    out = anthropic_to_ollama_chat(upstream, ollama_model="claude")
    assert out["done"] is True
    assert out["done_reason"] == "stop"
    assert out["message"] == {"role": "assistant", "content": "hello there"}
    assert out["prompt_eval_count"] == 5
    assert out["eval_count"] == 3
    assert out["model"] == "claude"


def test_anthropic_to_ollama_generate_basic():
    upstream = {
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 1, "output_tokens": 9},
    }
    out = anthropic_to_ollama_generate(upstream, ollama_model="m")
    assert out["response"] == "hi"
    assert out["done_reason"] == "length"
    assert out["context"] == []


def test_stream_translator_chat_flow():
    t = AnthropicStreamTranslator("claude", mode="chat")
    chunks = []
    chunks += t.feed_event(
        "message_start",
        {"message": {"usage": {"input_tokens": 4, "output_tokens": 0}}},
    )
    chunks += t.feed_event(
        "content_block_delta", {"delta": {"type": "text_delta", "text": "Hel"}}
    )
    chunks += t.feed_event(
        "content_block_delta", {"delta": {"type": "text_delta", "text": "lo"}}
    )
    chunks += t.feed_event(
        "message_delta",
        {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
    )
    chunks += t.feed_event("message_stop", {})

    text_chunks = [c for c in chunks if not c["done"]]
    final_chunks = [c for c in chunks if c["done"]]
    assert "".join(c["message"]["content"] for c in text_chunks) == "Hello"
    assert len(final_chunks) == 1
    final = final_chunks[0]
    assert final["done_reason"] == "stop"
    assert final["prompt_eval_count"] == 4
    assert final["eval_count"] == 2
    assert final["message"] == {"role": "assistant", "content": ""}


def test_stream_translator_generate_mode():
    t = AnthropicStreamTranslator("m", mode="generate")
    chunks = t.feed_event(
        "content_block_delta", {"delta": {"type": "text_delta", "text": "ok"}}
    )
    assert chunks[0]["response"] == "ok"
    final = t.feed_event("message_stop", {})[0]
    assert final["done"] is True
    assert "context" in final

def test_openai_assistant_tool_calls_preserved_in_history():
    payload = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "what is 2+2?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{\"expr\": \"2+2\"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "4"},
        ],
    }
    body = openai_chat_to_anthropic(payload, upstream_model="m", default_max_tokens=100)
    assert body["messages"][1]["role"] == "assistant"
    blocks = body["messages"][1]["content"]
    assert isinstance(blocks, list)
    tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
    assert tool_uses == [{"type": "tool_use", "id": "call_abc", "name": "calc", "input": {"expr": "2+2"}}]
    # tool_result must reference the same id
    assert body["messages"][2]["content"][0]["tool_use_id"] == "call_abc"

def test_openai_assistant_think_tag_preserved_as_thinking_block():
    payload = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>plan stuff</think>Hello!"},
            {"role": "user", "content": "go on"},
        ],
    }
    body = openai_chat_to_anthropic(payload, upstream_model="m", default_max_tokens=100)
    assistant = body["messages"][1]
    assert assistant["role"] == "assistant"
    blocks = assistant["content"]
    assert isinstance(blocks, list)
    assert blocks[0] == {"type": "thinking", "thinking": "plan stuff"}
    text_blocks = [b for b in blocks if b.get("type") == "text"]
    assert text_blocks and text_blocks[0]["text"] == "Hello!"


def test_openai_assistant_reasoning_content_field_preserved():
    payload = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "Hello!",
                "reasoning_content": "let me think",
            },
            {"role": "user", "content": "go on"},
        ],
    }
    body = openai_chat_to_anthropic(payload, upstream_model="m", default_max_tokens=100)
    blocks = body["messages"][1]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "let me think"}


def test_ollama_assistant_think_tag_preserved_as_thinking_block():
    system, msgs = ollama_messages_to_anthropic(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<think>reasoning</think>final"},
            {"role": "user", "content": "more"},
        ]
    )
    blocks = msgs[1]["content"]
    assert isinstance(blocks, list)
    assert blocks[0] == {"type": "thinking", "thinking": "reasoning"}
    assert any(b.get("type") == "text" and b.get("text") == "final" for b in blocks)

def test_thinking_cache_reinjects_via_tool_use_id():
    from fake_ollama.converters import (
        _clear_thinking_cache,
        anthropic_to_openai_chat,
        openai_chat_to_anthropic,
    )

    _clear_thinking_cache()
    # Simulate the upstream response we converted on turn 1.
    upstream_resp = {
        "id": "msg_1",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 7},
        "content": [
            {"type": "thinking", "thinking": "I should call calc.", "signature": "sigXYZ"},
            {"type": "text", "text": "Let me compute it."},
            {"type": "tool_use", "id": "call_abc", "name": "calc", "input": {"expr": "2+2"}},
        ],
    }
    anthropic_to_openai_chat(upstream_resp, openai_model="m", show_thinking=False)

    # Turn 2: the OpenAI client echoes the tool_calls back (without any
    # thinking content) and then provides the tool result.
    payload = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "what is 2+2?"},
            {
                "role": "assistant",
                "content": "Let me compute it.",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "calc", "arguments": "{\"expr\": \"2+2\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "4"},
        ],
    }
    body = openai_chat_to_anthropic(payload, upstream_model="m", default_max_tokens=100)
    blocks = body["messages"][1]["content"]
    # Cached thinking block (with signature) must be the first content block.
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == "I should call calc."
    assert blocks[0]["signature"] == "sigXYZ"
    # Followed by text + tool_use blocks.
    assert any(b.get("type") == "text" for b in blocks[1:])
    assert any(b.get("type") == "tool_use" and b.get("id") == "call_abc" for b in blocks)


def test_thinking_cache_recall_tolerates_reformatted_text():
    """Even when the client trims/lowercases/whitespace-collapses the
    echoed assistant text, the cached thinking block must still come back.
    """
    from fake_ollama.converters import (
        _clear_thinking_cache,
        anthropic_to_openai_chat,
        openai_chat_to_anthropic,
    )

    _clear_thinking_cache()
    upstream_resp = {
        "id": "msg_2",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 4},
        "content": [
            {"type": "thinking", "thinking": "Quietly thinking.", "signature": "sigQ"},
            {"type": "text", "text": "Hello, world!  This is a multi-line\n  reply."},
        ],
    }
    anthropic_to_openai_chat(upstream_resp, openai_model="m", show_thinking=False)

    payload = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "hi"},
            # Client reformatted: collapsed whitespace + lowercased.
            {"role": "assistant", "content": "hello, world! this is a multi-line reply."},
            {"role": "user", "content": "thanks"},
        ],
    }
    body = openai_chat_to_anthropic(payload, upstream_model="m", default_max_tokens=100)
    asst = body["messages"][1]
    assert isinstance(asst["content"], list)
    assert asst["content"][0]["type"] == "thinking"
    assert asst["content"][0]["signature"] == "sigQ"


def test_thinking_cache_last_fallback_when_text_completely_changed():
    from fake_ollama.converters import (
        _clear_thinking_cache,
        anthropic_to_openai_chat,
        openai_chat_to_anthropic,
    )

    _clear_thinking_cache()
    upstream_resp = {
        "id": "msg_3",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {"type": "thinking", "thinking": "deep thought", "signature": "sigZ"},
            {"type": "text", "text": "original answer"},
        ],
    }
    anthropic_to_openai_chat(upstream_resp, openai_model="m", show_thinking=False)

    # The client sent something completely unrelated as the assistant text
    # (e.g. a heavily edited summary). Last-write fallback must still inject
    # the cached thinking so DeepSeek does not 400.
    payload = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "totally different summary text"},
            {"role": "user", "content": "go on"},
        ],
    }
    body = openai_chat_to_anthropic(payload, upstream_model="m", default_max_tokens=100)
    asst = body["messages"][1]
    assert isinstance(asst["content"], list)
    assert asst["content"][0]["type"] == "thinking"
    assert asst["content"][0]["signature"] == "sigZ"

def test_openai_consecutive_tool_messages_merged_into_one_user_message():
    payload = {
        "model": "x",
        "messages": [
            {"role": "user", "content": "do A and B"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "A done"},
            {"role": "tool", "tool_call_id": "call_2", "content": "B done"},
        ],
    }
    body = openai_chat_to_anthropic(payload, upstream_model="m", default_max_tokens=100)
    # Should be exactly 3 messages: user, assistant(tool_use x2), user(tool_result x2).
    assert len(body["messages"]) == 3
    last = body["messages"][2]
    assert last["role"] == "user"
    assert [b["type"] for b in last["content"]] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in last["content"]] == ["call_1", "call_2"]
