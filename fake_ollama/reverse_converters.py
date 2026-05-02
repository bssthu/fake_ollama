"""Anthropic Messages API <-> Ollama chat conversions (reverse proxy).

This is the mirror of `converters.py`: it accepts requests in the Anthropic
Messages format and converts them to/from a local Ollama-compatible server.

Scope (MVP):
- Text, tool-calling content blocks, and base64 image blocks. Image blocks
    are forwarded to Ollama's per-message ``images`` array when possible; URL
    image sources are surfaced as a visible placeholder because Ollama's chat
    API expects base64 image data.
- Streaming and non-streaming requests.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_data_uri(value: str) -> str:
    if value.startswith("data:") and "," in value:
        return value.split(",", 1)[1]
    return value


def _image_data_from_block(block: Dict[str, Any]) -> Optional[str]:
    """Return base64 image data from an Anthropic/OpenAI-style image block."""
    source = block.get("source") if isinstance(block.get("source"), dict) else {}
    data = source.get("data")
    if isinstance(data, str) and data:
        return _strip_data_uri(data)
    url = source.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        return _strip_data_uri(url)

    image_url = block.get("image_url") if isinstance(block.get("image_url"), dict) else {}
    url = image_url.get("url")
    if isinstance(url, str) and url.startswith("data:"):
        return _strip_data_uri(url)
    return None


def _flatten_text_content(content: Any) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """Return (joined_text, tool_call_blocks, images) from Anthropic content."""
    if isinstance(content, str):
        return content, [], []
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    images: List[str] = []
    if not isinstance(content, list):
        return "", [], []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            # Drop on the way to Ollama; the local model has its own thinking.
            continue
        elif btype == "tool_use":
            tool_calls.append(block)
        elif btype in ("image", "image_url"):
            image = _image_data_from_block(block)
            if image:
                images.append(image)
            else:
                text_parts.append("[image omitted: Ollama requires base64 image data]")
    return "".join(text_parts), tool_calls, images


def _flatten_tool_result(content: Any) -> str:
    """Anthropic tool_result content -> plain string (Ollama wants a string)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
            elif isinstance(block, str):
                out.append(block)
        return "".join(out)
    if content is None:
        return ""
    return json.dumps(content)


def _system_to_string(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: List[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n\n".join(p for p in parts if p)
    return ""


# ---------------------------------------------------------------------------
# Anthropic request -> Ollama chat payload
# ---------------------------------------------------------------------------


def anthropic_to_ollama_chat(
    payload: Dict[str, Any],
    *,
    target_model: str,
    default_max_tokens: int = 1024,
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []

    sys = _system_to_string(payload.get("system"))
    if sys:
        messages.append({"role": "system", "content": sys})

    for msg in payload.get("messages") or []:
        role = msg.get("role", "user")
        content = msg.get("content")

        # Anthropic encodes tool_result inside a *user* message with content
        # blocks. Ollama wants a separate {"role":"tool"} message per result.
        if role == "user" and isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            text_buffer: List[str] = []
            image_buffer: List[str] = []

            def flush_user_message() -> None:
                nonlocal text_buffer, image_buffer
                if not text_buffer and not image_buffer:
                    return
                user_msg: Dict[str, Any] = {
                    "role": "user",
                    "content": "".join(text_buffer),
                }
                if image_buffer:
                    user_msg["images"] = list(image_buffer)
                messages.append(user_msg)
                text_buffer = []
                image_buffer = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_result":
                    flush_user_message()
                    messages.append(
                        {
                            "role": "tool",
                            "content": _flatten_tool_result(block.get("content")),
                            "tool_call_id": block.get("tool_use_id", ""),
                        }
                    )
                else:
                    text, _, images = _flatten_text_content([block])
                    if text:
                        text_buffer.append(text)
                    if images:
                        image_buffer.extend(images)
            flush_user_message()
            continue

        text, tool_calls, images = _flatten_text_content(content)
        if role == "assistant" and tool_calls:
            ollama_msg: Dict[str, Any] = {"role": "assistant", "content": text}
            if images:
                ollama_msg["images"] = images
            ollama_msg["tool_calls"] = [
                {
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": tc.get("input") or {},
                    },
                    "id": tc.get("id", ""),
                }
                for tc in tool_calls
            ]
            messages.append(ollama_msg)
        else:
            ollama_msg = {"role": role, "content": text}
            if images:
                ollama_msg["images"] = images
            messages.append(ollama_msg)

    # Tools
    tools_out: List[Dict[str, Any]] = []
    for t in payload.get("tools") or []:
        if not isinstance(t, dict):
            continue
        tools_out.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object"},
                },
            }
        )

    options: Dict[str, Any] = {}
    max_tokens = payload.get("max_tokens") or default_max_tokens
    if max_tokens:
        options["num_predict"] = int(max_tokens)
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if payload.get(src) is not None:
            options[dst] = payload[src]
    stops = payload.get("stop_sequences")
    if stops:
        options["stop"] = stops

    body: Dict[str, Any] = {"model": target_model, "messages": messages}
    if options:
        body["options"] = options
    if tools_out:
        body["tools"] = tools_out
    return body


# ---------------------------------------------------------------------------
# Ollama response -> Anthropic Messages response
# ---------------------------------------------------------------------------


_DONE_REASON_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "load": "end_turn",
    "tool_calls": "tool_use",
    "tool_use": "tool_use",
    None: "end_turn",
    "": "end_turn",
}


def _new_msg_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def ollama_chat_to_anthropic(
    response: Dict[str, Any],
    *,
    anthropic_model: str,
) -> Dict[str, Any]:
    msg = response.get("message") or {}
    text = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []

    blocks: List[Dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    for i, tc in enumerate(tool_calls):
        fn = (tc.get("function") if isinstance(tc, dict) else None) or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        elif args is None:
            args = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"call_{i}",
                "name": fn.get("name", ""),
                "input": args,
            }
        )

    done_reason = response.get("done_reason")
    if not done_reason and tool_calls:
        done_reason = "tool_use"
    stop_reason = _DONE_REASON_TO_STOP.get(done_reason, "end_turn")

    return {
        "id": _new_msg_id(),
        "type": "message",
        "role": "assistant",
        "model": anthropic_model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(response.get("prompt_eval_count") or 0),
            "output_tokens": int(response.get("eval_count") or 0),
        },
    }


# ---------------------------------------------------------------------------
# Streaming: Ollama NDJSON -> Anthropic SSE
# ---------------------------------------------------------------------------


def _sse(event: str, data: Dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode(
        "utf-8"
    )


async def ollama_stream_to_anthropic_sse(
    lines: AsyncIterator[bytes],
    *,
    anthropic_model: str,
) -> AsyncIterator[bytes]:
    """Translate a stream of Ollama NDJSON lines into Anthropic SSE events."""
    async for event, data in ollama_stream_to_anthropic_events(
        lines, anthropic_model=anthropic_model
    ):
        yield _sse(event, data)


async def ollama_stream_to_anthropic_events(
    lines: AsyncIterator[bytes],
    *,
    anthropic_model: str,
) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
    """Translate a stream of Ollama NDJSON lines into Anthropic event tuples."""
    msg_id = _new_msg_id()

    # message_start
    yield (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": anthropic_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    text_block_open = False
    output_tokens = 0
    input_tokens = 0
    stop_reason = "end_turn"
    pending_tool_calls: List[Dict[str, Any]] = []

    async for raw in lines:
        try:
            chunk = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except json.JSONDecodeError:
            continue

        msg = chunk.get("message") or {}
        delta_text = msg.get("content") or ""
        tool_calls_in_chunk = msg.get("tool_calls") or []

        if delta_text:
            if not text_block_open:
                yield (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
                text_block_open = True
            yield (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": delta_text},
                },
            )

        if tool_calls_in_chunk:
            pending_tool_calls.extend(tool_calls_in_chunk)

        if chunk.get("done"):
            if text_block_open:
                yield (
                    "content_block_stop",
                    {"type": "content_block_stop", "index": 0},
                )
                text_block_open = False

            # Emit tool_use blocks (Ollama only delivers them whole, not
            # streamed token-by-token).
            base_idx = 1 if (msg.get("content") or "") else 0
            for i, tc in enumerate(pending_tool_calls):
                fn = (tc.get("function") if isinstance(tc, dict) else None) or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                elif args is None:
                    args = {}
                idx = base_idx + i
                tu_id = tc.get("id") or f"call_{i}"
                yield (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": tu_id,
                            "name": fn.get("name", ""),
                            "input": {},
                        },
                    },
                )
                yield (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(args, ensure_ascii=False),
                        },
                    },
                )
                yield (
                    "content_block_stop",
                    {"type": "content_block_stop", "index": idx},
                )

            done_reason = chunk.get("done_reason")
            if not done_reason and pending_tool_calls:
                done_reason = "tool_use"
            stop_reason = _DONE_REASON_TO_STOP.get(done_reason, "end_turn")
            output_tokens = int(chunk.get("eval_count") or output_tokens)
            input_tokens = int(chunk.get("prompt_eval_count") or input_tokens)

            yield (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                },
            )
            yield ("message_stop", {"type": "message_stop"})
            return

    # Stream ended without a `done` flag (defensive).
    if text_block_open:
        yield (
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )
    yield (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        },
    )
    yield ("message_stop", {"type": "message_stop"})


__all__ = [
    "anthropic_to_ollama_chat",
    "ollama_chat_to_anthropic",
    "ollama_stream_to_anthropic_events",
    "ollama_stream_to_anthropic_sse",
]
