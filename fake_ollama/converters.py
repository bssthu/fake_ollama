"""Conversion helpers between Ollama and Anthropic Messages API formats."""

from __future__ import annotations

import base64 as _b64
import hashlib
import json
import re
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Option mapping
# ---------------------------------------------------------------------------

# Ollama option name -> Anthropic field name
_OPTION_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "num_predict": "max_tokens",
    "max_tokens": "max_tokens",
    "stop": "stop_sequences",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _detect_image_media_type(b64_data: str) -> str:
    """Sniff the MIME type of a base64-encoded image from its magic bytes.

    Falls back to image/png on any failure. Supports PNG, JPEG, GIF, WEBP -
    the four formats Anthropic's vision API accepts.
    """
    try:
        sample = _b64.b64decode(b64_data[:64], validate=False)
    except Exception:
        return "image/png"
    if sample.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if sample.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if sample.startswith(b"GIF87a") or sample.startswith(b"GIF89a"):
        return "image/gif"
    if sample[:4] == b"RIFF" and sample[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


# ---------------------------------------------------------------------------
# Ollama -> Anthropic
# ---------------------------------------------------------------------------


_THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


# ---------------------------------------------------------------------------
# Thinking cache
# ---------------------------------------------------------------------------
#
# Some upstreams (notably DeepSeek's thinking models) require that any
# `thinking` blocks emitted in a previous assistant turn be echoed back in
# the next request, otherwise they hard-fail with:
#
#     "content[].thinking in the thinking mode must be passed back to the API"
#
# OpenAI / Ollama clients (Copilot, Open WebUI, ...) do not preserve our
# `thinking` blocks across turns -- at best they keep the visible text
# (sometimes inside a <think>...</think> wrapper, often stripped entirely)
# and they certainly do not preserve the upstream-provided cryptographic
# `signature` field. So we cache the full thinking blocks server-side keyed
# by stable identifiers we know the client WILL echo back: tool_use ids
# (round-tripped as tool_call.id) and a hash of the visible assistant text.

_THINKING_CACHE_MAX = 512
_THINKING_CACHE: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()


def _normalise_thinking_blocks(
    blocks: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep only the fields Anthropic-compatible upstreams care about."""
    out: List[Dict[str, Any]] = []
    for b in blocks or []:
        if not isinstance(b, dict) or b.get("type") != "thinking":
            continue
        text = b.get("thinking") or b.get("text") or ""
        if not text:
            continue
        item: Dict[str, Any] = {"type": "thinking", "thinking": text}
        sig = b.get("signature")
        if sig:
            item["signature"] = sig
        out.append(item)
    return out


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8", errors="replace")).hexdigest()[:32]


def remember_thinking(
    blocks: List[Dict[str, Any]],
    *,
    tool_use_ids: Optional[Iterable[str]] = None,
    text: Optional[str] = None,
) -> None:
    """Cache thinking blocks under any provided keys."""
    norm = _normalise_thinking_blocks(blocks)
    if not norm:
        return
    keys: List[str] = []
    for tid in tool_use_ids or []:
        if tid:
            keys.append(f"tu:{tid}")
    if text and text.strip():
        keys.append(f"tx:{_hash_text(text)}")
    for key in keys:
        _THINKING_CACHE[key] = list(norm)
        _THINKING_CACHE.move_to_end(key)
    while len(_THINKING_CACHE) > _THINKING_CACHE_MAX:
        _THINKING_CACHE.popitem(last=False)


def recall_thinking(
    *,
    tool_use_ids: Optional[Iterable[str]] = None,
    text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Look up cached thinking blocks; tool ids take precedence over text."""
    for tid in tool_use_ids or []:
        if not tid:
            continue
        cached = _THINKING_CACHE.get(f"tu:{tid}")
        if cached:
            _THINKING_CACHE.move_to_end(f"tu:{tid}")
            return [dict(b) for b in cached]
    if text and text.strip():
        key = f"tx:{_hash_text(text)}"
        cached = _THINKING_CACHE.get(key)
        if cached:
            _THINKING_CACHE.move_to_end(key)
            return [dict(b) for b in cached]
    return []


def _clear_thinking_cache() -> None:
    """Test helper."""
    _THINKING_CACHE.clear()


def _split_thinking(text: str) -> Tuple[List[str], str]:
    """Pull out <think>...</think> sections; return (thoughts, remaining_text)."""
    if not text or "<think>" not in text:
        return [], text
    thoughts = [m.group(1).strip() for m in _THINK_TAG_RE.finditer(text)]
    rest = _THINK_TAG_RE.sub("", text).strip()
    return [t for t in thoughts if t], rest


def _prepend_assistant_thinking(
    blocks: List[Dict[str, Any]],
    *,
    thinking_text: Optional[str] = None,
    inline_thoughts: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Insert leading thinking blocks into an assistant content list.

    Some upstream models (e.g. DeepSeek in thinking mode) require that any
    thinking content emitted by the assistant in a previous turn be echoed
    back in the next request, otherwise they reject with
    `content[].thinking ... must be passed back to the API`.
    """
    thoughts: List[str] = []
    if thinking_text:
        thoughts.append(thinking_text.strip())
    if inline_thoughts:
        thoughts.extend(t for t in inline_thoughts if t)
    if not thoughts:
        return blocks
    leading = [{"type": "thinking", "thinking": t} for t in thoughts if t]
    return leading + blocks


def _content_to_anthropic(message: Dict[str, Any]) -> Any:
    """Convert one Ollama message's content into Anthropic content blocks."""
    text = message.get("content", "") or ""
    images = message.get("images") or []
    if not images:
        return text
    blocks: List[Dict[str, Any]] = []
    for img in images:
        # Ollama sends raw base64 strings; sniff the actual format from the
        # decoded magic bytes so we don't mislabel JPEGs / GIFs / WEBPs as
        # PNG (which can cause vision-capable upstreams to reject them).
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _detect_image_media_type(img),
                    "data": img,
                },
            }
        )
    if text:
        blocks.append({"type": "text", "text": text})
    return blocks


def ollama_messages_to_anthropic(
    messages: Iterable[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Split Ollama messages into Anthropic (system, messages)."""
    system_parts: List[str] = []
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            text = msg.get("content", "")
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            tr_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id")
                or msg.get("tool_use_id")
                or "",
                "content": msg.get("content", ""),
            }
            if (
                converted
                and converted[-1].get("role") == "user"
                and isinstance(converted[-1].get("content"), list)
                and all(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in converted[-1]["content"]
                )
            ):
                converted[-1]["content"].append(tr_block)
            else:
                converted.append({"role": "user", "content": [tr_block]})
            continue
        if role == "assistant":
            text_raw = msg.get("content") or ""
            inline_thoughts, text_clean = _split_thinking(text_raw)
            reasoning_text = msg.get("thinking") or msg.get("reasoning_content")
            cached_thinking = recall_thinking(
                tool_use_ids=[tc.get("id") for tc in (msg.get("tool_calls") or [])],
                text=text_clean,
            )
        if role == "assistant" and msg.get("tool_calls"):
            blocks: List[Dict[str, Any]] = []
            if text_clean:
                blocks.append({"type": "text", "text": text_clean})
            for i, tc in enumerate(msg.get("tool_calls") or []):
                fn = (tc.get("function") if isinstance(tc, dict) else None) or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    import json as _json
                    try:
                        args = _json.loads(args)
                    except Exception:
                        args = {"_raw": args}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or f"call_{i}",
                        "name": fn.get("name", ""),
                        "input": args or {},
                    }
                )
            blocks = _prepend_assistant_thinking(
                blocks,
                thinking_text=reasoning_text,
                inline_thoughts=inline_thoughts,
            )
            if cached_thinking:
                blocks = list(cached_thinking) + [
                    b for b in blocks if not (isinstance(b, dict) and b.get("type") == "thinking")
                ]
            converted.append({"role": "assistant", "content": blocks})
            continue
        if role == "assistant" and (inline_thoughts or reasoning_text or cached_thinking):
            blocks = (
                [{"type": "text", "text": text_clean}] if text_clean else []
            )
            blocks = _prepend_assistant_thinking(
                blocks,
                thinking_text=reasoning_text,
                inline_thoughts=inline_thoughts,
            )
            if cached_thinking:
                blocks = list(cached_thinking) + [
                    b for b in blocks if not (isinstance(b, dict) and b.get("type") == "thinking")
                ]
            converted.append({"role": "assistant", "content": blocks})
            continue
        if role not in ("user", "assistant"):
            role = "user"
        converted.append({"role": role, "content": _content_to_anthropic(msg)})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, converted


def ollama_chat_to_anthropic(
    payload: Dict[str, Any],
    *,
    upstream_model: str,
    default_max_tokens: int,
) -> Dict[str, Any]:
    """Build an Anthropic /v1/messages request body from an Ollama /api/chat payload."""
    system, messages = ollama_messages_to_anthropic(payload.get("messages") or [])
    body: Dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "stream": bool(payload.get("stream", True)),
    }
    if system:
        body["system"] = system

    options = payload.get("options") or {}
    for src, dst in _OPTION_MAP.items():
        if src in options and options[src] is not None:
            body[dst] = options[src]

    body.setdefault("max_tokens", default_max_tokens)

    if payload.get("format") == "json":
        # Best-effort hint; Anthropic has no native json mode.
        sys_extra = "Respond with a single valid JSON document and nothing else."
        body["system"] = (body.get("system") + "\n\n" + sys_extra) if body.get("system") else sys_extra

    if "tools" in payload and payload["tools"]:
        # Ollama uses the OpenAI tool schema:
        # [{type:"function", function:{name, description, parameters}}]
        # Anthropic expects [{name, description, input_schema}].
        anth_tools: List[Dict[str, Any]] = []
        for t in payload["tools"]:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") if "function" in t else t
            if not isinstance(fn, dict) or not fn.get("name"):
                continue
            anth_tools.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters")
                    or fn.get("input_schema")
                    or {"type": "object"},
                }
            )
        if anth_tools:
            body["tools"] = anth_tools

    return body


def ollama_generate_to_anthropic(
    payload: Dict[str, Any],
    *,
    upstream_model: str,
    default_max_tokens: int,
) -> Dict[str, Any]:
    """Build an Anthropic request from an Ollama /api/generate payload."""
    chat_payload: Dict[str, Any] = {
        "stream": bool(payload.get("stream", True)),
        "options": payload.get("options"),
        "format": payload.get("format"),
        "messages": [],
    }
    if payload.get("system"):
        chat_payload["messages"].append({"role": "system", "content": payload["system"]})
    prompt = payload.get("prompt", "")
    images = payload.get("images") or []
    chat_payload["messages"].append(
        {"role": "user", "content": prompt, "images": images}
    )
    return ollama_chat_to_anthropic(
        chat_payload,
        upstream_model=upstream_model,
        default_max_tokens=default_max_tokens,
    )


# ---------------------------------------------------------------------------
# Anthropic -> Ollama (non-streaming)
# ---------------------------------------------------------------------------


def _extract_text(content_blocks: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for block in content_blocks or []:
        if isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return "".join(out)


def _extract_thinking(content_blocks: List[Dict[str, Any]]) -> str:
    """Concatenated text of any `thinking` blocks in the response."""
    out: List[str] = []
    for block in content_blocks or []:
        if isinstance(block, dict) and block.get("type") == "thinking":
            # Anthropic uses block["thinking"]; some forks use block["text"].
            out.append(block.get("thinking") or block.get("text") or "")
    return "".join(out)


def _wrap_thinking(thinking_text: str, body_text: str) -> str:
    """Combine reasoning + visible body using the <think>...</think> convention.

    This format is widely supported by Open WebUI and similar Ollama frontends
    for collapsible reasoning display.
    """
    if not thinking_text:
        return body_text
    return f"<think>{thinking_text}</think>\n\n{body_text}"


def _stop_reason_to_done(reason: Optional[str]) -> str:
    mapping = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "stop",
    }
    return mapping.get(reason or "", "stop")


def anthropic_to_ollama_chat(
    response: Dict[str, Any],
    *,
    ollama_model: str,
    show_thinking: bool = True,
) -> Dict[str, Any]:
    usage = response.get("usage") or {}
    blocks = response.get("content") or []
    text = _extract_text(blocks)
    thinking = _extract_thinking(blocks) if show_thinking else ""
    tool_calls: List[Dict[str, Any]] = []
    tool_use_ids: List[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            if block.get("id"):
                tool_use_ids.append(block["id"])
            tool_calls.append(
                {
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": block.get("input") or {},
                    }
                }
            )
    # Cache full thinking blocks (with signature) for next-turn echo back.
    remember_thinking(blocks, tool_use_ids=tool_use_ids, text=text)
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": _wrap_thinking(thinking, text),
    }
    # Also surface raw reasoning on the dedicated `thinking` field that the
    # current Ollama API uses; harmless for clients that don't read it.
    if thinking:
        message["thinking"] = thinking
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "model": ollama_model,
        "created_at": _now_iso(),
        "message": message,
        "done": True,
        "done_reason": _stop_reason_to_done(response.get("stop_reason")),
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": int(usage.get("input_tokens") or 0),
        "prompt_eval_duration": 0,
        "eval_count": int(usage.get("output_tokens") or 0),
        "eval_duration": 0,
    }


def anthropic_to_ollama_generate(
    response: Dict[str, Any],
    *,
    ollama_model: str,
    show_thinking: bool = True,
) -> Dict[str, Any]:
    chat = anthropic_to_ollama_chat(
        response, ollama_model=ollama_model, show_thinking=show_thinking
    )
    return {
        "model": ollama_model,
        "created_at": chat["created_at"],
        "response": chat["message"]["content"],
        "done": True,
        "done_reason": chat["done_reason"],
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": chat["prompt_eval_count"],
        "prompt_eval_duration": 0,
        "eval_count": chat["eval_count"],
        "eval_duration": 0,
        "context": [],
    }


# ---------------------------------------------------------------------------
# Anthropic SSE -> Ollama NDJSON (streaming)
# ---------------------------------------------------------------------------


class AnthropicStreamTranslator:
    """Translate the Anthropic SSE event stream into Ollama-style chunks.

    Feed lines from the SSE response via :meth:`feed_line`; each call returns
    a list of dicts ready to be JSON-serialised as Ollama chunks.
    """

    def __init__(self, ollama_model: str, *, mode: str = "chat", show_thinking: bool = True) -> None:
        if mode not in ("chat", "generate"):
            raise ValueError("mode must be 'chat' or 'generate'")
        self.model = ollama_model
        self.mode = mode
        self.show_thinking = show_thinking
        self._event: Optional[str] = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._stop_reason: Optional[str] = None
        # Tracks which content_block indices correspond to thinking blocks
        # and whether we've emitted the opening <think> tag yet.
        self._thinking_indices: set = set()
        self._think_open = False
        # Accumulators used to remember server-side thinking blocks so we
        # can echo them back to the upstream on the next turn.
        self._thinking_blocks: List[Dict[str, Any]] = []
        self._cur_thinking_text: List[str] = []
        self._cur_thinking_signature: Optional[str] = None
        self._tool_use_ids: List[str] = []
        self._final_text: List[str] = []

    # -- helpers ---------------------------------------------------------

    def _text_chunk(self, text: str, *, thinking: str = "") -> Dict[str, Any]:
        base = {
            "model": self.model,
            "created_at": _now_iso(),
            "done": False,
        }
        if self.mode == "chat":
            msg: Dict[str, Any] = {"role": "assistant", "content": text}
            if thinking:
                msg["thinking"] = thinking
            base["message"] = msg
        else:
            base["response"] = text
        return base

    def _close_think_chunk(self) -> Optional[Dict[str, Any]]:
        if self._think_open:
            self._think_open = False
            return self._text_chunk("</think>\n\n")
        return None

    def _final_chunk(self) -> Dict[str, Any]:
        base = {
            "model": self.model,
            "created_at": _now_iso(),
            "done": True,
            "done_reason": _stop_reason_to_done(self._stop_reason),
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": self._input_tokens,
            "prompt_eval_duration": 0,
            "eval_count": self._output_tokens,
            "eval_duration": 0,
        }
        if self.mode == "chat":
            base["message"] = {"role": "assistant", "content": ""}
        else:
            base["response"] = ""
            base["context"] = []
        return base

    # -- public API ------------------------------------------------------

    def feed_event(self, event_type: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a parsed SSE event and return zero or more Ollama chunks."""
        out: List[Dict[str, Any]] = []
        if event_type == "message_start":
            usage = (data.get("message") or {}).get("usage") or {}
            self._input_tokens = int(usage.get("input_tokens") or 0)
            self._output_tokens = int(usage.get("output_tokens") or 0)
        elif event_type == "content_block_start":
            block = data.get("content_block") or {}
            idx = int(data.get("index", 0))
            if block.get("type") == "thinking":
                self._thinking_indices.add(idx)
                self._cur_thinking_text = []
                self._cur_thinking_signature = block.get("signature")
            elif block.get("type") == "tool_use":
                tid = block.get("id")
                if tid:
                    self._tool_use_ids.append(tid)
                close = self._close_think_chunk()
                if close is not None:
                    out.append(close)
            else:
                # Any non-thinking block ends the thinking section.
                close = self._close_think_chunk()
                if close is not None:
                    out.append(close)
        elif event_type == "content_block_stop":
            idx = int(data.get("index", 0))
            if idx in self._thinking_indices:
                close = self._close_think_chunk()
                if close is not None:
                    out.append(close)
                # Persist the completed thinking block.
                text = "".join(self._cur_thinking_text)
                if text:
                    blk: Dict[str, Any] = {"type": "thinking", "thinking": text}
                    if self._cur_thinking_signature:
                        blk["signature"] = self._cur_thinking_signature
                    self._thinking_blocks.append(blk)
                self._cur_thinking_text = []
                self._cur_thinking_signature = None
        elif event_type == "content_block_delta":
            delta = data.get("delta") or {}
            dtype = delta.get("type")
            idx = int(data.get("index", 0))
            if dtype == "text_delta":
                # Make sure we close any open <think> tag first.
                close = self._close_think_chunk()
                if close is not None:
                    out.append(close)
                text = delta.get("text", "")
                if text:
                    self._final_text.append(text)
                    out.append(self._text_chunk(text))
            elif dtype == "thinking_delta":
                text = delta.get("thinking", "")
                if not text:
                    return out
                # Always accumulate so we can echo back to upstream.
                self._cur_thinking_text.append(text)
                if not self.show_thinking:
                    return out
                if not self._think_open:
                    self._think_open = True
                    out.append(self._text_chunk("<think>" + text, thinking=text))
                else:
                    out.append(self._text_chunk(text, thinking=text))
            elif dtype == "signature_delta":
                # Some Anthropic models send the cryptographic signature
                # incrementally; capture it for thinking-block round-trip.
                sig = delta.get("signature")
                if sig:
                    self._cur_thinking_signature = sig
            # input_json_delta etc. are ignored here (handled by OpenAI translator).
        elif event_type == "message_delta":
            delta = data.get("delta") or {}
            if "stop_reason" in delta:
                self._stop_reason = delta.get("stop_reason")
            usage = data.get("usage") or {}
            if "output_tokens" in usage:
                self._output_tokens = int(usage["output_tokens"])
        elif event_type == "message_stop":
            close = self._close_think_chunk()
            if close is not None:
                out.append(close)
            # Cache thinking blocks so they can be echoed back next turn.
            if self._thinking_blocks:
                remember_thinking(
                    self._thinking_blocks,
                    tool_use_ids=self._tool_use_ids,
                    text="".join(self._final_text),
                )
            out.append(self._final_chunk())
        elif event_type == "error":
            err = data.get("error") or {}
            msg = err.get("message") or "upstream error"
            out.append(
                {
                    "model": self.model,
                    "created_at": _now_iso(),
                    "done": True,
                    "error": msg,
                }
            )
        return out


# ---------------------------------------------------------------------------
# OpenAI Chat Completions <-> Anthropic
# ---------------------------------------------------------------------------


# OpenAI sampling option name -> Anthropic field name
_OPENAI_OPTION_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "max_tokens",
    "max_completion_tokens": "max_tokens",
    "stop": "stop_sequences",
}


def _openai_message_to_anthropic_content(msg: Dict[str, Any]) -> Any:
    """Convert one OpenAI message's `content` into Anthropic content blocks."""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # OpenAI multi-part content: list of {type: text|image_url, ...}
    blocks: List[Dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "image_url":
            url = (part.get("image_url") or {}).get("url", "")
            if url.startswith("data:"):
                # data:image/png;base64,XXXX
                try:
                    header, b64 = url.split(",", 1)
                    media_type = header.split(";", 1)[0][len("data:"):] or "image/png"
                except ValueError:
                    media_type, b64 = "image/png", ""
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    }
                )
            elif url:
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    }
                )
    return blocks or ""


def openai_chat_to_anthropic(
    payload: Dict[str, Any],
    *,
    upstream_model: str,
    default_max_tokens: int,
) -> Dict[str, Any]:
    """Build an Anthropic /v1/messages request from an OpenAI chat completion."""
    system_parts: List[str] = []
    messages: List[Dict[str, Any]] = []
    for msg in payload.get("messages") or []:
        role = msg.get("role", "user")
        if role == "system" or role == "developer":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if content:
                system_parts.append(content)
            continue
        if role == "tool":
            # Map OpenAI tool result back to Anthropic tool_result block.
            # Anthropic requires ALL tool_results for a given assistant turn
            # to be packed into a single user message; if the previous entry
            # is already such a message, append to it instead of creating a
            # new one.
            tr_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            if (
                messages
                and messages[-1].get("role") == "user"
                and isinstance(messages[-1].get("content"), list)
                and all(
                    isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in messages[-1]["content"]
                )
            ):
                messages[-1]["content"].append(tr_block)
            else:
                messages.append({"role": "user", "content": [tr_block]})
            continue
        if role not in ("user", "assistant"):
            role = "user"
        anth_content = _openai_message_to_anthropic_content(msg)
        if role == "assistant":
            # Extract any <think>...</think> markers from text content and
            # the OpenAI-style `reasoning_content` field, so they can be
            # echoed back to upstreams that require thinking blocks to be
            # preserved across turns (e.g. DeepSeek's thinking mode).
            inline_thoughts: List[str] = []
            if isinstance(anth_content, str):
                thoughts, anth_content = _split_thinking(anth_content)
                inline_thoughts.extend(thoughts)
            elif isinstance(anth_content, list):
                stripped: List[Dict[str, Any]] = []
                for blk in anth_content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        ts, rest = _split_thinking(blk.get("text", ""))
                        inline_thoughts.extend(ts)
                        if rest:
                            stripped.append({"type": "text", "text": rest})
                    else:
                        stripped.append(blk)
                anth_content = stripped
            reasoning_text = msg.get("reasoning_content") or msg.get("reasoning")
            # Look up any cached thinking blocks (by tool_call ids first,
            # then by text hash). These blocks carry the upstream-issued
            # `signature` field that DeepSeek requires.
            assistant_text = ""
            if isinstance(anth_content, str):
                assistant_text = anth_content
            elif isinstance(anth_content, list):
                assistant_text = "".join(
                    b.get("text", "") for b in anth_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            cached_thinking = recall_thinking(
                tool_use_ids=[tc.get("id") for tc in (msg.get("tool_calls") or [])],
                text=assistant_text,
            )
        if role == "assistant" and msg.get("tool_calls"):
            # Preserve tool_use blocks so subsequent tool_result messages
            # have a matching tool_use_id in the previous assistant turn
            # (Anthropic strictly enforces this).
            blocks: List[Dict[str, Any]] = []
            if isinstance(anth_content, str):
                if anth_content:
                    blocks.append({"type": "text", "text": anth_content})
            elif isinstance(anth_content, list):
                blocks.extend(anth_content)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                elif args is None:
                    args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )
            blocks = _prepend_assistant_thinking(
                blocks,
                thinking_text=reasoning_text,
                inline_thoughts=inline_thoughts,
            )
            if cached_thinking:
                blocks = list(cached_thinking) + [
                    b for b in blocks if not (isinstance(b, dict) and b.get("type") == "thinking")
                ]
            messages.append({"role": role, "content": blocks})
            continue
        if role == "assistant" and (inline_thoughts or reasoning_text or cached_thinking):
            # Wrap text-only assistant content into a block list so we can
            # prepend the thinking blocks.
            if isinstance(anth_content, str):
                blocks = (
                    [{"type": "text", "text": anth_content}] if anth_content else []
                )
            elif isinstance(anth_content, list):
                blocks = list(anth_content)
            else:
                blocks = []
            blocks = _prepend_assistant_thinking(
                blocks,
                thinking_text=reasoning_text,
                inline_thoughts=inline_thoughts,
            )
            if cached_thinking:
                blocks = list(cached_thinking) + [
                    b for b in blocks if not (isinstance(b, dict) and b.get("type") == "thinking")
                ]
            messages.append({"role": role, "content": blocks})
            continue
        messages.append({"role": role, "content": anth_content})

    body: Dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "stream": bool(payload.get("stream", False)),
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)

    for src, dst in _OPENAI_OPTION_MAP.items():
        if src in payload and payload[src] is not None:
            body[dst] = payload[src]
    body.setdefault("max_tokens", default_max_tokens)

    if payload.get("tools"):
        # OpenAI tools: [{type:"function", function:{name,description,parameters}}]
        anth_tools: List[Dict[str, Any]] = []
        for t in payload["tools"]:
            fn = t.get("function") if isinstance(t, dict) else None
            if not fn:
                continue
            anth_tools.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object"},
                }
            )
        if anth_tools:
            body["tools"] = anth_tools

    return body


def anthropic_to_openai_chat(
    response: Dict[str, Any],
    *,
    openai_model: str,
    show_thinking: bool = True,
) -> Dict[str, Any]:
    usage = response.get("usage") or {}
    blocks = response.get("content") or []
    text = _extract_text(blocks)
    thinking = _extract_thinking(blocks) if show_thinking else ""
    finish = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(response.get("stop_reason") or "", "stop")

    visible = _wrap_thinking(thinking, text) if thinking else text
    message: Dict[str, Any] = {"role": "assistant", "content": visible or None}
    # Also surface raw reasoning on `reasoning_content` (DeepSeek / OpenAI
    # o-series convention); harmless for clients that don't read it.
    if thinking:
        message["reasoning_content"] = thinking
    tool_calls: List[Dict[str, Any]] = []
    tool_use_ids: List[str] = []
    for i, block in enumerate(blocks):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            import json as _json
            tid = block.get("id") or f"call_{i}"
            tool_use_ids.append(tid)
            tool_calls.append(
                {
                    "id": tid,
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": _json.dumps(block.get("input") or {}),
                    },
                }
            )
    if tool_calls:
        message["tool_calls"] = tool_calls
    # Cache full thinking blocks (with signature) for next-turn echo back.
    remember_thinking(blocks, tool_use_ids=tool_use_ids, text=text)

    return {
        "id": response.get("id") or "chatcmpl-fake",
        "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()),
        "model": openai_model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish}
        ],
        "usage": {
            "prompt_tokens": int(usage.get("input_tokens") or 0),
            "completion_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("input_tokens") or 0)
            + int(usage.get("output_tokens") or 0),
        },
    }


class OpenAIChatStreamTranslator:
    """Translate Anthropic SSE events into OpenAI chat.completion.chunk frames."""

    def __init__(self, openai_model: str, *, show_thinking: bool = True) -> None:
        self.model = openai_model
        self.show_thinking = show_thinking
        self._created = int(datetime.now(timezone.utc).timestamp())
        self._id = "chatcmpl-fake"
        self._stop_reason: Optional[str] = None
        self._sent_role = False
        # Track tool-use blocks by their Anthropic content_block index so we
        # can stream OpenAI tool_calls deltas with a stable per-call index.
        # { anthropic_block_index: openai_tool_call_index }
        self._tool_blocks: Dict[int, int] = {}
        self._next_tool_index = 0
        # Thinking-block tracking: surface as both <think>...</think>-wrapped
        # `content` and as `reasoning_content` deltas.
        self._thinking_indices: set = set()
        self._think_open = False
        # Server-side thinking accumulators (echoed back to upstream next turn).
        self._thinking_blocks: List[Dict[str, Any]] = []
        self._cur_thinking_text: List[str] = []
        self._cur_thinking_signature: Optional[str] = None
        self._tool_use_ids: List[str] = []
        self._final_text: List[str] = []

    def _frame(self, delta: Dict[str, Any], finish: Optional[str] = None) -> Dict[str, Any]:
        return {
            "id": self._id,
            "object": "chat.completion.chunk",
            "created": self._created,
            "model": self.model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish}
            ],
        }

    def _close_think_frame(self) -> Optional[Dict[str, Any]]:
        if self._think_open:
            self._think_open = False
            return self._frame({"content": "</think>\n\n"})
        return None

    def feed_event(self, event_type: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if event_type == "message_start":
            msg = data.get("message") or {}
            self._id = msg.get("id") or self._id
            out.append(self._frame({"role": "assistant", "content": ""}))
            self._sent_role = True
        elif event_type == "content_block_start":
            block = data.get("content_block") or {}
            idx = int(data.get("index", 0))
            if block.get("type") == "tool_use":
                tool_idx = self._next_tool_index
                self._next_tool_index += 1
                self._tool_blocks[idx] = tool_idx
                tid = block.get("id") or f"call_{tool_idx}"
                self._tool_use_ids.append(tid)
                import json as _json
                # Emit the opening tool_call frame with id/name; arguments will
                # stream in via subsequent input_json_delta events. Some tools
                # have no inputs at all (input={}) and never emit a delta, so
                # pre-seed arguments with the serialised initial input.
                initial_args = _json.dumps(block.get("input") or {}) if block.get("input") else ""
                out.append(
                    self._frame(
                        {
                            "tool_calls": [
                                {
                                    "index": tool_idx,
                                    "id": tid,
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": initial_args,
                                    },
                                }
                            ]
                        }
                    )
                )
            elif block.get("type") == "thinking":
                self._thinking_indices.add(idx)
                self._cur_thinking_text = []
                self._cur_thinking_signature = block.get("signature")
            else:
                # Plain text block opens; close any dangling <think> tag.
                close = self._close_think_frame()
                if close is not None:
                    out.append(close)
        elif event_type == "content_block_stop":
            idx = int(data.get("index", 0))
            if idx in self._thinking_indices:
                close = self._close_think_frame()
                if close is not None:
                    out.append(close)
                text = "".join(self._cur_thinking_text)
                if text:
                    blk: Dict[str, Any] = {"type": "thinking", "thinking": text}
                    if self._cur_thinking_signature:
                        blk["signature"] = self._cur_thinking_signature
                    self._thinking_blocks.append(blk)
                self._cur_thinking_text = []
                self._cur_thinking_signature = None
        elif event_type == "content_block_delta":
            delta = data.get("delta") or {}
            dtype = delta.get("type")
            if dtype == "text_delta":
                close = self._close_think_frame()
                if close is not None:
                    out.append(close)
                text = delta.get("text", "")
                if text:
                    self._final_text.append(text)
                    out.append(self._frame({"content": text}))
            elif dtype == "thinking_delta":
                text = delta.get("thinking", "")
                if not text:
                    return out
                self._cur_thinking_text.append(text)
                if not self.show_thinking:
                    return out
                if not self._think_open:
                    self._think_open = True
                    out.append(
                        self._frame({"content": "<think>" + text, "reasoning_content": text})
                    )
                else:
                    out.append(self._frame({"content": text, "reasoning_content": text}))
            elif dtype == "signature_delta":
                sig = delta.get("signature")
                if sig:
                    self._cur_thinking_signature = sig
            elif dtype == "input_json_delta":
                idx = int(data.get("index", 0))
                tool_idx = self._tool_blocks.get(idx)
                if tool_idx is not None:
                    partial = delta.get("partial_json", "")
                    if partial:
                        out.append(
                            self._frame(
                                {
                                    "tool_calls": [
                                        {
                                            "index": tool_idx,
                                            "function": {"arguments": partial},
                                        }
                                    ]
                                }
                            )
                        )
        elif event_type == "message_delta":
            d = data.get("delta") or {}
            if "stop_reason" in d:
                self._stop_reason = d.get("stop_reason")
        elif event_type == "message_stop":
            close = self._close_think_frame()
            if close is not None:
                out.append(close)
            # If we streamed any tool_use blocks, finish_reason must be
            # "tool_calls" so OpenAI clients (Copilot) actually invoke them.
            if self._tool_blocks and self._stop_reason in (None, "end_turn"):
                self._stop_reason = "tool_use"
            finish = {
                "end_turn": "stop",
                "stop_sequence": "stop",
                "max_tokens": "length",
                "tool_use": "tool_calls",
            }.get(self._stop_reason or "", "stop")
            if self._thinking_blocks:
                remember_thinking(
                    self._thinking_blocks,
                    tool_use_ids=self._tool_use_ids,
                    text="".join(self._final_text),
                )
            out.append(self._frame({}, finish=finish))
        elif event_type == "error":
            err = data.get("error") or {}
            out.append(
                self._frame({"content": f"[upstream error: {err.get('message','')}]"}, finish="stop")
            )
        return out
