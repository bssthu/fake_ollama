"""Exact, credential-scoped reasoning recovery. No text-only or tool-ID lookup.

Context variables belong to the request task (including its stream task). Only
irreversible digests and copies of upstream thinking blocks enter the cache.
"""
from __future__ import annotations

from collections import OrderedDict
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import hmac
import json
import re
import secrets
import threading
import time

from fastapi import HTTPException

_SECRET = secrets.token_bytes(32)
_LOCK = threading.RLock()
_EPOCH = 0
_MAX_ENTRIES = 512
_TTL_SECONDS = 900.0
_CACHE: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()


@dataclass(frozen=True)
class ThinkingScope:
    namespace: str
    epoch: int
    allow_unsigned: bool = False
    missing_policy: str = "omit"
    history: tuple = ()


_SCOPE: ContextVar[ThinkingScope | None] = ContextVar("thinking_scope", default=None)


def _digest(value) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hmac.new(_SECRET, raw, hashlib.sha256).hexdigest()


def bind_scope(*, principal: str, backend, model: str, configuration,
               allow_unsigned: bool = False, missing_policy: str = "omit"):
    """Bind only after authentication/routing; tokens are never stored in scope."""
    return _SCOPE.set(ThinkingScope(
        _digest([principal, backend, model, configuration]), _EPOCH,
        allow_unsigned, missing_policy,
    ))


def reset_scope(token) -> None:
    _SCOPE.reset(token)


def _canonical(message: dict) -> dict:
    msg = deepcopy(message)
    if msg.get("role") == "assistant":
        # These fields expose the same hidden thinking in different protocols.
        for name in ("thinking", "reasoning", "reasoning_content"):
            msg.pop(name, None)
        content = msg.get("content") or ""
        if isinstance(content, str) and "<think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        msg["content"] = content
        if msg.get("tool_calls"):
            calls = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        pass
                calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args})
            msg["tool_calls"] = calls
        else:
            msg.pop("tool_calls", None)
    return msg


def set_history(messages) -> None:
    scope = _SCOPE.get()
    if scope is not None:
        _SCOPE.set(replace(scope, history=tuple(_canonical(m) for m in messages)))


def _key(history, message) -> str | None:
    scope = _SCOPE.get()
    if scope is None or scope.epoch != _EPOCH or not history:
        return None
    return _digest([scope.namespace, scope.epoch, list(history), _canonical(message)])


def remember_thinking(blocks, *, tool_use_ids=None, text=None) -> None:
    scope = _SCOPE.get()
    if scope is None:
        return
    thinking = [deepcopy(b) for b in blocks if b.get("type") in ("thinking", "redacted_thinking")]
    if not thinking:
        return
    tools = [b for b in blocks if b.get("type") == "tool_use"]
    # Older stream callers lack complete tool content. Do not guess from IDs.
    if tool_use_ids and len(tools) != len(list(tool_use_ids)):
        return
    message = {"role": "assistant", "content": text or ""}
    if tools:
        message["tool_calls"] = [{"id": b.get("id"), "function": {
            "name": b.get("name"), "arguments": b.get("input") or {},
        }} for b in tools]
    key = _key(scope.history, message)
    if key is None:
        return
    with _LOCK:
        _prune()
        _CACHE[key] = (time.monotonic() + _TTL_SECONDS, thinking)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)


def _prune():
    now = time.monotonic()
    for key, (expires, _) in list(_CACHE.items()):
        if expires <= now:
            del _CACHE[key]


def recall_thinking(*, history=None, message=None, **_legacy) -> list[dict]:
    if history is None or message is None:
        return []
    key = _key([_canonical(m) for m in history], message)
    if key is None:
        return []
    with _LOCK:
        _prune()
        entry = _CACHE.get(key)
        if entry is None:
            return []
        _CACHE.move_to_end(key)
        return deepcopy(entry[1])


def invalidate() -> None:
    """Invalidate old entries AND in-flight writers at a configuration commit."""
    global _EPOCH
    with _LOCK:
        _EPOCH += 1
        _CACHE.clear()


def apply_policy(messages: list[dict]) -> None:
    scope = _SCOPE.get()
    allow_unsigned = scope.allow_unsigned if scope else False
    missing_policy = scope.missing_policy if scope else "omit"
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        blocks = msg.get("content")
        if isinstance(blocks, list):
            if not allow_unsigned:
                blocks = [b for b in blocks if b.get("type") != "thinking" or b.get("signature")]
                msg["content"] = blocks
            has_thinking = any(b.get("type") in ("thinking", "redacted_thinking") for b in blocks)
        else:
            has_thinking = False
        if not has_thinking and missing_policy == "error":
            raise HTTPException(400, "This backend requires recoverable thinking history; provide valid reasoning or restart the conversation.")
