"""Tool argument validation and ordered Ollama result association."""
from copy import deepcopy
import json
import uuid

from fastapi import HTTPException
import httpx


class UpstreamProtocolError(httpx.HTTPError):
    """A malformed upstream response, without including private payload data."""


def tool_arguments(value, *, upstream=True):
    try:
        result = json.loads(value) if isinstance(value, str) and value else value
    except (ValueError, TypeError):
        result = None
        if upstream:
            raise UpstreamProtocolError("Invalid or incomplete upstream tool arguments") from None
        raise HTTPException(400, "Invalid tool arguments JSON") from None
    if result is None and value in (None, ""):
        return {}
    if not isinstance(result, dict):
        if upstream:
            raise UpstreamProtocolError("Upstream tool arguments must be a JSON object")
        raise HTTPException(400, "Tool arguments must be a JSON object")
    return result


def validate_tool_blocks(blocks):
    """Non-stream Anthropic tools obey the same identity rules as SSE."""
    ids = set()
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        identifier, name = block.get("id"), block.get("name")
        if not isinstance(identifier, str) or not identifier or not isinstance(name, str) or not name:
            raise UpstreamProtocolError("Upstream tool call requires ID and name")
        if identifier in ids:
            raise UpstreamProtocolError("Duplicate upstream tool ID")
        ids.add(identifier)
        tool_arguments(block.get("input"))


def associate_ollama_results(messages):
    """Assign request-unique IDs; same-name calls are paired in protocol order."""
    result = deepcopy(list(messages))
    used = set()
    pending = []
    nonce = uuid.uuid4().hex
    count = 0
    for message in result:
        role = message.get("role", "user")
        if role == "tool":
            if not pending:
                raise HTTPException(400, "Tool result has no pending call")
            call = pending.pop(0)
            supplied = message.get("tool_call_id") or message.get("tool_use_id")
            if supplied and supplied != call["id"]:
                raise HTTPException(400, "Tool result ID does not match call order")
            if message.get("tool_name") and message["tool_name"] != call["function"]["name"]:
                raise HTTPException(400, "Tool result name does not match call order")
            message["tool_call_id"] = call["id"]
            continue
        if pending:
            raise HTTPException(400, "Missing tool results before next message")
        if role != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            identifier = call.get("id") or f"call_{nonce}_{count}"
            count += 1
            if identifier in used:
                raise HTTPException(400, "Duplicate tool call ID in request")
            used.add(identifier)
            call["id"] = identifier
            fn = call.get("function") or {}
            if not fn.get("name"):
                raise HTTPException(400, "Tool call requires a function name")
            fn["arguments"] = tool_arguments(fn.get("arguments"), upstream=False)
            pending.append(call)
    if pending:
        raise HTTPException(400, "Missing tool results at end of history")
    return result
