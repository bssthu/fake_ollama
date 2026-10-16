"""Shared validation for translated Anthropic streams and terminal outcomes."""
from copy import deepcopy
import json

from .protocol_tools import UpstreamProtocolError, tool_arguments


class AnthropicStreamState:
    def __init__(self):
        self.blocks = {}
        self.open_blocks = set()
        self.fragments = {}
        self.tool_indices = {}
        self.terminal = False
        self.failed = False

    def consume(self, event, data):
        if event == "error":
            self.terminal = self.failed = True
        elif event == "content_block_start":
            index = data.get("index", 0)
            if index in self.blocks:
                raise UpstreamProtocolError("Duplicate upstream content block index")
            block = deepcopy(data.get("content_block") or {})
            self.blocks[index] = block
            self.open_blocks.add(index)
            if block.get("type") == "tool_use":
                if not block.get("id") or not block.get("name"):
                    raise UpstreamProtocolError("Upstream tool call requires ID and name")
                if any(b is not block and b.get("id") == block["id"] for b in self.blocks.values()):
                    raise UpstreamProtocolError("Duplicate upstream tool ID")
                self.tool_indices[index] = len(self.tool_indices)
                self.fragments[index] = []
        elif event == "content_block_delta":
            index = data.get("index", 0)
            delta = data.get("delta") or {}
            block = self.blocks.get(index)
            # Some compatible upstreams omit text start events. Never allow
            # that relaxation for tool JSON: its name/ID must be established.
            if delta.get("type") == "input_json_delta":
                if index not in self.open_blocks or index not in self.fragments:
                    raise UpstreamProtocolError("Tool arguments without an open tool block")
                self.fragments[index].append(delta.get("partial_json", ""))
            elif block is not None:
                for kind, field in (("text_delta", "text"), ("thinking_delta", "thinking"), ("signature_delta", "signature")):
                    if delta.get("type") == kind:
                        block[field] = block.get(field, "") + delta.get(field, "")
        elif event == "content_block_stop":
            index = data.get("index", 0)
            if index not in self.open_blocks:
                raise UpstreamProtocolError("Upstream closed an unknown content block")
            if index in self.fragments:
                block = self.blocks[index]
                initial = tool_arguments(block.get("input"))
                partial = "".join(self.fragments[index])
                parsed = tool_arguments(partial) if partial else {}
                if any(key in initial and initial[key] != value for key, value in parsed.items()):
                    raise UpstreamProtocolError("Conflicting upstream tool arguments")
                block["input"] = {**initial, **parsed}
            self.open_blocks.remove(index)
        elif event == "message_stop":
            if self.open_blocks:
                raise UpstreamProtocolError("Upstream ended with incomplete content blocks")
            self.terminal = True

    @property
    def content(self):
        return list(self.blocks.values())


async def checked_openai_lines(lines):
    """Passthrough framing still requires the upstream terminal sentinel."""
    finished = False
    async for raw in lines:
        text = raw.decode() if isinstance(raw, bytes) else raw
        payload = text.strip()
        if payload.startswith("data:"):
            payload = payload[5:].strip()
        elif payload.startswith((":", "event:")) or not payload:
            yield text
            continue
        if payload == "[DONE]":
            if not finished:
                raise UpstreamProtocolError("Upstream stream terminated without finish reason")
            yield text
            return
        try:
            chunk = json.loads(payload)
        except ValueError:
            raise UpstreamProtocolError("Invalid upstream stream JSON") from None
        if chunk.get("error"):
            yield text
            return
        finished = finished or any(c.get("finish_reason") for c in chunk.get("choices", []))
        yield text
    raise UpstreamProtocolError("Upstream stream ended before completion")
