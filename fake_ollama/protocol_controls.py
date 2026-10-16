"""Translate caller constraints explicitly; reject controls we cannot express."""
from fastapi import HTTPException


def include_usage_requested(payload):
    options = payload.get("stream_options")
    if options is None:
        return False
    if not payload.get("stream") or not isinstance(options, dict):
        raise HTTPException(400, "stream_options requires a streaming request and an object")
    if set(options) - {"include_usage"}:
        raise HTTPException(400, "Unsupported stream_options for this protocol adapter")
    value = options.get("include_usage", False)
    if not isinstance(value, bool):
        raise HTTPException(400, "stream_options.include_usage must be boolean")
    return value


def openai_tool_controls(payload, body):
    include_usage_requested(payload)
    tools = payload.get("tools") or []
    names = set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise HTTPException(400, "This adapter supports function tools only")
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise HTTPException(400, "Function tools require a name")
        names.add(function["name"])
    parallel = payload.get("parallel_tool_calls")
    if parallel is not None and not isinstance(parallel, bool):
        raise HTTPException(400, "parallel_tool_calls must be boolean")
    choice = payload.get("tool_choice")
    if choice is None:
        if parallel is None:
            return
        choice = "auto" if names else "none"
    if isinstance(choice, str) and choice in ("none", "auto", "required"):
        result = {"type": {"required": "any"}.get(choice, choice)}
    elif (isinstance(choice, dict) and set(choice) == {"type", "function"} and choice.get("type") == "function"
          and isinstance(choice.get("function"), dict) and set(choice["function"]) == {"name"}):
        name = choice["function"]["name"]
        if not isinstance(name, str) or name not in names:
            raise HTTPException(400, "tool_choice must name a declared function")
        result = {"type": "tool", "name": name}
    else:
        raise HTTPException(400, "Unsupported tool_choice for Anthropic adapter")
    if result["type"] in ("any", "tool") and not names:
        raise HTTPException(400, "Forced tool use requires declared tools")
    # none already guarantees zero calls; Anthropic's none variant has no
    # disable_parallel_tool_use field, and either parallel policy is satisfied.
    if parallel is not None and result["type"] != "none":
        result["disable_parallel_tool_use"] = not parallel
    body["tool_choice"] = result


def anthropic_tool_controls(payload, body, *, ollama=False):
    value = payload.get("tool_choice")
    if value is None:
        return
    if not isinstance(value, dict):
        raise HTTPException(400, "Anthropic tool_choice must be an object")
    kind = value.get("type")
    if kind not in ("none", "auto", "any", "tool") or set(value) - {"type", "name", "disable_parallel_tool_use"}:
        raise HTTPException(400, "Unsupported Anthropic tool_choice")
    disabled = value.get("disable_parallel_tool_use")
    if disabled is not None and not isinstance(disabled, bool):
        raise HTTPException(400, "disable_parallel_tool_use must be boolean")
    names = {t.get("name") for t in payload.get("tools") or [] if isinstance(t, dict)}
    if kind in ("any", "tool") and not names:
        raise HTTPException(400, "Forced tool use requires declared tools")
    if kind == "tool" and (not isinstance(value.get("name"), str) or value["name"] not in names):
        raise HTTPException(400, "tool_choice must name a declared tool")
    if ollama:
        if kind == "none":
            body.pop("tools", None)
        elif kind != "auto" or disabled is True:
            raise HTTPException(400, "Ollama native chat cannot express forced or single-call tool constraints")
        return
    body["tool_choice"] = ({"type": "function", "function": {"name": value["name"]}}
                           if kind == "tool" else {"any": "required"}.get(kind, kind))
    if disabled is not None:
        body["parallel_tool_calls"] = not disabled
