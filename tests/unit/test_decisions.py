"""Offline codec contracts; neither endpoint is an alias for the other."""
from copy import deepcopy

import pytest

from fake_ollama.decisions import (
    DecisionAnswer, DecisionResult, DecisionValidationError,
    NativeSystemOneCodec as SystemOne, OpenAIDecisionsCodec as OpenAI,
    request_has_images,
)
from fake_ollama.protocol_tools import UpstreamProtocolError


def native_body():
    return {"model": "public", "state": {"ticket": "broken"}, "questions": {
        "urgent": {"type": "noul", "instructions": {"ask": "Urgent?"}, "criteria": {"true": ["Immediate"]}},
        "team": {"type": "choice", "instructions": "Team?", "criteria": {"a": None, "b": {"role": "support"}}},
        "severity": {"type": "score", "instructions": "Severity?", "criteria": ["low", {"label": "high"}]}},
    }


def openai_body():
    return {"model": "public", "input": "broken", "questions": [
        {"type": "predicate", "instructions": "Urgent?", "name": "same"},
        {"type": "choice", "instructions": "Team?", "name": "same", "choices": [{"value": True}, {"value": "true"}, {"value": False}]},
        {"type": "score", "instructions": "Severity?", "levels": [{"label": "low"}, {"label": "high", "description": "Broken"}]},
    ]}


def test_distinct_request_shapes_and_native_round_trip():
    assert SystemOne.encode_request(SystemOne.parse_request(native_body())) == native_body()
    assert OpenAI.encode_request(OpenAI.parse_request(openai_body())) == openai_body()
    with pytest.raises(DecisionValidationError):
        SystemOne.parse_request(openai_body())
    with pytest.raises(DecisionValidationError):
        OpenAI.parse_request(native_body())


def test_typed_choice_and_duplicate_names_survive_systemone_adapter():
    request = OpenAI.parse_request(openai_body())
    wire = SystemOne.encode_request(request, "wire-model")
    assert list(wire["questions"]) == ["q0", "q1", "q2"]
    criteria = wire["questions"]["q1"]["criteria"]
    assert [v["value"] for v in criteria.values()] == [True, "true", False]
    response = {"model": "wire-model", "answers": {
        "q0": {"type": "noul", "noul": .6},
        "q1": {"type": "choice", "choice": "option_1", "confidence": .2,
               "probabilities": {"option_0": .1, "option_1": .7, "option_2": .2}},
        "q2": {"type": "score", "score": .4, "confidence": .05,
               "probabilities": {"0": .6, "1": .4}, "legend": {"0": "low", "1": "high"}}}}
    result = SystemOne.decode_response(response, request)
    public = OpenAI.encode_response(result, request)
    assert public["model"] == "public"
    assert [a["name"] for a in public["answers"]] == ["same", "same", None]
    assert public["answers"][1]["choice"] == "true"
    assert type(public["answers"][1]["choice"]) is str
    assert public["answers"][1]["probabilities"] == [
        {"value": True, "probability": .1}, {"value": "true", "probability": .7}, {"value": False, "probability": .2}]
    assert public["answers"][2]["score"] == .4
    assert public["answers"][2]["confidence"] == .05
    assert "usage" not in public


def test_native_response_legend_and_partial_usage_preserved():
    request = SystemOne.parse_request(native_body())
    response = {"model": "wire", "answers": {
        "urgent": {"type": "noul", "noul": .3},
        "team": {"type": "choice", "choice": "a", "confidence": .8, "probabilities": {"a": .9, "b": .1}},
        "severity": {"type": "score", "score": .4, "confidence": .05, "probabilities": {"0": .6, "1": .4}, "legend": {"0": "LOW", "1": "HIGH"}}},
        "usage": {"input_tokens": 17}}
    public = SystemOne.encode_response(SystemOne.decode_response(response, request), request)
    assert public["usage"] == {"input_tokens": 17}
    assert public["decision_metadata"]["usage_complete"] is False
    assert public["decision_metadata"]["usage_missing_fields"] == ["output_tokens"]
    assert public["answers"]["severity"]["legend"] == response["answers"]["severity"]["legend"]
    assert public["answers"]["team"]["confidence"] == .8
    assert public["answers"]["team"]["probabilities"] == {"a": .9, "b": .1}


def test_refusal_is_explicit_and_not_fabricated_across_dialects():
    request = SystemOne.parse_request(native_body())
    wire = OpenAI.encode_request(request)
    assert isinstance(wire["input"], str)
    response = {"model": "wire", "answers": [
        {"type": "predicate", "name": "urgent", "probability": .7},
        {"type": "refusal", "name": "team"},
        {"type": "refusal", "name": "severity"}]}
    result = OpenAI.decode_response(response, request)
    assert OpenAI.encode_response(result, request)["answers"][1] == {"name": "team", "type": "refusal"}
    with pytest.raises(DecisionValidationError, match="refusal cannot be represented"):
        SystemOne.encode_response(result, request)


def test_startlux_none_is_not_an_openai_refusal():
    request = SystemOne.parse_request({"model": "m", "state": "s", "allow_none": True,
        "questions": {"q": {"type": "choice", "instructions": "q", "criteria": {"a": None, "b": None}}}})
    result = SystemOne.decode_response({"answers": {"q": {"type": "choice", "choice": None,
        "probabilities": {"a": .5, "b": .5}, "confidence": 0, "none_probability": .9, "details": {"matched": False}}}}, request)
    public = SystemOne.encode_response(result, request)
    assert public["answers"]["q"]["choice"] is None
    assert public["answers"]["q"]["none_probability"] == .9
    with pytest.raises(DecisionValidationError, match="none-of-the-above"):
        OpenAI.encode_response(result, request)


@pytest.mark.parametrize("mutation", [
    lambda b: b.update(questions={}),
    lambda b: b["questions"]["urgent"].update(instructions=1),
    lambda b: b["questions"]["severity"].update(criteria=["one"]),
    lambda b: b["questions"]["team"].update(criteria=["a", "b"]),
    lambda b: b.update(state=float("nan")),
    lambda b: b.update(stream=True),
])
def test_systemone_invalid_requests(mutation):
    body = native_body()
    mutation(body)
    with pytest.raises(DecisionValidationError):
        SystemOne.parse_request(body)


@pytest.mark.parametrize("choices", [[{"value": True}, {"value": True}], [{"value": 1}, {"value": 2}], [{"value": "one"}]])
def test_openai_choice_validation(choices):
    body = openai_body()
    body["questions"][1]["choices"] = choices
    with pytest.raises(DecisionValidationError):
        OpenAI.parse_request(body)


def test_openai_nested_metadata_is_not_silently_lost():
    body = openai_body()
    body["questions"][1]["choices"][0]["future_extension"] = {"a": 1}
    body["questions"][2]["levels"][0]["future_extension"] = True
    body["prompt_cache_key"] = "partition"
    request = OpenAI.parse_request(body)
    assert OpenAI.encode_request(request) == body
    with pytest.raises(DecisionValidationError):
        SystemOne.encode_request(request)


def test_openai_input_messages_and_tool_items_preserved():
    body = openai_body()
    body["input"] = [
        {"role": "system", "content": [{"type": "input_text", "text": "Context", "prompt_cache_breakpoint": {"mode": "explicit"}}]},
        {"type": "function_call", "call_id": "c", "name": "weather", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c", "output": "sunny"},
        {"type": "custom_tool_call", "call_id": "d", "name": "lookup", "input": "text"},
        {"type": "custom_tool_call_output", "call_id": "d", "output": [{"type": "input_text", "text": "result"}]},
        {"role": "assistant", "phase": "final_answer", "content": [{"type": "refusal", "refusal": "Cannot"}]},
        {"role": "user", "content": [{"type": "input_image", "image_url": "https://example.test/a.png"}]},
    ]
    request = OpenAI.parse_request(body)
    assert request_has_images(request)
    assert OpenAI.encode_request(request) == body
    with pytest.raises(DecisionValidationError, match="Image conversion"):
        SystemOne.encode_request(request)


def test_typesafe_assets_and_startlux_images_are_detected():
    body = native_body()
    body["state"] = "image-ref"
    body["assets"] = {"image-ref": {"type": "image", "format": "png", "data_base64": "AA=="}}
    request = SystemOne.parse_request(body)
    assert request_has_images(request)
    assert SystemOne.encode_request(request) == body
    with pytest.raises(DecisionValidationError, match="Image conversion"):
        OpenAI.encode_request(request)
    body.pop("assets")
    body["images"] = ["AA=="]
    assert request_has_images(SystemOne.parse_request(body))


@pytest.mark.parametrize("answer", [
    {"type": "choice", "name": "same", "choice": "missing"},
    {"type": "choice", "name": "wrong", "choice": True},
    {"type": "choice", "name": "same", "choice": True, "confidence": float("inf")},
    {"type": "choice", "name": "same", "choice": True, "probabilities": [{"value": True, "probability": 1.0}]},
])
def test_invalid_upstream_answer_fails_closed(answer):
    request = OpenAI.parse_request(openai_body())
    with pytest.raises(UpstreamProtocolError):
        OpenAI.decode_response({"answers": [{"type": "predicate", "name": "same", "probability": .5}, answer,
            {"type": "refusal", "name": None}]}, request)


def test_argmax_is_not_accepted_as_score_expectation():
    request = OpenAI.parse_request(openai_body())
    with pytest.raises(UpstreamProtocolError, match="expectation"):
        OpenAI.decode_response({"answers": [{"type": "predicate", "name": "same", "probability": .5},
            {"type": "refusal", "name": "same"},
            {"type": "score", "name": None, "score": 0, "probabilities": [
                {"label": "low", "value": 0, "probability": .6}, {"label": "high", "value": 1, "probability": .4}]}]}, request)


def test_missing_metrics_are_not_created_for_neutral_results():
    request = OpenAI.parse_request(openai_body())
    result = DecisionResult("model", [DecisionAnswer("q0", "predicate", probability=.8),
        DecisionAnswer("q1", "choice", value=True), DecisionAnswer("q2", "score", value=.5)],
        source="test", confidence_semantics="unavailable")
    with pytest.raises(UpstreamProtocolError, match="missing required confidence or probabilities"):
        OpenAI.encode_response(result, request)
    with pytest.raises(UpstreamProtocolError, match="missing required confidence or probabilities"):
        SystemOne.encode_response(result, request)


async def test_repeated_cancellation_joins_owned_inference_cleanup():
    import asyncio
    from types import SimpleNamespace
    from fake_ollama.decision_handlers import _decide_until_disconnect

    started, cleaning, release, cleaned = [asyncio.Event() for _ in range(4)]
    class Request:
        async def receive(self):
            await asyncio.Event().wait()
    class Client:
        async def decide(self, *args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                cleaned.set()
    task = asyncio.create_task(_decide_until_disconnect(Request(), Client(), None,
        SimpleNamespace(estimated_vram_gb=None, estimated_memory_gb=None)))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    await asyncio.wait_for(cleaning.wait(), 2)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not cleaned.is_set()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert cleaned.is_set()
