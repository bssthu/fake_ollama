"""Provider-neutral decisions and the two deliberately distinct public codecs.

Wire contracts: https://docs.typesafe.ai/api and
https://developers.openai.com/api/reference/resources/decisions/methods/create
No adapter synthesizes confidence, distributions, refusals, or token usage.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from typing import Any

from .protocol_tools import UpstreamProtocolError


class DecisionValidationError(ValueError):
    """Invalid public request or an unsupported lossless dialect conversion."""


@dataclass
class DecisionOption:
    value: Any
    description: Any = None
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionQuestion:
    id: str
    type: str
    instructions: Any
    choices: list[DecisionOption] = field(default_factory=list)
    levels: list[DecisionOption] = field(default_factory=list)
    name: str | None = None
    criteria: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionRequest:
    model: str
    input: Any
    questions: list[DecisionQuestion]
    dialect: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionAnswer:
    id: str
    type: str
    value: Any = None
    probability: float | None = None
    confidence: float | None = None
    probabilities: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionResult:
    model: str
    answers: list[DecisionAnswer]
    usage: dict[str, Any] | None = None
    source: str = "native"
    confidence_semantics: str = "provider_reported"
    metadata: dict[str, Any] = field(default_factory=dict)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecisionValidationError(message)


def _finite_json(value: Any, depth: int = 0, *, max_depth: int | None = 64) -> None:
    _require(max_depth is None or depth <= max_depth, "JSON nesting exceeds 64 levels")
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        _require(finite, "JSON numbers must be finite")
    elif isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str), "JSON object keys must be strings")
            _finite_json(item, depth + 1, max_depth=max_depth)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item, depth + 1, max_depth=max_depth)
    else:
        _require(value is None or isinstance(value, (str, int, bool)), "Value is not JSON")


def _content(value: Any, label: str, *, nullable: bool = False) -> None:
    _require(isinstance(value, (str, dict, list)) or (nullable and value is None),
             f"{label} must be a string, object, or array")
    _finite_json(value)


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _typed_key(value: Any) -> tuple[type, Any]:
    return type(value), value


def _base_request(payload: Any) -> None:
    _require(isinstance(payload, dict), "Request body must be a JSON object")
    _finite_json(payload, max_depth=None)
    _require(isinstance(payload.get("model"), str) and bool(payload["model"].strip()), "model must be a nonempty string")
    _require(payload.get("stream") in (None, False), "Decision streaming is not supported")


def request_has_images(request: DecisionRequest) -> bool:
    if request.options.get("assets") or request.options.get("images"):
        return True
    def image(value):
        if isinstance(value, dict):
            return value.get("type") == "input_image" or any(image(v) for v in value.values())
        return isinstance(value, list) and any(image(v) for v in value)
    return request.dialect == "openai_decisions" and image(request.input)


def _response_number(value: Any, label: str, *, probability: bool = False) -> float:
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise UpstreamProtocolError(f"Invalid upstream decision {label}")
    if probability and not 0 <= value <= 1:
        raise UpstreamProtocolError(f"Upstream decision {label} is outside [0, 1]")
    return value


def _optional_confidence(answer: dict) -> float | None:
    value = answer.get("confidence")
    return None if value is None else _response_number(value, "confidence", probability=True)


def _usage(payload: dict) -> dict | None:
    value = payload.get("usage")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise UpstreamProtocolError("Invalid upstream decision usage")
    def check(item):
        if isinstance(item, dict):
            for child in item.values():
                check(child)
        elif item is not None and (isinstance(item, bool) or not isinstance(item, int) or item < 0):
            raise UpstreamProtocolError("Invalid upstream decision token count")
    check(value)
    def known(item):
        return {key: known(child) if isinstance(child, dict) else child
                for key, child in item.items() if child is not None}
    return known(value) or None


def _distribution(entries: list[dict], options: list[DecisionOption], *, mass: float = 1.0) -> list[dict]:
    expected = {_typed_key(option.value) for option in options}
    actual = []
    for entry in entries:
        if not isinstance(entry, dict) or "value" not in entry:
            raise UpstreamProtocolError("Invalid upstream decision distribution")
        try:
            key = _typed_key(entry["value"])
            if key not in expected:
                raise UpstreamProtocolError("Unknown upstream decision distribution value")
            actual.append(key)
        except TypeError:
            raise UpstreamProtocolError("Invalid upstream decision distribution value") from None
        _response_number(entry.get("probability"), "probability", probability=True)
    if len(actual) != len(expected) or set(actual) != expected:
        raise UpstreamProtocolError("Incomplete or duplicate upstream decision distribution")
    if not math.isclose(sum(item["probability"] for item in entries), mass, abs_tol=0.001):
        raise UpstreamProtocolError("Upstream decision probabilities do not sum to one")
    # Preserve provider ordering and values, including typed bool/string choices.
    return deepcopy(entries)


def _check_score(value: Any, probabilities: list[dict] | None, count: int) -> float:
    score = _response_number(value, "score")
    if not 0 <= score <= count - 1:
        raise UpstreamProtocolError("Upstream decision score is outside the level range")
    if probabilities is not None:
        expected = sum(item["value"] * item["probability"] for item in probabilities)
        # Providers may round displayed score/distribution values independently.
        if not math.isclose(score, expected, abs_tol=0.02):
            raise UpstreamProtocolError("Upstream decision score is not the level expectation")
    return score


def _result(payload: Any, request: DecisionRequest, answers: list[DecisionAnswer], source: str) -> DecisionResult:
    model = payload.get("model", request.model)
    if not isinstance(model, str):
        raise UpstreamProtocolError("Invalid upstream decision model")
    return DecisionResult(model=model, answers=answers, usage=_usage(payload), source=source,
                          metadata={k: deepcopy(v) for k, v in payload.items() if k not in {"model", "answers", "usage", "decision_metadata"}})


def _ordered_answers(result: DecisionResult, request: DecisionRequest) -> list[DecisionAnswer]:
    if len(result.answers) != len(request.questions) or len({a.id for a in result.answers}) != len(result.answers):
        raise UpstreamProtocolError("Upstream decision answers do not match request")
    by_id = {a.id: a for a in result.answers}
    try:
        answers = [by_id[q.id] for q in request.questions]
    except KeyError:
        raise UpstreamProtocolError("Upstream decision answer ID does not match request") from None
    for question, answer in zip(request.questions, answers):
        if answer.type not in {question.type, "refusal"}:
            raise UpstreamProtocolError("Upstream decision answer type does not match request")
        if answer.type == "predicate":
            _response_number(answer.probability, "predicate probability", probability=True)
        elif answer.type in {"choice", "score"}:
            if answer.confidence is None or answer.probabilities is None:
                raise UpstreamProtocolError("Decision response is missing required confidence or probabilities; these values cannot be synthesized")
            _response_number(answer.confidence, "confidence", probability=True)
            _distribution(answer.probabilities, question.choices if answer.type == "choice" else question.levels)
    return answers


def _envelope(result: DecisionResult, request: DecisionRequest, answers: Any) -> dict:
    value = {"model": request.model, "answers": answers,
             "decision_metadata": {**deepcopy(result.metadata), "source": result.source,
                                   "confidence_semantics": result.confidence_semantics}}
    usage = _usage({"usage": result.usage})
    required = ["input_tokens", "output_tokens"]
    if request.dialect == "openai_decisions":
        required += ["total_tokens", "input_tokens_details.cached_tokens", "input_tokens_details.cache_write_tokens",
                     "output_tokens_details.reasoning_tokens"]
    def available(path):
        item = usage
        for key in path.split("."):
            if not isinstance(item, dict) or key not in item:
                return False
            item = item[key]
        return item is not None
    missing = [path for path in required if not available(path)]
    value["decision_metadata"]["usage_complete"] = not missing
    if missing:
        value["decision_metadata"]["usage_missing_fields"] = missing
    if usage is not None:
        value["usage"] = usage
    return value


def _native_choice_keys(question: DecisionQuestion) -> list[str]:
    # Opaque labels keep bool true and string "true" distinct on map-only APIs.
    if all(isinstance(option.value, str) for option in question.choices):
        return [option.value for option in question.choices]
    return [f"option_{i}" for i, _ in enumerate(question.choices)]


class NativeSystemOneCodec:
    dialect = "systemone"
    endpoint = "/v1/systemone"

    @staticmethod
    def parse_request(payload: Any) -> DecisionRequest:
        _base_request(payload)
        _require("input" not in payload, "SystemOne uses state, not input")
        _require("state" in payload, "state is required")
        _content(payload["state"], "state")
        raw = payload.get("questions")
        _require(isinstance(raw, dict) and bool(raw), "SystemOne questions must be a nonempty map")
        questions = []
        for identifier, item in raw.items():
            _require(isinstance(identifier, str), "Question IDs must be strings")
            _require(isinstance(item, dict), "Each question must be an object")
            kind = item.get("type")
            _require(isinstance(kind, str) and kind in {"noul", "choice", "score"}, "SystemOne question type must be noul, choice, or score")
            _require("instructions" in item, "Question instructions are required")
            _content(item["instructions"], "instructions")
            question = DecisionQuestion(identifier, "predicate" if kind == "noul" else kind, deepcopy(item["instructions"]), name=identifier)
            criteria = item.get("criteria")
            if kind == "noul" and criteria is not None:
                _require(isinstance(criteria, dict) and set(criteria) <= {"true", "false"}, "Noul criteria must describe true and/or false")
                for description in criteria.values():
                    _content(description, "Noul criteria")
                question.criteria = deepcopy(criteria)
            elif kind == "choice":
                _require(isinstance(criteria, dict) and 1 <= len(criteria) <= 255, "Choice criteria must be a map with 1–255 options")
                for choice, description in criteria.items():
                    _content(description, "Choice description", nullable=True)
                    question.choices.append(DecisionOption(choice, deepcopy(description)))
            elif kind == "score":
                _require(isinstance(criteria, list) and 2 <= len(criteria) <= 10, "Score criteria must contain 2–10 ordered levels")
                for i, description in enumerate(criteria):
                    _content(description, "Score level")
                    question.levels.append(DecisionOption(i, deepcopy(description), _text(description)))
            question.metadata = {k: deepcopy(v) for k, v in item.items() if k not in {"type", "instructions", "criteria"}}
            questions.append(question)
        options = {k: deepcopy(v) for k, v in payload.items() if k not in {"model", "state", "questions"}}
        if "assets" in options:
            assets = options["assets"]
            _require(isinstance(assets, dict), "assets must be an object")
            def strings(value):
                if isinstance(value, str):
                    yield value
                elif isinstance(value, dict):
                    for child in value.values():
                        yield from strings(child)
                elif isinstance(value, list):
                    for child in value:
                        yield from strings(child)
            content = list(strings([payload["state"], *[q["instructions"] for q in raw.values()], *[q.get("criteria") for q in raw.values()]]))
            for ref, asset in assets.items():
                _require(bool(ref.strip()) and any(ref in value for value in content), "Every image asset must have a referenced nonempty key")
                _require(isinstance(asset, dict) and asset.get("type", "image") == "image", "Only image assets are supported")
                _require("part" not in asset, "Multipart assets are not supported; use data_base64 JSON assets")
                _require(isinstance(asset.get("format"), str) and asset["format"] in {"png", "jpg", "jpeg"} and isinstance(asset.get("data_base64"), str), "Image assets require PNG/JPEG format and data_base64")
        return DecisionRequest(payload["model"], deepcopy(payload["state"]), questions, "systemone", options)

    @staticmethod
    def encode_request(request: DecisionRequest, model: str | None = None) -> dict:
        if request.dialect != "systemone":
            _require(not request_has_images(request), "Image conversion from OpenAI Decisions to SystemOne is unsupported; select the native dialect")
            _require(not (set(request.options) - {"think_mode", "stream"}), "OpenAI-only request options cannot be represented by SystemOne")
        questions = {}
        for question in request.questions:
            if request.dialect != "systemone":
                _require(not question.metadata and not any(o.metadata for o in [*question.choices, *question.levels]), "OpenAI question extensions cannot be represented by SystemOne")
            item = deepcopy(question.metadata)
            item.update(type="noul" if question.type == "predicate" else question.type,
                        instructions=deepcopy(question.instructions))
            if question.type == "predicate" and question.criteria is not None:
                item["criteria"] = deepcopy(question.criteria)
            elif question.type == "choice":
                item["criteria"] = {}
                for key, option in zip(_native_choice_keys(question), question.choices):
                    description = deepcopy(option.description)
                    if not all(isinstance(choice.value, str) for choice in question.choices):
                        description = {"value": option.value, "description": description}
                    item["criteria"][key] = description
            elif question.type == "score":
                _require(len(question.levels) <= 10, "SystemOne supports at most 10 score levels")
                item["criteria"] = [deepcopy(option.description) if request.dialect == "systemone" else
                                    ({"label": option.label, "description": option.description} if option.description is not None else option.label)
                                    for option in question.levels]
            questions[question.id] = item
        value = {"model": model or request.model, "state": deepcopy(request.input), "questions": questions}
        value.update(deepcopy(request.options))
        value.pop("stream", None)
        return value

    @staticmethod
    def decode_response(payload: Any, request: DecisionRequest) -> DecisionResult:
        if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
            raise UpstreamProtocolError("SystemOne upstream must return an answers map")
        if set(payload["answers"]) != {q.id for q in request.questions}:
            raise UpstreamProtocolError("SystemOne upstream answer IDs do not match request")
        answers = []
        for question in request.questions:
            item = payload["answers"][question.id]
            if not isinstance(item, dict):
                raise UpstreamProtocolError("Invalid SystemOne upstream answer")
            kind = item.get("type")
            if kind == "refusal":
                answers.append(DecisionAnswer(question.id, "refusal", metadata={k: deepcopy(v) for k, v in item.items() if k != "type"}))
                continue
            expected_kind = "noul" if question.type == "predicate" else question.type
            if kind != expected_kind:
                raise UpstreamProtocolError("SystemOne upstream answer type does not match request")
            answer = DecisionAnswer(question.id, question.type)
            if question.type == "predicate":
                answer.probability = _response_number(item.get("noul"), "noul", probability=True)
                answer.value = answer.probability
            else:
                answer.confidence = _optional_confidence(item)
                raw = item.get("probabilities")
                if raw is not None and not isinstance(raw, dict):
                    raise UpstreamProtocolError("SystemOne upstream probabilities must be a map")
                if question.type == "choice":
                    keys = _native_choice_keys(question)
                    mapping = dict(zip(keys, question.choices))
                    chosen = item.get("choice")
                    none_probability = item.get("none_probability")
                    if none_probability is not None:
                        none_probability = _response_number(none_probability, "none probability", probability=True)
                    if chosen is None and none_probability is not None:
                        answer.value = None
                    elif isinstance(chosen, str) and chosen in mapping:
                        answer.value = mapping[chosen].value
                    else:
                        raise UpstreamProtocolError("Unknown SystemOne upstream choice")
                    if raw is not None:
                        if set(raw) != set(keys):
                            raise UpstreamProtocolError("Incomplete SystemOne upstream choice distribution")
                        answer.probabilities = _distribution([{"value": mapping[key].value, "probability": value} for key, value in raw.items()], question.choices)
                else:
                    legend = item.get("legend")
                    if legend is not None and (not isinstance(legend, dict) or
                            set(legend) != {str(i) for i in range(len(question.levels))} or
                            any(not isinstance(description, (str, dict, list)) for description in legend.values())):
                        raise UpstreamProtocolError("Invalid SystemOne upstream score legend")
                    if raw is not None:
                        if set(raw) != {str(i) for i in range(len(question.levels))}:
                            raise UpstreamProtocolError("Incomplete SystemOne upstream score distribution")
                        answer.probabilities = _distribution([{"value": int(key), "label": question.levels[int(key)].label, "probability": value} for key, value in raw.items()], question.levels)
                    answer.value = _check_score(item.get("score"), answer.probabilities, len(question.levels))
            answer.metadata = {k: deepcopy(v) for k, v in item.items() if k not in {"type", "noul", "choice", "score", "confidence", "probabilities"}}
            answers.append(answer)
        return _result(payload, request, answers, "native_systemone")

    @staticmethod
    def encode_response(result: DecisionResult, request: DecisionRequest) -> dict:
        answers = {}
        for question, answer in zip(request.questions, _ordered_answers(result, request)):
            if answer.type == "refusal":
                raise DecisionValidationError("A provider refusal cannot be represented by the SystemOne response; use /v1/decisions")
            item = deepcopy(answer.metadata)
            item["type"] = "noul" if answer.type == "predicate" else answer.type
            if answer.type == "predicate":
                if answer.probability is not None:
                    item["noul"] = answer.probability
            elif answer.type in {"choice", "score"}:
                item[answer.type] = answer.value
                if answer.confidence is not None:
                    item["confidence"] = answer.confidence
                if answer.probabilities is not None:
                    item["probabilities"] = {str(p["value"]): p["probability"] for p in answer.probabilities}
                if answer.type == "score":
                    item.setdefault("legend", {str(option.value): deepcopy(option.description) for option in question.levels})
            answers[question.id] = item
        return _envelope(result, request, answers)


class OpenAIDecisionsCodec:
    dialect = "openai_decisions"
    endpoint = "/v1/decisions"

    @staticmethod
    def parse_request(payload: Any) -> DecisionRequest:
        _base_request(payload)
        _require("state" not in payload, "OpenAI Decisions uses input, not state")
        _require(isinstance(payload.get("input"), (str, list)), "input must be a string or an ordered array of input items")
        if isinstance(payload["input"], list):
            _validate_openai_input(payload["input"])
        raw = payload.get("questions")
        _require(isinstance(raw, list) and 1 <= len(raw) <= 200, "OpenAI Decisions questions must be an array of 1–200 questions")
        questions = []
        for i, item in enumerate(raw):
            _require(isinstance(item, dict), "Each question must be an object")
            kind = item.get("type")
            _require(isinstance(kind, str) and kind in {"predicate", "choice", "score"}, "OpenAI question type must be predicate, choice, or score")
            _require(isinstance(item.get("instructions"), str), "OpenAI question instructions must be a string")
            _require("name" not in item or isinstance(item["name"], str), "Question name must be a string when supplied")
            _require("choices" not in item or kind == "choice", "choices only applies to choice questions")
            _require("levels" not in item or kind == "score", "levels only applies to score questions")
            question = DecisionQuestion(f"q{i}", kind, item["instructions"], name=item.get("name"))
            if kind == "choice":
                choices = item.get("choices")
                _require(isinstance(choices, list) and 2 <= len(choices) <= 255, "choices must contain 2–255 options")
                seen = set()
                for choice in choices:
                    _require(isinstance(choice, dict) and isinstance(choice.get("value"), (str, bool)), "Choice value must be a string or boolean")
                    _require("description" not in choice or isinstance(choice["description"], str), "Choice description must be a string")
                    key = _typed_key(choice["value"])
                    _require(key not in seen, "Choice values must be unique, including their types")
                    seen.add(key)
                    question.choices.append(DecisionOption(choice["value"], choice.get("description"), metadata={k: deepcopy(v) for k, v in choice.items() if k not in {"value", "description"}}))
            elif kind == "score":
                levels = item.get("levels")
                _require(isinstance(levels, list) and len(levels) >= 2, "levels must contain at least two ordered levels")
                for j, level in enumerate(levels):
                    _require(isinstance(level, dict) and isinstance(level.get("label"), str), "Score level label must be a string")
                    _require("description" not in level or isinstance(level["description"], str), "Score level description must be a string")
                    question.levels.append(DecisionOption(j, level.get("description"), level["label"], {k: deepcopy(v) for k, v in level.items() if k not in {"label", "description"}}))
            question.metadata = {k: deepcopy(v) for k, v in item.items() if k not in {"type", "instructions", "name", "choices", "levels"}}
            questions.append(question)
        return DecisionRequest(payload["model"], deepcopy(payload["input"]), questions, "openai_decisions",
                               {k: deepcopy(v) for k, v in payload.items() if k not in {"model", "input", "questions"}})

    @staticmethod
    def encode_request(request: DecisionRequest, model: str | None = None) -> dict:
        if request.dialect != "openai_decisions":
            _require(not request_has_images(request), "Image conversion from SystemOne to OpenAI Decisions is unsupported; select the native dialect")
            _require(not (set(request.options) - {"think_mode", "stream"}), "SystemOne-only request options cannot be represented by OpenAI Decisions")
        _require(len(request.questions) <= 200, "OpenAI Decisions supports at most 200 questions")
        questions = []
        for question in request.questions:
            if request.dialect != "openai_decisions":
                _require(not question.metadata, "SystemOne question extensions cannot be represented by OpenAI Decisions")
            item = deepcopy(question.metadata)
            instructions = deepcopy(question.instructions)
            if question.criteria is not None:
                instructions = {"instructions": instructions, "criteria": question.criteria}
            item.update(type=question.type, instructions=_text(instructions))
            if question.name is not None:
                item["name"] = question.name
            if question.type == "choice":
                _require(len(question.choices) >= 2, "OpenAI Decisions requires at least two choices")
                item["choices"] = [{**deepcopy(option.metadata), "value": option.value, **({"description": _text(option.description)} if option.description is not None else {})} for option in question.choices]
            elif question.type == "score":
                item["levels"] = [{**deepcopy(option.metadata), "label": option.label if option.label is not None else str(option.value), **({"description": _text(option.description)} if option.description is not None else {})} for option in question.levels]
            questions.append(item)
        value = {"model": model or request.model,
                 "input": deepcopy(request.input) if request.dialect == "openai_decisions" else _text(request.input),
                 "questions": questions}
        value.update(deepcopy(request.options))
        value.pop("stream", None)
        return value

    @staticmethod
    def decode_response(payload: Any, request: DecisionRequest) -> DecisionResult:
        if not isinstance(payload, dict) or not isinstance(payload.get("answers"), list):
            raise UpstreamProtocolError("OpenAI Decisions upstream must return an answers array")
        if len(payload["answers"]) != len(request.questions):
            raise UpstreamProtocolError("OpenAI Decisions upstream answer count does not match request")
        answers = []
        for question, item in zip(request.questions, payload["answers"]):
            if not isinstance(item, dict) or item.get("name") != question.name:
                raise UpstreamProtocolError("OpenAI Decisions upstream answer name does not match request")
            kind = item.get("type")
            if kind == "refusal":
                answers.append(DecisionAnswer(question.id, "refusal", metadata={k: deepcopy(v) for k, v in item.items() if k not in {"type", "name"}}))
                continue
            if kind != question.type:
                raise UpstreamProtocolError("OpenAI Decisions upstream answer type does not match request")
            answer = DecisionAnswer(question.id, kind)
            if kind == "predicate":
                answer.probability = _response_number(item.get("probability"), "predicate probability", probability=True)
                answer.value = answer.probability
            else:
                answer.confidence = _optional_confidence(item)
                raw = item.get("probabilities")
                if raw is not None:
                    if not isinstance(raw, list):
                        raise UpstreamProtocolError("OpenAI Decisions upstream probabilities must be an array")
                    answer.probabilities = _distribution(raw, question.choices if kind == "choice" else question.levels)
                    if kind == "score" and any(entry.get("label") != question.levels[entry["value"]].label for entry in raw):
                        raise UpstreamProtocolError("OpenAI Decisions upstream score label does not match request")
                if kind == "choice":
                    chosen = item.get("choice")
                    if not isinstance(chosen, (str, bool)) or _typed_key(chosen) not in {_typed_key(option.value) for option in question.choices}:
                        raise UpstreamProtocolError("Unknown OpenAI Decisions upstream choice")
                    answer.value = chosen
                else:
                    answer.value = _check_score(item.get("score"), answer.probabilities, len(question.levels))
            answer.metadata = {k: deepcopy(v) for k, v in item.items() if k not in {"type", "name", "probability", "choice", "score", "confidence", "probabilities"}}
            answers.append(answer)
        return _result(payload, request, answers, "native_openai_decisions")

    @staticmethod
    def encode_response(result: DecisionResult, request: DecisionRequest) -> dict:
        answers = []
        for question, answer in zip(request.questions, _ordered_answers(result, request)):
            if answer.type == "choice" and (answer.value is None):
                raise DecisionValidationError("A provider none-of-the-above choice cannot be represented by the OpenAI Decisions response; use SystemOne")
            item = {k: deepcopy(v) for k, v in answer.metadata.items() if k != "legend"}
            item.update(type=answer.type, name=question.name)
            if answer.type == "predicate":
                if answer.probability is not None:
                    item["probability"] = answer.probability
            elif answer.type in {"choice", "score"}:
                item[answer.type] = answer.value
                if answer.confidence is not None:
                    item["confidence"] = answer.confidence
                if answer.probabilities is not None:
                    item["probabilities"] = deepcopy(answer.probabilities)
                    if answer.type == "score":
                        for entry in item["probabilities"]:
                            entry["label"] = question.levels[entry["value"]].label
            answers.append(item)
        return _envelope(result, request, answers)


def _validate_openai_input(items: list) -> None:
    calls = set()
    image_count = 0
    def parts(value: Any, role: str) -> None:
        nonlocal image_count
        if isinstance(value, str):
            return
        _require(isinstance(value, list), "Input content must be a string or array")
        allowed = {"input_text", "input_image"} if role in {"user", "tool"} else {"output_text", "refusal"} if role == "assistant" else {"input_text"}
        for part in value:
            _require(isinstance(part, dict) and isinstance(part.get("type"), str) and part["type"] in allowed, "Unsupported decision input content type for role")
            kind = part["type"]
            key = "image_url" if kind == "input_image" else "refusal" if kind == "refusal" else "text"
            _require(isinstance(part.get(key), str), f"Input {kind} requires {key}")
            if kind == "input_image":
                image_count += 1
                _require(part["image_url"].startswith(("data:image/", "https://", "http://")), "Images require a data URL or HTTP(S) URL")
    for item in items:
        _require(isinstance(item, dict), "Each input item must be an object")
        kind = item.get("type", "message")
        _require(isinstance(kind, str), "Input item type must be a string")
        if kind == "message":
            role = item.get("role")
            _require(isinstance(role, str) and role in {"user", "developer", "system", "assistant"}, "Unsupported decision input role")
            parts(item.get("content"), role)
        elif kind in {"function_call", "custom_tool_call"}:
            identifier = item.get("call_id")
            _require(isinstance(identifier, str) and identifier not in calls, "Tool calls require a unique call_id")
            _require(isinstance(item.get("name"), str), "Tool calls require a name")
            _require(isinstance(item.get("arguments" if kind == "function_call" else "input"), str), "Tool call content must be a string")
            calls.add(identifier)
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            _require(isinstance(item.get("call_id"), str) and item["call_id"] in calls, "Tool output must match an earlier call_id")
            parts(item.get("output"), "tool")
        else:
            raise DecisionValidationError("Unsupported decision input item type")
    _require(image_count <= 128, "At most 128 input images are supported")


CODECS = {"systemone": NativeSystemOneCodec, "openai_decisions": OpenAIDecisionsCodec}
