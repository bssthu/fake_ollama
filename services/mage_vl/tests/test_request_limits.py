"""Small in-memory body boundaries; no live service, media decoder or model."""
import asyncio
from dataclasses import replace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from mage_vl_adapter.api import create_app
from mage_vl_adapter.request import read_json_request
from mage_vl_adapter.settings import AdapterSettings


def _request(chunks, headers=()):
    pending = list(chunks)
    async def receive():
        data = pending.pop(0)
        return {"type": "http.request", "body": data, "more_body": bool(pending)}
    return Request({"type": "http", "headers": list(headers)}, receive), pending


@pytest.mark.parametrize("declared", [False, True])
async def test_complete_json_is_accepted_at_byte_limit(declared):
    body = b'{"n":12}'
    headers = [(b"content-length", str(len(body)).encode())] if declared else []
    request, _ = _request([body[:4], body[4:]], headers)
    assert await read_json_request(request, AdapterSettings(max_request_body_bytes=len(body))) == {"n": 12}


@pytest.mark.parametrize("length", [None, b"1", b"9"])
async def test_size_limit_covers_chunked_and_declared_bodies(length, monkeypatch):
    headers = [(b"content-length", length)] if length is not None else []
    request, pending = _request([b"1234", b"56789", b"unused"], headers)
    parse = Mock(side_effect=AssertionError("oversize body must not be parsed"))
    monkeypatch.setattr("mage_vl_adapter.request.json.loads", parse)
    with pytest.raises(HTTPException) as error:
        await read_json_request(request, AdapterSettings(max_request_body_bytes=8))
    assert error.value.status_code == 413
    assert len(pending) == (3 if length == b"9" else 1)
    parse.assert_not_called()


@pytest.mark.parametrize("header,status", [
    ((b"content-length", b"-1"), 400),
    ((b"content-length", b"invalid"), 400),
    ((b"content-encoding", b"gzip"), 415),
])
async def test_invalid_body_headers_are_rejected_without_reading(header, status):
    request, pending = _request([b"{}"], [header])
    with pytest.raises(HTTPException) as error:
        await read_json_request(request, AdapterSettings())
    assert error.value.status_code == status and pending == [b"{}"]


async def test_body_timeout_cancels_the_pending_read():
    cancelled = asyncio.Event()
    async def receive():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
    request = Request({"type": "http", "headers": []}, receive)
    with pytest.raises(HTTPException) as error:
        await read_json_request(request, AdapterSettings(request_body_timeout_seconds=0.01))
    assert error.value.status_code == 408
    assert cancelled.is_set()


def test_chat_route_limits_all_fields_before_preparing_media(tmp_path, monkeypatch):
    settings = AdapterSettings(temp_root=tmp_path, max_request_body_bytes=8)
    prepare = Mock(side_effect=AssertionError("media preparation must not run"))
    monkeypatch.setattr("mage_vl_adapter.api.prepare_request", prepare)
    with TestClient(create_app(settings, engine=Mock())) as client:
        response = client.post("/v1/chat/completions", json={"note": "ordinary field"})
    assert response.status_code == 413
    prepare.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_default_limit_accounts_for_base64_and_environment_overrides(monkeypatch):
    settings = AdapterSettings(max_video_bytes=7)
    assert settings.request_body_limit == 12 + 1024 * 1024
    assert replace(settings, max_request_body_bytes=20).request_body_limit == 20
    monkeypatch.setenv("MAGE_VL_MAX_REQUEST_BODY_BYTES", "2048")
    monkeypatch.setenv("MAGE_VL_REQUEST_BODY_TIMEOUT_SECONDS", "5")
    configured = AdapterSettings.from_env()
    assert configured.request_body_limit == 2048
    assert configured.request_body_timeout_seconds == 5


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_body_deadline_cannot_be_disabled(timeout):
    with pytest.raises(ValueError):
        AdapterSettings(request_body_timeout_seconds=timeout)
