"""Offline decision Playground DOM regression (optional Node.js + jsdom)."""

from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest


def test_decision_playground_dom(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the offline Playground DOM regression")
    available = subprocess.run(
        [node, "-e", "require.resolve('jsdom')"], capture_output=True, text=True, timeout=20,
    )
    if available.returncode:
        pytest.skip("Install jsdom or expose it through NODE_PATH for offline DOM QA")
    script = Path(__file__).resolve().parents[1] / "frontend" / "playground_decision.test.cjs"
    fixtures = tmp_path / "playground-requests.json"
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=30,
        env={**os.environ, "PLAYGROUND_REQUEST_FIXTURES": str(fixtures)},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # Exercise the actual public codec rather than trusting matching mock bodies.
    from fake_ollama.decisions import NativeSystemOneCodec, OpenAIDecisionsCodec
    from fake_ollama.decision_client import DecisionClient

    codecs = {"/v1/systemone": NativeSystemOneCodec, "/v1/decisions": OpenAIDecisionsCodec}
    requests = json.loads(fixtures.read_text(encoding="utf-8"))
    assert {request["endpoint"] for request in requests} == set(codecs)
    for request in requests:
        parsed = codecs[request["endpoint"]].parse_request(request["body"])
        assert parsed.questions
        # Verify image envelopes against the real backend capability gate too.
        validator = DecisionClient.__new__(DecisionClient)
        validator._capabilities = request["capabilities"]
        validator._validate_capabilities(parsed)
