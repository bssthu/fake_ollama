# Correctness and lifecycle remediation

Baseline: `57a6a5c`, clean working tree. Requested sequence: prerequisite stage 0,
then stages 1–6, with a separate local commit for each completed stage.
Production `config.json`, its backups, paid services and GPU workloads are excluded.
Tests use synthetic configuration and mocked upstreams. Integration, e2e and GPU
tests are excluded by default; live tests require explicit environment variables.

## Stage checklist

| Stage | Issue / evidence | Repair and regression evidence | Remaining boundary |
| --- | --- | --- | --- |
| 0 | Unknown/retired ports borrowed the first interface; nine new tests failed against the old implementation | Both router and interface lookup fail closed. Python-only `create_app(test_listener_fallback=True)` explicitly enables synthetic test routing and uses the correct surface's auth. Ten added tests | Real socket rebinding and listener snapshots belong to stage 4 |
| 1 | Global thinking cache uses text/tool ID and last-entry fallback | Pending | Request scope, signatures, TTL and invalidation |
| 2 | Tool and stream completion conversion | Pending | End-to-end mock protocol contracts |
| 3 | Cancellation / orphan prompt / execution lease ownership | Pending | Deterministic barriers and mocked ComfyUI |
| 4 | Reload publication, drain, actual listener security, atomic persistence | Pending | Resource generations and rollback |
| 5 | Backend and process-tree ownership | Pending | Windows execution; report POSIX skips |
| 6 | Tool controls, usage, bounded module extraction and documentation | Pending | Public compatibility |

## Stage 0 handoff

- Files: `fake_ollama/server.py`, `pyproject.toml`, `tests/conftest.py`,
  `tests/contract/test_listener_boundary.py`, and existing contract/unit callers
  that previously depended on implicit test routing.
- Red: `.venv/Scripts/python -m pytest -q tests/contract/test_listener_boundary.py
  --basetemp=.tmp/rectify-s0-red-2 -p no:cacheprovider --tb=short`:
  **9 failed**, confirming unknown/retired listener bypasses.
- Green: `.venv/Scripts/python -m pytest -q --basetemp=.tmp/rectify-s0-final
  -p no:cacheprovider --tb=short`: **495 passed, 5 deselected**, one existing
  Starlette/httpx deprecation warning, 26 seconds. Baseline 485 + 10 new tests.
- Initial sandbox runs failed on temporary-directory permissions before test
  execution; the successful offline runs used approved execution outside that
  sandbox. No production config or live service was used.
- Remaining: stages 1–6 above. No real-network/GPU acceptance is claimed.
