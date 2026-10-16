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
| 1 | Global thinking cache used text/tool ID and last-entry fallback; two revised negative tests failed on old code | Full normalized history + assistant content, scoped to credential/backend/wire model/config; explicit reasoning wins; bounded TTL/LRU; reload invalidates pending writers; signature policy | Exact replay within the same identity/history is intentionally the same association; truncated/edited history cannot recover thinking |
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

## Stage 1 handoff

- Files: `thinking_cache.py`, `converters.py`, `server.py`, `admin.py`,
  `config/models.py`; updated converter tests and new `test_thinking_isolation.py`
  and `test_thinking_boundary.py`.
- Chosen association: option (a), exact complete normalized client history plus
  current assistant content, including tool names, arguments and IDs. Normalization
  removes duplicate reasoning representations, normalizes tool JSON encoding and
  ignores protocol-only tool indices/types; it does not lowercase text, collapse
  whitespace, truncate history or use a last-entry fallback. No history/scope means
  no recovery. Stream callers without complete tool arguments cannot cache that turn
  until stage 2 supplies the complete tool block.
- Credential, source configuration, wire model, application identity and configuration
  are HMAC-derived with an ephemeral process secret. Keys contain no raw credentials
  or history. Cache holds at most 512 entries for 15 minutes; values are copied.
  Reload invalidates both stored entries and writers from the old epoch.
- Anthropic sources default to `allow_unsigned_thinking=false` and
  `missing_thinking_policy="omit"`. Compatible sources can explicitly allow unsigned
  reasoning and choose `"error"` when any assistant turn lacks recoverable thinking.
  Explicit reasoning is never replaced by a cached block; unsigned explicit reasoning
  is therefore omitted or rejected under a signed-only policy. Signatures received
  from the upstream are preserved; the gateway does not forge or validate signatures.
- Protocol references: [Anthropic extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking),
  [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling).
- Red: converter tests `-k 'rejects_reformatted or rejects_completely'`: **2 failed**.
- Full offline run: `.venv/Scripts/python -m pytest -q --basetemp=.tmp/rectify-s1-all
  -p no:cacheprovider --tb=short`: **522 passed, 5 deselected** (stage 0 + 27).
- Additional HTTP verification: same command with `tests/contract/test_thinking_boundary.py`
  and `--basetemp=.tmp/rectify-s1-http`: **5 passed**. It inspects actual mocked
  upstream payloads for both OpenAI/Ollama and stream/non-stream paths, credentials,
  models, history, changed upstream configuration, and pre-upstream policy errors.
- Remaining: stages 2–6; real upstream acceptance is NOT RUN.
