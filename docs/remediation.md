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
| 2 | Tools were dropped, result IDs mismatched, indices collided, errors/EOF became success; 14 new regressions failed on old code | Ordered result association; complete tool arguments/IDs/indices; monotonic reverse indices; thinking preservation; explicit terminal/error state | Live provider versions NOT RUN; tool JSON is buffered until validated |
| 3 | Acquired semaphore leaked on cancellation; cleanup held coordinator lock; cancelled ComfyUI prompt lost its owner | Joined permit cleanup; dedicated shutdown error; shielded bounded lease cleanup; prompt reconciliation and shared runtime/GPU quarantine | An unreachable runtime stays quarantined until verified recovery; no real GPU run |
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

## Stage 2 handoff

- Files: `converters.py`, `reverse_converters.py`, `server.py`; new
  `protocol_tools.py`, `stream_state.py`, `tests/unit/test_protocol_completion.py`,
  `tests/contract/test_protocol_tools.py`; corrected existing stream fixtures and
  the old disconnect-as-success assertion in `tests/contract/test_server.py`.
- Both Anthropic-to-Ollama routes now emit complete tool calls once with ID and
  `function.index`. JSON fragments and initial inputs are validated before tools
  are emitted. OpenAI argument frames are buffered until block completion to avoid
  emitting concatenated initial/partial JSON as a successful tool call.
- Ollama history preserves IDs, generates request-unique missing IDs, and pairs
  results in order, validating optional `tool_name`/IDs and result counts. Same-name
  calls remain supported. Invalid association returns 400 before upstream work.
- Reverse Ollama streaming allocates monotonic indices across thinking, text and
  tools. Actual tool calls determine `tool_use` completion even when Ollama says
  `done_reason="stop"`. Non-stream output preserves `message.thinking` too.
- Errors use protocol envelopes, never natural-language completion text. Missing
  `message_stop`, `done`, or `[DONE]` produces an error. Runtime exceptions are
  sanitized; task cancellation propagates. Duplicate terminal events cannot produce
  duplicate success. Tool-containing stream cache entries now include complete
  validated tool blocks.
- Red command: `.venv/Scripts/python -m pytest -q tests/unit/test_protocol_completion.py
  --basetemp=.tmp/rectify-s2-red -p no:cacheprovider --tb=short`: **14 failed**.
- Final: `.venv/Scripts/python -m pytest -q --basetemp=.tmp/rectify-s2-final
  -p no:cacheprovider --tb=short`: **575 passed, 5 deselected**, one existing warning.
  Adds 48 tests to the 527-test stage 1 total (522 full-run + 5 added HTTP tests).
- Endpoints cover four directions, stream/non-stream, pure/mixed text and tools,
  same-name multi-calls, split arguments, next-turn result IDs, error events, EOF,
  and non-httpx exceptions. Unit tests additionally cover cancellation propagation,
  malformed JSON, initial input merging and terminal idempotence.
- References: [Ollama tools](https://docs.ollama.com/capabilities/tool-calling),
  [Anthropic streaming](https://platform.claude.com/docs/en/build-with-claude/streaming),
  [OpenAI Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
- Remaining: stages 3–6; actual network/GPU execution is NOT RUN.

## Stage 3 handoff

- Files: `llama_cpp_client.py`, `comfyui_client.py`, `comfyui_jobs.py`,
  `lifecycle.py`, `vram.py`, `admin.py`, `dashboard.py`; queue tests plus new
  cancellation/prompt-ownership tests and a recovery endpoint contract.
- Semaphore cleanup cancels AND joins both child tasks, observes successful
  acquisition and releases exactly once. Shutdown uses a dedicated 503 resource
  exception, while actual task cancellation remains `CancelledError`.
- Execution cleanup has one shielded owner, performs IO outside the condition,
  and is bounded to 30 seconds. Failures remove the execution lease but retain
  GPU quarantine. Callbacks ignoring cancellation stay tracked and prevent
  concurrent cleanup retries. Dashboard snapshots show releasing/quarantined state.
- ComfyUI assigns a unique request client ID and proposed prompt ID. On uncertainty,
  the endpoint is quarantined across clients and reloads; GPU load and execution
  admission also pause. Recovery queries history/queue, deletes only a matching
  queued prompt and rechecks running/pending state. Running work is observed to
  completion without `/interrupt`. Lost acknowledgement on older servers that ignore
  proposed IDs can be reconciled via the unique client ID in queue/history.
- Recovery attempts are bounded to 30 seconds. Failure retains records and HTTP
  resources. An authenticated `POST /admin/recover-runtime` with
  `{"target":"configured-comfyui-target-name"}` retries verification and returns
  `recovered=true` only after confirmation; otherwise 503 and pending prompt status.
  Dashboard data includes `pending_comfyui_prompts` and GPU quarantine snapshots.
- Read-only local protocol verification: ComfyUI source HEAD
  `491e8454c105910d69dce830771f2748a17983d8`, `server.py` accepts client-provided
  prompt IDs, `/queue` supports ID deletion, and `/interrupt` includes a targeted
  option implemented via a global processing interrupt after an ID check. This
  implementation deliberately avoids that interrupt path for shared-runtime safety.
  No ComfyUI process was started or mutated during verification.
- Red: `test_cancellation_ownership.py`: **2 failed, 1 passed**; prompt ownership:
  **1 failed**. The subsequent cancellation test additionally verifies cleanup is
  executed once rather than restarted after cancellation.
- Full offline command with `--basetemp=.tmp/rectify-s3-all -p no:cacheprovider
  --tb=short`: **589 passed, 5 deselected** (575 + 14).
- Additional focused run with `test_cancellation_ownership.py`,
  `test_comfyui_prompt_ownership.py`, `test_dashboard.py`, `test_admin.py`:
  **51 passed**, including two added bounded/repeated-cancellation tests.
- Remaining: stages 4–6. Unreachable or purged history cannot prove completion:
  quarantine intentionally persists. Actual GPU, networking and process teardown
  acceptance is NOT RUN.
