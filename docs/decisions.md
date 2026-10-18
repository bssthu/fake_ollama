# Typed decision models

Decision targets have a dedicated protocol and client pool. They do not call
`/v1/chat/completions` or turn chat-generated text into fabricated probabilities.
Models are routed using the same source/model aliases, interface exposure,
authentication, listener boundaries and reload snapshots as other providers.

## Two public contracts

- `POST /v1/systemone`: TypeSafe/SystemOne `state` and a map of `questions`.
  Question kinds are `noul`, `choice`, and `score`.
- `POST /v1/decisions`: OpenAI Decisions `input` and an array of `questions`.
  Question kinds are `predicate`, `choice`, and `score`.

The two paths are not simple aliases. Their codecs preserve typed options,
ordered score levels, question IDs, labels, provider-reported probabilities
and usage where supplied. Native OpenAI refusals are preserved; a refusal that
cannot be represented by SystemOne produces an explicit conversion error.
Lossless text conversions are supported;
nonportable image/options combinations produce an explicit error. There is no
streaming transport: `stream: true` is rejected before dispatch.

The optional official StartLux server also calls its **SystemOne-schema** alias
`/v1/decisions`. A StartLux target therefore uses `dialect: "systemone"` and its
native endpoint path independently of the public API requested by a client.
See [optional StartLux setup](decision-startlux.md).

## Configuration

Start with one of the complete examples:

- [Externally managed native endpoints](../config/examples/decisions/external-native.json)
- [Managed local llama.cpp](../config/examples/decisions/managed-llama-cpp.json)
- [Optional official StartLux setup](decision-startlux.md)

Declare `decision_targets`, expose the model on the intended API
interface, and give clients that listener's token. A source model `alias` is the
internal/display name, `upstream_id` is the native wire ID, and an exposure
`alias` is the public API ID. No model-name whitelist is used: models such as D3
mini, Clef, Intern and Kev can be served by their own compatible native runtimes.
Model weights and their licenses remain the operator's responsibility.

`runtime` describes native release checks and optional launch behavior:

- `external`: arbitrary native decision HTTP service. Default behavior never
  starts or stops it. An explicit `start_command` can manage an existing adapter
  such as the repository's `services/laya` companion, provided it exposes the
  configured native decision dialect/route.
- `llama_cpp`: native SystemOne runtime. With `auto_start: true`, supply a local
  `binary_path` and `model_path` (GGUF).
- `vllm`: native SystemOne runtime. With `auto_start: true`, supply the selected
  Python interpreter and a complete local model directory. It launches the
  installed vLLM API server with the configured wire model IDs.
- `startlux`: optional official package in the selected Python environment;
  official code owns prompts, AutoThink and attention behavior. It is not a
  required dependency of fake_ollama.

Built-in managed launches require loopback HTTP with an explicit port and no
base URL prefix. Extra tuning arguments cannot override model/download,
listener identity or configuration-file arguments. Generic explicit companion
commands are operator-managed and should also consume only prepared local
model files. Secrets should not be put in shell commands or extra arguments.

## Local models only

There is no model-download feature, automatic fallback, or download opt-in.
Install the runtime yourself and download all weights, tokenizer/processor
files and model metadata separately. `model_path` must exist at actual start;
configuration loading and ordinary app startup do not access/download weights.
Built-in Python launchers set Hub/Transformers offline flags. Missing artifacts
fail rather than being fetched. A source URL is informational license/source
metadata, never a download instruction.

## Capabilities and runtime checks

`decision_capabilities` describes this deployment, with canonical question
kinds `predicate`, `choice`, `score`, modalities (`text`, optionally `image`),
explicit extension names and allowed `think_modes`. `image_format` declares the
native image wire shape: `typesafe_assets`, `startlux_images`, or
`openai_input_image` (default selected from the explicit runtime/dialect). Defaults are text-only and
no extensions/modes. Do not enable AutoThink on Instant based on its name.
Use the official model metadata and runtime configuration. Observed StartLux
health restrictions can narrow declared modes/modalities; they never add
features that were not configured.

Read-only `/openapi.json`, `/version` and `/props` probes run on use and after a
managed restart. StartLux also supplies health metadata. OpenAPI app/schema
versions are **not** treated as runtime versions. A published route is reported
as `route_advertised`; `endpoint_verified` becomes true only after decoding an
actual decision response. This proves that request's protocol support, not
all question types, every model, attention correctness or score calibration.

Dated upstream evidence, checked 2026-10-17:

- [llama.cpp v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0)
  (2026-10-05) includes native SystemOne.
- [vLLM native SystemOne #59299](https://github.com/vllm-project/vllm/pull/59299)
  merged on 2026-10-08, after stable v0.31.0 (2026-10-05).
  [#60696](https://github.com/vllm-project/vllm/pull/60696) added noul/score on
  2026-10-09. Native deployments need a build with the relevant capabilities.
- The official StartLux wrapper's support for vLLM 0.19.1/0.31.0 is separate
  from native vLLM decision endpoints; it must not be rejected for lacking the
  latter.

Unknown/custom versions produce warnings. Known older release information is
also advisory because custom deployments may backport features; the real native
request establishes support. Missing/unsupported routes (404/405/501) produce an
actionable backend error rather than a chat fallback. Runtime version, warnings,
source URL and license metadata are available in target telemetry. Provider
scores remain provider-reported: this integration does not claim calibration.

## Lifecycle and resources

Managed decisions use protocol-neutral process ownership, startup health checks,
concurrency admission, cancellation-safe request references, idle shutdown and
VRAM/RAM estimates from `model_profiles`. Configure realistic estimates for your
own model; example numbers are placeholders, not measured requirements.

Active requests keep their original settings/client generation through reload.
A changed client is gated while the old generation drains. Failed cleanup keeps
replacement startup blocked until cleanup can be retried. External stop commands
alone do not grant ownership; explicit `manage_external_runtime` and actual use
are required. Chat and decision views of the same physical runtime/model share
resource admission and do not double-count its loaded memory in the dashboard.
Public aliases of the same physical weights should resolve to the same
`upstream_id`; different wire model IDs are conservatively treated as distinct
models. Dynamic external servers retain separate per-wire-model reservations.
Stopping an owned process clears all of its tracked model reservations.

Disconnecting a caller cancels queued work. For a submitted decision, the gateway
keeps the connection, concurrency slot and runtime reference while draining the
response, for up to `stop_timeout_seconds` (30 seconds by default). Closing HTTP
alone does not confirm that StartLux or another provider stopped inference.
Read/write failures or an expired cancellation drain mark the shared runtime's
execution as unconfirmed and reject further requests with `runtime_quarantined`.
Reload cannot discard that state. A successful owned-runtime stop or verified
process-tree cleanup clears it. For an externally managed service, verify that
the outstanding work finished (or restart that service) before restarting the
gateway. Health 200 alone is not a completion acknowledgement.

Cancelling a managed cold start keeps its unconfirmed VRAM/RAM reservation while
the owned process can still load weights. Telemetry marks it as `loading`; it
becomes confirmed when a later health check succeeds, or is cleared after a
verified stop. Another waiter for the same model retains its admission.

Fresh StartLux health information can restore configured thinking or image
capabilities after a runtime change, such as Instant to AutoThink. A failed,
incomplete or redacted probe retains prior restrictions. Probes never enable
capabilities beyond `decision_capabilities`.

A missing optional StartLux package does not prevent ordinary startup. First
managed use reports the child setup error promptly, including which Python
package/environment is missing. Nothing installs itself in response to a request.

## Images and limits

SystemOne JSON images require PNG/JPEG base64 `assets`, references in question or
state content, the `TypeSafe-Danger-Zone-Images: true` request header, and explicit
image capability. OpenAI image blocks retain their own native structure. Native
StartLux `images` options are preserved only with declared image support.
Cross-dialect image conversion is rejected instead of silently dropping pixels.
The existing request-body limit and listener authentication remain enforced.

## Verification scope

Offline tests cover native codec round-trips, routing and authentication,
capability/version diagnostics, managed/external lifetimes, absent optional
packages, missing local models, cancellation, reload draining and shared resource
accounting. They do not establish GPU execution speed, installed runtime support,
model quality, attention settings or calibration. No live model service or GPU
was used for these tests.

Run the Python suite with `python -m pytest -p no:cacheprovider`. The cancellation
regression includes a bounded loopback TCP server whose work continues after a
client disconnects; it does not depend on a mock transport cancelling inference.
On Windows without symbolic-link privileges, the interpreter-symlink check is
skipped explicitly.

The DOM tests need Node.js and jsdom (test dependencies only):

```sh
npm install --prefix .tmp/decision-frontend --ignore-scripts --no-audit --no-fund jsdom@26.1.0
NODE_PATH="$PWD/.tmp/decision-frontend/node_modules" node --test --test-isolation=none tests/frontend/playground_decision.test.cjs
```

In PowerShell set `$env:NODE_PATH` to the absolute `node_modules` directory before
running the same `node` command. These are DOM checks, not browser visual or real
model acceptance.
