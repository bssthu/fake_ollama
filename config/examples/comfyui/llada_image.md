# LLaDA-Image / LLaDA-Image-Turbo

[llada_image.json](llada_image.json) supplies two `custom` ComfyUI targets and
their Playground profiles. Merge its targets, profiles and exposed models into
your existing configuration; retain your own access tokens. Workflow paths are
relative to the fake-ollama working directory, or can be made absolute.

These workflows use [T8's ComfyUI nodes](https://github.com/T8mars/Comfyui-LLaDa-Image-T8)
with the community INT8 ConvRot Mixed AIO files from
[t8star/LLaDa-Image-Comfy](https://huggingface.co/t8star/LLaDa-Image-Comfy).
Each file includes the text encoder, VAE and tokenizer and is approximately
27.65 GB. The quantization is experimental and is not lossless. The original
models are published by [inclusionAI](https://github.com/inclusionAI/LLaDA-Image).

Use ComfyUI 0.34.0 or later, the current T8 GitHub node code, and Comfy Kitchen
0.2.33. Put both AIO files in a directory registered as `checkpoints`. The local
installation uses `J:\Projects\LLM_Models\LLaDA-Image\models` through an
`extra_model_paths.yaml` file and a separate ComfyUI instance on port 21482.

Install the bundled [memory boundary node](../../../services/comfyui_memory/__init__.py)
as `ComfyUI/custom_nodes/comfyui_memory/__init__.py` before using these workflows,
then restart ComfyUI normally. Both positive and negative conditioning feed this
node before CFGGuider. It releases only their text encoder and its clones through
ComfyUI's model manager; conditioning tensors and the node cache remain intact.
The same boundary covers Base/Turbo generation and reference-image editing.

The launch examples also use `--disable-pinned-memory --vram-headroom 3`.
The first removes large host weight-transfer buffers; the second asks DynamicVRAM
to maintain 3 GiB of additional free VRAM, including usage by other applications.
This is a best-effort allocator margin, not a hard GPU memory cap. The existing
`--reserve-vram 2` remains for ComfyUI's other memory planning. These settings do
not change weights, precision, resolution, steps or sampling. Weight loading speed
depends on storage and system file caching; measurements below are specific to
this machine. Do not disable the conditioning cache to reduce the peak: doing so
would run the large text encoder again on every seed change.

| Model | Default resolution | Steps | CFG | Sampling |
| --- | --- | --- | --- | --- |
| `llada-image` | 1024 x 1024 | 50 | 5 | Euler with the LLaDA Base schedule |
| `llada-image-turbo` | 1024 x 1024 | 4 | 1 | T8's official Turbo stochastic flow sampler and schedule |

Playground exposes resolution, seed, steps, CFG and negative prompt for both
generation and editing; Base also exposes the sampler name. Turbo's sampler and
both model schedules are fixed by these workflows. Negative conditioning affects
classifier-free guidance when CFG differs from 1. Editing accepts one reference
image, resizes/crops it to the requested dimensions, and uses native LLaDA image
conditioning. It does not expose a denoise-strength control.

The local profiles use one image per API request, serial queued generations,
GPU exclusivity and automatic unloading. Their memory estimates are admission
settings, not benchmark results. A 24 GB GPU requires component offloading;
the complete AIO cannot remain on the GPU at once. The profiles use a 2 GiB
free-VRAM floor and dimensions divisible by 32 so generation and editing share
the same controls. Start at 1024 x 1024; larger settings need separate measurement.

The six upstream ComfyUI frontend workflows also include VQ generation. VQ is
available in ComfyUI and is not advertised as a separate Playground operation.
Turbo's four diffusion steps do not remove the preceding VQ token-generation
cost.

Open `http://127.0.0.1:21431/playground/`, enter an existing interface token,
load models, choose one of the two LLaDA models and expand **请求参数**.
The dedicated ComfyUI page is `http://127.0.0.1:21482/` while that runtime is running.

Initial validation before the memory optimization on 2026-09-08 used an RTX 4090
24 GB and 96 GB system RAM:
1024-square Base generation took 42.4 s and editing took 89.3 s; Turbo generation
through Playground took 16.0 s and editing took 18.1 s. A same-prompt Turbo seed
change took 4.8 s because text conditioning was cached. Full cold-start Turbo
API latency was 32.1 s. GPU peaks included the desktop and reached about 23.1 GiB.
The 1536-square request was rejected by the configured VRAM admission estimate
before execution; it was not an actual OOM measurement. Both model files passed
their published SHA256 checks. Raw local results and samples are in
`J:\Projects\LLM_Models\LLaDA-Image\validation`; these measurements cover the
tested fox prompt/edit only. Shared-runtime, workflow and API checks: 82 passed.

With the memory boundary, pinned memory disabled and DynamicVRAM headroom set to
3 GiB, the final 1024-square validation on the same day measured:

| Case | GPU peak before / after (GiB) | ComfyUI working-set peak before / after (GiB) | Time before / after (s) |
| --- | --- | --- | --- |
| User's Base prompt, uncached conditioning | 23.06 / 19.81 | 24.57 / 2.78 | 39.5 / 38.3 |
| Base reference-image editing | 22.71 / 20.06 | 27.52 / 2.58 | 89.3 / 83.5 |
| Turbo reference-image editing | 23.04 / 20.07 | 27.53 / 2.60 | 18.1 / 12.2 |

The first row's private commit fell from 47.72 to 21.60 GiB. Private commit is
virtual-memory commitment, not physical RAM usage, and must not be added to the
working set. GPU readings include the desktop. These are individual sampled runs,
not averages or memory guarantees; process and OS file-cache warmup differ.
The same-prompt Base seed change still ran all 50 steps (31.4 s) with cached
conditioning. Six fixed-input comparisons, covering both variants and both
operations, were pixel-identical to the original images.

Real Edge validation covered Playground auto-start and generation, plus loading
all six installed native workflows with the memory boundary connected. VQ
generation was not benchmarked. Both directions of Base/Turbo switching passed:
adaptive admission now unloads cached runtime weights when needed for a cold model
load, retaining the conditioning cache and verifying the actual free space before
submitting a prompt. The `keep` policy still prevents this cleanup. Related checks:
85 passed. Detailed reports, runtime logs and reproducible diagnostic scripts are
in `J:\Projects\LLM_Models\LLaDA-Image\validation\memory-optimization-20260908`.
