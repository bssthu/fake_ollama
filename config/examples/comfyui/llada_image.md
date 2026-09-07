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

Local validation on 2026-09-08 used an RTX 4090 24 GB and 96 GB system RAM:
1024-square Base generation took 42.4 s and editing took 89.3 s; Turbo generation
through Playground took 16.0 s and editing took 18.1 s. A same-prompt Turbo seed
change took 4.8 s because text conditioning was cached. Full cold-start Turbo
API latency was 32.1 s. GPU peaks included the desktop and reached about 23.1 GiB.
The 1536-square request was rejected by the configured VRAM admission estimate
before execution; it was not an actual OOM measurement. Both model files passed
their published SHA256 checks. Raw local results and samples are in
`J:\Projects\LLM_Models\LLaDA-Image\validation`; these measurements cover the
tested fox prompt/edit only. Shared-runtime, workflow and API checks: 82 passed.
