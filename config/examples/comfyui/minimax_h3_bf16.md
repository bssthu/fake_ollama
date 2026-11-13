# MiniMax H3 BF16 高精度入口

在现有 `minimax-h3` 量化入口旁新增 `minimax-h3-bf16`，使用同一套五种工作流。
配置片段见 [minimax_h3_bf16.json](minimax_h3_bf16.json)。将 target、profile、exposed model
分别合并到现有配置；示例路径需要替换，`context_ir_profile` 需要引用已配置的 profile。

## 精度与资源

| 组件 | 文件 | 大小 |
| --- | --- | --- |
| Base：文生、首帧、尾帧、首尾帧 | `minimax_h3_fl2va_pruned_bf16.safetensors` | 37.46 GiB |
| Ref2VA：参考图片、视频 | `minimax_h3_ref2va_pruned_bf16.safetensors` | 37.46 GiB |
| 共用文本编码器 | `qwen3vl_32b_minimax_h3_bf16.safetensors` | 47.97 GiB |
| 共用视频 VAE | `minimax_h3_video_vae_fp16.safetensors` | 4.85 GiB |
| 共用音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | 0.56 GiB |

BF16 和 FP16 都是 16 位浮点格式，此入口保留上游 BF16 权重，不强制转 FP16。
`pruned` 使用预计算 AdaLN 曲线表替换相应权重；它不是 INT8/FP8 量化，
也不是未裁剪原版，不能承诺与原版逐位一致。未裁剪 BF16 DiT 单份约 61.73 GiB，
连同 BF16 编码器和 VAE 约 115 GiB；这里选用每套约 91 GiB 的 Pruned BF16 组合。
两种 DiT 按任务选择，不要求同时驻留。

RTX 4090 24GB / 96GB RAM 需要 ComfyUI Dynamic VRAM 分层加载，部分权重留在系统内存，
计算时换入显存。保留 safetensors mmap，使用 `--cache-none --disable-pinned-memory`
减少缓存和锁页内存压力，`--bf16-text-enc` 保持文本编码器 BF16 计算。
使用 `--reserve-vram 2 --vram-headroom 3` 为桌面留出显存；其中 `--vram-headroom`
是 DynamicVRAM 的额外余量控制，不能仅依赖传统的 `--reserve-vram`。
这些参数针对支持 Dynamic VRAM 的 ComfyUI；`--lowvram` 在该模式下不控制层数。
显存不足时的换入换出会影响速度，升精度不保证提速或每个样例画质必然提升。

示例使用独立 `:21483`，原量化入口继续用 `:21481`。资源协调器将两者互斥执行；
BF16 profile 的 48 GiB RAM / 21 GiB VRAM 是准入预算，不是预分配量或峰值保证。
权重以 mmap 按需读取，文件总大小不等于常驻 RAM；该预算参考本机默认参数实测并保留余量。
请求完成后采用 `unload` 策略回收模型，仍需按分辨率、帧数和其他进程负载判断余量。

## 安装与目录

```powershell
.venv/Scripts/python.exe scripts/setup_h3_bf16.py --root J:/Projects/LLM_Models/MiniMax-H3
```

脚本仅在显式执行时下载，固定 Comfy-Org 仓库 revision，并对三份文件做 SHA-256 校验。
支持断点续传；下载中的 `.partial` 与分段记录不能改名为正式权重。已有正式文件校验不符时拒绝覆盖。
新增下载约 122.90 GiB，保留量化文件与已有 VAE，完成后生成 `bf16-manifest.json`。
若系统代理限制大文件速度而官方 CDN 可以直连，可加 `--direct-cdn --workers 12`：
仍从官方仓库解析下载地址，仅数据传输绕过系统代理，校验规则不变。

```text
MiniMax-H3/
  diffusion_models/  # FL2VA、Ref2VA 各自保留 INT8 和 Pruned BF16
  text_encoders/     # NVFP4-AWQ 与 BF16 并存
  vae/               # 两个入口共用，不重复下载
  bf16-manifest.json
  USAGE.md           # 本机路径及实际验证记录
  validation/
```

沿用 [额外模型路径配置](minimax_h3_extra_model_paths.yaml)。精度通过 target 的
`static_inputs` 分别覆盖五种工作流的加载器；Ref2VA 始终绑定自己的 DiT，
不能用 FL2VA 权重替代。原 preset 的量化默认值保持不变。

API 选择 `model: "minimax-h3-bf16"`；其他参数与 `minimax-h3` 相同。
Ref2VA 仍需显式 `video_mode: "ref2va"` 与 `prompt_mode: "raw"`。
首次验证可用 `size: "256x256"`、`num_frames: 22`、`steps: 2` 检查加载与完整输出，
这种短测试不代表 768p、124 帧、20 步的速度或画质验收。

来源：[Comfy-Org 权重](https://huggingface.co/Comfy-Org/MiniMax-H3)、
[MiniMax 官方集成说明](https://github.com/MiniMax-AI/awesome-minimax-h3-integration)。
