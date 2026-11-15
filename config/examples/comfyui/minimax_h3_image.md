# MiniMax H3-Image 本地图片入口

使用独立的新版 ComfyUI，将 H3-Image 文生图和参考图编辑接入 OpenAI Images API 与 Playground。
配置见 [minimax_h3_image.json](minimax_h3_image.json)，内置 preset 为 `minimax_h3_image`。
图片模型使用自己的 DiT 和图像 VAE，不能替换成 H3 视频组件。

## 两种精度

| API 模型名 | DiT | 文本编码器 | 图像 VAE |
| --- | --- | --- | --- |
| `minimax-h3-image` | Pruned W4A8，11.69 GiB | Qwen3-VL NVFP4-AWQ，14.61 GiB | FP16，1.09 GiB |
| `minimax-h3-image-bf16` | Pruned BF16，37.47 GiB | Qwen3-VL BF16，47.97 GiB | FP16，1.09 GiB |

BF16 入口的 DiT 和文本编码器不做低比特量化。`pruned` 使用预计算 AdaLN 表等处理，
不是未裁剪原版，不能称为与原版逐位一致的“满血版”。BF16 与 FP16 都是 16 位浮点格式，
这里保留上游 BF16 权重，不强制转换为 FP16。

24GB 显卡通过 ComfyUI Dynamic VRAM 分层加载，将部分权重留在系统内存并按需换入 GPU。
保留 safetensors mmap，使用 `--cache-none --disable-pinned-memory --bf16-text-enc`；
图像 VAE 文件为 FP16，示例通过 `--fp32-vae` 使用 FP32 计算。
`--vram-headroom 3` 为动态分配器保留余量。示例中的 RAM/VRAM 数值是请求准入预算，
不是显存预分配量或峰值保证。两个精度入口共用一个运行时，串行执行，并在完成后卸载模型。

## 安装

官方模板要求 ComfyUI 0.45.0+ 和模板包 0.11.104+。本次部署固定的上游源码版本为
`4a9636acd031525d37b1023af36085313b1a1543`；依赖使用该版本的 requirements。
建议单独安装，避免升级已有视频环境。

```powershell
.venv/Scripts/python.exe scripts/setup_h3_image.py --root J:/Projects/LLM_Models/MiniMax-H3-Image --direct-cdn --workers 12
```

下载器固定 Comfy-Org revision，对 W4A8、BF16 和 VAE 三份文件逐一校验 SHA-256，
新增约 50.25 GiB，支持 `.partial` 断点续传；校验成功后生成 `image-manifest.json`。
文本编码器复用既有 `MiniMax-H3/text_encoders`，不重复下载。
缺少编码器时，需按官方来源补齐；BF16 编码器也可通过已有的
[`setup_h3_bf16.py`](../../../scripts/setup_h3_bf16.py) 下载。

配置 `extra_model_paths.yaml`：

```yaml
h3_image:
  base_path: J:/Projects/LLM_Models/MiniMax-H3-Image
  diffusion_models: diffusion_models
  vae: vae
h3_shared_text:
  base_path: J:/Projects/LLM_Models/MiniMax-H3
  text_encoders: text_encoders
```

替换示例中的 `C:/path/to/H3-Image`，再将 targets、profiles 和 exposed models 合并到本机配置。
示例端口为 21484，两个 target 的启动命令和地址必须保持相同。

## 使用

Playground 提供 2K（2048×2048）和 4K（4096×4096）预设，均为原生输出。
宽高须为 32 的倍数；示例最多每次输出一张图。
工作流固定使用 euler / simple、6 steps、CFG 1、denoise 1，不提供无效的负面提示词和采样调节控件。
请求里的 steps、cfg 等未绑定字段不改变该固定采样流程。

```json
{
  "model": "minimax-h3-image",
  "prompt": "A detailed high-resolution photograph of ...",
  "size": "2048x2048",
  "seed": 20261115,
  "response_format": "b64_json"
}
```

将以上 JSON 发到 `POST /v1/images/generations`，模型名改成 `minimax-h3-image-bf16`
即可切换精度。`POST /v1/images/edits` 接受 multipart `image[]` 或 JSON base64 `images`。
最多 16 张参考图，保持上传顺序；在提示词中用 `<Picture 1>`、`<Picture 2>` 指定角色。
不同尺寸的参考图分别交给原生 conditioning 节点处理，不拼接成等尺寸 ImageBatch。
本地入口直接使用提示词，不自动启动提示词扩写模型。

推荐先从 2K 单参考图验证，再增加分辨率和参考图数量；16 张是接口容量，不是本机峰值性能承诺。
部署测量、生成图片和原生工作流记录放在模型目录 `validation/`，本机说明为 `USAGE.md`。

2026-11-15 本机 RTX 4090 / 96GB RAM 已完成 W4A8 和 Pruned BF16 的 2K、4K 文生图实测。
单参考图编辑也能执行，但茶壶变色样例在背景和材质上产生明显纹理伪影；改用 BF16 编码器、
BF16 DiT 或 FP32 VAE 计算均未消除。单独 VAE 重建没有同等伪影，原因仍未确定。
这是本次样例的画质限制，不能将成功返回 PNG 等同于编辑质量验收通过。

来源：[MiniMax 模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)、
[Comfy-Org 权重](https://huggingface.co/Comfy-Org/MiniMax-H3-Image)、
[ComfyUI 教程](https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image)。
