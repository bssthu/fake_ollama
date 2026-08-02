# scripts 使用说明

这里的脚本用于直接调用正在运行的 fake_ollama 本地 API。默认都会读取仓库根目录的 `config.json`：

- `base_url`：使用第一个 `api_interfaces` 的 `host` / `port`
- `api_key`：使用第一个 `api_interfaces[*].access_tokens[0]`
- 认证头：通过 `x-api-key` 发送

正常使用时不需要手动传 `--base-url` 和 `--api-key`。如果要调试其他实例，可以显式覆盖这两个参数。

## 1. 视频生成：call_video_generation.py

用途：调用 fake_ollama 的 OpenAI 兼容视频生成接口：

```text
POST /v1/videos/generations
```

当前主要用于 `joyai-echo` 这类 ComfyUI 视频生成模型。

### 文生视频

```powershell
python scripts\call_video_generation.py `
  --prompt "a cat walking through a neon city, cinematic" `
  --output .tmp\joyai-video.mp4
```

常用参数：

- `--model`：默认 `joyai-echo`
- `--size`：默认 `256x256`
- `--num-frames`：默认 `17`
- `--fps`：默认 `8.0`
- `--prefetch-count`：默认 `1`
- `--enable-tile` / `--no-enable-tile`：默认启用分块 VAE 解码，降低峰值显存
- `--enable-streaming` / `--no-enable-streaming`：模型内部权重流式加载，默认关闭；不是 HTTP streaming
- `--seed`：默认 `123`
- `--timeout`：默认 `7200`
- `--output`：输出 mp4 路径

### 图生视频

传入一张或多张参考图时，脚本会调用 image-to-video workflow：

```powershell
python scripts\call_video_generation.py `
  --prompt "make this image move gently, cinematic camera motion" `
  --image path\to\reference.png `
  --output .tmp\joyai-i2v.mp4
```

多参考图：

```powershell
python scripts\call_video_generation.py `
  --prompt "animate these references into one short shot" `
  --image ref1.png ref2.png `
  --output .tmp\joyai-multi-ref.mp4
```

返回结果默认要求 `response_format=b64_json`，脚本会解码第一个视频结果并写入 `--output`。

## 多行 prompt 写法

PowerShell 里推荐先把多行文本放进变量，再传给 `--prompt`。这里字符串里的换行会原样保留：

```powershell
$prompt = @"
请生成一个 4 秒短视频：
- 主体是一只橘猫
- 场景是雨夜霓虹街道
- 镜头缓慢推进
- 风格偏电影感，不要文字水印
"@

python scripts\call_video_generation.py `
  --prompt $prompt `
  --output .tmp\joyai-video.mp4
```

识别脚本也一样：

```powershell
$prompt = @"
请分析这个视频，并按下面结构回答：
1. 场景和主体
2. 动作变化
3. 可见文字
4. 不确定的地方
"@

python scripts\call_joyai_vl_recognition.py .tmp\joyai-test.mp4 `
  --prompt $prompt `
  --output .tmp\joyai-vl-result.txt
```

如果 prompt 很长，也可以放到文件里再读取：

```powershell
$prompt = Get-Content .tmp\prompt.txt -Raw -Encoding UTF8

python scripts\call_joyai_vl_recognition.py input.mp4 `
  --prompt $prompt `
  --output .tmp\result.txt
```

## 2. 视频/GIF/图片识别：call_joyai_vl_recognition.py

用途：调用 fake_ollama 的 OpenAI Chat Completions 接口：

```text
POST /v1/chat/completions
```

默认模型是 `joyai-vl-interaction`。这个模型的 adapter 接收的是 OpenAI 风格多模态 chat 消息。视频和 GIF 会先抽取若干 JPEG 帧，再按时间顺序作为多个 `image_url` 发送；静态图片会直接作为单个 `image_url` 发送。

### 识别视频

```powershell
python scripts\call_joyai_vl_recognition.py .tmp\joyai-test.mp4 `
  --prompt "请识别这个视频，描述主要画面、人物或物体、动作变化和任何可见文字。" `
  --output .tmp\joyai-vl-result.txt
```

### 识别 GIF

```powershell
python scripts\call_joyai_vl_recognition.py path\to\input.gif `
  --prompt "请描述这个 GIF 的动作变化。" `
  --output .tmp\gif-result.txt
```

### 识别静态图片

```powershell
python scripts\call_joyai_vl_recognition.py path\to\image.png `
  --prompt "请描述这张图片里的内容和文字。" `
  --output .tmp\image-result.txt
```

常用参数：

- `--model`：默认 `joyai-vl-interaction`
- `--media-kind auto|video|image`：默认 `auto`
- `--max-frames`：视频/GIF 抽帧数量，默认 `8`
- `--frame-width`：抽帧后的最大宽度，默认 `512`
- `--max-tokens`：默认 `768`
- `--temperature`：默认 `0.2`
- `--session-id`：JoyAI adapter 会话 id；默认每次运行生成一个新 id
- `--output`：输出路径；`.json` 后缀会保存完整响应，否则保存文本内容
- `--keep-frames`：保留抽出的 JPEG 帧用于调试；默认会清理临时帧

注意：当前 JoyAI vLLM 主服务在 `I:\Projects\vllm\start_joyai_vl_interaction.sh` 里通过
`--limit-mm-per-prompt '{"image":32,"video":1}'` 限制了单次 prompt 最多 32 张图片。
视频/GIF 识别脚本会把抽出的每一帧作为一张 `image_url` 发送，所以 `--max-frames`
不能超过 32。默认值是 8。

### ffmpeg

视频/GIF 识别需要 `ffmpeg` 抽帧。脚本查找顺序：

1. 优先使用 Windows PATH 里的 `ffmpeg`
2. 如果没有本机 `ffmpeg`，会从 `config.json` 里 JoyAI target 的 `start_command` 推断 WSL distro，并使用类似：

```powershell
wsl -d Ubuntu-24.04 --exec ffmpeg
```

也可以手动指定：

```powershell
python scripts\call_joyai_vl_recognition.py input.mp4 `
  --ffmpeg-command "wsl -d Ubuntu-24.04 --exec ffmpeg"
```

## 认证和安全边界

- 默认请求 fake_ollama 的 `api_interface`，并带 `x-api-key`。
- `generic_openai_targets[*].auth_token` 可以为空；它只影响 fake_ollama 到本机 adapter 的出站调用，不代表外部客户端可以绕过 fake_ollama 的 `access_tokens`。
- `call_joyai_vl_recognition.py` 会显式禁用 Python `urllib` 的系统代理，避免本地视频帧 payload 误走环境代理。
- 不建议把 JoyAI adapter 的 `127.0.0.1:8070` 改成裸露到局域网的地址。对外调用应通过 fake_ollama 的 API 入口和 token 控制。

## 排错

检查 fake_ollama 是否看到模型：

```powershell
python scripts\call_joyai_vl_recognition.py --help
python scripts\call_video_generation.py --help
```

常见日志位置：

- fake_ollama：`logs\fake_ollama.log`
- fake_ollama 请求日志：`logs\fake_ollama.requests.jsonl`
- JoyAI adapter 启动 stderr 捕获：`logs\generic-openai-8070.err.log`
- JoyAI WSL adapter 实际日志：`/tmp/joyai_vl_adapter.log`
- JoyAI WSL vLLM 主模型日志：`/tmp/joyai_vl_main.log`
