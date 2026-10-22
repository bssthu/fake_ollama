# scripts 使用说明

这里提供 fake_ollama API 调用示例和 ComfyUI 工作流构建工具。命令均在仓库根目录执行。
`examples/call_*.py` 调用示例默认读取仓库根目录的 `config.json`：

- `base_url`：使用第一个 `api_interfaces` 的 `host` / `port`
- `api_key`：使用第一个 `api_interfaces[*].access_tokens[0]`
- 认证头：通过 `x-api-key` 发送

API 调用示例正常使用时不需要手动传 `--base-url` 和 `--api-key`；调试其他实例时可显式覆盖。
这些脚本都会先读取 `--config` 文件（默认仓库根目录 `config.json`），即使显式传了 URL / key 也需要该文件存在。
`call_video_generation.py` 和 `call_joyai_vl_recognition.py` 要求非空 key，可由首个接口的 token 或显式 `--api-key` 提供；
`call_mage_vl_video.py` 也支持该接口没有 token 的情况。三个脚本都禁用 `urllib` 系统代理。
`--model` 必须是所选接口 `/v1/models` 返回的公开 ID；仅设置 source alias 不会自动产生同名的裸 ID。
[H3 Ref2VA 构建工具](#h3-ref2va-原生工作流) 只写出工作流 JSON，不读取 API 配置或凭据。
主服务启动参数见 [CLI 使用说明](../docs/cli.md)。

## 0. Mage-VL 本机视频分析：`examples/call_mage_vl_video.py`

先按 [`services/mage_vl/README.md`](../services/mage_vl/README.md) 安装独立服务并配置模型目录，
再从 [Windows 配置示例](../config/examples/config.windows-local.json) 合并 Mage-VL target、profile 和 exposure。
根目录的最小配置示例不包含 Mage-VL。设置 `auto_start: true` 和有效启动命令后，fake_ollama 才会在首次请求时启动适配器：

```powershell
python scripts\examples\call_mage_vl_video.py .tmp\joyai-test.mp4 `
  --prompt "请按时间顺序分析关键动作、场景变化和可见文字。" `
  --output .tmp\mage-vl-result.txt
```

默认按 8 秒窗口逐段处理、每段抽取 8 帧、最多分析 12 段。可用
`--segment-seconds`、`--frames-per-segment`、`--max-segments` 调整；
`--include-summary` 会在逐段结果后再生成整体总结。
其他默认值为 `--model mage-vl-local`、`--max-tokens 256`、`--temperature 0`、`--timeout 3600`。
脚本将整个视频作为 base64 JSON 上传，须同时满足网关和 Mage-VL 的整个请求体上限；`--output` 始终保存回答文本。
自动停机凭据由启动入口和停止脚本处理，不需要传入本脚本。

## 1. 视频生成

### 调用 fake_ollama API

脚本：`examples/call_video_generation.py`。

用途：调用 fake_ollama 的 OpenAI 兼容视频生成接口：

```text
POST /v1/videos/generations
```

适用于 `joyai-echo`、`minimax-h3` 等已配置的 ComfyUI 视频生成模型。

#### 文生视频

```powershell
python scripts\examples\call_video_generation.py `
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

#### 图生视频

传入一张或多张参考图时，脚本会调用 image-to-video workflow：

```powershell
python scripts\examples\call_video_generation.py `
  --prompt "make this image move gently, cinematic camera motion" `
  --image path\to\reference.png `
  --output .tmp\joyai-i2v.mp4
```

多参考图：

```powershell
python scripts\examples\call_video_generation.py `
  --prompt "animate these references into one short shot" `
  --image ref1.png ref2.png `
  --output .tmp\joyai-multi-ref.mp4
```

返回结果默认要求 `response_format=b64_json`，脚本会解码第一个视频结果并写入 `--output`。

### H3 Ref2VA 与任意帧关键帧

统一 API 已接入 Ref2VA，并沿用 fake_ollama 的资源准入与排队。使用独立 Ref2VA 权重，
target 需为 `minimax_h3` preset，`max_reference_images` 设为 9 可使用全部参考图槽位。

```powershell
python scripts\examples\call_video_generation.py `
  --model minimax-h3 --video-mode ref2va --prompt-mode raw `
  --prompt "The person in <Picture 1> follows the movement in <Video 1>." `
  --image identity.png --video motion_24fps.mp4 `
  --keyframe 36 middle.png --keyframe -1 ending.png `
  --size 768x768 --num-frames 124 --fps 24 --steps 20 `
  --output .tmp\h3-reference.mp4
```

- `--video-mode`: `auto|t2va|i2va|fl2va|l2va|ref2va`；`auto` 保持旧的 0／1／2 图语义。
- `--video` 可重复，最多 3 段；`--image` 最多 9 张，Ref2VA 至少需要一项参考素材。
- `--keyframe FRAME_INDEX IMAGE_PATH` 可重复，最多 16 项；帧号从 0 开始，`-1` 是末帧。
  Base 模式同样支持该参数，但不能与已有首／末帧约束占用同一位置。
- Ref2VA 和关键帧要求 `--prompt-mode raw`，不调用目前只支持 Base 模式的 Context-IR。
- H3 的宽高须是 32 的倍数，帧数为 `17k+5`，输出固定 24 FPS。
- 参考视频提前转换为 24 FPS、2–15 秒；目前只使用画面，不使用参考音轨。
- 关键帧需要服务端 `MiniMaxH3AddGuide`；[安装与回滚说明](../services/comfyui_h3_guides/README.md)。

Playground 中选择 `H3 video mode` → `Ref2VA` 后上传图片／视频。
高级参数 `Keyframes (JSON)` 可填写 `[{"image_index":0,"time_seconds":1.5}]`，
其中图片索引只计算图片，所选图片转为关键帧而不再作为普通参考图。
选择 Ref2VA 或填写关键帧会将 Prompt 处理方式设为 raw。

### H3 Ref2VA 原生工作流

[`examples/build_h3_ref_workflow.py`](examples/build_h3_ref_workflow.py) 生成可提交到 ComfyUI `/prompt` 的 JSON，
适用于用图片指定角色、用视频参考动作的生成任务。构建命令只写文件，不启动服务、不上传素材，也不提交推理。
模型路径配置见 [MiniMax H3 后端说明](../README.md#本机-minimax-h3-768p-后端)。

#### 准备运行环境与素材

1. 启动已配置公共模型目录的 H3 ComfyUI。检查 `GET /object_info/UNETLoader` 的候选列表包含
   `minimax_h3_ref2va_pruned_int8_convrot.safetensors`，并确认存在
   `MiniMaxH3ReferenceToVideo`、`LoadVideo`、`GetVideoComponents` 节点。
2. 将参考素材上传或放入**这个 ComfyUI 实例**的 input 目录。参数填写其 input 相对文件名，
   例如 `identity.png`、`motion_24fps.mp4`；不能填写客户端绝对路径或含 `..` 的路径。
3. 将参考视频预先转换为 24 fps。节点会截短过长视频，并将视频参考长度向下对齐 `17k+5` 网格；
   不会自动根据原始 FPS 保持时长。需要放慢动作时，先调整参考视频时间轴。
4. 创建非空 UTF-8 提示词文件，例如 `.tmp/h3-reference-prompt.txt`。
   用 `<Picture 1>`、`<Video 1>` 明确身份、着装和动作参考的分工；图片和视频分别从 1 编号，按各自参数顺序对应。

#### 构建工作流

```powershell
.\.venv\Scripts\python.exe scripts\examples\build_h3_ref_workflow.py `
  --prompt-file .tmp\h3-reference-prompt.txt `
  --image identity.png --image wardrobe.png --video motion_24fps.mp4 `
  --width 768 --height 768 --frames 141 --steps 25 --seed 2026092192 `
  --output .tmp\h3-reference-workflow.json
```

| 参数 | 默认值与限制 |
| --- | --- |
| `--image` / `--video` | 可重复，最多 9 张图和 3 段视频；至少提供一项参考素材 |
| `--width` / `--height` | 默认 768×768；均为 32 的正整数倍，总像素不超过 768×1344 |
| `--frames` | 默认 124；范围 5–362，满足 `17k+5`；输出固定 24 fps |
| `--steps` / `--seed` | 默认 20 步、种子 0；步数范围 1–100 |
| `--first-frame` | 可选首帧图片文件名；要求服务端已有 `MiniMaxH3AddGuide` |
| `--output` | 必填；写出 JSON，已有文件不会被覆盖 |

普通 Ref2VA 无需 `MiniMaxH3AddGuide`。使用 `--first-frame` 前应检查
`GET /object_info/MiniMaxH3AddGuide`，不能仅凭模型文件已安装就认定该节点可用。
构建工具目前不提供参考音频输入。

#### 提交与获取结果

原生 `/prompt` 请求直接进入 ComfyUI，不经过 fake_ollama 的请求资源协调器。
生成时应独占 H3 队列、串行提交，监测 GPU、RAM 和系统提交余量，保留至少 3 GiB 空闲显存；
不足时向目标 ComfyUI 请求 `/interrupt`。此工具不实施这些资源检查，应由提交端负责。

`21481` 是示例 H3 ComfyUI 端口，按实际运行实例调整：

```powershell
$comfyUrl = 'http://127.0.0.1:21481'
$workflow = Get-Content .tmp\h3-reference-workflow.json -Raw -Encoding UTF8 | ConvertFrom-Json
$body = @{ prompt = $workflow; client_id = [guid]::NewGuid().ToString() } | ConvertTo-Json -Depth 100
$result = Invoke-RestMethod "$comfyUrl/prompt" `
  -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
$result.prompt_id
```

通过 `GET /history/{prompt_id}` 查看执行状态和输出，再按输出的 `filename`、`subfolder`、`type`
调用 `/view` 下载。需要 fake_ollama 负责资源准入与排队时，使用上面的统一 API 入口。

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

python scripts\examples\call_video_generation.py `
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

python scripts\examples\call_joyai_vl_recognition.py .tmp\joyai-test.mp4 `
  --prompt $prompt `
  --output .tmp\joyai-vl-result.txt
```

如果 prompt 很长，也可以放到文件里再读取：

```powershell
$prompt = Get-Content .tmp\prompt.txt -Raw -Encoding UTF8

python scripts\examples\call_joyai_vl_recognition.py input.mp4 `
  --prompt $prompt `
  --output .tmp\result.txt
```

## 2. 视频/GIF/图片识别：`examples/call_joyai_vl_recognition.py`

用途：调用 fake_ollama 的 OpenAI Chat Completions 接口：

```text
POST /v1/chat/completions
```

默认模型是 `joyai-vl-interaction`。这个模型的 adapter 接收的是 OpenAI 风格多模态 chat 消息。视频和 GIF 会先抽取若干 JPEG 帧，再按时间顺序作为多个 `image_url` 发送；静态图片会直接作为单个 `image_url` 发送。

### 识别视频

```powershell
python scripts\examples\call_joyai_vl_recognition.py .tmp\joyai-test.mp4 `
  --prompt "请识别这个视频，描述主要画面、人物或物体、动作变化和任何可见文字。" `
  --output .tmp\joyai-vl-result.txt
```

### 识别 GIF

```powershell
python scripts\examples\call_joyai_vl_recognition.py path\to\input.gif `
  --prompt "请描述这个 GIF 的动作变化。" `
  --output .tmp\gif-result.txt
```

### 识别静态图片

```powershell
python scripts\examples\call_joyai_vl_recognition.py path\to\image.png `
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

注意：JoyAI vLLM 主服务通常通过
`--limit-mm-per-prompt '{"image":32,"video":1}'` 限制单次 prompt 最多 32 张图片。
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
python scripts\examples\call_joyai_vl_recognition.py input.mp4 `
  --ffmpeg-command "wsl -d Ubuntu-24.04 --exec ffmpeg"
```

## 认证和安全边界

- `call_*.py` 默认请求 fake_ollama 的 `api_interface`，并带 `x-api-key`；Ref2VA 构建工具仅写本地 JSON。
- `generic_openai_targets[*].auth_token` 可以为空；它只影响 fake_ollama 到本机 adapter 的出站调用，不代表外部客户端可以绕过 fake_ollama 的 `access_tokens`。
- 三个 HTTP 调用示例均显式禁用 Python `urllib` 的系统代理。
- 不建议把 JoyAI adapter 的 `127.0.0.1:8070` 改成裸露到局域网的地址。对外调用应通过 fake_ollama 的 API 入口和 token 控制。

## 排错

查看脚本参数（不会连接模型服务）：

```powershell
python scripts\examples\call_joyai_vl_recognition.py --help
python scripts\examples\call_video_generation.py --help
python scripts\examples\call_mage_vl_video.py --help
```

确认 fake_ollama 是否暴露目标模型，须实际查询所选接口的模型列表，例如：

```powershell
$headers = @{ 'x-api-key' = '替换为所选接口的 token' }
Invoke-RestMethod http://127.0.0.1:21435/v1/models -Headers $headers
```

以下是默认或示例部署的日志位置；后端日志以实际启动命令和运行环境为准：

- fake_ollama：`logs\fake_ollama.log`
- fake_ollama 请求日志：`logs\fake_ollama.requests.jsonl`
- JoyAI adapter 启动 stderr 捕获：`logs\generic-openai-8070.err.log`
- JoyAI WSL adapter 实际日志：`/tmp/joyai_vl_adapter.log`
- JoyAI WSL vLLM 主模型日志：`/tmp/joyai_vl_main.log`
