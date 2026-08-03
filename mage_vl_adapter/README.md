# Mage-VL 本机视频分析适配器

该适配器在 Windows 本机以 Transformers + PyTorch SDPA 加载
`microsoft/Mage-VL`，通过 `I:\Projects\Tools\ffmpeg\ffmpeg.exe` 将视频按时间窗
均匀抽帧，并提供 OpenAI 兼容的 `POST /v1/chat/completions`。

当前范围：本地视频文件或 Playground 摄像头短片段、单请求单视频、逐段流式
返回；不包含依赖 `mamba-ssm` 的 StreamMind 主动触发 Gate，也不接收远程 URL。

安装与启动：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_mage_vl.ps1
powershell -ExecutionPolicy Bypass -File scripts\start_mage_vl.ps1
```

优雅停止：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stop_mage_vl.ps1
```

通过 fake_ollama 验证：

```powershell
.\.venv\Scripts\python.exe scripts\call_mage_vl_video.py .tmp\joyai-test.mp4 `
  --prompt "请按时间顺序分析关键动作和场景变化。" `
  --output .tmp\mage-vl-result.txt
```

Playground 会把 capability 为 `video_understanding` 的模型显示为“视频分析”，
并使用 `video_url` data URL 内容部件上传文件。默认限制为 64 MiB、单个视频。
最多分析段数默认可选到 120；各段严格串行，因此增加段数只增加总耗时，
不会把多段视频张量同时留在显存。可通过 `MAGE_VL_MAX_SEGMENTS_LIMIT`
调整后端上限。启用整体总结时会按最多 24 段一批做分层总结，以限制每次
总结调用的上下文和 KV cache 峰值。

摄像头模式由浏览器通过 `getUserMedia` + `MediaRecorder` 实现。浏览器连续
生成独立的短视频窗口，并把每个窗口作为一次 `video_url` 请求串行交给适配器；
录制可与当前窗口的推理并行，但待处理队列最多保留一个窗口，新的窗口会替换
更旧的积压窗口。因此它属于“分窗近实时分析”，而不是模型原生逐帧流式输入，
运行时间和积压不会抬高 GPU 显存峰值。浏览器同时提供实际录制时长提示，以兼容
部分 `MediaRecorder` WebM 片段没有容器 Duration 元数据的情况。
