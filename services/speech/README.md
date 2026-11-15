# 语音翻译隔离运行环境

功能、配置、模型路径和本轮测试边界见 [即时语音翻译文档](../../docs/realtime-speech-translation.md)。

- `.venv`：Whisper、Voxtral 原生流式、Qwen3.5、TranslateGemma、Windows WASAPI、PyAV 内存解码。
- `qwen-asr/.venv`：官方 Qwen3-ASR 固定依赖，和新版 Transformers 隔离。
- 模型权重仅在指定 J/K 盘目录，不放入仓库。
- 安装：`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_speech_runtime.ps1`。
- 主配置需指定 `playground_speech_config` 并启用 Playground；`scripts/configure_speech.py` 只在语音配置不存在时创建文件和更新已有主配置，不负责启用页面。
- 新配置默认 `inference_enabled=false`；准备好权重和环境后显式改为 `true`。2026-11-01 的本机 GPU 验收记录见功能文档，当前就绪状态以配置和页面为准。安装与模型发现不会运行推理。
- Qwen 官方流式 API 需要 vLLM；此 Windows Transformers 环境保持自动语种的分段识别，推荐 16 秒窗口。
- 使用统一 server 启动参数 `--privacy-mode`；环境变量传入受管进程，内存传输不依赖客户端和服务器位于同一台电脑。
