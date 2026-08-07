# Mage-VL 本机视频分析服务

该组件是 fake-ollama 仓库内的独立伴生服务，使用自己的 Python/CUDA 环境，
通过 OpenAI-compatible `POST /v1/chat/completions` 提供视频分析。主服务仅通过
HTTP 和生命周期脚本管理它，不会把 PyTorch/Transformers 引入 fake-ollama 主环境。

## 安装

默认安装到 `%LOCALAPPDATA%\fake-ollama\mage-vl`，也可显式指定路径：

```powershell
powershell -ExecutionPolicy Bypass -File services\mage_vl\scripts\install.ps1 `
  -InstallRoot D:\models\mage-vl
```

脚本会创建独立虚拟环境、安装 CUDA PyTorch、以 editable 模式安装本服务并下载权重。
`uv`、基础 Python、代理和安装目录都可通过参数指定，不包含个人目录默认值。

## 启动与停止

```powershell
powershell -ExecutionPolicy Bypass -File services\mage_vl\scripts\start.ps1
powershell -ExecutionPolicy Bypass -File services\mage_vl\scripts\stop.ps1
```

可通过 `MAGE_VL_INSTALL_ROOT`、`MAGE_VL_PYTHON`、`MAGE_VL_MODEL_DIR`、
`MAGE_VL_FFMPEG` 和 `MAGE_VL_TEMP_DIR` 覆盖运行路径。服务安装后也可直接运行：

```powershell
python -m mage_vl_adapter --host 127.0.0.1 --port 8071
```

## 验证

通过 fake-ollama 调用：

```powershell
python scripts\examples\call_mage_vl_video.py input.mp4 `
  --prompt "请按时间顺序分析关键动作和场景变化。" `
  --output .tmp\mage-vl-result.txt
```

真实 Playground/Mage 链路验证位于
`services\mage_vl\validation\validate_integration.py`。默认测试不加载真实模型；
GPU、FFmpeg 和摄像头验证属于显式的 integration/e2e 层。

当前仍采用 FFmpeg 分窗均匀抽帧，而不是依赖 `mamba-ssm` 的 StreamMind Gate。
GPU 请求严格串行；增加分段数只增加顺序耗时，不会并行保留多段显存状态。
