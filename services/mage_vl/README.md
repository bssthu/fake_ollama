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

启动、停止命令无需额外参数：服务每次启动会生成独立停机令牌，`stop.ps1` 按端口自动读取，
只向本机直接连接发送，不使用代理或跟随重定向。Windows 令牌位于
`%LOCALAPPDATA%\fake-ollama\mage-vl\control\shutdown-<端口>.token`，目录 ACL 仅允许当前用户访问；
它与自定义安装、模型和临时文件路径无关，停止脚本须由启动服务的同一用户运行。
正常退出会清理令牌，异常退出留下的令牌在下次启动时轮换；同一用户的同一控制端口不允许两个实例同时占用。
POSIX 使用 `$XDG_STATE_HOME`（默认 `~/.local/state`）下的对应目录，目录/文件权限分别为 700/600。
`/shutdown` 校验实际回环连接、Host/Origin 和令牌，不信任代理转发头；鉴权只作用于停机接口，
不改变模型请求或主服务 Admin、Playground 的访问方式。自定义 HTTP 停机调用须读取上述文件并携带
`Authorization: Bearer <令牌>`；推荐继续使用停止脚本，升级前已运行的旧实例也可通过该脚本退出。
新保护在 Mage-VL 下次启动后生效。仅导入 `create_app()`、未通过服务入口启动的应用默认禁用 HTTP 停机。

视频默认上限为 64 MiB（`MAGE_VL_MAX_VIDEO_BYTES`）。聊天接口会在解析 JSON 前限制整个请求体：
默认大小为视频上限经 base64 编码后的大小加 1 MiB，覆盖文本和其他字段；
也可通过 `MAGE_VL_MAX_REQUEST_BODY_BYTES` 单独设置。无 Content-Length 的分块请求同样受限，
超过限制返回 413。`MAGE_VL_REQUEST_BODY_TIMEOUT_SECONDS` 默认 30 秒，限制整个请求体的接收时间，
超时返回 408；该时间不限制后续视频分析。压缩请求体返回 415。配置变更须重启伴生服务生效。

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
