# Local Laya companion for fake_ollama

独立的 CPU 服务，避免占用生成模型所需显存。使用固定 Laya 源码与多语言权重，运行时离线。

接口 `http://127.0.0.1:21461/v1/systemone`，模型 `laya-multilingual`，只监听回环地址。
使用 `runtime.py start|status|stop` 管理；stop 为带实例令牌的优雅退出，不强制终止进程。

该端口在 `runtime.py` / `service.py` 中固定。StartLux Windows 示例的 Instant 也使用 `21461`，两者不能同时照原样启动。
需要共存时，调整 [StartLux target 的 `base_url`](../../config/examples/decisions/startlux-windows-local.json) 到空闲端口；其管理启动会使用该端口。

本服务是 fake_ollama 项目的伴生组件。Jev 协议保持独立，不伪装成 Chat Completions。
权重约 322M，实际上下文限制由 checkpoint 决定；超长输入或题目返回 413，绝不静默截断。
中文聊天风险判断与概率未经任务校准，默认禁用推荐排名。

上游：https://github.com/NandhaKishorM/laya （Apache-2.0）
模型：https://huggingface.co/convaiinnovations/laya-multilingual （Apache-2.0）
精确源码与模型 revision、文件校验值见部署生成的 model-lock.json。

有 Python 与 uv 的 Windows 可运行 `setup.ps1` 复现固定源码/权重和依赖安装；该脚本最后会启动服务。
以下命令从 fake-ollama 仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File services\laya\setup.ps1
services\laya\.venv\Scripts\python.exe services\laya\runtime.py status
services\laya\.venv\Scripts\python.exe services\laya\runtime.py stop
# 已安装环境的再次启动
services\laya\.venv\Scripts\python.exe services\laya\runtime.py start
```

部署后运行 `services\laya\.venv\Scripts\python.exe services\laya\validate.py` 会调用真实 CPU 模型，验证合成推理及协议限制。
运行状态、PID、停止令牌和日志位于本目录的 `runtime/`；该目录不属于应提交的源码。
`requirements-lock.txt` 记录本次实际依赖；使用的 CPU PyTorch 源为 download.pytorch.org/whl/cpu。
此前本机 60 组中文评估 precision 34.9%、recall 88.3%，误报较多；这是历史样本结果，不作为已校准的自动决策服务。
