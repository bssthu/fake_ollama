# Local Laya companion for fake_ollama

独立的 CPU 服务，避免占用生成模型所需显存。使用固定 Laya 源码与多语言权重，运行时离线。

接口 `http://127.0.0.1:21461/v1/systemone`，模型 `laya-multilingual`，只监听回环地址。
使用 `runtime.py start|status|stop` 管理；stop 为带实例令牌的优雅退出，不强制终止进程。

本服务是 fake_ollama 项目的伴生组件。Jev 协议保持独立，不伪装成 Chat Completions。
权重约 322M，实际上下文限制由 checkpoint 决定；超长输入或题目返回 413，绝不静默截断。
中文聊天风险判断与概率未经任务校准，默认禁用推荐排名。

上游：https://github.com/NandhaKishorM/laya （Apache-2.0）
模型：https://huggingface.co/convaiinnovations/laya-multilingual （Apache-2.0）
精确源码与模型 revision、文件校验值见部署生成的 model-lock.json。

有 Python 与 uv 的 Windows 可运行 `setup.ps1` 复现固定源码/权重和依赖安装。
部署后运行 `.venv/Scripts/python.exe validate.py` 验证真实合成推理及协议限制。
`requirements-lock.txt` 记录本次实际依赖；使用的 CPU PyTorch 源为 download.pytorch.org/whl/cpu。
本次 60 组中文评估 precision 34.9%、recall 88.3%，误报较多，不作为已校准的自动决策服务。
