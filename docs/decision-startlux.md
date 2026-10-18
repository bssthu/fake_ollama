# 官方 StartLux 可选运行时

本集成使用 **StartLux 官方 Python 包** `startlux-decide`。fake_ollama 只做决策协议、路由和进程管理；prompt、read-out、temperature、AutoThink、Instant attention 全部交给官方实现。没有复制官方源码，也没有把决策模型当成普通 chat 模型重新实现。

普通 fake_ollama 安装和测试不依赖 StartLux、torch、transformers 或 vLLM。只有选择此运行时后启动的独立 Python 进程才导入官方包。启动脚本位于已打包的 `fake_ollama/startlux_companion.py`，不要求从源码仓库启动。

## 1. 由操作者安装可选环境

截至 **2026-10-17**，参考包版本为 `1.1.0`，官方源码固定到
[`83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01`](https://github.com/StartLuxLabs/StartLux-Decision/commit/83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01)。这里使用官方 Git 仓库，不假定有同名 PyPI 发布。

在适合所选引擎的机器上执行以下命令（这些命令是安装说明，fake_ollama 不会替你执行）：

```sh
python -m venv services/startlux/.venv
services/startlux/.venv/bin/python -m pip install -r services/startlux/requirements-vllm.txt
services/startlux/.venv/bin/python -m pip freeze > services/startlux/installed-packages.txt
```

`[vllm]` extra 在此源码中固定 vLLM `0.31.0`。请选择与显卡、驱动、操作系统匹配的 vLLM/PyTorch 环境。Windows 上 Python 路径通常为 `services/startlux/.venv/Scripts/python.exe`；原生 Windows 是否支持所选 GPU 引擎，以该引擎的平台说明为准，不要将路径替换当作平台支持保证。

如果使用官方 transformers 参考后端，改装 `requirements-transformers.txt`，目标设置 `runtime_backend: "transformers"`。可按硬件明确添加 `start_args: ["--device", "cpu", "--dtype", "float32"]`；CPU 可运行不代表速度或内存满足你的模型需求。

如果从已安装的 fake-ollama wheel 使用，可在任意独立环境执行同等的固定来源安装命令：

```sh
/path/to/optional/python -m pip install 'startlux-decide[vllm] @ git+https://github.com/StartLuxLabs/StartLux-Decision.git@83fe6c7d61a687a6e17f118b4f7d8f429f5d3a01'
```

安装、CUDA 配置、完整依赖锁定由操作者控制。随仓库提供的 requirements 固定了官方源码，而不是所有传递依赖的完整锁文件。保留实际安装的 `pip freeze` 和模型下载 revision，便于复现。

## 2. 提供你已下载的本地模型目录

本项目不打包权重、不下载模型、不接受模型许可，也不尝试填补缺失文件。请自行取得可合法使用的模型，并将 `model_path` 指向本地目录，保留原始：

- `config.json`、`decision_config.json`
- tokenizer、chat template 和所选引擎需要的 processor 文件
- 本机推理需要的全部权重及索引分片

禁止把 Hugging Face 仓库 ID、URL 或不存在的路径当作 `model_path`。已有本地目录可以使用相对路径（相对目标 `cwd`），但建议绝对路径。伴生进程在导入官方包前强制设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，并关闭 Hub telemetry；缺少模型/tokenizer 文件会报错，不会联网补下载。

这里的离线限制针对模型文件。明确配置的本地或远程推理引擎 HTTP 地址仍可使用，推理请求会发送到该引擎。

## 3. 先检查，不启动 GPU

在安装 fake_ollama 的 Python 环境中定位 companion 文件：

```sh
python -c "import pathlib, fake_ollama; print(pathlib.Path(fake_ollama.__file__).with_name('startlux_companion.py'))"
```

然后用独立环境 Python 执行这个绝对路径：

```sh
/path/to/optional/python /absolute/path/startlux_companion.py --check serve \
  --model /absolute/path/to/downloaded-model --backend vllm --launch-vllm
```

`--check` 读取已安装 distribution 版本，检查官方 CLI/API，调用官方模型元数据校验函数，打印 JSON 报告后退出。它不启动引擎、不加载权重、不做真实推理，也不证明 tokenizer 完整、显存足够、实际引擎设置正确或分数已经校准。报告的 `status` 是 `preflight_only`。正式启动时才由官方包加载 tokenizer、引擎和权重。

当前参考版本是 `1.1.0`，其他版本通过所需 API 检查时会警告而不是按版本白名单拒绝。缺包、缺选定后端依赖、模型目录错误会给出对应说明。

## 4. fake_ollama 管理启动

在你的配置中合并以下 `decision_targets` 项，并将路径替换为真实值。不要用它覆盖已有配置。

```json
{
  "decision_targets": [
    {
      "name": "startlux-local",
      "base_url": "http://127.0.0.1:8090",
      "runtime": "startlux",
      "runtime_backend": "vllm",
      "dialect": "systemone",
      "endpoint_path": "/v1/decisions",
      "auto_start": true,
      "python_path": "/absolute/path/to/startlux-venv/bin/python",
      "model_path": "/absolute/path/to/downloaded-model",
      "startup_timeout_seconds": 1800,
      "request_read_timeout_seconds": 300,
      "models": [{"name": "startlux-local"}],
      "start_args": ["--gpus", "0", "--max-model-len", "16384", "--no-images"],
      "decision_capabilities": {
        "question_types": ["predicate", "choice", "score"],
        "modalities": ["text"],
        "extensions": [],
        "think_modes": []
      },
      "source_url": "https://github.com/StartLuxLabs/StartLux-Decision",
      "license": "Official code: Apache-2.0; v1.1 weights: CC-BY-NC-4.0"
    }
  ]
}
```

此示例先声明基础文本能力。按你实际验证的模型/引擎扩展能力声明，尤其是图片、AutoThink 和供应商扩展；不要根据模型名字猜能力。官方引擎仍按模型目录决定默认推理方式。`models` 是路由用的名字，并不会选择或下载另一个模型。将决策模型暴露到需要的 API interface，调用决策端点；chat 路由不会替代决策协议。

客户端停止等待不代表官方服务停止推理。网关会在 `stop_timeout_seconds` 内继续收取已发送请求的响应并保留并发占用；响应丢失或等待到期时，运行时会进入待确认状态，阻止新请求。需确认自有进程树已停止，或由操作者确认外部服务完成/重启后再重启网关。冷启动被取消时，也会保留仍在加载的进程的资源预留。详见[生命周期与资源说明](decisions.md#lifecycle-and-resources)。

经官方模型元数据及运行时验证为 AutoThink 后，可设置 `extensions: ["think_mode"]` 和 `think_modes: ["never", "auto", "always"]` 来开放对应控制。Instant 最多声明 `think_modes: ["never"]`，不能开放 `auto` / `always`。去掉 `--no-images` 并验证实际引擎的图片路径后，才能在 `modalities` 中加入 `"image"`。官方 `/health` 的 Instant/图片禁用信息只用于进一步收紧配置能力，不会自动增加权限或功能。

- 管理启动只监听回环 HTTP 地址，端口从 `base_url` 取得；生成 argv 数组，不拼接 shell 命令。
- `runtime_backend=vllm` 默认交给官方 `--launch-vllm` 启动引擎，官方负责 Instant 插件和参数。也可明确在 `start_args` 提供 `--vllm-url http://engine:8000` 连接已有引擎，此时不会再自动启动该引擎；操作者负责它的版本、插件、模型和参数正确性。
- `start_args` 是官方 `serve` 的独立参数字符串，不是命令行文本。不能重写 model、backend、host、port 或 Python 身份；官方参数解析也禁用缩写绕过。
- 不要为伴生进程额外添加宽泛的 kill 命令。fake_ollama 的进程所有权、请求占用和回收机制负责自己启动的进程树；外部引擎的生命周期仍属于其操作者。

### 路径不是协议

官方 StartLux 的 `/v1/systemone` **与** `/v1/decisions` 都接受 `state` + `questions` 的 SystemOne body，因此两者都使用 `dialect: "systemone"`。不能仅凭 `/v1/decisions` 路径切到 OpenAI Decisions 的 `input` + typed questions 格式。

[官方 API 文档](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/api.md) 描述原生字段。`allow_none` 可在请求或 choice question 上设置；允许时 `choice` 可以为 null。保留 `none_probability`、`details` 等供应商数据；`none_probability` 是未校准的报警分数，不能当成统一置信度。跨协议无法表示的字段应显式拒绝，不可静默删除或改写。

## 5. 原生引擎与官方 wrapper 的版本区别

这些是 **2026-10-17 的上游证据**，不是对任意同名部署的保证：

- [llama.cpp v0.6.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.6.0) 于 2026-10-05 发布，包含原生 `/v1/systemone`。仍须确认模型类型及实际能力；旧版本、build 标签和定制 backport 需要探测。
- 原生 vLLM 的 [PR #59299](https://github.com/vllm-project/vllm/pull/59299) 于 2026-10-08 合入，比 [v0.31.0 stable](https://github.com/vllm-project/vllm/releases/tag/v0.31.0) 的 2026-10-05 发布晚。不能宣称原生 stable 0.31.0 自带该接口。采用包含相关提交的 main/nightly 时也要确认 endpoint、schema 和题型支持。
- **官方 StartLux wrapper** 是另一条路径：其参考插件已验证 vLLM `0.19.1` 和 `0.31.0`。它通过自己的服务器提供 SystemOne API，不依赖 vLLM 原生 `/v1/systemone`。[上游部署说明](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/deployment.md)

`verified_runtime_support` 返回带证据的 `supported` / `unsupported` / `unverified` 诊断。成功的决策协议能力探测可识别 backport，单个 health 200 或版本字符串不证明语义、题型、图片、校准或 Instant attention 正确。未知版本不会一律封禁。

## 6. llama.cpp / MLX 的官方包用法

内置管理启动目前覆盖官方 vLLM 与 transformers 后端。已有 llama.cpp/MLX 环境可运行官方服务，再配置 `runtime: "external"`、`auto_start: false`、`dialect: "systemone"` 指向它。也可手工运行 companion 文件的官方 `serve --backend llamacpp` / `serve --backend mlx`，保留本地模型和离线检查。必须自行启动和配置相应引擎。

### Instant + stock llama.cpp：不能只换个 GGUF

请按[上游 llama.cpp 后端说明](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/startlux_decide/backends/llamacpp.py)和[补丁说明](https://github.com/StartLuxLabs/StartLux-Decision/tree/main/patches/llama.cpp)显式配置：

1. 使用声明 non-causal attention 的 Instant GGUF，或适用于该架构的 `--override-kv qwen35.attention.causal=bool:false`。
2. stock server 使用 `--parallel 1 --ctx-checkpoints 0`，`-b` 和 `-ub` 至少覆盖最长完整 prompt。不要启用会改变 read-out 的 GPU sampling；不要让 prompt 被切分。
3. stock server 在官方 wrapper 侧明确设置 `STARTLUX_LLAMACPP_BIDIR=server`。默认 `auto` 对不报告 whole-prompt 支持的 server 会选择 per-request 模式；stock server 不支持该请求标记，不能假定会正确工作。
4. 图片需要匹配的 `mmproj`，并且 Instant 的文本和图片 embeddings 必须在同一 micro-batch 中读取。仅有 `mmproj` 不够：上游指出 stock server 仍可能拆开文本/图片；官方 patch 0003 通过 `/props` 的 `mtmd_prompt_whole_when_non_causal` 声明完整混合 prompt 能力。未验证时保持 `--no-images` 与文本能力声明。

本项目不会自动打补丁、转换 GGUF、强行声明正确 attention 或选择会悄悄改变模型行为的 stock 启动参数。

## 7. 许可与归属

- [StartLux 官方代码](https://github.com/StartLuxLabs/StartLux-Decision)及其 [LICENSE](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/LICENSE)：Apache-2.0。
- 此处讨论的 **StartLux-Decision v1.1 权重**按[上游许可说明](https://github.com/StartLuxLabs/StartLux-Decision#license)为 CC BY-NC 4.0，含非商业限制；商用授权请联系 StartLux。不同模型/revision 应重新核对其模型卡，不从代码许可证推断权重许可。
- 本集成不重新授权权重、不代替操作者接受许可、不附带任何权重。操作者负责用途、归属和许可合规。
- 归属：StartLux 与论文作者 Xialie Zhuang、Zunhai Su、Jia Pan、Ngai Wong、Guanhua Chen、Chuan-Wei Kuo；[技术报告](https://arxiv.org/abs/2610.17309)。

## 离线验证范围

```sh
python -m pytest tests/unit/test_startlux_launch.py
```

这些测试使用临时元数据和 mock 官方包/引擎版本，检查依赖隔离、路径与参数边界、预检及 CLI 委托。它们不安装官方包、不下载权重、不启动 GPU 进程，也不声称已经验证真实模型准确率、图片、AutoThink 或部署性能。
