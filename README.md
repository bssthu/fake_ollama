# fake_ollama

一个轻量的协议适配层，主要做两件事：

1. **正向**（Ollama 兼容入口 → 远端上游）：把 **Anthropic Messages API** 或 **OpenAI Chat Completions API** 兼容的上游（官方 / DeepSeek / Together / Groq / 自建网关 / claude-relay-service）伪装成一台本机 **Ollama** 服务，让只支持 Ollama 协议的客户端（GitHub Copilot 自定义 provider、IDE 插件、桌面 AI 软件）无缝调用 Claude / DeepSeek / GPT 等模型。
2. **反向**（Anthropic / OpenAI 兼容入口 → 本机模型服务）：把本机的 **Ollama**、**llama.cpp server** 或 **ComfyUI workflow** 包装成 Anthropic / OpenAI 兼容 API（含 `POST /v1/messages`、`POST /v1/chat/completions`、`POST /v1/images/generations`、`POST /v1/images/edits`），让只支持远端 API 的客户端也能调用本地模型。

附带一个零依赖的 Web 配置编辑器（`/admin`），以及可选的轻量模型调试页（`/playground`）。

## 架构概览

监听分为 Admin、Dashboard、Playground、Ollama 兼容入口和 API 接口五类。前三类各有独立端口，后两类支持多实例数组。Admin / Dashboard 共用独立的管理凭据；模型接口各自配置白名单和 token，Playground 按所选接口的权限提供模型调试。

```
┌──────────────────────────────────────────────────────────────────┐
│  fake_ollama 进程（asyncio.gather 同时跑多个 uvicorn）           │
│                                                                  │
│  admin listener          (默认 127.0.0.1:21433)                  │
│    /admin/*          Web 配置编辑器（独立管理凭据）              │
│                                                                  │
│  dashboard listener      (默认 127.0.0.1:21432)                  │
│    /dashboard/*      资源监控与回收（管理凭据）                  │
│                                                                  │
│  playground listener     (默认关闭；启用后 127.0.0.1:21431)     │
│    /playground/*、/v1/*   模型调试与调用                         │
│                                                                  │
│  ollama_interfaces[*]    （数组，可 0..N 个；典型: 21434）       │
│    /                 ping                                        │
│    /api/*            Ollama 兼容（正向）                         │
│    每个实例可独立配置 host / port / access_tokens / exposed_models│
│                                                                  │
│  api_interfaces[*]       （数组，可 0..N 个；典型: 21435）       │
│    /v1/messages      Anthropic 兼容                              │
│    /v1/messages/count_tokens                                     │
│    /v1/chat/completions   OpenAI 兼容                            │
│    /v1/images/generations OpenAI Images 兼容                      │
│    /v1/images/edits       OpenAI Images 兼容                      │
│    /v1/videos/generations 视频生成                               │
│    /v1/systemone、/v1/decisions  类型化决策                       │
│    /v1/embeddings    暂未实现，返回 501                          │
│    /v1/models                                                    │
│    每个实例可独立配置 host / port / access_tokens / exposed_models│
└──────────────────────────────────────────────────────────────────┘
```

- 每个 `ollama_interfaces[*]` / `api_interfaces[*]` 都是独立的 uvicorn 监听。你可以同时开多个 Ollama 接口（例如本机 21434 + 局域网 0.0.0.0:21437），每个挂不同的白名单和 token，端口不能重复。
- `access_tokens` 为空 = 模型请求无需 token，但仍校验 Host、实际端口及浏览器来源；非空 = 模型请求必须带 `x-api-key` 或 `Authorization: Bearer`。管理凭据独立配置，见[安全提示](#安全提示)。
- `exposed_models` 是该接口允许看到 / 请求的模型列表，元素是 `{model, target, alias?}` 三元组；不在白名单里的模型在该接口上 404。
- admin 接口（`admin_host` / `admin_port`）只服务 `/admin/*`，**不会**出现在任何 `ollama_interfaces` / `api_interfaces` 上；同样，`/api/*` 与 `/v1/*` 也**不会**出现在 admin 端口。
- Embeddings 尚未实现：`/api/embed`、`/api/embeddings` 和 `/v1/embeddings` 返回 501，不会转发到后端。
- Model Playground 默认关闭。启用后使用独立端口，按 `/playground/api/models` 返回的 capabilities/operations 调用聊天、视觉、视频分析、图片生成/编辑与视频生成接口；支持图片粘贴/拖放、本地视频选择，以及手机或 PC 摄像头的分窗近实时分析。API key 与交互记录仅保存在当前浏览器页面内存中。
- 可选 [OCR 功能](services/ocr/README.md) 已并入 `/playground/` 主页面：支持 MiniCPM-OCR、PP-OCR、HunyuanOCR 对比，可选择 Jev 区域复核和最多一次低分重试，并提供模型保留/释放与连续识别计时。通过 `playground_ocr_config` 指向隔离运行环境的本机配置；[调研与实测结果](docs/ocr-local-research.md) 包含 Hunyuan 旧 GGUF 的 EOS/EOT 问题与修复。
- 想让别的机器访问某个接口：把对应实例的 `host` 改成 `0.0.0.0`，或者保持 `127.0.0.1` + 在前面挂 Nginx/Caddy（推荐，可以加 TLS / 限流 / 客户端证书）。

## 特性一览

- **多上游路由**：把 Anthropic / OpenAI 兼容（DeepSeek / Together / Groq / 自建网关 / claude-relay-service…）合并到同一组接口
- **结构化模型标识 `{model, target}`**：每条暴露条目显式声明「上游显示名 + 来源 target 名」，相同名字但来源不同的模型不会再混淆；可选 `alias` 字段在客户端可见的公开 ID 上替换全名
- **每模型 profile**：capabilities / 上下文长度 / 思维链开关 / 输出上限——key 支持 `model@target`（最优先）、裸 `model`、tagless base 三级回退
- **按接口多实例暴露**：`ollama_interfaces[*]` 与 `api_interfaces[*]` 是独立数组，每个实例自带 `host` / `port` / `access_tokens` / `exposed_models`
- **来源命名清晰区分**：远端 `anthropic_upstreams` / `openai_upstreams`，本地 `ollama_targets` / `llama_cpp_targets` / `generic_openai_targets` / `comfyui_targets`，以及独立协议的 `decision_targets`
- **循环引用检测**：按模型转发图拒绝实际循环；允许最终落到外部上游的无环回指，并发出提示
- **本地 target 生命周期接管**：Ollama / llama.cpp / ComfyUI 都可配置 health check、按需启动脚本、启动超时、空闲回收
- **H3 Context-IR harness**：把自然语言和最多两张参考图转换成 MiniMax H3 Base Prompt；规划器可在本地纯文字/多模态模型与第三方 API 之间切换，并可串到 ComfyUI 视频生成之前
- **ComfyUI 图片后端**：可把 Z-Image-Turbo / Qwen-Image-Edit / SenseNova-U1 等 ComfyUI API workflow 暴露为 OpenAI 兼容图片生成 / 图片编辑接口。采用声明式 `preset` + bindings 结构，接入新模型只改配置 + JSON；多个图片模型可共用一个 ComfyUI 实例并由 VRAM 协调器互斥换出，避免爆显存。
- **本地显存治理**：`estimated_vram_gb` 控制模型加载准入；媒体模型还可配置请求瞬时余量、安全下限、自适应 GPU 权重卸载和全程独占租约。相同模型连续运行也会重新检查实时显存
- **本地内存预检**：部分模型（如 SenseNova / JoyAI 这类会把一部分计算 offload 到内存的 workflow）除显存外还需占用大量主机内存，可在 `model_profiles` 里填 `estimated_memory_gb`；逻辑与显存预检一致，启动前评估可用系统内存并尝试回收空闲模型
- **图片输入**：自动嗅探 base64 magic bytes（PNG/JPEG/GIF/WEBP）
- **无需额外前端依赖的 Web 编辑器**：按模型来源、接口、运行时和管理页面分组，提供字段说明、默认值回退、模型探测和 profile 自动补全
- **明确的失败状态**：连接失败 / 超时返回 502；响应已经开始时发送协议错误帧。上游错误或缺少结束标志不会伪装成正常完成；上游 HTTP 错误保留状态码。
- `pytest` + `httpx.MockTransport` 离线单测

## 热重载与运行时安全

Admin 的 Save & Reload 返回 `saved`、`applied`、`restart_required` 和配置代次。保存使用同目录临时文件、刷新落盘和原子替换；准备资源或替换文件失败时，旧文件和旧运行配置继续使用。请求从开始到流结束固定使用同一代配置与客户端，无关配置保存会复用原有连接。

| 变更 | 生效方式 |
| --- | --- |
| 模型白名单、profile、上游 URL/token、超时、系统代理 | 新请求使用新代配置；已有请求继续使用旧代 |
| 监听 host、port、接口名称/角色、新增监听 | 需重启才能绑定新 socket；不能通过保存把旧公网监听变成本机免鉴权 |
| 移除或改动模型接口端口 | 旧端口立即拒绝模型请求；新端口在重启后监听 |
| Admin/Dashboard 端口 | 重启前保留实际管理端口；Playground 改端口则关闭旧路由 |
| 相同端点的本地后端换代 | 旧代请求结束且清理完成前，新代返回 503，避免重叠启动 |

旧代长流不会被强行截断；30 秒后退休状态显示 `waiting_for_requests`。一批关闭操作最多等待 30 秒，未完成或失败的清理继续被跟踪。管理鉴权保护的 `GET /admin/runtime` 可查看状态，`POST /admin/retry-cleanup` 可重试失败的关闭。

本地服务区分外部实例、本进程启动的实例和明确授权管理的外部实例。单独配置 `stop_command` 不代表拥有服务；若需要管理外部服务，显式设置 target 的 `manage_external_runtime: true`，且仅在实际使用后才执行停止。`stop_timeout_seconds` 默认 30 秒（大于 0、最多 300）。同端点及 ComfyUI `runtime_group` 的客户端共享所有权；关闭一个客户端会保留其他客户端仍在使用的服务。停止失败保留资源用于恢复。Windows 使用 Job Object 跟踪子进程，包括启动器提前退出的情况；不会按进程名全局终止服务。

ComfyUI 请求取消、超时或提交应答丢失后，会核查 `/history` 和 `/queue`。只删除本请求仍在队列中的 prompt；正在运行的 prompt 等待确认完成，不调用全局 interrupt。无法确认时保留 prompt 记录并隔离 runtime/GPU，拒绝冲突新工作。可通过 Dashboard 状态及 `POST /admin/recover-runtime`（`{"target":"配置中的名称"}`）观察、重试恢复；无法访问或历史已被清理时，不会猜测任务已停止。

匿名 Playground 同时检查实际 loopback 监听、连接来源、Host 端口与同源 Origin，忽略伪造的 forwarded 头。合法反代可同时配置 `playground_trusted_proxy_ips`（直接代理 IP）和 `playground_allowed_origins`（完整 http(s) origin，无路径）；代理需保留公开 Host，页面导航不带 Origin 时仍检查该 Host 是否属于允许的 origin。内置启动器关闭 Uvicorn 的代理头改写；自行部署 ASGI 时也应使用 `--no-proxy-headers`，保留直接连接来源。公网监听默认仍需模型接口 token；修改 host 为 `127.0.0.1` 不能降低旧 socket 的鉴权要求。

Thinking 回查仅匹配同一凭据主体、后端、上游模型和完整规范化历史中的确切 assistant 轮次；不会用“最近一条”或相似文本补齐。明确提供的 reasoning 优先。缓存最多 512 项、保留 15 分钟，重载即失效。Anthropic 来源默认 `allow_unsigned_thinking: false`、`missing_thinking_policy: "omit"`；需要无签名 reasoning 的兼容端应显式开启前者，也可将缺失策略设为 `"error"`。网关保留收到的 signature，但不会伪造 signature。

整改证据、逐阶段测试和残余边界见 [整改清单](docs/remediation.md)。默认测试排除 integration/e2e/gpu；Windows 隔离进程测试另行运行。真实 GPU、付费上游、POSIX 进程树及本轮浏览器视觉验收未执行。

## 快速开始

```powershell
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -e ".[test]"

# 2. 准备配置
Copy-Item config.json.example config.json
# 编辑 config.json 填上游与监听

# 3. 启动
python -m fake_ollama
# 未激活 .venv 时
.\.venv\Scripts\python.exe -m fake_ollama
# 已 pip install -e .
fake-ollama
# 仅覆盖 admin listener
python -m fake_ollama --config ./config.json --admin-host 127.0.0.1 --admin-port 21433
```

> 注：CLI 不再有 `--host` / `--port` / `--external-host` / `--external-port`，因为 `ollama_interfaces` / `api_interfaces` 是数组，命令行无法表达；都在 `config.json` 里配置。

完整的命令行参数、日志选项和重启规则见 [CLI 使用说明](docs/cli.md)。示例中的上游模型名和凭据都是占位值，须换成实际可用的模型和 token。

启动后：
- Ollama 客户端连 `ollama_interfaces[*].host:port`（典型 `http://127.0.0.1:21434`）
- 浏览器打开 <http://127.0.0.1:21433/admin>
- Dashboard 默认位于 <http://127.0.0.1:21432/dashboard/>；Playground 需先启用，再打开 <http://127.0.0.1:21431/playground/>
- Anthropic / OpenAI 客户端连 `api_interfaces[*].host:port`（典型 `http://127.0.0.1:21435`），带对应 `access_tokens`

## 配置（config.json）

加载顺序（后者覆盖前者）：

1. 代码默认值
2. `config.json`（路径优先级：`--config` > `$FAKE_OLLAMA_CONFIG` > `./config.json`）
3. CLI 明确传入的 Admin / Dashboard / Playground 监听覆盖值（命令本身不写回文件，后续 Admin 保存以提交值为准）

主服务配置加载器通过 `FAKE_OLLAMA_CONFIG` 选择 JSON 文件，不从环境变量读取接口 token 或模型白名单，也不自动加载 `.env`。模型接口 host / port 在 JSON 中设置；管理页面监听可由上述 CLI 参数覆盖。伴生服务和显式启用的系统代理有各自的环境变量，见对应文档。

完整示例见 [config.json.example](./config.json.example)。下面分章节描述每块。

### 顶层结构

```jsonc
{
  // Admin UI
  "admin_enabled": true,
  "admin_host": "127.0.0.1",
  "admin_port": 21433,

  // Dashboard
  "dashboard_enabled": true,
  "dashboard_host": "127.0.0.1",
  "dashboard_port": 21432,
  // ... 见下

  // 轻量模型调试页（修改监听设置后重启进程）
  "playground_enabled": false,
  "playground_host": "127.0.0.1",
  "playground_port": 21431,

  // 共享设置
  "default_max_tokens": 4096,
  "timeout_seconds": 300.0,
  "use_system_proxy": false,
  "enforce_context_limit": true,
  "advertised_version": "0.6.4",

  // 远端上游
  "anthropic_upstreams": [ /* … */ ],
  "openai_upstreams":    [ /* … */ ],

  // 本机 backends
  "ollama_targets":      [ /* … */ ],
  "generic_openai_targets": [ /* … */ ],
  "decision_targets":    [ /* … */ ],
  "llama_cpp_defaults":  { /* … */ },
  "llama_cpp_targets":   [ /* … */ ],
  "comfyui_targets":     [ /* … */ ],
  "h3_context_ir_profiles": [ /* … */ ],

  // 对外接口（每项独立监听）
  "ollama_interfaces":   [ /* … */ ],
  "api_interfaces":      [ /* … */ ],

  // 每模型 profile
  "model_profiles":      [ /* … */ ]
}
```

### Sources（远端与本机的模型来源）

共有下表七类 source，`name` 在所有来源表之间唯一且不能包含 `@`。多模型来源使用 `models` 列表，每项包含 `name`、可选的源级 `alias` 和可选的 `upstream_id`。llama.cpp / ComfyUI 单模型 target 使用 `model` / `alias` 表达同样的显示关系。

```jsonc
{
  "name": "deepseek_remote",
  "base_url": "https://api.deepseek.com",
  "auth_token": "sk-...",
  "models": [
    { "name": "deepseek-chat" },                            // 暴露后默认公开 ID = "deepseek-chat@deepseek_remote"
    { "name": "deepseek-reasoner", "alias": "deepseek-r1" } // 源级显示名是 deepseek-r1；wire 仍是 deepseek-reasoner
  ]
}
```

- **源级显示名**：`ModelEntry.alias` 有值时使用它，否则使用 `name`；`exposed_models[*].model` 和 profile 引用这个显示名。显示名在同一来源内唯一，可以在不同来源重复。
- **公开 ID**：由接口 exposure 决定，默认为 `显示名@target`；只有 `exposed_models[*].alias` 才会把整个公开 ID 替换成简短别名。公开 ID 在同一接口内唯一。例如上述第二项默认是 `deepseek-r1@deepseek_remote`；要用裸 `deepseek-r1` 请求，还需在 exposure 设置同名 alias。
- **wire 名**：实际发给上游的是 `ModelEntry.upstream_id`，未设置时使用 `name`；源级和接口级 alias 都不改写 wire 名。
- **`model_map` 已移除**：使用 `models` 中的 `name` / `alias` / `upstream_id` 配置显示名与上游 ID。

各来源表的用途：

| 表名 | 协议 | 用途 |
| --- | --- | --- |
| `anthropic_upstreams` | Anthropic Messages | 远端 Anthropic 兼容 API（含 DeepSeek Anthropic 通道、claude-relay-service 等） |
| `openai_upstreams`    | OpenAI Chat Completions | 远端 OpenAI 兼容 API（DeepSeek OpenAI 通道、Together、Groq、自建网关等） |
| `ollama_targets`      | Ollama `/api/chat` | 本机或局域网 Ollama daemon；可生命周期接管 |
| `generic_openai_targets` | OpenAI Chat Completions | 带生命周期管理的 vLLM、SGLang、TGI 或 Mage-VL 等 adapter |
| `decision_targets`   | SystemOne / OpenAI Decisions | 原生类型化决策服务；独立于 Chat Completions |
| `llama_cpp_targets`   | OpenAI Chat Completions（llama.cpp server） | 一项 = 一个 llama.cpp server 进程 / 一个模型 / 一个端口 |
| `comfyui_targets`     | 图片 / 视频生成（ComfyUI workflow） | 一个模型 workflow target；多个 target 可以共用同一 ComfyUI 服务 |

#### anthropic_upstreams

```jsonc
{
  "name": "default",
  "base_url": "https://api.anthropic.com",
  "auth_token": "sk-ant-...",
  "models": [
    { "name": "claude-3-5-sonnet-20241022" },
    { "name": "claude-3-5-haiku-20241022", "alias": "claude-haiku" }
  ]
}
```

启动时会做基于「模型转发图」的循环检测，而不再只看 host:port：

- **case 1（允许，仅 WARNING）**：`base_url` 回指本进程的某个监听端口，但每一跳的模型名（alias）都不同，整条链路最终能落到一个真实的外部 upstream，不会无限递归。日志里会打印 `self-referential upstream is linear (no cycle): ...`，方便你确认这是有意为之。
- **case 2（拒绝启动）**：链路上至少有一跳重用了同一个公开模型名，导致请求会无限回到自己。会抛 `cycle detected in model-forwarding graph: ...`。Ollama 的 `:latest` 标签会被归一化，`qwen3` 与 `qwen3:latest` 视为同名。
- **管理端口误配（拒绝启动）**：如果 `base_url` 指向了 `admin` / dashboard 监听（不服务模型流量），直接报错 `cycle detected ... admin/dashboard listener`。

除了启动时的静态检测，运行时所有上游请求都会带上 `x-fake-ollama-forwarded-by` 头（包含本进程的随机 `INSTANCE_ID`）。如果某次请求绕一圈又回到自己（动态 DNS、反向代理改写等绕过静态检测的情况），中间件会直接返回 HTTP 508 `Loop Detected`，避免雪崩。

#### openai_upstreams

字段同 `anthropic_upstreams`，协议是 OpenAI Chat Completions（会自动拼 `/v1/chat/completions`），鉴权同时以 `Authorization: Bearer` 和 `x-api-key` 发送。

#### ollama_targets

```jsonc
{
  "name": "local",
  "base_url": "http://127.0.0.1:11434",
  "models": [
    { "name": "llama3.1:8b", "alias": "llama3.1" },
    { "name": "qwen2.5-coder" }
  ],

  "auto_start": false,
  "start_command": null,
  "stop_command": null,
  "idle_timeout_seconds": null,
  "startup_timeout_seconds": 60,
  "health_path": "/api/version",
  "cwd": null
}
```

- `auto_start: false`：fake-ollama 不自动启动 daemon。是否允许停止已存在的外部实例，另由 `manage_external_runtime` 与实际使用 / 所有权决定。
- `auto_start: true`：health check 失败时执行 `start_command`；`stop_command` / `idle_timeout_seconds` 控制具有所有权的实例回收。管理外部实例还需 `manage_external_runtime: true`；停止命令受 `stop_timeout_seconds` 限制。
- 模型名匹配按 Ollama 规则：`foo` 与 `foo:latest` 等价；省略 tag 只代表 `latest`。
- 每个本地模型的显存估算在 `model_profiles[*].estimated_vram_gb` 里配置。请求触发模型加载前，fake-ollama 调用 `nvidia-smi` 检查可用显存；不足时按 LRU 卸载其他空闲超过 60 秒的本地模型，每轮重新读真实 free VRAM。
- 媒体模型可再配置 `request_vram_headroom_gb`（已驻留模型的单次推理瞬时余量）、`min_free_vram_gb`（请求准入及运行观测下限）、`vram_cleanup_policy=keep|adaptive|unload` 和 `exclusive_gpu`。余量会按实际分辨率、视频帧数及 workflow 原生 batch 放大；非原生 batch 的串行循环不会重复放大。运行中会持续记录最低空闲显存；越过观测下限时不抢在 ComfyUI DynamicVRAM 分配器之前中断健康 prompt，而是在当前任务完成后清理运行时。真实 CUDA OOM 或 ComfyUI workflow 错误仍会使请求失败。
- ComfyUI target 默认以 `gpu_device + base_url` 作为运行时分组，也可用 `vram_runtime_group` 显式覆盖。组内 `/free`、进程退出和健康检查失败会使所有 sibling target 的显存/内存 reservation 一起失效。
- 多 GPU 主机不会把各卡空闲显存相加后做准入；默认协调器使用最紧张设备的读数并采用全局保守租约，避免“总空闲够、单卡放不下”仍启动任务。`gpu_device` 当前用于隔离运行时状态，不负责跨卡调度。
- 主机内存估算在 `model_profiles[*].estimated_memory_gb` 里配置，由独立的内存协调器按同样逻辑准入与回收：加载前读取系统可用内存，不足时卸载其他声明了 `estimated_memory_gb` 的空闲模型。Dashboard 的「Model Estimated Memory」面板与 Current Models 表里的 Est. Memory 列展示该维度。

#### generic_openai_targets 与 decision_targets

`generic_openai_targets` 用于带生命周期管理的 OpenAI-compatible 聊天 / 多模态 adapter。每项至少设置 `name`、`base_url` 和 `models`；`auth_token` 是网关到该 adapter 的出站凭据，与接口的 `access_tokens` 分开。

可配置 `auto_start`、`start_command`、`stop_command`、`cwd` 和空闲回收；默认 `startup_timeout_seconds=120`、`health_path="/health"`、`stop_timeout_seconds=30`。`max_concurrent_requests` 大于 0 时启用网关队列；`request_read_timeout_seconds` 留空沿用全局超时，非正值关闭出站读取超时。服务所有权仍遵循[生命周期规则](#热重载与运行时安全)。

Mage-VL 的安装、路径、请求大小限制及自动停机鉴权见[伴生服务说明](services/mage_vl/README.md)。将 [Windows 本地配置示例](config/examples/config.windows-local.json) 中相关 target、profile 和 exposure 合并进自己的配置，替换路径和凭据；最小 `config.json.example` 不包含本地模型服务。

`decision_targets` 使用独立的 `/v1/systemone` 或 `/v1/decisions` 协议，不能通过聊天端点调用。配置、原生运行时能力和生命周期边界见[决策模型说明](docs/decisions.md)；[StartLux Windows 示例](docs/decision-startlux-windows.md)和 [Laya 伴生服务](services/laya/README.md)分别说明其部署方式。

#### llama_cpp_targets

llama.cpp server 进程模型：**一个 target = 一个模型 = 一个端口**。要跑多个模型，加多个 target。

`llama_cpp_defaults` 集中配置所有 target 共用的默认值；target 自身字段可覆盖。

```jsonc
{
  "llama_cpp_defaults": {
    "auto_start": true,
    "idle_timeout_seconds": 1800,
    "startup_timeout_seconds": 600,
    "health_path": "/health",
    "cwd": "C:\\path\\to\\llama.cpp",
    "binary_path": "C:\\path\\to\\llama.cpp\\llama-server.exe",
    "gpu_layers": 99,
    "flash_attn": true
  },
  "llama_cpp_targets": [
    {
      "name": "qwen3.6-27b-iq4xs",
      "base_url": "http://127.0.0.1:21441",
      "model": "qwen3.6-27b-iq4xs",
      "alias": null,
      "model_path": "I:\\path\\to\\qwen.gguf",
      "mmproj_path": "I:\\path\\to\\mmproj.gguf",
      "ctx_size": 120000,
      "cache_type_k": "q8_0",
      "cache_type_v": "q8_0"
    }
  ]
}
```

`LlamaCppTarget.alias` 是源级显示名，替换白名单和 profile 引用中的 `model`；公开 ID 仍由 exposure 决定。发给 llama.cpp 的 wire 名为 `upstream_id`，未设置时使用 `model`。

#### comfyui_targets

LLaDA-Image Base / Turbo 的社区 INT8 ComfyUI 接入见 [配置说明](config/examples/comfyui/llada_image.md) 和 [双模型配置样例](config/examples/comfyui/llada_image.json)。两版均支持 Playground 文生图、单参考图编辑和真实节点参数调整。

视频 workflow 既可以使用内置 `joyai_echo` preset，也可以使用
`custom` / `comfyui_api` + `video_workflow_path` / `image_to_video_workflow_path`。
请求字段分别绑定在 `bindings.video`（文生视频）和 `bindings.i2v`（图生视频）；
图片可绑定到 `image`、`images` 或 `image_1`...`image_4`。请求附带图片且存在
I2V workflow 时自动走 I2V，否则走 T2V。

参数发现不另建一套容易漂移的模型配置：`WorkflowSpec.bindings` 仍是执行的事实来源，
服务端据此生成 `/playground/api/models` 中的 `operations[*].parameters`、默认值、范围、参考图上限和
可选 presets。Playground 动态渲染这些描述，因此 workflow 未绑定的参数不会显示，
也不会出现界面填写了 Steps/CFG、实际节点却没有收到的情况。JoyAI 的
`enable_tile` / `enable_streaming` 这类非 OpenAI 标准字段也沿同一条链路校验并写入节点。

For ComfyUI nodes that accept multiple references as one `IMAGE` batch (for
example JoyAI Echo's local ComfyUI node), bind `images` to the consumer IMAGE
input that is already connected to a `LoadImage` node. fake_ollama will upload
all reference files, clone the `LoadImage` node as needed, insert core
`ImageBatch` nodes, and feed the resulting batch into that input. Use
`max_reference_images` to cap the number of uploaded references for a target.

ComfyUI workflow 图片模型：一个 target 负责一个公开图片模型名，fake_ollama 通过 ComfyUI HTTP API 提交 workflow、轮询 history、读取 output 图片，并把结果包装成 OpenAI Images 兼容响应。

每个 target 用 **`preset`** 选择工作流形态（声明式绑定，新增模型只改配置 + JSON、不动代码）：

| preset | 适用模型 | 加载方式 / 特点 |
|---|---|---|
| `z_image_turbo`（默认） | Z-Image-Turbo | 独立 UNET/CLIP/VAE + KSampler；沿用 `diffusion_model` / `text_encoder_model` / `vae_model` / 各 `*_node_id` 字段 |
| `qwen_image_edit_aio` | Qwen-Image-Edit 整合检查点（如 Qwen-Rapid-AIO） | `CheckpointLoaderSimple` 一次加载 UNet+CLIP+VAE；文生图用 `CLIPTextEncode`，图生图用 `TextEncodeQwenImageEditPlus`（参考图经 conditioning 注入，编辑 `denoise=1.0`） |
| `sensenova_u1` | SenseNova-U1-8B（自定义节点 `ComfyUI_SenseNova_U1`） | 融合模型节点 + 融合采样节点；分辨率走 `target_pixels` 比例桶（请求 `size` 映射到最近比例，实际输出为该桶原生尺寸） |
| `joyai_echo` | JoyAI Echo（自定义节点 `ComfyUI_JoyAI_Echo`） | 内置 T2V + 多参考图 I2V；公开尺寸、seed、帧数、FPS、prefetch、分块解码与模型流式加载；默认使用 256² / 17 帧安全调试预设 |
| `minimax_h3` | MiniMax H3 Base / Ref2VA | 内置 T2V、首帧 I2V、首尾帧 FL2VA、末帧 L2VA 和 Ref2VA；Base / Ref2VA 使用各自的 Pruned INT8 ConvRot DiT，共用 Qwen3-VL-32B NVFP4-AWQ |

当前内置模型的推荐公开参数如下；表中默认值来自对应 target 配置，因而 API 缺省行为、
`/playground/api/models` 和 Playground 使用同一个值：

| 模型 | Playground / API 公开参数 | 推荐默认值 |
|---|---|---|
| Z-Image-Turbo | `size`、`n`、`seed`、`steps`、`cfg`、`sampler_name`、`scheduler`；编辑另有 `denoise` | 1024²、8 steps、CFG 1、`res_multistep` / `simple`；编辑 denoise 0.25 |
| Qwen-Image-Edit AIO | 同 Z-Image；编辑只接收 1 张参考图 | 1024²、6 steps、CFG 1、`euler_ancestral` / `beta`；编辑 denoise 1.0 |
| SenseNova-U1 | 原生宽高比桶、`n`、`seed`、`steps`、`cfg`；不显示 workflow 没有输入的 sampler/scheduler/denoise | 1:1、8 steps、CFG 1；实际像素由模型的原生比例桶决定 |
| JoyAI Echo | `size`、`seed`、`num_frames`、`fps`、`prefetch_count`、`enable_tile`、`enable_streaming` | 安全调试：256²、17 帧、8 FPS、prefetch 1、分块解码开；帧数必须为 `8k+1`，尺寸至少 256 且为 32 的倍数 |
| MiniMax H3 | `size`、`seed`、`steps`、`sampler_name`、`scheduler`、`num_frames`、`fps`，以及 H3 Context-IR 的模式/Planner 参数 | 1344×768、124 帧、24 FPS、20 steps、`res_multistep` / `simple`；帧数必须为 `17k+5` |

Playground 的图片生成、图片编辑和视频生成另有 **生成份数**（1～20）控制。它不会放大
单次 API 请求的 `n`，而是把每份作为独立的 `n=1` 请求串行排队：前一份完成并立即展示后，
自动重跑下一份；停止或中途失败时保留已经完成的结果。显式 Seed 会按份数依次加一；未填
Seed 时仍由 target 的 `seed_mode` 决定。H3 `prompt_mode=auto` 若首份调用 Planner 返回了
`revised_prompt`，后续份数会复用该结构化 Prompt 并以 `raw` 直通，避免重复加载 Planner；
`prompt_mode=enhance` 则仍按用户选择，每份都重新增强。标准 API 客户端直接传 `n` 的行为不变。

> **JoyAI Echo 节点兼容性**：已在本机验证的 `ComfyUI_JoyAI_Echo` 版本要求节点把
> `enable_tile` 写入模型实际读取的 `enable_tiles` 属性，并要求 VAE wrapper 将
> `tiled_decode()` 返回的分块迭代器沿时间维拼接后再进入后处理。旧版节点若未包含这两项
> 修复，会分别表现为勾选分块解码仍发生显存不足，或报
> `unsupported operand type(s) for +: 'generator' and 'int'`。

图片 preset 对应 `fake_ollama/workflows/<preset>_t2i.json` / `<preset>_i2i.json`，视频则对应 T2V / I2V / FL2VA / L2VA / Ref2VA workflow。要接入**其它** ComfyUI 模型，提供 API 格式 workflow JSON，再用 `bindings`（逻辑参数 → `[{node, input}]`）和 `static_inputs`（固定值，如模型文件名）声明落点即可；支持 `t2i`、`i2i`、`video`、`i2v`、`fl2va`、`l2va`、`ref2va` 七种绑定模式。

> **客户端可能按模型名判断是否图片模型**：部分客户端（如 CherryStudio）靠模型 id 的正则/子串匹配来决定走聊天接口还是 `/v1/images/*`——只认 `z-image*` / `qwen-image*` / `flux*` / `sd*` 等已知图片模型名。若把图片模型暴露成它不认识的名字（如 `sensenova`），它会当普通聊天模型发到 `/v1/chat/completions`，被 fake_ollama 以 400「use /v1/images/...」拒绝。对策：在 `exposed_models[*].alias` 里给这类模型起一个**包含已知图片模型关键词的别名**（例如 `sensenova-z-image`）；这只改客户端看到的 id，后端 target 与 `model_profiles` 仍按真实模型名匹配，不受影响。

> **多模型共用一个 ComfyUI 实例 + 显存互斥**：把多个 target 的 `base_url` / `start_command` 指向同一个 ComfyUI（如下例三个图片模型共用 `:21480`）。fake_ollama 会按 `GPU + base_url` 自动归并运行时状态：任一 target 卸载模型时会同步失效整组 reservation；每次推理持有全程执行租约，并按模型加载预算、请求瞬时余量和安全下限重新读取真实 free VRAM。显存不足时 `adaptive` 策略先卸载 GPU 权重但保留 ComfyUI 对象缓存，仍不足则在提交 `/prompt` 前返回 503。ComfyUI 的自动 offload 仍会使用，但不再被当作防止 OOM 的唯一保障。

> **24GB 卡（RTX 4090）实测性能与显存提示**：
> - **Qwen-Image-Edit AIO（fp8 28GB）**：显存**腾干净**时 DiT 完全驻留（文本编码器一次性放 CPU），文生图 ~5s/张（冷加载 ~25s）、图生图 ~15s/张；但若显存被占（DiT 放不下）会退化为**逐步从内存流式加载，单步从 ~0.7s 涨到 ~18s（约 110s/张），非常卡**。所以把它的 `estimated_vram_gb` 设得较高（20），让协调器先把 GPU 腾出来，确保驻留跑得快。本地只有 28GB fp8 整合检查点；想更省显存需另找更小的量化版本（如 Q4 GGUF 的 DiT ~12GB）。
> - **SenseNova-U1-8B**：原生只在 ~4MP（如 1:1=2048²）出图。GGUF 加载时会**反量化成 bf16（~16GB）**。`prefetch_count=0`（整模常驻）实测峰值约 24GB、刚好顶满 24GB 卡 → 触发 WDDM 往内存分页，单张 >400s 奇慢，**等同不可用**。故工作流用 `prefetch_count>=1` 的层流式（一次仅驻留 1 层，峰值仅 ~8-9GB，其中 ~6GB 是 2048² 激活），不占满显存；代价是每步都要把 ~16GB 主干从内存经 PCIe 搬一遍，较慢（~100s/张）。**注意**：当前节点实际用的 `SimpleLayerStreamingWrapper` 会**忽略 `prefetch_count` 的具体数值**（同步逐层加载、无异步预取），所以填 1 / 2 / 5 没有任何区别。`estimated_vram_gb` 按层流式实际峰值填（~13 留足余量即可），别按模型大小填——填高了只会让协调器无谓地挤出其它模型。

内置 Z-Image-Turbo workflow 默认使用 FP8 diffusion + 轻量 text encoder 的 ComfyUI 模型文件：

- `models/diffusion_models/z-image-turbo-fp8-e4m3fn.safetensors`
- `models/text_encoders/qwen_3_4b_fp4_mixed.safetensors`
- `models/vae/ae.safetensors`

示例（Z-Image-Turbo + Qwen-Image + SenseNova 共用同一个 ComfyUI 实例）：

```jsonc
{
  "comfyui_targets": [
    {
      "name": "z-image-turbo-comfyui",
      "base_url": "http://127.0.0.1:21480",
      "model": "z-image-turbo",
      "preset": "z_image_turbo",
      "auto_start": true,
      "start_command": "\"C:\\path\\to\\ComfyUI\\python\\python.exe\" -s main.py --listen 127.0.0.1 --port 21480",
      "cwd": "C:\\path\\to\\ComfyUI",
      "diffusion_model": "z-image-turbo-fp8-e4m3fn.safetensors",
      "text_encoder_model": "qwen_3_4b_fp4_mixed.safetensors",
      "vae_model": "ae.safetensors",
      "default_steps": 8,
      "default_cfg": 1.0,
      "default_sampler_name": "res_multistep",
      "default_scheduler": "simple",
      "default_denoise": 1.0,
      "default_edit_denoise": 0.25
    },
    {
      "name": "qwen-image-comfyui",
      "base_url": "http://127.0.0.1:21480",
      "model": "qwen-image",
      "preset": "qwen_image_edit_aio",
      "auto_start": true,
      "start_command": "\"C:\\path\\to\\ComfyUI\\python\\python.exe\" -s main.py --listen 127.0.0.1 --port 21480",
      "cwd": "C:\\path\\to\\ComfyUI",
      "default_steps": 6,
      "default_cfg": 1.0,
      "default_sampler_name": "euler_ancestral",
      "default_scheduler": "beta",
      "default_denoise": 1.0,
      "default_edit_denoise": 1.0,
      "output_prefix": "fake_ollama/qwen-image"
    },
    {
      "name": "sensenova-comfyui",
      "base_url": "http://127.0.0.1:21480",
      "model": "sensenova",
      "preset": "sensenova_u1",
      "auto_start": true,
      "start_command": "\"C:\\path\\to\\ComfyUI\\python\\python.exe\" -s main.py --listen 127.0.0.1 --port 21480",
      "cwd": "C:\\path\\to\\ComfyUI",
      "default_steps": 8,
      "default_cfg": 1.0,
      "output_prefix": "fake_ollama/sensenova"
    },
    {
      "name": "joyai-echo-comfyui",
      "base_url": "http://127.0.0.1:21480",
      "model": "joyai-echo",
      "preset": "joyai_echo",
      "auto_start": true,
      "start_command": "set \"PATH=C:\\path\\to\\ffmpeg;%PATH%\" && \"C:\\path\\to\\ComfyUI\\python\\python.exe\" -s main.py --listen 127.0.0.1 --port 21480",
      "cwd": "C:\\path\\to\\ComfyUI",
      "min_width": 256,
      "default_width": 256,
      "width_modulo": 32,
      "min_height": 256,
      "default_height": 256,
      "height_modulo": 32,
      "min_num_frames": 17,
      "default_num_frames": 17,
      "num_frames_offset": 1,
      "num_frames_modulo": 8,
      "min_frame_rate": 8,
      "default_frame_rate": 8,
      "default_prefetch_count": 1,
      "default_enable_tile": true,
      "default_enable_streaming": false,
      "max_reference_images": 5,
      "max_batch_size": 1,
      "prompt_timeout_seconds": 7200,
      "output_prefix": "fake_ollama/joyai-echo"
    }
  ],
  "api_interfaces": [
    {
      "name": "api",
      "host": "127.0.0.1",
      "port": 21435,
      "access_tokens": ["tk-..."],
      "exposed_models": [
        { "model": "z-image-turbo", "target": "z-image-turbo-comfyui", "alias": "z-image-turbo" },
        { "model": "qwen-image", "target": "qwen-image-comfyui", "alias": "qwen-image" },
        { "model": "sensenova", "target": "sensenova-comfyui", "alias": "sensenova-z-image" },
        { "model": "joyai-echo", "target": "joyai-echo-comfyui", "alias": "joyai-echo" }
      ]
    }
  ],
  "model_profiles": [
    {
      "model": "z-image-turbo",
      "target": "z-image-turbo-comfyui",
      "capabilities": ["image_generation", "image_edit"],
      "context_length": 4096,
      "estimated_vram_gb": 16,
      "request_vram_headroom_gb": 2,
      "min_free_vram_gb": 2,
      "vram_cleanup_policy": "adaptive"
    },
    {
      "model": "qwen-image",
      "target": "qwen-image-comfyui",
      "capabilities": ["image_generation", "image_edit"],
      "context_length": 4096,
      "estimated_vram_gb": 20,
      "request_vram_headroom_gb": 3,
      "min_free_vram_gb": 2,
      "vram_cleanup_policy": "adaptive"
    },
    {
      "model": "sensenova",
      "target": "sensenova-comfyui",
      "capabilities": ["image_generation", "image_edit"],
      "context_length": 4096,
      "estimated_vram_gb": 13,
      "estimated_memory_gb": 30,
      "request_vram_headroom_gb": 3,
      "min_free_vram_gb": 2,
      "vram_cleanup_policy": "adaptive"
    },
    {
      "model": "joyai-echo",
      "target": "joyai-echo-comfyui",
      "capabilities": ["video_generation"],
      "context_length": 4096,
      "estimated_vram_gb": 12,
      "estimated_memory_gb": 30,
      "request_vram_headroom_gb": 4,
      "min_free_vram_gb": 2,
      "vram_cleanup_policy": "adaptive",
      "exclusive_gpu": true
    }
  ]
}
```

OpenAI 兼容调用：

```powershell
$headers = @{ Authorization = "Bearer tk-..." }
$body = @{
  model = "z-image-turbo"
  prompt = "a clean product photo of a red cube"
  size = "512x512"
  response_format = "b64_json"
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:21435/v1/images/generations -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

`POST /v1/images/edits` 接受 OpenAI 风格 `multipart/form-data` 的 `image` 文件字段（也兼容 OpenAI 多图约定的 `image[]` / `image[0]` 字段名，并保留所有参考图），也接受 JSON base64 `image` 或 `images` 数组。常用覆盖参数：`size`、`n`、`seed`、`steps`、`cfg`、`sampler_name`、`scheduler`、`denoise`、`response_format`。

**随机种子（`seed_mode` / `seed`）**：请求体显式传 `seed` 时一律以请求为准；否则按 per-target 的 `seed_mode` 决定——`random`（默认，每次随机，取值限定在 0 ～ 2³²−1，以保证落在 JS/JSON 安全整数范围内、便于复现）、`fixed`（固定用 `seed`）、`increment`（从 `seed` 起按本次出图张数 `n` 递增）。`increment` 的计数器是进程内 per-target 状态，重启或 reload 后从 `seed` 重新开始。适合给不发 `seed` 的客户端（如 Cherry Studio 聊天生图）做可复现/递增出图。

#### 排队与超时（避免 502 ReadTimeout）

llama.cpp server 默认只有 `--parallel 1` 个解码 slot，并发请求会被上游 HTTP 层接收但串行 decode，前面的请求把后面的吃掉太久就会触发 fake_ollama 这一层的 httpx read timeout，被记成 `status=502 error=ReadTimeout`。

per-target / `llama_cpp_defaults` 上有两个旋钮专门解决这个：

- `max_concurrent_requests`：fake_ollama 内部用 `asyncio.Semaphore` 限制同时打上游的请求数，超出部分在内存里 FIFO 排队（占 `_request_refs`，不会被空闲回收当作 idle）。
  - **留空 / `0`（默认）→ 完全透传**，请求直接到 llama.cpp，由上游的 `--parallel` slot 队列调度。这是唯一能保证上下文切换阶段有 prefill 流水线的模式。
  - **正数 → 显式开启代理层队列**，cap 为该值；Ctrl+C 时整个队列会被一次性 cancel。

> 不从 `parallel` 自动继承：代理侧 cap=1 是严格串行，req B 必须等 req A 的 SSE [DONE] 之后才能开始处理，多一轮 fake_ollama↔llama.cpp 往返 + tokenization 延迟，实测比 cap=None 明显慢。需要代理层队列请显式配置。
- `request_read_timeout_seconds`：只覆盖 fake_ollama → llama.cpp 这一段的 httpx read timeout。
  - 留空 → 沿用全局 `timeout_seconds`。
  - `<=0` → read 不超时（适合排队 + 长生成场景，宁可一直等也不要 502）。
  - 正数 → 显式覆盖。

dashboard 表格新增 `Queued` 列，可以直接看每个本地模型当前排队的请求数。

### H3 Context-IR harness

`h3_context_ir_profiles` 是独立于公开聊天模型的编排层。配置内的 provider 只引用已经存在的 `{model, target}`，因此会复用原有的 Anthropic / OpenAI / Ollama 协议适配、token、超时、本地进程生命周期和显存/内存协调器，不保存第二份连接配置。Playground 也可按 profile 显式授权临时第三方连接。

```text
自然语言 + 0..2 张图
        │
        ▼
h3-context-ir-fake profile
        ├── text provider  ── 本地 Ollama/llama.cpp 或第三方 API
        └── image provider ── 本地/远端多模态 Chat API
        │
        ▼
校验后的 H3 Base IR ──► Base Prompt ──► 可选 ComfyUI H3 workflow
```

```jsonc
{
  "h3_context_ir_profiles": [
    {
      "name": "default",
      "alias": "h3-context-ir-fake",
      "allow_compatible_models": true,
      "allow_external_api": false,
      "providers": [
        {
          "name": "local-text",
          "model": "qwen-text",
          "target": "qwen-text-llama-cpp",
          "modalities": ["text"],
          "json_mode": true
        },
        {
          "name": "local-vl",
          "model": "qwen-vl",
          "target": "qwen-vl-llama-cpp",
          "modalities": ["text", "image"],
          "json_mode": true
        },
        {
          "name": "third-party",
          "model": "deepseek-chat",
          "target": "deepseek-openai",
          "modalities": ["text"],
          "json_mode": true
        }
      ],
      "default_text_provider": "local-text",
      "default_multimodal_provider": "local-vl",
      "max_attempts": 2,
      "failure_mode": "fallback",
      "default_duration_seconds": 5
    }
  ]
}
```

启用 Playground 后，`playground_visible: true` 的 profile 会显示为一个虚拟模型。它不会混入标准 `/v1/models`，也不需要加入接口的 `exposed_models`。Playground 调用 `POST /v1/videos/context-ir`；JSON 和 multipart 都支持以下字段：

- `model`：profile 的 `alias` 或 `name`。
- `prompt`：用户原始意图。
- `mode`：`auto`、`t2va`、`i2va`、`fl2va` 或 `l2va`；`auto` 根据参考图数量选择。
- `duration` / `duration_seconds`：4–15 秒。
- `provider`：可选推荐 provider 名，或 Playground 中列出的兼容模型；省略时，无图走 `default_text_provider`，有图走 `default_multimodal_provider`。
- `image` / `images`：最多两张，每张不超过 20 MiB。

`providers` 是 profile 明确推荐的稳定组合。所有 Planner（包括默认选择、推荐 provider 和视频生成时的编排）都必须在当前接口的 `exposed_models` 中授权。`allow_compatible_models: true` 时，Playground 还会在“自选兼容模型”分组中列出当前 API key 所选接口 `exposed_models` 里的其它聊天模型：必须声明 `completion`，带图能力则由 `vision` capability 决定；ComfyUI 和纯媒体模型不会进入列表。

`allow_external_api: true` 会再增加“临时第三方 API”选项。Playground 中选择 OpenAI-compatible 或 Anthropic-compatible，输入 URL 与 token 后，可通过第三方 `/v1/models` 自动识别模型并选择；若网关不提供模型列表，也可手动填写模型 ID。第三方模型端点通常不报告图片能力，因此需要明确选择“纯文字”或“多模态（可看图）”。URL 可填服务根路径、`.../v1` 或具体的 `.../v1/models` / `chat/completions` / `messages` 地址，服务端会规范化后调用。

临时 token 只保留在页面内存，通过单次请求的 `x-playground-upstream-key` 请求头传给 fake-ollama；不会写入 `config.json`、模型 discovery 或交互记录，请求数据日志也会将该头脱敏。临时外部连接只允许从独立 Playground 监听器发起，普通 API/Ollama 监听器不能使用。还必须在该 profile 的 `external_api_allowed_base_urls` 中逐项配置可信 base URL（例如 `["https://provider.example/api"]`）；规范化后精确匹配，不接受通配符，不跟随重定向。空列表拒绝所有临时第三方地址；填写回环或内网 URL 即明确授权访问该地址。管理员应只添加自己信任其 DNS 与服务的地址。

Playground 的预计显存、预计内存和上下文会随实际 Planner 选择即时更新；`auto` 会根据当前是否附图显示对应默认 Planner 的资源。若给纯文字 Planner 附图，界面会在发送前提醒，响应 `warnings` 也会说明：图片仍用于 `<Picture N>` 对齐，但不会发送给 Planner 分析。

模型只负责生成紧凑 JSON IR；harness 接管模式、时长、时间线验证、官方 Base Prompt 三段式渲染及失败重试。若 provider 不可用或连续返回无效 IR，`failure_mode: fallback` 会生成不丢失原始文字的一镜到底提示词，并在响应的 `fallback` / `warnings` 中明确标记；设为 `error` 则返回错误。

需要把它串到视频生成时，在对应 `comfyui_targets[*]` 上配置：

```jsonc
{
  "context_ir_profile": "default",
  "context_ir_prompt_mode": "auto"
}
```

随后 `POST /v1/videos/generations` 可用 `prompt_mode=raw|auto|enhance`、`context_ir_provider` 和 `context_ir_mode` 覆盖 target 默认值。`auto` 会保留已经是 H3 三段式的 prompt，普通自然语言才会增强；响应会附带实际送入 workflow 的 `revised_prompt`。只有真正接受 H3 Base Prompt 的视频 workflow 才应开启这项绑定。

### 本机 MiniMax H3 768p 后端

#### 模型与路径

仓库内置 `minimax_h3` preset，并在示例配置中把它作为独立 ComfyUI target 暴露为
`minimax-h3`。四种 HTTP 模式共用以下权重：

- `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`（19.53 GiB）
- `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`（14.61 GiB）
- `vae/minimax_h3_video_vae_fp16.safetensors`（4.85 GiB）
- `vae/minimax_h3_audio_vae_fp32.safetensors`（0.56 GiB）

模型根目录由本机配置决定。先把模板复制到被 Git 忽略的本机目录，再填写真实的
`base_path`：

```powershell
New-Item -ItemType Directory -Force config/local | Out-Null
Copy-Item config/examples/comfyui/minimax_h3_extra_model_paths.yaml config/local/minimax_h3_extra_model_paths.yaml
# 编辑 config/local/minimax_h3_extra_model_paths.yaml 中的 base_path
```

#### 服务与资源配置

H3 target 的 `start_command` 必须通过 `--extra-model-paths-config` 指向上述
`config/local/minimax_h3_extra_model_paths.yaml`，不要直接指向仍含占位路径的模板。示例 target 使用独立的
ComfyUI 主线实例和 `:21481`，不会升级或覆盖其他 ComfyUI 实例。启动参数保留 2 GiB
显存给系统；资源协调器按
`estimated_vram_gb=21`、`estimated_memory_gb=56` 做加载准入；连续推理另外使用
`request_vram_headroom_gb=6`、`min_free_vram_gb=2`、`vram_cleanup_policy=adaptive`
和 `exclusive_gpu=true`。默认参数下，上一轮结束后若空闲显存不足 8 GiB，会先卸载 GPU
权重但保留 ComfyUI 对象缓存，确认重新腾出至少 21 GiB 后才提交下一份视频。

RTX 4090（SM 8.9）没有原生 NVFP4 Tensor Core。最新版 ComfyUI 会将 NVFP4 标记为
emulated：磁盘/主存仍使用 NVFP4-AWQ 压缩权重，计算时按层反量化；因此能运行，但文本
编码速度不会获得 Blackwell 原生 NVFP4 的加速。INT8 ConvRot DiT 则继续走
`comfy-kitchen` 的量化路径。此差异影响速度和峰值内存，不会把量化模型悄悄换成另一套
权重。

当 `prompt_mode=auto|enhance` 先调用本地 Planner 时，服务会在规划结束后通过并发感知的
资源释放接口立即卸载由 fake-ollama 管理的 Planner，再执行 H3 target 配置的资源准入检查；因此
24 GiB 显卡不需要等待常规的 60 秒空闲回收窗口。远程/第三方 Planner 不受影响，仍在使用
或不由 fake-ollama 管理的本地进程也不会被强行停止。

#### HTTP 模式与调用

`video_mode` 的选择规则（旧字段 `context_ir_mode` 仍可用于四种 Base 模式；同时指定时不能冲突）：

- `auto`：0 张图 → T2V，1 张图 → 首帧 I2V，2 张图 → FL2VA。
- `t2va`：不接图片。
- `i2va`：1 张图接到 `first_frame`。
- `fl2va`：第 1 张接 `first_frame`，第 2 张接 `last_frame`。
- `l2va`：1 张图只接到 `last_frame`；由于与 I2V 同为一张图，必须显式选择。
- `ref2va`：显式使用 Ref2VA 权重，图片／视频作为身份、外观或动作参考；要求 `prompt_mode=raw`。

服务端会在调用 Planner 和 ComfyUI 之前校验模式与图片数；`prompt_mode=raw` 也不会绕过
这项校验。默认 124 帧约对应 5 秒输入时长，H3 只接受 `17k+5` 帧网格；24 FPS 固定。

```powershell
$headers = @{ Authorization = "Bearer tk-..." }
$body = @{
  model = "minimax-h3"
  prompt = "A paper boat moves through a rainy neon alley, with synchronized rain and footsteps."
  context_ir_mode = "t2va"
  size = "1344x768"
  num_frames = 124
  fps = 24
  response_format = "b64_json"
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:21435/v1/videos/generations `
  -Method Post -Headers $headers -ContentType "application/json" -Body $body
```

#### Ref2VA 参考生成

Ref2VA 使用 `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors`，
与上述 FL2VA 是不同权重，共用文本编码器和音视频 VAE。将它放到同一模型根目录的
`diffusion_models` 下，即可复用 `--extra-model-paths-config` 配置。

`minimax_h3` preset 的 `/v1/videos/generations` 和 Playground 支持 `video_mode=ref2va`。
设置 target 的 `max_reference_images=9` 可开放全部参考图槽位；原有 Base 模式仍按 0／1／2 张图校验。
可用 `reference_to_video_workflow_path` 覆盖模板，`bindings` / `static_inputs` 使用 `ref2va` 键。
内置模板使用 `MiniMaxH3ReferenceToVideo`，不会把 Ref2VA 权重替换进 FL2VA 图中。

| 输入 | JSON | multipart |
| --- | --- | --- |
| 参考图片，最多 9 张 | `images`: base64/data URL 数组，`filenames`: 文件名数组 | 重复 `image[]` 文件 |
| 参考视频，最多 3 段 | `videos`: base64/data URL 数组，`video_filenames`: 文件名数组 | 重复 `video[]` 文件 |
| 任意帧图片约束，最多 16 项 | `keyframes`: 对象数组 | `keyframes`: JSON 字符串 |

Ref2VA 至少需要一张参考图或一段参考视频。视频需预处理为 24 FPS、2–15 秒；
目前只使用视频画面，音轨及独立参考音频未接入。提示词用 `<Picture 1>` / `<Video 1>`
按各自输入顺序引用素材。Context-IR 尚不能规划参考模式，必须显式传 `prompt_mode=raw`。
统一 API 仍经过原有 GPU/RAM 准入、排队和运行中显存监控。

关键帧既可用于 Ref2VA，也可用于四种 Base 模式。例如：

```json
{
  "keyframes": [
    {"time_seconds": 1.5, "image_index": 1},
    {"frame_index": -1, "image": "data:image/png;base64,...", "filename": "end.png"}
  ]
}
```

每项使用 `frame_index` 或 `time_seconds` 二选一；帧号从 0 开始，负数从片尾倒数，秒数四舍五入到最近帧。
图片来源可用 `image` 的 base64/data URL、`image_index`（从 0 开始的图片上传顺序）或
`image_field`（独立 multipart 文件字段名）。被 `image_index` 选中的图片从普通参考／首尾帧输入中移出，
其余图片保留顺序。重复位置、越界位置和与 Base 首尾帧冲突的约束均返回 400，不能静默丢弃。
关键帧同样要求 `prompt_mode=raw`，并需要 ComfyUI 的 `MiniMaxH3AddGuide` 节点；
旧安装可按[关键帧补丁说明](services/comfyui_h3_guides/README.md)检查、备份和安装。

[统一 API 调用示例](scripts/README.md#h3-ref2va-与任意帧关键帧)包含本地文件上传命令。
[原生构建工具](scripts/README.md#h3-ref2va-原生工作流)仍可独立输出 `/prompt` 工作流，
但原生请求的资源管理由调用者负责。

本机安装、历史迁移验证和当前接入边界保存在模型根目录的 `USAGE.md`，
可按 `config/local/minimax_h3_extra_model_paths.yaml` 中的 `base_path` 查找。

### Interfaces（接口与白名单）

`ollama_interfaces` / `api_interfaces` 都是数组，每一项独立监听：

```jsonc
{
  "ollama_interfaces": [
    {
      "name": "ollama",
      "host": "127.0.0.1",
      "port": 21434,
      "access_tokens": [],
      "exposed_models": [
        { "model": "claude-3-5-sonnet-20241022", "target": "default" },
        { "model": "deepseek-r1",                "target": "deepseek_remote" }
      ]
    }
  ],
  "api_interfaces": [
    {
      "name": "api",
      "host": "127.0.0.1",
      "port": 21435,
      "access_tokens": [ "tk-..." ],
      "exposed_models": [
        { "model": "llama3.1",  "target": "local", "alias": "llama3.1" },
        { "model": "qwen3.6-27b-iq4xs", "target": "qwen3.6-27b-iq4xs", "alias": "qwen-vision" }
      ]
    }
  ]
}
```

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | string | 接口标识，用于日志、Admin 和运行时路由；修改监听器身份需重启 |
| `host` | string | 监听 host（`127.0.0.1` 或 `0.0.0.0`） |
| `port` | int | 所有启用的监听器必须使用不同端口，包括 Admin、Dashboard、Playground；即使 host 不同也不允许复用端口 |
| `access_tokens` | string[] | 该接口的 token 池；空 = 无需 token，仍须通过匿名请求的 Host / 来源校验 |
| `exposed_models` | array of `ExposureEntry` | 该接口允许看到 / 请求的模型 |

每个 `ExposureEntry` 字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `model` | string | source 的显示名：源级 `alias`，未设置时为 `name`；单模型 target 对应 `alias` 或 `model` |
| `target` | string | source 的 `name`，决定路由到哪 |
| `alias` | string? | 可选；非空时**作为该 exposure 的公开 ID**，替换 `model@target` 复合形式 |

- 公开 ID：`exposure.alias` 有值时使用它，否则使用 `"{exposure.model}@{exposure.target}"`。
- 客户端使用接口列出的公开 ID；没有 `@` 的裸名字只有显式 exposure alias 才能匹配，不会自动补 target。未暴露或未匹配返回 404。
- 同一个模型可以在多个接口里同时暴露，alias 可以不同。
- `/api/tags`（Ollama）与 `/v1/models`（OpenAI）只列出当前接口的公开 ID。
- mismatch 警告：Admin UI 会在你勾的 `exposed_models` 引用了不存在的 source / model 时高亮提示。

#### Tagless 回退匹配

为了兼容客户端把 `foo:latest` 写成 `foo` 的常见情况，请求 `model` 形如 `foo@target` 时，若没有精确匹配的 exposure，会在「该接口里 target 相同且未设 alias 的 exposure」里按 Ollama 规则（`foo` ≡ `foo:latest`）二次匹配，命中即路由。设了 `alias` 的 exposure 不参与该回退（alias 视为强精确）。

### 共享设置

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `default_max_tokens` | int | `4096` | 客户端没传时的默认输出上限 |
| `timeout_seconds` | float | `300` | 默认出站 HTTP 超时；可被 target 的读取超时等专用设置覆盖 |
| `use_system_proxy` | bool | `false` | 出站是否走系统代理 |
| `enforce_context_limit` | bool | `true` | 估算 token 超 `context_length` 直接 400 |
| `max_request_body_bytes` | int | `67108864` | 整个入口请求体上限（64 MiB）；包括 base64 与 JSON 开销，分块上传同样受限 |
| `request_body_timeout_seconds` | float | `60` | 接收请求体的时间上限，不限制后续推理时间 |
| `advertised_version` | string | `0.6.4` | 仅用于 Ollama 接口的 `/api/version` |
| `model_profiles` | list | `[]` | 每模型 capabilities / 上下文 / 思维链等。每项写 `model`（必填）+ 可选 `target`，两者拼起来作为最终 key：填 `target` 时为 `model@target` 仅覆盖该 target，不填则裸 `model` 适用于所有 target。查找时三级回退：`model@target` > 裸 `model` > tagless base。旧的 `{ "model@target": {...} }` dict 写法仍兼容 |

`model_profiles[*].capabilities` 是对外声明的功能标签，主要给 `/api/show`、Anthropic `/v1/models` 的标准能力映射和 `/playground/api/models` 做功能发现；它不会自动让底层模型获得该能力。`/api/show` 只输出 Ollama 标准的 `completion` / `tools` / `vision`，图片与视频标签只由 Playground 专用接口消费。常用值：

| capability | 使用场景 |
| --- | --- |
| `completion` | 文本补全 / 聊天模型。Ollama、llama.cpp、Anthropic/OpenAI chat 模型通常需要它。 |
| `tools` | 工具调用 / function calling。仅在底层模型或上游确实能处理工具调用时声明。 |
| `vision` | 聊天接口可接收图片输入，例如 `/api/chat` 的 `images` 或 OpenAI/Anthropic 多段图片消息。 |
| `video_understanding` | OpenAI 兼容聊天接口可接收视频输入，Playground 将其显示为单轮“视频分析”。 |
| `image_generation` | 图片生成模型，供 OpenAI 兼容 `POST /v1/images/generations` 使用。 |
| `image_edit` | 图片编辑 / image-to-image 模型，供 OpenAI 兼容 `POST /v1/images/edits` 使用。 |
| `video_generation` | 视频生成 / image-to-video 模型，供扩展接口 `POST /v1/videos/generations` 使用。 |

聊天模型一般至少包含 `completion`；纯媒体 workflow 可以不填 `completion`，只声明 `video_understanding`、`image_generation`、`image_edit` 或 `video_generation`。`/playground/api/models` 会返回原始 `capabilities` 和结构化 `operations`，包含 endpoint、是否流式、`history_mode`、图片/视频输入约束、实际绑定的参数 schema、默认值/范围及推荐 preset，并返回 `estimated_vram_gb` / `estimated_memory_gb` 供调试界面展示；Playground 完全由这些字段驱动。标准 `/v1/models` 根据 `anthropic-version` 请求头返回 Anthropic 或 OpenAI 原生结构，不包含这些调试扩展。

### Admin UI / Dashboard / Model Playground

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `admin_enabled` | bool | `true` | 是否注册 `/admin` |
| `management_access_tokens` | string[] | `[]` | Admin / Dashboard 的独立管理凭据；空列表仅允许受来源校验保护的本机页面免登录 |
| `admin_host` | string | `127.0.0.1` | admin listener 地址；远程管理需独立凭据 |
| `admin_port` | int | `21433` | admin listener 端口 |
| `dashboard_enabled` | bool | `true` | 是否启用资源监控 dashboard |
| `dashboard_host` / `dashboard_port` | | `127.0.0.1` / `21432` | dashboard listener |
| `dashboard_sample_interval_seconds` | float | `10.0` | 采样间隔 |
| `dashboard_retention_seconds` | float | `604800.0` | 历史保留窗口 |
| `dashboard_data_path` | string | `logs/dashboard_history.json` | 历史落盘路径 |
| `dashboard_model_reclaim_enabled` | bool | `false` | dashboard 是否显示模型回收 |
| `dashboard_reclaim_idle_seconds` | float | `20.0` | dashboard 关闭按钮所需的最小空闲秒数。和自动 LRU 回收的 60s 阈值独立——用户手动判断更宽松 |
| `vram_low_free_reclaim_enabled` | bool | `true` | 检测显存低水位时主动回收 |
| `vram_low_free_threshold_mib` | float | `200.0` | 低水位阈值（MiB） |
| `memory_low_free_reclaim_enabled` | bool | `true` | 检测系统内存低水位时主动回收（仅回收声明了 `estimated_memory_gb` 的模型） |
| `memory_low_free_threshold_mib` | float | `2048.0` | 内存低水位阈值（MiB） |
| `playground_enabled` | bool | `false` | 是否启用轻量流式模型调试页；交互记录仅保留在当前页面内存中，刷新后清空 |
| `playground_host` / `playground_port` | | `127.0.0.1` / `21431` | Playground 独立 listener；局域网访问时将 host 改为 `0.0.0.0`，修改后需重启进程 |
| `playground_trusted_proxy_ips` | string[] | `[]` | 允许匿名 Playground 经代理访问时，代理的实际直连 IP；不使用转发头推断 |
| `playground_allowed_origins` | string[] | `[]` | 与可信代理配套的精确公开 origin，例如 `https://models.example.com`；不能包含路径或尾部 `/` |

启用后打开 `http://127.0.0.1:21431/playground/`。本机直连且存在绑定回环地址的匿名模型接口时，可留空 API key；否则输入所选 interface 的 `access_tokens` 之一，加载它暴露的模型。页面按 capabilities/operations 提供聊天、视觉输入、视频分析、图片生成、图片编辑、视频生成或 Decision 模式，并按实际参数 schema 显示控制项。图片可粘贴、拖放或从文件选择；视频分析支持本地视频和手机 / PC 摄像头。摄像头按 `segment_seconds` 连续录制可独立解码的片段，录制与 Mage-VL 推理并行，GPU 请求串行；推理落后时只保留最新一个待处理片段并显示丢弃数。单段实时输出最多使用 128 个 token，且不执行整体总结。

手机访问时，把 `playground_host` 设为 `0.0.0.0`，重启 fake-ollama 后打开 `http://<PC 局域网 IP>:21431/playground/`，并按需放行 Windows 防火墙的 TCP 21431 入站。这里的 HTTP 方案仅完成应用链路：主流手机浏览器通常只在 HTTPS 安全上下文中开放 `getUserMedia`，因此未配置 HTTPS 时可能看得到页面但无法取得摄像头权限；PC 上的 `http://localhost` 通常可直接使用。不要把 Playground 端口直接暴露到互联网。

可用 Edge / Chrome 内置虚拟摄像头做完整页面烟测（要求 fake-ollama 已启动）：

```powershell
$env:FAKE_OLLAMA_CAMERA_TEST_API_KEY = "<access token>"
node scripts\validation\validate_playground_camera.mjs
```

脚本不需要 npm 依赖，不会打印 API key；它会用临时浏览器 profile 调用真实 Mage-VL，验证连续录制、单槽背压、结果历史和停止收尾后再关闭测试浏览器。

聊天模式按模型保存当前页面内的完整轮次，每次请求会回传先前的 user/assistant 消息；前端读取 `context_length` 和 `max_output_tokens` 估算预算，超过 90% 安全阈值时优先丢弃最早的完整轮次，必要时再截短当前输入。视频分析、图片生成、图片编辑和视频生成也会保留逐轮输入与结果，但其 `history_mode=single_turn`，接口只接收本次输入。

匿名 Playground 默认仅允许实际回环监听、回环直连客户端、正确端口的回环 Host 和同源请求，并从绑定回环的匿名接口中选择权限。直接通过局域网访问时需要有效的接口 token；无效 token 不会回退到匿名接口。若明确要让反向代理提供匿名入口，必须同时配置代理直连 IP `playground_trusted_proxy_ips` 和精确公开 origin `playground_allowed_origins`，并保留一个回环匿名模型接口。此配置明确授予该公开入口匿名模型访问能力；`Forwarded` / `X-Forwarded-*` 不能代替这两项配置。

## 内部 backends 视图

`Settings` 通过来源查询方法构造协议无关的 `Backend`，没有 `settings.backends` 属性：

```python
settings.all_sources()
# -> 配置中的 source 对象列表
settings.backend_by_name("deepseek_remote")
# -> Backend（未知来源返回 None）；含 name、protocol、kind、base_url、source 等字段
#    protocol: "anthropic" | "openai" | "ollama" | "comfyui" | "decision"
#    kind:     "remote" | "local"
#    source:   底层 AnthropicUpstream / OpenAIUpstream / OllamaTarget / GenericOpenAITarget / LlamaCppTarget / ComfyUITarget / DecisionTarget

settings.resolve_request("deepseek-r1@deepseek_remote", interface_name="api")
# -> 按接口解析「客户端请求 model」到 (backend, 源级显示名)
#    会查 exposed_models（含 tagless 回退），找不到抛出 ValueError
```

`resolve_request` 取代了旧版 `backend_for("model@target", surface=...)`：从「surface=internal/external」改成显式传 `interface_name`，因为接口现在是数组。

## 反向代理：把本地模型当远端 API 用

适合**只支持 Anthropic 或 OpenAI API** 的客户端调用本机 Ollama / llama.cpp / ComfyUI 模型，或把上游 Anthropic 做带鉴权的转发壳。

只要在某个 `api_interfaces[*].exposed_models` 里勾上对应模型（指向 `ollama_targets`、`llama_cpp_targets` 或 `comfyui_targets`），并设置好 `access_tokens`，反向代理就会按模型来源路由：

- 命中 `ollama_targets[*].models[*]` → 转换为 Ollama `/api/chat`，再翻译回 Anthropic / OpenAI 响应。
- 命中 `llama_cpp_targets[*]` → 调用 llama.cpp `/v1/chat/completions`；Anthropic 入口做格式转换，OpenAI 入口基本直通。
- 命中 `comfyui_targets[*]` → `POST /v1/images/generations` / `POST /v1/images/edits` 调用 ComfyUI workflow；文本 chat/messages 入口会返回 400 并提示改用图片接口。
- 命中 anthropic_upstreams / openai_upstreams 提供的模型 → 透传到该上游（带鉴权壳）。
- 未命中 / 未在白名单 → 404 `model '...' is not exposed on interface 'xxx'`。

`POST /v1/messages/count_tokens` 命中 Anthropic 来源时转发其计数接口；其他后端使用本地估算。Anthropic 上游缺少有效计数时返回 502，不报告虚假的 0。

支持的转换：

- 文本消息、`system`、`max_tokens` / `temperature` / `top_p` / `top_k` / `stop_sequences`
- **工具调用**：流式与非流式都保留名称、参数、ID 和结束原因。Ollama 输出包含 `function.index`；回传 `role=tool/tool_name` 按调用顺序关联，同名多次调用受支持。数量、顺序或名称冲突返回 400。流式工具 JSON 在校验完整后发出；不完整 JSON、上游错误或提前 EOF 输出错误，不报告成功。
- **工具控制**：OpenAI `none/auto/required/指定函数` 对应 Anthropic `none/auto/any/tool`；`parallel_tool_calls` 与 `disable_parallel_tool_use` 互相映射。原生 Ollama 无法表达强制或仅单次调用时返回 400；`none` 通过不提供工具实现。其他无法表达的 tool_choice 形式明确拒绝。
- **用量**：转换后的 OpenAI 流支持 `stream_options: {"include_usage": true}`，结束前发出 `choices: []` 的 usage 帧，其他帧 `usage: null`。转向 OpenAI 上游的流主动请求用量。未知计数省略，已知的 0 保留；只有输入、输出均已知才计算总数。全部未知时最终 OpenAI usage 为 null；流中断不会补造用量。原生 OpenAI 直通保留上游行为。
- Anthropic base64 `image` 块：转到 Ollama 时变成 `images` 数组；转到 llama.cpp 时变成 OpenAI `image_url` data URI

让 Claude Code 走它：

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:21435"
$env:ANTHROPIC_AUTH_TOKEN = "tk-填你的-api_interface-access_tokens-之一"
$env:ANTHROPIC_MODEL = "llama3.1"
claude
```

## 视觉输入（图片）

- `/api/chat`、`/api/generate`：消息里传 `images: ["<base64>", ...]`，服务端从 base64 magic bytes 嗅探 PNG / JPEG / GIF / WEBP。
- `/api/chat` 也兼容 OpenAI 风格多段 `content`：`{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`。
- `/v1/chat/completions` 与 `/v1/messages`：参见 OpenAI / Anthropic 官方多段 content 格式。
- `/v1/images/generations` / `/v1/images/edits`：走 `comfyui_targets`，用于图片生成和 image-to-image 编辑。
- `/v1/videos/generations`：走 `comfyui_targets`，用于 text-to-video；请求里带 `image` / `images` 且 target 配了 `image_to_video_workflow_path` 时走 image-to-video。
- 上游不支持图片时（如 DeepSeek text-only）会返回错误，fake-ollama 不预拦截。

## Web 配置编辑器（/admin）

启动后浏览器打开 <http://127.0.0.1:21433/admin>。左侧按 Model Sources、H3 Context-IR-fake、Interfaces、Runtime、Dashboard、Model Playground 和 Admin UI 分组：

- 每个字段左侧复选框 = 「是否写入 JSON」；取消勾选 = 回退默认值。
- 模型来源（含 `generic_openai_targets` / `decision_targets` / `comfyui_targets`）、接口和 `model_profiles` 都是可重复组，自带 +add / Remove。
- 视觉上 Ollama 和 llama.cpp source 卡片用不同强调色和图标分组，方便区分。
- **Detect models**：upstream / Ollama 卡片支持探测后合并到 `models`；llama.cpp 卡片支持探测后写入单个 `model`。
- **Exposure**：每个接口卡片里的 `exposed_models` 是结构化表格，`model` / `target` 都带 datalist 自动补全；mismatch（引用了不存在的 source / model）会高亮。
- **access_tokens**：每行带 Show / Generate；列表底部有 "+ generate token"。
- 顶部按钮：**Save & Reload** / **Discard & Reload from disk** / **Toggle raw JSON**。
- 校验失败会显示字段位置和错误原因，省略提交值以免回显密钥，配置不会被写入。

`/admin` 只挂在独立 admin listener 上，**不会**被任何 `ollama_interfaces` / `api_interfaces` 暴露。

## 测试

```powershell
pip install -e ".[test]"

# 默认离线单元和契约测试，排除 integration/e2e/gpu
pytest

# 单独运行隔离进程树测试；只启动短生命周期 Python 子进程
pytest tests/unit/test_owned_process_tree.py -m e2e

# live 集成测试需明确授权，并在当前进程环境中设置
# FAKE_OLLAMA_TEST_BASE_URL / FAKE_OLLAMA_TEST_AUTH_TOKEN
pytest -m integration
```

`tests/conftest.py` 不读取开发机 `.env`，在缺少环境凭据时自动跳过 `@pytest.mark.integration` 用例。普通测试使用临时合成配置，不读取项目根目录的真实 `config.json`。

## 手动验证示例

先替换配置中的上游模型与凭据，再从当前接口的模型列表选取公开 ID。以下使用 PowerShell 原生命令，避免 Windows PowerShell 的 `curl` 别名和 JSON 引号差异；聊天请求会实际调用所选模型。

```powershell
# Ollama 匿名接口；配置了 token 时也须传对应认证头
Invoke-RestMethod http://127.0.0.1:21434/api/tags
$ollamaModel = '替换为 /api/tags 返回的一个聊天模型名'
$body = @{ model = $ollamaModel; stream = $false; messages = @(@{ role = 'user'; content = 'hi' }) } | ConvertTo-Json -Depth 8
Invoke-RestMethod http://127.0.0.1:21434/api/chat -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))

# API 接口：默认返回 OpenAI 模型列表；带 anthropic-version 时返回 Anthropic 结构
$apiHeaders = @{ 'x-api-key' = '替换为此接口的 token' }
Invoke-RestMethod http://127.0.0.1:21435/v1/models -Headers $apiHeaders
$anthropicHeaders = @{ 'x-api-key' = $apiHeaders['x-api-key']; 'anthropic-version' = '2023-06-01' }
Invoke-RestMethod http://127.0.0.1:21435/v1/models -Headers $anthropicHeaders

# Playground 启用后，完整调试元数据在其独立端口提供
Invoke-RestMethod http://127.0.0.1:21431/playground/api/models -Headers $apiHeaders

$apiModel = '替换为 /v1/models 返回的一个聊天模型 ID'
$body = @{ model = $apiModel; max_tokens = 64; messages = @(@{ role = 'user'; content = 'hi' }) } | ConvertTo-Json -Depth 8
Invoke-RestMethod http://127.0.0.1:21435/v1/messages -Method Post -Headers $anthropicHeaders -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
Invoke-RestMethod http://127.0.0.1:21435/v1/chat/completions -Method Post -Headers $apiHeaders -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

## 请求数据日志

`logs/fake_ollama.log` 只放运行日志；请求 / 响应预览默认写到 `logs/fake_ollama.requests.jsonl`。

每行是一条 JSON 事件，核心字段：

- `request_id`：同一次入口请求贯穿入口 HTTP、后端请求、响应 chunk、返给 agent 的 chunk。
- `event`：`http_request_start` / `http_request_body` / `backend_request` / `backend_response_body` / `http_response_body` / `http_request_end` / `backend_error`。
- `body`：最多 64 KiB 的正文预览，超出时标记 `truncated: true`。入口响应的所有 chunk 共用 64 KiB 记录预算；后端单条记录同样限长。文本 UTF-8，非文本 base64。
- `headers`：`Authorization` / `x-api-key` / cookie 等敏感头只留 sha256 指纹。

只记录通过入口认证的 `/api/*` 与 `/v1/*`，不记录 `/admin/*`。默认 100MB × 10 个文件轮转。请求体在读取过程中累计限长，默认 `max_request_body_bytes: 67108864`（64 MiB）；`request_body_timeout_seconds: 60` 限制接收正文的时间，不限制推理时间。上传更大视频时需主动调高上限。

```powershell
python -m fake_ollama --request-data-log-file I:\path\requests.jsonl
python -m fake_ollama --no-request-data-log
```

需要避免将 API 调用内容记入文件时，可在启动服务时启用隐私模式（默认关闭）：

```powershell
python -m fake_ollama --config ./config.json --privacy-mode
```

隐私模式关闭请求 / 响应正文日志、可能包含上游报错内容的运行日志文件，以及网关代管的后端 stderr 文件捕获；终端日志保留。
它优先于 `--log-file` 和 `--request-data-log-file`，不创建或追加这些日志文件，也不删除已有日志；不传此参数时保持原有日志行为。
该开关仅对本次启动生效，不控制外部服务或启动脚本自行写入的日志、终端重定向，以及推理所需的临时文件或生成的媒体文件。更多选项见 [CLI 使用说明](docs/cli.md#日志)。

## 安全提示

- `.env` / `config.json` 已加入 `.gitignore`，不要提交真实 token。
- `logs/fake_ollama.requests.jsonl*` 会包含 prompt、工具参数、模型输出、图片 base64，应按敏感数据处理。
- 所有接口默认 `127.0.0.1`。要对外暴露：要么把对应实例的 `host` 改 `0.0.0.0`，要么保持 `127.0.0.1` + Nginx/Caddy 反代加 TLS。
- Admin / Dashboard 默认只允许回环监听、回环客户端与回环 Host。本机浏览器页面自动取得进程内管理凭据，所有管理 API 都验证该凭据；页面不缓存，跨站请求被拒绝。
- 免鉴权模型接口也会校验 Host 与实际监听端口，并拒绝跨源 Origin、跨站或仅同站的浏览器请求。本机 CLI、`127.0.0.1` / `localhost` 和 Playground 的同源访问仍可免填 token。非回环监听只接受配置的主机名或地址；通配监听接受实际连接的本机地址。通过其他域名或反向代理访问模型接口时，应配置接口 `access_tokens`。
- llama.cpp 启动日志和对外启动失败信息不输出完整命令或参数，避免泄露后端 `auth_token`；诊断保留目标名、退出码和配置检查提示。
- 远程访问或经反向代理提供管理页时，必须配置独立的 `management_access_tokens`，并使用 TLS。浏览器登录框的用户名可填 `admin`，密码填写管理 token；页面会自动认证后续 API 请求。命令行可使用 `Authorization: Bearer <management-token>` 或 `X-Management-Token`。模型接口的 token 不授予管理权限。修改管理凭据后旧页面失效，重新加载登录即可。
- 更新环境依赖后重启进程才会加载安全修复；端口隔离按 ASGI 原始路径判定，管理路由自身也独立验证监听端口及凭据。
- 反向代理 token：建议用 Web UI 的 Generate 生成 ≥24 字节随机串，不要复用其他系统 token。Token 池可放多个，便于按客户端轮换。

## 故障排查

- **`/v1/messages` 返回 404 `model '...' is not exposed on interface 'xxx'`**：该模型不在该 `api_interfaces[*].exposed_models` 里。去 admin UI 勾选。
- **`/api/chat` 返回 404 `model '...' is not exposed ...`**：客户端传的 model 不匹配公开 ID，也不在 tagless 回退里。先请求 `/api/tags` 获取当前接口的列表；源级 alias 不能代替接口级公开 ID。
- **返回 401**：该接口 `access_tokens` 非空，请求头里没带或带错 token。检查 `x-api-key` / `Authorization: Bearer`。
- **启动报 `cycle detected in model-forwarding graph`**：某条上游链路在同名模型上回到了自己，会无限递归。给链路上某一跳起一个不同的 alias 就能避免；`:latest` 标签会被归一化，重命名时注意区分。
- **启动报 `cycle detected ... admin/dashboard/playground listener`**：上游 URL 回指了本进程的管理或 Playground 端口，这些端口不能作为来源转发目标。改 URL 或删该来源。
- **运行时 HTTP 508 `loop detected`**：请求绕一圈又回到本进程（通常是反向代理 / 动态 DNS 绕过了启动期检测）。检查 upstream / 代理链。
- **启动报端口冲突**：接口或 Admin / Dashboard / Playground 复用了同一个端口；即使 host 不同也不允许。
- **匿名模型请求返回 403**：核对 Host、实际监听端口和浏览器 Origin。原生本机客户端使用 `127.0.0.1` / `localhost`；通过其他域名或代理访问时为接口配置 token。
- **上传返回 413 / 408**：分别表示整个请求体过大或接收超时。Mage-VL 还有独立上限，须同时核对网关配置和[伴生服务设置](services/mage_vl/README.md)，并为 base64 编码预留空间。
- **找不到日志文件**：默认相对 CWD：`logs/fake_ollama.log` 与 `logs/fake_ollama.requests.jsonl`。在项目根目录启动，或显式 `--log-file` / `--request-data-log-file`。
- **502 / 连不上上游**：`httpx` 默认会读 Windows 系统代理。装了 Clash / V2Ray 且上游是直连 IP 时，保持 `use_system_proxy: false`。
- **503 `Insufficient GPU VRAM`**：某个本地模型在 `model_profiles` 里配了 `estimated_vram_gb`，但可用显存不足。核对估算、等待模型空闲或手动释放显存；若需用停止命令管理外部服务，还须显式设置 `manage_external_runtime`。prompt 失联或关闭失败导致的隔离应先通过管理恢复接口核查，不能用降低估算绕过。
- **503 `Insufficient system RAM`**：某个本地模型配了 `estimated_memory_gb`，但当前可用系统内存不足。核对估算、等待空闲或释放内存；估算应以实际需求为依据。
- **缺少 thinking / signature**：回传完整、未编辑的历史或明确携带 reasoning；截断、改写的历史不会命中缓存。兼容端需要无签名 thinking 时显式配置 `allow_unsigned_thinking`；官方 Anthropic 的签名要求不能用这项配置绕过。

## 参考文档

- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) / [流式](https://docs.anthropic.com/en/docs/build-with-claude/streaming)
- [Ollama API 文档](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [DeepSeek Anthropic API 兼容性](https://api-docs.deepseek.com/zh-cn/guides/anthropic_api)
- [MiniMax H3 官方仓库](https://github.com/MiniMax-AI/MiniMax-H3) / [Prompt Writing Skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md) / [Base Prompt Guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt)
- [GitHub Copilot 自定义 Ollama provider](https://docs.github.com/en/copilot/customizing-copilot/extending-the-capabilities-of-github-copilot-in-your-ide)

## 免责声明（Disclaimer）

- 本项目只是一个**协议适配层**，不附带、不分发任何模型权重，也不与 Anthropic、Ollama、DeepSeek、OpenAI、GitHub 或任何商标持有人有关联或背书。
- 使用者**有义务**遵守所连接上游 API 提供商的服务条款（ToS）、Acceptable Use Policy 与所在司法辖区的法律法规。**不得**用于：
  - 绕过上游计费、配额、客户端识别等限制；
  - 转售 / 未经授权地代理上游服务；
  - 生成违反提供商内容政策的内容。
- 任何因使用本项目产生的费用、账号封禁、数据泄露或其他法律责任，**由使用者自行承担**，作者不承担任何明示或默示的担保责任。
- 在生产环境使用前，请自行评估并加固访问控制、密钥管理与日志脱敏。

## License

MIT — 详见 [LICENSE](./LICENSE)。

## Typed decision models

Dedicated `decision_targets` support SystemOne (`/v1/systemone`) and OpenAI Decisions (`/v1/decisions`) with explicit typed codecs, native runtime capability checks, and managed local lifecycles. See [decision configuration and examples](docs/decisions.md) and [optional official StartLux setup](docs/decision-startlux.md). Runtimes and model weights are prepared separately by the operator; decision runtime launches never fetch missing model files.
