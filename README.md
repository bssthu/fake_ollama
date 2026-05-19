# fake_ollama

一个轻量的协议适配层，主要做两件事：

1. **正向**（Ollama 兼容入口 → 远端上游）：把 **Anthropic Messages API** 或 **OpenAI Chat Completions API** 兼容的上游（官方 / DeepSeek / Together / Groq / 自建网关 / claude-relay-service）伪装成一台本机 **Ollama** 服务，让只支持 Ollama 协议的客户端（GitHub Copilot 自定义 provider、IDE 插件、桌面 AI 软件）无缝调用 Claude / DeepSeek / GPT 等模型。
2. **反向**（Anthropic / OpenAI 兼容入口 → 本机模型服务）：把本机的 **Ollama** 或 **llama.cpp server** 包装成 **Anthropic Messages API**（`POST /v1/messages`）和 OpenAI Chat Completions（`POST /v1/chat/completions`），让只支持远端 API 的客户端也能调用本地大模型。

附带一个零依赖的 Web 配置编辑器（`/admin`），不必再手改 JSON。

## 架构概览

监听被拆成三组：admin、Ollama 兼容入口（多实例数组）、API 接口（多实例数组）。这三组在配置里是独立的，安全边界、监听端口、白名单、token 都各自配置。

```
┌──────────────────────────────────────────────────────────────────┐
│  fake_ollama 进程（asyncio.gather 同时跑多个 uvicorn）           │
│                                                                  │
│  admin listener          (默认 127.0.0.1:21433)                  │
│    /admin/*          Web 配置编辑器（无内置鉴权）                │
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
│    /v1/embeddings    OpenAI 兼容                                 │
│    /v1/models                                                    │
│    每个实例可独立配置 host / port / access_tokens / exposed_models│
└──────────────────────────────────────────────────────────────────┘
```

- 每个 `ollama_interfaces[*]` / `api_interfaces[*]` 都是独立的 uvicorn 监听。你可以同时开多个 Ollama 接口（例如本机 21434 + 局域网 0.0.0.0:21434），每个挂不同的白名单和 token。
- `access_tokens` 为空 = 该接口不需鉴权；非空 = 该接口的所有请求都必须带 `x-api-key` 或 `Authorization: Bearer`。
- `exposed_models` 是该接口允许看到 / 请求的模型列表，元素是 `{model, target, alias?}` 三元组；不在白名单里的模型在该接口上 404。
- admin 接口（`admin_host` / `admin_port`）只服务 `/admin/*`，**不会**出现在任何 `ollama_interfaces` / `api_interfaces` 上；同样，`/api/*` 与 `/v1/*` 也**不会**出现在 admin 端口。
- 想让别的机器访问某个接口：把对应实例的 `host` 改成 `0.0.0.0`，或者保持 `127.0.0.1` + 在前面挂 Nginx/Caddy（推荐，可以加 TLS / 限流 / 客户端证书）。

## 特性一览

- **多上游路由**：把 Anthropic / OpenAI 兼容（DeepSeek / Together / Groq / 自建网关 / claude-relay-service…）合并到同一组接口
- **结构化模型标识 `{model, target}`**：每条暴露条目显式声明「上游显示名 + 来源 target 名」，相同名字但来源不同的模型不会再混淆；可选 `alias` 字段在客户端可见的公开 ID 上替换全名
- **每模型 profile**：capabilities / 上下文长度 / 思维链开关 / 输出上限——key 支持 `model@target`（最优先）、裸 `model`、tagless base 三级回退
- **按接口多实例暴露**：`ollama_interfaces[*]` 与 `api_interfaces[*]` 是独立数组，每个实例自带 `host` / `port` / `access_tokens` / `exposed_models`
- **来源命名清晰区分**：`anthropic_upstreams` / `openai_upstreams`（远端）+ `ollama_targets` / `llama_cpp_targets`（本机）
- **循环引用检测**：`anthropic_upstreams[*].base_url` 若指向 fake_ollama 自己的某个监听端口，启动时直接报错，避免转发死循环
- **本地 target 生命周期接管**：Ollama / llama.cpp 都可配置 health check、按需启动脚本、启动超时、空闲回收
- **本地显存预检**：本地模型可在 `model_profiles` 里填 `estimated_vram_gb`；启动前用 `nvidia-smi` 评估可用显存并尝试回收空闲模型
- **图片输入**：自动嗅探 base64 magic bytes（PNG/JPEG/GIF/WEBP）
- **零依赖 Web 编辑器**：按「Forward / Reverse / Shared / Admin UI」分组，字段说明、默认值回退、上游 detect-models、`model_profiles` key 自动补全
- **网络错误安全降级**：上游连接断开 / 超时统一返回 502 / 流式错误帧
- `pytest` + `httpx.MockTransport` 离线单测

## 快速开始

```powershell
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt

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

启动后：
- Ollama 客户端连 `ollama_interfaces[*].host:port`（典型 `http://127.0.0.1:21434`）
- 浏览器打开 <http://127.0.0.1:21433/admin>
- Anthropic / OpenAI 客户端连 `api_interfaces[*].host:port`（典型 `http://127.0.0.1:21435`），带对应 `access_tokens`

## 配置（config.json）

加载顺序（后者覆盖前者）：

1. 代码默认值
2. `config.json`（路径优先级：`--config` > `$FAKE_OLLAMA_CONFIG` > `./config.json`）

运行期只读一个环境变量：`FAKE_OLLAMA_CONFIG`，用于选中配置文件路径。所有 token / host / port / 白名单等都只能写在 `config.json`（或 Admin UI）里。

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
  "llama_cpp_defaults":  { /* … */ },
  "llama_cpp_targets":   [ /* … */ ],

  // 对外接口（每项独立监听）
  "ollama_interfaces":   [ /* … */ ],
  "api_interfaces":      [ /* … */ ],

  // 每模型 profile
  "model_profiles":      { /* … */ }
}
```

### Sources（远端与本机的模型来源）

四种 source，每一项有唯一 `name`（**不能包含 `@`**，且 4 张表里的 name 不能冲突）。每个 source 的 `models` 是 `ModelEntry` 列表，每条只表达「上游真实模型名 + 可选 alias」：

```jsonc
{
  "name": "deepseek_remote",
  "base_url": "https://api.deepseek.com",
  "auth_token": "sk-...",
  "models": [
    { "name": "deepseek-chat" },                            // 公开 ID 默认 = "deepseek-chat@deepseek_remote"
    { "name": "deepseek-reasoner", "alias": "deepseek-r1" } // 公开 ID = "deepseek-r1"；wire 还是 "deepseek-reasoner"
  ]
}
```

- **公开 ID（public_id）规则**：当 `alias` 为空时是复合 `name@source.name`；当 `alias` 非空时**就是** `alias`（不再带 `@target` 后缀）。alias 必须全局唯一，不能包含 `@`，也不能与其他 source 的 `name` / `alias` 冲突。
- **wire 名**：实际发给上游时永远是 `ModelEntry.name`，alias 只影响对外可见 ID。
- **`model_map` 已移除**：旧版的「显示名 → 上游真实名」映射改用 alias 直接挂在 models 列表里。

四张表的差异：

| 表名 | 协议 | 用途 |
| --- | --- | --- |
| `anthropic_upstreams` | Anthropic Messages | 远端 Anthropic 兼容 API（含 DeepSeek Anthropic 通道、claude-relay-service 等） |
| `openai_upstreams`    | OpenAI Chat Completions | 远端 OpenAI 兼容 API（DeepSeek OpenAI 通道、Together、Groq、自建网关等） |
| `ollama_targets`      | Ollama `/api/chat` | 本机或局域网 Ollama daemon；可生命周期接管 |
| `llama_cpp_targets`   | OpenAI Chat Completions（llama.cpp server） | 一项 = 一个 llama.cpp server 进程 / 一个模型 / 一个端口 |

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

启动时会做循环检测：如果 `base_url` 的 host:port 与本进程的某个 `ollama_interfaces[*]` / `api_interfaces[*]` / `admin` 监听重合，会直接报 `cycle detected` 拒绝启动。

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

- `auto_start: false`：fake-ollama 只转发，不接管 daemon。
- `auto_start: true`：health check 失败时执行 `start_command`；`stop_command` / `idle_timeout_seconds` 控制空闲回收。
- 模型名匹配按 Ollama 规则：`foo` 与 `foo:latest` 等价；省略 tag 只代表 `latest`。
- 每个本地模型的显存估算在 `model_profiles[*].estimated_vram_gb` 里配置。请求触发模型加载前，fake-ollama 调用 `nvidia-smi` 检查可用显存；不足时按 LRU 卸载其他空闲超过 60 秒的本地模型，每轮重新读真实 free VRAM。

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
    "cwd": "I:\\Projects\\llama.cpp",
    "binary_path": "I:\\Projects\\llama.cpp\\llama-server.exe",
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

`LlamaCppTarget.alias` 语义：当 `alias` 非空时，**它就是公开 display name**（替换 `model`），仍然作为模型在白名单里的 display；发给 llama.cpp 的 wire name 始终是 `model` 字段。

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
        { "model": "llama3.1",  "target": "local" },
        { "model": "qwen3.6-27b-iq4xs", "target": "qwen3.6-27b-iq4xs", "alias": "qwen-vision" }
      ]
    }
  ]
}
```

字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | string | 接口唯一标识，仅用于日志 / Admin UI |
| `host` | string | 监听 host（`127.0.0.1` 或 `0.0.0.0`） |
| `port` | int | 监听端口；启动时若与其他接口或 admin 端口冲突会报错 |
| `access_tokens` | string[] | 该接口的 token 池；空 = 该接口不鉴权 |
| `exposed_models` | array of `ExposureEntry` | 该接口允许看到 / 请求的模型 |

每个 `ExposureEntry` 字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `model` | string | source 里的 `ModelEntry.name`，必须与某个 source 的 `models[*].name` 完全匹配 |
| `target` | string | source 的 `name`，决定路由到哪 |
| `alias` | string? | 可选；非空时**作为该 exposure 的公开 ID**，替换 `model@target` 复合形式 |

- 公开 ID 优先级：`exposure.alias` > `model_entry.alias` > `"{model}@{target}"`
- 客户端必须用「公开 ID」请求；裸 model 名（无 `@`）会在做 tagless 匹配后路由（见下），仍找不到时 400。
- 同一个模型可以在多个接口里同时暴露，alias 可以不同。
- `/api/tags`（Ollama）与 `/v1/models`（OpenAI）只列出当前接口的公开 ID。
- mismatch 警告：Admin UI 会在你勾的 `exposed_models` 引用了不存在的 source / model 时高亮提示。

#### Tagless 回退匹配

为了兼容客户端把 `foo:latest` 写成 `foo` 的常见情况，请求 `model` 形如 `foo@target` 时，若没有精确匹配的 exposure，会在「该接口里 target 相同且未设 alias 的 exposure」里按 Ollama 规则（`foo` ≡ `foo:latest`）二次匹配，命中即路由。设了 `alias` 的 exposure 不参与该回退（alias 视为强精确）。

### 共享设置

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `default_max_tokens` | int | `4096` | 客户端没传时的默认输出上限 |
| `timeout_seconds` | float | `300` | 所有出站 HTTP 请求超时 |
| `use_system_proxy` | bool | `false` | 出站是否走系统代理 |
| `enforce_context_limit` | bool | `true` | 估算 token 超 `context_length` 直接 400 |
| `advertised_version` | string | `0.6.4` | 仅用于 Ollama 接口的 `/api/version` |
| `model_profiles` | object | `{}` | 每模型 capabilities / 上下文 / 思维链等。key 三级回退：`model@target` > 裸 `model` > tagless base |

### Admin UI / Dashboard

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `admin_enabled` | bool | `true` | 是否注册 `/admin` |
| `admin_host` | string | `127.0.0.1` | admin listener 地址（**无内置鉴权**） |
| `admin_port` | int | `21433` | admin listener 端口 |
| `dashboard_enabled` | bool | `true` | 是否启用资源监控 dashboard |
| `dashboard_host` / `dashboard_port` | | `127.0.0.1` / `21432` | dashboard listener |
| `dashboard_sample_interval_seconds` | float | `10.0` | 采样间隔 |
| `dashboard_retention_seconds` | float | `604800.0` | 历史保留窗口 |
| `dashboard_data_path` | string | `logs/dashboard_history.json` | 历史落盘路径 |
| `dashboard_model_reclaim_enabled` | bool | `true` | dashboard 是否显示模型回收 |
| `vram_low_free_reclaim_enabled` | bool | `true` | 检测显存低水位时主动回收 |
| `vram_low_free_threshold_mib` | float | `200.0` | 低水位阈值（MiB） |

## 内部 backends 视图

`Settings` 之上有一份协议无关的 `backends` 列表，路由代码用它统一查找：

```python
settings.backends
# -> List[Backend]，每项 (name, protocol, kind, base_url, source)
#    protocol: "anthropic" | "openai" | "ollama"
#    kind:     "remote" | "local"
#    source:   底层 AnthropicUpstream / OpenaiUpstream / OllamaTarget / LlamaCppTarget

settings.resolve_request("deepseek-r1", interface_name="api")
# -> 按某个接口的视角解析「客户端请求 model」到 (backend, ModelEntry)
#    会查 exposed_models（含 tagless 回退），找不到返回 None
```

`resolve_request` 取代了旧版 `backend_for("model@target", surface=...)`：从「surface=internal/external」改成显式传 `interface_name`，因为接口现在是数组。

## 反向代理：把本地模型当远端 API 用

适合**只支持 Anthropic 或 OpenAI API** 的客户端调用本机 Ollama / llama.cpp 模型，或把上游 Anthropic 做带鉴权的转发壳。

只要在某个 `api_interfaces[*].exposed_models` 里勾上对应模型（指向 `ollama_targets` 或 `llama_cpp_targets`），并设置好 `access_tokens`，反向代理 `POST /v1/messages`、`POST /v1/messages/count_tokens` 与 `POST /v1/chat/completions` 就开门工作：

- 命中 `ollama_targets[*].models[*]` → 转换为 Ollama `/api/chat`，再翻译回 Anthropic / OpenAI 响应。
- 命中 `llama_cpp_targets[*]` → 调用 llama.cpp `/v1/chat/completions`；Anthropic 入口做格式转换，OpenAI 入口基本直通。
- 命中 anthropic_upstreams / openai_upstreams 提供的模型 → 透传到该上游（带鉴权壳）。
- 未命中 / 未在白名单 → 404 `model '...' is not exposed on interface 'xxx'`。

`POST /v1/messages/count_tokens` 命中本地 target 时用本地估算；命中外露 upstream 时透传。

支持的转换：

- 文本消息、`system`、`max_tokens` / `temperature` / `top_p` / `top_k` / `stop_sequences`
- **工具调用**：`tools` + `tool_use` + `tool_result`（流式与非流式都支持）
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
- 上游不支持图片时（如 DeepSeek text-only）会返回错误，fake-ollama 不预拦截。

## Web 配置编辑器（/admin）

启动后浏览器打开 <http://127.0.0.1:21433/admin>。左侧侧栏按 「Forward Proxy / Reverse Proxy / Shared Settings / Admin UI」 分块：

- 每个字段左侧复选框 = 「是否写入 JSON」；取消勾选 = 回退默认值。
- `anthropic_upstreams` / `openai_upstreams` / `ollama_targets` / `llama_cpp_targets` / `ollama_interfaces` / `api_interfaces` / `model_profiles` 都是可重复组，自带 +add / Remove。
- 视觉上 Ollama 和 llama.cpp source 卡片用不同强调色和图标分组，方便区分。
- **Detect models**：upstream / Ollama 卡片支持探测后合并到 `models`；llama.cpp 卡片支持探测后写入单个 `model`。
- **Exposure**：每个接口卡片里的 `exposed_models` 是结构化表格，`model` / `target` 都带 datalist 自动补全；mismatch（引用了不存在的 source / model）会高亮。
- **access_tokens**：每行带 Show / Generate；列表底部有 "+ generate token"。
- 顶部按钮：**Save & Reload** / **Discard & Reload from disk** / **Toggle raw JSON**。
- 校验失败会显示 Pydantic 详细错误，配置不会被写入。

`/admin` 只挂在独立 admin listener 上，**不会**被任何 `ollama_interfaces` / `api_interfaces` 暴露。

## 测试

```powershell
pip install -e ".[test]"

# 离线单测
pytest

# live 集成测试（需要 .env 里有 FAKE_OLLAMA_TEST_BASE_URL / FAKE_OLLAMA_TEST_AUTH_TOKEN）
pytest -m integration
```

`tests/conftest.py` 在缺少凭据时自动跳过 `@pytest.mark.integration` 用例。

## 手动验证示例

```powershell
# Ollama 接口（默认 21434）
curl http://127.0.0.1:21434/api/tags
curl -X POST http://127.0.0.1:21434/api/chat `
  -H "Content-Type: application/json" `
  -d '{"model":"claude-3-5-sonnet-20241022@default","stream":false,"messages":[{"role":"user","content":"hi"}]}'

# API 接口（默认 21435，需 token）
curl http://127.0.0.1:21435/v1/models -H "x-api-key: tk-..."
curl -X POST http://127.0.0.1:21435/v1/messages `
  -H "Content-Type: application/json" -H "x-api-key: tk-..." `
  -d '{"model":"llama3.1","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'

# llama.cpp OpenAI 直通（同一个 API 接口）
curl -X POST http://127.0.0.1:21435/v1/chat/completions `
  -H "Content-Type: application/json" -H "x-api-key: tk-..." `
  -d '{"model":"qwen-vision","stream":false,"messages":[{"role":"user","content":"hi"}]}'
```

## 请求数据日志

`logs/fake_ollama.log` 只放运行日志；完整请求 / 响应数据默认写到 `logs/fake_ollama.requests.jsonl`。

每行是一条 JSON 事件，核心字段：

- `request_id`：同一次入口请求贯穿入口 HTTP、后端请求、响应 chunk、返给 agent 的 chunk。
- `event`：`http_request_start` / `http_request_body` / `backend_request` / `backend_response_body` / `http_response_body` / `http_request_end` / `backend_error`。
- `body`：完整请求体 / 响应体 / 流式 chunk。文本 UTF-8，非文本 base64。
- `headers`：`Authorization` / `x-api-key` / cookie 等敏感头只留 sha256 指纹。

只记录 `/api/*` 与 `/v1/*`，不记录 `/admin/*`。默认 100MB × 10 个文件轮转。

```powershell
python -m fake_ollama --request-data-log-file I:\path\requests.jsonl
python -m fake_ollama --no-request-data-log
```

## 安全提示

- `.env` / `config.json` 已加入 `.gitignore`，不要提交真实 token。
- `logs/fake_ollama.requests.jsonl*` 会包含 prompt、工具参数、模型输出、图片 base64，应按敏感数据处理。
- 所有接口默认 `127.0.0.1`。要对外暴露：要么把对应实例的 `host` 改 `0.0.0.0`，要么保持 `127.0.0.1` + Nginx/Caddy 反代加 TLS。
- **`/admin` 无任何鉴权**，默认绑 `127.0.0.1`。若改成 `0.0.0.0` 必须自己加一层鉴权反代，或 `"admin_enabled": false`。
- 反向代理 token：建议用 Web UI 的 Generate 生成 ≥24 字节随机串，不要复用其他系统 token。Token 池可放多个，便于按客户端轮换。

## 故障排查

- **`/v1/messages` 返回 404 `model '...' is not exposed on interface 'xxx'`**：该模型不在该 `api_interfaces[*].exposed_models` 里。去 admin UI 勾选。
- **`/api/chat` 返回 400 `unknown model '...'`**：客户端传的 model 不匹配任何公开 ID，也不在 tagless 回退里。先请求 `/api/tags` 获取当前接口的可用列表。
- **返回 401**：该接口 `access_tokens` 非空，请求头里没带或带错 token。检查 `x-api-key` / `Authorization: Bearer`。
- **启动报 `cycle detected`**：某个 `anthropic_upstreams[*].base_url` 指向了 fake_ollama 自己的某个监听端口。改 URL 或删该 upstream。
- **启动报端口冲突**：两个 interface（或与 admin / dashboard）共用了同一个 host:port。
- **找不到日志文件**：默认相对 CWD：`logs/fake_ollama.log` 与 `logs/fake_ollama.requests.jsonl`。在项目根目录启动，或显式 `--log-file` / `--request-data-log-file`。
- **502 / 连不上上游**：`httpx` 默认会读 Windows 系统代理。装了 Clash / V2Ray 且上游是直连 IP 时，保持 `use_system_proxy: false`。
- **503 `Insufficient GPU VRAM`**：某个本地模型在 `model_profiles` 里配了 `estimated_vram_gb`，但 `nvidia-smi` 显示当前可用显存不足；fake-ollama 已尝试回收空闲超过 60 秒的模型仍不够。降低估算、等模型空闲、配 `stop_command`，或手动释放显存。
- **400 thinking content must be passed back**（DeepSeek）：某轮启用 thinking 但下一轮历史没把 `thinking` 块带回。fake-ollama 已做缓存回查 + auto+show_thinking=false 时注入 disabled；要 thinking 请显式 `"thinking_mode": "enabled"`。

## 参考文档

- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages) / [流式](https://docs.anthropic.com/en/docs/build-with-claude/streaming)
- [Ollama API 文档](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
- [DeepSeek Anthropic API 兼容性](https://api-docs.deepseek.com/zh-cn/guides/anthropic_api)
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
