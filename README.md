# fake_ollama

一个轻量的协议适配层，主要做两件事：

1. **正向**（Ollama 兼容入口 → Anthropic 上游）：把 **Anthropic Messages API** 兼容的上游（官方 / DeepSeek / 自建网关 / claude-relay-service）伪装成一台本机 **Ollama** 服务，让只支持 Ollama 协议的客户端（GitHub Copilot 自定义 provider、IDE 插件、桌面 AI 软件）无缝调用 Claude / DeepSeek 等模型。
2. **反向**（Anthropic / OpenAI 兼容入口 → 本机模型服务）：把本机的 **Ollama** 或 **llama.cpp server** 包装成 **Anthropic Messages API**（`POST /v1/messages`）和 OpenAI Chat Completions（`POST /v1/chat/completions`），让只支持远端 API 的客户端也能调用本地大模型。

附带一个零依赖的 Web 配置编辑器（`/admin`），不必再手改 JSON。

## 三端口架构

配置页面、对内 Ollama 兼容入口、对外反向代理的安全边界不同，因此监听被拆成三个端口：

```
┌─────────────────────────────────────────────────────────┐
│  fake_ollama 进程（asyncio.gather 同时跑多个 uvicorn）  │
│                                                         │
│  admin listener     (默认 127.0.0.1:21433)              │
│    /admin/*     Web 配置编辑器（无内置鉴权）            │
│                                                         │
│  internal listener  (默认 127.0.0.1:21434)              │
│    /                                                    │
│    /api/*       Ollama 兼容（正向；本机消费）           │
│    /v1/chat/completions  /v1/embeddings                 │
│                                                         │
│  external listener  (可选；推荐 127.0.0.1:21435)        │
│    /v1/messages Anthropic 兼容（反向；可对外）          │
│    /v1/models                                           │
│    /v1/chat/completions  OpenAI 兼容（反向；可对外）    │
│    必须带 external_access_tokens 之一                   │
└─────────────────────────────────────────────────────────┘
```

- **admin listener**（默认启用）：只服务 `/admin/*`。配置页面没有内置鉴权，所以默认独立绑定 `127.0.0.1:21433`。不需要 UI 可设 `admin_enabled: false`。
- **internal listener**（必开）：服务正向调用方（本机 IDE / Copilot）。生产环境**保持 `127.0.0.1`**。
- **external listener**（可选）：单独承载反向代理 `/v1/messages`、`/v1/models`，以及反向模式的 `/v1/chat/completions`。`external_port` 设为 `null` 时，反向 `/v1/*` API 跟随 internal 端口（适合纯本机使用）；admin 仍默认在独立 admin 端口。
- 设置 `admin_port` 后，`/admin/*` **只在 admin 端口**可达；internal/external 端口都返回 404。把 `admin_port` 设为 `null` 才会恢复旧的“/admin 挂在 internal 端口”模式，不推荐。
- 设置 `external_port` 后，`/v1/messages` 与 `/v1/models` **只在 external 端口**可达；它们在 internal/admin 端口返回 404。`/v1/chat/completions` 在 internal/external 两个 API 端口都可达，但按端口分流：internal 端口走 upstream API，external 端口命中 `ollama_targets` / `llama_cpp_targets` 时走本机模型服务。
- 想让别的机器访问反向代理：把 `external_host` 改 `0.0.0.0`，或者保持 `127.0.0.1` + 在前面挂 Nginx/Caddy（推荐，可以加 TLS / 限流 / 客户端证书）。

## 特性一览

- **多上游路由**：把 Anthropic / DeepSeek / 自建网关合并到同一个 Ollama 端口
- **每模型 profile**：capabilities / 上下文长度 / 思维链开关 / 输出上限
- **每个模型可控外露**（`expose_external`，upstream / `ollama_targets` / `llama_cpp_targets` 都支持）：某些模型只想本机用、不想出现在反向代理 `/v1/models` 里？勾上即可
- **本地 target 生命周期接管**：Ollama / llama.cpp 都可配置 health check、按需启动脚本、启动超时、空闲回收；不配置时就只转发到你单独启动的服务
- **本地显存预检**：本地 Ollama / llama.cpp 模型可在 `model_profiles` 里填写 `estimated_vram_gb`；启动或加载前会用 `nvidia-smi` 评估可用显存，并尝试回收空闲本地模型
- **集中式访问 token**（`external_access_tokens`）：一个 token 池统一鉴权 external 端口上的 `/v1/messages`、`/v1/messages/count_tokens`、`/v1/models` 与 `/v1/chat/completions`
- **图片输入**：自动嗅探 base64 magic bytes（PNG/JPEG/GIF/WEBP），不再硬编码 `image/png`
- **零依赖 Web 编辑器**：字段说明 / 默认值回退 / 上游 detect-models / models 与 model_profiles key 自动补全
- `pytest` + `httpx.MockTransport` 离线单测
- **网络错误安全降级**：上游连接断开 / 超时等非 HTTP 错误统一返回 502 / 流式错误帧，不会抛出未处理异常

## 快速开始

```powershell
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 准备配置
Copy-Item config.json.example config.json
Copy-Item .env.example .env
# 编辑 config.json 填上游；token 推荐放 .env

# 3. 启动（已激活 .venv 时）
python -m fake_ollama
# 未激活 .venv 时，显式使用虚拟环境里的 Python
.\.venv\Scripts\python.exe -m fake_ollama
# 如果已经 pip install -e .，也可以用 console script
fake-ollama
# 可选：覆盖 internal listener 的 host/port
python -m fake_ollama --config ./config.json --host 127.0.0.1 --port 21434
# 可选：覆盖 admin listener
python -m fake_ollama --config ./config.json --admin-host 127.0.0.1 --admin-port 21433
```

说明：
- `python -m fake_ollama` 和 `.\.venv\Scripts\python.exe -m fake_ollama` 调的是同一个入口；前者依赖当前 shell 里的 `python` 已指向 `.venv`，后者直接指定虚拟环境解释器，更适合未激活环境、计划任务或服务脚本。
- 未执行 `pip install -e .` 时，请在项目根目录启动；这样源码包、`./config.json`、`./.env` 和默认日志路径都能按预期解析。
- 默认日志写到当前工作目录下的 `logs/fake_ollama.log`，同时输出到控制台。可用 `--log-file I:\path\fake_ollama.log` 指定固定日志路径，或用 `--no-log-file` 只保留控制台日志。

启动后：
- Ollama 客户端连 `http://127.0.0.1:21434`
- 浏览器打开 <http://127.0.0.1:21433/admin> 用 Web UI 调配置
- Anthropic 客户端连 **external listener**：`http://127.0.0.1:21435/v1/messages`（默认）

## 配置（config.json）

加载顺序（后者覆盖前者）：

1. 代码默认值
2. `config.json`（路径优先级：`--config` > `$FAKE_OLLAMA_CONFIG` > `./config.json`）
3. 环境变量 / `.env`（仅覆盖顶层标量；推荐只用来塞 token）

完整示例见 [config.json.example](./config.json.example)；下面只列结构。

### 顶层字段

配置页和示例 `config.json` 现在按四个层次排：

- **正向代理**：远端 Anthropic API -> 本机 Ollama 兼容入口
- **反向代理**：本机 Ollama / llama.cpp -> 对外 Anthropic / OpenAI 兼容 API
- **共享设置**：两条链路都会用到的运行默认值和模型 profile
- **Admin UI**：只影响 `/admin` 配置页面本身

#### 正向代理：远端 API -> 本机 Ollama

##### 本机 Ollama 兼容入口

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `host` | string | `127.0.0.1` | **internal** listener 地址（`/api/*`、`/v1/chat/completions`；OpenAI 端走 upstream） |
| `port` | int | `21434` | internal listener 端口 |
| `advertised_version` | string | `0.6.4` | **仅用于 Ollama 兼容入口**的 `/api/version` 返回值；不影响 `/v1/*` |

##### 远端上游

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `upstreams` | array | — | **必填**，至少一个 Anthropic 兼容上游（见下） |

#### 反向代理：本机模型 -> 对外 API

##### 对外 API listener 与鉴权

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `external_host` | string \| null | `null` | **external** listener 地址（`/v1/messages`、`/v1/models`、`/v1/chat/completions`；OpenAI 端命中本地 target 时走本机模型服务）。`null` = 不开独立 external 端口，反向 `/v1/*` API 跟随 internal 端口 |
| `external_port` | int \| null | `null` | external listener 端口 |
| `external_access_tokens` | string[] | `[]` | 访问 external `/v1/*` 反向代理需要的 token 池（`x-api-key` 或 `Authorization: Bearer`）。Web UI 可一键 Generate |

##### 本地模型 targets

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `ollama_targets` | array | `[]` | 反向代理的本机或远端 Ollama 服务（见下） |
| `llama_cpp_defaults` | object | `{}` | `llama_cpp_targets` 的全局默认值；target 同名字段可覆盖 |
| `llama_cpp_targets` | array | `[]` | 反向代理的 llama.cpp server（OpenAI 兼容，见下） |

#### 共享设置

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `default_max_tokens` | int | `4096` | 客户端没传 `num_predict` 时的默认 |
| `timeout_seconds` | float | `300` | 所有出站 HTTP 请求超时（上游与本地 target 都会用到） |
| `use_system_proxy` | bool | `false` | 所有出站 HTTP 请求是否走系统代理（Clash/V2Ray 用户保持 false） |
| `enforce_context_limit` | bool | `true` | 估算 token 超过 `context_length` 直接 400，避免误传巨 prompt |
| `model_profiles` | object | `{}` | 正向 / 反向共用的每模型 capabilities / 上下文 / 思维链等设置 |

#### Admin UI

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `admin_enabled` | bool | `true` | 是否注册 `/admin` Web UI |
| `admin_host` | string | `127.0.0.1` | **admin** listener 地址（`/admin/*`；无内置鉴权，建议保持 localhost） |
| `admin_port` | int \| null | `21433` | admin listener 端口。`null` = 把 `/admin` 挂回 internal 端口（旧行为，不推荐） |

#### 鉴权与端口的几种组合

| `external_port` | `external_access_tokens` | 行为 |
| --- | --- | --- |
| `null` | `[]` | 单端口；`/v1/*` 不需 token（**最宽松**，仅推荐绑 `127.0.0.1` 时） |
| `null` | 非空 | 单端口；`/v1/*` 需 token |
| 设置 | 非空 | external 独立端口；`/v1/*` 仅在 external 端口暴露并需 token（**推荐**） |
| 设置 | `[]` | external 独立端口；`/v1/*` 拒绝所有请求（启动时 WARN） |

### upstreams（正向：Ollama → Anthropic 上游）

每项是一个独立的 Anthropic 兼容端点：

```jsonc
{
  "name": "anthropic",                              // 唯一名字
  "base_url": "https://api.anthropic.com",           // 上游 base URL
  "auth_token": "sk-ant-...",                        // 鉴权 token（也可走 ANTHROPIC_AUTH_TOKEN）
  "models": ["claude-3-5-sonnet-20241022",           // 本机 Ollama 兼容入口可见、并路由到该 upstream 的显示名
             "claude-3-5-haiku-20241022"],
  "expose_external": ["claude-3-5-haiku-20241022"],  // 可选：哪些模型出现在反向代理 /v1/models
  "model_map": {                                     // 可选：显示名 → 上游真实模型 ID
    "claude-sonnet": "claude-3-5-sonnet-20241022"
  }
}
```

路由规则：
- `/api/tags`（正向 Ollama 端）：返回所有 upstream `models` 的并集（去重，保留首次出现位置）。**不**受 `expose_external` 限制——本机使用时全可见。
- 一次正向请求按 `model` 字段查找：第一个 `models` 中包含该名字的 upstream 胜出；没匹配回退第一个 upstream。
- `/v1/models`（反向 Anthropic 端）与 `/v1/messages` 透传：**只**返回 / 接受被 `expose_external` 允许的上游模型 + 被 `expose_external` 允许的本地 target 模型。

`expose_external` 语义：
- **不写该字段**（或为 `null`）：等同于"全部允许"，保持向后兼容。
- upstream / `ollama_targets` 使用列表：空数组 `[]` 表示全部隐藏；显式列表表示仅列出的模型对外暴露。
- `llama_cpp_targets` 因为每个 target 只有一个 `model`，所以使用布尔值：`true` 暴露，`false` 隐藏；target 不写则继承 `llama_cpp_defaults.expose_external`，全局也不写时按旧行为暴露。

模型名匹配按 Ollama 规则处理：`model` 与 `model:latest` 等价；省略 tag 只代表 `latest`，不会匹配其他显式 tag（例如 `model:q4_K_M` / `model:q2_k_p`）。

### ollama_targets（反向：Anthropic → 本机 Ollama）

```jsonc
{
  "name": "local",                                  // 唯一名字
  "base_url": "http://127.0.0.1:11434",              // 本机 Ollama URL
  "models": ["llama3.1", "qwen2.5-coder"],          // Anthropic 端可见的显示名
  "expose_external": ["llama3.1"],                  // 可选：哪些模型对外暴露（同 upstream 语义）
  "model_map": { "llama3.1": "llama3.1:8b" },       // 可选：显示名 → Ollama tag

  "auto_start": false,                               // true 时 health 失败会执行 start_command
  "start_command": null,                             // Windows 示例："\"C:\\Users\\a\\AppData\\Local\\Programs\\Ollama\\ollama.exe\" serve"
  "stop_command": null,
  "idle_timeout_seconds": null,                      // 通常留空，让 Ollama 自己卸载模型
  "startup_timeout_seconds": 60,
  "health_path": "/api/version",
  "cwd": null
}
```

ollama_targets 默认全部对外暴露（不写 `expose_external`，保持向后兼容）；想做"本机才能用"的隔离就把 `expose_external` 勾上并留空，或只勾选允许暴露的子集。访问鉴权统一走顶层的 `external_access_tokens`，**不再有 per-target 的 `api_token` 字段**——旧版 `api_token` 在加载时会被自动 hoist 到 `external_access_tokens` 并打 WARN，存盘后清理。

Ollama target 有两种运行方式：
- **单独运行模式**：`auto_start: false`（默认）。fake-ollama 只负责转发；Ollama Desktop / system service / 你自己的脚本负责 daemon 生命周期。
- **接管启动模式**：`auto_start: true` 且配置 `start_command`。fake-ollama 在请求到来且 `health_path` 不通时执行启动命令。`idle_timeout_seconds` 留空时不主动停止 daemon；填了以后只会停止 fake-ollama 自己启动的进程，或执行你配置的 `stop_command`。
- 进程开始关闭后（例如 Ctrl+C），fake-ollama 会拒绝新的请求工作并禁止再执行 `start_command`，避免在 shutdown drain 阶段把 Ollama 重新拉起。
- 每个本地模型的显存估算在 `model_profiles[模型名].estimated_vram_gb` 里配置，而不是在 target 上配置。请求命中该模型时，fake-ollama 会在触发本地模型加载前调用 `nvidia-smi` 汇总当前可用 GPU 显存。
- 如果显存不足，会找出其他已填写 `estimated_vram_gb`、当前无请求、且上次使用在 60 秒之前的本地模型，按最久未使用优先尝试释放；Ollama 会先用 `/api/generate` + `keep_alive: 0` 卸载单个模型，失败时再按可控进程 / `stop_command` 停掉整个 target。每次释放后都会重新读取 `nvidia-smi` 的当前可用显存，只有真实 free VRAM 达标才继续启动新模型。

请求头形式（与 Anthropic 官方一致）：

```
x-api-key: tk-...
# 或
Authorization: Bearer tk-...
```

详见下文「[反向代理](#反向代理把本地-ollama--llamacpp-当远端-api-用)」。

### llama_cpp_targets（反向：Anthropic / OpenAI → llama.cpp server）

llama.cpp server 自带 OpenAI 兼容接口，fake-ollama 会把多个 server 实例聚合到 external listener。`llama_cpp_targets` 是数组，但**每一项只代表一个模型、一个 llama.cpp server 进程、一个端口和一组启停脚本**；要配置多个 llama.cpp 模型，就新增多个 target，分别设置不同的 `base_url` / 端口、`start_command` / `stop_command`，以及唯一的 `model`。

`llama_cpp_defaults` 可以集中配置所有 target 的通用生命周期参数；target 里写了同名字段就覆盖全局值。`name` 仍然是 target 级标识，用于内部 client 字典、日志和去重，但通常不用写；不填时自动等于 `model`，它不是全局配置。

```jsonc
{
  "llama_cpp_defaults": {
    "expose_external": true,
    "auto_start": true,                           // health 失败时按需启动
    "idle_timeout_seconds": 1800,                 // 留空表示不做空闲回收
    "startup_timeout_seconds": 600,
    "health_path": "/health",
    "cwd": "I:\\Projects\\llama.cpp"
  },
  "llama_cpp_targets": [
    {
      "base_url": "http://127.0.0.1:21436",
      "auth_token": "",                          // 可选：llama.cpp --api-key
      "model": "qwen3.6-27b-hauhau-q2kp",         // 必填；一个 target = 一个模型进程
      "model_alias": null,                        // 可选：发送给 llama.cpp 的 --alias / OpenAI model ID
      "start_command": "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"I:\\Projects\\llama.cpp\\start_qwen36_server.ps1\" -ListenHost 127.0.0.1 -Port 21436",
      "stop_command": null                        // 可选；未填时只停止 fake-ollama 自己启动的进程
    }
  ]
}
```

路由行为：
- `POST /v1/chat/completions` 命中 `llama_cpp_targets[*].model` 时，基本直通 llama.cpp 的 OpenAI API，只把显示名映射到 `model_alias` 后的真实 alias。
- `POST /v1/messages` 命中同一模型时，会把 Anthropic Messages 请求转换成 OpenAI Chat Completions，再把响应转换回 Anthropic JSON / SSE。
- `llama_cpp_targets[*].model` 是单个字符串；旧版 `models: ["..."]` 会在读取时兼容，但新配置请不要再写列表。原因是 llama.cpp server 进程本身加载单个模型，端口、启动脚本、停止脚本和显存回收都应跟这个模型一一对应。
- 生命周期有两种模式：`auto_start: false` 是单独运行模式，fake-ollama 不接管进程；`auto_start: true` + `start_command` 是接管模式，fake-ollama 负责按需唤起该模型对应的 server 进程。`idle_timeout_seconds` 只在你配置后启用；没有 `stop_command` 时，fake-ollama 不会杀掉它没有亲自启动的外部 llama.cpp 进程。Windows 上由 fake-ollama 启动的 PowerShell/cmd 包装进程会按进程树停止，避免只结束 shell 而留下 `llama-server` 子进程。进程开始关闭后同样会禁止再执行 `start_command`。
- llama.cpp 模型同样从 `model_profiles[模型名].estimated_vram_gb` 读取显存估算。显存不足时，fake-ollama 可以停止空闲超过 60 秒、且由自己启动或配置了 `stop_command` 的其他 llama.cpp 模型 target；llama.cpp 没有通用的单模型卸载 API，所以回收单位就是该模型对应的 target 进程。释放后会重新读取 `nvidia-smi`，不会用 `estimated_vram_gb` 累加值当作真实可用显存。

#### Copilot 走 upstream 调用 Ollama 模型

如果希望 GitHub Copilot 仍然只连接 internal Ollama 兼容入口（`/api/chat`），但实际模型在本机或另一台机器的 Ollama 上，可以把一个 `upstreams` 项指向 fake_ollama 的 **external listener**：

- 同机部署：`upstreams[*].base_url = "http://127.0.0.1:<external_port>"`
- 跨机器部署：`upstreams[*].base_url = "http://<ollama-host>:<external_port>"`
- `upstreams[*].auth_token` 填目标 fake_ollama 的 `external_access_tokens` 之一
- 目标 fake_ollama 的 `ollama_targets[*].models` 或 `llama_cpp_targets[*].model` 包含这些模型

这样请求路径仍然是 upstream 语义：Copilot → internal `/api/chat` → upstream `/v1/messages` → 目标机器 Ollama / llama.cpp。不要为了图片能力把 internal `/api/chat` 直接改成本机 target，否则跨机器部署会失去这一层转发边界。

### model_profiles

key 是模型显示名，value 是该模型的元数据。GitHub Copilot 等客户端会读 `/api/show` 的 `capabilities` 决定模型在 UI 中是否可选、能否做 tool-calling / 视觉。

```jsonc
{
  "claude-3-5-sonnet-20241022": {
    "capabilities": ["completion", "tools", "vision"],
    "context_length": 200000,
    "max_output_tokens": 8192
  },
  "deepseek-v4-pro": {
    "capabilities": ["completion", "tools"],
    "context_length": 128000,
    "max_output_tokens": 8192,
    "thinking_mode": "enabled",
    "thinking_budget_tokens": 1024,
    "show_thinking": true
  },
  "llama3.1": {
    "capabilities": ["completion", "tools"],
    "context_length": 131072,
    "estimated_vram_gb": 6.0
  }
}
```

字段说明：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `capabilities` | `["completion","tools","vision"]` | 子集自这三者；**至少要有 `completion`**，否则 Copilot 会过滤掉 |
| `context_length` | `200000` | 总上下文 token 上限。服务端会做拦截（输入估算 + `max_tokens` 超过 → 400） |
| `max_output_tokens` | `null` | 可选；同时是 `max_tokens` 的硬上限 |
| `estimated_vram_gb` | `null` | 可选；本地 Ollama / llama.cpp 模型加载后预计占用的 GPU 显存（GB），用于启动前预检和空闲模型回收 |
| `thinking_mode` | `auto` | `auto` / `enabled` / `disabled`，控制是否注入 `thinking` 字段（仅 reasoning 模型有效） |
| `thinking_budget_tokens` | `1024` | `enabled` 时的预算（DeepSeek 会忽略） |
| `show_thinking` | `true` | 是否把上游 thinking 透传给客户端：用 `<think>...</think>` 包裹接到正文前面，并在 Ollama 响应里附 `message.thinking`、OpenAI 流式增量里附 `reasoning_content` |

> token 估算用 `字符数 / 3` 的保守启发式（中英都偏高估），目的是宁可早拦也不漏拦——它**不**保证与上游计费完全一致。如需关闭拦截设 `enforce_context_limit: false`。

> Web 编辑器里 `upstreams[*].models`、`ollama_targets[*].models`、`llama_cpp_targets[*].model` 与添加 `model_profiles` 时的 key 输入框都带 **HTML5 datalist 自动补全**。候选来自当前配置里的模型名、已探测到的模型名，以及已有 `model_profiles` key。

### 环境变量

`config.json` 的顶层标量都可被 `FAKE_OLLAMA_*` 覆盖（路径用 `FAKE_OLLAMA_CONFIG`），但**推荐只用 env 放敏感 token**：

| 变量 | 等价 |
| --- | --- |
| `FAKE_OLLAMA_CONFIG` | 同 `--config` |
| `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN` | 兼容旧版：自动建一个名为 `default` 的 upstream（如 JSON 已有同名 upstream，env 字段覆盖之） |
| `FAKE_OLLAMA_HOST` / `_PORT` | 覆盖 internal listener |
| `FAKE_OLLAMA_ADMIN_HOST` / `_ADMIN_PORT` | 覆盖 admin listener |
| `FAKE_OLLAMA_EXTERNAL_HOST` / `_EXTERNAL_PORT` | 覆盖 external listener |
| `FAKE_OLLAMA_EXTERNAL_ACCESS_TOKENS` | CSV 列表，覆盖 `external_access_tokens` |
| `FAKE_OLLAMA_DEFAULT_MAX_TOKENS` / `_TIMEOUT` / `_USE_SYSTEM_PROXY` / `_ENFORCE_CONTEXT_LIMIT` / `_ADVERTISED_VERSION` | 同名标量覆盖；其中 `_ADVERTISED_VERSION` 只影响 Ollama 兼容 `/api/version` |

## 反向代理：把本地 Ollama / llama.cpp 当远端 API 用

适合**只支持 Anthropic Messages API** 的客户端（如 Claude Code）调用本机 Ollama / llama.cpp 模型，或把上游 Anthropic 服务做带鉴权的转发壳。

只要在 `config.json` 里配上 `ollama_targets` 或 `llama_cpp_targets`（见上），并填了至少一个 `external_access_tokens`，反向代理 `POST /v1/messages`、`POST /v1/messages/count_tokens` 与 external 端口的 `POST /v1/chat/completions` 就开门工作：

- `model` 命中某个 `ollama_targets[*].models` → 转换为 Ollama `/api/chat`，再把响应翻译回 Anthropic 的 `message_*` SSE / 非流式 JSON。
- `model` 命中某个 `llama_cpp_targets[*].model` → 调用 llama.cpp `/v1/chat/completions`；Anthropic 入口会做格式转换，OpenAI 入口则基本直通。
- `model` 命中某个 upstream 的 `expose_external` 列表 → 透传到该 Anthropic 上游（相当于一个本机鉴权转发壳）。
- 其他情况 → 404 `model '...' is not exposed externally`。

`POST /v1/messages/count_tokens` 按 Anthropic 官方格式返回 `{"input_tokens": ...}`：命中本地 Ollama / llama.cpp target 时用本地估算，不会为了计数唤起模型；命中外露 upstream 时透传到上游的 count_tokens 接口，所以官方 Anthropic 端可以返回精确值。客户端附带的 `?beta=true` 会被接受；透传 upstream 时会保留这个 query。

端口分流规则：启用 `external_port` 后，internal 端口上的 `/v1/chat/completions` 保持 forward proxy 语义，走 upstream API；external 端口上的 `/v1/chat/completions` 走反向代理语义，优先命中 `ollama_targets`，再命中 `llama_cpp_targets`。

支持的转换：

- 文本消息、`system`、`max_tokens` / `temperature` / `top_p` / `top_k` / `stop_sequences`
- **工具调用**：`tools` + `tool_use` + `tool_result`（流式与非流式都支持；Ollama 仅在 `done` 时一次性返回 `tool_calls`，所以 `input_json_delta` 也是在 done 时一次性发出）
- Anthropic 的 base64 `image` 块：转到 Ollama 时变成消息里的 `images` 数组；转到 llama.cpp 时变成 OpenAI `image_url` data URI。URL 图片源可直接交给 llama.cpp，转给 Ollama 时会替换为占位文本（安全降级而非 500）。

让 Claude Code 走它（外部端口默认 21435）：

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:21435"
$env:ANTHROPIC_AUTH_TOKEN = "tk-填你的-external_access_tokens-之一"
$env:ANTHROPIC_MODEL = "llama3.1"
claude
```

## Web 配置编辑器（/admin）

启动后浏览器打开 <http://127.0.0.1:21433/admin>。左侧侧栏现在是二级目录：一级按“Forward Proxy / Reverse Proxy / Shared Settings / Admin UI”分块，二级再列具体配置组；每个字段一行：

- 左侧复选框 = **是否包含该字段**；取消勾选会从保存的 JSON 里移除，让它回退到默认值。
- 右侧是输入控件（按字段类型自适应：文本 / 数字 / 复选框 / 多行列表 / key-value 表 / 嵌套对象列表 / **从兄弟字段拉取的复选框列表**）。
- `upstreams`、`ollama_targets`、`llama_cpp_targets`、`model_profiles` 是可重复组，自带 +add / Remove。
- **Detect models**：upstream / Ollama 卡片会把探测结果合并或替换 `models` 字段；llama.cpp 卡片会把选择结果写入单个 `model` 字段。多个 llama.cpp 模型请拆成多个 target。
- **expose_external**：upstream、Ollama target 用模型列表筛选；llama.cpp target 用单个布尔开关，并可继承 `llama_cpp_defaults.expose_external`。
- **model_profiles 添加**：key 输入框带浏览器原生 datalist 自动补全，候选从 `upstreams` / `ollama_targets` 的 `models` 和 `llama_cpp_targets` 的 `model` 收集。
- **external_access_tokens**：每行带 Show / Generate 按钮；列表底部还有"+ generate token"直接追加随机 token。
- 顶部三个按钮：
  - **Save & Reload**：用 `Settings` 校验 → 写回磁盘 → 原子替换 `app.state.settings`，并重建上游连接池（旧连接异步关闭，正在进行的请求不受影响）。
  - **Discard & Reload from disk**：丢弃当前编辑，从磁盘重新加载。
  - **Toggle raw JSON**：当 schema 不够覆盖你的需求时，切回原始 JSON 文本框直接编辑。
- 校验失败会显示来自 Pydantic 的具体错误，配置不会被写入。

不需要这个 UI 的话设 `"admin_enabled": false`，相关路由不会注册。默认情况下 `/admin` 只挂在 admin listener 上，**不会被 internal 或 external 端口暴露**。只有显式设置 `"admin_port": null` 时才会回到旧的 internal 单端口模式。

## 视觉输入（图片）

- `/api/chat`、`/api/generate`：消息里传 `images: ["<base64>", ...]`，服务端从 base64 magic bytes 嗅探 PNG / JPEG / GIF / WEBP 并设置正确的 `media_type`。
- `/api/chat` 也兼容 OpenAI 风格的多段 `content`：`{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}` 会转成 Anthropic 图片块。
- `/v1/chat/completions`：`content` 用 OpenAI 风格 `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`，data URI 的 `media_type` 直接透传；也支持 HTTP(S) URL，转为 Anthropic `source.type=url` 块。
- `/v1/messages` 反向代理到本机 Ollama 时：Anthropic base64 `image` 块会转成 Ollama `images` 数组；反向代理到 llama.cpp 时会转成 OpenAI `image_url` data URI。external 端口的 `/v1/chat/completions` 命中本地 target 时也会保留对应协议的图片格式。
- 上游不支持图片时（如 DeepSeek）会返回错误，fake-ollama 不预拦截。

## 测试

```powershell
pip install -e ".[test]"

# 离线单测（默认）
pytest

# live 集成测试（需要 .env 里有有效 ANTHROPIC_BASE_URL/TOKEN）
pytest -m integration
```

`tests/conftest.py` 在缺少凭据时自动跳过 `@pytest.mark.integration` 用例，单测不会真正访问网络。

## 手动验证示例

```powershell
# Ollama 端（internal:21434）
curl http://127.0.0.1:21434/api/tags
curl -X POST http://127.0.0.1:21434/api/chat `
  -H "Content-Type: application/json" `
  -d '{"model":"claude-3-5-sonnet-20241022","stream":false,"messages":[{"role":"user","content":"hi"}]}'

# Anthropic 反向代理（external:21435，需 token）
curl http://127.0.0.1:21435/v1/models -H "x-api-key: tk-..."
curl -X POST http://127.0.0.1:21435/v1/messages `
  -H "Content-Type: application/json" -H "x-api-key: tk-..." `
  -d '{"model":"llama3.1","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
curl -X POST "http://127.0.0.1:21435/v1/messages/count_tokens?beta=true" `
  -H "Content-Type: application/json" -H "x-api-key: tk-..." `
  -d '{"model":"llama3.1","messages":[{"role":"user","content":"hi"}]}'

# llama.cpp OpenAI 端（同一个 external 端口）
curl -X POST http://127.0.0.1:21435/v1/chat/completions `
  -H "Content-Type: application/json" -H "x-api-key: tk-..." `
  -d '{"model":"qwen3.6-27b-hauhau-q2kp","stream":false,"messages":[{"role":"user","content":"hi"}]}'
```

## 安全提示

- `.env` / `config.json` 已加入 `.gitignore`，请勿提交真实 token。
- **internal listener 默认仅绑 `127.0.0.1`**；要局域网共享请显式 `--host 0.0.0.0` 并自行做访问控制。
- **external listener** 默认也是 `127.0.0.1`。要对外暴露反向代理：要么 `external_host: "0.0.0.0"`，要么保持 `127.0.0.1` + 用 Nginx/Caddy 反代（更推荐，可以加 TLS）。
- **`/admin` Web UI 没有任何鉴权**。它默认只挂在独立 admin listener 上，并默认绑定 `127.0.0.1`。如果你把 `admin_host` 改成 `0.0.0.0` 或设置 `admin_port: null`，必须自行加一层带认证的反向代理（nginx/Caddy basic auth 等），或直接 `"admin_enabled": false`。
- 反向代理 token：建议在 Web UI 用 Generate 生成 ≥24 字节的随机串，不要复用其他系统的 token。Token 池里可以放多个，便于按客户端轮换。

## 故障排查

- **/v1/messages 返回 404 `model '...' is not exposed externally`**：该模型来自 upstream、Ollama target 或 llama.cpp target，但其所属节点的 `expose_external` 没把它列进去。在 admin UI 里勾选，或干脆删掉 `expose_external` 字段恢复"全部暴露"。
- **/v1/messages 返回 401**：`external_access_tokens` 为空（且 `external_port` 已设置 → 启动时已 WARN），或请求头里的 token 不在池里。检查 `x-api-key` / `Authorization: Bearer` 是否带对了。
- **/v1/messages 在 internal 端口返回 404**：你启用了 `external_port`，反向代理已经只在 external 端口可达。请改连 external 端口。
- **启动后找不到日志文件**：默认日志路径是相对当前工作目录的 `logs/fake_ollama.log`。确认是在项目根目录启动，或显式传 `--log-file I:\Projects\fake_ollama\logs\fake_ollama.log`；如果传了 `--no-log-file`，则只会输出到控制台。
- **502 / 连不上上游**：`httpx` 默认会读 Windows 系统代理。装了 Clash / V2Ray 且上游是直连 IP 时，保持 `use_system_proxy: false`（默认）。
- **503 / `Insufficient GPU VRAM`**：某个本地 Ollama / llama.cpp 模型在 `model_profiles` 里配了 `estimated_vram_gb`，但 `nvidia-smi` 汇总的当前可用显存不足；fake-ollama 会尝试释放空闲 60 秒以上的可回收模型，并在释放后重新读取真实 free VRAM，仍不够才返回 503。Anthropic `/v1/messages` 会返回 `type=error`、`error.type=overloaded_error` 和英文 message；OpenAI/Ollama 入口会返回同样的英文错误文本。降低该模型的 `estimated_vram_gb`、等待旧模型空闲、配置有效的 `stop_command`，或手动释放显存后重试。
- **400 thinking content must be passed back**（DeepSeek）：模型在某轮启用了 thinking，但下一轮历史里没把 `thinking` 块带回。fake-ollama 已做了缓存回查 + 当 profile 是 `auto + show_thinking=false` 时主动注入 `thinking: {type:"disabled"}` 绕过；如果你确实想要 thinking，把对应 profile 设为 `"thinking_mode": "enabled"`。
- **503 `No available accounts: this group only allows Claude Code clients`**：上游（claude-relay-service 等）侧的账号池限制，要求请求来自 Claude Code 客户端且池里有可用账号。这种限制无法通过改请求体绕过，需在上游后台调整该 API Key 的客户端限制 / 账户池。

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
