# fake_ollama

一个轻量的协议适配层，主要做两件事：

1. **正向**（Ollama 兼容入口 → 远端上游）：把 **Anthropic Messages API** 或 **OpenAI Chat Completions API** 兼容的上游（官方 / DeepSeek / Together / Groq / 自建网关 / claude-relay-service）伪装成一台本机 **Ollama** 服务，让只支持 Ollama 协议的客户端（GitHub Copilot 自定义 provider、IDE 插件、桌面 AI 软件）无缝调用 Claude / DeepSeek / GPT 等模型。
2. **反向**（Anthropic / OpenAI 兼容入口 → 本机模型服务）：把本机的 **Ollama**、**llama.cpp server** 或 **ComfyUI workflow** 包装成 Anthropic / OpenAI 兼容 API（含 `POST /v1/messages`、`POST /v1/chat/completions`、`POST /v1/images/generations`、`POST /v1/images/edits`），让只支持远端 API 的客户端也能调用本地模型。

附带一个零依赖的 Web 配置编辑器（`/admin`），以及可选的轻量模型调试页（`/playground`）。

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
│    /v1/images/generations OpenAI Images 兼容                      │
│    /v1/images/edits       OpenAI Images 兼容                      │
│    /v1/embeddings    OpenAI 兼容                                 │
│    /v1/models                                                    │
│    每个实例可独立配置 host / port / access_tokens / exposed_models│
└──────────────────────────────────────────────────────────────────┘
```

- 每个 `ollama_interfaces[*]` / `api_interfaces[*]` 都是独立的 uvicorn 监听。你可以同时开多个 Ollama 接口（例如本机 21434 + 局域网 0.0.0.0:21434），每个挂不同的白名单和 token。
- `access_tokens` 为空 = 该接口不需鉴权；非空 = 该接口的所有请求都必须带 `x-api-key` 或 `Authorization: Bearer`。
- `exposed_models` 是该接口允许看到 / 请求的模型列表，元素是 `{model, target, alias?}` 三元组；不在白名单里的模型在该接口上 404。
- admin 接口（`admin_host` / `admin_port`）只服务 `/admin/*`，**不会**出现在任何 `ollama_interfaces` / `api_interfaces` 上；同样，`/api/*` 与 `/v1/*` 也**不会**出现在 admin 端口。
- Model Playground 默认关闭。启用后使用独立端口，按 `/v1/models` 返回的 capabilities/operations 调用聊天、视觉、图片生成/编辑与视频接口；支持粘贴、拖放和选择多张参考图。API key 仅保存在当前浏览器页面内存中。
- 想让别的机器访问某个接口：把对应实例的 `host` 改成 `0.0.0.0`，或者保持 `127.0.0.1` + 在前面挂 Nginx/Caddy（推荐，可以加 TLS / 限流 / 客户端证书）。

## 特性一览

- **多上游路由**：把 Anthropic / OpenAI 兼容（DeepSeek / Together / Groq / 自建网关 / claude-relay-service…）合并到同一组接口
- **结构化模型标识 `{model, target}`**：每条暴露条目显式声明「上游显示名 + 来源 target 名」，相同名字但来源不同的模型不会再混淆；可选 `alias` 字段在客户端可见的公开 ID 上替换全名
- **每模型 profile**：capabilities / 上下文长度 / 思维链开关 / 输出上限——key 支持 `model@target`（最优先）、裸 `model`、tagless base 三级回退
- **按接口多实例暴露**：`ollama_interfaces[*]` 与 `api_interfaces[*]` 是独立数组，每个实例自带 `host` / `port` / `access_tokens` / `exposed_models`
- **来源命名清晰区分**：`anthropic_upstreams` / `openai_upstreams`（远端）+ `ollama_targets` / `llama_cpp_targets` / `comfyui_targets`（本机）
- **循环引用检测**：`anthropic_upstreams[*].base_url` 若指向 fake_ollama 自己的某个监听端口，启动时直接报错，避免转发死循环
- **本地 target 生命周期接管**：Ollama / llama.cpp / ComfyUI 都可配置 health check、按需启动脚本、启动超时、空闲回收
- **ComfyUI 图片后端**：可把 Z-Image-Turbo / Qwen-Image-Edit / SenseNova-U1 等 ComfyUI API workflow 暴露为 OpenAI 兼容图片生成 / 图片编辑接口。采用声明式 `preset` + bindings 结构，接入新模型只改配置 + JSON；多个图片模型可共用一个 ComfyUI 实例并由 VRAM 协调器互斥换出，避免爆显存。
- **本地显存预检**：本地模型可在 `model_profiles` 里填 `estimated_vram_gb`；启动前用 `nvidia-smi` 评估可用显存并尝试回收空闲模型
- **本地内存预检**：部分模型（如 SenseNova / JoyAI 这类会把一部分计算 offload 到内存的 workflow）除显存外还需占用大量主机内存，可在 `model_profiles` 里填 `estimated_memory_gb`；逻辑与显存预检一致，启动前评估可用系统内存并尝试回收空闲模型
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
  "llama_cpp_defaults":  { /* … */ },
  "llama_cpp_targets":   [ /* … */ ],
  "comfyui_targets":     [ /* … */ ],

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
| `comfyui_targets`     | OpenAI Images（ComfyUI workflow） | 一项 = 一个 ComfyUI server / 一个图片模型 workflow target |

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

- `auto_start: false`：fake-ollama 只转发，不接管 daemon。
- `auto_start: true`：health check 失败时执行 `start_command`；`stop_command` / `idle_timeout_seconds` 控制空闲回收。
- 模型名匹配按 Ollama 规则：`foo` 与 `foo:latest` 等价；省略 tag 只代表 `latest`。
- 每个本地模型的显存估算在 `model_profiles[*].estimated_vram_gb` 里配置。请求触发模型加载前，fake-ollama 调用 `nvidia-smi` 检查可用显存；不足时按 LRU 卸载其他空闲超过 60 秒的本地模型，每轮重新读真实 free VRAM。
- 主机内存估算在 `model_profiles[*].estimated_memory_gb` 里配置，由独立的内存协调器按同样逻辑准入与回收：加载前读取系统可用内存，不足时卸载其他声明了 `estimated_memory_gb` 的空闲模型。Dashboard 的「Model Estimated Memory」面板与 Current Models 表里的 Est. Memory 列展示该维度。

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

#### comfyui_targets

视频 workflow 既可以使用内置 `joyai_echo` preset，也可以使用
`custom` / `comfyui_api` + `video_workflow_path` / `image_to_video_workflow_path`。
请求字段分别绑定在 `bindings.video`（文生视频）和 `bindings.i2v`（图生视频）；
图片可绑定到 `image`、`images` 或 `image_1`...`image_4`。请求附带图片且存在
I2V workflow 时自动走 I2V，否则走 T2V。

参数发现不另建一套容易漂移的模型配置：`WorkflowSpec.bindings` 仍是执行的事实来源，
服务端据此生成 `/v1/models[*].operations[*].parameters`、默认值、范围、参考图上限和
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

当前内置模型的推荐公开参数如下；表中默认值来自对应 target 配置，因而 API 缺省行为、
`/v1/models` 和 Playground 使用同一个值：

| 模型 | Playground / API 公开参数 | 推荐默认值 |
|---|---|---|
| Z-Image-Turbo | `size`、`n`、`seed`、`steps`、`cfg`、`sampler_name`、`scheduler`；编辑另有 `denoise` | 1024²、8 steps、CFG 1、`res_multistep` / `simple`；编辑 denoise 0.25 |
| Qwen-Image-Edit AIO | 同 Z-Image；编辑只接收 1 张参考图 | 1024²、6 steps、CFG 1、`euler_ancestral` / `beta`；编辑 denoise 1.0 |
| SenseNova-U1 | 原生宽高比桶、`n`、`seed`、`steps`、`cfg`；不显示 workflow 没有输入的 sampler/scheduler/denoise | 1:1、8 steps、CFG 1；实际像素由模型的原生比例桶决定 |
| JoyAI Echo | `size`、`seed`、`num_frames`、`fps`、`prefetch_count`、`enable_tile`、`enable_streaming` | 安全调试：256²、17 帧、8 FPS、prefetch 1、分块解码开；帧数必须为 `8k+1`，尺寸至少 256 且为 32 的倍数 |

> **JoyAI Echo 节点兼容性**：已在本机验证的 `ComfyUI_JoyAI_Echo` 版本要求节点把
> `enable_tile` 写入模型实际读取的 `enable_tiles` 属性，并要求 VAE wrapper 将
> `tiled_decode()` 返回的分块迭代器沿时间维拼接后再进入后处理。旧版节点若未包含这两项
> 修复，会分别表现为勾选分块解码仍发生显存不足，或报
> `unsupported operand type(s) for +: 'generator' and 'int'`。

图片 preset 对应 `fake_ollama/workflows/<preset>_t2i.json` / `<preset>_i2i.json`，视频则对应 T2V / I2V workflow。要接入**其它** ComfyUI 模型，提供 API 格式 workflow JSON，再用 `bindings`（逻辑参数 → `[{node, input}]`）和 `static_inputs`（固定值，如模型文件名）声明落点即可；支持 `t2i`、`i2i`、`video`、`i2v` 四种模式。

> **客户端可能按模型名判断是否图片模型**：部分客户端（如 CherryStudio）靠模型 id 的正则/子串匹配来决定走聊天接口还是 `/v1/images/*`——只认 `z-image*` / `qwen-image*` / `flux*` / `sd*` 等已知图片模型名。若把图片模型暴露成它不认识的名字（如 `sensenova`），它会当普通聊天模型发到 `/v1/chat/completions`，被 fake_ollama 以 400「use /v1/images/...」拒绝。对策：在 `exposed_models[*].alias` 里给这类模型起一个**包含已知图片模型关键词的别名**（例如 `sensenova-z-image`）；这只改客户端看到的 id，后端 target 与 `model_profiles` 仍按真实模型名匹配，不受影响。

> **多模型共用一个 ComfyUI 实例 + 显存互斥**：把多个 target 的 `base_url` / `start_command` 指向同一个 ComfyUI（如下例三个图片模型共用 `:21480`）。fake_ollama 的 VRAM 协调器按各模型 `model_profiles.estimated_vram_gb` 准入与 LRU 换出，大图模型运行前会先挤出空闲的 llama.cpp 模型；同实例内的图片模型切换由 ComfyUI 自身的智能内存负责（显存放不下时自动 offload 到内存）。这样在 24GB 卡上接入 28GB 的 Qwen 检查点也不会爆显存。

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
      "start_command": "\"I:\\Projects\\ComfyUI\\ComfyUI-aki-v3\\python\\python.exe\" -s main.py --listen 127.0.0.1 --port 21480",
      "cwd": "I:\\Projects\\ComfyUI\\ComfyUI-aki-v3\\ComfyUI",
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
      "start_command": "\"I:\\Projects\\ComfyUI\\ComfyUI-aki-v3\\python\\python.exe\" -s main.py --listen 127.0.0.1 --port 21480",
      "cwd": "I:\\Projects\\ComfyUI\\ComfyUI-aki-v3\\ComfyUI",
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
      "start_command": "\"I:\\Projects\\ComfyUI\\ComfyUI-aki-v3\\python\\python.exe\" -s main.py --listen 127.0.0.1 --port 21480",
      "cwd": "I:\\Projects\\ComfyUI\\ComfyUI-aki-v3\\ComfyUI",
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
      "start_command": "set \"PATH=I:\\Projects\\Tools\\ffmpeg;%PATH%\" && \"I:\\Projects\\ComfyUI\\ComfyUI-aki-v3\\python\\python.exe\" -s main.py --listen 127.0.0.1 --port 21480",
      "cwd": "I:\\Projects\\ComfyUI\\ComfyUI-aki-v3\\ComfyUI",
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
      "estimated_vram_gb": 16
    },
    {
      "model": "qwen-image",
      "target": "qwen-image-comfyui",
      "capabilities": ["image_generation", "image_edit"],
      "context_length": 4096,
      "estimated_vram_gb": 20
    },
    {
      "model": "sensenova",
      "target": "sensenova-comfyui",
      "capabilities": ["image_generation", "image_edit"],
      "context_length": 4096,
      "estimated_vram_gb": 13,
      "estimated_memory_gb": 30
    },
    {
      "model": "joyai-echo",
      "target": "joyai-echo-comfyui",
      "capabilities": ["video_generation"],
      "context_length": 4096,
      "estimated_vram_gb": 12,
      "estimated_memory_gb": 30
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
| `model_profiles` | list | `[]` | 每模型 capabilities / 上下文 / 思维链等。每项写 `model`（必填）+ 可选 `target`，两者拼起来作为最终 key：填 `target` 时为 `model@target` 仅覆盖该 target，不填则裸 `model` 适用于所有 target。查找时三级回退：`model@target` > 裸 `model` > tagless base。旧的 `{ "model@target": {...} }` dict 写法仍兼容 |

`model_profiles[*].capabilities` 是对外声明的功能标签，主要给 `/api/show`、`/api/tags`、`/v1/models` 和客户端做功能发现；它不会自动让底层模型获得该能力。常用值：

| capability | 使用场景 |
| --- | --- |
| `completion` | 文本补全 / 聊天模型。Ollama、llama.cpp、Anthropic/OpenAI chat 模型通常需要它。 |
| `tools` | 工具调用 / function calling。仅在底层模型或上游确实能处理工具调用时声明。 |
| `vision` | 聊天接口可接收图片输入，例如 `/api/chat` 的 `images` 或 OpenAI/Anthropic 多段图片消息。 |
| `image_generation` | 图片生成模型，供 OpenAI 兼容 `POST /v1/images/generations` 使用。 |
| `image_edit` | 图片编辑 / image-to-image 模型，供 OpenAI 兼容 `POST /v1/images/edits` 使用。 |
| `video_generation` | 视频生成 / image-to-video 模型，供扩展接口 `POST /v1/videos/generations` 使用。 |

聊天模型一般至少包含 `completion`；纯 ComfyUI 媒体 workflow 可以不填 `completion`，只声明 `image_generation` / `image_edit` / `video_generation`。`/v1/models` 除原始 `capabilities` 外还会返回结构化 `operations`，包含 endpoint、是否流式、图片输入约束、实际绑定的参数 schema、默认值/范围及推荐 preset，并返回 `estimated_vram_gb` 供调试界面展示；Playground 完全由这些字段驱动。

### Admin UI / Dashboard / Model Playground

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
| `dashboard_reclaim_idle_seconds` | float | `20.0` | dashboard 关闭按钮所需的最小空闲秒数。和自动 LRU 回收的 60s 阈值独立——用户手动判断更宽松 |
| `vram_low_free_reclaim_enabled` | bool | `true` | 检测显存低水位时主动回收 |
| `vram_low_free_threshold_mib` | float | `200.0` | 低水位阈值（MiB） |
| `memory_low_free_reclaim_enabled` | bool | `true` | 检测系统内存低水位时主动回收（仅回收声明了 `estimated_memory_gb` 的模型） |
| `memory_low_free_threshold_mib` | float | `2048.0` | 内存低水位阈值（MiB） |
| `playground_enabled` | bool | `false` | 是否启用轻量流式模型调试页；页面不保存历史记录或会话 |
| `playground_host` / `playground_port` | | `127.0.0.1` / `21431` | Playground 独立 listener；修改后需重启进程 |

启用后打开 `http://127.0.0.1:21431/playground/`，输入某个 interface 的 `access_tokens`，即可加载该 interface 暴露的模型。页面会读取模型的 capabilities/operations，自动提供聊天、视觉输入、图片生成、图片编辑或视频生成模式，并按 workflow 动态显示真正生效的参数与推荐 preset；图片可粘贴、拖放或从文件选择。若 interface 不要求鉴权，API key 可以留空。

## 内部 backends 视图

`Settings` 之上有一份协议无关的 `backends` 列表，路由代码用它统一查找：

```python
settings.backends
# -> List[Backend]，每项 (name, protocol, kind, base_url, source)
#    protocol: "anthropic" | "openai" | "ollama" | "comfyui"
#    kind:     "remote" | "local"
#    source:   底层 AnthropicUpstream / OpenaiUpstream / OllamaTarget / LlamaCppTarget / ComfyUITarget

settings.resolve_request("deepseek-r1", interface_name="api")
# -> 按某个接口的视角解析「客户端请求 model」到 (backend, ModelEntry)
#    会查 exposed_models（含 tagless 回退），找不到返回 None
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
- `/v1/images/generations` / `/v1/images/edits`：走 `comfyui_targets`，用于图片生成和 image-to-image 编辑。
- `/v1/videos/generations`：走 `comfyui_targets`，用于 text-to-video；请求里带 `image` / `images` 且 target 配了 `image_to_video_workflow_path` 时走 image-to-video。
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
- **启动报 `cycle detected in model-forwarding graph`**：某条上游链路在同名模型上回到了自己，会无限递归。给链路上某一跳起一个不同的 alias 就能避免；`:latest` 标签会被归一化，重命名时注意区分。
- **启动报 `cycle detected ... admin/dashboard listener`**：某个 `*_upstreams[*].base_url` 指向了 fake_ollama 自己的 admin / dashboard 端口（这些端口不服务模型流量）。改 URL 或删该 upstream。
- **运行时 HTTP 508 `loop detected`**：请求绕一圈又回到本进程（通常是反向代理 / 动态 DNS 绕过了启动期检测）。检查 upstream / 代理链。
- **启动报端口冲突**：两个 interface（或与 admin / dashboard）共用了同一个 host:port。
- **找不到日志文件**：默认相对 CWD：`logs/fake_ollama.log` 与 `logs/fake_ollama.requests.jsonl`。在项目根目录启动，或显式 `--log-file` / `--request-data-log-file`。
- **502 / 连不上上游**：`httpx` 默认会读 Windows 系统代理。装了 Clash / V2Ray 且上游是直连 IP 时，保持 `use_system_proxy: false`。
- **503 `Insufficient GPU VRAM`**：某个本地模型在 `model_profiles` 里配了 `estimated_vram_gb`，但 `nvidia-smi` 显示当前可用显存不足；fake-ollama 已尝试回收空闲超过 60 秒的模型仍不够。降低估算、等模型空闲、配 `stop_command`，或手动释放显存。
- **503 `Insufficient system RAM`**：某个本地模型配了 `estimated_memory_gb`，但当前可用系统内存不足；处理方式同上（降低估算、等模型空闲、释放内存）。
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
