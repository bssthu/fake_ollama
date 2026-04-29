# fake_ollama

把一个 **Anthropic Messages API** 兼容的上游（例如自建网关、第三方代理）伪装成一个本机 **Ollama** 服务，让只支持 Ollama 协议的客户端（如部分 IDE 插件、桌面 AI 软件）可以无缝调用 Claude 系列模型。

## 特性

- 监听本地 `127.0.0.1:21434`（默认刻意避开真正 Ollama 的 `11434`），可改
- 实现 Ollama 常用接口：
  - `GET  /` / `GET /api/version` / `GET /api/tags` / `GET /api/ps`
  - `POST /api/show`
  - `POST /api/chat`（流式 + 非流式）
  - `POST /api/generate`（流式 + 非流式）
  - `POST /api/embeddings`（返回 501，Anthropic 无此能力）
- 自动转换：
  - Ollama `messages` → Anthropic `system` + `messages`
  - Ollama `options.{temperature, top_p, top_k, num_predict, stop}` → Anthropic 对应字段
  - 多模态：Ollama `images`（base64）→ Anthropic `image` block
  - SSE → NDJSON 流式回包
- 配置全部通过环境变量 / `.env`，**密钥不写入代码**
- `pytest` + `httpx.MockTransport` 单元测试 + 可选的 live integration 测试

## 快速开始

```powershell
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 配置 .env
Copy-Item .env.example .env
# 然后用编辑器把 ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN 填好

# 3. 启动
python -m fake_ollama
# 或自定义端口
python -m fake_ollama --host 0.0.0.0 --port 21434
```

启动后即可用任何 Ollama 客户端连接 `http://127.0.0.1:21434`。

## 配置项（环境变量）

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `ANTHROPIC_BASE_URL` | 是 | 上游 base url，例如 `http://8.166.137.143:48085` |
| `ANTHROPIC_AUTH_TOKEN` | 是 | 上游鉴权 token；同时以 `x-api-key` 和 `Authorization: Bearer` 两种形式发送，兼容 Anthropic 原生与 OpenAI 风格代理 |
| `FAKE_OLLAMA_HOST` | 否 | 监听地址，默认 `127.0.0.1` |
| `FAKE_OLLAMA_PORT` | 否 | 监听端口，默认 `21434`（避开真正 Ollama 的 `11434`） |
| `FAKE_OLLAMA_ADVERTISED_VERSION` | 否 | `/api/version` 返回的版本号，默认 `0.6.4`。部分 Ollama 客户端会拒绝低版本服务（例如报 `Please upgrade to version 0.6.4 or higher`），按需调高 |
| `FAKE_OLLAMA_MODELS` | 否 | 暴露给 `/api/tags` 的模型名列表（逗号分隔） |
| `FAKE_OLLAMA_MODEL_MAP` | 否 | JSON 字典：`{"短名": "上游真实模型ID"}`，未命中则原样透传 |
| `FAKE_OLLAMA_MODEL_PROFILES` | 否 | JSON 对象：为每个模型声明 `capabilities`（`completion` / `tools` / `vision`，Copilot 等客户端会读它来判断该模型能否用作 chat / tool calling / vision）、`context_length`（最大上下文 token，同时用于在 UI 显示并拦截超长请求）、可选 `max_output_tokens`（覆盖 `num_predict` 默认值并设上限）。未配置的模型回退到 `["completion","tools","vision"]` + `200000` ctx |
| `FAKE_OLLAMA_ENFORCE_CONTEXT_LIMIT` | 否 | 默认 `true`：估算 `输入 token + max_tokens` 超过该模型 `context_length` 时直接 400 拦截，避免误传超长 prompt 产生高额费用。设 `false` 关闭 |
| `FAKE_OLLAMA_DEFAULT_MAX_TOKENS` | 否 | 客户端没传 `num_predict` 时使用 |
| `FAKE_OLLAMA_TIMEOUT` | 否 | 上游请求超时秒数，默认 300 |
| `FAKE_OLLAMA_USE_SYSTEM_PROXY` | 否 | 默认 `false`：上游 httpx 客户端**忽略** `HTTP_PROXY`/`HTTPS_PROXY`/Windows 系统代理；设为 `true` 才走（Clash/V2Ray 用户通常保持 `false`） |

## 每模型 capabilities / 上下文长度

GitHub Copilot 的 Ollama provider 会根据 `/api/show` 里的 `capabilities` 字段决定模型在「管理模型」UI 中是否显示，以及能否被用于 tool-calling / 视觉输入。`context_length` 也会展示在 UI 上。通过 `FAKE_OLLAMA_MODEL_PROFILES`（JSON）为每个模型单独配置：

```jsonc
// .env (单行；下面只是为了可读才换行)
FAKE_OLLAMA_MODEL_PROFILES={
  "claude-3-5-sonnet-20241022": {
    "capabilities": ["completion", "tools", "vision"],
    "context_length": 200000,
    "max_output_tokens": 8192
  },
  "claude-3-5-haiku-20241022": {
    "capabilities": ["completion", "tools"],
    "context_length": 200000,
    "max_output_tokens": 8192
  },
  "deepseek-v4-pro": {
    "capabilities": ["completion", "tools"],
    "context_length": 128000,
    "max_output_tokens": 8192,
    "thinking": "enabled",
    "thinking_budget_tokens": 1024,
    "show_thinking": true
  },
  "deepseek-v4-flash": {
    "capabilities": ["completion"],
    "context_length": 64000
  }
}
```

字段说明：
- `capabilities`：子集自 `["completion", "tools", "vision"]`。**至少要有 `completion`**，否则 Copilot 会把该模型过滤掉。`vision` 仅当上游模型支持图片输入时才声明（DeepSeek 当前不支持图片，Anthropic Claude 3.5+ 支持）。
- `context_length`：最大 *总* token 数（输入 + 输出）。除了在客户端 UI 显示，服务端还会做**校验**：若估算的输入 token + `max_tokens`/`num_predict` 超过该值，直接返回 `400`，**不**调用上游 —— 避免误传巨大 prompt 产生高额费用。
- `max_output_tokens`（可选）：覆盖该模型的默认 `num_predict`，并对客户端传入的 `max_tokens` 设上限。
- `thinking`（可选）：`auto` / `enabled` / `disabled`，控制是否在上游请求中注入 `thinking: {type:"enabled"|"disabled"}`。`auto`（默认）保持沉默，让客户端或上游自行决定。Reasoning 模型如 DeepSeek-V3.2 和 Claude 3.7+ 才有效；详见 [DeepSeek Anthropic API 兼容性文档](https://api-docs.deepseek.com/zh-cn/guides/anthropic_api)。
- `thinking_budget_tokens`（可选）：`thinking=enabled` 时的 token 预算，默认 `1024`。注意 DeepSeek 会忽略该字段，仅 Anthropic 真正生效。
- `show_thinking`（可选，默认 `true`）：是否把上游返回的 `thinking` 推理内容透传给客户端。`true` 时会把推理用 `<think>...</think>` 包裹后接到正文前面（Open WebUI、Cherry Studio 等会把 `<think>` 块折叠显示），同时在 Ollama `/api/chat` 响应里附上 `message.thinking` 字段、在 OpenAI `/v1/chat/completions` 流式增量里附上 `reasoning_content` 字段（DeepSeek / OpenAI o-series 约定）。

未在 `MODEL_PROFILES` 中列出的模型回退到 `["completion","tools","vision"]` + `context_length=200000` + `thinking=auto` + `show_thinking=true`。

如需关闭超长拦截，设置 `FAKE_OLLAMA_ENFORCE_CONTEXT_LIMIT=false`。

> 注意：token 估算使用约 `字符数 / 3` 的保守启发式（中文/英文都偏高估），目的是宁可早拦也不要漏拦。它**不**保证与 Anthropic 计费完全一致，但作为预算保险足够。

## 视觉输入（图片）

- `/api/chat`、`/api/generate`：在消息里传 `images: ["<base64>", ...]`，服务端会自动从 base64 magic bytes 嗅探 PNG / JPEG / GIF / WEBP 并设置正确的 `media_type`，**不**再硬编码 `image/png`。
- `/v1/chat/completions`：在 `content` 里传 OpenAI 风格的 `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}`，data URI 的 `media_type` 直接透传；也支持 HTTP(S) URL，转为 Anthropic `source.type=url` 块。
- 上游（如 DeepSeek）若不支持图片，会返回错误；fake-ollama 不会预拦截，只是消息形式正确。

## 测试

```powershell
pip install -e ".[test]"

# 仅跑离线单元测试（默认）
pytest

# 跑 live 集成测试（需要 .env 里有有效 ANTHROPIC_BASE_URL/TOKEN）
pytest -m integration
```

`tests/conftest.py` 会在缺少凭据时自动跳过 `@pytest.mark.integration` 标记的用例，单元测试不会真正访问网络。

## 手动验证示例

```powershell
# 列出模型
curl http://127.0.0.1:21434/api/tags

# 非流式 chat
curl -X POST http://127.0.0.1:21434/api/chat `
  -H "Content-Type: application/json" `
  -d '{"model":"claude-3-5-sonnet-20241022","stream":false,"messages":[{"role":"user","content":"hi"}]}'
```

## 安全提示

- `.env` 已加入 `.gitignore`，请勿提交真实 token。
- 默认仅绑定 `127.0.0.1`；如需局域网共享请显式 `--host 0.0.0.0` 并自行做访问控制。

## 故障排查

- **502 / 连不上上游**：`httpx` 默认会读 Windows 系统代理。如果你装了 Clash / V2Ray，且上游是直连 IP，请保持 `FAKE_OLLAMA_USE_SYSTEM_PROXY=false`（默认）。
- **503 `No available accounts: this group only allows Claude Code clients`**：这是上游（claude-relay-service 等）侧的账号池限制，要求请求必须来自 Claude Code 客户端，且当前池里有可用账号。这种限制无法通过修改请求体绕过——需要在上游后台调整该 API Key 的客户端限制 / 账户池。
