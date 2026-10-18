# StartLux v1.1 4B 本地接入

这份配置使用独立的官方 StartLux Transformers 后端，在 Windows CUDA 上加载原始 BF16 权重。
通用安装、协议和生命周期说明见 [decision-startlux.md](decision-startlux.md)。

示例配置见 [startlux-windows-local.json](../config/examples/decisions/startlux-windows-local.json)。将其中的 targets、profiles 和 exposed models 合并到现有配置，调整路径，并保留现有接口令牌。

在仓库根目录建立独立环境：

```powershell
python -m venv services/startlux/.venv
services/startlux/.venv/Scripts/python.exe -m pip install -r services/startlux/requirements-windows-cu128.txt
```

本机使用 Python 3.13、PyTorch 2.8.0+cu128、Transformers 5.8.1 和固定提交的 StartLux 1.1.0。`accelerate` 支持官方后端的 `device_map` 加载，`torchvision` / Pillow 支持图像 processor。完整环境版本记录在本机 `services/startlux/installed-packages.txt`。

Windows 的上述 PyTorch 构建没有 Flash Attention。companion 在启动官方 1.1.0 Transformers Instant 后端时，对双向 GQA 的 K/V heads 做等价重复，使其可使用 memory-efficient CUDA 内核。适配仅改变张量布局，官方代码继续负责 prompt、mask、校准、read-out 和启动自检；因果路径、CPU、vLLM 和 AutoThink 不受此适配影响。`--check` 仍只检查元数据。

## 模型与入口

| 模型 ID | 本地服务 | 思考模式 |
| --- | --- | --- |
| `startlux-decision-v1.1-4b-instant` | `127.0.0.1:21461` | `never` |
| `startlux-decision-v1.1-4b-autothink` | `127.0.0.1:21462` | `never` / `auto` / `always` |

通过 fake_ollama 的 API 接口调用上述模型；本地服务由网关按需启动和回收。
使用现有 API 接口的令牌。Playground 的 Decision 模式使用相同配置。

原始模型目录位于 `J:\Projects\LLM_Models\StartLux`：

- `StartLux-Decision-v1.1-4B-Instant`，HF revision `416088f77cfce09633906a3675c7cd2e5dabcf10`。
- `StartLux-Decision-v1.1-4B-AutoThink`，HF revision `23dd4c513ef387705bb13fd8caadc2d5c863082f`。

每个目录的 `download-manifest.json` 记录来源、文件大小和哈希。下载使用 curl；大文件可按 HTTP Range 分块，检查每块的 Content-Range，合并后核对完整 SHA256。保留 LICENSE、NOTICE、tokenizer、processor 和原始模型元数据。运行服务时禁止自动补下载。

## 请求示例

`POST http://127.0.0.1:21435/v1/systemone`，带现有 `Authorization: Bearer ...`：

```json
{
  "model": "startlux-decision-v1.1-4b-autothink",
  "state": {"ticket": "同一笔订单被重复扣款，请退还多扣的款项。"},
  "think_mode": "auto",
  "think_budget": 128,
  "return_details": true,
  "return_thought": true,
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "应该交给哪个团队处理？",
      "criteria": {"billing": "扣款与退款", "shipping": "物流配送", "technical": "登录与应用故障"}
    },
    "duplicate": {"type": "noul", "instructions": "是否发生了重复扣款？"},
    "severity": {
      "type": "score",
      "instructions": "问题对客户的影响程度？",
      "criteria": ["没有影响", "轻微不便", "资金损失"]
    }
  }
}
```

Instant 使用同样的题型，将模型名改成 `startlux-decision-v1.1-4b-instant`，并设置 `think_mode: "never"`，省略思考预算。图片使用顶层 `images: ["data:image/png;base64,..."]`。

网关的 `/v1/decisions` 使用另一套 OpenAI Decisions 请求结构：`input` 和 typed `questions` 数组。需要 StartLux 的 `gate`、`think_budget`、`return_thought` 或原生图片字段时，使用 `/v1/systemone`。这两个模型应调用决策端点。

## 运行边界

当前配置使用单并发、4,096 token 上下文、每请求最多 2 张图片、每张最多 262,144 像素。AutoThink 默认预算 512 token，请求上限 1,024 token。上下文包含协议模板、图片、问题和思考文本。

每个模型按 12 GiB 显存、16 GiB 系统内存做资源准入估算，空闲 300 秒回收。估算用于调度；实际占用随输入变化。避免同时加载其他大型模型挤占 GPU。官方 Transformers 后端用于参考推理，Windows 未安装线性注意力加速内核时，AutoThink 生成可能较慢。

模型权重遵循原仓库的 CC BY-NC 4.0，官方代码为 Apache-2.0。模型来源：[Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-Instant)、[AutoThink](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-4B-AutoThink)。

## 本机验证记录（2026-10-18）

在 RTX 4090、驱动 580.97 上，两个模型均以 `cuda:0` / BF16 完成真实 HTTP 网关调用：choice、predicate、score，OpenAI Decisions 格式转换，以及红色图片识别全部通过。Instant 拒绝 `always` 思考请求；AutoThink 的 `never`、`always`、`auto` 均通过，后两者返回实际思考文本；`auto` 用 gate=1 强制覆盖了思考分支。

Instant 官方启动自检：`causal_loop_max_diff=0`、`padding_invariance_max_diff=0`、`late_token_reach=2.4375`。额外 CUDA 数值回归 5 项通过。网关完整离线回归 930 项通过，3 项跳过（主环境未安装 OpenAI SDK、未安装 torch，以及 Windows 符号链接权限）；CUDA 回归在独立环境中单独执行。

短请求样本的热调用约 0.11–0.24 秒；AutoThink 96-token 预算样本约 3 秒。它们是功能验收样本，不代表长上下文、批量负载或模型准确率基准。默认 512-token 思考预算、4K 满上下文和多图片上限未做吞吐压测。

本机详细结果保存在 `config/local/startlux-validation-20261018.json`；测试结束后关闭测试网关及其模型进程，日常仍由正常 fake_ollama 启动入口按需加载。
