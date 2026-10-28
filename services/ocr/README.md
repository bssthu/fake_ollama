# 本机 OCR 与 Jev

入口：[Model Playground](http://127.0.0.1:21431/playground/)。OCR 已并入主页面，共用 API key、图片上传/粘贴、运行/停止及交互记录。连接后在模型列表选择 MiniCPM、PP-OCR、Hunyuan，能力选“识别”或“低分重识别”；选择“OCR 多模型对比”可同时比较，选择“OCR 候选复核”可复核已有候选。复核模型在单独的下拉框中选择。旧 `/playground/ocr` 地址跳转到主页面。

“连接并获取模型列表”只读取元数据，不加载权重。页面显示当前认证方式：实际本机监听、同源本机访问且配置存在本机匿名接口时可留空 key；填入无效 key 会被拒绝，不会退回匿名。模型在执行任务时加载。
连接时可使用现有模型 API 的 token，不是管理 token；若原有配置允许本机匿名入口，在受信任 loopback 上也可留空。页面不会持久保存凭据、原图或候选；工作进程的临时文件在完成、错误或取消后清理。

安装和实测结果：[OCR 调研与本机对比](../../docs/ocr-local-research.md)。

## 运行

本机配置为 `config/local/ocr.json`；主配置中的 `playground_ocr_config` 指向它。
完整服务启动后可直接使用此页面。仅需本机 Playground 时，在项目根目录运行：

```powershell
.venv\Scripts\python.exe scripts\run_ocr_playground.py --config config.json
```

该启动器只监听 `127.0.0.1:21431`，不启动配置中的其他监听器。已经运行完整服务时不要再占用同一端口。
所有 OCR API 经过原有 Playground 鉴权与本机匿名访问策略，没有新建绕过规则。配置此功能后，获准进入 Playground 的接口均可调用本机 OCR 引擎；OCR 引擎清单独立于聊天模型的暴露清单。

## 使用

- OCR 识别：上传或粘贴一个图片区域，选择文字、公式或表格及识别引擎。“复核模型”默认“不复核”，可选择已配置的 Jev。
- 多模型对比：一次选择多个引擎，依次识别；选择复核模型后再统一评分。当前包含 MiniCPM、PP-OCRv6 small、HunyuanOCR 1.5 普通版及 DFlash 版，DFlash 默认不勾选。
- 自动重试：首选引擎识别、Jev 评分；有效低分、识别失败、空结果或截断时，最多换另一个引擎再试一次。Jev 服务/渲染失败不冒充低分，也不触发无意义的识别重试。保留全部候选，按有效分数选择；未达阈值则人工核对。
- 候选复核：填写最多 8 个候选，明确选择复核模型。原图与区域类型必须对应候选；公式使用 LaTeX，表格使用 HTML。此模式不需要识别引擎。

自动重试、候选复核模式必须选择可用复核模型，缺失时禁止运行。识别、对比模式可选“不复核”，不会启动 Jev。结果记录保留实际选择的复核模型 ID、名称及阈值，切换选择不会改写旧记录。

“任务完成后”默认“释放模型”；选择“保留模型，供连续识别”后，复用已加载的 llama.cpp 识别模型、Jev 及 PP-OCR ONNX 实例。所有用过的模型保留到手动释放、取消任务、服务退出或 OCR 配置改变；关闭页面不会释放。状态区显示已保留模型并提供刷新、释放按钮。缓存属于整个 Playground 服务，各页面共用；释放期间或其他任务运行中不接受并发修改。切回“释放模型”后，在下一次运行开始时清理此前保留的模型。

“连续运行次数”支持 1–20 轮，顺序处理同一输入并保留各轮结果。显示首轮和后续平均浏览器往返耗时，各识别结果另列模型加载、模型是否复用及推理耗时；结果中的 `runtime_events` 也记录 Jev 加载/复用。测试 llama.cpp 时显式关闭请求提示词缓存（`cache_prompt=false`），保留模式仅复用模型实例。往返耗时包括请求传输、首次工作进程启动、图片处理、可选渲染/复核及清理，不等同于纯生成耗时。首轮若命中此前保留的模型，也会标明复用，不应视作冷启动。

这里只处理单个区域，不包含自动版面分析、PDF、多页拼接。复杂整页应先分区；官方 MiniCPM SDK 的 PP-DocLayoutV3 整页流水线是另一个部署层。

Jev 的分数不是已校准正确率。不同渲染字体、类别及后端会影响绝对分数；低分可以请求复核，不能证明识别错误，高分也不能保证无错字。结果中 `score=null` 和 `verdict` 用来区分失败与有效低分。

## 实现与资源生命周期

API：`GET /playground/api/ocr/models` 读取本机文件就绪状态、`worker_running`、`loaded_models` 和 `busy`；`POST /playground/api/ocr/release` 释放保留模型，忙时返回 409；两者均不触发模型加载且沿用接口鉴权。`POST /playground/api/ocr/run` 执行任务。请求示例（`image` 换成图片的 data URL 或纯 base64）：

```json
{
  "image": "data:image/png;base64,...",
  "mode": "auto",
  "category": "text",
  "engines": ["minicpm-ocr", "hunyuan-ocr-current"],
  "verifier": "minicpm-ocr-jev",
  "threshold": 0.9,
  "max_tokens": 4096,
  "keep_loaded": true
}
```

`mode` 支持 `recognize/compare/score/auto`；`score` 另传字符串数组 `candidates`，可传 `engines: []`。`verifier` 是配置中的 ID，省略或 `null` 表示不复核；`auto/score` 必填。原布尔字段 `verify` 已由该字段替代。未知复核 ID 返回 422，不可用模型返回 503，均不会启动工作进程。
响应包含 `results`（各候选的文字、有效分数或失败原因、耗时）、`best_index`、`decision`（`accept/review/unverified`）和 `retry_count`。最大 6 个识别引擎、8 个候选、单图 16 MP / 单边 8192 px；页面文件限制 12 MB。

网关不导入 PyTorch、Pillow 或 ONNX。通过 `python` 配置项启动独立 `fake_ollama.ocr_worker`，使用原有 Windows Job / POSIX 进程树管理。每次任务独占 OCR 队列；忙时返回 409；最多 900 秒。`keep_loaded` 默认 false，保持每阶段卸载、每次任务退出工作进程；为 true 时使用常驻工作进程及私有文件信箱，不额外开放服务端口。每轮图片/结果临时文件照常删除；Jev 渲染浏览器仍按每次评分创建和释放，模型权重保留。取消、断连、超时或工作进程故障时清理整个所属进程树；配置变化会重建工作进程。不会停止其他服务的进程。

Jev 直接使用 GGUF 的 `/v1/systemone` 分类接口：`state` 是候选文字，`files` 顺序必须为原图、候选渲染图，`questions.consistency` 为 `choice`、类别 `0/1`。只接受两类归一化概率中的 `1`，不能把它当文本生成模型。渲染调用固定版本 OpenBMB `RenderPool`，带本地 KaTeX 与语法预检查；阻断候选的网络资源请求，拒绝可执行 HTML。

PP-OCR 仅使用 RapidOCR 3.10.0 / ONNX CPU。权重和字典独立存放在 `K:\Projects\LLM_Models\PP-OCRv6-small-onnx`，复制后 SHA256 校验一致；运行不依赖 Clipaste 源码、安装目录或 .NET 桥。

旧 Hunyuan 基线及 .NET 对照项已移除，历史实验数据仍保留在调研报告。Hunyuan 直接使用官方任务提示词和正常 EOS/EOT，遵守页面最大输出限制。重新转换没有改变所比较的主模型及视觉投影张量，修复的是结束标记等 GGUF 元数据；DFlash 可单独选用。

## 配置复核模型

`config/local/ocr.json` 的 `verifiers` 是按 ID 索引的字典，页面从发现接口动态生成选项。目前支持的适配类型是 `minicpm_jev`，可配置多个兼容 Jev 权重；其他模型协议需要新增适配器，不能直接填聊天模型。

```json
{
  "verifiers": {
    "minicpm-ocr-jev": {
      "label": "MiniCPM-OCR-Jev / F16",
      "kind": "minicpm_jev",
      "model": "K:/Projects/LLM_Models/MiniCPM-OCR-Jev-gguf/MiniCPM-OCR-Jev-F16.gguf",
      "mmproj": "K:/Projects/LLM_Models/MiniCPM-OCR-Jev-gguf/mmproj-model-f16.gguf",
      "context": 8192,
      "renderer_source": "services/ocr/.upstream/MiniCPM-OCR-b33b474136088dc43a2341e94590c804ec0c16a5/scripts/minicpm-ocr-jev"
    }
  }
}
```

以上是需合并进完整配置的片段。原单一 `jev` / `jev_source` 已迁移为各复核模型配置；修改后重新获取模型列表即可刷新选项。配置可不含任何复核模型，普通识别仍可使用。

## 重建环境

权重下载（显式执行，网关不会自动下载）：

```powershell
.venv\Scripts\python.exe scripts\setup_ocr_models.py --root K:\Projects\LLM_Models
uv venv services\ocr\.venv --python 3.12
uv pip install --python services\ocr\.venv\Scripts\python.exe -r services\ocr\requirements.txt
$env:PLAYWRIGHT_BROWSERS_PATH = "$PWD\services\ocr\.browsers"
services\ocr\.venv\Scripts\python.exe -m playwright install chromium
```

OpenBMB 源码固定 `b33b474136088dc43a2341e94590c804ec0c16a5`，放在 `.upstream/MiniCPM-OCR-<revision>`。各复核模型的 `renderer_source` 指向其中 `scripts/minicpm-ocr-jev`。不要运行会改变系统字体或删除系统文件的 Linux 安装脚本；Windows 使用本机字体，实测校准单独报告。

llama.cpp 固定 `b11960`，CUDA 12.4 Windows 包及配套 cudart 解压到 `.runtime`。
官方发布现在通过 `v0.8.0/nightly-tag.txt` 指向每日构建；不要把只有该指针文件的 release 当成二进制包。

Hunyuan 当前官方快照是 `47644ecc4fc854efa4f505155158831f36773ee4`。
使用同一 `b11960` 源码中的 `convert_hf_to_gguf.py` 分别转换主模型 F16、`--mmproj` F16、`dflash` BF16（需要 `--target-model-dir`）。转换环境与 OCR 渲染环境分开。此版本转换器从 `generation_config.json` 写入 EOT；校验输出含 EOS 120020 和 EOT 120007。

本机 `K:\Projects\LLM_Models\minicpm-ocr-manifest.json`、`HunyuanOCR-1.5-HF/download-manifest.json`、`HunyuanOCR-1.5-b11960/conversion-manifest.json` 记录来源和 SHA256。
模型、浏览器、运行库与源码快照均忽略，不进入 Git。

## 验证

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_ocr_pipeline.py tests/contract/test_ocr_playground.py tests/contract/test_playground.py -q
services\ocr\.venv\Scripts\python.exe scripts\validation\ocr_samples.py
services\ocr\.venv\Scripts\python.exe scripts\validation\ocr_benchmark.py
services\ocr\.venv\Scripts\python.exe scripts\validation\ocr_browser.py
services\ocr\.venv\Scripts\python.exe scripts\validation\ocr_retention.py
```

浏览器验证使用真实 Chromium 的无头渲染，包含识别、评分、一次重试、取消。合成样本不是用户真实截图集，不应据此宣称整体精度排名。
