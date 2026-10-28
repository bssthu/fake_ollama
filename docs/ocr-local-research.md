# MiniCPM-OCR、Jev 与 Clipaste OCR 本机调研

2026-10-29；Windows / RTX 4090 24GB。已下载、接入并实际推理。入口为 [本机 Model Playground](http://127.0.0.1:21431/playground/)，OCR 与 Jev 直接出现在主页面模型和能力选择中，运行和配置说明见 [services/ocr](../services/ocr/README.md)。本次没有修改 Clipaste 工程或运行中的 Clipaste。

## 能力与接入边界

后续接入调整：已移除旧 Hunyuan GGUF 和 Clipaste RapidOcrNet 对照项，保留的 PP-OCR 权重已独立存入 K 盘模型目录，运行不依赖 Clipaste。复核器改为配置列表和页面下拉选择，识别/对比默认不复核，自动重试/候选复核需明确选择。本文下方 Clipaste 相关数据是此前实验记录，不代表当前可选引擎。

MiniCPM-OCR 1.0 是约 1.3B 的识别模型，支持中英文、表格和公式。MiniCPM-OCR Jev 是约 2B 的区域一致性分类器：输入原图区域、候选文字及候选渲染图，输出 0–1 分数。它可检查其他 OCR 模型的输出，不限 MiniCPM；不能替代 OCR，也不能直接返回修正文字。[官方模型介绍](https://github.com/OpenBMB/MiniCPM-OCR)

本机采用官方 F16 GGUF + llama.cpp b11960 CUDA 12.4。Jev 采用官方 SDK 渲染器（固定 OpenBMB commit、Playwright 1.62.0、Pillow 12.3.0、内置 KaTeX 0.16.47），专用 `/v1/systemone` 双类别分类接口；没有用聊天输出伪造分数。推理和渲染均在本机，Windows 字体环境与官方 Linux Docker 不相同。[部署说明](https://github.com/OpenBMB/MiniCPM-OCR/blob/main/docs/llamacpp_deploy_jev_en.md)

当前页面以**单区域**为单位，提供识别、单独评分、并排对比、低分换模型一次。自动模式保留两次结果，选择有效高分；仍未过阈值则人工核对。评分失败不作为低分。尚未接入自动版面分析、多页 PDF、整页区域拼接；不能把混合整页分数解释成逐区域检查结果。

## 下载位置与版本

| 内容 | 路径 / 版本 |
|---|---|
| MiniCPM-OCR F16 + projector | `K:\Projects\LLM_Models\MiniCPM-OCR-1.0-gguf`；HF revision `674e62bf4ef7560ba502fe9b366a2780028aef20` |
| Jev F16 + projector | `K:\Projects\LLM_Models\MiniCPM-OCR-Jev-gguf`；HF revision `2554d7c9173b4cb767c4ecfc3e886e824978965c` |
| Hunyuan 当前官方 HF 权重与 DFlash | `K:\Projects\LLM_Models\HunyuanOCR-1.5-HF`；revision `47644ecc4fc854efa4f505155158831f36773ee4` |
| Hunyuan 重新转换的 F16 / projector / DFlash BF16 | `K:\Projects\LLM_Models\HunyuanOCR-1.5-b11960` |
| Clipaste 基线 | 只读复用 `I:\Projects\clipaste\src\Clipaste` 的 PP-OCRv6 small 与 Hunyuan GGUF |

MiniCPM 与 Jev 四个权重文件约 6.39 GiB；下载均与 HF LFS SHA256 核对。Hunyuan 官方源权重也核对了 SHA256，转换输出另有清单。没有覆盖或删除旧模型。

截至调研时，官方 Hunyuan 仍为 **1.5**；PP-OCR 仍为 **v6**。RapidOCR Python 库已有 **3.10.0**，本机另行安装并与 Clipaste 的 **RapidOcrNet 3.0.0** 对比；两者是不同语言实现，不是直接的版本升级关系。没有发现需要将 Clipaste 的 v6 small 权重替换为“v7”的依据。[Hunyuan 官方仓库](https://github.com/Tencent-Hunyuan/HunyuanOCR)、[RapidOCR PyPI](https://pypi.org/project/rapidocr/)

## Hunyuan 重复输出的原因和更优做法

Clipaste 的 `VERSION.txt` 显示旧包为 2026-07-09 转换、llama.cpp b9941。旧 GGUF 只包含 `tokenizer.ggml.eos_token_id=120020`，缺少 `120007` 的结束标记。当前官方 `generation_config.json` 列出两个结束 token `[120007,120020]`。b11960 转换器保留 EOS 120020，并新增 EOT 120007。[官方生成配置](https://huggingface.co/tencent/HunyuanOCR/blob/main/generation_config.json)、[转换器](https://github.com/ggml-org/llama.cpp/blob/b11960/conversion/hunyuan.py)

控制实验用同一图片、同一旧权重、同一 b11960 服务，交叉测试自定义提示词、官方 structured_parse、官方 doc_parse，以及是否 `--jinja`：

| 运行配置 | 三种提示词结果 |
|---|---|
| 原停止配置 | 全部生成到 512 token 上限，重复尾文 |
| 仅 `--jinja` | 全部仍到 512 token 上限 |
| 仅运行时覆盖 EOS 为 120007 | 全部 42 token 正常 `stop` |
| EOS 覆盖 + `--jinja` | 全部 42 token 正常 `stop` |

重新转换后逐个张量 SHA256 比对：语言模型 **266/266** 相同，视觉投影 **447/447** 相同。这次修复的关键是转换元数据，不是模型权重精度提高。新版 GGUF 无需手工覆盖即正常结束。

**建议使用当前官方转换器生成的双停止标记 GGUF**，配匹配的 projector、官方任务提示词、greedy 生成、llama.cpp 的 `repeat_penalty` 参数。不要只靠删除重复尾文：Clipaste 的兜底在本次样本中既留下了残余重复，也出现过空结果。旧包紧急兼容可补 `--override-kv tokenizer.ggml.eot_token_id=int:120007`，但长期优先用重新转换并核验的文件；不要用 `--ignore-eos`。

官方客户端里还有 vLLM 风格 `repetition_penalty`；llama.cpp 原生字段为 `repeat_penalty`。新路径明确发送后者。官方表格、公式、纯文字使用各自固定任务提示词；本地文字模式采用 `提取图中的文字。`，不靠冗长自由提示词弥补停止配置错误。

第三方文件名带 `Updated` 不足以证明更新。检查发现 `ggml-org/HunyuanOCR-GGUF` 的最近修改为 2026-05-21，早于 1.5 发布；本次没有把它误当新版 1.5 下载替换。选择当前腾讯官方 HF 快照自行转换更容易核对来源。

## DFlash 实测

DFlash 已合并到 llama.cpp 上游（官方文档说明 b11103 起支持），不需要旧分支。配置为当前主模型 + 官方 draft GGUF、`--spec-type draft-dflash --spec-draft-n-max 15 --parallel 1 -fa on --jinja`。[官方 PC 部署说明](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/main/docs/llama_cpp.md)

下表为两次请求，包含图片编码、prefill、生成和 HTTP，不含模型加载；关闭 prompt cache。样本很少，仅说明本机可用。

| 样本 | 普通 AR | DFlash | 输出 |
|---|---:|---:|---|
| 合成中英短截图，42 输出 token | 0.242 / 0.200 秒 | 0.228 / 0.121 秒 | 文本完全相同 |
| 官方文档样例，约 970 输出 token | 4.982 / 4.844 秒 | 2.902 / 2.785 秒 | 均正常 stop；只观察到等价 LaTeX 下标括号差异，如 `S_i` / `S_{i}` |

长文档端到端约快 1.7 倍。不能声称在所有输入上逐 token 相同；短截图常被加载开销主导，因此保留普通版为默认可选基线，DFlash 单独列出。

## 三张合成截图的识别对比

样本为中英订单、18px 小字、轻度模糊发票，分别有 81、77、68 个去空白字符。CER 使用编辑距离 / 真值字符数，只忽略空白，不改标点或数字。它是小规模烟测，不代表真实截图、手写、复杂排版的总体排名。

| 引擎 | 中英截图 CER | 小字 CER | 模糊截图 CER | 中英截图识别阶段耗时（含加载） |
|---|---:|---:|---:|---:|
| MiniCPM-OCR 1.0 F16 | 0% | 1.30% | 1.47% | 2.69 秒 |
| Clipaste PP-OCRv6 / RapidOcrNet 3.0.0 | 0% | 1.30% | 0% | 2.26 秒 |
| 相同 PP-OCRv6 / RapidOCR 3.10.0 | 0% | 1.30% | 1.47% | 1.59 秒 |
| Clipaste 旧 Hunyuan GGUF + 原重复处理 | 343.21%（大量重复） | 100%（空结果） | 5.88% | 3.65 秒 |
| 当前重新转换 Hunyuan 1.5 F16 | 0% | 1.30% | 0% | 1.80 秒 |

新版 Hunyuan 在这三张图都正常结束，纯推理为 0.237–0.280 秒；MiniCPM 为 0.456–0.506 秒。PP 两条路径的初始化、预处理和组行不同；.NET 的记录包含首次模型初始化，不能直接用其内部 inference_seconds 与 GPU 纯推理数字比较。

本轮没有证据支持为了精度而替换 Clipaste 的 PP-OCRv6 实现。Hunyuan 的旧停止配置值得优先修正。MiniCPM 则是可直接运行的独立候选，特别适合继续评估结构化区域；本轮没有做大规模文档精度测评。

## Jev 的有效性与限制

用官方仓库的原始图和候选，实际经过本机渲染与分类：

| 官方样例 | 参考候选分数 | 第二候选分数 | 含义 |
|---|---:|---:|---|
| Putnam 文字列表 | 0.99518 | 0.00296 | 能区分正确列表与乱序结果 |
| Putnam 公式 | 0.97344 | 0.96841 | 两种等价 LaTeX 都得到高分 |
| 中文标准表格 | 0.05921 | 0.00124 | 排序正确，但正确参考的绝对分也很低 |

官方自带表格样例也记录了正确参考约 0.04075 的低分。**统一阈值无法覆盖所有类别**。另一方面，合成模糊截图出现一个错字时仍可能超过 0.96，所以高分也不能保证逐字正确。实际应先积累带人工标签的本地样本，按文字/公式/表格分别校准阈值，或优先用候选相对排序。[官方样例及说明](https://github.com/OpenBMB/MiniCPM-OCR/tree/main/scripts/minicpm-ocr-jev/examples)

更稳妥的流程是：快速 OCR → Jev 复核 → 有效低分时换一个 OCR → 保留并比较候选 → 未通过或评分失败则人工核对。不要无限次重跑确定性识别，也不要把 Jev 当自动纠错器。

## 验证证据

机器可读结果保存在 [ocr-local-results.json](reports/ocr-local-results.json)，包含完整预测、CER、EOS 对照、DFlash 请求耗时和输出差异；不含访问凭据或用户原图。合成图可用 `scripts/validation/ocr_samples.py` 重建，原始试验材料留在 `.tmp/ocr-research`。

本地模型真实推理、官方三类 Jev 样例、鉴权和决策逻辑测试均已运行。浏览器验证使用真实 Chromium 的无头渲染，不等同于人工桌面交互测试。每个任务结束后检查仅本机 Playground 仍监听，OCR 子模型释放。
