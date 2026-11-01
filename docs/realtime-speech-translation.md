# Playground 即时语音翻译

在 `/playground/` 先选择功能，再选择该功能的模型。语音翻译、OCR、对话等共用连接与功能入口；语音内部选择一个识别模型和一个翻译模型。三种音源为浏览器显式授权的共享音频、服务器 Windows 系统回放、用户选择的音视频文件。

用户于 2026-11-01 允许使用 GPU 后，已启用本机 `config/local/speech.json` 中的 `inference_enabled`。初版分段实现的四种 ASR 与三种翻译器共 12 个组合已完成真实 GPU 推理；本轮另验证七个模型的隐私内存路径、三种音源和 Voxtral 原生流式，具体范围见下方记录。默认选择 Whisper large-v3 turbo + Qwen3.5 2B，目标语言为中文；输入语言始终自动检测。查看页面、模型列表和设备列表不会加载模型。

常用的 [Playground](http://127.0.0.1:21431/playground/) 保持只监听本机。初版分段实现的 GPU 与生命周期记录保留在 [历史验收记录](reports/speech-live-results.json)；功能优先、内存传输和原生流式的验证记录见 [本轮改造记录](reports/speech-memory-streaming-results.json)。历史计时不代表新版的流式延迟。

六个新增模型共 58 个文件、36,272,535,869 字节已下载并通过完整哈希校验；复用已有 Qwen3.5-2B，无残留下载分片。最初不占用显卡的开发记录保留在 [开发验证记录](reports/speech-development-results.json)，其中的“未进行模型测试”只描述第一阶段，不代表后续 GPU 验收状态。

## 需求与实现边界

| 需求 | 实现 |
| --- | --- |
| 浏览器所在电脑的声音 | 点击开始后调用浏览器共享选择框；用户必须显式授权并分享音频，不上传画面。支持范围由浏览器与系统决定 |
| 服务器正在播放的声音 | Windows WASAPI loopback；可选默认播放设备或指定回放设备，不使用麦克风；无新数据包约 0.8 秒时也提交尾段 |
| 视频、音频文件 | 浏览器按解码器要求读取文件范围，PyAV 只解码首条音轨为 16 kHz 单声道 PCM；支持 seek，不整段读入内存或保存副本；默认文件上限 2 GiB |
| 字幕延迟 | Whisper 默认最长 8 秒；Qwen 推荐 16 秒，窗口可设 3–30 秒；Voxtral 显示增量原文，按独立间隔更新临时译文，在停顿或上下文上限处确认最终译文 |
| 输入语言未知、中途切换 | 不强制源语种，每个音频段重新执行自动识别；Whisper 开启 multilingual |
| 一段中混合多种语言 | Qwen ASR 和 Whisper turbo 已通过中英合成音频切换样本；其他模型及短促换语仍可能漏句，详见实测边界 |
| 选择目标语言 | 输入语言代码，例如 zh、en、ja、pt-BR；不支持的 TranslateGemma 方向会保留原文并显示翻译错误 |
| 原文先出现 | ASR 完成即发布该段原文，翻译完成后更新同一行 |
| 长时间运行 | 有界音频和字幕缓存，显示积压秒数；系统采集过载时明确停止，不静默跳段 |
| 文件完整处理 | 解码服从背压，不因处理慢丢弃文件中的片段；文件按处理速度前进，不与播放器同步 |
| 取消与释放 | 停止采集并处理余音，或立即停止并释放；页面断开 45 秒、服务关闭时清理自有进程 |

Voxtral 默认使用 Transformers 原生增量编码与解码，维护同一语句的编码器、卷积及解码器缓存，不会每收到一块音频就重新识别整段。页面可选 240、480、960、1200、2400 ms 识别延迟（默认 480 ms）；参数只修改进程内处理器，不修改模型文件。流式上下文默认上限 30 秒，可设 5–120 秒，停顿或到达上限后提交并重建缓存。识别延迟参数不是包含采集、网络、翻译的总延迟保证。该上游流式接口仍为实验性 API，因此本机固定已验证的 Transformers 5.22.1。

“临时译文更新间隔”独立于上述两个参数，默认 2 秒，可设 0.5–30 秒。首次出现尚未翻译的新原文后开始按真实时间计时，持续新增内容不会重置截止时间；到期后翻译当前语句的最新完整原文，更新同一字幕行。翻译执行期间识别继续，已有译文保留，并标注为临时译文或新内容待翻译。处理较慢时只合并保留最新待译内容，不累积逐词翻译任务。文件按最快速度解码，因此该间隔不是文件时间轴上的分段长度。

停顿、达到上下文上限或结束时，不再等待翻译间隔：等待正在进行的翻译结束，再校正最终原文对应的译文；若已经翻译过完全相同的原文则直接复用。只有最终确认的字幕进入后续翻译上下文。立即停止或断开会取消定时与正在执行的翻译，并释放受管进程。JSON 请求字段为 `partial_translation_interval_seconds`，仅原生流式模式使用；导出 SRT 时仍未确认的译文会明确标记为临时译文。

三个翻译模型均已用真实 Voxtral 文件输入验证，同一条字幕在结束前多次更新临时译文；默认 2 秒间隔下，首轮约在原文出现后 2.8–3.4 秒显示。这里是 26 秒样本按处理速度解码的观察值，不包含模型冷启动，不等同于实时采集的总延迟保证。回归及取消回收记录见 [流式临时译文验证](reports/speech-partial-translation-results.json)。

Qwen 官方 `init_streaming_state` / `streaming_transcribe` 当前只支持 vLLM；本机 Windows Transformers 运行时未启用该接口，页面明确标注分段模式。保留 auto 输入，推荐 16 秒窗口、至少约 8 秒语境后在停顿提交；播放结束、断流和手动停止仍立即提交短尾段。

采集层保留可调 RMS 能量阈值，默认 0.003。Whisper 另外用较保守的 Silero 阈值拦截整段非语音，但不裁剪传给模型的音频，避免删除高音对白及语言上下文；同时启用词时间与静音幻觉抑制。页面可以关闭非语音过滤作对照。不使用“感谢观看”或其译文黑名单，因为这些词也可能是真实台词。音乐、短句和夸张声线仍可能造成模型幻觉。

停顿切分前保留 300 ms，达到硬上限时下一段保留 400 ms 音频重叠；只在已知重叠边界去掉足够长的完全相同文字前缀。字幕时间是段级采集位置，不能用于声称逐词对齐。

## 模型与目录

| 选择项 | 后端 | 本机路径 |
| --- | --- | --- |
| Qwen3-ASR 1.7B | 官方 qwen-asr / Transformers 4.57.6 独立进程 | `J:/Projects/LLM_Models/Qwen/Qwen3-ASR-1.7B` |
| Whisper large-v3 | faster-whisper / CTranslate2 | `K:/Projects/LLM_Models/OpenAI/faster-whisper-large-v3` |
| Whisper large-v3 turbo | faster-whisper / CTranslate2 | `K:/Projects/LLM_Models/OpenAI/faster-whisper-large-v3-turbo` |
| Voxtral Mini 4B Realtime 2602 | Transformers 5.x，原生流式 / 可选分段 | `K:/Projects/LLM_Models/Mistral/Voxtral-Mini-4B-Realtime-2602` |
| Qwen3.5 2B | Transformers，复用已有权重 | `J:/Projects/LLM_Models/Qwen/Qwen3.5-2B` |
| Qwen3.5 4B | Transformers | `J:/Projects/LLM_Models/Qwen/Qwen3.5-4B` |
| TranslateGemma 4B | Transformers + CPU 文字语种检测 | `K:/Projects/LLM_Models/Google/translategemma-4b-it` |

只保留用户选中的一个识别模型和一个翻译模型，按会话运行；切换模型须结束原会话。进程启动不会自动下载权重，运行时使用 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和本地路径。

Whisper 权重为 SYSTRAN 的 large-v3 和 Mobius 的 turbo CTranslate2 格式；不是额外保存一份原始 Transformers 权重。Voxtral 只下载 Transformers 的 `model.safetensors`，不同时下载重复的 `consolidated.safetensors`。

Google 官方 TranslateGemma 仓库需要账号授权。本机未配置授权凭据，用户已选择公开社区分发；使用 [RedHatAI/translategemma-4b-it](https://huggingface.co/RedHatAI/translategemma-4b-it)，固定版本 `7da9de35f146086cc1e45e75563c5d62a079aaf6`。两块权重及 tokenizer/配置文件的哈希与 Google 上游 `10042cb0e6e7fdce748996a71dc3dc432a4e0c89` 一致，保留分发者的 README/NOTICE。模型仍遵守 Gemma 许可。

### 自动语言与 TranslateGemma

[TranslateGemma 官方模板](https://huggingface.co/google/translategemma-4b-it#usage)要求每条输入带 `source_lang_code` 和 `target_lang_code`，不能直接传 `auto`。本功能先按分句检测语种，避免同段日文把前面的中文全部吸收到日语标签中；相邻同语种再合并，保留上下文。每段分别构造合法模板，再合并译文，因此页面不要求用户指定源语言。检测切分不会拆开小数或千位分组数字。

文字语言检测对短句、专名和短促的句内换语可能不可靠。检测结果只标作推断，不等同于音频识别器给出的语种。无法识别或目标语言不受模板支持时，字幕行保留原文并显示错误，不默默改用英语，也不暗中换翻译模型。Qwen 翻译器可直接处理混语并使用最近四段的有界上下文；TranslateGemma 使用自身专用模板，不额外注入不符合模板的系统提示词。Voxtral 未提供可靠 LID 时显示 unknown，不编造语言标签。

## 配置与启动

软件依赖与主网关环境隔离：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_speech_runtime.ps1
.venv/Scripts/python.exe scripts/configure_speech.py
```

`configure_speech.py` 仅在配置不存在时创建 `config/local/speech.json`，并写入主配置的 `playground_speech_config`，不启用推理，不重启服务。已有配置不会被重复运行覆盖。Qwen ASR 环境单独固定官方依赖，避免改变主网关和其他模型服务的软件版本。

新安装默认不启用推理；准备好本地环境与权重后，将语音配置中的 `inference_enabled` 改为 `true`，正常重启/部署含新路由的 fake-ollama 代码。本机已经启用。单纯的语音配置内容在请求时重新读取；主配置的路径变更遵循项目已有配置加载规则。

```json
{
  "playground_speech_config": "I:/Projects/fake_ollama/config/local/speech.json"
}
```

音频任务继承 Playground 认证，WebSocket 首帧携带当前页面凭据，校验同源及真实监听端口；会话绑定凭据和页面随机标识。文件读取只接受所选文件的字节，不接受服务器任意文件路径或网络 URL，解码器禁止外部协议。兼容 HTTP 上传只在内存中保留最多 32 MiB，大文件使用页面按需读取。

`playground_server_audio_remote_enabled` 是独立的设置开关，默认 `false`，可在 Admin 的 Playground 设置中修改。默认只允许本机页面采集服务器声音；代理来源、非回环 Host 不视为本机用户。开启后仍要求有效 Playground 访问权限，不会自动修改监听地址。本轮没有开放局域网或公网监听。

## 生命周期、资源与隐私

- 点击开始后先加载两个模型，页面明确显示加载阶段；加载完成后才打开系统回放采集，避免把冷启动期间的音频无限排队。
- 服务端最多排队 40 个音频包，文件解码服从背压；浏览器与系统实时源过载时明确停止，不静默丢音频或转存磁盘。Voxtral 输入环形缓存上限约 128 KiB。
- “停止采集并处理余音”提交短尾段并排空队列。“立即停止并释放模型”取消推理和解码，停止整个自有进程树。
- 页面关闭时发送停止请求；即使请求没送达，服务端连续 45 秒没有收到该会话的请求也会清理。文件上传请求也续期。
- 语音输入、原文、译文和进程控制均通过有界内存通道传递，不再创建上传、PCM 或请求结果邮箱文件。内部 IPC 使用经随机密钥鉴权的临时回环 TCP 连接，接受自有子进程后立即关闭监听；Windows Job 仍负责整棵自有进程树。
- `--privacy-mode` 由 server 启动时统一设置，通过环境传给受管进程；页面只读显示状态。严格模式关闭字幕导出和 Dashboard 历史写入，普通 OCR 使用内存路径；未验证后端及仍需磁盘渲染的 Jev 拒绝执行。Python 模型进程额外安装写文件拦截，禁用内容日志和 CUDA 编译缓存。
- 这里保证应用与受管组件不保存会话内容；操作系统分页、休眠、转储、浏览器自身行为以及调用者的终端重定向不属于该开关能完全控制的范围。已有模型、原始文件与旧日志不删除。
- 翻译生成陷入重复或达到输出上限时，Qwen 使用同一模型去掉历史上下文重试一次，并在该字幕的提示中注明。再次失败则保留原文并显示错误；不会把截断译文标记为完成，也不会暗中换模型。
- 字幕不写持久化日志，语音 API 响应带 `Cache-Control: no-store`；服务端保留最多 2,000 个更新事件，浏览器保留最近 2,000 段。导出 JSON/SRT 只包含页面当前保留范围。SRT 包含原文及译文，失败或未完成译文有明确占位。
- 每段显示处理耗时；系统回放另显示采集段尾到译文发布的延迟，不含设备缓冲及约 0–0.2 秒的页面推送间隔。文件按最快处理速度解码，有界预读会保持非零积压，这不是实时声源持续落后。
- 模型返回空文字时，页面累计显示空结果段数；这可能是静音或音乐，也可能是漏识别，不能把已处理秒数当作识别覆盖率。

## 本轮内存与流式实测

2026-11-01 的完整离线回归、真实浏览器和 GPU 记录见 [本轮改造记录](reports/speech-memory-streaming-results.json)。验证包含七个模型在隐私模式下加载推理，PP-OCR 与 MiniCPM OCR 的内存识别、复用和释放；浏览器原生音频共享、拒绝授权、授权等待期间取消；真实 WASAPI 播放结束后自动提交尾段。

Voxtral 以同一进程连续运行 12 次 30 秒输入，结束后的线程数始终为 2，CUDA 已分配内存保持 8,867,886,592 字节，音频缓冲归零。完整 MKV 的 659.80 秒音频经过 23 次上下文提交，处理完成后队列清空并释放进程。流式输出中途取消与关闭页面分别约 0.83 / 0.33 秒释放资源。这些是已测时间范围内的结果，不是无限运行的无泄漏证明。

同一片段按实时速度送入，480 / 2400 ms 档首字分别约 2.30 / 4.20 秒；包含本次采集等待与推理，不包含冷启动和翻译。默认 480 ms 档对整段日语动画有 11 个上下文返回空文字。两个问题片段在整段和流式推理下同样接近空结果；把其中 60.2–90.2 秒的片段调至 2400 ms 后，从空结果恢复为 67 字输出。它说明延迟设置在此素材上会影响识别覆盖，但没有据此宣称逐字准确率；动画素材建议对比更长延迟。

## 下载和校验

`scripts/setup_speech_models.py --plan-only` 从模型仓库读取固定 revision、文件大小和哈希，写 `config/local/speech-downloads.json`；不启动推理。随后可用普通续传，或并发 Range 下载：

```powershell
.venv/Scripts/python.exe scripts/setup_speech_models.py --proxy http://127.0.0.1:6268 --plan-only
.venv/Scripts/python.exe scripts/download_speech_ranges.py --workers 16
```

大文件使用 `.partial`，并发下载另有有限数量的 32 MiB 分块和范围完成清单；这不是模型备份。每段验证 HTTP Content-Range 和大小；全文件 SHA-256（LFS）或 Git blob SHA-1 匹配后才改为正式文件名。中断后保留可续传数据，不复制已存在模型。本次大权重使用较快的公开 ModelScope 传输 URL，配置和元数据来自 Hugging Face；完成条件仍为清单内固定 Hugging Face 版本的相同哈希。最终逐文件结果在 `config/local/speech-download-results.json`。

已有 K 盘模型通过 `scripts/organize_local_models.ps1 -Apply` 同盘目录移动到 `DeepSeek`、`Hunyuan`、`OpenBMB`、`PaddlePaddle`。移动前后记录并比较文件清单、大小和最后修改时间，收据为 `config/local/model-move-receipt.json`。没有模型复制、备份或跨盘迁移。更新本项目 OCR 引用，以及项目 DeepSeek 启动器实际使用的 `K:/Projects/Xolotl-runtime/windows-4090.json`。历史测试报告和旧配置快照不作为活动配置重写。

## 初版分段实现的历史验收

本节保留最初 GPU 接入的数据；下述文件邮箱、轮询和历史计时属于旧实现。新版不使用这些内容临时文件，当前验证命令见文末。

测试机器为 Windows / RTX 4090 24 GiB。实际媒体为用户提供的 `R:/测试视频.mkv`：日语动画，有背景音乐、多位说话者和高音对白。短测试取 90 秒起的 8 秒或 26 秒；持续测试把整段音轨重复封装为文件，按处理速度运行，**持续文件处理不能作为持续系统回放的延迟数据**。真实 WASAPI 验证播放的是 26 秒片段。

混语音频另由 Windows 离线 Huihui/Zira 合成，先中文后英语，共约 13.80 秒。它证明测试样本的语种切换处理，不能代替自然句内混语的全面准确率评估。文字翻译还测试了中英日混合、数字、货币、否定和英译日。

已观察到的质量限制：

- Qwen ASR 把一个 8 秒日语片段误判为英语并输出罗马字；26 秒版本可以识别为日语。其他模型对很短的尾段也可能误判语种。
- Qwen ASR 和 Whisper turbo 保留了上述中英混合样本的两种语言；Whisper large-v3 和 Voxtral 对整段直接推理时漏掉了英语部分。自动检测不代表所有模型都能完整识别每种混语。
- 初版 Whisper 裁剪式 VAD 曾过滤掉可听见的日语对白；新版改为保守的整段非语音过滤并保留识别上下文。完全零静音直接返回空结果；音乐、噪声、短句仍可能导致模型幻觉。
- 低音量直接 PCM 测试中，Whisper 仍有输出，Voxtral 返回空文字。实际采集时还需结合静音阈值调整。
- 译文仍可能有专名、措辞或语种错误；短句提示词回归和 TranslateGemma 分句检测已经改善已发现的问题，但没有宣称经过全部语言的准确率验收。

首次模型加载与连续字幕延迟分开看。系统回放结果的 `source_latency_seconds` / `translation_latency_seconds` 从采集到段尾帧计至服务端发布，包含排队和推理；不含声音设备缓冲、浏览器下一次轮询，也不含从句子开头到分段结束的等待。文件源只记录处理耗时，避免把快速预读后的队列时间误报成真实播放延迟。

Windows 实测发现进度文件替换与结果邮箱读取/删除存在短暂共享冲突，已统一加入最多约 0.5 秒的有界重试。持续失败仍会报错并释放模型，不隐藏错误或跳过音频。三个回归用真实 Windows 独占文件句柄验证了读、删、替换恢复；全量离线测试为 **1,032 通过、4 跳过、17 排除**，字幕前端测试 2 项通过。

20 分钟持续文件处理已完成，未再出现 I/O 故障，并正常处理余音、释放全部模型。保留的最后 1,000 条记录中有 7 条 Qwen 2B 译文生成过长；其余片段继续处理，原文和错误均可见。随后新增生成重复保护及单次重试，用这些片段的原上下文逐一重放，7 条全部恢复，翻译耗时约 0.7–1.1 秒。**这项生成修复采用针对性回归和后续短时端到端复测，没有把修复前的 20 分钟结果写成修复后无错误的长测。**

修复后额外连续处理 2 分钟、184 段，翻译失败为 0；Qwen 4B 和 TranslateGemma 也分别回归通过。最终真实回放的 4 段样本中，段尾至服务端原文发布 P50/P95 为 0.661/0.676 秒，译文为 1.047/1.260 秒；默认 8 秒窗口还需加上分段等待，不能把该数字理解为从句子开头起的总延迟。样本数较小，不是所有语言或音源的延迟保证。

20 分钟测试采样的总显存峰值为 10,478 MiB，测试服务及子进程工作集峰值约 2,923 MiB；结束后只剩网关进程，工作集回落至约 86 MiB。真实异常路径检查包括：未结束 HTTP 上传约 0.004 秒可取消，加载中取消约 0.219 秒，页面失联约 46.3 秒自动回收，两个模型已加载时服务退出约 0.408 秒。损坏媒体报错后、正常停止、刷新和关闭页面均验证了资源释放。

当前重跑脚本（会使用真实模型；先确保显卡可用）：

```powershell
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONUTF8 = '1'
.venv/Scripts/python.exe scripts/validation/speech_refactor_models.py --models voxtral-mini-4b-realtime-2602 --streaming --privacy --repeat 12 --seconds 30
# 在单独终端启动隔离服务；验证结束通过 stop-server 文件正常关闭。
.venv/Scripts/python.exe scripts/validation/speech_server.py --privacy-mode
services/ocr/.venv/Scripts/python.exe scripts/validation/speech_refactor_browser.py --file .tmp/speech-live/japanese.wav
services/ocr/.venv/Scripts/python.exe scripts/validation/speech_refactor_browser.py --loopback .tmp/speech-live/japanese.wav
services/ocr/.venv/Scripts/python.exe scripts/validation/speech_browser_capture.py
services/ocr/.venv/Scripts/python.exe scripts/validation/ocr_private_memory.py
services/ocr/.venv/Scripts/python.exe scripts/validation/speech_disconnect.py
services/ocr/.venv/Scripts/python.exe scripts/validation/speech_partial_translation.py
.venv/Scripts/python.exe scripts/validation/speech_file_transport.py
```

浏览器脚本使用已有 OCR Playwright 环境。`speech_browser_capture.py` 使用完整 Chromium 和仅供测试的自动选择标签页开关，实际执行 `getDisplayMedia`、AudioWorklet、WebSocket、ASR 与翻译；另用浏览器原生拒绝授权开关验证不创建会话。产品代码始终需要用户点击并授权，不携带这些测试开关。

`speech_material_diagnostic.py` 对测试视频的不同窗口做对照；固定日语仅用于诊断，不进入生产 auto 模式。前 96 秒的固定 8 秒窗口中，Qwen 的 12 次语言标签有 6 次不为日语；16 秒窗口的 6 次标签有 1 次不为日语。画面字幕对照发现标题、短人名、道具名和夸张声线仍会出错。此统计只说明该样本的语言标签改善，不是逐字准确率或通用性能保证。Whisper 的裁剪式 VAD 曾在片头生成 “Thank you.” 并漏掉部分高音台词；新版采用整段非语音过滤、保留原音频上下文及静音幻觉抑制，仍不保证杜绝音乐和短句幻觉。

调研输入：用户提供的 `R:/realtime_speech_translation_research_2026-10-31.md`。适配接口依据：[Qwen 官方代码](https://github.com/QwenLM/Qwen3-ASR)、[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[Voxtral Transformers 接口](https://huggingface.co/docs/transformers/model_doc/voxtral_realtime)、[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B)。
