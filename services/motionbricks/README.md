# NVIDIA MotionBricks 骨骼动作生成

在 Playground 选择「骨骼动作生成」，再选 `motionbricks-soma`（人体）或 `motionbricks-g1`（Unitree G1）。输入框留空，点击「生成动作」即可运行步行样例。返回后可以播放、拖动进度条或下载动作 JSON。点击「下载输入姿态」可获得同一骨架的输入模板，修改后粘贴到输入框即可再次生成。

## API

向本机 API 监听的 `POST /v1/motions/generations` 发送 JSON，使用该监听配置的 API key。下面的 PowerShell 示例从环境变量读取凭据。

```powershell
$headers = @{ Authorization = "Bearer $env:FAKE_OLLAMA_API_KEY" }
$url = 'http://127.0.0.1:21435/v1/motions/generations'
$motion = Invoke-RestMethod -Method Post -Uri $url -Headers $headers -ContentType application/json -Body '{"model":"motionbricks-soma","example":"walk"}'
$motion | Select-Object model, fps, num_frames, pose_format

# 从上一段的最后四帧继续，只取一个目标姿态
$next = @{
  model = $motion.model
  context_poses = $motion.frames[-4..-1]
  target_poses = @(,$motion.input.target_poses[0])
}
Invoke-RestMethod -Method Post -Uri $url -Headers $headers -ContentType application/json -Body ($next | ConvertTo-Json -Depth 12 -Compress)
```

自定义输入需要连续的四个 `context_poses` 和一至四个 `target_poses`，相邻帧间隔为 1/30 秒。两套骨架都使用米、Z 向上和 XYZW 单位四元数。SOMA 每帧按返回的 `joint_names` 排列，每个关节含相对父关节的 XYZ 平移和 XYZW 旋转。G1 每帧为 36 个数，依次是根节点 XYZ、XYZW 和按 `joint_names` 排列的 29 个关节角（弧度）。

返回的 `frames` 是经过官方引擎端点校正的动作，可直接作为下一次输入。每段有 18 至 58 帧，`foot_contacts` 与帧一一对应。`preview.positions` 和 `preview.bones` 用于画世界空间的骨架。`input` 保存本次输入姿态。

## 部署

本机安装目录为 `J:\Projects\LLM_Models\MotionBricks`。`soma` 和 `g1` 保存 Hugging Face 原始 SafeTensors 与清单。`runtime` 保存 NVIDIA 官方代码、对应的 FP32 ONNX、独立 Python 环境及 TensorRT 计划缓存。每套骨架的首次启动会编译计划，后续请求复用缓存。

在装有 NVIDIA GPU、驱动、Git LFS、Hugging Face CLI 与 uv 的 Windows 主机上可按以下命令安装。

```powershell
$root = 'J:\Projects\LLM_Models\MotionBricks'
hf download nvidia/MotionBricks --revision 3cb362d9b95aa40462abcb4d8cfae9ffb6d97410 --local-dir $root
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/NVlabs/motionbricks.git "$root\runtime"
git -C "$root\runtime" checkout 54fc563aeaaf819a753d72cc9fdc6713989cdd43
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
Set-Location "$root\runtime"
git lfs pull --include="motionbricks/assets/onnx/*/*_fp32.onnx,motionbricks/assets/human/*,motionbricks/assets/robots/unitree_g1/**,assets/motions/*/locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.*"
uv sync --locked --python 3.12
```

将示例中的 `generic_openai_targets` 和 `model_profiles` 项追加到网关配置，再将 `exposed_models` 项追加到需要提供此能力的 API 监听。按本机目录调整 `cwd` 和启动命令中的脚本路径，然后启动网关。每套骨架使用一个按需启动的本地进程，默认空闲五分钟后释放。[Windows 配置片段](config.windows.json)

权重适用 NVIDIA Open Model Agreement。内置步行样例使用 NVIDIA 随附的 BONES 片段，适用单独的样本评估许可。[官方模型](https://huggingface.co/nvidia/MotionBricks) · [NVIDIA 推理代码与许可](https://github.com/NVlabs/motionbricks)
