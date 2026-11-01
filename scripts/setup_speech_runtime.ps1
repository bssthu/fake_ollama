param([string]$Proxy = 'http://127.0.0.1:6268')
$ErrorActionPreference = 'Stop'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$env:UV_CACHE_DIR = Join-Path $repoRoot '.tmp\uv-cache'
$env:HTTPS_PROXY = $Proxy
$env:HTTP_PROXY = $Proxy
$env:ALL_PROXY = $Proxy
Set-Location -LiteralPath $repoRoot
$torchWheel = Join-Path $repoRoot '.tmp\startlux-setup\torch-2.8.0+cu128-cp313-cp313-win_amd64.whl'
$torchSource = if (Test-Path -LiteralPath $torchWheel) { $torchWheel } else { 'torch==2.8.0+cu128' }
uv venv --allow-existing --python C:/Python313/python.exe services/speech/.venv
if ($LASTEXITCODE) { throw 'speech venv creation failed' }
uv pip install --python services/speech/.venv/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match $torchSource -r services/speech/requirements.txt
if ($LASTEXITCODE) { throw 'speech dependencies failed' }
uv venv --allow-existing --python C:/Python313/python.exe services/speech/qwen-asr/.venv
if ($LASTEXITCODE) { throw 'Qwen ASR venv creation failed' }
uv pip install --python services/speech/qwen-asr/.venv/Scripts/python.exe --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match $torchSource -r services/speech/requirements-qwen.txt
if ($LASTEXITCODE) { throw 'Qwen ASR dependencies failed' }
Write-Output 'Installed packages only. No model has been loaded.'
