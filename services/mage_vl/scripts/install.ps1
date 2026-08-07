# Install the standalone Mage-VL service and model environment.
param(
    [string]$InstallRoot = "",
    [string]$UvPath = "",
    [string]$BasePythonPath = "",
    [string]$ProxyUrl = "",
    [switch]$IncludeStreamingGate
)

$ErrorActionPreference = "Stop"
$serviceRoot = Split-Path -Parent $PSScriptRoot
if (-not $InstallRoot) {
    $InstallRoot = if ($env:MAGE_VL_INSTALL_ROOT) {
        $env:MAGE_VL_INSTALL_ROOT
    } else {
        Join-Path $env:LOCALAPPDATA "fake-ollama\mage-vl"
    }
}
if (-not $UvPath) {
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCommand) { $UvPath = $uvCommand.Source }
}
if (-not $BasePythonPath) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) { $BasePythonPath = $pythonCommand.Source }
}
$modelPath = Join-Path $InstallRoot "Mage-VL"
$venvPath = Join-Path $InstallRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $UvPath -PathType Leaf)) {
    throw "uv is missing: $UvPath"
}
if (-not (Test-Path -LiteralPath $BasePythonPath -PathType Leaf)) {
    throw "Base Python is missing: $BasePythonPath"
}
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null

if (-not (Test-Path -LiteralPath $modelPath)) {
    $env:GIT_LFS_SKIP_SMUDGE = "1"
    git clone https://huggingface.co/microsoft/Mage-VL $modelPath
    if ($LASTEXITCODE -ne 0) { throw "Mage-VL metadata clone failed." }
}
$venvHealthy = $false
if (
    (Test-Path -LiteralPath $pythonPath -PathType Leaf) -and
    (Test-Path -LiteralPath (Join-Path $venvPath "pyvenv.cfg") -PathType Leaf)
) {
    $venvConfig = Get-Content -LiteralPath (Join-Path $venvPath "pyvenv.cfg") -Raw
    $venvHealthy = $venvConfig -like "*$BasePythonPath*"
}
if (-not $venvHealthy) {
    & $UvPath venv --python $BasePythonPath --clear $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Mage-VL virtual environment creation failed." }
}

& $UvPath pip install --python $pythonPath `
    torch==2.12.1 torchvision==0.27.1 `
    --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch installation failed." }

& $UvPath pip install --python $pythonPath --editable $serviceRoot
if ($LASTEXITCODE -ne 0) { throw "Mage-VL adapter dependency installation failed." }

$downloadScript = Join-Path $PSScriptRoot "download_weights.ps1"
& $downloadScript -ModelPath $modelPath -ProxyUrl $ProxyUrl -IncludeStreamingGate:$IncludeStreamingGate
if ($LASTEXITCODE -ne 0) { throw "Mage-VL weight download failed." }

& $pythonPath -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "PyTorch CUDA validation failed." }

Write-Output "Mage-VL base installation is ready at $modelPath"
