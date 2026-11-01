param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$modelRoot = [IO.Path]::GetFullPath('K:\Projects\LLM_Models')
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$moves = [ordered]@{
    'DeepSeek-V4.1-Flash-1017-Engram' = 'DeepSeek'
    'DeepSeek-V4.1-Flash-1017-XQ-Uncensored' = 'DeepSeek'
    'HunyuanOCR-1.5-b11960' = 'Hunyuan'
    'HunyuanOCR-1.5-HF' = 'Hunyuan'
    'MiniCPM-OCR-1.0-gguf' = 'OpenBMB'
    'MiniCPM-OCR-Jev-gguf' = 'OpenBMB'
    'PP-OCRv6-small-onnx' = 'PaddlePaddle'
}
$records = @()
foreach ($name in $moves.Keys) {
    $source = [IO.Path]::GetFullPath((Join-Path $modelRoot $name))
    $destination = [IO.Path]::GetFullPath((Join-Path (Join-Path $modelRoot $moves[$name]) $name))
    if (-not $source.StartsWith($modelRoot + '\') -or -not $destination.StartsWith($modelRoot + '\')) { throw 'Path outside approved model root' }
    if ((Test-Path -LiteralPath $source) -and (Test-Path -LiteralPath $destination)) { throw "Both paths exist: $name" }
    $existing = if (Test-Path -LiteralPath $source) { $source } else { $destination }
    if (-not (Test-Path -LiteralPath $existing -PathType Container)) { throw "Missing model: $name" }
    $files = @(Get-ChildItem -LiteralPath $existing -Recurse -File | ForEach-Object {
        [pscustomobject]@{ path = $_.FullName.Substring($existing.Length); bytes = $_.Length; modified = $_.LastWriteTimeUtc.Ticks }
    })
    $records += [pscustomobject]@{ source = $source; destination = $destination; files = $files }
}
$records | Select-Object source,destination,@{n='files';e={$_.files.Count}} | Format-Table -AutoSize
if (-not $Apply) { return }
$receiptPath = Join-Path $repoRoot 'config\local\model-move-receipt.json'
if (-not (Test-Path -LiteralPath $receiptPath)) {
    [IO.File]::WriteAllText($receiptPath, ($records | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
}
foreach ($record in $records) {
    if (Test-Path -LiteralPath $record.source) {
        [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($record.destination)) | Out-Null
        # Same-volume directory rename. No copy, backup, deletion, or process stop.
        Move-Item -LiteralPath $record.source -Destination $record.destination
    }
    foreach ($entry in $record.files) {
        $file = Get-Item -LiteralPath ($record.destination + $entry.path)
        if ($file.Length -ne $entry.bytes -or $file.LastWriteTimeUtc.Ticks -ne $entry.modified) { throw "Moved file mismatch: $($file.FullName)" }
    }
}
$references = @(
    (Join-Path $repoRoot 'config\local\ocr.json'),
    (Join-Path $repoRoot 'scripts\validation\hunyuan_dflash.py'),
    (Join-Path $repoRoot 'docs\ocr-local-research.md'),
    (Join-Path $repoRoot 'services\ocr\README.md'),
    'K:\Projects\Xolotl-runtime\windows-4090.json',
    'K:\Projects\LLM_Models\minicpm-ocr-manifest.json'
)
foreach ($path in $references) {
    if (-not (Test-Path -LiteralPath $path)) { continue }
    $content = [IO.File]::ReadAllText($path)
    $updated = $content
    foreach ($name in $moves.Keys) {
        foreach ($separator in @('\', '\\', '/')) {
            $old = 'K:' + $separator + 'Projects' + $separator + 'LLM_Models' + $separator + $name
            $new = 'K:' + $separator + 'Projects' + $separator + 'LLM_Models' + $separator + $moves[$name] + $separator + $name
            $updated = $updated.Replace($old, $new)
        }
    }
    if ($updated -ne $content) {
        [IO.File]::WriteAllText($path, $updated, [Text.UTF8Encoding]::new($false))
        Write-Output "Updated references: $path"
    }
}
Write-Output 'Same-volume moves verified by file count, size and last-write time; receipt saved. No model copied or backed up.'
