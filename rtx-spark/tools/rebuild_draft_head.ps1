# Rebuilds the deployed Flash-Next draft head mtp-Qwen3.8-Flash-Next-Q8_0-output-Q4_K.gguf
# from the pinned public ggml-org source and checks it against the recorded hash.
# Recipe and per-tensor hashes: results/rebaseline-20261020/head-q4k/quantization.json
param(
    [string]$Runtime = 'C:\llama.cpp',
    [string]$Out = "$env:LOCALAPPDATA\LocalModelDeployer\custom-models\mtp-Qwen3.8-Flash-Next-Q8_0-output-Q4_K.gguf"
)
$ErrorActionPreference = 'Stop'
$srcUrl = 'https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF/resolve/92796c044d1e70d66a373598709092f2989f4155/mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
$srcSha = '6be8a94e2d94107b7f38fdc957c3460ffeb6ff6302edef92c50738682f9dc0e0'
$outSha = 'ee2207b567710b4f73035b83ed214ba470ee17a94355c678e17fd3c58e7ccd84'

if ((Test-Path $Out) -and (Get-FileHash $Out).Hash.ToLower() -eq $outSha) { Write-Output "already present and verified: $Out"; return }
$work = Join-Path $env:TEMP ('draft-head-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory $work | Out-Null
try {
    $src = Join-Path $work 'mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
    curl.exe -sSfL --retry 5 -o $src $srcUrl
    if ($LASTEXITCODE) { throw 'download failed' }
    if ((Get-FileHash $src).Hash.ToLower() -ne $srcSha) { throw 'source hash mismatch' }
    $tmp = Join-Path $work 'out.gguf'
    & "$Runtime\llama-quantize.exe" --allow-requantize --output-tensor-type q4_k --pure --max-buffer-size 512 $src $tmp Q8_0 8 *> (Join-Path $work 'quantize.log')
    if ($LASTEXITCODE) { throw "llama-quantize failed with $LASTEXITCODE" }
    if ((Get-FileHash $tmp).Hash.ToLower() -ne $outSha) { throw 'rebuilt file hash mismatch' }
    New-Item -ItemType Directory -Force (Split-Path $Out) | Out-Null
    Move-Item $tmp $Out -Force
    Write-Output "rebuilt and verified: $Out"
} finally {
    Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
}
