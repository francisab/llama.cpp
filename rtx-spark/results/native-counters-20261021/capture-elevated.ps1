param(
    [Parameter(Mandatory)][string]$Ncu,
    [Parameter(Mandatory)][string]$TestExe,
    [Parameter(Mandatory)][string]$Out
)

$ErrorActionPreference = 'Stop'
$protocol = Get-Content -LiteralPath "$PSScriptRoot\elevated-protocol.json" -Raw | ConvertFrom-Json
$Ncu = (Resolve-Path -LiteralPath $Ncu).Path
$TestExe = (Resolve-Path -LiteralPath $TestExe).Path
$Out = [IO.Path]::GetFullPath($Out)
New-Item -ItemType Directory -Path $Out | Out-Null
$started = Get-Date
$watch = [Diagnostics.Stopwatch]::StartNew()
$process = $null
$result = @{
    completed = $false
    cleanup_verified = $false
    driver_or_counter_access_policy_changed = $false
    deployment_changed = $false
}

function Get-GpuMemory {
    $value = & nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read GPU memory' }
    return [int]$value
}

try {
    $principal = [Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
    $result.elevated_admin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $result.elevated_admin) { throw 'This capture requires explicit user-approved elevation' }
    if ((Get-AuthenticodeSignature -LiteralPath $Ncu).Status -ne 'Valid') { throw 'Profiler signature is not valid' }
    if ((Get-FileHash -LiteralPath $Ncu).Hash -ne $protocol.runtime.profiler_sha256) { throw 'Profiler hash changed' }
    if ((Get-FileHash -LiteralPath $TestExe).Hash -ne $protocol.runtime.test_exe_sha256) { throw 'Test executable hash changed' }
    $cuda = Join-Path (Split-Path $TestExe) 'ggml-cuda.dll'
    if ((Get-FileHash -LiteralPath $cuda).Hash -ne $protocol.runtime.control_cuda_sha256) { throw 'Control CUDA hash changed' }
    if ((Get-GpuMemory) -ge 8000) { throw 'GPU is not idle' }
    $existing = Get-CimInstance Win32_Process -Filter "Name='test-backend-ops.exe'" |
        Where-Object { $_.ExecutablePath -eq $TestExe }
    if ($existing) { throw 'The isolated test executable is already running' }

    $fixture = (Resolve-Path -LiteralPath "$PSScriptRoot\..\q8-fourcol-warps-20261021\test-ops.txt").Path
    $arguments = @(
        '--config-file', 'off', '--target-processes', 'application-only',
        '--clock-control', 'none', '--cache-control', 'none', '--replay-mode', 'kernel',
        '--kernel-name-base', 'function', '--kernel-name', 'mul_mat_vec_q',
        '--launch-count', '1', '--kill', 'no'
    )
    foreach ($section in $protocol.collection.sections) { $arguments += @('--section', $section) }
    $arguments += @('--export', "$Out\q8", $TestExe, 'test', '-b', 'CUDA0',
                    '--test-file', $fixture, '-p', 'name=q8_k320_m10240_n4,')
    $arguments | ConvertTo-Json | Set-Content -LiteralPath "$Out\command.json" -Encoding utf8
    $quoted = ($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $process = Start-Process -FilePath $Ncu -ArgumentList $quoted -WorkingDirectory (Split-Path $TestExe) `
        -NoNewWindow -RedirectStandardOutput "$Out\stdout.log" -RedirectStandardError "$Out\stderr.log" -PassThru
    $result.profiler_pid = $process.Id
    if (-not $process.WaitForExit(1000 * $protocol.collection.timeout_seconds)) { throw 'Counter capture timed out' }
    $process.WaitForExit()
    $result.exit_code = $process.ExitCode
    if ($process.ExitCode -ne 0) { throw "Profiler exited with code $($process.ExitCode)" }
    if (-not (Test-Path -LiteralPath "$Out\q8.ncu-rep" -PathType Leaf)) { throw 'Profiler did not produce a report' }
    $result.completed = $true
} catch {
    $result.error = $_.Exception.Message
} finally {
    try {
        if ($null -ne $process) {
            $remaining = Get-CimInstance Win32_Process -Filter "Name='test-backend-ops.exe'" |
                Where-Object { $_.ExecutablePath -eq $TestExe -and $_.CreationDate -ge $started }
            foreach ($target in $remaining) {
                Stop-Process -Id $target.ProcessId -ErrorAction Stop
                $result.forced_target_cleanup = $true
                $result.completed = $false
            }
            if (-not $process.HasExited) {
                Stop-Process -Id $process.Id -ErrorAction Stop
                $result.forced_profiler_cleanup = $true
                $result.completed = $false
            }
        }
        $deadline = (Get-Date).AddSeconds(60)
        while ((Get-GpuMemory) -ge 8000) {
            if ((Get-Date) -ge $deadline) { throw 'GPU memory was not released' }
            Start-Sleep -Seconds 2
        }
        $result.gpu_memory_released_mib = Get-GpuMemory
        $result.cleanup_verified = $true
    } catch {
        $result.cleanup_error = $_.Exception.Message
        $result.completed = $false
    }
    $result.elapsed_seconds = $watch.Elapsed.TotalSeconds
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$Out\result.json" -Encoding utf8
}

if (-not $result.completed) { exit 1 }
