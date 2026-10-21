# Native Windows ARM64 + CUDA build of llama.cpp (no emulation, no cross toolchain).
# Needs: VS 2022 Build Tools with VC ARM64 tools, native ARM64 CUDA redist in $CudaRoot, ARM64 CMake + Ninja in $Tools.
param(
    [Parameter(Mandatory)][string]$Src,
    [Parameter(Mandatory)][string]$Out,
    [string]$CudaRoot = 'C:\cuda\v13.4',
    [string]$Tools = 'C:\buildtools',
    [string]$CudaArch = '121a-real',
    [string]$Cccl = 'v3.4.3',
    [string[]]$Targets = @('llama-server', 'llama-bench', 'llama-cli'),
    [string[]]$Extra = @()
)
$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
if (-not $vs) { throw 'VS with VC ARM64 tools not found' }
$cmake = Join-Path (Get-ChildItem $Tools -Directory -Filter 'cmake-*-windows-arm64' | Select-Object -First 1).FullName 'bin\cmake.exe'
$ninja = Join-Path $Tools 'ninja'
$lib = "$CudaRoot\lib\arm64"

$cfg = @(
    "-S `"$Src`"", "-B `"$Out`"", '-G Ninja', '-DCMAKE_BUILD_TYPE=Release',
    '-DGGML_NATIVE=ON', '-DGGML_CUDA=ON', "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch",
    "-DGGML_CUDA_CCCL_VERSION=$Cccl", "-DCUDAToolkit_ROOT=`"$CudaRoot`"",
    "-DCUDA_CUDART=`"$lib\cudart.lib`"", "-DCUDA_cudart_LIBRARY=`"$lib\cudart.lib`"",
    "-DCUDA_cublas_LIBRARY=`"$lib\cublas.lib`"", "-DCUDA_cublasLt_LIBRARY=`"$lib\cublasLt.lib`"",
    "-DCUDA_cuda_driver_LIBRARY=`"$lib\cuda.lib`"",
    '-DLLAMA_BUILD_TESTS=ON', '-DLLAMA_OPENSSL=OFF'
) + $Extra

$bat = @"
@echo off
call "$vs\VC\Auxiliary\Build\vcvarsall.bat" arm64 >nul || exit /b 1
set "PATH=$CudaRoot\bin;$ninja;%PATH%"
set "CUDA_PATH=$CudaRoot"
"$cmake" $($cfg -join ' ') || exit /b 2
"$cmake" --build "$Out" --target $($Targets -join ' ') || exit /b 3
"@
$tmp = Join-Path $env:TEMP ("build-" + [guid]::NewGuid().ToString('N') + '.cmd')
Set-Content -Path $tmp -Value $bat -Encoding ascii
try {
    & $env:ComSpec /c $tmp
    if ($LASTEXITCODE) { throw "build failed with exit code $LASTEXITCODE" }
} finally {
    Remove-Item $tmp -ErrorAction SilentlyContinue
}

# runtime DLLs next to the binaries
$bin = Join-Path $Out 'bin'
Copy-Item "$CudaRoot\bin\arm64\cudart64_13.dll", "$CudaRoot\bin\arm64\cublas64_13.dll", "$CudaRoot\bin\arm64\cublasLt64_13.dll" $bin -Force
Write-Host "built: $bin"
