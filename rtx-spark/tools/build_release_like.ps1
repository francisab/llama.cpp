# Release-like native Windows ARM64 build: Clang for CPU side and tools, MSVC + CUDA for ggml-cuda.dll only.
# Mirrors the official windows-cpu (arm64) and windows-cuda (13.4, arm64) jobs, merged into $Out\bin.
param(
    [Parameter(Mandatory)][string]$Src,
    [Parameter(Mandatory)][string]$Out,
    [string]$CudaRoot = 'C:\cuda\v13.4',
    [string]$Tools = 'C:\buildtools',
    [string]$CudaArch = '121a-real',
    # LLVM root with bin\clang.exe; its major version must match GGML_OPENMP_LLVM_VERSION (official: clang+llvm-*-aarch64-pc-windows-msvc release tarball)
    [string]$LlvmRoot = ''
)
$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
if (-not $LlvmRoot) { $LlvmRoot = (Get-ChildItem $Tools -Directory -Filter 'clang+llvm-*-aarch64-pc-windows-msvc' | Sort-Object Name | Select-Object -Last 1).FullName }
if (-not $LlvmRoot) { $LlvmRoot = "$vs\VC\Tools\Llvm\ARM64" }
$llvm = "$LlvmRoot\bin"
if (-not (Test-Path "$llvm\clang.exe")) { throw "ARM64 clang not found in $llvm" }
$cmake = Join-Path (Get-ChildItem $Tools -Directory -Filter 'cmake-*-windows-arm64' | Select-Object -First 1).FullName 'bin\cmake.exe'
$cpu = Join-Path $Out 'cpu'

# GGML_OPENMP_FETCH extracts libomp from the LLVM installer with 7-Zip; seed its cache from the matching LLVM toolchain instead
$ompVer = [regex]::Match((Get-Content "$Src\ggml\src\CMakeLists.txt" -Raw), 'GGML_OPENMP_LLVM_VERSION "([0-9.]+)"').Groups[1].Value
$clangVer = [regex]::Match((& "$llvm\clang.exe" --version | Select-Object -First 1), '([0-9]+)\.[0-9]+\.[0-9]+').Groups[1].Value
if ($ompVer.Split('.')[0] -ne $clangVer) { throw "clang $clangVer does not match LLVM OpenMP $ompVer" }
$omp = "$cpu\_deps\llvm-openmp-$ompVer-arm64"
if (Test-Path "$LlvmRoot\lib\libomp.lib") {
    New-Item -ItemType Directory -Force "$omp\lib", "$omp\bin", "$omp\include" | Out-Null
    Copy-Item "$LlvmRoot\lib\libomp.lib" "$omp\lib" -Force
    Copy-Item "$LlvmRoot\bin\libomp.dll" "$omp\bin" -Force
    Copy-Item "$LlvmRoot\lib\clang\$clangVer\include\omp.h" "$omp\include" -Force
}

$cfg = @(
    # release CMAKE_ARGS without LLAMA_CONNECT (it downloads a prebuilt binary)
    "-S `"$Src`"", "-B `"$cpu`"", '-G Ninja', '-DCMAKE_BUILD_TYPE=Release',
    "-DCMAKE_TOOLCHAIN_FILE=`"$Src\cmake\arm64-windows-llvm.cmake`"",
    '-DLLAMA_BUILD_BORINGSSL=ON', '-DGGML_NATIVE=OFF', '-DGGML_BACKEND_DL=ON',
    '-DGGML_CPU_ALL_VARIANTS=OFF', '-DGGML_OPENMP=ON', '-DGGML_OPENMP_FETCH=ON', '-DGGML_RPC=ON',
    '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_TOOLS=ON', '-DLLAMA_BUILD_SERVER=ON'
)
$bat = @"
@echo off
call "$vs\VC\Auxiliary\Build\vcvarsall.bat" arm64 >nul || exit /b 1
set "PATH=$llvm;$Tools\ninja;%PATH%"
"$cmake" $($cfg -join ' ') || exit /b 2
"$cmake" --build "$cpu" || exit /b 3
"@
$tmp = Join-Path $env:TEMP ("build-" + [guid]::NewGuid().ToString('N') + '.cmd')
Set-Content -Path $tmp -Value $bat -Encoding ascii
try {
    & $env:ComSpec /c $tmp
    if ($LASTEXITCODE) { throw "CPU build failed with exit code $LASTEXITCODE" }
} finally {
    Remove-Item $tmp -ErrorAction SilentlyContinue
}

$cuda = Join-Path $Out 'cuda'
& (Join-Path $PSScriptRoot 'build_native_cuda.ps1') -Src $Src -Out $cuda -CudaRoot $CudaRoot -Tools $Tools -CudaArch $CudaArch `
    -Targets ggml-cuda -Extra '-DGGML_BACKEND_DL=ON', '-DGGML_NATIVE=OFF', '-DGGML_CPU=OFF'

$bin = Join-Path $Out 'bin'
New-Item -ItemType Directory -Force $bin | Out-Null
Copy-Item "$cpu\bin\*" $bin -Recurse -Force
Copy-Item "$cuda\bin\ggml-cuda.dll", "$cuda\bin\cudart64_13.dll", "$cuda\bin\cublas64_13.dll", "$cuda\bin\cublasLt64_13.dll" $bin -Force
Write-Host "built: $bin"
