<#
.SYNOPSIS
  Build the patched source (<WorkDir>\src) without HIP and collect the
  overlay: every llama-level binary into <WorkDir>\overlay, their license
  texts into <WorkDir>\licenses.

.DESCRIPTION
  The flags are the ones Unsloth's Windows ROCm build uses, minus HIP. The
  overlay then runs on Unsloth's own ggml DLLs: the boundary between
  llama.dll and ggml is plain C resolved by name, so a llama.dll built
  without HIP binds to their ggml-hip build (gate.ps1 checks every imported
  name against their exports). Needs Visual Studio 2022 with the C++ tools;
  it uses the VS-bundled CMake and Ninja. No ROCm.

  Writes <WorkDir>\build-info.json (toolchain, versions, flags) for the
  descriptor.

.EXAMPLE
  .\build.ps1 -WorkDir $env:TEMP\fidim-dgo
#>
param(
    [Parameter(Mandatory = $true)] [string]$WorkDir,
    # clang: the VS-bundled clang through the source's x64-windows-llvm
    # toolchain file (upstream's Windows CPU recipe, and the compiler family
    # Unsloth uses). msvc: cl.exe. auto: clang when the VS "C++ Clang tools
    # for Windows" component is installed, else msvc.
    [ValidateSet('auto', 'clang', 'msvc')] [string]$Toolchain = 'auto',
    # fetch: BoringSSL at the version the source pins, cloned from
    # boringssl.googlesource.com at configure time, as Unsloth builds it.
    # off: no HTTPS in llama-common (downloads with -hf fail).
    # Any other value: a BoringSSL source folder to build instead of cloning.
    [string]$BoringSsl = 'fetch',
    [int]$Jobs = 0
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$WorkDir = (Resolve-Path -LiteralPath $WorkDir).Path
$src = Join-Path $WorkDir 'src'
$buildDir = Join-Path $WorkDir 'build'
if (-not (Test-Path -LiteralPath (Join-Path $WorkDir 'patch.stamp'))) { throw "the source in $src is not patched: run apply-patch.ps1 first" }
New-Item -ItemType Directory -Force $buildDir | Out-Null
Set-GitCeiling $src

$vs = Import-VsDevEnv
$vsVersion = & (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe') -path $vs -property catalog_productDisplayVersion
Write-Host "== Visual Studio $vsVersion, MSVC toolset $env:VCToolsVersion"
foreach ($tool in 'cmake.exe', 'ninja.exe') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool not on PATH after vcvars64: install the VS 'C++ CMake tools for Windows' component" }
}
$clang = Get-Command clang.exe -ErrorAction SilentlyContinue
if ($Toolchain -eq 'auto') { $Toolchain = if ($clang) { 'clang' } else { 'msvc' } }
if ($Toolchain -eq 'clang' -and -not $clang) {
    throw "clang.exe not on PATH after vcvars64: install the VS 'C++ Clang tools for Windows' component, or pass -Toolchain msvc"
}

# $common goes to CMake as is; the rest differ between what runs here
# ($actual, absolute paths) and what the descriptor records ($recorded).
$common = @('-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DGGML_HIP=OFF', '-DGGML_NATIVE=OFF',
            '-DGGML_OPENMP=OFF', '-DGGML_RPC=ON', '-DGGML_STATIC=OFF', '-DLLAMA_BUILD_TESTS=OFF')
$actual = @(); $recorded = @()
if ($Toolchain -eq 'clang') {
    $actual += '-DCMAKE_TOOLCHAIN_FILE=' + (Join-Path $src 'cmake\x64-windows-llvm.cmake')
    $recorded += '-DCMAKE_TOOLCHAIN_FILE=cmake/x64-windows-llvm.cmake'
} else {
    $actual += '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl'
    $recorded += '-DCMAKE_C_COMPILER=cl', '-DCMAKE_CXX_COMPILER=cl'
}
switch ($BoringSsl) {
    'fetch' { $actual += '-DLLAMA_BUILD_BORINGSSL=ON'; $recorded += '-DLLAMA_BUILD_BORINGSSL=ON' }
    'off' {
        # LLAMA_OPENSSL off too, or CMake could pick up an OpenSSL built for
        # another toolchain (Git for Windows ships one on PATH).
        $actual += '-DLLAMA_BUILD_BORINGSSL=OFF', '-DLLAMA_OPENSSL=OFF'
        $recorded += '-DLLAMA_BUILD_BORINGSSL=OFF', '-DLLAMA_OPENSSL=OFF'
    }
    default {
        $dir = (Resolve-Path -LiteralPath $BoringSsl).Path
        $actual += '-DLLAMA_BUILD_BORINGSSL=ON', "-DFETCHCONTENT_SOURCE_DIR_BORINGSSL=$dir"
        $recorded += '-DLLAMA_BUILD_BORINGSSL=ON', '-DFETCHCONTENT_SOURCE_DIR_BORINGSSL=<local BoringSSL source>'
    }
}
$configure = $common + $actual

Write-Host "== configure ($Toolchain, BoringSSL $BoringSsl)"
Invoke-Native 'cmake configure' { cmake -S $src -B $buildDir @configure }
$j = if ($Jobs -gt 0) { $Jobs } else { [Environment]::ProcessorCount }
Write-Host "== build (-j $j)"
$t0 = Get-Date
Invoke-Native 'cmake build' { cmake --build $buildDir --config Release -j $j }
$minutes = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
Write-Host "  built in $minutes min"

# Compiler identity from CMake's own probe.
function Get-CMakeCompiler([string]$Lang) {
    $f = Get-ChildItem -LiteralPath (Join-Path $buildDir 'CMakeFiles') -Directory |
        ForEach-Object { Join-Path $_.FullName "CMake${Lang}Compiler.cmake" } |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $f) { return $null }
    $t = Get-Content -Raw -LiteralPath $f
    $id = [regex]::Match($t, "set\(CMAKE_${Lang}_COMPILER_ID `"([^`"]*)`"\)").Groups[1].Value
    $ver = [regex]::Match($t, "set\(CMAKE_${Lang}_COMPILER_VERSION `"([^`"]*)`"\)").Groups[1].Value
    "$id $ver".Trim()
}
$boringLicense = Join-Path $buildDir '_deps\boringssl-src\LICENSE'
if ($BoringSsl -ne 'off' -and -not (Test-Path -LiteralPath $boringLicense)) { throw "BoringSSL was built in but $boringLicense is missing" }
$info = [ordered]@{
    runner        = if ($env:GITHUB_ACTIONS -eq 'true') { "github-actions $env:ImageOS $env:ImageVersion".Trim() } else { 'local' }
    toolchain     = $Toolchain
    c_compiler    = Get-CMakeCompiler 'C'
    cxx_compiler  = Get-CMakeCompiler 'CXX'
    msvc_toolset  = $env:VCToolsVersion
    vs_version    = [string]$vsVersion
    cmake         = ((cmake --version) | Select-Object -First 1) -replace '^cmake version\s*', ''
    cmake_flags   = $common + $recorded
    boringssl     = if ($BoringSsl -eq 'fetch' -or $BoringSsl -eq 'off') { $BoringSsl } else { 'local source' }
    build_minutes = $minutes
}
Write-Utf8NoBom (Join-Path $WorkDir 'build-info.json') (ConvertTo-PrettyJson $info)

# Collect: every llama-level binary; ggml and the rest stay Unsloth's.
$overlay = Join-Path $WorkDir 'overlay'
if (Test-Path -LiteralPath $overlay) { Remove-Item -Recurse -Force -LiteralPath $overlay }
New-Item -ItemType Directory -Force $overlay | Out-Null
$taken = @(); $left = @()
foreach ($f in Get-ChildItem -LiteralPath (Join-Path $buildDir 'bin') -File) {
    if ((Test-OverlayBinaryName $f.Name) -and ($KeepBaseFiles -notcontains $f.Name)) {
        Copy-Item -LiteralPath $f.FullName -Destination $overlay
        $taken += $f.Name
    } else {
        $left += $f.Name
    }
}
if ($taken.Count -eq 0) { throw "no llama binaries in $buildDir\bin" }
Write-Host "== collected $($taken.Count) files into $overlay"
Write-Host "  left to the base: $($left -join ', ')"

# License texts: llama.cpp's, those it keeps under licenses\ and beside its
# vendored libraries (vendor\hash\xxhash\LICENSE becomes LICENSE-xxhash),
# and BoringSSL's when it is linked in.
$lic = Join-Path $WorkDir 'licenses'
if (Test-Path -LiteralPath $lic) { Remove-Item -Recurse -Force -LiteralPath $lic }
New-Item -ItemType Directory -Force $lic | Out-Null
Copy-Item -LiteralPath (Join-Path $src 'LICENSE') -Destination (Join-Path $lic 'LICENSE-llama.cpp')
if (Test-Path -LiteralPath (Join-Path $src 'licenses')) {
    Get-ChildItem -LiteralPath (Join-Path $src 'licenses') -File | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $lic }
}
$vendored = Get-ChildItem -LiteralPath (Join-Path $src 'vendor') -Recurse -File |
    Where-Object { $_.Name -like 'LICENSE*' -or $_.Name -like 'COPYING*' }
foreach ($f in $vendored) { Copy-Item -LiteralPath $f.FullName -Destination (Join-Path $lic ('LICENSE-' + $f.Directory.Name + $f.Extension)) }
if ($BoringSsl -ne 'off') { Copy-Item -LiteralPath $boringLicense -Destination (Join-Path $lic 'LICENSE-boringssl') }
Write-Host "  license texts: $((Get-ChildItem -LiteralPath $lic -File | ForEach-Object { $_.Name }) -join ', ')"
