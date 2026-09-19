<#
.SYNOPSIS
  Check an overlay (<WorkDir>\overlay) against the Unsloth zip it will be
  installed over.

.DESCRIPTION
  1. Symbol closure (the hard gate): every name an overlay binary imports from
     ggml.dll, ggml-base.dll, ggml-cpu.dll or ggml-rpc.dll must be among that
     DLL's exports in the base zip, nothing may import by ordinal from them,
     and nothing may import ggml-hip.dll. Every other DLL an overlay binary
     imports must be in the overlay, in the base zip, or a Windows / VC++
     runtime DLL: a build that picked up some other library fails here.
  2. File set: the overlay replaces every llama-level file of the base zip
     (except llama-cvector-generator.exe, which stays Unsloth's) and adds none.
  3. With -Smoke: base and overlay assembled into one folder, then
     `llama-server --version` (loads every DLL, prints the build and commit)
     and the DiffusionGemma runner with no arguments (loads its DLLs, prints
     usage, exits). Neither loads a model. --version does initialise the HIP
     runtime, which enumerates the GPUs; nothing is allocated on them.

  Writes <WorkDir>\ggml-imports.json (what the overlay needs from each ggml
  DLL) for the descriptor.

.EXAMPLE
  .\gate.ps1 -WorkDir $env:TEMP\fidim-dgo -Smoke
#>
param(
    [Parameter(Mandatory = $true)] [string]$WorkDir,
    # Default: the zip fetch-base.ps1 downloaded (base.json).
    [string]$BaseZip = '',
    [switch]$Smoke,
    # Accept overlay binaries the base zip does not have (reported either way).
    [switch]$AllowExtra
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$WorkDir = (Resolve-Path -LiteralPath $WorkDir).Path
$base = Read-BaseInfo $WorkDir
if (-not $BaseZip) { $BaseZip = [string]$base.base_zip }
if (-not $BaseZip -or -not (Test-Path -LiteralPath $BaseZip)) { throw "no base zip: run fetch-base.ps1 with -Gfx, or pass -BaseZip" }
# --version must print the release's upstream build and a prefix of its source commit.
$ExpectBuild = [regex]::Match([string]$base.upstream_tag, '\d+').Value
$ExpectCommit = [string]$base.source_commit
$OverlayDir = Join-Path $WorkDir 'overlay'
$gate = Join-Path $WorkDir 'gate'
$baseTop = Join-Path $gate 'base'
if (Test-Path -LiteralPath $gate) { Remove-Item -Recurse -Force -LiteralPath $gate }
Write-Host "== gate: $(Split-Path -Leaf $BaseZip)"
Expand-ZipTopLevel $BaseZip $baseTop
if (-not (Get-Command dumpbin.exe -ErrorAction SilentlyContinue)) { Import-VsDevEnv | Out-Null }

$problems = New-Object System.Collections.Generic.List[string]

# ---- 1. symbol closure
function Get-Exports([string]$Dll) {
    $set = New-Object 'System.Collections.Generic.HashSet[string]'
    $inTable = $false
    foreach ($l in (dumpbin.exe /nologo /exports $Dll)) {
        if ($l -match '^\s+ordinal\s+hint\s+RVA\s+name') { $inTable = $true; continue }
        if (-not $inTable) { continue }
        if ($l -match '^\s+Summary') { break }
        if ($l -match '^\s+\d+\s+[0-9A-F]+\s+(?:[0-9A-F]{8}\s+)?(\S+)') { [void]$set.Add($Matches[1]) }
    }
    return , $set
}
# dll name (lower case) -> names imported by name; ordinals as "#<n>".
function Get-Imports([string]$File) {
    $map = @{}
    $cur = $null
    foreach ($l in (dumpbin.exe /nologo /imports $File)) {
        if ($l -match '^\s+Summary') { break }
        if ($l -match '^    (\S+\.(?:dll|exe))$') {
            $cur = $Matches[1].ToLowerInvariant()
            if (-not $map.ContainsKey($cur)) { $map[$cur] = New-Object System.Collections.Generic.List[string] }
            continue
        }
        if (-not $cur) { continue }
        if ($l -match '^\s+Ordinal\s+(\d+)$') { $map[$cur].Add('#' + $Matches[1]); continue }
        if ($l -match '^\s+[0-9A-F]+\s+(\S+)$') { $map[$cur].Add($Matches[1]) }
    }
    return $map
}

$ggml = 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-rpc.dll'
$exports = @{}
foreach ($d in $ggml) {
    $p = Join-Path $baseTop $d
    if (-not (Test-Path -LiteralPath $p)) { throw "the base zip has no $d" }
    $exports[$d] = Get-Exports $p
    Write-Host ("  {0,-14} exports {1}" -f $d, $exports[$d].Count)
}
$needed = @{}
foreach ($d in $ggml) { $needed[$d] = New-Object 'System.Collections.Generic.SortedSet[string]' }
$overlayNames = @(Get-ChildItem -LiteralPath $OverlayDir -File | ForEach-Object { $_.Name })
$baseNames = @(Get-ChildItem -LiteralPath $baseTop -File | ForEach-Object { $_.Name })
$system32 = Join-Path $env:SystemRoot 'System32'
foreach ($f in Get-ChildItem -LiteralPath $OverlayDir -File) {
    $imports = Get-Imports $f.FullName
    foreach ($dll in $imports.Keys) {
        if ($dll -eq 'ggml-hip.dll') { $problems.Add("$($f.Name) imports ggml-hip.dll"); continue }
        if ($ggml -contains $dll) {
            foreach ($name in $imports[$dll]) {
                if ($name.StartsWith('#')) { $problems.Add("$($f.Name) imports $dll by ordinal $name"); continue }
                if (-not $exports[$dll].Contains($name)) { $problems.Add("$($f.Name) imports $name from $dll, which does not export it") }
                [void]$needed[$dll].Add($name)
            }
            continue
        }
        $known = ($overlayNames -contains $dll) -or ($baseNames -contains $dll) -or $dll.StartsWith('api-ms-win-') -or
                 $dll.StartsWith('ext-ms-') -or (Test-Path -LiteralPath (Join-Path $system32 $dll)) -or
                 ($dll -match '^(msvcp140|vcruntime140|concrt140)(_\w+)?\.dll$')
        if (-not $known) { $problems.Add("$($f.Name) imports $dll, which is neither in the overlay, the base zip nor Windows") }
    }
}
$importsOut = [ordered]@{}
foreach ($d in $ggml) { if ($needed[$d].Count) { $importsOut[$d] = @($needed[$d]) } }
Write-Utf8NoBom (Join-Path $WorkDir 'ggml-imports.json') (ConvertTo-PrettyJson $importsOut)
foreach ($d in $importsOut.Keys) { Write-Host ("  overlay needs {0,4} names from {1}" -f $importsOut[$d].Count, $d) }

# ---- 2. file set
$baseLlama = @($baseNames | Where-Object { (Test-OverlayBinaryName $_) -and ($KeepBaseFiles -notcontains $_) })
$missing = @($baseLlama | Where-Object { $overlayNames -notcontains $_ })
$extra = @($overlayNames | Where-Object { $baseLlama -notcontains $_ })
$foreign = @($overlayNames | Where-Object { -not (Test-OverlayBinaryName $_) -or ($KeepBaseFiles -contains $_) })
foreach ($m in $missing) { $problems.Add("the base zip has $m but the overlay does not: it would run against the overlay's llama.dll") }
foreach ($x in $foreign) { $problems.Add("$x is not an overlay file (ggml, the ROCm runtime and $($KeepBaseFiles -join ', ') stay Unsloth's)") }
if ($extra.Count) {
    $msg = "the overlay adds files the base zip does not have: $($extra -join ', ')"
    if ($AllowExtra) { Write-Host "  note: $msg" } else { $problems.Add($msg) }
}
Write-Host "  file set: $($overlayNames.Count) overlay files replace $($baseLlama.Count) of the base's"

# ---- 3. smoke
if ($Smoke) {
    $asm = Join-Path $gate 'assembled'
    New-Item -ItemType Directory -Force $asm | Out-Null
    Copy-Item -Path (Join-Path $baseTop '*') -Destination $asm
    Copy-Item -Path (Join-Path $OverlayDir '*') -Destination $asm -Force
    # Run the way Llama FIDIM runs a bundled build: nothing else on PATH.
    function Invoke-Probe([string]$Exe, [string]$Arguments) {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $Exe
        $psi.Arguments = $Arguments
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.WorkingDirectory = $asm
        $psi.EnvironmentVariables['PATH'] = "$system32;$env:SystemRoot"
        $p = [System.Diagnostics.Process]::Start($psi)
        $err = $p.StandardError.ReadToEndAsync()
        $out = $p.StandardOutput.ReadToEnd()
        if (-not $p.WaitForExit(120000)) { $p.Kill(); throw "$Exe $Arguments did not exit within 120 s" }
        return @{ code = $p.ExitCode; text = ($out + $err.Result) }
    }
    $r = Invoke-Probe (Join-Path $asm 'llama-server.exe') '--version'
    Write-Host "  llama-server --version (exit $($r.code)):"
    $r.text.Trim() -split "`n" | ForEach-Object { Write-Host "    $($_.TrimEnd())" }
    $m = [regex]::Match($r.text, 'build\s+(\d+),\s*commit\s+([0-9a-fA-F]+)')
    if ($r.code -ne 0 -or -not $m.Success) { $problems.Add("llama-server --version failed (exit $($r.code))") }
    else {
        if ($ExpectBuild -and $m.Groups[1].Value -ne $ExpectBuild) { $problems.Add("--version reports build $($m.Groups[1].Value), expected $ExpectBuild") }
        if ($ExpectCommit -and -not $ExpectCommit.StartsWith($m.Groups[2].Value.ToLowerInvariant())) {
            $problems.Add("--version reports commit $($m.Groups[2].Value), not a prefix of $ExpectCommit (was the source built inside a git checkout?)")
        }
        if ($r.text -notmatch 'Llama FIDIM') { $problems.Add('--version does not carry the overlay fingerprint') }
    }
    $r = Invoke-Probe (Join-Path $asm 'llama-diffusion-gemma-visual-server.exe') ''
    Write-Host "  llama-diffusion-gemma-visual-server with no arguments (exit $($r.code)): $($r.text.Trim())"
    if ($r.code -ne 1 -or $r.text -notmatch 'usage:') { $problems.Add("the runner did not load and print its usage (exit $($r.code))") }
}

if ($problems.Count) {
    Write-Host "== gate FAILED:" -ForegroundColor Red
    $problems | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "== gate passed"
