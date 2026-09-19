<#
.SYNOPSIS
  Build an overlay on this machine the way the workflow does, so it can be
  validated before anything is published.

.DESCRIPTION
  fetch-base -> apply-patch -> build -> gate -> package, into OutDir:
    fidim-dg-overlay-<tag>-windows-x64.zip, fidim-overlay.json,
    dgpatch5.diff, SHA256SUMS
  Llama FIDIM installs that folder over the matching Unsloth zip with
    fidim update --channel unsloth --tag <tag> --install --overlay-from <OutDir>
  (see VALIDATION.md). Nothing here is signed or published, and nothing
  loads a model; the gate's smoke test runs `llama-server --version`, which
  initialises the HIP runtime (GPU enumeration only).

  Needs Visual Studio 2022 with the C++ tools and git. No ROCm.
  WorkDir must be short (the source nests ~190 characters deep) and outside
  any git checkout. A rerun on the same tag and patch builds incrementally.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File packaging\dg-overlay\scripts\build-local.ps1
.EXAMPLE
  powershell -File packaging\dg-overlay\scripts\build-local.ps1 -Tag b11030-mix-5ff778e -Toolchain msvc -BoringSsl off
#>
param(
    # An Unsloth release tag, or `latest`.
    [string]$Tag = 'latest',
    [string]$WorkDir = (Join-Path $env:TEMP 'fidim-dgo'),
    # Default: <WorkDir>\out
    [string]$OutDir = '',
    # The Windows ROCm zip the gate checks against (the overlay itself serves every GPU target).
    [string]$Gfx = 'gfx120X',
    [ValidateSet('auto', 'clang', 'msvc')] [string]$Toolchain = 'auto',
    # fetch (as the workflow and Unsloth build), off, or a BoringSSL source folder. See build.ps1.
    [string]$BoringSsl = 'fetch',
    [int]$Jobs = 0,
    # Skip the gate's smoke test (the symbol and file-set checks still run).
    [switch]$NoSmoke
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

if ($WorkDir.Length -gt 80) { throw "WorkDir is $($WorkDir.Length) characters; keep it under 80 (the source nests ~190 deep)" }
New-Item -ItemType Directory -Force $WorkDir | Out-Null
$WorkDir = (Resolve-Path -LiteralPath $WorkDir).Path
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
$inRepo = & git -C $WorkDir rev-parse --show-toplevel 2>$null
$ErrorActionPreference = $eap
if ($inRepo) { throw "$WorkDir is inside the git checkout $inRepo; the build would stamp that repository's commit" }
if (-not $OutDir) { $OutDir = Join-Path $WorkDir 'out' }

$t0 = Get-Date
& (Join-Path $PSScriptRoot 'fetch-base.ps1') -Tag $Tag -WorkDir $WorkDir -Gfx $Gfx
& (Join-Path $PSScriptRoot 'apply-patch.ps1') -WorkDir $WorkDir
& (Join-Path $PSScriptRoot 'build.ps1') -WorkDir $WorkDir -Toolchain $Toolchain -BoringSsl $BoringSsl -Jobs $Jobs
& (Join-Path $PSScriptRoot 'gate.ps1') -WorkDir $WorkDir -Smoke:(-not $NoSmoke)
if ($LASTEXITCODE -ne 0) { throw "the gate failed" }
& (Join-Path $PSScriptRoot 'package.ps1') -WorkDir $WorkDir -OutDir $OutDir

$base = Read-BaseInfo $WorkDir
Write-Host ""
Write-Host "== done in $([math]::Round(((Get-Date) - $t0).TotalMinutes, 1)) min: $OutDir"
Write-Host "Install it over Unsloth's $Gfx zip with Llama FIDIM (VALIDATION.md):"
Write-Host "  fidim update --channel unsloth --tag $($base.release_tag) --gfx $Gfx --install --overlay-from `"$OutDir`" --base-zip `"$($base.base_zip)`""
