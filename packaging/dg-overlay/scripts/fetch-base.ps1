<#
.SYNOPSIS
  Fetch the exact source of an Unsloth llama.cpp release, check it, and unpack
  it. Optionally fetch one of the release's Windows ROCm zips too.

.DESCRIPTION
  The release's git tag is not its source: Unsloth's CI merges the listed pull
  requests at build time and never pushes the result. The source the release
  was built from is its llama.cpp-source-commit-<sha>.tar.gz asset, which is
  checked against GitHub's digest and against the `exact-source` entry of the
  release's llama-prebuilt-sha256.json.

  Writes <WorkDir>\base.json (tag, source commit, ggml tree, the sha256 of
  every Windows ROCm zip) for package.ps1, and unpacks the source into
  <WorkDir>\src. Keep WorkDir short and outside any git checkout: the source
  nests files ~190 characters deep, and the build stamps its version from
  the first git repository it finds above the source.

.EXAMPLE
  .\fetch-base.ps1 -Tag b11030-mix-5ff778e -WorkDir $env:TEMP\fidim-dgo -Gfx gfx120X
#>
param(
    # An Unsloth release tag, or `latest`.
    [Parameter(Mandatory = $true)] [string]$Tag,
    [Parameter(Mandatory = $true)] [string]$WorkDir,
    # Also fetch app-<tag>-windows-x64-rocm-<Gfx>.zip (the gate needs one).
    [string]$Gfx = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$rel = Get-GitHubRelease $BaseRepo $Tag
$Tag = $rel.tag_name
Assert-UnslothTag $Tag
Write-Host "== base: $BaseRepo $Tag"
New-Item -ItemType Directory -Force $WorkDir | Out-Null
$WorkDir = (Resolve-Path -LiteralPath $WorkDir).Path
$dl = Join-Path $WorkDir 'dl'

$manifestPath = Save-ReleaseAsset $rel 'llama-prebuilt-manifest.json' $dl
$sumsPath = Save-ReleaseAsset $rel 'llama-prebuilt-sha256.json' $dl
$srcAsset = @($rel.assets | Where-Object { $_.name -match '^llama\.cpp-source-commit-[0-9a-f]{40}\.tar\.gz$' })
if ($srcAsset.Count -ne 1) {
    throw "release $Tag has $($srcAsset.Count) llama.cpp-source-commit-*.tar.gz assets, expected one (releases before b9585 ship no source)"
}
$srcName = $srcAsset[0].name
$tarball = Save-ReleaseAsset $rel $srcName $dl

$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$sums = Get-Content -Raw -LiteralPath $sumsPath | ConvertFrom-Json
$commit = [string]$manifest.source_commit
if ($commit -notmatch '^[0-9a-f]{40}$') { throw "llama-prebuilt-manifest.json has no full source_commit" }
if ($srcName -ne "llama.cpp-source-commit-$commit.tar.gz") { throw "$srcName does not name the manifest's source commit $commit" }
if ($sums.source_commit -ne $commit) { throw "llama-prebuilt-sha256.json names commit $($sums.source_commit), the manifest $commit" }
$entry = $sums.artifacts.PSObject.Properties[$srcName]
if (-not $entry -or $entry.Value.kind -ne 'exact-source') { throw "llama-prebuilt-sha256.json lists no exact-source entry for $srcName" }
$srcSha = Get-Sha256 $tarball
if ($entry.Value.sha256 -ne $srcSha) { throw "$srcName is $srcSha, llama-prebuilt-sha256.json says $($entry.Value.sha256)" }
Write-Host "  source $srcName sha256 $srcSha (GitHub digest and exact-source entry agree)"

# Every Windows ROCm zip of the release, per GPU target. One overlay serves
# them all: they share the source commit and differ only in HIP kernels.
$zips = [ordered]@{}
foreach ($p in $sums.artifacts.PSObject.Properties) {
    if ($p.Name -match ('^app-' + [regex]::Escape($Tag) + '-windows-x64-rocm-(.+)\.zip$')) {
        if ($p.Value.source_commit -ne $commit) { throw "$($p.Name) was built from $($p.Value.source_commit), not $commit" }
        $zips[$Matches[1]] = [string]$p.Value.sha256
    }
}
if ($zips.Count -eq 0) { throw "llama-prebuilt-sha256.json lists no Windows ROCm zips for $Tag" }

$baseZip = $null
if ($Gfx) {
    $gfxKey = @($zips.Keys | Where-Object { $_ -eq $Gfx })
    if ($gfxKey.Count -ne 1) { throw "release $Tag has no Windows ROCm zip for $Gfx; it has $($zips.Keys -join ', ')" }
    $zipName = "app-$Tag-windows-x64-rocm-$($gfxKey[0]).zip"
    $baseZip = Save-ReleaseAsset $rel $zipName $dl
    if ((Get-Sha256 $baseZip) -ne $zips[$gfxKey[0]]) { throw "$zipName does not match llama-prebuilt-sha256.json" }
}

# Unpack unless this exact source is already there (then the build can be
# incremental). apply-patch.ps1 records what it applied the same way.
$src = Join-Path $WorkDir 'src'
$stamp = Join-Path $WorkDir 'src.stamp'
$have = if (Test-Path -LiteralPath $stamp) { (Get-Content -Raw -LiteralPath $stamp).Trim() } else { '' }
if ($have -ne $srcSha -or -not (Test-Path -LiteralPath (Join-Path $src 'CMakeLists.txt'))) {
    if (Test-Path -LiteralPath $src) { Remove-Item -Recurse -Force -LiteralPath $src }
    Remove-Item -Force -ErrorAction SilentlyContinue -LiteralPath (Join-Path $WorkDir 'patch.stamp')
    New-Item -ItemType Directory -Force $src | Out-Null
    Write-Host "  unpacking into $src"
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    Invoke-Native "unpack $srcName" { & $tar -xzf $tarball -C $src --strip-components=1 }
    Write-Utf8NoBom $stamp $srcSha
} else {
    Write-Host "  source already unpacked in $src"
}

$info = [ordered]@{
    repo          = $BaseRepo
    release_tag   = $Tag
    source_commit = $commit
    source_asset  = $srcName
    source_sha256 = $srcSha
    upstream_tag  = [string]$manifest.upstream_tag
    ggml_tree     = [string]$manifest.ggml_tree
    ggml_version  = [string]$manifest.ggml_version
    zips          = $zips
    base_zip      = $baseZip
    base_gfx      = $Gfx
}
Write-Utf8NoBom (Join-Path $WorkDir 'base.json') (ConvertTo-PrettyJson $info)
Write-Host "  wrote $(Join-Path $WorkDir 'base.json')"
