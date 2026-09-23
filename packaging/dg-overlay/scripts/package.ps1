<#
.SYNOPSIS
  Package an overlay as release assets in OutDir: the zip, its descriptor
  (fidim-overlay.json), the patch and SHA256SUMS.

.DESCRIPTION
  The zip holds the overlay binaries at its top level (they go into the
  build's bin folder) and the license texts under licenses/. The descriptor
  pins the overlay to its base (release tag, source commit, the sha256 of
  every Windows ROCm zip of that release) and lists the sha256 and size of
  every file in the zip; Llama FIDIM refuses an install where any of them
  disagree. DESCRIPTOR.md documents the format.

  Reads <WorkDir>\overlay, licenses, base.json, build-info.json and
  ggml-imports.json, and refuses unless gate.ps1 passed on these overlay
  files (gate.pass). Run it after signing: the descriptor records the files
  as shipped.

.EXAMPLE
  .\package.ps1 -WorkDir $env:TEMP\fidim-dgo -OutDir $env:TEMP\fidim-dgo\out
#>
param(
    [Parameter(Mandatory = $true)] [string]$WorkDir,
    [Parameter(Mandatory = $true)] [string]$OutDir,
    # Default: patches\<PatchName>.diff next to this folder.
    [string]$Patch = '',
    # The repository the release goes to (the workflow passes its own).
    [string]$Repo = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$root = Split-Path -Parent $PSScriptRoot
if (-not $Patch) { $Patch = Join-Path $root "patches\$PatchName.diff" }
if (-not $Repo) { $Repo = $OverlayRepo }

$WorkDir = (Resolve-Path -LiteralPath $WorkDir).Path
$base = Read-BaseInfo $WorkDir
$Tag = [string]$base.release_tag
Assert-UnslothTag $Tag
Assert-GatePassed $WorkDir
function Read-Json([string]$Name, [string]$Producer) {
    $p = Join-Path $WorkDir $Name
    if (-not (Test-Path -LiteralPath $p)) { throw "no ${p}: run $Producer first" }
    Get-Content -Raw -LiteralPath $p | ConvertFrom-Json
}
$build = Read-Json 'build-info.json' 'build.ps1'
$imports = Read-Json 'ggml-imports.json' 'gate.ps1'
$overlay = Join-Path $WorkDir 'overlay'
$licenses = Join-Path $WorkDir 'licenses'

$binaries = @(Get-ChildItem -LiteralPath $overlay -File | Sort-Object Name)
foreach ($f in $binaries) {
    if (-not (Test-OverlayBinaryName $f.Name) -or ($KeepBaseFiles -contains $f.Name)) { throw "$($f.Name) does not belong in an overlay" }
}
foreach ($need in 'llama-server.exe', 'llama-diffusion-gemma-visual-server.exe', 'llama.dll') {
    if (-not (Test-Path -LiteralPath (Join-Path $overlay $need))) { throw "the overlay has no $need" }
}

# Signature state, as found on the files: all signed by one subject, or none.
$sig = Get-AuthenticodeSignature -LiteralPath (Join-Path $overlay 'llama.dll')
$signer = if ($sig.Status -eq 'Valid') { $sig.SignerCertificate.Subject } else { $null }
foreach ($f in $binaries) {
    $s = Get-AuthenticodeSignature -LiteralPath $f.FullName
    $same = if ($signer) { $s.Status -eq 'Valid' -and $s.SignerCertificate.Subject -eq $signer } else { $s.Status -eq 'NotSigned' }
    if (-not $same) { throw "$($f.Name) is $($s.Status) but llama.dll is $($sig.Status): sign every overlay file or none" }
}

if (Test-Path -LiteralPath $OutDir) { Remove-Item -Recurse -Force -LiteralPath $OutDir }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

# The notices, filled in for this release.
$notices = Get-Content -Raw -LiteralPath (Join-Path $root 'THIRD-PARTY-NOTICES.md')
$boringText = Join-Path $licenses 'LICENSE-boringssl'
$notices = $notices.Replace('{{PATCH}}', $PatchName).Replace('{{BASE_TAG}}', $Tag).Replace('{{SOURCE_COMMIT}}', [string]$base.source_commit)
if (Test-Path -LiteralPath $boringText) {
    # Named from the text that ships beside it, so the two cannot disagree.
    $notices = $notices.Replace('{{BORINGSSL}}', "$(Get-BoringSslLicenseName $boringText); statically linked into the binaries")
    $notices = $notices.Replace('{{BORINGSSL_TEXT}}', 'LICENSE-boringssl')
} else {
    $notices = $notices.Replace('{{BORINGSSL}}', 'not included in this build').Replace('{{BORINGSSL_TEXT}}', '-')
}
$noticesPath = Join-Path $OutDir 'THIRD-PARTY-NOTICES.md'
Write-Utf8NoBom $noticesPath $notices

# Zip entries in a stable order: binaries at the top, then licenses/.
$entries = @()
foreach ($f in $binaries) { $entries += , @($f.Name, $f.FullName) }
foreach ($f in (Get-ChildItem -LiteralPath $licenses -File | Sort-Object Name)) { $entries += , @("licenses/$($f.Name)", $f.FullName) }
$entries += , @('licenses/LICENSE-fidim-dg-overlay', (Join-Path $root 'LICENSE'))
$entries += , @('licenses/THIRD-PARTY-NOTICES.md', $noticesPath)

$zipName = Get-OverlayZipName $Tag
$zipPath = Join-Path $OutDir $zipName
$files = New-Object System.Collections.Generic.List[object]
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($e in $entries) {
        [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $e[1], $e[0], [System.IO.Compression.CompressionLevel]::Optimal)
        $files.Add([ordered]@{ name = $e[0]; sha256 = (Get-Sha256 $e[1]); size = (Get-Item -LiteralPath $e[1]).Length })
    }
} finally { $zip.Dispose() }
Remove-Item -LiteralPath $noticesPath

# The descriptor.
$patchOut = Join-Path $OutDir "$PatchName.diff"
Copy-Item -LiteralPath $Patch -Destination $patchOut
$runUrl = if ($env:GITHUB_RUN_ID) { "$env:GITHUB_SERVER_URL/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID" } else { $null }
$repoSha = $env:GITHUB_SHA
if (-not $repoSha) {
    $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $repoSha = (git -C $root rev-parse HEAD 2>$null) } finally { $ErrorActionPreference = $eap }
}
$zips = [ordered]@{}
foreach ($p in $base.zips.PSObject.Properties) { $zips[$p.Name] = [string]$p.Value }
$ggmlImports = [ordered]@{}
foreach ($p in $imports.PSObject.Properties) { $ggmlImports[$p.Name] = @($p.Value) }
$descriptor = [ordered]@{
    schema       = 1
    name         = $PatchName
    overlay_repo = $Repo
    release_tag  = Get-OverlayReleaseTag $Tag
    zip          = $zipName
    base         = [ordered]@{
        repo          = [string]$base.repo
        release_tag   = $Tag
        source_commit = [string]$base.source_commit
        source_asset  = [string]$base.source_asset
        source_sha256 = [string]$base.source_sha256
        upstream_tag  = [string]$base.upstream_tag
        ggml_tree     = [string]$base.ggml_tree
        zips          = $zips
    }
    patch        = [ordered]@{ file = "$PatchName.diff"; sha256 = (Get-Sha256 $patchOut); features = $PatchFeatures }
    build        = [ordered]@{
        runner           = $build.runner
        toolchain        = $build.toolchain
        c_compiler       = $build.c_compiler
        cxx_compiler     = $build.cxx_compiler
        msvc_toolset     = $build.msvc_toolset
        vs_version       = $build.vs_version
        cmake            = $build.cmake
        cmake_flags      = @($build.cmake_flags)
        boringssl        = $build.boringssl
        workflow_run_url = $runUrl
        overlay_repo_sha = $(if ($repoSha) { [string]$repoSha } else { $null })
        built_at         = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    }
    files        = $files
    ggml_imports = $ggmlImports
    signer       = $signer
}
Write-Utf8NoBom (Join-Path $OutDir $DescriptorName) (ConvertTo-PrettyJson $descriptor)

# SHA256SUMS: LF and two spaces, so `sha256sum -c SHA256SUMS` reads it.
$sums = foreach ($n in $zipName, $DescriptorName, "$PatchName.diff") { "$(Get-Sha256 (Join-Path $OutDir $n))  $n" }
Write-Utf8NoBom (Join-Path $OutDir 'SHA256SUMS') (($sums -join "`n") + "`n")

$mb = [math]::Round((Get-Item -LiteralPath $zipPath).Length / 1MB, 1)
Write-Host "== packaged $zipName ($mb MB, $($files.Count) files, $(if ($signer) { "signed by $signer" } else { 'unsigned' }))"
Get-ChildItem -LiteralPath $OutDir | ForEach-Object { Write-Host "  $($_.Name)" }
