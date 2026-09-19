# Shared settings and helpers, dot-sourced by the other scripts. Runs under
# Windows PowerShell 5.1 (a developer box) and PowerShell 7 (the CI runner).

# Where overlay releases are published. The one place to change it; the
# workflow passes its own repository instead, and Llama FIDIM has the same
# name in update.rs (OVERLAY_REPO, or `overlay_repo` in its config.json).
$OverlayRepo = 'Dixon-Cider/fidim-dg-overlay'
# The patch this repo carries. Release tags are `<PatchName>-<unsloth tag>`.
$PatchName = 'dgpatch5'
# The base every overlay is built against and installed over.
$BaseRepo = 'unslothai/llama.cpp'
$DescriptorName = 'fidim-overlay.json'
# What dgpatch5 changes in the runner, as Llama FIDIM names it (its
# discovery::dg_feature constants; unknown names are ignored there).
$PatchFeatures = @('dg-pkv-f16', 'dg-swa-ring', 'dg-fa-pad', 'dg-fa-turn-sizing', 'dg-step-fail-err',
                   'dg-frame-special', 'dg-prefill-reuse', 'dg-sc-splitk')

function Get-OverlayZipName([string]$Tag) { "fidim-dg-overlay-$Tag-windows-x64.zip" }
function Get-OverlayReleaseTag([string]$Tag) { "$PatchName-$Tag" }

# `b11030-mix-5ff778e`: the upstream build number and the hash of what
# Unsloth merged into it. Anything else is not a release we can build on.
function Assert-UnslothTag([string]$Tag) {
    if ($Tag -notmatch '^b\d+-mix-[0-9a-f]{7,40}$') { throw "'$Tag' is not an Unsloth release tag (b<n>-mix-<sha>)" }
}

# A file the overlay ships: every llama-level binary the build produces.
# ggml*, the ROCm/HIP runtime and everything else stay Unsloth's.
function Test-OverlayBinaryName([string]$Name) {
    $n = $Name.ToLowerInvariant()
    foreach ($p in 'ggml', 'amd', 'hip', 'roc', 'origami', 'lib') { if ($n.StartsWith($p)) { return $false } }
    if ($n -match '[\\/]') { return $false }
    return ($n -like 'llama*.exe') -or ($n -like 'llama*.dll') -or ($n -eq 'mtmd.dll')
}
# Unsloth's copy is kept: it imports ggml-hip.dll for its GPU path, which a
# build without HIP cannot reproduce.
$KeepBaseFiles = @('llama-cvector-generator.exe')

# Every script works in one folder:
#   dl\        downloads            src\       the patched source
#   build\     the CMake build      overlay\   the overlay binaries
#   licenses\  their license texts  gate\      the gate's scratch
#   base.json (fetch-base), build-info.json (build), ggml-imports.json (gate)
function Read-BaseInfo([string]$WorkDir) {
    $p = Join-Path $WorkDir 'base.json'
    if (-not (Test-Path -LiteralPath $p)) { throw "no ${p}: run fetch-base.ps1 first" }
    Get-Content -Raw -LiteralPath $p | ConvertFrom-Json
}

function Get-Sha256([string]$Path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }

# UTF-8 without a BOM and with LF line ends, so sha256sum and Llama FIDIM's
# JSON reader (serde_json does not skip a BOM) both take the files as-is.
function Write-Utf8NoBom([string]$Path, [string]$Text) {
    $Text = $Text -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

function ConvertTo-PrettyJson($Object) { $Object | ConvertTo-Json -Depth 12 }

# Run a native command and fail on a non-zero exit. Windows PowerShell turns
# a native command's stderr into errors under 'Stop', so lower it meanwhile.
function Invoke-Native([string]$What, [scriptblock]$Command) {
    $eap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Command; $code = $LASTEXITCODE } finally { $ErrorActionPreference = $eap }
    if ($code -ne 0) { throw "$What failed (exit $code)" }
}

if ($PSVersionTable.PSVersion.Major -lt 6) {
    # Windows PowerShell's .NET Framework may still offer TLS 1.0 first.
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
}

function Get-GitHubHeaders {
    $h = @{ 'User-Agent' = 'fidim-dg-overlay'; 'Accept' = 'application/vnd.github+json' }
    $token = if ($env:GH_TOKEN) { $env:GH_TOKEN } elseif ($env:GITHUB_TOKEN) { $env:GITHUB_TOKEN } else { $null }
    if ($token) { $h['Authorization'] = "Bearer $token" }
    return $h
}

function Get-GitHubRelease([string]$Repo, [string]$Tag) {
    $url = if ($Tag -eq 'latest') { "https://api.github.com/repos/$Repo/releases/latest" }
           else { "https://api.github.com/repos/$Repo/releases/tags/$Tag" }
    Invoke-RestMethod -Uri $url -Headers (Get-GitHubHeaders)
}

# Download one release asset and check it against GitHub's sha256 digest.
# Returns the local path. curl.exe ships with Windows 10 and later, and
# streams large files far faster than Invoke-WebRequest under 5.1.
function Save-ReleaseAsset($Release, [string]$Name, [string]$Dir) {
    $asset = @($Release.assets | Where-Object { $_.name -eq $Name })
    if ($asset.Count -ne 1) { throw "release $($Release.tag_name) has no asset $Name" }
    $asset = $asset[0]
    New-Item -ItemType Directory -Force $Dir | Out-Null
    $out = Join-Path $Dir $Name
    $want = if ($asset.digest) { ([string]$asset.digest) -replace '^sha256:', '' } else { $null }
    if ((Test-Path -LiteralPath $out) -and $want -and ((Get-Sha256 $out) -eq $want)) {
        Write-Host "  $Name already downloaded"
        return $out
    }
    Write-Host "  downloading $Name ($([math]::Round($asset.size / 1MB)) MB)"
    Invoke-Native "download $Name" { curl.exe -fsSL --retry 3 -o $out $asset.browser_download_url }
    $got = Get-Sha256 $out
    if (-not $want) { throw "$Name has no published digest; refusing to use it unchecked" }
    if ($got -ne $want) { throw "sha256 mismatch for ${Name}: GitHub publishes $want, the download is $got" }
    return $out
}

# Keep git from finding a repository above `Dir`: `git apply` would patch
# relative to that repository, and the build stamps the version from it.
function Set-GitCeiling([string]$Dir) {
    $env:GIT_CEILING_DIRECTORIES = (Split-Path -Parent ([System.IO.Path]::GetFullPath($Dir)))
}

# Read a flat zip's top-level files (no folders) into a directory.
function Expand-ZipTopLevel([string]$Zip, [string]$Dest, [string[]]$Only = $null) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    New-Item -ItemType Directory -Force $Dest | Out-Null
    $z = [System.IO.Compression.ZipFile]::OpenRead($Zip)
    try {
        foreach ($e in $z.Entries) {
            if ($e.FullName.Contains('/') -or $e.FullName.Contains('\') -or -not $e.Name) { continue }
            if ($Only -and ($Only -notcontains $e.Name)) { continue }
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($e, (Join-Path $Dest $e.Name), $true)
        }
    } finally { $z.Dispose() }
}

# The Visual Studio 2022 toolset (MSVC 14.4x, the redist floor Unsloth's
# builds use), imported into this process like a Developer prompt would.
function Import-VsDevEnv {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { throw "vswhere.exe not found: install Visual Studio 2022 Build Tools with the C++ workload" }
    $vc = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
    # Prefer an instance with the bundled CMake and Ninja.
    $vs = & $vswhere -version '[17.0,18.0)' -products '*' -requires $vc Microsoft.VisualStudio.Component.VC.CMake.Project -latest -property installationPath
    if (-not $vs) { $vs = & $vswhere -version '[17.0,18.0)' -products '*' -requires $vc -latest -property installationPath }
    if (-not $vs) { throw "no Visual Studio 2022 with the x64 C++ tools (VS 2026 is not used: its STL raises the redist floor)" }
    $vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
    # vcvars probes for tools it may not find and says so on stderr; whether
    # it worked shows in the variables it sets.
    $lines = cmd.exe /c "`"$vcvars`" >nul 2>nul && set"
    foreach ($l in $lines) {
        $i = $l.IndexOf('=')
        if ($i -gt 0) { [Environment]::SetEnvironmentVariable($l.Substring(0, $i), $l.Substring($i + 1), 'Process') }
    }
    if (-not $env:VCToolsVersion) { throw "$vcvars did not set up the MSVC tools" }
    return $vs
}
