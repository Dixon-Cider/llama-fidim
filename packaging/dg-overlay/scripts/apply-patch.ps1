<#
.SYNOPSIS
  Apply the runner patch to the unpacked Unsloth source (<WorkDir>\src) and
  replace Unsloth's build fingerprint with the overlay's.

.DESCRIPTION
  `git apply --check` first, so a tag the patch no longer fits fails before
  anything changes. The fingerprint is the text --version prints after the
  target ("Windows x86_64 (Compiled by the Unsloth team)"); binaries built
  here say they are the Llama FIDIM overlay instead. BUILD_NUMBER and
  BUILD_COMMIT stay as Unsloth stamped them, so the version line still names
  the release's build and source commit.

  Rerunning it with the same patch and tag does nothing. With an edited
  patch, or another tag of the same source, it unpacks the release's source
  again from dl\ first: the patch applies to that, not to a patched tree.

.EXAMPLE
  .\apply-patch.ps1 -WorkDir $env:TEMP\fidim-dgo
#>
param(
    [Parameter(Mandatory = $true)] [string]$WorkDir,
    # Default: patches\<PatchName>.diff next to this folder.
    [string]$Patch = ''
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$WorkDir = (Resolve-Path -LiteralPath $WorkDir).Path
$base = Read-BaseInfo $WorkDir
$Tag = $base.release_tag
if (-not $Patch) { $Patch = Join-Path (Split-Path -Parent $PSScriptRoot) "patches\$PatchName.diff" }
$Patch = (Resolve-Path -LiteralPath $Patch).Path
$src = Join-Path $WorkDir 'src'
$patchSha = Get-Sha256 $Patch
$stamp = Join-Path $WorkDir 'patch.stamp'
$want = "$patchSha $Tag"
$have = if (Test-Path -LiteralPath $stamp) { (Get-Content -Raw -LiteralPath $stamp).Trim() } else { '' }
if ($have -eq $want) {
    Write-Host "== $PatchName already applied to $src"
    return
}
if ($have) {
    # The tree carries something else: another revision of the patch, the
    # same source under another tag (the fingerprint names the tag), or an
    # apply that did not finish. The patch fits the release's source, not
    # that, so start again from the tarball.
    Write-Host "== $src carries another patch or tag ($have): unpacking the release's source again"
    $tarball = Join-Path $WorkDir "dl\$($base.source_asset)"
    if (-not (Test-Path -LiteralPath $tarball) -or (Get-Sha256 $tarball) -ne [string]$base.source_sha256) {
        throw "$tarball is missing or is not the source base.json names: run fetch-base.ps1 again"
    }
    Expand-BaseSource $WorkDir $tarball ([string]$base.source_sha256)
}

Write-Host "== applying $(Split-Path -Leaf $Patch) ($patchSha) to $Tag"
Set-GitCeiling $src
Push-Location -LiteralPath $src
try {
    # autocrlf off: a CRLF rewrite of the patched files would not match the
    # next patch's context. longpaths: the tree nests deep.
    Invoke-Native "git apply --check (the patch no longer fits ${Tag}: rebase it)" {
        git -c core.autocrlf=false -c core.longpaths=true apply --check $Patch
    }
    # From here on the tree is no longer the release's source: a rerun that
    # finds this stamp unpacks it again instead of patching a patched tree.
    Write-Utf8NoBom $stamp "applying $want"
    Invoke-Native 'git apply' { git -c core.autocrlf=false -c core.longpaths=true apply $Patch }
} finally { Pop-Location }

# A patch applied relative to some other repository would leave this tree
# unchanged without an error; check for something only the patch adds.
if (-not (Select-String -LiteralPath (Join-Path $src 'include\llama.h') -SimpleMatch 'llama_diffusion_fa_turn_bytes' -Quiet)) {
    throw "the patch did not land in $src (include\llama.h lacks llama_diffusion_fa_turn_bytes)"
}

$bi = Join-Path $src 'cmake\build-info.cmake'
$text = [System.IO.File]::ReadAllText($bi)
$ours = 'set(BUILD_TARGET "${BUILD_TARGET} (Llama FIDIM ' + $PatchName + ' overlay on Unsloth ' + $Tag + ')")'
$comment = '# Llama FIDIM overlay fingerprint: shows in --version and strings.'
$lines = [System.Collections.Generic.List[string]]($text -split "`n")
$replaced = $false
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i] -match 'Compiled by the Unsloth team') { $lines[$i] = $ours; $replaced = $true }
    elseif ($lines[$i] -match '^#\s*Unsloth fingerprint') { $lines[$i] = $comment }
}
$text = $lines -join "`n"
if (-not $replaced) {
    # Unsloth dropped its fingerprint: add ours anyway.
    $text = $text.TrimEnd("`n") + "`n`n$comment`n$ours`n"
}
[System.IO.File]::WriteAllText($bi, $text, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "  build fingerprint: (Llama FIDIM $PatchName overlay on Unsloth $Tag)"
Write-Utf8NoBom $stamp $want
