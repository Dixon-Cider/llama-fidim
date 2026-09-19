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
if ((Test-Path -LiteralPath $stamp) -and ((Get-Content -Raw -LiteralPath $stamp).Trim() -eq "$patchSha $Tag")) {
    Write-Host "== $PatchName already applied to $src"
    return
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
Write-Utf8NoBom $stamp "$patchSha $Tag"
