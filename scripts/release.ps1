<#
.SYNOPSIS
  Cut a Llama FIDIM release: set the version everywhere it lives, date the
  changelog's Unreleased section, commit, and tag vX.Y.Z. Pushes nothing.

  First list what the release changes under "## [Unreleased]" in
  CHANGELOG.md. Pushing the tag afterwards is what publishes: the release
  workflow builds the zips and attaches them, with that changelog section
  as the release notes.

  Versions follow semver. Before 1.0: a new feature or a change in
  behavior bumps the minor version (0.2.0 -> 0.3.0), a fix alone bumps the
  patch (0.2.0 -> 0.2.1).
.PARAMETER Version
  The new version, X.Y.Z, above the current one.
.PARAMETER Title
  One line for the tag message ("vX.Y.Z: <Title>").
.PARAMETER DryRun
  Check everything and show what would change, writing nothing.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\release.ps1 0.3.0 -DryRun
  powershell -ExecutionPolicy Bypass -File scripts\release.ps1 0.3.0 -Title "router autoload for diffusion profiles"
#>
param(
  [Parameter(Mandatory = $true, Position = 0)][string]$Version,
  [string]$Title,
  [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$utf8 = New-Object System.Text.UTF8Encoding($false)

# git and cargo report progress and warnings on stderr, which Windows
# PowerShell turns into a terminating error under 'Stop'. Judge them by
# their exit code instead, and show what they printed when they fail (a
# failing test reports on stdout).
function Invoke-Native([string]$Exe, [string[]]$Argv) {
  $eap = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try { $out = & $Exe @Argv } finally { $ErrorActionPreference = $eap }
  if ($LASTEXITCODE -ne 0) {
    if ($out) { $out | Out-Host }
    throw "$Exe $($Argv -join ' ') exited $LASTEXITCODE"
  }
  $out
}

# Replace the first match's version, keeping what the pattern's two groups
# capture around it.
function Set-FirstVersion([string]$Text, [string]$Pattern, [string]$What) {
  $re = [regex]$Pattern
  if (-not $re.IsMatch($Text)) { throw "no version field in $What" }
  $re.Replace($Text, ('${1}' + $Version + '${2}'), 1)
}

if ($Version -notmatch '^\d+\.\d+\.\d+$') { throw "the version must be X.Y.Z, not '$Version'" }
$tag = "v$Version"

$paths = [ordered]@{
  cargo   = 'Cargo.toml'
  package = 'ui/package.json'
  tauri   = 'ui/src-tauri/tauri.conf.json'
  log     = 'CHANGELOG.md'
}
$text = @{}
foreach ($k in $paths.Keys) { $text[$k] = [IO.File]::ReadAllText((Join-Path $repo $paths[$k])) }

$cargoVersion = '(?m)^(version = ")\d+\.\d+\.\d+(")'
$jsonVersion = '(?m)^(\s*"version":\s*")\d+\.\d+\.\d+(")'
$current = [regex]::Match($text.cargo, '(?m)^version = "(\d+\.\d+\.\d+)"').Groups[1].Value
if (-not $current) { throw 'no version line in Cargo.toml' }
foreach ($k in 'package', 'tauri') {
  $v = [regex]::Match($text[$k], '(?m)^\s*"version":\s*"(\d+\.\d+\.\d+)"').Groups[1].Value
  if ($v -ne $current) { throw "$($paths[$k]) says '$v' but Cargo.toml says $current; fix that first" }
}
if ([version]$Version -le [version]$current) { throw "$Version is not above the current version $current" }
$repoUrl = [regex]::Match($text.cargo, '(?m)^repository = "([^"]+)"').Groups[1].Value
if (-not $repoUrl) { throw 'no repository URL in Cargo.toml' }

Push-Location $repo
try {
  $branch = Invoke-Native git @('rev-parse', '--abbrev-ref', 'HEAD') | Select-Object -First 1
  if ($branch -ne 'main') { throw "releases are cut from main; this checkout is on '$branch'" }
  # The check below builds the working tree, so it must be exactly what gets
  # committed: no changes and no untracked files (a new source file the
  # commit lacks would build here and break the tagged commit).
  $dirty = Invoke-Native git @('status', '--porcelain')
  if ($dirty) { throw "commit, stash or ignore these first:`n$($dirty -join "`n")" }
  $touched = @('Cargo.lock') + @($paths.Values)
  foreach ($f in $touched) {
    if (-not (Invoke-Native git @('ls-files', '--', $f))) { throw "$f is not tracked; commit it first" }
  }
  if (Invoke-Native git @('tag', '--list', $tag)) { throw "tag $tag already exists" }

  # The changelog: "## [Unreleased]" keeps its heading, and what was under
  # it becomes the dated section for this version.
  $nl = "`n"
  if ($text.log.Contains("`r`n")) { $nl = "`r`n" }
  $head = [regex]::Match($text.log, '(?m)^## \[Unreleased\][^\r\n]*\r?\n')
  if (-not $head.Success) { throw 'CHANGELOG.md has no "## [Unreleased]" heading' }
  $rest = $text.log.Substring($head.Index + $head.Length)
  $notes = $rest
  $next = [regex]::Match($notes, '(?m)^## \[')
  if ($next.Success) { $notes = $notes.Substring(0, $next.Index) }
  $refs = [regex]::Match($notes, '(?m)^\[[^\]]+\]: ')
  if ($refs.Success) { $notes = $notes.Substring(0, $refs.Index) }
  if (-not $notes.Trim()) { throw 'the Unreleased section of CHANGELOG.md is empty: list what this release changes first' }
  $date = Get-Date -Format 'yyyy-MM-dd'
  $log = $text.log.Substring(0, $head.Index) + "## [Unreleased]$nl$nl## [$Version] - $date$nl" + $rest
  $links = "[Unreleased]: $repoUrl/compare/$tag...HEAD$nl[$Version]: $repoUrl/compare/v$current...$tag"
  $unreleasedLink = [regex]::Match($log, '(?m)^\[Unreleased\]: \S+')
  if ($unreleasedLink.Success) {
    $log = $log.Remove($unreleasedLink.Index, $unreleasedLink.Length).Insert($unreleasedLink.Index, $links)
  } else {
    $log = $log.TrimEnd() + $nl + $nl + $links + $nl
  }

  $new = @{
    cargo   = Set-FirstVersion $text.cargo $cargoVersion 'Cargo.toml'
    package = Set-FirstVersion $text.package $jsonVersion 'ui/package.json'
    tauri   = Set-FirstVersion $text.tauri $jsonVersion 'ui/src-tauri/tauri.conf.json'
    log     = $log
  }

  Write-Host "== $current -> $Version" -ForegroundColor Cyan
  Write-Host "   Cargo.toml, Cargo.lock, ui/package.json, ui/src-tauri/tauri.conf.json"
  Write-Host "   CHANGELOG.md: ## [$Version] - $date"
  Write-Host "   commit 'chore(release): $tag', tag $tag"
  Write-Host "== release notes" -ForegroundColor Cyan
  Write-Host $notes.Trim()
  if ($DryRun) {
    Write-Host "`n(dry run: nothing written)"
    return
  }

  # Every file's bytes as they are now, to put back if anything fails.
  $restore = @{}
  foreach ($f in $touched) { $restore[$f] = [IO.File]::ReadAllBytes((Join-Path $repo $f)) }
  $committed = $false
  try {
    foreach ($k in $paths.Keys) { [IO.File]::WriteAllText((Join-Path $repo $paths[$k]), $new[$k], $utf8) }
    Write-Host "== Cargo.lock" -ForegroundColor Cyan
    Invoke-Native cargo @('update', '--workspace', '--offline') | Out-Host
    Write-Host "== checking every copy of the version agrees" -ForegroundColor Cyan
    Invoke-Native cargo @('test', '-q', '-p', 'fidim-core', '--lib', 'build_info') | Out-Host
    Invoke-Native git (@('add', '--') + $touched) | Out-Null
    Invoke-Native git @('commit', '-q', '-m', "chore(release): $tag") | Out-Null
    $committed = $true
  } finally {
    # A failure before the commit puts every file back as it was read.
    if (-not $committed) {
      # Each file on its own, and only when it changed: one that cannot be
      # written must not keep the others from being restored.
      $failed = @()
      foreach ($f in $restore.Keys) {
        $p = Join-Path $repo $f
        try {
          $was = [Convert]::ToBase64String($restore[$f])
          if ([Convert]::ToBase64String([IO.File]::ReadAllBytes($p)) -ne $was) { [IO.File]::WriteAllBytes($p, $restore[$f]) }
        } catch { $failed += $f }
      }
      $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
      & git reset -q -- $touched 2>$null
      $ErrorActionPreference = $eap
      if ($failed) {
        Write-Host "== nothing released, but could not restore $($failed -join ', '): git checkout -- $($failed -join ' ')" -ForegroundColor Red
      } else {
        Write-Host "== nothing released: the files are back as they were" -ForegroundColor Yellow
      }
    }
  }
  $sha = Invoke-Native git @('rev-parse', '--short=7', 'HEAD') | Select-Object -First 1
  # From a file: a title with quotes survives no command line intact.
  $message = $tag
  if ($Title) { $message = "${tag}: $Title" }
  $msgFile = [IO.Path]::GetTempFileName()
  try {
    [IO.File]::WriteAllText($msgFile, $message + "`n", $utf8)
    Invoke-Native git @('tag', '-a', $tag, '-F', $msgFile) | Out-Null
  } catch {
    Write-Host "the release commit $sha exists but tagging failed: $_" -ForegroundColor Red
    Write-Host "finish with:  git tag -a $tag -m `"$tag`"      or undo with:  git reset --hard HEAD~1"
    throw
  } finally {
    Remove-Item -LiteralPath $msgFile -ErrorAction SilentlyContinue
  }

  Write-Host ""
  Write-Host "released $tag locally (commit $sha). Nothing was pushed." -ForegroundColor Green
  Write-Host "To publish:  git push origin main  then  git push origin $tag"
  Write-Host "  (the tag builds the release zips; its notes are the $Version changelog section)"
  Write-Host "Reinstall (scripts\install.ps1) so the installed copy reads $Version."
} finally {
  Pop-Location
}
