<#
.SYNOPSIS
  Exercise the installer hooks (ui\src-tauri\windows\hooks.nsh) without
  installing anything. A small NSIS program inserts the same NSIS_HOOK_*
  macros the installer runs, with $INSTDIR, the old install folder and the
  old Start Menu shortcut pointed into a temporary folder, while copies of
  fidim.exe run from there as stand-ins for keep-alive helpers and
  diffusion servers. Then the results are checked.

.DESCRIPTION
  Needs the NSIS that Tauri downloads (%LOCALAPPDATA%\tauri\NSIS) and the
  utils.nsh Tauri writes beside its installer script, so build the
  installer once first:
    cd ui; pnpm tauri build --bundles nsis --config src-tauri/tauri.release.conf.json
  Nothing outside the temporary folder is touched. The stand-in processes
  run `fidim keepalive` with a one-hour interval and FIDIM_HOME pointed at
  the temporary folder, so they sleep and send nothing; they are stopped
  at the end.

  NOTE: ASCII only, deliberately. Windows PowerShell 5.1 reads .ps1 as ANSI
  unless the file has a BOM.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\test-installer-hooks.ps1
#>
[CmdletBinding()]
param(
  [string]$Makensis = (Join-Path $env:LOCALAPPDATA 'tauri\NSIS\makensis.exe'),
  # Default: target\release\nsis\x64 in this checkout.
  [string]$TauriNsisDir,
  # Default: target\release\fidim.exe in this checkout.
  [string]$Fidim
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (-not $TauriNsisDir) { $TauriNsisDir = Join-Path $repo 'target\release\nsis\x64' }
if (-not $Fidim) { $Fidim = Join-Path $repo 'target\release\fidim.exe' }
$hooks = Join-Path $repo 'ui\src-tauri\windows\hooks.nsh'
foreach ($f in $Makensis, (Join-Path $TauriNsisDir 'utils.nsh'), $Fidim, $hooks) {
  if (-not (Test-Path $f)) { throw "missing $f (build the installer once first; see the help)" }
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("fidim-hooks-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
$inst = Join-Path $tmp 'Llama FIDIM'
$old = Join-Path $tmp 'Programs\LlamaFIDIM'
$sm = Join-Path $tmp 'Start Menu'
$lnk = Join-Path $sm 'Llama FIDIM.lnk'
New-Item -ItemType Directory -Force $inst, $old, $sm, (Join-Path $tmp 'home') | Out-Null

$failures = New-Object System.Collections.Generic.List[string]
function Check([bool]$ok, [string]$what) {
  if ($ok) { Write-Host "  ok   $what" } else { Write-Host "  FAIL $what" -ForegroundColor Red; $failures.Add($what) }
}
# The image name tasklist reports, as supervise.rs reads it for `fidim stop`.
function Image([int]$procId) {
  $line = & tasklist /FI "PID eq $procId" /NH /FO CSV | Where-Object { $_ -match ('"' + $procId + '"') }
  if ($line) { ($line -split '","')[0].Trim('"') } else { $null }
}

$procs = @()
$env:FIDIM_HOME = Join-Path $tmp 'home'
function Start-Helper([string]$path) {
  Copy-Item $Fidim $path -Force
  $p = Start-Process -FilePath $path -WindowStyle Hidden -PassThru `
    -ArgumentList @('keepalive', '--port', '9', '--interval', '3600', '--server-pid', "$PID")
  $script:procs += $p
  Start-Sleep -Milliseconds 300
  $p
}

$handles = @()
try {
  # The NSIS program: the installer's order, PREINSTALL then POSTINSTALL,
  # or the uninstaller's PREUNINSTALL.
  $nsi = Join-Path $tmp 'hooks-test.nsi'
  $utils = Join-Path $TauriNsisDir 'utils.nsh'
  @"
Unicode true
RequestExecutionLevel user
SilentInstall silent
OutFile "hooks-test.exe"
Var OLD
Var LNK
Var PHASE
!define FIDIM_OLD_INSTALL_DIR "`$OLD"
!define FIDIM_OLD_SHORTCUT "`$LNK"
!include LogicLib.nsh
!include FileFunc.nsh
!include x64.nsh
!include "$utils"
!include "Win\COM.nsh"
!include "Win\Propkey.nsh"
!include "$hooks"
Section
  `${GetParameters} `$0
  `${GetOptions} `$0 "/INST=" `$INSTDIR
  `${GetOptions} `$0 "/OLD=" `$OLD
  `${GetOptions} `$0 "/LNK=" `$LNK
  `${GetOptions} `$0 "/PHASE=" `$PHASE
  `${If} `$PHASE == "install"
    !insertmacro NSIS_HOOK_PREINSTALL
    !insertmacro NSIS_HOOK_POSTINSTALL
  `${Else}
    !insertmacro NSIS_HOOK_PREUNINSTALL
  `${EndIf}
SectionEnd
"@ | Set-Content $nsi -Encoding UTF8
  Push-Location $tmp
  try {
    $out = & $Makensis -V2 -INPUTCHARSET UTF8 $nsi 2>&1
    if ($LASTEXITCODE -ne 0) { $out | Write-Host; throw "makensis failed ($LASTEXITCODE)" }
  } finally { Pop-Location }
  $exe = Join-Path $tmp 'hooks-test.exe'
  function Run-Hooks([string]$phase) {
    $p = Start-Process -FilePath $exe -Wait -PassThru `
      -ArgumentList @("/PHASE=$phase", "/INST=`"$inst`"", "/OLD=`"$old`"", "/LNK=`"$lnk`"")
    if ($p.ExitCode -ne 0) { throw "hooks-test.exe /PHASE=$phase exited $($p.ExitCode)" }
  }
  function Aside([string]$dir, [string]$name) {
    @(Get-ChildItem $dir -Filter "$name.old-*" -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match ('^' + [regex]::Escape($name) + '\.old-\d{14}(-\d+)?$') -and $_.Name -notmatch '-19990101000000$' })
  }
  function New-Lnk([string]$target) {
    $sc = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    $sc.TargetPath = $target
    $sc.WorkingDirectory = Split-Path $target
    $sc.Save()
  }

  Write-Host "== 1. install over running helpers, retiring an old script install"
  # A server an earlier install moved aside, still running (renamed while
  # it runs, the way the hooks do it).
  $c = Start-Helper (Join-Path $inst 'fidim-dg.exe')
  Rename-Item (Join-Path $inst 'fidim-dg.exe') 'fidim-dg.exe.old-19990101000000'
  Copy-Item $Fidim (Join-Path $inst 'fidim.exe.old-19990101000000')     # one that has exited
  $a = Start-Helper (Join-Path $inst 'fidim.exe')                       # a keep-alive helper
  $b = Start-Helper (Join-Path $inst 'fidim-dg.exe')                    # a diffusion server
  $d = Start-Helper (Join-Path $old 'fidim.exe')                        # a helper from the old folder
  Copy-Item $Fidim (Join-Path $old 'fidim-dg.exe')
  Copy-Item $Fidim (Join-Path $old 'llama-fidim.exe')
  New-Lnk (Join-Path $old 'llama-fidim.exe')                            # what install.ps1 wrote
  Run-Hooks 'install'
  Check (-not (Test-Path (Join-Path $inst 'fidim.exe'))) 'running fidim.exe left its name'
  Check (-not (Test-Path (Join-Path $inst 'fidim-dg.exe'))) 'running fidim-dg.exe left its name'
  Check ((Aside $inst 'fidim.exe').Count -eq 1) 'fidim.exe moved aside as fidim.exe.old-<yyyyMMddHHmmss>'
  Check ((Aside $inst 'fidim-dg.exe').Count -eq 1) 'fidim-dg.exe moved aside as fidim-dg.exe.old-<yyyyMMddHHmmss>'
  Check (-not (Test-Path (Join-Path $inst 'fidim.exe.old-19990101000000'))) 'an exited leftover is deleted'
  Check (Test-Path (Join-Path $inst 'fidim-dg.exe.old-19990101000000')) 'a running leftover stays'
  foreach ($p in $a, $b, $c, $d) { Check (-not $p.HasExited) "helper pid $($p.Id) still running" }
  Check ((Image $a.Id) -eq 'fidim.exe') "a renamed keep-alive still reads as fidim.exe in tasklist (got '$(Image $a.Id)')"
  Check ((Image $b.Id) -eq 'fidim-dg.exe') "a renamed server still reads as fidim-dg.exe in tasklist (got '$(Image $b.Id)')"
  Check (-not (Test-Path $lnk)) 'the old Start Menu entry is removed'
  Check (-not (Test-Path (Join-Path $old 'llama-fidim.exe'))) 'old llama-fidim.exe deleted'
  Check (-not (Test-Path (Join-Path $old 'fidim-dg.exe'))) 'old fidim-dg.exe deleted'
  Check ((Aside $old 'fidim.exe').Count -eq 1) 'old running fidim.exe moved aside'
  Check (Test-Path $old) 'the old folder stays while something runs from it'

  Write-Host "== 2. a shortcut that points elsewhere is left alone; same-second renames do not collide"
  New-Lnk (Join-Path $inst 'llama-fidim.exe')
  $e = Start-Helper (Join-Path $inst 'fidim.exe')
  # Hold every name this second and the next few would get, the way a
  # running copy renamed a moment earlier would.
  $now = Get-Date
  foreach ($s in 0..5) {
    $f = Join-Path $inst ('fidim.exe.old-' + $now.AddSeconds($s).ToString('yyyyMMddHHmmss'))
    if (-not (Test-Path $f)) { Set-Content $f 'held' }
    $handles += [IO.File]::Open($f, 'Open', 'Read', 'Read')
  }
  Run-Hooks 'install'
  Check (Test-Path $lnk) 'a Start Menu entry for the new folder stays'
  Check (-not (Test-Path (Join-Path $inst 'fidim.exe'))) 'the second running fidim.exe left its name'
  Check (@(Get-ChildItem $inst -Filter 'fidim.exe.old-*' | Where-Object { $_.Name -match '^fidim\.exe\.old-\d{14}-\d+$' }).Count -eq 1) 'it took a -2 style name instead of an existing one'
  Check (-not $e.HasExited) 'and it keeps running'

  Write-Host "== 3. uninstall moves running helpers aside too"
  $g = Start-Helper (Join-Path $inst 'fidim.exe')
  Run-Hooks 'uninstall'
  Check (-not (Test-Path (Join-Path $inst 'fidim.exe'))) 'running fidim.exe left its name'
  Check (-not $g.HasExited) 'and it keeps running'

  Write-Host "== 4. once everything has exited, the next install cleans up"
  foreach ($h in $handles) { $h.Dispose() }
  $handles = @()
  foreach ($p in $procs) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
  foreach ($p in $procs) { $p.WaitForExit(5000) | Out-Null }
  Run-Hooks 'install'
  Check (@(Get-ChildItem $inst -Filter '*.old-*').Count -eq 0) 'no renamed copies left in the install folder'
  Check (-not (Test-Path $old)) 'the old folder is gone'
} finally {
  foreach ($h in $handles) { $h.Dispose() }
  foreach ($p in $procs) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
  Start-Sleep -Milliseconds 300
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item Env:\FIDIM_HOME -ErrorAction SilentlyContinue
}

if ($failures.Count) {
  Write-Host "$($failures.Count) check(s) failed" -ForegroundColor Red
  exit 1
}
Write-Host "all hook checks passed" -ForegroundColor Green
exit 0
