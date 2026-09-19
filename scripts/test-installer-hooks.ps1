<#
.SYNOPSIS
  Exercise the installer hooks (ui\src-tauri\windows\hooks.nsh) without
  installing anything. A small NSIS program runs the same NSIS_HOOK_*
  macros the installer runs, in the installer's order around Tauri's own
  "app is running" check, with $INSTDIR, the old install folder and the
  old Start Menu shortcut pointed into a temporary folder, while copies of
  fidim.exe run from there as stand-ins for keep-alive helpers, diffusion
  servers and the app itself. Then the results are checked.

.DESCRIPTION
  Needs the NSIS that Tauri downloads (%LOCALAPPDATA%\tauri\NSIS) and the
  utils.nsh and English.nsh Tauri writes beside its installer script, so
  build the installer once first:
    cd ui; pnpm tauri build --bundles nsis --config src-tauri/tauri.release.conf.json
  Nothing outside the temporary folder is touched. Tauri's check closes
  the app by executable name, every copy the user runs; here the app is a
  stand-in with a name of its own (fidim-hooks-gui-<id>.exe), so a running
  Llama FIDIM is never closed. The stand-in processes run `fidim keepalive`
  with a one-hour interval and FIDIM_HOME pointed at the temporary folder,
  so they sleep and send nothing; they are stopped at the end.

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
$utils = Join-Path $TauriNsisDir 'utils.nsh'
$english = Join-Path $TauriNsisDir 'English.nsh'
# Where Tauri keeps its nsis_tauri_utils plug-in, which the check calls.
$plugins = Join-Path (Split-Path $Makensis) 'Plugins\x86-unicode\additional'
foreach ($f in $Makensis, $utils, $english, (Join-Path $plugins 'nsis_tauri_utils.dll'), $Fidim, $hooks) {
  if (-not (Test-Path $f)) { throw "missing $f (build the installer once first; see the help)" }
}

$id = [guid]::NewGuid().ToString('N').Substring(0, 8)
$tmp = Join-Path ([IO.Path]::GetTempPath()) "fidim-hooks-$id"
$inst = Join-Path $tmp 'Llama FIDIM'
$old = Join-Path $tmp 'Programs\LlamaFIDIM'
$sm = Join-Path $tmp 'Start Menu'
$lnk = Join-Path $sm 'Llama FIDIM.lnk'
$guiName = "fidim-hooks-gui-$id"
New-Item -ItemType Directory -Force $inst, $old, $sm, (Join-Path $tmp 'home'), (Join-Path $tmp 'gui') | Out-Null

$failures = New-Object System.Collections.Generic.List[string]
function Check([bool]$ok, [string]$what) {
  if ($ok) { Write-Host "  ok   $what" } else { Write-Host "  FAIL $what" -ForegroundColor Red; $failures.Add($what) }
}
# The image name tasklist reports, as supervise.rs reads it for `fidim stop`.
function Image([int]$procId) {
  $line = & tasklist /FI "PID eq $procId" /NH /FO CSV | Where-Object { $_ -match ('"' + $procId + '"') }
  if ($line) { ($line -split '","')[0].Trim('"') } else { $null }
}

# A process whose DACL denies PROCESS_TERMINATE cannot be killed by its own
# user, like an app window started elevated: Tauri's check then fails to
# close it and aborts, the same Abort a Cancel in its prompt takes.
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class FidimHooksTestAcl {
  [DllImport("advapi32.dll", SetLastError = true)]
  static extern bool SetKernelObjectSecurity(IntPtr handle, uint info, byte[] sd);
  public static void Set(IntPtr handle, string sddl) {
    var sd = new System.Security.AccessControl.RawSecurityDescriptor(sddl);
    var bytes = new byte[sd.BinaryLength];
    sd.GetBinaryForm(bytes, 0);
    if (!SetKernelObjectSecurity(handle, 4 /* DACL_SECURITY_INFORMATION */, bytes))
      throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
  }
}
"@
$userSid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
function Set-Killable($p, [bool]$killable) {
  # Through the handle Start-Process kept, which has every right.
  $deny = if ($killable) { '' } else { '(D;;0x1;;;WD)' }
  [FidimHooksTestAcl]::Set($p.Handle, "D:$deny(A;;0x1FFFFF;;;$userSid)")
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
  # The NSIS program: Tauri's Install section (SetOutPath, PREINSTALL, the
  # app check, the file copies, POSTINSTALL) or its Uninstall section
  # (PREUNINSTALL, the app check, the deletes), with small files standing
  # in for the copies.
  $nsi = Join-Path $tmp 'hooks-test.nsi'
  @"
Unicode true
RequestExecutionLevel user
SilentInstall silent
OutFile "hooks-test.exe"
!define INSTALLMODE "currentUser"
!define PRODUCTNAME "Llama FIDIM hooks test"
!define VERSION "0.0.0"
!define MAINBINARYNAME "$guiName"
Var PassiveMode
Var OLD
Var LNK
Var PHASE
!define FIDIM_OLD_INSTALL_DIR "`$OLD"
!define FIDIM_OLD_SHORTCUT "`$LNK"
!addplugindir "$plugins"
!include LogicLib.nsh
!include FileFunc.nsh
!include x64.nsh
LoadLanguageFile "`${NSISDIR}\Contrib\Language files\English.nlf"
!include "$utils"
!include "$english"
!include "Win\COM.nsh"
!include "Win\Propkey.nsh"
!include "$hooks"
!macro NEW_FILE NAME
  FileOpen `$0 "`$INSTDIR\`${NAME}" w
  FileWrite `$0 "new"
  FileClose `$0
!macroend
Section
  StrCpy `$PassiveMode 0
  `${GetParameters} `$0
  `${GetOptions} `$0 "/INST=" `$INSTDIR
  `${GetOptions} `$0 "/OLD=" `$OLD
  `${GetOptions} `$0 "/LNK=" `$LNK
  `${GetOptions} `$0 "/PHASE=" `$PHASE
  `${If} `$PHASE == "install"
    SetOutPath `$INSTDIR
    !insertmacro NSIS_HOOK_PREINSTALL
    !insertmacro CheckIfAppIsRunning "`${MAINBINARYNAME}.exe" "`${PRODUCTNAME}"
    !insertmacro NEW_FILE "llama-fidim.exe"
    !insertmacro NEW_FILE "fidim-dg.exe"
    !insertmacro NEW_FILE "fidim.exe"
    !insertmacro NEW_FILE "uninstall.exe"
    !insertmacro NSIS_HOOK_POSTINSTALL
  `${Else}
    !insertmacro NSIS_HOOK_PREUNINSTALL
    !insertmacro CheckIfAppIsRunning "`${MAINBINARYNAME}.exe" "`${PRODUCTNAME}"
    Delete "`$INSTDIR\llama-fidim.exe"
    Delete "`$INSTDIR\fidim-dg.exe"
    Delete "`$INSTDIR\fidim.exe"
    Delete "`$INSTDIR\uninstall.exe"
    RMDir "`$INSTDIR"
  `${EndIf}
SectionEnd
"@ | Set-Content $nsi -Encoding UTF8
  Push-Location $tmp
  try {
    $out = & $Makensis -V2 -INPUTCHARSET UTF8 $nsi 2>&1
    if ($LASTEXITCODE -ne 0) { $out | Write-Host; throw "makensis failed ($LASTEXITCODE)" }
  } finally { Pop-Location }
  $exe = Join-Path $tmp 'hooks-test.exe'
  # Exit code 2 is an install the script aborted.
  function Run-Hooks([string]$phase, [int]$exit = 0, [string]$at = $inst) {
    $p = Start-Process -FilePath $exe -Wait -PassThru `
      -ArgumentList @("/PHASE=$phase", "/INST=`"$at`"", "/OLD=`"$old`"", "/LNK=`"$lnk`"")
    if ($p.ExitCode -ne $exit) { throw "hooks-test.exe /PHASE=$phase exited $($p.ExitCode), expected $exit" }
  }
  function Aside([string]$dir, [string]$name) {
    @(Get-ChildItem $dir -Filter "$name.old-*" -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match ('^' + [regex]::Escape($name) + '\.old-\d{14}(-\d+)?$') -and $_.Name -notmatch '-19990101000000$' })
  }
  # A file the install phase wrote, standing in for one the installer copies.
  function Is-New([string]$path) {
    (Test-Path $path) -and ((Get-Content $path -Raw -ErrorAction SilentlyContinue) -eq 'new')
  }
  function New-Lnk([string]$target) {
    $sc = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)
    $sc.TargetPath = $target
    $sc.WorkingDirectory = Split-Path $target
    $sc.Save()
  }
  # Every file in the two folders with its size and time, and the shortcut.
  function Snapshot {
    $items = @(Get-ChildItem $inst, $old -Force -ErrorAction SilentlyContinue |
      ForEach-Object { '{0}|{1}|{2}' -f $_.FullName, $_.Length, $_.LastWriteTimeUtc.Ticks })
    ($items + "lnk=$(Test-Path $lnk)") -join "`n"
  }

  Write-Host "== 1. an app window that cannot be closed stops the install before anything changes"
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
  $gui = Start-Helper (Join-Path $tmp "gui\$guiName.exe")               # the app
  Set-Killable $gui $false
  $before = Snapshot
  Run-Hooks 'install' 2
  Check ((Snapshot) -eq $before) 'install: nothing in either folder or the Start Menu changed'
  Run-Hooks 'uninstall' 2
  Check ((Snapshot) -eq $before) 'uninstall: nothing changed either'
  $fresh = Join-Path $tmp 'Fresh FIDIM'
  Run-Hooks 'install' 2 $fresh
  Check (-not (Test-Path $fresh)) 'a first install leaves no empty folder behind'
  Check (-not $gui.HasExited) 'the app is still running'
  Set-Killable $gui $true

  Write-Host "== 2. install over running helpers, retiring an old script install"
  Run-Hooks 'install'
  Check ($gui.WaitForExit(5000)) 'the app was closed'
  Check (Is-New (Join-Path $inst 'fidim.exe')) 'a running fidim.exe made room for the new one'
  Check (Is-New (Join-Path $inst 'fidim-dg.exe')) 'a running fidim-dg.exe made room for the new one'
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

  Write-Host "== 3. a shortcut that points elsewhere is left alone; same-second renames do not collide"
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
  Check (Is-New (Join-Path $inst 'fidim.exe')) 'the second running fidim.exe made room for the new one'
  Check (@(Get-ChildItem $inst -Filter 'fidim.exe.old-*' | Where-Object { $_.Name -match '^fidim\.exe\.old-\d{14}-\d+$' }).Count -eq 1) 'it took a -2 style name instead of an existing one'
  Check (-not $e.HasExited) 'and it keeps running'

  Write-Host "== 4. uninstall moves running helpers aside too"
  $g = Start-Helper (Join-Path $inst 'fidim.exe')
  Run-Hooks 'uninstall'
  Check (-not (Test-Path (Join-Path $inst 'fidim.exe'))) 'running fidim.exe left its name'
  Check (-not $g.HasExited) 'and it keeps running'

  Write-Host "== 5. once everything has exited, the next install cleans up"
  foreach ($h in $handles) { $h.Dispose() }
  $handles = @()
  foreach ($p in $procs) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
  foreach ($p in $procs) { $p.WaitForExit(5000) | Out-Null }
  Run-Hooks 'install'
  Check (@(Get-ChildItem $inst -Filter '*.old-*').Count -eq 0) 'no renamed copies left in the install folder'
  Check (-not (Test-Path $old)) 'the old folder is gone'
} finally {
  foreach ($h in $handles) { $h.Dispose() }
  foreach ($p in $procs) {
    if (-not $p.HasExited) { try { Set-Killable $p $true } catch { } }
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
  }
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
