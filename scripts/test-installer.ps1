<#
.SYNOPSIS
  Install Llama FIDIM's setup.exe silently for the current user, check what
  it did, upgrade to it from an older version both ways the installer can,
  exercise its hooks against running helpers and a stand-in for an old
  script install, then uninstall it silently and check that too.

.DESCRIPTION
  THIS CHANGES THE CURRENT USER'S REAL SETUP. It installs into
  %LOCALAPPDATA%\Llama FIDIM, closes every running Llama FIDIM window,
  writes Start Menu and desktop shortcuts and an uninstall entry, and
  creates and then retires a stand-in old install in
  %LOCALAPPDATA%\Programs\LlamaFIDIM. The release workflow runs it on a
  fresh GitHub runner. Run it yourself only on a machine you can throw
  away (a VM, Windows Sandbox): it refuses to start without
  -ThrowawayMachine, and when Llama FIDIM is installed there already, by
  the installer or by scripts\install.ps1.
  (scripts\test-installer-hooks.ps1 tests the hooks alone and is safe
  anywhere.)

  The older version is the same installer built again as version 0.0.1
  from the installer.nsi Tauri generated (-NsisDir), so its uninstaller
  carries the same hooks. Upgrades from it are tested both ways they
  happen:
  - silently (setup.exe /S, also /P and Tauri's updater): the new version
    installs over the old one;
  - interactively: the reinstall page's default, "Uninstall before
    installing", runs the installed version's uninstaller in place
    (uninstall.exe _?=<folder>) and waits for it, then installs. NSIS skips
    pages in a silent run, so the test runs that uninstaller the same way
    itself, silently, and then setup.exe /S.

  The helpers it starts are `fidim keepalive` with a one-hour interval and
  FIDIM_HOME in a temporary folder: they sleep, send nothing, and are
  stopped at the end.

  NOTE: ASCII only, deliberately. Windows PowerShell 5.1 reads .ps1 as ANSI
  unless the file has a BOM.

.PARAMETER Setup
  The setup.exe to test.
.PARAMETER Version
  The version it must install, e.g. 0.3.0.
.PARAMETER Binaries
  A folder with fidim.exe, fidim-dg.exe and llama-fidim.exe for the
  stand-in old install (the release zip's folder).
.PARAMETER NsisDir
  The folder where Tauri wrote the installer script setup.exe was built
  from (target\release\nsis\x64), for the older version.
.PARAMETER Makensis
  The makensis.exe Tauri used (Tauri downloads it to
  %LOCALAPPDATA%\tauri\NSIS).
.PARAMETER Signed
  Also require a valid Authenticode signature on setup.exe and on every
  .exe and .dll it installs, uninstall.exe included.
.PARAMETER Signer
  With -Signed: the signing certificate's subject must carry CN=<Signer>.
.EXAMPLE
  scripts\test-installer.ps1 -ThrowawayMachine -Setup .\llama-fidim-v0.3.0-win-x64-setup.exe -Version 0.3.0 -Binaries .\llama-fidim-v0.3.0-win-x64 -NsisDir target\release\nsis\x64
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$Setup,
  [Parameter(Mandatory)][string]$Version,
  [Parameter(Mandatory)][string]$Binaries,
  [Parameter(Mandatory)][string]$NsisDir,
  [string]$Makensis = (Join-Path $env:LOCALAPPDATA 'tauri\NSIS\makensis.exe'),
  [switch]$Signed,
  [string]$Signer,
  [switch]$ThrowawayMachine
)
$ErrorActionPreference = 'Stop'
if (-not $ThrowawayMachine) {
  throw "this installs for the current user and retires an existing install; pass -ThrowawayMachine on a machine you can throw away (see the help)"
}
$Setup = (Resolve-Path $Setup).Path
$Binaries = (Resolve-Path $Binaries).Path
$NsisDir = (Resolve-Path $NsisDir).Path
$OlderVersion = '0.0.1'
$inst = Join-Path $env:LOCALAPPDATA 'Llama FIDIM'
$old = Join-Path $env:LOCALAPPDATA 'Programs\LlamaFIDIM'
$uninstKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Llama FIDIM'
# Where the installer records its folder; the reinstall page reads it.
$productKey = 'HKCU:\Software\Dixon-Cider\Llama FIDIM'
$startLnk = Join-Path ([Environment]::GetFolderPath('Programs')) 'Llama FIDIM.lnk'
$deskLnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Llama FIDIM.lnk'
foreach ($p in $inst, $old, $uninstKey) {
  if (Test-Path $p) { throw "$p exists: Llama FIDIM is installed here already; use a clean machine" }
}
foreach ($f in $Makensis, (Join-Path $NsisDir 'installer.nsi')) {
  if (-not (Test-Path $f)) { throw "missing $f" }
}

$failures = New-Object System.Collections.Generic.List[string]
function Check([bool]$ok, [string]$what) {
  if ($ok) { Write-Host "  ok   $what" } else { Write-Host "  FAIL $what" -ForegroundColor Red; $failures.Add($what) }
}
function Lnk-Target([string]$lnk) {
  if (-not (Test-Path $lnk)) { return $null }
  (New-Object -ComObject WScript.Shell).CreateShortcut($lnk).TargetPath
}
function Run-Setup([string]$exe = $Setup) {
  $p = Start-Process -FilePath $exe -ArgumentList '/S' -Wait -PassThru
  Check ($p.ExitCode -eq 0) "$(Split-Path -Leaf $exe) /S exited 0 (got $($p.ExitCode))"
}
function Aside([string]$dir, [string]$name) {
  @(Get-ChildItem $dir -Filter "$name.old-*" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match ('^' + [regex]::Escape($name) + '\.old-\d{14}(-\d+)?$') })
}
function Installed-Version {
  $key = Get-ItemProperty $uninstKey -ErrorAction SilentlyContinue
  if ($key) { $key.DisplayVersion } else { $null }
}
function Signature-Ok([string]$file) {
  $s = Get-AuthenticodeSignature $file
  $ok = $s.Status -eq 'Valid'
  if ($ok -and $Signer) { $ok = $s.SignerCertificate.Subject -match ('(^|, )CN="?' + [regex]::Escape($Signer) + '"?(,|$)') }
  if (-not $ok) { Write-Host "       $file : $($s.Status) $($s.SignerCertificate.Subject)" }
  $ok
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("fidim-setup-test-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force $tmp | Out-Null
$env:FIDIM_HOME = $tmp
$procs = @()
function Start-Helper([string]$exe) {
  $p = Start-Process -FilePath $exe -WindowStyle Hidden -PassThru `
    -ArgumentList @('keepalive', '--port', '9', '--interval', '3600', '--server-pid', "$PID")
  $script:procs += $p
  Start-Sleep -Milliseconds 500
  $p
}

# The same installer as version $OlderVersion: Tauri's installer script
# with a lower version, compiled again with nothing to sign.
function Build-OlderSetup([string]$out) {
  $text = [IO.File]::ReadAllText((Join-Path $NsisDir 'installer.nsi'))
  $edits = [ordered]@{
    '(?m)^!define VERSION "[^"\r\n]*"'             = "!define VERSION `"$OlderVersion`""
    '(?m)^!define VERSIONWITHBUILD "[^"\r\n]*"'    = "!define VERSIONWITHBUILD `"$OlderVersion.0`""
    '(?m)^!define OUTFILE "[^"\r\n]*"'             = "!define OUTFILE `"$out`""
    '(?m)^!define UNINSTALLERSIGNCOMMAND [^\r\n]*' = '!define UNINSTALLERSIGNCOMMAND ""'
  }
  foreach ($re in $edits.Keys) {
    if (-not [regex]::IsMatch($text, $re)) { throw "installer.nsi has no line matching $re" }
    $text = [regex]::Replace($text, $re, $edits[$re].Replace('$', '$$'))
  }
  # Beside Tauri's script: NSIS looks for its relative includes there.
  $nsi = Join-Path $NsisDir 'installer-older.nsi'
  [IO.File]::WriteAllText($nsi, $text, (New-Object System.Text.UTF8Encoding($true)))
  try {
    # makensis reports errors on stderr; read them, do not throw on them.
    $ErrorActionPreference = 'Continue'
    $log = & $Makensis -V2 -INPUTCHARSET UTF8 -OUTPUTCHARSET UTF8 $nsi 2>&1
    $code = $LASTEXITCODE
  } finally { Remove-Item $nsi -ErrorAction SilentlyContinue }
  if ($code -ne 0 -or -not (Test-Path $out)) { $log | Write-Host; throw "makensis failed for the $OlderVersion installer ($code)" }
}

try {
  if ($Signed) {
    Write-Host "== signature of the installer itself"
    Check (Signature-Ok $Setup) "setup.exe is signed$(if ($Signer) { " by $Signer" })"
  }

  $olderSetup = Join-Path $tmp "llama-fidim-v$OlderVersion-setup.exe"
  Build-OlderSetup $olderSetup
  Check ((Get-Item $olderSetup).VersionInfo.ProductVersion -eq $OlderVersion) "built the same installer as version $OlderVersion"

  Write-Host "== 1. first install (version $OlderVersion), over a running helper of an old script install"
  New-Item -ItemType Directory -Force $old | Out-Null
  foreach ($exe in 'fidim.exe', 'fidim-dg.exe', 'llama-fidim.exe') { Copy-Item (Join-Path $Binaries $exe) $old }
  # The Start Menu entry exactly as install.ps1 wrote it.
  $sc = (New-Object -ComObject WScript.Shell).CreateShortcut($startLnk)
  $sc.TargetPath = Join-Path $old 'llama-fidim.exe'
  $sc.WorkingDirectory = $old
  $sc.Save()
  $oldHelper = Start-Helper (Join-Path $old 'fidim.exe')
  Run-Setup $olderSetup
  foreach ($exe in 'llama-fidim.exe', 'fidim.exe', 'fidim-dg.exe', 'uninstall.exe') {
    Check (Test-Path (Join-Path $inst $exe)) "installed $exe"
  }
  Check ((Installed-Version) -eq $OlderVersion) "DisplayVersion is $OlderVersion (got '$(Installed-Version)')"
  $want = Join-Path $inst 'llama-fidim.exe'
  Check ((Lnk-Target $startLnk) -eq $want) "the old Start Menu entry was replaced by one that starts $want (got '$(Lnk-Target $startLnk)')"
  Check (-not $oldHelper.HasExited) 'the old install''s helper keeps running'
  Check ((Aside $old 'fidim.exe').Count -eq 1) 'its fidim.exe was moved aside'
  Check (-not (Test-Path (Join-Path $old 'llama-fidim.exe')) -and -not (Test-Path (Join-Path $old 'fidim-dg.exe'))) 'the old install''s other files are gone'

  Write-Host "== 2. silent upgrade to $Version over a running helper; the old install's helper has exited"
  Stop-Process -Id $oldHelper.Id -Force
  $oldHelper.WaitForExit(5000) | Out-Null
  $helper = Start-Helper (Join-Path $inst 'fidim.exe')
  $aside = (Aside $inst 'fidim.exe').Count
  Run-Setup
  foreach ($exe in 'llama-fidim.exe', 'fidim.exe', 'fidim-dg.exe', 'uninstall.exe') {
    Check (Test-Path (Join-Path $inst $exe)) "installed $exe"
  }
  $key = Get-ItemProperty $uninstKey -ErrorAction SilentlyContinue
  Check ($null -ne $key) 'uninstall entry written'
  if ($key) {
    Check ($key.DisplayName -eq 'Llama FIDIM') "DisplayName is Llama FIDIM (got '$($key.DisplayName)')"
    Check ($key.DisplayVersion -eq $Version) "DisplayVersion is $Version (got '$($key.DisplayVersion)')"
    Check ($key.Publisher -eq 'Dixon-Cider') "Publisher is Dixon-Cider (got '$($key.Publisher)')"
  }
  Check ((Lnk-Target $startLnk) -eq $want) "Start Menu entry starts $want (got '$(Lnk-Target $startLnk)')"
  Check ((Lnk-Target $deskLnk) -eq $want) "desktop shortcut starts $want"
  $v = & (Join-Path $inst 'fidim.exe') --version
  Check ($LASTEXITCODE -eq 0 -and $v -match ('^fidim ' + [regex]::Escape($Version) + '\b')) "installed fidim --version reports $Version (got '$v')"
  & (Join-Path $inst 'fidim-dg.exe') --help | Out-Null
  Check ($LASTEXITCODE -eq 0) 'installed fidim-dg --help exits 0'
  foreach ($exe in 'fidim.exe', 'fidim-dg.exe', 'llama-fidim.exe') {
    $vi = (Get-Item (Join-Path $inst $exe)).VersionInfo
    Check ($vi.ProductName -eq 'Llama FIDIM' -and $vi.CompanyName -eq 'Dixon-Cider' -and $vi.ProductVersion -like "$Version*") `
      "$exe version resource: '$($vi.ProductName)' by '$($vi.CompanyName)' $($vi.ProductVersion)"
  }
  if ($Signed) {
    foreach ($f in Get-ChildItem $inst -Recurse -Include *.exe, *.dll) { Check (Signature-Ok $f.FullName) "signed: $($f.Name)" }
  }
  Check ((Aside $inst 'fidim.exe').Count -eq $aside + 1) 'the running fidim.exe was moved aside'
  Check (-not $helper.HasExited) 'and keeps running'
  Check (-not (Test-Path $old)) 'the old install folder is gone'

  Write-Host "== 3. upgrade the interactive way: the installed $OlderVersion's uninstaller first, with a helper running"
  Run-Setup $olderSetup
  Check ((Installed-Version) -eq $OlderVersion) "back on $OlderVersion (got '$(Installed-Version)')"
  $helper2 = Start-Helper (Join-Path $inst 'fidim.exe')
  $aside = (Aside $inst 'fidim.exe').Count
  # What the reinstall page runs: the registered uninstall command, with
  # the registered folder after _?= (unquoted, last), which runs it in
  # place so that it can be waited for.
  $uninstaller = (Get-ItemProperty $uninstKey).UninstallString.Trim('"')
  $folder = (Get-Item $productKey).GetValue('')
  Check ($folder -eq $inst) "the registered install folder is $inst (got '$folder')"
  $p = Start-Process -FilePath $uninstaller -ArgumentList "/S _?=$folder" -Wait -PassThru
  # The reinstall page reports "Unable to uninstall!" and stops unless
  # both hold.
  Check ($p.ExitCode -eq 0) "the $OlderVersion uninstaller exited 0 (got $($p.ExitCode))"
  Check (-not (Test-Path (Join-Path $inst 'llama-fidim.exe'))) 'it removed llama-fidim.exe'
  Check ((Aside $inst 'fidim.exe').Count -eq $aside + 1) 'it moved the running fidim.exe aside'
  Check (-not $helper2.HasExited) 'which keeps running'
  Run-Setup
  Check ((Installed-Version) -eq $Version) "DisplayVersion is $Version (got '$(Installed-Version)')"
  foreach ($exe in 'llama-fidim.exe', 'fidim.exe', 'fidim-dg.exe', 'uninstall.exe') {
    Check (Test-Path (Join-Path $inst $exe)) "installed $exe"
  }
  $v = & (Join-Path $inst 'fidim.exe') --version
  Check ($LASTEXITCODE -eq 0 -and $v -match ('^fidim ' + [regex]::Escape($Version) + '\b')) "installed fidim --version reports $Version (got '$v')"
  Check ((Lnk-Target $startLnk) -eq $want) 'the Start Menu entry is back'
  Check (-not $helper2.HasExited) 'the helper survived the upgrade'

  Write-Host "== 4. uninstall with a helper running"
  $helper3 = Start-Helper (Join-Path $inst 'fidim.exe')
  # The uninstaller copies itself to %TEMP%, starts the copy and returns at
  # once, so its exit code says nothing; wait for the copy's last step, the
  # uninstall entry going away, and judge by what is left.
  Start-Process -FilePath (Join-Path $inst 'uninstall.exe') -ArgumentList '/S' -Wait
  $deadline = (Get-Date).AddSeconds(120)
  while ((Test-Path $uninstKey) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
  Start-Sleep -Seconds 2
  Check (-not (Test-Path $uninstKey)) 'uninstall entry removed'
  foreach ($exe in 'llama-fidim.exe', 'fidim.exe', 'fidim-dg.exe', 'uninstall.exe') {
    Check (-not (Test-Path (Join-Path $inst $exe))) "$exe removed"
  }
  Check (-not $helper3.HasExited) 'the running helper survived the uninstall'
  Check (-not (Test-Path $startLnk)) 'Start Menu entry removed'
  Check (-not (Test-Path $deskLnk)) 'desktop shortcut removed'
} finally {
  foreach ($p in $procs) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
  foreach ($p in $procs) { $p.WaitForExit(5000) | Out-Null }
  # What the helpers kept alive: renamed copies in both folders.
  foreach ($d in $inst, $old) { Remove-Item $d -Recurse -Force -ErrorAction SilentlyContinue }
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item Env:\FIDIM_HOME -ErrorAction SilentlyContinue
}

if ($failures.Count) {
  Write-Host "$($failures.Count) installer check(s) failed" -ForegroundColor Red
  exit 1
}
Write-Host "installer checks passed" -ForegroundColor Green
exit 0
