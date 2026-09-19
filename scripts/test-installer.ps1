<#
.SYNOPSIS
  Install Llama FIDIM's setup.exe silently for the current user, check what
  it did, exercise its hooks against running helpers and a stand-in for an
  old script install, then uninstall it silently and check that too.

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
.PARAMETER Signed
  Also require a valid Authenticode signature on setup.exe and on every
  .exe and .dll it installs, uninstall.exe included.
.PARAMETER Signer
  With -Signed: the signing certificate's subject must carry CN=<Signer>.
.EXAMPLE
  scripts\test-installer.ps1 -ThrowawayMachine -Setup .\llama-fidim-v0.3.0-win-x64-setup.exe -Version 0.3.0 -Binaries .\llama-fidim-v0.3.0-win-x64
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$Setup,
  [Parameter(Mandatory)][string]$Version,
  [Parameter(Mandatory)][string]$Binaries,
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
$inst = Join-Path $env:LOCALAPPDATA 'Llama FIDIM'
$old = Join-Path $env:LOCALAPPDATA 'Programs\LlamaFIDIM'
$uninstKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Llama FIDIM'
$startLnk = Join-Path ([Environment]::GetFolderPath('Programs')) 'Llama FIDIM.lnk'
$deskLnk = Join-Path ([Environment]::GetFolderPath('Desktop')) 'Llama FIDIM.lnk'
foreach ($p in $inst, $old, $uninstKey) {
  if (Test-Path $p) { throw "$p exists: Llama FIDIM is installed here already; use a clean machine" }
}

$failures = New-Object System.Collections.Generic.List[string]
function Check([bool]$ok, [string]$what) {
  if ($ok) { Write-Host "  ok   $what" } else { Write-Host "  FAIL $what" -ForegroundColor Red; $failures.Add($what) }
}
function Lnk-Target([string]$lnk) {
  if (-not (Test-Path $lnk)) { return $null }
  (New-Object -ComObject WScript.Shell).CreateShortcut($lnk).TargetPath
}
function Run-Setup {
  $p = Start-Process -FilePath $Setup -ArgumentList '/S' -Wait -PassThru
  Check ($p.ExitCode -eq 0) "setup.exe /S exited 0 (got $($p.ExitCode))"
}
function Aside([string]$dir, [string]$name) {
  @(Get-ChildItem $dir -Filter "$name.old-*" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match ('^' + [regex]::Escape($name) + '\.old-\d{14}(-\d+)?$') })
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

try {
  if ($Signed) {
    Write-Host "== signature of the installer itself"
    Check (Signature-Ok $Setup) "setup.exe is signed$(if ($Signer) { " by $Signer" })"
  }

  Write-Host "== 1. first install, over a running helper of an old script install"
  New-Item -ItemType Directory -Force $old | Out-Null
  foreach ($exe in 'fidim.exe', 'fidim-dg.exe', 'llama-fidim.exe') { Copy-Item (Join-Path $Binaries $exe) $old }
  # The Start Menu entry exactly as install.ps1 wrote it.
  $sc = (New-Object -ComObject WScript.Shell).CreateShortcut($startLnk)
  $sc.TargetPath = Join-Path $old 'llama-fidim.exe'
  $sc.WorkingDirectory = $old
  $sc.Save()
  $oldHelper = Start-Helper (Join-Path $old 'fidim.exe')
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
  $want = Join-Path $inst 'llama-fidim.exe'
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
  Check (-not $oldHelper.HasExited) 'the old install''s helper keeps running'
  Check ((Aside $old 'fidim.exe').Count -eq 1) 'its fidim.exe was moved aside'
  Check (-not (Test-Path (Join-Path $old 'llama-fidim.exe')) -and -not (Test-Path (Join-Path $old 'fidim-dg.exe'))) 'the old install''s other files are gone'

  Write-Host "== 2. reinstall over a running helper; the old install's helper has exited"
  Stop-Process -Id $oldHelper.Id -Force
  $oldHelper.WaitForExit(5000) | Out-Null
  $helper = Start-Helper (Join-Path $inst 'fidim.exe')
  Run-Setup
  Check (Test-Path (Join-Path $inst 'fidim.exe')) 'a fresh fidim.exe is in place'
  Check ((Aside $inst 'fidim.exe').Count -eq 1) 'the running one was moved aside'
  Check (-not $helper.HasExited) 'and keeps running'
  Check (-not (Test-Path $old)) 'the old install folder is gone'

  Write-Host "== 3. uninstall with a helper running"
  $helper2 = Start-Helper (Join-Path $inst 'fidim.exe')
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
  Check (-not $helper2.HasExited) 'the running helper survived the uninstall'
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
