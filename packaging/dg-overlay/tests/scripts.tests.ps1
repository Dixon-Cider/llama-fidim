<#
.SYNOPSIS
  Self-tests for the overlay scripts that need no network, no Visual Studio
  and no build: re-applying an edited patch, the BoringSSL license lookup,
  and the gate stamp package.ps1 requires. Needs git and Windows' tar.exe.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tests\scripts.tests.ps1
#>
$ErrorActionPreference = 'Stop'
$scripts = Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts'
. (Join-Path $scripts 'common.ps1')

$root = Join-Path ([System.IO.Path]::GetTempPath()) "fidim-dgo-tests-$PID"
if (Test-Path -LiteralPath $root) { Remove-Item -Recurse -Force -LiteralPath $root }
New-Item -ItemType Directory -Force $root | Out-Null
$failures = New-Object System.Collections.Generic.List[string]

function Assert([bool]$Cond, [string]$What) { if (-not $Cond) { throw "assertion failed: $What" } }
function Assert-Throws([scriptblock]$Block, [string]$Like) {
    try { & $Block } catch {
        if ($_.Exception.Message -notlike "*$Like*") { throw "threw '$($_.Exception.Message)', expected *$Like*" }
        return
    }
    throw "did not throw (expected *$Like*)"
}
function Test-Case([string]$Name, [scriptblock]$Block) {
    try { & $Block; Write-Host "ok   $Name" }
    catch { $failures.Add("$Name`: $($_.Exception.Message)"); Write-Host "FAIL $Name`: $($_.Exception.Message)" -ForegroundColor Red }
}
function Write-Lf([string]$Path, [string]$Text) {
    New-Item -ItemType Directory -Force (Split-Path -Parent $Path) | Out-Null
    [System.IO.File]::WriteAllText($Path, ($Text -replace "`r`n", "`n"), (New-Object System.Text.UTF8Encoding($false)))
}
# Run a script in its own PowerShell, as build-local.ps1's callers do; returns its output and exit code.
function Invoke-Script([string]$Script, [string[]]$Arguments) {
    $ps = (Get-Process -Id $PID).Path
    $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try { $out = & $ps -NoProfile -ExecutionPolicy Bypass -File (Join-Path $scripts $Script) @Arguments 2>&1 | Out-String }
    finally { $ErrorActionPreference = $eap }
    return @{ code = $LASTEXITCODE; text = $out }
}

# ---- a miniature release: its source tarball, base.json and two revisions of a patch
$tag = 'b11030-mix-5ff778e'
$tree = Join-Path $root 'tree\llama.cpp-src'
Write-Lf (Join-Path $tree 'CMakeLists.txt') "project(llama)`n"
Write-Lf (Join-Path $tree 'include\llama.h') "#pragma once`n// test header`nint llama_one(void);`n"
Write-Lf (Join-Path $tree 'cmake\build-info.cmake') "set(BUILD_NUMBER 11030)`n# Unsloth fingerprint`nset(BUILD_TARGET `"`${BUILD_TARGET} (Compiled by the Unsloth team)`")`n"
$work = Join-Path $root 'w'
$dl = Join-Path $work 'dl'
New-Item -ItemType Directory -Force $dl | Out-Null
$asset = 'llama.cpp-source-commit-6ba30d05b140ebb0baeded27d7d9b843c5b71ff1.tar.gz'
$tarball = Join-Path $dl $asset
& (Join-Path $env:SystemRoot 'System32\tar.exe') -czf $tarball -C (Split-Path -Parent $tree) (Split-Path -Leaf $tree)
if ($LASTEXITCODE -ne 0) { throw "tar failed" }
$srcSha = Get-Sha256 $tarball
function Write-BaseJson([string]$Tag) {
    Write-Utf8NoBom (Join-Path $work 'base.json') (ConvertTo-PrettyJson ([ordered]@{
        repo = $BaseRepo; release_tag = $Tag; source_commit = '6ba30d05b140ebb0baeded27d7d9b843c5b71ff1'
        source_asset = $asset; source_sha256 = $srcSha; upstream_tag = 'b11030'; zips = @{}
    }))
}
$body = @"
diff --git a/include/llama.h b/include/llama.h
--- a/include/llama.h
+++ b/include/llama.h
@@ -1,3 +1,4 @@
 #pragma once
 // test header
 int llama_one(void);
+int llama_diffusion_fa_turn_bytes(void);

"@
$patch1 = Join-Path $root 'dgpatch5.diff'
Write-Lf $patch1 ("dgpatch5, first revision`n`n" + $body)
# Only the preamble differs: it applies to the release's source exactly as
# the first does, and not at all to a tree that already carries the first.
$patch2 = Join-Path $root 'dgpatch5-r2.diff'
Write-Lf $patch2 ("dgpatch5, second revision`n`n" + $body)
$src = Join-Path $work 'src'
function Get-Fingerprints {
    @(Get-Content -LiteralPath (Join-Path $src 'cmake\build-info.cmake') | Where-Object { $_ -like 'set(BUILD_TARGET*Llama FIDIM*' })
}

Test-Case 'the release source unpacks and records its stamp' {
    Write-BaseJson $tag
    Write-Lf (Join-Path $work 'patch.stamp') 'left over'
    Expand-BaseSource $work $tarball $srcSha
    Assert (Test-Path -LiteralPath (Join-Path $src 'include\llama.h')) 'src\include\llama.h'
    Assert ((Get-Content -Raw -LiteralPath (Join-Path $work 'src.stamp')).Trim() -eq $srcSha) 'src.stamp names the tarball'
    Assert (-not (Test-Path -LiteralPath (Join-Path $work 'patch.stamp'))) 'an unpacked tree carries no patch'
}

Test-Case 'apply-patch applies, and a rerun with the same patch and tag does nothing' {
    $r = Invoke-Script 'apply-patch.ps1' @('-WorkDir', $work, '-Patch', $patch1)
    Assert ($r.code -eq 0) "exit $($r.code): $($r.text)"
    Assert ((Get-Content -Raw -LiteralPath (Join-Path $work 'patch.stamp')).Trim() -eq "$(Get-Sha256 $patch1) $tag") 'patch.stamp'
    $f = @(Get-Fingerprints)
    Assert ($f.Count -eq 1 -and $f[0] -like "*overlay on Unsloth $tag*") "one fingerprint: $($f -join ' | ')"
    $r = Invoke-Script 'apply-patch.ps1' @('-WorkDir', $work, '-Patch', $patch1)
    Assert ($r.code -eq 0 -and $r.text -like '*already applied*') "rerun: $($r.text)"
}

Test-Case 'an edited patch is applied to the release source, not to the patched tree' {
    # The reported failure: git apply --check of the new revision against the
    # tree the first one patched fails, blaming the release.
    Set-GitCeiling $src
    Push-Location -LiteralPath $src
    try {
        $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        git -c core.autocrlf=false apply --check $patch2 2>&1 | Out-Null
        $code = $LASTEXITCODE
        $ErrorActionPreference = $eap
    } finally { Pop-Location }
    Assert ($code -ne 0) 'the second revision does not apply on top of the first'
    $r = Invoke-Script 'apply-patch.ps1' @('-WorkDir', $work, '-Patch', $patch2)
    Assert ($r.code -eq 0) "exit $($r.code): $($r.text)"
    Assert ($r.text -like '*unpacking the release*') "re-unpacked: $($r.text)"
    Assert ((Get-Content -Raw -LiteralPath (Join-Path $work 'patch.stamp')).Trim() -eq "$(Get-Sha256 $patch2) $tag") 'patch.stamp names the new revision'
    $decls = @(Select-String -LiteralPath (Join-Path $src 'include\llama.h') -SimpleMatch 'llama_diffusion_fa_turn_bytes')
    Assert ($decls.Count -eq 1) "patched once, not twice ($($decls.Count))"
    Assert (@(Get-Fingerprints).Count -eq 1) 'one fingerprint'
}

Test-Case 'the same source under another tag is unpacked again for its fingerprint' {
    $other = 'b11031-mix-5ff778e'
    Write-BaseJson $other
    try {
        $r = Invoke-Script 'apply-patch.ps1' @('-WorkDir', $work, '-Patch', $patch2)
        Assert ($r.code -eq 0) "exit $($r.code): $($r.text)"
        $f = @(Get-Fingerprints)
        Assert ($f.Count -eq 1 -and $f[0] -like "*overlay on Unsloth $other*") "fingerprint: $($f -join ' | ')"
    } finally { Write-BaseJson $tag }
}

Test-Case 'an apply that did not finish is started again from the release source' {
    Write-Lf (Join-Path $work 'patch.stamp') "applying $(Get-Sha256 $patch1) $tag"
    $r = Invoke-Script 'apply-patch.ps1' @('-WorkDir', $work, '-Patch', $patch1)
    Assert ($r.code -eq 0 -and $r.text -like '*unpacking the release*') "exit $($r.code): $($r.text)"
    Assert ((Get-Content -Raw -LiteralPath (Join-Path $work 'patch.stamp')).Trim() -eq "$(Get-Sha256 $patch1) $tag") 'patch.stamp'
}

Test-Case 'a patch that does not fit the release source is reported as such' {
    $bad = Join-Path $root 'bad.diff'
    Write-Lf $bad ($body -replace 'int llama_one\(void\);', 'int llama_two(void);')
    $r = Invoke-Script 'apply-patch.ps1' @('-WorkDir', $work, '-Patch', $bad)
    Assert ($r.code -ne 0 -and $r.text -like '*no longer fits*') "exit $($r.code): $($r.text)"
    # The release source was restored first and is untouched: the next run applies cleanly.
    $r = Invoke-Script 'apply-patch.ps1' @('-WorkDir', $work, '-Patch', $patch1)
    Assert ($r.code -eq 0) "exit $($r.code): $($r.text)"
}

Test-Case 'the BoringSSL license is found where each mode builds it' {
    $build = Join-Path $root 'build'
    Assert ($null -eq (Get-BoringSslSourceDir 'off' $build)) 'off: none'
    Assert ((Get-BoringSslSourceDir 'fetch' $build) -eq (Join-Path $build '_deps\boringssl-src')) 'fetch: the FetchContent clone'
    $folder = Join-Path $root 'boringssl'
    New-Item -ItemType Directory -Force $folder | Out-Null
    Assert ((Get-BoringSslSourceDir $folder $build) -eq (Resolve-Path -LiteralPath $folder).Path) 'a folder: that folder, not build\_deps'
}

Test-Case 'the notices name BoringSSL by the license text it ships' {
    $apache = Join-Path $root 'LICENSE-apache'
    Write-Lf $apache "`n                                 Apache License`n                           Version 2.0, January 2004`n                        http://www.apache.org/licenses/`n"
    Assert ((Get-BoringSslLicenseName $apache) -eq 'Apache License 2.0') 'Apache 2.0'
    $old = Join-Path $root 'LICENSE-openssl'
    Write-Lf $old "BoringSSL is a fork of OpenSSL. As such, large parts of it fall under OpenSSL licensing.`n"
    Assert-Throws { Get-BoringSslLicenseName $old } 'update THIRD-PARTY-NOTICES'
    $tpl = Get-Content -Raw -LiteralPath (Join-Path (Split-Path -Parent $PSScriptRoot) 'THIRD-PARTY-NOTICES.md')
    Assert ($tpl -notmatch 'ISC') 'the template no longer says ISC'
    Assert ($tpl -match '\| BoringSSL \| \{\{BORINGSSL\}\} \| \{\{BORINGSSL_TEXT\}\} \|') 'the template fills the license in'
}

Test-Case 'package.ps1 refuses an overlay the gate has not passed' {
    $ov = Join-Path $work 'overlay'
    foreach ($n in 'llama.dll', 'llama-server.exe', 'llama-diffusion-gemma-visual-server.exe') { Write-Lf (Join-Path $ov $n) 'x' }
    Write-Utf8NoBom (Join-Path $work 'ggml-imports.json') '{}'
    Write-Utf8NoBom (Join-Path $work 'build-info.json') '{}'
    # A failed gate leaves ggml-imports.json behind no longer; even if one is
    # there, no gate.pass means no package.
    $out = Join-Path $root 'out'
    $r = Invoke-Script 'package.ps1' @('-WorkDir', $work, '-OutDir', $out, '-Patch', $patch1)
    Assert ($r.code -ne 0 -and $r.text -like '*gate has not passed*') "exit $($r.code): $($r.text)"
    Assert (-not (Test-Path -LiteralPath $out)) 'nothing packaged'
    # Passed on these files: accepted (names, not bytes: signing changes bytes).
    Write-GatePass $work @('LLAMA.DLL', 'llama-server.exe', 'llama-diffusion-gemma-visual-server.exe') 'app.zip' $false
    Assert-GatePassed $work
    Write-Lf (Join-Path $ov 'llama.dll') 'signed bytes'
    Assert-GatePassed $work
    # Files added or removed since: refused.
    Write-Lf (Join-Path $ov 'llama-extra-impl.dll') 'x'
    Assert-Throws { Assert-GatePassed $work } 'changed since the gate passed'
    Remove-Item -LiteralPath (Join-Path $ov 'llama-extra-impl.dll')
    Remove-Item -LiteralPath (Join-Path $ov 'llama.dll')
    Assert-Throws { Assert-GatePassed $work } 'changed since the gate passed'
}

Remove-Item -Recurse -Force -LiteralPath $root -ErrorAction SilentlyContinue
if ($failures.Count) {
    Write-Host "$($failures.Count) failed" -ForegroundColor Red
    exit 1
}
Write-Host 'all passed'
