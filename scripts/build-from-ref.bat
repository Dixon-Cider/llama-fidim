@echo off
REM Build any llama.cpp git ref from source (a fork's branch, an upstream pull
REM request, a tag, a commit) with HIP, into an output directory. Invoked by
REM `fidim update --source --remote ...` (and the model wizard) as:
REM     build-from-ref.bat <checkout> <remote> <ref> <sha> <out> <gpu_targets>
REM
REM <checkout>     FIDIM's own clone (created here as a partial clone of
REM                upstream when missing). Nobody else's checkout is touched:
REM                the build runs in a detached worktree beside it.
REM <remote> <ref> fetched with `git fetch <remote> +<ref>:refs/fidim/<sha>`;
REM                the build stops unless the fetched commit is exactly <sha>
REM                (40 digits). Give <ref> fully qualified (refs/heads/...,
REM                refs/tags/..., refs/pull/<n>/head) or as the commit itself:
REM                git fetches a bare name that is both a branch and a tag as
REM                the tag.
REM <out>          receives bin\ (executables and DLLs) and source\ (the three
REM                table files FIDIM reads to know what the build supports).
REM <gpu_targets>  comma-separated, e.g. gfx1201 or gfx1100,gfx1201.
REM
REM Toolchain: FIDIM passes what its doctor found in FIDIM_VS, FIDIM_ROCM,
REM FIDIM_CLANG_DIR, FIDIM_CMAKE, FIDIM_NINJA_DIR and optionally
REM FIDIM_VCVARS_VER; run by hand, the newest Visual Studio with C++ tools and
REM HIP_PATH are used. FIDIM_WORKTREE overrides the worktree location.
REM FIDIM_STOP_AFTER=worktree or configure ends the run early (with exit 0,
REM after cleaning up), for tests.
REM
REM The flags configure upstream trees of every layout: before b5269 (May
REM 2025) llama-server, llama-quantize and llama-tokenize were examples, and
REM from b5064 to b7736 LLAMA_CURL defaulted to ON and configure stopped
REM without libcurl. On Windows every tree compiles its HIP code as C++ with
REM --offload-arch from GPU_TARGETS (ROCm's hip-config), so that one flag
REM picks the GPU in every era. Trees before b5872 (July 2025) use hipBLAS
REM types ROCm 7 removed; with a ROCm 7 HIP SDK the build stops after
REM configure (exit 76) instead of failing minutes into compiling.
REM
REM Exit codes: 64 usage, 66 toolchain missing, 67 clone, 68 fetch,
REM 70 ref moved off <sha>, 71 worktree, 72 vcvars, 73 configure, 74 build,
REM 75 copy, 76 the tree predates the HIP SDK's hipBLAS.
setlocal enableextensions
if "%~6"=="" (
  echo usage: %~nx0 ^<checkout^> ^<remote^> ^<ref^> ^<sha^> ^<out^> ^<gpu_targets^>
  exit /b 64
)
set "SRC=%~1"
set "REMOTE=%~2"
set "REF=%~3"
set "SHA=%~4"
set "OUT=%~5"
set "GPUS=%~6"
set "GPUS=%GPUS:,=;%"
if "%FIDIM_UPSTREAM%"=="" set "FIDIM_UPSTREAM=https://github.com/ggml-org/llama.cpp"
if "%FIDIM_WORKTREE%"=="" set "FIDIM_WORKTREE=%SRC%\..\.wt-%SHA:~0,8%"
set "WT=%FIDIM_WORKTREE%"
REM The fetched commit gets a ref of its own: FETCH_HEAD is one file for
REM every fetch into the clone.
set "PIN=refs/fidim/%SHA%"
REM No credential helper and no password prompt: a missing or private
REM repository fails at once instead of opening a sign-in window. The -c
REM options reach the git processes git itself starts (the lazy fetches of
REM a partial clone's file contents).
set GIT=git -c "credential.helper=" -c "core.askPass="
set "GIT_TERMINAL_PROMPT=0"
set "GCM_INTERACTIVE=never"
set "GIT_ASKPASS="
set "SSH_ASKPASS="

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if "%FIDIM_VS%"=="" for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "FIDIM_VS=%%i"
if "%FIDIM_ROCM%"=="" set "FIDIM_ROCM=%HIP_PATH%"
if "%FIDIM_ROCM%"=="" set "FIDIM_ROCM=C:\Program Files\AMD\ROCm\7.1"
if "%FIDIM_ROCM:~-1%"=="\" set "FIDIM_ROCM=%FIDIM_ROCM:~0,-1%"
if "%FIDIM_CLANG_DIR%"=="" set "FIDIM_CLANG_DIR=%FIDIM_ROCM%\bin"
if "%FIDIM_CMAKE%"=="" set "FIDIM_CMAKE=%FIDIM_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if "%FIDIM_NINJA_DIR%"=="" set "FIDIM_NINJA_DIR=%FIDIM_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
set "VCVARS_ARGS="
if not "%FIDIM_VCVARS_VER%"=="" set "VCVARS_ARGS=-vcvars_ver=%FIDIM_VCVARS_VER%"

if not exist "%FIDIM_VS%\VC\Auxiliary\Build\vcvars64.bat" ( echo NO_VCVARS in "%FIDIM_VS%" & exit /b 66 )
if not exist "%FIDIM_CMAKE%" ( echo NO_CMAKE at "%FIDIM_CMAKE%" & exit /b 66 )
if not exist "%FIDIM_CLANG_DIR%\clang++.exe" ( echo NO_HIP_CLANG in "%FIDIM_CLANG_DIR%" & exit /b 66 )

echo === STEP clone %SRC%
REM Cloned beside its place and moved in only once complete: a clone that
REM is killed half way leaves a .git with no commits, and a fetch into that
REM has nothing to offer the server, which then sends every version of
REM every file. (No nested blocks here: cmd.exe loses an `exit /b` code
REM given inside a block that has more commands after it.)
if exist "%SRC%\.git" goto :fetch
if exist "%SRC%" ( echo NOT_A_CLONE: %SRC% exists but is not a git clone; remove it & exit /b 67 )
if exist "%SRC%.part" rmdir /s /q "%SRC%.part"
%GIT% clone --filter=blob:none --no-checkout --single-branch "%FIDIM_UPSTREAM%" "%SRC%.part"
if errorlevel 1 ( echo CLONE_FAILED & rmdir /s /q "%SRC%.part" 2>nul & exit /b 67 )
move "%SRC%.part" "%SRC%" >nul || ( ping -n 3 127.0.0.1 >nul & move "%SRC%.part" "%SRC%" >nul )
if not exist "%SRC%\.git" ( echo CLONE_FAILED: could not move the clone into %SRC% & exit /b 67 )

:fetch
echo === STEP fetch %REMOTE% %REF%
REM Upstream first, from the partial clone's own remote (commits and trees
REM only): the ref's fetch below then sends only what the ref adds.
%GIT% -C "%SRC%" fetch --no-tags --quiet origin
if errorlevel 1 ( echo FETCH_FAILED: upstream & exit /b 68 )
%GIT% -C "%SRC%" update-ref -d "%PIN%" >nul 2>&1
%GIT% -C "%SRC%" fetch --no-tags "%REMOTE%" "+%REF%:%PIN%"
if errorlevel 1 ( echo FETCH_FAILED & exit /b 68 )
set "GOT="
for /f "delims=" %%h in ('git -C "%SRC%" rev-parse --verify --quiet "%PIN%^{commit}"') do set "GOT=%%h"
if /i not "%GOT%"=="%SHA%" (
  echo SHA_MISMATCH: %REF% is now %GOT%, not the pinned %SHA%
  call :cleanup
  exit /b 70
)

echo === STEP worktree %WT%
if exist "%WT%" (
  %GIT% -C "%SRC%" worktree remove --force "%WT%" >nul 2>&1
  if exist "%WT%" rmdir /s /q "%WT%"
)
%GIT% -C "%SRC%" worktree prune
REM --quiet: git's checkout progress would arrive as a hundred lines.
%GIT% -C "%SRC%" worktree add --quiet --detach "%WT%" %SHA%
if errorlevel 1 ( echo WORKTREE_FAILED & call :cleanup & exit /b 71 )
if /i "%FIDIM_STOP_AFTER%"=="worktree" ( call :cleanup & echo === BUILD_EXIT 0 stopped after the worktree & exit /b 0 )

REM Before upstream b5269 the three targets were examples.
set "EXAMPLES=OFF"
if not exist "%WT%\tools\server\CMakeLists.txt" set "EXAMPLES=ON"

echo === STEP configure (HIP, %GPUS%, Release, examples %EXAMPLES%)
REM vcvars64 runs vswhere by name; without the installer folder on PATH it
REM prints "'vswhere.exe' is not recognized" (harmless, but noise in the log).
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%FIDIM_VS%\VC\Auxiliary\Build\vcvars64.bat" %VCVARS_ARGS% >nul
if errorlevel 1 ( echo VCVARS_FAILED & call :cleanup & exit /b 72 )
set "HIP_PATH=%FIDIM_ROCM%"
set "PATH=%FIDIM_CLANG_DIR%;%FIDIM_ROCM%\bin;%FIDIM_NINJA_DIR%;%PATH%"
"%FIDIM_CMAKE%" -S "%WT%" -B "%WT%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON ^
  "-DGPU_TARGETS=%GPUS%" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=%EXAMPLES% -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF
if errorlevel 1 ( echo CONFIGURE_FAILED & call :cleanup & exit /b 73 )
if /i "%FIDIM_STOP_AFTER%"=="configure" ( call :cleanup & echo === BUILD_EXIT 0 stopped after configure & exit /b 0 )

REM ROCm 7's hipBLAS dropped hipblasDatatype_t; ggml's HIP code moved to the
REM new compute types in upstream b5872 (it defines cublasComputeType_t as
REM either, by HIP version). A tree that only knows the old type cannot
REM compile here. The define moved between these files over the years; the
REM old one's comment names the new type, so only the defines are matched.
set "SDK_HIPBLAS_H=%FIDIM_ROCM%\include\hipblas\hipblas.h"
set "OLD_HIPBLAS="
set "NEW_HIPBLAS="
for %%f in (ggml\src\ggml-cuda\vendors\hip.h ggml-cuda\vendors\hip.h ggml-cuda\common.cuh ggml-cuda.cu) do (
  if exist "%WT%\%%f" findstr /r /c:"define cublasComputeType_t  *hipblasDatatype_t" "%WT%\%%f" >nul 2>&1 && set "OLD_HIPBLAS=1"
  if exist "%WT%\%%f" findstr /r /c:"define cublasComputeType_t  *hipblasComputeType_t" "%WT%\%%f" >nul 2>&1 && set "NEW_HIPBLAS=1"
)
if defined NEW_HIPBLAS set "OLD_HIPBLAS="
if not exist "%SDK_HIPBLAS_H%" set "OLD_HIPBLAS="
if defined OLD_HIPBLAS findstr /c:"hipblasDatatype_t" "%SDK_HIPBLAS_H%" >nul 2>&1 && set "OLD_HIPBLAS="
if defined OLD_HIPBLAS (
  echo PRE_ROCM7_TREE: this tree's HIP code predates upstream b5872 and uses hipBLAS types the HIP SDK at %FIDIM_ROCM% no longer has; build a newer ref, or use a ROCm 6 HIP SDK
  call :cleanup
  exit /b 76
)

echo === STEP build llama-server llama-quantize llama-tokenize
"%FIDIM_CMAKE%" --build "%WT%\build" --target llama-server llama-quantize llama-tokenize -j
if errorlevel 1 ( echo BUILD_FAILED & call :cleanup & exit /b 74 )

echo === STEP copy %OUT%
if not exist "%OUT%\bin" mkdir "%OUT%\bin"
copy /y "%WT%\build\bin\*.exe" "%OUT%\bin\" >nul
if errorlevel 1 ( echo COPY_FAILED & call :cleanup & exit /b 75 )
copy /y "%WT%\build\bin\*.dll" "%OUT%\bin\" >nul
if errorlevel 1 ( echo COPY_FAILED & call :cleanup & exit /b 75 )
mkdir "%OUT%\source\src" "%OUT%\source\ggml\include" 2>nul
for %%f in (src\llama-arch.cpp src\llama-vocab.cpp src\llama.cpp ggml\include\ggml.h) do if exist "%WT%\%%f" copy /y "%WT%\%%f" "%OUT%\source\%%f" >nul

call :cleanup
echo === BUILD_EXIT 0 (%REF% @ %SHA%, out %OUT%)
exit /b 0

:cleanup
echo === STEP cleanup %WT%
%GIT% -C "%SRC%" worktree remove --force "%WT%" >nul 2>&1
if exist "%WT%" rmdir /s /q "%WT%"
%GIT% -C "%SRC%" worktree prune
%GIT% -C "%SRC%" update-ref -d "%PIN%" >nul 2>&1
exit /b 0
