@echo off
REM Build any llama.cpp git ref from source (a fork's branch, an upstream pull
REM request, a tag, a commit) with HIP, into an output directory. Invoked by
REM `fidim update --source --remote ...` (and the model wizard) as:
REM     build-from-ref.bat <checkout> <remote> <ref> <sha> <out> <gpu_targets>
REM
REM <checkout>     FIDIM's own clone (created here as a partial clone of
REM                upstream when missing). Nobody else's checkout is touched:
REM                the build runs in a detached worktree beside it.
REM <remote> <ref> fetched with `git fetch <remote> <ref>`; the build stops
REM                unless the fetched commit is exactly <sha> (40 digits).
REM <out>          receives bin\ (executables and DLLs) and source\ (the three
REM                table files FIDIM reads to know what the build supports).
REM <gpu_targets>  comma-separated, e.g. gfx1201 or gfx1100,gfx1201.
REM
REM Toolchain: FIDIM passes what its doctor found in FIDIM_VS, FIDIM_ROCM,
REM FIDIM_CLANG_DIR, FIDIM_CMAKE, FIDIM_NINJA_DIR and optionally
REM FIDIM_VCVARS_VER; run by hand, the newest Visual Studio with C++ tools and
REM HIP_PATH are used. FIDIM_WORKTREE overrides the worktree location.
REM
REM Exit codes: 64 usage, 66 toolchain missing, 67 clone, 68 fetch,
REM 70 ref moved off <sha>, 71 worktree, 72 vcvars, 73 configure, 74 build,
REM 75 copy.
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
if not exist "%SRC%\.git" (
  git clone --filter=blob:none --no-checkout --single-branch "%FIDIM_UPSTREAM%" "%SRC%"
  if errorlevel 1 ( echo CLONE_FAILED & exit /b 67 )
)

echo === STEP fetch %REMOTE% %REF%
git -C "%SRC%" fetch --no-tags "%REMOTE%" "%REF%"
if errorlevel 1 ( echo FETCH_FAILED & exit /b 68 )
set "GOT="
for /f "delims=" %%h in ('git -C "%SRC%" rev-parse "FETCH_HEAD^{commit}"') do set "GOT=%%h"
if /i not "%GOT%"=="%SHA%" (
  echo SHA_MISMATCH: %REF% is now %GOT%, not the pinned %SHA%
  exit /b 70
)

echo === STEP worktree %WT%
if exist "%WT%" (
  git -C "%SRC%" worktree remove --force "%WT%" >nul 2>&1
  if exist "%WT%" rmdir /s /q "%WT%"
)
git -C "%SRC%" worktree prune
REM --quiet: git's checkout progress would arrive as a hundred lines.
git -C "%SRC%" worktree add --quiet --detach "%WT%" %SHA%
if errorlevel 1 ( echo WORKTREE_FAILED & call :cleanup & exit /b 71 )

echo === STEP configure (HIP, %GPUS%, Release)
REM vcvars64 runs vswhere by name; without the installer folder on PATH it
REM prints "'vswhere.exe' is not recognized" (harmless, but noise in the log).
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%FIDIM_VS%\VC\Auxiliary\Build\vcvars64.bat" %VCVARS_ARGS% >nul
if errorlevel 1 ( echo VCVARS_FAILED & call :cleanup & exit /b 72 )
set "HIP_PATH=%FIDIM_ROCM%"
set "PATH=%FIDIM_CLANG_DIR%;%FIDIM_ROCM%\bin;%FIDIM_NINJA_DIR%;%PATH%"
"%FIDIM_CMAKE%" -S "%WT%" -B "%WT%\build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON ^
  "-DGPU_TARGETS=%GPUS%" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ^
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_CCACHE=OFF
if errorlevel 1 ( echo CONFIGURE_FAILED & call :cleanup & exit /b 73 )

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
git -C "%SRC%" worktree remove --force "%WT%" >nul 2>&1
if exist "%WT%" rmdir /s /q "%WT%"
git -C "%SRC%" worktree prune
exit /b 0
