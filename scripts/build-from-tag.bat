@echo off
REM Build one llama.cpp tag from source, HIP for gfx1201, into an immutable
REM output directory. Invoked by `llamactl update --source` as:
REM     build-from-tag.bat <llama.cpp checkout> <tag> <output dir>
REM The checkout's HEAD is restored afterwards, so a pinned checkout stays
REM pinned. Output dir gets bin\llama-server.exe (+ DLLs) like every other build.
REM
REM Prereqs (one-time, see BUILD_NOTES.md): ROCm 7.1 HIP SDK with the cmath
REM wrapper patch applied, VS 2026 Build Tools. Override with env vars
REM LLAMACTL_ROCM / LLAMACTL_VS if they move.
setlocal enableextensions
if "%~3"=="" (
  echo usage: %~nx0 ^<checkout^> ^<tag^> ^<output dir^>
  exit /b 64
)
set "SRC=%~1"
set "TAG=%~2"
set "OUT=%~3"
if "%LLAMACTL_ROCM%"=="" set "LLAMACTL_ROCM=C:\Program Files\AMD\ROCm\7.1"
if "%LLAMACTL_VS%"=="" set "LLAMACTL_VS=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
set "CMAKE=%LLAMACTL_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA_DIR=%LLAMACTL_VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

if not exist "%SRC%\.git" ( echo NOT_A_CHECKOUT %SRC% & exit /b 65 )
if exist "%OUT%\bin\llama-server.exe" ( echo ALREADY_BUILT %OUT% & exit /b 0 )

call "%LLAMACTL_VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo VCVARS_FAILED & exit /b 66 )
set "PATH=%LLAMACTL_ROCM%\bin;%NINJA_DIR%;%PATH%"
cd /d "%SRC%"

for /f "delims=" %%h in ('git rev-parse --abbrev-ref HEAD') do set "PREV=%%h"
if "%PREV%"=="HEAD" for /f "delims=" %%h in ('git rev-parse HEAD') do set "PREV=%%h"
echo === previous HEAD: %PREV%

echo === fetch tags
git fetch --tags origin
if errorlevel 1 ( echo FETCH_FAILED & exit /b 67 )

echo === checkout %TAG%
git checkout --quiet %TAG%
if errorlevel 1 ( echo CHECKOUT_FAILED %TAG% & exit /b 68 )

echo === configure %OUT% (HIP, gfx1201, Release)
"%CMAKE%" -S . -B "%OUT%" -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_CCACHE=OFF ^
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
set "RC=%errorlevel%"
if not "%RC%"=="0" ( echo CONFIGURE_FAILED & git checkout --quiet %PREV% & exit /b 69 )

echo === build llama-server (pulls in mtmd)
"%CMAKE%" --build "%OUT%" --target llama-server -j
set "RC=%errorlevel%"

echo === restore %PREV%
git checkout --quiet %PREV%

echo === BUILD_EXIT %RC% (tag %TAG%, out %OUT%)
exit /b %RC%
