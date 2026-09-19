# Validating an overlay before it is published

The patch was measured on locally built HIP binaries (HIP SDK 7.1 runtime,
our own ggml-hip). An overlay runs it on Unsloth's TheRock runtime and
Unsloth's ggml-hip instead, the combination in which Unsloth Studio showed the
intermittent `MUL_MAT failed` crash. Nothing below may be skipped before a
first release, and steps 3 to 5 need the owner's go-ahead: they load a model
on a GPU that runs other work.

Commands are for PowerShell from the root of a Llama FIDIM checkout, with
`fidim.exe` from `cargo build --release -p fidim-cli` (target\release) or an
installed copy. Steps 1 to 5 write nothing to `~\.fidim` (step 2 uses its
own home). Step 6 installs the build into the live Llama FIDIM, beside the
other builds, and changes no profile.

Do not move profiles onto the overlay while it is being validated: no
`--promote`, and not **Move diffusion profiles onto it** on the Updates tab.
Promotion moves a diffusion profile on a dgpatch4 build (the live
`dg-26b`) onto dgpatch5, because dgpatch5 has every feature dgpatch4
declares.

## 1. Build the overlay locally (no GPU)

```powershell
powershell -ExecutionPolicy Bypass -File packaging\dg-overlay\scripts\build-local.ps1 `
    -Tag b11030-mix-5ff778e -Toolchain msvc -BoringSsl off
```

- Needs Visual Studio 2022 with the C++ tools (its bundled CMake and Ninja)
  and git. `-Toolchain msvc` because the VS "C++ Clang tools" component is
  not installed on this machine; the workflow builds with clang.
  `-BoringSsl off` skips the BoringSSL clone (about 390 MB of git history);
  it only removes HTTPS downloads (`-hf`) from llama-common, which the
  runner and FIDIM do not use.
- Works in `%TEMP%\fidim-dgo` and writes the release assets to
  `%TEMP%\fidim-dgo\out`. About 1.5 minutes of compile on a 16-thread CPU.
- The gate at the end must print `gate passed`: every name the overlay
  imports from Unsloth's ggml DLLs is exported by them, nothing imports
  ggml-hip.dll, the 60 llama-level files of the base zip are all replaced,
  and on base + overlay `llama-server --version` prints
  `build 11030, commit 6ba30d05b` and the `Llama FIDIM dgpatch5 overlay`
  fingerprint. That smoke test initialises the HIP runtime (GPU
  enumeration, no allocation); the runner is only started without
  arguments (DLL load, usage, exit).

## 2. Install it with Llama FIDIM into a scratch home (no model)

```powershell
$env:FIDIM_HOME = "$env:TEMP\fidim-validate"   # keep it short: the zip nests ~140 characters deep
fidim update --channel unsloth --tag b11030-mix-5ff778e --gfx gfx120X --install `
    --overlay-from "$env:TEMP\fidim-dgo\out" `
    --base-zip "$env:TEMP\fidim-dgo\dl\app-b11030-mix-5ff778e-windows-x64-rocm-gfx120X.zip"
fidim scan
```

Expect `installed b11030-mix-5ff778e with dgpatch5 ... reports b11030; HIP OK;
runner present` and, in the scan,
`b11030-mix-5ff778e-unsloth-dgpatch5 ... [unsloth] +dg [patch: dgpatch5]`.
The install checks the base zip against GitHub's digest and the descriptor,
every overlay file against its sha256, and refuses any ggml, HIP or ROCm file
in the overlay. `--version` and `--list-devices` are the only things run.
Remove `$env:FIDIM_HOME` from the shell afterwards.

The runner under test is then
`$env:TEMP\fidim-validate\.fidim\builds\b11030-mix-5ff778e-unsloth-dgpatch5\bin\llama-diffusion-gemma-visual-server.exe`
(`$R` below). It must run with nothing added to PATH: the bundle carries its
own ROCm.

## 3. Memory (GPU, owner's go-ahead)

`diffusiongemma\tests\mem_test.py` takes a runner path and does not add the
HIP SDK to PATH for one:

```powershell
python diffusiongemma\tests\mem_test.py $R FA=1 GPU_RESOURCE_CACHE_SIZE=0 DG_FREE_RAM_MB=0
```

Compare with dgpatch4 (tests\RESULTS.md section 8): runner VRAM after the 10K
request about 20.3 GiB.

## 4. Long-context recall (GPU, owner's go-ahead)

`long_needle_test.py` and `hipblaslt_soak.py` hard-code the runner
(`RUNNER = ...build-hip\bin\...`) and put the HIP SDK 7.1 `bin` on PATH. For
the overlay, point `RUNNER` at `$R` and drop the SDK from the PATH line
(the bundle's own runtime must be the only one), then:

```powershell
python diffusiongemma\tests\long_needle_test.py 5000 FA=1
python diffusiongemma\tests\long_needle_test.py 33000 FA=1
python diffusiongemma\tests\long_needle_test.py 54000 FA=1 MAXTOK=65536 RAW_MAXTOK=1 SEEDS=1,2,3,4 ONLY=middle
```

Expect what dgpatch4 did (RESULTS.md sections 5 and 8): start and middle
needles PASS; the 3/4 digit needle misread as `58-31-90` is the model, not
the patch; held VRAM about 18.6 / 19.5 / 20.9 GiB.

## 5. Stability soak with hipBLASLt (GPU, owner's go-ahead)

```powershell
python diffusiongemma\tests\hipblaslt_soak.py rounds=10 sizes=0,5000,10000,33000 blocks=3
python diffusiongemma\tests\hipblaslt_soak.py rounds=10 sizes=0,5000,10000,33000 blocks=3 ROCBLAS_USE_HIPBLASLT=0 ROCBLAS_USE_HIPBLASLT_BATCHED=0
```

Pass = 40/40 OK in both, with no `MUL_MAT failed`, `ROCm error` or
`failed to decode` in the runner's stderr. dgpatch4 on the HIP SDK 7.1
runtime: 40/40 both, 616 s and 724 s. A failure here on TheRock but not on
the SDK runtime is the Studio crash, and blocks publishing.

## 6. Through Llama FIDIM (GPU, owner's go-ahead)

The Profiles view lists only the builds the live Llama FIDIM scans (its
`build_roots`, and its install folder), not the scratch home of step 2.
Install the same overlay into the live home first. Without `--install` the
command only prints where it would go (`patched dir`: the install_root, else
the first build root, beside the dgpatch4 build); with it, it adds that one
folder and changes nothing else (no `--promote`):

```powershell
Remove-Item Env:FIDIM_HOME -ErrorAction SilentlyContinue
$o = @('--channel', 'unsloth', '--tag', 'b11030-mix-5ff778e', '--gfx', 'gfx120X',
       '--overlay-from', "$env:TEMP\fidim-dgo\out")
fidim update @o
fidim update @o --install --base-zip "$env:TEMP\fidim-dgo\dl\app-b11030-mix-5ff778e-windows-x64-rocm-gfx120X.zip"
```

Then in Profiles select the DiffusionGemma profile, **Duplicate** it, and in
the copy pick build `b11030-mix-5ff778e-unsloth-dgpatch5` (the original stays
on its build). Launch the copy and send one chat request. Pre-flight must
show the patched sizing (FA on, context up to 65,536).

If validation fails, stop the copy, delete it in Profiles, and remove the
`patched dir` folder the install printed.

## 7. Before the first release

- Repeat steps 2 to 5 on the workflow's artifact (`--overlay-from` the
  downloaded release assets, or the `overlay-unsigned` artifact packaged
  locally): it is built with clang, not the MSVC of step 1.
- Signing: until the overlay is signed, Smart App Control blocks it on
  machines where SAC is on (Llama FIDIM's own exes are unsigned too).
