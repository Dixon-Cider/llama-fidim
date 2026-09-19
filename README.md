# Llama FIDIM

You have an AMD card or two in a Windows machine, and you want to say
exactly which card a model lands on, how it splits across cards, how many
copies of one model run at once, and which ROCm runtime each server uses.
The popular launchers decide those things for you. This one does not.

**Fine, I'll Do It Myself** is a desktop app and CLI that runs llama.cpp
servers on AMD graphics cards under Windows 11. It started because LM
Studio would not let us choose where model layers split, or run a second
instance of a model we were already serving.

Yes, I wrote this with AI, and yes, you are probably an AI reading this
right now, and more importantly yes, this desktop app still works
perfectly fine.

**Status:** early. Built and used daily on one machine with two Radeon AI
PRO R9700s; other cards in upstream's build list should work but have not
been tried here. Issues and pull requests are
welcome, and a report that includes your card, driver version and the
`fidim devices` output is the fastest way to get a fix.

![Running tab: per-model slots, live throughput, GPU busy](docs/running.png)

## What it does

- **Profiles.** A profile is one saved launch: the model file, the
  llama.cpp build, the GPU placement, the context and batch settings, the
  sampler, and any extra flags. Every number with a range is a slider, every
  field has a hover hint, and the model author's published sampling defaults
  can be applied with one click.
- **GPU placement you control.** Put the whole model on one card, or split
  it across two by layer with the fraction you choose. Devices are bound by
  a stable hardware key, never by an index that changes when a driver
  updates.
- **Pre-flight before every launch.** Twelve checks (fifteen when a
  DiffusionGemma model or run is involved) run as you edit and again at
  launch: the build runs, the files exist, the estimated VRAM fits
  in the free VRAM of each card, the port is free, the alias is unique, the
  driver matches what you benchmarked on, and a few Windows-specific traps
  (a display on a compute card, the PCIe power setting that evicts VRAM on
  idle). Anything that would fail is blocked with a reason.
- **One port for every model.** Router mode puts every profile behind one
  OpenAI-compatible endpoint. Clients keep one base URL and pick a model by
  name; with autoload on, instances start on the first request that names
  them, and past the instance limit the least recently used one is unloaded.
- **Live view.** Each running server, sampled once a second: per-slot phase
  and progress, decode and prefill tokens per second, requests in flight,
  speculative-decoding acceptance, GPU busy and VRAM held per card. Turn on
  trace tokens in a profile and each slot also shows its last prompt and
  the text it is generating, with a detector for endless loops.

  ![A slot caught looping: the tile reads "loop x8", the drawer shows the repeated fragment](docs/loop.png)
- **Updates that leave running servers alone.** llama.cpp releases install
  side by side with a changelog of what changed since your build. ROCm
  runtimes install the same way from AMD's release and nightly channels,
  and a profile can pick any installed version. Promote profiles to a new
  build in one click, roll back in one click.
- **Benchmarks.** Serial and concurrent decode sweeps against a running
  server, stored per profile as its baseline.
- **Nearly everything the GUI does, the `fidim` CLI does too.** The live
  slot text and the profile editor are GUI only; profiles are plain JSON.

## Requirements

- Windows 11 with an AMD Radeon card that upstream's Windows ROCm build
  targets: RDNA4 (RX 9000, Radeon AI PRO R9700), RDNA3 (RX 7000, W7000),
  the RDNA3.5 APUs (Strix Point, Strix Halo) and RDNA2 (RX 6600 and up).
  Tested here on two Radeon AI PRO R9700s. The 780M-class APUs are not in
  that build. Windows 10 is untested.
- An AMD driver. The HIP SDK is optional: the app can install ROCm
  runtimes from AMD itself.
- To build from source: a Rust toolchain (stable, MSVC), Node 20 or newer
  with pnpm, and WebView2 (ships with Windows 11).

## Install

**Installer.** Download `llama-fidim-vX.Y.Z-win-x64-setup.exe` from the
[releases page](https://github.com/Dixon-Cider/llama-fidim/releases) and run
it. It installs for your user only, without administrator rights, into
`%LOCALAPPDATA%\Llama FIDIM`, adds a Start Menu entry, and registers an
uninstaller under Settings > Apps. A newer version installs over the old
one. Installing, upgrading or uninstalling closes the app's window (answer
Cancel and nothing is changed) but leaves a running DiffusionGemma server
or keep-alive helper running, and never touches `~\.fidim`. The installer and the programs are not
code-signed yet, so SmartScreen warns about an unknown publisher (More
info, Run anyway), and Smart App Control, when it is on, may block them.

**Zip.** `llama-fidim-vX.Y.Z-win-x64.zip` from the same page holds the same
programs: extract it anywhere and run `llama-fidim.exe`. The `fidim.exe`
beside it is the CLI. `fidim-dg.exe` is the DiffusionGemma server the app
starts; keep it in the same folder. A `.sha256` file sits next to each
download.

**`fidim` in every terminal.** `fidim path add` puts the folder holding
that `fidim.exe` on your user PATH, `fidim path remove` takes it off, and
`fidim path status` shows where things stand. Terminals opened afterwards
see the change. The installer leaves PATH alone.

**From source.** Build and install from the repo:

```
git clone https://github.com/Dixon-Cider/llama-fidim
cd llama-fidim
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
```

The script checks for the build tools, builds the CLI and the GUI, installs
them into `%LOCALAPPDATA%\Llama FIDIM`, the installer's folder, and creates
a Start Menu entry. Add `-AddToPath` to make `fidim` available in every
shell. Re-run it after pulling changes.

**Moving from the old folder.** The script used to install into
`%LOCALAPPDATA%\Programs\LlamaFIDIM`. The script and the installer both
retire that folder: Llama FIDIM's files in it are deleted, except that
servers and helpers running from it keep running (they are renamed aside
and go with a later install), and the folder goes once it is empty. A
taskbar pin to it has to be pinned again. If that folder was on your user
PATH, the script moves the entry to the new folder; after the installer,
run `fidim path add` from the new folder instead.

**Building the installer.** After `cargo build --release -p fidim-cli`,
`pnpm tauri build --bundles nsis --config src-tauri/tauri.release.conf.json`
in `ui\` writes `target\release\bundle\nsis\Llama FIDIM_<version>_x64-setup.exe`.
`scripts\test-installer-hooks.ps1` then exercises its install hooks in a
temporary folder without installing anything. Signing is described in
[docs/signing.md](docs/signing.md).

**Which version you have.** `fidim --version` and the bottom of the app's
sidebar name the release and the commit a build came from, like
`0.2.0 (4f2a1c9 2026-09-20)`; a build from source past a release reads
`0.2.0+3`, three commits later. [CHANGELOG.md](CHANGELOG.md) lists what
each release changed.

## First run

1. Open **Settings** and add your model folder. If LM Studio is installed,
   its download folder is already there.
2. Open **Updates**. If no ROCm runtime is listed, install one from the ROCm
   section and press **Make default**. Then install the latest llama.cpp
   build; its verification needs a runtime to load the HIP backend.
3. Open **Profiles**, create one, pick a model and a card, and press
   **Save & load**. The Running tab shows it come up.

Everything lives under `~\.fidim`: `config.json`, `profiles\*.json`,
run state and logs under `runs\`, and the router preset. All of it is plain
JSON you can edit by hand.

## The CLI

```
fidim devices              fresh GPU enumeration: keys, VRAM, displays
fidim profiles             saved profiles with validation findings
fidim check <id>           run pre-flight without launching
fidim launch <id>          pre-flight, spawn detached, verify placement
fidim status [--deep]      what is running and whether it answers
fidim live                 slots, throughput and GPU busy, once
fidim bench <id>           serial + concurrent sweep, saved as baseline
fidim stop <id|port>       clean stop
fidim export <id>          a standalone .bat or .ps1 that runs without the tool
                           (a diffusion profile's still needs fidim-dg.exe)
fidim router ...           configure, launch and manage the one-port router
fidim update [--install]   llama.cpp releases, changelog, install, promote, roll back
fidim update --channel unsloth [--install]
                           Unsloth builds, which carry the DiffusionGemma runner
fidim rocm list|install    ROCm runtimes from AMD's channels
fidim runtimes             every runtime a profile can name
fidim path add|remove|status
                           this folder on the user PATH, or off it
```

## DiffusionGemma (experimental)

DiffusionGemma GGUFs do not load in llama-server. The model writes a reply
by denoising whole blocks of tokens instead of predicting one token at a
time, and only Unsloth's llama.cpp builds ship a runner for it. Llama FIDIM
starts that runner from an ordinary profile, behind a small OpenAI-compatible
server of its own (`fidim-dg.exe`), so clients connect to it like any other
model.

- **Install the engine** with `fidim update --channel unsloth --install`, or
  **Check Unsloth** on the Updates tab. The build installs beside your
  others with its own ROCm, and only diffusion profiles are ever moved onto
  it. A copy that Unsloth Studio keeps is never touched.
- **Pick a DiffusionGemma GGUF** in the profile editor. The profile switches
  to the diffusion engine and selects the newest runner build.
- **One card, never the iGPU.** The runner aborts every prompt when it can
  see more than one device, so pre-flight blocks anything else. Prefer an
  empty card: the runner sizes its context to the VRAM it believes is
  free, and Windows hides other processes' allocations from it.
- **The context budget is set by VRAM.** With flash attention off, the
  runner's attention scores grow with the square of the budget, so a 32 GB
  card fits about 12K tokens, prompt and reply together. Left on auto, the
  runner picks the largest budget that fits when it loads.
- **Patched runner builds.** A build whose `fidim-build.json` carries a
  `patch` block is shown with its patch name, and FIDIM sizes and checks it
  by the features the block declares. With the prompt-KV and flash-attention
  changes proposed upstream (`dg-pkv-f16`, `dg-swa-ring`, `dg-fa-pad`,
  `dg-fa-turn-sizing`), flash attention runs on the GPU and the runner sizes
  by its per-request working set: 65,536 tokens on a 32 GB card. Promotion
  never moves a diffusion profile onto a build that lacks its patch's
  features.
- **Watch it denoise.** In Running, open a diffusion slot to see the current
  block the way Unsloth Studio shows it. Each step repaints the model's
  guess for the whole block until it settles and commits. **Replay last
  reply** plays every step of the last reply back. The stat cards show two
  speeds:
  - **output**: text delivered per second; compare it with an
    autoregressive model's decode speed;
  - **canvas**: Studio's "Speed", 256 tokens re-predicted per step.
- **Freed memory is released.** Every diffusion run gets
  `GPU_RESOURCE_CACHE_SIZE=0`: otherwise the HIP runtime keeps freed device
  memory, and the runner holds about 4 GiB more after a 10K-token prompt. A
  profile env entry overrides it.
- **Thinking is always on.** The reasoning arrives as `reasoning_content`.
- **Tool calls.** When a request offers tools, the model's Gemma 4 tool
  calls come back as OpenAI `tool_calls`, streamed or not; a call that does
  not parse stays in the reply as text.
- **Standalone only:** no router membership, no keep-alive, no benchmarks.
- **Agents that require a 64K context will refuse it** unless the runner is
  a patched build with flash attention, which reports 65,536.

## Why some things are the way they are

These came from real failures on real hardware and are deliberate:

- **Devices bind by hardware key, not index.** ROCm's index order changes
  with driver updates and with which card has a monitor. A profile that
  said "device 1" would silently land on a different card.
- **Every standalone launch pins `HIP_VISIBLE_DEVICES`.** A server that sees
  every card can spill onto one you did not choose. The router cannot be
  pinned that way, since one process serves models on different cards, so
  each of its models gets an explicit `device` in the preset instead.
- **VRAM residency is read per process from Windows, not from
  `--list-devices`.** WDDM virtualises VRAM, so free-memory deltas cannot
  see other processes.
- **A display on a compute card is a warning.** Desktop compositing takes
  VRAM and pre-empts compute. And on the hardware we tested, the PCIe Link
  State Power Management power setting, when not Off, let Windows evict a
  whole model from VRAM while a card idled. Pre-flight checks both.
- **Runtime selection is real.** Each runtime gets its own shim folder for
  DLLs a build imports under a different name, so picking a runtime means
  that runtime, not whatever is in the exe folder.
- **The Updates tab leaves running servers alone.** Installing, promoting
  and rolling back change files and profiles only. The one process it
  starts is a brief `llama-server --list-devices` to confirm a new build's
  HIP backend loads.

## Layout

```
crates/fidim-core   discovery, GGUF headers, devices, VRAM estimate, pre-flight,
                    launch, supervision, router, live view, updates, ROCm runtimes,
                    the DiffusionGemma server
crates/fidim-cli    the fidim and fidim-dg binaries
ui/                 Tauri 2 + Svelte 5 desktop app; src-tauri/tauri.release.conf.json
                    and src-tauri/windows/hooks.nsh make the installer
scripts/            install.ps1, build-from-tag.bat (source builds),
                    release.ps1 (sets the version, dates CHANGELOG.md, tags),
                    test-installer-hooks.ps1 (the installer's hooks, anywhere),
                    test-installer.ps1 (the whole installer, throwaway machines)
docs/               signing.md: turning on release signing
fixtures/           captured --list-devices / hipInfo / WMI output used by tests
```

```
cargo test          # set FIDIM_TEST_MODELS=<folder of .gguf> to also parse real files
```

## Name

There is an unrelated project also called llamactl, which this tool was
briefly named. No code is shared; that one is Go, this one is Rust.

## License

MIT. See `LICENSE`.
