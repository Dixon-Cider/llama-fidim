# Changelog

What changed in each release of Llama FIDIM, newest first. Versions follow
[semantic versioning](https://semver.org): until 1.0, a release with new
features or changed behavior bumps the minor version (0.2.0 -> 0.3.0) and a
release of fixes alone bumps the patch (0.2.0 -> 0.2.1).

`fidim --version` and the bottom of the app's sidebar show which build you
are running: `0.2.0 (4f2a1c9 2026-09-20)` is the 0.2.0 release itself,
`0.2.0+3` is a build from source three commits past it (`+?` when a clone
without tags cannot count them), and `-modified` after the commit marks a
build with uncommitted changes.

## [Unreleased]

### Added

- **Builds of any git ref.** `fidim update --source --remote <url> --ref
  <branch|pull/N/head|commit>` compiles a llama.cpp fork's branch, an
  upstream pull request or any commit for this machine's GPU. The commit
  is pinned first and checked after the fetch; the build runs in FIDIM's
  own clone under `~/.fidim/src`, never in your checkouts; stopping it
  (Ctrl+C) stops every process it started; and the result is labelled by
  where it came from (`ifm-ai K2Horizon fork @42adf01`) and never ranked
  or promoted as an upstream release.
- **Toolchain doctor.** `fidim toolchain` checks Visual Studio's C++ tools,
  git, CMake, Ninja and the HIP SDK, and compiles a test file to catch the
  MSVC `<cmath>` clash with HIP clang (llama.cpp#22570) before a build
  spends minutes finding it. Every source build runs it first.
- **Which build can load a model.** The core can now tell whether a build
  knows a model's architecture, pre-tokenizer and tensor types, from the
  tables in its `llama.dll` or from llama.cpp's source at any commit, and
  plan how to get one that does: an installed build, the newest upstream
  release, an Unsloth mix, an upstream pull request, or a fork the model
  card links. The model wizard builds on this.
- **Pre-flight check 16.** A launch on a build that does not know the
  model's architecture is blocked before llama-server fails with "unknown
  model architecture"; a pre-tokenizer or tensor type it could not confirm
  is a warning.
- `github_token` in `config.json` (or `GITHUB_TOKEN`) for GitHub API
  lookups; without one, answers are cached to stay within 60 requests an
  hour.

## [0.2.0] - 2026-09-18

### Added

- **DiffusionGemma engine (experimental).** A profile whose model is a
  DiffusionGemma GGUF runs through `fidim-dg.exe`, a new server that wraps
  Unsloth's diffusion runner in an OpenAI-compatible API: chat completions
  with streaming and reasoning, models, health, slots and metrics. It
  restarts the runner after a crash, retries a request that had not
  streamed anything yet, queues requests, and reports ready only once the
  runner is on the expected card with every layer offloaded.
- **Tool calls from DiffusionGemma.** When a request offers tools, the
  model's Gemma 4 tool calls come back as OpenAI `tool_calls`, streamed or
  not, so agent clients run them instead of reading them as text.
- **Unsloth builds.** `fidim update --channel unsloth` and a card in
  Updates install Unsloth's ROCm builds, which carry the diffusion runner,
  and move diffusion profiles onto them.
- **Patched runner builds.** A build can declare the runner fixes it
  carries. The VRAM estimate, the pre-flight checks and the flash attention
  toggle follow them, and a diffusion profile is never promoted onto a
  build that lacks them.
- **Watching a reply denoise.** A diffusion slot's drawer in Running shows
  the block as it denoises, laid out like Unsloth Studio's player, and
  replays every step of the last reply with pause, scrub and speed
  controls.
- A diffusion profile editor: one GPU, an automatic or explicit context
  budget, reply budget, seed, the hipBLASLt safeguard and flash attention.
- Pre-flight checks 13 to 15, wherever a DiffusionGemma model, profile or
  run is involved: the engine can load the model (a llama-server profile on
  a DiffusionGemma GGUF is blocked), a card shared with a diffusion run, and
  a diffusion profile's context budget.
- Two speeds for a diffusion run: output tokens per second, comparable to
  autoregressive decode, and canvas tokens per second, the number Unsloth
  Studio calls Speed.
- Version tracking: this changelog, the commit in `fidim --version`, the
  version in the sidebar, and `scripts/release.ps1` to cut a release.

### Changed

- Draft models and vision projectors pair with a model by name across
  model folders, and the draft picker offers every draft.
- Diffusion runs release the HIP runtime's memory cache
  (`GPU_RESOURCE_CACHE_SIZE=0`) unless the profile sets it.
- Router lists llama-server profiles only.

### Fixed

- The release checksum file uses LF line endings, so `sha256sum -c`
  verifies it.
- When a port is taken, pre-flight and the router name the process holding
  it even when it listens on IPv6 or Windows is not in English; they said
  "an unknown process".
- A run's VRAM and GPU busy (the Live view, the launch placement check) no
  longer count processes whose pid merely starts with the run's pid, or
  stray processes that only look like the router's children because
  Windows reused a pid.
- The Router tab's "newest" build is the numerically newest (b10819 over
  b9817).
- A slider that can be switched off no longer draws its off label over its
  minimum.

## [0.1.0] - 2026-09-06

First public release.

- Profiles: one saved launch per JSON file, edited with sliders and hints,
  with the model author's sampling defaults a click away.
- GPU placement by stable hardware key: a whole model on one card, or a
  layer split across two at the fraction you choose.
- Twelve pre-flight checks as you edit and again at launch, from VRAM fit
  per card to the Windows display and PCIe power traps.
- Router mode: every profile behind one OpenAI-compatible port, with
  autoload and least-recently-used unloading.
- Live view: per-slot phase and progress, decode and prefill rates,
  speculative acceptance, GPU busy and VRAM per card, and trace tokens with
  loop detection.
- llama.cpp builds and ROCm runtimes installed side by side, with promote
  and roll back; benchmarks stored per profile; the `fidim` CLI for nearly
  all of it.

[Unreleased]: https://github.com/Dixon-Cider/llama-fidim/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Dixon-Cider/llama-fidim/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Dixon-Cider/llama-fidim/releases/tag/v0.1.0
