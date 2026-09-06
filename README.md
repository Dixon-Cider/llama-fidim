# Llama FIDIM

Local control surface for llama.cpp servers on a multi-GPU Windows
workstation. Replaces the hand-edited `start-gemma-*.bat` launchers with a
tool that knows the failure modes and refuses to launch configurations that
will fail. Spec: the "Llama FIDIM — Build & Configuration Manager Spec" artifact
(v1.1, multi-GPU amendments).

## Layout

- `crates/fidim-core` — all logic: discovery, GGUF header parsing, device
  enumeration/stable keys, VRAM estimator, pre-flight checks, launch
  composition, supervision, benchmarks, script export.
- `crates/fidim-cli` — the `fidim` binary (thin clap wrapper).
- `fixtures/` — captured `--list-devices` / `hipInfo` / WMI output from the
  target machine (2x Radeon AI PRO R9700 + Ryzen iGPU).

## Commands

```
fidim scan                 # discover builds + models (mmproj/MTP paired)
fidim devices              # fresh GPU enumeration: keys, VRAM, displays
fidim profiles             # list saved profiles with validation findings
fidim check <id>           # run the 11-check pre-flight without launching
fidim launch <id>          # pre-flight, spawn detached, verify placement
fidim status [--deep]      # re-attach; health is healthy/not-generating/dead
fidim bench <id>           # warmups + serial + concurrent sweep -> baseline
fidim stop <id|port>       # clean stop
fidim export <id>          # standalone .bat/.ps1 (runs without Llama FIDIM)
fidim logs <id>            # log path + tail
```

Config lives at `~/.fidim/config.json` (build/model roots, ROCm bin);
profiles at `~/.fidim/profiles/*.json` (hand-editable, schema v1, plural
`devices` array); run state + logs at `~/.fidim/runs/`.

## Non-negotiable invariants (from hard-won failures)

- Devices bind by **stable key** (`pci:<hw>:busNN`), never by index. On this
  machine ROCm enumerates R9700=0, **iGPU=1**, R9700=2.
- Every launch pins `HIP_VISIBLE_DEVICES` to exactly the resolved devices —
  llama-server never sees the full enumeration (R-13).
- Residency truth is PDH `GPU Process Memory` per LUID. WDDM virtualizes
  VRAM: `--list-devices` free-memory deltas cannot see other processes.
  Split layers become dedicated-resident on first inference; `Total
  Committed` proves placement immediately.
- Cold-cache first benchmarks are recorded but never become baselines (R-08).

## Build

```
cargo build --release
cargo test              # set FIDIM_TEST_MODELS=<folder of .gguf> to also parse real files
```
