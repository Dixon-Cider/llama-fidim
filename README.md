# llamactl

Local control surface for llama.cpp servers on a multi-GPU Windows
workstation. Replaces the hand-edited `start-gemma-*.bat` launchers with a
tool that knows the failure modes and refuses to launch configurations that
will fail. Spec: the "llamactl — Build & Configuration Manager Spec" artifact
(v1.1, multi-GPU amendments).

## Layout

- `crates/llamactl-core` — all logic: discovery, GGUF header parsing, device
  enumeration/stable keys, VRAM estimator, pre-flight checks, launch
  composition, supervision, benchmarks, script export.
- `crates/llamactl-cli` — the `llamactl` binary (thin clap wrapper).
- `fixtures/` — captured `--list-devices` / `hipInfo` / WMI output from the
  target machine (2x Radeon AI PRO R9700 + Ryzen iGPU).

## Commands

```
llamactl scan                 # discover builds + models (mmproj/MTP paired)
llamactl devices              # fresh GPU enumeration: keys, VRAM, displays
llamactl profiles             # list saved profiles with validation findings
llamactl check <id>           # run the 11-check pre-flight without launching
llamactl launch <id>          # pre-flight, spawn detached, verify placement
llamactl status [--deep]      # re-attach; health is healthy/not-generating/dead
llamactl bench <id>           # warmups + serial + concurrent sweep -> baseline
llamactl stop <id|port>       # clean stop
llamactl export <id>          # standalone .bat/.ps1 (runs without llamactl)
llamactl logs <id>            # log path + tail
llamactl seed                 # starter profiles from the old batch files
```

Config lives at `~/.llamactl/config.json` (build/model roots, ROCm bin);
profiles at `~/.llamactl/profiles/*.json` (hand-editable, schema v1, plural
`devices` array); run state + logs at `~/.llamactl/runs/`.

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
cargo test              # includes live parses of E:\models when present
```
