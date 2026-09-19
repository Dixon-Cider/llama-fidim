# fidim-dg-overlay

Llama FIDIM's DiffusionGemma runner patch, built for Unsloth's llama.cpp
releases and published as an overlay.

Unsloth's Windows ROCm builds carry the DiffusionGemma runner. The patch here
(dgpatch5) changes the runner and llama.dll; it does not touch ggml. So an
overlay is just the llama-level binaries of Unsloth's build (llama.dll,
llama-common.dll, mtmd.dll, the tools and the runner), rebuilt from the same
release's source with the patch applied. Llama FIDIM lays it over Unsloth's
zip of that release: the runner gets the patch, and ggml, the HIP backend and
AMD's ROCm libraries stay exactly as Unsloth ships them.

This folder lives in the Llama FIDIM repository as `packaging/dg-overlay` and
is laid out as the root of its own repository, where the workflow runs.

## What the patch does

- The prompt-KV store is F16, and sliding-window layers keep a ring of
  `n_swa-1 + n_ubatch` rows instead of every prompt position.
- Flash attention pads keys to the kernel's 256-key stride, so the 512-wide
  full-attention heads run on the GPU; under flash attention the runner sizes
  its context by `llama_diffusion_fa_turn_bytes` instead of the N² score
  tensor (up to 65,536 tokens on a 32 GB card).
- Each block extends the prompt-KV store instead of prefilling the whole
  prompt again.
- The self-conditioning matmul runs as 32 vocabulary slices.
- A failed denoise step ends the block with `ERR gen` instead of committing
  a stale canvas; `DG_FRAME_SPECIAL=1` frames keep special tokens.
- The diffusion CLI's chunked-prefill sizing and the repetition-loop trim
  are fixed.

`patches/dgpatch5.diff` applies with `git apply` to the source tarball of
Unsloth release b11030-mix-5ff778e. It is the local dgpatch4 without its test
hooks (store poisoning, logit dumps, the op profiler, the pool trim and its
ggml-cuda changes, the env switches for the store type, the ring and the
split count).

## How an overlay fits its base

- The base is one Unsloth release. Its git tag is not its source (Unsloth
  merges pull requests at build time and never pushes the result); the
  release's `llama.cpp-source-commit-<sha>.tar.gz` asset is, pinned by its
  sha256 in GitHub's digest and in the release's `llama-prebuilt-sha256.json`.
- The overlay is built from that tarball with Unsloth's Windows ROCm flags
  minus HIP. llama.dll talks to ggml through a plain C interface resolved by
  name, so it binds to Unsloth's ggml-hip build. It also compiles ggml's
  enums and struct layouts in from the headers, so an overlay fits only the
  release it was built from: it is rebuilt for every tag.
- Every llama-level binary of the base is replaced (60 files for b11030),
  except `llama-cvector-generator.exe`, which stays Unsloth's: it imports
  ggml-hip for its GPU path.
- The gate (`scripts/gate.ps1`) checks every name the overlay imports from
  Unsloth's ggml DLLs against their exports, that nothing imports
  ggml-hip.dll, and that the overlay replaces every llama-level file of the
  base. Only an overlay the gate passed is packaged (`gate.pass`).
- The descriptor (`fidim-overlay.json`, see DESCRIPTOR.md) pins the release,
  the source commit and the sha256 of every Windows ROCm zip of the release,
  and lists every file of the overlay. Llama FIDIM refuses an install where
  any of them disagree.

## Releases

One release per Unsloth tag, named `dgpatch5-<unsloth tag>`:

| Asset | |
| --- | --- |
| `fidim-dg-overlay-<tag>-windows-x64.zip` | the binaries, and their license texts under `licenses/` |
| `fidim-overlay.json` | the descriptor |
| `dgpatch5.diff` | the patch they were built with |
| `SHA256SUMS` | the three above |

Install one with Llama FIDIM: Updates, Unsloth builds, "Install with FIDIM
runner patch", or

```
fidim update --channel unsloth --install --overlay
```

It installs beside the plain Unsloth build as `<tag>-unsloth-dgpatch5`.

## Building

### The workflow

`.github/workflows/build-overlay.yml`, run by hand with an Unsloth tag (or
`latest`) and a compiler (clang, as upstream's Windows recipe and Unsloth, or
msvc):

1. **build** (windows-2022, Visual Studio 2022, no ROCm): fetch and check the
   base, apply the patch (a patch that no longer applies opens an issue),
   build, and gate against the release's gfx120X zip. A gate run before
   that one also starts `llama-server --version` on base + overlay; whether
   AMD's HIP runtime loads on a runner without an AMD GPU is not known, so
   that step only reports. The GitHub token reaches only the steps that call
   GitHub, never the build or the smoke test, which run code from Unsloth's
   release.
2. **sign**, only when the repository variable `AS_ACCOUNT` is set: Azure
   Artifact Signing through GitHub OIDC, in the `release-signing`
   environment (give it a required reviewer). It needs the variables
   `AS_ENDPOINT`, `AS_ACCOUNT` and `AS_PROFILE` and the secrets
   `AZURE_CLIENT_ID`, `AZURE_TENANT_ID` and `AZURE_SUBSCRIPTION_ID`; the Entra
   federated credential's subject is
   `repo:<owner>/<repo>:environment:release-signing`. Only the files built
   here are signed. Without signing the overlay ships unsigned, and Smart App
   Control blocks unsigned DLLs where it is on.
3. **release**: package, attest build provenance, and create the release
   (never marked latest). A tag that already has a release is refused.

### On a Windows machine

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build-local.ps1 -Tag b11030-mix-5ff778e
```

Needs Visual Studio 2022 with the C++ tools (its bundled CMake and Ninja are
used) and git; no ROCm. Options: `-Toolchain clang|msvc` (auto picks clang
when the VS "C++ Clang tools" component is installed), `-BoringSsl off` to
build llama-common without HTTPS instead of cloning BoringSSL, or
`-BoringSsl <folder>` to build a BoringSSL source you already have,
`-WorkDir` (short, and outside any git checkout: the build stamps its version
from the first repository above the source). The release assets land in
`<WorkDir>\out`; VALIDATION.md is what to run on them before publishing.

| Script | Does |
| --- | --- |
| `scripts/fetch-base.ps1` | the release's source tarball (checked), unpacked; optionally one of its Windows ROCm zips; `base.json` |
| `scripts/apply-patch.ps1` | `git apply --check`, `git apply`, the overlay's build fingerprint; an edited patch, or another tag, starts again from the unpacked tarball |
| `scripts/build.ps1` | configure and build, collect the overlay binaries and license texts |
| `scripts/gate.ps1` | symbol closure and file set against the base zip; `-Smoke` runs it; `gate.pass` when it passes |
| `scripts/package.ps1` | the zip, the descriptor, the patch, SHA256SUMS; only after the gate passed |
| `scripts/build-local.ps1` | all of the above in order |
| `scripts/common.ps1` | shared settings: the repository name (`$OverlayRepo`), the patch name, the file allowlist |
| `tests/scripts.tests.ps1` | self-tests of the above that need no network or build (the workflow runs them first) |

## The repository name

Set once in `scripts/common.ps1` (`$OverlayRepo`); the workflow publishes to
the repository it runs in. Llama FIDIM looks for releases in
`Dixon-Cider/fidim-dg-overlay` (`OVERLAY_REPO` in
`crates/fidim-core/src/overlay.rs`), or in the repository named by
`"overlay_repo"` in `~/.fidim/config.json`.

## Rebasing the patch

When a new Unsloth release changes the files the patch touches, `git apply
--check` fails. Unpack the new source, apply the patch with `git apply
--3way` (in a scratch git repository holding the old and new trees) or by
hand, regenerate `patches/dgpatch5.diff` with `git diff`, and run
build-local.ps1 (on the same WorkDir it unpacks the release's source again
for the new patch) and VALIDATION.md again before publishing.

## License

MIT, see LICENSE. The binaries in each overlay are llama.cpp (MIT) and the
libraries it builds in; `licenses/THIRD-PARTY-NOTICES.md` in the zip lists
them (template: THIRD-PARTY-NOTICES.md).
