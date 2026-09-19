# fidim-overlay.json (schema 1)

Every overlay release carries a descriptor next to its zip. It pins the
overlay to the one Unsloth release it fits and lists every file in the zip.
`scripts/package.ps1` writes it; Llama FIDIM reads it
(`crates/fidim-core/src/overlay.rs`, `Descriptor`) and ignores fields it does
not know, so producers may add some.

```json
{
  "schema": 1,
  "name": "dgpatch5",
  "overlay_repo": "Dixon-Cider/fidim-dg-overlay",
  "release_tag": "dgpatch5-b11030-mix-5ff778e",
  "zip": "fidim-dg-overlay-b11030-mix-5ff778e-windows-x64.zip",
  "base": {
    "repo": "unslothai/llama.cpp",
    "release_tag": "b11030-mix-5ff778e",
    "source_commit": "6ba30d05b140ebb0baeded27d7d9b843c5b71ff1",
    "source_asset": "llama.cpp-source-commit-6ba30d05b140ebb0baeded27d7d9b843c5b71ff1.tar.gz",
    "source_sha256": "44cca07c...adfc",
    "upstream_tag": "b11030",
    "ggml_tree": "677df46a6800d1070bb787aaded017229583ce9b",
    "zips": { "gfx120X": "c780a9bd...9557", "gfx1151": "22016e77...82e8", "...": "..." }
  },
  "patch": {
    "file": "dgpatch5.diff",
    "sha256": "2822f06e...131f",
    "features": ["dg-pkv-f16", "dg-swa-ring", "dg-fa-pad", "dg-fa-turn-sizing",
                 "dg-step-fail-err", "dg-frame-special", "dg-prefill-reuse", "dg-sc-splitk"]
  },
  "build": {
    "runner": "github-actions win22 20260914.1",
    "toolchain": "clang",
    "c_compiler": "Clang 19.1.5", "cxx_compiler": "Clang 19.1.5",
    "msvc_toolset": "14.44.35207", "vs_version": "17.14.x", "cmake": "3.31.6-msvc6",
    "cmake_flags": ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "..."],
    "boringssl": "fetch",
    "workflow_run_url": "https://github.com/<repo>/actions/runs/<id>",
    "overlay_repo_sha": "<commit of this repository>",
    "built_at": "2026-09-19T00:24:18Z"
  },
  "files": [
    { "name": "llama.dll", "sha256": "81884604...0fee", "size": 2753024 },
    { "name": "licenses/LICENSE-llama.cpp", "sha256": "...", "size": 1078 }
  ],
  "ggml_imports": { "ggml.dll": ["ggml_backend_dev_by_type", "..."], "ggml-base.dll": ["..."] },
  "signer": null
}
```

(Hashes shortened here; the file has them in full. The `build` values are
illustrative.)

## Fields

| Field | Meaning |
| --- | --- |
| `schema` | 1. A reader refuses any other value. |
| `name` | The patch. Llama FIDIM installs only the patch it was built for (`OVERLAY_PATCH`). |
| `overlay_repo`, `release_tag`, `zip` | Where the overlay was published and its zip's name. Informational. |
| `base.repo` | Always `unslothai/llama.cpp`. |
| `base.release_tag` | The Unsloth release the overlay was built from and fits. |
| `base.source_commit` | The full commit of that release's source tarball. `llama-server --version` prints its first nine characters. |
| `base.source_asset`, `base.source_sha256` | The source tarball, as checked against GitHub's digest and the release's `llama-prebuilt-sha256.json` (`exact-source`). |
| `base.upstream_tag`, `base.ggml_tree` | From the release's `llama-prebuilt-manifest.json`. Informational. |
| `base.zips` | GPU target to the sha256 of that release's `app-<tag>-windows-x64-rocm-<gfx>.zip`, for every Windows ROCm zip (from `llama-prebuilt-sha256.json`). One overlay serves them all: they share the source commit. |
| `patch.file`, `patch.sha256` | The patch the binaries were built with, published beside the zip. |
| `patch.features` | What the patch changes, by Llama FIDIM's `discovery::dg_feature` names. Recorded in the build's manifest; the VRAM estimate, pre-flight and promotion follow them. |
| `build` | How it was built. Informational. |
| `files` | Every entry of the zip with its sha256 and size: the binaries at the top level (they go into the build's `bin`), license texts under `licenses/`. |
| `ggml_imports` | What the overlay imports from each of Unsloth's ggml DLLs, by name (the gate checked each against the base's exports). Informational. |
| `signer` | The Authenticode subject every binary is signed with, or `null` when unsigned. |

## What Llama FIDIM checks

Before the Unsloth zip is downloaded:
- the descriptor and the zip match GitHub's digests (a published overlay
  asset without a digest is refused);
- `schema` is 1, `name` is the patch it installs, `base.repo` is
  `unslothai/llama.cpp` and `base.release_tag` is the release being
  installed;
- `base.zips` has an entry for the GPU target;
- every file name is a llama-level binary (`llama*.exe`, `llama*.dll`,
  `mtmd.dll`; never `ggml*`, `amd*`, `hip*`, `roc*`, `origami*`, `lib*`) or a
  plain file under `licenses/`, listed once, and the runner,
  `llama-server.exe` and `llama.dll` are among them.

Then:
- the Unsloth zip matches GitHub's digest and `base.zips[gfx]`;
- the patch, when published, matches `patch.sha256`;
- the overlay zip holds exactly the listed files, and each one's sha256 and
  size match as it is written;
- after the install, `llama-server --version` prints a prefix of
  `base.source_commit`; otherwise the build is removed.

A local overlay (`--overlay-from`) has no GitHub digests to match; every
other check applies.

The build lands in `<install_root>\<tag>-unsloth-dgpatch5` with the
descriptor, the patch and the license texts (`overlay-licenses\`) beside
`bin`, and a manifest whose `patch` block is `{name, base_commit,
features, patch_sha256}`.
