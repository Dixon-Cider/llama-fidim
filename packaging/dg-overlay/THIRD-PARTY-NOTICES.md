# Third-party notices

This archive is the Llama FIDIM {{PATCH}} overlay for Unsloth's llama.cpp
release {{BASE_TAG}}: the llama-level binaries (llama.dll, llama-common.dll,
mtmd.dll, the tools and the DiffusionGemma runner) built from that release's
published source, commit {{SOURCE_COMMIT}}, with the {{PATCH}} patch applied.

It is installed over Unsloth's Windows ROCm zip of the same release. The
ggml DLLs, the HIP backend and AMD's ROCm libraries in that zip are not part
of this archive and are not modified; their notices are Unsloth's and AMD's.

| Component | License | Text in this folder |
| --- | --- | --- |
| llama.cpp and ggml (the ggml authors), including the changes Unsloth merged into its release source | MIT | LICENSE-llama.cpp |
| nlohmann/json | MIT | LICENSE-jsonhpp |
| cpp-httplib | MIT | LICENSE-cpp-httplib |
| xxHash | BSD 2-Clause | LICENSE-xxhash |
| rotate-bits | MIT | LICENSE-rotate-bits.md |
| sha1, sha256 (llama.cpp vendor/hash) | Public domain | LICENSE-sha256 |
| stb_image, miniaudio | Public domain or MIT, at the user's choice | in their headers in the llama.cpp source |
| subprocess.h (sheredom) | The Unlicense | in its header in the llama.cpp source |
| BoringSSL | OpenSSL / ISC-style; {{BORINGSSL}} | LICENSE-boringssl |
| The {{PATCH}} patch and the overlay build scripts | MIT, Copyright (c) 2026 Dixon-Cider | LICENSE-fidim-dg-overlay |

llama-server embeds llama.cpp's web UI (tools/ui in the llama.cpp source,
MIT), which the build downloads prebuilt from ggml-org's llama-ui bucket on
Hugging Face, as upstream builds do.
