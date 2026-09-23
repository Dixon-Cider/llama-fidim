# GitHub API captures

Responses from api.github.com on 2026-09-18, trimmed to the fields the
parsers in `compat::github` and `compat::plan` read (the compare's file list
keeps names only). Contributor user names in commit messages and fork names
are replaced with placeholders.

| file | request |
| --- | --- |
| `repo-mbzuai-ifm-llama.cpp.json` | `GET /repos/MBZUAI-IFM/llama.cpp` (answered through a 301 by ifm-ai/llama.cpp) |
| `compare-master-ifm-ai-42adf019.json` | `GET /repos/ggml-org/llama.cpp/compare/master...ifm-ai:llama.cpp:42adf019...?per_page=10` |
| `pull-27752.json` | `GET /repos/ggml-org/llama.cpp/pulls/27752` |
| `search-prs-inkling.json` | `GET /search/issues?q=is:pr repo:ggml-org/llama.cpp inkling` |
| `search-prs-k2-horizon.json` | `GET /search/issues?q=is:pr repo:ggml-org/llama.cpp "k2-horizon"` |
| `releases-upstream-b11046.json` | `GET /repos/ggml-org/llama.cpp/releases?per_page=5` (first two, Windows assets only) |
| `release-unsloth-b11030.json` | `GET /repos/unslothai/llama.cpp/releases?per_page=1` (Windows ROCm assets only) |
