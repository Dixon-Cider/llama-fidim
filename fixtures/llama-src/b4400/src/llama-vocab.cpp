// Trimmed excerpt of ggml-org/llama.cpp at tag b4400 (from before src/llama-arch.cpp existed), src/llama-vocab.cpp, lines 1-12.
// llama.cpp is MIT licensed, Copyright (c) 2023-2026 The ggml authors; see ../README.md.
#include "llama-vocab.h"

#include "unicode.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cstdarg>
#include <cstring>
#include <forward_list>
#include <queue>
