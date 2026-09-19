# llama.cpp source excerpts

Trimmed copies of the three tables `compat::probe_source` reads, captured on
2026-09-18 from raw.githubusercontent.com, used by the parser tests:

| folder | source |
| --- | --- |
| `b11046/` | ggml-org/llama.cpp, tag b11046 (commit 60081bb2b5b3), the newest release that day |
| `ifm-ai-42adf019/` | ifm-ai/llama.cpp (formerly MBZUAI-IFM), branch model/K2Horizon at 42adf019f760: adds the `k2-horizon` architecture and pre-tokenizer |
| `b4400/` | ggml-org/llama.cpp, tag b4400: the older layout with both tables in `src/llama.cpp` |

Each file keeps only the lines named in its first comment: the
`LLM_ARCH_NAMES` table, the BPE pre-tokenizer dispatch of the vocab loader,
and `enum ggml_type`.

llama.cpp is distributed under the MIT License:

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
