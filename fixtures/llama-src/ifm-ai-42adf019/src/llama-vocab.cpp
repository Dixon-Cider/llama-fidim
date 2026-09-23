// Trimmed excerpt of ifm-ai/llama.cpp at 42adf019f760 (branch model/K2Horizon), src/llama-vocab.cpp, lines 2133-2396.
// llama.cpp is MIT licensed, Copyright (c) 2023-2026 The ggml authors; see ../README.md.
        // for now, only BPE models have pre-tokenizers
        if (type == LLAMA_VOCAB_TYPE_BPE) {
            add_space_prefix = false;
            escape_whitespaces = false;
            clean_spaces = true;
            if (tokenizer_pre.empty()) {
                LLAMA_LOG_WARN("%s: missing pre-tokenizer type, using: 'default'\n", __func__);
                LLAMA_LOG_WARN("%s:                                             \n", __func__);
                LLAMA_LOG_WARN("%s: ************************************        \n", __func__);
                LLAMA_LOG_WARN("%s: GENERATION QUALITY WILL BE DEGRADED!        \n", __func__);
                LLAMA_LOG_WARN("%s: CONSIDER REGENERATING THE MODEL             \n", __func__);
                LLAMA_LOG_WARN("%s: ************************************        \n", __func__);
                LLAMA_LOG_WARN("%s:                                             \n", __func__);
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (tokenizer_pre == "default") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (tokenizer_pre == "minicpm5") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_MINICPM5;
                ignore_merges = true;
            } else if (
                    tokenizer_pre == "llama3"   ||
                    tokenizer_pre == "llama-v3" ||
                    tokenizer_pre == "llama-bpe"||
                    tokenizer_pre == "falcon3"  ||
                    tokenizer_pre == "falcon-h1" ||
                    tokenizer_pre == "pixtral"  ||
                    tokenizer_pre == "midm-2.0" ||
                    tokenizer_pre == "lfm2"     ||
                    tokenizer_pre == "jina-v5-nano") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_LLAMA3;
                ignore_merges = true;
                add_bos = true;
            } else if (
                    tokenizer_pre == "deepseek-llm") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-coder") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-v3") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "youtu") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_YOUTU;
                clean_spaces = false;
                ignore_merges = true;
            } else if (
                    tokenizer_pre == "falcon") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_FALCON;
            } else if (
                    tokenizer_pre == "mpt") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_MPT;
            } else if (
                    tokenizer_pre == "starcoder") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_STARCODER;
            } else if (
                    tokenizer_pre == "gpt-2"   ||
                    tokenizer_pre == "phi-2"   ||
                    tokenizer_pre == "jina-es" ||
                    tokenizer_pre == "jina-de" ||
                    tokenizer_pre == "gigachat"   ||
                    tokenizer_pre == "jina-v2-es" ||
                    tokenizer_pre == "jina-v2-de" ||
                    tokenizer_pre == "a.x-4.0" ||
                    tokenizer_pre == "mellum"  ||
                    tokenizer_pre == "modern-bert") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT2;
            } else if (
                    tokenizer_pre == "jais-2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_JAIS2;
            } else if (
                    tokenizer_pre == "gemma4" ||
                    tokenizer_pre == "granite-embed-multi-311m") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GEMMA4;
                escape_whitespaces = true;
            } else if (
                    tokenizer_pre == "sarvam-moe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_SARVAM_MOE;
                escape_whitespaces = true;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "jina-v1-en" ||
                    tokenizer_pre == "jina-v2-code" ||
                    tokenizer_pre == "roberta-bpe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT2;
                add_sep = true;
            } else if (
                    tokenizer_pre == "whitespace") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_WHITESPACE;
                normalizer_opts.lowercase = false;
            } else if (
                    tokenizer_pre == "refact") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_REFACT;
            } else if (
                tokenizer_pre == "command-r") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_COMMAND_R;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "qwen2" ||
                    tokenizer_pre == "deepseek-r1-qwen" ||
                    tokenizer_pre == "kormo" ||
                    tokenizer_pre == "f2llmv2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_QWEN2;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "qwen35") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_QWEN35;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "stablelm2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_STABLELM2;
            } else if (
                tokenizer_pre == "olmo") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_OLMO;
            } else if (
                tokenizer_pre == "dbrx") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DBRX;
            } else if (
                tokenizer_pre == "smaug-bpe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_SMAUG;
            } else if (
                tokenizer_pre == "poro-chat") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_PORO;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "glm4" ||
                tokenizer_pre == "chatglm-bpe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_CHATGLM4;
                special_bos_id = LLAMA_TOKEN_NULL;
            } else if (
                tokenizer_pre == "viking") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_VIKING;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "jais") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_JAIS;
            } else if (
                tokenizer_pre == "tekken") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_TEKKEN;
                clean_spaces = false;
                ignore_merges = true;
                add_bos = true;
            } else if (
                tokenizer_pre == "smollm") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_SMOLLM;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "codeshell") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_CODESHELL;
            } else if (
                tokenizer_pre == "bloom") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_BLOOM;
            } else if (
                tokenizer_pre == "gpt3-finnish") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH;
            } else if (
                tokenizer_pre == "exaone") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_EXAONE;
            } else if (
                tokenizer_pre == "exaone4") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT2;
            } else if (
                tokenizer_pre == "exaone-moe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_EXAONE_MOE;
            } else if (
                tokenizer_pre == "chameleon") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_CHAMELEON;
                add_bos = true;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "minerva-7b") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_MINERVA;
            } else if (
                tokenizer_pre == "megrez") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_QWEN2;
            } else if (
                tokenizer_pre == "gpt-4o" ||
                tokenizer_pre == "llama4" ||
                tokenizer_pre == "kanana2" ||
                tokenizer_pre == "talkie") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT4O;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "granite-embed-multi-97m") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GRANITE_EMB_MULTI;
                clean_spaces = false;
                ignore_merges = true;
            } else if (
                tokenizer_pre == "tiny_aya" ||
                tokenizer_pre == "cohere2moe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_TINY_AYA;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "superbpe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_SUPERBPE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "trillion") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_TRILLION;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "granite-docling") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GRANITE_DOCLING;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "bailingmoe" ||
                tokenizer_pre == "bailingmoe2" ||
                tokenizer_pre == "llada-moe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_BAILINGMOE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "seed-coder") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_SEED_CODER;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "hunyuan") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_HUNYUAN;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "hunyuan-dense") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "joyai-llm") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "kimi-k2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_KIMI_K2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "grok-2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GROK_2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "afmoe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_AFMOE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "laguna") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_LAGUNA;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "minimax-m2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "solar-open") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_SOLAR_OPEN;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "mellum2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_MELLUM2;
            } else if (
                tokenizer_pre == "k2-horizon") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_K2_HORIZON;
                clean_spaces = false;
            } else {
                throw std::runtime_error(format("unknown pre-tokenizer type: '%s'", tokenizer_pre.c_str()));
            }
        } else if (type == LLAMA_VOCAB_TYPE_SPM) {
