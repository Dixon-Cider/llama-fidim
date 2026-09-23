// Trimmed excerpt of ggml-org/llama.cpp at tag b4400 (from before src/llama-arch.cpp existed), src/llama.cpp, lines 205-261 and 6588-6714.
// llama.cpp is MIT licensed, Copyright (c) 2023-2026 The ggml authors; see ../README.md.
static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {
    { LLM_ARCH_LLAMA,            "llama"            },
    { LLM_ARCH_DECI,             "deci"            },
    { LLM_ARCH_FALCON,           "falcon"           },
    { LLM_ARCH_GROK,             "grok"             },
    { LLM_ARCH_GPT2,             "gpt2"             },
    { LLM_ARCH_GPTJ,             "gptj"             },
    { LLM_ARCH_GPTNEOX,          "gptneox"          },
    { LLM_ARCH_MPT,              "mpt"              },
    { LLM_ARCH_BAICHUAN,         "baichuan"         },
    { LLM_ARCH_STARCODER,        "starcoder"        },
    { LLM_ARCH_REFACT,           "refact"           },
    { LLM_ARCH_BERT,             "bert"             },
    { LLM_ARCH_NOMIC_BERT,       "nomic-bert"       },
    { LLM_ARCH_JINA_BERT_V2,     "jina-bert-v2"     },
    { LLM_ARCH_BLOOM,            "bloom"            },
    { LLM_ARCH_STABLELM,         "stablelm"         },
    { LLM_ARCH_QWEN,             "qwen"             },
    { LLM_ARCH_QWEN2,            "qwen2"            },
    { LLM_ARCH_QWEN2MOE,         "qwen2moe"         },
    { LLM_ARCH_QWEN2VL,          "qwen2vl"          },
    { LLM_ARCH_PHI2,             "phi2"             },
    { LLM_ARCH_PHI3,             "phi3"             },
    { LLM_ARCH_PLAMO,            "plamo"            },
    { LLM_ARCH_CODESHELL,        "codeshell"        },
    { LLM_ARCH_ORION,            "orion"            },
    { LLM_ARCH_INTERNLM2,        "internlm2"        },
    { LLM_ARCH_MINICPM,          "minicpm"          },
    { LLM_ARCH_MINICPM3,         "minicpm3"         },
    { LLM_ARCH_GEMMA,            "gemma"            },
    { LLM_ARCH_GEMMA2,           "gemma2"           },
    { LLM_ARCH_STARCODER2,       "starcoder2"       },
    { LLM_ARCH_MAMBA,            "mamba"            },
    { LLM_ARCH_XVERSE,           "xverse"           },
    { LLM_ARCH_COMMAND_R,        "command-r"        },
    { LLM_ARCH_DBRX,             "dbrx"             },
    { LLM_ARCH_OLMO,             "olmo"             },
    { LLM_ARCH_OLMO2,            "olmo2"            },
    { LLM_ARCH_OLMOE,            "olmoe"            },
    { LLM_ARCH_OPENELM,          "openelm"          },
    { LLM_ARCH_ARCTIC,           "arctic"           },
    { LLM_ARCH_DEEPSEEK,         "deepseek"         },
    { LLM_ARCH_DEEPSEEK2,        "deepseek2"        },
    { LLM_ARCH_CHATGLM,          "chatglm"          },
    { LLM_ARCH_BITNET,           "bitnet"           },
    { LLM_ARCH_T5,               "t5"               },
    { LLM_ARCH_T5ENCODER,        "t5encoder"        },
    { LLM_ARCH_JAIS,             "jais"             },
    { LLM_ARCH_NEMOTRON,         "nemotron"         },
    { LLM_ARCH_EXAONE,           "exaone"           },
    { LLM_ARCH_RWKV6,            "rwkv6"            },
    { LLM_ARCH_GRANITE,          "granite"          },
    { LLM_ARCH_GRANITE_MOE,      "granitemoe"       },
    { LLM_ARCH_CHAMELEON,        "chameleon"        },
    { LLM_ARCH_WAVTOKENIZER_DEC, "wavtokenizer-dec" },
    { LLM_ARCH_UNKNOWN,          "(unknown)"        },
};

// [...]

        // for now, only BPE models have pre-tokenizers
        if (vocab.type == LLAMA_VOCAB_TYPE_BPE) {
            vocab.tokenizer_add_space_prefix = false;
            vocab.tokenizer_clean_spaces = true;
            if (tokenizer_pre.empty()) {
                LLAMA_LOG_WARN("%s: missing pre-tokenizer type, using: 'default'\n", __func__);
                LLAMA_LOG_WARN("%s:                                             \n", __func__);
                LLAMA_LOG_WARN("%s: ************************************        \n", __func__);
                LLAMA_LOG_WARN("%s: GENERATION QUALITY WILL BE DEGRADED!        \n", __func__);
                LLAMA_LOG_WARN("%s: CONSIDER REGENERATING THE MODEL             \n", __func__);
                LLAMA_LOG_WARN("%s: ************************************        \n", __func__);
                LLAMA_LOG_WARN("%s:                                             \n", __func__);
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (tokenizer_pre == "default") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (
                    tokenizer_pre == "llama3"   ||
                    tokenizer_pre == "llama-v3" ||
                    tokenizer_pre == "llama-bpe"||
                    tokenizer_pre == "falcon3") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_LLAMA3;
                vocab.tokenizer_ignore_merges = true;
                vocab.tokenizer_add_bos = true;
            } else if (
                    tokenizer_pre == "deepseek-llm") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-coder") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                    tokenizer_pre == "falcon") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_FALCON;
            } else if (
                    tokenizer_pre == "mpt") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_MPT;
            } else if (
                    tokenizer_pre == "starcoder") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_STARCODER;
            } else if (
                    tokenizer_pre == "gpt-2"   ||
                    tokenizer_pre == "phi-2"   ||
                    tokenizer_pre == "jina-es" ||
                    tokenizer_pre == "jina-de" ||
                    tokenizer_pre == "gigachat"   ||
                    tokenizer_pre == "jina-v1-en" ||
                    tokenizer_pre == "jina-v2-es" ||
                    tokenizer_pre == "jina-v2-de" ||
                    tokenizer_pre == "jina-v2-code" ||
                    tokenizer_pre == "roberta-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_GPT2;
            } else if (
                    tokenizer_pre == "refact") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_REFACT;
            } else if (
                tokenizer_pre == "command-r") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_COMMAND_R;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "qwen2") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_QWEN2;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "stablelm2") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_STABLELM2;
            } else if (
                tokenizer_pre == "olmo") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_OLMO;
            } else if (
                tokenizer_pre == "dbrx") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_DBRX;
            } else if (
                tokenizer_pre == "smaug-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_SMAUG;
            } else if (
                tokenizer_pre == "poro-chat") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_PORO;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "chatglm-bpe") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CHATGLM4;
                vocab.special_bos_id = LLAMA_TOKEN_NULL;
            } else if (
                tokenizer_pre == "viking") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_VIKING;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "jais") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_JAIS;
            } else if (
                tokenizer_pre == "tekken") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_TEKKEN;
                vocab.tokenizer_clean_spaces = false;
                vocab.tokenizer_ignore_merges = true;
                vocab.tokenizer_add_bos = true;
            } else if (
                tokenizer_pre == "smollm") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_SMOLLM;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "codeshell") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CODESHELL;
            } else if (
                tokenizer_pre == "bloom") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_BLOOM;
            } else if (
                tokenizer_pre == "gpt3-finnish") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH;
            } else if (
                tokenizer_pre == "exaone") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_EXAONE;
            } else if (
                tokenizer_pre == "chameleon") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_CHAMELEON;
                vocab.tokenizer_add_bos = true;
                vocab.tokenizer_clean_spaces = false;
            } else if (
                tokenizer_pre == "minerva-7b") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_MINERVA;
            } else if (
                tokenizer_pre == "megrez") {
                vocab.type_pre = LLAMA_VOCAB_PRE_TYPE_QWEN2;
            } else {
                throw std::runtime_error(format("unknown pre-tokenizer type: '%s'", tokenizer_pre.c_str()));
            }
        } else if (vocab.type == LLAMA_VOCAB_TYPE_SPM) {
