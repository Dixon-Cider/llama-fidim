//! The SGLang engine (Linux): profile settings, launch plan, pre-flight,
//! launch, and the model-name router that fronts several servers.
//!
//! SGLang is a python server (`python -m sglang.launch_server`) from a venv,
//! one card per instance (`HIP_VISIBLE_DEVICES`). It has no `/slots`; the
//! router (`model_router.py` in the tools dir) is what answers FIDIM's live
//! view in llama-server's shape, so every member sits behind the router and
//! FIDIM launches members first, then the router.
//!
//! Everything box-specific (venv, tools dir, ROCm env) lives in
//! `Config.sglang`; everything model-specific in `Profile.sglang`.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::devices::Device;
use crate::launch::LaunchPlan;
use crate::platform::Platform;
use crate::preflight::{CheckResult, Outcome};
use crate::profile::{Engine, Profile};
use crate::supervise::{self, RunState};
use crate::{Error, Result};

// ------------------------------------------------------------- settings ----

/// Where SGLang lives on this host (`~/.fidim/config.json` → `"sglang"`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SgLangHost {
    /// The venv with sglang and torch installed (its `bin/python` launches everything).
    pub venv: PathBuf,
    /// Where the router script and the memory guard live. None = FIDIM writes
    /// its bundled copies to `<config dir>/sglang/` and uses those; set it to
    /// use your own (e.g. a checkout with a tuned `tunableop.csv`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools_dir: Option<PathBuf>,
    /// Extra `PYTHONPATH` entries (the AOT kernel build dir).
    #[serde(default)]
    pub pythonpath: Vec<PathBuf>,
    /// Environment every server and the router get (GPU_ARCHS, SGLANG_USE_AITER, ...).
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Working directory for the servers (TunableOp and profiler paths are
    /// relative to it in the scripts); default: the tools dir's parent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
}

/// The scripts FIDIM ships for the SGLang engine, written to the tools dir
/// when it has no copy (or an older one).
const BUNDLED: &[(&str, &str)] = &[
    ("model_router.py", include_str!("../assets/sglang/model_router.py")),
    ("memguard/sitecustomize.py", include_str!("../assets/sglang/sitecustomize.py")),
];

impl SgLangHost {
    pub fn python(&self) -> PathBuf {
        self.venv.join("bin").join("python")
    }
    /// The tools dir in use: the configured one, else `<config dir>/sglang`.
    pub fn tools(&self) -> PathBuf {
        self.tools_dir.clone().unwrap_or_else(|| Config::config_dir().join("sglang"))
    }
    /// Make sure the bundled scripts exist in the tools dir. A configured
    /// tools dir is left alone when it already has them (the user's copies
    /// win); the default dir is kept in sync with this build.
    pub fn ensure_tools(&self) -> Result<PathBuf> {
        let dir = self.tools();
        let managed = self.tools_dir.is_none();
        for (rel, body) in BUNDLED {
            let p = dir.join(rel);
            let stale = match std::fs::read_to_string(&p) {
                Ok(cur) => managed && cur != *body,
                Err(_) => true,
            };
            if stale {
                if let Some(parent) = p.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
                }
                std::fs::write(&p, body).map_err(|e| Error::io(&p, e))?;
            }
        }
        Ok(dir)
    }
    /// The ROCm SDK inside the venv (`_rocm_sdk_devel` wheel), if present.
    pub fn rocm_home(&self) -> Option<PathBuf> {
        let sp = self.venv.join("lib");
        let rd = std::fs::read_dir(&sp).ok()?;
        for e in rd.flatten() {
            let p = e.path().join("site-packages").join("_rocm_sdk_devel");
            if p.is_dir() {
                return Some(p);
            }
        }
        None
    }
    pub fn router_script(&self) -> PathBuf {
        self.tools().join("model_router.py")
    }
}

/// Speculative decoding (the MTP head inside the checkpoint, "NEXTN").
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SgSpec {
    #[serde(default = "d_nextn")]
    pub algorithm: String,
    #[serde(default = "d_steps")]
    pub steps: u32,
    #[serde(default = "d_topk")]
    pub topk: u32,
    #[serde(default = "d_draft")]
    pub draft_tokens: u32,
    /// `--speculative-token-map`: a torch-saved id tensor restricting the
    /// draft lm_head to a hot vocabulary (hot_tokens.py).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_map: Option<PathBuf>,
    /// `--speculative-draft-model-path`: the separate draft checkpoint EAGLE,
    /// EAGLE3, STANDALONE and DFLASH need (NEXTN and NGRAM take none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_model: Option<PathBuf>,
    /// `--speculative-draft-window-size`: positions a DFLASH draft attends to. With the
    /// gfx1201 draft ring (sglang-gfx1201 dflash patch) its KV pool is then window-sized
    /// instead of one slot per target token.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_window: Option<u32>,
    /// `--speculative-draft-kv-cache-dtype`: the draft's own KV precision (default: the target's).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_kv_cache_dtype: Option<String>,
}

impl SgSpec {
    /// Algorithms that load a second checkpoint from `draft_model`.
    pub fn needs_draft_model(&self) -> bool {
        matches!(self.algorithm.to_ascii_uppercase().as_str(), "EAGLE" | "EAGLE3" | "STANDALONE" | "DFLASH")
    }
    /// DFLASH drafts a whole block per pass with a block-diffusion model; SGLang derives
    /// its steps and top-k from the block size (`draft_tokens`).
    pub fn is_dflash(&self) -> bool {
        self.algorithm.eq_ignore_ascii_case("DFLASH")
    }
    pub fn is_ngram(&self) -> bool {
        self.algorithm.eq_ignore_ascii_case("NGRAM")
    }
    pub fn is_nextn(&self) -> bool {
        self.algorithm.eq_ignore_ascii_case("NEXTN")
    }
}

fn d_nextn() -> String {
    "NEXTN".into()
}
fn d_steps() -> u32 {
    3
}
fn d_topk() -> u32 {
    1
}
fn d_draft() -> u32 {
    4
}

/// Per-profile SGLang settings (`Profile.sglang`). Context length is the
/// profile's `runtime.ctx_total`, the served model name its `server.alias`,
/// the card its single `devices[0]`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SgLangCfg {
    /// `--mem-fraction-static`: share of VRAM SGLang takes for weights + KV pool.
    /// A card that drives displays must leave >= 1.5 GB (see preflight).
    #[serde(default = "d_mem_fraction")]
    pub mem_fraction: f64,
    #[serde(default = "d_chunk")]
    pub chunked_prefill: u32,
    #[serde(default = "d_slots")]
    pub max_running_requests: u32,
    /// `--max-mamba-cache-size` for GatedDeltaNet models (slots x 5 by default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mamba_slots: Option<u32>,
    #[serde(default = "d_kv")]
    pub kv_cache_dtype: String,
    #[serde(default = "d_attn")]
    pub attention_backend: String,
    #[serde(default = "d_dtype")]
    pub dtype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spec: Option<SgSpec>,
    /// `--mamba-ssm-dtype`: precision of a hybrid model's recurrent state (and its
    /// verify snapshots); bfloat16 halves the state pool. Default: SGLang's (float32).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mamba_ssm_dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_parser: Option<String>,
    /// In-process host RSS guard per process (memguard/sitecustomize.py), GB.
    #[serde(default = "d_memguard")]
    pub memguard_gb: u32,
    #[serde(default = "d_true")]
    pub enable_metrics: bool,
    /// `--sleep-on-idle`: block in a socket poll between requests instead of
    /// spinning (SGLang's default burns a full core per idle scheduler).
    #[serde(default = "d_true")]
    pub sleep_on_idle: bool,
    /// `TORCHINDUCTOR_COMPILE_THREADS`: Triton/inductor kernel-compile worker
    /// processes. The default is one per CPU core, each a resident torch
    /// import; a small pool costs only first-launch compile time.
    #[serde(default = "d_compile_threads")]
    pub compile_threads: u32,
    /// Extra model names the router maps to this member (`ddg` -> `dd`).
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub extra_args: Vec<String>,
}

fn d_mem_fraction() -> f64 {
    0.85
}
fn d_chunk() -> u32 {
    8192
}
fn d_slots() -> u32 {
    2
}
fn d_kv() -> String {
    "fp8_e4m3".into()
}
fn d_attn() -> String {
    "triton".into()
}
fn d_dtype() -> String {
    "bfloat16".into()
}
fn d_memguard() -> u32 {
    12
}
fn d_compile_threads() -> u32 {
    2
}
fn d_true() -> bool {
    true
}

impl Default for SgLangCfg {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).expect("defaults")
    }
}

/// `--attention-backend` values with NVIDIA-only kernels.
pub const NVIDIA_ONLY_ATTENTION: &[&str] = &["flashinfer", "fa3", "fa4", "trtllm_mha", "trtllm_mla", "flashmla", "cutedsl_mla", "hpc_ops"];
/// AMD backends whose kernels target Instinct (gfx9) only.
pub const AMD_GFX9_ATTENTION: &[&str] = &["aiter", "wave"];

/// Minimum free VRAM a display-driving card must keep: below it the desktop's
/// allocations evict the compute process (KFD evicted_ms climbs, decode drops
/// to 3 tok/s — measured 2026-09-20/21 on the R9700 with two displays).
pub const DISPLAY_HEADROOM_BYTES: u64 = 3 * 1024 * 1024 * 1024 / 2;

// ------------------------------------------------------- model sizing ----

/// What the KV-cache estimate needs from a model, whichever format it is in.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ModelShape {
    pub weights_bytes: u64,
    pub attention_layers: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub label: String,
    /// fp32 recurrent state per request slot for hybrid GDN/SSM models
    /// (linear layers x ssm.inner_size x ssm.state_size x 4 B); None = unknown.
    pub state_bytes_per_slot: Option<u64>,
    /// Only some layers carry KV (GatedDeltaNet/SSM hybrid): SGLang gives the
    /// recurrent state its own pool (`--max-mamba-cache-size`).
    #[serde(default)]
    pub hybrid: bool,
    /// Built-in next-token predict layers (what NEXTN drafts with); None = unknown.
    #[serde(default)]
    pub nextn_layers: Option<u64>,
}

/// Fallback per-slot state when the model does not describe it: the 27B GDN value.
const DEFAULT_STATE_BYTES_PER_SLOT: u64 = 150 << 20;

impl ModelShape {
    /// From a GGUF header: `full_attention_interval` marks hybrid GDN models
    /// (only every N-th layer carries KV).
    pub fn from_gguf(h: &crate::gguf::GgufHeader, weights_bytes: u64) -> Option<Self> {
        let layers = h.block_count?;
        let attention_layers = match h.full_attention_interval {
            Some(i) if i > 1 => layers / i,
            _ => layers,
        };
        let kv_heads = h.head_count_kv.or(h.head_count)?;
        let head_dim = h.key_length.or_else(|| h.embedding_length.zip(h.head_count).filter(|(_, n)| *n > 0).map(|(e, n)| e / n))?;
        // Checked against SGLang's own "ssm_state size" log: 27B 151 MB, 35B-A3B 60-63 MB per slot.
        let state_bytes_per_slot = match (h.full_attention_interval, h.ssm_inner_size, h.ssm_state_size) {
            (Some(i), Some(inner), Some(state)) if i > 1 => Some((layers - attention_layers) * inner * state * 4),
            _ => None,
        };
        let hybrid = matches!(h.full_attention_interval, Some(i) if i > 1);
        Some(Self { weights_bytes, attention_layers, kv_heads, head_dim, label: h.architecture.clone().unwrap_or_default(), state_bytes_per_slot, hybrid, nextn_layers: h.nextn_predict_layers })
    }
    pub fn from_hf(i: &crate::discovery::HfModelInfo) -> Option<Self> {
        Some(Self {
            weights_bytes: i.weights_bytes,
            attention_layers: i.attention_layers.or(i.num_layers)?,
            kv_heads: i.num_kv_heads?,
            head_dim: i.head_dim?,
            label: i.architecture.clone().or_else(|| i.model_type.clone()).unwrap_or_default(),
            state_bytes_per_slot: None,
            hybrid: matches!((i.attention_layers, i.num_layers), (Some(a), Some(n)) if a < n),
            nextn_layers: None,
        })
    }
    /// Read the model a profile points at (a .gguf file or a checkpoint dir).
    pub fn of_model(path: &Path) -> Option<Self> {
        if path.is_dir() {
            return crate::discovery::hf_model_dir(path).and_then(|m| m.hf.as_ref().and_then(Self::from_hf));
        }
        let h = crate::gguf::read_header(path).ok()?;
        let mut bytes = h.file_size;
        // split GGUF: count every shard
        if let Some((prefix, _, _)) = path.file_name().and_then(|n| n.to_str()).and_then(crate::gguf::split_name) {
            if let Some(dir) = path.parent() {
                bytes = std::fs::read_dir(dir)
                    .map(|rd| rd.flatten().filter(|e| e.file_name().to_string_lossy().starts_with(prefix) && e.path().extension().is_some_and(|x| x == "gguf")).filter_map(|e| e.metadata().ok()).map(|m| m.len()).sum())
                    .unwrap_or(bytes);
            }
        }
        Self::from_gguf(&h, bytes)
    }
    /// Bytes of KV cache per token: 2 (K,V) x layers x heads x dim x elem.
    pub fn kv_bytes_per_token(&self, kv_dtype: &str) -> u64 {
        let cells = 2 * self.attention_layers * self.kv_heads * self.head_dim;
        match kv_dtype {
            // packed 4-bit plus one byte of scale per 16 values
            d if d.starts_with("fp4") || d == "nvfp4" => cells * 9 / 16,
            d if d.starts_with("fp8") || d.starts_with("e4m3") || d.starts_with("e5m2") || d == "int8" => cells,
            _ => cells * 2,
        }
    }
}

/// Activations, CUDA graphs and allocator cache SGLang grows on top of the
/// static pool under long-context load: the 27B at 0.85 held 31.1 GiB of
/// its card with a 104K-token prompt in flight (2026-09-26), 4 GiB over.
pub const RUNTIME_OVERHEAD_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// How the static pool divides up for a profile on a device.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PoolBudget {
    pub pool_bytes: u64,
    pub weights_bytes: u64,
    /// Draft head or draft model weights (speculative decoding).
    pub draft_bytes: u64,
    /// Recurrent-state pool of a hybrid model, spec scratch included.
    pub state_bytes: u64,
    /// State slots: the configured count, or SGLang's own sizing when auto.
    pub state_slots: Option<u64>,
    pub state_auto: bool,
    pub kv_budget_bytes: u64,
    pub kv_bytes_per_token: u64,
    /// Tokens the pool can hold in KV (SGLang's max_total_num_tokens, roughly).
    pub tokens_in_budget: u64,
    pub ctx: u64,
}

/// What loaded weights take on the card over their file size (dequant
/// scratch, torch context): 18.86 GB for a 17.2 GiB GGUF, 21.83 for 19.8.
const WEIGHTS_VRAM_NUM: u64 = 11;
const WEIGHTS_VRAM_DEN: u64 = 10;
/// An MTP draft head as SGLang loads it for NEXTN (1.07 GB measured on the 27B).
const NEXTN_DRAFT_BYTES: u64 = 1 << 30;
/// SGLang's `mamba_full_memory_ratio`: with no `--max-mamba-cache-size` it
/// gives the state pool 0.9 / 1.9 of what is left after the weights.
const MAMBA_AUTO_SHARE: f64 = 0.9 / 1.9;

/// SGLang's own arithmetic (kv_cache_configurator.py): it loads the weights,
/// keeps `1 - mem_fraction` of the card back, gives a hybrid model's state
/// pool its share, and fills the rest with KV. `others_bytes` is VRAM other
/// processes hold on the card now. Checked against both servers on this box:
/// 27B GGUF (fp4 KV, NEXTN, 10 slots) 204K predicted vs 208,379 allocated;
/// 35B-A3B (fp8 KV, 40 slots) 438K vs 428,128.
/// The split on ONE card. `tp` > 1 (one card per profile device) shards the
/// weights, the draft, the recurrent state and the KV heads across the cards.
pub fn pool_budget(s: &SgLangCfg, shape: &ModelShape, device: &Device, ctx: u64, others_bytes: u64, tp: u64) -> PoolBudget {
    let tp = tp.max(1);
    let pool = estimated_bytes(s, device);
    let weights = shape.weights_bytes / WEIGHTS_VRAM_DEN * WEIGHTS_VRAM_NUM / tp;
    let spec = s.spec.as_ref();
    let dflash_kv = spec.and_then(|sp| dflash_draft_kv(sp, s, tp));
    let draft = match spec {
        Some(sp) if sp.is_nextn() => NEXTN_DRAFT_BYTES,
        Some(sp) if sp.needs_draft_model() => sp.draft_model.as_deref().map(path_bytes).unwrap_or(0) / WEIGHTS_VRAM_DEN * WEIGHTS_VRAM_NUM,
        _ => 0,
    } / tp
        + dflash_kv.map_or(0, |k| k.fixed_bytes);
    // SGLang starts from what is free (the card minus other processes) and
    // keeps (1 - mem_fraction) of the whole card back.
    let base = (256u64 << 20).max(others_bytes);
    let rest = pool.saturating_sub(base).saturating_sub(weights).saturating_sub(draft);
    let draft_tokens = spec.map(|sp| sp.draft_tokens.max(1) as u64).unwrap_or(0);
    let (state_bytes, state_slots, state_auto) = if shape.hybrid {
        let per_slot = shape.state_bytes_per_slot.unwrap_or(DEFAULT_STATE_BYTES_PER_SLOT) / tp;
        // the state is fp32 unless --mamba-ssm-dtype says otherwise
        let per_slot = match s.mamba_ssm_dtype.as_deref() {
            Some("bfloat16" | "float16" | "bf16" | "fp16") => per_slot / 2,
            _ => per_slot,
        };
        match s.mamba_slots {
            Some(k) => {
                let k = k as u64;
                // +1 padding slot; spec verify keeps a snapshot per draft token
                // for every running request (bf16-ish, ~0.9 of the fp32 state).
                let scratch = if spec.is_some() {
                    per_slot * 9 / 10 * ((s.max_running_requests as u64).min(k / state_slots_per_request(s)) + 1) * draft_tokens
                } else {
                    0
                };
                ((k + 1) * per_slot + scratch, Some(k), false)
            }
            None => {
                let budget = (rest as f64 * MAMBA_AUTO_SHARE) as u64;
                let per = per_slot * (1 + draft_tokens);
                (budget, Some(budget / per.max(1)), true)
            }
        }
    } else {
        (0, None, false)
    };
    let kv_budget = rest.saturating_sub(state_bytes);
    let mut per_token = (shape.kv_bytes_per_token(&s.kv_cache_dtype) / tp).max(1);
    // NEXTN's draft layer keeps its own KV for every token of the pool.
    if spec.is_some_and(SgSpec::is_nextn) && shape.attention_layers > 0 {
        per_token = per_token * (shape.attention_layers + shape.nextn_layers.unwrap_or(1)) / shape.attention_layers;
    }
    // A DFLASH draft without a window keeps its KV per pool token too.
    per_token += dflash_kv.map_or(0, |k| k.per_token_bytes);
    PoolBudget {
        pool_bytes: pool,
        weights_bytes: weights,
        draft_bytes: draft,
        state_bytes,
        state_slots,
        state_auto,
        kv_budget_bytes: kv_budget,
        kv_bytes_per_token: per_token,
        tokens_in_budget: kv_budget / per_token,
        ctx,
    }
}

/// Draft KV cost of a DFLASH draft model on one card.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DflashDraftKv {
    /// Window-sized pool (`draft_window` set): a ring per request row plus the tail cache.
    pub fixed_bytes: u64,
    /// No window: one draft KV cell per target pool token.
    pub per_token_bytes: u64,
}

/// Tail-cache slots of the gfx1201 draft ring (SGLANG_DFLASH_DRAFT_TAIL_SLOTS default).
const DFLASH_DRAFT_TAIL_SLOTS: u64 = 32768;

/// From the draft checkpoint's config.json: layers x KV heads x head_dim, at the draft KV dtype.
/// None when the algorithm is not DFLASH or the config is unreadable.
pub fn dflash_draft_kv(sp: &SgSpec, s: &SgLangCfg, tp: u64) -> Option<DflashDraftKv> {
    if !sp.is_dflash() {
        return None;
    }
    let cfg: serde_json::Value = serde_json::from_slice(&std::fs::read(sp.draft_model.as_ref()?.join("config.json")).ok()?).ok()?;
    let n = |k: &str| cfg.get(k).and_then(serde_json::Value::as_u64);
    let layers = n("num_hidden_layers")?;
    let heads = n("num_attention_heads")?;
    let kv_heads = n("num_key_value_heads").unwrap_or(heads);
    let head_dim = n("head_dim").or_else(|| Some(n("hidden_size")? / heads.max(1)))?;
    let draft = ModelShape {
        weights_bytes: 0,
        attention_layers: layers,
        kv_heads,
        head_dim,
        label: String::new(),
        state_bytes_per_slot: None,
        hybrid: false,
        nextn_layers: None,
    };
    let cell = draft.kv_bytes_per_token(sp.draft_kv_cache_dtype.as_deref().unwrap_or(&s.kv_cache_dtype)) / tp.max(1);
    Some(match sp.draft_window {
        Some(w) => {
            // rows: the request pool (max running + spare); each ring holds window + 2 blocks
            let rows = s.max_running_requests as u64 + 2;
            let ring = w as u64 + 2 * sp.draft_tokens.max(1) as u64;
            DflashDraftKv { fixed_bytes: cell * (1 + rows * ring + DFLASH_DRAFT_TAIL_SLOTS), per_token_bytes: 0 }
        }
        None => DflashDraftKv { fixed_bytes: 0, per_token_bytes: cell },
    })
}

/// State slots SGLang holds per running request on a hybrid model
/// (`_calculate_mamba_ratio`): 3 plus a 2-slot ping-pong buffer under the
/// overlap scheduler; 1 without the radix cache. It caps
/// `--max-running-requests` at slots / ratio (qwenMoE with 24 slots came up
/// at 4 requests, 2026-09-26).
pub fn state_slots_per_request(s: &SgLangCfg) -> u64 {
    let has = |f: &str| s.extra_args.iter().any(|a| a == f);
    if has("--disable-radix-cache") {
        1
    } else if has("--disable-overlap-schedule") {
        4
    } else {
        5
    }
}

/// Bytes of a file, or of every file under a checkpoint folder.
fn path_bytes(p: &Path) -> u64 {
    if p.is_dir() {
        std::fs::read_dir(p)
            .map(|rd| rd.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum())
            .unwrap_or(0)
    } else {
        std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
    }
}

// ------------------------------------------------ installs (venvs) ----

/// A python environment that imports sglang.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SgLangInstall {
    pub venv: PathBuf,
    pub python: PathBuf,
    pub sglang_version: Option<String>,
    pub torch_version: Option<String>,
    /// `rocm <ver>`, `cuda <ver>` or None for CPU-only torch.
    pub device: Option<String>,
    pub configured: bool,
}

/// Run the venv's python and ask it about sglang and torch (~2 s).
pub fn probe_install(venv: &Path) -> Option<SgLangInstall> {
    let python = if venv.join("bin").join("python").is_file() {
        venv.join("bin").join("python")
    } else if venv.join("Scripts").join("python.exe").is_file() {
        venv.join("Scripts").join("python.exe")
    } else if venv.is_file() {
        venv.to_path_buf()
    } else {
        return None;
    };
    let mut cmd = std::process::Command::new(&python);
    cmd.args([
        "-c",
        "import json\ntry:\n import sglang; s=getattr(sglang,'__version__',None)\nexcept Exception: s=None\ntry:\n import torch; t=torch.__version__; d=('rocm '+torch.version.hip) if getattr(torch.version,'hip',None) else (('cuda '+torch.version.cuda) if torch.version.cuda else None)\nexcept Exception: t=None; d=None\nprint(json.dumps({'sglang':s,'torch':t,'device':d}))",
    ]);
    cmd.env("PYTHONWARNINGS", "ignore").stdin(std::process::Stdio::null());
    crate::launch::hide_console(&mut cmd);
    let out = cmd.output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().rev().find(|l| l.starts_with('{'))?;
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let sglang_version = v["sglang"].as_str().map(str::to_string)?;
    let venv_dir = if venv.is_file() { venv.parent().and_then(|p| p.parent()).unwrap_or(venv).to_path_buf() } else { venv.to_path_buf() };
    Some(SgLangInstall {
        venv: venv_dir,
        python,
        sglang_version: Some(sglang_version),
        torch_version: v["torch"].as_str().map(str::to_string),
        device: v["device"].as_str().map(str::to_string),
        configured: false,
    })
}

/// Candidate venv directories: the configured one, `roots`, `~/.venvs/*`,
/// `~/venv`, `~/.venv`, and every `*env*` or `*venv*` directory one level
/// under the home directory and under `~/Claude`-style project folders.
fn candidate_venvs(cfg: &Config, roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(h) = &cfg.sglang {
        out.push(h.venv.clone());
    }
    for r in roots {
        if r.join("bin").join("python").is_file() || r.join("Scripts").join("python.exe").is_file() {
            out.push(r.clone());
        } else if let Ok(rd) = std::fs::read_dir(r) {
            out.extend(rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()));
        }
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from);
    if let Some(home) = home {
        for sub in [".venvs", "venvs"] {
            if let Ok(rd) = std::fs::read_dir(home.join(sub)) {
                out.extend(rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()));
            }
        }
        for sub in ["venv", ".venv", "sglang-env", "sglang"] {
            out.push(home.join(sub));
        }
        // one and two levels down: <home>/*env* and <home>/*/*env*
        if let Ok(rd) = std::fs::read_dir(&home) {
            for e in rd.flatten().filter(|e| e.path().is_dir()) {
                let name = e.file_name().to_string_lossy().to_lowercase();
                if name.contains("env") {
                    out.push(e.path());
                }
                if !name.starts_with('.') {
                    if let Ok(rd2) = std::fs::read_dir(e.path()) {
                        for e2 in rd2.flatten().filter(|x| x.path().is_dir()) {
                            let n2 = e2.file_name().to_string_lossy().to_lowercase();
                            if n2.contains("env") {
                                out.push(e2.path());
                            }
                            if let Ok(rd3) = std::fs::read_dir(e2.path()) {
                                for e3 in rd3.flatten().filter(|x| x.path().is_dir()) {
                                    let n3 = e3.file_name().to_string_lossy().to_lowercase();
                                    if n3.contains("env") {
                                        out.push(e3.path());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    out.into_iter().filter(|p| (p.join("bin").join("python").is_file() || p.join("Scripts").join("python.exe").is_file()) && seen.insert(p.clone())).collect()
}

/// Every install found, probed in parallel; the configured one first.
pub fn discover_installs(cfg: &Config, roots: &[PathBuf]) -> Vec<SgLangInstall> {
    let cands = candidate_venvs(cfg, roots);
    let configured = cfg.sglang.as_ref().map(|h| h.venv.clone());
    let found: Vec<SgLangInstall> = std::thread::scope(|s| {
        let handles: Vec<_> = cands.iter().map(|c| s.spawn(move || probe_install(c))).collect();
        handles.into_iter().filter_map(|h| h.join().ok().flatten()).collect()
    });
    let mut out: Vec<SgLangInstall> = found
        .into_iter()
        .map(|mut i| {
            i.configured = configured.as_ref().is_some_and(|c| c == &i.venv);
            i
        })
        .collect();
    out.sort_by_key(|i| (!i.configured, i.venv.clone()));
    out
}

/// `python3 -m venv <dir>` then `pip install sglang` (+ the torch index for
/// the flavor). Streams pip's output to `progress`. Returns the venv path.
pub fn install(dir: &Path, flavor: &str, pip_args: &[String], progress: &mut dyn FnMut(String)) -> Result<PathBuf> {
    let python_sys = if cfg!(windows) { "python" } else { "python3" };
    if !dir.join("bin").join("python").is_file() && !dir.join("Scripts").join("python.exe").is_file() {
        progress(format!("creating venv {}", dir.display()));
        let st = std::process::Command::new(python_sys).args(["-m", "venv"]).arg(dir).status().map_err(|e| Error::Platform(format!("{python_sys} -m venv: {e}")))?;
        if !st.success() {
            return Err(Error::Platform(format!("{python_sys} -m venv {} failed", dir.display())));
        }
    }
    let python = if cfg!(windows) { dir.join("Scripts").join("python.exe") } else { dir.join("bin").join("python") };
    let mut args: Vec<String> = vec!["-m".into(), "pip".into(), "install".into(), "--upgrade".into(), "pip".into(), "sglang[all]".into()];
    match flavor {
        "rocm" => args.extend(["--extra-index-url".into(), "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1/".into()]),
        "cpu" => args.extend(["--extra-index-url".into(), "https://download.pytorch.org/whl/cpu".into()]),
        _ => {}
    }
    args.extend(pip_args.iter().cloned());
    progress(format!("{} {}", python.display(), args.join(" ")));
    let mut child = std::process::Command::new(&python)
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| Error::Platform(format!("pip: {e}")))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let err_thread = stderr.map(|e| std::thread::spawn(move || { use std::io::Read; let mut s = String::new(); let _ = std::io::BufReader::new(e).read_to_string(&mut s); s }));
    if let Some(o) = stdout {
        use std::io::BufRead;
        for line in std::io::BufReader::new(o).lines().map_while(std::result::Result::ok) {
            progress(line);
        }
    }
    let st = child.wait().map_err(|e| Error::Platform(format!("pip: {e}")))?;
    let err_text = err_thread.and_then(|t| t.join().ok()).unwrap_or_default();
    if !st.success() {
        return Err(Error::Platform(format!("pip install failed: {}", err_text.lines().rev().take(8).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join(" | "))));
    }
    probe_install(dir).map(|i| i.venv).ok_or_else(|| Error::Platform("installed, but the venv's python does not import sglang".into()))
}

pub fn host_of(cfg: &Config) -> Result<&SgLangHost> {
    cfg.sglang.as_ref().ok_or_else(|| Error::Config("config.json has no \"sglang\" section (venv, tools_dir)".into()))
}

pub fn cfg_of(p: &Profile) -> SgLangCfg {
    p.sglang.clone().unwrap_or_default()
}

// ---------------------------------------------------------------- plan ----

fn join_paths(parts: &[PathBuf]) -> String {
    parts.iter().map(|p| p.to_string_lossy().into_owned()).collect::<Vec<_>>().join(":")
}

/// The server command for `p` on `devices` (one tensor-parallel rank per card).
pub fn compose(cfg: &Config, p: &Profile, devices: &[&Device], runs_dir: &Path) -> Result<LaunchPlan> {
    let host = host_of(cfg)?;
    let s = cfg_of(p);
    if devices.is_empty() {
        return Err(Error::Config("no card to launch on".into()));
    }
    let model = p.model.path.to_string_lossy().into_owned();
    let mut args: Vec<String> = vec![
        "-m".into(),
        "sglang.launch_server".into(),
        "--model-path".into(),
        model,
        "--dtype".into(),
        s.dtype.clone(),
        "--attention-backend".into(),
        s.attention_backend.clone(),
        "--tp".into(),
        devices.len().to_string(),
        "--host".into(),
        p.server.host.clone(),
        "--port".into(),
        p.server.port.to_string(),
        "--served-model-name".into(),
        crate::router::model_id(p),
        "--mem-fraction-static".into(),
        format!("{:.2}", s.mem_fraction),
        "--context-length".into(),
        p.runtime.ctx_total.to_string(),
        "--max-running-requests".into(),
        s.max_running_requests.to_string(),
        "--kv-cache-dtype".into(),
        s.kv_cache_dtype.clone(),
        "--chunked-prefill-size".into(),
        s.chunked_prefill.to_string(),
    ];
    if let Some(m) = s.mamba_slots {
        args.extend(["--max-mamba-cache-size".into(), m.to_string()]);
    }
    if let Some(sp) = &s.spec {
        args.extend(["--speculative-algorithm".into(), sp.algorithm.clone()]);
        if !sp.is_dflash() {
            args.extend(["--speculative-num-steps".into(), sp.steps.to_string()]);
            // NGRAM sets its own top-k (the n-gram search breadth) and ignores ours.
            if !sp.is_ngram() {
                args.extend(["--speculative-eagle-topk".into(), sp.topk.to_string()]);
            }
        }
        args.extend(["--speculative-num-draft-tokens".into(), sp.draft_tokens.to_string()]);
        if let Some(m) = &sp.token_map {
            args.extend(["--speculative-token-map".into(), m.to_string_lossy().into_owned()]);
        }
        if let Some(m) = sp.draft_model.as_ref().filter(|_| sp.needs_draft_model()) {
            args.extend(["--speculative-draft-model-path".into(), m.to_string_lossy().into_owned()]);
            // SGLang loads the draft with the target's format and quantization: a safetensors
            // draft beside a GGUF target would be read as GGUF.
            let is_gguf = |p: &Path| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
            if is_gguf(&p.model.path) && !is_gguf(m.as_path()) {
                args.extend([
                    "--speculative-draft-load-format".into(),
                    "auto".into(),
                    "--speculative-draft-model-quantization".into(),
                    "unquant".into(),
                ]);
            }
        }
        if let Some(w) = sp.draft_window {
            args.extend(["--speculative-draft-window-size".into(), w.to_string()]);
        }
        if let Some(d) = &sp.draft_kv_cache_dtype {
            args.extend(["--speculative-draft-kv-cache-dtype".into(), d.clone()]);
        }
    }
    if let Some(d) = &s.mamba_ssm_dtype {
        args.extend(["--mamba-ssm-dtype".into(), d.clone()]);
    }
    if let Some(r) = &s.reasoning_parser {
        args.extend(["--reasoning-parser".into(), r.clone()]);
    }
    if let Some(t) = &s.tool_call_parser {
        args.extend(["--tool-call-parser".into(), t.clone()]);
    }
    if s.enable_metrics {
        args.push("--enable-metrics".into());
    }
    if s.sleep_on_idle {
        args.push("--sleep-on-idle".into());
    }
    args.extend(s.extra_args.iter().cloned());

    let visibility_env = devices.iter().map(|d| d.hip_index.to_string()).collect::<Vec<_>>().join(",");
    let mut env: Vec<(String, String)> = Vec::new();
    let rocm = host.rocm_home();
    if let Some(r) = &rocm {
        for k in ["ROCM_HOME", "ROCM_PATH", "HIP_PATH"] {
            env.push((k.into(), r.to_string_lossy().into_owned()));
        }
    }
    let tools = host.ensure_tools()?;
    let mut pythonpath = vec![tools.join("memguard")];
    pythonpath.extend(host.pythonpath.iter().cloned());
    env.push(("PYTHONPATH".into(), join_paths(&pythonpath)));
    env.push(("SGL_MEMGUARD_GB".into(), s.memguard_gb.to_string()));
    env.push((
        "SGL_MEMGUARD_LOG".into(),
        runs_dir.join(format!("memguard-{}-{}.log", p.id, p.server.port)).to_string_lossy().into_owned(),
    ));
    env.push(("SGLANG_MAMBA_CONV_DTYPE".into(), s.dtype.clone()));
    env.push(("TORCHINDUCTOR_COMPILE_THREADS".into(), s.compile_threads.max(1).to_string()));
    // A pre-tuned hipBLASLt/rocBLAS kernel table, when the tools dir has one
    // (tuning inside the server is pathological during graph capture).
    let tunable = tools.join("tunableop.csv");
    if tunable.is_file() {
        env.push(("PYTORCH_TUNABLEOP_ENABLED".into(), "1".into()));
        env.push(("PYTORCH_TUNABLEOP_TUNING".into(), "0".into()));
        env.push(("PYTORCH_TUNABLEOP_FILENAME".into(), tunable.to_string_lossy().into_owned()));
    }
    // Idle CPU: torch exports `rocprofiler_configure` (Kineto), so
    // rocprofiler-register loads rocprofiler-sdk at hsa_init; its 4096-signal
    // pool exhausts KFD's per-process event limit, one queue signal is left
    // without an interrupt event, and ROCr's AsyncEventsLoop polls instead of
    // sleeping -- one core at 100% per GPU process (launcher and scheduler)
    // while the server is idle (ROCm 7.14 / 10.0). Skipping the registration
    // costs only GPU kernel events in torch.profiler / `/start_profile`; the
    // host or profile env can set it back to 1 for a profiling run.
    env.push(("ROCPROFILER_REGISTER_ENABLED".into(), "0".into()));
    for (k, v) in &host.env {
        env.push((k.clone(), v.clone()));
    }
    // Tensor parallelism across cards without RCCL: card 1 here sits behind a chipset link
    // with no PCIe AtomicOps, where RCCL's kernels cannot be dispatched. hostcomm (the
    // tools-dir SGLang patch) all-reduces through pinned host memory instead.
    if devices.len() > 1 && !p.env.contains_key("SGLANG_HOSTCOMM") {
        env.push(("SGLANG_HOSTCOMM".into(), "1".into()));
    }
    for (k, v) in &p.env {
        env.push((k.clone(), v.clone()));
    }
    env.push(("HIP_VISIBLE_DEVICES".into(), visibility_env.clone()));
    // PATH: the ROCm bin (hipcc for JIT kernels) and the venv bin (ninja).
    let mut prepend = Vec::new();
    if let Some(r) = &rocm {
        prepend.push(r.join("bin"));
    }
    prepend.push(host.venv.join("bin"));
    let path = std::env::var("PATH").unwrap_or_default();
    env.push(("PATH".into(), format!("{}:{}", join_paths(&prepend), path)));

    Ok(LaunchPlan { exe: host.python(), args, env, path_prepend: None, visibility_env })
}

/// The `fidim` CLI binary, which carries the Rust router (`fidim router serve`):
/// this executable when it is the CLI, a sibling of the app, or one on PATH.
pub fn fidim_cli_exe() -> Option<PathBuf> {
    let name = if cfg!(windows) { "fidim.exe" } else { "fidim" };
    if let Some(p) = supervise::helper_exe(name) {
        return Some(p);
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(name)).find(|p| p.is_file())
}

/// The router command: one OpenAI port in front of the members. The Rust
/// router in the CLI binary when it can be found, else the bundled Python
/// script in the venv.
pub fn router_plan(cfg: &Config, rc: &crate::router::RouterConfig, members: &[&Profile]) -> Result<LaunchPlan> {
    let host = host_of(cfg)?;
    let rust = fidim_cli_exe();
    let mut args: Vec<String> = match &rust {
        Some(_) => vec!["router".into(), "serve".into()],
        None => {
            host.ensure_tools()?;
            vec![host.router_script().to_string_lossy().into_owned()]
        }
    };
    args.extend(["--host".into(), rc.host.clone(), "--port".into(), rc.port.to_string()]);
    let mut default: Option<String> = None;
    for m in members {
        let id = crate::router::model_id(m);
        args.extend(["--route".into(), format!("{id}=http://{}:{}", m.server.host, m.server.port)]);
        for a in &cfg_of(m).aliases {
            args.extend(["--alias".into(), format!("{a}={id}")]);
        }
        default.get_or_insert(id);
    }
    let default = default.ok_or_else(|| Error::Config("router has no members — add profiles first".into()))?;
    args.extend(["--default".into(), default]);
    let mut env: Vec<(String, String)> = host.env.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    env.push(("PATH".into(), format!("{}:{}", host.venv.join("bin").to_string_lossy(), std::env::var("PATH").unwrap_or_default())));
    let exe = rust.unwrap_or_else(|| host.python());
    Ok(LaunchPlan { exe, args, env, path_prepend: None, visibility_env: String::new() })
}

// ----------------------------------------------------------- pre-flight ----

fn check(id: &'static str, n: u8, title: &'static str, outcome: Outcome) -> CheckResult {
    CheckResult { id, spec_number: n, title, outcome }
}

/// Bytes the launch will take on the card: SGLang reserves `mem_fraction` of
/// the card's total for weights and pool, then grows ~3 GB of activations
/// and allocator cache under long-context load.
/// `--max-total-tokens N` from the profile's extra args. SGLang then allocates a KV pool of
/// N tokens instead of filling its whole static share, so two servers can share the cards
/// (dd + qwenMoE at --tp 2, both at share 0.95, launched in either order).
pub fn max_total_tokens(s: &SgLangCfg) -> Option<u64> {
    let a = &s.extra_args;
    a.windows(2)
        .find(|w| w[0] == "--max-total-tokens")
        .and_then(|w| w[1].parse().ok())
        .or_else(|| a.iter().find_map(|x| x.strip_prefix("--max-total-tokens=").and_then(|v| v.parse().ok())))
}

/// VRAM the server holds on ONE card once its pools are allocated: the static share, or,
/// with a token cap and a readable model, only weights + draft + state + the capped KV.
pub fn need_bytes(s: &SgLangCfg, device: &Device, shape: Option<&ModelShape>, ctx: u64, tp: u64) -> u64 {
    let share = estimated_bytes(s, device);
    match (max_total_tokens(s), shape) {
        (Some(cap), Some(shape)) => {
            let b = pool_budget(s, shape, device, ctx, 0, tp);
            ((256u64 << 20) + b.weights_bytes + b.draft_bytes + b.state_bytes + cap * b.kv_bytes_per_token).min(share)
        }
        _ => share,
    }
}

pub fn estimated_bytes(s: &SgLangCfg, device: &Device) -> u64 {
    (device.total_mib as f64 * 1024.0 * 1024.0 * s.mem_fraction) as u64
}

/// VRAM free on `d` for launching `p`: free now plus what a running
/// instance of the same profile holds (the launch replaces it).
pub fn free_for_launch(cfg: &Config, p: &Profile, d: &Device, platform: &dyn Platform) -> u64 {
    let mut free = d.free_mib * 1024 * 1024;
    for r in supervise::reattach(&cfg.runs_dir) {
        if r.alive && r.state.profile_id == p.id {
            let mut pids = vec![r.state.pid];
            pids.extend(crate::platform::process_descendants(r.state.pid));
            for pid in pids {
                for m in platform.gpu_process_memory(pid).unwrap_or_default() {
                    if Some(m.luid_low) == d.luid_low {
                        free += m.dedicated_bytes;
                    }
                }
            }
        }
    }
    free.min(d.total_mib * 1024 * 1024)
}

/// The pool split for `p` on `d` (None when the model cannot be read).
pub fn budget_for(cfg: &Config, p: &Profile, d: &Device, platform: &dyn Platform) -> Option<PoolBudget> {
    let shape = ModelShape::of_model(&p.model.path)?;
    let others = (d.total_mib * 1024 * 1024).saturating_sub(free_for_launch(cfg, p, d, platform));
    Some(pool_budget(&cfg_of(p), &shape, d, p.runtime.ctx_total, others, tp_of(p)))
}

pub fn preflight(
    cfg: &Config,
    p: &Profile,
    devices: Option<&[&Device]>,
    platform: &dyn Platform,
    running_aliases: &[String],
) -> Vec<CheckResult> {
    let s = cfg_of(p);
    let mut out = Vec::new();
    // Per-card checks run on every card of a tensor-parallel profile.
    let devs: Vec<&Device> = devices.map(<[&Device]>::to_vec).unwrap_or_default();
    let device = devs.first().copied();

    // 1: the engine runs
    out.push(match host_of(cfg) {
        Err(e) => check("build-runs", 1, "SGLang venv and tools exist", Outcome::Block(e.to_string())),
        Ok(h) => {
            let py = h.python();
            if !py.is_file() {
                check("build-runs", 1, "SGLang venv and tools exist", Outcome::Block(format!("no python at {}", py.display())))
            } else {
                match h.ensure_tools() {
                    Err(e) => check("build-runs", 1, "SGLang venv and tools exist", Outcome::Block(format!("tools dir: {e}"))),
                    Ok(dir) if !dir.join("model_router.py").is_file() => check("build-runs", 1, "SGLang venv and tools exist", Outcome::Block(format!("no model_router.py in {}", dir.display()))),
                    Ok(_) => check("build-runs", 1, "SGLang venv and tools exist", Outcome::Pass),
                }
            }
        }
    });

    // 2: model files
    let mp = &p.model.path;
    let mut missing = Vec::new();
    if !mp.exists() {
        missing.push(mp.to_string_lossy().into_owned());
    } else if mp.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
        // GGUF needs the sidecar config/tokenizer next to it (SGLang reads the
        // .gguf as the model and the directory as the HF config).
        if let Some(dir) = mp.parent() {
            for f in ["config.json", "tokenizer.json"] {
                if !dir.join(f).is_file() {
                    missing.push(dir.join(f).to_string_lossy().into_owned());
                }
            }
        }
    }
    let mut spec_problem = None;
    if let Some(sp) = &s.spec {
        if let Some(m) = &sp.token_map {
            if !m.is_file() {
                missing.push(m.to_string_lossy().into_owned());
            }
        }
        if sp.needs_draft_model() {
            match &sp.draft_model {
                None => spec_problem = Some(format!("{} needs a draft model; set one, or use NEXTN or NGRAM", sp.algorithm)),
                Some(m) if !m.exists() => missing.push(m.to_string_lossy().into_owned()),
                _ => {}
            }
        }
    }
    out.push(check(
        "files-exist",
        2,
        "Model, sidecar and draft files exist",
        match (missing.is_empty(), spec_problem) {
            (_, Some(m)) if missing.is_empty() => Outcome::Block(m),
            (true, _) => Outcome::Pass,
            (false, Some(m)) => Outcome::Block(format!("missing: {}; {m}", missing.join(", "))),
            (false, None) => Outcome::Block(format!("missing: {}", missing.join(", "))),
        },
    ));

    // 3: device resolves
    out.push(match device {
        Some(d) => check("devices-resolve", 3, "Device key resolves to a card", Outcome::Pass).with_note(d),
        None => check(
            "devices-resolve",
            3,
            "Device key resolves to a card",
            Outcome::Block(format!(
                "no card matches {}",
                p.devices.first().map(|d| d.key.as_str()).unwrap_or("(no device in profile)")
            )),
        ),
    });

    // 4: the attention kernel exists for this card's vendor
    for d in devs.iter().copied() {
        let b = s.attention_backend.as_str();
        let amd = d.stable_key.contains("VEN_1002");
        let nvidia = d.stable_key.contains("VEN_10DE");
        out.push(check(
            "attention-backend",
            4,
            "Attention kernel runs on this card",
            if amd && NVIDIA_ONLY_ATTENTION.contains(&b) {
                Outcome::Block(format!("{b} is an NVIDIA kernel; on {} use triton", d.name))
            } else if amd && AMD_GFX9_ATTENTION.contains(&b) && !d.name.contains("Instinct") {
                Outcome::Warn(format!("{b} is built for Instinct (gfx9) cards; on {} use triton", d.name))
            } else if nvidia && AMD_GFX9_ATTENTION.contains(&b) {
                Outcome::Block(format!("{b} is an AMD kernel; on {} use triton or flashinfer", d.name))
            } else {
                Outcome::Pass
            },
        ));
    }

    // 6: VRAM fits (free now + what our own previous instance holds)
    let mut others_bytes = 0u64;
    let shape6 = ModelShape::of_model(&p.model.path);
    let capped = max_total_tokens(&s).is_some();
    for d in devs.iter().copied() {
        let need = need_bytes(&s, d, shape6.as_ref(), p.runtime.ctx_total, tp_of(p));
        let free = free_for_launch(cfg, p, d, platform);
        others_bytes = others_bytes.max((d.total_mib * 1024 * 1024).saturating_sub(free));
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        out.push(check(
            "vram-fits",
            6,
            "Static pool fits in free VRAM",
            if need > free {
                Outcome::Block(if capped {
                    format!("{}: SGLang will hold ~{:.1} GiB (weights, state, capped KV) but only {:.1} GiB is free", d.name, gib(need), gib(free))
                } else {
                    format!(
                        "{}: SGLang will reserve {:.1} GiB ({:.0}% of {:.1}) but only {:.1} GiB is free",
                        d.name, gib(need), s.mem_fraction * 100.0, gib(d.total_mib * 1024 * 1024), gib(free)
                    )
                })
            } else {
                Outcome::Pass
            },
        ));
        // display headroom: the static pool is taken at launch; activations,
        // CUDA graphs and the allocator cache grow on top of it under load.
        let total = d.total_mib * 1024 * 1024;
        // With a token cap the server takes only `need`; whatever the other servers on this
        // card hold is gone too, so measure the headroom from what is free.
        let headroom = if capped { free.saturating_sub(need) } else { total.saturating_sub(need) };
        let loaded = headroom.saturating_sub(RUNTIME_OVERHEAD_BYTES);
        let safe_fraction = 1.0 - (DISPLAY_HEADROOM_BYTES + RUNTIME_OVERHEAD_BYTES) as f64 / total as f64;
        out.push(check(
            "display-headroom",
            10,
            "A display-driving card keeps VRAM headroom",
            match &d.display {
                Some(_) if headroom < DISPLAY_HEADROOM_BYTES => Outcome::Block(format!(
                    "{} drives a display and VRAM share {:.2} leaves only {:.2} GiB; the desktop's allocations \
                     then evict the server (3 tok/s, KFD evicted_ms climbing). Keep >= {:.1} GiB: share <= {:.2}",
                    d.name, s.mem_fraction, gib(headroom), gib(DISPLAY_HEADROOM_BYTES), safe_fraction
                )),
                Some(_) if loaded < DISPLAY_HEADROOM_BYTES => Outcome::Warn(format!(
                    "{} drives a display: {:.2} GiB free at launch, ~{:.1} GiB once long prompts grow the server \
                     (~{:.0} GiB of activations and graphs); eviction likely under load. Share <= {:.2} keeps {:.1} GiB",
                    d.name, gib(headroom), gib(loaded), gib(RUNTIME_OVERHEAD_BYTES), safe_fraction, gib(DISPLAY_HEADROOM_BYTES)
                )),
                Some(_) => Outcome::Note(format!("{} drives a display; {:.2} GiB free at launch, ~{:.1} GiB under load", d.name, gib(headroom), gib(loaded))),
                None => Outcome::Pass,
            },
        ));
    }

    // 7: does the model + KV for the context fit inside the pool?
    if let (Some(d), Some(shape)) = (device, ModelShape::of_model(&p.model.path)) {
        let b = pool_budget(&s, &shape, d, p.runtime.ctx_total, others_bytes, tp_of(p));
        let gib = |x: u64| x as f64 / (1u64 << 30) as f64;
        let state = match (b.state_slots, b.state_auto) {
            (Some(n), true) => format!(", state pool {:.1} GiB (auto, ~{n} slots)", gib(b.state_bytes)),
            (Some(n), false) => format!(", state pool {:.1} GiB ({n} slots)", gib(b.state_bytes)),
            _ => String::new(),
        };
        let draft = if b.draft_bytes > 0 { format!(", draft {:.1} GiB", gib(b.draft_bytes)) } else { String::new() };
        let split = format!("weights {:.1} GiB{draft}{state}", gib(b.weights_bytes));
        out.push(check(
            "kv-budget",
            7,
            "Weights and KV cache fit in the static pool",
            if b.kv_budget_bytes == 0 || b.tokens_in_budget < 1024 {
                Outcome::Block(format!(
                    "{}: {split} leave no room for KV in the {:.1} GiB pool ({:.0}% of the card); raise VRAM share, cut state slots, use a smaller quant or a bigger card",
                    shape.label, gib(b.pool_bytes), s.mem_fraction * 100.0
                ))
            } else if b.tokens_in_budget < b.ctx {
                Outcome::Warn(format!(
                    "{}: {split}; ~{} KV tokens fit ({} B/token, {}), fewer than the {} context: one request cannot reach it",
                    shape.label, b.tokens_in_budget, b.kv_bytes_per_token, s.kv_cache_dtype, b.ctx
                ))
            } else {
                Outcome::Note(format!(
                    "{}: {split}; ~{} KV tokens fit ({} B/token, {}) in the {:.1} GiB pool",
                    shape.label, b.tokens_in_budget, b.kv_bytes_per_token, s.kv_cache_dtype, gib(b.pool_bytes)
                ))
            },
        ));
        if let (Some(k), false) = (b.state_slots, b.state_auto) {
            let ratio = state_slots_per_request(&s);
            let cap = k / ratio;
            if cap < s.max_running_requests as u64 {
                out.push(check(
                    "state-slots",
                    11,
                    "State slots cover the concurrent requests",
                    Outcome::Warn(format!(
                        "{k} state slots at {ratio} per request: SGLang caps concurrent requests at {cap}, not {}; set state slots to {}",
                        s.max_running_requests,
                        ratio * s.max_running_requests as u64
                    )),
                ));
            }
        }
        if shape.nextn_layers == Some(0) && s.spec.as_ref().is_some_and(SgSpec::is_nextn) {
            out.push(check("spec-head", 5, "Speculative decoding has a draft", Outcome::Block(format!("{}: NEXTN drafts with the model's own predict layers and this checkpoint has none; use NGRAM or a draft model", shape.label))));
        }
    }

    // 8: port
    let holder = crate::launch::port_holder(&p.server.host, p.server.port, &cfg.runs_dir);
    out.push(check(
        "port-free",
        8,
        "Requested port is free or held by a Llama FIDIM server",
        match &holder {
            None => Outcome::Pass,
            Some(h) if h.profile_id.is_some() => Outcome::Note(format!(
                "port {} is held by FIDIM profile {} — it will be stopped and replaced",
                p.server.port,
                h.profile_id.as_deref().unwrap_or("?")
            )),
            Some(h) => Outcome::Block(format!(
                "port {} is in use by {}{} — not a Llama FIDIM server",
                p.server.port,
                h.process_name.as_deref().unwrap_or("an unknown process"),
                h.pid.map(|x| format!(" (pid {x})")).unwrap_or_default()
            )),
        },
    ));

    // 9: alias
    let id = crate::router::model_id(p);
    out.push(check(
        "alias-unique",
        9,
        "Served model name is unique among running servers",
        if running_aliases.iter().any(|a| a == &id) && holder.as_ref().and_then(|h| h.profile_id.as_deref()) != Some(p.id.as_str()) {
            Outcome::Block(format!("another running server already serves `{id}`"))
        } else {
            Outcome::Pass
        },
    ));

    // context sanity
    out.push(check(
        "context-set",
        12,
        "Context length is set",
        if p.runtime.ctx_total == 0 { Outcome::Block("runtime.ctx_total is 0".into()) } else { Outcome::Pass },
    ));
    out
}

trait WithNote {
    fn with_note(self, d: &Device) -> Self;
}
impl WithNote for CheckResult {
    fn with_note(mut self, d: &Device) -> Self {
        if matches!(self.outcome, Outcome::Pass) {
            self.outcome = Outcome::Note(format!("{} (HIP {}, bus {})", d.name, d.hip_index, d.bus_number.unwrap_or(0)));
        }
        self
    }
}

// --------------------------------------------------------------- launch ----

pub struct SgLaunch {
    pub state: RunState,
    pub results: Vec<CheckResult>,
    pub replaced: Option<String>,
    pub plan: LaunchPlan,
}

pub fn devices_now(cfg: &Config, platform: &dyn Platform) -> Result<Vec<Device>> {
    Ok(crate::devices::from_adapters(&platform.video_adapters()?, &cfg.integrated_name_patterns))
}

pub fn resolve_device<'d>(p: &Profile, devices: &'d [Device]) -> Option<&'d Device> {
    resolve_devices(p, devices).and_then(|v| v.into_iter().next())
}

/// Every card of the profile, in order; None when any key fails to resolve.
/// More than one card = tensor parallelism across them (`--tp N`).
pub fn resolve_devices<'d>(p: &Profile, devices: &'d [Device]) -> Option<Vec<&'d Device>> {
    if p.devices.is_empty() {
        return None;
    }
    p.devices
        .iter()
        .map(|pd| {
            crate::devices::resolve_key(pd.key.as_str(), devices, &[])
                .ok()
                .and_then(|r| devices.iter().find(|d| d.stable_key == r.device.stable_key))
        })
        .collect()
}

/// SGLang tensor-parallel width: one rank per profile device.
pub fn tp_of(p: &Profile) -> u64 {
    p.devices.len().max(1) as u64
}

pub fn running_aliases(cfg: &Config) -> Vec<String> {
    supervise::reattach(&cfg.runs_dir).into_iter().filter(|r| r.alive).map(|r| r.state.alias).collect()
}

/// Pre-flight, replace a previous FIDIM holder of the port, spawn, wait for
/// `/v1/models`. `override_blocks` launches past Block findings.
pub fn launch(
    cfg: &Config,
    p: &Profile,
    platform: &dyn Platform,
    override_blocks: bool,
    ready_timeout: Duration,
) -> Result<SgLaunch> {
    let devices = devices_now(cfg, platform)?;
    let devs = resolve_devices(p, &devices);
    let results = preflight(cfg, p, devs.as_deref(), platform, &running_aliases(cfg));
    if crate::preflight::any_block(&results) && !override_blocks {
        return Err(Error::Config(format!(
            "pre-flight blocked: {}",
            results
                .iter()
                .filter(|r| matches!(r.outcome, Outcome::Block(_)))
                .map(|r| match &r.outcome {
                    Outcome::Block(m) => format!("[{}] {m}", r.id),
                    _ => String::new(),
                })
                .collect::<Vec<_>>()
                .join("; ")
        )));
    }
    let devs = devs.ok_or_else(|| Error::Config("device does not resolve".into()))?;
    let device = devs[0];
    let plan = compose(cfg, p, &devs, &cfg.runs_dir)?;

    let mut replaced = None;
    for run in supervise::reattach(&cfg.runs_dir).into_iter().filter(|r| r.alive && r.state.port == p.server.port) {
        supervise::stop(&run.state, &cfg.runs_dir)?;
        replaced = Some(run.state.profile_id.clone());
    }
    if replaced.is_some() {
        crate::launch::wait_port_free(&p.server.host, p.server.port, Duration::from_secs(30))?;
    }
    let s = cfg_of(p);
    let cold = supervise::is_cold_start(&Config::config_dir(), &p.model.path);
    let state = supervise::spawn(
        &plan,
        p,
        &cfg.runs_dir,
        devs.iter().map(|d| d.stable_key.clone()).collect(),
        devs.iter().map(|d| d.free_mib).collect(),
        cold,
        need_bytes(&s, device, ModelShape::of_model(&p.model.path).as_ref(), p.runtime.ctx_total, tp_of(p)),
    )?;
    supervise::wait_ready_in(&state, ready_timeout, Some(&cfg.runs_dir))?;
    let _ = supervise::record_model_loaded(&Config::config_dir(), &p.model.path);
    Ok(SgLaunch { state, results, replaced, plan })
}

/// Launch the router (and first every member marked load_on_startup that is
/// not running). Returns the router's run state and what was started.
pub fn launch_router(
    cfg: &Config,
    rc: &crate::router::RouterConfig,
    profiles: &[Profile],
    platform: &dyn Platform,
    ready_timeout: Duration,
) -> Result<(RunState, Vec<String>, Vec<String>)> {
    let members: Vec<&Profile> = rc
        .members
        .iter()
        .map(|m| {
            profiles
                .iter()
                .find(|p| p.id == m.profile_id)
                .ok_or_else(|| Error::Config(format!("router member `{}` is not a saved profile", m.profile_id)))
        })
        .collect::<Result<_>>()?;
    if let Some(bad) = members.iter().find(|p| !p.engine.is_sglang()) {
        return Err(Error::Config(format!("`{}` is not an SGLang profile; the SGLang router only fronts SGLang members", bad.id)));
    }
    let mut started = Vec::new();
    let runs = supervise::reattach(&cfg.runs_dir);
    for (m, member) in rc.members.iter().zip(&members) {
        match runs.iter().find(|r| r.alive && r.state.profile_id == member.id) {
            // Started elsewhere and still loading: the router's readiness
            // needs it answering, and starting it again would replace it.
            Some(r) if r.loading => supervise::wait_ready_in(&r.state, ready_timeout, Some(&cfg.runs_dir))?,
            Some(_) => {}
            None if m.load_on_startup => {
                launch(cfg, member, platform, false, ready_timeout)?;
                started.push(member.id.clone());
            }
            None => {}
        }
    }
    let plan = router_plan(cfg, rc, &members)?;
    let mut replaced = Vec::new();
    for run in supervise::reattach(&cfg.runs_dir).into_iter().filter(|r| r.alive && r.state.port == rc.port) {
        supervise::stop(&run.state, &cfg.runs_dir)?;
        replaced.push(run.state.profile_id.clone());
    }
    if !replaced.is_empty() {
        crate::launch::wait_port_free(&rc.host, rc.port, Duration::from_secs(20))?;
    }
    let profile: Profile = serde_json::from_value(serde_json::json!({
        "schema": 1, "id": crate::router::ROUTER_ID, "name": "router", "engine": "sglang",
        "build": { "path": "" }, "model": { "path": "" }, "devices": [],
        "server": { "port": rc.port, "alias": crate::router::ROUTER_ID, "host": rc.host },
        "runtime": { "ctx_total": 0 },
    }))?;
    let keys: Vec<String> = members.iter().flat_map(|m| m.devices.iter().map(|d| d.key.clone())).collect();
    let state = supervise::spawn(&plan, &profile, &cfg.runs_dir, keys, vec![], false, 0)?;
    supervise::wait_ready_in(&state, Duration::from_secs(30), Some(&cfg.runs_dir))?;
    Ok((state, started, replaced))
}

pub fn is_sglang_router(rc: &crate::router::RouterConfig, profiles: &[Profile]) -> bool {
    !rc.members.is_empty()
        && rc.members.iter().all(|m| profiles.iter().any(|p| p.id == m.profile_id && p.engine == Engine::SgLang))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(total_mib: u64) -> Device {
        Device {
            stable_key: "pci:VEN_1002&DEV_7551:bus03".into(), name: "R9700".into(), hip_index: 0, backend: "ROCm".into(),
            total_mib, free_mib: total_mib, integrated: false, bus_number: Some(3), driver_version: None, display: None,
            luid_low: Some(3), correlation_assumed: false,
        }
    }

    #[test]
    fn gdn_state_per_slot_from_gguf_header() {
        // qwen35moe (35B-A3B): 40 blocks, every 4th full attention, inner 4096, state 128
        let h = crate::gguf::GgufHeader {
            block_count: Some(40), full_attention_interval: Some(4), ssm_inner_size: Some(4096),
            ssm_state_size: Some(128), head_count_kv: Some(2), key_length: Some(256),
            ..Default::default()
        };
        let s = ModelShape::from_gguf(&h, 21 << 30).unwrap();
        assert_eq!(s.attention_layers, 10);
        assert_eq!(s.state_bytes_per_slot, Some(30 * 4096 * 128 * 4)); // ~63 MB; SGLang logs 60 MB
    }

    #[test]
    fn hf_config_nested_text_and_hybrid_layers() {
        let cfg = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
            "text_config": { "hidden_size": 5120, "num_hidden_layers": 64, "num_attention_heads": 40,
                             "num_key_value_heads": 4, "head_dim": 256, "full_attention_interval": 4,
                             "max_position_embeddings": 262144, "vocab_size": 248320 },
            "vision_config": {}, "quantization_config": { "quant_method": "awq" }, "torch_dtype": "bfloat16"
        });
        let i = crate::discovery::HfModelInfo::from_config(&cfg, 18 << 30);
        assert_eq!(i.attention_layers, Some(16));
        assert_eq!(i.num_kv_heads, Some(4));
        assert_eq!(i.head_dim, Some(256));
        assert_eq!(i.quantization.as_deref(), Some("awq"));
        assert!(i.has_vision);
        let shape = ModelShape::from_hf(&i).unwrap();
        assert_eq!(shape.kv_bytes_per_token("fp8_e4m3"), 2 * 16 * 4 * 256);
        assert_eq!(shape.kv_bytes_per_token("bf16"), 2 * 2 * 16 * 4 * 256);
    }

    #[test]
    fn hf_layer_types_count_attention() {
        let cfg = serde_json::json!({ "model_type": "x", "num_hidden_layers": 4, "num_attention_heads": 8,
            "hidden_size": 1024, "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"] });
        let i = crate::discovery::HfModelInfo::from_config(&cfg, 0);
        assert_eq!(i.attention_layers, Some(2));
        assert_eq!(i.head_dim, Some(128));
    }

    #[test]
    fn pool_budget_matches_the_measured_pool() {
        // dd: 27B GGUF (17.2 GiB), fp4 KV, NEXTN 3/1/4, 10 state slots, 2
        // requests, 0.85 of an R9700: SGLang allocated 208,379 tokens.
        let s: SgLangCfg = serde_json::from_value(serde_json::json!({
            "mem_fraction": 0.85, "mamba_slots": 10, "max_running_requests": 2,
            "kv_cache_dtype": "fp4_mx_block16", "spec": { "algorithm": "NEXTN" } })).unwrap();
        let shape = ModelShape { weights_bytes: (17.2 * (1u64 << 30) as f64) as u64, attention_layers: 16, kv_heads: 4, head_dim: 256, label: "qwen35".into(), state_bytes_per_slot: Some(151 << 20), hybrid: true, nextn_layers: Some(1) };
        let b = pool_budget(&s, &shape, &dev(32624), 262144, 0, 1);
        assert!((190_000..225_000).contains(&b.tokens_in_budget), "{}", b.tokens_in_budget);
        assert!(b.tokens_in_budget < b.ctx);
        // qwenMoE: 35B-A3B (19.8 GiB), fp8 KV, 40 slots, 8 requests, 0.90: 428,128.
        let s: SgLangCfg = serde_json::from_value(serde_json::json!({
            "mem_fraction": 0.90, "mamba_slots": 40, "max_running_requests": 8, "kv_cache_dtype": "fp8_e4m3" })).unwrap();
        let shape = ModelShape { weights_bytes: (19.8 * (1u64 << 30) as f64) as u64, attention_layers: 10, kv_heads: 2, head_dim: 256, label: "qwen35moe".into(), state_bytes_per_slot: Some(30 * 4096 * 128 * 4), hybrid: true, nextn_layers: Some(0) };
        let b = pool_budget(&s, &shape, &dev(32624), 262144, 0, 1);
        assert!((405_000..450_000).contains(&b.tokens_in_budget), "{}", b.tokens_in_budget);
        assert!(b.tokens_in_budget > b.ctx);
        // auto state slots: SGLang gives the state pool 0.9/1.9 of the rest.
        let s: SgLangCfg = serde_json::from_value(serde_json::json!({ "mem_fraction": 0.90, "kv_cache_dtype": "fp8_e4m3" })).unwrap();
        let b2 = pool_budget(&s, &shape, &dev(32624), 262144, 0, 1);
        assert!(b2.state_auto && b2.state_bytes > b.state_bytes && b2.tokens_in_budget < b.tokens_in_budget);
    }

    #[test]
    fn pool_budget_per_card_under_tensor_parallelism() {
        // dd at --tp 2 over both R9700s (2026-10-09): 0.85, 15 state slots, 3 requests,
        // NEXTN 4 draft tokens; SGLang allocated 1,475,208 tokens per rank (heads split,
        // so each rank's per-token KV is half and every token lives on both cards).
        let s: SgLangCfg = serde_json::from_value(serde_json::json!({
            "mem_fraction": 0.85, "mamba_slots": 15, "max_running_requests": 3,
            "kv_cache_dtype": "fp4_mx_block16", "spec": { "algorithm": "NEXTN", "draft_tokens": 4 } })).unwrap();
        let shape = ModelShape { weights_bytes: (17.2 * (1u64 << 30) as f64) as u64, attention_layers: 16, kv_heads: 4, head_dim: 256, label: "qwen35".into(), state_bytes_per_slot: Some(151 << 20), hybrid: true, nextn_layers: Some(1) };
        let one = pool_budget(&s, &shape, &dev(32624), 262144, 0, 1);
        let two = pool_budget(&s, &shape, &dev(32624), 262144, 0, 2);
        assert!(two.weights_bytes * 2 <= one.weights_bytes + 1 && two.kv_bytes_per_token * 2 <= one.kv_bytes_per_token + 1);
        assert!((1_350_000..1_700_000).contains(&two.tokens_in_budget), "{}", two.tokens_in_budget);
    }

    #[test]
    fn compose_tensor_parallel_over_both_cards() {
        let mut d1 = dev(32624);
        d1.stable_key = "pci:VEN_1002&DEV_7551:bus08".into();
        d1.hip_index = 1;
        let d0 = dev(32624);
        let dir = std::env::temp_dir().join(format!("fidim-tp-{}", std::process::id()));
        let tools = dir.join("tools");
        std::fs::create_dir_all(&tools).unwrap();
        let mut cfg = Config::default_for_machine();
        cfg.sglang = Some(serde_json::from_value(serde_json::json!({ "venv": dir.join("venv"), "tools_dir": tools })).unwrap());
        let p: Profile = serde_json::from_value(serde_json::json!({
            "schema": 1, "engine": "sglang", "id": "tp", "name": "tp",
            "build": { "path": "" }, "model": { "path": "/m/x.gguf" },
            "devices": [ { "key": "pci:VEN_1002&DEV_7551:bus03" }, { "key": "pci:VEN_1002&DEV_7551:bus08" } ],
            "server": { "port": 30010, "alias": "tp", "host": "127.0.0.1" },
            "runtime": { "ctx_total": 262144 } })).unwrap();
        let plan = compose(&cfg, &p, &[&d0, &d1], &dir).unwrap();
        let tp = plan.args.iter().position(|a| a == "--tp").unwrap();
        assert_eq!(plan.args[tp + 1], "2");
        assert_eq!(plan.visibility_env, "0,1");
        assert!(plan.env.iter().any(|(k, v)| k == "SGLANG_HOSTCOMM" && v == "1"));
        assert!(plan.env.iter().any(|(k, v)| k == "HIP_VISIBLE_DEVICES" && v == "0,1"));
        // one card: no hostcomm, --tp 1
        let single = compose(&cfg, &p, &[&d0], &dir).unwrap();
        assert!(!single.env.iter().any(|(k, _)| k == "SGLANG_HOSTCOMM"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compose_dflash_draft_beside_a_gguf_target() {
        let d0 = dev(32624);
        let dir = std::env::temp_dir().join(format!("fidim-dflash-{}", std::process::id()));
        let tools = dir.join("tools");
        std::fs::create_dir_all(&tools).unwrap();
        let mut cfg = Config::default_for_machine();
        cfg.sglang = Some(serde_json::from_value(serde_json::json!({ "venv": dir.join("venv"), "tools_dir": tools })).unwrap());
        let p: Profile = serde_json::from_value(serde_json::json!({
            "schema": 1, "engine": "sglang", "id": "df", "name": "df",
            "build": { "path": "" }, "model": { "path": "/m/x.gguf" },
            "devices": [ { "key": "pci:VEN_1002&DEV_7551:bus03" } ],
            "server": { "port": 30011, "alias": "df", "host": "127.0.0.1" },
            "runtime": { "ctx_total": 262144 },
            "sglang": { "mamba_ssm_dtype": "bfloat16", "spec": { "algorithm": "DFLASH", "draft_tokens": 8,
                "draft_model": "/m/DFlash2", "draft_window": 2048, "draft_kv_cache_dtype": "fp8_e4m3" } } })).unwrap();
        let a = compose(&cfg, &p, &[&d0], &dir).unwrap().args;
        let val = |f: &str| a.iter().position(|x| x == f).map(|i| a[i + 1].clone());
        assert_eq!(val("--speculative-algorithm").as_deref(), Some("DFLASH"));
        assert_eq!(val("--speculative-num-draft-tokens").as_deref(), Some("8"));
        assert_eq!(val("--speculative-draft-model-path").as_deref(), Some("/m/DFlash2"));
        assert_eq!(val("--speculative-draft-load-format").as_deref(), Some("auto"));
        assert_eq!(val("--speculative-draft-model-quantization").as_deref(), Some("unquant"));
        assert_eq!(val("--speculative-draft-window-size").as_deref(), Some("2048"));
        assert_eq!(val("--speculative-draft-kv-cache-dtype").as_deref(), Some("fp8_e4m3"));
        assert_eq!(val("--mamba-ssm-dtype").as_deref(), Some("bfloat16"));
        // the block size is the draft; SGLang derives steps and top-k
        assert!(val("--speculative-num-steps").is_none() && val("--speculative-eagle-topk").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn pool_budget_dflash_window_and_bf16_state() {
        // dd at --tp 2 (2026-10-11): 0.85, 64 state slots, 4 requests, DFLASH 8 tokens with a
        // 2048 window and fp8 draft KV, bf16 SSM state: SGLang allocated 1,187,734 tokens per
        // rank and a 43,089-slot draft pool (5 KB per slot per rank).
        let dir = std::env::temp_dir().join(format!("fidim-dflash-kv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"num_hidden_layers": 5, "num_attention_heads": 32,
            "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 5120}"#).unwrap();
        let spec = |w: Option<u32>, ssm: Option<&str>| -> SgLangCfg {
            serde_json::from_value(serde_json::json!({
                "mem_fraction": 0.85, "mamba_slots": 64, "max_running_requests": 4, "kv_cache_dtype": "fp4_mx_block16",
                "mamba_ssm_dtype": ssm, "spec": { "algorithm": "DFLASH", "draft_tokens": 8, "draft_model": dir,
                "draft_window": w, "draft_kv_cache_dtype": "fp8_e4m3" } })).unwrap()
        };
        let shape = ModelShape { weights_bytes: (17.2 * (1u64 << 30) as f64) as u64, attention_layers: 16, kv_heads: 4, head_dim: 256, label: "qwen35".into(), state_bytes_per_slot: Some(151 << 20), hybrid: true, nextn_layers: Some(1) };
        let s = spec(Some(2048), Some("bfloat16"));
        let k = dflash_draft_kv(s.spec.as_ref().unwrap(), &s, 2).unwrap();
        assert_eq!(k.per_token_bytes, 0);
        assert_eq!(k.fixed_bytes, 5120 * (1 + 6 * 2064 + 32768));
        let windowed = pool_budget(&s, &shape, &dev(32624), 262144, 0, 2);
        // without the window every pool token carries a draft cell
        let unwindowed = pool_budget(&spec(None, Some("bfloat16")), &shape, &dev(32624), 262144, 0, 2);
        assert!(unwindowed.kv_bytes_per_token > windowed.kv_bytes_per_token);
        assert!(unwindowed.tokens_in_budget * 3 < windowed.tokens_in_budget * 2);
        // fp32 state: twice the state pool, fewer tokens
        let fp32 = pool_budget(&spec(Some(2048), None), &shape, &dev(32624), 262144, 0, 2);
        assert!(fp32.state_bytes > windowed.state_bytes * 19 / 10 && fp32.tokens_in_budget < windowed.tokens_in_budget);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn defaults_and_plan_shape() {
        let s = SgLangCfg::default();
        assert_eq!(s.mem_fraction, 0.85);
        assert_eq!(s.kv_cache_dtype, "fp8_e4m3");
        let p: Profile = serde_json::from_value(serde_json::json!({
            "schema": 1, "id": "dd", "name": "dd", "engine": "sglang",
            "build": { "path": "" }, "model": { "path": "/models/x.gguf" },
            "devices": [{ "key": "pci:VEN_1002&DEV_7551:bus03" }],
            "server": { "port": 30000, "alias": "dd", "host": "127.0.0.1" },
            "runtime": { "ctx_total": 262144 },
            "sglang": { "mamba_slots": 10, "spec": { "token_map": "/m/hot.pt" }, "aliases": ["ddg"] }
        }))
        .unwrap();
        assert!(p.engine.is_sglang());
        let s = cfg_of(&p);
        assert_eq!(s.mamba_slots, Some(10));
        assert_eq!(s.spec.as_ref().unwrap().steps, 3);
        assert_eq!(s.aliases, vec!["ddg".to_string()]);
    }
}
