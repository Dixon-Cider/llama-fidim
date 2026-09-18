//! VRAM estimation (R-04): weights + KV cache + compute buffer + overhead,
//! per device under the split proportions. Deliberately conservative, and
//! every estimate carries its assumptions so Block/Warn messages can show
//! the arithmetic instead of a bare number.

use serde::Serialize;

use crate::gguf::GgufHeader;
use crate::profile::{Runtime, SplitMode};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Bytes per element of a KV cache type. Quantised caches carry scale
/// blocks, hence the fractional values (q8_0: 1 byte + 1/16 scale, q4_0:
/// half a byte + 1/16 scale). Unknown types estimate as f16 — conservative.
fn kv_bytes_per_elem(kv_type: &str) -> (f64, bool) {
    match kv_type {
        "f32" => (4.0, true),
        "f16" | "bf16" => (2.0, true),
        "q8_0" => (1.0625, true),
        "q5_1" => (0.75, true),
        "q5_0" => (0.6875, true),
        "q4_1" => (0.625, true),
        "q4_0" => (0.5625, true),
        _ => (2.0, false),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct DeviceEstimate {
    pub key: String,
    pub fraction: f64,
    pub weights_bytes: u64,
    pub kv_bytes: u64,
    pub compute_bytes: u64,
    pub overhead_bytes: u64,
    pub total_bytes: u64,
}

impl DeviceEstimate {
    pub fn total_gib(&self) -> f64 {
        self.total_bytes as f64 / GIB
    }
    /// The arithmetic, human-readable — shown in Block/Warn messages.
    pub fn breakdown(&self) -> String {
        format!(
            "{:.2} GiB = weights {:.2} + kv {:.2} + compute {:.2} + overhead {:.2}",
            self.total_bytes as f64 / GIB,
            self.weights_bytes as f64 / GIB,
            self.kv_bytes as f64 / GIB,
            self.compute_bytes as f64 / GIB,
            self.overhead_bytes as f64 / GIB,
        )
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct VramEstimate {
    pub per_device: Vec<DeviceEstimate>,
    pub total_bytes: u64,
    /// Everything the formula had to assume (missing metadata, heuristic
    /// compute-buffer size). Surfaced alongside results.
    pub assumptions: Vec<String>,
}

pub struct EstimateInput<'a> {
    pub header: &'a GgufHeader,
    pub runtime: &'a Runtime,
    /// (stable key, fraction) per device, fractions summing to 1.0.
    pub devices: Vec<(String, f64)>,
    pub split_mode: Option<SplitMode>,
    /// Index into `devices` of the main GPU (compute buffer lives there).
    pub main_index: usize,
    /// Auxiliary files resident alongside the model, attributed to main.
    pub mmproj_bytes: u64,
    pub draft_bytes: u64,
}

pub fn estimate(input: &EstimateInput) -> VramEstimate {
    let mut assumptions = Vec::new();
    let h = input.header;
    let r = input.runtime;

    // ---- weights ------------------------------------------------------
    // Resident weights ≈ file size (quantised weights load as stored).
    let block_count = h.block_count.unwrap_or_else(|| {
        assumptions.push("layer count missing from GGUF; assuming full offload".into());
        0
    });
    let offload_fraction = if block_count == 0 || r.n_gpu_layers as u64 >= block_count + 1 {
        1.0
    } else {
        r.n_gpu_layers as f64 / (block_count + 1) as f64
    };
    let weights_total = (h.file_size as f64 * offload_fraction) as u64;

    // ---- KV cache -----------------------------------------------------
    // Per token per layer: 2 (K+V) × head_count_kv × head_dim × bytes/elem.
    // SWA architectures (sliding_window_pattern = N: every Nth layer is
    // full-attention) grow only the full layers with context; SWA layers cap
    // at min(per-slot ctx, window) per slot. This is why 64K-per-slot context
    // cost only 2.8 GB on the Gemma-4 MoE (measured 2026-07-29).
    let (kv_k, known_k) = kv_bytes_per_elem(&r.kv_type_k);
    let (kv_v, known_v) = kv_bytes_per_elem(&r.kv_type_v);
    if !known_k || !known_v {
        assumptions.push(format!(
            "unknown KV type ({} / {}) estimated as f16",
            r.kv_type_k, r.kv_type_v
        ));
    }
    // Head dims: `attention.key_length`/`value_length` are authoritative
    // (Gemma-4: 512, with 256 SWA variants — NOT embedding/head_count).
    let head_count = h.head_count.unwrap_or(0);
    let fallback_dim =
        if head_count > 0 { h.embedding_length.unwrap_or(0) / head_count } else { 0 };
    let k_dim = h.key_length.unwrap_or(fallback_dim);
    let v_dim = h.value_length.unwrap_or(fallback_dim);
    let k_dim_swa = h.key_length_swa.unwrap_or(k_dim);
    let v_dim_swa = h.value_length_swa.unwrap_or(v_dim);
    if h.key_length.is_none() {
        assumptions
            .push(format!("attention.key_length missing; using embedding/head_count = {fallback_dim}"));
    }
    // KV heads per layer: modern architectures store an array; older a scalar.
    let kv_heads_for = |layer: usize| -> Option<u64> {
        if let Some(arr) = &h.head_count_kv_per_layer {
            arr.get(layer).copied()
        } else if let Some(s) = h.head_count_kv {
            Some(s)
        } else {
            None
        }
    };
    // Sliding-vs-full per layer: bool[block_count] (true = sliding) on modern
    // models; scalar pattern N (every Nth full) on older; else all full.
    let sliding_for = |layer: usize| -> bool {
        if let Some(flags) = &h.swa_layer_flags {
            flags.get(layer).copied().unwrap_or(false)
        } else if let Some(p) = h.sliding_window_pattern {
            p > 1 && (layer as u64 + 1) % p != 0
        } else {
            false
        }
    };
    let slots = r.slots.max(1) as u64;
    let per_slot_ctx = r.ctx_total / slots;
    let window = h.sliding_window.unwrap_or(0);
    let swa_tokens = if window > 0 { per_slot_ctx.min(window) * slots } else { r.ctx_total };

    let kv_estimable = block_count > 0
        && (0..block_count as usize).all(|i| kv_heads_for(i).is_some())
        && (k_dim + v_dim) > 0;
    // Hybrid architectures (Qwen 3.5+/3.8, Qwen3-Next): only every Nth layer
    // is full attention with a KV cache; the rest are linear-attention
    // layers whose state is fixed per slot regardless of context —
    // llama.cpp's n_embd_r + n_embd_s, in f32:
    //   r = (conv_kernel - 1) x (inner + 2 x groups x state)
    //   s = state x inner
    let full_interval = h.full_attention_interval.unwrap_or(0);
    let is_recurrent = |layer: usize| full_interval > 1 && (layer as u64 + 1) % full_interval != 0;
    let recurrent_bytes_per_slot: f64 = if full_interval > 1 {
        let inner = h.ssm_inner_size.unwrap_or(0) as f64;
        let state = h.ssm_state_size.unwrap_or(0) as f64;
        let conv = h.ssm_conv_kernel.unwrap_or(1) as f64;
        let groups = h.ssm_group_count.unwrap_or(1) as f64;
        ((conv - 1.0).max(0.0) * (inner + 2.0 * groups * state) + state * inner) * 4.0
    } else {
        0.0
    };
    if full_interval > 1 {
        assumptions.push(format!(
            "hybrid attention: KV cache on every {full_interval}th layer only; the other layers hold a fixed \
             {:.1} MiB recurrent state per slot",
            recurrent_bytes_per_slot / (1024.0 * 1024.0)
        ));
    }

    let kv_total = if !kv_estimable {
        assumptions.push("attention metadata incomplete; KV cache not estimable — treating as 0, DO NOT trust for tight fits".into());
        0u64
    } else {
        let mut total = 0.0f64;
        for layer in 0..block_count as usize {
            if is_recurrent(layer) {
                total += recurrent_bytes_per_slot * slots as f64;
                continue;
            }
            let heads = kv_heads_for(layer).unwrap_or(0) as f64;
            let (dims, tokens) = if sliding_for(layer) && window > 0 {
                ((k_dim_swa + v_dim_swa) as f64, swa_tokens as f64)
            } else {
                ((k_dim + v_dim) as f64, r.ctx_total as f64)
            };
            // Per token per layer: heads x (K dim + V dim), K at kv_k bytes,
            // V at kv_v bytes — approximated as the mean since dims match.
            total += heads * dims * ((kv_k + kv_v) / 2.0) * tokens;
        }
        total as u64
    };

    // ---- compute buffer ------------------------------------------------
    // Heuristic, calibrated against measured configurations on the target
    // machine; scales with the physical batch (its actual driver). Listed
    // as an assumption because it is one.
    let compute_total = ((0.75 + 0.5 * (r.batch_physical as f64 / 512.0)) * GIB) as u64;
    assumptions.push(format!(
        "compute buffer heuristic: 0.75 GiB + 0.5 GiB x (batch_physical {} / 512)",
        r.batch_physical
    ));
    let overhead_per_device = (0.4 * GIB) as u64; // HIP context + fragmentation

    // ---- distribute across devices -------------------------------------
    // Layer split distributes weights and their KV together (KV lives with
    // its layer); the compute buffer and auxiliary models sit on main.
    let mut per_device = Vec::new();
    for (i, (key, fraction)) in input.devices.iter().enumerate() {
        let is_main = i == input.main_index;
        let weights = (weights_total as f64 * fraction) as u64
            + if is_main { input.mmproj_bytes + input.draft_bytes } else { 0 };
        let kv = (kv_total as f64 * fraction) as u64;
        let compute = if is_main { compute_total } else { compute_total / 4 };
        let total = weights + kv + compute + overhead_per_device;
        per_device.push(DeviceEstimate {
            key: key.clone(),
            fraction: *fraction,
            weights_bytes: weights,
            kv_bytes: kv,
            compute_bytes: compute,
            overhead_bytes: overhead_per_device,
            total_bytes: total,
        });
    }
    if input.devices.len() > 1 {
        assumptions.push(
            "non-main devices carry a quarter compute buffer (activation transfer staging)".into(),
        );
    }
    let total_bytes = per_device.iter().map(|d| d.total_bytes).sum();
    VramEstimate { per_device, total_bytes, assumptions }
}

// ------------------------------------------------------------ diffusion ----

/// Which sizing rules the build's DiffusionGemma runner follows. Stock
/// Unsloth unless the build's manifest `patch` declares features
/// (`discovery::dg_feature`); plus whether FIDIM launches it with the HIP
/// runtime cache off (`GPU_RESOURCE_CACHE_SIZE=0`, its default unless the
/// profile env sets the key, see `launch::compose_diffusion`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DgRunner {
    /// The prompt-KV store is F16 with FA on or off (stock: F32 unless FA).
    pub pkv_f16: bool,
    /// Sliding layers keep a ring of `n_swa-1 + n_ubatch` store rows.
    pub swa_ring: bool,
    /// FA=1 runs the 512-dim heads on the GPU (stock: they fall back to the CPU).
    pub fa_pad: bool,
    /// Under FA the runner sizes by `llama_diffusion_fa_turn_bytes` up to
    /// min(n_ctx_train, 65536), with a 2048-token prefill chunk.
    pub fa_turn_sizing: bool,
    pub runtime_cache_off: bool,
}

impl DgRunner {
    /// Unsloth's runner as shipped, launched the way FIDIM launches it.
    pub const STOCK: DgRunner =
        DgRunner { pkv_f16: false, swa_ring: false, fa_pad: false, fa_turn_sizing: false, runtime_cache_off: true };

    pub fn from_build(meta: &crate::discovery::BuildMeta, runtime_cache_off: bool) -> Self {
        use crate::discovery::dg_feature as f;
        DgRunner {
            pkv_f16: meta.has_feature(f::PKV_F16),
            swa_ring: meta.has_feature(f::SWA_RING),
            fa_pad: meta.has_feature(f::FA_PAD),
            fa_turn_sizing: meta.has_feature(f::FA_PAD) && meta.has_feature(f::FA_TURN_SIZING),
            runtime_cache_off,
        }
    }

    /// Whether a run with `flash_attn` sizes by the per-turn estimate rather
    /// than the N² score tensor.
    pub fn fa_sized(&self, flash_attn: bool) -> bool {
        flash_attn && self.fa_turn_sizing
    }
}

/// How a DiffusionGemma run's context budget (MAXTOK) will come out: shown
/// by pre-flight check 15 and read by the profile editor.
#[derive(Debug, Clone, Serialize)]
pub struct DiffusionSizing {
    /// Tokens denoised per block (`diffusion.canvas_length`).
    pub canvas: u32,
    /// What the runner's auto-size picks with the card otherwise empty.
    /// None = no candidate fits; the runner then falls back to its floor.
    pub predicted_auto_maxtok: Option<u32>,
    /// The budget this launch runs with: `ctx_total` when the runner takes
    /// it, else the prediction (else the floor), never above a refused
    /// `ctx_total`.
    pub maxtok_used: u32,
    /// A reply needs at least one whole canvas of the budget.
    pub max_prompt_tokens: u32,
    /// The part of the prompt-KV store that grows with the prompt: K and V
    /// of every layer that keeps each position (all layers on the stock
    /// runner, F32 unless FA; only the full-attention layers, F16, on a
    /// runner with the sliding ring). Ring rows are a fixed part of
    /// `request_bytes`.
    pub pkv_bytes_per_token: u64,
    /// What a request adds over the load at `maxtok_used`, kept by the
    /// runner afterwards (the estimate's KV term).
    pub request_bytes: u64,
    /// The runner sizes by its per-turn FA estimate (patched build, FA on)
    /// instead of the N² score tensor.
    pub fa_sized: bool,
    /// Largest budget the runner considers: min(n_ctx_train, 65536).
    pub ctx_ceiling: u32,
    /// Whether an explicit `ctx_total` passes the runner's own gate; None
    /// when the runner auto-sizes.
    pub explicit_fits: Option<bool>,
    /// An explicit `ctx_total` above `ctx_ceiling` under FA sizing: the
    /// runner ignores it and auto-sizes.
    pub explicit_capped: bool,
    /// NGL covers every block plus the output layer.
    pub full_offload: bool,
}

/// The runner's MAXTOK candidates, largest first (VS:182).
const DG_MAXTOK_CANDIDATES: [u64; 12] =
    [65536, 49152, 40960, 32768, 24576, 20480, 16384, 12288, 8192, 6144, 4096, 2048];

/// Prefill buffers that grow with the square of the prompt, per token², with
/// the HIP runtime keeping freed memory (its default). Measured, not derived:
/// see `estimate_diffusion`.
const DG_PREFILL_BYTES_PER_TOKEN_SQ: u64 = 53;

/// The same with `GPU_RESOURCE_CACHE_SIZE=0`: 12.4-13.9 B/token² measured at
/// 5,014, 5,230 and 10,012 prompt tokens; the largest is kept.
const DG_PREFILL_BYTES_PER_TOKEN_SQ_CACHE_OFF: u64 = 14;

/// What the first request adds regardless of prompt length (measured).
const DG_FIRST_REQUEST_BYTES: u64 = 400 * 1024 * 1024;

/// The runner's smallest auto-sized context: four canvases, never below 2048.
fn dg_floor(canvas: u64) -> u64 {
    (canvas.max(1) * 4).max(2048)
}

/// Largest budget the runner considers (VS:182); under FA sizing also its
/// cap for an explicit MAXTOK.
fn dg_ceiling(n_ctx_train: u64) -> u64 {
    if n_ctx_train > 0 { n_ctx_train.min(65536) } else { 65536 }
}

/// The runner's prefill chunk (n_ubatch) for a context of `n` (VS make_cparams).
fn dg_ubatch(n_head: u64, n: u64, fa_sized: bool) -> u64 {
    let n = n.max(1);
    if fa_sized {
        let ub = n.min(2048);
        if ub > 256 { ub - ub % 256 } else { ub }
    } else {
        n.min(((1u64 << 30) / (n_head.max(1) * n)).clamp(256, 2048))
    }
}

fn prefill_sq(runner: DgRunner) -> u64 {
    if runner.runtime_cache_off { DG_PREFILL_BYTES_PER_TOKEN_SQ_CACHE_OFF } else { DG_PREFILL_BYTES_PER_TOKEN_SQ }
}

/// The MAXTOK the runner's VRAM-gated auto-size settles on (VS:180-212),
/// given the bytes left on the card after the weights, for a run that sizes
/// by the N² score tensor (the stock runner, or any runner with FA off).
/// With DG_FREE_RAM_MB=0 (FIDIM's full-offload default) that pass is the
/// only one, so this is deterministic up to the probe's own allocation
/// check. Here a zero budget simply fits nothing.
pub fn predict_auto_maxtok(n_head: u64, canvas: u64, n_ctx_train: u64, budget_bytes: u64) -> Option<u32> {
    let n_head = n_head.max(1);
    predict_with(canvas, n_ctx_train, budget_bytes, |n| n_head as f64 * n as f64 * n as f64 * 4.0)
}

/// The candidate walk both gates share: the first whole-canvas candidate at
/// or under the ceiling whose turn needs at most 90% of the budget.
fn predict_with(canvas: u64, n_ctx_train: u64, budget_bytes: u64, turn: impl Fn(u64) -> f64) -> Option<u32> {
    let canvas = canvas.max(1);
    let ceil = dg_ceiling(n_ctx_train);
    let floor = dg_floor(canvas);
    for raw in DG_MAXTOK_CANDIDATES {
        if raw > ceil {
            continue;
        }
        let n = raw / canvas * canvas; // whole canvases only
        if n < floor {
            break;
        }
        if turn(n) <= budget_bytes as f64 * 0.9 {
            return Some(n as u32);
        }
    }
    None
}

/// Sliding-vs-full for one layer, as `estimate` reads it.
fn layer_is_sliding(h: &GgufHeader, layer: usize) -> bool {
    if let Some(flags) = &h.swa_layer_flags {
        flags.get(layer).copied().unwrap_or(false)
    } else if let Some(p) = h.sliding_window_pattern {
        p > 1 && !(layer as u64 + 1).is_multiple_of(p)
    } else {
        false
    }
}

/// The prompt-KV store's geometry: per layer, K+V elements per position and
/// whether the layer keeps a ring (sliding, on a runner with the ring).
struct DgStore {
    layers: Vec<(u64, bool)>,
    /// `n_swa - 1`; 0 when no layer keeps a ring.
    swa_reach: u64,
    elt: u64,
}

impl DgStore {
    /// Store bytes for `p` positions with a `ub`-token prefill chunk
    /// (diffusion-gemma.cpp dg_ensure_pkv_store).
    fn bytes(&self, p: u64, ub: u64) -> u64 {
        let ring_rows = self.swa_reach + ub;
        self.layers.iter().map(|&(row, ring)| row * self.elt * if ring { p.min(ring_rows) } else { p }).sum()
    }
    /// The part that grows with every position.
    fn per_token(&self) -> u64 {
        self.layers.iter().filter(|(_, ring)| !ring).map(|(row, _)| row * self.elt).sum()
    }
    /// Widest layer that attends over every key (the DECODE working copies).
    fn widest_full_row(&self) -> u64 {
        self.layers.iter().filter(|(_, ring)| !ring).map(|(row, _)| *row).max().unwrap_or(0)
    }
}

/// VRAM estimate and context sizing for a DiffusionGemma run on one card
/// with `free_mib` free before the load. The terms beyond the weights are
/// partly inferred and say so in `assumptions`, so a Block or Warn built on
/// them shows how much is measured.
pub fn estimate_diffusion(
    h: &GgufHeader,
    n_gpu_layers: u32,
    ctx_total: u64,
    flash_attn: bool,
    runner: DgRunner,
    device_key: &str,
    free_mib: u64,
) -> (VramEstimate, DiffusionSizing) {
    const MIB: u64 = 1024 * 1024;
    let mut assumptions = Vec::new();

    let canvas = h.diffusion_canvas_length.filter(|c| *c > 0).unwrap_or_else(|| {
        assumptions.push("diffusion.canvas_length missing; assuming 256".into());
        256
    });
    let block_count = h.block_count.unwrap_or_else(|| {
        assumptions.push("layer count missing from GGUF; assuming full offload".into());
        0
    });
    // Every block plus the output layer.
    let full_offload = block_count == 0 || n_gpu_layers as u64 > block_count;
    let offload_fraction =
        if full_offload { 1.0 } else { n_gpu_layers as f64 / (block_count + 1) as f64 };
    let weights = (h.file_size as f64 * offload_fraction) as u64;

    let n_head = h.head_count.unwrap_or_else(|| {
        assumptions.push("attention.head_count missing; the context prediction assumes 1 head".into());
        1
    });
    let vocab = h.vocab_size.unwrap_or_else(|| {
        assumptions.push("tokenizer vocabulary size missing; logits and sc_embT terms omitted".into());
        0
    });

    // Prompt-KV store (diffusion-gemma.cpp dg_ensure_pkv_store): K and V of
    // n_embd_head_k(l) x n_head_kv(l) per position on every layer; F32 on the
    // stock runner unless FA, F16 on a patched one, whose sliding layers keep
    // a ring.
    let fallback_dim = h.embedding_length.unwrap_or(0).checked_div(n_head).unwrap_or(0);
    let k_full = h.key_length.unwrap_or(fallback_dim);
    let k_swa = h.key_length_swa.unwrap_or(k_full);
    if h.key_length.is_none() {
        assumptions
            .push(format!("attention.key_length missing; using embedding/head_count = {fallback_dim}"));
    }
    let n_swa = h.sliding_window.unwrap_or(0);
    let ring = runner.swa_ring && n_swa > 1;
    if runner.swa_ring && !ring {
        assumptions.push("sliding_window missing from GGUF; sizing the store without the sliding ring".into());
    }
    let mut kv_heads_assumed = false;
    let layers: Vec<(u64, bool)> = (0..block_count as usize)
        .map(|layer| {
            let heads = match (&h.head_count_kv_per_layer, h.head_count_kv) {
                (Some(arr), _) if layer < arr.len() => arr[layer],
                (_, Some(s)) => s,
                _ => {
                    kv_heads_assumed = true;
                    n_head
                }
            };
            let sliding = layer_is_sliding(h, layer);
            let dim = if sliding { k_swa } else { k_full };
            (dim * heads * 2, ring && sliding)
        })
        .collect();
    if kv_heads_assumed {
        assumptions.push("KV head count missing for some layers; assuming one per attention head".into());
    }
    let store = DgStore {
        layers,
        swa_reach: if ring { n_swa - 1 } else { 0 },
        elt: if runner.pkv_f16 || flash_attn { 2 } else { 4 },
    };

    // The runner sizes against the VRAM its device reports once the weights
    // are resident, which is this card's free memory minus the weights.
    let fa_sized = runner.fa_sized(flash_attn);
    let logits_bytes = canvas * vocab * 4;
    // Mirrors llama_diffusion_fa_turn_bytes: the F16 store over n keys, three
    // F32 copies of the widest full-attention K/V over n keys, and the
    // device-resident self-conditioning logits.
    let fa_turn = |n: u64| -> u64 {
        store.bytes(n, dg_ubatch(n_head, n, true)) + store.widest_full_row() * 4 * 3 * n + logits_bytes
    };
    let scores = |n: u64| -> f64 { n_head.max(1) as f64 * n as f64 * n as f64 * 4.0 };
    let n_ctx_train = h.context_length.unwrap_or(0);
    let ceiling = dg_ceiling(n_ctx_train);
    let budget = (free_mib * MIB).saturating_sub(weights);
    let predicted = if fa_sized {
        predict_with(canvas, n_ctx_train, budget, |n| fa_turn(n) as f64)
    } else {
        predict_auto_maxtok(n_head, canvas, n_ctx_train, budget)
    };
    let floor = dg_floor(canvas);
    let explicit_capped = fa_sized && ctx_total > ceiling;
    // The runner honours an explicit budget when its turn fits 90% of what is
    // left (VS explicit branch); FIDIM runs a full offload with no RAM budget.
    let explicit_fits = (ctx_total > 0 && !explicit_capped).then(|| {
        let need = if fa_sized { fa_turn(ctx_total) as f64 } else { scores(ctx_total) };
        need <= budget as f64 * 0.9
    });
    // A refused budget degrades through the same probe, never above it; a
    // capped one is ignored for auto.
    let auto_used = predicted.map(u64::from).unwrap_or(floor);
    let maxtok_used = match explicit_fits {
        Some(true) => ctx_total,
        Some(false) => auto_used.min(ctx_total),
        None => auto_used,
    };
    if (ctx_total == 0 || explicit_capped) && predicted.is_none() {
        assumptions.push(format!(
            "no context candidate fits the VRAM left after the weights; the runner falls back to its \
             {floor}-token floor"
        ));
    }
    let prompt_tokens = maxtok_used.saturating_sub(canvas);
    let ub = dg_ubatch(n_head, maxtok_used, fa_sized);

    let pkv_bytes_per_token = store.per_token();
    let kv_bytes = if fa_sized {
        // Measured on gfx1201 (patched b11027, FA, cache off) over the load:
        // +1.21 GiB after a 10K prompt and +2.88 GiB after 60.6K, against
        // 1.27 and 3.38 from this term.
        fa_turn(maxtok_used)
    } else {
        // What a request adds on top of the load, which the runner keeps after
        // the request ends: the prompt-KV store plus prefill working buffers
        // that grow faster than the prompt, plus a fixed first-request part.
        // Fitted to the stock runner's resident VRAM on gfx1201 (b11027, FA
        // off). Runtime cache on: 5,026 prompt tokens -> +3.88 GiB, 10,526 ->
        // +10.0 GiB. Cache off: 5,014 -> +2.82, 5,230 -> +2.91, 10,012 -> +5.75.
        DG_FIRST_REQUEST_BYTES
            + store.bytes(prompt_tokens, ub)
            + prefill_sq(runner) * prompt_tokens * prompt_tokens
    };

    // Compute: the reserve at the runner's prefill chunk plus the canvas
    // logits plus a fixed part, which together match the measured 566.01 MiB
    // reserve (and the 17.98 GiB patched load at a 2048 chunk); and the
    // self-conditioning embedding copy (sc_embT), inferred.
    let n_embd = h.embedding_length.unwrap_or(0);
    let sc_emb_t = vocab * n_embd * 2;
    let compute_bytes = n_head * ub * ub * 4 + logits_bytes + 54 * MIB + sc_emb_t;
    let overhead_bytes = (0.4 * GIB) as u64;

    if fa_sized {
        assumptions.push(format!(
            "per-request working set (the patched runner's own FA sizing; measured on gfx1201 5-17% under \
             it): allocated per request and kept, not at load — up to {:.2} GiB at MAXTOK {maxtok_used} (F16 \
             store {pkv_bytes_per_token} B/token + sliding rings {:.0} MiB, K/V working copies, canvas logits)",
            kv_bytes as f64 / GIB,
            store.bytes(maxtok_used, ub).saturating_sub(pkv_bytes_per_token * maxtok_used) as f64 / MIB as f64
        ));
    } else {
        assumptions.push(format!(
            "prompt working set (measured on gfx1201, FA off{}): allocated per request and kept, not at load \
             — up to {:.2} GiB for a {prompt_tokens}-token prompt (prompt-KV store {pkv_bytes_per_token} \
             B/token{} + prefill buffers {} B/token² + 0.4 GiB); a lower context budget shrinks it \
             quadratically",
            if runner.pkv_f16 || ring { "; F16/ring store of the patched runner, prefill term from stock" } else { "" },
            kv_bytes as f64 / GIB,
            if ring { " + sliding rings" } else { "" },
            prefill_sq(runner),
        ));
    }
    if !runner.runtime_cache_off {
        assumptions.push(
            "the profile env sets GPU_RESOURCE_CACHE_SIZE: the HIP runtime keeps freed device memory, so \
             long prompts hold more (measured +3.9 GiB after 10K tokens on the stock runner)"
                .into(),
        );
    }
    assumptions.push(format!(
        "self-conditioning embedding copy (vocab x n_embd x 2 B = {:.0} MiB) is inferred; the rest of the \
         compute term matches the measured 566 MiB reserve",
        sc_emb_t as f64 / MIB as f64
    ));
    assumptions.push(
        "the runner's auto-size reads free VRAM through WDDM, which cannot see other processes' \
         allocations"
            .into(),
    );

    let total_bytes = weights + kv_bytes + compute_bytes + overhead_bytes;
    let estimate = VramEstimate {
        per_device: vec![DeviceEstimate {
            key: device_key.to_string(),
            fraction: 1.0,
            weights_bytes: weights,
            kv_bytes,
            compute_bytes,
            overhead_bytes,
            total_bytes,
        }],
        total_bytes,
        assumptions,
    };
    let clamp_u32 = |v: u64| v.min(u32::MAX as u64) as u32;
    let sizing = DiffusionSizing {
        canvas: clamp_u32(canvas),
        predicted_auto_maxtok: predicted,
        maxtok_used: clamp_u32(maxtok_used),
        max_prompt_tokens: clamp_u32(prompt_tokens),
        pkv_bytes_per_token,
        request_bytes: kv_bytes,
        fa_sized,
        ctx_ceiling: clamp_u32(ceiling),
        explicit_fits,
        explicit_capped,
        full_offload,
    };
    (estimate, sizing)
}

/// Compute auto split fractions proportional to free VRAM.
pub fn auto_fractions(free_mib: &[u64]) -> Vec<f64> {
    let total: u64 = free_mib.iter().sum();
    if total == 0 {
        return free_mib.iter().map(|_| 1.0 / free_mib.len().max(1) as f64).collect();
    }
    free_mib.iter().map(|f| *f as f64 / total as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf;
    use crate::profile::Runtime;

    fn worker_pool_runtime() -> Runtime {
        serde_json::from_value(serde_json::json!({
            "ctx_total": 393216u64, "slots": 6, "kv_type_k": "q8_0", "kv_type_v": "q8_0",
            "batch_logical": 2048, "batch_physical": 256
        }))
        .unwrap()
    }

    /// Calibration against the strongest available ground truth: the QAT
    /// 26B-A4B worker-pool configuration measured at 19.87 GB resident on
    /// 2026-07-29 (np=6, ctx=384K, q8_0 KV, ub=256). Runs only when the real
    /// model file is present.
    #[test]
    fn calibrates_against_measured_worker_pool() {
        let path = std::path::Path::new(
            "E:/models/unsloth/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        );
        if !path.exists() {
            eprintln!("real model absent; calibration test skipped");
            return;
        }
        let header = gguf::read_header(path).unwrap();
        let est = estimate(&EstimateInput {
            header: &header,
            runtime: &worker_pool_runtime(),
            devices: vec![("pci:test:bus08".into(), 1.0)],
            split_mode: None,
            main_index: 0,
            mmproj_bytes: 0,
            draft_bytes: 0,
        });
        let total = est.per_device[0].total_gib();
        let measured = 19.87;
        let error = (total - measured).abs() / measured;
        eprintln!("estimate {total:.2} GiB vs measured {measured} GiB ({:.1}% off)", error * 100.0);
        eprintln!("breakdown: {}", est.per_device[0].breakdown());
        assert!(
            error < 0.15,
            "estimator drifted {:.1}% from the measured baseline: {}",
            error * 100.0,
            est.per_device[0].breakdown()
        );
    }

    #[test]
    fn split_distributes_weights_and_kv_but_not_compute() {
        let path = std::path::Path::new(
            "E:/models/lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf",
        );
        if !path.exists() {
            eprintln!("real model absent; split test skipped");
            return;
        }
        let header = gguf::read_header(path).unwrap();
        let est = estimate(&EstimateInput {
            header: &header,
            runtime: &worker_pool_runtime(),
            devices: vec![("pci:a:bus03".into(), 0.5), ("pci:a:bus08".into(), 0.5)],
            split_mode: Some(SplitMode::Layer),
            main_index: 0,
            mmproj_bytes: 0,
            draft_bytes: 0,
        });
        assert_eq!(est.per_device.len(), 2);
        let (main, second) = (&est.per_device[0], &est.per_device[1]);
        // Weights split evenly; compute concentrated on main.
        let wdiff = (main.weights_bytes as f64 - second.weights_bytes as f64).abs()
            / main.weights_bytes as f64;
        assert!(wdiff < 0.01, "weights should split evenly");
        assert!(main.compute_bytes > second.compute_bytes);
        // Each half of a 34 GB model must individually fit a 32 GB card.
        assert!(main.total_gib() < 32.0, "main: {}", main.breakdown());
        assert!(second.total_gib() < 32.0, "second: {}", second.breakdown());
    }

    #[test]
    fn auto_fractions_follow_free_vram() {
        let f = auto_fractions(&[32472, 16000]);
        assert!((f[0] + f[1] - 1.0).abs() < 1e-9);
        assert!(f[0] > f[1]);
    }

    /// DiffusionGemma 26B-A4B Q4_K_M as its header reads: 30 layers, 16
    /// heads, every 6th layer full attention (2 KV heads, 512-dim) and the
    /// rest sliding (8 KV heads, 256-dim).
    fn diffusion_gemma_header() -> GgufHeader {
        let swa: Vec<bool> = (0..30).map(|l| (l + 1) % 6 != 0).collect();
        let kv: Vec<u64> = swa.iter().map(|s| if *s { 8 } else { 2 }).collect();
        serde_json::from_value(serde_json::json!({
            "path": "E:/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
            "file_size": 16_806_810_208u64, "gguf_version": 3, "tensor_count": 0,
            "architecture": "diffusion-gemma", "block_count": 30, "context_length": 262144,
            "embedding_length": 2816, "head_count": 16, "head_count_kv_per_layer": kv,
            "key_length": 512, "value_length": 512, "key_length_swa": 256, "value_length_swa": 256,
            "sliding_window": 1024,
            "swa_layer_flags": swa, "diffusion_canvas_length": 256, "vocab_size": 262144,
            "metadata": {}
        }))
        .unwrap()
    }

    /// The stock runner with the HIP runtime cache left on: the conditions
    /// the original measurements were taken under.
    const STOCK_CACHE_ON: DgRunner = DgRunner { runtime_cache_off: false, ..DgRunner::STOCK };

    /// The patched runner as the dgpatch manifest declares it, launched by FIDIM.
    fn patched() -> DgRunner {
        use crate::discovery::{dg_feature as f, BuildMeta, BuildPatch};
        let meta = BuildMeta {
            patch: Some(BuildPatch {
                name: "dgpatch".into(),
                features: [f::PKV_F16, f::SWA_RING, f::FA_PAD, f::FA_TURN_SIZING, f::STEP_FAIL_ERR]
                    .map(String::from)
                    .to_vec(),
                ..Default::default()
            }),
            ..Default::default()
        };
        DgRunner::from_build(&meta, true)
    }

    /// Hand arithmetic, following the runner (VS:180-212) and the prompt-KV
    /// allocator: 32472 MiB free - 16,806,810,208 B of weights leaves
    /// 17,242,549,664 B; 90% of that is 15.52 GB, so 16384 (16 x 16384^2 x 4
    /// = 17.18 GB of scores) is skipped and 12288 (9.66 GB) is taken. Per
    /// prompt token: 25 sliding layers x 256 x 8 x 2 x 4 B + 5 full layers x
    /// 512 x 2 x 2 x 4 B = 409,600 + 40,960 = 450,560 B.
    #[test]
    fn diffusion_sizing() {
        let h = diffusion_gemma_header();
        assert_eq!(predict_auto_maxtok(16, 256, 262144, 17_242_549_664), Some(12288));
        // The candidate ceiling follows a short training context.
        assert_eq!(predict_auto_maxtok(16, 256, 8192, u64::MAX / 8), Some(8192));
        // Nothing left after the weights: nothing fits.
        assert_eq!(predict_auto_maxtok(16, 256, 262144, 0), None);

        let (est, s) = estimate_diffusion(&h, 99, 0, false, STOCK_CACHE_ON, "pci:x:bus08", 32472);
        assert_eq!(s.predicted_auto_maxtok, Some(12288));
        assert_eq!(s.maxtok_used, 12288);
        assert_eq!(s.max_prompt_tokens, 12288 - 256);
        assert_eq!(s.canvas, 256);
        assert_eq!(s.pkv_bytes_per_token, 450_560);
        assert!(!s.fa_sized);
        assert_eq!(s.ctx_ceiling, 65536);
        assert_eq!(s.explicit_fits, None);
        assert!(s.full_offload);
        assert_eq!(est.per_device.len(), 1);
        let d = &est.per_device[0];
        assert_eq!(d.key, "pci:x:bus08");
        assert_eq!(d.fraction, 1.0);
        assert_eq!(d.weights_bytes, 16_806_810_208);
        let p = 12288 - 256;
        assert_eq!(d.kv_bytes, 400 * 1024 * 1024 + 450_560 * p + 53 * p * p);
        assert_eq!(s.request_bytes, d.kv_bytes);
        // 566 MiB reserve (256 + 256 + 54) + sc_embT (262144 x 2816 x 2 B).
        assert_eq!(d.compute_bytes, 566 * 1024 * 1024 + 262_144 * 2816 * 2);
        assert_eq!(est.total_bytes, d.total_bytes);
        assert!(est.assumptions.iter().any(|a| a.contains("prompt working set (measured")), "{:?}", est.assumptions);
        assert!(est.assumptions.iter().any(|a| a.contains("GPU_RESOURCE_CACHE_SIZE")));
        assert!(est.assumptions.iter().any(|a| a.contains("WDDM")));

        // FIDIM's default launch turns the runtime cache off: only the
        // prefill term changes.
        let (est, _) = estimate_diffusion(&h, 99, 0, false, DgRunner::STOCK, "k", 32472);
        assert_eq!(est.per_device[0].kv_bytes, 400 * 1024 * 1024 + 450_560 * p + 14 * p * p);
        assert!(!est.assumptions.iter().any(|a| a.contains("GPU_RESOURCE_CACHE_SIZE")));

        // f16 store under flash attention halves the per-token cost, and the
        // stock runner still sizes by the scores tensor.
        let (_, s) = estimate_diffusion(&h, 99, 0, true, DgRunner::STOCK, "k", 32472);
        assert_eq!(s.pkv_bytes_per_token, 225_280);
        assert_eq!(s.predicted_auto_maxtok, Some(12288));
        assert!(!s.fa_sized);

        // Partial offload: NGL 30 leaves the output layer on the CPU.
        let (est, s) = estimate_diffusion(&h, 30, 0, false, DgRunner::STOCK, "k", 32472);
        assert!(!s.full_offload);
        assert!(est.per_device[0].weights_bytes < 16_806_810_208);

        // An explicit budget is used as given, and checked against the gate.
        let (est, s) = estimate_diffusion(&h, 99, 8192, false, STOCK_CACHE_ON, "k", 32472);
        assert_eq!(s.maxtok_used, 8192);
        assert_eq!(s.predicted_auto_maxtok, Some(12288));
        assert_eq!(s.explicit_fits, Some(true));
        let p = 8192 - 256;
        assert_eq!(est.per_device[0].kv_bytes, 400 * 1024 * 1024 + 450_560 * p + 53 * p * p);
        // 14000 is no candidate but passes the gate (12.5 GB of scores against 15.5).
        let (_, s) = estimate_diffusion(&h, 99, 14000, false, DgRunner::STOCK, "k", 32472);
        assert_eq!(s.explicit_fits, Some(true));
        let (_, s) = estimate_diffusion(&h, 99, 32768, false, DgRunner::STOCK, "k", 32472);
        assert_eq!(s.explicit_fits, Some(false));
        assert!(!s.explicit_capped, "only FA sizing caps an explicit budget");
        // A refused budget is sized at what the runner degrades it to.
        assert_eq!(s.maxtok_used, 12288);
    }

    /// The per-request term against the stock runner's measured resident VRAM
    /// on gfx1201 (b11027, Q4_K_M, FA off), taken over the load (17.90 GiB,
    /// against 17.98 estimated). Runtime cache on: +3.88 GiB after a
    /// 5,026-token prompt and +10.0 GiB after 10,526. Cache off (fresh runner
    /// per prompt): +2.82 after 5,014, +2.91 after 5,230, +5.75 after 10,012.
    #[test]
    fn diffusion_prompt_term_matches_measurements() {
        let h = diffusion_gemma_header();
        let cases = [
            (STOCK_CACHE_ON, 5_026u64, 3.88),
            (STOCK_CACHE_ON, 10_526, 10.0),
            (DgRunner::STOCK, 5_014, 2.82),
            (DgRunner::STOCK, 5_230, 2.91),
            (DgRunner::STOCK, 10_012, 5.75),
        ];
        for (runner, prompt, measured_gib) in cases {
            let (est, _) = estimate_diffusion(&h, 99, prompt + 256, false, runner, "k", 32472);
            let got = est.per_device[0].kv_bytes as f64 / GIB;
            assert!(
                (got / measured_gib - 1.0).abs() < 0.05,
                "{prompt} tokens ({runner:?}): {got:.2} GiB vs {measured_gib}"
            );
        }
        let (est, _) = estimate_diffusion(&h, 99, 0, false, DgRunner::STOCK, "k", 32472);
        let d = &est.per_device[0];
        let at_load = (d.weights_bytes + d.compute_bytes + d.overhead_bytes) as f64 / GIB;
        assert!((at_load / 17.90 - 1.0).abs() < 0.02, "at load {at_load:.2} GiB vs 17.90");
    }

    /// The patched runner with FA on sizes by its own per-turn estimate
    /// (llama_diffusion_fa_turn_bytes). By hand for 26B-A4B: the F16 store
    /// keeps 5 full layers x 512 x 2 x 2 x 2 B = 20,480 B per position, the
    /// 25 sliding layers a ring of 1023 + 2048 rows x 8,192 B = 628,940,800 B,
    /// the DECODE copies take 2,048 x 4 x 3 = 24,576 B per key, and the canvas
    /// logits 256 x 262,144 x 4 B.
    #[test]
    fn patched_runner_fa_sizing() {
        let h = diffusion_gemma_header();
        let r = patched();
        assert!(r.pkv_f16 && r.swa_ring && r.fa_pad && r.fa_turn_sizing && r.runtime_cache_off);
        let turn = |n: u64| 45_056 * n + 628_940_800 + 268_435_456;

        let (est, s) = estimate_diffusion(&h, 99, 0, true, r, "k", 32472);
        assert!(s.fa_sized);
        assert_eq!(s.predicted_auto_maxtok, Some(65536), "the runner reached 65,536 on this card");
        assert_eq!(s.maxtok_used, 65536);
        assert_eq!(s.pkv_bytes_per_token, 20_480);
        assert_eq!(s.request_bytes, turn(65536));
        let d = &est.per_device[0];
        assert_eq!(d.kv_bytes, turn(65536));
        // The reserve at the 2048-token FA chunk is the stock 566 MiB one;
        // the runner measured 17.98 GiB at load.
        assert_eq!(d.compute_bytes, 566 * 1024 * 1024 + 262_144 * 2816 * 2);
        let at_load = (d.weights_bytes + d.compute_bytes + d.overhead_bytes) as f64 / GIB;
        assert!((at_load / 17.98 - 1.0).abs() < 0.02, "at load {at_load:.2} GiB vs 17.98");
        assert!(est.assumptions.iter().any(|a| a.contains("patched runner's own FA sizing")), "{:?}", est.assumptions);

        // The runner, told 3000 MiB were free after the weights
        // (DG_FREE_VRAM_MB), picked 40,960: the same walk here.
        let free_mib = 16_806_810_208u64.div_ceil(1024 * 1024) + 3000;
        let (_, s) = estimate_diffusion(&h, 99, 0, true, r, "k", free_mib);
        assert_eq!(s.predicted_auto_maxtok, Some(40960));

        // Measured growth over the load stays under the estimate, within 20%:
        // +1.21 GiB after a 10,012-token prompt, +2.88 after 60,618.
        for (prompt, measured_gib) in [(10_012u64, 1.21), (60_618, 2.88)] {
            let (_, s) = estimate_diffusion(&h, 99, prompt + 256, true, r, "k", 32472);
            let got = s.request_bytes as f64 / GIB;
            assert!(got >= measured_gib && got <= measured_gib * 1.25, "{prompt}: {got:.2} GiB vs {measured_gib}");
            assert_eq!(s.explicit_fits, Some(true));
        }

        // An explicit budget the runner's gate refuses degrades to the prediction.
        let (_, s) = estimate_diffusion(&h, 99, 65536, true, r, "k", free_mib);
        assert_eq!(s.explicit_fits, Some(false));
        assert_eq!(s.maxtok_used, 40960);
        assert_eq!(s.request_bytes, turn(40960));

        // Above the FA ceiling the runner ignores an explicit budget and auto-sizes.
        let (_, s) = estimate_diffusion(&h, 99, 70_000, true, r, "k", 32472);
        assert!(s.explicit_capped);
        assert_eq!(s.explicit_fits, None);
        assert_eq!(s.maxtok_used, 65536);

        // FA off on the same build: the scores gate again, but the F16 ring store.
        let (est, s) = estimate_diffusion(&h, 99, 0, false, r, "k", 32472);
        assert!(!s.fa_sized);
        assert_eq!(s.predicted_auto_maxtok, Some(12288));
        assert_eq!(s.pkv_bytes_per_token, 20_480);
        let p = 12288 - 256;
        // The FA-off chunk at 12288 is 2048, so the ring is 1023 + 2048 rows.
        assert_eq!(est.per_device[0].kv_bytes, 400 * 1024 * 1024 + 20_480 * p + 628_940_800 + 14 * p * p);

        // Without a sliding window in the header the ring is not assumed.
        let mut h2 = diffusion_gemma_header();
        h2.sliding_window = None;
        let (est, s) = estimate_diffusion(&h2, 99, 0, true, r, "k", 32472);
        assert_eq!(s.pkv_bytes_per_token, 225_280);
        assert!(est.assumptions.iter().any(|a| a.contains("sliding_window missing")));
    }

    /// The same numbers from the real file's header, when it is present.
    #[test]
    fn diffusion_sizing_matches_the_real_header() {
        let path = std::path::Path::new(
            "E:/models/unsloth/hub/models--unsloth--diffusiongemma-26B-A4B-it-GGUF/snapshots/\
             f4183a2c7a354128d02545752303c4354d165bf0/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
        );
        if !path.exists() {
            eprintln!("real model absent; diffusion header check skipped");
            return;
        }
        let h = gguf::read_header(path).unwrap();
        let (est, s) = estimate_diffusion(&h, 99, 0, false, DgRunner::STOCK, "k", 32472);
        eprintln!("sizing {s:?}; breakdown {}", est.per_device[0].breakdown());
        assert_eq!(s.predicted_auto_maxtok, Some(12288));
        assert_eq!(s.pkv_bytes_per_token, 450_560);
        assert_eq!(s.canvas, 256);
        let (_, s) = estimate_diffusion(&h, 99, 0, true, patched(), "k", 32472);
        assert_eq!(s.predicted_auto_maxtok, Some(65536));
        assert_eq!(s.pkv_bytes_per_token, 20_480);
        assert_eq!(s.request_bytes, 45_056 * 65536 + 628_940_800 + 268_435_456);
    }

    #[test]
    fn missing_metadata_is_conservative_and_announced() {
        // Header with nothing in it: KV unestimable → announced loudly.
        let bytes = {
            let mut v = Vec::new();
            v.extend_from_slice(b"GGUF");
            v.extend_from_slice(&3u32.to_le_bytes());
            v.extend_from_slice(&0u64.to_le_bytes());
            v.extend_from_slice(&0u64.to_le_bytes());
            v
        };
        let dir = std::env::temp_dir().join("fidim-est-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("empty-{}.gguf", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let header = gguf::read_header(&path).unwrap();
        let est = estimate(&EstimateInput {
            header: &header,
            runtime: &worker_pool_runtime(),
            devices: vec![("k".into(), 1.0)],
            split_mode: None,
            main_index: 0,
            mmproj_bytes: 0,
            draft_bytes: 0,
        });
        assert!(est.assumptions.iter().any(|a| a.contains("DO NOT trust")));
        std::fs::remove_file(path).ok();
    }
}
