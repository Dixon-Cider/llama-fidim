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

/// How a DiffusionGemma run's context budget (MAXTOK) will come out: shown
/// by pre-flight check 15 and read by the profile editor.
#[derive(Debug, Clone, Serialize)]
pub struct DiffusionSizing {
    /// Tokens denoised per block (`diffusion.canvas_length`).
    pub canvas: u32,
    /// What the runner's auto-size picks with the card otherwise empty.
    /// None = no candidate fits; the runner then falls back to its floor.
    pub predicted_auto_maxtok: Option<u32>,
    /// The budget this launch runs with: `ctx_total` when set, else the
    /// prediction (else the floor).
    pub maxtok_used: u32,
    /// A reply needs at least one whole canvas of the budget.
    pub max_prompt_tokens: u32,
    /// The prompt-KV store per prompt token: K and V for every layer, f32
    /// (f16 under flash attention).
    pub pkv_bytes_per_token: u64,
    /// NGL covers every block plus the output layer.
    pub full_offload: bool,
}

/// The runner's MAXTOK candidates, largest first (VS:182).
const DG_MAXTOK_CANDIDATES: [u64; 12] =
    [65536, 49152, 40960, 32768, 24576, 20480, 16384, 12288, 8192, 6144, 4096, 2048];

/// Prefill buffers that grow with the square of the prompt, per token².
/// Measured, not derived: see `estimate_diffusion`.
const DG_PREFILL_BYTES_PER_TOKEN_SQ: u64 = 53;

/// What the first request adds regardless of prompt length (measured).
const DG_FIRST_REQUEST_BYTES: u64 = 400 * 1024 * 1024;

/// The runner's smallest auto-sized context: four canvases, never below 2048.
fn dg_floor(canvas: u64) -> u64 {
    (canvas.max(1) * 4).max(2048)
}

/// The MAXTOK the runner's VRAM-gated auto-size settles on (VS:180-212),
/// given the bytes left on the card after the weights. With
/// DG_FREE_RAM_MB=0 (FIDIM's full-offload default) that pass is the only
/// one, so this is deterministic up to the probe's own allocation check.
/// The runner skips the scores gate only when it cannot read VRAM at all;
/// here a zero budget simply fits nothing.
pub fn predict_auto_maxtok(n_head: u64, canvas: u64, n_ctx_train: u64, budget_bytes: u64) -> Option<u32> {
    let n_head = n_head.max(1);
    let canvas = canvas.max(1);
    let ceil = if n_ctx_train > 0 { n_ctx_train.min(65536) } else { 65536 };
    let floor = dg_floor(canvas);
    for raw in DG_MAXTOK_CANDIDATES {
        if raw > ceil {
            continue;
        }
        let n = raw / canvas * canvas; // whole canvases only
        if n < floor {
            break;
        }
        // The fp32 [n_head, N, N] scores buffer (FA off) must fit in 90%.
        let scores = n_head as f64 * n as f64 * n as f64 * 4.0;
        if scores <= budget_bytes as f64 * 0.9 {
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

/// VRAM estimate and context sizing for a DiffusionGemma run on one card
/// with `free_mib` free before the load. The terms beyond the weights are
/// partly inferred and say so in `assumptions`, so a Block or Warn built on
/// them shows how much is measured.
pub fn estimate_diffusion(
    h: &GgufHeader,
    n_gpu_layers: u32,
    ctx_total: u64,
    flash_attn: bool,
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

    // The runner sizes against the VRAM its device reports once the weights
    // are resident, which is this card's free memory minus the weights.
    let n_head = h.head_count.unwrap_or_else(|| {
        assumptions.push("attention.head_count missing; the context prediction assumes 1 head".into());
        1
    });
    let budget = (free_mib * MIB).saturating_sub(weights);
    let predicted = predict_auto_maxtok(n_head, canvas, h.context_length.unwrap_or(0), budget);
    let floor = dg_floor(canvas);
    let maxtok_used = if ctx_total > 0 { ctx_total } else { predicted.map(u64::from).unwrap_or(floor) };
    if ctx_total == 0 && predicted.is_none() {
        assumptions.push(format!(
            "no context candidate fits the VRAM left after the weights; the runner falls back to its \
             {floor}-token floor"
        ));
    }
    let prompt_tokens = maxtok_used.saturating_sub(canvas);

    // Prompt-KV store (diffusion-gemma.cpp dg_ensure_pkv_store): K and V of
    // n_embd_head_k(l) x n_head_kv(l) per prompt token on every layer.
    let fallback_dim = h.embedding_length.unwrap_or(0).checked_div(n_head).unwrap_or(0);
    let k_full = h.key_length.unwrap_or(fallback_dim);
    let k_swa = h.key_length_swa.unwrap_or(k_full);
    if h.key_length.is_none() {
        assumptions
            .push(format!("attention.key_length missing; using embedding/head_count = {fallback_dim}"));
    }
    let mut kv_heads_assumed = false;
    let elt: u64 = if flash_attn { 2 } else { 4 };
    let mut pkv_bytes_per_token = 0u64;
    for layer in 0..block_count as usize {
        let heads = match (&h.head_count_kv_per_layer, h.head_count_kv) {
            (Some(arr), _) if layer < arr.len() => arr[layer],
            (_, Some(s)) => s,
            _ => {
                kv_heads_assumed = true;
                n_head
            }
        };
        let dim = if layer_is_sliding(h, layer) { k_swa } else { k_full };
        pkv_bytes_per_token += dim * heads * 2 * elt;
    }
    if kv_heads_assumed {
        assumptions.push("KV head count missing for some layers; assuming one per attention head".into());
    }
    // What a request adds on top of the load, which the runner keeps after
    // the request ends: the prompt-KV store plus prefill working buffers that
    // grow faster than the prompt, plus a fixed first-request part. Fitted to
    // the runner's resident VRAM on gfx1201 (b11027, FA off): 5,026 prompt
    // tokens -> +3.88 GiB and 10,526 -> +10.0 GiB over the load, within ~4%.
    let kv_bytes = DG_FIRST_REQUEST_BYTES
        + pkv_bytes_per_token * prompt_tokens
        + DG_PREFILL_BYTES_PER_TOKEN_SQ * prompt_tokens * prompt_tokens;

    // Compute: the reserve at the runner's chunked ubatch plus the canvas
    // logits plus a fixed part, which together match the measured 566.01 MiB
    // reserve; and the self-conditioning embedding copy (sc_embT), inferred.
    let vocab = h.vocab_size.unwrap_or_else(|| {
        assumptions.push("tokenizer vocabulary size missing; logits and sc_embT terms omitted".into());
        0
    });
    let n_embd = h.embedding_length.unwrap_or(0);
    let n = maxtok_used.max(1);
    let ub = n.min(((1u64 << 30) / (n_head.max(1) * n)).clamp(256, 2048));
    let sc_emb_t = vocab * n_embd * 2;
    let compute_bytes = n_head * ub * ub * 4 + canvas * vocab * 4 + 54 * MIB + sc_emb_t;
    let overhead_bytes = (0.4 * GIB) as u64;

    assumptions.push(format!(
        "prompt working set (measured on gfx1201, FA off): allocated per request and kept, not at load \
         — up to {:.2} GiB for a {prompt_tokens}-token prompt (prompt-KV store {pkv_bytes_per_token} B/token \
         + prefill buffers {DG_PREFILL_BYTES_PER_TOKEN_SQ} B/token² + 0.4 GiB); a lower context budget \
         shrinks it quadratically",
        kv_bytes as f64 / GIB
    ));
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
            "swa_layer_flags": swa, "diffusion_canvas_length": 256, "vocab_size": 262144,
            "metadata": {}
        }))
        .unwrap()
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

        let (est, s) = estimate_diffusion(&h, 99, 0, false, "pci:x:bus08", 32472);
        assert_eq!(s.predicted_auto_maxtok, Some(12288));
        assert_eq!(s.maxtok_used, 12288);
        assert_eq!(s.max_prompt_tokens, 12288 - 256);
        assert_eq!(s.canvas, 256);
        assert_eq!(s.pkv_bytes_per_token, 450_560);
        assert!(s.full_offload);
        assert_eq!(est.per_device.len(), 1);
        let d = &est.per_device[0];
        assert_eq!(d.key, "pci:x:bus08");
        assert_eq!(d.fraction, 1.0);
        assert_eq!(d.weights_bytes, 16_806_810_208);
        let p = 12288 - 256;
        assert_eq!(d.kv_bytes, 400 * 1024 * 1024 + 450_560 * p + 53 * p * p);
        // 566 MiB reserve (256 + 256 + 54) + sc_embT (262144 x 2816 x 2 B).
        assert_eq!(d.compute_bytes, 566 * 1024 * 1024 + 262_144 * 2816 * 2);
        assert_eq!(est.total_bytes, d.total_bytes);
        assert!(est.assumptions.iter().any(|a| a.contains("prompt working set (measured")), "{:?}", est.assumptions);
        assert!(est.assumptions.iter().any(|a| a.contains("WDDM")));

        // f16 store under flash attention halves the per-token cost.
        let (_, s) = estimate_diffusion(&h, 99, 0, true, "k", 32472);
        assert_eq!(s.pkv_bytes_per_token, 225_280);

        // Partial offload: NGL 30 leaves the output layer on the CPU.
        let (est, s) = estimate_diffusion(&h, 30, 0, false, "k", 32472);
        assert!(!s.full_offload);
        assert!(est.per_device[0].weights_bytes < 16_806_810_208);

        // An explicit budget is used as given.
        let (est, s) = estimate_diffusion(&h, 99, 8192, false, "k", 32472);
        assert_eq!(s.maxtok_used, 8192);
        assert_eq!(s.predicted_auto_maxtok, Some(12288));
        let p = 8192 - 256;
        assert_eq!(est.per_device[0].kv_bytes, 400 * 1024 * 1024 + 450_560 * p + 53 * p * p);
    }

    /// The per-request term against the runner's measured resident VRAM on
    /// gfx1201 (b11027, Q4_K_M, FA off), taken over the load: +3.88 GiB after
    /// a 5,026-token prompt and +10.0 GiB after 10,526. The load itself
    /// measured 17.90 GiB against 17.98 estimated.
    #[test]
    fn diffusion_prompt_term_matches_measurements() {
        let h = diffusion_gemma_header();
        for (prompt, measured_gib) in [(5_026u64, 3.88), (10_526, 10.0)] {
            let (est, _) = estimate_diffusion(&h, 99, prompt + 256, false, "k", 32472);
            let got = est.per_device[0].kv_bytes as f64 / GIB;
            assert!((got / measured_gib - 1.0).abs() < 0.05, "{prompt} tokens: {got:.2} GiB vs {measured_gib}");
        }
        let (est, _) = estimate_diffusion(&h, 99, 0, false, "k", 32472);
        let d = &est.per_device[0];
        let at_load = (d.weights_bytes + d.compute_bytes + d.overhead_bytes) as f64 / GIB;
        assert!((at_load / 17.90 - 1.0).abs() < 0.02, "at load {at_load:.2} GiB vs 17.90");
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
        let (est, s) = estimate_diffusion(&h, 99, 0, false, "k", 32472);
        eprintln!("sizing {s:?}; breakdown {}", est.per_device[0].breakdown());
        assert_eq!(s.predicted_auto_maxtok, Some(12288));
        assert_eq!(s.pkv_bytes_per_token, 450_560);
        assert_eq!(s.canvas, 256);
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
