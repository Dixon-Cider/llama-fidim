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
        let dir = std::env::temp_dir().join("llamactl-est-tests");
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
