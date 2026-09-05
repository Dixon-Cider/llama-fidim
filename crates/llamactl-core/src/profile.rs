//! Profile schema v1 (spec §06) — JSON on disk, one file per profile,
//! hand-editable, unknown fields preserved. The `devices` array is plural
//! from day one (v1.1 amendment); a single-device profile is a one-element
//! list with `split_mode: null`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub schema: u32,
    pub id: String,
    pub name: String,
    pub build: BuildRef,
    pub model: ModelRef,
    pub devices: Vec<DeviceRef>,
    /// `None` = single device. `Layer` is the supported multi-GPU path;
    /// `Row` is experimental until measured on this stack (R-14).
    #[serde(default)]
    pub split_mode: Option<SplitMode>,
    /// Index into `devices[]`, in remapped visibility-pinned order (R-13).
    #[serde(default)]
    pub main_device: u32,
    /// Named ROCm runtime (see `runtime::discover`); None = config default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rocm_runtime: Option<String>,
    pub server: ServerCfg,
    pub runtime: Runtime,
    #[serde(default)]
    pub sampling: Sampling,
    #[serde(default)]
    pub chat: Chat,
    /// Arbitrary env passthrough — hardware workarounds change without
    /// tool releases (§06).
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Measured baseline, written by the benchmark runner (R-09). Kept as
    /// loose JSON until M8 gives it a shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub notes: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildRef {
    pub path: PathBuf,
    #[serde(default)]
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    pub path: PathBuf,
    #[serde(default)]
    pub mmproj: Option<PathBuf>,
    #[serde(default)]
    pub draft: Option<DraftRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftRef {
    pub path: PathBuf,
    #[serde(default)]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRef {
    /// Stable key (`pci:<hw>:busNN`) — never an index (R-03).
    pub key: String,
    /// Proportion of the model on this device. `None` on all entries =
    /// auto-split by free VRAM at launch.
    #[serde(default)]
    pub split_fraction: Option<f64>,
    /// Informational only, never authoritative (§06).
    #[serde(default)]
    pub resolved_index_last_launch: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitMode {
    Layer,
    Row,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCfg {
    pub port: u16,
    pub alias: String,
    #[serde(default = "default_host")]
    pub host: String,
}

fn default_host() -> String {
    "127.0.0.1".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Runtime {
    #[serde(default = "default_ngl")]
    pub n_gpu_layers: u32,
    pub ctx_total: u64,
    #[serde(default = "default_slots")]
    pub slots: u32,
    #[serde(default = "default_kv_type")]
    pub kv_type_k: String,
    #[serde(default = "default_kv_type")]
    pub kv_type_v: String,
    #[serde(default = "default_flash_attn")]
    pub flash_attn: String,
    /// `-b`: shared per-iteration budget for decode + prefill (R-10).
    #[serde(default = "default_batch_logical")]
    pub batch_logical: u32,
    /// `-ub`: sizes the compute buffer (R-10).
    #[serde(default = "default_batch_physical")]
    pub batch_physical: u32,
    #[serde(default = "default_true")]
    pub cont_batching: bool,
    #[serde(default)]
    pub kv_unified: bool,
    #[serde(default)]
    pub cache_reuse: Option<u32>,
    /// Unknown llama-server flags pass through without a tool update (§07).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_flags: Vec<String>,
}

fn default_ngl() -> u32 { 99 }
fn default_slots() -> u32 { 1 }
fn default_kv_type() -> String { "f16".into() }
fn default_flash_attn() -> String { "on".into() }
fn default_batch_logical() -> u32 { 2048 }
fn default_batch_physical() -> u32 { 512 }
fn default_true() -> bool { true }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Sampling {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_multiplier: Option<f64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Chat {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_thinking: Option<bool>,
}

// ------------------------------------------------------------ load/save ----

impl Profile {
    pub fn load(path: &Path) -> Result<Profile> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        let p: Profile = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        if p.schema > SCHEMA_VERSION {
            return Err(Error::Config(format!(
                "{}: schema {} is newer than this tool understands ({SCHEMA_VERSION})",
                path.display(),
                p.schema
            )));
        }
        Ok(p)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).map_err(|e| Error::io(path, e))
    }

    pub fn load_all(dir: &Path) -> Result<Vec<Profile>> {
        let mut out = Vec::new();
        if !dir.exists() {
            return Ok(out);
        }
        let entries = std::fs::read_dir(dir).map_err(|e| Error::io(dir, e))?;
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("json")) {
                out.push(Profile::load(&p)?);
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }
}

// ----------------------------------------------------------- validation ----

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub severity: Severity,
    pub code: &'static str,
    pub message: String,
}

fn finding(severity: Severity, code: &'static str, message: String) -> Finding {
    Finding { severity, code, message }
}

/// Parameter-interaction validation (R-10) plus structural checks. Pure —
/// no filesystem or device access — so the profile editor can run it live.
pub fn validate(p: &Profile) -> Vec<Finding> {
    let mut out = Vec::new();
    let err = Severity::Error;
    let warn = Severity::Warning;

    if p.devices.is_empty() {
        out.push(finding(err, "no-devices", "profile targets no devices".into()));
    }
    if p.devices.len() > 1 && p.split_mode.is_none() {
        out.push(finding(
            err,
            "split-mode-missing",
            format!(
                "{} devices listed but split_mode is null — multi-device requires \"layer\" (or experimental \"row\")",
                p.devices.len()
            ),
        ));
    }
    if p.devices.len() == 1 && p.split_mode.is_some() {
        out.push(finding(
            warn,
            "split-mode-single",
            "split_mode set on a single-device profile has no effect".into(),
        ));
    }
    if p.split_mode == Some(SplitMode::Row) {
        out.push(finding(
            warn,
            "row-split-experimental",
            "row split is unmeasured on this stack (Windows/gfx1201, no P2P DMA) — benchmark before trusting it".into(),
        ));
    }
    if p.main_device as usize >= p.devices.len().max(1) {
        out.push(finding(
            err,
            "main-device-range",
            format!("main_device {} is out of range for {} devices", p.main_device, p.devices.len()),
        ));
    }
    // Split fractions: all set (summing to ~1) or none set (auto by free VRAM).
    let set: Vec<f64> = p.devices.iter().filter_map(|d| d.split_fraction).collect();
    if !set.is_empty() {
        if set.len() != p.devices.len() {
            out.push(finding(
                err,
                "split-fractions-partial",
                "split_fraction must be set on every device or on none (none = auto by free VRAM)".into(),
            ));
        } else if p.devices.len() > 1 {
            let sum: f64 = set.iter().sum();
            if (sum - 1.0).abs() > 0.01 {
                out.push(finding(
                    err,
                    "split-fractions-sum",
                    format!("split_fractions sum to {sum:.3}, expected 1.0"),
                ));
            }
        }
    }

    // Batch coupling (R-10). The failure this encodes: a launcher derived
    // -b and -ub from one variable; lowering -ub to shrink the compute
    // buffer silently throttled the shared decode+prefill budget.
    let r = &p.runtime;
    if r.batch_physical > r.batch_logical {
        out.push(finding(
            err,
            "batch-physical-exceeds-logical",
            format!(
                "batch_physical {} > batch_logical {} — the physical batch can never exceed the logical budget",
                r.batch_physical, r.batch_logical
            ),
        ));
    }
    if r.batch_logical == r.batch_physical && r.batch_logical < 2048 {
        out.push(finding(
            warn,
            "batch-coupled-low",
            format!(
                "batch_logical and batch_physical are both {} — if the intent was to shrink the compute \
                 buffer, lower batch_physical only and leave batch_logical at 2048 (shared decode+prefill budget)",
                r.batch_logical
            ),
        ));
    }
    // Context is divided across slots (R-10).
    if r.slots == 0 {
        out.push(finding(err, "slots-zero", "slots must be at least 1".into()));
    } else {
        if r.ctx_total % r.slots as u64 != 0 {
            out.push(finding(
                warn,
                "ctx-not-divisible",
                format!(
                    "ctx_total {} is not divisible by {} slots — per-slot context truncates to {}",
                    r.ctx_total,
                    r.slots,
                    r.ctx_total / r.slots as u64
                ),
            ));
        }
        let per_slot = r.ctx_total / r.slots as u64;
        if per_slot < 4096 {
            out.push(finding(
                warn,
                "per-slot-ctx-small",
                format!(
                    "per-slot context is only {per_slot} tokens ({} total / {} slots) — raising slots shrinks per-slot context unless ctx_total rises with it",
                    r.ctx_total, r.slots
                ),
            ));
        }
    }
    for kv in [&r.kv_type_k, &r.kv_type_v] {
        if !["f16", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1", "bf16", "f32"].contains(&kv.as_str()) {
            out.push(finding(warn, "kv-type-unknown", format!("unrecognised KV cache type {kv:?}")));
        }
    }
    if p.server.alias.trim().is_empty() {
        out.push(finding(err, "alias-empty", "server alias must not be empty".into()));
    }
    if p.server.port < 1024 {
        out.push(finding(warn, "port-privileged", format!("port {} is in the privileged range", p.server.port)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_profile() -> Profile {
        serde_json::from_value(serde_json::json!({
            "schema": 1,
            "id": "worker-pool",
            "name": "Subagent worker pool",
            "build": { "path": "C:/b/build-hip-vision", "version": "b9817" },
            "model": { "path": "E:/models/m.gguf" },
            "devices": [ { "key": "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08" } ],
            "server": { "port": 9701, "alias": "worker-pool" },
            "runtime": { "ctx_total": 393216, "slots": 6, "kv_type_k": "q8_0",
                         "kv_type_v": "q8_0", "batch_logical": 2048, "batch_physical": 256 }
        }))
        .unwrap()
    }

    #[test]
    fn valid_single_device_profile_has_no_findings() {
        let f = validate(&base_profile());
        assert!(f.is_empty(), "unexpected findings: {f:?}");
    }

    #[test]
    fn multi_device_without_split_mode_is_an_error() {
        let mut p = base_profile();
        p.devices.push(DeviceRef {
            key: "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus03".into(),
            split_fraction: None,
            resolved_index_last_launch: None,
        });
        let f = validate(&p);
        assert!(f.iter().any(|x| x.code == "split-mode-missing" && x.severity == Severity::Error));
    }

    #[test]
    fn split_fractions_must_sum_to_one() {
        let mut p = base_profile();
        p.split_mode = Some(SplitMode::Layer);
        p.devices[0].split_fraction = Some(0.7);
        p.devices.push(DeviceRef {
            key: "pci:X:bus03".into(),
            split_fraction: Some(0.7),
            resolved_index_last_launch: None,
        });
        let f = validate(&p);
        assert!(f.iter().any(|x| x.code == "split-fractions-sum"));
    }

    /// The exact run-server.bat UB bug: `-b` and `-ub` driven from one
    /// variable, throttling the shared prefill/decode budget.
    #[test]
    fn ub_bug_regression_warns_on_coupled_low_batches() {
        let mut p = base_profile();
        p.runtime.batch_logical = 256;
        p.runtime.batch_physical = 256;
        let f = validate(&p);
        assert!(f.iter().any(|x| x.code == "batch-coupled-low"));
        // The correct configuration (2048 / 256) does not warn.
        let ok = validate(&base_profile());
        assert!(!ok.iter().any(|x| x.code == "batch-coupled-low"));
    }

    #[test]
    fn round_trip_preserves_unknown_fields_and_split() {
        let raw = serde_json::json!({
            "schema": 1,
            "id": "big-split",
            "name": "60GB model across both cards",
            "build": { "path": "C:/b" },
            "model": { "path": "E:/models/big.gguf" },
            "devices": [
                { "key": "pci:A:bus03", "split_fraction": 0.5 },
                { "key": "pci:A:bus08", "split_fraction": 0.5 }
            ],
            "split_mode": "layer",
            "main_device": 0,
            "server": { "port": 9700, "alias": "big" },
            "runtime": { "ctx_total": 65536 },
            "future_field": { "keep": true }
        });
        let p: Profile = serde_json::from_value(raw).unwrap();
        assert_eq!(p.split_mode, Some(SplitMode::Layer));
        assert_eq!(p.devices.len(), 2);
        let out = serde_json::to_value(&p).unwrap();
        assert_eq!(out["future_field"]["keep"], true);
        assert_eq!(out["split_mode"], "layer");
    }

    #[test]
    fn newer_schema_is_refused() {
        let dir = std::env::temp_dir().join(format!("llamactl-prof-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("future.json");
        std::fs::write(
            &path,
            r#"{"schema": 99, "id": "x", "name": "x",
               "build": {"path": "C:/b"}, "model": {"path": "C:/m.gguf"},
               "devices": [], "server": {"port": 1, "alias": "x"},
               "runtime": {"ctx_total": 1}}"#,
        )
        .unwrap();
        assert!(Profile::load(&path).is_err());
        std::fs::remove_dir_all(dir).ok();
    }
}
