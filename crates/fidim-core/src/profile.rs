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
    /// Which server runs this profile. Omitted when llama-server, so every
    /// existing profile file re-saves byte-identically.
    #[serde(default, skip_serializing_if = "Engine::is_llama_server")]
    pub engine: Engine,
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
    /// Keep-alive interval in seconds: a 1-token request this often keeps
    /// the adapter busy so WDDM never evicts the model when the displays
    /// power off (measured 2026-09-05: evicted within 20 s without it).
    /// None = config default; 0 = off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keep_alive_seconds: Option<u32>,
    pub server: ServerCfg,
    pub runtime: Runtime,
    #[serde(default)]
    pub sampling: Sampling,
    /// Speculative decoding. None = off (or legacy `model.draft.enabled`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative: Option<Speculative>,
    /// DiffusionGemma-only settings with no llama-server equivalent. The
    /// shared ones reuse `runtime`: `ctx_total` is MAXTOK (0 = the runner
    /// auto-sizes) and `n_gpu_layers` is NGL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diffusion: Option<DiffusionCfg>,
    /// SGLang engine settings (engine = "sglang"); None = engine defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sglang: Option<crate::sglang::SgLangCfg>,
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

/// The server process a profile launches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Engine {
    #[default]
    LlamaServer,
    /// Unsloth's DiffusionGemma runner behind FIDIM's `fidim-dg.exe` shim.
    DiffusionGemma,
    /// An SGLang server (Linux): launched through sglang-gfx1201's guarded
    /// launcher, fronted by model_router.py, which answers `/slots` and
    /// `/metrics` in llama-server's shape.
    #[serde(rename = "sglang")]
    SgLang,
    /// A value this build does not know (a typo, or a newer FIDIM). Kept so
    /// the profile still loads; it never launches and is never re-saved.
    #[serde(other)]
    Unknown,
}

impl Engine {
    pub fn is_llama_server(&self) -> bool {
        *self == Engine::LlamaServer
    }
    pub fn is_diffusion(&self) -> bool {
        *self == Engine::DiffusionGemma
    }
    pub fn is_sglang(&self) -> bool {
        *self == Engine::SgLang
    }
    pub fn label(&self) -> &'static str {
        match self {
            Engine::LlamaServer => "llama-server",
            Engine::DiffusionGemma => "diffusion-gemma",
            Engine::SgLang => "sglang",
            Engine::Unknown => "unknown",
        }
    }
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
    /// `-t`: CPU threads for the non-offloaded work. None = llama.cpp default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Speculative decoding configuration, mapped onto llama-server's
/// `--spec-type` family (same flags on every build from b9553 to b10771).
///
/// - `mtp`: the model's built-in Multi-Token Prediction head
///   (`nextn_predict_layers` in the GGUF, e.g. Qwen 3.8), or an external MTP
///   head file in `model.draft` (Gemma 4's `MTP/` sidecars).
/// - `draft`: a separate small draft model in `model.draft`.
/// - `dflash`: a DFlash draft file in `model.draft`.
/// - `ngram`: model-free n-gram drafting (`ngram-mod`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Speculative {
    /// `off`, `mtp`, `draft`, `dflash`, `ngram`.
    #[serde(default = "default_spec_mode")]
    pub mode: String,
    /// `--spec-draft-n-max` (engine default 3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_max: Option<u32>,
    /// `--spec-draft-n-min` (engine default 0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_min: Option<u32>,
    /// `--spec-draft-p-min` (engine default 0.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p_min: Option<f64>,
}

fn default_spec_mode() -> String {
    "off".into()
}

impl Default for Speculative {
    fn default() -> Self {
        Speculative { mode: default_spec_mode(), n_max: None, n_min: None, p_min: None }
    }
}

impl Speculative {
    /// llama-server's `--spec-type` value for this mode, None when off.
    pub fn spec_type(&self) -> Option<&'static str> {
        match self.mode.as_str() {
            "mtp" => Some("draft-mtp"),
            "draft" => Some("draft-simple"),
            "dflash" => Some("draft-dflash"),
            "ngram" => Some("ngram-mod"),
            _ => None,
        }
    }
    /// Whether this mode needs a file in `model.draft`.
    pub fn needs_draft_file(&self) -> bool {
        matches!(self.mode.as_str(), "draft" | "dflash")
    }
}

/// Settings for the DiffusionGemma engine that llama-server has no field for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffusionCfg {
    /// ROCBLAS_USE_HIPBLASLT(_BATCHED)=0: without it the first denoise step
    /// intermittently fails with "MUL_MAT failed / ROCm error: invalid
    /// argument"; measured no speed cost.
    #[serde(default = "default_true")]
    pub hipblaslt_safeguard: bool,
    /// FA=1. Separate from `runtime.flash_attn`, whose default is "on":
    /// the 512-dim heads fall back to the CPU on HIP, so this defaults off.
    #[serde(default, skip_serializing_if = "is_false")]
    pub flash_attn: bool,
    /// Reply budget when the client sends no max_tokens; the runner spends
    /// it in whole 256-token canvas blocks.
    #[serde(default = "default_dg_max_tokens")]
    pub default_max_tokens: u32,
    /// Fixed seed for every request; None = a random seed per request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
}

fn default_dg_max_tokens() -> u32 { 2048 }
fn is_false(b: &bool) -> bool { !*b }

impl Default for DiffusionCfg {
    /// Must equal the serde defaults: a profile with no `diffusion` section
    /// and one with `"diffusion": {}` have to behave the same.
    fn default() -> Self {
        DiffusionCfg {
            hipblaslt_safeguard: default_true(),
            flash_attn: false,
            default_max_tokens: default_dg_max_tokens(),
            seed: None,
        }
    }
}

/// Env keys FIDIM composes for every diffusion run. A profile.env entry for
/// any of them could silently add a device (the runner's unified path aborts
/// every prompt), move the run to another card, or fake free memory, so
/// validate refuses them.
pub const DG_OWNED_ENV: &[&str] = &[
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "NGL",
    "MAXTOK",
    "FA",
    "DG_FREE_VRAM_MB",
    "DG_FREE_RAM_MB",
    "GGML_BACKEND_PATH",
    "GGML_CUDA_DEVICES",
    "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
];

/// Test hooks of the locally built patched DiffusionGemma runners (dgpatch4
/// and earlier): they corrupt the prompt-KV store, change its layout under
/// the sizing model, stop the runner, time every graph node, or change the
/// self-conditioning matmul. The helper strips them from the runner's
/// environment; validate warns when a profile sets one. The published
/// dgpatch5 overlay has none of them.
/// (DG_POOL_TRIM passes through: a default-off memory option of those local
/// builds, and inert on overlay and stock builds, which do not carry it.)
pub const DG_TEST_HOOK_ENV: &[&str] = &[
    "DG_PKV_TYPE",
    "DG_SWA_WINDOW",
    "DG_POISON",
    "DG_RING_POISON",
    "DG_DUMP_LOGITS",
    "DG_EXIT_AFTER_DUMP",
    "DG_PROFILE",
    "DG_SC_SPLITK",
    "DG_SC_SPLITK_CHECK",
];

/// Windows env names are case-insensitive (and so is Rust's `Command` env
/// there), so `hip_visible_devices` overrides `HIP_VISIBLE_DEVICES`.
pub fn env_has_key(env: &BTreeMap<String, String>, key: &str) -> bool {
    env.keys().any(|k| k.eq_ignore_ascii_case(key))
}

/// The HIP runtime's resource cache, which keeps freed device memory. FIDIM
/// sends 0 to the DiffusionGemma runner unless the profile env sets the key.
pub const DG_RUNTIME_CACHE_KEY: &str = "GPU_RESOURCE_CACHE_SIZE";

/// Whether a diffusion run has the HIP runtime cache off: FIDIM's default,
/// or the profile env setting the key to 0 itself.
pub fn dg_runtime_cache_off(env: &BTreeMap<String, String>) -> bool {
    env.iter().find(|(k, _)| k.eq_ignore_ascii_case(DG_RUNTIME_CACHE_KEY)).is_none_or(|(_, v)| v.trim() == "0")
}

impl Profile {
    /// The diffusion settings in effect: the explicit section, else defaults.
    pub fn diffusion_effective(&self) -> DiffusionCfg {
        self.diffusion.clone().unwrap_or_default()
    }

    /// The effective speculative config: the explicit section, else the
    /// legacy `model.draft.enabled` translated (MTP sidecar -> `mtp`,
    /// anything else -> `draft`), else off.
    pub fn speculative_effective(&self) -> Speculative {
        if let Some(s) = &self.speculative {
            return s.clone();
        }
        if let Some(d) = &self.model.draft {
            if d.enabled {
                let mode = if d.path.to_string_lossy().to_lowercase().contains("mtp") { "mtp" } else { "draft" };
                return Speculative { mode: mode.into(), n_max: None, n_min: None, p_min: None };
            }
        }
        Speculative::default()
    }
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
        // `#[serde(other)]` would write a mistyped engine back as "unknown",
        // destroying what the user typed. Every save path goes through here,
        // promote included.
        if self.engine == Engine::Unknown {
            return Err(Error::Config(format!(
                "refusing to save profile `{}`: unknown engine; fix the \"engine\" field by hand",
                self.id
            )));
        }
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

// --------------------------------------------------------------- factory ----

/// The first port a new profile gets; the ones below are left to
/// hand-made profiles and the router (the GUI's New starts here too).
pub const FIRST_NEW_PORT: u16 = 9710;
/// A new llama-server profile's context when the user picks none and the
/// estimate allows more.
pub const NEW_PROFILE_CTX: u64 = 32_768;

/// What `new_for_model` takes beyond the model and build.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NewProfileOpts {
    /// The context the user picked; None = the estimate's, capped at
    /// `NEW_PROFILE_CTX`. Always capped at the model's trained context.
    #[serde(default)]
    pub ctx: Option<u64>,
    /// Largest context that fits one card (`catalog::ChoiceFit::max_ctx_one_card`).
    #[serde(default)]
    pub fit_ctx: Option<u64>,
    /// Layer-split over the two largest cards: the model does not fit one.
    #[serde(default)]
    pub split: bool,
    /// A projector to pair; None = the first found beside the model.
    #[serde(default)]
    pub mmproj: Option<PathBuf>,
    /// A draft or MTP head to decode speculatively with.
    #[serde(default)]
    pub draft: Option<PathBuf>,
    /// The speculative mode for `draft` when the caller knows it (the
    /// wizard reads it from the file's path in its repo, whose `MTP/`
    /// folder the download flattens away); None = from the file name
    /// (`draft_mode_of`).
    #[serde(default)]
    pub draft_mode: Option<String>,
    /// Stable keys of cards a running server holds: a new profile goes on
    /// an idle card when there is one.
    #[serde(default)]
    pub busy_cards: Vec<String>,
    /// Ports running servers listen on, besides the profiles' own.
    #[serde(default)]
    pub taken_ports: Vec<u16>,
    /// Display name; None = the model's file name.
    #[serde(default)]
    pub name: Option<String>,
    /// Free text for `notes`, e.g. where the model came from.
    #[serde(default)]
    pub notes: Option<String>,
}

/// The speculative mode a draft file is for, from its path in a repo (or
/// its file name): `dflash` for a DFlash draft, `mtp` for an MTP head
/// (`mtp` in the name, or in an `MTP/` folder), `draft` for a separate
/// small model. None for the heads llama-server runs with spec types a
/// profile cannot name yet (EAGLE3: `draft-eagle3`, DSpark:
/// `draft-dspark`): as `draft` they would be loaded as a standalone model,
/// which they are not. Only the file name and folder names equal to `mtp`
/// count, never a folder that merely contains the letters.
pub fn draft_mode_of(path: &str) -> Option<&'static str> {
    let segs: Vec<&str> = path.split(['/', '\\']).filter(|s| !s.is_empty()).collect();
    let name = segs.last().map(|s| s.to_ascii_lowercase()).unwrap_or_default();
    let in_mtp_dir = segs.iter().rev().skip(1).any(|d| d.eq_ignore_ascii_case("mtp"));
    if name.starts_with("eagle") || name.contains("eagle3") || name.contains("dspark") {
        None
    } else if name.contains("dflash") {
        Some("dflash")
    } else if in_mtp_dir || name.contains("mtp") {
        Some("mtp")
    } else {
        Some("draft")
    }
}

/// What the head `draft_mode_of` refuses is, for a message.
pub fn unsupported_draft_kind(path: &str) -> &'static str {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path).to_ascii_lowercase();
    if name.contains("dspark") { "a DSpark head" } else { "an EAGLE3 head" }
}

/// `K2-Horizon-7B-Q4_K_M` -> `k2-horizon-7b-q4-k-m`: lowercase letters,
/// digits and single dashes, at most 48 characters.
pub fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    let out: String = out.trim_end_matches('-').chars().take(48).collect();
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() { "model".into() } else { out }
}

/// A new profile for `model` on `build`, as the Profiles editor's New then
/// a model pick would make it (same defaults: f16 KV, flash attention on,
/// -b 2048 / -ub 512, one slot, the first projector found beside the
/// model; a diffusion model gets the editor's diffusion defaults), with
/// what the editor leaves to the user filled in:
/// - an id from the model's file name, unique among `existing` (case
///   ignored, as file names are), used as the alias too;
/// - the first free port from `FIRST_NEW_PORT` up that no profile and no
///   running server uses;
/// - the context from the estimate, capped at `NEW_PROFILE_CTX` unless
///   the user picked one, and at the model's trained context;
/// - every layer on the GPU, the output layer included (-ngl above the
///   block count);
/// - an idle discrete card (the first discrete card when all are busy),
///   or the two largest layer-split when the model needs both.
///
/// llama-server always gets `--jinja` from the launch path, so the
/// model's chat template applies without a flag here.
pub fn new_for_model(
    cfg: &crate::config::Config,
    model: &crate::discovery::Model,
    build: &crate::discovery::Build,
    devices: &[crate::devices::Device],
    existing: &[Profile],
    opts: &NewProfileOpts,
) -> Profile {
    use crate::discovery::dg_feature;

    let h = model.header.as_ref();
    let file_name = model.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let stem = crate::gguf::split_name(&file_name)
        .map(|(prefix, _, _)| prefix.to_string())
        .unwrap_or_else(|| file_name.strip_suffix(".gguf").or_else(|| file_name.strip_suffix(".GGUF")).unwrap_or(&file_name).to_string());

    // A unique id: the slug, then -2, -3, ...
    let base_id = slug(&stem);
    let taken_id = |id: &str| existing.iter().any(|p| p.id.eq_ignore_ascii_case(id));
    let mut id = base_id.clone();
    let mut n = 2;
    while taken_id(&id) {
        id = format!("{base_id}-{n}");
        n += 1;
    }

    let mut port = FIRST_NEW_PORT;
    while existing.iter().any(|p| p.server.port == port) || opts.taken_ports.contains(&port) {
        port = port.saturating_add(1);
        if port == u16::MAX {
            break;
        }
    }

    let engine = model.engine;
    let diffusion = engine.is_diffusion();

    // Cards: discrete ones, or every one when integrated graphics may be
    // bound and there is no discrete card.
    let mut cards: Vec<&crate::devices::Device> = devices.iter().filter(|d| !d.integrated).collect();
    if cards.is_empty() && cfg.allow_integrated {
        cards = devices.iter().collect();
    }
    let one = |k: &str| DeviceRef { key: k.to_string(), split_fraction: None, resolved_index_last_launch: None };
    let (device_refs, split_mode) = if opts.split && !diffusion && cards.len() >= 2 {
        let mut by_size = cards.clone();
        by_size.sort_by(|a, b| b.total_mib.cmp(&a.total_mib).then_with(|| b.free_mib.cmp(&a.free_mib)));
        (vec![one(&by_size[0].stable_key), one(&by_size[1].stable_key)], Some(SplitMode::Layer))
    } else {
        let pick = cards.iter().find(|d| !opts.busy_cards.contains(&d.stable_key)).or(cards.first());
        (pick.map(|d| vec![one(&d.stable_key)]).unwrap_or_default(), None)
    };

    let model_ctx = h.and_then(|h| h.context_length).filter(|c| *c > 0);
    let ctx_total = if diffusion {
        // MAXTOK 0: the runner sizes its context to the card at load.
        0
    } else {
        let want = opts.ctx.filter(|c| *c > 0).unwrap_or_else(|| {
            opts.fit_ctx.filter(|c| *c > 0).map_or(NEW_PROFILE_CTX, |f| f.min(NEW_PROFILE_CTX))
        });
        let capped = model_ctx.map_or(want, |m| want.min(m));
        // Whole 256-token steps, as the editor's slider moves.
        if capped >= 512 { capped / 256 * 256 } else { capped.max(1) }
    };
    let block_count = h.and_then(|h| h.block_count).unwrap_or(0);
    let n_gpu_layers = u32::try_from(block_count + 1).unwrap_or(u32::MAX).max(99);

    let mmproj = if diffusion { None } else { opts.mmproj.clone().or_else(|| model.mmproj_candidates.first().cloned()) };
    let (draft, speculative) = match (&opts.draft, diffusion) {
        (Some(d), false) => {
            // The file name only: the folders above are the model's, and a
            // repo named "...mtp..." says nothing about this file.
            let name = d.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let mode = opts.draft_mode.clone().or_else(|| draft_mode_of(&name).map(String::from));
            match mode {
                Some(mode) => (
                    Some(DraftRef { path: d.clone(), enabled: true }),
                    Some(Speculative { mode, n_max: None, n_min: None, p_min: None }),
                ),
                // A head no mode runs: kept in the profile, switched off.
                None => (Some(DraftRef { path: d.clone(), enabled: false }), None),
            }
        }
        _ => (None, None),
    };
    let diffusion_cfg = diffusion.then(|| DiffusionCfg {
        // Flash attention follows the build, as a build pick does in the editor.
        flash_attn: build.patch.as_ref().is_some_and(|p| p.has(dg_feature::FA_PAD)),
        ..DiffusionCfg::default()
    });

    Profile {
        schema: SCHEMA_VERSION,
        engine,
        name: opts.name.clone().filter(|n| !n.trim().is_empty()).unwrap_or_else(|| stem.clone()),
        build: BuildRef { path: build.path.clone(), version: build.version.clone() },
        model: ModelRef { path: model.path.clone(), mmproj, draft },
        devices: device_refs,
        split_mode,
        main_device: 0,
        rocm_runtime: None,
        keep_alive_seconds: None,
        server: ServerCfg { port, alias: id.clone(), host: default_host() },
        runtime: Runtime {
            n_gpu_layers,
            ctx_total,
            slots: default_slots(),
            kv_type_k: default_kv_type(),
            kv_type_v: default_kv_type(),
            flash_attn: default_flash_attn(),
            batch_logical: default_batch_logical(),
            batch_physical: default_batch_physical(),
            cont_batching: true,
            kv_unified: false,
            cache_reuse: None,
            threads: None,
            extra_flags: Vec::new(),
        },
        sampling: Sampling::default(),
        speculative,
        diffusion: diffusion_cfg,
        sglang: None,
        chat: Chat::default(),
        env: BTreeMap::new(),
        baseline: None,
        notes: opts.notes.clone().unwrap_or_default(),
        extra: serde_json::Map::new(),
        id,
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

    if p.engine == Engine::Unknown {
        out.push(finding(
            Severity::Error,
            "engine-unknown",
            "engine is not one this build knows (expected \"llama-server\", \"diffusion-gemma\" or \"sglang\"); \
             fix the \"engine\" field by hand: the profile cannot launch or be saved until then"
                .into(),
        ));
        return out;
    }
    if p.engine.is_diffusion() {
        validate_diffusion(p, &mut out);
        validate_server(p, &mut out);
        return out;
    }
    if p.engine.is_sglang() {
        validate_sglang(p, &mut out);
        validate_server(p, &mut out);
        return out;
    }

    let spec = p.speculative_effective();
    if !matches!(spec.mode.as_str(), "off" | "mtp" | "draft" | "dflash" | "ngram") {
        out.push(finding(
            Severity::Error,
            "spec-mode",
            format!("speculative.mode `{}` is not one of off/mtp/draft/dflash/ngram", spec.mode),
        ));
    }
    if spec.needs_draft_file() && p.model.draft.as_ref().map_or(true, |d| d.path.as_os_str().is_empty()) {
        out.push(finding(
            Severity::Error,
            "spec-draft-file",
            format!("speculative mode `{}` needs a draft file in model.draft", spec.mode),
        ));
    }
    if let (Some(lo), Some(hi)) = (spec.n_min, spec.n_max) {
        if lo > hi {
            out.push(finding(Severity::Error, "spec-range", format!("speculative n_min {lo} exceeds n_max {hi}")));
        }
    }
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
    validate_server(p, &mut out);
    out
}

/// Rules for the listening socket, shared by every engine.
fn validate_server(p: &Profile, out: &mut Vec<Finding>) {
    if p.server.alias.trim().is_empty() {
        out.push(finding(Severity::Error, "alias-empty", "server alias must not be empty".into()));
    }
    if p.server.port < 1024 {
        out.push(finding(
            Severity::Warning,
            "port-privileged",
            format!("port {} is in the privileged range", p.server.port),
        ));
    }
}

/// DiffusionGemma rules. Batch, slot and KV rules do not apply to the
/// runner; instead a few settings that are harmless for llama-server break
/// every prompt here, and those are Errors.
fn validate_sglang(p: &Profile, out: &mut Vec<Finding>) {
    let s = crate::sglang::cfg_of(p);
    if p.devices.len() != 1 {
        out.push(finding(Severity::Error, "sglang-one-device", format!("an SGLang profile runs on exactly one card ({} listed)", p.devices.len())));
    }
    if p.runtime.ctx_total == 0 {
        out.push(finding(Severity::Error, "sglang-context", "runtime.ctx_total (the --context-length) is 0".into()));
    }
    if !(0.3..=0.98).contains(&s.mem_fraction) {
        out.push(finding(Severity::Error, "sglang-mem-fraction", format!("sglang.mem_fraction {} is outside 0.30..0.98", s.mem_fraction)));
    }
    if s.max_running_requests == 0 {
        out.push(finding(Severity::Error, "sglang-slots", "sglang.max_running_requests is 0".into()));
    }
    if p.model.path.as_os_str().is_empty() {
        out.push(finding(Severity::Error, "sglang-model", "model.path is empty".into()));
    }
}

fn validate_diffusion(p: &Profile, out: &mut Vec<Finding>) {
    let err = Severity::Error;
    let warn = Severity::Warning;
    let dg = p.diffusion_effective();

    if p.devices.len() != 1 {
        out.push(finding(
            err,
            "dg-single-device",
            format!(
                "the diffusion engine needs exactly one device (profile lists {}): more than one visible \
                 device takes the runner's unified path, which aborts every prompt",
                p.devices.len()
            ),
        ));
    }
    let owned: Vec<&str> = p
        .env
        .keys()
        .filter(|k| DG_OWNED_ENV.iter().any(|o| k.eq_ignore_ascii_case(o)))
        .map(String::as_str)
        .collect();
    if !owned.is_empty() {
        out.push(finding(
            err,
            "dg-env-owned",
            format!(
                "env sets {}: Llama FIDIM composes these for every diffusion run (card pinning, NGL, \
                 MAXTOK, FA, memory sizing), and an override can add a device or move the run to another \
                 card; remove them from the profile env",
                owned.join(", ")
            ),
        ));
    }
    let draft_on = p.model.draft.as_ref().is_some_and(|d| d.enabled);
    if p.speculative_effective().mode != "off" || draft_on {
        out.push(finding(
            err,
            "dg-speculative",
            "speculative decoding and draft models do not apply to the diffusion engine; set speculative \
             to off and disable model.draft"
                .into(),
        ));
    }
    if !p.model.path.to_string_lossy().is_ascii() {
        out.push(finding(
            err,
            "dg-path-ascii",
            format!(
                "model path {} is not ASCII: the diffusion runner opens it through narrow argv/fopen and \
                 cannot reach it; rename or move the file",
                p.model.path.display()
            ),
        ));
    }
    if dg.default_max_tokens == 0 {
        out.push(finding(err, "dg-max-tokens", "diffusion.default_max_tokens must be at least 1".into()));
    }

    let mut ignored: Vec<String> = Vec::new();
    if p.model.mmproj.as_ref().is_some_and(|m| !m.as_os_str().is_empty()) {
        ignored.push("model.mmproj".into());
    }
    let s = &p.sampling;
    for (name, set) in [
        ("temperature", s.temperature.is_some()),
        ("top_p", s.top_p.is_some()),
        ("top_k", s.top_k.is_some()),
        ("min_p", s.min_p.is_some()),
        ("dry_multiplier", s.dry_multiplier.is_some()),
        ("repeat_penalty", s.repeat_penalty.is_some()),
        ("presence_penalty", s.presence_penalty.is_some()),
    ] {
        if set {
            ignored.push(format!("sampling.{name}"));
        }
    }
    ignored.extend(s.extra.keys().map(|k| format!("sampling.{k}")));
    if p.split_mode.is_some() {
        ignored.push("split_mode".into());
    }
    if p.main_device != 0 {
        ignored.push("main_device".into());
    }
    if p.runtime.slots != 1 {
        ignored.push("runtime.slots".into());
    }
    if !p.runtime.extra_flags.is_empty() {
        ignored.push("runtime.extra_flags".into());
    }
    if p.runtime.cache_reuse.is_some() {
        ignored.push("runtime.cache_reuse".into());
    }
    // rocm_runtime is not listed: a build without its own ROCm does run the
    // diffusion runner on it, and validate cannot see which kind of build
    // this is (the editor clears it for a bundled one).
    if !ignored.is_empty() {
        out.push(finding(
            warn,
            "dg-ignored",
            format!("not used by the diffusion engine: {}", ignored.join(", ")),
        ));
    }
    if p.chat.enable_thinking == Some(false) {
        out.push(finding(
            warn,
            "dg-thinking",
            "chat.enable_thinking=false cannot be honoured: the runner's request file has no \
             chat_template_kwargs, so the model always thinks; the thought channel is split into \
             reasoning_content"
                .into(),
        ));
    }
    if p.keep_alive_seconds.is_some_and(|s| s > 0) {
        out.push(finding(
            warn,
            "dg-keepalive",
            "keep_alive_seconds is not used for the diffusion engine: each keep-alive request would run a \
             whole denoise block"
                .into(),
        ));
    }
    // diffusion.flash_attn is judged at pre-flight (check 15), which knows
    // whether the build's runner pads keys for it; validate cannot see the build.
    let hooks: Vec<&str> = p
        .env
        .keys()
        .filter(|k| DG_TEST_HOOK_ENV.iter().any(|h| k.eq_ignore_ascii_case(h)))
        .map(String::as_str)
        .collect();
    if !hooks.is_empty() {
        out.push(finding(
            warn,
            "dg-test-hook-env",
            format!(
                "env sets {}: the runner's test hooks are never passed to it (they corrupt the prompt-KV store, \
                 change its layout or stop the runner)",
                hooks.join(", ")
            ),
        ));
    }
    if let Some((k, v)) = p.env.iter().find(|(k, _)| k.eq_ignore_ascii_case(DG_RUNTIME_CACHE_KEY)) {
        if v.trim() != "0" {
            out.push(finding(
                warn,
                "dg-resource-cache-env",
                format!(
                    "env sets {k}={v}: the HIP runtime keeps freed device memory, so the runner holds more \
                     after long prompts than FIDIM's default of 0 (measured +3.9 GiB after 10K tokens)"
                ),
            ));
        }
    }
    if !dg.hipblaslt_safeguard {
        out.push(finding(
            warn,
            "dg-safeguard-off",
            "diffusion.hipblaslt_safeguard is off: the first denoise step intermittently fails with \
             'MUL_MAT failed / ROCm error: invalid argument' when rocBLAS routes through hipBLASLt"
                .into(),
        ));
    } else {
        let overrides: Vec<&str> = p
            .env
            .keys()
            .filter(|k| k.to_ascii_uppercase().starts_with("ROCBLAS_USE_HIPBLASLT"))
            .map(String::as_str)
            .collect();
        if !overrides.is_empty() {
            out.push(finding(
                warn,
                "dg-hipblaslt-env",
                format!(
                    "env sets {}: the profile env wins over the hipBLASLt safeguard's value",
                    overrides.join(", ")
                ),
            ));
        }
    }
    let ctx = p.runtime.ctx_total;
    if ctx != 0 && (ctx < 2048 || ctx % 256 != 0 || ctx > 65536) {
        out.push(finding(
            warn,
            "dg-context",
            format!(
                "runtime.ctx_total {ctx} is the diffusion context budget (MAXTOK, 0 = auto-size): expected a \
                 multiple of 256 between 2048 and 65536; with flash attention off the runner's scores buffer \
                 grows with N², so large budgets do not fit"
            ),
        ));
    }
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

    /// A profile exactly as the pre-engine code saved it (every optional
    /// section present). Loading and saving it again must not change a byte:
    /// the engine fields stay invisible on llama-server profiles.
    const LEGACY_SAVED: &str = r#"{
  "schema": 1,
  "id": "daily-driver",
  "name": "Daily driver",
  "build": {
    "path": "C:/llama.cpp/b10819-rocm",
    "version": "b10819"
  },
  "model": {
    "path": "E:/models/qwen.gguf",
    "mmproj": null,
    "draft": {
      "path": "E:/models/MTP/mtp.gguf",
      "enabled": false
    }
  },
  "devices": [
    {
      "key": "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus03",
      "split_fraction": null,
      "resolved_index_last_launch": 0
    }
  ],
  "split_mode": null,
  "main_device": 0,
  "rocm_runtime": "default",
  "keep_alive_seconds": 0,
  "server": {
    "port": 1234,
    "alias": "dd",
    "host": "127.0.0.1"
  },
  "runtime": {
    "n_gpu_layers": 99,
    "ctx_total": 262144,
    "slots": 2,
    "kv_type_k": "q4_0",
    "kv_type_v": "q4_0",
    "flash_attn": "on",
    "batch_logical": 2048,
    "batch_physical": 512,
    "cont_batching": true,
    "kv_unified": false,
    "cache_reuse": 256,
    "threads": 8,
    "extra_flags": [
      "--no-mmap"
    ]
  },
  "sampling": {
    "temperature": 0.6,
    "top_k": 20,
    "future_sampler": 1
  },
  "speculative": {
    "mode": "mtp",
    "n_max": 3
  },
  "chat": {
    "enable_thinking": false
  },
  "env": {
    "GPU_MAX_HW_QUEUES": "1"
  },
  "notes": "hand notes",
  "future_field": {
    "keep": true
  }
}"#;

    #[test]
    fn engine_serde_legacy_profile_resaves_byte_identically() {
        let p: Profile = serde_json::from_str(LEGACY_SAVED).unwrap();
        assert_eq!(p.engine, Engine::LlamaServer);
        assert!(p.diffusion.is_none());
        let text = serde_json::to_string_pretty(&p).unwrap();
        assert_eq!(text, LEGACY_SAVED);
        assert!(!text.contains("\"engine\"") && !text.contains("\"diffusion\""));
        // Through the real save path as well.
        let dir = std::env::temp_dir().join(format!("fidim-prof-legacy-{}", std::process::id()));
        let path = dir.join("daily-driver.json");
        p.save(&path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), LEGACY_SAVED);
        std::fs::remove_dir_all(dir).ok();
    }

    fn diffusion_profile() -> Profile {
        serde_json::from_value(serde_json::json!({
            "schema": 1,
            "engine": "diffusion-gemma",
            "id": "dg-26b",
            "name": "DiffusionGemma 26B",
            "build": { "path": "C:/b/b11027-mix-3e83366-unsloth" },
            "model": { "path": "E:/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf" },
            "devices": [ { "key": "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08" } ],
            "server": { "port": 2345, "alias": "dg" },
            "runtime": { "ctx_total": 0 },
            "diffusion": { "hipblaslt_safeguard": true, "default_max_tokens": 2048 }
        }))
        .unwrap()
    }

    #[test]
    fn engine_serde_diffusion_round_trips_kebab_case() {
        let p = diffusion_profile();
        assert_eq!(p.engine, Engine::DiffusionGemma);
        assert_eq!(p.diffusion_effective(), DiffusionCfg::default());
        let text = serde_json::to_string_pretty(&p).unwrap();
        assert!(text.starts_with("{\n  \"schema\": 1,\n  \"engine\": \"diffusion-gemma\",\n"), "{text}");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["diffusion"], serde_json::json!({ "hipblaslt_safeguard": true, "default_max_tokens": 2048 }));
        let back: Profile = serde_json::from_str(&text).unwrap();
        assert_eq!(back.engine, Engine::DiffusionGemma);
        assert_eq!(back.diffusion, p.diffusion);
        assert_eq!(serde_json::to_string_pretty(&back).unwrap(), text);

        // An empty section means the defaults, same as no section.
        let mut raw = serde_json::to_value(&p).unwrap();
        raw["diffusion"] = serde_json::json!({});
        let empty: Profile = serde_json::from_value(raw).unwrap();
        assert_eq!(empty.diffusion, Some(DiffusionCfg::default()));
        let mut none = p.clone();
        none.diffusion = None;
        assert_eq!(none.diffusion_effective(), DiffusionCfg::default());

        // Non-default values survive; flash_attn=false and seed=None stay out.
        let mut q = p.clone();
        q.diffusion = Some(DiffusionCfg { flash_attn: true, seed: Some(7), ..DiffusionCfg::default() });
        let v = serde_json::to_value(&q).unwrap();
        assert_eq!(v["diffusion"]["flash_attn"], true);
        assert_eq!(v["diffusion"]["seed"], 7);
    }

    #[test]
    fn engine_serde_unknown_engine_loads_but_never_saves() {
        let mut raw = serde_json::to_value(diffusion_profile()).unwrap();
        raw["engine"] = "difusion-gema".into();
        let p: Profile = serde_json::from_value(raw).unwrap();
        assert_eq!(p.engine, Engine::Unknown);
        assert!(!p.extra.contains_key("engine"));

        let f = validate(&p);
        assert_eq!(f.len(), 1, "{f:?}");
        assert_eq!((f[0].code, f[0].severity), ("engine-unknown", Severity::Error));

        let dir = std::env::temp_dir().join(format!("fidim-prof-unknown-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dg-26b.json");
        std::fs::write(&path, "hand-typed original").unwrap();
        match p.save(&path) {
            Err(Error::Config(msg)) => assert!(msg.contains("unknown engine"), "{msg}"),
            other => panic!("expected a Config error, got {other:?}"),
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hand-typed original");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn engine_serde_missing_ctx_total_still_fails_to_parse() {
        for engine in ["llama-server", "diffusion-gemma"] {
            let r = serde_json::from_value::<Profile>(serde_json::json!({
                "schema": 1, "engine": engine, "id": "x", "name": "x",
                "build": { "path": "C:/b" }, "model": { "path": "E:/m.gguf" },
                "devices": [], "server": { "port": 9701, "alias": "x" },
                "runtime": { "slots": 1 }
            }));
            assert!(r.is_err(), "{engine}: a profile without ctx_total must not load");
        }
    }

    fn codes(f: &[Finding], severity: Severity) -> Vec<&'static str> {
        f.iter().filter(|x| x.severity == severity).map(|x| x.code).collect()
    }

    #[test]
    fn validate_diffusion_healthy_profile_has_no_findings() {
        let f = validate(&diffusion_profile());
        assert!(f.is_empty(), "unexpected findings: {f:?}");
    }

    #[test]
    fn validate_diffusion_errors() {
        let only_error = |p: &Profile, code: &str| {
            let errs = codes(&validate(p), Severity::Error);
            assert_eq!(errs, vec![code], "for {code}");
        };

        let mut p = diffusion_profile();
        p.devices.push(DeviceRef { key: "pci:X:bus03".into(), split_fraction: None, resolved_index_last_launch: None });
        only_error(&p, "dg-single-device");
        let mut p = diffusion_profile();
        p.devices.clear();
        only_error(&p, "dg-single-device");

        let mut p = diffusion_profile();
        p.env.insert("hip_visible_devices".into(), "1".into());
        only_error(&p, "dg-env-owned");
        assert!(env_has_key(&p.env, "HIP_VISIBLE_DEVICES"));
        assert!(!env_has_key(&p.env, "CUDA_VISIBLE_DEVICES"));

        let mut p = diffusion_profile();
        p.speculative = Some(Speculative { mode: "mtp".into(), ..Speculative::default() });
        only_error(&p, "dg-speculative");
        let mut p = diffusion_profile();
        p.model.draft = Some(DraftRef { path: "E:/models/MTP/x.gguf".into(), enabled: true });
        only_error(&p, "dg-speculative");
        let mut p = diffusion_profile();
        p.speculative = Some(Speculative::default());
        p.model.draft = Some(DraftRef { path: "E:/models/d.gguf".into(), enabled: false });
        assert!(codes(&validate(&p), Severity::Error).is_empty(), "off + disabled draft is fine");

        let mut p = diffusion_profile();
        p.model.path = "E:/modèles/diffusiongemma.gguf".into();
        only_error(&p, "dg-path-ascii");

        let mut p = diffusion_profile();
        p.diffusion.as_mut().unwrap().default_max_tokens = 0;
        only_error(&p, "dg-max-tokens");
    }

    #[test]
    fn validate_diffusion_warnings() {
        let only_warning = |p: &Profile, code: &str| {
            let f = validate(p);
            assert!(codes(&f, Severity::Error).is_empty(), "{code}: {f:?}");
            assert_eq!(codes(&f, Severity::Warning), vec![code], "for {code}");
        };

        let mut p = diffusion_profile();
        p.model.mmproj = Some("E:/models/mmproj.gguf".into());
        only_warning(&p, "dg-ignored");
        assert!(validate(&p)[0].message.contains("model.mmproj"));

        let mut p = diffusion_profile();
        p.chat.enable_thinking = Some(false);
        only_warning(&p, "dg-thinking");
        p.chat.enable_thinking = Some(true);
        assert!(validate(&p).is_empty());

        let mut p = diffusion_profile();
        p.keep_alive_seconds = Some(5);
        only_warning(&p, "dg-keepalive");
        p.keep_alive_seconds = Some(0);
        assert!(validate(&p).is_empty());

        // Flash attention is judged at pre-flight, which knows the build.
        let mut p = diffusion_profile();
        p.diffusion.as_mut().unwrap().flash_attn = true;
        assert!(validate(&p).is_empty());

        // The runner's test hooks never reach it; setting one is a mistake.
        let mut p = diffusion_profile();
        p.env.insert("dg_pkv_type".into(), "f32".into());
        only_warning(&p, "dg-test-hook-env");
        let mut p = diffusion_profile();
        p.env.insert("DG_PROFILE".into(), "2".into());
        only_warning(&p, "dg-test-hook-env");

        // FIDIM turns the HIP runtime cache off; a profile that turns it on is told the cost.
        let mut p = diffusion_profile();
        p.env.insert("GPU_RESOURCE_CACHE_SIZE".into(), "0".into());
        assert!(validate(&p).is_empty());
        assert!(dg_runtime_cache_off(&p.env));
        p.env.insert("GPU_RESOURCE_CACHE_SIZE".into(), "1024".into());
        only_warning(&p, "dg-resource-cache-env");
        assert!(!dg_runtime_cache_off(&p.env));
        assert!(dg_runtime_cache_off(&BTreeMap::new()));

        let mut p = diffusion_profile();
        p.diffusion.as_mut().unwrap().hipblaslt_safeguard = false;
        only_warning(&p, "dg-safeguard-off");
        // With the safeguard off, the env is the user's call: no double warning.
        p.env.insert("ROCBLAS_USE_HIPBLASLT".into(), "1".into());
        only_warning(&p, "dg-safeguard-off");

        let mut p = diffusion_profile();
        p.env.insert("rocblas_use_hipblaslt_batched".into(), "1".into());
        only_warning(&p, "dg-hipblaslt-env");

        for ctx in [3000u64, 1024, 65536 + 256] {
            let mut p = diffusion_profile();
            p.runtime.ctx_total = ctx;
            only_warning(&p, "dg-context");
        }
        for ctx in [0u64, 2048, 12288, 65536] {
            let mut p = diffusion_profile();
            p.runtime.ctx_total = ctx;
            assert!(validate(&p).is_empty(), "ctx_total {ctx}");
        }

        // Every ignored llama-server knob lands in one finding.
        let mut p = diffusion_profile();
        p.sampling.temperature = Some(1.0);
        p.sampling.extra.insert("xtc_probability".into(), 0.5.into());
        p.split_mode = Some(SplitMode::Layer);
        p.main_device = 1;
        p.runtime.extra_flags = vec!["--no-mmap".into()];
        p.runtime.cache_reuse = Some(256);
        let f = validate(&p);
        let ignored: Vec<&Finding> = f.iter().filter(|x| x.code == "dg-ignored").collect();
        assert_eq!(ignored.len(), 1, "{f:?}");
        for part in [
            "sampling.temperature",
            "sampling.xtc_probability",
            "split_mode",
            "main_device",
            "runtime.extra_flags",
            "runtime.cache_reuse",
        ] {
            assert!(ignored[0].message.contains(part), "missing {part}: {}", ignored[0].message);
        }

        // A runtime is used whenever the build does not bundle its own ROCm,
        // which validate cannot see: never reported as ignored.
        let mut p = diffusion_profile();
        p.rocm_runtime = Some("rocm-7.14.0".into());
        assert!(validate(&p).is_empty(), "{:?}", validate(&p));
    }

    #[test]
    fn validate_diffusion_skips_llama_server_batch_slot_and_kv_rules() {
        let mut p = diffusion_profile();
        p.runtime.batch_logical = 256;
        p.runtime.batch_physical = 512;
        p.runtime.slots = 3;
        p.runtime.ctx_total = 4096;
        p.runtime.kv_type_k = "mystery".into();
        let f = validate(&p);
        for code in [
            "batch-physical-exceeds-logical",
            "batch-coupled-low",
            "ctx-not-divisible",
            "per-slot-ctx-small",
            "kv-type-unknown",
            "slots-zero",
        ] {
            assert!(!f.iter().any(|x| x.code == code), "{code} reported for a diffusion profile: {f:?}");
        }
        assert_eq!(codes(&f, Severity::Warning), vec!["dg-ignored"], "slots != 1 is the only note");

        // The shared server rules still apply.
        p.runtime.slots = 1;
        p.server.alias = " ".into();
        p.server.port = 80;
        assert_eq!(codes(&validate(&p), Severity::Error), vec!["alias-empty"]);
        assert_eq!(codes(&validate(&p), Severity::Warning), vec!["port-privileged"]);
    }

    #[test]
    fn validate_llama_server_results_unchanged_by_explicit_engine() {
        let mut coupled = base_profile();
        coupled.runtime.batch_logical = 256;
        coupled.runtime.batch_physical = 256;
        coupled.server.port = 80;
        for p in [base_profile(), coupled] {
            let implicit: Vec<_> = validate(&p).into_iter().map(|x| (x.code, x.severity, x.message)).collect();
            let mut explicit = p.clone();
            explicit.engine = Engine::LlamaServer;
            let explicit: Vec<_> = validate(&explicit).into_iter().map(|x| (x.code, x.severity, x.message)).collect();
            assert_eq!(implicit, explicit);
        }
        let mut raw = serde_json::to_value(base_profile()).unwrap();
        raw["engine"] = "llama-server".into();
        let p: Profile = serde_json::from_value(raw).unwrap();
        assert_eq!(p.engine, Engine::LlamaServer);
        assert!(serde_json::to_value(&p).unwrap().get("engine").is_none());
    }

    #[test]
    fn newer_schema_is_refused() {
        let dir = std::env::temp_dir().join(format!("fidim-prof-{}", std::process::id()));
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

    // ---------------------------------------------------------- factory ----

    mod factory {
        use super::super::*;
        use crate::config::Config;
        use crate::devices::Device;
        use crate::discovery::{Build, Channel, Model};

        fn card(key: &str, total_mib: u64, integrated: bool) -> Device {
            serde_json::from_value(serde_json::json!({
                "stable_key": key, "name": "card", "hip_index": 0, "backend": "ROCm",
                "total_mib": total_mib, "free_mib": total_mib, "integrated": integrated, "bus_number": 3,
                "driver_version": null, "display": null, "luid_low": null, "correlation_assumed": false
            }))
            .unwrap()
        }

        fn cards() -> Vec<Device> {
            vec![card("pci:a:bus03", 32624, false), card("pci:igpu:bus19", 12381, true), card("pci:a:bus08", 32624, false)]
        }

        fn model(file: &str, arch: &str, ctx: u64, blocks: u64) -> Model {
            let canvas = if arch == "diffusion-gemma" { Some(256) } else { None };
            let header: crate::gguf::GgufHeader = serde_json::from_value(serde_json::json!({
                "path": file, "file_size": 1, "gguf_version": 3, "tensor_count": 0, "architecture": arch,
                "block_count": blocks, "context_length": ctx, "diffusion_canvas_length": canvas,
                "metadata": {}
            }))
            .unwrap();
            Model {
                path: PathBuf::from(file),
                file_size: 1,
                modified_unix: None,
                engine: header.engine(),
                format: "gguf".into(),
                hf: None,
                header: Some(header),
                header_error: None,
                mmproj_candidates: vec![PathBuf::from(r"E:\m\mmproj-F16.gguf")],
                draft_candidates: vec![],
                shards: vec![],
                source: None,
            }
        }

        fn build(tag: &str) -> Build {
            let path = PathBuf::from(format!(r"C:\b\{tag}"));
            Build {
                server_exe: path.join("bin").join("llama-server.exe"),
                path,
                tag: tag.into(),
                version: Some("b10984".into()),
                commit: None,
                version_error: None,
                channel: Channel::Upstream,
                bundled_runtime: false,
                release_tag: None,
                patch: None,
                runner_exe: None,
                git: None,
            }
        }

        fn cfg() -> Config {
            Config::default_for_machine()
        }

        #[cfg(windows)]
        #[test]
        fn a_new_profile_mirrors_the_editors_defaults() {
            let m = model(r"E:\m\IFM\K2-Horizon-7B-GGUF\K2-Horizon-7B-Q4_K_M.gguf", "k2-horizon", 524_288, 36);
            let p = new_for_model(&cfg(), &m, &build("b10984-rocm"), &cards(), &[], &NewProfileOpts::default());
            assert_eq!(p.id, "k2-horizon-7b-q4-k-m");
            assert_eq!(p.server.alias, p.id);
            assert_eq!(p.name, "K2-Horizon-7B-Q4_K_M");
            assert_eq!(p.server.port, FIRST_NEW_PORT);
            assert_eq!(p.engine, Engine::LlamaServer);
            assert_eq!((p.build.path.clone(), p.build.version.as_deref()), (PathBuf::from(r"C:\b\b10984-rocm"), Some("b10984")));
            assert_eq!(p.devices.len(), 1);
            assert_eq!(p.devices[0].key, "pci:a:bus03", "the first discrete card, never the iGPU");
            assert_eq!(p.split_mode, None);
            let r = &p.runtime;
            assert_eq!(r.ctx_total, NEW_PROFILE_CTX);
            assert_eq!((r.kv_type_k.as_str(), r.kv_type_v.as_str(), r.flash_attn.as_str()), ("f16", "f16", "on"));
            assert_eq!((r.batch_logical, r.batch_physical, r.slots), (2048, 512, 1));
            assert!(r.cont_batching && !r.kv_unified);
            assert_eq!(r.n_gpu_layers, 99);
            assert_eq!(p.model.mmproj.as_deref(), Some(Path::new(r"E:\m\mmproj-F16.gguf")), "the first projector beside it");
            assert!(p.model.draft.is_none() && p.speculative.is_none() && p.diffusion.is_none());
            assert!(validate(&p).is_empty(), "{:?}", validate(&p));
            // It saves and loads like any profile.
            let back: Profile = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
            assert_eq!(back.id, p.id);
        }

        #[cfg(windows)]
        #[test]
        fn ids_and_ports_are_unique() {
            let m = model(r"E:\m\a\K2-Horizon-7B-Q4_K_M.gguf", "k2-horizon", 524_288, 36);
            let b = build("b");
            let mut existing: Vec<Profile> = Vec::new();
            let opts = NewProfileOpts { taken_ports: vec![9711], ..Default::default() };
            for _ in 0..3 {
                let p = new_for_model(&cfg(), &m, &b, &cards(), &existing, &opts);
                existing.push(p);
            }
            let ids: Vec<&str> = existing.iter().map(|p| p.id.as_str()).collect();
            assert_eq!(ids, ["k2-horizon-7b-q4-k-m", "k2-horizon-7b-q4-k-m-2", "k2-horizon-7b-q4-k-m-3"]);
            let ports: Vec<u16> = existing.iter().map(|p| p.server.port).collect();
            assert_eq!(ports, [9710, 9712, 9713], "9711 belongs to a running server");
            // Case does not make an id new (Windows file names ignore it).
            let mut shouting = existing[0].clone();
            shouting.id = "K2-HORIZON-7B-Q4-K-M-4".into();
            existing.push(shouting);
            assert_eq!(new_for_model(&cfg(), &m, &b, &cards(), &existing, &opts).id, "k2-horizon-7b-q4-k-m-5");
            // A split model is named by its set, not its first shard.
            let split = model(r"E:\m\g\gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf", "gemma4", 262_144, 30);
            assert_eq!(new_for_model(&cfg(), &split, &b, &cards(), &[], &opts).id, "gemma-4-26b-a4b-it-bf16");
            assert_eq!(slug("...\u{dc}n\u{ef}code \u{2603} name!!"), "n-code-name");
            assert_eq!(slug("---"), "model");
            assert_eq!(slug(&"x".repeat(80)).len(), 48);
        }

        #[test]
        fn context_from_the_estimate_capped() {
            let m = model(r"E:\m\a\small.gguf", "llama", 131_072, 32);
            let b = build("b");
            let ctx = |opts: NewProfileOpts| new_for_model(&cfg(), &m, &b, &cards(), &[], &opts).runtime.ctx_total;
            assert_eq!(ctx(NewProfileOpts { fit_ctx: Some(200_704), ..Default::default() }), 32_768, "capped at 32K");
            assert_eq!(ctx(NewProfileOpts { fit_ctx: Some(20_480), ..Default::default() }), 20_480, "what fits");
            assert_eq!(ctx(NewProfileOpts { ctx: Some(65_536), fit_ctx: Some(20_480), ..Default::default() }), 65_536, "the user's pick");
            assert_eq!(ctx(NewProfileOpts { ctx: Some(1_000_000), ..Default::default() }), 131_072, "never past the trained context");
            assert_eq!(ctx(NewProfileOpts { ctx: Some(10_000), ..Default::default() }), 9_984, "whole 256-token steps");
            let tiny = model(r"E:\m\a\tiny.gguf", "llama", 2048, 12);
            assert_eq!(new_for_model(&cfg(), &tiny, &b, &cards(), &[], &NewProfileOpts::default()).runtime.ctx_total, 2048);
            // More than 98 blocks: -ngl still covers the output layer.
            let deep = model(r"E:\m\a\deep.gguf", "llama", 8192, 120);
            assert_eq!(new_for_model(&cfg(), &deep, &b, &cards(), &[], &NewProfileOpts::default()).runtime.n_gpu_layers, 121);
        }

        #[test]
        fn cards_idle_split_and_integrated() {
            let m = model(r"E:\m\a\m-Q8_0.gguf", "llama", 131_072, 32);
            let b = build("b");
            let busy = NewProfileOpts { busy_cards: vec!["pci:a:bus03".into()], ..Default::default() };
            let p = new_for_model(&cfg(), &m, &b, &cards(), &[], &busy);
            assert_eq!(p.devices[0].key, "pci:a:bus08", "an idle card first");
            let all_busy = NewProfileOpts { busy_cards: vec!["pci:a:bus03".into(), "pci:a:bus08".into()], ..Default::default() };
            assert_eq!(new_for_model(&cfg(), &m, &b, &cards(), &[], &all_busy).devices[0].key, "pci:a:bus03");
            let mut big = cards();
            big[2].total_mib = 49_152;
            let p = new_for_model(&cfg(), &m, &b, &big, &[], &NewProfileOpts { split: true, ..Default::default() });
            let keys: Vec<&str> = p.devices.iter().map(|d| d.key.as_str()).collect();
            assert_eq!(keys, ["pci:a:bus08", "pci:a:bus03"], "the two largest, largest (main) first");
            assert_eq!(p.split_mode, Some(SplitMode::Layer));
            assert!(validate(&p).is_empty(), "{:?}", validate(&p));
            // An APU-only machine: the iGPU only when allowed.
            let apu = vec![card("pci:igpu:bus19", 98_304, true)];
            assert!(new_for_model(&cfg(), &m, &b, &apu, &[], &NewProfileOpts::default()).devices.is_empty());
            let mut allow = cfg();
            allow.allow_integrated = true;
            assert_eq!(new_for_model(&allow, &m, &b, &apu, &[], &NewProfileOpts::default()).devices.len(), 1);
        }

        #[test]
        fn drafts_and_projectors() {
            let m = model(r"E:\m\g\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf", "gemma4", 262_144, 30);
            let b = build("b");
            let opts = NewProfileOpts {
                mmproj: Some(PathBuf::from(r"E:\m\g\mmproj-BF16.gguf")),
                draft: Some(PathBuf::from(r"E:\m\g\mtp-gemma-4-26B-A4B-it.gguf")),
                ..Default::default()
            };
            let p = new_for_model(&cfg(), &m, &b, &cards(), &[], &opts);
            assert_eq!(p.model.mmproj.as_deref(), Some(Path::new(r"E:\m\g\mmproj-BF16.gguf")), "the one asked for");
            let d = p.model.draft.as_ref().unwrap();
            assert!(d.enabled);
            assert_eq!(p.speculative.as_ref().unwrap().mode, "mtp");
            assert_eq!(p.speculative_effective().spec_type(), Some("draft-mtp"));
            let plain = NewProfileOpts { draft: Some(PathBuf::from(r"E:\m\g\small-draft.gguf")), ..Default::default() };
            let p = new_for_model(&cfg(), &m, &b, &cards(), &[], &plain);
            assert_eq!(p.speculative.as_ref().unwrap().mode, "draft");
            assert!(validate(&p).is_empty(), "{:?}", validate(&p));
            let with = |draft: &str, mode: Option<&str>| {
                let opts = NewProfileOpts { draft: Some(PathBuf::from(draft)), draft_mode: mode.map(String::from), ..Default::default() };
                new_for_model(&cfg(), &m, &b, &cards(), &[], &opts)
            };
            // A folder named for MTP holds a plain draft: the file decides.
            let p = with(r"E:\m\someone\qwen3-mtp-drafts\draft-qwen3-0.6b-Q8_0.gguf", None);
            assert_eq!(p.speculative.as_ref().unwrap().mode, "draft");
            // DFlash drafts get their own spec type.
            let p = with(r"E:\m\g\dflash-gpt-oss-20b-Q8_0.gguf", None);
            assert_eq!(p.speculative_effective().spec_type(), Some("draft-dflash"));
            assert!(validate(&p).is_empty(), "{:?}", validate(&p));
            // An EAGLE3 head is not a draft model: kept, switched off.
            for head in [r"E:\m\g\eagle3-gpt-oss-20b-Q8_0.gguf", r"E:\m\g\dspark-qwen3-8b.gguf"] {
                let p = with(head, None);
                assert!(p.speculative.is_none(), "{head}");
                assert!(!p.model.draft.as_ref().unwrap().enabled, "{head}");
                assert_eq!(p.speculative_effective().spec_type(), None, "{head}");
            }
            // The caller's mode wins (an MTP/ folder flattened away).
            let p = with(r"E:\m\g\gemma-4-26B-A4B-it-Q8_0-head.gguf", Some("mtp"));
            assert_eq!(p.speculative.as_ref().unwrap().mode, "mtp");
        }

        #[test]
        fn draft_modes_by_name() {
            for (path, mode) in [
                ("MTP/gemma-4-26B-A4B-it-Q8_0.gguf", Some("mtp")),
                ("mtp-gemma-4-26B-A4B-it.gguf", Some("mtp")),
                (r"E:\m\u\g\MTP\mtp-x.gguf", Some("mtp")),
                ("draft-qwen3-0.6b-Q8_0.gguf", Some("draft")),
                ("mtp-things/draft-qwen3.gguf", Some("draft")),
                ("dflash-qwen3-8b-Q8_0.gguf", Some("dflash")),
                ("Qwen3-8B-DFlash-b16.gguf", Some("dflash")),
                ("eagle3-gpt-oss-20b-Q8_0.gguf", None),
                ("EAGLE3/llama-3.1-8b-eagle3-f16.gguf", None),
                ("dspark-qwen3-8b.gguf", None),
            ] {
                assert_eq!(draft_mode_of(path), mode, "{path}");
            }
            assert_eq!(unsupported_draft_kind("eagle3-x.gguf"), "an EAGLE3 head");
            assert_eq!(unsupported_draft_kind("x/dspark-x.gguf"), "a DSpark head");
        }

        #[test]
        fn a_diffusion_model_gets_the_diffusion_engine() {
            let m = model(r"E:\m\u\diffusiongemma-26B-A4B-it-Q4_K_M.gguf", "diffusion-gemma", 262_144, 30);
            let mut b = build("b11027-mix-unsloth");
            b.channel = Channel::Unsloth;
            let opts = NewProfileOpts {
                busy_cards: vec!["pci:a:bus03".into()],
                draft: Some(PathBuf::from(r"E:\m\u\mtp.gguf")),
                split: true,
                fit_ctx: Some(65_536),
                ..Default::default()
            };
            let p = new_for_model(&cfg(), &m, &b, &cards(), &[], &opts);
            assert_eq!(p.engine, Engine::DiffusionGemma);
            assert_eq!(p.runtime.ctx_total, 0, "MAXTOK auto");
            assert_eq!(p.devices.len(), 1, "one card, never a split");
            assert_eq!(p.devices[0].key, "pci:a:bus08", "an idle card");
            assert!(p.model.mmproj.is_none() && p.model.draft.is_none() && p.speculative.is_none());
            let dg = p.diffusion.as_ref().unwrap();
            assert!(dg.hipblaslt_safeguard && !dg.flash_attn);
            assert_eq!(dg.default_max_tokens, 2048);
            assert!(!validate(&p).iter().any(|f| f.severity == Severity::Error), "{:?}", validate(&p));
            // A runner that pads keys for flash attention gets it on.
            b.patch = Some(crate::discovery::BuildPatch { name: "dgpatch".into(), features: vec!["dg-fa-pad".into()], ..Default::default() });
            assert!(new_for_model(&cfg(), &m, &b, &cards(), &[], &opts).diffusion.unwrap().flash_attn);
        }
    }
}
