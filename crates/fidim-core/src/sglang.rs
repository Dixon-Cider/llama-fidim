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
        Some(Self { weights_bytes, attention_layers, kv_heads, head_dim, label: h.architecture.clone().unwrap_or_default(), state_bytes_per_slot })
    }
    pub fn from_hf(i: &crate::discovery::HfModelInfo) -> Option<Self> {
        Some(Self {
            weights_bytes: i.weights_bytes,
            attention_layers: i.attention_layers.or(i.num_layers)?,
            kv_heads: i.num_kv_heads?,
            head_dim: i.head_dim?,
            label: i.architecture.clone().or_else(|| i.model_type.clone()).unwrap_or_default(),
            state_bytes_per_slot: None,
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
/// static pool under long-context load (measured ~3 GB on a 27B).
pub const RUNTIME_OVERHEAD_BYTES: u64 = 3 * 1024 * 1024 * 1024;

/// How the static pool divides up for a profile on a device.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PoolBudget {
    pub pool_bytes: u64,
    pub weights_bytes: u64,
    pub kv_budget_bytes: u64,
    pub kv_bytes_per_token: u64,
    /// Tokens the pool can hold in KV (SGLang's max_total_num_tokens, roughly).
    pub tokens_in_budget: u64,
    pub ctx: u64,
}

pub fn pool_budget(s: &SgLangCfg, shape: &ModelShape, device: &Device, ctx: u64) -> PoolBudget {
    let pool = estimated_bytes(s, device);
    let per_token = shape.kv_bytes_per_token(&s.kv_cache_dtype).max(1);
    // Calibrated on the 27B GGUF at 0.85: SGLang reported 117,990 tokens for a
    // 27.1 GiB pool with 17.2 GiB of weights, i.e. it keeps ~5 GiB for CUDA
    // graphs, activations, the vision tower and its own reserve, plus ~150 MiB
    // per GatedDeltaNet state slot. This is an approximation, marked "~".
    let per_slot = shape.state_bytes_per_slot.unwrap_or(DEFAULT_STATE_BYTES_PER_SLOT);
    let reserve = (4_500u64 << 20) + s.mamba_slots.unwrap_or(0) as u64 * per_slot;
    let kv_budget = pool.saturating_sub(shape.weights_bytes).saturating_sub(reserve);
    PoolBudget { pool_bytes: pool, weights_bytes: shape.weights_bytes, kv_budget_bytes: kv_budget, kv_bytes_per_token: per_token, tokens_in_budget: kv_budget / per_token, ctx }
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

/// The server command for `p` on `device`.
pub fn compose(cfg: &Config, p: &Profile, device: &Device, runs_dir: &Path) -> Result<LaunchPlan> {
    let host = host_of(cfg)?;
    let s = cfg_of(p);
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
        "1".into(),
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
        args.extend([
            "--speculative-algorithm".into(),
            sp.algorithm.clone(),
            "--speculative-num-steps".into(),
            sp.steps.to_string(),
            "--speculative-eagle-topk".into(),
            sp.topk.to_string(),
            "--speculative-num-draft-tokens".into(),
            sp.draft_tokens.to_string(),
        ]);
        if let Some(m) = &sp.token_map {
            args.extend(["--speculative-token-map".into(), m.to_string_lossy().into_owned()]);
        }
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

    let visibility_env = device.hip_index.to_string();
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
pub fn estimated_bytes(s: &SgLangCfg, device: &Device) -> u64 {
    (device.total_mib as f64 * 1024.0 * 1024.0 * s.mem_fraction) as u64
}

pub fn preflight(
    cfg: &Config,
    p: &Profile,
    device: Option<&Device>,
    platform: &dyn Platform,
    running_aliases: &[String],
) -> Vec<CheckResult> {
    let s = cfg_of(p);
    let mut out = Vec::new();

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
    if let Some(sp) = &s.spec {
        if let Some(m) = &sp.token_map {
            if !m.is_file() {
                missing.push(m.to_string_lossy().into_owned());
            }
        }
    }
    out.push(check(
        "files-exist",
        2,
        "Model, sidecar and token-map files exist",
        if missing.is_empty() { Outcome::Pass } else { Outcome::Block(format!("missing: {}", missing.join(", "))) },
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

    // 6: VRAM fits (free now + what our own previous instance holds)
    if let Some(d) = device {
        let need = estimated_bytes(&s, d);
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
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        out.push(check(
            "vram-fits",
            6,
            "Static pool fits in free VRAM",
            if need > free {
                Outcome::Block(format!(
                    "{}: SGLang will reserve {:.1} GiB ({:.0}% of {:.1}) but only {:.1} GiB is free",
                    d.name, gib(need), s.mem_fraction * 100.0, gib(d.total_mib * 1024 * 1024), gib(free)
                ))
            } else {
                Outcome::Pass
            },
        ));
        // display headroom
        let total = d.total_mib * 1024 * 1024;
        let headroom = total.saturating_sub(need);
        out.push(check(
            "display-headroom",
            10,
            "A display-driving card keeps VRAM headroom",
            match (&d.display, headroom) {
                (Some(_), h) if h < DISPLAY_HEADROOM_BYTES => Outcome::Block(format!(
                    "{} drives a display and mem_fraction {:.2} leaves only {:.2} GiB; the desktop's allocations \
                     then evict the compute process (3 tok/s, KFD evicted_ms climbing). Keep >= {:.1} GiB: \
                     mem_fraction <= {:.2}",
                    d.name, s.mem_fraction, gib(h), gib(DISPLAY_HEADROOM_BYTES),
                    1.0 - DISPLAY_HEADROOM_BYTES as f64 / total as f64
                )),
                (Some(_), h) => Outcome::Note(format!("{} drives a display; {:.2} GiB headroom", d.name, gib(h))),
                (None, _) => Outcome::Pass,
            },
        ));
    }

    // 7: does the model + KV for the context fit inside the pool?
    if let (Some(d), Some(shape)) = (device, ModelShape::of_model(&p.model.path)) {
        let b = pool_budget(&s, &shape, d, p.runtime.ctx_total);
        let gib = |x: u64| x as f64 / (1u64 << 30) as f64;
        out.push(check(
            "kv-budget",
            7,
            "Weights and KV cache fit in the static pool",
            if b.kv_budget_bytes == 0 || b.tokens_in_budget < 1024 {
                Outcome::Block(format!(
                    "{}: weights {:.1} GiB do not fit the {:.1} GiB pool ({:.0}% of the card) with room for KV; raise mem_fraction, use a smaller quant or a bigger card",
                    shape.label, gib(b.weights_bytes), gib(b.pool_bytes), s.mem_fraction * 100.0
                ))
            } else if b.tokens_in_budget < b.ctx {
                Outcome::Warn(format!(
                    "{}: after {:.1} GiB of weights the pool holds ~{} KV tokens ({} B/token, {}), less than the {} context; SGLang will cap max_total_tokens and retract long requests",
                    shape.label, gib(b.weights_bytes), b.tokens_in_budget, b.kv_bytes_per_token, s.kv_cache_dtype, b.ctx
                ))
            } else {
                Outcome::Note(format!(
                    "{}: weights {:.1} GiB + KV {:.1} GiB for {} tokens ({} B/token, {}) in a {:.1} GiB pool; ~{} tokens fit",
                    shape.label, gib(b.weights_bytes), gib(b.kv_bytes_per_token * b.ctx), b.ctx, b.kv_bytes_per_token, s.kv_cache_dtype, gib(b.pool_bytes), b.tokens_in_budget
                ))
            },
        ));
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
    let key = p.devices.first()?.key.as_str();
    crate::devices::resolve_key(key, devices, &[]).ok().and_then(|r| devices.iter().find(|d| d.stable_key == r.device.stable_key))
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
    let device = resolve_device(p, &devices);
    let results = preflight(cfg, p, device, platform, &running_aliases(cfg));
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
    let device = device.ok_or_else(|| Error::Config("device does not resolve".into()))?;
    let plan = compose(cfg, p, device, &cfg.runs_dir)?;

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
        vec![device.stable_key.clone()],
        vec![device.free_mib],
        cold,
        estimated_bytes(&s, device),
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
        let alive = runs.iter().any(|r| r.alive && r.state.profile_id == member.id);
        if m.load_on_startup && !alive {
            launch(cfg, member, platform, false, ready_timeout)?;
            started.push(member.id.clone());
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
        // dd on the R9700 at 0.85 with 10 mamba slots: SGLang gave 117,990 tokens.
        let s: SgLangCfg = serde_json::from_value(serde_json::json!({ "mem_fraction": 0.85, "mamba_slots": 10 })).unwrap();
        let shape = ModelShape { weights_bytes: (17.2 * (1u64 << 30) as f64) as u64, attention_layers: 16, kv_heads: 4, head_dim: 256, label: "qwen35".into(), state_bytes_per_slot: None };
        let b = pool_budget(&s, &shape, &dev(32624), 262144);
        assert!((100_000..170_000).contains(&b.tokens_in_budget), "{}", b.tokens_in_budget);
        assert!(b.tokens_in_budget < b.ctx);
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
