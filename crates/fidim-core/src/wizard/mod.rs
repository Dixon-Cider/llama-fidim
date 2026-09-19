//! The model wizard ("Get a model"): one orchestrator the GUI and the CLI
//! both call.
//!
//! - `inspect` looks at a Hugging Face repo: its files grouped into quant
//!   choices, the model's GGUF header read with HTTP range requests (the
//!   first 64 KiB; the whole header only when the architecture keys sit past
//!   the tokenizer), the VRAM estimate of every choice on this machine's
//!   cards, which installed build can load the model, and, when none can,
//!   the plan for getting one (`compat::plan_build`). A repo with no GGUF is
//!   explained, with the GGUF quantizations of it (or of its base model)
//!   the Hub knows.
//! - `plan` turns a choice into exact steps: an install or a source build
//!   (a pull request's or a fork's code needs the user's consent), each
//!   download with its destination, the disk and path findings, and the
//!   profile to create. It reads the chosen file's whole header, so the
//!   build is checked against the model's tensor types too.
//! - `run` executes a plan: build or install, download (resumable, size and
//!   SHA-256 checked, a `fidim-source.json` written beside the files), then
//!   the profile. It never launches a model: the GPUs are shared, and
//!   loading stays an explicit action of the user's.
//!
//! Every side effect goes through `Env`, so the orchestration is tested
//! with fakes and no network; `LiveEnv` is the real machine.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::catalog::{self, Catalog, Choice, ChoiceFit, Fit, FitOptions};
use crate::compat::{self, BuildPlan, GitRef, ModelNeeds, PlanAction, PlanInputs, PlanStep, Support};
use crate::config::Config;
use crate::devices::Device;
use crate::discovery::{self, Build, Channel, Model, ModelSource};
use crate::fetch;
use crate::gguf::{GgufHeader, ReadMode, DIFFUSION_ARCHES};
use crate::hub::{self, Gated, RepoFile, RepoHit, RepoInfo, SearchQuery};
use crate::profile::{self, Engine, NewProfileOpts, Profile};
use crate::update::{self, BuildProgress, InstallReport, SourceRef};
use crate::{Error, Result};

#[cfg(test)]
mod tests;

const GIB: f64 = (1u64 << 30) as f64;
/// Free space a source build wants where FIDIM keeps its llama.cpp clone
/// (`update::source_checkout_dir`, under the config folder, usually on C:):
/// the worktree and build tree at the peak (measured 0.5 GB for the K2
/// Horizon fork, one GPU target), with room to spare.
const SOURCE_SCRATCH_BYTES: u64 = 1 << 30;
/// Below this much free there, a source build is not started: the measured
/// peak, which would leave the drive full.
const SOURCE_SCRATCH_MIN: u64 = 600 << 20;
/// What a source build installs (0.1 GB), staged beside the install first.
const SOURCE_INSTALL_BYTES: u64 = 256 << 20;
/// A prebuilt install: the zips (up to ~0.5 GB) and what they unpack to
/// (~0.9-1.1 GB).
const PREBUILT_BYTES: u64 = 3 << 29;

// ------------------------------------------------------------------ types ----

/// How much a note matters: an `Error` blocks the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Error,
    Warning,
    Info,
}

/// One thing the user should know, with a stable code for tests and UIs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Note {
    pub level: Level,
    pub code: String,
    pub message: String,
}

impl Note {
    fn new(level: Level, code: &str, message: impl Into<String>) -> Self {
        Note { level, code: code.into(), message: message.into() }
    }
}

impl From<profile::Finding> for Note {
    fn from(f: profile::Finding) -> Self {
        let level = match f.severity {
            profile::Severity::Error => Level::Error,
            profile::Severity::Warning => Level::Warning,
        };
        Note { level, code: f.code.into(), message: f.message }
    }
}

/// What a repo holds, as far as llama.cpp is concerned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "format", rename_all = "snake_case")]
pub enum RepoKind {
    /// GGUF files to choose from.
    Gguf,
    /// Weights for transformers (safetensors, PyTorch `.bin`), no GGUF.
    Safetensors,
    /// A LoRA / PEFT adapter, not a model.
    Adapter,
    /// Quantized for another runtime: FP8, NVFP4, AWQ, GPTQ, EXL2, MLX...
    OtherFormat(String),
    /// No model files at all.
    Empty,
}

/// An installed build and whether it can load the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildVerdict {
    pub path: PathBuf,
    /// The build's name in tables: its git label, else its folder name.
    pub name: String,
    pub channel: Channel,
    pub version: Option<String>,
    /// `llama-server --version` failed: the build does not run.
    pub broken: bool,
    pub support: Support,
}

/// A git ref a build could come from (a fork the model card links, an
/// upstream pull request), with what a person needs to judge it before
/// building its code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildSource {
    /// Canonical repository, after GitHub's renames and transfers.
    pub owner: String,
    pub repo: String,
    /// The link as the card wrote it, when GitHub now sends it elsewhere.
    pub linked_as: Option<String>,
    pub git_ref: String,
    pub sha: String,
    pub url: String,
    pub is_upstream: bool,
    pub is_fork_of_upstream: bool,
    pub ahead_by: Option<u32>,
    pub behind_by: Option<u32>,
    /// Newest commit subjects, newest last.
    pub subjects: Vec<String>,
    pub pr: Option<compat::PrInfo>,
    pub support: Option<Support>,
}

/// The candidate sources a build plan weighed.
fn sources_of(inputs: &PlanInputs) -> Vec<BuildSource> {
    let mut out: Vec<BuildSource> = inputs
        .card_refs
        .iter()
        .filter_map(|c| {
            let r = c.resolved.as_ref()?;
            Some(BuildSource {
                owner: r.owner.clone(),
                repo: r.repo.clone(),
                linked_as: r.redirected.then(|| format!("{}/{}", r.requested.owner, r.requested.repo)),
                git_ref: r.git_ref.clone(),
                sha: r.sha.clone(),
                url: r.html_url(),
                is_upstream: r.is_upstream,
                is_fork_of_upstream: r.is_fork_of_ggml,
                ahead_by: r.ahead_by,
                behind_by: r.behind_by,
                subjects: r.head_subjects.clone(),
                pr: r.pr.clone(),
                support: c.support.clone(),
            })
        })
        .collect();
    for p in &inputs.prs {
        if out.iter().any(|s| s.sha.eq_ignore_ascii_case(&p.pr.head_sha)) {
            continue;
        }
        out.push(BuildSource {
            owner: compat::UPSTREAM_OWNER.into(),
            repo: compat::UPSTREAM_REPO.into(),
            linked_as: None,
            git_ref: format!("pull/{}/head", p.pr.number),
            sha: p.pr.head_sha.to_ascii_lowercase(),
            url: p.pr.html_url.clone(),
            is_upstream: true,
            is_fork_of_upstream: false,
            ahead_by: None,
            behind_by: None,
            subjects: Vec::new(),
            pr: Some(p.pr.clone()),
            support: Some(p.support.clone()),
        });
    }
    out
}

/// A model folder and the room left on its drive.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RootInfo {
    pub path: PathBuf,
    pub exists: bool,
    pub free_bytes: Option<u64>,
}

/// Everything `inspect` found out about a repo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepoView {
    /// What the user typed.
    pub input: String,
    pub repo: String,
    /// The commit everything here describes; downloads are pinned to it.
    pub sha: String,
    /// The revision asked for (a branch, tag or commit); None = the default branch.
    pub rev: Option<String>,
    pub kind: RepoKind,
    pub info: RepoInfo,
    pub catalog: Catalog,
    /// Explanations: a repo llama.cpp cannot run, gating, lookups that failed.
    pub notes: Vec<Note>,
    /// GGUF quantizations of this repo (or of its base model) on the Hub,
    /// for a repo with no GGUF.
    pub derivatives: Vec<RepoHit>,
    pub derivatives_of: Option<String>,
    /// The header of `header_of`'s first file (the architecture keys are the
    /// same for every quant of a model); its `file_size` is that choice's.
    pub header: Option<GgufHeader>,
    pub header_of: Option<String>,
    pub header_error: Option<String>,
    pub needs: Option<ModelNeeds>,
    /// The choice a pasted file URL named.
    pub preselect: Option<String>,
    /// Each draft's speculative mode (`mtp`, `draft`, `dflash`) by its
    /// repo path; None for a head profiles cannot run (EAGLE3, DSpark).
    #[serde(default)]
    pub draft_modes: BTreeMap<String, Option<String>>,
    /// This machine's GPUs, as the estimates saw them.
    pub devices: Vec<Device>,
    pub fits: Vec<ChoiceFit>,
    pub recommended: Option<String>,
    pub builds: Vec<BuildVerdict>,
    /// The best installed build that can load the model.
    pub usable_build: Option<PathBuf>,
    /// How to get a build, when no installed one can load the model.
    pub build_plan: Option<BuildPlan>,
    /// The forks and pull requests that plan weighed.
    pub build_sources: Vec<BuildSource>,
    pub model_roots: Vec<RootInfo>,
}

/// Which build to use.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum BuildChoice {
    /// An installed build that loads the model, else the plan's recommendation.
    #[default]
    Auto,
    /// No install or build step; the profile goes on the best installed build.
    Skip,
    /// This installed build.
    Installed(PathBuf),
    /// The build plan's recommendation (0) or its n-th alternative.
    Plan(usize),
}

/// What the user picked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRequest {
    /// A choice label from the view's catalog; None = the recommendation.
    #[serde(default)]
    pub choice: Option<String>,
    /// Repo path of a projector to download too.
    #[serde(default)]
    pub mmproj: Option<String>,
    /// Repo path of a draft / MTP head to download too.
    #[serde(default)]
    pub draft: Option<String>,
    /// Model folder to save into; None = the first model folder.
    #[serde(default)]
    pub dest_root: Option<PathBuf>,
    #[serde(default)]
    pub build: BuildChoice,
    /// Create a profile at the end.
    #[serde(default = "yes")]
    pub profile: bool,
    /// The profile's context; None = the estimate's, capped at 32768.
    #[serde(default)]
    pub ctx: Option<u64>,
}

fn yes() -> bool {
    true
}

impl Default for PlanRequest {
    fn default() -> Self {
        PlanRequest { choice: None, mmproj: None, draft: None, dest_root: None, build: BuildChoice::Auto, profile: true, ctx: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileRole {
    Model,
    Mmproj,
    Draft,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StepAction {
    /// Install a prebuilt or compile from source (`plan.action` says which).
    Build { plan: PlanStep },
    Download {
        role: FileRole,
        file: RepoFile,
        dest: PathBuf,
        /// Bytes already on disk: the whole file, or a `.part` to resume.
        have: u64,
        /// The file is already in place: it is verified, not downloaded.
        present: bool,
    },
    Profile { id: String, path: PathBuf },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub title: String,
    pub detail: String,
    /// Runs code nobody reviewed for this machine: needs the user's consent.
    pub needs_consent: bool,
    pub action: StepAction,
}

/// The build the new profile will use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedBuild {
    pub path: PathBuf,
    pub name: String,
    /// Already installed (else a step of the plan installs or builds it).
    pub installed: bool,
    pub support: Option<Support>,
}

/// A toolchain doctor finding, for a plan with a source build.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolFinding {
    pub id: String,
    pub title: String,
    /// `pass`, `note`, `warn` or `block`.
    pub outcome: String,
    pub message: String,
    pub fix: Option<String>,
}

impl From<&crate::toolchain::Finding> for ToolFinding {
    fn from(f: &crate::toolchain::Finding) -> Self {
        use crate::preflight::Outcome;
        let (outcome, message) = match &f.outcome {
            Outcome::Pass => ("pass", f.detail.clone()),
            Outcome::Note(m) => ("note", m.clone()),
            Outcome::Warn(m) => ("warn", m.clone()),
            Outcome::Block(m) => ("block", m.clone()),
        };
        ToolFinding { id: f.id.into(), title: f.title.into(), outcome: outcome.into(), message, fix: f.fix.clone() }
    }
}

/// Exactly what `run` will do.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WizardPlan {
    pub repo: String,
    pub sha: String,
    pub choice: Choice,
    pub mmproj: Option<RepoFile>,
    pub draft: Option<RepoFile>,
    pub dest_root: PathBuf,
    /// `<root>\<owner>\<repo>`: where every file goes.
    pub dest_dir: PathBuf,
    pub steps: Vec<Step>,
    pub notes: Vec<Note>,
    /// An `Error` note: `run` refuses the plan.
    pub blocked: bool,
    pub needs_consent: bool,
    /// What the user consents to, in one sentence.
    pub consent: Option<String>,
    /// Bytes the downloads still have to fetch, and the files' total.
    pub download_bytes: u64,
    pub total_bytes: u64,
    pub engine: Engine,
    pub gated: Gated,
    pub needs: Option<ModelNeeds>,
    /// The chosen file's header: whole (tensor types known) when it could be read.
    pub header: Option<GgufHeader>,
    /// The chosen file's estimate, with the projector and draft counted.
    pub fit: Option<ChoiceFit>,
    pub build: Option<PlannedBuild>,
    /// The build plan the build step came from, for its alternatives.
    pub build_plan: Option<BuildPlan>,
    /// The forks and pull requests it weighed (owner, commit, distance
    /// from upstream, newest commit subjects).
    pub build_sources: Vec<BuildSource>,
    pub build_choice: BuildChoice,
    pub toolchain: Vec<ToolFinding>,
    /// The profile `run` will create (its id and port are chosen again
    /// then, against the profiles that exist by that time).
    pub profile: Option<Profile>,
    pub profile_opts: NewProfileOpts,
    pub devices: Vec<Device>,
}

impl WizardPlan {
    /// The files `run` downloads, as (dest, size).
    pub fn download_dests(&self) -> Vec<(PathBuf, u64)> {
        self.steps
            .iter()
            .filter_map(|s| match &s.action {
                StepAction::Download { dest, file, .. } => Some((dest.clone(), file.size)),
                _ => None,
            })
            .collect()
    }

    /// A step runs code nobody reviewed for this machine (see
    /// `step_needs_consent`: judged by what the steps do, not by the
    /// plan's `needs_consent`).
    pub fn requires_consent(&self) -> bool {
        self.steps.iter().any(step_needs_consent)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepStatus {
    Pending,
    Running,
    Done,
    Failed,
    Skipped,
}

/// One progress report from `run`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WizardEvent {
    /// Index into the plan's steps.
    pub step: usize,
    pub status: StepStatus,
    /// What the step is doing: `downloading`, `hashing`, or a build's
    /// `doctor`, `fetch`, `configure`, `build`, `install`, `verify`...
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub file: Option<String>,
    #[serde(default)]
    pub done: Option<u64>,
    #[serde(default)]
    pub total: Option<u64>,
    /// Bytes per second while downloading.
    #[serde(default)]
    pub bps: Option<f64>,
    /// A line of build or install output, or the error of a failed step.
    #[serde(default)]
    pub line: Option<String>,
}

impl WizardEvent {
    fn status(step: usize, status: StepStatus) -> Self {
        WizardEvent { step, status, stage: None, file: None, done: None, total: None, bps: None, line: None }
    }
}

/// The latest state of each step of a running job, rebuilt from its
/// events, so a view that comes back can draw it again.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct JobProgress {
    pub steps: Vec<StepProgress>,
    /// The last lines of build and install output.
    pub log: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepProgress {
    pub status: StepStatus,
    pub stage: Option<String>,
    pub file: Option<String>,
    pub done: Option<u64>,
    pub total: Option<u64>,
    pub bps: Option<f64>,
    /// The error of a failed step.
    pub error: Option<String>,
}

/// Build and install output kept for a view that comes back.
const LOG_KEEP: usize = 400;

impl JobProgress {
    pub fn new(plan: &WizardPlan) -> Self {
        let pending = StepProgress {
            status: StepStatus::Pending,
            stage: None,
            file: None,
            done: None,
            total: None,
            bps: None,
            error: None,
        };
        JobProgress { steps: vec![pending; plan.steps.len()], log: Vec::new() }
    }

    pub fn apply(&mut self, ev: &WizardEvent) {
        if let Some(line) = &ev.line {
            if ev.status != StepStatus::Failed {
                self.log.push(line.clone());
                let over = self.log.len().saturating_sub(LOG_KEEP);
                self.log.drain(..over);
            }
        }
        let Some(s) = self.steps.get_mut(ev.step) else { return };
        s.status = ev.status;
        if ev.stage.is_some() {
            s.stage = ev.stage.clone();
        }
        if ev.file.is_some() {
            s.file = ev.file.clone();
        }
        if ev.done.is_some() || ev.total.is_some() {
            s.done = ev.done;
            s.total = ev.total;
        }
        s.bps = ev.bps.or(if ev.status == StepStatus::Running { s.bps } else { None });
        if ev.status == StepStatus::Failed {
            s.error = ev.line.clone();
        }
    }
}

/// What `run` did.
#[derive(Debug, Clone, Serialize)]
pub struct WizardResult {
    pub repo: String,
    pub sha: String,
    /// The file llama.cpp is given (the first shard of a split model).
    pub model_path: PathBuf,
    pub files: Vec<PathBuf>,
    pub build: Option<InstallReport>,
    pub build_path: Option<PathBuf>,
    pub profile: Option<Profile>,
    pub profile_path: Option<PathBuf>,
    pub warnings: Vec<String>,
}

/// A search result, with whether an installed build knows its architecture
/// (None: no architecture on the Hub's summary, or no build installed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    #[serde(flatten)]
    pub hit: RepoHit,
    pub arch_known: Option<bool>,
}

/// `fidim models needs`: what a model needs from a build and who has it.
#[derive(Debug, Clone, Serialize)]
pub struct NeedsReport {
    pub target: String,
    pub repo: Option<String>,
    pub needs: ModelNeeds,
    /// The largest tensor type in the file, when its tensor table was read
    /// (`needs.max_type_id` has it only when a build might lack it).
    pub max_tensor_type: Option<u32>,
    pub builds: Vec<BuildVerdict>,
    pub usable_build: Option<PathBuf>,
    pub build_plan: Option<BuildPlan>,
    pub notes: Vec<Note>,
}

// -------------------------------------------------------------------- env ----

/// The wizard's side effects. `LiveEnv` is the machine and the network.
pub trait Env {
    fn search(&self, q: &SearchQuery) -> Result<Vec<RepoHit>>;
    fn model_info(&self, repo: &str, rev: Option<&str>) -> Result<RepoInfo>;
    fn list_files(&self, repo: &str, sha: &str) -> Result<Vec<RepoFile>>;
    fn readme(&self, repo: &str, rev: &str) -> Result<Option<String>>;
    /// The header of a model published as `files` (see `hub::remote_model_header`).
    fn header(&self, repo: &str, sha: &str, files: &[RepoFile], mode: ReadMode) -> Result<GgufHeader>;
    fn derivatives(&self, base_repo: &str) -> Result<Vec<RepoHit>>;
    fn has_hf_token(&self) -> bool;
    fn auth_check(&self, repo: &str) -> Result<()>;

    fn builds(&self) -> Vec<Build>;
    fn devices(&self) -> Result<Vec<Device>>;
    /// GPU target(s) of this machine's cards, e.g. `gfx1201`.
    fn gpu_targets(&self) -> Option<String>;
    /// Stable keys of cards a running server holds.
    fn busy_cards(&self) -> Vec<String>;
    /// Ports running servers listen on.
    fn taken_ports(&self) -> Vec<u16>;
    fn profiles(&self) -> Vec<Profile>;

    fn probe(&self, build: &Build, needs: &ModelNeeds) -> Support;
    fn gather(&self, needs: &ModelNeeds, installed: &[Build], card_refs: &[GitRef], gfx: &str, hint: Option<&str>) -> PlanInputs;
    fn doctor(&self, gfx: &str) -> Vec<crate::toolchain::Finding>;

    fn download(
        &self,
        repo: &str,
        sha: &str,
        file: &RepoFile,
        dest: &Path,
        progress: &mut dyn FnMut(&fetch::Progress),
        cancel: &AtomicBool,
    ) -> Result<u64>;
    /// Install a prebuilt; `cancel` stops it while its zips download.
    fn install_upstream(&self, tag: &str, progress: &mut dyn FnMut(String), cancel: &AtomicBool) -> Result<InstallReport>;
    fn install_unsloth(
        &self,
        tag: &str,
        gfx: &str,
        progress: &mut dyn FnMut(String),
        cancel: &AtomicBool,
    ) -> Result<InstallReport>;
    fn build_source(
        &self,
        src: &SourceRef,
        gpu_targets: &str,
        progress: &mut dyn FnMut(BuildProgress),
        cancel: &AtomicBool,
    ) -> Result<InstallReport>;
    /// Save a new profile; an existing file of that id is never overwritten.
    fn save_profile(&self, p: &Profile) -> Result<PathBuf>;
    /// Bytes free on the drive of `path` (see `disk::free_bytes`); None
    /// when it cannot be told.
    fn free_bytes(&self, path: &Path) -> Option<u64>;
}

/// The real machine: the Hub, GitHub, the installed builds, the GPUs.
pub struct LiveEnv {
    pub cfg: Config,
    /// Cards the caller already enumerated (the GUI keeps them for a few
    /// seconds); None = enumerate them when asked.
    pub devices: Option<Vec<Device>>,
    /// GPU target(s) for source builds given by the user (`--gfx`); None =
    /// what hipInfo or the card names say.
    pub gfx: Option<String>,
}

impl LiveEnv {
    pub fn new(cfg: Config) -> Self {
        LiveEnv { cfg, devices: None, gfx: None }
    }

    pub fn with_devices(cfg: Config, devices: Option<Vec<Device>>) -> Self {
        LiveEnv { cfg, devices, gfx: None }
    }
}

/// Whole headers already read in this process, by `repo@sha/first file`:
/// the GUI plans again on every change of a choice.
fn full_headers() -> &'static Mutex<HashMap<String, GgufHeader>> {
    static CACHE: OnceLock<Mutex<HashMap<String, GgufHeader>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// When the doctor ran for a GPU target, and what it found.
type DoctorAnswers = HashMap<String, (Instant, Vec<crate::toolchain::Finding>)>;

/// Toolchain doctor answers by GPU target: a test compile takes seconds.
fn doctor_cache() -> &'static Mutex<DoctorAnswers> {
    static CACHE: OnceLock<Mutex<DoctorAnswers>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}
const DOCTOR_TTL: Duration = Duration::from_secs(600);

/// The installed builds, as last scanned: every build's `--version` runs in
/// a scan (about 0.3 s each), and a search, an inspect and a plan each want
/// the list. Kept a minute, for the same roots; `forget_builds` drops it
/// when a build is added.
type BuildScan = (Instant, Vec<PathBuf>, Option<PathBuf>, Vec<Build>);
fn builds_cache() -> &'static Mutex<Option<BuildScan>> {
    static CACHE: OnceLock<Mutex<Option<BuildScan>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}
const BUILDS_TTL: Duration = Duration::from_secs(60);

/// Scan the builds afresh next time (one was installed or removed).
pub fn forget_builds() {
    *builds_cache().lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// The build `--list-devices` runs from: the newest upstream release that
/// runs, else any that runs, else the first (as the CLI and the GUI pick it).
pub fn enumeration_build(builds: &[Build]) -> Option<&Build> {
    let newest = |upstream_only: bool| {
        builds
            .iter()
            .filter(|b| !upstream_only || b.channel == Channel::Upstream)
            .filter(|b| b.version.is_some())
            .max_by_key(|b| b.version.as_deref().and_then(update::version_number).unwrap_or(0))
    };
    newest(true).or_else(|| newest(false)).or(builds.first())
}

/// The GPU target(s) of this machine's discrete cards: ROCm's hipInfo when
/// the configured runtime has it, else a guess from the card names.
pub fn machine_gpu_targets(cfg: &Config) -> Option<String> {
    let hipinfo = cfg
        .rocm_bin
        .as_ref()
        .map(|b| b.join("hipInfo.exe"))
        .filter(|p| p.is_file())
        .and_then(|exe| crate::launch::run_capture(&exe, &[], cfg.rocm_bin.as_deref()).ok());
    #[cfg(windows)]
    let names: Vec<String> = {
        use crate::platform::Platform;
        crate::platform::WindowsPlatform.video_adapters().unwrap_or_default().into_iter().map(|a| a.name).collect()
    };
    #[cfg(not(windows))]
    let names: Vec<String> = Vec::new();
    update::default_gpu_targets(hipinfo.as_deref(), &names)
}

impl Env for LiveEnv {
    fn search(&self, q: &SearchQuery) -> Result<Vec<RepoHit>> {
        hub::search(&self.cfg, q)
    }
    fn model_info(&self, repo: &str, rev: Option<&str>) -> Result<RepoInfo> {
        hub::model_info(&self.cfg, repo, rev)
    }
    fn list_files(&self, repo: &str, sha: &str) -> Result<Vec<RepoFile>> {
        hub::list_files(&self.cfg, repo, sha)
    }
    fn readme(&self, repo: &str, rev: &str) -> Result<Option<String>> {
        hub::readme(&self.cfg, repo, rev)
    }
    fn header(&self, repo: &str, sha: &str, files: &[RepoFile], mode: ReadMode) -> Result<GgufHeader> {
        let key = format!("{repo}@{sha}/{}", files.first().map(|f| f.path.as_str()).unwrap_or(""));
        if mode == ReadMode::Full {
            if let Some(h) = full_headers().lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
                return Ok(h.clone());
            }
        }
        let h = hub::remote_model_header(&self.cfg, repo, sha, files, mode)?;
        if mode == ReadMode::Full {
            full_headers().lock().unwrap_or_else(|e| e.into_inner()).insert(key, h.clone());
        }
        Ok(h)
    }
    fn derivatives(&self, base_repo: &str) -> Result<Vec<RepoHit>> {
        hub::derivatives(&self.cfg, base_repo)
    }
    fn has_hf_token(&self) -> bool {
        hub::token(&self.cfg).is_some()
    }
    fn auth_check(&self, repo: &str) -> Result<()> {
        hub::auth_check(&self.cfg, repo)
    }
    fn builds(&self) -> Vec<Build> {
        let roots = self.cfg.build_roots_effective();
        let rocm = self.cfg.rocm_bin.clone();
        if let Some((at, r, rb, builds)) = builds_cache().lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
            if at.elapsed() < BUILDS_TTL && *r == roots && *rb == rocm {
                return builds.clone();
            }
        }
        let builds = discovery::scan_builds(&roots, rocm.as_deref());
        *builds_cache().lock().unwrap_or_else(|e| e.into_inner()) = Some((Instant::now(), roots, rocm, builds.clone()));
        builds
    }
    fn devices(&self) -> Result<Vec<Device>> {
        if let Some(d) = &self.devices {
            return Ok(d.clone());
        }
        #[cfg(windows)]
        {
            let builds = self.builds();
            let build = enumeration_build(&builds)
                .ok_or_else(|| Error::Config("no llama.cpp build is installed to list the GPUs with".into()))?;
            crate::launch::enumerate_devices(&self.cfg, &build.server_exe, &crate::platform::WindowsPlatform)
        }
        #[cfg(not(windows))]
        Ok(Vec::new())
    }
    fn gpu_targets(&self) -> Option<String> {
        self.gfx.clone().or_else(|| machine_gpu_targets(&self.cfg))
    }
    fn busy_cards(&self) -> Vec<String> {
        crate::supervise::reattach(&self.cfg.runs_dir)
            .into_iter()
            .filter(|r| r.alive)
            .flat_map(|r| r.state.device_keys)
            .collect()
    }
    fn taken_ports(&self) -> Vec<u16> {
        crate::supervise::reattach(&self.cfg.runs_dir).into_iter().filter(|r| r.alive).map(|r| r.state.port).collect()
    }
    fn profiles(&self) -> Vec<Profile> {
        Profile::load_all(&self.cfg.profile_dir).unwrap_or_default()
    }
    fn probe(&self, build: &Build, needs: &ModelNeeds) -> Support {
        compat::probe_build(build, needs)
    }
    fn gather(&self, needs: &ModelNeeds, installed: &[Build], card_refs: &[GitRef], gfx: &str, hint: Option<&str>) -> PlanInputs {
        compat::gather_plan_inputs(&self.cfg, needs, installed, card_refs, gfx, hint, None)
    }
    fn doctor(&self, gfx: &str) -> Vec<crate::toolchain::Finding> {
        if let Some((at, f)) = doctor_cache().lock().unwrap_or_else(|e| e.into_inner()).get(gfx) {
            if at.elapsed() < DOCTOR_TTL {
                return f.clone();
            }
        }
        let f = crate::toolchain::doctor(&self.cfg, gfx);
        doctor_cache().lock().unwrap_or_else(|e| e.into_inner()).insert(gfx.to_string(), (Instant::now(), f.clone()));
        f
    }
    fn download(
        &self,
        repo: &str,
        sha: &str,
        file: &RepoFile,
        dest: &Path,
        progress: &mut dyn FnMut(&fetch::Progress),
        cancel: &AtomicBool,
    ) -> Result<u64> {
        let mut req = fetch::FetchReq::new(hub::resolve_url(repo, sha, &file.path));
        req.token = hub::token(&self.cfg);
        req.expected_size = Some(file.size).filter(|s| *s > 0);
        req.expected_sha256 = file.sha256.clone();
        fetch::download_resumable(&req, dest, progress, cancel)
    }
    fn install_upstream(&self, tag: &str, progress: &mut dyn FnMut(String), cancel: &AtomicBool) -> Result<InstallReport> {
        let release = update::release_by_tag(tag)?;
        let r = update::install_prebuilt_cancellable(&self.cfg, &release, progress, cancel);
        forget_builds();
        r
    }
    fn install_unsloth(
        &self,
        tag: &str,
        gfx: &str,
        progress: &mut dyn FnMut(String),
        cancel: &AtomicBool,
    ) -> Result<InstallReport> {
        let release = update::unsloth_release_by_tag(tag)?;
        let r = update::install_unsloth_cancellable(&self.cfg, &release, gfx, progress, cancel);
        forget_builds();
        r
    }
    fn build_source(
        &self,
        src: &SourceRef,
        gpu_targets: &str,
        progress: &mut dyn FnMut(BuildProgress),
        cancel: &AtomicBool,
    ) -> Result<InstallReport> {
        let r = update::build_from_ref(&self.cfg, src, gpu_targets, progress, cancel);
        forget_builds();
        r
    }
    fn save_profile(&self, p: &Profile) -> Result<PathBuf> {
        save_new_profile(&self.cfg.profile_dir, p)
    }
    fn free_bytes(&self, path: &Path) -> Option<u64> {
        crate::disk::free_bytes(path)
    }
}

/// Write `p` as `<dir>\<id>.json`, refusing to replace a profile that
/// already has that id.
pub fn save_new_profile(dir: &Path, p: &Profile) -> Result<PathBuf> {
    let path = dir.join(format!("{}.json", p.id));
    if path.exists() {
        return Err(Error::Config(format!("a profile named {} already exists ({})", p.id, path.display())));
    }
    p.save(&path)?;
    Ok(path)
}

// ---------------------------------------------------------------- helpers ----

/// The engine an architecture needs.
fn engine_of_arch(arch: &str) -> Engine {
    if DIFFUSION_ARCHES.contains(&arch) {
        Engine::DiffusionGemma
    } else {
        Engine::LlamaServer
    }
}

/// Tensor types every build the wizard deals in knows: GGML_TYPE_COUNT at
/// upstream b4400 (December 2024), through BF16, every K and IQ quant and
/// the ternary types. A source build for the ROCm 7 HIP SDK is b5872 or
/// newer. Only a file with a newer type (MXFP4 is 39) needs the check.
pub const TYPES_EVERY_BUILD_KNOWS: u32 = 39;

/// What a model needs from a build, from its header: the architecture, the
/// pre-tokenizer of a BPE (`gpt2`) vocabulary (llama.cpp ignores it for any
/// other), and the largest tensor type when the tensor table was read and it
/// is one a build might lack (see `TYPES_EVERY_BUILD_KNOWS`).
pub fn needs_of(h: &GgufHeader) -> Option<ModelNeeds> {
    let arch = h.architecture.clone().filter(|a| !a.trim().is_empty())?;
    let tok_model = h
        .tokenizer_model
        .clone()
        .or_else(|| h.metadata.get("tokenizer.ggml.model").and_then(|v| v.as_str()).map(str::to_string));
    let pre = h
        .tokenizer_pre
        .clone()
        .or_else(|| h.metadata.get("tokenizer.ggml.pre").and_then(|v| v.as_str()).map(str::to_string));
    let pre = if tok_model.as_deref() == Some("gpt2") { pre } else { None };
    let types = h.max_tensor_type().filter(|t| *t >= TYPES_EVERY_BUILD_KNOWS);
    Some(ModelNeeds::new(arch, pre, types, h.engine()))
}

/// A header as a view or plan carries it: the parsed fields only. The raw
/// metadata (a chat template runs to tens of KB) is for parsing and never
/// read after it, and a float in it that JSON cannot hold (NaN) would make
/// the view fail its trip back from the web view.
fn slim(mut h: GgufHeader) -> GgufHeader {
    h.metadata.clear();
    h.tensors.clear();
    h
}

fn build_name(b: &Build) -> String {
    b.git.as_ref().map(|g| g.display()).unwrap_or_else(|| b.tag.clone())
}

/// Installed builds best first: upstream (newest first), git builds,
/// Unsloth, anything else; a build that does not run last.
fn build_rank(b: &Build) -> (bool, u8, std::cmp::Reverse<u32>) {
    let ch = match b.channel {
        Channel::Upstream => 0,
        Channel::Git => 1,
        Channel::Unsloth => 2,
        Channel::Other => 3,
    };
    let n = b.version.as_deref().and_then(update::version_number).unwrap_or(0);
    (b.version_error.is_some(), ch, std::cmp::Reverse(n))
}

fn verdicts(env: &dyn Env, needs: &ModelNeeds, builds: &[Build]) -> Vec<BuildVerdict> {
    let mut sorted: Vec<&Build> = builds.iter().collect();
    sorted.sort_by_key(|b| build_rank(b));
    sorted
        .into_iter()
        .map(|b| BuildVerdict {
            path: b.path.clone(),
            name: build_name(b),
            channel: b.channel,
            version: b.version.clone(),
            broken: b.version_error.is_some(),
            support: env.probe(b, needs),
        })
        .collect()
}

/// The best build that certainly loads the model (verdicts come sorted).
fn best_usable(v: &[BuildVerdict]) -> Option<&BuildVerdict> {
    v.iter().find(|b| !b.broken && b.support.is_yes())
}

/// `p` is `root` or inside it, compared as Windows does: without case,
/// either separator.
fn under(p: &Path, root: &Path) -> bool {
    let norm = |x: &Path| x.to_string_lossy().replace('/', "\\").trim_end_matches('\\').to_lowercase();
    let (p, root) = (norm(p), norm(root));
    p == root || p.starts_with(&format!("{root}\\"))
}

fn same_path(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| p.to_string_lossy().replace('/', "\\").trim_end_matches('\\').to_lowercase();
    norm(a) == norm(b)
}

/// A name to search upstream pull requests by: the base model's (or this
/// repo's) name without its size and format tokens. `IFM/K2-Horizon-MoVA-36B-A4B`
/// -> `K2 Horizon MoVA`.
pub fn model_hint(repo: &str, info: Option<&RepoInfo>) -> Option<String> {
    let base = info
        .and_then(|i| i.base_models.iter().find(|(rel, _)| rel == "quantized" || rel.is_empty()).map(|(_, r)| r.clone()))
        .unwrap_or_else(|| repo.to_string());
    let name = base.rsplit('/').next().unwrap_or(&base);
    let mut words: Vec<&str> = Vec::new();
    for tok in name.split(['-', '_', ' ', '.']) {
        let t = tok.to_ascii_lowercase();
        let size = t.len() > 1
            && (t.ends_with('b') || t.ends_with('m'))
            && t.chars().any(|c| c.is_ascii_digit())
            && t[..t.len() - 1].chars().all(|c| c.is_ascii_digit() || c == '.' || c == 'x' || c == 'a');
        if size || ["gguf", "it", "instruct", "chat", "base", "fp8", "awq", "gptq"].contains(&t.as_str()) {
            break;
        }
        if !tok.is_empty() {
            words.push(tok);
        }
    }
    let hint = words.join(" ");
    (hint.len() >= 3).then_some(hint)
}

/// A format for other runtimes, from the repo's tags and the words of its name.
fn other_format(info: &RepoInfo) -> Option<String> {
    let lower_tags: Vec<String> = info.tags.iter().map(|t| t.to_ascii_lowercase()).collect();
    let name = info.id.rsplit('/').next().unwrap_or(&info.id).to_ascii_lowercase();
    let words: Vec<&str> = name.split(['-', '_', '.']).collect();
    if info.library_name.as_deref().is_some_and(|l| l.eq_ignore_ascii_case("mlx")) || lower_tags.iter().any(|t| t == "mlx") {
        return Some("MLX".into());
    }
    let markers = [
        ("nvfp4", "NVFP4"),
        ("fp8", "FP8"),
        ("awq", "AWQ"),
        ("gptq", "GPTQ"),
        ("exl2", "EXL2"),
        ("exl3", "EXL3"),
        ("bnb", "bitsandbytes"),
        ("bitsandbytes", "bitsandbytes"),
        ("onnx", "ONNX"),
        ("mlx", "MLX"),
        ("compressed-tensors", "compressed-tensors"),
    ];
    markers
        .iter()
        .find(|(m, _)| lower_tags.iter().any(|t| t == m) || words.contains(m))
        .map(|(_, label)| label.to_string())
}

/// What a repo holds for llama.cpp.
pub fn repo_kind(info: &RepoInfo, cat: &Catalog) -> RepoKind {
    if !cat.choices.is_empty() {
        return RepoKind::Gguf;
    }
    let names: Vec<String> = info.siblings.iter().map(|f| f.name().to_ascii_lowercase()).collect();
    let adapter = info.library_name.as_deref().is_some_and(|l| l.eq_ignore_ascii_case("peft"))
        || info.tags.iter().any(|t| t.starts_with("base_model:adapter:"))
        || names.iter().any(|n| n == "adapter_config.json");
    if adapter {
        return RepoKind::Adapter;
    }
    if let Some(f) = other_format(info) {
        return RepoKind::OtherFormat(f);
    }
    let weights = names.iter().any(|n| {
        n.ends_with(".safetensors") || (n.starts_with("pytorch_model") && n.ends_with(".bin")) || n.ends_with(".pt")
    });
    if weights {
        RepoKind::Safetensors
    } else {
        RepoKind::Empty
    }
}

/// A size in the unit that reads best: KiB, MiB or GiB.
pub fn human_size(b: u64) -> String {
    let b = b as f64;
    if b >= GIB {
        format!("{:.1} GiB", b / GIB)
    } else if b >= 1024.0 * 1024.0 {
        format!("{:.1} MiB", b / (1024.0 * 1024.0))
    } else {
        format!("{:.0} KiB", (b / 1024.0).ceil())
    }
}

fn gib(b: u64) -> String {
    human_size(b)
}

/// Someone else's text (a model card's gating terms) as one short line:
/// runs of whitespace, line breaks and control characters made one space,
/// at most `max` characters, as commit subjects are kept.
fn one_line(text: &str, max: usize) -> String {
    let words: Vec<&str> = text.split(|c: char| c.is_whitespace() || c.is_control()).filter(|w| !w.is_empty()).collect();
    words.join(" ").chars().take(max).collect()
}

fn short(sha: &str) -> &str {
    sha.get(..7).unwrap_or(sha)
}

/// llama.cpp links in the repo's model card, else in its base model's.
/// The cards are untrusted: only URLs are taken from them.
fn card_refs_for(env: &dyn Env, repo: &str, sha: &str, info: Option<&RepoInfo>, notes: &mut Vec<Note>) -> Vec<GitRef> {
    let mut refs = match env.readme(repo, sha) {
        Ok(Some(text)) => compat::card_refs(&text),
        Ok(None) => Vec::new(),
        Err(e) => {
            notes.push(Note::new(Level::Warning, "card-unread", format!("{repo}'s model card could not be read: {e}")));
            Vec::new()
        }
    };
    if refs.is_empty() {
        let bases = info.map(|i| i.base_models.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>()).unwrap_or_default();
        for base in bases.iter().take(2) {
            if let Ok(Some(text)) = env.readme(base, "main") {
                refs = compat::card_refs(&text);
                if !refs.is_empty() {
                    notes.push(Note::new(
                        Level::Info,
                        "card-of-base",
                        format!("llama.cpp links taken from the base model's card ({base})"),
                    ));
                    break;
                }
            }
        }
    }
    refs
}

/// The projector to take when the user asks for one without naming it:
/// F16, then BF16, then Q8_0, then F32 (twice the size for nothing), then
/// whatever there is.
pub fn default_mmproj(cat: &Catalog) -> Option<&RepoFile> {
    let rank = |f: &RepoFile| {
        let n = f.name().to_ascii_lowercase();
        ["f16", "bf16", "q8_0", "f32"].iter().position(|q| n.contains(&format!("-{q}")) || n.contains(&format!("_{q}"))).unwrap_or(4)
    };
    cat.mmproj.iter().min_by_key(|f| (rank(f), f.size))
}

/// The draft to take when the user asks for one without naming it: one
/// named for the chosen quant, else a Q8_0 one, else the smallest; never a
/// head profiles cannot run (`profile::draft_mode_of`).
pub fn default_draft<'a>(cat: &'a Catalog, choice: &Choice) -> Option<&'a RepoFile> {
    let lower = |f: &RepoFile| f.name().to_ascii_lowercase();
    let quant = choice.quant.as_deref().map(str::to_ascii_lowercase);
    let usable = || cat.drafts.iter().filter(|f| profile::draft_mode_of(&f.path).is_some());
    quant
        .and_then(|q| usable().find(|f| lower(f).contains(&q)))
        .or_else(|| usable().find(|f| lower(f).contains("q8_0")))
        .or_else(|| usable().min_by_key(|f| f.size))
}

fn root_infos(cfg: &Config) -> Vec<RootInfo> {
    cfg.model_roots
        .iter()
        .map(|p| RootInfo { path: p.clone(), exists: p.is_dir(), free_bytes: crate::disk::free_bytes(p) })
        .collect()
}

/// Add `path` to the model folders (creating it), for downloads to land
/// where the scan finds them. False when it is already one.
pub fn add_model_root(cfg: &mut Config, path: &Path) -> Result<bool> {
    let text = path.to_string_lossy();
    let trimmed = PathBuf::from(text.trim().trim_end_matches(['\\', '/']));
    if !trimmed.is_absolute() {
        return Err(Error::InvalidInput(format!("{} is not an absolute folder path", path.display())));
    }
    if trimmed.is_file() {
        return Err(Error::InvalidInput(format!("{} is a file, not a folder", trimmed.display())));
    }
    if cfg.model_roots.iter().any(|r| same_path(r, &trimmed)) {
        return Ok(false);
    }
    std::fs::create_dir_all(&trimmed).map_err(|e| Error::io(&trimmed, e))?;
    cfg.model_roots.push(trimmed);
    Ok(true)
}

/// The model folders with the space left on each.
pub fn model_roots(cfg: &Config) -> Vec<RootInfo> {
    root_infos(cfg)
}

// ------------------------------------------------------------------ search ----

pub fn search(cfg: &Config, q: &SearchQuery) -> Result<Vec<SearchHit>> {
    search_with(&LiveEnv::new(cfg.clone()), q)
}

/// Hub search, each hit marked with whether an installed build knows its
/// architecture (the llama.dll probe, cached per build).
pub fn search_with(env: &dyn Env, q: &SearchQuery) -> Result<Vec<SearchHit>> {
    let hits = env.search(q)?;
    let builds: Vec<Build> = env.builds().into_iter().filter(|b| b.version_error.is_none()).collect();
    let mut known: HashMap<String, Option<bool>> = HashMap::new();
    Ok(hits
        .into_iter()
        .map(|hit| {
            let arch_known = hit.arch.as_ref().and_then(|a| {
                *known.entry(a.clone()).or_insert_with(|| {
                    if builds.is_empty() {
                        return None;
                    }
                    let needs = ModelNeeds::new(a.clone(), None, None, engine_of_arch(a));
                    Some(builds.iter().any(|b| !env.probe(b, &needs).is_no()))
                })
            });
            SearchHit { hit, arch_known }
        })
        .collect())
}

// ----------------------------------------------------------------- inspect ----

pub fn inspect(cfg: &Config, input: &str, rev: Option<&str>) -> Result<RepoView> {
    inspect_with(&LiveEnv::new(cfg.clone()), cfg, input, rev)
}

/// Look at a repo (see the module notes).
pub fn inspect_with(env: &dyn Env, cfg: &Config, input: &str, rev: Option<&str>) -> Result<RepoView> {
    let r = hub::parse_repo_input(input).ok_or_else(|| {
        Error::InvalidInput(format!(
            "{:?} is not a Hugging Face model: paste owner/name or a huggingface.co link",
            input.trim()
        ))
    })?;
    let rev = rev.map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).or(r.rev.clone());
    let mut info = env.model_info(&r.repo, rev.as_deref())?;
    if info.siblings.is_empty() {
        info.siblings = env.list_files(&r.repo, &info.sha)?;
    }
    let repo = info.id.clone();
    let sha = info.sha.clone();
    let catalog = catalog::classify(&info.siblings);
    let kind = repo_kind(&info, &catalog);
    let draft_modes = catalog.drafts.iter().map(|f| (f.path.clone(), profile::draft_mode_of(&f.path).map(String::from))).collect();
    let mut notes: Vec<Note> = Vec::new();
    let mut view = RepoView {
        input: input.trim().to_string(),
        repo: repo.clone(),
        sha: sha.clone(),
        rev: rev.clone(),
        kind: kind.clone(),
        info: info.clone(),
        catalog,
        notes: Vec::new(),
        derivatives: Vec::new(),
        derivatives_of: None,
        header: None,
        header_of: None,
        header_error: None,
        needs: None,
        preselect: None,
        draft_modes,
        devices: Vec::new(),
        fits: Vec::new(),
        recommended: None,
        builds: Vec::new(),
        usable_build: None,
        build_plan: None,
        build_sources: Vec::new(),
        model_roots: root_infos(cfg),
    };

    if kind != RepoKind::Gguf {
        let base = info.base_models.first().map(|(_, b)| b.clone());
        let (message, of) = match &kind {
            RepoKind::Safetensors => (
                format!(
                    "{repo} holds weights for transformers (safetensors), not GGUF. llama.cpp loads only GGUF; \
                     converting takes Python and the matching build's convert script. GGUF quantizations of it \
                     published on the Hub are listed instead."
                ),
                Some(repo.clone()),
            ),
            RepoKind::Adapter => (
                format!(
                    "{repo} is a LoRA / PEFT adapter, not a model: llama.cpp cannot run it on its own, and \
                     profiles do not apply adapters.{}",
                    base.as_ref().map(|b| format!(" GGUF quantizations of its base model {b} are listed instead.")).unwrap_or_default()
                ),
                base.clone(),
            ),
            RepoKind::OtherFormat(f) => (
                format!(
                    "{repo} is quantized as {f}, a format for other runtimes (vLLM, TensorRT-LLM, MLX, ...): \
                     llama.cpp cannot load it. GGUF quantizations of {} are listed instead.",
                    base.as_deref().unwrap_or(&repo)
                ),
                Some(base.clone().unwrap_or_else(|| repo.clone())),
            ),
            _ => {
                let only_aux = !view.catalog.mmproj.is_empty() || !view.catalog.drafts.is_empty();
                (
                    if only_aux {
                        format!("{repo} holds only projectors or draft heads, no model to run")
                    } else {
                        format!("{repo} holds no model files")
                    },
                    None,
                )
            }
        };
        notes.push(Note::new(Level::Error, "not-gguf", message));
        if let Some(of) = of {
            match env.derivatives(&of) {
                Ok(d) => {
                    if d.is_empty() {
                        notes.push(Note::new(
                            Level::Info,
                            "no-derivatives",
                            format!("the Hub lists no GGUF repo that declares {of} as its base model"),
                        ));
                    }
                    view.derivatives = d;
                }
                Err(e) => notes.push(Note::new(Level::Warning, "derivatives-failed", format!("GGUF quantizations of {of}: {e}"))),
            }
            view.derivatives_of = Some(of);
        }
        view.notes = notes;
        return Ok(view);
    }

    if info.gated != Gated::No {
        let how = if info.gated == Gated::Manual { " (the owner approves each request by hand)" } else { "" };
        notes.push(Note::new(
            Level::Info,
            "gated",
            format!(
                "{repo} is gated: accept its terms at {}/{repo}{how} before downloading{}",
                hub::DEFAULT_ENDPOINT,
                info.gated_prompt
                    .as_deref()
                    .map(|p| one_line(p, 400))
                    .filter(|p| !p.is_empty())
                    .map(|p| format!(". The terms: {p}"))
                    .unwrap_or_default()
            ),
        ));
        if !env.has_hf_token() {
            notes.push(Note::new(
                Level::Warning,
                "no-token",
                "no Hugging Face token is set, and a gated repo downloads only with one: set it in Settings (or HF_TOKEN)",
            ));
        }
    }
    if !view.catalog.ignored.is_empty() {
        let incomplete = view.catalog.ignored.iter().filter(|(_, why)| why.starts_with("incomplete split")).count();
        if incomplete > 0 {
            notes.push(Note::new(
                Level::Warning,
                "incomplete-split",
                format!("{incomplete} file(s) belong to split sets with parts missing in the repo; they are not offered"),
            ));
        }
    }

    // A pasted file link names the choice.
    view.preselect = r
        .path
        .as_ref()
        .and_then(|p| view.catalog.choices.iter().find(|c| c.files.iter().any(|f| &f.path == p)).map(|c| c.label.clone()));

    // The header of one choice: the architecture keys are the same for
    // every quant. 64 KiB first; the whole header only when those keys come
    // after the tokenizer.
    let first = view
        .preselect
        .as_ref()
        .and_then(|l| view.catalog.choices.iter().find(|c| &c.label == l))
        .or(view.catalog.choices.first())
        .cloned()
        .expect("a GGUF repo has a choice");
    let header = match env.header(&repo, &sha, &first.files, ReadMode::UntilTokenizer) {
        Ok(h) if h.block_count.is_none() || h.architecture.is_none() => {
            env.header(&repo, &sha, &first.files, ReadMode::Full).or(Ok(h))
        }
        other => other,
    };
    match header {
        Ok(h) => {
            view.header_of = Some(first.label.clone());
            view.needs = needs_of(&h);
            view.header = Some(slim(h));
        }
        Err(e) => {
            view.header_error = Some(e.to_string());
            notes.push(Note::new(
                Level::Warning,
                "header-unread",
                format!("the GGUF header of {} could not be read, so nothing could be estimated: {e}", first.first_file),
            ));
        }
    }
    if view.needs.is_none() {
        // The Hub's own summary still names the architecture.
        if let Some(arch) = info.gguf.as_ref().and_then(|g| g.architecture.clone()) {
            view.needs = Some(ModelNeeds::new(arch.clone(), None, None, engine_of_arch(&arch)));
        }
    }

    match env.devices() {
        Ok(d) => view.devices = d,
        Err(e) => notes.push(Note::new(Level::Warning, "no-devices", format!("could not list the GPUs, so fits are unknown: {e}"))),
    }
    if let Some(h) = &view.header {
        view.fits = catalog::estimate_choices(h, &view.catalog.choices, &view.devices, &FitOptions::default());
        view.recommended = view.preselect.clone().or_else(|| catalog::recommend(&view.fits));
        if view.recommended.is_none() && !view.devices.is_empty() {
            notes.push(Note::new(
                Level::Warning,
                "nothing-fits",
                "no file is estimated to fit these cards at the default context, even split over two",
            ));
        }
    }

    if let Some(needs) = view.needs.clone() {
        let builds = env.builds();
        view.builds = verdicts(env, &needs, &builds);
        view.usable_build = best_usable(&view.builds).map(|b| b.path.clone());
        if view.usable_build.is_none() {
            let refs = card_refs_for(env, &repo, &sha, Some(&info), &mut notes);
            let gfx = env.gpu_targets().unwrap_or_default();
            let hint = model_hint(&repo, Some(&info));
            let inputs = env.gather(&needs, &builds, &refs, &gfx, hint.as_deref());
            view.build_plan = Some(compat::plan_build(cfg, &needs, &inputs));
            view.build_sources = sources_of(&inputs);
        }
    } else {
        notes.push(Note::new(Level::Warning, "no-arch", "the model's architecture is unknown, so no build could be checked"));
    }
    view.notes = notes;
    Ok(view)
}

// -------------------------------------------------------------------- plan ----

pub fn plan(cfg: &Config, view: &RepoView, req: &PlanRequest) -> Result<WizardPlan> {
    let env = LiveEnv::with_devices(cfg.clone(), Some(view.devices.clone()).filter(|d| !d.is_empty()));
    plan_with(&env, cfg, view, req)
}

/// A build step, the build the profile uses, and the plan they came from.
struct BuildPick {
    step: Option<PlanStep>,
    build: Option<PlannedBuild>,
    plan: Option<BuildPlan>,
    sources: Vec<BuildSource>,
}

fn consent_text(step: &PlanStep) -> Option<String> {
    step.needs_consent.then(|| consent_sentence(&step.action))
}

/// What running `action` means, in one sentence, for the consent box.
fn consent_sentence(action: &PlanAction) -> String {
    match action {
        PlanAction::BuildFork { owner, repo, source, .. } => format!(
            "Building {} runs code from {owner}/{repo} (commit {}) that nobody reviewed for your machine.",
            source.git_source().display(),
            short(&source.sha)
        ),
        PlanAction::BuildPr { number, source, .. } => format!(
            "Building pull request #{number} runs code from {} (commit {}) that nobody reviewed for your machine.",
            source.remote_url.trim_start_matches("https://github.com/"),
            short(&source.sha)
        ),
        PlanAction::InstallUnsloth { tag, .. } => format!(
            "Unsloth's {tag} merges upstream pull requests nobody reviewed for your machine, and runs their code."
        ),
        _ => "This step runs code nobody reviewed for your machine.".to_string(),
    }
}

/// Whether a step runs code nobody reviewed for this machine. Building a
/// pull request's or a fork's code always does, whatever the flags of the
/// plan say (a plan that went to a web view and came back may have lost
/// them); an Unsloth mix does when the plan picked it for an unmerged pull
/// request, which only its flag records.
pub fn step_needs_consent(s: &Step) -> bool {
    s.needs_consent
        || match &s.action {
            StepAction::Build { plan: ps } => {
                ps.needs_consent || matches!(ps.action, PlanAction::BuildPr { .. } | PlanAction::BuildFork { .. })
            }
            _ => false,
        }
}

/// The directory a build step installs into.
fn step_install_dir(step: &PlanStep) -> Option<&Path> {
    match &step.action {
        PlanAction::InstallUpstream { install_dir, .. }
        | PlanAction::InstallUnsloth { install_dir, .. }
        | PlanAction::BuildPr { install_dir, .. }
        | PlanAction::BuildFork { install_dir, .. } => Some(install_dir),
        _ => None,
    }
}

fn step_title(step: &PlanStep) -> String {
    match &step.action {
        PlanAction::InstallUpstream { tag, .. } => format!("Install upstream {tag} (prebuilt)"),
        PlanAction::InstallUnsloth { tag, gfx, .. } => format!("Install Unsloth {tag} ({gfx})"),
        PlanAction::BuildPr { number, source, gpu_targets, .. } => {
            format!("Build pull request #{number} at {} for {gpu_targets}", short(&source.sha))
        }
        PlanAction::BuildFork { source, gpu_targets, .. } => {
            format!("Build {} for {gpu_targets}", source.git_source().display())
        }
        PlanAction::UseInstalled { name, .. } => format!("Use {name}"),
        PlanAction::Unsupported { .. } => "No build".into(),
    }
}

#[allow(clippy::too_many_arguments)]
fn pick_build(
    env: &dyn Env,
    cfg: &Config,
    view: &RepoView,
    needs: Option<&ModelNeeds>,
    builds: &[Build],
    choice: &BuildChoice,
    notes: &mut Vec<Note>,
) -> Result<BuildPick> {
    let Some(needs) = needs else {
        notes.push(Note::new(Level::Warning, "build-unchecked", "the model's architecture is unknown: no build was checked"));
        let b = enumeration_build(builds);
        return Ok(BuildPick {
            step: None,
            build: b.map(|b| PlannedBuild { path: b.path.clone(), name: build_name(b), installed: true, support: None }),
            plan: None,
            sources: Vec::new(),
        });
    };
    let v = verdicts(env, needs, builds);
    let installed = |bv: &BuildVerdict| PlannedBuild {
        path: bv.path.clone(),
        name: bv.name.clone(),
        installed: true,
        support: Some(bv.support.clone()),
    };
    match choice {
        BuildChoice::Skip => {
            let b = best_usable(&v).or_else(|| v.iter().find(|b| !b.broken));
            match b {
                Some(b) if !b.support.is_yes() => notes.push(Note::new(
                    Level::Warning,
                    "build-skipped",
                    format!(
                        "no installed build is known to load it; the profile goes on {} and pre-flight will say \
                         what is missing",
                        b.name
                    ),
                )),
                None => notes.push(Note::new(Level::Warning, "no-build", "no llama.cpp build is installed: no profile is created")),
                _ => {}
            }
            return Ok(BuildPick { step: None, build: b.map(installed), plan: None, sources: Vec::new() });
        }
        BuildChoice::Installed(path) => {
            let b = v
                .iter()
                .find(|b| same_path(&b.path, path))
                .ok_or_else(|| Error::InvalidInput(format!("no installed build at {}", path.display())))?;
            match &b.support {
                Support::No { missing } => notes.push(Note::new(
                    Level::Error,
                    "build-cannot-load",
                    format!("{} cannot load this model: it lacks {}", b.name, compat::describe_missing(needs, missing)),
                )),
                Support::Unknown(why) => {
                    notes.push(Note::new(Level::Warning, "build-unverified", format!("{} may not load it: {why}", b.name)))
                }
                Support::Yes => {}
            }
            if b.broken {
                notes.push(Note::new(Level::Error, "build-broken", format!("{} does not run (--version failed)", b.name)));
            }
            return Ok(BuildPick { step: None, build: Some(installed(b)), plan: None, sources: Vec::new() });
        }
        BuildChoice::Auto => {
            if let Some(b) = best_usable(&v) {
                return Ok(BuildPick { step: None, build: Some(installed(b)), plan: None, sources: Vec::new() });
            }
        }
        BuildChoice::Plan(_) => {}
    }

    // The build plan: the one inspect made when it was made for the same
    // needs, else a fresh one (GitHub answers are cached for minutes, and
    // source tables for good).
    let (bp, sources) = match &view.build_plan {
        Some(bp) if &bp.needs == needs => (bp.clone(), view.build_sources.clone()),
        _ => {
            let mut card_notes = Vec::new();
            let refs = card_refs_for(env, &view.repo, &view.sha, Some(&view.info), &mut card_notes);
            notes.extend(card_notes.into_iter().filter(|n| n.level != Level::Info));
            let gfx = env.gpu_targets().unwrap_or_default();
            let hint = model_hint(&view.repo, Some(&view.info));
            let inputs = env.gather(needs, builds, &refs, &gfx, hint.as_deref());
            (compat::plan_build(cfg, needs, &inputs), sources_of(&inputs))
        }
    };
    let index = match choice {
        BuildChoice::Plan(i) => *i,
        _ => 0,
    };
    let step = if index == 0 {
        bp.step.clone()
    } else {
        bp.alternatives
            .get(index - 1)
            .cloned()
            .ok_or_else(|| Error::InvalidInput(format!("the build plan has no alternative {index}")))?
    };
    for w in &step.warnings {
        notes.push(Note::new(Level::Warning, "build-warning", w.clone()));
    }
    let pick = match &step.action {
        PlanAction::UseInstalled { path, name } => {
            let support = v.iter().find(|b| same_path(&b.path, path)).map(|b| b.support.clone());
            BuildPick {
                build: Some(PlannedBuild { path: path.clone(), name: name.clone(), installed: true, support }),
                step: None,
                plan: Some(bp),
                sources,
            }
        }
        PlanAction::Unsupported { reason } => {
            notes.push(Note::new(
                Level::Warning,
                "no-build-found",
                format!("{reason} The files can still be downloaded; no profile is created until a build loads the model."),
            ));
            BuildPick { step: None, build: None, plan: Some(bp), sources }
        }
        _ => {
            let dir = step_install_dir(&step).map(Path::to_path_buf).unwrap_or_default();
            let name = match &step.action {
                PlanAction::BuildPr { source, .. } | PlanAction::BuildFork { source, .. } => source.git_source().display(),
                _ => dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            };
            let support = if step.verified { Some(Support::Yes) } else { None };
            BuildPick {
                build: Some(PlannedBuild { path: dir, name, installed: false, support }),
                step: Some(step),
                plan: Some(bp),
                sources,
            }
        }
    };
    Ok(pick)
}

/// A build that does not exist yet, as the profile preview names it.
fn future_build(step: &PlanStep, pb: &PlannedBuild) -> Build {
    let (channel, git, runner) = match &step.action {
        PlanAction::BuildPr { source, .. } | PlanAction::BuildFork { source, .. } => (Channel::Git, Some(source.git_source()), None),
        PlanAction::InstallUnsloth { .. } => {
            (Channel::Unsloth, None, Some(pb.path.join("bin").join(discovery::RUNNER_EXE)))
        }
        _ => (Channel::Upstream, None, None),
    };
    let version = match &step.action {
        PlanAction::InstallUpstream { tag, .. } => Some(tag.clone()),
        _ => None,
    };
    Build {
        path: pb.path.clone(),
        tag: pb.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        server_exe: pb.path.join("bin").join("llama-server.exe"),
        version,
        commit: None,
        version_error: None,
        channel,
        bundled_runtime: channel == Channel::Unsloth,
        release_tag: None,
        patch: None,
        runner_exe: runner,
        git,
    }
}

/// The drive part of a path (`E:`), to tell whether two folders share a disk.
fn drive_of(p: &Path) -> Option<String> {
    match p.components().next()? {
        std::path::Component::Prefix(pre) => Some(pre.as_os_str().to_string_lossy().to_ascii_uppercase()),
        _ => None,
    }
}

/// Where a build step writes and about how much, one entry per drive:
/// (a folder on it, bytes, what for). A source build's worktree and build
/// tree sit beside FIDIM's llama.cpp clone (under the config folder,
/// usually on C:), whatever the install folder is.
fn build_space(ps: &PlanStep, install_dir: &Path) -> Vec<(PathBuf, u64, &'static str)> {
    let parts: Vec<(PathBuf, u64, &'static str)> = match &ps.action {
        PlanAction::BuildPr { .. } | PlanAction::BuildFork { .. } => vec![
            (update::source_checkout_dir(), SOURCE_SCRATCH_BYTES, "the source checkout and build tree"),
            (install_dir.to_path_buf(), SOURCE_INSTALL_BYTES, "the install"),
        ],
        _ => vec![(install_dir.to_path_buf(), PREBUILT_BYTES, "the zips and the install")],
    };
    let mut out: Vec<(PathBuf, u64, &'static str)> = Vec::new();
    for (dir, bytes, what) in parts {
        match out.iter_mut().find(|(d, _, _)| drive_of(d).is_some() && drive_of(d) == drive_of(&dir)) {
            Some(e) => {
                e.1 += bytes;
                e.2 = "the source checkout, build tree and install";
            }
            None => out.push((dir, bytes, what)),
        }
    }
    out
}

/// Refuse a download the drive has no room for now: the plan checked when
/// it was made, and a plan can wait (on the Get step, at "Go ahead?")
/// while other downloads fill the drive. `from`: the first step still to
/// run; what the downloads from there still have to write, and 5%, must
/// be free.
fn check_download_room(env: &dyn Env, plan: &WizardPlan, from: usize) -> Result<()> {
    let rest: Vec<(PathBuf, u64)> = plan.steps[from.min(plan.steps.len())..]
        .iter()
        .filter_map(|s| match &s.action {
            StepAction::Download { dest, file, .. } => Some((dest.clone(), file.size)),
            _ => None,
        })
        .collect();
    let bytes = crate::disk::bytes_to_fetch(&rest);
    if bytes == 0 {
        return Ok(());
    }
    match env.free_bytes(&plan.dest_root) {
        Some(free) if free < crate::disk::with_margin(bytes) => Err(Error::InvalidInput(format!(
            "the downloads still need {} (plus 5%) and the drive holding {} has {} free now: make room there, \
             or plan again with another model folder (finished and partial downloads are kept)",
            gib(bytes),
            plan.dest_root.display(),
            gib(free)
        ))),
        _ => Ok(()),
    }
}

/// Refuse a source build that would fill the drive of FIDIM's llama.cpp
/// clone: less free there than the build's measured peak.
fn check_build_room(env: &dyn Env, ps: &PlanStep) -> Result<()> {
    if !matches!(ps.action, PlanAction::BuildPr { .. } | PlanAction::BuildFork { .. }) {
        return Ok(());
    }
    let dir = update::source_checkout_dir();
    match env.free_bytes(&dir) {
        Some(free) if free < SOURCE_SCRATCH_MIN => Err(Error::InvalidInput(format!(
            "a source build needs about {} of scratch space on the drive of {} and it has {} free: make room \
             there first",
            gib(SOURCE_SCRATCH_BYTES),
            dir.display(),
            gib(free)
        ))),
        _ => Ok(()),
    }
}

/// Turn a choice into exact steps (see the module notes).
pub fn plan_with(env: &dyn Env, cfg: &Config, view: &RepoView, req: &PlanRequest) -> Result<WizardPlan> {
    if view.kind != RepoKind::Gguf {
        return Err(Error::InvalidInput(format!("{} has no GGUF file to download", view.repo)));
    }
    hub::validate_repo(&view.repo)?;
    let label = req
        .choice
        .clone()
        .or_else(|| view.recommended.clone())
        .or_else(|| view.catalog.choices.first().map(|c| c.label.clone()))
        .ok_or_else(|| Error::InvalidInput(format!("{} offers no file to choose", view.repo)))?;
    let choice = view
        .catalog
        .choices
        .iter()
        .find(|c| c.label == label || c.quant.as_deref().is_some_and(|q| q.eq_ignore_ascii_case(&label)))
        .cloned()
        .ok_or_else(|| {
            let all: Vec<&str> = view.catalog.choices.iter().map(|c| c.label.as_str()).collect();
            Error::InvalidInput(format!("{} has no file {label:?}; it offers: {}", view.repo, all.join(", ")))
        })?;
    let aux = |path: &Option<String>, list: &[RepoFile], what: &str| -> Result<Option<RepoFile>> {
        match path {
            None => Ok(None),
            Some(p) => list
                .iter()
                .find(|f| &f.path == p || f.name().eq_ignore_ascii_case(p))
                .cloned()
                .map(Some)
                .ok_or_else(|| Error::InvalidInput(format!("{} has no {what} {p:?}", view.repo))),
        }
    };
    let mmproj = aux(&req.mmproj, &view.catalog.mmproj, "projector")?;
    let draft = aux(&req.draft, &view.catalog.drafts, "draft")?;
    // The mode comes from the file's place in the repo: the download
    // flattens an `MTP/` folder away. A head no profile mode runs is
    // refused rather than set up as a draft model it is not.
    let draft_mode = match &draft {
        Some(f) => Some(
            profile::draft_mode_of(&f.path)
                .ok_or_else(|| {
                    Error::InvalidInput(format!(
                        "{} is {}: llama-server runs it with a speculative type profiles do not have yet, and as                          a plain draft model it would not load; pick another draft, or none",
                        f.path,
                        profile::unsupported_draft_kind(&f.path)
                    ))
                })?
                .to_string(),
        ),
        None => None,
    };
    let dest_root = req
        .dest_root
        .clone()
        .or_else(|| cfg.model_roots.first().cloned())
        .ok_or_else(|| Error::InvalidInput("no model folder: add one in Settings, or pass --dest".into()))?;
    if !dest_root.is_absolute() {
        return Err(Error::InvalidInput(format!("{} is not an absolute folder path", dest_root.display())));
    }
    // One separator throughout the paths shown and saved.
    #[cfg(windows)]
    let dest_root = PathBuf::from(dest_root.to_string_lossy().replace('/', "\\"));
    let mut notes: Vec<Note> = Vec::new();

    // The chosen file's whole header: tensor types, for the build check.
    let mut header = view.header.clone();
    let mut needs = view.needs.clone();
    if view.header.is_some() {
        match env.header(&view.repo, &view.sha, &choice.files, ReadMode::Full) {
            Ok(h) => {
                needs = needs_of(&h).or(needs);
                header = Some(slim(h));
            }
            Err(e) => {
                if let Some(h) = header.as_mut() {
                    h.file_size = choice.total_size;
                }
                notes.push(Note::new(
                    Level::Warning,
                    "types-unchecked",
                    format!("the tensor types of {} were not checked (its whole header could not be read): {e}", choice.label),
                ));
            }
        }
    }
    let engine = header.as_ref().map(|h| h.engine()).or(needs.as_ref().map(|n| n.engine)).unwrap_or_default();

    // The estimate of this choice, projector and draft included.
    let fit = header.as_ref().map(|h| {
        let opts = FitOptions {
            ctx: req.ctx,
            mmproj_bytes: mmproj.as_ref().map_or(0, |f| f.size),
            draft_bytes: draft.as_ref().map_or(0, |f| f.size),
            ..FitOptions::default()
        };
        catalog::estimate_choices(h, std::slice::from_ref(&choice), &view.devices, &opts).remove(0)
    });
    let fits = |f: Fit| matches!(f, Fit::Fits | Fit::Tight);
    let one_card = fit.as_ref().is_some_and(|f| fits(f.one_card.fit));
    let split = !one_card && fit.as_ref().is_some_and(|f| fits(f.two_card_split.fit));
    if let Some(f) = &fit {
        if !view.devices.is_empty() && !one_card && !split {
            notes.push(Note::new(
                Level::Warning,
                "does-not-fit",
                format!(
                    "{} is estimated not to fit at context {}: {}. Pick a smaller file or a shorter context.",
                    choice.label, f.ctx, f.one_card.detail
                ),
            ));
        }
    }

    // The build.
    let builds = env.builds();
    let pick = pick_build(env, cfg, view, needs.as_ref(), &builds, &req.build, &mut notes)?;
    let mut toolchain: Vec<ToolFinding> = Vec::new();
    let mut steps: Vec<Step> = Vec::new();
    let mut consent = None;
    if let Some(ps) = &pick.step {
        if let PlanAction::BuildPr { gpu_targets, .. } | PlanAction::BuildFork { gpu_targets, .. } = &ps.action {
            if gpu_targets.trim().is_empty() {
                notes.push(Note::new(
                    Level::Error,
                    "no-gpu-target",
                    "this machine's GPU target is unknown (ROCm's hipInfo was not found and the card names are not \
                     known ones), so a source build cannot start: point Settings > fallback runtime folder at a HIP \
                     SDK bin folder (hipInfo.exe is read from there), or give the target on the command line: \
                     `fidim models get <repo> --gfx gfx1201`",
                ));
            } else {
                let first = gpu_targets.split(',').next().unwrap_or("").trim().to_string();
                for f in env.doctor(&first) {
                    let tf = ToolFinding::from(&f);
                    match tf.outcome.as_str() {
                        "block" => notes.push(Note::new(
                            Level::Error,
                            "toolchain",
                            format!("{}: {}{}", tf.title, tf.message, tf.fix.as_deref().map(|x| format!(" Fix: {x}")).unwrap_or_default()),
                        )),
                        "warn" => notes.push(Note::new(Level::Warning, "toolchain", format!("{}: {}", tf.title, tf.message))),
                        _ => {}
                    }
                    toolchain.push(tf);
                }
            }
        }
        let mut step = Step {
            title: step_title(ps),
            detail: ps.explanation.clone(),
            needs_consent: false,
            action: StepAction::Build { plan: ps.clone() },
        };
        step.needs_consent = step_needs_consent(&step);
        consent = step.needs_consent.then(|| consent_text(ps).unwrap_or_else(|| consent_sentence(&ps.action)));
        steps.push(step);
    }

    // The downloads.
    let mut files: Vec<(FileRole, RepoFile)> = choice.files.iter().map(|f| (FileRole::Model, f.clone())).collect();
    files.extend(mmproj.iter().map(|f| (FileRole::Mmproj, f.clone())));
    files.extend(draft.iter().map(|f| (FileRole::Draft, f.clone())));
    let mut dests: Vec<(PathBuf, u64)> = Vec::new();
    for (role, f) in &files {
        hub::validate_repo_path(&f.path)?;
        let dest = catalog::dest_path(&dest_root, &view.repo, &f.path);
        let present = dest.is_file();
        let have = if present {
            f.size
        } else {
            std::fs::metadata(fetch::part_path(&dest)).map(|m| m.len().min(f.size)).unwrap_or(0)
        };
        let what = match role {
            FileRole::Model if files.iter().filter(|(r, _)| *r == FileRole::Model).count() > 1 => "part of the model",
            FileRole::Model => "the model",
            FileRole::Mmproj => "the vision projector",
            FileRole::Draft => "the draft model",
        };
        let detail = if present {
            format!("{} is already there; its size and SHA-256 are checked instead", dest.display())
        } else if have > 0 {
            format!("resumes at {} of {} into {}", gib(have), gib(f.size), dest.display())
        } else {
            format!("{} into {}", gib(f.size), dest.display())
        };
        steps.push(Step {
            title: format!("Download {} ({what})", f.name()),
            detail,
            needs_consent: false,
            action: StepAction::Download { role: *role, file: f.clone(), dest: dest.clone(), have, present },
        });
        dests.push((dest, f.size));
    }
    let download_bytes = crate::disk::bytes_to_fetch(&dests);
    let total_bytes: u64 = dests.iter().map(|(_, s)| s).sum();
    let rel: Vec<PathBuf> = dests.iter().map(|(d, _)| d.strip_prefix(&dest_root).map(Path::to_path_buf).unwrap_or_else(|_| d.clone())).collect();
    notes.extend(crate::disk::check_destination(&dest_root, &rel, download_bytes, engine.is_diffusion()).into_iter().map(Note::from));
    if !cfg.model_roots.iter().any(|r| under(&dest_root, r)) {
        notes.push(Note::new(
            Level::Warning,
            "dest-not-scanned",
            format!(
                "{} is not one of your model folders: the Profiles picker will not list the model (the new profile \
                 still points at it). Add the folder in Settings to have it scanned.",
                dest_root.display()
            ),
        ));
    }
    if view.info.gated != Gated::No && !env.has_hf_token() {
        notes.push(Note::new(
            Level::Warning,
            "no-token",
            format!("{} is gated and no Hugging Face token is set: the download will be refused", view.repo),
        ));
    }
    // Room for the build itself, drive by drive: a source build's checkout
    // and build tree beside FIDIM's llama.cpp clone, the install in the
    // install folder (and the download too, on a drive they share).
    if let (Some(ps), Some(pb)) = (&pick.step, &pick.build) {
        for (dir, need, what) in build_space(ps, &pb.path) {
            let with_download = drive_of(&dir).is_some() && drive_of(&dir) == drive_of(&dest_root);
            let need_there = need + if with_download { download_bytes } else { 0 };
            if let Some(free) = env.free_bytes(&dir) {
                if free < need_there {
                    notes.push(Note::new(
                        Level::Warning,
                        "build-space",
                        format!(
                            "the build needs about {} on the drive of {} ({what}{}); it has {} free",
                            gib(need_there),
                            dir.display(),
                            if with_download { ", with the download" } else { "" },
                            gib(free)
                        ),
                    ));
                }
            }
        }
    }

    // The profile.
    let dest_dir = catalog::dest_path(&dest_root, &view.repo, "x").parent().map(Path::to_path_buf).unwrap_or_default();
    let first_dest = dests.first().map(|(d, _)| d.clone()).unwrap_or_default();
    let mmproj_dest = mmproj.as_ref().map(|f| catalog::dest_path(&dest_root, &view.repo, &f.path));
    let draft_dest = draft.as_ref().map(|f| catalog::dest_path(&dest_root, &view.repo, &f.path));
    let profile_opts = NewProfileOpts {
        ctx: req.ctx,
        fit_ctx: fit.as_ref().and_then(|f| f.max_ctx_one_card),
        split,
        mmproj: mmproj_dest.clone(),
        draft: draft_dest.clone(),
        draft_mode,
        busy_cards: env.busy_cards(),
        taken_ports: env.taken_ports(),
        name: None,
        notes: Some(format!("From {} at {} ({}), by the model wizard.", view.repo, short(&view.sha), choice.label)),
    };
    let mut profile_preview = None;
    if req.profile {
        if let Some(pb) = &pick.build {
            let build = builds
                .iter()
                .find(|b| same_path(&b.path, &pb.path))
                .cloned()
                .or_else(|| pick.step.as_ref().map(|s| future_build(s, pb)));
            if let Some(build) = build {
                let model = planned_model(&first_dest, &choice, &view.repo, &view.sha, header.clone(), engine, &dests, mmproj_dest.clone(), draft_dest.clone());
                let p = profile::new_for_model(cfg, &model, &build, &view.devices, &env.profiles(), &profile_opts);
                steps.push(Step {
                    title: format!("Create profile {}", p.id),
                    detail: format!(
                        "on {} with {}, port {}, context {}; nothing is launched",
                        pb.name,
                        if p.devices.len() > 1 { "a layer split over two cards".to_string() } else { "one card".to_string() },
                        p.server.port,
                        if p.runtime.ctx_total == 0 { "auto".to_string() } else { p.runtime.ctx_total.to_string() }
                    ),
                    needs_consent: false,
                    action: StepAction::Profile { id: p.id.clone(), path: cfg.profile_dir.join(format!("{}.json", p.id)) },
                });
                for f in profile::validate(&p).into_iter().filter(|f| f.severity == profile::Severity::Error) {
                    notes.push(Note::new(Level::Warning, "profile", format!("the new profile will need a fix: {}", f.message)));
                }
                profile_preview = Some(p);
            }
        }
    }

    let blocked = notes.iter().any(|n| n.level == Level::Error);
    let needs_consent = steps.iter().any(|s| s.needs_consent);
    Ok(WizardPlan {
        repo: view.repo.clone(),
        sha: view.sha.clone(),
        choice,
        mmproj,
        draft,
        dest_root,
        dest_dir,
        steps,
        notes,
        blocked,
        needs_consent,
        consent,
        download_bytes,
        total_bytes,
        engine,
        gated: view.info.gated,
        needs,
        header,
        fit,
        build: pick.build,
        build_plan: pick.plan,
        build_sources: pick.sources,
        build_choice: req.build.clone(),
        toolchain,
        profile: profile_preview,
        profile_opts,
        devices: view.devices.clone(),
    })
}

/// The model as it will be on disk, for the profile.
#[allow(clippy::too_many_arguments)]
fn planned_model(
    first: &Path,
    choice: &Choice,
    repo: &str,
    sha: &str,
    header: Option<GgufHeader>,
    engine: Engine,
    dests: &[(PathBuf, u64)],
    mmproj: Option<PathBuf>,
    draft: Option<PathBuf>,
) -> Model {
    let model_files: Vec<PathBuf> = dests.iter().take(choice.files.len()).map(|(d, _)| d.clone()).collect();
    Model {
        path: first.to_path_buf(),
        file_size: choice.total_size,
        modified_unix: None,
        header: header.map(|mut h| {
            h.file_size = choice.total_size;
            h
        }),
        header_error: None,
        engine,
        mmproj_candidates: mmproj.into_iter().collect(),
        draft_candidates: draft.into_iter().collect(),
        shards: if model_files.len() > 1 { model_files } else { Vec::new() },
        source: Some(ModelSource {
            repo: repo.to_string(),
            commit: sha.to_string(),
            path: choice.first_file.clone(),
            sha256: choice.files.first().and_then(|f| f.sha256.clone()),
        }),
    }
}

// --------------------------------------------------------------------- run ----

pub fn run(
    cfg: &Config,
    devices: Option<Vec<Device>>,
    plan: &WizardPlan,
    consent: bool,
    progress: &mut dyn FnMut(&WizardEvent),
    cancel: &AtomicBool,
) -> Result<WizardResult> {
    run_with(&LiveEnv::with_devices(cfg.clone(), devices), cfg, plan, consent, progress, cancel)
}

/// The refusals `run` makes before it does anything: a plan with errors,
/// consent missing for code nobody reviewed, and a plan that does not say
/// what `plan` would have (a repo, commit or destination changed by hand).
///
/// Consent is required by what a step does (building a pull request or a
/// fork), not by the flags the plan carries. The app does not take plans
/// back from its web view (it runs the copy `plan` made), but this check
/// holds for any caller.
pub fn check_runnable(plan: &WizardPlan, consent: bool) -> Result<()> {
    if plan.blocked {
        let errors: Vec<&str> = plan.notes.iter().filter(|n| n.level == Level::Error).map(|n| n.message.as_str()).collect();
        return Err(Error::InvalidInput(format!("the plan cannot run: {}", errors.join("; "))));
    }
    if !consent {
        if let Some(s) = plan.steps.iter().find(|s| step_needs_consent(s)) {
            // The sentence is made from the step itself, not taken from the
            // plan's `consent`, which is only text.
            let sentence = match &s.action {
                StepAction::Build { plan: ps } => consent_sentence(&ps.action),
                _ => "A step runs code nobody reviewed for your machine.".to_string(),
            };
            return Err(Error::InvalidInput(format!(
                "{sentence} Nothing was done: confirm it first (the consent box in the app, --allow-fork in the CLI)."
            )));
        }
    }
    hub::validate_repo(&plan.repo)?;
    if plan.sha.len() != 40 || !plan.sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::InvalidInput(format!("{:?} is not a commit", plan.sha)));
    }
    for s in &plan.steps {
        if let StepAction::Download { file, dest, .. } = &s.action {
            hub::validate_repo_path(&file.path)?;
            if catalog::dest_path(&plan.dest_root, &plan.repo, &file.path) != *dest {
                return Err(Error::InvalidInput(format!("{} is not where {} belongs", dest.display(), file.path)));
            }
        }
    }
    Ok(())
}

/// Execute a plan (see the module notes). Stops at the first step that
/// fails; downloads keep their `.part` for the next try, and a build that
/// is already installed is only verified again.
pub fn run_with(
    env: &dyn Env,
    cfg: &Config,
    plan: &WizardPlan,
    consent: bool,
    progress: &mut dyn FnMut(&WizardEvent),
    cancel: &AtomicBool,
) -> Result<WizardResult> {
    check_runnable(plan, consent)?;
    let mut result = WizardResult {
        repo: plan.repo.clone(),
        sha: plan.sha.clone(),
        model_path: plan.download_dests().first().map(|(d, _)| d.clone()).unwrap_or_default(),
        files: Vec::new(),
        build: None,
        build_path: plan.build.as_ref().filter(|b| b.installed).map(|b| b.path.clone()),
        profile: None,
        profile_path: None,
        warnings: Vec::new(),
    };
    // A gated or vanished repo fails here, before a build of minutes; so
    // does a drive that filled up since the plan was made.
    if plan.steps.iter().any(|s| matches!(s.action, StepAction::Download { .. })) {
        env.auth_check(&plan.repo)?;
        check_download_room(env, plan, 0)?;
    }
    for (i, step) in plan.steps.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        progress(&WizardEvent::status(i, StepStatus::Running));
        let outcome = run_step(env, cfg, plan, i, step, &mut result, progress, cancel);
        match outcome {
            Ok(status) => progress(&WizardEvent::status(i, status)),
            Err(e) => {
                progress(&WizardEvent { line: Some(e.to_string()), ..WizardEvent::status(i, StepStatus::Failed) });
                return Err(e);
            }
        }
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn run_step(
    env: &dyn Env,
    cfg: &Config,
    plan: &WizardPlan,
    i: usize,
    step: &Step,
    result: &mut WizardResult,
    progress: &mut dyn FnMut(&WizardEvent),
    cancel: &AtomicBool,
) -> Result<StepStatus> {
    match &step.action {
        StepAction::Build { plan: ps } => {
            check_build_room(env, ps)?;
            let mut line = |stage: &str, text: String| {
                progress(&WizardEvent { stage: Some(stage.into()), line: Some(text), ..WizardEvent::status(i, StepStatus::Running) })
            };
            let outcome = match &ps.action {
                PlanAction::InstallUpstream { tag, .. } => env.install_upstream(tag, &mut |l| line("install", l), cancel),
                PlanAction::InstallUnsloth { tag, gfx, .. } => env.install_unsloth(tag, gfx, &mut |l| line("install", l), cancel),
                PlanAction::BuildPr { source, gpu_targets, .. } | PlanAction::BuildFork { source, gpu_targets, .. } => {
                    env.build_source(
                        source,
                        gpu_targets,
                        &mut |p: BuildProgress| {
                            progress(&WizardEvent {
                                stage: Some(p.step.clone()),
                                done: p.done.map(u64::from),
                                total: p.total.map(u64::from),
                                line: Some(p.line),
                                ..WizardEvent::status(i, StepStatus::Running)
                            })
                        },
                        cancel,
                    )
                }
                PlanAction::UseInstalled { .. } | PlanAction::Unsupported { .. } => return Ok(StepStatus::Skipped),
            };
            // Stopped by the user: `cancelled`, as a stopped download is,
            // whatever words the build used for it (build_from_ref reports
            // its cleanup), so the step reads "stopped", not "failed".
            let report = match outcome {
                Err(_) if cancel.load(Ordering::Relaxed) => return Err(Error::Cancelled),
                r => r?,
            };
            if !report.verify.hip_ok {
                result.warnings.push(format!(
                    "{} is installed but its HIP backend did not load in the check (--list-devices): {}",
                    report.tag,
                    report.verify.detail.trim()
                ));
            }
            result.build_path = Some(report.dir.clone());
            result.build = Some(report);
            Ok(StepStatus::Done)
        }
        StepAction::Download { file, dest, .. } => {
            // Again before each file: another program may have written to
            // the drive while the build or the files before this ran.
            check_download_room(env, plan, i)?;
            let name = file.name().to_string();
            env.download(
                &plan.repo,
                &plan.sha,
                file,
                dest,
                &mut |p: &fetch::Progress| {
                    let stage = match p.stage {
                        fetch::Stage::Hashing => "hashing",
                        fetch::Stage::Downloading => "downloading",
                    };
                    progress(&WizardEvent {
                        stage: Some(stage.into()),
                        file: Some(name.clone()),
                        done: Some(p.done),
                        total: p.total.or(Some(file.size)),
                        bps: Some(p.bps),
                        ..WizardEvent::status(i, StepStatus::Running)
                    })
                },
                cancel,
            )?;
            // Where it came from, beside it: a failure here costs only the
            // provenance, not the download.
            if let Some(dir) = dest.parent() {
                if let Err(e) = discovery::write_source_sidecar(dir, &plan.repo, &plan.sha, std::slice::from_ref(file)) {
                    result.warnings.push(format!("{}: {e}", discovery::SOURCE_SIDECAR));
                }
            }
            result.files.push(dest.clone());
            Ok(StepStatus::Done)
        }
        StepAction::Profile { .. } => {
            let Some(build_path) = result.build_path.clone() else {
                result.warnings.push("no build to put the profile on: none was created".into());
                return Ok(StepStatus::Skipped);
            };
            let build = env.builds().into_iter().find(|b| same_path(&b.path, &build_path)).or_else(|| {
                // Not found by a scan (a build root the config does not
                // list): name it from what was installed.
                let pb = plan.build.clone()?;
                let step = plan.steps.iter().find_map(|s| match &s.action {
                    StepAction::Build { plan } => Some(plan.clone()),
                    _ => None,
                })?;
                let mut b = future_build(&step, &PlannedBuild { path: build_path.clone(), ..pb });
                b.version = result.build.as_ref().and_then(|r| r.verify.version.clone()).or(b.version);
                Some(b)
            });
            let Some(build) = build else {
                result.warnings.push(format!("{} is not a build the scan finds: no profile was created", build_path.display()));
                return Ok(StepStatus::Skipped);
            };
            let dests = plan.download_dests();
            let first = dests.first().map(|(d, _)| d.clone()).unwrap_or_default();
            // The header of what is on disk now (the whole model), else the plan's.
            let header = discovery::read_model_header(&first).ok().or_else(|| plan.header.clone());
            let model = planned_model(
                &first,
                &plan.choice,
                &plan.repo,
                &plan.sha,
                header,
                plan.engine,
                &dests,
                plan.profile_opts.mmproj.clone(),
                plan.profile_opts.draft.clone(),
            );
            let devices = env.devices().unwrap_or_else(|_| plan.devices.clone());
            let opts = NewProfileOpts { busy_cards: env.busy_cards(), taken_ports: env.taken_ports(), ..plan.profile_opts.clone() };
            let p = profile::new_for_model(cfg, &model, &build, &devices, &env.profiles(), &opts);
            let path = env.save_profile(&p)?;
            result.profile_path = Some(path);
            result.profile = Some(p);
            Ok(StepStatus::Done)
        }
    }
}

// ------------------------------------------------------------------- needs ----

pub fn needs(cfg: &Config, target: &str) -> Result<NeedsReport> {
    needs_with(&LiveEnv::new(cfg.clone()), cfg, target)
}

/// What a model (a local GGUF, or a repo) needs from a build, every
/// installed build's answer, and the build plan when none can load it.
pub fn needs_with(env: &dyn Env, cfg: &Config, target: &str) -> Result<NeedsReport> {
    let mut notes = Vec::new();
    let local = Path::new(target.trim());
    let (header, repo, sha, info) = if local.is_file() {
        let h = discovery::read_model_header(local)?;
        let src = discovery::model_source(local);
        let repo = src.as_ref().map(|s| s.repo.clone()).or_else(|| crate::hf::repo_from_layout(&cfg.model_roots, local));
        (h, repo, src.map(|s| s.commit), None)
    } else {
        let r = hub::parse_repo_input(target).ok_or_else(|| {
            Error::InvalidInput(format!("{target:?} is neither a GGUF file nor a Hugging Face repo"))
        })?;
        let mut info = env.model_info(&r.repo, r.rev.as_deref())?;
        if info.siblings.is_empty() {
            info.siblings = env.list_files(&r.repo, &info.sha)?;
        }
        let cat = catalog::classify(&info.siblings);
        let choice = r
            .path
            .as_ref()
            .and_then(|p| cat.choices.iter().find(|c| c.files.iter().any(|f| &f.path == p)))
            .or(cat.choices.first())
            .ok_or_else(|| Error::InvalidInput(format!("{} has no GGUF file", info.id)))?;
        let h = env.header(&info.id, &info.sha, &choice.files, ReadMode::Full)?;
        notes.push(Note::new(Level::Info, "read-from", format!("read from {} ({})", choice.first_file, choice.label)));
        (h, Some(info.id.clone()), Some(info.sha.clone()), Some(info))
    };
    let needs = needs_of(&header).ok_or_else(|| Error::InvalidInput(format!("{target} names no architecture")))?;
    let builds = env.builds();
    let v = verdicts(env, &needs, &builds);
    let usable_build = best_usable(&v).map(|b| b.path.clone());
    let build_plan = if usable_build.is_none() {
        let refs = match &repo {
            Some(r) => card_refs_for(env, r, sha.as_deref().unwrap_or("main"), info.as_ref(), &mut notes),
            None => Vec::new(),
        };
        let gfx = env.gpu_targets().unwrap_or_default();
        let hint = repo.as_deref().and_then(|r| model_hint(r, info.as_ref()));
        Some(compat::plan_build(cfg, &needs, &env.gather(&needs, &builds, &refs, &gfx, hint.as_deref())))
    } else {
        None
    };
    Ok(NeedsReport {
        target: target.to_string(),
        repo,
        max_tensor_type: header.max_tensor_type(),
        needs,
        builds: v,
        usable_build,
        build_plan,
        notes,
    })
}
