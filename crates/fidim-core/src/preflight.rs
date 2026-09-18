//! Pre-flight check sequence (spec §04, v1.1). Every check exists because the
//! corresponding failure happened on the target machine and cost hours.
//!
//! Checks are pure functions over a `LaunchContext` assembled by the launch
//! path — the same objects power the profile editor's live validation, so the
//! editor and the launcher can never disagree. Any Block prevents launch; the
//! user may override with an explicit confirmation that names the risk.

use std::path::PathBuf;

use serde::Serialize;

use crate::devices::Device;
use crate::estimate::{DiffusionSizing, VramEstimate};
use crate::gguf::GgufHeader;
use crate::platform::SystemCommit;
use crate::profile::{Engine, Profile};

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "outcome", content = "message", rename_all = "lowercase")]
pub enum Outcome {
    Pass,
    /// Advisory only (§04 "Note").
    Note(String),
    Warn(String),
    Block(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    /// Stable id, e.g. `vram-fits` — override confirmations reference it.
    pub id: &'static str,
    /// §04 row number, for traceability to the spec.
    pub spec_number: u8,
    pub title: &'static str,
    pub outcome: Outcome,
}

impl CheckResult {
    pub fn blocks(&self) -> bool {
        matches!(self.outcome, Outcome::Block(_))
    }
}

/// The process listening on a profile's port, when it is not free.
#[derive(Debug, Clone, Serialize)]
pub struct PortHolder {
    pub pid: Option<u32>,
    pub process_name: Option<String>,
    /// Set when the holder is a live Llama FIDIM run (from its state file).
    pub profile_id: Option<String>,
}

/// A device the launch path resolved from a profile key.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedDevice {
    pub profile_key: String,
    pub device: Device,
    pub fraction: f64,
    /// Set when resolution re-bound a stale key (renumbering event).
    pub rebound: bool,
}

/// What the model header says about the engine it needs (check 13).
#[derive(Debug, Clone, Serialize)]
pub struct ModelFacts {
    pub architecture: Option<String>,
    pub is_diffusion: bool,
    /// The DiffusionGemma runner can load it (its own architecture, with
    /// the canvas key).
    pub runner_supported: bool,
    pub block_count: Option<u64>,
}

impl ModelFacts {
    pub fn from_header(h: &GgufHeader) -> Self {
        ModelFacts {
            architecture: h.architecture.clone(),
            is_diffusion: h.is_diffusion(),
            runner_supported: h.runner_supported(),
            block_count: h.block_count,
        }
    }
}

/// What a diffusion launch needs besides the build's llama-server (check 1)
/// and how its context will be sized (check 15).
#[derive(Debug, Clone, Serialize)]
pub struct DiffusionPreflight {
    /// `fidim-dg.exe` beside the running Llama FIDIM; None = not installed.
    pub helper_exe: Option<PathBuf>,
    /// The running Llama FIDIM's folder, where the helper is looked for.
    pub helper_dir: Option<PathBuf>,
    pub runner_exe: PathBuf,
    pub runner_present: bool,
    /// `bin/ggml-vulkan.dll` is present in the build.
    pub vulkan_backend: bool,
    /// Where the helper writes request files (`<runs_dir>/dg-<id>-<port>`),
    /// and why the runner could not open them there (non-ASCII, too long).
    pub req_prefix: PathBuf,
    pub req_prefix_error: Option<String>,
    /// The helper's `%TEMP%` fallback, judged the same way. Both unusable =
    /// the helper exits 7 at launch, so check 1 Blocks first.
    pub req_fallback: PathBuf,
    pub req_fallback_error: Option<String>,
    pub sizing: Option<DiffusionSizing>,
}

/// A live run on one of this launch's cards (check 14).
#[derive(Debug, Clone, Serialize)]
pub struct CoResident {
    pub profile_id: String,
    pub device_key: String,
    pub engine: Engine,
}

/// Everything §04 needs, assembled by the launch path (or a fake in tests).
#[derive(Debug, Clone, Serialize)]
pub struct LaunchContext {
    pub profile: Profile,
    /// config.allow_integrated: binding an iGPU is a warning, not a block.
    pub allow_integrated: bool,
    /// `--version` output of the build binary; None = it failed to run.
    pub build_version_output: Option<String>,
    /// Model + auxiliary files that must exist, with existence pre-checked
    /// (the context builder does the I/O; checks stay pure).
    pub missing_files: Vec<String>,
    pub resolved: Vec<ResolvedDevice>,
    /// Keys that failed to resolve, with the error text.
    pub unresolved: Vec<(String, String)>,
    pub estimate: Option<VramEstimate>,
    pub commit: SystemCommit,
    /// Bytes already promised by servers that are still loading: their VRAM
    /// is reserved but not yet fully charged to commit or resident in RAM.
    /// Without this, launching two large servers back-to-back sees the
    /// second one's headroom as if the first had allocated nothing — the
    /// exact hole that thrashed the machine on 2026-08-01.
    pub inflight_reserved_bytes: u64,
    /// The exact `HIP_VISIBLE_DEVICES` value the launcher will set.
    pub visibility_env: String,
    /// Aliases of currently-running servers.
    pub running_aliases: Vec<String>,
    /// Who holds the requested port. None = free. A holder with a
    /// `profile_id` is one of Llama FIDIM's own servers (launch takes the port
    /// over by stopping it first); anything else is foreign and blocks.
    pub port_holder: Option<PortHolder>,
    /// Driver/SDK now vs the profile baseline's recorded versions.
    pub driver_now: Option<String>,
    pub driver_baseline: Option<String>,
    pub sdk_now: Option<String>,
    pub sdk_baseline: Option<String>,
    /// PCI Express Link State Power Management, AC index of the active power
    /// plan: 0 = Off, 1 = Moderate, 2 = Maximum. None = could not read.
    pub pcie_aspm: Option<u32>,
    /// From the model header; None = unreadable.
    pub model_facts: Option<ModelFacts>,
    /// Set for diffusion profiles.
    pub diffusion: Option<DiffusionPreflight>,
    /// Live runs on this launch's cards, except one on this profile's port
    /// (launch takes that one over).
    pub co_resident: Vec<CoResident>,
}

pub fn run_all(ctx: &LaunchContext) -> Vec<CheckResult> {
    let mut out = vec![
        check_build(ctx),
        check_files(ctx),
        check_resolution(ctx),
        check_discrete(ctx),
        check_visibility(ctx),
        check_vram(ctx),
        check_commit(ctx),
        check_port(ctx),
        check_alias(ctx),
        check_display(ctx),
        check_versions(ctx),
        check_pcie_aspm(ctx),
    ];
    // The engine checks only appear where a diffusion model, profile or run
    // is involved, so every llama-server-only launch keeps its 12 results.
    let diffusion = ctx.profile.engine.is_diffusion();
    if diffusion || ctx.model_facts.as_ref().is_some_and(|m| m.is_diffusion) {
        out.push(check_engine_model(ctx));
    }
    if !ctx.co_resident.is_empty() && (diffusion || ctx.co_resident.iter().any(|c| c.engine.is_diffusion())) {
        out.push(check_card_sharing(ctx));
    }
    if diffusion {
        out.push(check_diffusion_context(ctx));
    }
    out
}

pub fn any_block(results: &[CheckResult]) -> bool {
    results.iter().any(|r| r.blocks())
}

// ---------------------------------------------------------------- checks ----

/// For a diffusion profile: what is missing for the helper and runner to
/// start, first match wins. None = the llama-server checks below decide.
fn diffusion_build_block(ctx: &LaunchContext) -> Option<String> {
    let Some(d) = &ctx.diffusion else {
        return Some("the launch path did not describe the diffusion runner; nothing to check it against".into());
    };
    if d.helper_exe.is_none() {
        // Most people install from the release zip and have no install script.
        let dir = d.helper_dir.as_ref().map(|p| p.display().to_string());
        return Some(format!(
            "{exe} not found in {}; it ships beside llama-fidim.exe and fidim.exe in the release zip, so keep \
             the three in one folder (or run scripts\\install.ps1 from a source checkout)",
            dir.as_deref().unwrap_or("the running Llama FIDIM's folder"),
            exe = crate::supervise::DG_HELPER_EXE
        ));
    }
    if !d.runner_present {
        let tag = ctx
            .profile
            .build
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| ctx.profile.build.path.display().to_string());
        return Some(format!(
            "build {tag} has no {}; install one with `fidim update --channel unsloth --install` (or the Updates tab)",
            crate::discovery::RUNNER_EXE
        ));
    }
    if d.vulkan_backend {
        return Some(
            "build has ggml-vulkan.dll: Vulkan devices are not hidden by HIP_VISIBLE_DEVICES, so the runner \
             would see 2 devices and abort every prompt"
                .into(),
        );
    }
    if let (Some(e1), Some(e2)) = (&d.req_prefix_error, &d.req_fallback_error) {
        return Some(format!(
            "the runner cannot open request files under {} ({e1}) or under the %TEMP% fallback {} ({e2}); \
             fidim-dg.exe would exit 7 at launch. Point FIDIM_HOME or %TEMP% at a short ASCII path",
            d.req_prefix.display(),
            d.req_fallback.display()
        ));
    }
    None
}

fn check_build(ctx: &LaunchContext) -> CheckResult {
    if ctx.profile.engine.is_diffusion() {
        if let Some(msg) = diffusion_build_block(ctx) {
            return CheckResult {
                id: "build-runs",
                spec_number: 1,
                title: "Build binary exists and runs",
                outcome: Outcome::Block(msg),
            };
        }
    }
    // For diffusion too: the runner shares llama-server's ROCm DLLs, so a
    // llama-server that runs proves they load.
    let outcome = match &ctx.build_version_output {
        // The binary runs. Also catch a build directory that was rebuilt or
        // replaced underneath the profile: the stored version is what the
        // baseline was measured against, so drift makes those numbers moot.
        Some(text) => {
            let actual = crate::discovery::parse_version_output(text).map(|(v, _)| v);
            match (&ctx.profile.build.version, actual) {
                (Some(want), Some(have)) if want != &have => Outcome::Note(format!(
                    "profile records build {want} but the binary reports {have} — the build \
                     directory changed underneath the profile; re-bench before trusting its baseline"
                )),
                _ => Outcome::Pass,
            }
        }
        None => Outcome::Block(format!(
            "build binary {} did not run — reinstall or rescan builds",
            ctx.profile.build.path.display()
        )),
    };
    // The helper falls back to %TEMP% when the runs-dir prefix is unusable
    // for the runner: say where the conversation files will live.
    let outcome = match (&ctx.diffusion, outcome) {
        (Some(d), Outcome::Pass) if ctx.profile.engine.is_diffusion() && d.req_prefix_error.is_some() => {
            Outcome::Note(format!(
                "request files will live under %TEMP% ({}) because {} is unusable for the runner ({})",
                d.req_fallback.display(),
                d.req_prefix.display(),
                d.req_prefix_error.as_deref().unwrap_or("")
            ))
        }
        (_, o) => o,
    };
    CheckResult { id: "build-runs", spec_number: 1, title: "Build binary exists and runs", outcome }
}

fn check_files(ctx: &LaunchContext) -> CheckResult {
    let outcome = if ctx.missing_files.is_empty() {
        Outcome::Pass
    } else {
        Outcome::Block(format!("missing file(s): {}", ctx.missing_files.join(", ")))
    };
    CheckResult {
        id: "files-exist",
        spec_number: 2,
        title: "Model and paired mmproj / draft files exist",
        outcome,
    }
}

fn check_resolution(ctx: &LaunchContext) -> CheckResult {
    let outcome = if !ctx.unresolved.is_empty() {
        Outcome::Block(
            ctx.unresolved
                .iter()
                .map(|(k, e)| format!("{k}: {e}"))
                .collect::<Vec<_>>()
                .join("; "),
        )
    } else if ctx.resolved.iter().any(|r| r.rebound) {
        let rebound: Vec<&str> = ctx
            .resolved
            .iter()
            .filter(|r| r.rebound)
            .map(|r| r.profile_key.as_str())
            .collect();
        Outcome::Warn(format!(
            "device renumbering detected — key(s) {} re-bound by hardware model; the profile will be updated",
            rebound.join(", ")
        ))
    } else {
        Outcome::Pass
    };
    CheckResult {
        id: "devices-resolve",
        spec_number: 3,
        title: "Every target device resolves from its stable key",
        outcome,
    }
}

fn check_discrete(ctx: &LaunchContext) -> CheckResult {
    let igpus: Vec<String> = ctx
        .resolved
        .iter()
        .filter(|r| r.device.integrated)
        .map(|r| format!("{} ({})", r.device.name, r.profile_key))
        .collect();
    let outcome = if igpus.is_empty() {
        Outcome::Pass
    } else if ctx.profile.engine.is_diffusion() {
        Outcome::Block(format!(
            "target resolves to integrated graphics: {} — the diffusion bundle targets gfx1200/gfx1201 only; \
             pick a discrete card",
            igpus.join(", ")
        ))
    } else if ctx.allow_integrated {
        Outcome::Warn(format!(
            "target resolves to integrated graphics: {} — allowed by Settings; expect a fraction of a discrete card's throughput",
            igpus.join(", ")
        ))
    } else {
        Outcome::Block(format!(
            "target resolves to integrated graphics: {} — an iGPU runs inference at a fraction of a discrete card's speed with no error. \
             Turn on 'allow integrated graphics' in Settings if that is what you want",
            igpus.join(", ")
        ))
    };
    CheckResult {
        id: "discrete-only",
        spec_number: 4,
        title: "Every resolved device is a discrete compute GPU",
        outcome,
    }
}

/// v1.1 check 5 (R-13): the composed visibility env must contain exactly the
/// resolved devices' indices — nothing more (an unpinned split would shear
/// layers onto the iGPU), nothing less.
fn check_visibility(ctx: &LaunchContext) -> CheckResult {
    let expected: Vec<String> =
        ctx.resolved.iter().map(|r| r.device.hip_index.to_string()).collect();
    let expected_str = expected.join(",");
    let outcome = if ctx.profile.engine.is_diffusion() && ctx.resolved.len() != 1 {
        Outcome::Block(format!(
            "the diffusion runner needs exactly one visible device, but the profile resolves to {}: with more \
             than one it takes its unified path, which aborts every prompt",
            ctx.resolved.len()
        ))
    } else if ctx.resolved.is_empty() {
        Outcome::Block("no resolved devices to pin visibility to".into())
    } else if ctx.visibility_env == expected_str {
        Outcome::Pass
    } else {
        Outcome::Block(format!(
            "HIP_VISIBLE_DEVICES would be \"{}\" but the resolved device set requires exactly \"{}\" — \
             an unpinned or mismatched visibility lets llama-server see devices the profile never chose",
            ctx.visibility_env, expected_str
        ))
    };
    CheckResult {
        id: "visibility-pinned",
        spec_number: 5,
        title: "Composed device visibility matches the resolved set exactly",
        outcome,
    }
}

fn check_vram(ctx: &LaunchContext) -> CheckResult {
    let Some(est) = &ctx.estimate else {
        return CheckResult {
            id: "vram-fits",
            spec_number: 6,
            title: "Estimated VRAM fits in free VRAM per device",
            outcome: Outcome::Warn("no VRAM estimate available (model header unreadable)".into()),
        };
    };
    // A diffusion profile at auto (ctx_total = 0) lets the runner size its
    // context budget to the VRAM it sees at load, so the load fits by
    // construction. The estimate's kv term there is the working set of a
    // full-budget prompt, which the runner keeps after the request but never
    // budgets for: report it, and Block only when the load itself does not fit.
    let auto_sized = ctx.profile.engine.is_diffusion() && ctx.profile.runtime.ctx_total == 0;
    let mut worst: Option<(f64, f64, String)> = None;
    for (d, r) in est.per_device.iter().zip(&ctx.resolved) {
        let free_bytes = r.device.free_mib as f64 * 1024.0 * 1024.0;
        if free_bytes <= 0.0 {
            continue;
        }
        let ratio = d.total_bytes as f64 / free_bytes;
        let load_ratio = d.total_bytes.saturating_sub(d.kv_bytes) as f64 / free_bytes;
        let msg = format!(
            "{}: estimated {} vs {:.2} GiB free ({:.0}%)",
            r.device.name,
            d.breakdown(),
            free_bytes / (1024.0 * 1024.0 * 1024.0),
            ratio * 100.0
        );
        if worst.as_ref().map(|(w, _, _)| ratio > *w).unwrap_or(true) {
            worst = Some((ratio, load_ratio, msg));
        }
    }
    let outcome = match worst {
        Some((_, load, msg)) if auto_sized && load > 1.0 => Outcome::Block(format!(
            "{msg} — the weights and buffers alone exceed free VRAM, so no context budget fits and the runner \
             fails to load"
        )),
        Some((r, _, msg)) if auto_sized && r > 1.0 => Outcome::Warn(format!(
            "{msg} — the load fits (the runner sizes MAXTOK to the card at load), but the working set of a \
             full-budget prompt would not; set an explicit context budget (MAXTOK) below the auto prediction so \
             the largest prompt stays on the card"
        )),
        Some((r, _, msg)) if auto_sized && r > 0.90 => Outcome::Note(format!(
            "{msg} — the runner sizes MAXTOK to the card at load, so this fits; the figure is the working set \
             of a full-budget prompt, kept after the request. An explicit context budget (MAXTOK) caps it"
        )),
        Some((r, _, msg)) if r > 1.0 => Outcome::Block(format!(
            "{msg} — spill costs ~3.4x prefill and presents as \"the model got slow\", not an error"
        )),
        Some((r, _, msg)) if r > 0.90 => Outcome::Warn(format!(
            "{msg} — over 90% leaves no headroom for desktop compositing if a display attaches to this card"
        )),
        Some(_) => Outcome::Pass,
        None => Outcome::Warn("free VRAM unknown for target devices".into()),
    };
    CheckResult {
        id: "vram-fits",
        spec_number: 6,
        title: "Estimated VRAM fits in free VRAM per device",
        outcome,
    }
}

const GIB_F: f64 = 1024.0 * 1024.0 * 1024.0;

fn gib(b: u64) -> f64 {
    b as f64 / GIB_F
}

/// Everything this launch will add: its own estimate plus anything already
/// promised by servers still loading.
fn projected_add(ctx: &LaunchContext) -> u64 {
    ctx.estimate.as_ref().map(|e| e.total_bytes).unwrap_or(0) + ctx.inflight_reserved_bytes
}

/// Commit headroom, and the pagefile growth that may happen near the ceiling.
///
/// Commit is the ceiling that binds. A loaded server charges substantial
/// commit for its GPU allocation (measured 2026-08-01: llama-server holding
/// 19.8 GiB of VRAM reported 17.7 GiB of private bytes). The exact VRAM ->
/// commit ratio is NOT established — two runs on a machine in active use
/// disagreed (+8.4 and +24 GiB system-wide) — so this check gates on the
/// system total rather than on any assumed per-byte ratio.
///
/// What IS established is that the charge is a reservation, not a transfer:
/// the same server's resident working set was 11.6 GiB against 19.8 GiB of
/// VRAM, and the machine stayed fully responsive at 0.6 GiB of free RAM
/// while commit sat at ~52% of the limit. Hence there is deliberately NO
/// physical-RAM gate anywhere in this pipeline: a RAM gate would have
/// blocked that healthy state, and models much larger than RAM load fine
/// when VRAM holds them and commit can promise them.
///
/// Two hazards as projected commit rises:
///   1. Above the *allocated pagefile*, Windows must extend it. Extension is
///      synchronous and disk-heavy and can stall the machine. HYPOTHESIS for
///      the 2026-08-01 freeze — the limit did move 88.9 -> 114.9 GiB during
///      that session, but the growth was never tied to the launch by
///      timestamp. Treated as a risk to warn about, not a proven mechanism.
///   2. Above ~90% of the limit, allocations fail or WDDM evicts VRAM — the
///      documented 88.9 -> 1.8 tok/s collapse. This one is observed.
fn check_commit(ctx: &LaunchContext) -> CheckResult {
    let add = projected_add(ctx);
    let projected = ctx.commit.charge_bytes + add;
    let limit = ctx.commit.limit_bytes.max(1);
    let frac = projected as f64 / limit as f64;
    let inflight_note = if ctx.inflight_reserved_bytes > 0 {
        format!(
            " (includes {:.1} GiB reserved by a server still loading)",
            gib(ctx.inflight_reserved_bytes)
        )
    } else {
        String::new()
    };
    let head = format!(
        "projected commit {:.1} GiB of {:.1} GiB limit ({:.0}%){inflight_note}",
        gib(projected),
        gib(limit),
        frac * 100.0
    );

    // Growth threshold: how close to the ceiling before Windows extends.
    // Windows starts extending well before the limit is reached, so treat
    // the top ~12% of a growable pagefile as the danger band.
    let pagefile = ctx.commit.pagefile_allocated_bytes;
    let growth_risk = pagefile > 0
        && ctx.commit.pagefile_can_grow
        && frac > 0.88
        && frac <= 0.90;

    let outcome = if frac > 0.90 {
        if ctx.commit.pagefile_can_grow {
            Outcome::Block(format!(
                "{head} — past ~90% Windows evicts GPU allocations, which collapses decode roughly 50x with \
                 nothing in any log (observed). Covering this may also require extending the {:.0} GiB \
                 pagefile, and extension is synchronous and disk-heavy — a suspected cause of a multi-minute \
                 machine stall on 2026-08-01, though not proven. Remedies: stop another server, or set an \
                 explicit fixed pagefile size so Windows never extends mid-launch (System Properties > \
                 Performance > Virtual memory; needs elevation — this tool will not change it for you)",
                gib(pagefile)
            ))
        } else {
            Outcome::Block(format!(
                "{head} — the pagefile is a fixed {:.0} GiB and cannot grow, so this allocation fails \
                 outright rather than slowing down. Remedies: stop another server, use a smaller \
                 quantisation, or raise the pagefile size (needs elevation)",
                gib(pagefile)
            ))
        }
    } else if growth_risk {
        Outcome::Warn(format!(
            "{head} — close enough to the ceiling that Windows may need to extend the {:.0} GiB pagefile \
             during load. Extension is synchronous and can stall the machine while it writes; a fixed \
             pagefile size avoids the risk entirely",
            gib(pagefile)
        ))
    } else {
        Outcome::Pass
    };
    CheckResult {
        id: "commit-headroom",
        spec_number: 7,
        title: "Projected system commit stays under 90% of the limit",
        outcome,
    }
}

/// Check 8. Serving every model on one well-known port is the normal way
/// to use this box (clients point at :1234 regardless of which model is
/// up), so a Llama FIDIM server already on the port is a Warn and launch
/// replaces it. Only a process Llama FIDIM does not know blocks.
fn check_port(ctx: &LaunchContext) -> CheckResult {
    let port = ctx.profile.server.port;
    let pid = |h: &PortHolder| h.pid.map(|p| format!(" (pid {p})")).unwrap_or_default();
    let outcome = match &ctx.port_holder {
        None => Outcome::Pass,
        Some(h) if h.profile_id.is_some() => Outcome::Warn(format!(
            "port {port} is serving Llama FIDIM profile `{}`{} — launching stops that server first and takes the port over",
            h.profile_id.as_deref().unwrap_or("?"),
            pid(h)
        )),
        Some(h) => Outcome::Block(format!(
            "port {port} is in use by {}{} — not a Llama FIDIM server; stop it or pick another port",
            h.process_name.as_deref().unwrap_or("an unknown process"),
            pid(h)
        )),
    };
    CheckResult { id: "port-free", spec_number: 8, title: "Requested port is free or held by a Llama FIDIM server", outcome }
}

fn check_alias(ctx: &LaunchContext) -> CheckResult {
    let alias = &ctx.profile.server.alias;
    let outcome = if ctx.running_aliases.iter().any(|a| a == alias) {
        Outcome::Warn(format!(
            "alias {alias:?} is already served by a running server — clients routing by model name \
             will be ambiguous"
        ))
    } else {
        Outcome::Pass
    };
    CheckResult {
        id: "alias-unique",
        spec_number: 9,
        title: "Alias is unique across running servers",
        outcome,
    }
}

fn check_display(ctx: &LaunchContext) -> CheckResult {
    let driving: Vec<String> = ctx
        .resolved
        .iter()
        .filter_map(|r| {
            r.device.display.as_ref().map(|m| {
                format!("{} ({}x{}@{})", r.device.name, m.width, m.height, m.refresh_hz)
            })
        })
        .collect();
    let outcome = if driving.is_empty() {
        Outcome::Pass
    } else {
        Outcome::Warn(format!(
            "target device(s) driving a display: {} — desktop compositing has consumed 2.68 GB and \
             preempted compute in the past; one display per compute card is deliberate on this box, so this is the accepted cost",
            driving.join(", ")
        ))
    };
    CheckResult {
        id: "no-display",
        spec_number: 10,
        title: "No target device is driving a display",
        outcome,
    }
}

/// Check 12: PCI Express Link State Power Management must be Off. Measured
/// 2026-09-05 on this box: with it at Moderate, an idle card whose display
/// powers off has its whole allocation evicted to system RAM within 20 s
/// (25.6 GB, 4-6 s to re-page); at Off the model stays resident with no
/// traffic at all. The keep-alive helper masks this, so it is a Warn, but
/// the setting is the actual fix. Diffusion runs get no keep-alive (every
/// request is a whole denoise block), so their text omits that remedy.
fn check_pcie_aspm(ctx: &LaunchContext) -> CheckResult {
    let keepalive = if ctx.profile.engine.is_diffusion() {
        ""
    } else {
        ", or enable the keep-alive interval on this profile as a workaround"
    };
    let outcome = match ctx.pcie_aspm {
        Some(0) => Outcome::Pass,
        Some(n) => Outcome::Warn(format!(
            "PCI Express Link State Power Management is {} — an idle card with its display off will be \
             evicted from VRAM into system RAM; set it to Off in the power plan (measured 2026-09-05){keepalive}",
            match n { 1 => "Moderate".to_string(), 2 => "Maximum power savings".to_string(), o => format!("index {o}") }
        )),
        None => Outcome::Note("PCI Express Link State Power Management could not be read (powercfg)".into()),
    };
    CheckResult {
        id: "pcie-aspm-off",
        spec_number: 12,
        title: "PCIe link-state power management is Off",
        outcome,
    }
}

fn check_versions(ctx: &LaunchContext) -> CheckResult {
    let mut changes = Vec::new();
    if let (Some(now), Some(then)) = (&ctx.driver_now, &ctx.driver_baseline) {
        if now != then {
            changes.push(format!("driver {then} -> {now}"));
        }
    }
    if let (Some(now), Some(then)) = (&ctx.sdk_now, &ctx.sdk_baseline) {
        if now != then {
            changes.push(format!("SDK {then} -> {now}"));
        }
    }
    let outcome = if changes.is_empty() {
        Outcome::Pass
    } else {
        Outcome::Note(format!(
            "environment changed since baselines were measured ({}) — stored numbers are unverified",
            changes.join(", ")
        ))
    };
    CheckResult {
        id: "versions-match",
        spec_number: 11,
        title: "Driver and SDK match the profile's baselines",
        outcome,
    }
}

/// Check 13: the profile's engine can load the model. The editor switches
/// engines when a model is picked, so a mismatch means a hand-edited or
/// re-pointed profile.
fn check_engine_model(ctx: &LaunchContext) -> CheckResult {
    let arch = ctx
        .model_facts
        .as_ref()
        .and_then(|m| m.architecture.clone())
        .unwrap_or_else(|| "an unknown architecture".into());
    let outcome = match (&ctx.model_facts, ctx.profile.engine.is_diffusion()) {
        (Some(m), false) if m.is_diffusion => Outcome::Block(format!(
            "llama-server cannot load {arch}; pick this model again in the editor so it switches to the \
             diffusion engine"
        )),
        (Some(m), true) if !m.is_diffusion => {
            Outcome::Block("the diffusion runner exits with \"not a diffusion model\"".into())
        }
        (Some(m), true) if !m.runner_supported => Outcome::Block(format!(
            "no runner for architecture {arch}; only diffusion-gemma is supported"
        )),
        _ => Outcome::Pass,
    };
    CheckResult { id: "engine-matches-model", spec_number: 13, title: "Engine can load this model", outcome }
}

/// Check 14: a diffusion run and another model on one card. The runner sizes
/// its context from the VRAM WDDM reports free, which does not include other
/// processes' allocations, and its prompt-KV store is allocated per request
/// after load, so pre-flight's own free-VRAM figure understates the clash.
fn check_card_sharing(ctx: &LaunchContext) -> CheckResult {
    let mut others: Vec<&str> = Vec::new();
    let mut diffusion_runs: Vec<&str> = Vec::new();
    for c in &ctx.co_resident {
        if !others.contains(&c.profile_id.as_str()) {
            others.push(&c.profile_id);
        }
        if c.engine.is_diffusion() && !diffusion_runs.contains(&c.profile_id.as_str()) {
            diffusion_runs.push(&c.profile_id);
        }
    }
    let outcome = if ctx.profile.engine.is_diffusion() && ctx.profile.runtime.ctx_total == 0 {
        Outcome::Warn(format!(
            "card also hosts {}; the runner auto-sizes its context to the VRAM it believes is free, and WDDM \
             hides other processes' allocations, so it will oversubscribe. Pick an empty card",
            others.join(", ")
        ))
    } else if ctx.profile.engine.is_diffusion() {
        let store = ctx
            .diffusion
            .as_ref()
            .and_then(|d| d.sizing.as_ref())
            .map(|s| {
                format!(
                    " (up to {:.1} GiB)",
                    (s.pkv_bytes_per_token * s.max_prompt_tokens as u64) as f64 / GIB_F
                )
            })
            .unwrap_or_default();
        Outcome::Warn(format!(
            "card also hosts {}; this profile allocates its prompt-KV store{store} per request, after load; \
             sharing can push either model into shared memory",
            others.join(", ")
        ))
    } else {
        Outcome::Warn(format!(
            "{} on this card sized its context to the card at load and allocates its prompt-KV store per \
             request; sharing can push either model into shared memory",
            diffusion_runs.join(", ")
        ))
    };
    CheckResult {
        id: "diffusion-card-sharing",
        spec_number: 14,
        title: "Card is not shared with a diffusion run",
        outcome,
    }
}

/// Check 15: what context budget (MAXTOK) the runner will end up with.
fn check_diffusion_context(ctx: &LaunchContext) -> CheckResult {
    let sizing = ctx.diffusion.as_ref().and_then(|d| d.sizing.as_ref());
    let outcome = match sizing {
        None => Outcome::Note(
            "context budget not predicted (model header unreadable or no device resolved); the runner decides \
             at load"
                .into(),
        ),
        Some(s) => {
            let ctx_total = ctx.profile.runtime.ctx_total;
            let mut warns = Vec::new();
            match s.predicted_auto_maxtok {
                Some(p) if ctx_total > p as u64 => warns.push(format!(
                    "explicit budget {ctx_total} is above what fits (auto MAXTOK predicted ≈ {p}); the runner \
                     will degrade it at load"
                )),
                None if ctx_total > 0 => warns.push(format!(
                    "explicit budget {ctx_total}: no context fits the VRAM left after the weights; the runner \
                     will degrade it at load, down to its floor, or fail to load"
                )),
                None => warns.push(format!(
                    "no context candidate fits the VRAM left after the weights; the runner falls back to its \
                     floor ({} tokens) or fails to load",
                    s.maxtok_used
                )),
                _ => {}
            }
            if !s.full_offload {
                warns.push(
                    "NGL < block_count+1: partial offload is slow and the context is sized against system RAM"
                        .into(),
                );
            }
            if !warns.is_empty() {
                Outcome::Warn(warns.join("; "))
            } else if ctx_total > 0 {
                Outcome::Note(format!(
                    "explicit MAXTOK {ctx_total} (largest prompt ≈ {} tokens); auto would pick ≈ {}",
                    s.max_prompt_tokens,
                    s.predicted_auto_maxtok.map(|p| p.to_string()).unwrap_or_else(|| "?".into())
                ))
            } else {
                let p = s.predicted_auto_maxtok.unwrap_or(s.maxtok_used);
                Outcome::Note(format!(
                    "auto MAXTOK predicted ≈ {p} (largest prompt ≈ {} tokens); the runner decides at load",
                    p.saturating_sub(s.canvas)
                ))
            }
        }
    };
    CheckResult { id: "diffusion-context", spec_number: 15, title: "Diffusion context budget", outcome }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{DisplayMode, OsAdapter};
    use crate::devices::{correlate, parse_list_devices};
    use crate::platform::SystemCommit;
    use crate::profile::Profile;

    const REAL_LIST: &str = "Available devices:\n  ROCm0: AMD Radeon AI PRO R9700 (32624 MiB, 32472 MiB free)\n  ROCm1: AMD Radeon(TM) Graphics (12381 MiB, 12099 MiB free)\n  ROCm2: AMD Radeon AI PRO R9700 (32624 MiB, 32472 MiB free)\n";

    fn devices() -> Vec<crate::devices::Device> {
        let adapters = vec![
            OsAdapter {
                name: "AMD Radeon(TM) Graphics".into(),
                pnp_device_id: r"PCI\VEN_1002&DEV_13C0&SUBSYS_7D781462&REV_CB\4&1&0&0041".into(),
                driver_version: "32.0.21045.1000".into(),
                bus_number: Some(19),
                display: None,
                luid_low: Some(0x1DCEC),
            },
            OsAdapter {
                name: "AMD Radeon AI PRO R9700".into(),
                pnp_device_id: r"PCI\VEN_1002&DEV_7551&SUBSYS_54131849&REV_C0\6&2&0&09".into(),
                driver_version: "32.0.31035.1003".into(),
                bus_number: Some(3),
                display: Some(DisplayMode { width: 2560, height: 1440, refresh_hz: 144 }),
                luid_low: Some(0x1621C),
            },
            OsAdapter {
                name: "AMD Radeon AI PRO R9700".into(),
                pnp_device_id: r"PCI\VEN_1002&DEV_7551&SUBSYS_54131849&REV_C0\8&3&0&11".into(),
                driver_version: "32.0.31035.1003".into(),
                bus_number: Some(8),
                display: None,
                luid_low: Some(0x1B592),
            },
        ];
        correlate(
            &parse_list_devices(REAL_LIST).unwrap(),
            &adapters,
            &[],
            &["Radeon(TM) Graphics".to_string()],
        )
    }

    fn profile() -> Profile {
        serde_json::from_value(serde_json::json!({
            "schema": 1, "id": "t", "name": "t",
            "build": { "path": "C:/b" }, "model": { "path": "E:/m.gguf" },
            "devices": [ { "key": "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08" } ],
            "server": { "port": 9701, "alias": "t" },
            "runtime": { "ctx_total": 8192, "slots": 1 }
        }))
        .unwrap()
    }

    fn resolved_for(idx: u32, fraction: f64) -> ResolvedDevice {
        let d = devices().into_iter().find(|d| d.hip_index == idx).unwrap();
        ResolvedDevice {
            profile_key: d.stable_key.clone(),
            device: d,
            fraction,
            rebound: false,
        }
    }

    fn healthy_ctx() -> LaunchContext {
        LaunchContext {
            allow_integrated: false,
            profile: profile(),
            build_version_output: Some("version: 9817 (5397c3619)".into()),
            missing_files: vec![],
            resolved: vec![resolved_for(2, 1.0)],
            unresolved: vec![],
            estimate: None,
            commit: SystemCommit {
                limit_bytes: 100 * 1024 * 1024 * 1024,
                charge_bytes: 30 * 1024 * 1024 * 1024,
                physical_total_bytes: 64 * 1024 * 1024 * 1024,
                physical_available_bytes: 48 * 1024 * 1024 * 1024,
                pagefile_allocated_bytes: 36 * 1024 * 1024 * 1024,
                pagefile_can_grow: true,
            },
            inflight_reserved_bytes: 0,
            visibility_env: "2".into(),
            running_aliases: vec![],
            port_holder: None,
            driver_now: Some("32.0.31035.1003".into()),
            driver_baseline: None,
            sdk_now: None,
            sdk_baseline: None,
            pcie_aspm: Some(0),
            model_facts: None,
            diffusion: None,
            co_resident: vec![],
        }
    }

    #[test]
    fn port_held_by_our_server_warns_and_foreign_blocks() {
        let mut ctx = healthy_ctx();
        ctx.port_holder = Some(PortHolder { pid: Some(7), process_name: Some("llama-server.exe".into()), profile_id: Some("daily-driver".into()) });
        let r = run_all(&ctx);
        let port = r.iter().find(|c| c.id == "port-free").unwrap();
        assert!(matches!(port.outcome, Outcome::Warn(_)), "{port:?}");
        assert!(!any_block(&r));
        ctx.port_holder = Some(PortHolder { pid: Some(8), process_name: Some("LM Studio.exe".into()), profile_id: None });
        let r = run_all(&ctx);
        let port = r.iter().find(|c| c.id == "port-free").unwrap();
        assert!(matches!(port.outcome, Outcome::Block(_)), "{port:?}");
        assert!(any_block(&r));
    }

    #[test]
    fn healthy_context_passes_everything() {
        let results = run_all(&healthy_ctx());
        assert_eq!(results.len(), 12);
        assert!(!any_block(&results), "{results:?}");
    }

    fn estimate_of(bytes: u64) -> crate::estimate::VramEstimate {
        crate::estimate::VramEstimate {
            per_device: vec![],
            total_bytes: bytes,
            assumptions: vec![],
        }
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    /// The machine as measured 2026-08-01 while idle.
    fn measured_idle_commit() -> SystemCommit {
        SystemCommit {
            limit_bytes: 114 * GIB + GIB / 2,   // 114.9 GiB
            charge_bytes: 31 * GIB + GIB / 5,   // 31.2 GiB
            physical_total_bytes: 30 * GIB + GIB * 9 / 10, // 30.9 GiB
            physical_available_bytes: 11 * GIB, // 11.0 GiB
            pagefile_allocated_bytes: 84 * GIB,
            pagefile_can_grow: true,
        }
    }

    /// A model far larger than physical RAM must launch. Measured: a 20 GB
    /// model charges +19.6 GiB commit but costs only ~8 GiB of RAM, and
    /// other loaders do this routinely on this box. An earlier revision
    /// gated on RAM 1:1 and would have blocked this — it was wrong.
    #[test]
    fn model_larger_than_physical_ram_is_allowed() {
        let mut ctx = healthy_ctx();
        ctx.commit = measured_idle_commit(); // 30.9 GiB RAM, 11.0 GiB free
        ctx.estimate = Some(estimate_of(34 * GIB)); // bigger than ALL of RAM
        let results = run_all(&ctx);
        assert!(
            !any_block(&results),
            "a model larger than RAM must still launch — VRAM holds it and commit promises it: {results:?}"
        );
    }

    /// Commit near the ceiling is the real hazard, and the message must name
    /// the pagefile-extension stall rather than the old eviction-only story.
    #[test]
    fn commit_over_limit_blocks_and_names_pagefile_growth() {
        let mut ctx = healthy_ctx();
        ctx.commit = SystemCommit { charge_bytes: 95 * GIB, ..measured_idle_commit() };
        ctx.estimate = Some(estimate_of(15 * GIB)); // 110 of 114.9 = 96%
        let results = run_all(&ctx);
        let commit = results.iter().find(|r| r.id == "commit-headroom").unwrap();
        assert!(commit.blocks(), "{commit:?}");
        if let Outcome::Block(msg) = &commit.outcome {
            assert!(msg.contains("extend"), "must explain the growth stall: {msg}");
            assert!(msg.contains("fixed pagefile size"), "must offer the remedy: {msg}");
        }
    }

    /// A fixed-size pagefile cannot stall on growth, so the explanation and
    /// remedy differ — it fails outright instead.
    #[test]
    fn fixed_pagefile_reports_hard_failure_not_a_stall() {
        let mut ctx = healthy_ctx();
        ctx.commit = SystemCommit {
            charge_bytes: 95 * GIB,
            pagefile_can_grow: false,
            ..measured_idle_commit()
        };
        ctx.estimate = Some(estimate_of(15 * GIB));
        let commit = run_all(&ctx).into_iter().find(|r| r.id == "commit-headroom").unwrap();
        assert!(commit.blocks());
        if let Outcome::Block(msg) = &commit.outcome {
            assert!(msg.contains("cannot grow"), "{msg}");
            assert!(msg.contains("fails outright"), "{msg}");
        }
    }

    /// The back-to-back launch hole: a spawned-but-not-ready server has
    /// claimed commit it has not finished charging, so a second launch
    /// evaluated in that window sees phantom headroom.
    #[test]
    fn inflight_reservation_counts_against_a_concurrent_launch() {
        let mut ctx = healthy_ctx();
        // Charge reflects only what the loading server has committed so far.
        ctx.commit = SystemCommit { charge_bytes: 80 * GIB, ..measured_idle_commit() };
        ctx.estimate = Some(estimate_of(13 * GIB)); // 93 of 114.9 = 81%, fine

        let commit = run_all(&ctx).into_iter().find(|r| r.id == "commit-headroom").unwrap();
        assert!(!commit.blocks(), "81% is genuinely fine: {commit:?}");

        // But a server still loading has 20 GiB more to charge: 113 of 114.9.
        ctx.inflight_reserved_bytes = 20 * GIB;
        let commit = run_all(&ctx).into_iter().find(|r| r.id == "commit-headroom").unwrap();
        assert!(commit.blocks(), "must count the loading server: {commit:?}");
        if let Outcome::Block(msg) = &commit.outcome {
            assert!(msg.contains("still loading"), "must name the in-flight server: {msg}");
        }
    }

    /// The everyday single-server launch on the real machine must not regress.
    #[test]
    fn measured_everyday_launch_passes() {
        let mut ctx = healthy_ctx();
        ctx.commit = measured_idle_commit();
        ctx.estimate = Some(estimate_of(20 * GIB)); // the worker pool
        let results = run_all(&ctx);
        assert!(!any_block(&results), "the everyday launch must not regress: {results:?}");
    }

    #[test]
    fn igpu_bind_is_blocked() {
        let mut ctx = healthy_ctx();
        ctx.resolved = vec![resolved_for(1, 1.0)];
        ctx.visibility_env = "1".into();
        let results = run_all(&ctx);
        let discrete = results.iter().find(|r| r.id == "discrete-only").unwrap();
        assert!(discrete.blocks(), "{discrete:?}");
    }

    #[test]
    fn visibility_mismatch_is_blocked() {
        let mut ctx = healthy_ctx();
        // Split profile resolved to 0 and 2, but the env would expose everything.
        ctx.resolved = vec![resolved_for(0, 0.5), resolved_for(2, 0.5)];
        ctx.visibility_env = "0,1,2".into();
        let results = run_all(&ctx);
        let vis = results.iter().find(|r| r.id == "visibility-pinned").unwrap();
        assert!(vis.blocks(), "{vis:?}");
        // Correctly pinned passes.
        ctx.visibility_env = "0,2".into();
        let results = run_all(&ctx);
        assert!(!results.iter().find(|r| r.id == "visibility-pinned").unwrap().blocks());
    }

    #[test]
    fn commit_exhaustion_is_blocked_with_pagefile_remedy() {
        let mut ctx = healthy_ctx();
        // 30 GiB estimate on a machine with 95% commit already used.
        let bytes = 30u64 * 1024 * 1024 * 1024;
        ctx.commit = SystemCommit {
            limit_bytes: 100 * 1024 * 1024 * 1024,
            charge_bytes: 65 * 1024 * 1024 * 1024,
            ..measured_idle_commit()
        };
        ctx.estimate = Some(crate::estimate::VramEstimate {
            per_device: vec![],
            total_bytes: bytes,
            assumptions: vec![],
        });
        let results = run_all(&ctx);
        let commit = results.iter().find(|r| r.id == "commit-headroom").unwrap();
        assert!(commit.blocks());
        if let Outcome::Block(msg) = &commit.outcome {
            assert!(msg.contains("pagefile"), "remedy must name the pagefile: {msg}");
        }
    }

    #[test]
    fn display_attached_warns_but_does_not_block() {
        let mut ctx = healthy_ctx();
        ctx.resolved = vec![resolved_for(0, 1.0)]; // bus 3 card drives 2560x1440@144
        ctx.visibility_env = "0".into();
        let results = run_all(&ctx);
        let disp = results.iter().find(|r| r.id == "no-display").unwrap();
        assert!(matches!(disp.outcome, Outcome::Warn(_)), "{disp:?}");
        assert!(!any_block(&results));
    }

    #[test]
    fn vram_overflow_blocks_and_shows_arithmetic() {
        let mut ctx = healthy_ctx();
        // A fabricated 40 GiB single-device estimate against a 32 GiB card.
        let forty = 40u64 * 1024 * 1024 * 1024;
        ctx.estimate = Some(crate::estimate::VramEstimate {
            per_device: vec![crate::estimate::DeviceEstimate {
                key: ctx.resolved[0].profile_key.clone(),
                fraction: 1.0,
                weights_bytes: forty,
                kv_bytes: 0,
                compute_bytes: 0,
                overhead_bytes: 0,
                total_bytes: forty,
            }],
            total_bytes: forty,
            assumptions: vec![],
        });
        let results = run_all(&ctx);
        let vram = results.iter().find(|r| r.id == "vram-fits").unwrap();
        assert!(vram.blocks());
        if let Outcome::Block(msg) = &vram.outcome {
            assert!(msg.contains("weights"), "message must show the arithmetic: {msg}");
        }
    }

    #[test]
    fn driver_change_notes_baselines_unverified() {
        let mut ctx = healthy_ctx();
        ctx.driver_baseline = Some("32.0.30000.9000".into());
        let results = run_all(&ctx);
        let v = results.iter().find(|r| r.id == "versions-match").unwrap();
        assert!(matches!(v.outcome, Outcome::Note(_)), "{v:?}");
    }

    // ------------------------------------------------------------ diffusion ----

    fn diffusion_facts() -> ModelFacts {
        ModelFacts {
            architecture: Some("diffusion-gemma".into()),
            is_diffusion: true,
            runner_supported: true,
            block_count: Some(30),
        }
    }

    fn sizing(predicted: Option<u32>, maxtok_used: u32, full_offload: bool) -> DiffusionSizing {
        DiffusionSizing {
            canvas: 256,
            predicted_auto_maxtok: predicted,
            maxtok_used,
            max_prompt_tokens: maxtok_used - 256,
            pkv_bytes_per_token: 450_560,
            full_offload,
        }
    }

    /// dg-26b as the editor creates it: one card (bus08), auto context, the
    /// Unsloth build with its runner, and fidim-dg.exe installed.
    fn healthy_diffusion_ctx() -> LaunchContext {
        let mut ctx = healthy_ctx();
        ctx.profile.engine = Engine::DiffusionGemma;
        ctx.profile.runtime.ctx_total = 0;
        ctx.model_facts = Some(diffusion_facts());
        ctx.diffusion = Some(DiffusionPreflight {
            helper_exe: Some("C:/Programs/LlamaFIDIM/fidim-dg.exe".into()),
            helper_dir: Some("C:/Programs/LlamaFIDIM".into()),
            runner_exe: "C:/b/bin/llama-diffusion-gemma-visual-server.exe".into(),
            runner_present: true,
            vulkan_backend: false,
            req_prefix: "C:/Users/u/.fidim/runs/dg-dg-26b-9760".into(),
            req_prefix_error: None,
            req_fallback: "C:/Users/u/AppData/Local/Temp/fidim-dg-9760".into(),
            req_fallback_error: None,
            sizing: Some(sizing(Some(12288), 12288, true)),
        });
        ctx
    }

    fn outcome_of<'a>(results: &'a [CheckResult], id: &str) -> &'a Outcome {
        &results.iter().find(|r| r.id == id).unwrap_or_else(|| panic!("no {id} in {results:?}")).outcome
    }

    #[test]
    fn diffusion_checks() {
        // The fixture: no Block, and check 15 notes the predicted budget.
        let results = run_all(&healthy_diffusion_ctx());
        assert!(!any_block(&results), "{results:?}");
        assert_eq!(results.len(), 14, "13 and 15 appear for diffusion; 14 only when a card is shared");
        match outcome_of(&results, "diffusion-context") {
            Outcome::Note(m) => assert!(m.contains("12288") && m.contains("12032"), "{m}"),
            o => panic!("{o:?}"),
        }
        assert_eq!(outcome_of(&results, "engine-matches-model"), &Outcome::Pass);

        // Check 1: a missing helper, a missing runner or a Vulkan backend each Block.
        let block_msg = |ctx: &LaunchContext| match outcome_of(&run_all(ctx), "build-runs") {
            Outcome::Block(m) => m.clone(),
            o => panic!("{o:?}"),
        };
        let mut ctx = healthy_diffusion_ctx();
        ctx.diffusion.as_mut().unwrap().helper_exe = None;
        let m = block_msg(&ctx);
        assert!(m.contains("fidim-dg.exe not found in C:/Programs/LlamaFIDIM"), "{m}");
        assert!(m.contains("release zip") && m.contains("install.ps1"), "{m}");
        let mut ctx = healthy_diffusion_ctx();
        ctx.diffusion.as_mut().unwrap().runner_present = false;
        assert!(block_msg(&ctx).contains("--channel unsloth"));
        let mut ctx = healthy_diffusion_ctx();
        ctx.diffusion.as_mut().unwrap().vulkan_backend = true;
        assert!(block_msg(&ctx).contains("ggml-vulkan.dll"));

        // Check 4: the iGPU Blocks even when Settings allow integrated graphics.
        let mut ctx = healthy_diffusion_ctx();
        ctx.allow_integrated = true;
        ctx.resolved = vec![resolved_for(1, 1.0)];
        ctx.visibility_env = "1".into();
        assert!(matches!(outcome_of(&run_all(&ctx), "discrete-only"), Outcome::Block(_)));
        let mut llama = ctx.clone();
        llama.profile.engine = Engine::LlamaServer;
        assert!(matches!(outcome_of(&run_all(&llama), "discrete-only"), Outcome::Warn(_)));

        // Check 5: two resolved devices Block even when correctly pinned.
        let mut ctx = healthy_diffusion_ctx();
        ctx.resolved = vec![resolved_for(0, 0.5), resolved_for(2, 0.5)];
        ctx.visibility_env = "0,2".into();
        assert!(matches!(outcome_of(&run_all(&ctx), "visibility-pinned"), Outcome::Block(_)));

        // Check 13: engine and model disagree, either way.
        let mut ctx = healthy_ctx();
        ctx.model_facts = Some(diffusion_facts());
        match outcome_of(&run_all(&ctx), "engine-matches-model") {
            Outcome::Block(m) => assert!(m.contains("llama-server cannot load diffusion-gemma"), "{m}"),
            o => panic!("{o:?}"),
        }
        let mut ctx = healthy_diffusion_ctx();
        ctx.model_facts = Some(ModelFacts {
            architecture: Some("gemma4".into()),
            is_diffusion: false,
            runner_supported: false,
            block_count: Some(30),
        });
        assert!(matches!(outcome_of(&run_all(&ctx), "engine-matches-model"), Outcome::Block(_)));
        let mut ctx = healthy_diffusion_ctx();
        ctx.model_facts = Some(ModelFacts { architecture: Some("llada".into()), runner_supported: false, ..diffusion_facts() });
        match outcome_of(&run_all(&ctx), "engine-matches-model") {
            Outcome::Block(m) => assert!(m.contains("only diffusion-gemma"), "{m}"),
            o => panic!("{o:?}"),
        }

        // Check 14: an auto-sized diffusion profile next to any run, and a
        // llama-server profile on a card that hosts a diffusion run.
        let mut ctx = healthy_diffusion_ctx();
        ctx.co_resident = vec![CoResident {
            profile_id: "daily-driver".into(),
            device_key: ctx.resolved[0].device.stable_key.clone(),
            engine: Engine::LlamaServer,
        }];
        match outcome_of(&run_all(&ctx), "diffusion-card-sharing") {
            Outcome::Warn(m) => assert!(m.contains("daily-driver") && m.contains("WDDM"), "{m}"),
            o => panic!("{o:?}"),
        }
        let mut ctx = healthy_ctx();
        ctx.co_resident = vec![CoResident {
            profile_id: "dg-26b".into(),
            device_key: ctx.resolved[0].device.stable_key.clone(),
            engine: Engine::DiffusionGemma,
        }];
        match outcome_of(&run_all(&ctx), "diffusion-card-sharing") {
            Outcome::Warn(m) => assert!(m.contains("dg-26b"), "{m}"),
            o => panic!("{o:?}"),
        }
        // Two llama-server runs sharing a card are not this check's business.
        ctx.co_resident[0].engine = Engine::LlamaServer;
        assert_eq!(run_all(&ctx).len(), 12);

        // Check 15: an explicit budget above the prediction, and partial offload.
        let mut ctx = healthy_diffusion_ctx();
        ctx.profile.runtime.ctx_total = 16384;
        ctx.diffusion.as_mut().unwrap().sizing = Some(sizing(Some(12288), 16384, true));
        match outcome_of(&run_all(&ctx), "diffusion-context") {
            Outcome::Warn(m) => assert!(m.contains("above what fits"), "{m}"),
            o => panic!("{o:?}"),
        }
        let mut ctx = healthy_diffusion_ctx();
        ctx.diffusion.as_mut().unwrap().sizing = Some(sizing(Some(12288), 12288, false));
        match outcome_of(&run_all(&ctx), "diffusion-context") {
            Outcome::Warn(m) => assert!(m.contains("partial offload"), "{m}"),
            o => panic!("{o:?}"),
        }

        // Check 12: no keep-alive remedy for diffusion, and no stray runs of spaces.
        let mut ctx = healthy_diffusion_ctx();
        ctx.pcie_aspm = Some(1);
        match outcome_of(&run_all(&ctx), "pcie-aspm-off") {
            Outcome::Warn(m) => {
                assert!(!m.contains("keep-alive"), "{m}");
                assert!(!m.contains("   "), "{m}");
            }
            o => panic!("{o:?}"),
        }
        let mut ctx = healthy_ctx();
        ctx.pcie_aspm = Some(1);
        match outcome_of(&run_all(&ctx), "pcie-aspm-off") {
            Outcome::Warn(m) => {
                assert!(m.contains("(measured 2026-09-05), or enable the keep-alive interval"), "{m}");
                assert!(!m.contains("   "), "{m}");
            }
            o => panic!("{o:?}"),
        }
    }

    /// Check 1 mirrors the helper's request-path resolution: a fallback to
    /// %TEMP% is a Note naming it; neither path usable is a Block before the
    /// launch instead of exit 7 after it (a non-ASCII user name breaks both).
    #[test]
    fn diffusion_request_path_checks() {
        let mut ctx = healthy_diffusion_ctx();
        let d = ctx.diffusion.as_mut().unwrap();
        d.req_prefix = "C:/Users/Pål/.fidim/runs/dg-dg-26b-9760".into();
        d.req_prefix_error = crate::diffusion::protocol::check_req_prefix(&d.req_prefix).err();
        assert!(d.req_prefix_error.is_some());
        match outcome_of(&run_all(&ctx), "build-runs") {
            Outcome::Note(m) => assert!(m.contains("%TEMP%") && m.contains("fidim-dg-9760"), "{m}"),
            o => panic!("{o:?}"),
        }
        let d = ctx.diffusion.as_mut().unwrap();
        d.req_fallback = "C:/Users/Pål/AppData/Local/Temp/fidim-dg-9760".into();
        d.req_fallback_error = crate::diffusion::protocol::check_req_prefix(&d.req_fallback).err();
        match outcome_of(&run_all(&ctx), "build-runs") {
            Outcome::Block(m) => assert!(m.contains("exit 7") && m.contains("ASCII"), "{m}"),
            o => panic!("{o:?}"),
        }
    }

    /// Check 6 for a diffusion profile at auto: the runner sizes MAXTOK to the
    /// card at load, so a full-budget working set near or over the card is a
    /// Note or a Warn, never a Block; only the load itself Blocks. With an
    /// explicit budget, and for llama-server, the usual thresholds apply.
    #[test]
    fn diffusion_vram_at_auto_is_not_a_block() {
        fn with_estimate(ctx: &mut LaunchContext, weights: u64, kv: u64) {
            let key = ctx.resolved[0].profile_key.clone();
            ctx.estimate = Some(crate::estimate::VramEstimate {
                per_device: vec![crate::estimate::DeviceEstimate {
                    key,
                    fraction: 1.0,
                    weights_bytes: weights,
                    kv_bytes: kv,
                    compute_bytes: 0,
                    overhead_bytes: 0,
                    total_bytes: weights + kv,
                }],
                total_bytes: weights + kv,
                assumptions: vec![],
            });
        }
        let free = healthy_diffusion_ctx().resolved[0].device.free_mib * 1024 * 1024;
        assert!(free > 0);
        // 55% load + 42% working set = 97%: Note at auto, Warn when explicit.
        let mut ctx = healthy_diffusion_ctx();
        with_estimate(&mut ctx, free * 55 / 100, free * 42 / 100);
        match outcome_of(&run_all(&ctx), "vram-fits") {
            Outcome::Note(m) => assert!(m.contains("sizes MAXTOK"), "{m}"),
            o => panic!("{o:?}"),
        }
        ctx.profile.runtime.ctx_total = 8192;
        assert!(matches!(outcome_of(&run_all(&ctx), "vram-fits"), Outcome::Warn(_)));
        // 55% + 50% = 105%: Warn at auto (set an explicit budget), Block when explicit.
        let mut ctx = healthy_diffusion_ctx();
        with_estimate(&mut ctx, free * 55 / 100, free * 50 / 100);
        match outcome_of(&run_all(&ctx), "vram-fits") {
            Outcome::Warn(m) => assert!(m.contains("explicit context budget"), "{m}"),
            o => panic!("{o:?}"),
        }
        ctx.profile.runtime.ctx_total = 8192;
        assert!(matches!(outcome_of(&run_all(&ctx), "vram-fits"), Outcome::Block(_)));
        // The load alone over the card Blocks at auto too.
        let mut ctx = healthy_diffusion_ctx();
        with_estimate(&mut ctx, free * 105 / 100, 0);
        match outcome_of(&run_all(&ctx), "vram-fits") {
            Outcome::Block(m) => assert!(m.contains("no context budget fits"), "{m}"),
            o => panic!("{o:?}"),
        }
        // llama-server keeps the old thresholds.
        let mut ctx = healthy_ctx();
        with_estimate(&mut ctx, free * 55 / 100, free * 42 / 100);
        assert!(matches!(outcome_of(&run_all(&ctx), "vram-fits"), Outcome::Warn(_)));
    }
}
