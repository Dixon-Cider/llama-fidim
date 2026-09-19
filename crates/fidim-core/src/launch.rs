//! Launch composition: profile → exact command line + environment (R-13,
//! §07 "configuration is passed as CLI arguments and environment variables").
//!
//! `LaunchPlan` is the single source of truth consumed by the spawner, the
//! script exporter, and the parity tests — what you export is exactly what
//! runs.

use std::net::TcpListener;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::config::Config;
use crate::devices::{self, Device};
use crate::discovery;
use crate::estimate::{self, DiffusionSizing, EstimateInput, VramEstimate};
use crate::gguf;
use crate::platform::Platform;
use crate::preflight::{CoResident, DiffusionPreflight, LaunchContext, ModelFacts, PortHolder, ResolvedDevice};
use crate::profile::{env_has_key, Profile, SplitMode};
use crate::{Error, Result};

#[derive(Debug, Clone, Serialize)]
pub struct LaunchPlan {
    pub exe: PathBuf,
    pub args: Vec<String>,
    /// Environment set for the child (on top of the inherited env).
    pub env: Vec<(String, String)>,
    /// ROCm bin dir prepended to PATH.
    pub path_prepend: Option<PathBuf>,
    /// The composed visibility pin — also asserted by pre-flight check 5.
    pub visibility_env: String,
}

impl LaunchPlan {
    /// Full command line for display/state files (args quoted when needed).
    pub fn command_line(&self) -> String {
        let mut parts = vec![quote_arg(&self.exe.to_string_lossy())];
        parts.extend(self.args.iter().map(|a| quote_arg(a)));
        parts.join(" ")
    }
}

fn quote_arg(a: &str) -> String {
    if a.contains(' ') || a.contains('"') {
        format!("\"{}\"", a.replace('"', "\\\""))
    } else {
        a.to_string()
    }
}

/// Compose the command and environment for a profile whose devices are
/// already resolved. `resolved` must be in profile-device order.
pub fn compose(profile: &Profile, resolved: &[ResolvedDevice]) -> LaunchPlan {
    let mut args: Vec<String> = Vec::new();
    let m = &profile.model;
    args.push("-m".into());
    args.push(m.path.to_string_lossy().into_owned());
    if let Some(mmproj) = &m.mmproj {
        args.push("--mmproj".into());
        args.push(mmproj.to_string_lossy().into_owned());
    }
    if let Some(draft) = &m.draft {
        if draft.enabled {
            // The file only; the mode and draft-token knobs come from
            // profile.speculative (emitted after sampling, below).
            args.push("-md".into());
            args.push(draft.path.to_string_lossy().into_owned());
        }
    }

    let r = &profile.runtime;
    args.extend(["-ngl".into(), r.n_gpu_layers.to_string()]);
    args.extend(["-c".into(), r.ctx_total.to_string()]);
    args.extend(["-np".into(), r.slots.to_string()]);
    args.extend(["-fa".into(), r.flash_attn.clone()]);
    args.extend(["-ctk".into(), r.kv_type_k.clone()]);
    args.extend(["-ctv".into(), r.kv_type_v.clone()]);
    args.extend(["-b".into(), r.batch_logical.to_string()]);
    args.extend(["-ub".into(), r.batch_physical.to_string()]);
    if r.cont_batching {
        args.push("-cb".into());
    }
    if r.kv_unified {
        args.push("--kv-unified".into());
    }
    if let Some(reuse) = r.cache_reuse {
        args.extend(["--cache-reuse".into(), reuse.to_string()]);
    }

    // Multi-device split (R-14). Fractions and --main-gpu refer to the
    // REMAPPED order created by the visibility pin: device i in the profile
    // list is in-process device i.
    if resolved.len() > 1 {
        let mode = match profile.split_mode {
            Some(SplitMode::Row) => "row",
            _ => "layer",
        };
        args.extend(["--split-mode".into(), mode.into()]);
        let fractions: Vec<String> =
            resolved.iter().map(|d| format!("{:.3}", d.fraction)).collect();
        args.extend(["--tensor-split".into(), fractions.join(",")]);
        args.extend(["--main-gpu".into(), profile.main_device.to_string()]);
    }

    let s = &profile.sampling;
    if let Some(t) = s.temperature {
        args.extend(["--temp".into(), format_num(t)]);
    }
    if let Some(k) = s.top_k {
        args.extend(["--top-k".into(), k.to_string()]);
    }
    if let Some(p) = s.top_p {
        args.extend(["--top-p".into(), format_num(p)]);
    }
    if let Some(p) = s.min_p {
        args.extend(["--min-p".into(), format_num(p)]);
    }
    if let Some(d) = s.dry_multiplier {
        args.extend(["--dry-multiplier".into(), format_num(d)]);
    }
    if let Some(p) = s.repeat_penalty {
        args.extend(["--repeat-penalty".into(), format_num(p)]);
    }
    if let Some(p) = s.presence_penalty {
        args.extend(["--presence-penalty".into(), format_num(p)]);
    }
    if let Some(t) = r.threads {
        args.extend(["-t".into(), t.to_string()]);
    }

    // Speculative decoding (same flag family on every build since b9553).
    // `-md <draft>` for external heads/drafts is emitted with the model
    // block above whenever model.draft is enabled.
    let spec = profile.speculative_effective();
    if let Some(t) = spec.spec_type() {
        args.extend(["--spec-type".into(), t.into()]);
        if let Some(n) = spec.n_max {
            args.extend(["--spec-draft-n-max".into(), n.to_string()]);
        }
        if let Some(n) = spec.n_min {
            args.extend(["--spec-draft-n-min".into(), n.to_string()]);
        }
        if let Some(p) = spec.p_min {
            args.extend(["--spec-draft-p-min".into(), format_num(p)]);
        }
    }

    args.push("--jinja".into());
    args.push("--metrics".into());
    // Enables the /slots endpoint the Running view reads (V-3). Local-only
    // servers; the endpoint's prompt-exposure caveat does not apply here.
    args.push("--slots".into());
    args.extend(["--alias".into(), profile.server.alias.clone()]);
    args.extend(["--host".into(), profile.server.host.clone()]);
    args.extend(["--port".into(), profile.server.port.to_string()]);

    // Unknown flags pass through without a tool update (§07).
    args.extend(r.extra_flags.iter().cloned());

    // ---- environment ---------------------------------------------------
    // Visibility pin (R-13): global HIP indices of the resolved devices; the
    // process then sees them remapped to 0..n in list order.
    let visibility: Vec<String> =
        resolved.iter().map(|d| d.device.hip_index.to_string()).collect();
    let visibility_env = visibility.join(",");
    let mut env: Vec<(String, String)> =
        vec![("HIP_VISIBLE_DEVICES".into(), visibility_env.clone())];
    for (k, v) in &profile.env {
        env.push((k.clone(), v.clone()));
    }
    if profile.chat.enable_thinking == Some(false) {
        env.push((
            "LLAMA_ARG_CHAT_TEMPLATE_KWARGS".into(),
            r#"{"enable_thinking": false}"#.into(),
        ));
    }

    LaunchPlan {
        exe: profile.build.path.join("bin").join("llama-server.exe"),
        args,
        env,
        path_prepend: None,
        visibility_env,
    }
}

fn format_num(v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

// ------------------------------------------------------------ diffusion ----

/// The paths and facts `compose_diffusion` needs beyond the profile, found
/// by the launch path (or given by tests).
#[derive(Debug, Clone)]
pub struct DiffusionPaths {
    pub helper_exe: PathBuf,
    pub runner_exe: PathBuf,
    pub req_prefix: PathBuf,
    /// PCI bus of the chosen card; the helper refuses a runner on another.
    pub expect_bus: Option<u32>,
    pub build_tag: Option<String>,
    pub full_offload: bool,
}

/// The helper's `--build-tag`: the release tag (else the profile's recorded
/// version), plus `+<patch>` for a patched build, which shares its base's
/// release tag, so /health and the run log tell them apart.
pub fn dg_build_tag(meta: &discovery::BuildMeta, version: Option<&str>) -> Option<String> {
    let tag = meta.release_tag.as_deref().or(version)?;
    Some(match &meta.patch {
        Some(p) => format!("{tag}+{}", p.label()),
        None => tag.to_string(),
    })
}

/// The DiffusionGemma runner inside the profile's build.
pub fn runner_exe(p: &Profile) -> PathBuf {
    p.build.path.join("bin").join(discovery::RUNNER_EXE)
}

/// Where a diffusion run's request files go: `runs_dir/dg-<id>-<port>`, to
/// which the helper appends `-<task>-<attempt>.req`.
pub fn req_prefix(runs_dir: &Path, p: &Profile) -> PathBuf {
    req_prefix_for(runs_dir, &p.id, p.server.port)
}

/// `req_prefix` from a run state's identity (stop has no profile).
pub fn req_prefix_for(runs_dir: &Path, profile_id: &str, port: u16) -> PathBuf {
    runs_dir.join(format!("dg-{profile_id}-{port}"))
}

/// Compose the `fidim-dg.exe` command for a diffusion profile. Pure. The
/// runner reads everything but the model path from its environment, so the
/// env carries the settings: profile.env first, then every key FIDIM owns,
/// which therefore wins (Windows env names are case-insensitive, and so is
/// `Command::env` there).
pub fn compose_diffusion(p: &Profile, resolved: &[ResolvedDevice], d: &DiffusionPaths) -> LaunchPlan {
    let dg = p.diffusion_effective();
    let mut args: Vec<String> = vec![
        "--runner".into(),
        d.runner_exe.to_string_lossy().into_owned(),
        "--model".into(),
        p.model.path.to_string_lossy().into_owned(),
        "--host".into(),
        p.server.host.clone(),
        "--port".into(),
        p.server.port.to_string(),
        "--alias".into(),
        p.server.alias.clone(),
        "--req-prefix".into(),
        d.req_prefix.to_string_lossy().into_owned(),
        "--default-max-tokens".into(),
        dg.default_max_tokens.to_string(),
    ];
    if let Some(seed) = dg.seed {
        args.extend(["--seed".into(), seed.to_string()]);
    }
    if let Some(bus) = d.expect_bus {
        args.extend(["--expect-bus".into(), bus.to_string()]);
    }
    if let Some(tag) = &d.build_tag {
        args.extend(["--build-tag".into(), tag.clone()]);
    }

    // One card only (check 5): the runner's unified multi-device path
    // aborts every prompt. The HIP runtime honours CUDA_VISIBLE_DEVICES as
    // well, so both carry the same index.
    let visibility_env = resolved.first().map(|r| r.device.hip_index.to_string()).unwrap_or_default();
    let mut env: Vec<(String, String)> = p.env.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    env.push(("HIP_VISIBLE_DEVICES".into(), visibility_env.clone()));
    env.push(("CUDA_VISIBLE_DEVICES".into(), visibility_env.clone()));
    // Always sent: the runner's own default is 0, a CPU run.
    env.push(("NGL".into(), p.runtime.n_gpu_layers.to_string()));
    env.push(("MAXTOK".into(), p.runtime.ctx_total.to_string()));
    env.push(("FA".into(), if dg.flash_attn { "1" } else { "0" }.into()));
    // With the whole model on the card, size the context against VRAM only,
    // so auto MAXTOK does not depend on how much RAM happens to be free.
    if d.full_offload {
        env.push(("DG_FREE_RAM_MB".into(), "0".into()));
    }
    if dg.hipblaslt_safeguard {
        for k in ["ROCBLAS_USE_HIPBLASLT", "ROCBLAS_USE_HIPBLASLT_BATCHED"] {
            if !env_has_key(&p.env, k) {
                env.push((k.into(), "0".into()));
            }
        }
    }
    // The HIP runtime otherwise keeps freed device memory cached: the runner
    // held 3.9 GiB more after a 10K-token prompt, at no measured speed gain.
    // The profile env may still set it (its value wins, see above).
    if !env_has_key(&p.env, crate::profile::DG_RUNTIME_CACHE_KEY) {
        env.push((crate::profile::DG_RUNTIME_CACHE_KEY.into(), "0".into()));
    }
    // Step drafts that keep the thinking markers, for the live view. A stock
    // runner ignores the key.
    if !env_has_key(&p.env, "DG_FRAME_SPECIAL") {
        env.push(("DG_FRAME_SPECIAL".into(), "1".into()));
    }

    LaunchPlan { exe: d.helper_exe.clone(), args, env, path_prepend: None, visibility_env }
}

// -------------------------------------------------------- context builder ----

/// Everything the launch path measured while assembling a context, kept so
/// callers (CLI/GUI) can show it.
pub struct PreparedLaunch {
    pub context: LaunchContext,
    pub plan: LaunchPlan,
    pub devices_now: Vec<Device>,
}

/// Enumerate devices using the given server binary's own `--list-devices`
/// (the canonical index space) correlated with OS adapters + hipInfo.
pub fn enumerate_devices(
    cfg: &Config,
    server_exe: &Path,
    platform: &dyn Platform,
) -> Result<Vec<Device>> {
    // A build that bundles its own ROCm enumerates the way it launches: on
    // its own DLLs with nothing on PATH. The default runtime would mix two
    // ROCm versions in one process, and a mixed load lists no devices at all.
    if bundles_runtime(server_exe) {
        return enumerate_devices_with(cfg, server_exe, platform, None);
    }
    // Probe with the default runtime (and its shims) so the HIP backend
    // loads the same way a launch would; the bare fallback folder is the
    // last resort.
    // An in-folder shim from an older install shadows whatever runtime is
    // on PATH (the exe folder wins DLL resolution) and mixes two ROCm
    // versions in one process; retire it before every probe.
    crate::update::retire_build_shims(server_exe);
    let prepend = crate::runtime::default_prepend(cfg, server_exe).or_else(|| cfg.rocm_bin.clone());
    enumerate_devices_with(cfg, server_exe, platform, prepend.as_deref())
}

/// Whether the build `<dir>/bin/llama-server.exe` belongs to says, in its
/// FIDIM manifest, that it ships its own ROCm.
fn bundles_runtime(server_exe: &Path) -> bool {
    server_exe
        .parent()
        .and_then(Path::parent)
        .is_some_and(|dir| discovery::read_build_meta(dir).bundled_runtime)
}

/// `enumerate_devices` against a specific runtime search path (a profile's
/// chosen ROCm runtime). hipInfo correlation still comes from the system
/// SDK, which is the only runtime that ships it.
pub fn enumerate_devices_with(
    cfg: &Config,
    server_exe: &Path,
    platform: &dyn Platform,
    runtime_path: Option<&Path>,
) -> Result<Vec<Device>> {
    let listed_text = run_capture(server_exe, &["--list-devices"], runtime_path)?;
    let listed = devices::parse_list_devices(&listed_text)?;
    let adapters = platform.video_adapters()?;
    let hipinfo = cfg
        .rocm_bin
        .as_ref()
        .map(|b| b.join("hipInfo.exe"))
        .filter(|p| p.is_file())
        .and_then(|exe| run_capture(&exe, &[], cfg.rocm_bin.as_deref()).ok())
        .map(|t| devices::parse_hipinfo(&t))
        .unwrap_or_default();
    Ok(devices::correlate(&listed, &adapters, &hipinfo, &cfg.integrated_name_patterns))
}

/// Keep a probe/helper process from opening a console window. From a GUI
/// process every `Command` child gets its own console by default, so a
/// profile selection (build probe + --list-devices + hipInfo) flashed a
/// burst of windows and stole keyboard focus. Output is captured through
/// pipes either way, so nothing is lost.
pub fn hide_console(cmd: &mut std::process::Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    {
        let _ = cmd;
    }
}

pub fn run_capture(exe: &Path, args: &[&str], rocm_bin: Option<&Path>) -> Result<String> {
    let mut cmd = std::process::Command::new(exe);
    cmd.args(args);
    hide_console(&mut cmd);
    if let Some(rocm) = rocm_bin {
        let path = std::env::var_os("PATH").unwrap_or_default();
        let mut joined = rocm.as_os_str().to_owned();
        joined.push(";");
        joined.push(path);
        cmd.env("PATH", joined);
    }
    let out = cmd
        .output()
        .map_err(|e| Error::BuildBinary { path: exe.to_path_buf(), detail: e.to_string() })?;
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok(text)
}

/// ROCm SDK version: `hipconfig --version`, falling back to the install
/// directory name.
pub fn sdk_version(rocm_bin: Option<&Path>) -> Option<String> {
    let bin = rocm_bin?;
    let hipconfig = bin.join("hipconfig.exe");
    if hipconfig.is_file() {
        if let Ok(out) = run_capture(&hipconfig, &["--version"], rocm_bin) {
            let v = out.trim().to_string();
            if !v.is_empty() {
                return Some(v);
            }
        }
    }
    bin.parent().and_then(|p| p.file_name()).map(|n| format!("ROCm {}", n.to_string_lossy()))
}

/// Pre-gathered slow inputs for `prepare_with_inputs`: the UI caches these
/// (device enumeration ~2-4s, build probe ~1s) so the profile editor can
/// re-run pre-flight live. Launches always gather fresh (R-03).
pub struct PrepareInputs {
    pub devices_now: Vec<Device>,
    pub build_version_output: Option<String>,
}

impl PrepareInputs {
    pub fn gather(cfg: &Config, profile: &Profile, platform: &dyn Platform) -> Result<Self> {
        let server_exe = profile.build.path.join("bin").join("llama-server.exe");
        // Probe and enumerate with the runtime this profile will launch
        // under, so a backend that only loads against one runtime is judged
        // against that one. A build that bundles its own ROCm runs with
        // nothing on PATH, and its device indices come from that bundle.
        let meta = discovery::read_build_meta(&profile.build.path);
        let runtime_path = if meta.bundled_runtime {
            None
        } else {
            crate::runtime::path_prepend(cfg, profile.rocm_runtime.as_deref())?
        };
        let build_version_output =
            run_capture(&server_exe, &["--version"], runtime_path.as_deref())
                .ok()
                .filter(|t| discovery::parse_version_output(t).is_some());
        let devices_now = if build_version_output.is_some() {
            enumerate_devices_with(cfg, &server_exe, platform, runtime_path.as_deref())?
        } else {
            Vec::new()
        };
        Ok(PrepareInputs { devices_now, build_version_output })
    }
}

/// Build the full pre-flight context for a profile: resolve devices, compute
/// fractions, estimate VRAM, read commit, probe the build, check the port.
/// Gathers fresh inputs — the correct choice for a real launch.
pub fn prepare(
    cfg: &Config,
    profile: &Profile,
    platform: &dyn Platform,
    running_aliases: Vec<String>,
) -> Result<PreparedLaunch> {
    let inputs = PrepareInputs::gather(cfg, profile, platform)?;
    prepare_with_inputs(cfg, profile, platform, running_aliases, inputs)
}

/// `prepare` with the slow inputs supplied by the caller (UI live checks).
pub fn prepare_with_inputs(
    cfg: &Config,
    profile: &Profile,
    platform: &dyn Platform,
    running_aliases: Vec<String>,
    inputs: PrepareInputs,
) -> Result<PreparedLaunch> {
    let PrepareInputs { devices_now, build_version_output } = inputs;
    let diffusion = profile.engine.is_diffusion();
    let meta = discovery::read_build_meta(&profile.build.path);

    // File existence (check 2). The diffusion runner takes neither a
    // projector nor a draft, so a stale path there does not matter.
    let mut missing = Vec::new();
    let mut check_file = |p: &Path| {
        if !p.is_file() {
            missing.push(p.to_string_lossy().into_owned());
        }
    };
    check_file(&profile.model.path);
    if !diffusion {
        if let Some(mm) = &profile.model.mmproj {
            check_file(mm);
        }
        if let Some(d) = &profile.model.draft {
            if d.enabled {
                check_file(&d.path);
            }
        }
    }

    // Device resolution (check 3).
    let mut resolved: Vec<ResolvedDevice> = Vec::new();
    let mut unresolved: Vec<(String, String)> = Vec::new();
    let mut taken: Vec<String> = Vec::new();
    for dref in &profile.devices {
        match devices::resolve_key(&dref.key, &devices_now, &taken) {
            Ok(res) => {
                taken.push(res.device.stable_key.clone());
                resolved.push(ResolvedDevice {
                    profile_key: dref.key.clone(),
                    device: res.device,
                    fraction: 0.0, // filled below
                    rebound: res.rebound_from.is_some(),
                });
            }
            Err(e) => unresolved.push((dref.key.clone(), e.to_string())),
        }
    }

    // Split fractions: explicit, or auto by free VRAM (R-14).
    let explicit: Vec<Option<f64>> = profile.devices.iter().map(|d| d.split_fraction).collect();
    if resolved.len() == profile.devices.len() && !resolved.is_empty() {
        if explicit.iter().all(|f| f.is_some()) {
            for (r, f) in resolved.iter_mut().zip(&explicit) {
                r.fraction = f.unwrap();
            }
        } else {
            let free: Vec<u64> = resolved.iter().map(|r| r.device.free_mib).collect();
            let fractions = estimate::auto_fractions(&free);
            for (r, f) in resolved.iter_mut().zip(fractions) {
                r.fraction = f;
            }
        }
        if resolved.len() == 1 {
            resolved[0].fraction = 1.0;
        }
    }

    // VRAM estimate (check 6), and what the header says about the engine
    // the model needs (check 13). A split model's weights are every shard.
    let header = crate::discovery::read_model_header(&profile.model.path).ok();
    let model_facts = header.as_ref().map(ModelFacts::from_header);
    let mut sizing: Option<DiffusionSizing> = None;
    let estimate: Option<VramEstimate> = match &header {
        None => None,
        Some(h) if diffusion => resolved.first().and_then(|r| {
            let (est, s) = estimate::estimate_diffusion(
                h,
                profile.runtime.n_gpu_layers,
                profile.runtime.ctx_total,
                profile.diffusion_effective().flash_attn,
                estimate::DgRunner::from_build(&meta, crate::profile::dg_runtime_cache_off(&profile.env)),
                &r.profile_key,
                r.device.free_mib,
            );
            sizing = Some(s);
            // More than one card blocks at check 5; a per-card figure for
            // only the first would read as a fit.
            (resolved.len() == 1).then_some(est)
        }),
        Some(h) => Some(llama_server_estimate(profile, &resolved, h)),
    };

    let commit = platform.system_commit()?;
    let runs = crate::supervise::reattach(&cfg.runs_dir);
    let inflight_reserved_bytes = crate::supervise::inflight_of(&runs);
    let port_holder = port_holder(&profile.server.host, profile.server.port, &cfg.runs_dir);
    // Live runs on this launch's cards. A run on this profile's own port is
    // left out: launching stops it first to take the port over.
    let keys: Vec<&str> = resolved.iter().map(|r| r.device.stable_key.as_str()).collect();
    let co_resident: Vec<CoResident> = runs
        .iter()
        .filter(|r| r.alive && r.state.port != profile.server.port)
        .flat_map(|r| {
            r.state.device_keys.iter().filter(|k| keys.contains(&k.as_str())).map(move |k| CoResident {
                profile_id: r.state.profile_id.clone(),
                device_key: k.clone(),
                engine: r.state.engine,
            })
        })
        .collect();

    // Driver/SDK now vs baseline (check 11). A bundled build has no runtime
    // to resolve: it runs on the DLLs in its own folder.
    let driver_now = resolved.iter().find_map(|r| r.device.driver_version.clone());
    let runtime = if meta.bundled_runtime {
        None
    } else {
        Some(crate::runtime::resolve(cfg, profile.rocm_runtime.as_deref())?)
    };
    // SDK identity for the baseline fingerprint: hipconfig where the runtime
    // has one (the HIP SDK), else the runtime's name + version.
    let sdk_now = match &runtime {
        Some(runtime) => sdk_version(runtime.dirs.first().map(|p| p.as_path())).or_else(|| {
            Some(format!(
                "{}{}",
                runtime.name,
                runtime.version.as_ref().map(|v| format!(" {v}")).unwrap_or_default()
            ))
        }),
        None => Some(format!(
            "bundled ROCm ({})",
            meta.release_tag.clone().unwrap_or_else(|| {
                profile.build.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
            })
        )),
    };
    let baseline = profile.baseline.as_ref();
    let driver_baseline =
        baseline.and_then(|b| b.get("driver")).and_then(|v| v.as_str()).map(String::from);
    let sdk_baseline =
        baseline.and_then(|b| b.get("sdk")).and_then(|v| v.as_str()).map(String::from);

    let (plan, diffusion_pre) = if diffusion {
        let runner = runner_exe(profile);
        let helper = crate::supervise::helper_exe(crate::supervise::DG_HELPER_EXE);
        let helper_dir = std::env::current_exe().ok().and_then(|me| me.parent().map(Path::to_path_buf));
        let paths = DiffusionPaths {
            // A missing helper blocks at check 1; the plan still names where
            // it would be.
            helper_exe: helper.clone().unwrap_or_else(|| match &helper_dir {
                Some(d) => d.join(crate::supervise::DG_HELPER_EXE),
                None => PathBuf::from(crate::supervise::DG_HELPER_EXE),
            }),
            runner_exe: runner.clone(),
            req_prefix: req_prefix(&cfg.runs_dir, profile),
            expect_bus: resolved.first().and_then(|r| r.device.bus_number),
            build_tag: dg_build_tag(&meta, profile.build.version.as_deref()),
            full_offload: sizing.as_ref().is_some_and(|s| s.full_offload),
        };
        let mut plan = compose_diffusion(profile, &resolved, &paths);
        // The helper passes its PATH on to the runner, which lives in the build.
        plan.path_prepend = runtime.as_ref().and_then(|rt| crate::runtime::prepend_for(rt, &runner));
        // The helper judges these two paths the same way at startup and
        // exits 7 when neither is usable; check 1 says so before the launch.
        let req_fallback = crate::diffusion::req_fallback_prefix(profile.server.port);
        let pre = DiffusionPreflight {
            helper_exe: helper,
            helper_dir,
            runner_present: runner.is_file(),
            vulkan_backend: profile.build.path.join("bin").join("ggml-vulkan.dll").is_file(),
            runner_exe: runner,
            req_prefix_error: crate::diffusion::protocol::check_req_prefix(&paths.req_prefix).err(),
            req_prefix: paths.req_prefix.clone(),
            req_fallback_error: crate::diffusion::protocol::check_req_prefix(&req_fallback).err(),
            req_fallback,
            sizing,
            runner: estimate::DgRunner::from_build(&meta, crate::profile::dg_runtime_cache_off(&profile.env)),
        };
        (plan, Some(pre))
    } else {
        let mut plan = compose(profile, &resolved);
        // Shims an older install copied into the build folder would shadow the
        // chosen runtime; retire them, then prepend the runtime's own shim dir
        // and its DLL dirs (joined with `;`).
        crate::update::retire_build_shims(&plan.exe);
        plan.path_prepend = runtime.as_ref().and_then(|rt| crate::runtime::prepend_for(rt, &plan.exe));
        (plan, None)
    };

    let context = LaunchContext {
        profile: profile.clone(),
        allow_integrated: cfg.allow_integrated,
        build_version_output,
        missing_files: missing,
        resolved,
        unresolved,
        estimate,
        commit,
        inflight_reserved_bytes,
        visibility_env: plan.visibility_env.clone(),
        running_aliases,
        port_holder,
        driver_now,
        driver_baseline,
        sdk_now,
        sdk_baseline,
        pcie_aspm: pcie_aspm_ac(),
        model_facts,
        diffusion: diffusion_pre,
        co_resident,
    };
    Ok(PreparedLaunch { context, plan, devices_now })
}

/// The llama-server VRAM estimate for a profile whose devices are resolved.
fn llama_server_estimate(profile: &Profile, resolved: &[ResolvedDevice], h: &gguf::GgufHeader) -> VramEstimate {
    let mmproj_bytes = profile
        .model
        .mmproj
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    let draft_bytes = profile
        .model
        .draft
        .as_ref()
        .filter(|d| d.enabled)
        .and_then(|d| std::fs::metadata(&d.path).ok())
        .map(|m| m.len())
        .unwrap_or(0);
    estimate::estimate(&EstimateInput {
        header: h,
        runtime: &profile.runtime,
        devices: resolved.iter().map(|r| (r.profile_key.clone(), r.fraction)).collect(),
        split_mode: profile.split_mode,
        main_index: (profile.main_device as usize).min(resolved.len().saturating_sub(1)),
        mmproj_bytes,
        draft_bytes,
    })
}

/// AC index of "PCI Express > Link State Power Management" on the active
/// power plan (0 = Off, 1 = Moderate, 2 = Maximum). Read via powercfg; None
/// when it cannot be read. See preflight check 12 for why it matters.
pub fn pcie_aspm_ac() -> Option<u32> {
    let mut cmd = std::process::Command::new("powercfg");
    cmd.args(["/q", "SCHEME_CURRENT", "SUB_PCIEXPRESS", "ASPM"]);
    hide_console(&mut cmd);
    let out = cmd.output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    parse_powercfg_ac_index(&text)
}

/// `Current AC Power Setting Index: 0x00000001` -> 1.
pub fn parse_powercfg_ac_index(text: &str) -> Option<u32> {
    let line = text.lines().find(|l| l.contains("Current AC Power Setting Index"))?;
    let hex = line.split(':').nth(1)?.trim().trim_start_matches("0x");
    u32::from_str_radix(hex, 16).ok()
}

/// Who holds `port`, if anyone: a live Llama FIDIM run (by its state file),
/// else whatever netstat says is listening there.
pub fn port_holder(host: &str, port: u16, runs_dir: &Path) -> Option<PortHolder> {
    if port_is_free(host, port) {
        return None;
    }
    let ours = crate::supervise::reattach(runs_dir)
        .into_iter()
        .find(|r| r.alive && r.state.port == port);
    let pid = ours.as_ref().map(|r| r.state.pid).or_else(|| listening_pid(port));
    let process_name = pid.and_then(process_name);
    Some(PortHolder { pid, process_name, profile_id: ours.map(|r| r.state.profile_id) })
}

/// PID of the process LISTENING on a TCP port (netstat -ano).
pub fn listening_pid(port: u16) -> Option<u32> {
    parse_netstat_listener(&netstat_tcp()?, port)
}

/// Whether `pid` has a LISTENING socket on `port`. Checks every listener
/// rather than the first: one process on 127.0.0.1:N and another on [::1]:N
/// can coexist. False when netstat cannot run.
pub fn listens_on(port: u16, pid: u32) -> bool {
    netstat_tcp().is_some_and(|t| parse_netstat_listeners(&t, port).contains(&pid))
}

fn netstat_tcp() -> Option<String> {
    // No `-p tcp`: that lists IPv4 sockets only, and a server bound to
    // "localhost" or "::1" listens on IPv6. Without `-p` both families print
    // as "TCP" rows (UDP rows are filtered out by the parser).
    let mut cmd = std::process::Command::new("netstat");
    cmd.arg("-ano");
    hide_console(&mut cmd);
    let out = cmd.output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `  TCP    127.0.0.1:1234    0.0.0.0:0    LISTENING    3776` -> 3776.
pub fn parse_netstat_listener(text: &str, port: u16) -> Option<u32> {
    parse_netstat_listeners(text, port).into_iter().next()
}

/// Every pid listening on `port`, IPv4 or IPv6, in netstat order. A
/// listener is recognised by its unconnected foreign address, not by the
/// state column: Windows translates "LISTENING" (ABHÖREN, …), so matching
/// the English word made every run look dead on a non-English system.
pub fn parse_netstat_listeners(text: &str, port: u16) -> Vec<u32> {
    let suffix = format!(":{port}");
    text.lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            let unconnected = f.get(2).is_some_and(|a| *a == "0.0.0.0:0" || *a == "[::]:0");
            if f.len() >= 5 && f[0].eq_ignore_ascii_case("TCP") && f[1].ends_with(&suffix) && unconnected {
                f.last()?.parse().ok()
            } else {
                None
            }
        })
        .collect()
}

fn process_name(pid: u32) -> Option<String> {
    let mut cmd = std::process::Command::new("tasklist");
    cmd.args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"]);
    hide_console(&mut cmd);
    let out = cmd.output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| l.starts_with('"'))?;
    line.split(',').next().map(|s| s.trim_matches('"').to_string())
}

/// Poll until `port` is free or `timeout` elapses (used after a takeover stop).
pub fn wait_port_free(host: &str, port: u16, timeout: std::time::Duration) -> Result<()> {
    let start = std::time::Instant::now();
    while !port_is_free(host, port) {
        if start.elapsed() > timeout {
            return Err(Error::Platform(format!("port {port} still in use {timeout:?} after stopping its server")));
        }
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
    Ok(())
}

pub fn port_is_free(host: &str, port: u16) -> bool {
    TcpListener::bind((host, port)).is_ok()
}

/// Final commit gate, run immediately before spawn.
///
/// Pre-flight can be minutes stale by the time a launch actually fires (a
/// model finished loading elsewhere, another launch started, the user opened
/// something large). Re-reading commit here closes that window. Deliberately
/// commit-only: physical RAM is not a gate anywhere in this pipeline, per the
/// 2026-08-01 measurement in `platform::SystemCommit`.
pub fn final_commit_gate(
    cfg: &Config,
    platform: &dyn Platform,
    estimate: Option<&VramEstimate>,
    overridden: bool,
) -> Result<()> {
    if overridden {
        return Ok(());
    }
    let commit = platform.system_commit()?;
    let inflight = crate::supervise::inflight_reserved_bytes(&cfg.runs_dir);
    let add = estimate.map(|e| e.total_bytes).unwrap_or(0) + inflight;
    if add == 0 || commit.limit_bytes == 0 {
        return Ok(());
    }
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let projected = commit.charge_bytes + add;
    if projected as f64 > commit.limit_bytes as f64 * 0.90 {
        let inflight_note = if inflight > 0 {
            format!(" ({:.1} GiB of it reserved by a server still loading)", gib(inflight))
        } else {
            String::new()
        };
        return Err(Error::Platform(format!(
            "aborted at the final commit gate: conditions changed since pre-flight — projected commit \
             {:.1} GiB of {:.1} GiB limit exceeds 90%{inflight_note}. Stop another server or wait for one \
             to finish loading, then retry; --override-blocks proceeds anyway",
            gib(projected),
            gib(commit.limit_bytes)
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::Device;

    fn device(idx: u32, key: &str, integrated: bool) -> Device {
        Device {
            stable_key: key.into(),
            name: "AMD Radeon AI PRO R9700".into(),
            hip_index: idx,
            backend: "ROCm".into(),
            total_mib: 32624,
            free_mib: 32472,
            integrated,
            bus_number: Some(3),
            driver_version: Some("32.0.31035.1003".into()),
            display: None,
            luid_low: Some(0x1B592),
            correlation_assumed: false,
        }
    }

    fn worker_profile() -> Profile {
        serde_json::from_value(serde_json::json!({
            "schema": 1, "id": "worker-pool", "name": "Worker pool",
            "build": { "path": "C:/llama.cpp/build-hip-vision", "version": "b9817" },
            "model": { "path": "E:/models/model.gguf" },
            "devices": [ { "key": "pci:A:bus08" } ],
            "server": { "port": 9701, "alias": "gemma-4-worker" },
            "runtime": {
                "ctx_total": 393216, "slots": 6, "kv_type_k": "q8_0", "kv_type_v": "q8_0",
                "batch_logical": 2048, "batch_physical": 256, "flash_attn": "on"
            },
            "sampling": { "temperature": 1.0, "top_p": 0.95, "top_k": 64 },
            "chat": { "enable_thinking": false },
            "env": { "GPU_MAX_HW_QUEUES": "1", "ROCBLAS_USE_HIPBLASLT": "0" }
        }))
        .unwrap()
    }

    fn resolved_single() -> Vec<ResolvedDevice> {
        vec![ResolvedDevice {
            profile_key: "pci:A:bus08".into(),
            device: device(2, "pci:A:bus08", false),
            fraction: 1.0,
            rebound: false,
        }]
    }

    /// Parity with run-server.bat: the composed command must carry every
    /// flag the batch file passes for the worker-pool configuration.
    #[test]
    fn composes_worker_pool_equivalent_command() {
        let plan = compose(&worker_profile(), &resolved_single());
        let cmd = plan.command_line();
        for expected in [
            "-m E:/models/model.gguf",
            "-ngl 99",
            "-c 393216",
            "-np 6",
            "-fa on",
            "-ctk q8_0",
            "-ctv q8_0",
            "-b 2048",
            "-ub 256",
            "-cb",
            "--temp 1.0",
            "--top-k 64",
            "--top-p 0.95",
            "--jinja",
            "--metrics",
            "--alias gemma-4-worker",
            "--host 127.0.0.1",
            "--port 9701",
        ] {
            assert!(cmd.contains(expected), "missing {expected:?} in: {cmd}");
        }
        // Visibility pinned to the GLOBAL index (2), like the batch file.
        assert_eq!(plan.visibility_env, "2");
        assert!(plan.env.contains(&("HIP_VISIBLE_DEVICES".into(), "2".into())));
        assert!(plan.env.contains(&("GPU_MAX_HW_QUEUES".into(), "1".into())));
        assert!(plan
            .env
            .contains(&("LLAMA_ARG_CHAT_TEMPLATE_KWARGS".into(), r#"{"enable_thinking": false}"#.into())));
        // No split flags on a single device.
        assert!(!cmd.contains("--split-mode"));
        assert!(!cmd.contains("--tensor-split"));
    }

    #[test]
    fn composes_split_flags_in_remapped_order() {
        let mut p = worker_profile();
        p.devices = vec![
            serde_json::from_value(serde_json::json!({"key": "pci:A:bus03", "split_fraction": 0.6})).unwrap(),
            serde_json::from_value(serde_json::json!({"key": "pci:A:bus08", "split_fraction": 0.4})).unwrap(),
        ];
        p.split_mode = Some(SplitMode::Layer);
        p.main_device = 0;
        let resolved = vec![
            ResolvedDevice {
                profile_key: "pci:A:bus03".into(),
                device: device(0, "pci:A:bus03", false),
                fraction: 0.6,
                rebound: false,
            },
            ResolvedDevice {
                profile_key: "pci:A:bus08".into(),
                device: device(2, "pci:A:bus08", false),
                fraction: 0.4,
                rebound: false,
            },
        ];
        let plan = compose(&p, &resolved);
        let cmd = plan.command_line();
        // Global indices pinned; in-process remap makes them 0 and 1.
        assert_eq!(plan.visibility_env, "0,2");
        assert!(cmd.contains("--split-mode layer"));
        assert!(cmd.contains("--tensor-split 0.600,0.400"));
        assert!(cmd.contains("--main-gpu 0"), "main-gpu is in REMAPPED space: {cmd}");
    }

    #[test]
    fn extra_flags_pass_through_verbatim() {
        let mut p = worker_profile();
        p.runtime.extra_flags = vec![
            "--spec-type".into(),
            "draft-mtp".into(),
            "--some-future-flag".into(),
        ];
        let plan = compose(&p, &resolved_single());
        let cmd = plan.command_line();
        assert!(cmd.contains("--spec-type draft-mtp"));
        assert!(cmd.contains("--some-future-flag"));
    }

    fn diffusion_profile(env: serde_json::Value) -> Profile {
        serde_json::from_value(serde_json::json!({
            "schema": 1, "engine": "diffusion-gemma", "id": "dg-26b", "name": "DiffusionGemma",
            "build": { "path": "C:/fidim/b11027-mix-3e83366-unsloth", "version": "b11027" },
            "model": { "path": "E:/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf" },
            "devices": [ { "key": "pci:A:bus08" } ],
            "server": { "port": 9760, "alias": "diffusiongemma" },
            "runtime": { "ctx_total": 0, "n_gpu_layers": 99 },
            "diffusion": { "default_max_tokens": 1024, "seed": 7 },
            "chat": { "enable_thinking": false },
            "env": env
        }))
        .unwrap()
    }

    fn diffusion_paths(full_offload: bool) -> DiffusionPaths {
        DiffusionPaths {
            helper_exe: "C:/Programs/LlamaFIDIM/fidim-dg.exe".into(),
            runner_exe: "C:/fidim/b11027-mix-3e83366-unsloth/bin/llama-diffusion-gemma-visual-server.exe".into(),
            req_prefix: "C:/Users/u/.fidim/runs/dg-dg-26b-9760".into(),
            expect_bus: Some(8),
            build_tag: Some("b11027-mix-3e83366".into()),
            full_offload,
        }
    }

    #[test]
    fn compose_diffusion_golden() {
        let p = diffusion_profile(serde_json::json!({ "GPU_MAX_HW_QUEUES": "1" }));
        let plan = compose_diffusion(&p, &resolved_single(), &diffusion_paths(true));
        assert_eq!(plan.exe, PathBuf::from("C:/Programs/LlamaFIDIM/fidim-dg.exe"));
        assert_eq!(
            plan.args,
            [
                "--runner",
                "C:/fidim/b11027-mix-3e83366-unsloth/bin/llama-diffusion-gemma-visual-server.exe",
                "--model",
                "E:/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
                "--host",
                "127.0.0.1",
                "--port",
                "9760",
                "--alias",
                "diffusiongemma",
                "--req-prefix",
                "C:/Users/u/.fidim/runs/dg-dg-26b-9760",
                "--default-max-tokens",
                "1024",
                "--seed",
                "7",
                "--expect-bus",
                "8",
                "--build-tag",
                "b11027-mix-3e83366",
            ]
        );
        let env: Vec<(&str, &str)> = plan.env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        assert_eq!(
            env,
            [
                ("GPU_MAX_HW_QUEUES", "1"),
                ("HIP_VISIBLE_DEVICES", "2"),
                ("CUDA_VISIBLE_DEVICES", "2"),
                ("NGL", "99"),
                ("MAXTOK", "0"),
                ("FA", "0"),
                ("DG_FREE_RAM_MB", "0"),
                ("ROCBLAS_USE_HIPBLASLT", "0"),
                ("ROCBLAS_USE_HIPBLASLT_BATCHED", "0"),
                ("GPU_RESOURCE_CACHE_SIZE", "0"),
                ("DG_FRAME_SPECIAL", "1"),
            ]
        );
        // enable_thinking=false cannot reach the runner (validate warns).
        assert!(!plan.env.iter().any(|(k, _)| k == "LLAMA_ARG_CHAT_TEMPLATE_KWARGS"));
        assert_eq!(plan.path_prepend, None);
        assert_eq!(plan.visibility_env, "2");

        // profile.env wins for the hipBLASLt keys, whatever their case.
        let p = diffusion_profile(serde_json::json!({ "rocblas_use_hipblaslt": "1" }));
        let plan = compose_diffusion(&p, &resolved_single(), &diffusion_paths(false));
        let keys: Vec<&str> = plan.env.iter().map(|(k, _)| k.as_str()).collect();
        assert!(!keys.contains(&"ROCBLAS_USE_HIPBLASLT"), "{keys:?}");
        assert!(keys.contains(&"ROCBLAS_USE_HIPBLASLT_BATCHED"), "{keys:?}");
        // Partial offload: the runner may size against RAM, so no override.
        assert!(!keys.contains(&"DG_FREE_RAM_MB"), "{keys:?}");

        // Safeguard off, flash attention on, no seed/bus/tag.
        let mut p = diffusion_profile(serde_json::json!({}));
        p.diffusion = Some(crate::profile::DiffusionCfg {
            hipblaslt_safeguard: false,
            flash_attn: true,
            ..Default::default()
        });
        let d = DiffusionPaths { expect_bus: None, build_tag: None, ..diffusion_paths(true) };
        let plan = compose_diffusion(&p, &resolved_single(), &d);
        assert!(plan.env.contains(&("FA".into(), "1".into())));
        assert!(!plan.env.iter().any(|(k, _)| k.starts_with("ROCBLAS")));
        // The runtime cache goes off with or without the safeguard...
        assert!(plan.env.contains(&("GPU_RESOURCE_CACHE_SIZE".into(), "0".into())));
        // ...unless the profile env sets it, whatever its case.
        let mut p2 = p.clone();
        p2.env.insert("gpu_resource_cache_size".into(), "256".into());
        let plan2 = compose_diffusion(&p2, &resolved_single(), &d);
        let cache: Vec<&(String, String)> =
            plan2.env.iter().filter(|(k, _)| k.eq_ignore_ascii_case("GPU_RESOURCE_CACHE_SIZE")).collect();
        assert_eq!(cache, [&("gpu_resource_cache_size".to_string(), "256".to_string())]);
        let cmd = plan.command_line();
        assert!(cmd.contains("--default-max-tokens 2048") && !cmd.contains("--seed") && !cmd.contains("--expect-bus"), "{cmd}");
        assert_eq!(runner_exe(&p), PathBuf::from("C:/fidim/b11027-mix-3e83366-unsloth/bin/llama-diffusion-gemma-visual-server.exe"));

        // --build-tag names a patch, which shares its base's release tag.
        let mut meta = discovery::BuildMeta { release_tag: Some("b11027-mix-3e83366".into()), ..Default::default() };
        assert_eq!(dg_build_tag(&meta, Some("b11027")).as_deref(), Some("b11027-mix-3e83366"));
        meta.patch = Some(discovery::BuildPatch { name: "dgpatch".into(), ..Default::default() });
        assert_eq!(dg_build_tag(&meta, None).as_deref(), Some("b11027-mix-3e83366+dgpatch"));
        meta.patch = Some(discovery::BuildPatch::default());
        assert_eq!(dg_build_tag(&meta, None).as_deref(), Some("b11027-mix-3e83366+patched"));
        let bare = discovery::BuildMeta::default();
        assert_eq!(dg_build_tag(&bare, Some("b11027")).as_deref(), Some("b11027"));
        assert_eq!(dg_build_tag(&bare, None), None);
        assert_eq!(req_prefix(Path::new("C:/r"), &p), PathBuf::from("C:/r/dg-dg-26b-9760"));
    }

    #[test]
    fn port_free_detects_bound_port() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!port_is_free("127.0.0.1", port));
        drop(listener);
        assert!(port_is_free("127.0.0.1", port));
    }
}

#[cfg(test)]
mod aspm_tests {
    #[test]
    fn parses_powercfg_ac_index() {
        let text = concat!(
            "  Current AC Power Setting Index: 0x00000001", "
",
            "  Current DC Power Setting Index: 0x00000002", "
"
        );
        assert_eq!(super::parse_powercfg_ac_index(text), Some(1));
        assert_eq!(super::parse_powercfg_ac_index("nothing here"), None);
    }
}

#[cfg(test)]
mod bundle_tests {
    #[test]
    fn bundled_builds_are_recognised_from_the_server_path() {
        let dir = std::env::temp_dir().join(format!("fidim-bundle-{}", std::process::id()));
        let build = dir.join("b11027-mix-3e83366-unsloth");
        let exe = build.join("bin").join("llama-server.exe");
        std::fs::create_dir_all(build.join("bin")).unwrap();
        assert!(!super::bundles_runtime(&exe), "no manifest: an upstream build on the default runtime");
        std::fs::write(
            build.join(crate::update::MANIFEST_NAME),
            r#"{"channel":"unsloth","bundled_runtime":true,"release_tag":"b11027-mix-3e83366"}"#,
        )
        .unwrap();
        assert!(super::bundles_runtime(&exe));
        assert!(!super::bundles_runtime(std::path::Path::new("llama-server.exe")));
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod port_tests {
    #[test]
    fn parses_netstat_listener() {
        // `netstat -ano` (no -p): IPv4 and IPv6 TCP rows, then UDP rows.
        let text = "\nActive Connections\n\n  Proto  Local Address          Foreign Address        State           PID\n\
            \x20 TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1234\n\
            \x20 TCP    127.0.0.1:9701         0.0.0.0:0              LISTENING       3776\n\
            \x20 TCP    127.0.0.1:9701         127.0.0.1:5000         ESTABLISHED     3776\n\
            \x20 TCP    [::1]:9701             [::]:0                 LISTENING       4100\n\
            \x20 TCP    [::1]:9701             [::1]:50202            ESTABLISHED     4100\n\
            \x20 UDP    0.0.0.0:9701           *:*                                    5436\n";
        assert_eq!(super::parse_netstat_listener(text, 9701), Some(3776));
        assert_eq!(super::parse_netstat_listener(text, 9702), None);
        assert_eq!(super::parse_netstat_listeners(text, 9701), vec![3776, 4100]);
        assert!(super::parse_netstat_listeners(text, 970).is_empty());
        // A helper bound to "localhost" listens on IPv6 only.
        let v6 = "  TCP    [::]:135               [::]:0                 LISTENING       2196\n\
                  \x20 TCP    [::1]:18766            [::]:0                 LISTENING       4100\n";
        assert_eq!(super::parse_netstat_listeners(v6, 18766), vec![4100]);
        // The state column is translated on non-English Windows.
        let de = "  TCP    127.0.0.1:9701         0.0.0.0:0              ABH\u{00d6}REN         3776\n\
                  \x20 TCP    127.0.0.1:9701         127.0.0.1:5000         HERGESTELLT     3776\n";
        assert_eq!(super::parse_netstat_listeners(de, 9701), vec![3776]);
    }

    /// The real netstat must see an IPv6 listener: a diffusion helper bound
    /// to "localhost" or "::1" otherwise reads as CRASHED and is never stopped.
    #[cfg(windows)]
    #[test]
    fn listens_on_sees_ipv4_and_ipv6_listeners() {
        let me = std::process::id();
        let v4 = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        assert!(super::listens_on(v4.local_addr().unwrap().port(), me));
        let Ok(v6) = std::net::TcpListener::bind("[::1]:0") else { return }; // no IPv6 stack
        let port = v6.local_addr().unwrap().port();
        assert!(super::listens_on(port, me), "[::1]:{port} is not seen");
        assert!(!super::listens_on(port, me.wrapping_add(1)));
    }
}
