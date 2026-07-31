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
use crate::estimate::{self, EstimateInput, VramEstimate};
use crate::gguf;
use crate::platform::Platform;
use crate::preflight::{LaunchContext, ResolvedDevice};
use crate::profile::{Profile, SplitMode};
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
            // Speculative-decoding specifics (--spec-type etc.) are
            // build-dependent; they belong in runtime.extra_flags.
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

    args.push("--jinja".into());
    args.push("--metrics".into());
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
    let listed_text = run_capture(server_exe, &["--list-devices"], cfg.rocm_bin.as_deref())?;
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

pub fn run_capture(exe: &Path, args: &[&str], rocm_bin: Option<&Path>) -> Result<String> {
    let mut cmd = std::process::Command::new(exe);
    cmd.args(args);
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

/// Build the full pre-flight context for a profile: resolve devices, compute
/// fractions, estimate VRAM, read commit, probe the build, check the port.
pub fn prepare(
    cfg: &Config,
    profile: &Profile,
    platform: &dyn Platform,
    running_aliases: Vec<String>,
) -> Result<PreparedLaunch> {
    // Build probe.
    let server_exe = profile.build.path.join("bin").join("llama-server.exe");
    let build_version_output = run_capture(&server_exe, &["--version"], cfg.rocm_bin.as_deref())
        .ok()
        .filter(|t| discovery::parse_version_output(t).is_some());

    // File existence (check 2).
    let mut missing = Vec::new();
    let mut check_file = |p: &Path| {
        if !p.is_file() {
            missing.push(p.to_string_lossy().into_owned());
        }
    };
    check_file(&profile.model.path);
    if let Some(mm) = &profile.model.mmproj {
        check_file(mm);
    }
    if let Some(d) = &profile.model.draft {
        if d.enabled {
            check_file(&d.path);
        }
    }

    // Device resolution (check 3) against a fresh enumeration.
    let devices_now = if build_version_output.is_some() {
        enumerate_devices(cfg, &server_exe, platform)?
    } else {
        Vec::new()
    };
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

    // VRAM estimate (check 6).
    let estimate: Option<VramEstimate> = gguf::read_header(&profile.model.path).ok().map(|h| {
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
            header: &h,
            runtime: &profile.runtime,
            devices: resolved.iter().map(|r| (r.profile_key.clone(), r.fraction)).collect(),
            split_mode: profile.split_mode,
            main_index: (profile.main_device as usize).min(resolved.len().saturating_sub(1)),
            mmproj_bytes,
            draft_bytes,
        })
    });

    let commit = platform.system_commit()?;
    let port_free = port_is_free(&profile.server.host, profile.server.port);

    // Driver/SDK now vs baseline (check 11).
    let driver_now = resolved.iter().find_map(|r| r.device.driver_version.clone());
    let sdk_now = sdk_version(cfg.rocm_bin.as_deref());
    let baseline = profile.baseline.as_ref();
    let driver_baseline =
        baseline.and_then(|b| b.get("driver")).and_then(|v| v.as_str()).map(String::from);
    let sdk_baseline =
        baseline.and_then(|b| b.get("sdk")).and_then(|v| v.as_str()).map(String::from);

    let plan = compose(profile, &resolved);
    let mut plan = plan;
    plan.path_prepend = cfg.rocm_bin.clone();

    let context = LaunchContext {
        profile: profile.clone(),
        build_version_output,
        missing_files: missing,
        resolved,
        unresolved,
        estimate,
        commit,
        visibility_env: plan.visibility_env.clone(),
        running_aliases,
        port_free,
        driver_now,
        driver_baseline,
        sdk_now,
        sdk_baseline,
    };
    Ok(PreparedLaunch { context, plan, devices_now })
}

pub fn port_is_free(host: &str, port: u16) -> bool {
    TcpListener::bind((host, port)).is_ok()
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

    #[test]
    fn port_free_detects_bound_port() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!port_is_free("127.0.0.1", port));
        drop(listener);
        assert!(port_is_free("127.0.0.1", port));
    }
}
