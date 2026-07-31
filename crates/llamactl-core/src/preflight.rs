//! Pre-flight check sequence (spec §04, v1.1). Every check exists because the
//! corresponding failure happened on the target machine and cost hours.
//!
//! Checks are pure functions over a `LaunchContext` assembled by the launch
//! path — the same objects power the profile editor's live validation, so the
//! editor and the launcher can never disagree. Any Block prevents launch; the
//! user may override with an explicit confirmation that names the risk.

use serde::Serialize;

use crate::devices::Device;
use crate::estimate::VramEstimate;
use crate::platform::SystemCommit;
use crate::profile::Profile;

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

/// A device the launch path resolved from a profile key.
#[derive(Debug, Clone, Serialize)]
pub struct ResolvedDevice {
    pub profile_key: String,
    pub device: Device,
    pub fraction: f64,
    /// Set when resolution re-bound a stale key (renumbering event).
    pub rebound: bool,
}

/// Everything §04 needs, assembled by the launch path (or a fake in tests).
#[derive(Debug, Clone, Serialize)]
pub struct LaunchContext {
    pub profile: Profile,
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
    /// The exact `HIP_VISIBLE_DEVICES` value the launcher will set.
    pub visibility_env: String,
    /// Aliases of currently-running servers.
    pub running_aliases: Vec<String>,
    pub port_free: bool,
    /// Driver/SDK now vs the profile baseline's recorded versions.
    pub driver_now: Option<String>,
    pub driver_baseline: Option<String>,
    pub sdk_now: Option<String>,
    pub sdk_baseline: Option<String>,
}

pub fn run_all(ctx: &LaunchContext) -> Vec<CheckResult> {
    vec![
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
    ]
}

pub fn any_block(results: &[CheckResult]) -> bool {
    results.iter().any(|r| r.blocks())
}

// ---------------------------------------------------------------- checks ----

fn check_build(ctx: &LaunchContext) -> CheckResult {
    let outcome = match &ctx.build_version_output {
        Some(_) => Outcome::Pass,
        None => Outcome::Block(format!(
            "build binary {} did not run — reinstall or rescan builds",
            ctx.profile.build.path.display()
        )),
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
    } else {
        Outcome::Block(format!(
            "target resolves to integrated graphics: {} — on this machine the iGPU sits at index 1 \
             between the two discrete cards; binding it runs inference at ~1/20th throughput with no error",
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
    let outcome = if ctx.resolved.is_empty() {
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
    let mut worst: Option<(f64, String)> = None;
    for (d, r) in est.per_device.iter().zip(&ctx.resolved) {
        let free_bytes = r.device.free_mib as f64 * 1024.0 * 1024.0;
        if free_bytes <= 0.0 {
            continue;
        }
        let ratio = d.total_bytes as f64 / free_bytes;
        let msg = format!(
            "{}: estimated {} vs {:.2} GiB free ({:.0}%)",
            r.device.name,
            d.breakdown(),
            free_bytes / (1024.0 * 1024.0 * 1024.0),
            ratio * 100.0
        );
        if worst.as_ref().map(|(w, _)| ratio > *w).unwrap_or(true) {
            worst = Some((ratio, msg));
        }
    }
    let outcome = match worst {
        Some((r, msg)) if r > 1.0 => Outcome::Block(format!(
            "{msg} — spill costs ~3.4x prefill and presents as \"the model got slow\", not an error"
        )),
        Some((r, msg)) if r > 0.90 => Outcome::Warn(format!(
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

fn check_commit(ctx: &LaunchContext) -> CheckResult {
    let projected = ctx.commit.charge_bytes
        + ctx.estimate.as_ref().map(|e| e.total_bytes).unwrap_or(0);
    let limit = ctx.commit.limit_bytes.max(1);
    let frac = projected as f64 / limit as f64;
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let outcome = if frac > 0.90 {
        Outcome::Block(format!(
            "projected commit {:.1} GiB of {:.1} GiB limit ({:.0}%) — past ~90% Windows evicts GPU \
             allocations to the pagefile: the server stays up but decode collapses (~50x observed) with \
             nothing in any log. Remedy: enlarge the pagefile (Settings > System > About > Advanced \
             system settings > Performance > Virtual memory) — needs no reboot to grow",
            gib(projected),
            gib(limit),
            frac * 100.0
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

fn check_port(ctx: &LaunchContext) -> CheckResult {
    let outcome = if ctx.port_free {
        Outcome::Pass
    } else {
        Outcome::Block(format!("port {} is already in use", ctx.profile.server.port))
    };
    CheckResult { id: "port-free", spec_number: 8, title: "Requested port is free", outcome }
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
             preempted compute in the past; move the cable to the iGPU outputs for clean numbers",
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
            },
            OsAdapter {
                name: "AMD Radeon AI PRO R9700".into(),
                pnp_device_id: r"PCI\VEN_1002&DEV_7551&SUBSYS_54131849&REV_C0\6&2&0&09".into(),
                driver_version: "32.0.31035.1003".into(),
                bus_number: Some(3),
                display: Some(DisplayMode { width: 2560, height: 1440, refresh_hz: 144 }),
            },
            OsAdapter {
                name: "AMD Radeon AI PRO R9700".into(),
                pnp_device_id: r"PCI\VEN_1002&DEV_7551&SUBSYS_54131849&REV_C0\8&3&0&11".into(),
                driver_version: "32.0.31035.1003".into(),
                bus_number: Some(8),
                display: None,
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
            profile: profile(),
            build_version_output: Some("version: 9817 (5397c3619)".into()),
            missing_files: vec![],
            resolved: vec![resolved_for(2, 1.0)],
            unresolved: vec![],
            estimate: None,
            commit: SystemCommit {
                limit_bytes: 100 * 1024 * 1024 * 1024,
                charge_bytes: 30 * 1024 * 1024 * 1024,
            },
            visibility_env: "2".into(),
            running_aliases: vec![],
            port_free: true,
            driver_now: Some("32.0.31035.1003".into()),
            driver_baseline: None,
            sdk_now: None,
            sdk_baseline: None,
        }
    }

    #[test]
    fn healthy_context_passes_everything() {
        let results = run_all(&healthy_ctx());
        assert_eq!(results.len(), 11);
        assert!(!any_block(&results), "{results:?}");
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
}
