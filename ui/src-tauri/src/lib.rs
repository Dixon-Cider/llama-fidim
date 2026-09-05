//! Tauri command layer: thin wrappers over llamactl-core. The GUI runs the
//! SAME check objects and launch path as the CLI — the editor and the
//! launcher can never disagree (spec V-2).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use llamactl_core::config::Config;
use llamactl_core::devices::Device;
use llamactl_core::launch::{self, PrepareInputs};
use llamactl_core::platform::{Platform, WindowsPlatform};
use llamactl_core::profile::{self, Profile};
use llamactl_core::supervise;
use llamactl_core::update::{self, PromoteScope};
use llamactl_core::{bench, discovery, export, preflight};
use tauri::Emitter;

/// Cached slow inputs for live pre-flight (device enumeration ~2-4s, build
/// probe ~1s). Real launches never read this cache — they enumerate fresh
/// per R-03.
struct UiCache {
    devices: Option<(Instant, Vec<Device>)>,
    build_probes: HashMap<PathBuf, Option<String>>,
}

struct AppState {
    /// Arc so command bodies can move a handle onto the blocking pool.
    /// Everything touching WMI MUST run there: Tauri's main thread holds
    /// STA COM for WebView2, and CoInitializeEx(MTA) on it fails with
    /// RPC_E_CHANGED_MODE (observed live).
    cache: Arc<Mutex<UiCache>>,
}

const DEVICE_TTL: Duration = Duration::from_secs(15);

fn cfg() -> Result<Config, String> {
    Config::load_or_init().map_err(|e| e.to_string())
}

/// Run `f` on the async runtime's blocking pool and wait for it.
async fn blocking<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(|e| e.to_string())?
}

fn cached_devices(
    cache_arc: &Arc<Mutex<UiCache>>,
    cfg: &Config,
    refresh: bool,
) -> Result<Vec<Device>, String> {
    {
        let cache = cache_arc.lock().unwrap();
        if !refresh {
            if let Some((at, devices)) = &cache.devices {
                if at.elapsed() < DEVICE_TTL {
                    return Ok(devices.clone());
                }
            }
        }
    }
    let builds = discovery::scan_builds(&cfg.build_roots, cfg.rocm_bin.as_deref());
    let build = builds
        .iter()
        .filter(|b| b.version.is_some())
        .max_by(|a, b| a.version.cmp(&b.version))
        .or(builds.first())
        .ok_or("no builds found under configured build_roots")?;
    let devices = launch::enumerate_devices(cfg, &build.server_exe, &WindowsPlatform)
        .map_err(|e| e.to_string())?;
    let mut cache = cache_arc.lock().unwrap();
    cache.devices = Some((Instant::now(), devices.clone()));
    Ok(devices)
}

fn cached_build_probe(
    cache_arc: &Arc<Mutex<UiCache>>,
    cfg: &Config,
    build_path: &PathBuf,
) -> Option<String> {
    {
        let cache = cache_arc.lock().unwrap();
        if let Some(v) = cache.build_probes.get(build_path) {
            return v.clone();
        }
    }
    let exe = build_path.join("bin").join("llama-server.exe");
    let probe = launch::run_capture(&exe, &["--version"], cfg.rocm_bin.as_deref())
        .ok()
        .filter(|t| discovery::parse_version_output(t).is_some());
    let mut cache = cache_arc.lock().unwrap();
    cache.build_probes.insert(build_path.clone(), probe.clone());
    probe
}

fn running_aliases(cfg: &Config) -> Vec<String> {
    supervise::reattach(&cfg.runs_dir)
        .into_iter()
        .filter(|r| r.alive)
        .map(|r| r.state.alias)
        .collect()
}

// ---------------------------------------------------------------- commands ----

#[tauri::command]
fn scan() -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    let builds = discovery::scan_builds(&cfg.build_roots, cfg.rocm_bin.as_deref());
    let models = discovery::scan_models(&cfg.model_roots);
    Ok(serde_json::json!({ "builds": builds, "models": models }))
}

#[tauri::command]
async fn devices(
    state: tauri::State<'_, AppState>,
    refresh: bool,
) -> Result<serde_json::Value, String> {
    let cache = state.cache.clone();
    blocking(move || devices_blocking(&cache, refresh)).await
}

fn devices_blocking(
    cache: &Arc<Mutex<UiCache>>,
    refresh: bool,
) -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    let devices = cached_devices(cache, &cfg, refresh)?;
    let runs = supervise::reattach(&cfg.runs_dir);
    let occupancy: Vec<serde_json::Value> = devices
        .iter()
        .map(|d| {
            let by: Vec<&str> = runs
                .iter()
                .filter(|r| r.alive && r.state.device_keys.contains(&d.stable_key))
                .map(|r| r.state.profile_id.as_str())
                .collect();
            serde_json::json!({ "device": d, "occupied_by": by })
        })
        .collect();
    Ok(serde_json::Value::Array(occupancy))
}

#[tauri::command]
fn list_profiles() -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    let profiles = Profile::load_all(&cfg.profile_dir).map_err(|e| e.to_string())?;
    let rows: Vec<serde_json::Value> = profiles
        .iter()
        .map(|p| {
            serde_json::json!({
                "profile": p,
                "findings": profile::validate(p),
            })
        })
        .collect();
    Ok(serde_json::Value::Array(rows))
}

#[tauri::command]
fn save_profile(p: Profile) -> Result<(), String> {
    let cfg = cfg()?;
    p.save(&cfg.profile_dir.join(format!("{}.json", p.id))).map_err(|e| e.to_string())
}

#[tauri::command]
fn delete_profile(id: String) -> Result<(), String> {
    let cfg = cfg()?;
    let path = cfg.profile_dir.join(format!("{id}.json"));
    std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Live pre-flight for the editor: validation + full check pipeline against
/// cached device enumeration. Returns everything the editor renders.
#[tauri::command]
async fn live_check(
    state: tauri::State<'_, AppState>,
    p: Profile,
) -> Result<serde_json::Value, String> {
    let cache = state.cache.clone();
    blocking(move || live_check_blocking(&cache, p)).await
}

fn live_check_blocking(
    cache: &Arc<Mutex<UiCache>>,
    p: Profile,
) -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    let findings = profile::validate(&p);
    let devices_now = cached_devices(cache, &cfg, false).unwrap_or_default();
    let build_version_output = cached_build_probe(cache, &cfg, &p.build.path);
    let inputs = PrepareInputs { devices_now, build_version_output };
    let prepared =
        launch::prepare_with_inputs(&cfg, &p, &WindowsPlatform, running_aliases(&cfg), inputs)
            .map_err(|e| e.to_string())?;
    let results = preflight::run_all(&prepared.context);
    Ok(serde_json::json!({
        "findings": findings,
        "results": results,
        "estimate": prepared.context.estimate,
        "resolved": prepared.context.resolved,
        "command_line": prepared.plan.command_line(),
        "env": prepared.plan.env,
        "commit": prepared.context.commit,
    }))
}

/// Launch: full fresh prepare (R-03), spawn detached, wait ready, verify
/// placement via PDH. Long-running — async so the UI stays responsive.
#[tauri::command]
async fn launch_profile(id: String, override_blocks: bool) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || do_launch(&id, override_blocks))
        .await
        .map_err(|e| e.to_string())?
}

fn do_launch(id: &str, override_blocks: bool) -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    let platform = WindowsPlatform;
    let path = cfg.profile_dir.join(format!("{id}.json"));
    let mut profile = Profile::load(&path).map_err(|e| e.to_string())?;
    let findings = profile::validate(&profile);
    if findings.iter().any(|f| f.severity == profile::Severity::Error) {
        return Err("profile has validation errors — fix them in the editor first".into());
    }
    let prepared = launch::prepare(&cfg, &profile, &platform, running_aliases(&cfg))
        .map_err(|e| e.to_string())?;
    let results = preflight::run_all(&prepared.context);
    if preflight::any_block(&results) && !override_blocks {
        return Ok(serde_json::json!({ "blocked": true, "results": results }));
    }
    // Persist rebound keys + informational indices.
    let mut changed = false;
    for (dref, res) in profile.devices.iter_mut().zip(&prepared.context.resolved) {
        if res.rebound {
            dref.key = res.device.stable_key.clone();
            changed = true;
        }
        if dref.resolved_index_last_launch != Some(res.device.hip_index) {
            dref.resolved_index_last_launch = Some(res.device.hip_index);
            changed = true;
        }
    }
    if changed {
        profile.save(&path).map_err(|e| e.to_string())?;
    }
    let device_keys: Vec<String> =
        prepared.context.resolved.iter().map(|r| r.device.stable_key.clone()).collect();
    let free_before: Vec<u64> =
        prepared.context.resolved.iter().map(|r| r.device.free_mib).collect();
    let cold = supervise::is_cold_start(&Config::config_dir(), &profile.model.path);
    launch::final_commit_gate(
        &cfg,
        &platform,
        prepared.context.estimate.as_ref(),
        override_blocks,
    )
    .map_err(|e| e.to_string())?;
    let state = supervise::spawn(
        &prepared.plan,
        &profile,
        &cfg.runs_dir,
        device_keys,
        free_before,
        cold,
        prepared.context.estimate.as_ref().map(|e| e.total_bytes).unwrap_or(0),
    )
    .map_err(|e| e.to_string())?;
    supervise::wait_ready_in(&state, Duration::from_secs(420), Some(&cfg.runs_dir))
        .map_err(|e| e.to_string())?;
    supervise::record_model_loaded(&Config::config_dir(), &profile.model.path)
        .map_err(|e| e.to_string())?;

    // Placement verification (committed proves placement; dedicated fills on
    // first inference).
    let mem = platform.gpu_process_memory(state.pid).unwrap_or_default();
    let placement: Vec<serde_json::Value> = prepared
        .context
        .resolved
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let expected = prepared
                .context
                .estimate
                .as_ref()
                .and_then(|e| e.per_device.get(i))
                .map(|d| d.total_bytes)
                .unwrap_or(0);
            let m = r.device.luid_low.and_then(|l| mem.iter().find(|x| x.luid_low == l));
            serde_json::json!({
                "key": r.device.stable_key,
                "expected_bytes": expected,
                "dedicated_bytes": m.map(|x| x.dedicated_bytes),
                "committed_bytes": m.map(|x| x.committed_bytes),
            })
        })
        .collect();
    Ok(serde_json::json!({
        "blocked": false,
        "results": results,
        "state": state,
        "cold_start": cold,
        "placement": placement,
    }))
}

#[tauri::command]
async fn status(deep: bool) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let mut runs = supervise::reattach(&cfg.runs_dir);
        if deep {
            for r in &mut runs {
                if r.alive {
                    r.health = supervise::probe_health(&r.state, true);
                }
            }
        }
        serde_json::to_value(&runs).map_err(|e| e.to_string())
    })
    .await
}

/// Slot occupancy from the server's own /slots endpoint (V-3).
#[tauri::command]
fn slots(port: u16, host: String) -> Result<serde_json::Value, String> {
    match supervise::http_get(&host, port, "/slots", Duration::from_secs(3)) {
        Ok((200, body)) => {
            let start = body.find('[').unwrap_or(0);
            let end = body.rfind(']').map(|i| i + 1).unwrap_or(body.len());
            serde_json::from_str(&body[start..end]).map_err(|e| e.to_string())
        }
        Ok((code, _)) => Err(format!("HTTP {code}")),
        Err(e) => Err(e.to_string()),
    }
}

#[tauri::command]
fn stop_run(target: String) -> Result<(), String> {
    let cfg = cfg()?;
    let runs = supervise::reattach(&cfg.runs_dir);
    let run = runs
        .into_iter()
        .find(|r| r.state.profile_id == target || r.state.port.to_string() == target)
        .ok_or(format!("no run matches {target}"))?;
    supervise::stop(&run.state, &cfg.runs_dir).map_err(|e| e.to_string())
}

#[tauri::command]
fn read_log(target: String, tail: usize, filter: String) -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    let runs = supervise::reattach(&cfg.runs_dir);
    let run = runs
        .into_iter()
        .find(|r| r.state.profile_id == target || r.state.port.to_string() == target)
        .ok_or(format!("no run matches {target}"))?;
    let text = std::fs::read_to_string(&run.state.log_path).unwrap_or_default();
    let lines: Vec<&str> = text
        .lines()
        .filter(|l| filter.is_empty() || l.to_lowercase().contains(&filter.to_lowercase()))
        .collect();
    let tail_lines: Vec<&str> = lines.iter().rev().take(tail).rev().copied().collect();
    Ok(serde_json::json!({
        "path": run.state.log_path,
        "lines": tail_lines,
        "total_matching": lines.len(),
    }))
}

/// List all runs (alive + crashed) whose logs exist, for the Logs view.
#[tauri::command]
fn log_sources() -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    let runs = supervise::reattach(&cfg.runs_dir);
    let rows: Vec<serde_json::Value> = runs
        .iter()
        .map(|r| {
            serde_json::json!({
                "profile_id": r.state.profile_id,
                "port": r.state.port,
                "alive": r.alive,
                "log_path": r.state.log_path,
            })
        })
        .collect();
    Ok(serde_json::Value::Array(rows))
}

#[tauri::command]
async fn bench_profile(
    id: String,
    concurrency: Option<u32>,
    tokens: u32,
) -> Result<serde_json::Value, String> {
    tauri::async_runtime::spawn_blocking(move || do_bench(&id, concurrency, tokens))
        .await
        .map_err(|e| e.to_string())?
}

fn do_bench(id: &str, concurrency: Option<u32>, tokens: u32) -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    let platform = WindowsPlatform;
    let profile =
        Profile::load(&cfg.profile_dir.join(format!("{id}.json"))).map_err(|e| e.to_string())?;
    let runs = supervise::reattach(&cfg.runs_dir);
    let run = runs
        .into_iter()
        .find(|r| r.state.profile_id == id && r.alive)
        .ok_or("no live run for this profile — launch it first")?;
    let opts = bench::BenchOptions {
        warmups: 2,
        max_tokens: tokens,
        concurrency: concurrency.unwrap_or(profile.runtime.slots).max(1),
        ..Default::default()
    };
    let sweep = bench::run_sweep(&run.state.host, run.state.port, &run.state.alias, &opts)
        .map_err(|e| e.to_string())?;

    let mem = platform.gpu_process_memory(run.state.pid).unwrap_or_default();
    let adapters = platform.video_adapters().unwrap_or_default();
    let per_device_vram_gb: Vec<f64> = run
        .state
        .device_keys
        .iter()
        .map(|key| {
            adapters
                .iter()
                .find(|a| {
                    llamactl_core::devices::stable_key(&a.pnp_device_id, a.bus_number) == *key
                })
                .and_then(|a| a.luid_low)
                .and_then(|l| mem.iter().find(|m| m.luid_low == l))
                .map(|m| {
                    m.dedicated_bytes.max(m.committed_bytes) as f64 / (1024.0 * 1024.0 * 1024.0)
                })
                .unwrap_or(0.0)
        })
        .collect();
    let vram_gb: f64 = per_device_vram_gb.iter().sum();
    let serial = sweep.serial.decode_tok_s.unwrap_or(sweep.serial.wall_tok_s);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let driver = adapters
        .iter()
        .find(|a| {
            run.state.device_keys.iter().any(|k| {
                llamactl_core::devices::stable_key(&a.pnp_device_id, a.bus_number) == *k
            })
        })
        .map(|a| a.driver_version.clone());
    let baseline = serde_json::json!({
        "measured_at": bench::iso8601_utc(now),
        "measured_at_unix": now,
        "build_version": profile.build.version,
        "driver": driver,
        "sdk": launch::sdk_version(cfg.rocm_bin.as_deref()),
        "split": profile.split_mode.map(|m| serde_json::json!({
            "mode": m,
            "fractions": profile.devices.iter().map(|d| d.split_fraction).collect::<Vec<_>>(),
        })),
        "vram_gb": (vram_gb * 100.0).round() / 100.0,
        "per_device_vram_gb": per_device_vram_gb.iter().map(|g| (g * 100.0).round() / 100.0).collect::<Vec<_>>(),
        "serial_tok_s": (serial * 10.0).round() / 10.0,
        "concurrent": sweep.concurrent.as_ref().map(|c| serde_json::json!({
            "n": c.n,
            "aggregate_tok_s": (c.aggregate_tok_s * 10.0).round() / 10.0,
            "decode_aggregate_tok_s": (c.decode_aggregate_tok_s * 10.0).round() / 10.0,
            "per_stream_tok_s": (c.per_stream_tok_s * 10.0).round() / 10.0,
        })),
        "cold_cache": run.state.cold_start,
        "profile_fingerprint": bench::profile_fingerprint(&profile),
        "bench": { "warmups": 2, "tokens": tokens },
    });
    let hist_dir = Config::config_dir().join("benchmarks");
    std::fs::create_dir_all(&hist_dir).map_err(|e| e.to_string())?;
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(hist_dir.join(format!("{id}.jsonl")))
        .map_err(|e| e.to_string())?;
    writeln!(f, "{baseline}").map_err(|e| e.to_string())?;
    let saved = if run.state.cold_start {
        false
    } else {
        let mut profile = profile;
        profile.baseline = Some(baseline.clone());
        profile
            .save(&cfg.profile_dir.join(format!("{id}.json")))
            .map_err(|e| e.to_string())?;
        true
    };
    Ok(serde_json::json!({ "baseline": baseline, "saved": saved, "cold": run.state.cold_start }))
}

#[tauri::command]
fn bench_history(id: String) -> Result<serde_json::Value, String> {
    let path = Config::config_dir().join("benchmarks").join(format!("{id}.jsonl"));
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let rows: Vec<serde_json::Value> =
        text.lines().filter_map(|l| serde_json::from_str(l).ok()).collect();
    Ok(serde_json::Value::Array(rows))
}

#[tauri::command]
async fn export_profile(id: String, format: String) -> Result<serde_json::Value, String> {
    blocking(move || export_blocking(&id, &format)).await
}

fn export_blocking(id: &str, format: &str) -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    let platform = WindowsPlatform;
    let profile =
        Profile::load(&cfg.profile_dir.join(format!("{id}.json"))).map_err(|e| e.to_string())?;
    let prepared = launch::prepare(&cfg, &profile, &platform, running_aliases(&cfg))
        .map_err(|e| e.to_string())?;
    let (text, ext) = match format {
        "ps1" => (export::to_ps1(&profile, &prepared.plan), "ps1"),
        _ => (export::to_bat(&profile, &prepared.plan), "bat"),
    };
    let out = cfg.profile_dir.join(format!("{id}.{ext}"));
    std::fs::write(&out, &text).map_err(|e| e.to_string())?;
    Ok(serde_json::json!({ "path": out, "text": text }))
}

// ---------------------------------------------------------------- updates ----

/// Latest upstream release vs newest installed build. Network + a build
/// probe, so it runs on the blocking pool.
#[tauri::command]
async fn update_check() -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let c = update::check(&cfg).map_err(|e| e.to_string())?;
        serde_json::to_value(c).map_err(|e| e.to_string())
    })
    .await
}

/// Install `tag` (latest when None) as prebuilt or from source. Progress
/// lines stream to the window as `update-progress` events. Never launches.
#[tauri::command]
async fn update_install(
    app: tauri::AppHandle,
    tag: Option<String>,
    source: bool,
) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let release = match &tag {
            Some(t) => update::release_by_tag(t),
            None => update::latest_release(),
        }
        .map_err(|e| e.to_string())?;
        let mut progress = |line: String| {
            let _ = app.emit("update-progress", line);
        };
        let r = if source {
            update::build_from_source(&cfg, &release.tag, &mut progress)
        } else {
            update::install_prebuilt(&cfg, &release, &mut progress)
        }
        .map_err(|e| e.to_string())?;
        serde_json::to_value(r).map_err(|e| e.to_string())
    })
    .await
}

/// Re-point profiles onto an installed build directory. `from_path` limits
/// the scope to profiles on that build; `all` moves every unpinned profile.
#[tauri::command]
async fn update_promote(
    to_path: String,
    to_version: Option<String>,
    from_path: Option<String>,
    all: bool,
) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let scope = if all {
            PromoteScope::All
        } else if let Some(f) = from_path {
            PromoteScope::FromBuild(PathBuf::from(f))
        } else {
            PromoteScope::All
        };
        let r = update::promote(&cfg, &PathBuf::from(to_path), to_version, scope).map_err(|e| e.to_string())?;
        serde_json::to_value(r).map_err(|e| e.to_string())
    })
    .await
}

#[tauri::command]
async fn update_rollback() -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let r = update::rollback(&cfg).map_err(|e| e.to_string())?;
        serde_json::to_value(r).map_err(|e| e.to_string())
    })
    .await
}

/// ROCm runtimes a profile can name. Filesystem probing only.
#[tauri::command]
fn list_runtimes() -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    serde_json::to_value(llamactl_core::runtime::discover(&cfg)).map_err(|e| e.to_string())
}

#[tauri::command]
fn update_history() -> Result<serde_json::Value, String> {
    let h = update::load_history().map_err(|e| e.to_string())?;
    serde_json::to_value(h).map_err(|e| e.to_string())
}

pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            cache: Arc::new(Mutex::new(UiCache { devices: None, build_probes: HashMap::new() })),
        })
        .invoke_handler(tauri::generate_handler![
            scan,
            devices,
            list_profiles,
            save_profile,
            delete_profile,
            live_check,
            launch_profile,
            status,
            slots,
            stop_run,
            read_log,
            log_sources,
            bench_profile,
            bench_history,
            export_profile,
            update_check,
            update_install,
            update_promote,
            update_rollback,
            update_history,
            list_runtimes,
        ])
        .run(tauri::generate_context!())
        .expect("error while running llamactl UI");
}
