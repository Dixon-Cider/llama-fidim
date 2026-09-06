//! Tauri command layer: thin wrappers over fidim-core. The GUI runs the
//! SAME check objects and launch path as the CLI — the editor and the
//! launcher can never disagree (spec V-2).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fidim_core::config::Config;
use fidim_core::devices::Device;
use fidim_core::launch::{self, PrepareInputs};
use fidim_core::platform::{Platform, WindowsPlatform};
use fidim_core::profile::{self, Profile};
use fidim_core::supervise;
use fidim_core::update::{self, PromoteScope};
use fidim_core::{bench, discovery, export, preflight};
use tauri::Emitter;

/// Cached slow inputs for live pre-flight (device enumeration ~2-4s, build
/// probe ~1s). Real launches never read this cache — they enumerate fresh
/// per R-03.
struct UiCache {
    devices: Option<(Instant, Vec<Device>)>,
    build_probes: HashMap<PathBuf, Option<String>>,
    /// Build + model scan; probing every build costs ~0.3 s each.
    scan: Option<(Instant, serde_json::Value)>,
}

const SCAN_TTL: Duration = Duration::from_secs(60);

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
    let builds = discovery::scan_builds(&cfg.build_roots_effective(), cfg.rocm_bin.as_deref());
    // Newest by release NUMBER — a string compare ranks b9817 above b10771.
    let build = builds
        .iter()
        .filter(|b| b.version.is_some())
        .max_by_key(|b| b.version.as_deref().and_then(fidim_core::update::version_number).unwrap_or(0))
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
async fn scan(state: tauri::State<'_, AppState>, refresh: Option<bool>) -> Result<serde_json::Value, String> {
    let cache = state.cache.clone();
    let refresh = refresh.unwrap_or(false);
    blocking(move || {
        if !refresh {
            let c = cache.lock().unwrap();
            if let Some((at, v)) = &c.scan {
                if at.elapsed() < SCAN_TTL {
                    return Ok(v.clone());
                }
            }
        }
        let cfg = cfg()?;
        let builds = discovery::scan_builds(&cfg.build_roots_effective(), cfg.rocm_bin.as_deref());
        let models = discovery::scan_models(&cfg.model_roots);
        let v = serde_json::json!({ "builds": builds, "models": models });
        cache.lock().unwrap().scan = Some((Instant::now(), v.clone()));
        Ok(v)
    })
    .await
}

/// The model author's published sampling defaults (generation_config.json).
#[tauri::command]
async fn creator_defaults(model_path: String) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let d = fidim_core::hf::creator_defaults(&cfg, &PathBuf::from(model_path)).map_err(|e| e.to_string())?;
        serde_json::to_value(d).map_err(|e| e.to_string())
    })
    .await
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
    // Trace what the editor's check saw, so a PASS here that a real launch
    // contradicts can be diagnosed from ~/.fidim/ui.log.
    {
        let dev: Vec<String> = prepared
            .context
            .resolved
            .iter()
            .map(|r| {
                format!(
                    "{}[free {} MiB, display {}]",
                    r.device.stable_key.rsplit(':').next().unwrap_or("?"),
                    r.device.free_mib,
                    r.device.display.is_some()
                )
            })
            .collect();
        let est = prepared
            .context
            .estimate
            .as_ref()
            .map(|e| format!("{:.2} GiB", e.total_bytes as f64 / (1u64 << 30) as f64))
            .unwrap_or_else(|| "none".into());
        let blocks: Vec<String> = results
            .iter()
            .filter(|r| matches!(r.outcome, preflight::Outcome::Block(_)))
            .map(|r| r.spec_number.to_string())
            .collect();
        let _ = ui_log(format!(
            "live_check {}: devices={} estimate={} blocks=[{}]",
            p.id,
            dev.join(","),
            est,
            blocks.join(",")
        ));
    }
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
    // Port takeover (check 8 warned): stop the Llama FIDIM server on this port first.
    let mut replaced: Option<String> = None;
    if let Some(h) = &prepared.context.port_holder {
        if h.profile_id.is_some() {
            if let Some(run) = supervise::reattach(&cfg.runs_dir)
                .into_iter()
                .find(|r| r.alive && r.state.port == profile.server.port)
            {
                supervise::stop(&run.state, &cfg.runs_dir).map_err(|e| e.to_string())?;
                launch::wait_port_free(&profile.server.host, profile.server.port, Duration::from_secs(20))
                    .map_err(|e| e.to_string())?;
                replaced = Some(run.state.profile_id.clone());
            }
        }
    }
    let cold = supervise::is_cold_start(&Config::config_dir(), &profile.model.path);
    launch::final_commit_gate(
        &cfg,
        &platform,
        prepared.context.estimate.as_ref(),
        override_blocks,
    )
    .map_err(|e| e.to_string())?;
    let mut state = supervise::spawn(
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
    let ka = profile.keep_alive_seconds.unwrap_or(cfg.keep_alive_seconds);
    let keepalive = match supervise::spawn_keepalive(&mut state, &cfg.runs_dir, ka) {
        Ok(Some(kp)) => serde_json::json!({ "interval_s": ka, "pid": kp }),
        Ok(None) => serde_json::json!({ "interval_s": 0 }),
        Err(e) => serde_json::json!({ "error": e.to_string() }),
    };

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
        "keepalive": keepalive,
        "replaced": replaced,
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
                    fidim_core::devices::stable_key(&a.pnp_device_id, a.bus_number) == *key
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
                fidim_core::devices::stable_key(&a.pnp_device_id, a.bus_number) == *k
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

// --------------------------------------------------------------- settings ----

/// The tool's own configuration (roots, runtime, install paths) plus where
/// it lives, for the Settings view.
#[tauri::command]
fn get_config() -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    Ok(serde_json::json!({
        "path": Config::config_path(),
        "config": cfg,
    }))
}

/// Save the configuration and invalidate every cache that depends on it.
#[tauri::command]
fn save_config(state: tauri::State<'_, AppState>, config: Config) -> Result<(), String> {
    config.save(&Config::config_path()).map_err(|e| e.to_string())?;
    let mut c = state.cache.lock().unwrap();
    c.devices = None;
    c.build_probes.clear();
    c.scan = None;
    Ok(())
}

/// Append a line from the web view to `~/.fidim/ui.log` — the only way
/// a failure inside the GUI becomes visible outside it.
#[tauri::command]
fn ui_log(line: String) -> Result<(), String> {
    use std::io::Write;
    let p = Config::config_dir().join("ui.log");
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&p).map_err(|e| e.to_string())?;
    writeln!(f, "{ts} {line}").map_err(|e| e.to_string())
}

/// ROCm runtimes a profile can name. Filesystem probing only.
#[tauri::command]
fn list_runtimes() -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    serde_json::to_value(fidim_core::runtime::discover(&cfg)).map_err(|e| e.to_string())
}

#[tauri::command]
fn update_history() -> Result<serde_json::Value, String> {
    let h = update::load_history().map_err(|e| e.to_string())?;
    serde_json::to_value(h).map_err(|e| e.to_string())
}

// ----------------------------------------------------------------- router ----

#[tauri::command]
fn router_get() -> Result<serde_json::Value, String> {
    let rc = fidim_core::router::load_config().map_err(|e| e.to_string())?;
    serde_json::to_value(rc).map_err(|e| e.to_string())
}

#[tauri::command]
fn router_save(rc: fidim_core::router::RouterConfig) -> Result<(), String> {
    fidim_core::router::save_config(&rc).map_err(|e| e.to_string())
}

/// Render the preset file for a (possibly unsaved) router config.
#[tauri::command]
async fn router_ini(state: tauri::State<'_, AppState>, rc: fidim_core::router::RouterConfig) -> Result<serde_json::Value, String> {
    let cache = state.cache.clone();
    blocking(move || {
        let cfg = cfg()?;
        let profiles = Profile::load_all(&cfg.profile_dir).map_err(|e| e.to_string())?;
        let devices = cached_devices(&cache, &cfg, false)?;
        let r = fidim_core::router::render_ini(&rc, &profiles, &devices).map_err(|e| e.to_string())?;
        serde_json::to_value(r).map_err(|e| e.to_string())
    })
    .await
}

#[tauri::command]
async fn router_launch() -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let rc = fidim_core::router::load_config().map_err(|e| e.to_string())?;
        let profiles = Profile::load_all(&cfg.profile_dir).map_err(|e| e.to_string())?;
        let builds = discovery::scan_builds(&cfg.build_roots_effective(), cfg.rocm_bin.as_deref());
        let build_dir = match &rc.build {
            Some(b) => b.clone(),
            None => update::newest_installed(&builds).ok_or("no installed build")?.path,
        };
        let devices = launch::enumerate_devices(&cfg, &build_dir.join("bin").join("llama-server.exe"), &WindowsPlatform)
            .map_err(|e| e.to_string())?;
        let r = fidim_core::router::launch(&cfg, &rc, &profiles, &devices, &build_dir, Duration::from_secs(180))
            .map_err(|e| e.to_string())?;
        serde_json::to_value(r).map_err(|e| e.to_string())
    })
    .await
}

#[tauri::command]
fn router_status() -> Result<serde_json::Value, String> {
    let cfg = cfg()?;
    let run = supervise::reattach(&cfg.runs_dir).into_iter().find(|r| r.state.profile_id == fidim_core::router::ROUTER_ID);
    Ok(match run {
        Some(r) => serde_json::json!({ "alive": r.alive, "state": r.state }),
        None => serde_json::json!({ "alive": false }),
    })
}

#[tauri::command]
async fn router_models() -> Result<serde_json::Value, String> {
    blocking(move || {
        let rc = fidim_core::router::load_config().map_err(|e| e.to_string())?;
        let ms = fidim_core::router::models(&rc.host, rc.port).map_err(|e| e.to_string())?;
        serde_json::to_value(ms).map_err(|e| e.to_string())
    })
    .await
}

#[tauri::command]
async fn router_load(id: String) -> Result<(), String> {
    blocking(move || {
        let rc = fidim_core::router::load_config().map_err(|e| e.to_string())?;
        fidim_core::router::load_model(&rc.host, rc.port, &id).map_err(|e| e.to_string())
    })
    .await
}

#[tauri::command]
async fn router_unload(id: String) -> Result<(), String> {
    blocking(move || {
        let rc = fidim_core::router::load_config().map_err(|e| e.to_string())?;
        fidim_core::router::unload_model(&rc.host, rc.port, &id).map_err(|e| e.to_string())
    })
    .await
}

// ------------------------------------------------------------------- live ----

/// One poll of everything running: per run, slot phases + metrics (per
/// model behind a router), resident VRAM, and per-card GPU busy.
#[tauri::command]
async fn live(state: tauri::State<'_, AppState>) -> Result<serde_json::Value, String> {
    let cache = state.cache.clone();
    blocking(move || {
        let cfg = cfg()?;
        let platform = WindowsPlatform;
        let devices = cached_devices(&cache, &cfg, false).unwrap_or_default();
        let card_of = |luid: u64| devices.iter().find(|d| d.luid_low == Some(luid)).map(|d| d.stable_key.clone());
        let util = platform.gpu_utilization().unwrap_or_default();
        // Per-card busy: sum over every process on that adapter.
        let mut cards: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
        for u in &util {
            if let Some(k) = card_of(u.luid_low) {
                *cards.entry(k).or_insert(0.0) += u.percent;
            }
        }
        let runs: Vec<serde_json::Value> = supervise::reattach(&cfg.runs_dir)
            .into_iter()
            .map(|r| {
                let mut samples = Vec::new();
                if r.alive {
                    if r.state.profile_id == fidim_core::router::ROUTER_ID {
                        if let Ok(ms) = fidim_core::router::models(&r.state.host, r.state.port) {
                            for m in ms.iter().filter(|m| m.status == "loaded") {
                                samples.push(fidim_core::live::sample(&r.state.host, r.state.port, Some(&m.id)));
                            }
                        }
                    } else {
                        samples.push(fidim_core::live::sample(&r.state.host, r.state.port, None));
                    }
                }
                // The router's model instances are child processes; their
                // VRAM and GPU time belong to the router run.
                let mut pids = vec![r.state.pid];
                if r.alive {
                    pids.extend(fidim_core::platform::process_descendants(r.state.pid));
                }
                let mut resident: Vec<serde_json::Value> = Vec::new();
                for pid in &pids {
                    for m in platform.gpu_process_memory(*pid).unwrap_or_default() {
                        resident.push(serde_json::json!({ "pid": pid, "card": card_of(m.luid_low), "dedicated_bytes": m.dedicated_bytes, "committed_bytes": m.committed_bytes }));
                    }
                }
                let busy: f64 = util.iter().filter(|u| pids.contains(&u.pid)).map(|u| u.percent).sum();
                serde_json::json!({
                    "run": r,
                    "pids": pids,
                    "samples": samples,
                    "resident": resident,
                    "gpu_busy_percent": if busy <= 0.0 { 0.0 } else { busy.min(100.0) },
                })
            })
            .collect();
        let cards_json: Vec<serde_json::Value> = devices
            .iter()
            .filter(|d| !d.integrated)
            .map(|d| serde_json::json!({ "key": d.stable_key, "name": d.name, "busy_percent": cards.get(&d.stable_key).copied().unwrap_or(0.0).clamp(0.0, 100.0), "total_mib": d.total_mib }))
            .collect();
        Ok(serde_json::json!({ "runs": runs, "cards": cards_json }))
    })
    .await
}

// ------------------------------------------------------------ rocm runtimes ----

#[tauri::command]
async fn rocm_families() -> Result<serde_json::Value, String> {
    blocking(|| {
        let f = fidim_core::rocm::families().map_err(|e| e.to_string())?;
        serde_json::to_value(f).map_err(|e| e.to_string())
    })
    .await
}

#[tauri::command]
async fn rocm_available(state: tauri::State<'_, AppState>, family: String) -> Result<serde_json::Value, String> {
    let cache = state.cache.clone();
    blocking(move || {
        let cfg = cfg()?;
        // Empty = guess from the cards, else RDNA4.
        let family = if family.trim().is_empty() {
            let names: Vec<String> = cached_devices(&cache, &cfg, false).unwrap_or_default().into_iter().map(|d| d.name).collect();
            fidim_core::rocm::guess_family(&names).unwrap_or_else(|| "gfx120X-all".to_string())
        } else {
            family
        };
        let (runtimes, problems) = fidim_core::rocm::available(&family).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({ "family": family, "runtimes": runtimes, "problems": problems }))
    })
    .await
}

#[tauri::command]
async fn rocm_install(app: tauri::AppHandle, runtime: fidim_core::rocm::AvailableRuntime) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let mut progress = |line: String| {
            let _ = app.emit("update-progress", line);
        };
        let dir = fidim_core::rocm::install(&cfg, &runtime, &mut progress).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({ "dir": dir, "name": format!("rocm-{}", runtime.version) }))
    })
    .await
}

#[tauri::command]
async fn rocm_remove(version: String) -> Result<(), String> {
    blocking(move || {
        let cfg = cfg()?;
        fidim_core::rocm::remove(&cfg, &version).map_err(|e| e.to_string())
    })
    .await
}

pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            cache: Arc::new(Mutex::new(UiCache { devices: None, build_probes: HashMap::new(), scan: None })),
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
            creator_defaults,
            get_config,
            save_config,
            ui_log,
            router_get,
            router_save,
            router_ini,
            router_launch,
            router_status,
            router_models,
            router_load,
            router_unload,
            live,
            rocm_families,
            rocm_available,
            rocm_install,
            rocm_remove,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Llama FIDIM UI");
}
