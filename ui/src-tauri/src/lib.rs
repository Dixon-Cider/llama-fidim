//! Tauri command layer: thin wrappers over fidim-core. The GUI runs the
//! SAME check objects and launch path as the CLI — the editor and the
//! launcher can never disagree (spec V-2).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fidim_core::chat::{self, store as chat_store};
use fidim_core::config::Config;
use fidim_core::devices::Device;
use fidim_core::launch::{self, PrepareInputs};
use fidim_core::platform::{Platform, WindowsPlatform};
use fidim_core::profile::{self, Engine, Profile};
use fidim_core::supervise;
use fidim_core::update::{self, PromoteScope};
use fidim_core::{bench, discovery, export, preflight};
use tauri::ipc::Channel;
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
    /// Chat streams in flight, by the id the web view gave each, so Stop
    /// (and a reload's cancel-all) can reach them.
    chats: ChatStreams,
    /// Model wizard jobs, running and finished, so the Models view can
    /// come back to one after navigating away.
    wizard: WizardJobs,
}

type WizardJobs = Arc<Mutex<HashMap<String, WizardJob>>>;

/// One wizard run: its plan, the latest state of each step (rebuilt from
/// the same events the view gets), and how it ended.
struct WizardJob {
    plan: fidim_core::wizard::WizardPlan,
    progress: fidim_core::wizard::JobProgress,
    cancel: Arc<std::sync::atomic::AtomicBool>,
    started_unix: u64,
    consent: bool,
    finished: Option<serde_json::Value>,
}

impl WizardJob {
    fn snapshot(&self, id: &str) -> serde_json::Value {
        serde_json::json!({
            "job": id,
            "plan": self.plan,
            "progress": self.progress,
            "started_unix": self.started_unix,
            "consent": self.consent,
            "cancelling": self.cancel.load(std::sync::atomic::Ordering::Relaxed) && self.finished.is_none(),
            "finished": self.finished,
        })
    }
}

type ChatStreams = Arc<Mutex<HashMap<String, Arc<chat::Cancel>>>>;

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
    // Upstream first: a fork build reports a higher upstream number and
    // enumerates under its own bundled runtime.
    let newest = |upstream_only: bool| {
        builds
            .iter()
            .filter(|b| !upstream_only || b.channel == discovery::Channel::Upstream)
            .filter(|b| b.version.is_some())
            .max_by_key(|b| b.version.as_deref().and_then(fidim_core::update::version_number).unwrap_or(0))
    };
    let build = newest(true)
        .or_else(|| newest(false))
        .or(builds.first())
        .ok_or("no builds found under configured build_roots")?;
    let devices = launch::enumerate_devices(cfg, &build.server_exe, &WindowsPlatform)
        .map_err(|e| e.to_string())?;
    let mut cache = cache_arc.lock().unwrap();
    cache.devices = Some((Instant::now(), devices.clone()));
    Ok(devices)
}

/// Forget everything derived from the installed builds, so a build that
/// was just installed shows up in the pickers without waiting for a TTL.
fn invalidate_build_caches(cache: &Arc<Mutex<UiCache>>) {
    fidim_core::wizard::forget_builds();
    let mut c = cache.lock().unwrap();
    c.scan = None;
    c.devices = None;
    c.build_probes.clear();
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
    // A build that bundles its ROCm runs with nothing on PATH, as a launch does.
    let prefix = if discovery::read_build_meta(build_path).bundled_runtime { None } else { cfg.rocm_bin.as_deref() };
    let probe = launch::run_capture(&exe, &["--version"], prefix)
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
        let (models, aux) = discovery::scan_models_and_aux(&cfg.model_roots);
        let v = serde_json::json!({ "builds": builds, "models": models, "drafts": aux.drafts, "mmproj": aux.mmproj });
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
        // Diffusion profiles: helper/runner presence and the predicted
        // context budget the editor shows as the "auto" placeholder.
        "diffusion": prepared.context.diffusion,
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
    // first inference), over the run's process tree: a diffusion helper's
    // runner child holds the model.
    let mem = fidim_core::platform::sum_gpu_memory(&platform, &fidim_core::platform::run_pids(state.pid))
        .unwrap_or_default();
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
    bench::ensure_benchable(&profile).map_err(|e| e.to_string())?;
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

    let mem = fidim_core::platform::sum_gpu_memory(&platform, &fidim_core::platform::run_pids(run.state.pid))
        .unwrap_or_default();
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
    state: tauri::State<'_, AppState>,
    tag: Option<String>,
    source: bool,
) -> Result<serde_json::Value, String> {
    let cache = state.cache.clone();
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
        invalidate_build_caches(&cache);
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

/// Card names for the Unsloth GPU-target guess. WMI, so blocking pool only.
fn adapter_names() -> Vec<String> {
    WindowsPlatform.video_adapters().unwrap_or_default().into_iter().map(|a| a.name).collect()
}

/// Latest (or `tag`) Unsloth fork release, the zip for this machine's GPU
/// target, whether the runner-patch overlay is published for it, and the
/// fork builds already installed. Network + a build scan.
#[tauri::command]
async fn unsloth_check(tag: Option<String>, gfx: Option<String>) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let c = update::check_unsloth(&cfg, &adapter_names(), gfx.as_deref(), tag.as_deref())
            .map_err(|e| e.to_string())?;
        serde_json::to_value(c).map_err(|e| e.to_string())
    })
    .await
}

/// Download, digest-check, install and verify an Unsloth fork build; with
/// `overlay`, the build with the published runner-patch overlay laid over
/// it. Progress streams as `update-progress`. Never starts the runner.
#[tauri::command]
async fn unsloth_install(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    tag: Option<String>,
    gfx: Option<String>,
    overlay: Option<bool>,
) -> Result<serde_json::Value, String> {
    let cache = state.cache.clone();
    blocking(move || {
        let cfg = cfg()?;
        let release = match &tag {
            Some(t) => update::unsloth_release_by_tag(t),
            None => update::latest_unsloth_release(),
        }
        .map_err(|e| e.to_string())?;
        let gfx = update::unsloth_gfx(&cfg, &adapter_names(), gfx.as_deref());
        let mut progress = |line: String| {
            let _ = app.emit("update-progress", line);
        };
        let r = if overlay.unwrap_or(false) {
            let source = fidim_core::overlay::OverlaySource::Published;
            fidim_core::overlay::install_unsloth_overlay(&cfg, &release, &gfx, &source, None, &mut progress)
        } else {
            update::install_unsloth(&cfg, &release, &gfx, &mut progress)
        }
        .map_err(|e| e.to_string())?;
        invalidate_build_caches(&cache);
        serde_json::to_value(r).map_err(|e| e.to_string())
    })
    .await
}

/// Which diffusion profiles `unsloth_promote` would move onto a fork build,
/// off which runner patch, and why the others stay. Changes nothing; the
/// Updates view shows it for confirmation.
#[tauri::command]
async fn unsloth_promote_preview(to_path: String) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let r = update::promote_preview(&cfg, &PathBuf::from(to_path), &PromoteScope::All).map_err(|e| e.to_string())?;
        serde_json::to_value(r).map_err(|e| e.to_string())
    })
    .await
}

/// Move diffusion profiles onto a fork build: the confirmed `ids` of a
/// preview, or (without ids) every profile the promote rules allow; they
/// skip llama-server ones and anything pinned.
#[tauri::command]
async fn unsloth_promote(to_path: String, to_version: Option<String>, ids: Option<Vec<String>>) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let scope = ids.map(PromoteScope::Ids).unwrap_or(PromoteScope::All);
        let r = update::promote(&cfg, &PathBuf::from(to_path), to_version, scope).map_err(|e| e.to_string())?;
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
    invalidate_build_caches(&state.cache);
    Ok(())
}

/// This build's version and commit, for the sidebar.
#[tauri::command]
fn app_version() -> serde_json::Value {
    fidim_core::build_info::json()
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
        // A server started with an API key answers /slots and /metrics
        // only with it; Copy endpoint tells other programs they need one.
        let profiles = Profile::load_all(&cfg.profile_dir).unwrap_or_default();
        let member = |id: &str| profiles.iter().find(|p| fidim_core::router::model_id(p) == id);
        let runs: Vec<serde_json::Value> = supervise::reattach(&cfg.runs_dir)
            .into_iter()
            .map(|r| {
                let mut samples = Vec::new();
                let mut keyed_models = Vec::new();
                let profile = profiles.iter().find(|p| p.id == r.state.profile_id);
                if r.alive {
                    if r.state.profile_id == fidim_core::router::ROUTER_ID {
                        if let Ok(ms) = fidim_core::router::models(&r.state.host, r.state.port) {
                            for m in &ms {
                                if member(&m.id).is_some_and(chat::requires_api_key) {
                                    keyed_models.push(m.id.clone());
                                }
                            }
                            for m in ms.iter().filter(|m| m.status == "loaded") {
                                let key = member(&m.id).and_then(chat::api_key);
                                samples.push(fidim_core::live::sample_with_key(&r.state.host, r.state.port, Some(&m.id), key.as_deref()));
                            }
                        }
                    } else {
                        let key = profile.and_then(chat::api_key);
                        samples.push(fidim_core::live::sample_with_key(&r.state.host, r.state.port, None, key.as_deref()));
                    }
                }
                let has_api_key = r.state.profile_id != fidim_core::router::ROUTER_ID && profile.is_some_and(chat::requires_api_key);
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
                    "has_api_key": has_api_key,
                    "keyed_models": keyed_models,
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

/// One server's slots and metrics, for a view that follows it faster than
/// the whole-app `live` poll (the diffusion canvas refreshes ~5x a second).
#[tauri::command]
async fn live_one(host: String, port: u16) -> Result<serde_json::Value, String> {
    blocking(move || serde_json::to_value(fidim_core::live::sample(&host, port, None)).map_err(|e| e.to_string())).await
}

/// A diffusion server's steps of its current (or last) reply, for replay.
#[tauri::command]
async fn dg_frames(host: String, port: u16) -> Result<serde_json::Value, String> {
    blocking(move || {
        match fidim_core::supervise::http_get(&host, port, "/frames", std::time::Duration::from_secs(5)) {
            Ok((200, body)) => serde_json::from_str(&body).map_err(|e| format!("/frames: {e}")),
            Ok((code, _)) => Err(format!("/frames HTTP {code} (a diffusion server from an older Llama FIDIM?)")),
            Err(e) => Err(format!("/frames: {e}")),
        }
    })
    .await
}

// ------------------------------------------------------------------- chat ----

/// Which server a chat talks to: a run Llama FIDIM started, and a model
/// when the run is the router. The web view never sends an address: Rust
/// resolves host, port and any API key from the run and its profile, so
/// script in the page cannot aim a request anywhere else.
#[derive(serde::Deserialize)]
struct ChatTarget {
    run: String,
    #[serde(default)]
    model: Option<String>,
}

struct ResolvedTarget {
    host: String,
    port: u16,
    engine: Engine,
    /// The model id the server expects.
    model: String,
    api_key: Option<String>,
    router: bool,
}

fn resolve_target(cfg: &Config, t: &ChatTarget) -> Result<ResolvedTarget, String> {
    let run = supervise::reattach(&cfg.runs_dir)
        .into_iter()
        .find(|r| r.alive && r.state.profile_id == t.run)
        .ok_or_else(|| format!("{} is not running", t.run))?;
    let profiles = Profile::load_all(&cfg.profile_dir).unwrap_or_default();
    let s = &run.state;
    let host = chat::connect_host(&s.host).to_string();
    if s.profile_id == fidim_core::router::ROUTER_ID {
        let model = t.model.clone().filter(|m| !m.trim().is_empty()).ok_or("pick a model on the router")?;
        let member = profiles.iter().find(|p| fidim_core::router::model_id(p) == model);
        Ok(ResolvedTarget {
            host,
            port: s.port,
            engine: Engine::LlamaServer,
            api_key: member.and_then(chat::api_key),
            model,
            router: true,
        })
    } else {
        let profile = profiles.iter().find(|p| p.id == s.profile_id);
        Ok(ResolvedTarget {
            host,
            port: s.port,
            engine: s.engine,
            model: s.alias.clone(),
            api_key: profile.and_then(chat::api_key),
            router: false,
        })
    }
}

/// Removes a stream's cancel handle however the command ends, but never a
/// newer stream that reused the id.
struct StreamGuard {
    streams: ChatStreams,
    id: String,
    cancel: Arc<chat::Cancel>,
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        let mut m = self.streams.lock().unwrap_or_else(|e| e.into_inner());
        if m.get(&self.id).is_some_and(|c| Arc::ptr_eq(c, &self.cancel)) {
            m.remove(&self.id);
        }
    }
}

/// Stream one reply. Events go to `on_event` in order and always end with
/// done, error or cancelled; the return value summarises the same. Only the
/// shape of a stream is logged, never its text.
#[tauri::command]
async fn chat_send(
    state: tauri::State<'_, AppState>,
    stream_id: String,
    target: ChatTarget,
    body: serde_json::Value,
    on_event: Channel<chat::ChatEvent>,
) -> Result<serde_json::Value, String> {
    let cancel = Arc::new(chat::Cancel::new());
    {
        let mut m = state.chats.lock().unwrap_or_else(|e| e.into_inner());
        if m.contains_key(&stream_id) {
            return Err(format!("chat stream {stream_id} is already running"));
        }
        m.insert(stream_id.clone(), cancel.clone());
    }
    let guard = StreamGuard { streams: state.chats.clone(), id: stream_id, cancel: cancel.clone() };
    blocking(move || {
        let _guard = guard;
        let cfg = cfg()?;
        let t = resolve_target(&cfg, &target)?;
        let n_msgs = body.get("messages").and_then(|m| m.as_array()).map_or(0, Vec::len);
        let body = chat::request_body(&body, &t.model, t.engine).map_err(|e| e.to_string())?;
        let started = Instant::now();
        let sum = chat::stream_chat(&t.host, t.port, &body, t.api_key.as_deref(), &cancel, chat::DEFAULT_IDLE, &mut |ev| {
            // A reloaded web view no longer listens; the stream still ends
            // on its own, and a reload cancels every stream at boot.
            let _ = on_event.send(ev);
        });
        let outcome = if sum.cancelled {
            "stopped".to_string()
        } else if sum.error.is_some() {
            format!("error {}", sum.status.map_or("-".into(), |s| s.to_string()))
        } else {
            format!("finish={}", sum.finish_reason.as_deref().unwrap_or("-"))
        };
        let _ = ui_log(format!(
            "chat {}{} messages={n_msgs} -> {outcome} in {} ms",
            target.run,
            if t.router { format!("/{}", t.model) } else { String::new() },
            started.elapsed().as_millis()
        ));
        serde_json::to_value(sum).map_err(|e| e.to_string())
    })
    .await
}

/// Stop one stream. False when it had already ended.
#[tauri::command]
fn chat_cancel(state: tauri::State<'_, AppState>, stream_id: String) -> bool {
    let m = state.chats.lock().unwrap_or_else(|e| e.into_inner());
    match m.get(&stream_id) {
        Some(c) => {
            c.cancel();
            true
        }
        None => false,
    }
}

/// Stop every stream: the web view calls this once at boot, so a reload
/// does not leave replies streaming to nobody.
#[tauri::command]
fn chat_cancel_all(state: tauri::State<'_, AppState>) -> usize {
    let m = state.chats.lock().unwrap_or_else(|e| e.into_inner());
    m.values().for_each(|c| c.cancel());
    m.len()
}

/// Everything the chat can talk to right now.
#[tauri::command]
async fn chat_targets() -> Result<serde_json::Value, String> {
    blocking(|| {
        let cfg = cfg()?;
        let profiles = Profile::load_all(&cfg.profile_dir).unwrap_or_default();
        let runs = supervise::reattach(&cfg.runs_dir);
        serde_json::to_value(chat::targets(&runs, &profiles)).map_err(|e| e.to_string())
    })
    .await
}

/// A target's server-side facts: context, default sampler, capabilities.
/// A router model that is not loaded is not asked; one evicted between the
/// check and the question gets an error, not a load (`autoload=false`).
#[tauri::command]
async fn chat_props(target: ChatTarget) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let t = resolve_target(&cfg, &target)?;
        if t.router {
            let status = fidim_core::router::models(&t.host, t.port)
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|m| m.id == t.model)
                .map(|m| m.status)
                .unwrap_or_else(|| "unknown".into());
            if status != "loaded" {
                return Ok(serde_json::json!({ "loaded": false, "status": status }));
            }
        }
        let mut v = chat::props(&t.host, t.port, t.engine, t.router.then_some(t.model.as_str()), t.api_key.as_deref())
            .map_err(|e| e.to_string())?;
        v["loaded"] = true.into();
        Ok(v)
    })
    .await
}

#[tauri::command]
async fn chat_list() -> Result<serde_json::Value, String> {
    blocking(|| serde_json::to_value(chat_store::list(&chat_store::chats_dir())).map_err(|e| e.to_string())).await
}

#[tauri::command]
async fn chat_load(id: String) -> Result<serde_json::Value, String> {
    blocking(move || chat_store::load(&chat_store::chats_dir(), &id).map_err(|e| e.to_string())).await
}

/// Save a conversation; false (nothing written) when Settings has saving off.
#[tauri::command]
async fn chat_save(conv: serde_json::Value) -> Result<bool, String> {
    blocking(move || {
        if !cfg()?.save_chats {
            return Ok(false);
        }
        chat_store::save(&chat_store::chats_dir(), &conv).map_err(|e| e.to_string())?;
        Ok(true)
    })
    .await
}

#[tauri::command]
async fn chat_delete(id: String) -> Result<bool, String> {
    blocking(move || chat_store::delete(&chat_store::chats_dir(), &id).map_err(|e| e.to_string())).await
}

#[tauri::command]
async fn chat_delete_all() -> Result<usize, String> {
    blocking(|| chat_store::delete_all(&chat_store::chats_dir()).map_err(|e| e.to_string())).await
}

/// Per-profile chat defaults (system prompt, sampler overrides, thinking).
#[tauri::command]
async fn chat_presets_get() -> Result<serde_json::Value, String> {
    blocking(|| Ok(chat_store::load_presets(&chat_store::presets_path()))).await
}

#[tauri::command]
async fn chat_presets_save(presets: serde_json::Value) -> Result<(), String> {
    blocking(move || chat_store::save_presets(&chat_store::presets_path(), &presets).map_err(|e| e.to_string())).await
}

// ----------------------------------------------------------------- wizard ----

/// Hugging Face search, each hit marked with whether an installed build
/// knows its architecture.
#[tauri::command]
async fn hub_search(query: String, limit: Option<u32>, all: Option<bool>) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let q = fidim_core::hub::SearchQuery {
            text: query,
            limit: limit.unwrap_or(30),
            gguf_only: !all.unwrap_or(false),
            ..Default::default()
        };
        let hits = fidim_core::wizard::search(&cfg, &q).map_err(|e| e.to_string())?;
        serde_json::to_value(hits).map_err(|e| e.to_string())
    })
    .await
}

/// A repo's files, the fit of each on these cards, and which build loads
/// it (or the plan for one). Header range reads only; nothing downloads.
#[tauri::command]
async fn wizard_inspect(
    state: tauri::State<'_, AppState>,
    input: String,
    rev: Option<String>,
) -> Result<serde_json::Value, String> {
    let cache = state.cache.clone();
    blocking(move || {
        let cfg = cfg()?;
        let env = fidim_core::wizard::LiveEnv { devices: cached_devices(&cache, &cfg, false).ok(), cfg: cfg.clone() };
        let view = fidim_core::wizard::inspect_with(&env, &cfg, &input, rev.as_deref()).map_err(|e| e.to_string())?;
        let _ = ui_log(format!(
            "wizard inspect {} @ {}: {:?}, {} choices, usable={}",
            view.repo,
            view.sha.get(..7).unwrap_or(&view.sha),
            view.kind,
            view.catalog.choices.len(),
            view.usable_build.is_some()
        ));
        serde_json::to_value(view).map_err(|e| e.to_string())
    })
    .await
}

/// The exact steps for a choice: build or install, downloads, the profile.
#[tauri::command]
async fn wizard_plan(
    view: fidim_core::wizard::RepoView,
    request: fidim_core::wizard::PlanRequest,
) -> Result<serde_json::Value, String> {
    blocking(move || {
        let cfg = cfg()?;
        let plan = fidim_core::wizard::plan(&cfg, &view, &request).map_err(|e| e.to_string())?;
        serde_json::to_value(plan).map_err(|e| e.to_string())
    })
    .await
}

/// Start a plan on its own thread; progress arrives as `wizard-progress`
/// events and the end as `wizard-done`. The job outlives the view that
/// started it (`wizard_jobs` finds it again). A plan that needs consent is
/// refused here, before anything runs, unless `consent` is true.
#[tauri::command]
fn wizard_start(
    app: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    plan: fidim_core::wizard::WizardPlan,
    consent: bool,
) -> Result<String, String> {
    fidim_core::wizard::check_runnable(&plan, consent).map_err(|e| e.to_string())?;
    let jobs = state.wizard.clone();
    let cache = state.cache.clone();
    let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let id = {
        let mut m = jobs.lock().unwrap_or_else(|e| e.into_inner());
        // Two jobs never write the same file.
        let mine = plan.download_dests();
        for (other, job) in m.iter().filter(|(_, j)| j.finished.is_none()) {
            let theirs = job.plan.download_dests();
            if let Some((d, _)) = mine.iter().find(|(d, _)| theirs.iter().any(|(t, _)| t == d)) {
                return Err(format!("job {other} is already downloading {}", d.display()));
            }
        }
        let started_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let id = format!("w{started_unix}-{}", m.len() + 1);
        m.insert(
            id.clone(),
            WizardJob {
                progress: fidim_core::wizard::JobProgress::new(&plan),
                plan: plan.clone(),
                cancel: cancel.clone(),
                started_unix,
                consent,
                finished: None,
            },
        );
        id
    };
    let _ = ui_log(format!("wizard start {id}: {} {} ({} steps, consent={consent})", plan.repo, plan.choice.label, plan.steps.len()));
    let job = id.clone();
    std::thread::spawn(move || {
        let outcome = (|| -> Result<fidim_core::wizard::WizardResult, String> {
            let cfg = cfg()?;
            let devices = cached_devices(&cache, &cfg, false).ok();
            fidim_core::wizard::run(
                &cfg,
                devices,
                &plan,
                consent,
                &mut |e| {
                    if let Some(j) = jobs.lock().unwrap_or_else(|p| p.into_inner()).get_mut(&job) {
                        j.progress.apply(e);
                    }
                    let mut payload = serde_json::to_value(e).unwrap_or_default();
                    payload["job"] = job.clone().into();
                    let _ = app.emit("wizard-progress", payload);
                },
                &cancel,
            )
            .map_err(|e| e.to_string())
        })();
        // A download or a build changes what the pickers should list.
        invalidate_build_caches(&cache);
        let done = match &outcome {
            Ok(r) => serde_json::json!({ "job": job, "ok": true, "result": r }),
            Err(e) => serde_json::json!({ "job": job, "ok": false, "error": e }),
        };
        let _ = ui_log(format!(
            "wizard {job} {}",
            match &outcome {
                Ok(r) => format!("done: profile {}", r.profile.as_ref().map(|p| p.id.as_str()).unwrap_or("none")),
                Err(e) => format!("failed: {e}"),
            }
        ));
        if let Some(j) = jobs.lock().unwrap_or_else(|p| p.into_inner()).get_mut(&job) {
            j.finished = Some(done.clone());
        }
        let _ = app.emit("wizard-done", done);
    });
    Ok(id)
}

/// Ask a job to stop: a download keeps its `.part` for a resume, a build
/// stops its compilers and cleans up. False when it had already ended.
#[tauri::command]
fn wizard_cancel(state: tauri::State<'_, AppState>, job: String) -> bool {
    let m = state.wizard.lock().unwrap_or_else(|e| e.into_inner());
    match m.get(&job) {
        Some(j) if j.finished.is_none() => {
            j.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            true
        }
        _ => false,
    }
}

/// Every job this session, oldest first, with its plan and progress.
#[tauri::command]
fn wizard_jobs(state: tauri::State<'_, AppState>) -> serde_json::Value {
    let m = state.wizard.lock().unwrap_or_else(|e| e.into_inner());
    let mut jobs: Vec<(&String, &WizardJob)> = m.iter().collect();
    jobs.sort_by_key(|(id, j)| (j.started_unix, (*id).clone()));
    serde_json::Value::Array(jobs.into_iter().map(|(id, j)| j.snapshot(id)).collect())
}

/// Drop a finished job from the list.
#[tauri::command]
fn wizard_forget(state: tauri::State<'_, AppState>, job: String) -> bool {
    let mut m = state.wizard.lock().unwrap_or_else(|e| e.into_inner());
    if m.get(&job).is_some_and(|j| j.finished.is_some()) {
        m.remove(&job);
        true
    } else {
        false
    }
}

/// The model folders with the space left on each drive.
#[tauri::command]
async fn wizard_roots() -> Result<serde_json::Value, String> {
    blocking(|| {
        let cfg = cfg()?;
        serde_json::to_value(fidim_core::wizard::model_roots(&cfg)).map_err(|e| e.to_string())
    })
    .await
}

/// Add a model folder (created if missing) to the configuration, so a
/// download there is found by the scan.
#[tauri::command]
async fn wizard_add_root(state: tauri::State<'_, AppState>, path: String) -> Result<serde_json::Value, String> {
    let cache = state.cache.clone();
    blocking(move || {
        let mut cfg = cfg()?;
        if fidim_core::wizard::add_model_root(&mut cfg, std::path::Path::new(&path)).map_err(|e| e.to_string())? {
            cfg.save(&Config::config_path()).map_err(|e| e.to_string())?;
            invalidate_build_caches(&cache);
            let _ = ui_log(format!("wizard: added model folder {path}"));
        }
        serde_json::to_value(fidim_core::wizard::model_roots(&cfg)).map_err(|e| e.to_string())
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
        // Links in chat replies open in the browser only through our own
        // Open action; the plugin's click interception stays off.
        .plugin(tauri_plugin_opener::Builder::new().open_js_links_on_click(false).build())
        .manage(AppState {
            cache: Arc::new(Mutex::new(UiCache { devices: None, build_probes: HashMap::new(), scan: None })),
            chats: Arc::new(Mutex::new(HashMap::new())),
            wizard: Arc::new(Mutex::new(HashMap::new())),
        })
        .invoke_handler(tauri::generate_handler![
            scan,
            devices,
            list_profiles,
            save_profile,
            delete_profile,
            live_check,
            live_one,
            dg_frames,
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
            unsloth_check,
            unsloth_install,
            unsloth_promote_preview,
            unsloth_promote,
            list_runtimes,
            creator_defaults,
            get_config,
            save_config,
            ui_log,
            app_version,
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
            chat_send,
            chat_cancel,
            chat_cancel_all,
            chat_targets,
            chat_props,
            chat_list,
            chat_load,
            chat_save,
            chat_delete,
            chat_delete_all,
            chat_presets_get,
            chat_presets_save,
            hub_search,
            wizard_inspect,
            wizard_plan,
            wizard_start,
            wizard_cancel,
            wizard_jobs,
            wizard_forget,
            wizard_roots,
            wizard_add_root,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Llama FIDIM UI");
}
