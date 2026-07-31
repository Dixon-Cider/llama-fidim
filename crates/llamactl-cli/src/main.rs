use std::path::PathBuf;
use std::time::Duration;

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use llamactl_core::config::Config;
use llamactl_core::discovery::{self, Build};
use llamactl_core::launch;
use llamactl_core::platform::{Platform, WindowsPlatform};
use llamactl_core::preflight::{self, Outcome};
use llamactl_core::profile::{self, Profile};
use llamactl_core::supervise::{self, Health};
use llamactl_core::{export, gguf};

#[derive(Parser)]
#[command(name = "llamactl", version, about = "llama.cpp build/config manager")]
struct Cli {
    /// Emit JSON instead of tables.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Scan configured roots for builds and models.
    Scan,
    /// Enumerate GPUs fresh: stable keys, indices, VRAM, displays.
    Devices {
        #[arg(long)]
        build: Option<String>,
    },
    /// List saved profiles with validation findings.
    Profiles,
    /// Run the pre-flight sequence for a profile without launching.
    Check { profile_id: String },
    /// Pre-flight then launch a profile; waits for readiness.
    Launch {
        profile_id: String,
        /// Proceed past Block findings (the confirmation that names the risk).
        #[arg(long)]
        override_blocks: bool,
        /// Seconds to wait for /v1/models before declaring failure.
        #[arg(long, default_value = "300")]
        ready_timeout: u64,
    },
    /// Stop a running server by profile id or port.
    Stop { target: String },
    /// Show running servers (re-attaches from state files).
    Status {
        /// Also probe generation with a 1-token completion.
        #[arg(long)]
        deep: bool,
    },
    /// Export a profile as a standalone script.
    Export {
        profile_id: String,
        /// bat or ps1.
        #[arg(long, default_value = "bat")]
        format: String,
        /// Output path (default: <profile>.<ext> in the current directory).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Print the log path and last lines for a run.
    Logs {
        target: String,
        #[arg(long, default_value = "40")]
        tail: usize,
    },
    /// Write starter profiles translated from the existing batch files.
    Seed,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load_or_init().context("loading llamactl config")?;
    let platform = WindowsPlatform;
    match cli.command {
        Cmd::Scan => cmd_scan(&cfg, cli.json),
        Cmd::Devices { build } => cmd_devices(&cfg, build.as_deref(), cli.json),
        Cmd::Profiles => cmd_profiles(&cfg, cli.json),
        Cmd::Check { profile_id } => cmd_check(&cfg, &platform, &profile_id, cli.json),
        Cmd::Launch { profile_id, override_blocks, ready_timeout } => {
            cmd_launch(&cfg, &platform, &profile_id, override_blocks, ready_timeout)
        }
        Cmd::Stop { target } => cmd_stop(&cfg, &target),
        Cmd::Status { deep } => cmd_status(&cfg, deep, cli.json),
        Cmd::Export { profile_id, format, out } => {
            cmd_export(&cfg, &platform, &profile_id, &format, out)
        }
        Cmd::Logs { target, tail } => cmd_logs(&cfg, &target, tail),
        Cmd::Seed => cmd_seed(&cfg),
    }
}

// ------------------------------------------------------------------ scan ----

fn cmd_scan(cfg: &Config, json: bool) -> anyhow::Result<()> {
    let builds = discovery::scan_builds(&cfg.build_roots, cfg.rocm_bin.as_deref());
    let models = discovery::scan_models(&cfg.model_roots);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({"builds": builds, "models": models}))?
        );
        return Ok(());
    }
    println!("BUILDS ({})", builds.len());
    for b in &builds {
        let ver = b.version.as_deref().unwrap_or("?");
        let commit = b.commit.as_deref().unwrap_or("-");
        let err = b
            .version_error
            .as_deref()
            .map(|e| format!("  [BROKEN: {e}]"))
            .unwrap_or_default();
        println!("  {:<22} {:<7} {:<11} {}{err}", b.tag, ver, commit, b.path.display());
    }
    println!("\nMODELS ({})", models.len());
    for m in &models {
        let gb = m.file_size as f64 / (1024.0 * 1024.0 * 1024.0);
        let (arch, quant, layers) = match &m.header {
            Some(h) => (
                h.architecture.clone().unwrap_or_else(|| "?".into()),
                h.file_type.map(gguf::file_type_name).unwrap_or_else(|| "?".into()),
                h.block_count.map(|b| b.to_string()).unwrap_or_else(|| "?".into()),
            ),
            None => ("PARSE-ERROR".into(), "-".into(), "-".into()),
        };
        let extras = format!(
            "{}{}",
            if m.mmproj_candidates.is_empty() { "" } else { " +mmproj" },
            if m.draft_candidates.is_empty() { "" } else { " +draft" },
        );
        println!("  {:>7.2} GB  {:<10} {:<8} {:>3}L{}  {}", gb, arch, quant, layers, extras,
            m.path.display());
        if let Some(e) = &m.header_error {
            println!("           [header error: {e}]");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- devices ----

fn cmd_devices(cfg: &Config, build_tag: Option<&str>, json: bool) -> anyhow::Result<()> {
    let platform = WindowsPlatform;
    let builds = discovery::scan_builds(&cfg.build_roots, cfg.rocm_bin.as_deref());
    let build = pick_build(&builds, build_tag)?;
    let devices = launch::enumerate_devices(cfg, &build.server_exe, &platform)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&devices)?);
        return Ok(());
    }
    // Occupancy from run states.
    let runs = supervise::reattach(&cfg.runs_dir);
    println!(
        "{:<4} {:<26} {:<44} {:>9} {:>9}  {:<9} {:<16} {:<16} {}",
        "IDX", "NAME", "STABLE KEY", "TOTAL", "FREE", "CLASS", "DISPLAY", "DRIVER", "OCCUPIED BY"
    );
    for d in &devices {
        let class = if d.integrated { "iGPU" } else { "discrete" };
        let display = d
            .display
            .as_ref()
            .map(|m| format!("{}x{}@{}", m.width, m.height, m.refresh_hz))
            .unwrap_or_else(|| "-".into());
        let occupied: Vec<&str> = runs
            .iter()
            .filter(|r| r.alive && r.state.device_keys.contains(&d.stable_key))
            .map(|r| r.state.profile_id.as_str())
            .collect();
        let assumed = if d.correlation_assumed { "~" } else { " " };
        println!(
            "{}{:<3} {:<26} {:<44} {:>7}MB {:>7}MB  {:<9} {:<16} {:<16} {}",
            assumed,
            format!("{}{}", d.backend, d.hip_index),
            d.name,
            d.stable_key,
            d.total_mib,
            d.free_mib,
            class,
            display,
            d.driver_version.as_deref().unwrap_or("-"),
            if occupied.is_empty() { "-".into() } else { occupied.join(",") },
        );
    }
    println!("\n~ = identical-name correlation by bus order (verified at launch by residency check)");
    Ok(())
}

fn pick_build<'b>(builds: &'b [Build], tag: Option<&str>) -> anyhow::Result<&'b Build> {
    if builds.is_empty() {
        bail!("no builds found under configured build_roots");
    }
    match tag {
        Some(t) => builds
            .iter()
            .find(|b| b.tag == t)
            .with_context(|| format!("no build tagged {t}")),
        None => Ok(builds
            .iter()
            .filter(|b| b.version.is_some())
            .max_by(|a, b| a.version.cmp(&b.version))
            .unwrap_or(&builds[0])),
    }
}

// --------------------------------------------------------------- profiles ----

fn cmd_profiles(cfg: &Config, json: bool) -> anyhow::Result<()> {
    let profiles = Profile::load_all(&cfg.profile_dir)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&profiles)?);
        return Ok(());
    }
    if profiles.is_empty() {
        println!(
            "no profiles in {} — `llamactl seed` writes starters from the existing batch files",
            cfg.profile_dir.display()
        );
        return Ok(());
    }
    for p in &profiles {
        let findings = profile::validate(p);
        let devs = p.devices.len();
        let split = match (devs, p.split_mode) {
            (n, Some(m)) if n > 1 => format!(" split={m:?}"),
            _ => String::new(),
        };
        println!(
            "{:<16} port {:<6} alias {:<18} {} device(s){split}  [{}]",
            p.id,
            p.server.port,
            p.server.alias,
            devs,
            p.build.version.as_deref().unwrap_or("?")
        );
        for f in findings {
            println!("    {:?}: {} ({})", f.severity, f.message, f.code);
        }
    }
    Ok(())
}

fn load_profile(cfg: &Config, id: &str) -> anyhow::Result<Profile> {
    let path = cfg.profile_dir.join(format!("{id}.json"));
    Profile::load(&path).with_context(|| format!("loading profile {id} from {}", path.display()))
}

// ------------------------------------------------------------------ check ----

fn print_results(results: &[preflight::CheckResult]) {
    for r in results {
        let (tag, msg) = match &r.outcome {
            Outcome::Pass => ("PASS ", String::new()),
            Outcome::Note(m) => ("NOTE ", m.clone()),
            Outcome::Warn(m) => ("WARN ", m.clone()),
            Outcome::Block(m) => ("BLOCK", m.clone()),
        };
        println!("  [{tag}] {:>2}. {}", r.spec_number, r.title);
        if !msg.is_empty() {
            for line in textwrap(&msg, 96) {
                println!("           {line}");
            }
        }
    }
}

fn textwrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if !current.is_empty() && current.len() + word.len() + 1 > width {
            lines.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

fn prepare(
    cfg: &Config,
    platform: &dyn Platform,
    profile_id: &str,
) -> anyhow::Result<(Profile, launch::PreparedLaunch)> {
    let profile = load_profile(cfg, profile_id)?;
    let findings = profile::validate(&profile);
    for f in &findings {
        println!("  [{:?}] {} ({})", f.severity, f.message, f.code);
    }
    if findings.iter().any(|f| f.severity == profile::Severity::Error) {
        bail!("profile {profile_id} has validation errors — fix the profile JSON first");
    }
    let running_aliases: Vec<String> = supervise::reattach(&cfg.runs_dir)
        .into_iter()
        .filter(|r| r.alive)
        .map(|r| r.state.alias)
        .collect();
    let prepared = launch::prepare(cfg, &profile, platform, running_aliases)?;
    Ok((profile, prepared))
}

fn cmd_check(
    cfg: &Config,
    platform: &dyn Platform,
    profile_id: &str,
    json: bool,
) -> anyhow::Result<()> {
    let (_, prepared) = prepare(cfg, platform, profile_id)?;
    let results = preflight::run_all(&prepared.context);
    if json {
        println!("{}", serde_json::to_string_pretty(&results)?);
        return Ok(());
    }
    println!("pre-flight for {profile_id}:");
    print_results(&results);
    if let Some(est) = &prepared.context.estimate {
        println!("\n  estimate assumptions:");
        for a in &est.assumptions {
            println!("    - {a}");
        }
    }
    println!("\n  command: {}", prepared.plan.command_line());
    println!(
        "  env: {}",
        prepared
            .plan
            .env
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    if preflight::any_block(&results) {
        println!("\nBLOCKED — fix the findings above, or launch with --override-blocks to accept the named risks");
    }
    Ok(())
}

// ----------------------------------------------------------------- launch ----

fn cmd_launch(
    cfg: &Config,
    platform: &dyn Platform,
    profile_id: &str,
    override_blocks: bool,
    ready_timeout: u64,
) -> anyhow::Result<()> {
    let (mut profile, prepared) = prepare(cfg, platform, profile_id)?;
    let results = preflight::run_all(&prepared.context);
    print_results(&results);
    if preflight::any_block(&results) {
        if !override_blocks {
            bail!("pre-flight blocked launch; rerun with --override-blocks to accept the named risks");
        }
        println!("\n!! overriding {} Block finding(s) — the risks above are accepted by --override-blocks",
            results.iter().filter(|r| r.blocks()).count());
    }

    // Persist rebound keys + resolved indices (informational field).
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
        profile.save(&cfg.profile_dir.join(format!("{}.json", profile.id)))?;
        println!("profile updated with re-resolved device bindings");
    }

    let device_keys: Vec<String> =
        prepared.context.resolved.iter().map(|r| r.device.stable_key.clone()).collect();
    let free_before: Vec<u64> =
        prepared.context.resolved.iter().map(|r| r.device.free_mib).collect();
    let state = supervise::spawn(&prepared.plan, &profile, &cfg.runs_dir, device_keys, free_before)?;
    println!(
        "launched pid {} on port {} — waiting for /v1/models (log: {})",
        state.pid,
        state.port,
        state.log_path.display()
    );
    supervise::wait_ready(&state, Duration::from_secs(ready_timeout))?;
    println!("ready.");
    verify_residency(platform, &state, &prepared);
    Ok(())
}

/// Residency ground truth (acceptance §09): PDH per-process dedicated GPU
/// memory, attributed to physical cards via adapter LUIDs. WDDM virtualizes
/// VRAM, so free-memory deltas from `--list-devices` cannot see another
/// process's allocations — per-process counters can.
fn verify_residency(
    platform: &dyn Platform,
    state: &supervise::RunState,
    prepared: &launch::PreparedLaunch,
) {
    let mem = match platform.gpu_process_memory(state.pid) {
        Ok(m) => m,
        Err(e) => {
            println!("  residency: GPU counters unavailable ({e}) — cannot verify placement");
            return;
        }
    };
    let mut misplaced = false;
    for (i, r) in prepared.context.resolved.iter().enumerate() {
        let expected_mib = prepared
            .context
            .estimate
            .as_ref()
            .and_then(|est| est.per_device.get(i))
            .map(|d| d.total_bytes / (1024 * 1024))
            .unwrap_or(0);
        let entry = r
            .device
            .luid_low
            .and_then(|luid| mem.iter().find(|m| m.luid_low == luid));
        match entry {
            Some(m) => {
                let dedicated = m.dedicated_bytes / (1024 * 1024);
                let committed = m.committed_bytes / (1024 * 1024);
                // Committed proves placement immediately; dedicated fills on
                // the first forward pass (WDDM residency is on-demand).
                let placed = expected_mib == 0
                    || (committed.max(dedicated) as f64) >= expected_mib as f64 * 0.5;
                let verdict = if !placed {
                    misplaced = true;
                    "!! FAR BELOW estimate — placement suspect"
                } else if (dedicated as f64) < expected_mib as f64 * 0.5 {
                    "placed (becomes VRAM-resident on first inference)"
                } else {
                    "resident on the intended card"
                };
                println!(
                    "  residency {}: {dedicated} MiB resident / {committed} MiB committed (estimated {expected_mib} MiB) — {verdict}",
                    r.device.stable_key
                );
            }
            None => {
                misplaced = true;
                println!(
                    "  residency {}: NO memory attributed to this card (estimated {expected_mib} MiB)",
                    r.device.stable_key
                );
            }
        }
    }
    // Memory on cards the profile never chose is the silent-misplacement case.
    for m in &mem {
        let known = prepared
            .context
            .resolved
            .iter()
            .any(|r| r.device.luid_low == Some(m.luid_low));
        if !known && m.dedicated_bytes.max(m.committed_bytes) > 512 * 1024 * 1024 {
            misplaced = true;
            println!(
                "  residency: !! {} MiB on UNINTENDED adapter luid 0x{:X}",
                m.dedicated_bytes.max(m.committed_bytes) / (1024 * 1024),
                m.luid_low
            );
        }
    }
    if !misplaced {
        println!("  placement verified: every allocation sits on a profile-selected card");
    }
}

// ------------------------------------------------------------- stop/status ----

fn find_run(cfg: &Config, target: &str) -> anyhow::Result<supervise::AttachedRun> {
    let runs = supervise::reattach(&cfg.runs_dir);
    runs.into_iter()
        .find(|r| r.state.profile_id == target || r.state.port.to_string() == target)
        .with_context(|| format!("no run state matches {target:?} (try `llamactl status`)"))
}

fn cmd_stop(cfg: &Config, target: &str) -> anyhow::Result<()> {
    let run = find_run(cfg, target)?;
    supervise::stop(&run.state, &cfg.runs_dir)?;
    println!("stopped {} (pid {}, port {})", run.state.profile_id, run.state.pid, run.state.port);
    Ok(())
}

fn cmd_status(cfg: &Config, deep: bool, json: bool) -> anyhow::Result<()> {
    let mut runs = supervise::reattach(&cfg.runs_dir);
    if deep {
        for r in &mut runs {
            if r.alive {
                r.health = supervise::probe_health(&r.state, true);
            }
        }
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&runs)?);
        return Ok(());
    }
    if runs.is_empty() {
        println!("no runs (state dir: {})", cfg.runs_dir.display());
        return Ok(());
    }
    for r in &runs {
        let health = match (&r.crashed, &r.health) {
            (true, _) => "CRASHED (state file kept; see log)".into(),
            (false, Health::Healthy) => "healthy".to_string(),
            (false, Health::RespondingNotGenerating) => "responding but NOT generating".into(),
            (false, Health::Dead) => "dead (process alive, HTTP down)".into(),
        };
        let uptime = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|n| n.as_secs().saturating_sub(r.state.started_unix))
            .unwrap_or(0);
        println!(
            "{:<16} pid {:<7} port {:<6} alias {:<18} up {:>6}s  {}",
            r.state.profile_id, r.state.pid, r.state.port, r.state.alias, uptime, health
        );
        println!("    devices {} | vis {} | log {}",
            r.state.device_keys.join(", "),
            r.state.visibility_env,
            r.state.log_path.display());
    }
    Ok(())
}

// ----------------------------------------------------------- export/logs ----

fn cmd_export(
    cfg: &Config,
    platform: &dyn Platform,
    profile_id: &str,
    format: &str,
    out: Option<PathBuf>,
) -> anyhow::Result<()> {
    let (profile, prepared) = prepare(cfg, platform, profile_id)?;
    let (text, ext) = match format {
        "bat" => (export::to_bat(&profile, &prepared.plan), "bat"),
        "ps1" => (export::to_ps1(&profile, &prepared.plan), "ps1"),
        other => bail!("unknown format {other:?} (bat|ps1)"),
    };
    let out = out.unwrap_or_else(|| PathBuf::from(format!("{profile_id}.{ext}")));
    std::fs::write(&out, text).with_context(|| format!("writing {}", out.display()))?;
    println!("exported {} -> {}", profile_id, out.display());
    Ok(())
}

fn cmd_logs(cfg: &Config, target: &str, tail: usize) -> anyhow::Result<()> {
    let run = find_run(cfg, target)?;
    println!("log: {}", run.state.log_path.display());
    let text = std::fs::read_to_string(&run.state.log_path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    for line in lines.iter().rev().take(tail).rev() {
        println!("{line}");
    }
    Ok(())
}

// ------------------------------------------------------------------- seed ----

/// Starter profiles translated from tools/start-gemma-*.bat — the parity
/// baseline for the export acceptance test.
fn cmd_seed(cfg: &Config) -> anyhow::Result<()> {
    let worker = serde_json::json!({
        "schema": 1,
        "id": "worker-pool",
        "name": "Subagent worker pool (from start-gemma-workers.bat)",
        "build": { "path": "C:\\Users\\me\\Documents\\Claude\\Projects\\AMD GPU Programming\\llama.cpp\\build-hip-vision", "version": "b9817" },
        "model": {
            "path": "E:\\models\\unsloth\\gemma-4-26B-A4B-it-qat-GGUF\\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
            "draft": { "path": "E:\\models\\unsloth\\gemma-4-26B-A4B-it-qat-GGUF\\MTP\\mtp-gemma-4-26B-A4B-it-Q8_0.gguf", "enabled": false }
        },
        "devices": [ { "key": "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08" } ],
        "server": { "port": 9701, "alias": "gemma-4-worker", "host": "127.0.0.1" },
        "runtime": {
            "n_gpu_layers": 99, "ctx_total": 393216, "slots": 6,
            "kv_type_k": "q8_0", "kv_type_v": "q8_0", "flash_attn": "on",
            "batch_logical": 2048, "batch_physical": 256, "cont_batching": true
        },
        "sampling": { "temperature": 1.0, "top_p": 0.95, "top_k": 64, "dry_multiplier": 0.8 },
        "chat": { "enable_thinking": false },
        "env": { "GPU_MAX_HW_QUEUES": "1", "ROCBLAS_USE_HIPBLASLT": "0" },
        "notes": "Knee is at 6 slots; 6->8 buys +2.4% aggregate for -23% per-stream. Swept 2026-07-29."
    });
    let orchestrator = serde_json::json!({
        "schema": 1,
        "id": "orchestrator",
        "name": "Orchestrator 31B (from start-gemma-moe.bat lineage)",
        "build": { "path": "C:\\Users\\me\\Documents\\Claude\\Projects\\AMD GPU Programming\\llama.cpp\\build-hip", "version": "b9553" },
        "model": {
            "path": "E:\\models\\gemma-4-31B\\gemma-4-31B-it-UD-Q5_K_XL.gguf",
            "draft": { "path": "E:\\models\\gemma-4-31B\\MTP\\gemma-4-31B-it-Q8_0-MTP.gguf", "enabled": false }
        },
        "devices": [ { "key": "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus03" } ],
        "server": { "port": 9700, "alias": "gemma-4", "host": "127.0.0.1" },
        "runtime": {
            "n_gpu_layers": 99, "ctx_total": 215040, "slots": 1,
            "kv_type_k": "q4_0", "kv_type_v": "q4_0", "flash_attn": "on",
            "batch_logical": 2048, "batch_physical": 512, "cont_batching": true
        },
        "sampling": { "dry_multiplier": 0.8 },
        "chat": { "enable_thinking": false },
        "env": { "GPU_MAX_HW_QUEUES": "1", "ROCBLAS_USE_HIPBLASLT": "0" },
        "notes": "q4_0 KV validated to 246K multi-hop; ~29.25GB at 210K ctx, ~43 t/s decode."
    });
    std::fs::create_dir_all(&cfg.profile_dir)?;
    for p in [worker, orchestrator] {
        let parsed: Profile = serde_json::from_value(p)?;
        let path = cfg.profile_dir.join(format!("{}.json", parsed.id));
        if path.exists() {
            println!("kept existing {}", path.display());
            continue;
        }
        parsed.save(&path)?;
        println!("wrote {}", path.display());
    }
    println!("\nNOTE: verify the draft-model paths — seed guesses the MTP filenames; `llamactl scan` shows the real ones.");
    Ok(())
}
