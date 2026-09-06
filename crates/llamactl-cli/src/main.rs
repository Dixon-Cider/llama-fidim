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
enum RouterCmd {
    /// Show the router config and rendered preset file.
    Show,
    /// Add a profile as a member (its alias becomes the model id).
    Add {
        profile_id: String,
        #[arg(long)]
        load_on_startup: bool,
    },
    /// Remove a member.
    Remove { profile_id: String },
    /// Change router settings.
    Set {
        #[arg(long)]
        port: Option<u16>,
        #[arg(long)]
        models_max: Option<u32>,
        #[arg(long)]
        autoload: Option<bool>,
        /// Build directory, or `newest`.
        #[arg(long)]
        build: Option<String>,
    },
    /// Write the preset file, take the port over, launch, wait for readiness.
    Launch {
        #[arg(long, default_value = "120")]
        ready_timeout: u64,
    },
    /// Stop the router (and its model instances).
    Stop,
    /// Models the running router knows, with load state.
    Status,
    /// Load a model on the running router.
    Load { model_id: String },
    /// Unload a model from the running router.
    Unload { model_id: String },
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
    /// List ROCm runtimes (HIP SDK, installed from AMD, LM Studio, manual) a profile can name.
    Runtimes,
    /// ROCm runtimes from AMD: list what is installable, install one, remove one.
    Rocm {
        #[command(subcommand)]
        cmd: RocmCmd,
    },
    /// The model author's published sampling defaults (Hugging Face generation_config.json).
    CreatorDefaults { model_path: PathBuf },
    /// Parse one GGUF header and report what the scanner would see (and how long it took).
    Gguf { path: PathBuf },
    /// One-shot live view: phase, progress and throughput of every running server.
    Live,
    /// One OpenAI-compatible port serving every member profile (llama-server router mode).
    Router {
        #[command(subcommand)]
        cmd: RouterCmd,
    },
    /// (internal) keep a server's VRAM resident: 1-token request every --interval s until --server-pid exits.
    #[command(hide = true)]
    Keepalive {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long)]
        port: u16,
        #[arg(long, default_value = "5")]
        interval: u32,
        #[arg(long)]
        server_pid: u32,
    },
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
    /// Benchmark a running server and store the baseline on its profile.
    Bench {
        profile_id: String,
        /// Concurrent streams (default: the profile's slot count).
        #[arg(long)]
        concurrency: Option<u32>,
        /// Decode length per measured request.
        #[arg(long, default_value = "256")]
        tokens: u32,
        #[arg(long, default_value = "2")]
        warmups: u32,
        /// Skip writing the baseline (measure and print only).
        #[arg(long)]
        no_save: bool,
    },
    /// Check upstream llama.cpp releases; optionally install, promote, roll back.
    /// Never launches a server or loads a model.
    Update {
        /// Download + verify the prebuilt Windows ROCm build (latest, or --tag).
        #[arg(long)]
        install: bool,
        /// Build from source with config.source_build_script instead of prebuilt.
        #[arg(long)]
        source: bool,
        /// Re-point profiles onto the installed build (default scope: profiles
        /// on the previous newest build; --all for every unpinned profile).
        #[arg(long)]
        promote: bool,
        #[arg(long)]
        all: bool,
        /// Undo the most recent promotion.
        #[arg(long)]
        rollback: bool,
        /// Specific release tag (default: latest).
        #[arg(long)]
        tag: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load_or_init().context("loading llamactl config")?;
    let platform = WindowsPlatform;
    match cli.command {
        Cmd::Scan => cmd_scan(&cfg, cli.json),
        Cmd::Devices { build } => cmd_devices(&cfg, build.as_deref(), cli.json),
        Cmd::Profiles => cmd_profiles(&cfg, cli.json),
        Cmd::Runtimes => cmd_runtimes(&cfg, cli.json),
        Cmd::Keepalive { host, port, interval, server_pid } => {
            supervise::run_keepalive(&host, port, interval, server_pid);
            Ok(())
        }
        Cmd::Live => cmd_live(&cfg, cli.json),
        Cmd::Router { cmd } => cmd_router(&cfg, cli.json, cmd),
        Cmd::Gguf { path } => {
            let t = std::time::Instant::now();
            let h = gguf::read_header(&path)?;
            let ms = t.elapsed().as_millis();
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&h)?);
            } else {
                println!(
                    "{} ms  {}  arch={} layers={} ctx={} ft={} keys={} sampling(temp={:?} top_k={:?} top_p={:?}) repo={:?}",
                    ms,
                    path.display(),
                    h.architecture.as_deref().unwrap_or("?"),
                    h.block_count.unwrap_or(0),
                    h.context_length.unwrap_or(0),
                    h.file_type.unwrap_or(0),
                    h.metadata.len(),
                    h.sampling_temp,
                    h.sampling_top_k,
                    h.sampling_top_p,
                    h.source_repo
                );
            }
            Ok(())
        }
        Cmd::CreatorDefaults { model_path } => {
            let d = llamactl_core::hf::creator_defaults(&cfg, &model_path)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&d)?);
            } else {
                println!("{}  ({}{})", d.repo, if d.from_cache { "cached, " } else { "" }, d.url);
                let f = |v: Option<f64>| v.map(|x| x.to_string()).unwrap_or_else(|| "-".into());
                println!("  temperature {}  top_p {}  top_k {}  min_p {}  repetition_penalty {}",
                    f(d.temperature), f(d.top_p), d.top_k.map(|k| k.to_string()).unwrap_or_else(|| "-".into()),
                    f(d.min_p), f(d.repetition_penalty));
            }
            Ok(())
        }
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
        Cmd::Bench { profile_id, concurrency, tokens, warmups, no_save } => {
            cmd_bench(&cfg, &platform, &profile_id, concurrency, tokens, warmups, no_save)
        }
        Cmd::Rocm { cmd } => cmd_rocm(&cfg, cli.json, cmd),
        Cmd::Update { install, source, promote, all, rollback, tag } => {
            cmd_update(&cfg, cli.json, install, source, promote, all, rollback, tag)
        }
    }
}

// -------------------------------------------------------------- runtimes ----

#[derive(Subcommand, Debug)]
enum RocmCmd {
    /// Versions installable from AMD's release and nightly channels, newest first.
    List {
        /// GPU family for nightlies (default: config.rocm_family, else gfx120X-all).
        #[arg(long)]
        family: Option<String>,
    },
    /// Download and unpack a version into <install root>\rocm\<version>.
    Install {
        version: String,
        #[arg(long)]
        family: Option<String>,
    },
    /// Delete an installed version (never the default).
    Remove { version: String },
}

fn cmd_rocm(cfg: &Config, json: bool, cmd: RocmCmd) -> anyhow::Result<()> {
    use llamactl_core::rocm;
    let fam = |f: Option<String>| f.or_else(|| cfg.rocm_family.clone()).unwrap_or_else(|| "gfx120X-all".to_string());
    match cmd {
        RocmCmd::List { family } => {
            let family = fam(family);
            let (list, problems) = rocm::available(&family)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "family": family, "runtimes": list, "problems": problems }))?);
                return Ok(());
            }
            let installed: Vec<String> = rocm::installed(cfg).into_iter().map(|(_, m)| m.version).collect();
            println!("{:<22} {:<8} {}", "VERSION", "CHANNEL", "STATE");
            for a in &list {
                println!("{:<22} {:<8} {}", a.version, a.channel, if installed.contains(&a.version) { "installed" } else { "" });
            }
            for p in problems {
                eprintln!("note: {p}");
            }
            println!("\nnightlies are for family {family}; `llamactl rocm install <version>` puts one under {}", rocm::runtimes_root(cfg).display());
        }
        RocmCmd::Install { version, family } => {
            let (list, _) = rocm::available(&fam(family))?;
            let a = list
                .into_iter()
                .find(|a| a.version == version)
                .ok_or_else(|| anyhow::anyhow!("ROCm {version} is not on either channel; see `llamactl rocm list`"))?;
            let mut progress = |line: String| println!("{line}");
            let dir = rocm::install(cfg, &a, &mut progress)?;
            println!("installed {} -> select it as runtime `rocm-{}`", dir.display(), a.version);
        }
        RocmCmd::Remove { version } => {
            rocm::remove(cfg, &version)?;
            println!("removed ROCm {version}");
        }
    }
    Ok(())
}

fn cmd_runtimes(cfg: &Config, json: bool) -> anyhow::Result<()> {
    let all = llamactl_core::runtime::discover(cfg);
    if json {
        println!("{}", serde_json::to_string_pretty(&all)?);
        return Ok(());
    }
    println!("{:<34} {:<9} {:<9} {:<5} {}", "NAME", "SOURCE", "VERSION", "OK", "DIRS");
    for r in &all {
        println!(
            "{:<34} {:<9} {:<9} {:<5} {}{}",
            r.name,
            r.source,
            r.version.as_deref().unwrap_or("-"),
            if r.available { "yes" } else { "NO" },
            r.dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(" ; "),
            if r.is_default { "   <- default" } else { "" }
        );
    }
    println!("\nselect per profile with \"rocm_runtime\": \"<name>\"; change the default with config.default_runtime.");
    Ok(())
}

// ---------------------------------------------------------------- update ----

#[allow(clippy::too_many_arguments)]
fn cmd_update(
    cfg: &Config,
    json: bool,
    install: bool,
    source: bool,
    promote: bool,
    all: bool,
    rollback: bool,
    tag: Option<String>,
) -> anyhow::Result<()> {
    use llamactl_core::update::{self, PromoteScope};

    if rollback {
        let r = update::rollback(cfg)?;
        if json {
            println!("{}", serde_json::to_string_pretty(&r)?);
        } else {
            println!("rolled back promotion from {}", r.batch_at_unix);
            for e in &r.restored {
                println!("  {:<20} -> {} ({})", e.profile_id, e.from.path.display(), e.from.version.as_deref().unwrap_or("?"));
            }
            for (id, why) in &r.skipped {
                println!("  {id:<20} skipped: {why}");
            }
        }
        return Ok(());
    }

    let builds = discovery::scan_builds(&cfg.build_roots, cfg.rocm_bin.as_deref());
    let check = match &tag {
        Some(t) => update::check_against(cfg, &builds, update::release_by_tag(t)?)?,
        None => update::check(cfg)?,
    };
    let release = check.latest.clone();
    let previous = check.newest_installed.clone();
    if !json {
        println!("upstream latest : {} ({})", check.latest.tag, check.latest.published_at);
        match &previous {
            Some(n) => println!(
                "newest installed: {} at {}{}",
                n.version,
                n.path.display(),
                check.behind.map(|b| format!("  [{b} releases behind]")).unwrap_or_default()
            ),
            None => println!("newest installed: none"),
        }
        println!("install dir     : {}{}", check.install_dir.display(), if check.already_installed { "  [present]" } else { "" });
        if !check.changes.is_empty() {
            println!("what changed{}:", if check.changes_complete { "" } else { " (newest 100 only)" });
            for c in &check.changes {
                println!("  {:<8} {}  {}", c.tag, c.published_at.get(..10).unwrap_or(""), c.title);
            }
        }
        if let Some(e) = &check.asset_error {
            println!("prebuilt        : unavailable — {e}");
        }
    }
    if !install && !promote {
        if json {
            println!("{}", serde_json::to_string_pretty(&check)?);
        } else if !check.update_available {
            println!("up to date.");
        } else {
            println!("run with --install to fetch it (add --source to compile instead).");
        }
        return Ok(());
    }

    let mut progress = |line: String| {
        if !json {
            println!("  {line}");
        }
    };
    let report = if install {
        let r = if source {
            update::build_from_source(cfg, &release.tag, &mut progress)?
        } else {
            update::install_prebuilt(cfg, &release, &mut progress)?
        };
        if !json {
            let v = &r.verify;
            println!(
                "installed {} ({}) at {} — binary reports {}; HIP {}",
                r.tag,
                r.source,
                r.dir.display(),
                v.version.as_deref().unwrap_or("?"),
                if v.hip_ok { "OK" } else { "NOT LOADED" }
            );
            for d in &v.devices {
                println!("    {}{}  {}  {} MiB", d.backend, d.index, d.name, d.total_mib);
            }
            if !v.detail.is_empty() {
                println!("    {}", v.detail.trim().replace('\n', "\n    "));
            }
        }
        Some(r)
    } else {
        None
    };

    if promote {
        let (to_dir, to_version) = match &report {
            Some(r) => (r.dir.clone(), r.verify.version.clone()),
            None => {
                // --promote without --install: promote onto the newest
                // installed build (already verified by the scan).
                let n = update::newest_installed(&builds).context("no installed build to promote onto")?;
                (n.path.clone(), Some(n.version.clone()))
            }
        };
        if let Some(r) = &report {
            if !r.verify.hip_ok {
                bail!("refusing to promote onto {}: the HIP backend did not load (see detail above)", r.dir.display());
            }
        }
        let scope = if all {
            PromoteScope::All
        } else {
            match &previous {
                Some(p) if !llamactl_core::update::version_number(&p.version).is_none() => {
                    PromoteScope::FromBuild(p.path.clone())
                }
                _ => PromoteScope::All,
            }
        };
        let r = update::promote(cfg, &to_dir, to_version, scope)?;
        if json {
            println!("{}", serde_json::to_string_pretty(&r)?);
        } else {
            println!("promoted {} profile(s) onto {}:", r.batch.entries.len(), to_dir.display());
            for e in &r.batch.entries {
                println!("  {:<20} {} -> {}", e.profile_id, e.from.version.as_deref().unwrap_or("?"), e.to.version.as_deref().unwrap_or("?"));
            }
            for (id, why) in &r.skipped {
                println!("  {id:<20} skipped: {why}");
            }
            println!("nothing was launched — bench a profile when the GPUs are free; `llamactl update --rollback` undoes this.");
        }
    } else if json {
        if let Some(r) = &report {
            println!("{}", serde_json::to_string_pretty(r)?);
        }
    }
    Ok(())
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
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        println!("\n  VRAM estimate: {:.2} GiB total", est.total_bytes as f64 / GIB);
        for d in &est.per_device {
            let free = prepared
                .context
                .resolved
                .iter()
                .find(|r| r.device.stable_key == d.key)
                .map(|r| r.device.free_mib as f64 / 1024.0);
            println!(
                "    {}: {:.2} GiB = weights {:.2} + kv {:.2} + compute {:.2} + overhead {:.2}{}",
                d.key.rsplit(':').next().unwrap_or(&d.key),
                d.total_bytes as f64 / GIB,
                d.weights_bytes as f64 / GIB,
                d.kv_bytes as f64 / GIB,
                d.compute_bytes as f64 / GIB,
                d.overhead_bytes as f64 / GIB,
                free.map(|f| format!("  (free {f:.2} GiB, {:.0}%)", 100.0 * d.total_bytes as f64 / GIB / f)).unwrap_or_default()
            );
        }
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
    // Port takeover (check 8 warned): a llamactl server on this port is
    // stopped first so the new model answers on the same well-known port.
    if let Some(h) = &prepared.context.port_holder {
        if h.profile_id.is_some() {
            if let Some(run) = supervise::reattach(&cfg.runs_dir)
                .into_iter()
                .find(|r| r.alive && r.state.port == profile.server.port)
            {
                supervise::stop(&run.state, &cfg.runs_dir)?;
                println!("stopped {} (pid {}) to take over port {}", run.state.profile_id, run.state.pid, profile.server.port);
                launch::wait_port_free(&profile.server.host, profile.server.port, Duration::from_secs(20))?;
            }
        }
    }
    let cold_start = supervise::is_cold_start(&Config::config_dir(), &profile.model.path);
    if cold_start {
        println!(
            "note: first load of this model file since it changed — the first benchmark will be \
             marked cold-cache (R-08)"
        );
    }
    // Final gate: pre-flight may be minutes stale by now.
    launch::final_commit_gate(
        cfg,
        platform,
        prepared.context.estimate.as_ref(),
        override_blocks,
    )?;
    let state = supervise::spawn(
        &prepared.plan,
        &profile,
        &cfg.runs_dir,
        device_keys,
        free_before,
        cold_start,
        prepared.context.estimate.as_ref().map(|e| e.total_bytes).unwrap_or(0),
    )?;
    println!(
        "launched pid {} on port {} — waiting for /v1/models (log: {})",
        state.pid,
        state.port,
        state.log_path.display()
    );
    supervise::wait_ready_in(&state, Duration::from_secs(ready_timeout), Some(&cfg.runs_dir))?;
    println!("ready.");
    supervise::record_model_loaded(&Config::config_dir(), &profile.model.path)?;
    let mut state = state;
    let ka = profile.keep_alive_seconds.unwrap_or(cfg.keep_alive_seconds);
    match supervise::spawn_keepalive(&mut state, &cfg.runs_dir, ka) {
        Ok(Some(kp)) => println!("  keep-alive every {ka} s (pid {kp}) - VRAM stays resident when the displays power off"),
        Ok(None) => {}
        Err(e) => println!("  keep-alive NOT started: {e}"),
    }
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
            (false, Health::Healthy) => match r.state.keepalive_pid {
                Some(kp) if supervise::process_alive(kp) => "healthy + keep-alive".to_string(),
                Some(_) => "healthy, keep-alive DIED".to_string(),
                None => "healthy".to_string(),
            },
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

// ------------------------------------------------------------------ bench ----

fn cmd_bench(
    cfg: &Config,
    platform: &dyn Platform,
    profile_id: &str,
    concurrency: Option<u32>,
    tokens: u32,
    warmups: u32,
    no_save: bool,
) -> anyhow::Result<()> {
    let profile = load_profile(cfg, profile_id)?;
    let run = find_run(cfg, profile_id)?;
    if run.crashed {
        bail!("run for {profile_id} has crashed — relaunch before benchmarking");
    }
    let n = concurrency.unwrap_or(profile.runtime.slots).max(1);
    let opts = llamactl_core::bench::BenchOptions {
        warmups,
        max_tokens: tokens,
        concurrency: n,
        ..Default::default()
    };
    println!(
        "sweeping {} on port {}: {} warmups, serial + {}x{} tokens{}",
        profile_id,
        run.state.port,
        warmups,
        n,
        tokens,
        if run.state.cold_start { "  [COLD-CACHE RUN]" } else { "" }
    );
    let sweep =
        llamactl_core::bench::run_sweep(&run.state.host, run.state.port, &run.state.alias, &opts)?;

    // Measured VRAM from the PDH counters, per profile device.
    let mem = platform.gpu_process_memory(run.state.pid).unwrap_or_default();
    let adapters = platform.video_adapters().unwrap_or_default();
    let key_to_luid: Vec<(String, Option<u64>)> = run
        .state
        .device_keys
        .iter()
        .map(|key| {
            let luid = adapters
                .iter()
                .find(|a| {
                    llamactl_core::devices::stable_key(&a.pnp_device_id, a.bus_number) == *key
                })
                .and_then(|a| a.luid_low);
            (key.clone(), luid)
        })
        .collect();
    let per_device_vram_gb: Vec<f64> = key_to_luid
        .iter()
        .map(|(_, luid)| {
            luid.and_then(|l| mem.iter().find(|m| m.luid_low == l))
                .map(|m| m.dedicated_bytes.max(m.committed_bytes) as f64 / (1024.0 * 1024.0 * 1024.0))
                .unwrap_or(0.0)
        })
        .collect();
    let vram_gb: f64 = per_device_vram_gb.iter().sum();

    let serial_tok_s = sweep.serial.decode_tok_s.unwrap_or(sweep.serial.wall_tok_s);
    println!("  serial: {serial_tok_s:.1} tok/s ({} tokens in {:.1}s wall)",
        sweep.serial.tokens, sweep.serial.wall_seconds);
    if let Some(c) = &sweep.concurrent {
        println!(
            "  concurrent n={}: wall aggregate {:.1} tok/s, decode aggregate {:.1} tok/s, per-stream {:.1} tok/s",
            c.n, c.aggregate_tok_s, c.decode_aggregate_tok_s, c.per_stream_tok_s
        );
    }
    println!("  resident VRAM: {vram_gb:.2} GiB ({})",
        per_device_vram_gb.iter().map(|g| format!("{g:.2}")).collect::<Vec<_>>().join(" + "));

    // Assemble the baseline record (§06 schema shape).
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let driver = adapters
        .iter()
        .find(|a| {
            key_to_luid.iter().any(|(k, l)| {
                l.is_some()
                    && *l == a.luid_low
                    && run.state.device_keys.contains(k)
            })
        })
        .map(|a| a.driver_version.clone());
    let sdk = launch::sdk_version(cfg.rocm_bin.as_deref());
    let split = profile.split_mode.map(|m| {
        serde_json::json!({
            "mode": m,
            "fractions": profile.devices.iter().map(|d| d.split_fraction).collect::<Vec<_>>(),
        })
    });
    let baseline = serde_json::json!({
        "measured_at": llamactl_core::bench::iso8601_utc(now),
        "measured_at_unix": now,
        "build_version": profile.build.version,
        "driver": driver,
        "sdk": sdk,
        "split": split,
        "vram_gb": (vram_gb * 100.0).round() / 100.0,
        "per_device_vram_gb": per_device_vram_gb.iter().map(|g| (g * 100.0).round() / 100.0).collect::<Vec<_>>(),
        "serial_tok_s": (serial_tok_s * 10.0).round() / 10.0,
        "concurrent": sweep.concurrent.as_ref().map(|c| serde_json::json!({
            "n": c.n,
            "aggregate_tok_s": (c.aggregate_tok_s * 10.0).round() / 10.0,
            "decode_aggregate_tok_s": (c.decode_aggregate_tok_s * 10.0).round() / 10.0,
            "per_stream_tok_s": (c.per_stream_tok_s * 10.0).round() / 10.0,
        })),
        "cold_cache": run.state.cold_start,
        "profile_fingerprint": llamactl_core::bench::profile_fingerprint(&profile),
        "bench": { "warmups": warmups, "tokens": tokens },
    });

    // History always gets the record; the profile baseline only warm runs.
    let hist_dir = Config::config_dir().join("benchmarks");
    std::fs::create_dir_all(&hist_dir)?;
    let hist_path = hist_dir.join(format!("{profile_id}.jsonl"));
    use std::io::Write as _;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&hist_path)?;
    writeln!(f, "{}", serde_json::to_string(&baseline)?)?;
    println!("  appended to history: {}", hist_path.display());

    if no_save {
        println!("  baseline NOT saved (--no-save)");
    } else if run.state.cold_start {
        println!(
            "  baseline NOT saved: cold-cache run (R-08) — a fresh model file measured 2.4x slow \
             once and nearly became a recorded number. Stop, relaunch, and bench again for a warm \
             baseline."
        );
    } else {
        let mut profile = profile;
        profile.baseline = Some(baseline);
        profile.save(&cfg.profile_dir.join(format!("{profile_id}.json")))?;
        println!("  baseline saved to profile {profile_id}");
    }
    Ok(())
}

// ------------------------------------------------------------------- seed ----

/// Starter profiles translated from tools/start-gemma-*.bat — the parity
/// baseline for the export acceptance test.
// ---------------------------------------------------------------- router ----

fn cmd_router(cfg: &Config, json: bool, cmd: RouterCmd) -> anyhow::Result<()> {
    use llamactl_core::router::{self, RouterMember};
    let mut rc = router::load_config()?;
    match cmd {
        RouterCmd::Show => {
            let profiles = Profile::load_all(&cfg.profile_dir)?;
            let builds = discovery::scan_builds(&cfg.build_roots, cfg.rocm_bin.as_deref());
            let platform = WindowsPlatform;
            let exe = pick_build(&builds, None)?.server_exe.clone();
            let devices = launch::enumerate_devices(cfg, &exe, &platform)?;
            let ini = router::render_ini(&rc, &profiles, &devices);
            if json {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({ "config": rc, "ini": ini.as_ref().map(|r| r.text.clone()).ok() }))?);
            } else {
                println!("router: http://{}:{}  models-max {}  autoload {}  build {}", rc.host, rc.port, rc.models_max, rc.autoload,
                    rc.build.as_ref().map(|b| b.display().to_string()).unwrap_or_else(|| "newest".into()));
                for m in &rc.members {
                    println!("  member {:<20} load-on-startup {}", m.profile_id, m.load_on_startup);
                }
                match ini {
                    Ok(r) => { println!("\n--- {} ---\n{}", router::ini_path().display(), r.text); if !r.env_conflicts.is_empty() { println!("env conflicts (first wins): {:?}", r.env_conflicts); } }
                    Err(e) => println!("\n(preset not renderable: {e})"),
                }
            }
        }
        RouterCmd::Add { profile_id, load_on_startup } => {
            Profile::load(&cfg.profile_dir.join(format!("{profile_id}.json"))).context("no such profile")?;
            rc.members.retain(|m| m.profile_id != profile_id);
            rc.members.push(RouterMember { profile_id: profile_id.clone(), load_on_startup });
            router::save_config(&rc)?;
            println!("added {profile_id} ({} members)", rc.members.len());
        }
        RouterCmd::Remove { profile_id } => {
            rc.members.retain(|m| m.profile_id != profile_id);
            router::save_config(&rc)?;
            println!("removed {profile_id} ({} members)", rc.members.len());
        }
        RouterCmd::Set { port, models_max, autoload, build } => {
            if let Some(p) = port { rc.port = p; }
            if let Some(m) = models_max { rc.models_max = m; }
            if let Some(a) = autoload { rc.autoload = a; }
            if let Some(b) = build { rc.build = if b == "newest" { None } else { Some(PathBuf::from(b)) }; }
            router::save_config(&rc)?;
            println!("saved: port {} models-max {} autoload {}", rc.port, rc.models_max, rc.autoload);
        }
        RouterCmd::Launch { ready_timeout } => {
            let profiles = Profile::load_all(&cfg.profile_dir)?;
            let builds = discovery::scan_builds(&cfg.build_roots, cfg.rocm_bin.as_deref());
            let build_dir = match &rc.build {
                Some(b) => b.clone(),
                None => llamactl_core::update::newest_installed(&builds).context("no installed build")?.path,
            };
            let platform = WindowsPlatform;
            let devices = launch::enumerate_devices(cfg, &build_dir.join("bin").join("llama-server.exe"), &platform)?;
            let r = router::launch(cfg, &rc, &profiles, &devices, &build_dir, Duration::from_secs(ready_timeout))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&r)?);
            } else {
                if !r.replaced.is_empty() { println!("stopped {} to take over port {}", r.replaced.join(", "), rc.port); }
                println!("router ready: http://{}:{}/v1  pid {}  models: {}", rc.host, rc.port, r.state.pid, r.sections.join(", "));
                println!("preset: {}", r.ini_path.display());
                if !r.env_conflicts.is_empty() { println!("env conflicts (first wins): {:?}", r.env_conflicts); }
            }
        }
        RouterCmd::Stop => {
            let runs = supervise::reattach(&cfg.runs_dir);
            match runs.into_iter().find(|r| r.state.profile_id == router::ROUTER_ID) {
                Some(r) => { supervise::stop(&r.state, &cfg.runs_dir)?; println!("stopped router (pid {})", r.state.pid); }
                None => println!("router is not running"),
            }
        }
        RouterCmd::Status => {
            let ms = router::models(&rc.host, rc.port)?;
            if json { println!("{}", serde_json::to_string_pretty(&ms)?); } else {
                for m in &ms {
                    println!("  {:<40} {}{}", m.id, m.status, if m.failed { format!(" FAILED exit {:?}", m.exit_code) } else { String::new() });
                }
            }
        }
        RouterCmd::Load { model_id } => { router::load_model(&rc.host, rc.port, &model_id)?; println!("loading {model_id}"); }
        RouterCmd::Unload { model_id } => { router::unload_model(&rc.host, rc.port, &model_id)?; println!("unloaded {model_id}"); }
    }
    Ok(())
}

// ------------------------------------------------------------------ live ----

fn cmd_live(cfg: &Config, json: bool) -> anyhow::Result<()> {
    use llamactl_core::live;
    let platform = WindowsPlatform;
    let util = platform.gpu_utilization().unwrap_or_default();
    let mut rows = Vec::new();
    for r in supervise::reattach(&cfg.runs_dir).into_iter().filter(|r| r.alive) {
        let samples: Vec<live::LiveSample> = if r.state.profile_id == llamactl_core::router::ROUTER_ID {
            llamactl_core::router::models(&r.state.host, r.state.port)
                .unwrap_or_default()
                .iter()
                .filter(|m| m.status == "loaded")
                .map(|m| live::sample(&r.state.host, r.state.port, Some(&m.id)))
                .collect()
        } else {
            vec![live::sample(&r.state.host, r.state.port, None)]
        };
        let busy: f64 = util.iter().filter(|u| u.pid == r.state.pid).map(|u| u.percent).sum();
        rows.push((r, samples, if busy <= 0.0 { 0.0 } else { busy.min(100.0) }));
    }
    if json {
        let v: Vec<serde_json::Value> = rows.iter().map(|(r, s, b)| serde_json::json!({ "run": r, "samples": s, "gpu_busy_percent": b })).collect();
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    for (r, samples, busy) in &rows {
        println!("{:<16} :{:<5} pid {:<6} gpu busy {:>5.1}%", r.state.profile_id, r.state.port, r.state.pid, busy);
        for s in samples {
            let m = &s.metrics;
            let g = |k: &str| m.get(k).copied().unwrap_or(0.0);
            let busy_slots = s.slots.iter().filter(|x| x.is_processing).count();
            println!(
                "  {:<24} {:<8} slots {}/{}  in-flight {}  queued {}  last decode {:.1} tok/s  last prompt {:.0} tok/s{}",
                s.model.as_deref().unwrap_or("(standalone)"),
                s.phase,
                busy_slots,
                s.slots.len(),
                g("requests_processing"),
                g("requests_deferred"),
                g("predicted_tokens_seconds"),
                g("prompt_tokens_seconds"),
                if g("spec_decode_num_draft_tokens_total") > 0.0 {
                    format!("  draft acceptance {:.0}%", 100.0 * g("spec_decode_num_accepted_tokens_total") / g("spec_decode_num_draft_tokens_total"))
                } else { String::new() }
            );
            for x in s.slots.iter().filter(|x| x.is_processing) {
                match x.phase.as_str() {
                    "prefill" => println!("    slot {} prefill {}/{} tokens ({:.0}%)", x.id, x.n_prompt_tokens_processed, x.n_prompt_tokens, 100.0 * x.prefill_fraction),
                    _ => println!("    slot {} decode {} generated, {} remain, ctx {:.0}%", x.id, x.n_decoded, x.n_remain, 100.0 * x.ctx_fraction),
                }
            }
            if let Some(e) = &s.error { println!("    ({e})"); }
        }
    }
    if rows.is_empty() { println!("nothing running"); }
    Ok(())
}
