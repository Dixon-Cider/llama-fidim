use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context};
use clap::{Parser, Subcommand};
use llamactl_core::config::Config;
use llamactl_core::devices::{self, Device};
use llamactl_core::discovery::{self, Build};
use llamactl_core::gguf;
use llamactl_core::platform::{Platform, WindowsPlatform};

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
    /// Enumerate GPUs fresh and show stable keys, indices, VRAM, displays.
    Devices {
        /// Build tag to enumerate with (default: newest version).
        #[arg(long)]
        build: Option<String>,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cfg = Config::load_or_init().context("loading llamactl config")?;
    match cli.command {
        Cmd::Scan => cmd_scan(&cfg, cli.json),
        Cmd::Devices { build } => cmd_devices(&cfg, build.as_deref(), cli.json),
    }
}

fn cmd_scan(cfg: &Config, json: bool) -> anyhow::Result<()> {
    let builds = discovery::scan_builds(&cfg.build_roots, cfg.rocm_bin.as_deref());
    let models = discovery::scan_models(&cfg.model_roots);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "builds": builds,
                "models": models,
            }))?
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

fn cmd_devices(cfg: &Config, build_tag: Option<&str>, json: bool) -> anyhow::Result<()> {
    let devices = enumerate_devices(cfg, build_tag)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&devices)?);
        return Ok(());
    }
    println!(
        "{:<4} {:<26} {:<44} {:>9} {:>9}  {:<9} {:<16} {}",
        "IDX", "NAME", "STABLE KEY", "TOTAL", "FREE", "CLASS", "DISPLAY", "DRIVER"
    );
    for d in &devices {
        let class = if d.integrated { "iGPU" } else { "discrete" };
        let display = d
            .display
            .as_ref()
            .map(|m| format!("{}x{}@{}", m.width, m.height, m.refresh_hz))
            .unwrap_or_else(|| "-".into());
        let assumed = if d.correlation_assumed { "~" } else { " " };
        println!(
            "{}{:<3} {:<26} {:<44} {:>7}MB {:>7}MB  {:<9} {:<16} {}",
            assumed,
            format!("{}{}", d.backend, d.hip_index),
            d.name,
            d.stable_key,
            d.total_mib,
            d.free_mib,
            class,
            display,
            d.driver_version.as_deref().unwrap_or("-"),
        );
    }
    println!("\n~ = identical-name correlation by bus order (verified at launch by residency check)");
    Ok(())
}

/// Enumerate devices with the chosen build's own `--list-devices` (the
/// canonical index space), correlated against OS adapters and hipInfo.
fn enumerate_devices(cfg: &Config, build_tag: Option<&str>) -> anyhow::Result<Vec<Device>> {
    let builds = discovery::scan_builds(&cfg.build_roots, cfg.rocm_bin.as_deref());
    let build = pick_build(&builds, build_tag)?;
    let listed_text = run_with_rocm(&build.server_exe, &["--list-devices"], cfg)?;
    let listed = devices::parse_list_devices(&listed_text)?;

    let platform = WindowsPlatform;
    let adapters = platform.video_adapters()?;

    let hipinfo = cfg
        .rocm_bin
        .as_ref()
        .map(|bin| bin.join("hipInfo.exe"))
        .filter(|p| p.is_file())
        .and_then(|exe| run_with_rocm(&exe, &[], cfg).ok())
        .map(|text| devices::parse_hipinfo(&text))
        .unwrap_or_default();

    Ok(devices::correlate(&listed, &adapters, &hipinfo, &cfg.integrated_name_patterns))
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

fn run_with_rocm(exe: &PathBuf, args: &[&str], cfg: &Config) -> anyhow::Result<String> {
    let mut cmd = Command::new(exe);
    cmd.args(args);
    if let Some(rocm) = &cfg.rocm_bin {
        let path = std::env::var_os("PATH").unwrap_or_default();
        let mut joined = rocm.as_os_str().to_owned();
        joined.push(";");
        joined.push(path);
        cmd.env("PATH", joined);
    }
    let out = cmd.output().with_context(|| format!("running {}", exe.display()))?;
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    Ok(text)
}
