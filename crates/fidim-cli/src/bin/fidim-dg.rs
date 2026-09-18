//! fidim-dg — Llama FIDIM's OpenAI-compatible front for the DiffusionGemma
//! runner. FIDIM launches it like llama-server; all logic lives in
//! `fidim_core::diffusion`. It never reads FIDIM's config or home dir: the
//! command line and the environment FIDIM composes are all it knows.

use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use fidim_core::diffusion::{self, ServeConfig};

#[derive(Parser)]
#[command(
    name = "fidim-dg",
    version = fidim_core::build_info::LONG,
    about = "Serve a DiffusionGemma GGUF over an OpenAI-compatible API (started by Llama FIDIM)",
    after_help = "Environment: NGL (GPU layers, default 0) and MAXTOK (context budget, 0 = auto) \
                  are read here for the device guard and passed on to the runner, like the rest of \
                  the environment (FA, GPU_RESOURCE_CACHE_SIZE, ...). Keys that add devices or fake \
                  memory, and a patched runner's DG_* test hooks, are stripped."
)]
struct Args {
    /// llama-diffusion-gemma-visual-server.exe
    #[arg(long)]
    runner: PathBuf,
    /// The DiffusionGemma GGUF.
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long)]
    port: u16,
    /// Model id reported to clients.
    #[arg(long)]
    alias: String,
    /// Request files are written as <prefix>-<task>-<attempt>.req.
    #[arg(long)]
    req_prefix: PathBuf,
    #[arg(long, default_value_t = 2048)]
    default_max_tokens: u32,
    /// Seed for requests that bring none (default: random per request).
    #[arg(long, allow_hyphen_values = true)]
    seed: Option<i64>,
    /// PCI bus number the runner must report (its log prints it in hex).
    #[arg(long)]
    expect_bus: Option<u32>,
    #[arg(long)]
    build_tag: Option<String>,
    #[arg(long, default_value_t = 8)]
    queue_depth: usize,
    /// Seconds to wait for the runner's READY.
    #[arg(long, default_value_t = 900)]
    load_timeout: u64,
    /// Seconds of runner silence during a request before it is restarted.
    #[arg(long, default_value_t = 180)]
    watchdog: u64,
}

/// An unset or unparsable value is 0, the runner's own default.
fn env_u32(key: &str) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .map(|n| n.clamp(0, u32::MAX as i64) as u32)
        .unwrap_or(0)
}

fn main() {
    // A panic outside a connection thread must end the process: exiting
    // closes the job object, which kills the runner. A connection thread's
    // panic unwinds and drops only that connection. Never `eprintln!` here —
    // it panics itself when the log handle is gone.
    std::panic::set_hook(Box::new(|info| {
        let _ = writeln!(std::io::stderr().lock(), "fidim-dg: panic: {info}");
        if diffusion::panic_is_fatal() {
            std::process::exit(diffusion::EXIT_PANIC);
        }
    }));
    diffusion::job::quiet_error_mode();

    let args = match Args::try_parse() {
        Ok(a) => a,
        Err(e) => {
            // clap's own exit code 2 would read as "port bind failed".
            let _ = e.print();
            let code = match e.kind() {
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion => 0,
                _ => diffusion::EXIT_SETUP,
            };
            std::process::exit(code);
        }
    };
    let cfg = ServeConfig {
        runner: args.runner,
        model: args.model,
        host: args.host,
        port: args.port,
        alias: args.alias,
        req_prefix: args.req_prefix,
        default_max_tokens: args.default_max_tokens,
        seed: args.seed,
        expect_bus: args.expect_bus,
        build_tag: args.build_tag,
        queue_depth: args.queue_depth,
        load_timeout: Duration::from_secs(args.load_timeout),
        watchdog: Duration::from_secs(args.watchdog.max(1)),
        ngl: env_u32("NGL"),
        maxtok_env: env_u32("MAXTOK"),
    };
    std::process::exit(diffusion::serve(cfg));
}
