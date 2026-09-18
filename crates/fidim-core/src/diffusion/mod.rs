//! `fidim-dg`: Llama FIDIM's OpenAI shim for Unsloth's DiffusionGemma
//! runner (`llama-diffusion-gemma-visual-server.exe`).
//!
//! FIDIM starts the helper exactly like llama-server (detached, stdout and
//! stderr to the run log). The helper binds the profile port first, owns the
//! runner through piped stdio inside a kill-on-close Job Object, forwards the
//! runner's stderr into the log, and serves a trimmed OpenAI surface over a
//! std::net HTTP/1.1 server: /health, /v1/models, /v1/chat/completions,
//! /slots and /metrics. `/v1/models` answers 503 until the runner is READY
//! and the device guard has passed, so FIDIM's readiness poll, in-flight
//! reservations, timeouts and port takeover all work unchanged.
//!
//! - `protocol`: the runner's stdout records and stderr facts (pure).
//! - `openai`: request planning and response shaping (pure).
//! - `engine`: the worker thread that owns the runner.
//! - `http`: the HTTP layer and the endpoints.
//! - `job`: the Job Object and error-mode plumbing.

pub mod engine;
pub mod http;
pub mod job;
pub mod openai;
pub mod protocol;

use std::fmt;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::gguf::GgufHeader;

/// Helper exit codes (shim spec §0). FIDIM reports any non-zero exit as
/// "exited during load — see log" or CRASHED; the log line says which.
pub const EXIT_BIND: i32 = 2;
/// The runner exited, hit EOF or timed out before its first READY.
pub const EXIT_LOAD: i32 = 3;
pub const EXIT_GUARD: i32 = 4;
/// The runner is on a different PCI bus than `--expect-bus`.
pub const EXIT_BUS: i32 = 5;
/// Three runner deaths with no successful job in between.
pub const EXIT_BREAKER: i32 = 6;
/// Bad arguments, unreadable GGUF, unusable request path, job object failure.
pub const EXIT_SETUP: i32 = 7;
pub const EXIT_PANIC: i32 = 8;

#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub runner: PathBuf,
    pub model: PathBuf,
    pub host: String,
    pub port: u16,
    /// Model id clients see.
    pub alias: String,
    /// Request files are `<req_prefix>-<id_task>-<attempt>.req`.
    pub req_prefix: PathBuf,
    pub default_max_tokens: u32,
    /// Profile seed: pins every request that does not bring its own.
    pub seed: Option<i64>,
    pub expect_bus: Option<u32>,
    pub build_tag: Option<String>,
    pub queue_depth: usize,
    pub load_timeout: Duration,
    pub watchdog: Duration,
    /// NGL from the helper's environment (FIDIM always sets it; default 0).
    pub ngl: u32,
    /// MAXTOK from the helper's environment (0 = the runner auto-sizes).
    pub maxtok_env: u32,
}

/// What the helper needs from the GGUF header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInfo {
    pub canvas: u32,
    pub block_count: u64,
    pub n_ctx_train: u64,
    pub vocab: u64,
}

impl ModelInfo {
    pub fn from_header(h: &GgufHeader) -> Self {
        ModelInfo {
            // The runner's own fallback is to refuse the model; 256 is what
            // every published DiffusionGemma file carries.
            canvas: h.diffusion_canvas_length.unwrap_or(256).clamp(1, u32::MAX as u64) as u32,
            block_count: h.block_count.unwrap_or(0),
            n_ctx_train: h.context_length.unwrap_or(0),
            vocab: h.vocab_size.unwrap_or(0),
        }
    }
}

/// Text for FIDIM's live view (`/slots` `prompt` and `generated`): the
/// request's conversation, the answer committed so far and the current
/// block's latest draft. Kept after the job, like llama-server's last
/// generation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LiveText {
    pub prompt: String,
    /// The cumulative answer of the last accepted `C` line.
    pub committed: String,
    /// The block being denoised, as of its latest step; empty between blocks.
    pub draft: String,
    /// Every step of the job, for the GUI's replay (`/frames`); the oldest
    /// are dropped past `FRAMES_MAX_BYTES` of text.
    pub frames: std::collections::VecDeque<FrameRec>,
    pub frames_bytes: usize,
    pub frames_dropped: u64,
}

/// One denoise step as the replay shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameRec {
    pub block: u32,
    pub step: u32,
    pub total: u32,
    pub text: String,
}

/// Replay history cap: a long reply (hundreds of blocks) keeps its latest
/// steps only.
pub const FRAMES_MAX_BYTES: usize = 8 << 20;

impl LiveText {
    /// Record one step: the draft for /slots and a replay frame.
    pub fn push_frame(&mut self, block: u32, step: u32, total: u32, text: String) {
        self.frames_bytes += text.len();
        self.frames.push_back(FrameRec { block, step, total, text: text.clone() });
        while self.frames_bytes > FRAMES_MAX_BYTES {
            match self.frames.pop_front() {
                Some(f) => {
                    self.frames_bytes -= f.text.len();
                    self.frames_dropped += 1;
                }
                None => break,
            }
        }
        self.draft = text;
    }
}

/// Where the current (or last) job is, for /slots and the stream comments.
#[derive(Debug, Clone, PartialEq)]
pub struct Progress {
    pub id_task: u64,
    /// `loading`, `idle`, `prefill` or `denoise`.
    pub state: &'static str,
    pub block: u32,
    pub step: u32,
    pub total: u32,
    pub n_blocks: u32,
    /// Prompt tokens of the last finished job (the helper cannot tokenize).
    pub n_prompt: u64,
    /// Denoise steps so far in this attempt, over all blocks (each one
    /// re-predicts the whole canvas: Studio's "canvas tok/s").
    pub steps_done: u32,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Metrics {
    pub prompt_tokens_total: u64,
    pub tokens_predicted_total: u64,
    pub predicted_seconds_total: f64,
    /// Denoise steps, the closest thing diffusion has to a decode call.
    pub n_decode_total: u64,
    pub last_predicted_tps: f64,
    pub blocks_total: u64,
    /// Canvas tokens predicted, over every step: canvas x steps.
    pub canvas_tokens_total: u64,
    /// The last job's canvas x steps / wall: what Unsloth Studio shows as
    /// its headline "Speed".
    pub last_canvas_tps: f64,
}

/// State shared by the engine worker and the HTTP threads. The read-only
/// endpoints only ever read this; they never wait on the engine.
pub struct Shared {
    pub(crate) ready: AtomicBool,
    pub(crate) ever_ready: AtomicBool,
    /// The engine worker has returned; nothing will serve a job again.
    pub(crate) exited: AtomicBool,
    /// Stop accepting connections.
    pub(crate) shutdown: AtomicBool,
    pub(crate) maxtok: AtomicU32,
    /// From READY: the runner's own vocabulary size.
    pub(crate) n_vocab: AtomicU32,
    pub(crate) restarts: AtomicU32,
    /// Jobs sent to the engine and not yet started.
    pub(crate) queued: AtomicUsize,
    pub(crate) processing: AtomicBool,
    pub(crate) child_pid: AtomicU32,
    pub(crate) connections: AtomicUsize,
    /// Enqueue order: taken under the lock together with the send, so the
    /// ticket order is the channel order.
    pub(crate) tickets: Mutex<u64>,
    /// Jobs the engine has taken off its queue (run or skipped).
    pub(crate) served: AtomicU64,
    pub(crate) progress: Mutex<Progress>,
    pub(crate) live: Mutex<LiveText>,
    pub(crate) metrics: Mutex<Metrics>,
    pub(crate) rng: Mutex<XorShift64>,
}

impl Shared {
    pub(crate) fn new() -> Self {
        Shared {
            ready: AtomicBool::new(false),
            ever_ready: AtomicBool::new(false),
            exited: AtomicBool::new(false),
            shutdown: AtomicBool::new(false),
            maxtok: AtomicU32::new(0),
            n_vocab: AtomicU32::new(0),
            restarts: AtomicU32::new(0),
            queued: AtomicUsize::new(0),
            processing: AtomicBool::new(false),
            child_pid: AtomicU32::new(0),
            connections: AtomicUsize::new(0),
            tickets: Mutex::new(0),
            served: AtomicU64::new(0),
            progress: Mutex::new(Progress {
                id_task: 0,
                state: "loading",
                block: 0,
                step: 0,
                total: 0,
                n_blocks: 0,
                n_prompt: 0,
                steps_done: 0,
            }),
            live: Mutex::new(LiveText::default()),
            metrics: Mutex::new(Metrics::default()),
            rng: Mutex::new(XorShift64::seeded()),
        }
    }

    pub(crate) fn progress(&self) -> MutexGuard<'_, Progress> {
        lock(&self.progress)
    }
    pub(crate) fn live(&self) -> MutexGuard<'_, LiveText> {
        lock(&self.live)
    }
    pub(crate) fn metrics(&self) -> MutexGuard<'_, Metrics> {
        lock(&self.metrics)
    }
}

/// A poisoned lock only means another thread panicked: outside a connection
/// thread that already exits fidim-dg, and a connection thread only reads
/// or bumps counters here, so the data is still good.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether a panic on the current thread must end the helper (fidim-dg's
/// panic hook asks). A connection thread unwinds to its `catch_unwind` and
/// loses only its own connection; anywhere else (the engine, the accept
/// loop, the pipe readers) the helper can no longer be trusted, and exiting
/// closes the job object, which kills the runner.
pub fn panic_is_fatal() -> bool {
    std::thread::current().name() != Some(http::CONN_THREAD)
}

/// xorshift64 for response ids and request seeds: no `rand` dependency,
/// and nothing here needs cryptographic randomness.
#[derive(Debug, Clone)]
pub struct XorShift64(u64);

impl XorShift64 {
    pub fn new(seed: u64) -> Self {
        XorShift64(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    /// Seeded from the clock and the pid, so two helpers started in the same
    /// instant still differ.
    pub fn seeded() -> Self {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        Self::new(nanos ^ ((std::process::id() as u64) << 32) ^ 0xD1B5_4A32_D192_ED03)
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

pub(crate) fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Write one line to stderr, which is the run log. Never `eprintln!`: it
/// panics when the write fails (a full disk, a closed log), and a panic
/// ends the helper and its runner. Tests go through `eprintln!` so the
/// harness captures the output.
pub(crate) fn stderr_line(line: &str) {
    #[cfg(test)]
    eprintln!("{line}");
    #[cfg(not(test))]
    {
        use std::io::Write;
        let _ = writeln!(std::io::stderr().lock(), "{line}");
    }
}

/// A helper line in the run log, prefixed `fidim-dg: `.
pub(crate) fn log(args: fmt::Arguments) {
    stderr_line(&format!("fidim-dg: {args}"));
}

macro_rules! dglog {
    ($($t:tt)*) => { $crate::diffusion::log(format_args!($($t)*)) };
}
pub(crate) use dglog;

// ------------------------------------------------------------ request files ----

/// The helper's fallback request prefix, `%TEMP%\fidim-dg-<port>`, when the
/// runs-dir prefix is unusable for the runner. Pre-flight check 1 judges the
/// same path, and `stop` cleans it, so all three agree on where it is.
pub fn req_fallback_prefix(port: u16) -> PathBuf {
    std::env::temp_dir().join(format!("fidim-dg-{port}"))
}

/// Check the request prefix and make sure its directory exists; fall back to
/// `req_fallback_prefix` (never ProgramData: request files hold the
/// conversation, and the runs dir and TEMP are both user-only).
pub fn resolve_req_prefix(p: &Path, port: u16) -> Result<PathBuf, String> {
    fn usable(p: &Path) -> Result<(), String> {
        protocol::check_req_prefix(p)?;
        let dir = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))
    }
    match usable(p) {
        Ok(()) => Ok(p.to_path_buf()),
        Err(first) => {
            let fallback = req_fallback_prefix(port);
            dglog!("request path {} is unusable ({first}); using {}", p.display(), fallback.display());
            usable(&fallback)
                .map(|()| fallback.clone())
                .map_err(|second| format!("{first}; fallback {}: {second}", fallback.display()))
        }
    }
}

/// `<prefix>-<id_task>-<attempt>.req`.
pub(crate) fn req_file(prefix: &Path, id_task: u64, attempt: u32) -> PathBuf {
    let name = prefix.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    prefix.with_file_name(format!("{name}-{id_task}-{attempt}.req"))
}

/// Delete every `<prefix>-<n>-<n>.req`. Matching the two numbers keeps a
/// profile `a` on port 1 from deleting the files of a profile `a-1`.
pub fn remove_request_files(prefix: &Path) {
    let Some(name) = prefix.file_name().map(|n| n.to_string_lossy().into_owned()) else { return };
    let dir = prefix.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let head = format!("{name}-");
    for e in entries.flatten() {
        let file = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = file.strip_prefix(&head).and_then(|r| r.strip_suffix(".req")) else { continue };
        let mut parts = rest.split('-');
        let numeric = |s: Option<&str>| s.is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()));
        if numeric(parts.next()) && numeric(parts.next()) && parts.next().is_none() {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

// ------------------------------------------------------------- entry points ----

/// Run the helper to completion; the return value is the process exit code.
pub fn serve(mut cfg: ServeConfig) -> i32 {
    // First, so every run's log says which helper it was, even one that
    // fails to start.
    dglog!("fidim-dg {}", crate::build_info::LONG);
    // Bind before anything else: FIDIM's stop/reattach identify the helper by
    // the port it listens on, and a second FIDIM launch must see it taken.
    let listener = match TcpListener::bind((cfg.host.as_str(), cfg.port)) {
        Ok(l) => l,
        Err(e) => {
            dglog!("cannot listen on {}:{}: {e}", cfg.host, cfg.port);
            return EXIT_BIND;
        }
    };
    if !cfg.runner.is_file() {
        dglog!("runner {} does not exist", cfg.runner.display());
        return EXIT_SETUP;
    }
    let info = match crate::gguf::read_header(&cfg.model) {
        Ok(h) => ModelInfo::from_header(&h),
        Err(e) => {
            dglog!("cannot read the model header: {e}");
            return EXIT_SETUP;
        }
    };
    match resolve_req_prefix(&cfg.req_prefix, cfg.port) {
        Ok(p) => cfg.req_prefix = p,
        Err(e) => {
            dglog!("no usable request path: {e}");
            return EXIT_SETUP;
        }
    }
    remove_request_files(&cfg.req_prefix);
    let job = match job::KillOnCloseJob::new() {
        Ok(j) => j,
        Err(e) => {
            dglog!("cannot create the kill-on-close job object: {e}");
            return EXIT_SETUP;
        }
    };
    // FA and the HIP runtime cache decide how the runner sizes its context
    // and what it holds after a request: log what the run actually got.
    let env_or = |k: &str| std::env::var(k).unwrap_or_else(|_| "unset".into());
    dglog!(
        "serving '{}' on {}:{}; runner {}; NGL={} MAXTOK={} FA={} GPU_RESOURCE_CACHE_SIZE={} expect-bus={} canvas={} \
         layers={}",
        cfg.alias,
        cfg.host,
        cfg.port,
        cfg.runner.display(),
        cfg.ngl,
        cfg.maxtok_env,
        env_or("FA"),
        env_or("GPU_RESOURCE_CACHE_SIZE"),
        cfg.expect_bus.map(|b| format!("{b:02x}")).unwrap_or_else(|| "any".into()),
        info.canvas,
        info.block_count,
    );
    let spawner = engine::RealSpawner { runner: cfg.runner.clone(), model: cfg.model.clone(), job };
    match start(listener, cfg, info, Box::new(spawner)) {
        Ok(h) => h.join(),
        Err(e) => {
            dglog!("cannot start: {e}");
            EXIT_SETUP
        }
    }
}

/// Start the engine worker and the accept loop on an already-bound
/// listener. Tests pass a listener on 127.0.0.1:0 and a fake spawner.
pub fn start(
    listener: TcpListener,
    cfg: ServeConfig,
    info: ModelInfo,
    spawner: Box<dyn engine::Spawner>,
) -> std::io::Result<ServerHandle> {
    let addr = listener.local_addr()?;
    let shared = Arc::new(Shared::new());
    let cfg = Arc::new(cfg);
    let (tx, rx) = mpsc::channel();

    let engine = {
        let (cfg, shared, tx) = (cfg.clone(), shared.clone(), tx.clone());
        std::thread::Builder::new()
            .name("dg-engine".into())
            .spawn(move || engine::run_engine(&cfg, &info, shared, spawner, rx, tx))?
    };
    let ctx = Arc::new(http::Ctx { shared: shared.clone(), tx: tx.clone(), cfg, info, created: unix_now() });
    let accept = std::thread::Builder::new().name("dg-accept".into()).spawn(move || http::accept_loop(listener, ctx))?;
    Ok(ServerHandle { addr, shared, tx, engine: Some(engine), accept: Some(accept) })
}

pub struct ServerHandle {
    pub addr: SocketAddr,
    shared: Arc<Shared>,
    tx: Sender<engine::EngineMsg>,
    engine: Option<JoinHandle<i32>>,
    accept: Option<JoinHandle<()>>,
}

impl ServerHandle {
    /// Graceful stop: QUIT the runner, stop accepting.
    pub fn shutdown(&self) {
        let _ = self.tx.send(engine::EngineMsg::Shutdown);
        self.stop_accepting();
    }

    fn stop_accepting(&self) {
        self.shared.shutdown.store(true, Ordering::SeqCst);
        // `accept` blocks; a throwaway connection wakes it to see the flag.
        let mut wake = self.addr;
        if wake.ip().is_unspecified() {
            wake.set_ip(match wake {
                SocketAddr::V4(_) => std::net::Ipv4Addr::LOCALHOST.into(),
                SocketAddr::V6(_) => std::net::Ipv6Addr::LOCALHOST.into(),
            });
        }
        let _ = TcpStream::connect_timeout(&wake, Duration::from_secs(1));
    }

    /// Wait for the engine worker; its code is the helper's exit code. The
    /// listener is closed afterwards so FIDIM sees the port free.
    pub fn join(mut self) -> i32 {
        let code = self.engine.take().map(|h| h.join().unwrap_or(EXIT_PANIC)).unwrap_or(0);
        self.stop_accepting();
        if let Some(a) = self.accept.take() {
            let _ = a.join();
        }
        code
    }
}

#[cfg(test)]
mod serve_e2e;
