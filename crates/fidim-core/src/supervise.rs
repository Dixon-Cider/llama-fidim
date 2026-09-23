//! Server supervision (R-07, §07 "servers outlive the UI"): detached spawn
//! with log capture, readiness by polling `/v1/models`, tri-state health,
//! stop, and re-attach after the tool restarts.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::launch::LaunchPlan;
use crate::profile::{Engine, Profile};
use crate::{Error, Result};

/// The diffusion engine's server: FIDIM's shim, which owns the runner.
pub const DG_HELPER_EXE: &str = "fidim-dg.exe";

/// State file written next to the logs for every launched server — the
/// re-attach path reads these back after the tool restarts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunState {
    pub profile_id: String,
    pub pid: u32,
    pub port: u16,
    pub host: String,
    pub alias: String,
    pub started_unix: u64,
    pub log_path: PathBuf,
    pub command_line: String,
    pub visibility_env: String,
    /// Stable keys of the devices this run occupies, in profile order.
    pub device_keys: Vec<String>,
    /// Free VRAM (MiB) per device just before launch — residency
    /// verification compares against a fresh enumeration after readiness.
    pub free_mib_before: Vec<u64>,
    /// True when the model file had never been loaded since its last
    /// modification (R-08): the first benchmark against this run is a
    /// cold-cache measurement and must not become a saved baseline.
    #[serde(default)]
    pub cold_start: bool,
    /// Total bytes this run is expected to occupy (VRAM estimate). Concurrent
    /// launches read this to account for memory a still-loading server has
    /// reserved but not yet charged.
    #[serde(default)]
    pub estimated_bytes: u64,
    /// Set once `/v1/models` answers. Until then the run is "in flight" and
    /// its full `estimated_bytes` counts against a subsequent launch.
    #[serde(default)]
    pub ready: bool,
    /// PID of the keep-alive helper (`fidim keepalive`) started for this
    /// run, if any. Killed by `stop`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub keepalive_pid: Option<u32>,
    /// What `pid` is: llama-server, or the `fidim-dg.exe` helper that owns a
    /// DiffusionGemma runner. Omitted for llama-server, so state files from
    /// before this field (and router states) read as llama-server.
    #[serde(default, skip_serializing_if = "Engine::is_llama_server")]
    pub engine: Engine,
}

/// Bytes reserved by runs that are alive but not yet finished loading.
/// A loading server has claimed its memory without having charged it, so a
/// concurrent pre-flight that ignores it sees phantom headroom.
pub fn inflight_reserved_bytes(runs_dir: &Path) -> u64 {
    inflight_of(&reattach(runs_dir))
}

/// `inflight_reserved_bytes` over runs already re-attached.
pub fn inflight_of(runs: &[AttachedRun]) -> u64 {
    runs.iter().filter(|r| r.alive && !r.state.ready).map(|r| r.state.estimated_bytes).sum()
}

impl RunState {
    pub fn state_path(runs_dir: &Path, profile_id: &str, port: u16) -> PathBuf {
        runs_dir.join(format!("{profile_id}-{port}.json"))
    }

    pub fn save(&self, runs_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(runs_dir).map_err(|e| Error::io(runs_dir, e))?;
        let p = Self::state_path(runs_dir, &self.profile_id, self.port);
        std::fs::write(&p, serde_json::to_string_pretty(self)?).map_err(|e| Error::io(&p, e))
    }
}

// ------------------------------------------------- model load history ----

/// `<config-dir>/model-history.json`: model path → unix time of the last
/// successful load. Drives the cold-cache heuristic (R-08): a model counts
/// cold until one recorded load postdates its mtime.
fn model_history_path(config_dir: &Path) -> PathBuf {
    config_dir.join("model-history.json")
}

fn load_model_history(config_dir: &Path) -> std::collections::BTreeMap<String, u64> {
    std::fs::read_to_string(model_history_path(config_dir))
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default()
}

/// Whether a launch of `model_path` would be a cold-cache first load.
pub fn is_cold_start(config_dir: &Path, model_path: &Path) -> bool {
    let mtime = std::fs::metadata(model_path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let history = load_model_history(config_dir);
    match history.get(&model_path.to_string_lossy().to_lowercase()) {
        Some(last_loaded) => *last_loaded < mtime,
        None => true,
    }
}

/// Record a successful load (call after readiness, not after spawn — a
/// crashed load warms nothing).
pub fn record_model_loaded(config_dir: &Path, model_path: &Path) -> Result<()> {
    let mut history = load_model_history(config_dir);
    let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    history.insert(model_path.to_string_lossy().to_lowercase(), now);
    std::fs::create_dir_all(config_dir).map_err(|e| Error::io(config_dir, e))?;
    let p = model_history_path(config_dir);
    std::fs::write(&p, serde_json::to_string_pretty(&history)?).map_err(|e| Error::io(&p, e))
}

/// Spawn the plan detached: new process group, no console window, stdout and
/// stderr appended to a log file. Closing Llama FIDIM later must not kill it.
pub fn spawn(
    plan: &LaunchPlan,
    profile: &Profile,
    runs_dir: &Path,
    device_keys: Vec<String>,
    free_mib_before: Vec<u64>,
    cold_start: bool,
    estimated_bytes: u64,
) -> Result<RunState> {
    std::fs::create_dir_all(runs_dir).map_err(|e| Error::io(runs_dir, e))?;
    let started_unix =
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let log_path = runs_dir.join(format!("{}-{}-{started_unix}.log", profile.id, profile.server.port));
    let log = std::fs::File::create(&log_path).map_err(|e| Error::io(&log_path, e))?;
    let log_err = log.try_clone().map_err(|e| Error::io(&log_path, e))?;

    let mut cmd = std::process::Command::new(&plan.exe);
    cmd.args(&plan.args);
    for (k, v) in &plan.env {
        cmd.env(k, v);
    }
    if let Some(rocm) = &plan.path_prepend {
        let path = std::env::var_os("PATH").unwrap_or_default();
        let mut joined = rocm.as_os_str().to_owned();
        joined.push(if cfg!(windows) { ";" } else { ":" });
        joined.push(path);
        cmd.env("PATH", joined);
    }
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(log_err));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
        stop_inheriting_std_handles();
    }
    let child = cmd
        .spawn()
        .map_err(|e| Error::BuildBinary { path: plan.exe.clone(), detail: e.to_string() })?;

    let state = RunState {
        profile_id: profile.id.clone(),
        pid: child.id(),
        port: profile.server.port,
        host: profile.server.host.clone(),
        alias: profile.server.alias.clone(),
        started_unix,
        log_path,
        command_line: plan.command_line(),
        visibility_env: plan.visibility_env.clone(),
        device_keys,
        free_mib_before,
        cold_start,
        estimated_bytes,
        ready: false,
        keepalive_pid: None,
        engine: profile.engine,
    };
    state.save(runs_dir)?;
    Ok(state)
}

// ------------------------------------------------------------------ HTTP ----

/// Minimal HTTP/1.1 GET — avoids an async client dependency for what is a
/// localhost status poll. Returns (status code, body).
pub fn http_get(host: &str, port: u16, path: &str, timeout: Duration) -> Result<(u16, String)> {
    http_get_auth(host, port, path, timeout, None)
}

/// `http_get` with a bearer token: a llama-server started with `--api-key`
/// answers 401 on everything but `/health` without one.
pub fn http_get_auth(host: &str, port: u16, path: &str, timeout: Duration, bearer: Option<&str>) -> Result<(u16, String)> {
    let auth = match bearer {
        // A control character would end the header line.
        Some(k) if k.chars().any(char::is_control) => {
            return Err(Error::Config("the API key contains a control character".into()));
        }
        Some(k) => format!("Authorization: Bearer {k}\r\n"),
        None => String::new(),
    };
    // A server bound to 0.0.0.0 is reached over loopback; IPv6 in brackets.
    let addr = crate::chat::host_port(host, port);
    let stream = TcpStream::connect_timeout(
        &addr
            .parse()
            .map_err(|e| Error::Platform(format!("bad address {addr}: {e}")))?,
        timeout,
    )
    .map_err(|e| Error::Platform(format!("connect {addr}: {e}")))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    let mut stream = stream;
    let req = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\n{auth}Connection: close\r\n\r\n");
    stream
        .write_all(req.as_bytes())
        .map_err(|e| Error::Platform(format!("send {addr}: {e}")))?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).ok(); // best-effort: server may cut the stream
    parse_response(&buf, &addr)
}

/// POST with a JSON body — used by the generation probe and the benchmark
/// runner. Returns (status, body).
pub fn http_post_json(
    host: &str,
    port: u16,
    path: &str,
    json: &str,
    timeout: Duration,
) -> Result<(u16, String)> {
    let addr = crate::chat::host_port(host, port);
    let stream = TcpStream::connect_timeout(
        &addr
            .parse()
            .map_err(|e| Error::Platform(format!("bad address {addr}: {e}")))?,
        timeout,
    )
    .map_err(|e| Error::Platform(format!("connect {addr}: {e}")))?;
    stream.set_read_timeout(Some(timeout)).ok();
    stream.set_write_timeout(Some(timeout)).ok();
    let mut stream = stream;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{json}",
        json.len()
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| Error::Platform(format!("send {addr}: {e}")))?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).ok();
    parse_response(&buf, &addr)
}

/// Split a raw HTTP/1.1 response into (status, body). Decodes
/// `Transfer-Encoding: chunked` — llama-server (and the router proxy)
/// chunk anything beyond a few KB, and a `/slots` body for 8 slots crosses
/// a chunk boundary, which left a hex chunk-size line in the middle of the
/// JSON ("expected ident at column 3940" on the Running tab, 2026-09-05).
fn parse_response(buf: &[u8], addr: &str) -> Result<(u16, String)> {
    let split = buf.windows(4).position(|w| w == b"\r\n\r\n");
    let (head, body) = match split {
        Some(i) => (&buf[..i], &buf[i + 4..]),
        None => (buf, &buf[buf.len()..]),
    };
    let head = String::from_utf8_lossy(head);
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::Platform(format!("no HTTP status from {addr}")))?;
    let chunked = head.lines().any(|l| {
        let l = l.to_ascii_lowercase();
        l.trim_start().starts_with("transfer-encoding:") && l.contains("chunked")
    });
    let body = if chunked { dechunk(body) } else { body.to_vec() };
    Ok((status, String::from_utf8_lossy(&body).into_owned()))
}

/// Concatenate the data of a chunked body. Tolerant: a truncated or
/// malformed stream yields whatever was decoded so far.
fn dechunk(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.len());
    loop {
        let Some(nl) = b.windows(2).position(|w| w == b"\r\n") else { break };
        let line = String::from_utf8_lossy(&b[..nl]);
        let size_hex = line.split(';').next().unwrap_or("").trim();
        let Ok(size) = usize::from_str_radix(size_hex, 16) else { break };
        b = &b[nl + 2..];
        if size == 0 {
            break;
        }
        if b.len() < size {
            out.extend_from_slice(b);
            break;
        }
        out.extend_from_slice(&b[..size]);
        b = &b[size..];
        if b.starts_with(b"\r\n") {
            b = &b[2..];
        }
    }
    out
}

/// Clear the inherit flag on this process's own stdin/stdout/stderr before
/// spawning the detached server. Rust's `Command` passes
/// `bInheritHandles = TRUE` whenever stdio is configured, and every
/// inheritable handle in the parent comes along — including the pipe a
/// shell created to capture OUR output. The server then holds that pipe's
/// write end for its whole lifetime and the shell's reader never sees EOF:
/// `fidim launch … | Out-String` (2026-09-02, 2.5 h hang) and
/// `fidim launch … | tail` both deadlocked this way. The server's own
/// stdio is set explicitly to the log file, so it loses nothing.
#[cfg(windows)]
pub(crate) fn stop_inheriting_std_handles() {
    use windows::Win32::Foundation::{SetHandleInformation, HANDLE_FLAGS, HANDLE_FLAG_INHERIT};
    use windows::Win32::System::Console::{
        GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
    };
    for which in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
        // Best effort: a missing console handle (GUI process) is not an error.
        if let Ok(h) = unsafe { GetStdHandle(which) } {
            if !h.is_invalid() {
                let _ = unsafe { SetHandleInformation(h, HANDLE_FLAG_INHERIT.0, HANDLE_FLAGS(0)) };
            }
        }
    }
}

/// Handle inheritance is a Windows-only hazard; elsewhere there is nothing to clear.
#[cfg(not(windows))]
pub(crate) fn stop_inheriting_std_handles() {}

/// Wait for `/v1/models` to answer 200 (R-07: readiness by HTTP, never by
/// log-scraping). Large models take minutes to load.
pub fn wait_ready(state: &RunState, deadline: Duration) -> Result<()> {
    wait_ready_in(state, deadline, None)
}

/// `wait_ready`, additionally clearing the run's in-flight reservation once
/// it answers (pass the runs dir so the state file is rewritten).
pub fn wait_ready_in(
    state: &RunState,
    deadline: Duration,
    runs_dir: Option<&Path>,
) -> Result<()> {
    let start = Instant::now();
    loop {
        if !process_alive(state.pid) {
            return Err(Error::Platform(format!(
                "server process {} exited during load — see log {}",
                state.pid,
                state.log_path.display()
            )));
        }
        // SGLang answers /v1/models before its warm-up and CUDA-graph capture
        // finish; /health turns 200 only at "fired up and ready to roll".
        let ready_path = if state.engine.is_sglang() { "/health" } else { "/v1/models" };
        if let Ok((200, _)) = http_get(&state.host, state.port, ready_path, Duration::from_secs(2))
        {
            if let Some(dir) = runs_dir {
                let mut ready_state = state.clone();
                ready_state.ready = true;
                ready_state.save(dir)?;
            }
            return Ok(());
        }
        if start.elapsed() > deadline {
            return Err(Error::Platform(format!(
                "server did not become ready within {deadline:?} — see log {}",
                state.log_path.display()
            )));
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Health has three states, not two (§05 V-3).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Health {
    Healthy,
    /// Answers `/v1/models` but a 1-token completion fails — a real observed
    /// state, reported distinctly (R-07).
    RespondingNotGenerating,
    Dead,
}

/// Probe health. `deep` adds the 1-token generation probe; without it a
/// responding server is reported healthy.
pub fn probe_health(state: &RunState, deep: bool) -> Health {
    let responding = matches!(
        http_get(&state.host, state.port, "/v1/models", Duration::from_secs(3)),
        Ok((200, _))
    );
    if !responding {
        return Health::Dead;
    }
    if deep {
        let ok = if state.engine.is_diffusion() {
            // Never generate: the smallest diffusion request is a whole
            // 256-token denoise block. The helper's /health answers 200 only
            // while a guarded runner is up, and 503 while it respawns.
            matches!(http_get(&state.host, state.port, "/health", Duration::from_secs(5)), Ok((200, _)))
        } else {
            let body = format!(
                r#"{{"model":"{}","prompt":"Hi","max_tokens":1,"stream":false}}"#,
                state.alias
            );
            matches!(
                http_post_json(&state.host, state.port, "/v1/completions", &body, Duration::from_secs(60)),
                Ok((200, _))
            )
        };
        if !ok {
            return Health::RespondingNotGenerating;
        }
    }
    Health::Healthy
}

pub fn process_alive(pid: u32) -> bool {
    process_image(pid).is_some()
}

/// Linux: the executable name from `/proc/<pid>/comm` (what `tasklist`
/// reports on Windows, minus the `.exe`). For an interpreter, argv[1]'s file
/// name (`model_router.py`) is the image that means something.
#[cfg(not(windows))]
pub fn process_image(pid: u32) -> Option<String> {
    let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).ok()?;
    let comm = comm.trim();
    if comm.is_empty() {
        return None;
    }
    if comm.starts_with("python") {
        if let Ok(cmd) = std::fs::read(format!("/proc/{pid}/cmdline")) {
            let args: Vec<&[u8]> = cmd.split(|b| *b == 0).filter(|a| !a.is_empty()).collect();
            // `python -m pkg.module` -> "pkg.module"; `python script.py` -> "script.py"
            let arg = match args.get(1) {
                Some(a) if *a == b"-m" => args.get(2),
                a => a,
            };
            if let Some(script) = arg {
                let script = String::from_utf8_lossy(script);
                if let Some(name) = std::path::Path::new(script.as_ref()).file_name() {
                    return Some(name.to_string_lossy().to_string());
                }
            }
        }
    }
    Some(comm.to_string())
}

#[cfg(windows)]

/// Image name of a live process (`llama-server.exe`), or None when no
/// process has that pid. tasklist CSV: `"image","pid","session",...`.
pub fn process_image(pid: u32) -> Option<String> {
    let mut cmd = std::process::Command::new("tasklist");
    cmd.args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"]);
    crate::launch::hide_console(&mut cmd);
    let out = cmd.output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().find(|l| l.contains(&format!("\"{pid}\"")))?;
    Some(line.trim_start_matches('"').split('"').next().unwrap_or("").to_string())
}

/// True only when `pid` is alive AND is the kind of process we started.
/// A pid from a run state that is weeks old can belong to anything by now
/// (Windows reuses pids), so "alive" for a run means "still our server",
/// never merely "some process has this number".
pub fn process_alive_as(pid: u32, image_contains: &str) -> bool {
    process_image(pid).map(|img| img.to_ascii_lowercase().contains(image_contains)).unwrap_or(false)
}

/// Whether a run's pid is still the server it started, per engine:
/// - llama-server: the image name, as before.
/// - diffusion: the exact image `fidim-dg.exe` (a substring would also match
///   `llama-fidim.exe` and the keep-alive `fidim.exe`) AND that pid is the one
///   listening on the run's port. The helper binds before anything else, and
///   a reused pid on another helper listens elsewhere.
/// - unknown: never ours to kill.
pub fn run_alive(s: &RunState) -> bool {
    match s.engine {
        Engine::LlamaServer => process_alive_as(s.pid, "llama-server"),
        Engine::DiffusionGemma => {
            process_image(s.pid).is_some_and(|img| img.eq_ignore_ascii_case(DG_HELPER_EXE))
                && crate::launch::listens_on(s.port, s.pid)
        }
        // The router (or a bare server) is a python process; what matters is
        // that the recorded pid still owns the port.
        Engine::SgLang => process_alive(s.pid) && crate::launch::listens_on(s.port, s.pid),
        Engine::Unknown => false,
    }
}

/// The keep-alive helper's image: `fidim.exe`, or `llamactl.exe` from before
/// the rename. Exact, because `llama-fidim.exe` (the GUI) contains "fidim".
pub(crate) fn is_keepalive_image(img: &str) -> bool {
    img.eq_ignore_ascii_case("fidim.exe")
        || img.eq_ignore_ascii_case("llamactl.exe")
        || img.eq_ignore_ascii_case("fidim")
        || img.eq_ignore_ascii_case("llamactl")
}

/// Stop a server. llama-server has no shutdown endpoint; on Windows a
/// process-tree terminate is the clean stop (§04 clean-stop requirement).
/// For a diffusion run the tree is the helper and its runner; the runner's
/// kill-on-close job is the second line of defence.
///
/// A run whose process is gone (or whose pid now belongs to something
/// else) is only forgotten: the state file goes, nothing is killed.
#[cfg(not(windows))]
pub fn stop(state: &RunState, runs_dir: &Path) -> Result<()> {
    // Children first (a python launcher's scheduler/detokenizer hold the GPU),
    // then the root; SIGKILL, as the Windows path does with /F.
    fn kill_tree(pid: u32) {
        for c in crate::platform::process_descendants(pid) {
            unsafe { libc_kill(c) };
        }
        unsafe { libc_kill(pid) };
    }
    unsafe fn libc_kill(pid: u32) {
        extern "C" {
            fn kill(pid: i32, sig: i32) -> i32;
        }
        kill(pid as i32, 9);
    }
    if let Some(kp) = state.keepalive_pid {
        if process_image(kp).is_some_and(|img| is_keepalive_image(&img)) {
            kill_tree(kp);
        }
    }
    if run_alive(state) {
        kill_tree(state.pid);
        for _ in 0..50 {
            if !process_alive(state.pid) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    if state.engine.is_diffusion() {
        remove_diffusion_requests(state, runs_dir);
    }
    forget(state, runs_dir);
    Ok(())
}

#[cfg(windows)]
pub fn stop(state: &RunState, runs_dir: &Path) -> Result<()> {
    if let Some(kp) = state.keepalive_pid {
        if process_image(kp).is_some_and(|img| is_keepalive_image(&img)) {
            let mut k = std::process::Command::new("taskkill");
            k.args(["/PID", &kp.to_string(), "/F"]);
            crate::launch::hide_console(&mut k);
            let _ = k.output();
        }
    }
    if run_alive(state) {
        let mut cmd = std::process::Command::new("taskkill");
        cmd.args(["/PID", &state.pid.to_string(), "/T", "/F"]);
        crate::launch::hide_console(&mut cmd);
        let out = cmd.output().map_err(|e| Error::Platform(format!("taskkill: {e}")))?;
        if !out.status.success() {
            let text = String::from_utf8_lossy(&out.stderr).into_owned();
            // Already gone counts as stopped.
            if !text.contains("not found") && !text.contains("not be found") {
                return Err(Error::Platform(format!("taskkill failed: {text}")));
            }
        }
    }
    if state.engine.is_diffusion() {
        remove_diffusion_requests(state, runs_dir);
    }
    forget(state, runs_dir);
    Ok(())
}

/// Request files hold conversations, and a killed helper cannot clean up
/// the one in flight. Also the helper's %TEMP% fallback, used when the runs
/// dir path is unusable for the runner (non-ASCII or too long).
fn remove_diffusion_requests(state: &RunState, runs_dir: &Path) {
    crate::diffusion::remove_request_files(&crate::launch::req_prefix_for(runs_dir, &state.profile_id, state.port));
    crate::diffusion::remove_request_files(&crate::diffusion::req_fallback_prefix(state.port));
}

/// Drop the run state without touching any process.
pub fn forget(state: &RunState, runs_dir: &Path) {
    let p = RunState::state_path(runs_dir, &state.profile_id, state.port);
    std::fs::remove_file(&p).ok();
}

/// A run state re-checked against reality.
#[derive(Debug, Clone, Serialize)]
pub struct AttachedRun {
    pub state: RunState,
    pub alive: bool,
    pub health: Health,
    /// True when the state file's PID is dead — a crash while the tool was
    /// away (its log outlives it for diagnosis).
    pub crashed: bool,
}

/// Re-discover servers from state files (§07: reopening the tool re-attaches).
/// State files whose process died stay on disk, flagged as crashed, until the
/// user clears them — their logs are the crash evidence.
pub fn reattach(runs_dir: &Path) -> Vec<AttachedRun> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(runs_dir) else {
        return out;
    };
    for e in entries.flatten() {
        let p = e.path();
        if !p.extension().is_some_and(|x| x.eq_ignore_ascii_case("json")) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        let Ok(state) = serde_json::from_str::<RunState>(text.trim_start_matches('\u{feff}')) else { continue };
        let alive = run_alive(&state);
        let health = if alive { probe_health(&state, false) } else { Health::Dead };
        out.push(AttachedRun { crashed: !alive, alive, health, state });
    }
    out.sort_by_key(|r| r.state.port);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_get_talks_to_a_real_socket() {
        use std::io::Write as _;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let t = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = s.read(&mut buf);
            s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"object\":\"ok\"}")
                .unwrap();
        });
        let (status, body) = http_get("127.0.0.1", port, "/v1/models", Duration::from_secs(2)).unwrap();
        assert_eq!(status, 200);
        assert!(body.contains("ok"));
        t.join().unwrap();
    }

    #[test]
    fn chunked_bodies_are_reassembled() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n7\r\n[{\"id\":\r\n3;ext=1\r\n0}]\r\n0\r\n\r\n";
        let (status, body) = parse_response(raw, "x").unwrap();
        assert_eq!(status, 200);
        assert_eq!(body, "[{\"id\":0}]");
        // Content-Length bodies pass through untouched.
        let (_, body) = parse_response(b"HTTP/1.1 404 Not Found\r\nContent-Length: 2\r\n\r\n{}", "x").unwrap();
        assert_eq!(body, "{}");
        // Truncated chunk: keep what arrived rather than failing the poll.
        let (_, body) = parse_response(b"HTTP/1.1 200 OK\r\ntransfer-encoding: Chunked\r\n\r\n10\r\n[{\"id\"", "x").unwrap();
        assert_eq!(body, "[{\"id\"");
    }

    #[test]
    fn process_alive_matches_reality() {
        assert!(process_alive(std::process::id()));
        // Our own image is the test binary, not llama-server.
        let img = process_image(std::process::id()).unwrap().to_ascii_lowercase();
        assert!(if cfg!(windows) { img.ends_with(".exe") } else { img.starts_with("fidim_core") }, "{img}");
        assert!(!process_alive_as(std::process::id(), "llama-server"));
        assert!(!process_alive_as(4_000_000, "llama-server"));
        // PID 4 is System; PID 0 idle — use an absurd value instead.
        assert!(!process_alive(4_000_000));
    }

    #[test]
    fn cold_start_until_a_load_postdates_mtime() {
        let dir = std::env::temp_dir().join(format!("fidim-hist-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        // Never loaded → cold.
        assert!(is_cold_start(&dir, &model));
        // Recorded load (now > mtime) → warm.
        record_model_loaded(&dir, &model).unwrap();
        assert!(!is_cold_start(&dir, &model));
        // Re-download (fresh mtime after the recorded load) → cold again.
        std::thread::sleep(Duration::from_millis(1100));
        std::fs::write(&model, b"GGUF2").unwrap();
        assert!(is_cold_start(&dir, &model));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn run_state_round_trips() {
        let dir = std::env::temp_dir().join(format!("fidim-runs-{}", std::process::id()));
        let state = RunState {
            profile_id: "worker-pool".into(),
            pid: 1234,
            port: 9701,
            host: "127.0.0.1".into(),
            alias: "worker".into(),
            started_unix: 1_700_000_000,
            log_path: dir.join("worker-pool-9701.log"),
            command_line: "llama-server.exe -m model.gguf".into(),
            visibility_env: "2".into(),
            device_keys: vec!["pci:A:bus08".into()],
            free_mib_before: vec![32472],
            cold_start: false,
            estimated_bytes: 20 * 1024 * 1024 * 1024,
            ready: true,
            keepalive_pid: None,
            engine: Engine::LlamaServer,
        };
        state.save(&dir).unwrap();
        let attached = reattach(&dir);
        assert_eq!(attached.len(), 1);
        assert_eq!(attached[0].state.alias, "worker");
        assert!(attached[0].crashed, "pid 1234 should not be our live process");
        std::fs::remove_dir_all(dir).ok();
    }

    fn diffusion_state(dir: &Path, pid: u32) -> RunState {
        RunState {
            profile_id: "dg-26b".into(),
            pid,
            port: 9760,
            host: "127.0.0.1".into(),
            alias: "dg".into(),
            started_unix: 1_700_000_000,
            log_path: dir.join("dg-26b-9760.log"),
            command_line: "fidim-dg.exe --runner r.exe".into(),
            visibility_env: "2".into(),
            device_keys: vec!["pci:A:bus08".into()],
            free_mib_before: vec![32472],
            cold_start: false,
            estimated_bytes: 0,
            ready: true,
            keepalive_pid: None,
            engine: Engine::DiffusionGemma,
        }
    }

    #[test]
    fn engine_identity() {
        let dir = std::env::temp_dir().join(format!("fidim-runs-engine-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();

        // A state file from before the field reads as llama-server, and a
        // llama-server state still saves without the key.
        let legacy = r#"{"profile_id":"w","pid":1,"port":9701,"host":"127.0.0.1","alias":"w",
            "started_unix":0,"log_path":"w.log","command_line":"llama-server.exe","visibility_env":"2",
            "device_keys":[],"free_mib_before":[]}"#;
        let s: RunState = serde_json::from_str(legacy).unwrap();
        assert_eq!(s.engine, Engine::LlamaServer);
        assert!(!serde_json::to_string(&s).unwrap().contains("engine"));

        // A diffusion state round-trips through its file.
        let s = diffusion_state(&dir, std::process::id());
        s.save(&dir).unwrap();
        let text = std::fs::read_to_string(RunState::state_path(&dir, "dg-26b", 9760)).unwrap();
        assert!(text.contains(r#""engine": "diffusion-gemma""#), "{text}");
        let back: RunState = serde_json::from_str(&text).unwrap();
        assert_eq!(back.engine, Engine::DiffusionGemma);

        // Our own pid is alive but it is the test binary, not fidim-dg.exe:
        // the run is crashed, never ours to kill.
        let attached = reattach(&dir);
        assert_eq!(attached.len(), 1);
        assert!(attached[0].crashed && !attached[0].alive, "{:?}", attached[0]);
        assert!(!run_alive(&s));
        let unknown = RunState { engine: Engine::Unknown, ..s.clone() };
        assert!(!run_alive(&unknown));

        // No keep-alive for diffusion runs, whatever the interval.
        let mut ka = s.clone();
        assert_eq!(spawn_keepalive(&mut ka, &dir, 5).unwrap(), None);
        assert_eq!(ka.keepalive_pid, None);

        // stop forgets the run (killing nothing: it is not alive) and deletes
        // its request files, but only its own.
        let mine = dir.join("dg-dg-26b-9760-3-1.req");
        let other = dir.join("dg-dg-26b-97601-3-1.req");
        std::fs::write(&mine, b"{}").unwrap();
        std::fs::write(&other, b"{}").unwrap();
        stop(&s, &dir).unwrap();
        assert!(!mine.exists());
        assert!(other.exists());
        assert!(!RunState::state_path(&dir, "dg-26b", 9760).exists());
        std::fs::remove_dir_all(&dir).ok();

        // The keep-alive kill matches exact images only.
        assert!(is_keepalive_image("FIDIM.EXE"));
        assert!(is_keepalive_image("fidim.exe"));
        assert!(is_keepalive_image("llamactl.exe"));
        assert!(!is_keepalive_image("llama-fidim.exe"));
        assert!(!is_keepalive_image("fidim-dg.exe"));
    }
}


// ---------------------------------------------------------------- keep-alive ----

/// A companion binary shipped with Llama FIDIM: the running executable when
/// that is `name`, else an existing `name` beside it (install.ps1 and
/// `target\<profile>` both put the CLI, the GUI and fidim-dg side by side).
pub fn helper_exe(name: &str) -> Option<PathBuf> {
    let me = std::env::current_exe().ok()?;
    if me.file_name().is_some_and(|n| n.eq_ignore_ascii_case(name)) {
        return Some(me);
    }
    let sibling = me.parent()?.join(name);
    sibling.is_file().then_some(sibling)
}

/// The CLI binary that hosts the `keepalive` loop: `fidim.exe` beside the
/// running executable (the GUI ships next to it), or the executable itself.
pub fn keepalive_exe() -> Option<std::path::PathBuf> {
    helper_exe("fidim.exe")
}

/// Arguments for the helper process (kept separate so they are testable).
pub fn keepalive_args(host: &str, port: u16, interval_s: u32, server_pid: u32) -> Vec<String> {
    vec![
        "keepalive".into(),
        "--host".into(),
        host.into(),
        "--port".into(),
        port.to_string(),
        "--interval".into(),
        interval_s.to_string(),
        "--server-pid".into(),
        server_pid.to_string(),
    ]
}

/// Start the detached keep-alive helper for a ready run and record its PID
/// in the run state. Why: with no lit display an idle adapter is powered
/// down and WDDM evicts every allocation first (measured 2026-09-05: 25.6 GB
/// gone within 20 s of the monitors switching off; a 1-token request every
/// 5 s held it for the full test). `interval_s == 0` = disabled.
///
/// Never for a diffusion run: its smallest request is a whole 256-token
/// denoise block, and the helper has no `/completion` to post to.
pub fn spawn_keepalive(state: &mut RunState, runs_dir: &Path, interval_s: u32) -> Result<Option<u32>> {
    if state.engine.is_diffusion() {
        return Ok(None);
    }
    if interval_s == 0 {
        return Ok(None);
    }
    let exe = keepalive_exe().ok_or_else(|| {
        Error::Platform("keep-alive helper fidim.exe not found beside the running executable".into())
    })?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.args(keepalive_args(&state.host, state.port, interval_s, state.pid))
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
        stop_inheriting_std_handles();
    }
    let child = cmd.spawn().map_err(|e| Error::Platform(format!("spawn keep-alive: {e}")))?;
    state.keepalive_pid = Some(child.id());
    state.save(runs_dir)?;
    Ok(Some(child.id()))
}

/// The keep-alive loop itself (runs inside `fidim keepalive`). Exits when
/// the server process is gone. Request failures are ignored: a busy or
/// restarting server is not a reason to stop trying.
pub fn run_keepalive(host: &str, port: u16, interval_s: u32, server_pid: u32) {
    let body = serde_json::json!({ "prompt": "hi", "n_predict": 1, "cache_prompt": false });
    let interval = Duration::from_secs(interval_s.max(1) as u64);
    // `process_alive` shells out to tasklist; one hiccup there must not
    // take the helper down with it. Only three consecutive "gone" verdicts
    // end the loop.
    let mut gone_streak = 0u32;
    loop {
        std::thread::sleep(interval);
        if process_alive(server_pid) {
            gone_streak = 0;
        } else {
            gone_streak += 1;
            if gone_streak >= 3 {
                return;
            }
            continue;
        }
        let _ = http_post_json(host, port, "/completion", &body.to_string(), Duration::from_secs(120));
    }
}

#[cfg(test)]
mod keepalive_tests {
    use super::*;

    #[test]
    fn keepalive_args_round_trip() {
        let a = keepalive_args("127.0.0.1", 1234, 5, 42);
        assert_eq!(
            a,
            ["keepalive", "--host", "127.0.0.1", "--port", "1234", "--interval", "5", "--server-pid", "42"]
        );
    }
}
