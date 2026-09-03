//! Server supervision (R-07, §07 "servers outlive the UI"): detached spawn
//! with log capture, readiness by polling `/v1/models`, tri-state health,
//! stop, and re-attach after the tool restarts.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::launch::LaunchPlan;
use crate::profile::Profile;
use crate::{Error, Result};

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
}

/// Bytes reserved by runs that are alive but not yet finished loading.
/// A loading server has claimed its memory without having charged it, so a
/// concurrent pre-flight that ignores it sees phantom headroom.
pub fn inflight_reserved_bytes(runs_dir: &Path) -> u64 {
    reattach(runs_dir)
        .iter()
        .filter(|r| r.alive && !r.state.ready)
        .map(|r| r.state.estimated_bytes)
        .sum()
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
/// stderr appended to a log file. Closing llamactl later must not kill it.
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
        joined.push(";");
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
    };
    state.save(runs_dir)?;
    Ok(state)
}

// ------------------------------------------------------------------ HTTP ----

/// Minimal HTTP/1.1 GET — avoids an async client dependency for what is a
/// localhost status poll. Returns (status code, body).
pub fn http_get(host: &str, port: u16, path: &str, timeout: Duration) -> Result<(u16, String)> {
    let addr = format!("{host}:{port}");
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
    let req = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    stream
        .write_all(req.as_bytes())
        .map_err(|e| Error::Platform(format!("send {addr}: {e}")))?;
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).ok(); // best-effort: server may cut the stream
    let text = String::from_utf8_lossy(&buf);
    let status = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::Platform(format!("no HTTP status from {addr}")))?;
    let body = text.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
    Ok((status, body))
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
    let addr = format!("{host}:{port}");
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
    let text = String::from_utf8_lossy(&buf);
    let status = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::Platform(format!("no HTTP status from {addr}")))?;
    let body = text.split_once("\r\n\r\n").map(|(_, b)| b.to_string()).unwrap_or_default();
    Ok((status, body))
}

/// Clear the inherit flag on this process's own stdin/stdout/stderr before
/// spawning the detached server. Rust's `Command` passes
/// `bInheritHandles = TRUE` whenever stdio is configured, and every
/// inheritable handle in the parent comes along — including the pipe a
/// shell created to capture OUR output. The server then holds that pipe's
/// write end for its whole lifetime and the shell's reader never sees EOF:
/// `llamactl launch … | Out-String` (2026-09-02, 2.5 h hang) and
/// `llamactl launch … | tail` both deadlocked this way. The server's own
/// stdio is set explicitly to the log file, so it loses nothing.
#[cfg(windows)]
fn stop_inheriting_std_handles() {
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
        if let Ok((200, _)) = http_get(&state.host, state.port, "/v1/models", Duration::from_secs(2))
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
        let body = format!(
            r#"{{"model":"{}","prompt":"Hi","max_tokens":1,"stream":false}}"#,
            state.alias
        );
        let ok = matches!(
            http_post_json(&state.host, state.port, "/v1/completions", &body, Duration::from_secs(60)),
            Ok((200, _))
        );
        if !ok {
            return Health::RespondingNotGenerating;
        }
    }
    Health::Healthy
}

pub fn process_alive(pid: u32) -> bool {
    // tasklist filters by PID; output contains the PID only when it exists.
    let out = std::process::Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).contains(&format!("\"{pid}\"")),
        Err(_) => false,
    }
}

/// Stop a server. llama-server has no shutdown endpoint; on Windows a
/// process-tree terminate is the clean stop (§04 clean-stop requirement).
pub fn stop(state: &RunState, runs_dir: &Path) -> Result<()> {
    let out = std::process::Command::new("taskkill")
        .args(["/PID", &state.pid.to_string(), "/T", "/F"])
        .output()
        .map_err(|e| Error::Platform(format!("taskkill: {e}")))?;
    if !out.status.success() {
        let text = String::from_utf8_lossy(&out.stderr).into_owned();
        // Already gone counts as stopped.
        if !text.contains("not found") && !text.contains("not be found") {
            return Err(Error::Platform(format!("taskkill failed: {text}")));
        }
    }
    let p = RunState::state_path(runs_dir, &state.profile_id, state.port);
    std::fs::remove_file(&p).ok();
    Ok(())
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
        let Ok(state) = serde_json::from_str::<RunState>(&text) else { continue };
        let alive = process_alive(state.pid);
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
    fn process_alive_matches_reality() {
        assert!(process_alive(std::process::id()));
        // PID 4 is System; PID 0 idle — use an absurd value instead.
        assert!(!process_alive(4_000_000));
    }

    #[test]
    fn cold_start_until_a_load_postdates_mtime() {
        let dir = std::env::temp_dir().join(format!("llamactl-hist-{}", std::process::id()));
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
        let dir = std::env::temp_dir().join(format!("llamactl-runs-{}", std::process::id()));
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
        };
        state.save(&dir).unwrap();
        let attached = reattach(&dir);
        assert_eq!(attached.len(), 1);
        assert_eq!(attached[0].state.alias, "worker");
        assert!(attached[0].crashed, "pid 1234 should not be our live process");
        std::fs::remove_dir_all(dir).ok();
    }
}
