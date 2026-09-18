//! The engine worker: one thread that owns the runner process and turns
//! its line protocol into per-job events (shim spec §3-§4).
//!
//! Everything arrives on ONE channel — jobs from the HTTP threads, the
//! runner's stdout lines and stderr facts (tagged with the spawn generation
//! so a dead runner's leftovers are ignored), and shutdown — so the state
//! machine never has to choose between blocking reads.
//!
//! The runner is strictly one request at a time and has no request ids;
//! the protocol stays in sync only because every request is read to its
//! terminal record before the next is sent (Unsloth's driver desyncs here,
//! visual_engine.py:331-337).

use std::collections::VecDeque;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::openai::exit_hex;
use super::protocol::{self, EngineRequest, ErrLine, Fact, Line, Stats};
use super::{dglog, job, lock, ModelInfo, ServeConfig, Shared};
use super::{EXIT_BREAKER, EXIT_BUS, EXIT_GUARD, EXIT_LOAD};

/// The runner prints its `ready (…)` line on stderr just before READY on
/// stdout (VS:281-286); the two pipes are read by different threads.
const GUARD_WAIT: Duration = Duration::from_secs(5);
/// Absorbs the multi-line `ERR parse` dumps (chat.cpp:616 appends the
/// whole tools JSON to the exception text).
const ERR_QUIET: Duration = Duration::from_millis(200);
/// An unrecognised ERR may or may not be followed by DONE.
const ERR_UNKNOWN_WAIT: Duration = Duration::from_millis(500);
/// The runner's LOG_ERR is asynchronous: at DONE, a step failure can still
/// be on its way through stderr.
const STDERR_CATCHUP: Duration = Duration::from_millis(200);
/// A dying runner gets this long to exit on its own before it is killed.
const EXIT_WAIT: Duration = Duration::from_secs(5);
/// Never spawn while the old runner lives: two ~17 GB processes on one card
/// oversubscribe WDDM.
const KILL_WAIT: Duration = Duration::from_secs(30);
/// Runner deaths with no successful job in between before giving up.
const BREAKER: u32 = 3;
const TAIL_LINES: usize = 40;

/// Env keys that can add devices or fake memory under the runner
/// (ggml-backend-reg.cpp:600-604; ggml-cuda.cu:279-289, 5100).
const SCRUBBED_ENV: &[&str] =
    &["GGML_BACKEND_PATH", "GGML_CUDA_DEVICES", "GGML_CUDA_ENABLE_UNIFIED_MEMORY", "DG_FREE_VRAM_MB"];

// -------------------------------------------------------------- interfaces ----

pub trait Spawner: Send {
    /// Start a runner whose output is sent on `tx` tagged with `gen`.
    /// `maxtok_override` pins MAXTOK on respawns so n_ctx never grows under
    /// a live client.
    fn spawn(
        &mut self,
        gen: u64,
        maxtok_override: Option<u32>,
        tx: Sender<EngineMsg>,
    ) -> io::Result<Box<dyn EngineChild>>;
}

pub trait EngineChild: Send {
    fn send_line(&mut self, l: &str) -> io::Result<()>;
    fn pid(&self) -> Option<u32>;
    fn kill_and_wait(&mut self, timeout: Duration) -> Option<i32>;
    fn try_exit_code(&mut self) -> Option<i32>;
    fn stderr_tail(&self) -> Vec<String>;
    /// True once the stderr reader has hit EOF, i.e. every fact the runner
    /// printed before dying has been sent. Crash classification waits for it.
    fn stderr_drained(&self) -> bool {
        true
    }
}

pub enum EngineMsg {
    Job(Job),
    Child { gen: u64, out: ChildOut },
    Shutdown,
}

pub enum ChildOut {
    Line(Vec<u8>),
    Eof,
    Fact(Fact),
}

pub struct Job {
    pub req: EngineRequest,
    pub seed_pinned: bool,
    pub stream: bool,
    pub events: Sender<EngineEvent>,
    /// Set by the HTTP thread when the client went away. Only checked while
    /// the job is queued: there is no mid-request cancel in the runner.
    pub cancelled: Arc<AtomicBool>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    Started { id_task: u64, seed: i32 },
    Commit { block: u32, text: String },
    Stats(Stats),
    TooLong { needed: u32, budget: u32, after_commit: bool },
    /// A retry starts from nothing, with this seed.
    Restarted { seed: i32 },
    Done,
    Failed(EngineFailure),
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineFailure {
    Parse(String),
    EmptyPrompt,
    BadReqFile,
    Gen,
    StepFailed,
    Oom,
    Crashed { exit: Option<i32>, detail: String },
    Watchdog(u64),
    /// An `ERR` this build does not know.
    Runner(String),
    Unavailable,
}

// ------------------------------------------------------------ device guard ----

/// The positive-evidence device guard (shim spec §2.6): the helper refuses
/// to become ready unless the runner's own log proves it is on exactly one
/// ROCm device on the expected bus, on the single-device path, fully
/// offloaded. Returns a summary, or `(exit code, reason)`.
pub(crate) fn evaluate_guard(
    facts: &[Fact],
    ngl: u32,
    expect_bus: Option<u32>,
    block_count: u64,
) -> Result<String, (i32, String)> {
    let devices: Vec<(&str, Option<u32>)> = facts
        .iter()
        .filter_map(|f| match f {
            Fact::UsingDevice { name, bus } => Some((name.as_str(), *bus)),
            _ => None,
        })
        .collect();
    if devices.len() != 1 {
        return Err((
            EXIT_GUARD,
            format!(
                "the runner reported {} 'using device' lines, expected exactly one; more than one device \
                 takes the multi-device path, which aborts every prompt, and none means a CPU run",
                devices.len()
            ),
        ));
    }
    let (name, bus) = devices[0];
    if !name.starts_with("ROCm") {
        return Err((EXIT_GUARD, format!("the runner is using device {name}, not a ROCm device")));
    }
    if let Some(expected) = expect_bus {
        if bus != Some(expected) {
            let seen = bus.map(|b| format!("bus {b:02x}")).unwrap_or_else(|| "an unknown bus".into());
            return Err((
                EXIT_BUS,
                format!("visibility index resolved to a different card: the runner is on {seen}, expected bus {expected:02x}"),
            ));
        }
    }
    let Some((runner_ngl, kv_on)) = facts.iter().rev().find_map(|f| match f {
        Fact::RunnerReady { ngl, kv_cache_on } => Some((*ngl, *kv_cache_on)),
        _ => None,
    }) else {
        return Err((EXIT_GUARD, "the runner's 'ready (… kv_cache=…)' line never appeared on stderr".into()));
    };
    if !kv_on {
        return Err((
            EXIT_GUARD,
            "the runner reports kv_cache=off: it is not on the single-device path (VS:273-277)".into(),
        ));
    }
    if runner_ngl != Some(ngl) {
        let seen = runner_ngl.map(|n| n.to_string()).unwrap_or_else(|| "nothing".into());
        return Err((EXIT_GUARD, format!("the runner reports NGL={seen}, FIDIM asked for NGL={ngl}")));
    }
    let offload = facts.iter().rev().find_map(|f| match f {
        Fact::Offloaded { done, total } => Some((*done, *total)),
        _ => None,
    });
    // Full offload = every repeating layer plus the output layer: NGL ≥ block_count + 1.
    let full = ngl as u64 > block_count;
    if full {
        match offload {
            Some((d, t)) if d == t => {}
            Some((d, t)) => {
                return Err((EXIT_GUARD, format!("only {d}/{t} layers were offloaded although NGL={ngl} asks for all")))
            }
            None => return Err((EXIT_GUARD, "the runner never reported its 'offloaded X/Y layers' line".into())),
        }
    }
    let bus_text = bus.map(|b| format!("bus {b:02x}")).unwrap_or_else(|| "unknown bus".into());
    let offload_text = offload.map(|(d, t)| format!(", offloaded {d}/{t}")).unwrap_or_default();
    Ok(format!("{name} on {bus_text}, NGL={ngl}, kv_cache=on{offload_text}"))
}

// ------------------------------------------------------------------ worker ----

enum Next {
    Line(Line),
    Eof,
    Timeout,
    Shutdown,
    Job,
}

/// `Err(code)` ends the helper with `code` (0 = graceful shutdown).
type Flow<T> = Result<T, i32>;

enum Boot {
    /// The runner died, timed out or could not start before READY.
    Died(String),
    Exit(i32),
}

enum Outcome {
    /// The job's terminal event has been sent.
    Finished,
    Retry,
}

struct Run {
    id_task: u64,
    attempt: u32,
    seed: i32,
    retry_left: bool,
}

#[derive(Default)]
struct AttemptState {
    /// F records since the last C.
    frames: u32,
    commit_sent: bool,
    stale: bool,
    toolong: Option<(u32, u32)>,
    stats: Option<Stats>,
    err_gen: bool,
}

struct Worker<'a> {
    cfg: &'a ServeConfig,
    shared: Arc<Shared>,
    info: &'a ModelInfo,
    spawner: Box<dyn Spawner>,
    rx: Receiver<EngineMsg>,
    tx: Sender<EngineMsg>,
    gen: u64,
    spawned_once: bool,
    child: Option<Box<dyn EngineChild>>,
    pending: VecDeque<Job>,
    /// Stderr facts of the current runner, in arrival order.
    facts: Vec<Fact>,
    pushback: Option<Next>,
    shutdown_seen: bool,
    id_task: u64,
    crash_streak: u32,
    /// The first READY's MAXTOK; every respawn is pinned to it.
    pinned_maxtok: Option<u32>,
    stray_lines: u64,
    live_req: Option<PathBuf>,
    last_tail: Vec<String>,
}

/// The §4 state machine. Returns the helper's exit code.
pub fn run_engine(
    cfg: &ServeConfig,
    info: &ModelInfo,
    shared: Arc<Shared>,
    spawner: Box<dyn Spawner>,
    rx: Receiver<EngineMsg>,
    tx: Sender<EngineMsg>,
) -> i32 {
    let mut w = Worker {
        cfg,
        shared,
        info,
        spawner,
        rx,
        tx,
        gen: 0,
        spawned_once: false,
        child: None,
        pending: VecDeque::new(),
        facts: Vec::new(),
        pushback: None,
        shutdown_seen: false,
        id_task: 0,
        crash_streak: 0,
        pinned_maxtok: None,
        stray_lines: 0,
        live_req: None,
        last_tail: Vec::new(),
    };
    let code = match w.run() {
        Ok(()) => 0,
        Err(c) => c,
    };
    w.teardown(code);
    if code == 0 {
        dglog!("stopped");
    } else {
        dglog!("exiting with code {code}");
    }
    code
}

fn code_text(code: Option<i32>) -> String {
    code.map(exit_hex).unwrap_or_else(|| "unknown".into())
}

/// The line that best explains a crash: the last one naming an error.
fn crash_detail(tail: &[String]) -> String {
    let hit = tail.iter().rev().find(|l| {
        let low = l.to_ascii_lowercase();
        low.contains("rocm error") || low.contains("ggml_assert") || low.contains("failed") || low.contains("error")
    });
    hit.map(|l| l.trim().chars().take(300).collect()).unwrap_or_default()
}

/// STATUS_STACK_BUFFER_OVERRUN: how the MSVC runtime's `abort()` ends a
/// process (__fastfail).
const EXIT_FASTFAIL: i32 = 0xC000_0409_u32 as i32;

/// The Windows runner links the MSVC runtime, where an uncaught C++
/// exception goes terminate → abort → __fastfail: exit 0xC0000409 and
/// nothing on stderr (no libstdc++ "terminate called", no what()).
/// GGML_ASSERT and ROCm errors end the same way but print their reason
/// first. The runner's one uncaught throw on a request path is dump() of an
/// invalid-UTF-8 canvas (VS:81, 360), a crash that depends on the seed, so a
/// silent fastfail is classified as that.
fn silent_fastfail(code: Option<i32>, tail: &[String]) -> bool {
    code == Some(EXIT_FASTFAIL) && crash_detail(tail).is_empty()
}

/// Says "retrying" only when a restart really follows: once the circuit
/// breaker trips, the helper gives up instead.
fn death_line(run: &Run, code: Option<i32>, hint: Option<&str>, restart: bool) -> String {
    format!(
        "runner died during #{} attempt {} (exit {}{}){}",
        run.id_task,
        run.attempt,
        code_text(code),
        hint.map(|h| format!(", {h}")).unwrap_or_default(),
        if restart { "; retrying after a restart" } else { "" }
    )
}

impl Worker<'_> {
    fn run(&mut self) -> Flow<()> {
        self.boot(true)?;
        loop {
            if let Some(job) = self.pending.pop_front() {
                self.run_job(job)?;
                continue;
            }
            match self.next(None, true) {
                Next::Job | Next::Timeout => {}
                Next::Shutdown => return Err(0),
                Next::Line(_) => self.stray_lines += 1,
                Next::Eof => self.idle_death()?,
            }
        }
    }

    /// The next thing that needs attention. Jobs are queued on the way;
    /// with `wake_on_job` their arrival is itself a wake-up.
    fn next(&mut self, deadline: Option<Instant>, wake_on_job: bool) -> Next {
        if let Some(n) = self.pushback.take() {
            return n;
        }
        if self.shutdown_seen {
            return Next::Shutdown;
        }
        loop {
            let msg = match deadline {
                None => match self.rx.recv() {
                    Ok(m) => m,
                    Err(_) => return Next::Shutdown,
                },
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        return Next::Timeout;
                    }
                    match self.rx.recv_timeout(d - now) {
                        Ok(m) => m,
                        Err(RecvTimeoutError::Timeout) => return Next::Timeout,
                        Err(RecvTimeoutError::Disconnected) => return Next::Shutdown,
                    }
                }
            };
            match msg {
                EngineMsg::Job(j) => {
                    self.pending.push_back(j);
                    if wake_on_job {
                        return Next::Job;
                    }
                }
                EngineMsg::Shutdown => {
                    self.shutdown_seen = true;
                    return Next::Shutdown;
                }
                EngineMsg::Child { gen, .. } if gen != self.gen => {}
                EngineMsg::Child { out: ChildOut::Fact(f), .. } => self.facts.push(f),
                EngineMsg::Child { out: ChildOut::Line(b), .. } => return Next::Line(protocol::parse_line(&b)),
                EngineMsg::Child { out: ChildOut::Eof, .. } => return Next::Eof,
            }
        }
    }

    /// Let stderr facts arrive for `dur`; the first stdout event ends the
    /// wait and is kept for the caller's next read.
    fn settle(&mut self, dur: Duration) {
        let deadline = Instant::now() + dur;
        loop {
            match self.next(Some(deadline), false) {
                Next::Timeout => return,
                Next::Job => {}
                other => {
                    self.pushback = Some(other);
                    return;
                }
            }
        }
    }

    /// Discard stdout lines until `dur` passes with none.
    fn quiet(&mut self, dur: Duration) {
        let mut deadline = Instant::now() + dur;
        loop {
            match self.next(Some(deadline), false) {
                Next::Line(_) => deadline = Instant::now() + dur,
                Next::Timeout => return,
                Next::Job => {}
                other => {
                    self.pushback = Some(other);
                    return;
                }
            }
        }
    }

    /// After an unknown ERR: read to DONE, or until `dur` of silence.
    fn await_done(&mut self, dur: Duration) {
        let mut deadline = Instant::now() + dur;
        loop {
            match self.next(Some(deadline), false) {
                Next::Line(Line::Done) | Next::Timeout => return,
                Next::Line(_) => deadline = Instant::now() + dur,
                Next::Job => {}
                other => {
                    self.pushback = Some(other);
                    return;
                }
            }
        }
    }

    /// Pick up whatever the channel already holds without waiting.
    fn collect_queued_messages(&mut self) {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                EngineMsg::Job(j) => self.pending.push_back(j),
                EngineMsg::Shutdown => self.shutdown_seen = true,
                EngineMsg::Child { gen, out: ChildOut::Fact(f) } if gen == self.gen => self.facts.push(f),
                // Lines and EOF of a runner that is already gone.
                EngineMsg::Child { .. } => {}
            }
        }
    }

    /// Make sure the current runner is dead: wait `grace` for it to exit,
    /// then kill it. Returns its exit code. A runner that outlives the kill
    /// (stuck in a driver call) ends the helper instead: every caller would
    /// boot a replacement next to it, and two ~17 GB runners oversubscribe
    /// the card. Exiting closes the job object, so Windows finishes the old
    /// runner off whenever its call returns.
    fn reap(&mut self, grace: Duration) -> Flow<Option<i32>> {
        let Some(mut child) = self.child.take() else { return Ok(None) };
        let deadline = Instant::now() + grace;
        let mut code = child.try_exit_code();
        while code.is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
            code = child.try_exit_code();
        }
        if code.is_none() {
            code = child.kill_and_wait(KILL_WAIT);
            if code.is_none() {
                self.shared.child_pid.store(0, Ordering::SeqCst);
                self.shared.ready.store(false, Ordering::SeqCst);
                dglog!(
                    "the runner (pid {}) is still alive {} s after it was killed; exiting rather than loading \
                     a second copy next to it",
                    child.pid().map(|p| p.to_string()).unwrap_or_else(|| "?".into()),
                    KILL_WAIT.as_secs()
                );
                return Err(EXIT_BREAKER);
            }
        }
        // Crash hints are often the last thing it printed.
        let drained_by = Instant::now() + Duration::from_secs(1);
        while !child.stderr_drained() && Instant::now() < drained_by {
            std::thread::sleep(Duration::from_millis(20));
        }
        self.collect_queued_messages();
        self.last_tail = child.stderr_tail();
        self.shared.child_pid.store(0, Ordering::SeqCst);
        self.shared.ready.store(false, Ordering::SeqCst);
        Ok(code)
    }

    fn log_death(&self, what: &str) {
        dglog!("{what}");
        let skip = self.last_tail.len().saturating_sub(10);
        for l in &self.last_tail[skip..] {
            dglog!("  | {l}");
        }
    }

    // ---------------------------------------------------------- lifecycle ----

    /// Spawn until a runner passes READY and the device guard. The first
    /// start fails the helper (exit 3); later ones retry until the breaker.
    fn boot(&mut self, initial: bool) -> Flow<()> {
        loop {
            match self.start_runner() {
                Ok(()) => return Ok(()),
                Err(Boot::Exit(code)) => return Err(code),
                Err(Boot::Died(why)) => {
                    self.log_death(&why);
                    if initial {
                        return Err(EXIT_LOAD);
                    }
                    self.crash_streak += 1;
                    if self.crash_streak >= BREAKER {
                        dglog!("circuit breaker: {BREAKER} runner deaths without a successful request; giving up");
                        return Err(EXIT_BREAKER);
                    }
                }
            }
        }
    }

    fn start_runner(&mut self) -> Result<(), Boot> {
        // A shutdown picked up while reaping must not load a fresh 16 GB.
        if self.shutdown_seen {
            return Err(Boot::Exit(0));
        }
        self.gen += 1;
        self.facts.clear();
        self.pushback = None;
        self.shared.ready.store(false, Ordering::SeqCst);
        self.shared.progress().state = "loading";
        if self.spawned_once {
            self.shared.restarts.fetch_add(1, Ordering::SeqCst);
        }
        self.spawned_once = true;

        let child = self
            .spawner
            .spawn(self.gen, self.pinned_maxtok, self.tx.clone())
            .map_err(|e| Boot::Died(format!("cannot start the runner: {e}")))?;
        let pid = child.pid();
        self.shared.child_pid.store(pid.unwrap_or(0), Ordering::SeqCst);
        self.child = Some(child);
        dglog!(
            "runner started (pid {}{})",
            pid.map(|p| p.to_string()).unwrap_or_else(|| "?".into()),
            self.pinned_maxtok.map(|m| format!(", MAXTOK pinned to {m}")).unwrap_or_default()
        );

        let deadline = Instant::now() + self.cfg.load_timeout;
        let (n_vocab, maxtok) = loop {
            match self.next(Some(deadline), false) {
                Next::Line(Line::Ready { n_vocab, maxtok }) => break (n_vocab, maxtok),
                Next::Line(_) | Next::Job => {}
                Next::Eof => {
                    let code = self.reap(EXIT_WAIT).map_err(Boot::Exit)?;
                    return Err(Boot::Died(format!("the runner exited before READY (exit {})", code_text(code))));
                }
                Next::Timeout => {
                    self.reap(Duration::ZERO).map_err(Boot::Exit)?;
                    return Err(Boot::Died(format!(
                        "the runner did not print READY within {} s",
                        self.cfg.load_timeout.as_secs()
                    )));
                }
                Next::Shutdown => return Err(Boot::Exit(0)),
            }
        };

        self.device_guard()?;

        // Builds before the auto-sizer print `READY <n_vocab>` only.
        let maxtok = maxtok
            .filter(|&m| m > 0)
            .or(self.pinned_maxtok)
            .or((self.cfg.maxtok_env > 0).then_some(self.cfg.maxtok_env))
            .unwrap_or(0);
        if self.pinned_maxtok.is_none() && maxtok > 0 {
            self.pinned_maxtok = Some(maxtok);
        }
        self.shared.maxtok.store(maxtok, Ordering::SeqCst);
        self.shared.n_vocab.store(n_vocab, Ordering::SeqCst);
        self.shared.progress().state = "idle";
        self.shared.ready.store(true, Ordering::SeqCst);
        self.shared.ever_ready.store(true, Ordering::SeqCst);
        dglog!("ready: n_vocab={n_vocab} MAXTOK={maxtok}");
        Ok(())
    }

    fn device_guard(&mut self) -> Result<(), Boot> {
        if self.cfg.ngl == 0 {
            dglog!("device guard skipped: NGL=0 runs the model on the CPU");
            return Ok(());
        }
        let deadline = Instant::now() + GUARD_WAIT;
        while !self.facts.iter().any(|f| matches!(f, Fact::RunnerReady { .. })) {
            match self.next(Some(deadline), false) {
                Next::Timeout => break,
                Next::Line(_) => self.stray_lines += 1,
                Next::Job => {}
                Next::Eof => {
                    let code = self.reap(EXIT_WAIT).map_err(Boot::Exit)?;
                    return Err(Boot::Died(format!("the runner exited right after READY (exit {})", code_text(code))));
                }
                Next::Shutdown => return Err(Boot::Exit(0)),
            }
        }
        match evaluate_guard(&self.facts, self.cfg.ngl, self.cfg.expect_bus, self.info.block_count) {
            Ok(summary) => {
                dglog!("device guard passed: {summary}");
                Ok(())
            }
            Err((code, why)) => {
                dglog!("device guard failed: {why}; refusing to serve");
                self.reap(Duration::ZERO).map_err(Boot::Exit)?;
                Err(Boot::Exit(code))
            }
        }
    }

    fn idle_death(&mut self) -> Flow<()> {
        let code = self.reap(EXIT_WAIT)?;
        self.crash_streak += 1;
        self.log_death(&format!("runner exited while idle (exit {}); restarting it", code_text(code)));
        if self.crash_streak >= BREAKER {
            dglog!("circuit breaker: {BREAKER} runner deaths without a successful request; giving up");
            return Err(EXIT_BREAKER);
        }
        self.boot(false)
    }

    fn teardown(&mut self, code: i32) {
        if let Some(mut c) = self.child.take() {
            if code == 0 && c.send_line("QUIT").is_ok() {
                let deadline = Instant::now() + EXIT_WAIT;
                while c.try_exit_code().is_none() && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            if c.try_exit_code().is_none() {
                c.kill_and_wait(KILL_WAIT);
            }
        }
        self.shared.exited.store(true, Ordering::SeqCst);
        self.collect_queued_messages();
        for j in self.pending.drain(..) {
            let _ = self.shared.queued.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |q| q.checked_sub(1));
            let _ = j.events.send(EngineEvent::Failed(EngineFailure::Unavailable));
        }
        self.drop_req();
        super::remove_request_files(&self.cfg.req_prefix);
        self.shared.ready.store(false, Ordering::SeqCst);
        self.shared.processing.store(false, Ordering::SeqCst);
        self.shared.child_pid.store(0, Ordering::SeqCst);
    }

    // --------------------------------------------------------------- jobs ----

    fn drop_req(&mut self) {
        if let Some(p) = self.live_req.take() {
            let _ = std::fs::remove_file(p);
        }
    }

    fn run_job(&mut self, job: Job) -> Flow<()> {
        let _ = self.shared.queued.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |q| q.checked_sub(1));
        self.shared.served.fetch_add(1, Ordering::SeqCst);
        if job.cancelled.load(Ordering::SeqCst) {
            dglog!("skipped a queued request: its client disconnected");
            return Ok(());
        }
        if self.stray_lines > 0 {
            dglog!("dropped {} stray runner line(s) that arrived outside a request", self.stray_lines);
            self.stray_lines = 0;
        }
        self.id_task += 1;
        let mut run = Run { id_task: self.id_task, attempt: 1, seed: job.req.seed, retry_left: true };
        // Progress first: a /slots poll that sees is_processing must also
        // see this job's id_task.
        self.set_prefill(run.id_task, job.req.n_blocks);
        self.shared.processing.store(true, Ordering::SeqCst);
        let _ = job.events.send(EngineEvent::Started { id_task: run.id_task, seed: run.seed });
        let result = loop {
            match self.attempt(&job, &mut run) {
                Ok(Outcome::Finished) => break Ok(()),
                Ok(Outcome::Retry) => {
                    run.attempt += 1;
                    // A pinned seed is the client's; a random one moves on so
                    // a seed-dependent failure is not replayed.
                    if !job.seed_pinned {
                        run.seed = run.seed.saturating_add(1);
                    }
                    let _ = job.events.send(EngineEvent::Restarted { seed: run.seed });
                }
                Err(code) => {
                    // The client gets an answer even when the helper is going down.
                    let _ = job.events.send(EngineEvent::Failed(EngineFailure::Unavailable));
                    break Err(code);
                }
            }
        };
        self.shared.processing.store(false, Ordering::SeqCst);
        self.shared.progress().state = if self.shared.ready.load(Ordering::SeqCst) { "idle" } else { "loading" };
        result
    }

    fn attempt(&mut self, job: &Job, run: &mut Run) -> Flow<Outcome> {
        let path = super::req_file(&self.cfg.req_prefix, run.id_task, run.attempt);
        let req = EngineRequest { seed: run.seed, ..job.req.clone() };
        // Fully written and closed before its path is sent.
        if let Err(e) = std::fs::write(&path, req.to_json()) {
            dglog!("#{} cannot write the request file {}: {e}", run.id_task, path.display());
            return self.fail(job, EngineFailure::BadReqFile);
        }
        self.live_req = Some(path.clone());
        let mark = self.facts.len();
        self.set_prefill(run.id_task, job.req.n_blocks);
        let sent = match self.child.as_mut() {
            Some(c) => c.send_line(&path.to_string_lossy()),
            None => Err(io::ErrorKind::BrokenPipe.into()),
        };
        if sent.is_err() {
            return self.crash_during_job(job, run, mark, false);
        }

        let mut st = AttemptState::default();
        let mut last_output = Instant::now();
        loop {
            match self.next(Some(last_output + self.cfg.watchdog), false) {
                Next::Line(l) => {
                    last_output = Instant::now();
                    match l {
                        Line::Frame { block, step, total } => {
                            st.frames += 1;
                            let mut p = self.shared.progress();
                            p.state = "denoise";
                            p.block = block;
                            p.step = step;
                            p.total = total;
                        }
                        Line::Commit { block, text } => {
                            // A block with no F frame means its step-0 decode
                            // failed and the canvas is stale — possibly from the
                            // previous conversation (diffusion.cpp:561-563, 682).
                            if st.frames == 0 || st.stale {
                                if !st.stale {
                                    dglog!("#{} block {block} committed without a denoise step: stale canvas", run.id_task);
                                }
                                st.stale = true;
                            } else {
                                let _ = job.events.send(EngineEvent::Commit { block, text });
                                st.commit_sent = true;
                            }
                            st.frames = 0;
                        }
                        Line::Stats(s) => st.stats = Some(s),
                        Line::Err(e) => match e {
                            ErrLine::TooLong { needed, budget } => st.toolong = Some((needed, budget)),
                            ErrLine::Gen => st.err_gen = true,
                            ErrLine::Unknown(m) => {
                                self.await_done(ERR_UNKNOWN_WAIT);
                                self.drop_req();
                                return self.fail(job, EngineFailure::Runner(m));
                            }
                            terminal => {
                                self.quiet(ERR_QUIET);
                                self.drop_req();
                                let f = match terminal {
                                    ErrLine::Parse(m) => EngineFailure::Parse(m),
                                    ErrLine::EmptyPrompt => EngineFailure::EmptyPrompt,
                                    _ => EngineFailure::BadReqFile,
                                };
                                return self.fail(job, f);
                            }
                        },
                        Line::Done => return self.at_done(job, run, mark, st),
                        Line::Ready { .. } | Line::Other => {}
                    }
                }
                Next::Eof => return self.crash_during_job(job, run, mark, st.commit_sent),
                Next::Timeout => return self.watchdog(job, run),
                Next::Job => {}
                Next::Shutdown => {
                    self.drop_req();
                    return Err(0);
                }
            }
        }
    }

    fn set_prefill(&self, id_task: u64, n_blocks: u32) {
        let mut p = self.shared.progress();
        p.id_task = id_task;
        p.state = "prefill";
        p.block = 0;
        p.step = 0;
        p.total = 0;
        p.n_blocks = n_blocks;
    }

    fn fail(&mut self, job: &Job, f: EngineFailure) -> Flow<Outcome> {
        let _ = job.events.send(EngineEvent::Failed(f));
        Ok(Outcome::Finished)
    }

    fn at_done(&mut self, job: &Job, run: &mut Run, mark: usize, st: AttemptState) -> Flow<Outcome> {
        self.settle(STDERR_CATCHUP);
        self.drop_req();
        let step_failed = st.stale || self.facts.get(mark..).is_some_and(|f| f.contains(&Fact::StepFailure));
        if step_failed {
            // Same runner, no respawn: the runner itself is fine.
            let can_retry = run.retry_left && (!job.stream || !st.commit_sent);
            dglog!(
                "#{} attempt {}: a denoise step failed{}",
                run.id_task,
                run.attempt,
                if can_retry { "; retrying on the same runner" } else { "" }
            );
            if can_retry {
                run.retry_left = false;
                return Ok(Outcome::Retry);
            }
            return self.fail(job, EngineFailure::StepFailed);
        }
        // The runner answered, so it is healthy whatever became of the request.
        self.crash_streak = 0;
        if st.err_gen && st.stats.is_none() {
            return self.fail(job, EngineFailure::Gen);
        }
        if let Some((needed, budget)) = st.toolong {
            let _ = job.events.send(EngineEvent::TooLong { needed, budget, after_commit: st.commit_sent });
        }
        if let Some(s) = st.stats {
            self.record(&s);
            let _ = job.events.send(EngineEvent::Stats(s));
        }
        let _ = job.events.send(EngineEvent::Done);
        Ok(Outcome::Finished)
    }

    fn record(&mut self, s: &Stats) {
        {
            let mut m = self.shared.metrics();
            m.prompt_tokens_total += s.prompt_n;
            m.tokens_predicted_total += s.predicted_n;
            m.predicted_seconds_total += s.wall_ms / 1000.0;
            m.n_decode_total += s.steps as u64;
            m.blocks_total += s.blocks as u64;
            m.last_predicted_tps = if s.wall_ms > 0.0 { s.predicted_n as f64 / (s.wall_ms / 1000.0) } else { 0.0 };
        }
        self.shared.progress().n_prompt = s.prompt_n;
        // READY without MAXTOK: the first STATS says what the budget is.
        if self.shared.maxtok.load(Ordering::SeqCst) == 0 && s.n_ctx > 0 {
            self.shared.maxtok.store(s.n_ctx, Ordering::SeqCst);
            self.pinned_maxtok.get_or_insert(s.n_ctx);
        }
    }

    fn crash_during_job(&mut self, job: &Job, run: &mut Run, mark: usize, commit_sent: bool) -> Flow<Outcome> {
        let code = self.reap(EXIT_WAIT)?;
        self.drop_req();
        self.crash_streak += 1;
        let hint = self
            .facts
            .get(mark..)
            .unwrap_or(&[])
            .iter()
            .find_map(|f| match f {
                Fact::CrashHint(h) => Some(*h),
                _ => None,
            })
            .or_else(|| silent_fastfail(code, &self.last_tail).then_some("utf8"));
        // A streamed commit cannot be taken back; a pinned seed would replay
        // a seed-dependent (utf8) crash; an OOM repeats whatever the seed.
        let retry = run.retry_left
            && !(job.stream && commit_sent)
            && match hint {
                Some("oom") => false,
                Some("utf8") => !job.seed_pinned,
                _ => true,
            };
        let tripped = self.crash_streak >= BREAKER;
        self.log_death(&death_line(run, code, hint, retry && !tripped));
        let failure = match hint {
            Some("oom") => EngineFailure::Oom,
            _ => EngineFailure::Crashed { exit: code, detail: crash_detail(&self.last_tail) },
        };
        if tripped {
            let _ = job.events.send(EngineEvent::Failed(failure));
            dglog!("circuit breaker: {BREAKER} runner deaths without a successful request; giving up");
            return Err(EXIT_BREAKER);
        }
        if !retry {
            // Answer now; the reload takes a while.
            let _ = job.events.send(EngineEvent::Failed(failure));
        }
        self.boot(false)?;
        if retry {
            run.retry_left = false;
            Ok(Outcome::Retry)
        } else {
            Ok(Outcome::Finished)
        }
    }

    fn watchdog(&mut self, job: &Job, run: &Run) -> Flow<Outcome> {
        let secs = self.cfg.watchdog.as_millis().div_ceil(1000) as u64;
        dglog!("#{} no runner output for {secs} s; killing and restarting it", run.id_task);
        self.reap(Duration::ZERO)?;
        self.drop_req();
        self.crash_streak += 1;
        let _ = job.events.send(EngineEvent::Failed(EngineFailure::Watchdog(secs)));
        if self.crash_streak >= BREAKER {
            dglog!("circuit breaker: {BREAKER} runner deaths without a successful request; giving up");
            return Err(EXIT_BREAKER);
        }
        self.boot(false)?;
        Ok(Outcome::Finished)
    }
}

// ------------------------------------------------------------ real runner ----

pub struct RealSpawner {
    pub runner: PathBuf,
    pub model: PathBuf,
    pub job: job::KillOnCloseJob,
}

impl Spawner for RealSpawner {
    fn spawn(
        &mut self,
        gen: u64,
        maxtok_override: Option<u32>,
        tx: Sender<EngineMsg>,
    ) -> io::Result<Box<dyn EngineChild>> {
        let mut cmd = Command::new(&self.runner);
        cmd.arg(&self.model).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        for k in SCRUBBED_ENV {
            cmd.env_remove(k);
        }
        if let Some(m) = maxtok_override {
            cmd.env("MAXTOK", m.to_string());
        }
        crate::launch::hide_console(&mut cmd);
        // The helper's own stdio is the run log; the runner must not hold it.
        crate::supervise::stop_inheriting_std_handles();
        let mut child = cmd.spawn()?;
        if let Err(e) = self.job.assign(&child) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(io::Error::new(e.kind(), format!("cannot put the runner in the kill-on-close job: {e}")));
        }
        let (Some(stdout), Some(stderr)) = (child.stdout.take(), child.stderr.take()) else {
            let _ = child.kill();
            return Err(io::Error::other("runner pipes missing"));
        };
        let stdin = child.stdin.take();
        let tail = Arc::new(Mutex::new(VecDeque::new()));
        let drained = Arc::new(AtomicBool::new(false));

        let out_tx = tx.clone();
        let started = std::thread::Builder::new().name("dg-stdout".into()).spawn(move || {
            let mut r = BufReader::new(stdout);
            let mut buf = Vec::new();
            loop {
                buf.clear();
                match r.read_until(b'\n', &mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        while matches!(buf.last(), Some(b'\n' | b'\r')) {
                            buf.pop();
                        }
                        let line = std::mem::take(&mut buf);
                        if out_tx.send(EngineMsg::Child { gen, out: ChildOut::Line(line) }).is_err() {
                            return;
                        }
                    }
                }
            }
            let _ = out_tx.send(EngineMsg::Child { gen, out: ChildOut::Eof });
        });
        let forward = {
            let (tail, drained) = (tail.clone(), drained.clone());
            started.and_then(|_| {
                std::thread::Builder::new()
                    .name("dg-stderr".into())
                    .spawn(move || forward_stderr(stderr, gen, tx, tail, drained))
            })
        };
        if let Err(e) = forward {
            let _ = child.kill();
            return Err(e);
        }
        Ok(Box::new(RealChild { child, stdin, tail, drained }))
    }
}

/// Collapses runs of identical lines: the runner logs "cannot decode
/// batches with this context (calling encode() instead)" on every step.
#[derive(Debug, Default)]
pub(crate) struct Collapser {
    last: Option<String>,
    repeats: u32,
}

impl Collapser {
    /// The lines to write for `line`: the first of a run at once, the rest
    /// counted and summarised when the run ends.
    pub(crate) fn push(&mut self, line: &str) -> Vec<String> {
        if self.last.as_deref() == Some(line) {
            self.repeats += 1;
            return Vec::new();
        }
        let mut out = Vec::new();
        out.extend(self.finish());
        out.push(line.to_string());
        self.last = Some(line.to_string());
        out
    }

    /// ASCII on purpose: Windows PowerShell 5.1 reads the log in the ANSI
    /// code page and would garble an ellipsis or a multiplication sign.
    pub(crate) fn finish(&mut self) -> Option<String> {
        let n = std::mem::take(&mut self.repeats);
        (n > 0).then(|| format!("...(repeated {n}x)"))
    }
}

/// Per-step noise that would push the useful lines out of the crash tail.
fn tail_worthy(line: &str) -> bool {
    !line.trim().is_empty() && !line.contains("cannot decode batches") && !line.contains("set_causal_attn")
}

fn forward_stderr(
    stderr: std::process::ChildStderr,
    gen: u64,
    tx: Sender<EngineMsg>,
    tail: Arc<Mutex<VecDeque<String>>>,
    drained: Arc<AtomicBool>,
) {
    let mut r = BufReader::new(stderr);
    let mut buf = Vec::new();
    let mut collapse = Collapser::default();
    loop {
        buf.clear();
        match r.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&buf).trim_end_matches(['\r', '\n']).to_string();
        // Keep reading whatever happens to the log: a runner blocked on a
        // full stderr pipe stops generating.
        for out in collapse.push(&line) {
            super::stderr_line(&out);
        }
        if tail_worthy(&line) {
            let mut t = lock(&tail);
            if t.len() == TAIL_LINES {
                t.pop_front();
            }
            t.push_back(line.clone());
        }
        if let Some(f) = protocol::parse_stderr_fact(&line) {
            let _ = tx.send(EngineMsg::Child { gen, out: ChildOut::Fact(f) });
        }
    }
    if let Some(out) = collapse.finish() {
        super::stderr_line(&out);
    }
    drained.store(true, Ordering::SeqCst);
}

struct RealChild {
    child: Child,
    stdin: Option<ChildStdin>,
    tail: Arc<Mutex<VecDeque<String>>>,
    drained: Arc<AtomicBool>,
}

impl EngineChild for RealChild {
    fn send_line(&mut self, l: &str) -> io::Result<()> {
        let stdin = self.stdin.as_mut().ok_or(io::ErrorKind::BrokenPipe)?;
        stdin.write_all(format!("{l}\n").as_bytes())?;
        stdin.flush()
    }

    fn pid(&self) -> Option<u32> {
        Some(self.child.id())
    }

    fn kill_and_wait(&mut self, timeout: Duration) -> Option<i32> {
        let _ = self.child.kill();
        self.stdin = None;
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(c) = self.try_exit_code() {
                return Some(c);
            }
            if Instant::now() >= deadline {
                return None;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn try_exit_code(&mut self) -> Option<i32> {
        self.child.try_wait().ok().flatten().map(|s| s.code().unwrap_or(-1))
    }

    fn stderr_tail(&self) -> Vec<String> {
        lock(&self.tail).iter().cloned().collect()
    }

    fn stderr_drained(&self) -> bool {
        self.drained.load(Ordering::SeqCst)
    }
}

impl Drop for RealChild {
    fn drop(&mut self) {
        if self.try_exit_code().is_none() {
            let _ = self.child.kill();
        }
    }
}

// -------------------------------------------------------------- test double ----

#[cfg(test)]
pub(crate) mod fake {
    //! Runners as threads driven by a script: the script sees each request
    //! file the engine sends and answers with stdout lines, stderr lines
    //! (turned into facts), EOF or silence.

    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::Duration;

    use serde_json::Value;

    use super::{ChildOut, EngineChild, EngineMsg, Spawner};
    use crate::diffusion::protocol::parse_stderr_fact;

    pub(crate) type Script = Arc<dyn Fn(FakeCtx) + Send + Sync>;

    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct SpawnRecord {
        pub gen: u64,
        pub maxtok_override: Option<u32>,
    }

    pub(crate) struct FakeSpawner {
        script: Script,
        spawns: Arc<Mutex<Vec<SpawnRecord>>>,
    }

    impl FakeSpawner {
        pub(crate) fn new(script: impl Fn(FakeCtx) + Send + Sync + 'static) -> (Self, Arc<Mutex<Vec<SpawnRecord>>>) {
            let spawns = Arc::new(Mutex::new(Vec::new()));
            (FakeSpawner { script: Arc::new(script), spawns: spawns.clone() }, spawns)
        }
    }

    #[derive(Default)]
    struct FakeState {
        exited: AtomicBool,
        eof_sent: AtomicBool,
        killed: AtomicBool,
        code: AtomicI32,
        tail: Mutex<Vec<String>>,
    }

    pub(crate) struct FakeReq {
        pub path: PathBuf,
        pub body: Value,
    }

    pub(crate) struct FakeCtx {
        pub gen: u64,
        /// 0 for the first runner, 1 for the first respawn, …
        pub index: usize,
        tx: Sender<EngineMsg>,
        stdin: Receiver<String>,
        state: Arc<FakeState>,
    }

    impl FakeCtx {
        pub fn line(&self, s: &str) {
            let _ = self.tx.send(EngineMsg::Child { gen: self.gen, out: ChildOut::Line(s.as_bytes().to_vec()) });
        }

        pub fn stderr(&self, s: &str) {
            self.state.tail.lock().unwrap().push(s.to_string());
            if let Some(f) = parse_stderr_fact(s) {
                let _ = self.tx.send(EngineMsg::Child { gen: self.gen, out: ChildOut::Fact(f) });
            }
        }

        /// The process exits with `code`: stdout closes.
        pub fn eof(&self, code: i32) {
            self.state.code.store(code, Ordering::SeqCst);
            self.state.exited.store(true, Ordering::SeqCst);
            if !self.state.eof_sent.swap(true, Ordering::SeqCst) {
                let _ = self.tx.send(EngineMsg::Child { gen: self.gen, out: ChildOut::Eof });
            }
        }

        /// The device facts of a healthy single-card load, then READY.
        pub fn ready(&self, maxtok: u32) {
            self.device_facts(8, true, 31, 31);
            self.line(&format!("READY 262144 {maxtok}"));
        }

        pub fn device_facts(&self, bus: u32, kv_on: bool, done: u32, total: u32) {
            self.stderr(&format!(
                "llama_prepare_model_devices: using device ROCm0 (AMD Radeon AI PRO R9700) (0000:{bus:02x}:00.0) - 32472 MiB free"
            ));
            self.stderr(&format!("load_tensors: offloaded {done}/{total} layers to GPU"));
            self.stderr(&format!(
                "diffusion-gemma-visual-server ready (n_vocab=262144, canvas=256, MAXTOK=12288, NGL=99, gpu_sampling=on sample_reduce=on kv_cache={})",
                if kv_on { "on" } else { "off" }
            ));
        }

        /// Wait for the next request path; None on QUIT or when killed.
        pub fn next_request(&self) -> Option<FakeReq> {
            match self.stdin.recv() {
                Ok(l) if l == "QUIT" => None,
                Ok(l) => {
                    let path = PathBuf::from(&l);
                    let body = std::fs::read_to_string(&path)
                        .ok()
                        .and_then(|t| serde_json::from_str(&t).ok())
                        .unwrap_or(Value::Null);
                    Some(FakeReq { path, body })
                }
                Err(_) => None,
            }
        }

        /// A well-formed one-block answer.
        pub fn reply(&self, text: &str) {
            self.line("F 0 0 48 \"x\"");
            self.line(&format!("C 0 {}", serde_json::to_string(text).unwrap()));
            self.stats(5, text.len() as u64, 1);
            self.line("DONE");
        }

        pub fn stats(&self, prompt_n: u64, predicted_n: u64, blocks: u32) {
            self.line(&format!(
                "STATS prompt_n={prompt_n} predicted_n={predicted_n} prompt_prepare_ms=1.0 wall_ms=500.0 decode_ms=450.0 blocks={blocks} steps={} canvas=256 n_ctx=12288",
                blocks * 16
            ));
        }

        /// Serve `reply(text(n))` for every request until QUIT or kill.
        pub fn serve(&self, text: impl Fn(usize) -> String) {
            let mut n = 0;
            while self.next_request().is_some() {
                self.reply(&text(n));
                n += 1;
            }
        }

        /// Block until QUIT or kill, answering nothing.
        pub fn hang(&self) {
            while let Ok(l) = self.stdin.recv() {
                if l == "QUIT" {
                    break;
                }
            }
        }
    }

    struct FakeChild {
        stdin: Option<Sender<String>>,
        state: Arc<FakeState>,
        pid: u32,
    }

    impl EngineChild for FakeChild {
        fn send_line(&mut self, l: &str) -> std::io::Result<()> {
            if self.state.exited.load(Ordering::SeqCst) {
                return Err(std::io::ErrorKind::BrokenPipe.into());
            }
            match &self.stdin {
                Some(s) => s.send(l.to_string()).map_err(|_| std::io::ErrorKind::BrokenPipe.into()),
                None => Err(std::io::ErrorKind::BrokenPipe.into()),
            }
        }
        fn pid(&self) -> Option<u32> {
            Some(self.pid)
        }
        fn kill_and_wait(&mut self, _timeout: Duration) -> Option<i32> {
            self.state.killed.store(true, Ordering::SeqCst);
            self.stdin = None;
            if !self.state.exited.swap(true, Ordering::SeqCst) {
                self.state.code.store(1, Ordering::SeqCst);
            }
            Some(self.state.code.load(Ordering::SeqCst))
        }
        fn try_exit_code(&mut self) -> Option<i32> {
            self.state.exited.load(Ordering::SeqCst).then(|| self.state.code.load(Ordering::SeqCst))
        }
        fn stderr_tail(&self) -> Vec<String> {
            self.state.tail.lock().unwrap().clone()
        }
    }

    impl Spawner for FakeSpawner {
        fn spawn(
            &mut self,
            gen: u64,
            maxtok_override: Option<u32>,
            tx: Sender<EngineMsg>,
        ) -> std::io::Result<Box<dyn EngineChild>> {
            let index = {
                let mut s = self.spawns.lock().unwrap();
                s.push(SpawnRecord { gen, maxtok_override });
                s.len() - 1
            };
            let (stdin_tx, stdin_rx) = mpsc::channel();
            let state = Arc::new(FakeState::default());
            let ctx = FakeCtx { gen, index, tx: tx.clone(), stdin: stdin_rx, state: state.clone() };
            let script = self.script.clone();
            let st = state.clone();
            std::thread::spawn(move || {
                script(ctx);
                // Returning from the script is the process exiting.
                if !st.killed.load(Ordering::SeqCst) && !st.eof_sent.swap(true, Ordering::SeqCst) {
                    st.exited.store(true, Ordering::SeqCst);
                    let _ = tx.send(EngineMsg::Child { gen, out: ChildOut::Eof });
                }
            });
            Ok(Box::new(FakeChild { stdin: Some(stdin_tx), state, pid: 40_000 + index as u32 }))
        }
    }

    /// A counting semaphore so tests can step a fake runner deterministically.
    #[derive(Clone, Default)]
    pub(crate) struct Gate(Arc<(Mutex<u32>, Condvar)>);

    impl Gate {
        pub(crate) fn open(&self) {
            let (m, c) = &*self.0;
            *m.lock().unwrap() += 1;
            c.notify_all();
        }

        /// Take one permit (gives up after 20 s so a broken test fails
        /// instead of hanging).
        pub(crate) fn wait(&self) {
            let (m, c) = &*self.0;
            let mut n = m.lock().unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            while *n == 0 {
                let left = deadline.saturating_duration_since(std::time::Instant::now());
                if left.is_zero() {
                    return;
                }
                n = c.wait_timeout(n, left).unwrap().0;
            }
            *n -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{FakeCtx, FakeSpawner, Gate, SpawnRecord};
    use super::*;
    use crate::diffusion::{ModelInfo, ServeConfig, Shared};
    use serde_json::json;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;

    fn test_dir(name: &str) -> PathBuf {
        static N: AtomicUsize = AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!(
            "fidim-dg-{}-{name}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn cfg(dir: &std::path::Path) -> ServeConfig {
        ServeConfig {
            runner: "fake-runner.exe".into(),
            model: "fake.gguf".into(),
            host: "127.0.0.1".into(),
            port: 0,
            alias: "dg-test".into(),
            req_prefix: dir.join("dg-test-0"),
            default_max_tokens: 2048,
            seed: None,
            expect_bus: Some(8),
            build_tag: Some("b1-test".into()),
            queue_depth: 8,
            load_timeout: Duration::from_secs(5),
            watchdog: Duration::from_secs(5),
            ngl: 99,
            maxtok_env: 0,
        }
    }

    fn info() -> ModelInfo {
        ModelInfo { canvas: 256, block_count: 30, n_ctx_train: 262_144, vocab: 262_144 }
    }

    struct Harness {
        shared: Arc<Shared>,
        tx: Sender<EngineMsg>,
        done: Receiver<i32>,
        spawns: Arc<Mutex<Vec<SpawnRecord>>>,
        dir: PathBuf,
    }

    fn harness(tweak: impl FnOnce(&mut ServeConfig), script: impl Fn(FakeCtx) + Send + Sync + 'static) -> Harness {
        let dir = test_dir("engine");
        let mut c = cfg(&dir);
        tweak(&mut c);
        let shared = Arc::new(Shared::new());
        let (tx, rx) = mpsc::channel();
        let (spawner, spawns) = FakeSpawner::new(script);
        let (done_tx, done) = mpsc::channel();
        {
            let (shared, tx) = (shared.clone(), tx.clone());
            std::thread::spawn(move || {
                let code = run_engine(&c, &info(), shared, Box::new(spawner), rx, tx);
                let _ = done_tx.send(code);
            });
        }
        Harness { shared, tx, done, spawns, dir }
    }

    impl Harness {
        fn submit(&self, seed: i32, stream: bool, seed_pinned: bool) -> Receiver<EngineEvent> {
            let (etx, erx) = mpsc::channel();
            self.shared.queued.fetch_add(1, Ordering::SeqCst);
            let req = EngineRequest {
                seed,
                n_blocks: 2,
                messages: json!([{"role": "user", "content": "hi"}]),
                tools: None,
            };
            let job = Job { req, seed_pinned, stream, events: etx, cancelled: Arc::new(AtomicBool::new(false)) };
            self.tx.send(EngineMsg::Job(job)).unwrap();
            erx
        }

        fn exit_code(&self) -> i32 {
            self.done.recv_timeout(Duration::from_secs(20)).expect("engine did not exit")
        }

        fn stop(&self) -> i32 {
            let _ = self.tx.send(EngineMsg::Shutdown);
            self.exit_code()
        }

        fn spawns(&self) -> Vec<SpawnRecord> {
            self.spawns.lock().unwrap().clone()
        }

        fn wait_ready(&self) {
            let deadline = Instant::now() + Duration::from_secs(10);
            while !self.shared.ready.load(Ordering::SeqCst) {
                assert!(Instant::now() < deadline, "engine never became ready");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            let _ = self.tx.send(EngineMsg::Shutdown);
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// Events up to and including the terminal one.
    fn collect(rx: &Receiver<EngineEvent>) -> Vec<EngineEvent> {
        let mut out = Vec::new();
        loop {
            let ev = rx.recv_timeout(Duration::from_secs(20)).expect("job never finished");
            let end = matches!(ev, EngineEvent::Done | EngineEvent::Failed(_));
            out.push(ev);
            if end {
                return out;
            }
        }
    }

    fn commits(evs: &[EngineEvent]) -> Vec<String> {
        evs.iter()
            .filter_map(|e| match e {
                EngineEvent::Commit { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    // ------------------------------------------------------ ready_and_guard ----

    #[test]
    fn ready_and_guard_passes_on_one_rocm_device() {
        let h = harness(|_| {}, |cx| {
            cx.ready(12288);
            cx.hang();
        });
        h.wait_ready();
        assert!(h.shared.ever_ready.load(Ordering::SeqCst));
        assert_eq!(h.shared.maxtok.load(Ordering::SeqCst), 12288);
        assert_eq!(h.shared.child_pid.load(Ordering::SeqCst), 40_000);
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn ready_and_guard_two_devices_exit_4() {
        let h = harness(|_| {}, |cx| {
            cx.stderr("f: using device ROCm1 (AMD Radeon AI PRO R9700) (0000:03:00.0) - 32472 MiB free");
            cx.ready(12288);
            cx.hang();
        });
        assert_eq!(h.exit_code(), EXIT_GUARD);
        assert!(!h.shared.ever_ready.load(Ordering::SeqCst));
    }

    #[test]
    fn ready_and_guard_wrong_bus_exit_5() {
        let h = harness(|_| {}, |cx| {
            cx.device_facts(3, true, 31, 31);
            cx.line("READY 262144 12288");
            cx.hang();
        });
        assert_eq!(h.exit_code(), EXIT_BUS);
    }

    #[test]
    fn ready_and_guard_kv_off_exit_4() {
        let h = harness(|_| {}, |cx| {
            cx.device_facts(8, false, 31, 31);
            cx.line("READY 262144 12288");
            cx.hang();
        });
        assert_eq!(h.exit_code(), EXIT_GUARD);
    }

    #[test]
    fn ready_and_guard_no_device_facts_exit_4() {
        let h = harness(|_| {}, |cx| {
            cx.stderr("diffusion-gemma-visual-server ready (n_vocab=262144, canvas=256, MAXTOK=12288, NGL=99, kv_cache=on)");
            cx.line("READY 262144 12288");
            cx.hang();
        });
        assert_eq!(h.exit_code(), EXIT_GUARD);
    }

    #[test]
    fn ready_and_guard_eof_before_ready_exit_3() {
        let h = harness(|_| {}, |cx| {
            cx.stderr("failed to load model");
            cx.eof(1);
        });
        assert_eq!(h.exit_code(), EXIT_LOAD);
        assert_eq!(h.spawns().len(), 1, "the first load is never retried");
    }

    #[test]
    fn ready_and_guard_skipped_at_ngl_0() {
        let h = harness(|c| c.ngl = 0, |cx| {
            cx.line("READY 262144 4096");
            cx.hang();
        });
        h.wait_ready();
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn guard_rules_pure() {
        let dev = |bus| Fact::UsingDevice { name: "ROCm0".into(), bus: Some(bus) };
        let ready = Fact::RunnerReady { ngl: Some(99), kv_cache_on: true };
        let full = Fact::Offloaded { done: 31, total: 31 };
        assert!(evaluate_guard(&[dev(8), full.clone(), ready.clone()], 99, Some(8), 30).is_ok());
        assert!(evaluate_guard(&[dev(8), full.clone(), ready.clone()], 99, None, 30).is_ok());
        // Partial offload while NGL asks for everything.
        let partial = Fact::Offloaded { done: 29, total: 31 };
        assert_eq!(evaluate_guard(&[dev(8), partial.clone(), ready.clone()], 99, Some(8), 30).unwrap_err().0, 4);
        // A real partial offload (NGL 20 of 31) does not need the full line.
        let r20 = Fact::RunnerReady { ngl: Some(20), kv_cache_on: true };
        assert!(evaluate_guard(&[dev(8), partial, r20], 20, Some(8), 30).is_ok());
        // NGL mismatch.
        assert_eq!(evaluate_guard(&[dev(8), full.clone(), ready.clone()], 50, Some(8), 30).unwrap_err().0, 4);
        // Not ROCm.
        let vk = Fact::UsingDevice { name: "Vulkan0".into(), bus: Some(8) };
        assert_eq!(evaluate_guard(&[vk, full.clone(), ready.clone()], 99, Some(8), 30).unwrap_err().0, 4);
        // Unknown bus with an expectation is a different card as far as we know.
        let nobus = Fact::UsingDevice { name: "ROCm0".into(), bus: None };
        assert_eq!(evaluate_guard(&[nobus, full, ready], 99, Some(8), 30).unwrap_err().0, 5);
        // Missing ready line.
        assert_eq!(evaluate_guard(&[dev(8)], 99, Some(8), 30).unwrap_err().0, 4);
    }

    // ------------------------------------------------------------ protocol ----

    #[test]
    fn toolong_then_clean() {
        let h = harness(|_| {}, |cx| {
            cx.ready(12288);
            let mut n = 0;
            while cx.next_request().is_some() {
                if n == 0 {
                    cx.line("F 0 0 48 \"x\"");
                    cx.line("C 0 \"partial answer\"");
                    cx.line("ERR toolong 12544 12288");
                    cx.stats(12000, 256, 1);
                    cx.line("DONE");
                } else {
                    cx.reply("second");
                }
                n += 1;
            }
        });
        let a = collect(&h.submit(1, false, false));
        assert!(matches!(a[0], EngineEvent::Started { id_task: 1, .. }));
        assert_eq!(commits(&a), vec!["partial answer".to_string()]);
        assert!(a.contains(&EngineEvent::TooLong { needed: 12544, budget: 12288, after_commit: true }));
        assert!(a.iter().any(|e| matches!(e, EngineEvent::Stats(s) if s.prompt_n == 12000)));
        assert_eq!(a.last(), Some(&EngineEvent::Done));
        let b = collect(&h.submit(1, false, false));
        assert_eq!(commits(&b), vec!["second".to_string()]);
        assert!(!b.iter().any(|e| matches!(e, EngineEvent::TooLong { .. })));
        assert_eq!(b.last(), Some(&EngineEvent::Done));
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn toolong_on_block_0_has_no_commit() {
        let h = harness(|_| {}, |cx| {
            cx.ready(12288);
            while cx.next_request().is_some() {
                cx.line("ERR toolong 12544 12288");
                cx.line("DONE");
            }
        });
        let a = collect(&h.submit(1, true, false));
        assert!(a.contains(&EngineEvent::TooLong { needed: 12544, budget: 12288, after_commit: false }));
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn err_parse_multiline_terminal() {
        let h = harness(|_| {}, |cx| {
            cx.ready(12288);
            let mut n = 0;
            while cx.next_request().is_some() {
                if n == 0 {
                    // chat.cpp dumps the tools JSON after the message; no DONE.
                    cx.line("ERR parse Failed to parse tools: Missing tool type; tools = [");
                    cx.line("  {");
                    cx.line("    \"function\": {}");
                    cx.line("  }");
                    cx.line("]");
                } else {
                    cx.reply("fine");
                }
                n += 1;
            }
        });
        let a = collect(&h.submit(1, false, false));
        match a.last() {
            Some(EngineEvent::Failed(EngineFailure::Parse(m))) => assert!(m.starts_with("Failed to parse tools")),
            other => panic!("{other:?}"),
        }
        let b = collect(&h.submit(1, false, false));
        assert_eq!(commits(&b), vec!["fine".to_string()]);
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn stale_canvas_is_retried_on_the_same_runner() {
        let h = harness(|_| {}, |cx| {
            cx.ready(12288);
            let mut n = 0;
            while cx.next_request().is_some() {
                if n == 0 {
                    // Step 0 failed: no F frame, a stale canvas committed.
                    cx.line("C 0 \"answer to the previous conversation\"");
                    cx.stats(5, 256, 1);
                    cx.line("DONE");
                } else {
                    cx.reply("fresh");
                }
                n += 1;
            }
        });
        let a = collect(&h.submit(10, false, false));
        assert!(a.contains(&EngineEvent::Restarted { seed: 11 }));
        assert_eq!(commits(&a), vec!["fresh".to_string()]);
        assert_eq!(a.last(), Some(&EngineEvent::Done));
        assert_eq!(h.spawns().len(), 1, "a step failure never respawns");
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn stepfailure_fact_at_done_retries_once() {
        let h = harness(|_| {}, |cx| {
            cx.ready(12288);
            let mut n = 0;
            while cx.next_request().is_some() {
                if n < 3 {
                    cx.line("F 0 0 48 \"x\"");
                    cx.stderr("diffusion_generate: failed to decode at step 3");
                    cx.line("C 0 \"garbled\"");
                    cx.stats(5, 256, 1);
                    cx.line("DONE");
                } else {
                    cx.reply("ok");
                }
                n += 1;
            }
        });
        // Fails twice: one retry, then StepFailed.
        let a = collect(&h.submit(1, false, false));
        assert_eq!(a.iter().filter(|e| matches!(e, EngineEvent::Restarted { .. })).count(), 1);
        assert_eq!(a.last(), Some(&EngineEvent::Failed(EngineFailure::StepFailed)));
        // Third failure, then success on the retry.
        let b = collect(&h.submit(1, false, false));
        assert_eq!(b.last(), Some(&EngineEvent::Done));
        assert!(b.iter().any(|e| matches!(e, EngineEvent::Restarted { .. })));
        assert_eq!(h.spawns().len(), 1);
        assert_eq!(h.stop(), 0);
    }

    // ---------------------------------------------------------- crash rules ----

    #[test]
    fn crash_before_commit_restarts_and_retries_with_seed_plus_1() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        let h = harness(|_| {}, move |cx| {
            cx.ready(12288);
            while let Some(r) = cx.next_request() {
                s2.lock().unwrap().push(r.body["seed"].as_i64().unwrap());
                if cx.index == 0 {
                    cx.stderr("ROCm error: unspecified launch failure");
                    cx.eof(-1073740791);
                    return;
                }
                cx.reply("after restart");
            }
        });
        let a = collect(&h.submit(100, false, false));
        assert!(a.contains(&EngineEvent::Restarted { seed: 101 }));
        assert_eq!(commits(&a), vec!["after restart".to_string()]);
        assert_eq!(*seen.lock().unwrap(), vec![100, 101]);
        assert_eq!(h.spawns().len(), 2);
        assert_eq!(h.shared.restarts.load(Ordering::SeqCst), 1);
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn stream_crash_after_commit_is_not_retried() {
        let h = harness(|_| {}, |cx| {
            cx.ready(12288);
            while cx.next_request().is_some() {
                if cx.index == 0 {
                    cx.line("F 0 0 48 \"x\"");
                    cx.line("C 0 \"half\"");
                    cx.stderr("D:\\ggml-cuda.cu:97: ROCm error: invalid argument");
                    cx.eof(-1073740791);
                    return;
                }
                cx.reply("next");
            }
        });
        let a = collect(&h.submit(1, true, false));
        assert_eq!(commits(&a), vec!["half".to_string()]);
        assert!(!a.iter().any(|e| matches!(e, EngineEvent::Restarted { .. })));
        match a.last() {
            Some(EngineEvent::Failed(EngineFailure::Crashed { exit, detail })) => {
                assert_eq!(*exit, Some(-1073740791));
                assert!(detail.contains("ROCm error"), "{detail}");
            }
            other => panic!("{other:?}"),
        }
        // The respawned runner serves the next request.
        let b = collect(&h.submit(1, true, false));
        assert_eq!(commits(&b), vec!["next".to_string()]);
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn oom_hint_is_not_retried() {
        let h = harness(|_| {}, |cx| {
            cx.ready(12288);
            while cx.next_request().is_some() {
                if cx.index == 0 {
                    cx.stderr("D:\\src\\models\\diffusion-gemma.cpp:818: GGML_ASSERT(m.pkv_buf != nullptr) failed");
                    cx.eof(-1073740791);
                    return;
                }
                cx.reply("x");
            }
        });
        let a = collect(&h.submit(1, false, false));
        assert!(!a.iter().any(|e| matches!(e, EngineEvent::Restarted { .. })));
        assert_eq!(a.last(), Some(&EngineEvent::Failed(EngineFailure::Oom)));
        let e = crate::diffusion::openai::failure_error(&EngineFailure::Oom);
        assert_eq!(e.kind, "exceed_context_size_error");
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn utf8_crash_with_pinned_seed_is_not_retried() {
        let h = harness(|_| {}, |cx| {
            cx.ready(12288);
            while cx.next_request().is_some() {
                if cx.index == 0 {
                    cx.stderr("terminate called after throwing an instance of 'nlohmann::json_abi_v3_12_0::detail::type_error'");
                    cx.stderr("  what():  [json.exception.type_error.316] invalid UTF-8 byte at index 12: 0xE2");
                    cx.eof(3);
                    return;
                }
                cx.reply("x");
            }
        });
        let a = collect(&h.submit(7, false, true));
        assert!(!a.iter().any(|e| matches!(e, EngineEvent::Restarted { .. })));
        assert!(matches!(a.last(), Some(EngineEvent::Failed(EngineFailure::Crashed { .. }))));
        assert_eq!(h.stop(), 0);
    }

    /// The same crash as the Windows runner really reports it: the MSVC
    /// runtime fastfails with 0xC0000409 and prints nothing at all.
    #[test]
    fn silent_fastfail_with_pinned_seed_is_not_retried() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        let h = harness(|_| {}, move |cx| {
            cx.ready(12288);
            while let Some(r) = cx.next_request() {
                s2.lock().unwrap().push((cx.index, r.body["seed"].as_i64().unwrap()));
                if cx.index == 0 {
                    cx.eof(EXIT_FASTFAIL);
                    return;
                }
                cx.reply("after restart");
            }
        });
        let a = collect(&h.submit(7, false, true));
        assert!(!a.iter().any(|e| matches!(e, EngineEvent::Restarted { .. })), "{a:?}");
        match a.last() {
            Some(EngineEvent::Failed(EngineFailure::Crashed { exit, detail })) => {
                assert_eq!(*exit, Some(EXIT_FASTFAIL));
                assert_eq!(detail, "");
            }
            other => panic!("{other:?}"),
        }
        // The respawn serves the next request; the crashing one was not replayed.
        assert_eq!(commits(&collect(&h.submit(7, false, true))), vec!["after restart".to_string()]);
        assert_eq!(*seen.lock().unwrap(), vec![(0, 7), (1, 7)]);
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn crash_classification_and_log_line() {
        // Silent 0xC0000409 = the uncaught throw; one that printed its reason is not.
        assert!(silent_fastfail(Some(EXIT_FASTFAIL), &[]));
        assert!(silent_fastfail(Some(EXIT_FASTFAIL), &["load_tensors: offloaded 31/31 layers to GPU".into()]));
        assert!(!silent_fastfail(Some(EXIT_FASTFAIL), &["D:\\ggml.c:97: GGML_ASSERT(x) failed".into()]));
        assert!(!silent_fastfail(Some(EXIT_FASTFAIL), &["ROCm error: invalid argument".into()]));
        assert!(!silent_fastfail(Some(-1073741819), &[]));
        assert!(!silent_fastfail(None, &[]));
        // "retrying" only when a restart follows (not when the breaker trips).
        let run = Run { id_task: 2, attempt: 1, seed: 7, retry_left: true };
        assert_eq!(
            death_line(&run, Some(EXIT_FASTFAIL), Some("utf8"), true),
            "runner died during #2 attempt 1 (exit 0xC0000409, utf8); retrying after a restart"
        );
        assert_eq!(death_line(&run, Some(EXIT_FASTFAIL), None, false), "runner died during #2 attempt 1 (exit 0xC0000409)");
    }

    #[test]
    fn three_consecutive_deaths_trip_the_breaker() {
        let h = harness(|_| {}, |cx| {
            if cx.index > 0 {
                // Every respawn dies before READY.
                cx.eof(1);
                return;
            }
            cx.ready(12288);
            if cx.next_request().is_some() {
                cx.eof(-1073740791);
            }
        });
        let a = h.submit(1, false, false);
        assert_eq!(h.exit_code(), EXIT_BREAKER);
        assert!(matches!(collect(&a).last(), Some(EngineEvent::Failed(_))));
        assert_eq!(h.spawns().len(), 3);
    }

    #[test]
    fn idle_crash_respawns_and_reports_health() {
        use crate::diffusion::serve_e2e::{get, start_fake};
        let (die, respawn) = (Gate::default(), Gate::default());
        let (d, r) = (die.clone(), respawn.clone());
        let srv = start_fake(|_| {}, move |cx| {
            if cx.index == 1 {
                r.wait();
            }
            cx.ready(12288);
            if cx.index == 0 {
                d.wait();
                cx.eof(-1073741819);
                return;
            }
            cx.hang();
        });
        srv.wait_status("/health", 200);
        // Die while idle; the respawn then waits on its gate.
        die.open();
        let deadline = Instant::now() + Duration::from_secs(10);
        while srv.spawns().len() < 2 {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(get(srv.port(), "/health").0, 503);
        // /v1/models keeps answering during a respawn.
        assert_eq!(get(srv.port(), "/v1/models").0, 200);
        respawn.open();
        srv.wait_status("/health", 200);
        let (_, body) = get(srv.port(), "/health");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["restarts"], 1);
        assert_eq!(v["child_pid"], 40_001);
        assert_eq!(srv.stop(), 0);
    }

    #[test]
    fn respawn_pins_maxtok() {
        let h = harness(|_| {}, |cx| {
            cx.ready(if cx.index == 0 { 12288 } else { 8192 });
            if cx.index == 0 {
                cx.eof(1);
                return;
            }
            cx.hang();
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while h.spawns().len() < 2 || h.shared.maxtok.load(Ordering::SeqCst) != 8192 {
            assert!(Instant::now() < deadline, "no respawn");
            std::thread::sleep(Duration::from_millis(5));
        }
        let s = h.spawns();
        assert_eq!(s[0].maxtok_override, None);
        assert_eq!(s[1].maxtok_override, Some(12288));
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn watchdog_fails_the_job_and_respawns() {
        let h = harness(|c| c.watchdog = Duration::from_millis(200), |cx| {
            cx.ready(12288);
            if cx.index == 0 {
                // Swallow the request and go silent.
                let _ = cx.next_request();
                cx.hang();
                return;
            }
            cx.serve(|_| "alive".into());
        });
        let a = collect(&h.submit(1, false, false));
        assert_eq!(a.last(), Some(&EngineEvent::Failed(EngineFailure::Watchdog(1))));
        let b = collect(&h.submit(1, false, false));
        assert_eq!(commits(&b), vec!["alive".to_string()]);
        assert_eq!(h.spawns().len(), 2);
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn request_file_lifecycle() {
        let seen: Arc<Mutex<Vec<(PathBuf, bool, serde_json::Value)>>> = Arc::new(Mutex::new(Vec::new()));
        let s2 = seen.clone();
        let h = harness(|_| {}, move |cx| {
            cx.ready(12288);
            while let Some(r) = cx.next_request() {
                let exists = r.path.is_file();
                s2.lock().unwrap().push((r.path.clone(), exists, r.body.clone()));
                if cx.index == 0 {
                    cx.eof(1);
                    return;
                }
                cx.reply("done");
            }
        });
        let a = collect(&h.submit(5, false, false));
        assert_eq!(a.last(), Some(&EngineEvent::Done));
        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 2);
        let (p1, e1, b1) = &seen[0];
        let (p2, e2, b2) = &seen[1];
        assert!(*e1 && *e2, "the file exists when the runner reads it");
        assert_ne!(p1, p2, "every attempt gets its own file");
        assert!(p1.file_name().unwrap().to_string_lossy().ends_with("-1-1.req"));
        assert!(p2.file_name().unwrap().to_string_lossy().ends_with("-1-2.req"));
        assert_eq!(b1["seed"], 5);
        assert_eq!(b2["seed"], 6);
        assert_eq!(b1["n_blocks"], 2);
        assert_eq!(b1["messages"][0]["content"], "hi");
        assert!(!p1.exists(), "deleted after the crash");
        assert!(!p2.exists(), "deleted after DONE");
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn cancelled_queued_jobs_are_skipped() {
        let h = harness(|_| {}, |cx| {
            cx.ready(12288);
            cx.serve(|n| format!("answer {n}"));
        });
        h.wait_ready();
        let (etx, erx) = mpsc::channel();
        h.shared.queued.fetch_add(1, Ordering::SeqCst);
        let job = Job {
            req: EngineRequest { seed: 1, n_blocks: 1, messages: json!([]), tools: None },
            seed_pinned: false,
            stream: false,
            events: etx,
            cancelled: Arc::new(AtomicBool::new(true)),
        };
        h.tx.send(EngineMsg::Job(job)).unwrap();
        let b = collect(&h.submit(1, false, false));
        assert_eq!(commits(&b), vec!["answer 0".to_string()]);
        assert!(erx.try_recv().is_err(), "a cancelled job gets no events");
        assert_eq!(h.shared.queued.load(Ordering::SeqCst), 0);
        assert_eq!(h.stop(), 0);
    }

    #[test]
    fn collapser_summarises_runs() {
        let mut c = Collapser::default();
        assert_eq!(c.push("a"), vec!["a"]);
        assert!(c.push("a").is_empty());
        assert!(c.push("a").is_empty());
        assert_eq!(c.push("b"), vec!["...(repeated 2x)".to_string(), "b".to_string()]);
        assert_eq!(c.finish(), None);
        assert!(c.push("b").is_empty());
        assert_eq!(c.finish(), Some("...(repeated 1x)".to_string()));
        assert!(!tail_worthy("decode: cannot decode batches with this context (calling encode() instead)"));
        assert!(!tail_worthy("set_causal_attn: value = 0"));
        assert!(tail_worthy("failed to load model"));
    }

    #[test]
    fn crash_detail_prefers_error_lines() {
        let tail: Vec<String> = ["load_tensors: ok", "ggml_cuda_compute_forward: MUL_MAT failed", "done"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(crash_detail(&tail), "ggml_cuda_compute_forward: MUL_MAT failed");
        assert_eq!(crash_detail(&[]), "");
    }

    /// RealSpawner against a batch file standing in for the runner: CRLF
    /// stdout, stderr facts, stdout EOF at exit, the exit code, and job
    /// assignment (spawn fails if it does not succeed).
    #[cfg(windows)]
    #[test]
    fn real_pipe() {
        let dir = test_dir("realpipe");
        let runner = dir.join("fake-runner.cmd");
        std::fs::write(
            &runner,
            "@echo off\r\n\
             echo llama_prepare_model_devices: using device ROCm0 (AMD Radeon AI PRO R9700) (0000:08:00.0) - 32472 MiB free 1>&2\r\n\
             echo decode: cannot decode batches with this context (calling encode() instead) 1>&2\r\n\
             echo decode: cannot decode batches with this context (calling encode() instead) 1>&2\r\n\
             echo load_tensors: offloaded 31/31 layers to GPU 1>&2\r\n\
             echo READY 262144 12288\r\n\
             exit /b 3\r\n",
        )
        .unwrap();
        let mut sp = RealSpawner {
            runner: runner.clone(),
            model: dir.join("model.gguf"),
            job: job::KillOnCloseJob::new().expect("job object"),
        };
        let (tx, rx) = mpsc::channel();
        let mut child = sp.spawn(7, Some(12288), tx).expect("spawn + job assignment");
        assert!(child.pid().is_some());
        let mut lines = Vec::new();
        let mut facts = Vec::new();
        loop {
            match rx.recv_timeout(Duration::from_secs(20)).expect("runner output") {
                EngineMsg::Child { gen, out } => {
                    assert_eq!(gen, 7);
                    match out {
                        ChildOut::Line(b) => lines.push(b),
                        ChildOut::Fact(f) => facts.push(f),
                        ChildOut::Eof => break,
                    }
                }
                _ => unreachable!(),
            }
        }
        assert_eq!(lines, vec![b"READY 262144 12288".to_vec()], "CR stripped");
        assert_eq!(protocol::parse_line(&lines[0]), Line::Ready { n_vocab: 262144, maxtok: Some(12288) });
        let deadline = Instant::now() + Duration::from_secs(10);
        let code = loop {
            if let Some(c) = child.try_exit_code() {
                break c;
            }
            assert!(Instant::now() < deadline, "runner never exited");
            std::thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(code, 3);
        let deadline = Instant::now() + Duration::from_secs(10);
        while !child.stderr_drained() {
            assert!(Instant::now() < deadline, "stderr never drained");
            std::thread::sleep(Duration::from_millis(20));
        }
        // Facts may trail the stdout EOF; everything is in by the drain.
        while let Ok(EngineMsg::Child { out: ChildOut::Fact(f), .. }) = rx.try_recv() {
            facts.push(f);
        }
        assert!(facts.contains(&Fact::UsingDevice { name: "ROCm0".into(), bus: Some(8) }));
        assert!(facts.contains(&Fact::Offloaded { done: 31, total: 31 }));
        let tail = child.stderr_tail();
        assert!(tail.iter().any(|l| l.contains("using device ROCm0")));
        assert!(!tail.iter().any(|l| l.contains("cannot decode batches")), "noise stays out of the tail");
        assert_eq!(child.kill_and_wait(Duration::from_secs(5)), Some(3));
        drop(child);
        let _ = std::fs::remove_dir_all(dir);
    }
}
