//! Minimal HTTP/1.1 over std::net for the diffusion shim (shim spec §6-§9),
//! plus the endpoints themselves.
//!
//! One thread per connection, every response `Connection: close`. JSON
//! bodies carry a Content-Length; SSE streams are chunked. The read-only
//! endpoints only read `Shared` and never wait on the engine.

use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::engine::{EngineEvent, EngineMsg, Job};
use super::openai::{
    self, chat_chunk, chat_completion, delta_json, exceed_context, failure_error, finish_reason, shape_final,
    timings, usage_chunk, ApiError, ChatPlan, RespMeta, StreamShaper, extract_tool_calls, tool_calls_json,
};
use super::protocol::Stats;
use super::{dglog, lock, unix_now, ModelInfo, ServeConfig, Shared};

const MAX_HEAD: usize = 64 * 1024;
const MAX_BODY: usize = 32 * 1024 * 1024;
const MAX_CONNECTIONS: usize = 64;
const READ_TIMEOUT: Duration = Duration::from_secs(30);
const WRITE_TIMEOUT: Duration = Duration::from_secs(120);
/// How often a waiting handler checks its client and the engine.
const TICK: Duration = Duration::from_millis(250);
/// SSE comment interval while no data flows.
const KEEPALIVE: Duration = Duration::from_secs(2);
/// A stream's headers wait at most this long, so an early failure can still
/// be a plain HTTP error.
const HEADER_DELAY: Duration = Duration::from_secs(5);

/// Paths the shim serves (llama-server's aliases included).
const ROUTES: &[&str] = &[
    "/health",
    "/v1/health",
    "/v1/models",
    "/models",
    "/v1/chat/completions",
    "/chat/completions",
    "/slots",
    "/metrics",
    "/frames",
];

/// What every connection thread needs.
pub struct Ctx {
    pub(crate) shared: Arc<Shared>,
    pub(crate) tx: Sender<EngineMsg>,
    pub(crate) cfg: Arc<ServeConfig>,
    pub(crate) info: ModelInfo,
    /// Unix time the helper started (`/v1/models` `created`).
    pub(crate) created: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub method: String,
    /// Routing path: no query string, no trailing slash.
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

// ----------------------------------------------------------------- reading ----

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn io_status(e: &io::Error) -> (u16, &'static str) {
    match e.kind() {
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => (408, "request timed out"),
        _ => (400, "could not read the request"),
    }
}

/// Read more bytes into `buf`; EOF mid-request is an error.
fn fill(s: &mut TcpStream, buf: &mut Vec<u8>) -> Result<(), (u16, &'static str)> {
    let mut tmp = [0u8; 16 * 1024];
    match s.read(&mut tmp) {
        Ok(0) => Err((400, "incomplete request")),
        Ok(n) => {
            buf.extend_from_slice(&tmp[..n]);
            Ok(())
        }
        Err(e) => Err(io_status(&e)),
    }
}

pub fn read_request(s: &mut TcpStream) -> Result<Request, (u16, &'static str)> {
    let mut buf = Vec::with_capacity(4096);
    let head_end = loop {
        if let Some(i) = find(&buf, b"\r\n\r\n") {
            break i;
        }
        if buf.len() > MAX_HEAD {
            return Err((431, "request header fields too large"));
        }
        fill(s, &mut buf)?;
    };
    if head_end > MAX_HEAD {
        return Err((431, "request header fields too large"));
    }
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let rest = buf[head_end + 4..].to_vec();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let (Some(method), Some(target)) = (first.next(), first.next()) else {
        return Err((400, "malformed request line"));
    };
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    let mut req = Request { method: method.to_string(), path: route_path(target), headers, body: Vec::new() };

    let chunked = req.header("transfer-encoding").is_some_and(|v| v.to_ascii_lowercase().contains("chunked"));
    let length = match req.header("content-length") {
        Some(v) => Some(v.parse::<usize>().map_err(|_| (400, "invalid Content-Length"))?),
        None => None,
    };
    if length.is_some_and(|n| n > MAX_BODY) {
        return Err((413, "request body too large"));
    }
    if (chunked || length.is_some_and(|n| n > 0))
        && req.header("expect").is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
    {
        s.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").map_err(|e| io_status(&e))?;
    }
    req.body = if chunked {
        read_chunked(s, rest)?
    } else if let Some(n) = length {
        let mut body = rest;
        while body.len() < n {
            fill(s, &mut body)?;
        }
        body.truncate(n);
        body
    } else {
        Vec::new()
    };
    Ok(req)
}

/// `/v1/models/?x=1` → `/v1/models`.
fn route_path(target: &str) -> String {
    let p = target.split(['?', '#']).next().unwrap_or("");
    let p = if p.len() > 1 { p.trim_end_matches('/') } else { p };
    if p.is_empty() { "/".into() } else { p.to_string() }
}

/// Decode a chunked body as it arrives, stopping at the last chunk: a
/// client that keeps the connection open must not stall us waiting for EOF.
fn read_chunked(s: &mut TcpStream, mut buf: Vec<u8>) -> Result<Vec<u8>, (u16, &'static str)> {
    let mut body = Vec::new();
    loop {
        let line = take_line(s, &mut buf)?;
        let size_hex = line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16).map_err(|_| (400, "invalid chunk size"))?;
        if size == 0 {
            // Optional trailers, then the blank line.
            while !take_line(s, &mut buf)?.is_empty() {}
            return Ok(body);
        }
        // The size is the client's: bound it before any arithmetic, or a
        // chunk of ffffffffffffffff overflows `body.len() + size` and
        // `size + 2` below.
        if size > MAX_BODY.saturating_sub(body.len()) {
            return Err((413, "request body too large"));
        }
        while buf.len() < size + 2 {
            fill(s, &mut buf)?;
        }
        if &buf[size..size + 2] != b"\r\n" {
            return Err((400, "malformed chunk"));
        }
        body.extend_from_slice(&buf[..size]);
        buf.drain(..size + 2);
    }
}

fn take_line(s: &mut TcpStream, buf: &mut Vec<u8>) -> Result<String, (u16, &'static str)> {
    loop {
        if let Some(i) = find(buf, b"\r\n") {
            let line = String::from_utf8_lossy(&buf[..i]).into_owned();
            buf.drain(..i + 2);
            return Ok(line);
        }
        if buf.len() > 4096 {
            return Err((400, "malformed chunk"));
        }
        fill(s, buf)?;
    }
}

// ----------------------------------------------------------------- writing ----

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    }
}

fn write_body(s: &mut TcpStream, status: u16, ctype: &str, body: &[u8], extra: &[(&str, String)]) -> io::Result<()> {
    let mut out = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n",
        reason(status),
        body.len()
    );
    for (k, v) in extra {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("\r\n");
    let mut bytes = out.into_bytes();
    bytes.extend_from_slice(body);
    s.write_all(&bytes)
}

pub fn write_json(s: &mut TcpStream, status: u16, body: &Value, extra: &[(&str, String)]) -> io::Result<()> {
    write_body(s, status, "application/json", body.to_string().as_bytes(), extra)
}

/// Prometheus text for `/metrics`.
pub fn write_text(s: &mut TcpStream, status: u16, text: &str) -> io::Result<()> {
    write_body(s, status, "text/plain; version=0.0.4", text.as_bytes(), &[])
}

fn write_error(s: &mut TcpStream, e: &ApiError) {
    let extra: Vec<(&str, String)> = match e.status {
        // Queue full or still loading: coming back later is the right move.
        503 => vec![("Retry-After", "5".into())],
        // Every 500 is an engine failure the helper has already retried
        // itself. openai-python re-sends any 5xx twice unless told not to,
        // and each replay of a repeatable crash is another runner death
        // toward the 3-death circuit breaker that unloads the model.
        500 => vec![("x-should-retry", "false".into())],
        _ => vec![],
    };
    let _ = write_json(s, e.status, &e.body(), &extra);
}

/// Server-sent events over `Transfer-Encoding: chunked`, one chunk per event
/// so nothing sits in a buffer between them.
pub struct Sse<'a> {
    s: &'a mut TcpStream,
}

impl<'a> Sse<'a> {
    pub fn new(s: &'a mut TcpStream) -> Self {
        Sse { s }
    }

    pub fn start(&mut self) -> io::Result<()> {
        self.s.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\
              Cache-Control: no-cache\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n",
        )
    }

    fn chunk(&mut self, payload: &str) -> io::Result<()> {
        self.s.write_all(format!("{:x}\r\n{payload}\r\n", payload.len()).as_bytes())
    }

    pub fn data(&mut self, data: &str) -> io::Result<()> {
        self.chunk(&format!("data: {data}\n\n"))
    }

    pub fn comment(&mut self, text: &str) -> io::Result<()> {
        self.chunk(&format!(": {text}\n\n"))
    }

    pub fn end(&mut self) -> io::Result<()> {
        self.s.write_all(b"0\r\n\r\n")
    }
}

/// Half-close, then drain briefly: closing a socket with unread input makes
/// Windows send RST, which can destroy the tail of the response in flight.
pub fn close_gracefully(s: TcpStream) {
    let _ = s.shutdown(Shutdown::Write);
    let _ = s.set_read_timeout(Some(Duration::from_millis(50)));
    let deadline = Instant::now() + Duration::from_millis(250);
    let mut total = 0usize;
    let mut buf = [0u8; 4096];
    while Instant::now() < deadline && total < 64 * 1024 {
        match (&s).read(&mut buf) {
            Ok(0) => break,
            Ok(n) => total += n,
            Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
            Err(_) => break,
        }
    }
}

/// A non-blocking peek: an orderly close or a hard error means the client
/// has gone. Pipelined bytes count as present.
pub fn client_gone(s: &TcpStream) -> bool {
    if s.set_nonblocking(true).is_err() {
        return false;
    }
    let mut b = [0u8; 1];
    let gone = match s.peek(&mut b) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != io::ErrorKind::WouldBlock,
    };
    let _ = s.set_nonblocking(false);
    gone
}

// ------------------------------------------------------------------ server ----

/// The name of every connection thread. fidim-dg's panic hook lets a panic
/// here end only its own connection (`diffusion::panic_is_fatal`).
pub(crate) const CONN_THREAD: &str = "dg-conn";

/// Frees the connection slot however the handler ends.
struct Slot(Arc<Ctx>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.shared.connections.fetch_sub(1, Ordering::SeqCst);
    }
}

pub fn accept_loop(listener: TcpListener, ctx: Arc<Ctx>) {
    for conn in listener.incoming() {
        if ctx.shared.shutdown.load(Ordering::SeqCst) {
            break;
        }
        let mut s = match conn {
            Ok(s) => s,
            Err(_) => {
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
        };
        if ctx.shared.connections.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            ctx.shared.connections.fetch_sub(1, Ordering::SeqCst);
            let _ = s.set_write_timeout(Some(Duration::from_secs(5)));
            write_error(&mut s, &ApiError::unavailable("too many open connections"));
            close_gracefully(s);
            continue;
        }
        let slot = Slot(ctx.clone());
        let spawned = std::thread::Builder::new().name(CONN_THREAD.into()).spawn(move || {
            // A bug in one request must cost that connection, not the loaded
            // model: the unwind stops here and the socket is dropped.
            let handled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle_conn(s, &slot.0)));
            if handled.is_err() {
                dglog!("a connection handler panicked; dropped that connection");
            }
            drop(slot);
        });
        if spawned.is_err() {
            dglog!("cannot start a connection thread; dropping the connection");
        }
    }
}

fn handle_conn(mut s: TcpStream, ctx: &Ctx) {
    let _ = s.set_nodelay(true);
    let _ = s.set_read_timeout(Some(READ_TIMEOUT));
    let _ = s.set_write_timeout(Some(WRITE_TIMEOUT));
    match read_request(&mut s) {
        Ok(req) => route(&mut s, &req, ctx),
        Err((status, msg)) => {
            let kind = if status >= 500 { "server_error" } else { "invalid_request_error" };
            write_error(&mut s, &ApiError::new(status, kind, msg));
        }
    }
    close_gracefully(s);
}

fn route(s: &mut TcpStream, req: &Request, ctx: &Ctx) {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health" | "/v1/health") => health(s, ctx),
        ("GET", "/v1/models" | "/models") => models(s, ctx),
        ("POST", "/v1/chat/completions" | "/chat/completions") => chat(s, req, ctx),
        ("GET", "/slots") => slots(s, ctx),
        ("GET", "/metrics") => metrics(s, ctx),
        ("GET", "/frames") => frames(s, ctx),
        (_, p) if ROUTES.contains(&p) => {
            let allow = if p.ends_with("completions") { "POST" } else { "GET" };
            let e = ApiError::new(405, "invalid_request_error", format!("{} is not allowed on {p}", req.method));
            let _ = write_json(s, 405, &e.body(), &[("Allow", allow.into())]);
        }
        (_, p) => write_error(s, &ApiError::not_found(format!("no endpoint {p}"))),
    }
}

fn loading() -> ApiError {
    ApiError::unavailable("Loading model")
}

fn health(s: &mut TcpStream, ctx: &Ctx) {
    let sh = &ctx.shared;
    if !sh.ready.load(Ordering::SeqCst) {
        return write_error(s, &loading());
    }
    let body = json!({
        "status": "ok",
        "engine": "diffusion-gemma",
        "pid": std::process::id(),
        "child_pid": sh.child_pid.load(Ordering::SeqCst),
        "maxtok": sh.maxtok.load(Ordering::SeqCst),
        "restarts": sh.restarts.load(Ordering::SeqCst),
        "queued": sh.queued.load(Ordering::SeqCst),
        "build_tag": ctx.cfg.build_tag,
        "version": crate::build_info::LONG,
    });
    let _ = write_json(s, 200, &body, &[]);
}

/// 503 until the first READY, then 200 for good — a respawn must not make
/// FIDIM's readiness poll or a client's model list flap. n_ctx is the
/// runner's real budget, never inflated.
fn models(s: &mut TcpStream, ctx: &Ctx) {
    if !ctx.shared.ever_ready.load(Ordering::SeqCst) {
        return write_error(s, &loading());
    }
    let n_vocab = match ctx.shared.n_vocab.load(Ordering::SeqCst) {
        0 => ctx.info.vocab,
        n => n as u64,
    };
    let body = json!({
        "object": "list",
        "data": [{
            "id": ctx.cfg.alias,
            "object": "model",
            "created": ctx.created,
            "owned_by": "llama-fidim",
            "meta": {
                "n_ctx": ctx.shared.maxtok.load(Ordering::SeqCst),
                "n_ctx_train": ctx.info.n_ctx_train,
                "n_vocab": n_vocab,
                "canvas": ctx.info.canvas,
                "diffusion": true,
            },
        }],
    });
    let _ = write_json(s, 200, &body, &[]);
}

/// Decoded tokens so far in the current job: whole blocks plus the canvas
/// share of the current step, 0 before the first frame (so FIDIM's live
/// view shows "prefill" first, then "decode").
pub(crate) fn n_decoded(p: &super::Progress, canvas: u32) -> u64 {
    if p.state != "denoise" || p.total == 0 {
        return 0;
    }
    let canvas = canvas as u64;
    let within = (canvas as f64 * (p.step as f64 + 1.0) / p.total as f64).round() as u64;
    p.block as u64 * canvas + within.min(canvas)
}

/// One slot in llama-server's shape (live.rs parses it), with the text
/// llama-server shows under LLAMA_SERVER_SLOTS_DEBUG: the conversation as
/// `prompt`, the committed answer plus the current block's draft as
/// `generated`. `generated_committed_chars` marks where the draft starts, so
/// the loop detector skips it (an early draft is often repetitive noise).
fn slots(s: &mut TcpStream, ctx: &Ctx) {
    let p = ctx.shared.progress().clone();
    let live = ctx.shared.live().clone();
    let processing = ctx.shared.processing.load(Ordering::SeqCst);
    let decoded = if processing { n_decoded(&p, ctx.info.canvas) } else { 0 };
    let remain = p.n_blocks as i64 * ctx.info.canvas as i64 - decoded as i64;
    let body = json!([{
        "id": 0,
        "id_task": p.id_task,
        "n_ctx": ctx.shared.maxtok.load(Ordering::SeqCst),
        "is_processing": processing,
        "n_prompt_tokens": p.n_prompt,
        "n_prompt_tokens_processed": p.n_prompt,
        "n_prompt_tokens_cache": 0,
        "next_token": [{
            "has_next_token": processing,
            "n_decoded": decoded,
            "n_remain": remain.max(0),
        }],
        "prompt": live.prompt,
        "generated": format!("{}{}", live.committed, live.draft),
        "generated_committed_chars": live.committed.chars().count(),
        // Where the denoise is, for FIDIM's canvas view.
        "diffusion": {
            "block": p.block,
            "n_blocks": p.n_blocks,
            "step": p.step,
            "total": p.total,
            "steps_done": p.steps_done,
            "canvas": ctx.info.canvas,
            "state": p.state,
        },
    }]);
    let _ = write_json(s, 200, &body, &[]);
}

/// Every denoise step of the current (or last) job, for the GUI's replay:
/// `{id_task, canvas, dropped, frames: [{b, s, t, x}]}`.
fn frames(s: &mut TcpStream, ctx: &Ctx) {
    let id_task = ctx.shared.progress().id_task;
    let live = ctx.shared.live();
    let frames: Vec<Value> =
        live.frames.iter().map(|f| json!({ "b": f.block, "s": f.step, "t": f.total, "x": f.text })).collect();
    let body = json!({ "id_task": id_task, "canvas": ctx.info.canvas, "dropped": live.frames_dropped, "frames": frames });
    drop(live);
    let _ = write_json(s, 200, &body, &[]);
}

fn metrics(s: &mut TcpStream, ctx: &Ctx) {
    let m = ctx.shared.metrics().clone();
    let sh = &ctx.shared;
    let lines = [
        ("prompt_tokens_total", m.prompt_tokens_total as f64),
        ("tokens_predicted_total", m.tokens_predicted_total as f64),
        ("tokens_predicted_seconds_total", m.predicted_seconds_total),
        ("n_decode_total", m.n_decode_total as f64),
        ("predicted_tokens_seconds", m.last_predicted_tps),
        ("diffusion_canvas_tokens_total", m.canvas_tokens_total as f64),
        ("diffusion_canvas_tokens_seconds", m.last_canvas_tps),
        ("requests_processing", u8::from(sh.processing.load(Ordering::SeqCst)) as f64),
        ("requests_deferred", sh.queued.load(Ordering::SeqCst) as f64),
        ("diffusion_restarts_total", sh.restarts.load(Ordering::SeqCst) as f64),
        ("diffusion_maxtok", sh.maxtok.load(Ordering::SeqCst) as f64),
    ];
    let mut text = String::new();
    for (k, v) in lines {
        text.push_str(&format!("llamacpp:{k} {v}\n"));
    }
    let _ = write_text(s, 200, &text);
}

// -------------------------------------------------------------------- chat ----

struct Enqueued {
    events: Receiver<EngineEvent>,
    cancelled: Arc<AtomicBool>,
    ticket: u64,
    /// Another job was running or waiting when this one arrived.
    behind: bool,
}

/// What the §10 log line needs.
struct Summary {
    id_task: Option<u64>,
    stream: bool,
    n_blocks: u32,
    seed: i32,
    attempt: u32,
    /// 0 = the client went away.
    status: u16,
    finish: &'static str,
    stats: Option<Stats>,
    wait_ms: Option<u128>,
}

impl Summary {
    fn new(plan: &ChatPlan) -> Self {
        Summary {
            id_task: None,
            stream: plan.stream,
            n_blocks: plan.n_blocks,
            seed: plan.req.seed,
            attempt: 1,
            status: 0,
            finish: "-",
            stats: None,
            wait_ms: None,
        }
    }

    fn started(&mut self, id_task: u64, seed: i32, t_enq: Instant) {
        self.id_task = Some(id_task);
        self.seed = seed;
        self.wait_ms = Some(t_enq.elapsed().as_millis());
    }

    fn restarted(&mut self, seed: i32) {
        self.attempt += 1;
        self.seed = seed;
    }

    fn error(mut self, s: &mut TcpStream, e: &ApiError) -> Self {
        write_error(s, e);
        self.status = e.status;
        self
    }

    /// Message text is never logged, only its shape.
    fn log(&self) {
        let id = self.id_task.map(|i| i.to_string()).unwrap_or_else(|| "-".into());
        let status = if self.status == 0 { "gone".to_string() } else { self.status.to_string() };
        let (p, g, wall) = self
            .stats
            .as_ref()
            .map(|s| (s.prompt_n.to_string(), s.predicted_n.to_string(), format!("{:.0}", s.wall_ms)))
            .unwrap_or_else(|| ("-".into(), "-".into(), "-".into()));
        let wait = self.wait_ms.map(|w| w.to_string()).unwrap_or_else(|| "-".into());
        dglog!(
            "#{id} stream={} blocks={} seed={} attempt={} -> {status} finish={} P={p} G={g} wall={wall} wait={wait}",
            self.stream,
            self.n_blocks,
            self.seed,
            self.attempt,
            self.finish
        );
    }
}

fn plan_for(ctx: &Ctx, body: &Value, maxtok: u32) -> Result<ChatPlan, ApiError> {
    let mut rng = lock(&ctx.shared.rng);
    openai::plan_chat(body, ctx.cfg.default_max_tokens, ctx.cfg.seed, ctx.info.canvas, maxtok, &mut rng)
}

fn new_meta(ctx: &Ctx) -> RespMeta {
    let mut rng = lock(&ctx.shared.rng);
    let (a, b) = (rng.next_u64(), rng.next_u64());
    RespMeta { id: format!("chatcmpl-{a:016x}{:08x}", b as u32), created: unix_now(), model: ctx.cfg.alias.clone() }
}

/// Requests that arrive while the model loads wait for it (FIDIM's own
/// readiness gate makes this rare). `Err(None)` = the client left.
fn wait_until_loaded(s: &TcpStream, ctx: &Ctx) -> Result<(), Option<ApiError>> {
    let deadline = Instant::now() + ctx.cfg.load_timeout + Duration::from_secs(30);
    loop {
        if ctx.shared.ever_ready.load(Ordering::SeqCst) {
            return Ok(());
        }
        if ctx.shared.exited.load(Ordering::SeqCst) {
            return Err(Some(ApiError::unavailable("the diffusion engine failed to start (see the log)")));
        }
        if Instant::now() >= deadline {
            return Err(Some(loading()));
        }
        if client_gone(s) {
            return Err(None);
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn enqueue(ctx: &Ctx, plan: &ChatPlan) -> Result<Enqueued, ApiError> {
    let depth = ctx.cfg.queue_depth.max(1);
    let sh = &ctx.shared;
    if sh.queued.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |q| (q < depth).then_some(q + 1)).is_err() {
        return Err(ApiError::unavailable(format!("the diffusion queue is full ({depth} waiting); retry shortly")));
    }
    let (etx, erx) = mpsc::channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    let job = Job {
        req: plan.req.clone(),
        seed_pinned: plan.seed_pinned,
        stream: plan.stream,
        events: etx,
        cancelled: cancelled.clone(),
    };
    let mut tickets = lock(&sh.tickets);
    let ticket = *tickets;
    let behind = sh.processing.load(Ordering::SeqCst) || ticket > sh.served.load(Ordering::SeqCst);
    if ctx.tx.send(EngineMsg::Job(job)).is_err() {
        drop(tickets);
        let _ = sh.queued.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |q| q.checked_sub(1));
        return Err(ApiError::unavailable("the diffusion engine is not running"));
    }
    *tickets += 1;
    Ok(Enqueued { events: erx, cancelled, ticket, behind })
}

/// 1 = next in line.
fn queue_position(ctx: &Ctx, ticket: u64) -> u64 {
    ticket.saturating_sub(ctx.shared.served.load(Ordering::SeqCst)) + 1
}

fn denoising(ctx: &Ctx, id_task: Option<u64>) -> bool {
    let p = ctx.shared.progress();
    id_task == Some(p.id_task) && p.state == "denoise"
}

fn progress_comment(ctx: &Ctx, id_task: u64) -> String {
    let p = ctx.shared.progress();
    if p.id_task == id_task && p.state == "denoise" {
        format!("dg {}/{} {}/{}", p.block + 1, p.n_blocks, p.step + 1, p.total)
    } else if !ctx.shared.ready.load(Ordering::SeqCst) {
        "dg loading".into()
    } else {
        "dg prefill".into()
    }
}

fn chat(s: &mut TcpStream, req: &Request, ctx: &Ctx) {
    let body: Value = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => return write_error(s, &ApiError::invalid(format!("the request body is not valid JSON: {e}"))),
    };
    if !ctx.shared.ever_ready.load(Ordering::SeqCst) {
        // Reject a bad request now rather than after the load.
        if let Err(e) = plan_for(ctx, &body, 0) {
            return write_error(s, &e);
        }
        match wait_until_loaded(s, ctx) {
            Ok(()) => {}
            Err(Some(e)) => return write_error(s, &e),
            Err(None) => return,
        }
    }
    let plan = match plan_for(ctx, &body, ctx.shared.maxtok.load(Ordering::SeqCst)) {
        Ok(p) => p,
        Err(e) => return write_error(s, &e),
    };
    let t_enq = Instant::now();
    let q = match enqueue(ctx, &plan) {
        Ok(q) => q,
        Err(e) => {
            let sum = Summary::new(&plan).error(s, &e);
            return sum.log();
        }
    };
    let sum = if plan.stream { stream_chat(s, ctx, &plan, q, t_enq) } else { plain_chat(s, ctx, &plan, q, t_enq) };
    sum.log();
}

fn plain_chat(s: &mut TcpStream, ctx: &Ctx, plan: &ChatPlan, q: Enqueued, t_enq: Instant) -> Summary {
    let mut sum = Summary::new(plan);
    let mut text = String::new();
    let mut toolong_after = false;
    loop {
        match q.events.recv_timeout(TICK) {
            Ok(EngineEvent::Started { id_task, seed }) => sum.started(id_task, seed, t_enq),
            Ok(EngineEvent::Commit { text: t, .. }) => text = t,
            Ok(EngineEvent::Stats(st)) => sum.stats = Some(st),
            Ok(EngineEvent::TooLong { needed, budget, after_commit }) => {
                if !after_commit {
                    return sum.error(s, &exceed_context(needed, budget, ctx.info.canvas));
                }
                // The committed text stands; finish says "length".
                toolong_after = true;
            }
            Ok(EngineEvent::Restarted { seed }) => {
                text.clear();
                sum.restarted(seed);
            }
            Ok(EngineEvent::Done) => {
                let (reasoning, content, hit) = shape_final(&text, plan.raw_reasoning, &plan.stop);
                // The model writes tool calls as Gemma 4 text; clients expect OpenAI tool_calls.
                let (content, calls) =
                    if plan.req.tools.is_some() { extract_tool_calls(&content) } else { (content, Vec::new()) };
                let finish = if calls.is_empty() {
                    finish_reason(toolong_after, sum.stats.as_ref(), ctx.info.canvas, plan.n_blocks, hit)
                } else {
                    "tool_calls"
                };
                let v = chat_completion(&new_meta(ctx), &content, &reasoning, &calls, finish, sum.stats.as_ref(), sum.seed);
                let _ = write_json(s, 200, &v, &[]);
                sum.status = 200;
                sum.finish = finish;
                return sum;
            }
            Ok(EngineEvent::Failed(f)) => return sum.error(s, &failure_error(&f)),
            Err(RecvTimeoutError::Timeout) => {
                if client_gone(s) {
                    // Skipped if still queued; otherwise it runs to DONE unseen.
                    q.cancelled.store(true, Ordering::SeqCst);
                    return sum;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                return sum.error(s, &ApiError::unavailable("the diffusion engine stopped"));
            }
        }
    }
}

fn stream_chat(s: &mut TcpStream, ctx: &Ctx, plan: &ChatPlan, q: Enqueued, t_enq: Instant) -> Summary {
    let mut sum = Summary::new(plan);
    let mut shaper = StreamShaper::new(plan.raw_reasoning, plan.stop.clone(), plan.req.tools.is_some());
    let mut first = None;

    // Hold the headers back until there is something to stream (or the
    // wait gets long), so an early failure is still a plain HTTP error. A
    // job queued behind another gets them at once, with queue comments.
    if !q.behind {
        loop {
            match q.events.recv_timeout(TICK) {
                Ok(EngineEvent::Started { id_task, seed }) => sum.started(id_task, seed, t_enq),
                Ok(EngineEvent::Restarted { seed }) => {
                    shaper.reset();
                    sum.restarted(seed);
                }
                Ok(EngineEvent::Stats(st)) => sum.stats = Some(st),
                Ok(EngineEvent::Failed(f)) => return sum.error(s, &failure_error(&f)),
                Ok(EngineEvent::TooLong { needed, budget, after_commit: false }) => {
                    return sum.error(s, &exceed_context(needed, budget, ctx.info.canvas));
                }
                Ok(ev) => {
                    first = Some(ev);
                    break;
                }
                Err(RecvTimeoutError::Timeout) => {
                    if denoising(ctx, sum.id_task) || t_enq.elapsed() >= HEADER_DELAY {
                        break;
                    }
                    if client_gone(s) {
                        q.cancelled.store(true, Ordering::SeqCst);
                        return sum;
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return sum.error(s, &ApiError::unavailable("the diffusion engine stopped"));
                }
            }
        }
    }

    let mut sse = Sse::new(s);
    if stream_body(&mut sse, ctx, plan, &q, t_enq, &mut shaper, first, &mut sum).is_err() {
        // A write failed: the client is gone. The job finishes unseen.
        q.cancelled.store(true, Ordering::SeqCst);
        sum.status = 0;
    }
    sum
}

fn stream_error(sse: &mut Sse, e: &ApiError) -> io::Result<()> {
    // Never as content text: a client would show it as the model's answer.
    sse.data(&e.body().to_string())?;
    sse.data("[DONE]")?;
    sse.end()
}

#[allow(clippy::too_many_arguments)]
fn stream_body(
    sse: &mut Sse,
    ctx: &Ctx,
    plan: &ChatPlan,
    q: &Enqueued,
    t_enq: Instant,
    shaper: &mut StreamShaper,
    first: Option<EngineEvent>,
    sum: &mut Summary,
) -> io::Result<()> {
    let meta = new_meta(ctx);
    let chunk = |delta: Value| chat_chunk(&meta, delta, None, None).to_string();
    sse.start()?;
    sse.data(&chunk(json!({ "role": "assistant", "content": null })))?;
    if sum.id_task.is_none() && q.behind {
        sse.comment(&format!("queued {}", queue_position(ctx, q.ticket)))?;
    }
    let mut last_write = Instant::now();
    let mut pending = first;
    let mut text = String::new();
    let mut toolong_after = false;
    loop {
        let ev = match pending.take() {
            Some(ev) => ev,
            None => match q.events.recv_timeout(TICK) {
                Ok(ev) => ev,
                Err(RecvTimeoutError::Timeout) => {
                    if last_write.elapsed() >= KEEPALIVE {
                        let c = match sum.id_task {
                            None => format!("queued {}", queue_position(ctx, q.ticket)),
                            Some(id) => progress_comment(ctx, id),
                        };
                        sse.comment(&c)?;
                        last_write = Instant::now();
                    }
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let e = ApiError::unavailable("the diffusion engine stopped");
                    sum.status = e.status;
                    return stream_error(sse, &e);
                }
            },
        };
        match ev {
            EngineEvent::Started { id_task, seed } => sum.started(id_task, seed, t_enq),
            EngineEvent::Restarted { seed } => {
                shaper.reset();
                text.clear();
                sum.restarted(seed);
            }
            EngineEvent::Commit { text: t, .. } => {
                let (deltas, hit) = shaper.on_commit(&t);
                text = t;
                for d in &deltas {
                    sse.data(&chunk(delta_json(d)))?;
                    last_write = Instant::now();
                }
                if hit {
                    // Stop string: finish now; the job completes in the background.
                    sse.data(&chat_chunk(&meta, json!({}), Some("stop"), None).to_string())?;
                    sse.data("[DONE]")?;
                    sse.end()?;
                    sum.status = 200;
                    sum.finish = "stop";
                    return Ok(());
                }
            }
            EngineEvent::Stats(st) => sum.stats = Some(st),
            EngineEvent::TooLong { after_commit: true, .. } => toolong_after = true,
            EngineEvent::TooLong { needed, budget, .. } => {
                let e = exceed_context(needed, budget, ctx.info.canvas);
                sum.status = e.status;
                return stream_error(sse, &e);
            }
            EngineEvent::Done => {
                let (deltas, calls) = shaper.finish(&text);
                for d in &deltas {
                    sse.data(&chunk(delta_json(d)))?;
                }
                if !calls.is_empty() {
                    sse.data(&chunk(json!({ "tool_calls": tool_calls_json(&meta.id, &calls, true) })))?;
                }
                let (_, _, hit) = shape_final(&text, plan.raw_reasoning, &plan.stop);
                let finish = if calls.is_empty() {
                    finish_reason(toolong_after, sum.stats.as_ref(), ctx.info.canvas, plan.n_blocks, hit)
                } else {
                    "tool_calls"
                };
                let t = sum.stats.as_ref().map(|st| ("timings", timings(st, sum.seed)));
                sse.data(&chat_chunk(&meta, json!({}), Some(finish), t).to_string())?;
                if plan.include_usage {
                    sse.data(&usage_chunk(&meta, sum.stats.as_ref(), sum.seed).to_string())?;
                }
                sse.data("[DONE]")?;
                sse.end()?;
                sum.status = 200;
                sum.finish = finish;
                return Ok(());
            }
            EngineEvent::Failed(f) => {
                let e = failure_error(&f);
                sum.status = e.status;
                return stream_error(sse, &e);
            }
        }
    }
}
