//! M28 — the chat box's client: one streaming `/v1/chat/completions` request
//! to a server Llama FIDIM started (a llama-server, a model behind the
//! router, or a DiffusionGemma run's fidim-dg), turned into events the GUI
//! renders as they arrive.
//!
//! std only, like `supervise::http_get`. Three layers, each testable alone:
//! `wire::ChunkedReader` decodes the chunked body, `wire::SseParser` splits
//! it into SSE data and comments, and `Translator` maps those onto
//! `ChatEvent`s, coalescing deltas so the view re-renders a few dozen times a
//! second rather than once per token. `stream_chat` drives them.
//!
//! Cancellation. The socket is non-blocking and the reader waits for bytes
//! in `WSAPoll` with a short timeout, checking the cancel flag between waits;
//! on cancel it simply drops (closes) the socket, which is what makes
//! llama-server free the slot. Two simpler designs do not work on Windows,
//! both checked on this machine: `shutdown()` from another thread (even on
//! the original handle) does not wake a thread blocked in `recv`, and a
//! receive that times out under SO_RCVTIMEO leaves the connection in an
//! indeterminate state (Microsoft's words), so polling a flag between short
//! read timeouts would risk the stream itself. The idle timeout (no bytes at
//! all for 10 minutes) is separate from the cancel poll.
//!
//! Message text is never logged here or by the callers.

pub mod store;
pub mod wire;

use std::io::{self, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::live::LiveSample;
use crate::profile::{DiffusionCfg, Engine, Profile, Sampling};
use crate::supervise::{AttachedRun, RunState};
use crate::{Error, Result};

use wire::{ChunkedReader, Prefixed, SseItem, SseParser};

/// No bytes at all for this long ends a stream (llama-server sends nothing
/// while it prefills unless asked for progress; a router may be loading).
pub const DEFAULT_IDLE: Duration = Duration::from_secs(10 * 60);
/// Deltas are held back at most this long, so the view renders in batches.
const FLUSH: Duration = Duration::from_millis(30);
/// Longest single wait for bytes; the cancel flag is checked between waits.
const POLL: Duration = Duration::from_millis(100);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_HEAD: usize = 64 * 1024;
const MAX_ERROR_BODY: usize = 1024 * 1024;
const MAX_JSON_BODY: usize = 64 * 1024 * 1024;

// -------------------------------------------------------------- addresses ----

/// The address to connect to for a server bound to `host`: a wildcard bind
/// (0.0.0.0, ::) is reached over loopback; connecting to the wildcard itself
/// fails on Windows.
pub fn connect_host(host: &str) -> &str {
    match host.trim() {
        "0.0.0.0" | "" => "127.0.0.1",
        "::" | "[::]" => "::1",
        h => h,
    }
}

/// `host:port` for a URL or a Host header, IPv6 in brackets.
pub fn host_port(host: &str, port: u16) -> String {
    let h = connect_host(host).trim_start_matches('[').trim_end_matches(']');
    if h.contains(':') { format!("[{h}]:{port}") } else { format!("{h}:{port}") }
}

pub fn socket_addr(host: &str, port: u16) -> Result<SocketAddr> {
    let hp = host_port(host, port);
    if let Ok(a) = hp.parse() {
        return Ok(a);
    }
    // A name: localhost is the only one a FIDIM profile would use.
    if connect_host(host).eq_ignore_ascii_case("localhost") {
        return Ok(SocketAddr::from(([127, 0, 0, 1], port)));
    }
    Err(Error::Platform(format!("bad address {hp}: the host must be an IP address")))
}

/// The OpenAI base URL clients use for a server bound to `host`.
pub fn base_url(host: &str, port: u16) -> String {
    format!("http://{}/v1", host_port(host, port))
}

/// Whether only this PC can reach a server bound to `host`.
pub fn is_loopback(host: &str) -> bool {
    let h = host.trim().trim_start_matches('[').trim_end_matches(']');
    h.eq_ignore_ascii_case("localhost") || h.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// Where a llama-server profile's API keys come from. llama-server accepts
/// every key from every source, so any one of them authenticates.
#[derive(Debug, Default, PartialEq)]
struct KeySources {
    /// `--api-key` values and `LLAMA_API_KEY`: comma-separated lists.
    lists: Vec<String>,
    /// `--api-key-file` and `LLAMA_ARG_API_KEY_FILE`: one key per line.
    files: Vec<String>,
}

fn key_sources(p: &Profile) -> KeySources {
    let mut ks = KeySources::default();
    if !p.engine.is_llama_server() {
        return ks;
    }
    let flags = &p.runtime.extra_flags;
    for (i, f) in flags.iter().enumerate() {
        let next = || flags.get(i + 1).cloned();
        match f.as_str() {
            "--api-key" => ks.lists.extend(next()),
            "--api-key-file" => ks.files.extend(next()),
            _ => {
                if let Some(v) = f.strip_prefix("--api-key=") {
                    ks.lists.push(v.to_string());
                } else if let Some(v) = f.strip_prefix("--api-key-file=") {
                    ks.files.push(v.to_string());
                }
            }
        }
    }
    let env = |name: &str| p.env.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.clone());
    ks.lists.extend(env("LLAMA_API_KEY").filter(|v| !v.is_empty()));
    ks.files.extend(env("LLAMA_ARG_API_KEY_FILE").filter(|v| !v.is_empty()));
    ks
}

/// llama.cpp's `parse_csv_row`: split on commas outside double quotes; `""`
/// inside quotes is one quote.
fn csv_fields(s: &str) -> Vec<String> {
    let (mut out, mut field, mut quoted) = (Vec::new(), String::new(), false);
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        match c {
            '"' if quoted && it.peek() == Some(&'"') => {
                field.push('"');
                it.next();
            }
            '"' if quoted => quoted = false,
            '"' if field.is_empty() => quoted = true,
            ',' if !quoted => out.push(std::mem::take(&mut field)),
            c => field.push(c),
        }
    }
    out.push(field);
    out
}

/// The keys in a key file as llama-server reads it: one per line, blank
/// lines and lines starting with `#` skipped. A relative path is taken from
/// this process's folder, as the server takes it from its own.
fn file_keys(path: &str) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new() };
    // The server reads the file in text mode: CRLF is a line end.
    text.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l)).filter(|l| !l.is_empty() && !l.starts_with('#')).map(str::to_string).collect()
}

/// Whether a llama-server profile requires an API key: it sets
/// `--api-key`, `--api-key-file` or their environment variables. Other
/// programs then need the key too (the Endpoint card says so).
pub fn requires_api_key(p: &Profile) -> bool {
    key_sources(p) != KeySources::default()
}

/// A key the chat can send to a llama-server profile that requires one, so
/// JS never holds it: the first usable one from `--api-key` (a
/// comma-separated list), `LLAMA_API_KEY`, then the key files. A key with
/// control characters is skipped: it goes into a header line.
pub fn api_key(p: &Profile) -> Option<String> {
    let usable = |v: &String| !v.is_empty() && !v.chars().any(char::is_control);
    let ks = key_sources(p);
    let listed = ks.lists.iter().flat_map(|l| csv_fields(l));
    let filed = ks.files.iter().flat_map(|f| file_keys(f));
    listed.chain(filed).find(usable)
}

// ---------------------------------------------------------------- request ----

/// The body sent upstream: the GUI's messages and sampler overrides, plus
/// the fields streaming depends on. llama-server also gets prompt progress
/// (otherwise nothing arrives until the first token) and per-token timings;
/// fidim-dg would ignore them, so they are not sent there.
pub fn request_body(body: &Value, model: &str, engine: Engine) -> Result<Value> {
    let Some(obj) = body.as_object() else {
        return Err(Error::Config("the chat request must be a JSON object".into()));
    };
    if !obj.get("messages").and_then(Value::as_array).is_some_and(|m| !m.is_empty()) {
        return Err(Error::Config("the chat request needs a non-empty 'messages' array".into()));
    }
    let mut out = obj.clone();
    out.insert("model".into(), model.into());
    out.insert("stream".into(), true.into());
    let mut so = out.get("stream_options").and_then(Value::as_object).cloned().unwrap_or_default();
    so.insert("include_usage".into(), true.into());
    out.insert("stream_options".into(), Value::Object(so));
    if engine.is_llama_server() {
        out.insert("return_progress".into(), true.into());
        out.insert("timings_per_token".into(), true.into());
    }
    Ok(Value::Object(out))
}

// ----------------------------------------------------------------- events ----

/// What the GUI hears while a reply streams. Exactly one of `Done`, `Error`
/// or `Cancelled` ends every stream.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ChatEvent {
    /// The server accepted the request and started the stream.
    Open { status: u16 },
    /// fidim-dg: waiting behind other requests; 1 = next.
    Queued { position: u64 },
    /// fidim-dg: the job started as this task (matches `/slots` and `/frames`).
    Task { id_task: u64 },
    /// fidim-dg's keep-alive progress while no text flows. `stage` is
    /// `loading`, `prefill` or `denoise`; block and step are 1-based.
    Progress {
        stage: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        block: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        n_blocks: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        step: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        total: Option<u32>,
    },
    /// llama-server prompt processing (`return_progress`).
    Prefill { processed: u64, total: u64, cache: u64, time_ms: f64 },
    /// Text since the last delta. Reasoning is `reasoning_content`.
    Delta {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
    },
    /// Every tool call so far, merged from the streamed fragments.
    ToolCalls { calls: Value },
    /// Live generation stats (llama-server `timings_per_token`).
    Timings { timings: Value },
    Done {
        finish_reason: Option<String>,
        timings: Option<Value>,
        usage: Option<Value>,
        model: Option<String>,
    },
    Error { status: Option<u16>, message: String },
    Cancelled,
}

/// How a stream ended, for the command's return value.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct StreamSummary {
    pub finish_reason: Option<String>,
    pub timings: Option<Value>,
    pub usage: Option<Value>,
    pub cancelled: bool,
    pub error: Option<String>,
    pub status: Option<u16>,
    pub id_task: Option<u64>,
}

/// A stream's stop button. `stream_chat` checks it at least every 100 ms.
#[derive(Debug, Default)]
pub struct Cancel {
    flag: AtomicBool,
}

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// fidim-dg's SSE comments: `queued N`, `dg task N`, `dg B/N S/T`,
/// `dg prefill`, `dg loading`. Anything else (llama-server's `:` pings) is
/// not an event.
pub fn comment_event(c: &str) -> Option<ChatEvent> {
    let c = c.trim();
    if let Some(n) = c.strip_prefix("queued ") {
        return Some(ChatEvent::Queued { position: n.trim().parse().ok()? });
    }
    let rest = c.strip_prefix("dg ")?.trim();
    if let Some(id) = rest.strip_prefix("task ") {
        return Some(ChatEvent::Task { id_task: id.trim().parse().ok()? });
    }
    let stage = |s: &str| ChatEvent::Progress { stage: s.into(), block: None, n_blocks: None, step: None, total: None };
    match rest {
        "prefill" => return Some(stage("prefill")),
        "loading" => return Some(stage("loading")),
        _ => {}
    }
    let mut it = rest.split_whitespace();
    let (b, s) = (it.next()?.split_once('/')?, it.next()?.split_once('/')?);
    Some(ChatEvent::Progress {
        stage: "denoise".into(),
        block: Some(b.0.parse().ok()?),
        n_blocks: Some(b.1.parse().ok()?),
        step: Some(s.0.parse().ok()?),
        total: Some(s.1.parse().ok()?),
    })
}

/// An upstream error object (`{"code","message","type"}`, or a bare string).
fn error_of(e: &Value) -> (Option<u16>, String) {
    let status = e.get("code").and_then(Value::as_u64).filter(|c| (100..600).contains(c)).map(|c| c as u16);
    let msg = e
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| e.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| e.to_string());
    (status, msg)
}

/// SSE items in, ordered events out. Text deltas, tool-call fragments and
/// live timings are held until `flush`; every other event flushes them first
/// so the order the server sent is kept.
#[derive(Debug, Default)]
pub struct Translator {
    content: String,
    reasoning: String,
    live_timings: Option<Value>,
    calls: Vec<Value>,
    calls_dirty: bool,
    pub finish_reason: Option<String>,
    pub timings: Option<Value>,
    pub usage: Option<Value>,
    pub model: Option<String>,
    /// `data: [DONE]` arrived.
    pub done: bool,
}

impl Translator {
    pub fn has_pending(&self) -> bool {
        !self.content.is_empty() || !self.reasoning.is_empty() || self.calls_dirty || self.live_timings.is_some()
    }

    pub fn flush(&mut self, out: &mut Vec<ChatEvent>) {
        if !self.content.is_empty() || !self.reasoning.is_empty() {
            let take = |s: &mut String| Some(std::mem::take(s)).filter(|t| !t.is_empty());
            out.push(ChatEvent::Delta { content: take(&mut self.content), reasoning: take(&mut self.reasoning) });
        }
        if std::mem::take(&mut self.calls_dirty) {
            out.push(ChatEvent::ToolCalls { calls: Value::Array(self.calls.clone()) });
        }
        if let Some(t) = self.live_timings.take() {
            out.push(ChatEvent::Timings { timings: t });
        }
    }

    fn ordered(&mut self, ev: ChatEvent, out: &mut Vec<ChatEvent>) {
        self.flush(out);
        out.push(ev);
    }

    /// Feed one item. `Err` = the server reported an error in the stream.
    pub fn feed(&mut self, item: SseItem, out: &mut Vec<ChatEvent>) -> std::result::Result<(), (Option<u16>, String)> {
        match item {
            SseItem::Comment(c) => {
                if let Some(ev) = comment_event(&c) {
                    self.ordered(ev, out);
                }
                Ok(())
            }
            SseItem::Data(d) => self.data(&d, out),
        }
    }

    fn data(&mut self, d: &str, out: &mut Vec<ChatEvent>) -> std::result::Result<(), (Option<u16>, String)> {
        let d = d.trim();
        if d == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        // Not JSON: nothing a chat client can use.
        let Ok(v) = serde_json::from_str::<Value>(d) else { return Ok(()) };
        if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
            self.flush(out);
            return Err(error_of(e));
        }
        if let Some(m) = v.get("model").and_then(Value::as_str) {
            self.model = Some(m.to_string());
        }
        if let Some(p) = v.get("prompt_progress").filter(|p| p.is_object()) {
            let n = |k: &str| p.get(k).and_then(Value::as_u64).unwrap_or(0);
            let ev = ChatEvent::Prefill {
                processed: n("processed"),
                total: n("total"),
                cache: n("cache"),
                time_ms: p.get("time_ms").and_then(Value::as_f64).unwrap_or(0.0),
            };
            self.ordered(ev, out);
        }
        if let Some(c) = v.get("choices").and_then(|c| c.get(0)) {
            // `delta` while streaming; `message` when a server answered with
            // one plain completion instead.
            for part in [c.get("delta"), c.get("message")].into_iter().flatten() {
                if let Some(t) = part.get("reasoning_content").and_then(Value::as_str) {
                    self.reasoning.push_str(t);
                }
                if let Some(t) = part.get("content").and_then(Value::as_str) {
                    self.content.push_str(t);
                }
                if let Some(tc) = part.get("tool_calls").and_then(Value::as_array) {
                    self.merge_calls(tc);
                }
            }
            if let Some(f) = c.get("finish_reason").and_then(Value::as_str) {
                self.finish_reason = Some(f.to_string());
            }
        }
        if let Some(t) = v.get("timings").filter(|t| t.is_object()) {
            self.timings = Some(t.clone());
            self.live_timings = Some(t.clone());
        }
        if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(u.clone());
        }
        Ok(())
    }

    /// OpenAI streams a tool call as fragments keyed by `index`: the name
    /// once, the arguments in pieces to concatenate.
    fn merge_calls(&mut self, parts: &[Value]) {
        for (i, p) in parts.iter().enumerate() {
            let idx = p.get("index").and_then(Value::as_u64).map_or(i, |x| x as usize);
            if idx >= 64 {
                continue;
            }
            while self.calls.len() <= idx {
                self.calls.push(json!({ "id": null, "type": "function", "function": { "name": "", "arguments": "" } }));
            }
            let c = &mut self.calls[idx];
            if let Some(id) = p.get("id").and_then(Value::as_str) {
                c["id"] = id.into();
            }
            if let Some(f) = p.get("function") {
                for k in ["name", "arguments"] {
                    match f.get(k) {
                        Some(Value::String(s)) => {
                            let cur = c["function"][k].as_str().unwrap_or("").to_string();
                            c["function"][k] = (cur + s).into();
                        }
                        // Some servers send arguments as an object.
                        Some(v @ Value::Object(_)) => c["function"][k] = v.to_string().into(),
                        _ => {}
                    }
                }
            }
            self.calls_dirty = true;
        }
    }

    /// The stream ended: flush and close with `Done`, or say it was cut short.
    pub fn finish(&mut self, out: &mut Vec<ChatEvent>) -> std::result::Result<(), (Option<u16>, String)> {
        self.flush(out);
        if self.done || self.finish_reason.is_some() {
            out.push(ChatEvent::Done {
                finish_reason: self.finish_reason.clone(),
                timings: self.timings.clone(),
                usage: self.usage.clone(),
                model: self.model.clone(),
            });
            Ok(())
        } else {
            Err((None, "the server closed the stream before the reply finished".into()))
        }
    }
}

// --------------------------------------------------------------- streaming ----

enum End {
    Cancelled,
    Failed { status: Option<u16>, message: String },
}

fn failed(status: Option<u16>, message: impl Into<String>) -> End {
    End::Failed { status, message: message.into() }
}

/// Hands events to the caller and remembers how the stream ended.
struct Emitter<'a> {
    on: &'a mut dyn FnMut(ChatEvent),
    summary: StreamSummary,
}

impl Emitter<'_> {
    fn send(&mut self, evs: &mut Vec<ChatEvent>) {
        for ev in evs.drain(..) {
            match &ev {
                ChatEvent::Task { id_task } => self.summary.id_task = Some(*id_task),
                ChatEvent::Done { finish_reason, timings, usage, .. } => {
                    self.summary.finish_reason.clone_from(finish_reason);
                    self.summary.timings.clone_from(timings);
                    self.summary.usage.clone_from(usage);
                }
                ChatEvent::Error { status, message } => {
                    self.summary.status = *status;
                    self.summary.error = Some(message.clone());
                }
                ChatEvent::Cancelled => self.summary.cancelled = true,
                _ => {}
            }
            (self.on)(ev);
        }
    }
}

/// Wait until `s` has bytes (or EOF, or an error) to read, at most `t`.
/// WSAPoll leaves the socket alone: no receive is ever timed out.
#[cfg(windows)]
fn wait_readable(s: &TcpStream, t: Duration) {
    use std::os::windows::io::AsRawSocket;
    use windows::Win32::Networking::WinSock::{WSAPoll, POLLRDNORM, SOCKET, WSAPOLLFD, WSAPOLL_EVENT_FLAGS};
    let mut fd = WSAPOLLFD { fd: SOCKET(s.as_raw_socket() as usize), events: POLLRDNORM, revents: WSAPOLL_EVENT_FLAGS(0) };
    let ms = t.as_millis().clamp(1, i32::MAX as u128) as i32;
    // SAFETY: one valid WSAPOLLFD for a socket this function borrows.
    if unsafe { WSAPoll(&mut fd, 1, ms) } < 0 {
        // Never spin on a failing poll.
        std::thread::sleep(t);
    }
}

/// Elsewhere the non-blocking read is simply retried on a short sleep.
#[cfg(not(windows))]
fn wait_readable(_s: &TcpStream, t: Duration) {
    std::thread::sleep(t.min(Duration::from_millis(10)));
}

/// The cancel flag and the idle clock, consulted whenever the socket has
/// nothing to read.
struct Pace<'a> {
    cancel: &'a Cancel,
    idle: Duration,
    last_data: Instant,
}

impl Pace<'_> {
    fn check(&self) -> std::result::Result<(), End> {
        if self.cancel.is_cancelled() {
            return Err(End::Cancelled);
        }
        if self.last_data.elapsed() >= self.idle {
            return Err(failed(None, format!("no data from the server for {}; gave up", human(self.idle))));
        }
        Ok(())
    }

    fn wait(&self, sock: &TcpStream, max: Duration) -> std::result::Result<(), End> {
        self.check()?;
        let left = self.idle.saturating_sub(self.last_data.elapsed());
        wait_readable(sock, max.min(POLL).min(left).max(Duration::from_millis(1)));
        self.check()
    }
}

fn human(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 60 && s % 60 == 0 { format!("{} min", s / 60) } else if s > 0 { format!("{s} s") } else { format!("{} ms", d.as_millis()) }
}

struct Head {
    status: u16,
    chunked: bool,
    length: Option<u64>,
    content_type: String,
}

fn parse_head(raw: &[u8]) -> Option<Head> {
    let text = String::from_utf8_lossy(raw);
    let mut lines = text.split("\r\n");
    let status = lines.next()?.split_whitespace().nth(1)?.parse().ok()?;
    let mut h = Head { status, chunked: false, length: None, content_type: String::new() };
    for l in lines {
        let Some((k, v)) = l.split_once(':') else { continue };
        let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
        match k.as_str() {
            "transfer-encoding" => h.chunked = v.to_ascii_lowercase().contains("chunked"),
            "content-length" => h.length = v.parse().ok(),
            "content-type" => h.content_type = v.to_ascii_lowercase(),
            _ => {}
        }
    }
    Some(h)
}

/// The response body, framed as the headers say.
enum Body<'a> {
    Chunked(ChunkedReader<&'a TcpStream>),
    Length(Prefixed<&'a TcpStream>, u64),
    ToEof(Prefixed<&'a TcpStream>),
}

impl Read for Body<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        match self {
            Body::Chunked(r) => r.read(out),
            Body::Length(r, left) => {
                if *left == 0 {
                    return Ok(0);
                }
                let k = (*left).min(out.len() as u64) as usize;
                let n = r.read(&mut out[..k])?;
                *left -= n as u64;
                Ok(n)
            }
            Body::ToEof(r) => r.read(out),
        }
    }
}

/// Read one read's worth from a non-blocking source, waiting as needed.
/// Ok(0) = end of body.
fn read_some(r: &mut impl Read, sock: &TcpStream, pace: &mut Pace, buf: &mut [u8]) -> std::result::Result<usize, End> {
    loop {
        if pace.cancel.is_cancelled() {
            return Err(End::Cancelled);
        }
        match r.read(buf) {
            Ok(n) => {
                if n > 0 {
                    pace.last_data = Instant::now();
                }
                return Ok(n);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => pace.wait(sock, POLL)?,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            // A body cut short reads as its end; the caller decides whether
            // what arrived was complete.
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(0),
            Err(e) => return Err(failed(None, format!("reading the response: {e}"))),
        }
    }
}

/// A whole (bounded) body as text: error pages and plain JSON answers.
fn read_text(body: &mut Body, sock: &TcpStream, pace: &mut Pace, max: usize) -> std::result::Result<String, End> {
    let mut out = Vec::new();
    let mut buf = [0u8; 16 * 1024];
    while out.len() < max {
        match read_some(body, sock, pace, &mut buf)? {
            0 => break,
            n => out.extend_from_slice(&buf[..n]),
        }
    }
    out.truncate(max);
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// The message of a non-2xx answer: the llama-server error envelope, or the
/// start of whatever the body says.
pub fn error_message(status: u16, body: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body.trim()) {
        if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
            return error_of(e).1;
        }
        if let Some(m) = v.get("message").and_then(Value::as_str) {
            return m.to_string();
        }
    }
    let t = body.trim();
    if t.is_empty() {
        format!("HTTP {status}")
    } else {
        t.chars().take(300).collect()
    }
}

/// Stream one chat completion from `host:port`. `body` is sent as given:
/// shape it with `request_body` first. Events go to `on` in order; the last
/// one is always `Done`, `Error` or `Cancelled`.
pub fn stream_chat(
    host: &str,
    port: u16,
    body: &Value,
    api_key: Option<&str>,
    cancel: &Cancel,
    idle: Duration,
    on: &mut dyn FnMut(ChatEvent),
) -> StreamSummary {
    let mut out = Emitter { on, summary: StreamSummary::default() };
    let mut tr = Translator::default();
    let end = run_stream(host, port, body, api_key, cancel, idle, &mut tr, &mut out);
    // The socket is closed by now: a cancelled server sees it before the GUI
    // hears `Cancelled`.
    let mut evs = Vec::new();
    let end = match end {
        Ok(()) => tr.finish(&mut evs).map_err(|(status, message)| End::Failed { status, message }),
        Err(e) => {
            tr.flush(&mut evs);
            Err(e)
        }
    };
    match end {
        Ok(()) => {}
        Err(End::Cancelled) => evs.push(ChatEvent::Cancelled),
        Err(End::Failed { status, message }) => evs.push(ChatEvent::Error { status, message }),
    }
    out.send(&mut evs);
    out.summary
}

#[allow(clippy::too_many_arguments)]
fn run_stream(
    host: &str,
    port: u16,
    body: &Value,
    api_key: Option<&str>,
    cancel: &Cancel,
    idle: Duration,
    tr: &mut Translator,
    out: &mut Emitter,
) -> std::result::Result<(), End> {
    if cancel.is_cancelled() {
        return Err(End::Cancelled);
    }
    let addr = socket_addr(host, port).map_err(|e| failed(None, e.to_string()))?;
    let sock = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
        .map_err(|e| failed(None, format!("cannot connect to {addr}: {e}")))?;
    let _ = sock.set_nodelay(true);
    let _ = sock.set_write_timeout(Some(WRITE_TIMEOUT));
    let payload = body.to_string();
    let mut req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
         Accept: text/event-stream\r\nConnection: close\r\nContent-Length: {}\r\n",
        host_port(host, port),
        payload.len()
    );
    if let Some(k) = api_key {
        req.push_str(&format!("Authorization: Bearer {k}\r\n"));
    }
    req.push_str("\r\n");
    (&sock)
        .write_all(req.as_bytes())
        .and_then(|_| (&sock).write_all(payload.as_bytes()))
        .map_err(|e| failed(None, format!("sending the request to {addr}: {e}")))?;
    sock.set_nonblocking(true).map_err(|e| failed(None, format!("socket: {e}")))?;
    let mut pace = Pace { cancel, idle, last_data: Instant::now() };

    // Head: everything up to the blank line; what follows starts the body.
    let mut raw = Vec::new();
    let mut buf = [0u8; 16 * 1024];
    let (head, rest) = loop {
        if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = parse_head(&raw[..i]).ok_or_else(|| failed(None, "the server's answer is not HTTP"))?;
            break (head, raw[i + 4..].to_vec());
        }
        if raw.len() > MAX_HEAD {
            return Err(failed(None, "the server's response headers are too large"));
        }
        let mut s = &sock;
        match read_some(&mut s, &sock, &mut pace, &mut buf)? {
            0 => return Err(failed(None, "the server closed the connection without answering")),
            n => raw.extend_from_slice(&buf[..n]),
        }
    };
    let mut body = if head.chunked {
        Body::Chunked(ChunkedReader::with_prefix(&sock, rest))
    } else if let Some(n) = head.length {
        Body::Length(Prefixed::new(rest, &sock), n)
    } else {
        Body::ToEof(Prefixed::new(rest, &sock))
    };

    if !(200..300).contains(&head.status) {
        let text = read_text(&mut body, &sock, &mut pace, MAX_ERROR_BODY)?;
        return Err(failed(Some(head.status), error_message(head.status, &text)));
    }
    out.send(&mut vec![ChatEvent::Open { status: head.status }]);

    let mut evs = Vec::new();
    if !head.content_type.contains("text/event-stream") {
        // A server that answered with one plain completion.
        let text = read_text(&mut body, &sock, &mut pace, MAX_JSON_BODY)?;
        if serde_json::from_str::<Value>(text.trim()).is_err() {
            return Err(failed(Some(head.status), "the server answered, but not with a chat completion"));
        }
        tr.feed(SseItem::Data(text), &mut evs).map_err(|(status, message)| End::Failed { status, message })?;
        tr.done = true;
        out.send(&mut evs);
        return Ok(());
    }

    let mut sse = SseParser::new();
    let mut items = Vec::new();
    let mut last_flush = Instant::now();
    loop {
        if cancel.is_cancelled() {
            return Err(End::Cancelled);
        }
        let n = match body.read(&mut buf) {
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                // Nothing to read: send held-back deltas when due, else wait.
                if tr.has_pending() && last_flush.elapsed() >= FLUSH {
                    tr.flush(&mut evs);
                    out.send(&mut evs);
                    last_flush = Instant::now();
                    continue;
                }
                let max = if tr.has_pending() { FLUSH.saturating_sub(last_flush.elapsed()) } else { POLL };
                pace.wait(&sock, max)?;
                continue;
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => 0,
            Err(e) => return Err(failed(None, format!("reading the stream: {e}"))),
        };
        if n == 0 {
            sse.finish(&mut items);
        } else {
            pace.last_data = Instant::now();
            sse.push(&buf[..n], &mut items);
        }
        for it in items.drain(..) {
            if let Err((status, message)) = tr.feed(it, &mut evs) {
                out.send(&mut evs);
                return Err(End::Failed { status, message });
            }
        }
        out.send(&mut evs);
        if n == 0 || tr.done {
            return Ok(());
        }
        if tr.has_pending() && last_flush.elapsed() >= FLUSH {
            tr.flush(&mut evs);
            out.send(&mut evs);
            last_flush = Instant::now();
        }
    }
}

// ------------------------------------------------------------------- props ----

/// Sampler keys the chat's inspector shows from `/props`.
const PROPS_PARAMS: &[&str] = &[
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "dry_multiplier",
    "max_tokens",
    "n_predict",
    "seed",
    "stop",
    "reasoning_format",
];

/// llama-server's `/props`, trimmed to what the chat uses: the per-slot
/// context, the sampler the server applies by default (the profile's flags
/// included), vision/audio support and the chat template's capabilities.
/// The template text itself is left out.
pub fn parse_props(v: &Value) -> Value {
    let dgs = v.get("default_generation_settings").cloned().unwrap_or(Value::Null);
    let params = dgs.get("params").unwrap_or(&dgs);
    let mut p = Map::new();
    for k in PROPS_PARAMS {
        if let Some(x) = params.get(*k) {
            p.insert((*k).into(), x.clone());
        }
    }
    json!({
        "engine": "llama-server",
        "n_ctx": dgs.get("n_ctx").or_else(|| v.get("n_ctx")).and_then(Value::as_u64),
        "total_slots": v.get("total_slots").and_then(Value::as_u64),
        "params": p,
        "modalities": v.get("modalities").cloned().unwrap_or_else(|| json!({})),
        "caps": v.get("chat_template_caps").cloned().unwrap_or_else(|| json!({})),
        "model_alias": v.get("model_alias").cloned().unwrap_or(Value::Null),
        "build_info": v.get("build_info").cloned().unwrap_or(Value::Null),
    })
}

/// fidim-dg has no `/props`: its `/v1/models` meta carries the context
/// budget (MAXTOK) and the canvas.
pub fn parse_dg_models(v: &Value) -> Value {
    let m = v.get("data").and_then(|d| d.get(0)).cloned().unwrap_or(Value::Null);
    let meta = m.get("meta").cloned().unwrap_or(Value::Null);
    json!({
        "engine": "diffusion-gemma",
        "n_ctx": meta.get("n_ctx").and_then(Value::as_u64),
        "n_ctx_train": meta.get("n_ctx_train").and_then(Value::as_u64),
        "canvas": meta.get("canvas").and_then(Value::as_u64),
        "diffusion": meta.get("diffusion").and_then(Value::as_bool).unwrap_or(true),
        "model_alias": m.get("id").cloned().unwrap_or(Value::Null),
        "params": {},
        "caps": {},
        "modalities": {},
    })
}

/// Fetch the props of a target. `model` names a router model, asked with
/// `autoload=false` so the question never loads it; `api_key` is the
/// profile's (`api_key`): a keyed llama-server answers 401 without it.
pub fn props(host: &str, port: u16, engine: Engine, model: Option<&str>, api_key: Option<&str>) -> Result<Value> {
    let host = connect_host(host);
    let t = Duration::from_secs(5);
    let path = if engine.is_diffusion() { "/v1/models".to_string() } else { crate::live::query("/props", model) };
    let (code, body) = crate::supervise::http_get_auth(host, port, &path, t, api_key)?;
    if code != 200 {
        return Err(Error::Platform(format!("{path} answered {code}: {}", error_message(code, &body))));
    }
    let v: Value = serde_json::from_str(&body)?;
    Ok(if engine.is_diffusion() { parse_dg_models(&v) } else { parse_props(&v) })
}

// ----------------------------------------------------------------- targets ----

/// One thing the chat can talk to: a standalone run, or one model behind
/// the router.
#[derive(Debug, Clone, Serialize)]
pub struct Target {
    /// `run` or `run/model`, stable across refreshes.
    pub key: String,
    pub run: String,
    pub model: Option<String>,
    pub engine: Engine,
    pub label: String,
    pub profile_id: Option<String>,
    /// What clients send as `model`.
    pub model_id: String,
    /// Where to connect (a wildcard bind mapped to loopback).
    pub host: String,
    /// What the server is bound to.
    pub bind_host: String,
    pub port: u16,
    pub base_url: String,
    pub loopback: bool,
    /// Router models: `loaded`, `unloaded`, `loading`…; a standalone run is `loaded`.
    pub status: String,
    pub slots_busy: Option<usize>,
    pub slots_total: Option<usize>,
    /// Per-slot context from `/slots` (MAXTOK for DiffusionGemma).
    pub n_ctx: Option<u64>,
    pub sampling: Option<Sampling>,
    pub enable_thinking: Option<bool>,
    pub diffusion: Option<DiffusionCfg>,
    pub model_path: Option<PathBuf>,
    pub has_api_key: bool,
}

/// A target from what is known about it (pure).
pub fn target_for(
    state: &RunState,
    profile: Option<&Profile>,
    model: Option<&str>,
    status: &str,
    sample: Option<&LiveSample>,
) -> Target {
    let model_id = model.map(str::to_string).unwrap_or_else(|| state.alias.clone());
    let label = profile.map(|p| if p.name.trim().is_empty() { p.id.clone() } else { p.name.clone() }).unwrap_or_else(|| {
        model.map(str::to_string).unwrap_or_else(|| state.profile_id.clone())
    });
    let slots = sample.filter(|s| s.error.is_none()).map(|s| &s.slots);
    let engine = if model.is_some() { Engine::LlamaServer } else { state.engine };
    Target {
        key: match model {
            Some(m) => format!("{}/{m}", state.profile_id),
            None => state.profile_id.clone(),
        },
        run: state.profile_id.clone(),
        model: model.map(str::to_string),
        engine,
        label,
        profile_id: profile.map(|p| p.id.clone()),
        model_id,
        host: connect_host(&state.host).to_string(),
        bind_host: state.host.clone(),
        port: state.port,
        base_url: base_url(&state.host, state.port),
        loopback: is_loopback(&state.host),
        status: status.to_string(),
        slots_busy: slots.map(|s| s.iter().filter(|x| x.is_processing).count()),
        slots_total: slots.map(Vec::len),
        n_ctx: slots.and_then(|s| s.first()).map(|x| x.n_ctx).filter(|n| *n > 0),
        sampling: profile.filter(|p| p.engine.is_llama_server()).map(|p| p.sampling.clone()),
        enable_thinking: profile.and_then(|p| p.chat.enable_thinking),
        diffusion: profile.filter(|p| p.engine.is_diffusion()).map(Profile::diffusion_effective),
        model_path: profile.map(|p| p.model.path.clone()),
        has_api_key: profile.is_some_and(requires_api_key),
    }
}

/// Every chat target among the live runs: one per standalone server, one
/// per router model (loaded or not; the router loads one on first use).
/// Polls each: the router's model list and `/slots` of what is loaded
/// (with the profile's API key, and never loading a model).
pub fn targets(runs: &[AttachedRun], profiles: &[Profile]) -> Vec<Target> {
    let mut out = Vec::new();
    for r in runs.iter().filter(|r| r.alive) {
        let s = &r.state;
        let host = connect_host(&s.host);
        if s.profile_id == crate::router::ROUTER_ID {
            let Ok(models) = crate::router::models(host, s.port) else { continue };
            for m in models {
                let member = profiles.iter().find(|p| crate::router::model_id(p) == m.id);
                let key = member.and_then(api_key);
                let sample =
                    (m.status == "loaded").then(|| crate::live::sample_with_key(host, s.port, Some(&m.id), key.as_deref()));
                out.push(target_for(s, member, Some(&m.id), &m.status, sample.as_ref()));
            }
        } else {
            let profile = profiles.iter().find(|p| p.id == s.profile_id);
            let key = profile.and_then(api_key);
            let sample = crate::live::sample_with_key(host, s.port, None, key.as_deref());
            out.push(target_for(s, profile, None, "loaded", Some(&sample)));
        }
    }
    out
}

#[cfg(test)]
mod tests;
