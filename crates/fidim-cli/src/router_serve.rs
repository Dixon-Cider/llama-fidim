//! `fidim router serve`: one OpenAI-compatible port in front of several
//! SGLang servers, with a llama-server-shaped live view for Llama FIDIM.
//!
//! This is the Rust port of `crates/fidim-core/assets/sglang/model_router.py`
//! (the spec; its behaviour is reproduced request for request so FIDIM's
//! Running tab works unchanged against SGLang, and the router needs no
//! Python). Every request is routed by the JSON `model` field (aliases
//! resolve to a route, unknown models fall back to `--default`), streamed
//! through (SSE included), and the tracked completion endpoints update
//! per-model "slots" the way llama-server reports them:
//!
//!   GET /models          llama-server router shape (status.value = loaded)
//!   GET /slots?model=X   per-slot phase, prompt/decode progress, text tails
//!   GET /metrics?model=X llamacpp:* counters plus the backend's sglang:* lines
//!   GET /v1/models       merged backend lists, ids replaced by route names
//!   GET /health          200 only when every backend is healthy (cached)
//!   GET /fidim/state     everything above as one JSON
//!
//! Pools: `~/.fidim/router-remotes.json` (or `--remotes-file`) can add remote
//! OpenAI-compatible servers to a route (e.g. a DGX Spark behind `dd`). The
//! router then picks a member per chat/completion request: one conversation
//! stays on one member (prompt-cache reuse), new conversations go local
//! while the local server has a free slot and spill to a remote with room,
//! and a remote never gets more than its `max_inflight`. A remote that
//! refuses or fails before anything reached the client is retried locally.
//! See `RemoteSpec` and `Router::try_pick`.
//!
//! HTTP/1.1 over hyper; TLS (rustls) only toward remote members. The layout follows the Python file:
//! `Slot`/`ModelStats` (per-model bookkeeping), `SseTracker` (feeds streamed
//! chunks into a slot and strips the injected usage chunk), the pure
//! helpers (`prompt_text`, `aggregate_backend_metrics`), then `Router` (the
//! HTTP handlers) and `run`.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::convert::Infallible;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use anyhow::{bail, Context as _};
use bytes::Bytes;
use http::{header, HeaderMap, HeaderValue, Method, Request, Response, StatusCode, Uri};
use http_body::{Body, Frame, SizeHint};
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full, Limited};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_rustls::HttpsConnector;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::client::legacy::Client;
use hyper_util::rt::{TokioExecutor, TokioIo};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::net::TcpListener;

/// Headers that describe one hop and are never forwarded (either way).
const HOP: [&str; 5] = ["host", "content-length", "transfer-encoding", "connection", "keep-alive"];
/// POST paths whose requests occupy a slot.
const TRACKED: [&str; 3] = ["/v1/chat/completions", "/v1/completions", "/generate"];
/// Chars of prompt/generated text kept in a slot view.
const TEXT_TAIL: usize = 6000;
/// aiohttp's `client_max_size` in the Python router.
const MAX_REQUEST_BODY: usize = 256 * 1024 * 1024;
/// A non-streamed tracked reply is parsed for usage/content up to this size.
const MAX_COLLECT: usize = 4 * 1024 * 1024;

const JSON_CT: &str = "application/json; charset=utf-8";
const TEXT_CT: &str = "text/plain; charset=utf-8";

type OutBody = BoxBody<Bytes, hyper::Error>;
type HttpClient = Client<HttpsConnector<HttpConnector>, Full<Bytes>>;

/// Seconds since the router started (monotonic; the Python uses
/// `time.time()`, but only differences are ever exposed).
fn now() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

fn timestamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    // Civil date from days since the epoch (Howard Hinnant's algorithm); no
    // chrono dependency for one log line.
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}", rem / 3600, (rem / 60) % 60, rem % 60)
}

fn log_line(level: &str, msg: &str) {
    eprintln!("{} {level} {msg}", timestamp());
}

// ------------------------------------------------------ Python helpers ----

/// Python truthiness of a JSON value (None/False/0/""/[]/{} are false).
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// A non-empty string value (`x or ""` semantics for string fields).
fn truthy_str(v: Option<&Value>) -> Option<&str> {
    v.and_then(Value::as_str).filter(|s| !s.is_empty())
}

/// Python `int(x)` for a JSON number or numeric string; None when it would raise.
fn py_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f.trunc() as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        Value::Bool(b) => Some(*b as i64),
        _ => None,
    }
}

/// Python `repr(float)`: shortest round-trip digits, `.0` on integral
/// values, exponent form outside 1e-4..1e16.
pub fn py_float(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = format!("{v:e}");
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-4..16).contains(&exp) {
        let s = format!("{v}");
        if s.contains('.') { s } else { format!("{s}.0") }
    } else {
        format!("{mant}e{}{:02}", if exp < 0 { "-" } else { "+" }, exp.abs())
    }
}

/// Python `json.dumps(v)` with the default separators and `ensure_ascii`
/// (the token estimate counts its characters, so the spacing matters).
pub fn py_dumps(v: &Value) -> String {
    let mut out = String::new();
    py_dumps_into(v, &mut out);
    out
}

fn py_dumps_into(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else {
                let f = n.as_f64().unwrap_or(0.0);
                out.push_str(&match f {
                    f if f.is_nan() => "NaN".to_string(),
                    f if f.is_infinite() && f > 0.0 => "Infinity".to_string(),
                    f if f.is_infinite() => "-Infinity".to_string(),
                    f => py_float(f),
                });
            }
        }
        Value::String(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    '\u{8}' => out.push_str("\\b"),
                    '\u{c}' => out.push_str("\\f"),
                    c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                        let mut buf = [0u16; 2];
                        for unit in c.encode_utf16(&mut buf) {
                            out.push_str(&format!("\\u{unit:04x}"));
                        }
                    }
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_dumps_into(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_dumps_into(&Value::String(k.clone()), out);
                out.push_str(": ");
                py_dumps_into(x, out);
            }
            out.push('}');
        }
    }
}

fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Keep the last `n` chars of `s` (char-safe).
fn tail(s: &str, n: usize) -> &str {
    let count = char_len(s);
    if count <= n {
        s
    } else {
        let skip = count - n;
        let (idx, _) = s.char_indices().nth(skip).unwrap_or((s.len(), ' '));
        &s[idx..]
    }
}

/// A message's `content`: the string, or the `text` parts of a list joined by spaces.
fn text_of(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a
            .iter()
            .filter(|x| x.is_object() && x.get("type").and_then(Value::as_str) == Some("text"))
            .map(|x| x.get("text").and_then(Value::as_str).unwrap_or(""))
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Last user turn (or the raw prompt) and a token estimate for the whole
/// request (chars / 4, at least 1).
pub fn prompt_text(body: &Map<String, Value>) -> (String, u64) {
    let dumps_or_empty = |v: Option<&Value>| if truthy(v) { py_dumps(v.unwrap()) } else { "\"\"".to_string() };
    if let Some(msgs) = body.get("messages").filter(|m| truthy(Some(m))).and_then(Value::as_array) {
        let mut total = 0usize;
        for m in msgs {
            let (content, tool_calls) = match m.as_object() {
                Some(o) => (o.get("content"), o.get("tool_calls")),
                None => (None, None),
            };
            total += char_len(&text_of(content)) + dumps_or_empty(tool_calls).len();
        }
        total += dumps_or_empty(body.get("tools")).len();
        let last = msgs
            .iter()
            .rev()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
            .map(|m| text_of(m.get("content")))
            .unwrap_or_default();
        return (last, (total / 4).max(1) as u64);
    }
    let p = body.get("prompt").filter(|v| truthy(Some(v))).or_else(|| body.get("text").filter(|v| truthy(Some(v))));
    let p = match p {
        Some(Value::String(s)) => s.clone(),
        Some(v) => py_dumps(v),
        None => String::new(),
    };
    let n = (char_len(&p) / 4).max(1) as u64;
    (p, n)
}

// ------------------------------------------------------------- slots ----

/// One in-flight (or finished) request, in llama-server's slot vocabulary.
#[derive(Debug, Clone)]
pub struct Slot {
    pub id: usize,
    pub id_task: i64,
    pub busy: bool,
    pub t0: f64,
    /// When the first generated text arrived (prefill ended).
    pub t_first: Option<f64>,
    pub t_end: f64,
    pub n_prompt: u64,
    pub n_prompt_exact: bool,
    /// The raw chars/4 estimate `n_prompt` started from (before the
    /// member's learned scale), to learn the scale once usage is exact.
    pub n_prompt_est: u64,
    pub processed: u64,
    pub prompt: String,
    pub generated: String,
    pub n_decoded: u64,
    pub n_decoded_exact: bool,
    pub max_tokens: i64,
    pub gen_chars: u64,
}

impl Slot {
    pub fn new(id: usize) -> Self {
        Slot {
            id,
            id_task: -1,
            busy: false,
            t0: 0.0,
            t_first: None,
            t_end: 0.0,
            n_prompt: 0,
            n_prompt_exact: false,
            n_prompt_est: 0,
            processed: 0,
            prompt: String::new(),
            generated: String::new(),
            n_decoded: 0,
            n_decoded_exact: false,
            max_tokens: -1,
            gen_chars: 0,
        }
    }

    /// Decoded tokens as reported: exact when known, else the chunk count
    /// or a chars/4 estimate, whichever is larger, while the slot is busy.
    fn decoded_now(&self) -> u64 {
        if self.n_decoded_exact || !self.busy {
            self.n_decoded
        } else {
            self.n_decoded.max(self.gen_chars / 4)
        }
    }

    /// `speculative`: the member runs a draft (SGLang `speculative_algorithm`).
    pub fn view(&self, n_ctx: u64, speculative: bool) -> Value {
        let decoded = self.decoded_now();
        json!({
            "id": self.id,
            "id_task": self.id_task,
            "n_ctx": n_ctx,
            "is_processing": self.busy,
            "speculative": speculative,
            "n_prompt_tokens": self.n_prompt,
            "n_prompt_tokens_processed": if self.busy && decoded == 0 { self.processed } else { self.n_prompt },
            "n_prompt_tokens_cache": 0,
            // false while n_prompt_tokens is still the estimate (usage not in yet)
            "n_prompt_tokens_exact": self.n_prompt_exact || self.n_prompt == 0,
            "next_token": [{
                "has_next_token": self.busy,
                "n_decoded": decoded,
                "n_remain": if self.max_tokens > 0 { self.max_tokens - decoded as i64 } else { -1 },
            }],
            "prompt": tail(&self.prompt, TEXT_TAIL),
            "generated": tail(&self.generated, TEXT_TAIL),
        })
    }

    /// Text generated for this request (a streamed delta or the final
    /// message): starts the decode clock on the first one.
    fn add_text(&mut self, text: &str) {
        if self.t_first.is_none() {
            self.t_first = Some(now());
            self.processed = self.n_prompt;
        }
        self.generated.push_str(text);
        self.gen_chars += char_len(text) as u64;
        self.n_decoded += 1;
    }

    fn apply_usage(&mut self, usage: &Map<String, Value>) {
        if truthy(usage.get("prompt_tokens")) {
            if let Some(n) = usage.get("prompt_tokens").and_then(py_int) {
                self.n_prompt = n.max(0) as u64;
                self.n_prompt_exact = true;
            }
        }
        if let Some(v) = usage.get("completion_tokens").filter(|v| !v.is_null()) {
            if let Some(n) = py_int(v) {
                self.n_decoded = n.max(0) as u64;
                self.n_decoded_exact = true;
            }
        }
    }
}

/// Per-model bookkeeping: slots, totals and what the backend told us about itself.
#[derive(Debug)]
pub struct ModelStats {
    pub name: String,
    pub url: String,
    pub slots: Vec<Slot>,
    pub n_ctx: u64,
    pub n_slots: usize,
    pub tokens_predicted_total: u64,
    pub predicted_seconds: f64,
    pub prompt_tokens_total: u64,
    pub prompt_seconds: f64,
    pub requests_total: u64,
    /// The member runs speculative decoding (from `/get_server_info`).
    pub speculative: bool,
    /// Real prompt tokens per chars/4-estimated token, learned from the
    /// usage of finished requests (None until one reported it).
    pub prompt_scale: Option<f64>,
    task_counter: i64,
    info_at: Option<f64>,
}

impl ModelStats {
    pub fn new(name: &str, url: &str) -> Self {
        ModelStats {
            name: name.into(),
            url: url.into(),
            slots: vec![],
            n_ctx: 0,
            n_slots: 0,
            tokens_predicted_total: 0,
            predicted_seconds: 0.0,
            prompt_tokens_total: 0,
            prompt_seconds: 0.0,
            requests_total: 0,
            speculative: false,
            prompt_scale: None,
            task_counter: 0,
            info_at: None,
        }
    }

    /// Whether `/get_server_info` should be (re)read: never read, or older than 60 s.
    fn info_stale(&self) -> bool {
        self.slots.is_empty() || self.info_at.map(|t| now() - t >= 60.0).unwrap_or(true)
    }

    /// Apply a `/get_server_info` reply (None: the request failed, keep the
    /// slot count we have, at least one).
    fn apply_info(&mut self, info: Option<&Value>) {
        let want = match info {
            Some(d) => {
                let pick = |k: &str| d.get(k).filter(|v| truthy(Some(v))).and_then(py_int);
                // What one request can hold: the KV pool bounds it below the
                // model's nominal window (SGLang: max_req_input_len < context_length).
                self.n_ctx = pick("max_req_input_len")
                    .or_else(|| pick("context_length"))
                    .or_else(|| pick("max_total_num_tokens"))
                    .unwrap_or(0)
                    .max(0) as u64;
                self.speculative = truthy(d.get("speculative_algorithm"));
                // The scheduler's own cap wins over the flag: on a hybrid model
                // SGLang lowers it to state slots / 5 (qwenMoE: flag 8, runs 4).
                let effective = d
                    .get("internal_states")
                    .and_then(|v| v.as_array())
                    .filter(|a| !a.is_empty())
                    .map(|a| a.iter().filter_map(|st| st.get("effective_max_running_requests_per_dp").and_then(py_int)).sum::<i64>())
                    .filter(|n| *n > 0);
                effective.or_else(|| pick("max_running_requests")).unwrap_or(1).max(0) as usize
            }
            None => self.n_slots.max(1),
        };
        while self.slots.len() < want {
            self.slots.push(Slot::new(self.slots.len()));
        }
        // Drop idle slots past the count (a busy one stays until it finishes).
        while self.slots.len() > want.max(1) && self.slots.last().is_some_and(|s| !s.busy) {
            self.slots.pop();
        }
        self.n_slots = want;
        self.info_at = Some(now());
    }

    /// Occupy a free slot (or grow) for a new request; returns its index.
    pub fn take(&mut self) -> usize {
        let idx = match self.slots.iter().position(|s| !s.busy) {
            Some(i) => i,
            None => {
                self.slots.push(Slot::new(self.slots.len()));
                self.slots.len() - 1
            }
        };
        self.task_counter += 1;
        self.requests_total += 1;
        let id_task = self.task_counter;
        let s = &mut self.slots[idx];
        *s = Slot::new(s.id);
        s.busy = true;
        s.id_task = id_task;
        s.t0 = now();
        idx
    }

    /// The request in slot `idx` is over: fold its numbers into the totals.
    pub fn release(&mut self, idx: usize) {
        let t = now();
        let Some(s) = self.slots.get_mut(idx) else { return };
        s.t_end = t;
        s.busy = false;
        if !s.n_decoded_exact {
            s.n_decoded = s.n_decoded.max(s.gen_chars / 4);
        }
        self.tokens_predicted_total += s.n_decoded;
        if let Some(t_first) = s.t_first {
            self.predicted_seconds += (t - t_first).max(0.0);
            self.prompt_seconds += (t_first - s.t0).max(0.0);
        }
        self.prompt_tokens_total += s.n_prompt;
        if s.n_prompt_exact && s.n_prompt_est > 0 && s.n_prompt > 0 {
            let r = (s.n_prompt as f64 / s.n_prompt_est as f64).clamp(0.25, 4.0);
            self.prompt_scale = Some(match self.prompt_scale {
                Some(p) => 0.7 * p + 0.3 * r,
                None => r,
            });
        }
    }

    /// A new request's prompt size before the backend reports it: the
    /// chars/4 estimate, scaled by what this member's usage has shown.
    pub fn estimate_prompt(&self, est: u64) -> u64 {
        ((est as f64 * self.prompt_scale.unwrap_or(1.0)).round() as u64).max(1)
    }

    /// Slots still prefilling (busy, nothing generated yet).
    fn prefilling(&self) -> Vec<usize> {
        self.slots.iter().enumerate().filter(|(_, s)| s.busy && s.t_first.is_none()).map(|(i, _)| i).collect()
    }

    /// SGLang has no per-request prefill counter; with one request
    /// prefilling, the backend's pending-token count is that request's
    /// remaining prefill. With several, call each half done.
    fn apply_prefill(&mut self, pending: u64) {
        let pre = self.prefilling();
        if pre.len() == 1 {
            let s = &mut self.slots[pre[0]];
            s.processed = s.processed.max(s.n_prompt.saturating_sub(pending));
        } else {
            for i in pre {
                let s = &mut self.slots[i];
                s.processed = s.processed.max((s.n_prompt as f64 * 0.5) as u64);
            }
        }
    }

    /// The llamacpp:* family. llama-server counts live: what in-flight slots
    /// have produced so far is included, so a 1 Hz sampler sees the rate
    /// while a request is still decoding.
    pub fn metrics_text(&self) -> String {
        let t = now();
        let mut live_tokens = 0u64;
        let mut live_secs = 0.0f64;
        let mut live_prompt = 0u64;
        for s in &self.slots {
            if let (true, Some(t_first)) = (s.busy, s.t_first) {
                live_tokens += if s.n_decoded_exact { s.n_decoded } else { s.n_decoded.max(s.gen_chars / 4) };
                live_secs += t - t_first;
                live_prompt += s.n_prompt;
            }
        }
        let processing = self.slots.iter().filter(|s| s.busy).count();
        format!(
            "# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.\n\
             # TYPE llamacpp:tokens_predicted_total counter\n\
             llamacpp:tokens_predicted_total {}\n\
             # TYPE llamacpp:tokens_predicted_seconds_total counter\n\
             llamacpp:tokens_predicted_seconds_total {:.3}\n\
             # TYPE llamacpp:predicted_tokens_seconds gauge\n\
             llamacpp:predicted_tokens_seconds {:.3}\n\
             # TYPE llamacpp:prompt_tokens_total counter\n\
             llamacpp:prompt_tokens_total {}\n\
             # TYPE llamacpp:prompt_seconds_total counter\n\
             llamacpp:prompt_seconds_total {:.3}\n\
             # TYPE llamacpp:requests_processing gauge\n\
             llamacpp:requests_processing {}\n\
             # TYPE llamacpp:n_decode_total counter\n\
             llamacpp:n_decode_total {}\n",
            self.tokens_predicted_total + live_tokens,
            self.predicted_seconds + live_secs,
            // llama-server's gauge is tok/s (the seconds are the _total above)
            if self.predicted_seconds + live_secs > 0.0 {
                (self.tokens_predicted_total + live_tokens) as f64 / (self.predicted_seconds + live_secs)
            } else {
                0.0
            },
            self.prompt_tokens_total + live_prompt,
            self.prompt_seconds,
            processing,
            self.requests_total,
        )
    }

    pub fn views(&self) -> Vec<Value> {
        self.slots.iter().map(|s| s.view(self.n_ctx, self.speculative)).collect()
    }

    pub fn state_json(&self) -> Value {
        json!({
            "url": self.url,
            "n_ctx": self.n_ctx,
            "slots": self.views(),
            "tokens_predicted_total": self.tokens_predicted_total,
            "predicted_seconds": self.predicted_seconds,
            "prompt_tokens_total": self.prompt_tokens_total,
            "prompt_seconds": self.prompt_seconds,
            "requests_total": self.requests_total,
        })
    }
}

// ------------------------------------------------------- SSE tracker ----

/// Feed raw SSE bytes; updates the slot from OpenAI-style chunks and strips
/// a usage-only chunk the client did not ask for.
#[derive(Debug)]
pub struct SseTracker {
    buf: Vec<u8>,
    want_usage: bool,
}

impl SseTracker {
    pub fn new(want_usage: bool) -> Self {
        SseTracker { buf: Vec::new(), want_usage }
    }

    /// Bytes from the backend in; the bytes the client gets out (whole
    /// events only, a partial event waits for the next chunk).
    pub fn feed(&mut self, chunk: &[u8], slot: &mut Slot) -> Vec<u8> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some(pos) = self.buf.windows(2).position(|w| w == b"\n\n") {
            let rest = self.buf.split_off(pos + 2);
            let mut event = std::mem::replace(&mut self.buf, rest);
            event.truncate(pos);
            if self.consume(&event, slot) {
                out.extend_from_slice(&event);
                out.extend_from_slice(b"\n\n");
            }
        }
        out
    }

    /// End of stream: whatever is left without its terminating blank line.
    pub fn flush(&mut self, slot: &mut Slot) -> Vec<u8> {
        let rest = std::mem::take(&mut self.buf);
        if !rest.is_empty() && self.consume(&rest, slot) {
            rest
        } else {
            Vec::new()
        }
    }

    /// Update the slot from one event; false means "do not pass it on".
    fn consume(&self, event: &[u8], slot: &mut Slot) -> bool {
        for line in event.split(|b| *b == b'\n') {
            let Some(payload) = line.strip_prefix(b"data:") else { continue };
            let payload = payload.trim_ascii();
            if payload == b"[DONE]" {
                return true;
            }
            let Ok(d) = serde_json::from_slice::<Value>(payload) else { return true };
            let Some(d) = d.as_object() else { return true };
            let usage = d.get("usage");
            let choices = d.get("choices").filter(|c| truthy(Some(c)));
            if let Some(list) = choices.and_then(Value::as_array) {
                for c in list.iter().filter_map(Value::as_object) {
                    let delta = c.get("delta").and_then(Value::as_object);
                    let mut text = String::new();
                    if let Some(delta) = delta {
                        text.push_str(truthy_str(delta.get("content")).unwrap_or(""));
                        text.push_str(truthy_str(delta.get("reasoning_content")).unwrap_or(""));
                    }
                    if text.is_empty() {
                        if let Some(t) = truthy_str(c.get("text")) {
                            text.push_str(t);
                        }
                    }
                    if !text.is_empty() {
                        slot.add_text(&text);
                    }
                }
            }
            if let Some(u) = usage.filter(|u| truthy(Some(u))).and_then(Value::as_object) {
                slot.apply_usage(u);
                if choices.is_none() && !self.want_usage {
                    return false; // our injected usage chunk: keep it from the client
                }
            }
        }
        true
    }
}

// -------------------------------------------------- backend metrics ----

/// The backend's own Prometheus text: only `sglang:` lines, labels
/// stripped, histogram buckets dropped, `_sum`/`_count`/`_total` names
/// summed across label groups and everything else last-wins. Output order
/// is first appearance, values printed like Python floats.
pub fn aggregate_backend_metrics(raw: &str) -> String {
    let mut order: Vec<String> = Vec::new();
    let mut agg: HashMap<String, f64> = HashMap::new();
    for line in raw.lines() {
        if !line.starts_with("sglang:") || line.contains("_bucket{") {
            continue;
        }
        let first = line.split(' ').next().unwrap_or("");
        let name = if first.contains('{') { line.split('{').next().unwrap_or("") } else { first };
        let value = line.rsplit(' ').next().unwrap_or("");
        let Ok(v) = value.parse::<f64>() else { continue };
        let summed = name.ends_with("_sum") || name.ends_with("_count") || name.ends_with("_total");
        match agg.get_mut(name) {
            Some(cur) => *cur = if summed { *cur + v } else { v },
            None => {
                order.push(name.to_string());
                agg.insert(name.to_string(), v);
            }
        }
    }
    let mut out = String::new();
    for k in order {
        out.push_str(&k);
        out.push(' ');
        out.push_str(&py_float(agg[&k]));
        out.push('\n');
    }
    out
}

// ---------------------------------------------------- tracked bodies ----

enum TrackMode {
    /// Streamed reply: feed every chunk through the tracker.
    Sse(SseTracker),
    /// Non-streamed reply: pass through, keep a copy to parse at the end.
    Collect(Vec<u8>),
}

/// A taken slot that is released when dropped, unless it was handed on to
/// the response body (`disarm`). hyper drops the proxy future when the
/// client disconnects while the backend is still working on the response
/// headers (a non-streamed request the client timed out on); before this
/// guard such a slot stayed busy forever, showing phantom in-flight requests
/// and keeping the prefill-progress polling of `/get_load` running.
struct SlotGuard {
    stats: Arc<Mutex<ModelStats>>,
    idx: usize,
    armed: bool,
}

impl SlotGuard {
    fn new(stats: Arc<Mutex<ModelStats>>, idx: usize) -> Self {
        SlotGuard { stats, idx, armed: true }
    }

    /// Hand the slot on to `TrackedBody`, which releases it itself.
    fn disarm(mut self) -> (Arc<Mutex<ModelStats>>, usize) {
        self.armed = false;
        (self.stats.clone(), self.idx)
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if self.armed {
            Router::lock_stats(&self.stats).release(self.idx);
        }
    }
}

/// The backend's response body on its way to the client, updating the
/// slot as it flows and releasing the slot when it ends (or is dropped:
/// a client that went away releases too).
struct TrackedBody {
    /// None: the whole reply is in `prefix` (a buffered remote reply).
    inner: Option<Incoming>,
    /// Bytes already read from the backend (a remote's first SSE event, or
    /// its whole non-streamed reply), passed on before `inner`.
    prefix: Option<Bytes>,
    mode: TrackMode,
    stats: Arc<Mutex<ModelStats>>,
    slot: usize,
    status_ok: bool,
    done: bool,
    released: bool,
}

impl TrackedBody {
    /// One chunk of the backend body: into the slot, and what the client gets.
    fn track(&mut self, data: Bytes) -> Bytes {
        match &mut self.mode {
            TrackMode::Collect(buf) => {
                if buf.len() < MAX_COLLECT {
                    buf.extend_from_slice(&data);
                }
                data
            }
            TrackMode::Sse(tracker) => {
                let mut st = self.stats.lock().unwrap_or_else(|e| e.into_inner());
                match st.slots.get_mut(self.slot) {
                    Some(slot) => Bytes::from(tracker.feed(&data, slot)),
                    None => data,
                }
            }
        }
    }

    /// End of the backend body: flush the tracker (its tail is returned so
    /// the client gets it), parse a collected reply, release the slot.
    fn finish(&mut self) -> Vec<u8> {
        if self.released {
            return Vec::new();
        }
        self.released = true;
        let stats = self.stats.clone();
        let mut st = stats.lock().unwrap_or_else(|e| e.into_inner());
        let idx = self.slot;
        let tail = match &mut self.mode {
            TrackMode::Sse(tracker) => match st.slots.get_mut(idx) {
                Some(slot) => tracker.flush(slot),
                None => Vec::new(),
            },
            TrackMode::Collect(buf) => {
                if self.status_ok {
                    if let Some(slot) = st.slots.get_mut(idx) {
                        apply_final_reply(slot, buf);
                    }
                }
                Vec::new()
            }
        };
        st.release(idx);
        tail
    }
}

/// A non-streamed completion reply: text and usage from the JSON.
fn apply_final_reply(slot: &mut Slot, body: &[u8]) {
    let Ok(d) = serde_json::from_slice::<Value>(body) else { return };
    let Some(d) = d.as_object() else { return };
    let empty = Map::new();
    let ch = d
        .get("choices")
        .filter(|c| truthy(Some(c)))
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .and_then(Value::as_object)
        .unwrap_or(&empty);
    let msg = ch.get("message").and_then(Value::as_object).unwrap_or(&empty);
    let content = truthy_str(msg.get("content"))
        .or_else(|| truthy_str(ch.get("text")))
        .or_else(|| truthy_str(d.get("text")))
        .unwrap_or("");
    slot.generated = format!("{}{}", truthy_str(msg.get("reasoning_content")).unwrap_or(""), content);
    slot.gen_chars = char_len(&slot.generated) as u64;
    // no stream: the whole wall time counts as decode (no prefill split)
    if slot.t_first.is_none() {
        slot.t_first = Some(slot.t0);
    }
    if let Some(u) = d.get("usage").and_then(Value::as_object) {
        // The Python applies completion_tokens before prompt_tokens; the
        // order does not matter, both are independent fields.
        slot.apply_usage(u);
    }
}

impl Body for TrackedBody {
    type Data = Bytes;
    type Error = hyper::Error;

    fn poll_frame(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Bytes>, hyper::Error>>> {
        let this = self.get_mut();
        loop {
            if this.done {
                return Poll::Ready(None);
            }
            if let Some(data) = this.prefix.take() {
                let out = this.track(data);
                if out.is_empty() {
                    continue;
                }
                return Poll::Ready(Some(Ok(Frame::data(out))));
            }
            let Some(inner) = this.inner.as_mut() else {
                this.done = true;
                let tail = this.finish();
                return if tail.is_empty() { Poll::Ready(None) } else { Poll::Ready(Some(Ok(Frame::data(tail.into())))) };
            };
            match Pin::new(inner).poll_frame(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    this.done = true;
                    let tail = this.finish();
                    return if tail.is_empty() { Poll::Ready(None) } else { Poll::Ready(Some(Ok(Frame::data(tail.into())))) };
                }
                Poll::Ready(Some(Err(e))) => {
                    this.done = true;
                    this.finish();
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(Some(Ok(frame))) => match frame.into_data() {
                    Err(other) => return Poll::Ready(Some(Ok(other))),
                    Ok(data) => {
                        let out = this.track(data);
                        if out.is_empty() {
                            continue;
                        }
                        return Poll::Ready(Some(Ok(Frame::data(out))));
                    }
                },
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        self.done
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::default()
    }
}

impl Drop for TrackedBody {
    fn drop(&mut self) {
        self.finish();
    }
}

// ------------------------------------------------------------ router ----

/// What `fidim router serve` was told on the command line.
#[derive(Debug, Clone)]
pub struct Opts {
    pub host: String,
    pub port: u16,
    /// (model name, base URL) in command-line order.
    pub routes: Vec<(String, String)>,
    /// (alias, model name).
    pub aliases: Vec<(String, String)>,
    pub default: String,
    /// Remote members that add capacity to a route (router-remotes.json).
    pub remotes: Vec<RemoteSpec>,
    /// Seconds a conversation waits for its own member's slot before it may
    /// move (a move costs one full prefill elsewhere).
    pub sticky_wait: f64,
}

/// One remote OpenAI-compatible server in a route's pool, from
/// `router-remotes.json`:
///
/// ```json
/// {"sticky_wait_secs": 15,
///  "remotes": [{"name": "spark", "pool": "dd",
///               "base_url": "https://spark.example.com/v1", "served_model": "qwen3.8-27b",
///               "api_key_file": "~/.secrets/spark_key.txt", "max_inflight": 2,
///               "context_length": 262144}]}
/// ```
///
/// The remote is also listed by `name` in `/models`, so FIDIM's Running tab
/// shows its slots (`/slots?model=<name>`) and counters
/// (`/metrics?model=<name>`): what this router sent there, plus the queue
/// the remote's own `/health` reports (`inflight`, `waiting`; every client
/// of the remote, not just this router). A request for model `<name>` goes
/// to that remote only (no local fallback).
///
/// Requests for `pool` (aliases included) may go there with `model` set to
/// `served_model` and `Authorization: Bearer <key>`; only
/// `/v1/chat/completions` and `/v1/completions` are pooled. A client sends
/// `x-fidim-local-only: 1` to keep a request local, and `x-fidim-session` to
/// name its conversation (else the system prompt and first user message do).
/// `touch ~/.fidim/remote-<name>.off` takes a remote out until removed.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RemoteSpec {
    pub name: String,
    pub pool: String,
    pub base_url: String,
    pub served_model: String,
    #[serde(default)]
    pub api_key_file: Option<String>,
    /// Requests the remote takes at once (its gateway refuses more).
    #[serde(default = "one")]
    pub max_inflight: usize,
    /// Its context window, for the live view (`n_ctx`).
    #[serde(default)]
    pub context_length: Option<u64>,
}

fn one() -> usize {
    1
}

#[derive(Debug, Default, Deserialize)]
struct RemotesFile {
    #[serde(default)]
    remotes: Vec<RemoteSpec>,
    #[serde(default)]
    sticky_wait_secs: Option<f64>,
}

/// `~/` at the start of a path, expanded (key files are written by hand).
fn expand_home(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), std::env::var_os("HOME")) {
        (Some(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => PathBuf::from(p),
    }
}

impl Opts {
    /// The `--route name=url`, `--alias alias=name`, `--default name` flags.
    pub fn parse_args(host: &str, port: u16, route: &[String], alias: &[String], default: &str) -> anyhow::Result<Self> {
        let mut routes: Vec<(String, String)> = Vec::new();
        for r in route {
            let (name, url) = r.split_once('=').with_context(|| format!("--route {r}: expected model=http://host:port"))?;
            let url = url.trim().trim_end_matches('/').to_string();
            let uri: Uri = url.parse().with_context(|| format!("--route {r}: bad URL"))?;
            if uri.scheme_str() != Some("http") || uri.host().is_none() {
                bail!("--route {r}: the URL must be http://host:port (plain HTTP, no path)");
            }
            routes.retain(|(n, _)| n != name);
            routes.push((name.to_string(), url));
        }
        if routes.is_empty() {
            bail!("no --route given");
        }
        let mut aliases: Vec<(String, String)> = Vec::new();
        for a in alias {
            let (k, v) = a.split_once('=').with_context(|| format!("--alias {a}: expected alias=model"))?;
            aliases.retain(|(n, _)| n != k);
            aliases.push((k.to_string(), v.to_string()));
        }
        if !routes.iter().any(|(n, _)| n == default) {
            bail!("--default {default} is not one of the routes");
        }
        Ok(Opts { host: host.to_string(), port, routes, aliases, default: default.to_string(), remotes: vec![], sticky_wait: 15.0 })
    }

    /// Add the remote members in `path` when it exists. A bad file is
    /// logged and ignored: the router still serves its local routes.
    pub fn with_remotes_file(mut self, path: &Path) -> Self {
        if !path.exists() {
            return self;
        }
        match self.parse_remotes(path) {
            Ok(()) => {}
            Err(e) => {
                log_line("WARNING", &format!("{}: {e:#}; serving without remote members", path.display()));
                self.remotes.clear();
            }
        }
        self
    }

    fn parse_remotes(&mut self, path: &Path) -> anyhow::Result<()> {
        let text = std::fs::read_to_string(path).context("read")?;
        let f: RemotesFile = serde_json::from_str(&text).context("parse")?;
        for mut r in f.remotes {
            r.base_url = r.base_url.trim().trim_end_matches('/').to_string();
            let uri: Uri = r.base_url.parse().with_context(|| format!("remote {}: bad base_url", r.name))?;
            if !matches!(uri.scheme_str(), Some("http" | "https")) || uri.host().is_none() {
                bail!("remote {}: base_url must be http(s)://host[:port]/path", r.name);
            }
            if r.name.is_empty() || self.routes.iter().any(|(n, _)| *n == r.name) || self.remotes.iter().any(|x| x.name == r.name) {
                bail!("remote `{}`: the name must be unique and not a route", r.name);
            }
            if !self.routes.iter().any(|(n, _)| *n == r.pool) {
                bail!("remote {}: pool `{}` is not a route", r.name, r.pool);
            }
            if r.max_inflight == 0 || r.served_model.is_empty() {
                bail!("remote {}: max_inflight must be at least 1 and served_model set", r.name);
            }
            self.remotes.push(r);
        }
        if let Some(w) = f.sticky_wait_secs {
            self.sticky_wait = w.clamp(0.0, 300.0);
        }
        Ok(())
    }
}

/// A remote member at run time.
struct Remote {
    spec: RemoteSpec,
    key: Option<String>,
    /// Slots for the live view; `max_inflight` of them.
    stats: Arc<Mutex<ModelStats>>,
    /// Last `/models` probe listed `served_model`.
    healthy: AtomicBool,
    /// After a refused or failed request: not picked until then (router clock).
    cooling_until: Mutex<f64>,
    last_error: Mutex<String>,
    off_file: PathBuf,
    /// (when, inflight, waiting) from the remote's own `/health`.
    queue: Mutex<Option<(f64, f64, f64)>>,
    /// When FIDIM last asked for its metrics (drives `queue_poller`).
    viewed_at: Mutex<f64>,
}

/// How long a refused or failed remote sits out.
const REMOTE_COOLDOWN: f64 = 30.0;

impl Remote {
    fn new(spec: RemoteSpec, config_dir: &Path) -> Self {
        let key = spec.api_key_file.as_deref().and_then(|f| match std::fs::read_to_string(expand_home(f)) {
            Ok(k) if !k.trim().is_empty() => Some(k.trim().to_string()),
            Ok(_) => {
                log_line("WARNING", &format!("remote {}: key file {f} is empty", spec.name));
                None
            }
            Err(e) => {
                log_line("WARNING", &format!("remote {}: key file {f}: {e}", spec.name));
                None
            }
        });
        let mut st = ModelStats::new(&spec.name, &spec.base_url);
        st.slots = (0..spec.max_inflight).map(Slot::new).collect();
        st.n_slots = spec.max_inflight;
        st.n_ctx = spec.context_length.unwrap_or(0);
        let off_file = config_dir.join(format!("remote-{}.off", spec.name));
        Remote {
            spec,
            key,
            stats: Arc::new(Mutex::new(st)),
            healthy: AtomicBool::new(false),
            cooling_until: Mutex::new(0.0),
            last_error: Mutex::new(String::new()),
            off_file,
            queue: Mutex::new(None),
            viewed_at: Mutex::new(-1e9),
        }
    }

    fn busy(&self) -> usize {
        Router::lock_stats(&self.stats).slots.iter().filter(|s| s.busy).count()
    }

    fn cooling(&self) -> bool {
        now() < *self.cooling_until.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Up and switched on. A cooling remote (it refused recently) still
    /// serves the conversations already on it (moving them costs a prefill),
    /// but takes no new ones.
    fn usable(&self, sticky: bool) -> bool {
        self.healthy.load(Ordering::Relaxed) && (sticky || !self.cooling()) && !self.off_file.exists()
    }

    fn has_room(&self, sticky: bool) -> bool {
        self.usable(sticky) && self.busy() < self.spec.max_inflight
    }

    /// A request it took failed before reaching the client: sit out a while.
    fn fail(&self, reason: &str) {
        *self.cooling_until.lock().unwrap_or_else(|e| e.into_inner()) = now() + REMOTE_COOLDOWN;
        *self.last_error.lock().unwrap_or_else(|e| e.into_inner()) = reason.to_string();
        if reason.starts_with("HTTP 401") || reason.starts_with("HTTP 403") {
            self.healthy.store(false, Ordering::Relaxed);
        }
        log_line("WARNING", &format!("remote {}: {reason}; no new conversations for {REMOTE_COOLDOWN:.0} s", self.spec.name));
    }

    fn state_json(&self) -> Value {
        let mut v = Router::lock_stats(&self.stats).state_json();
        v["max_inflight"] = json!(self.spec.max_inflight);
        v["served_model"] = json!(self.spec.served_model);
        v["healthy"] = json!(self.healthy.load(Ordering::Relaxed));
        v["cooling"] = json!(self.cooling());
        v["switched_off"] = json!(self.off_file.exists());
        v["key_loaded"] = json!(self.key.is_some());
        v["last_error"] = json!(self.last_error.lock().unwrap_or_else(|e| e.into_inner()).clone());
        v
    }
}

/// Which member a conversation is on.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    Local,
    Remote(usize),
}

/// A member chosen for one request, its slot already taken.
#[derive(Debug, PartialEq)]
enum Pick {
    Local(usize),
    /// (remote index, slot index, why: new/sticky/spill)
    Remote(usize, usize, &'static str),
}

/// Conversation key -> (member, last used), router clock.
#[derive(Default)]
struct Sticky {
    map: HashMap<u64, (Target, f64)>,
}

const STICKY_TTL: f64 = 1800.0;
const STICKY_MAX: usize = 4096;

impl Sticky {
    fn get(&self, key: Option<u64>) -> Option<Target> {
        let t = now();
        key.and_then(|k| self.map.get(&k)).filter(|(_, at)| t - at < STICKY_TTL).map(|(tg, _)| *tg)
    }

    fn put(&mut self, key: Option<u64>, target: Target) {
        let Some(k) = key else { return };
        let t = now();
        if self.map.len() >= STICKY_MAX {
            self.map.retain(|_, (_, at)| t - *at < STICKY_TTL);
            if self.map.len() >= STICKY_MAX {
                // still full of live entries: drop the oldest quarter
                let mut ats: Vec<f64> = self.map.values().map(|(_, at)| *at).collect();
                ats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let cut = ats[ats.len() / 4];
                self.map.retain(|_, (_, at)| *at > cut);
            }
        }
        self.map.insert(k, (target, t));
    }
}

/// The conversation a request belongs to: the client's `x-fidim-session`,
/// else the model, system prompt and first user message (an agent's later
/// turns repeat them). None for a request with neither.
fn sticky_key(headers: &HeaderMap, body: &Map<String, Value>) -> Option<u64> {
    let mut h = DefaultHasher::new();
    if let Some(v) = headers.get("x-fidim-session").filter(|v| !v.is_empty()) {
        "session".hash(&mut h);
        v.as_bytes().hash(&mut h);
        return Some(h.finish());
    }
    let msgs = body.get("messages").and_then(Value::as_array)?;
    let first = |role: &str| msgs.iter().find(|m| m.get("role").and_then(Value::as_str) == Some(role)).map(|m| text_of(m.get("content")));
    let (system, user) = (first("system"), first("user"));
    if system.is_none() && user.is_none() {
        return None;
    }
    "prefix".hash(&mut h);
    body.get("model").and_then(Value::as_str).unwrap_or("").hash(&mut h);
    for part in [system, user].into_iter().flatten() {
        let cut: String = part.chars().take(8000).collect();
        cut.hash(&mut h);
    }
    Some(h.finish())
}

/// `x-fidim-local-only: 1|true|yes`.
fn local_only(headers: &HeaderMap) -> bool {
    headers
        .get("x-fidim-local-only")
        .and_then(|v| v.to_str().ok())
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

/// The remote URL for a pooled request path (`/v1` is in the remote's base).
fn remote_target(base: &str, uri: &Uri) -> String {
    let path = uri.path();
    let rest = path.strip_prefix("/v1").unwrap_or(path);
    match uri.query() {
        Some(q) => format!("{base}{rest}?{q}"),
        None => format!("{base}{rest}"),
    }
}

/// The first SSE event of a remote reply is an error (a gateway refusal
/// such as `{"error": {"type": "rate_limited"}}` sent with HTTP 200):
/// its message, else None.
fn sse_error(event: &[u8]) -> Option<String> {
    for line in event.split(|b| *b == b'\n') {
        let Some(payload) = line.strip_prefix(b"data:") else { continue };
        let Ok(d) = serde_json::from_slice::<Value>(payload.trim_ascii()) else { continue };
        if let Some(e) = d.get("error").filter(|e| truthy(Some(e))) {
            let msg = e.get("message").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| e.to_string());
            return Some(msg);
        }
    }
    None
}

/// Fill a just-taken slot from the request and build the body to send:
/// `model` replaced for a remote, and a stream asked for its usage (the
/// tracker strips that chunk again unless the client asked for it).
fn start_slot(st: &Arc<Mutex<ModelStats>>, idx: usize, p: &Map<String, Value>, body: &Bytes, model: Option<&str>) -> (SlotGuard, Option<SseTracker>, Bytes) {
    let (prompt, n_prompt) = prompt_text(p);
    let max_tokens = p
        .get("max_tokens")
        .filter(|v| truthy(Some(v)))
        .or_else(|| p.get("max_completion_tokens").filter(|v| truthy(Some(v))))
        .and_then(py_int)
        .unwrap_or(-1);
    {
        let mut s = Router::lock_stats(st);
        let scaled = s.estimate_prompt(n_prompt);
        if let Some(sl) = s.slots.get_mut(idx) {
            sl.prompt = prompt;
            sl.n_prompt_est = n_prompt;
            sl.n_prompt = scaled;
            sl.max_tokens = max_tokens;
        }
    }
    let guard = SlotGuard::new(st.clone(), idx);
    let stream = truthy(p.get("stream"));
    if !stream && model.is_none() {
        return (guard, None, body.clone());
    }
    let mut out = p.clone();
    if let Some(m) = model {
        out.insert("model".into(), Value::String(m.to_string()));
    }
    let mut tracker = None;
    if stream {
        let mut so = out.get("stream_options").and_then(Value::as_object).cloned().unwrap_or_default();
        let want_usage = truthy(so.get("include_usage"));
        so.insert("include_usage".into(), Value::Bool(true));
        out.insert("stream_options".into(), Value::Object(so));
        tracker = Some(SseTracker::new(want_usage));
    }
    (guard, tracker, Bytes::from(serde_json::to_vec(&Value::Object(out)).unwrap_or_default()))
}

/// Everything the handlers share.
pub struct Router {
    opts: Opts,
    stats: HashMap<String, Arc<Mutex<ModelStats>>>,
    remotes: Vec<Remote>,
    sticky: Mutex<Sticky>,
    healthy: HashMap<String, AtomicBool>,
    /// name -> (when, aggregated text), 1 s cache of the backend's /metrics.
    backend_metrics: Mutex<HashMap<String, (f64, String)>>,
    client: HttpClient,
}

impl Router {
    pub fn new(opts: Opts) -> Arc<Self> {
        let stats = opts.routes.iter().map(|(n, u)| (n.clone(), Arc::new(Mutex::new(ModelStats::new(n, u))))).collect();
        let healthy = opts.routes.iter().map(|(n, _)| (n.clone(), AtomicBool::new(false))).collect();
        let https = hyper_rustls::HttpsConnectorBuilder::new()
            .with_provider_and_webpki_roots(rustls::crypto::ring::default_provider())
            .expect("TLS provider")
            .https_or_http()
            .enable_http1()
            .build();
        let client = Client::builder(TokioExecutor::new()).build::<_, Full<Bytes>>(https);
        let config_dir = fidim_core::config::Config::config_dir();
        let remotes = opts.remotes.iter().map(|r| Remote::new(r.clone(), &config_dir)).collect();
        Arc::new(Router {
            opts,
            stats,
            remotes,
            sticky: Mutex::new(Sticky::default()),
            healthy,
            backend_metrics: Mutex::new(HashMap::new()),
            client,
        })
    }

    fn lock_stats(st: &Mutex<ModelStats>) -> std::sync::MutexGuard<'_, ModelStats> {
        st.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn alias_target<'a>(&'a self, model: &'a str) -> &'a str {
        self.opts.aliases.iter().find(|(a, _)| a == model).map(|(_, t)| t.as_str()).unwrap_or(model)
    }

    fn route_url(&self, name: &str) -> Option<&str> {
        self.opts.routes.iter().find(|(n, _)| n == name).map(|(_, u)| u.as_str())
    }

    /// The backend for a model name: alias resolved, unknown -> default.
    fn backend_for<'a>(&'a self, model: &'a str) -> (&'a str, &'a str) {
        let m = self.alias_target(model);
        match self.route_url(m) {
            Some(u) => (u, m),
            None => (self.route_url(&self.opts.default).unwrap_or(""), self.opts.default.as_str()),
        }
    }

    fn is_healthy(&self, name: &str) -> bool {
        self.healthy.get(name).map(|h| h.load(Ordering::Relaxed)).unwrap_or(false)
    }

    /// The models a `?model=` query names: one (alias resolved; none when
    /// unknown), or all of them.
    fn stats_for(&self, model: Option<&str>) -> Vec<Arc<Mutex<ModelStats>>> {
        match model {
            None => self.opts.routes.iter().filter_map(|(n, _)| self.stats.get(n).cloned()).collect(),
            Some(m) => match self.remote_named(m) {
                Some(r) => vec![r.stats.clone()],
                None => self.stats.get(self.alias_target(m)).cloned().into_iter().collect(),
            },
        }
    }

    fn remote_named(&self, name: &str) -> Option<&Remote> {
        self.remotes.iter().find(|r| r.spec.name == name)
    }

    /// A remote's own queue (`inflight`, `waiting` from its `/health`) as the
    /// SGLang gauge names FIDIM shows. Read from the cache `queue_poller`
    /// fills (that `/health` takes ~2 s); asking marks the remote as watched.
    fn remote_queue_text(&self, r: &Remote) -> String {
        *r.viewed_at.lock().unwrap_or_else(|e| e.into_inner()) = now();
        let q = r.queue.lock().unwrap_or_else(|e| e.into_inner());
        match *q {
            Some((at, inflight, waiting)) if now() - at < 20.0 => {
                format!("sglang:num_running_reqs {}\nsglang:num_queue_reqs {}\n", py_float(inflight), py_float(waiting))
            }
            _ => String::new(),
        }
    }

    /// Keeps a remote's queue numbers fresh: back to back (~2 s apart) while
    /// FIDIM looks at it (asked within the last 30 s), else once a minute.
    async fn queue_poller(self: Arc<Self>, i: usize) {
        let r = &self.remotes[i];
        let Some(origin) = r.spec.base_url.parse::<Uri>().ok().and_then(|u| Some(format!("{}://{}", u.scheme_str()?, u.authority()?))) else {
            return;
        };
        let mut last_poll = -1e9;
        loop {
            let watched = now() - *r.viewed_at.lock().unwrap_or_else(|e| e.into_inner()) < 30.0;
            if !watched && now() - last_poll < 60.0 {
                tokio::time::sleep(Duration::from_secs(2)).await;
                continue;
            }
            last_poll = now();
            if let Some((StatusCode::OK, _, body)) = self.get_raw_auth(format!("{origin}/health"), r.key.as_deref(), Duration::from_secs(8)).await {
                if let Ok(v) = serde_json::from_slice::<Value>(&body) {
                    let num = |k: &str| v.get(k).and_then(Value::as_f64);
                    if let (Some(a), Some(b)) = (num("inflight"), num("waiting")) {
                        *r.queue.lock().unwrap_or_else(|e| e.into_inner()) = Some((now(), a, b));
                    }
                }
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    }


    // ---- backend calls ----

    async fn get_raw(&self, url: String, timeout: Duration) -> Option<(StatusCode, HeaderMap, Bytes)> {
        self.get_raw_auth(url, None, timeout).await
    }

    async fn get_raw_auth(&self, url: String, bearer: Option<&str>, timeout: Duration) -> Option<(StatusCode, HeaderMap, Bytes)> {
        let uri: Uri = url.parse().ok()?;
        let mut req = Request::get(uri).body(Full::new(Bytes::new())).ok()?;
        if let Some(k) = bearer {
            req.headers_mut().insert(header::AUTHORIZATION, HeaderValue::from_str(&format!("Bearer {k}")).ok()?);
        }
        tokio::time::timeout(timeout, async {
            let resp = self.client.request(req).await.ok()?;
            let (parts, body) = resp.into_parts();
            let bytes = body.collect().await.ok()?.to_bytes();
            Some((parts.status, parts.headers, bytes))
        })
        .await
        .ok()
        .flatten()
    }

    async fn get_json(&self, url: String, timeout: Duration) -> Option<Value> {
        let (_, _, body) = self.get_raw(url, timeout).await?;
        serde_json::from_slice(&body).ok()
    }

    /// `/get_server_info` into the model's slot count and context size,
    /// re-read every 60 s.
    async fn ensure_info(&self, st: &Arc<Mutex<ModelStats>>) {
        let url = {
            let s = Self::lock_stats(st);
            if !s.info_stale() {
                return;
            }
            format!("{}/get_server_info", s.url)
        };
        let info = self.get_json(url, Duration::from_secs(3)).await;
        Self::lock_stats(st).apply_info(info.as_ref());
    }

    /// A member's per-request token limit (fresh `/get_server_info`), if known.
    async fn member_n_ctx(&self, name: &str) -> Option<u64> {
        let st = self.stats.get(name)?;
        self.ensure_info(st).await;
        Some(Self::lock_stats(st).n_ctx)
    }

    async fn prefill_progress(&self, st: &Arc<Mutex<ModelStats>>) {
        let url = {
            let s = Self::lock_stats(st);
            if s.prefilling().is_empty() {
                return;
            }
            format!("{}/get_load", s.url)
        };
        let Some(load) = self.get_json(url, Duration::from_secs(2)).await else { return };
        let Some(items) = load.as_array() else { return };
        let mut pending = 0i64;
        for x in items {
            match x.get("num_pending_tokens").filter(|v| truthy(Some(v))) {
                Some(v) => match py_int(v) {
                    Some(n) => pending += n,
                    None => return,
                },
                None => {}
            }
        }
        Self::lock_stats(st).apply_prefill(pending.max(0) as u64);
    }

    /// The backend's own Prometheus metrics (--enable-metrics), aggregated, cached 1 s.
    async fn backend_metrics_text(&self, st: &Arc<Mutex<ModelStats>>) -> String {
        let (name, url) = {
            let s = Self::lock_stats(st);
            (s.name.clone(), s.url.clone())
        };
        {
            let cache = self.backend_metrics.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((t, text)) = cache.get(&name) {
                if now() - t < 1.0 {
                    return text.clone();
                }
            }
        }
        let raw = match self.get_raw(format!("{url}/metrics"), Duration::from_millis(1500)).await {
            Some((status, _, body)) if status == StatusCode::OK => String::from_utf8_lossy(&body).into_owned(),
            _ => String::new(),
        };
        let text = aggregate_backend_metrics(&raw);
        self.backend_metrics.lock().unwrap_or_else(|e| e.into_inner()).insert(name, (now(), text.clone()));
        text
    }

    /// Backend liveness into a cache, for `/health` and per-model status.
    ///
    /// Probes `/ready`: on SGLang it is a plain HTTP answer (tokenizer up,
    /// scheduler reported ready). SGLang's `/health` instead generates a token
    /// on the GPU for every call (~15-20 J each at idle here, a GPU wake every
    /// poll), so it is only the fallback for backends without `/ready`
    /// (llama-server answers 404 there). Every 2 s while any backend is down or
    /// any request is in flight or ended within the last minute; every 30 s
    /// when everything is idle and healthy.
    async fn health_poller(self: Arc<Self>) {
        const BUSY: Duration = Duration::from_secs(2);
        const IDLE: Duration = Duration::from_secs(30);
        const ACTIVE_WINDOW: f64 = 60.0;
        let mut has_ready: HashMap<String, bool> = HashMap::new();
        loop {
            let mut all_ok = true;
            for (name, url) in &self.opts.routes {
                let ok = self.probe_backend(name, url, &mut has_ready).await;
                if let Some(h) = self.healthy.get(name) {
                    h.store(ok, Ordering::Relaxed);
                }
                all_ok &= ok;
            }
            let period = if !all_ok || self.recently_active(ACTIVE_WINDOW) { BUSY } else { IDLE };
            tokio::time::sleep(period).await;
        }
    }

    /// One liveness probe: `/ready`, or `/health` once the backend showed it
    /// has no `/ready` (404). The choice is remembered per backend.
    async fn probe_backend(&self, name: &str, url: &str, has_ready: &mut HashMap<String, bool>) -> bool {
        let ok_status = |r: &Option<(StatusCode, HeaderMap, Bytes)>| matches!(r, Some((s, _, _)) if *s == StatusCode::OK);
        if *has_ready.get(name).unwrap_or(&true) {
            let r = self.get_raw(format!("{url}/ready"), Duration::from_secs(4)).await;
            if !matches!(r, Some((s, _, _)) if s == StatusCode::NOT_FOUND) {
                return ok_status(&r);
            }
            has_ready.insert(name.to_string(), false);
        }
        ok_status(&self.get_raw(format!("{url}/health"), Duration::from_secs(4)).await)
    }

    /// A remote member's liveness: `GET {base}/models` lists `served_model`.
    /// Every 60 s when idle, 20 s while it is in use, 15 s while it is down.
    async fn remote_poller(self: Arc<Self>, i: usize) {
        loop {
            let r = &self.remotes[i];
            let res = self.get_raw_auth(format!("{}/models", r.spec.base_url), r.key.as_deref(), Duration::from_secs(10)).await;
            let (ok, why) = match &res {
                None => (false, "no answer".to_string()),
                Some((st, _, body)) if *st == StatusCode::OK => {
                    let listed = serde_json::from_slice::<Value>(body)
                        .ok()
                        .and_then(|v| v.get("data").and_then(Value::as_array).cloned())
                        .is_some_and(|a| a.iter().any(|m| m.get("id").and_then(Value::as_str) == Some(r.spec.served_model.as_str())));
                    (listed, if listed { String::new() } else { format!("{} not listed", r.spec.served_model) })
                }
                Some((st, _, _)) => (false, format!("HTTP {}", st.as_u16())),
            };
            let was = r.healthy.swap(ok, Ordering::Relaxed);
            if was != ok {
                if ok {
                    log_line("INFO", &format!("remote {} healthy ({})", r.spec.name, r.spec.base_url));
                } else {
                    log_line("WARNING", &format!("remote {} unavailable: {why}", r.spec.name));
                }
            }
            if !ok {
                *r.last_error.lock().unwrap_or_else(|e| e.into_inner()) = why;
            }
            let in_use = {
                let t = now();
                Self::lock_stats(&r.stats).slots.iter().any(|sl| sl.busy || (sl.t_end > 0.0 && t - sl.t_end < 60.0))
            };
            let period = if !ok { 15 } else if in_use { 20 } else { 60 };
            tokio::time::sleep(Duration::from_secs(period)).await;
        }
    }

    /// The remotes that may take this request: pooled with its route, and a
    /// pooled path, and the client did not ask to stay local.
    fn remotes_for(&self, route: &str, path: &str, headers: &HeaderMap) -> Vec<usize> {
        if !matches!(path, "/v1/chat/completions" | "/v1/completions") || local_only(headers) {
            return vec![];
        }
        self.remotes.iter().enumerate().filter(|(_, r)| r.spec.pool == route).map(|(i, _)| i).collect()
    }

    /// Choose a member for a pooled request, waiting up to `sticky_wait` for
    /// the conversation's own member when it is full and another could take it.
    async fn choose(&self, st: &Arc<Mutex<ModelStats>>, pool: &[usize], key: Option<u64>) -> Pick {
        let start = now();
        loop {
            let waited = now() - start >= self.opts.sticky_wait;
            if let Some(p) = self.try_pick(st, pool, key, waited) {
                return p;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }

    /// One pick attempt (None: wait and try again). Rules:
    /// - a conversation stays on its member while that member has a slot;
    /// - its member full: wait (until `waited`) when another member could
    ///   take it, then move; its remote unusable: move now;
    /// - new conversations: local while it has a free slot, else a remote
    ///   with room, else local (SGLang queues);
    /// - a remote never gets more than `max_inflight` (the slot is taken
    ///   here, under the lock).
    fn try_pick(&self, st: &Arc<Mutex<ModelStats>>, pool: &[usize], key: Option<u64>, waited: bool) -> Option<Pick> {
        let mut sticky = self.sticky.lock().unwrap_or_else(|e| e.into_inner());
        let local_room = {
            let s = Self::lock_stats(st);
            s.slots.iter().filter(|x| x.busy).count() < s.n_slots.max(1)
        };
        let remote_with_room = || pool.iter().copied().find(|&i| self.remotes[i].has_room(false));
        let fresh = |prefer_remote: bool| -> (Target, &'static str) {
            if local_room && !prefer_remote {
                return (Target::Local, "new");
            }
            match remote_with_room() {
                Some(i) => (Target::Remote(i), "spill"),
                None => (Target::Local, "new"),
            }
        };
        let (target, why) = match sticky.get(key) {
            None => fresh(false),
            Some(Target::Local) if local_room => (Target::Local, "sticky"),
            Some(Target::Local) => {
                if remote_with_room().is_none() {
                    (Target::Local, "sticky") // nowhere else: queue locally
                } else if !waited {
                    return None;
                } else {
                    fresh(true)
                }
            }
            Some(Target::Remote(i)) if pool.contains(&i) => {
                let r = &self.remotes[i];
                if r.has_room(true) {
                    (Target::Remote(i), "sticky")
                } else if !r.usable(true) {
                    fresh(false)
                } else if !waited {
                    return None;
                } else {
                    fresh(false)
                }
            }
            Some(Target::Remote(_)) => fresh(false),
        };
        sticky.put(key, target);
        Some(match target {
            Target::Local => Pick::Local(Self::lock_stats(st).take()),
            Target::Remote(i) => Pick::Remote(i, Self::lock_stats(&self.remotes[i].stats).take(), why),
        })
    }

    /// Whether any member has a request in flight or finished one within
    /// `window` seconds (drives the health poll rate).
    fn recently_active(&self, window: f64) -> bool {
        let t = now();
        self.stats.values().any(|st| {
            let s = Self::lock_stats(st);
            s.slots.iter().any(|sl| sl.busy || (sl.t_end > 0.0 && t - sl.t_end < window))
        })
    }

    // ---- handlers ----

    async fn handle(self: Arc<Self>, req: Request<Incoming>) -> Result<Response<OutBody>, Infallible> {
        let path = req.uri().path().to_string();
        let is_get = matches!(*req.method(), Method::GET | Method::HEAD);
        let resp = if is_get {
            match path.as_str() {
                "/v1/models" => self.models_v1().await,
                p if p.starts_with("/v1/models/") => self.model_v1(&p["/v1/models/".len()..]).await,
                "/models" => self.models_router(),
                "/health" => self.health(),
                "/slots" => self.slots(query_param(req.uri(), "model")).await,
                "/metrics" => self.metrics(query_param(req.uri(), "model")).await,
                "/fidim/state" => self.fidim_state().await,
                "/fidim/pool" => self.pool_state(),
                _ => self.proxy(req).await,
            }
        } else {
            self.proxy(req).await
        };
        Ok(resp)
    }

    /// Merged backend /v1/models: every entry with `id` replaced by the
    /// route name and a status, plus one entry per alias.
    async fn models_v1(&self) -> Response<OutBody> {
        let loaded = || json!({"value": "loaded", "failed": false, "exit_code": null});
        let mut data: Vec<Value> = Vec::new();
        for (name, url) in &self.opts.routes {
            let fallback = || {
                json!({"id": name, "object": "model", "owned_by": "router",
                       "status": {"value": "unloaded", "failed": true, "exit_code": null}})
            };
            let entries = match self.get_json(format!("{url}/v1/models"), Duration::from_secs(10)).await {
                Some(v) => match v.get("data") {
                    None => Some(vec![]),
                    Some(Value::Array(a)) => Some(a.clone()),
                    Some(_) => None,
                },
                None => None,
            };
            match entries {
                None => {
                    log_line("WARNING", &format!("backend {name} /v1/models failed"));
                    data.push(fallback());
                }
                Some(list) => {
                    for m in list {
                        match m {
                            Value::Object(mut o) => {
                                o.insert("id".into(), Value::String(name.clone()));
                                o.insert("status".into(), loaded());
                                if let Some(n) = self.member_n_ctx(name).await.filter(|n| *n > 0) {
                                    // Real per-request limit (KV pool), not the model's nominal
                                    // window; clients such as Hermes read n_ctx to size compaction.
                                    o.insert("n_ctx".into(), json!(n));
                                    o.insert("context_length".into(), json!(n));
                                }
                                data.push(Value::Object(o));
                            }
                            _ => {
                                log_line("WARNING", &format!("backend {name} /v1/models failed: entry is not an object"));
                                data.push(fallback());
                                break;
                            }
                        }
                    }
                }
            }
        }
        for (alias, target) in &self.opts.aliases {
            let mut o = json!({"id": alias, "object": "model", "owned_by": "router", "routed_to": target, "status": loaded()});
            if let Some(n) = self.member_n_ctx(target).await.filter(|n| *n > 0) {
                o["n_ctx"] = json!(n);
                o["context_length"] = json!(n);
            }
            data.push(o);
        }
        json_response(StatusCode::OK, &json!({"object": "list", "data": data}))
    }

    /// `/v1/models/{name}`: the backend's entry for the member (alias resolved),
    /// with the real per-request limit, since clients probe this path first.
    async fn model_v1(&self, name: &str) -> Response<OutBody> {
        let target = self.alias_target(name).to_string();
        let Some(url) = self.route_url(&target) else {
            return json_response(StatusCode::NOT_FOUND, &json!({"error": {"message": format!("model `{name}` is not served here"), "type": "not_found"}}));
        };
        let mut o = match self.get_json(format!("{url}/v1/models/{target}"), Duration::from_secs(10)).await {
            Some(Value::Object(o)) => o,
            _ => serde_json::Map::new(),
        };
        o.insert("id".into(), Value::String(name.to_string()));
        o.entry("object").or_insert(json!("model"));
        if target != name {
            o.insert("routed_to".into(), Value::String(target.clone()));
        }
        if let Some(n) = self.member_n_ctx(&target).await.filter(|n| *n > 0) {
            o.insert("n_ctx".into(), json!(n));
            o.insert("context_length".into(), json!(n));
        }
        json_response(StatusCode::OK, &Value::Object(o))
    }

    /// llama-server router-mode /models: only real members, with status.value (cached health).
    fn models_router(&self) -> Response<OutBody> {
        let data: Vec<Value> = self
            .opts
            .routes
            .iter()
            .map(|(name, _)| {
                let ok = self.is_healthy(name);
                json!({"id": name, "object": "model", "owned_by": "sglang",
                       "status": {"value": if ok { "loaded" } else { "unloaded" }, "failed": !ok, "exit_code": null}})
            })
            .chain(self.remotes.iter().map(|r| {
                let ok = r.usable(true);
                json!({"id": r.spec.name, "object": "model", "owned_by": "remote", "pool": r.spec.pool,
                       "status": {"value": if ok { "loaded" } else { "unloaded" }, "failed": !ok, "exit_code": null}})
            }))
            .collect();
        json_response(StatusCode::OK, &json!({"object": "list", "data": data}))
    }

    fn health(&self) -> Response<OutBody> {
        let bad: Vec<&str> = self.opts.routes.iter().filter(|(n, _)| !self.is_healthy(n)).map(|(n, _)| n.as_str()).collect();
        if bad.is_empty() {
            text_response(StatusCode::OK, "ok")
        } else {
            text_response(StatusCode::SERVICE_UNAVAILABLE, &format!("{} unhealthy", bad.join(", ")))
        }
    }

    async fn slots(&self, model: Option<String>) -> Response<OutBody> {
        if let Some(r) = model.as_deref().and_then(|m| self.remote_named(m)) {
            return json_response(StatusCode::OK, &Value::Array(Self::lock_stats(&r.stats).views()));
        }
        let mut out: Vec<Value> = Vec::new();
        for st in self.stats_for(model.as_deref()) {
            self.ensure_info(&st).await;
            self.prefill_progress(&st).await;
            out.extend(Self::lock_stats(&st).views());
        }
        json_response(StatusCode::OK, &Value::Array(out))
    }

    async fn metrics(&self, model: Option<String>) -> Response<OutBody> {
        if let Some(r) = model.as_deref().and_then(|m| self.remote_named(m)) {
            let mut text = Self::lock_stats(&r.stats).metrics_text();
            text.push_str(&self.remote_queue_text(r));
            return text_response(StatusCode::OK, &text);
        }
        let mut text = String::new();
        for st in self.stats_for(model.as_deref()) {
            text.push_str(&Self::lock_stats(&st).metrics_text());
            text.push_str(&self.backend_metrics_text(&st).await);
        }
        text_response(StatusCode::OK, &text)
    }

    async fn fidim_state(&self) -> Response<OutBody> {
        let mut out = Map::new();
        for (name, _) in &self.opts.routes {
            let Some(st) = self.stats.get(name) else { continue };
            self.ensure_info(st).await;
            self.prefill_progress(st).await;
            let mut v = Self::lock_stats(st).state_json();
            let pool: Map<String, Value> =
                self.remotes.iter().filter(|r| r.spec.pool == *name).map(|r| (r.spec.name.clone(), r.state_json())).collect();
            if !pool.is_empty() {
                v["pool"] = Value::Object(pool);
            }
            out.insert(name.clone(), v);
        }
        json_response(StatusCode::OK, &Value::Object(out))
    }

    /// `/fidim/pool`: every remote member and how many conversations are pinned where.
    fn pool_state(&self) -> Response<OutBody> {
        let (mut local, mut remote) = (0usize, vec![0usize; self.remotes.len()]);
        {
            let t = now();
            let sticky = self.sticky.lock().unwrap_or_else(|e| e.into_inner());
            for (tg, at) in sticky.map.values() {
                if t - at < STICKY_TTL {
                    match tg {
                        Target::Local => local += 1,
                        Target::Remote(i) => remote[*i] += 1,
                    }
                }
            }
        }
        let remotes: Vec<Value> = self
            .remotes
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut v = r.state_json();
                v["name"] = json!(r.spec.name);
                v["pool"] = json!(r.spec.pool);
                v["conversations"] = json!(remote[i]);
                v
            })
            .collect();
        json_response(StatusCode::OK, &json!({"sticky_wait_secs": self.opts.sticky_wait, "local_conversations": local, "remotes": remotes}))
    }

    /// Everything else: forward to the backend the `model` field names,
    /// streaming the reply back; tracked completion requests occupy a slot.
    /// A route with remote members may send a tracked request to one of them
    /// (`try_pick`); a remote that fails before answering is retried locally.
    async fn proxy(&self, req: Request<Incoming>) -> Response<OutBody> {
        let (parts, body) = req.into_parts();
        let body = match Limited::new(body, MAX_REQUEST_BODY).collect().await {
            Ok(c) => c.to_bytes(),
            Err(e) if e.downcast_ref::<http_body_util::LengthLimitError>().is_some() => {
                return text_response(StatusCode::PAYLOAD_TOO_LARGE, &format!("Maximum request body size {MAX_REQUEST_BODY} exceeded"));
            }
            Err(e) => return text_response(StatusCode::BAD_REQUEST, &format!("request body: {e}")),
        };
        let parsed: Option<Map<String, Value>> =
            if body.is_empty() { None } else { serde_json::from_slice::<Value>(&body).ok().and_then(|v| v.as_object().cloned()) };
        let model: String = parsed
            .as_ref()
            .and_then(|p| p.get("model"))
            .filter(|m| truthy(Some(m)))
            .and_then(Value::as_str)
            .unwrap_or(&self.opts.default)
            .to_string();
        // A request for a remote by name (FIDIM's chat with "spark") goes there only.
        let direct = self.remotes.iter().position(|r| r.spec.name == model);
        let (url, resolved) = match direct {
            Some(i) => self.backend_for(&self.remotes[i].spec.pool),
            None => self.backend_for(&model),
        };
        let url = url.to_string();
        let resolved = resolved.to_string();

        let mut headers = HeaderMap::new();
        for (k, v) in parts.headers.iter() {
            if !HOP.contains(&k.as_str()) {
                headers.append(k.clone(), v.clone());
            }
        }
        let path_q = parts.uri.path_and_query().map(|p| p.as_str()).unwrap_or("/").to_string();

        let tracked = parts.method == Method::POST && TRACKED.contains(&parts.uri.path()) && parsed.is_some();
        let st = if tracked { self.stats.get(&resolved).cloned() } else { None };
        let Some(st) = st else {
            return self.send(&url, &resolved, &parts.method, &path_q, headers, body, None).await;
        };
        let p = parsed.unwrap_or_default();
        if let Some(i) = direct.filter(|_| matches!(parts.uri.path(), "/v1/chat/completions" | "/v1/completions")) {
            return self.proxy_direct(i, &parts.uri, &headers, &p, &body).await;
        }
        self.ensure_info(&st).await;
        let pool = self.remotes_for(&resolved, parts.uri.path(), &parts.headers);
        let idx = if pool.is_empty() {
            Self::lock_stats(&st).take()
        } else {
            let key = sticky_key(&parts.headers, &p);
            match self.choose(&st, &pool, key).await {
                Pick::Local(idx) => idx,
                Pick::Remote(i, idx, why) => {
                    let r = &self.remotes[i];
                    let by = if parts.headers.contains_key("x-fidim-session") { "session header" } else { "prompt prefix" };
                    log_line("INFO", &format!("pool {resolved} -> {} ({why}, conversation by {by})", r.spec.name));
                    match self.try_remote(r, idx, &parts.uri, &headers, &p, &body).await {
                        Ok(resp) => return resp,
                        Err(reason) => {
                            r.fail(&reason);
                            log_line("INFO", &format!("pool {resolved}: retrying locally"));
                            self.sticky.lock().unwrap_or_else(|e| e.into_inner()).put(key, Target::Local);
                            Self::lock_stats(&st).take()
                        }
                    }
                }
            }
        };
        let (guard, tracker, body) = start_slot(&st, idx, &p, &body, None);
        self.send(&url, &resolved, &parts.method, &path_q, headers, body, Some((guard, tracker))).await
    }

    /// A request naming a remote: that remote or an error (no local fallback).
    async fn proxy_direct(&self, i: usize, uri: &Uri, headers: &HeaderMap, p: &Map<String, Value>, body: &Bytes) -> Response<OutBody> {
        let r = &self.remotes[i];
        let idx = {
            let _sticky = self.sticky.lock().unwrap_or_else(|e| e.into_inner());
            if !r.has_room(true) {
                let why = if !r.usable(true) { "unavailable" } else { "busy (all its slots are in use)" };
                return json_response(
                    StatusCode::SERVICE_UNAVAILABLE,
                    &json!({"error": {"message": format!("remote {} is {why}", r.spec.name), "type": "unavailable"}}),
                );
            }
            Self::lock_stats(&r.stats).take()
        };
        log_line("INFO", &format!("remote {} (requested by name)", r.spec.name));
        match self.try_remote(r, idx, uri, headers, p, body).await {
            Ok(resp) => resp,
            Err(reason) => {
                r.fail(&reason);
                json_response(StatusCode::BAD_GATEWAY, &json!({"error": {"message": format!("remote {}: {reason}", r.spec.name), "type": "bad_gateway"}}))
            }
        }
    }

    /// Forward one request to `base` and stream the reply back; a tracked
    /// request's slot rides along on the response body.
    #[allow(clippy::too_many_arguments)]
    async fn send(
        &self,
        base: &str,
        label: &str,
        method: &Method,
        path_q: &str,
        headers: HeaderMap,
        body: Bytes,
        track: Option<(SlotGuard, Option<SseTracker>)>,
    ) -> Response<OutBody> {
        let target = format!("{base}{path_q}");
        let uri: Uri = match target.parse() {
            Ok(u) => u,
            Err(e) => {
                return text_response(StatusCode::BAD_GATEWAY, &format!("bad backend URL {target}: {e}"));
            }
        };
        let mut out_req = Request::builder().method(method.clone()).uri(uri).body(Full::new(body)).expect("valid request");
        *out_req.headers_mut() = headers;
        let resp = match self.client.request(out_req).await {
            Ok(r) => r,
            Err(e) => {
                // `track` drops here: the slot is released.
                return text_response(StatusCode::BAD_GATEWAY, &format!("backend {label} ({base}): {e}"));
            }
        };
        let (rparts, rbody) = resp.into_parts();
        tracked_response(rparts, Some(rbody), None, track)
    }

    /// Send a pooled request to remote `r` (slot `idx` already taken there).
    /// Ok: the response, already committed to the remote. Err: why it should
    /// go local instead; nothing has reached the client (the remote refused,
    /// failed, or its first SSE event was an error).
    async fn try_remote(&self, r: &Remote, idx: usize, uri: &Uri, client_headers: &HeaderMap, p: &Map<String, Value>, body: &Bytes) -> Result<Response<OutBody>, String> {
        let (guard, tracker, rbody) = start_slot(&r.stats, idx, p, body, Some(&r.spec.served_model));
        let stream = tracker.is_some();
        let target = remote_target(&r.spec.base_url, uri);
        let mut headers = HeaderMap::new();
        for (k, v) in client_headers.iter() {
            let name = k.as_str();
            if name != "authorization" && !name.starts_with("x-fidim-") {
                headers.append(k.clone(), v.clone());
            }
        }
        if let Some(k) = &r.key {
            let v = HeaderValue::from_str(&format!("Bearer {k}")).map_err(|_| "API key is not a valid header value".to_string())?;
            headers.insert(header::AUTHORIZATION, v);
        }
        let uri: Uri = target.parse().map_err(|e| format!("bad URL {target}: {e}"))?;
        let mut req = Request::builder().method(Method::POST).uri(uri).body(Full::new(rbody)).map_err(|e| format!("request: {e}"))?;
        *req.headers_mut() = headers;
        // Streams answer once prefill is done (the first token): long prompts take minutes there.
        let wait = Duration::from_secs(900);
        let resp = match tokio::time::timeout(wait, self.client.request(req)).await {
            Err(_) => return Err("no reply in time".into()),
            Ok(Err(e)) => return Err(format!("connect: {e}")),
            Ok(Ok(r)) => r,
        };
        let (rparts, mut rbody) = resp.into_parts();
        if rparts.status != StatusCode::OK {
            let snippet = match tokio::time::timeout(Duration::from_secs(5), Limited::new(rbody, 4096).collect()).await {
                Ok(Ok(b)) => String::from_utf8_lossy(&b.to_bytes()).chars().take(200).collect::<String>(),
                _ => String::new(),
            };
            return Err(format!("HTTP {} {}", rparts.status.as_u16(), snippet.trim()));
        }
        if !stream {
            let all = match tokio::time::timeout(wait, Limited::new(rbody, MAX_REQUEST_BODY).collect()).await {
                Err(_) => return Err("reply not finished in time".into()),
                Ok(Err(e)) => return Err(format!("read: {e}")),
                Ok(Ok(b)) => b.to_bytes(),
            };
            let refused = serde_json::from_slice::<Value>(&all)
                .ok()
                .filter(|v| v.get("choices").is_none())
                .and_then(|v| v.get("error").filter(|e| truthy(Some(e))).map(|e| e.to_string()));
            if let Some(e) = refused {
                return Err(format!("error reply: {}", e.chars().take(200).collect::<String>()));
            }
            return Ok(tracked_response(rparts, None, Some(all), Some((guard, tracker))));
        }
        // A stream: read up to its first event before committing to this member.
        let mut buf: Vec<u8> = Vec::new();
        let mut ended = false;
        while !buf.windows(2).any(|w| w == b"\n\n") {
            match tokio::time::timeout(wait, rbody.frame()).await {
                Err(_) => return Err("no first event in time".into()),
                Ok(None) => {
                    ended = true;
                    break;
                }
                Ok(Some(Err(e))) => return Err(format!("read: {e}")),
                Ok(Some(Ok(frame))) => {
                    if let Ok(d) = frame.into_data() {
                        buf.extend_from_slice(&d);
                    }
                }
            }
        }
        if buf.iter().all(|b| b.is_ascii_whitespace()) {
            return Err("empty reply".into());
        }
        let first_end = buf.windows(2).position(|w| w == b"\n\n").unwrap_or(buf.len());
        if let Some(msg) = sse_error(&buf[..first_end]) {
            return Err(format!("refused: {msg}"));
        }
        Ok(tracked_response(rparts, if ended { None } else { Some(rbody) }, Some(Bytes::from(buf)), Some((guard, tracker))))
    }
}

/// A backend reply on its way to the client: status and headers (hop
/// headers dropped), the body tracked into the slot when there is one.
fn tracked_response(
    rparts: http::response::Parts,
    rbody: Option<Incoming>,
    prefix: Option<Bytes>,
    track: Option<(SlotGuard, Option<SseTracker>)>,
) -> Response<OutBody> {
    let mut builder = Response::builder().status(rparts.status);
    if let Some(h) = builder.headers_mut() {
        for (k, v) in rparts.headers.iter() {
            if !HOP.contains(&k.as_str()) {
                h.append(k.clone(), v.clone());
            }
        }
    }
    let is_sse = rparts.status == StatusCode::OK
        && rparts.headers.get(header::CONTENT_TYPE).map(|v| String::from_utf8_lossy(v.as_bytes()).contains("text/event-stream")).unwrap_or(false);
    let out_body: OutBody = match track {
        // untracked replies are never pre-read
        None => match rbody {
            Some(b) => b.boxed(),
            None => full(prefix.unwrap_or_default()),
        },
        Some((guard, tracker)) => {
            let (st, idx) = guard.disarm();
            let mode = match tracker {
                Some(t) if is_sse => TrackMode::Sse(t),
                _ => TrackMode::Collect(Vec::new()),
            };
            TrackedBody { inner: rbody, prefix, mode, stats: st, slot: idx, status_ok: rparts.status == StatusCode::OK, done: false, released: false }
                .boxed()
        }
    };
    builder.body(out_body).unwrap_or_else(|e| text_response(StatusCode::BAD_GATEWAY, &format!("response: {e}")))
}

// ----------------------------------------------------------- helpers ----

fn full(bytes: impl Into<Bytes>) -> OutBody {
    Full::new(bytes.into()).map_err(|never: Infallible| match never {}).boxed()
}

fn json_response(status: StatusCode, v: &Value) -> Response<OutBody> {
    let body = serde_json::to_vec(v).unwrap_or_default();
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, HeaderValue::from_static(JSON_CT))
        .body(full(body))
        .expect("static response")
}

fn text_response(status: StatusCode, text: &str) -> Response<OutBody> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, HeaderValue::from_static(TEXT_CT))
        .body(full(text.to_string()))
        .expect("static response")
}

/// Percent-decode a query component (`+` is a space, as aiohttp reads it).
fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let hex = &s[i + 1..i + 3];
                match u8::from_str_radix(hex, 16) {
                    Ok(b) => {
                        out.push(b);
                        i += 2;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The first `key=value` of the query string, decoded.
fn query_param(uri: &Uri, key: &str) -> Option<String> {
    uri.query()?.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        (url_decode(k) == key).then(|| url_decode(v))
    })
}

// ------------------------------------------------------------- serve ----

/// Serve on an already bound listener until the task is dropped.
pub async fn serve(router: Arc<Router>, listener: TcpListener) -> anyhow::Result<()> {
    tokio::spawn(router.clone().health_poller());
    for i in 0..router.remotes.len() {
        tokio::spawn(router.clone().remote_poller(i));
        tokio::spawn(router.clone().queue_poller(i));
    }
    loop {
        let (stream, _) = listener.accept().await.context("accept")?;
        let router = router.clone();
        tokio::spawn(async move {
            let io = TokioIo::new(stream);
            let svc = service_fn(move |req| router.clone().handle(req));
            // A client that hangs up mid-stream is not an error worth a log line.
            let _ = hyper::server::conn::http1::Builder::new().serve_connection(io, svc).await;
        });
    }
}

/// `fidim router serve`: bind, log one line, run until killed.
pub fn run(opts: Opts) -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().context("tokio runtime")?;
    rt.block_on(async {
        let listener = TcpListener::bind((opts.host.as_str(), opts.port)).await.with_context(|| format!("bind {}:{}", opts.host, opts.port))?;
        let addr = listener.local_addr().map(|a| a.to_string()).unwrap_or_else(|_| format!("{}:{}", opts.host, opts.port));
        let routes: Vec<String> = opts.routes.iter().map(|(n, u)| format!("{n}={u}")).collect();
        let aliases: Vec<String> = opts.aliases.iter().map(|(a, t)| format!("{a}={t}")).collect();
        let remotes: Vec<String> =
            opts.remotes.iter().map(|r| format!("{}->{} ({} x{})", r.pool, r.name, r.base_url, r.max_inflight)).collect();
        println!(
            "{} INFO router listening on http://{addr}  routes [{}]  aliases [{}]  default {}  remotes [{}]",
            timestamp(),
            routes.join(" "),
            aliases.join(" "),
            opts.default,
            remotes.join(" ")
        );
        let router = Router::new(opts);
        serve(router, listener).await
    })
}

// ------------------------------------------------------------- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_guard_releases_on_drop_unless_disarmed() {
        let st = Arc::new(Mutex::new(ModelStats::new("m", "http://127.0.0.1:1")));
        // dropped (client went away before the response headers): slot freed
        let idx = st.lock().unwrap().take();
        drop(SlotGuard::new(st.clone(), idx));
        assert!(!st.lock().unwrap().slots[idx].busy);
        // disarmed (handed to the response body): still busy until the body ends
        let idx = st.lock().unwrap().take();
        let (st2, idx2) = SlotGuard::new(st.clone(), idx).disarm();
        assert_eq!(idx2, idx);
        assert!(st2.lock().unwrap().slots[idx].busy);
    }

    fn fresh_slot() -> Slot {
        let mut s = Slot::new(0);
        s.busy = true;
        s.t0 = now();
        s.n_prompt = 10;
        s
    }

    #[test]
    fn python_float_repr() {
        assert_eq!(py_float(0.0), "0.0");
        assert_eq!(py_float(12345.0), "12345.0");
        assert_eq!(py_float(0.5), "0.5");
        assert_eq!(py_float(1e16), "1e+16");
        assert_eq!(py_float(1.5e-5), "1.5e-05");
        assert_eq!(py_float(0.0001), "0.0001");
        assert_eq!(py_float(1e15), "1000000000000000.0");
        assert_eq!(py_float(f64::NAN), "nan");
    }

    #[test]
    fn python_dumps_spacing_and_ascii() {
        let v = json!({"a": [1, 2.0, "x\"y"], "b": {"c": null, "d": true}, "e": "é😀"});
        assert_eq!(py_dumps(&v), "{\"a\": [1, 2.0, \"x\\\"y\"], \"b\": {\"c\": null, \"d\": true}, \"e\": \"\\u00e9\\ud83d\\ude00\"}");
        assert_eq!(py_dumps(&json!("")), "\"\"");
    }

    #[test]
    fn prompt_text_chat_and_completion() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "an answer"},
                {"role": "user", "content": [{"type": "text", "text": "second"}, {"type": "image_url", "image_url": {}}, {"type": "text", "text": "question"}]},
            ]
        });
        let (last, n) = prompt_text(body.as_object().unwrap());
        assert_eq!(last, "second question");
        // chars: 8 + 14 + 9 + 15 = 46, plus json.dumps("") twice per message (2 chars * 4)
        // plus json.dumps("") for tools (2) = 56 -> 14 tokens.
        assert_eq!(n, 14);

        let body = json!({"prompt": "hello world!"});
        assert_eq!(prompt_text(body.as_object().unwrap()), ("hello world!".to_string(), 3));
        let body = json!({"text": ""});
        assert_eq!(prompt_text(body.as_object().unwrap()), (String::new(), 1));
        let body = json!({"prompt": [1, 2, 3]});
        assert_eq!(prompt_text(body.as_object().unwrap()), ("[1, 2, 3]".to_string(), 2));
        let body = json!({"messages": [{"role": "user", "content": "x", "tool_calls": [{"id": "1"}]}], "tools": [{"type": "function"}]});
        // 1 + len('[{"id": "1"}]')=13 + len('[{"type": "function"}]')=22 = 36 -> 9
        assert_eq!(prompt_text(body.as_object().unwrap()).1, 9);
    }

    #[test]
    fn slot_view_shape() {
        let mut s = fresh_slot();
        s.id_task = 7;
        s.max_tokens = 100;
        s.processed = 4;
        s.prompt = "p".repeat(7000);
        let v = s.view(4096, true);
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            ["id", "id_task", "n_ctx", "is_processing", "speculative", "n_prompt_tokens", "n_prompt_tokens_processed", "n_prompt_tokens_cache", "n_prompt_tokens_exact", "next_token", "prompt", "generated"]
        );
        assert_eq!(v["n_ctx"], 4096);
        assert_eq!(v["is_processing"], true);
        assert_eq!(v["speculative"], true);
        // prefill: processed shown while nothing is decoded
        assert_eq!(v["n_prompt_tokens_processed"], 4);
        assert_eq!(v["next_token"][0]["has_next_token"], true);
        assert_eq!(v["next_token"][0]["n_decoded"], 0);
        assert_eq!(v["next_token"][0]["n_remain"], 100);
        assert_eq!(v["prompt"].as_str().unwrap().len(), TEXT_TAIL);
        // decode: prompt counts as fully processed, chars/4 estimate wins over chunk count
        s.add_text(&"g".repeat(40));
        let v = s.view(4096, true);
        assert_eq!(v["n_prompt_tokens_processed"], 10);
        assert_eq!(v["next_token"][0]["n_decoded"], 10);
        assert_eq!(v["next_token"][0]["n_remain"], 90);
        // idle after release: exact count, n_remain from it
        s.busy = false;
        s.n_decoded_exact = true;
        s.n_decoded = 3;
        let v = s.view(0, false);
        assert_eq!(v["speculative"], false);
        assert_eq!(v["next_token"][0]["n_decoded"], 3);
        assert_eq!(v["next_token"][0]["has_next_token"], false);
        assert_eq!(v["next_token"][0]["n_remain"], 97);
        // FIDIM's own parser accepts it
        let parsed = fidim_core::live::parse_slots(&serde_json::to_string(&json!([v])).unwrap()).unwrap();
        assert_eq!(parsed[0].n_decoded, 3);
    }

    #[test]
    fn sse_tracker_feeds_strips_and_reports_usage() {
        let mut slot = fresh_slot();
        let mut t = SseTracker::new(false);
        let c1 = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n";
        let c2 = b"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"lo\"}}]}\n\ndata: {\"choices\":[{\"delta\":{}}]}\n\n";
        let usage = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7}}\n\n";
        let done = b"data: [DONE]\n\n";
        // partial chunks: nothing until the blank line completes an event
        let (a, b) = c1.split_at(20);
        assert!(t.feed(a, &mut slot).is_empty());
        assert_eq!(t.feed(b, &mut slot), c1.to_vec());
        assert_eq!(slot.generated, "Hel");
        assert_eq!(slot.n_decoded, 1);
        assert!(slot.t_first.is_some());
        assert_eq!(slot.processed, 10);
        assert_eq!(t.feed(c2, &mut slot), c2.to_vec());
        assert_eq!(slot.generated, "Hello");
        assert_eq!(slot.n_decoded, 2);
        // the injected usage chunk is swallowed, but its numbers land
        assert!(t.feed(usage, &mut slot).is_empty());
        assert_eq!((slot.n_prompt, slot.n_prompt_exact, slot.n_decoded, slot.n_decoded_exact), (11, true, 7, true));
        assert_eq!(t.feed(done, &mut slot), done.to_vec());
        // a trailing partial event is flushed as-is
        assert_eq!(t.feed(b"data: {\"x\":1}", &mut slot), b"".to_vec());
        assert_eq!(t.flush(&mut slot), b"data: {\"x\":1}".to_vec());
        assert!(t.flush(&mut slot).is_empty());

        // the client asked for usage: the chunk passes
        let mut slot = fresh_slot();
        let mut t = SseTracker::new(true);
        assert_eq!(t.feed(usage, &mut slot), usage.to_vec());
        assert_eq!(slot.n_decoded, 7);
        // usage alongside choices (a final content chunk) passes even when not asked for
        let mut t = SseTracker::new(false);
        let both = b"data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}],\"usage\":{\"completion_tokens\":9}}\n\n";
        assert_eq!(t.feed(both, &mut slot), both.to_vec());
        assert_eq!(slot.n_decoded, 9);
        // completions-style `text`, non-JSON and comment lines pass untouched
        let mut slot = fresh_slot();
        let mut t = SseTracker::new(false);
        let text = b": keep-alive\r\ndata: {\"choices\":[{\"text\":\"abc\"}]}\r\n\n";
        assert_eq!(t.feed(text, &mut slot), text.to_vec());
        assert_eq!(slot.generated, "abc");
        let junk = b"data: not json\n\n";
        assert_eq!(t.feed(junk, &mut slot), junk.to_vec());
    }

    #[test]
    fn backend_metrics_aggregation() {
        let raw = "\
# HELP sglang:prompt_tokens_total Number of prefill tokens processed.
# TYPE sglang:prompt_tokens_total counter
sglang:prompt_tokens_total{model_name=\"a\",tp_rank=\"0\"} 10.0
sglang:prompt_tokens_total{model_name=\"a\",tp_rank=\"1\"} 20.0
sglang:num_running_reqs{model_name=\"a\"} 1.0
sglang:num_running_reqs{model_name=\"b\"} 3.0
sglang:e2e_request_latency_seconds_bucket{le=\"0.5\",model_name=\"a\"} 4.0
sglang:e2e_request_latency_seconds_sum{model_name=\"a\"} 1.25
sglang:e2e_request_latency_seconds_count{model_name=\"a\"} 4.0
sglang:e2e_request_latency_seconds_sum{model_name=\"b\"} 0.25
sglang:e2e_request_latency_seconds_count{model_name=\"b\"} 1.0
sglang:bad_value{x=\"1\"} notanumber
sglang:no_labels 2.5
vllm:other 1.0
";
        let text = aggregate_backend_metrics(raw);
        assert_eq!(
            text,
            "sglang:prompt_tokens_total 30.0\n\
             sglang:num_running_reqs 3.0\n\
             sglang:e2e_request_latency_seconds_sum 1.5\n\
             sglang:e2e_request_latency_seconds_count 5.0\n\
             sglang:no_labels 2.5\n"
        );
        let m = fidim_core::live::parse_metrics(&text);
        assert_eq!(m["prompt_tokens_total"], 30.0);
        assert_eq!(aggregate_backend_metrics(""), "");
    }

    #[test]
    fn model_stats_lifecycle_and_metrics() {
        let mut st = ModelStats::new("dd", "http://127.0.0.1:1");
        assert!(st.info_stale());
        st.apply_info(Some(&json!({"max_running_requests": 2, "context_length": 4096})));
        assert_eq!((st.slots.len(), st.n_ctx), (2, 4096));
        assert!(!st.info_stale());
        let i = st.take();
        let j = st.take();
        let k = st.take(); // grows past max_running_requests
        assert_eq!((i, j, k, st.slots.len(), st.requests_total), (0, 1, 2, 3, 3));
        assert_eq!(st.slots[0].id_task, 1);
        st.slots[i].n_prompt = 8;
        st.apply_prefill(3);
        assert_eq!(st.slots[i].processed, 4); // several prefilling: half
        st.release(j);
        st.release(k);
        st.apply_prefill(3);
        assert_eq!(st.slots[i].processed, 5); // one prefilling: n_prompt - pending
        st.slots[i].add_text("hello world!");
        let m = fidim_core::live::parse_metrics(&st.metrics_text());
        assert_eq!(m["tokens_predicted_total"], 3.0); // live: max(1 chunk, 12 chars / 4)
        assert_eq!(m["prompt_tokens_total"], 8.0);
        assert_eq!(m["requests_processing"], 1.0);
        assert_eq!(m["n_decode_total"], 3.0);
        st.release(i);
        assert_eq!(st.tokens_predicted_total, 3);
        assert_eq!(st.prompt_tokens_total, 8);
        let v = st.state_json();
        assert_eq!(v["requests_total"], 3);
        assert_eq!(v["slots"].as_array().unwrap().len(), 3);
        assert!(v["slots"][0]["is_processing"] == false);
        // a failed info read keeps the slot count
        st.apply_info(None);
        assert_eq!(st.n_slots, 2);
        // a non-streamed reply
        let idx = st.take();
        apply_final_reply(&mut st.slots[idx], br#"{"choices":[{"message":{"reasoning_content":"think ","content":"answer"}}],"usage":{"prompt_tokens":5,"completion_tokens":6}}"#);
        assert_eq!(st.slots[idx].generated, "think answer");
        assert!(st.slots[idx].t_first.is_some());
        st.release(idx);
        assert_eq!((st.tokens_predicted_total, st.prompt_tokens_total), (9, 13));
        // exact usage teaches the prompt estimate this member's ratio
        assert_eq!(st.estimate_prompt(100), 100);
        let idx = st.take();
        st.slots[idx].n_prompt_est = 10;
        apply_final_reply(&mut st.slots[idx], br#"{"choices":[{"text":"x"}],"usage":{"prompt_tokens":12,"completion_tokens":1}}"#);
        st.release(idx);
        assert_eq!(st.estimate_prompt(100), 120);
        // the draft flag comes from the server info
        assert!(!st.speculative);
        st.apply_info(Some(&json!({"max_running_requests": 2, "speculative_algorithm": "EAGLE"})));
        assert!(st.speculative);
    }

    #[test]
    fn opts_and_query_parsing() {
        let o = Opts::parse_args("127.0.0.1", 1234, &["dd=http://127.0.0.1:30000/".into(), "q=http://h:1".into()], &["ddg=dd".into()], "dd").unwrap();
        assert_eq!(o.routes, vec![("dd".to_string(), "http://127.0.0.1:30000".to_string()), ("q".to_string(), "http://h:1".to_string())]);
        assert_eq!(o.aliases, vec![("ddg".to_string(), "dd".to_string())]);
        assert!(Opts::parse_args("h", 1, &["dd=http://x:1".into()], &[], "nope").is_err());
        assert!(Opts::parse_args("h", 1, &["dd".into()], &[], "dd").is_err());
        assert!(Opts::parse_args("h", 1, &[], &[], "dd").is_err());
        let r = Router::new(o);
        assert_eq!(r.backend_for("ddg"), ("http://127.0.0.1:30000", "dd"));
        assert_eq!(r.backend_for("q"), ("http://h:1", "q"));
        assert_eq!(r.backend_for("zzz"), ("http://127.0.0.1:30000", "dd"));
        assert_eq!(r.stats_for(Some("ddg")).len(), 1);
        assert_eq!(r.stats_for(Some("zzz")).len(), 0);
        assert_eq!(r.stats_for(None).len(), 2);
        let u: Uri = "/slots?autoload=false&model=a%2Fb%3Ac+d".parse().unwrap();
        assert_eq!(query_param(&u, "model").as_deref(), Some("a/b:c d"));
        assert_eq!(query_param(&u, "autoload").as_deref(), Some("false"));
        assert_eq!(query_param(&u, "x"), None);
        let u: Uri = "/slots".parse().unwrap();
        assert_eq!(query_param(&u, "model"), None);
    }

    // ---- integration smoke test: a fake SGLang behind the router ----

    /// A body fed from a channel, so the fake backend can pace its chunks.
    struct ChannelBody(tokio::sync::mpsc::Receiver<Bytes>);
    impl Body for ChannelBody {
        type Data = Bytes;
        type Error = Infallible;
        fn poll_frame(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
            self.0.poll_recv(cx).map(|o| o.map(|b| Ok(Frame::data(b))))
        }
    }

    type TestBody = BoxBody<Bytes, Infallible>;

    fn tb(s: impl Into<Bytes>) -> TestBody {
        Full::new(s.into()).boxed()
    }

    async fn fake_backend(req: Request<Incoming>) -> Result<Response<TestBody>, Infallible> {
        let path = req.uri().path().to_string();
        let resp = match (req.method().clone(), path.as_str()) {
            (Method::GET, "/get_server_info") => Response::builder()
                .header("content-type", "application/json")
                .body(tb(r#"{"max_running_requests": 2, "context_length": 4096}"#))
                .unwrap(),
            (Method::GET, "/health") => Response::new(tb("ok")),
            (Method::GET, "/get_load") => Response::new(tb(r#"[{"num_pending_tokens": 3}]"#)),
            (Method::GET, "/v1/models") => Response::new(tb(r#"{"object":"list","data":[{"id":"real/model","object":"model","owned_by":"sglang"}]}"#)),
            (Method::GET, "/v1/models/dd") => Response::new(tb(r#"{"id":"dd","object":"model","owned_by":"sglang","max_model_len":262144}"#)),
            (Method::GET, "/metrics") => Response::new(tb(
                // (`generation_tokens_total`, not `prompt_tokens_total`: FIDIM's parse_metrics
                // keys both families by the name after the colon, and the llamacpp line must win here.)
                "sglang:generation_tokens_total{model_name=\"a\",tp=\"0\"} 10.0\nsglang:generation_tokens_total{model_name=\"a\",tp=\"1\"} 20.0\nsglang:x_bucket{le=\"1\"} 1.0\n",
            )),
            (Method::POST, "/v1/chat/completions") => {
                let body = req.into_body().collect().await.unwrap().to_bytes();
                let v: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(v["stream_options"]["include_usage"], true, "usage injected");
                assert_eq!(v["model"], "ddg", "body forwarded as sent");
                let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(8);
                tokio::spawn(async move {
                    for word in ["Hel", "lo ", "there"] {
                        let _ = tx.send(Bytes::from(format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{word}\"}}}}]}}\n\n"))).await;
                        tokio::time::sleep(Duration::from_millis(250)).await;
                    }
                    let _ = tx.send(Bytes::from("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7}}\n\ndata: [DONE]\n\n")).await;
                });
                Response::builder().header("content-type", "text/event-stream").body(ChannelBody(rx).boxed()).unwrap()
            }
            (Method::POST, "/v1/completions") => Response::builder()
                .header("content-type", "application/json")
                .body(tb(r#"{"choices":[{"text":"yo"}],"usage":{"prompt_tokens":2,"completion_tokens":3}}"#))
                .unwrap(),
            _ => Response::builder().status(404).body(tb("nope")).unwrap(),
        };
        Ok(resp)
    }

    /// A free port above 40000 (ports below are production servers).
    async fn free_listener() -> TcpListener {
        for port in 41000..42000u16 {
            if let Ok(l) = TcpListener::bind(("127.0.0.1", port)).await {
                return l;
            }
        }
        panic!("no free port in 41000..42000");
    }

    async fn get(client: &Client<HttpConnector, Full<Bytes>>, url: &str) -> (StatusCode, String) {
        let r = client.get(url.parse().unwrap()).await.unwrap();
        let (p, b) = r.into_parts();
        (p.status, String::from_utf8_lossy(&b.collect().await.unwrap().to_bytes()).into_owned())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "starts two local HTTP servers; run with --ignored"]
    async fn smoke_router_in_front_of_fake_backend() {
        let backend = free_listener().await;
        let backend_url = format!("http://{}", backend.local_addr().unwrap());
        tokio::spawn(async move {
            loop {
                let (s, _) = backend.accept().await.unwrap();
                tokio::spawn(async move {
                    let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(s), service_fn(fake_backend)).await;
                });
            }
        });
        let listener = free_listener().await;
        let base = format!("http://{}", listener.local_addr().unwrap());
        let opts = Opts::parse_args("127.0.0.1", 0, &[format!("dd={backend_url}")], &["ddg=dd".into()], "dd").unwrap();
        let router = Router::new(opts);
        let server = tokio::spawn(serve(router, listener));

        let client: Client<HttpConnector, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();

        // streaming chat through the alias
        let req = Request::post(format!("{base}/v1/chat/completions"))
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(r#"{"model":"ddg","stream":true,"max_tokens":50,"messages":[{"role":"user","content":"hello there"}]}"#)))
            .unwrap();
        let resp = client.request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers().get("content-type").unwrap().to_str().unwrap().contains("text/event-stream"));
        let mut body = resp.into_body();
        let reader = tokio::spawn(async move {
            let mut got = Vec::new();
            while let Some(f) = body.frame().await {
                if let Ok(d) = f.unwrap().into_data() {
                    got.extend_from_slice(&d);
                }
            }
            String::from_utf8(got).unwrap()
        });
        tokio::time::sleep(Duration::from_millis(350)).await;
        let (code, slots) = get(&client, &format!("{base}/slots?model=ddg&autoload=false")).await;
        assert_eq!(code, StatusCode::OK);
        let v: Value = serde_json::from_str(&slots).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 2, "{slots}");
        assert_eq!(v[0]["is_processing"], true);
        assert_eq!(v[0]["n_ctx"], 4096);
        assert_eq!(v[0]["prompt"], "hello there");
        assert!(v[0]["next_token"][0]["n_decoded"].as_u64().unwrap() >= 1, "{slots}");
        assert_eq!(v[0]["next_token"][0]["has_next_token"], true);
        assert_eq!(v[1]["is_processing"], false);
        let parsed = fidim_core::live::parse_slots(&slots).unwrap();
        assert_eq!(parsed[0].phase, "decode");
        let (_, m) = get(&client, &format!("{base}/metrics?model=dd")).await;
        assert_eq!(fidim_core::live::parse_metrics(&m)["requests_processing"], 1.0, "{m}");

        // /v1/models carries the real per-request limit for clients that size compaction by it
        let (_, models) = get(&client, &format!("{base}/v1/models")).await;
        let mv: Value = serde_json::from_str(&models).unwrap();
        let dd = mv["data"].as_array().unwrap().iter().find(|m| m["id"] == "dd").unwrap();
        assert_eq!((dd["n_ctx"].as_u64(), dd["context_length"].as_u64()), (Some(4096), Some(4096)), "{models}");
        let ddg = mv["data"].as_array().unwrap().iter().find(|m| m["id"] == "ddg").unwrap();
        assert_eq!(ddg["n_ctx"].as_u64(), Some(4096), "{models}");
        // the per-model path clients probe first carries it too, alias resolved
        let (code, one) = get(&client, &format!("{base}/v1/models/ddg")).await;
        let ov: Value = serde_json::from_str(&one).unwrap();
        assert_eq!(code, StatusCode::OK, "{one}");
        assert_eq!((ov["id"].as_str(), ov["routed_to"].as_str(), ov["n_ctx"].as_u64(), ov["max_model_len"].as_u64()), (Some("ddg"), Some("dd"), Some(4096), Some(262144)), "{one}");
        let (code, _) = get(&client, &format!("{base}/v1/models/nope")).await;
        assert_eq!(code, StatusCode::NOT_FOUND);

        let text = reader.await.unwrap();
        assert!(text.contains("data: [DONE]"), "{text}");
        assert_eq!(text.matches("\"delta\"").count(), 3, "{text}");
        assert!(!text.contains("completion_tokens"), "usage chunk leaked: {text}");
        assert!(!text.contains("\"choices\":[]"), "{text}");

        // the slot is free and the counters carry the exact usage
        tokio::time::sleep(Duration::from_millis(50)).await;
        let (_, m) = get(&client, &format!("{base}/metrics?model=dd")).await;
        let pm = fidim_core::live::parse_metrics(&m);
        assert_eq!(pm["tokens_predicted_total"], 7.0, "{m}");
        assert_eq!(pm["prompt_tokens_total"], 11.0, "{m}");
        assert_eq!(pm["requests_processing"], 0.0, "{m}");
        assert_eq!(pm["n_decode_total"], 1.0, "{m}");
        assert!(pm["tokens_predicted_seconds_total"] > 0.3, "{m}");
        assert!(pm["predicted_tokens_seconds"] > 0.0 && pm["predicted_tokens_seconds"] < 7.0 / 0.3, "{m}");
        assert!(m.contains("sglang:generation_tokens_total 30.0\n"), "{m}");
        assert!(!m.contains("_bucket"), "{m}");
        let (_, slots) = get(&client, &format!("{base}/slots")).await;
        let v: Value = serde_json::from_str(&slots).unwrap();
        assert_eq!(v[0]["is_processing"], false);
        assert_eq!(v[0]["generated"], "Hello there");
        assert_eq!(v[0]["next_token"][0]["n_decoded"], 7);
        assert_eq!(v[0]["next_token"][0]["n_remain"], 43);
        let (_, unknown) = get(&client, &format!("{base}/slots?model=zzz")).await;
        assert_eq!(unknown, "[]");

        // non-streamed completion through the default route
        let req = Request::post(format!("{base}/v1/completions"))
            .body(Full::new(Bytes::from(r#"{"model":"dd","prompt":"hi"}"#)))
            .unwrap();
        let resp = client.request(req).await.unwrap();
        let (p, b) = resp.into_parts();
        assert_eq!(p.status, StatusCode::OK);
        let b = b.collect().await.unwrap().to_bytes();
        assert!(b.starts_with(b"{\"choices\""));
        tokio::time::sleep(Duration::from_millis(50)).await;
        let (_, state) = get(&client, &format!("{base}/fidim/state")).await;
        let v: Value = serde_json::from_str(&state).unwrap();
        assert_eq!(v["dd"]["requests_total"], 2, "{state}");
        assert_eq!(v["dd"]["tokens_predicted_total"], 10, "{state}");
        assert_eq!(v["dd"]["prompt_tokens_total"], 13, "{state}");
        assert_eq!(v["dd"]["url"], backend_url);

        // health / models (the poller runs at startup, then every 2 s)
        let mut ok = false;
        for _ in 0..30 {
            if get(&client, &format!("{base}/health")).await.0 == StatusCode::OK {
                ok = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(ok, "health never turned green");
        assert_eq!(get(&client, &format!("{base}/health")).await.1, "ok");
        let models = fidim_core::router::models("127.0.0.1", base.rsplit(':').next().unwrap().parse().unwrap()).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!((models[0].id.as_str(), models[0].status.as_str(), models[0].failed), ("dd", "loaded", false));
        let (_, v1) = get(&client, &format!("{base}/v1/models")).await;
        let v: Value = serde_json::from_str(&v1).unwrap();
        let d = v["data"].as_array().unwrap();
        assert_eq!(d.len(), 2, "{v1}");
        assert_eq!(d[0]["id"], "dd");
        assert_eq!(d[0]["owned_by"], "sglang");
        assert_eq!(d[0]["status"]["value"], "loaded");
        assert_eq!(d[1]["id"], "ddg");
        assert_eq!(d[1]["routed_to"], "dd");

        // untracked passthrough keeps the backend's status and body
        let (code, body) = get(&client, &format!("{base}/get_load")).await;
        assert_eq!((code, body.as_str()), (StatusCode::OK, r#"[{"num_pending_tokens": 3}]"#));
        let (code, _) = get(&client, &format!("{base}/nowhere")).await;
        assert_eq!(code, StatusCode::NOT_FOUND);

        server.abort();
    }

    #[test]
    fn slot_count_follows_the_schedulers_cap() {
        let mut st = ModelStats::new("qwenMoE", "http://127.0.0.1:1");
        st.apply_info(Some(&json!({"max_running_requests": 8})));
        assert_eq!(st.slots.len(), 8);
        st.apply_info(Some(&json!({"max_running_requests": 8,
            "internal_states": [{"effective_max_running_requests_per_dp": 4}]})));
        assert_eq!(st.slots.len(), 4);
    }


    // ---- pools ----

    fn spec(name: &str, max: usize) -> RemoteSpec {
        RemoteSpec {
            name: name.into(),
            pool: "dd".into(),
            base_url: "http://127.0.0.1:1/v1".into(),
            served_model: "served-x".into(),
            api_key_file: None,
            max_inflight: max,
            context_length: Some(8192),
        }
    }

    /// A router with route `dd` (`local_slots` slots) and the given remotes, all healthy.
    fn pool_router(local_slots: usize, remotes: Vec<RemoteSpec>) -> Arc<Router> {
        let mut opts = Opts::parse_args("127.0.0.1", 0, &["dd=http://127.0.0.1:1".into()], &[], "dd").unwrap();
        opts.remotes = remotes;
        let r = Router::new(opts);
        Router::lock_stats(&r.stats["dd"]).apply_info(Some(&json!({"max_running_requests": local_slots})));
        for x in &r.remotes {
            x.healthy.store(true, Ordering::Relaxed);
        }
        r
    }

    #[test]
    fn pick_prefers_local_spills_and_sticks() {
        let r = pool_router(1, vec![spec("spark", 2)]);
        let st = r.stats["dd"].clone();
        let pool = [0usize];
        // new conversation, local free -> local
        let a = r.try_pick(&st, &pool, Some(1), false).unwrap();
        assert_eq!(a, Pick::Local(0));
        // local full -> a new conversation spills to the remote
        let b = r.try_pick(&st, &pool, Some(2), false).unwrap();
        assert!(matches!(b, Pick::Remote(0, _, "spill")), "{b:?}");
        // conversation 2 stays on the remote even once local frees up
        Router::lock_stats(&st).release(0);
        Router::lock_stats(&r.remotes[0].stats).release(0);
        let b2 = r.try_pick(&st, &pool, Some(2), false).unwrap();
        assert!(matches!(b2, Pick::Remote(0, _, "sticky")), "{b2:?}");
        // conversation 1 (local) waits while local is full and the remote has room...
        let c = r.try_pick(&st, &pool, Some(3), false).unwrap(); // takes local
        assert!(matches!(c, Pick::Local(_)));
        assert_eq!(r.try_pick(&st, &pool, Some(1), false), None);
        // ...and moves after the wait
        assert!(matches!(r.try_pick(&st, &pool, Some(1), true), Some(Pick::Remote(0, _, "spill"))));
        // remote at max_inflight (2): a new conversation queues locally, never a third remote request
        assert!(matches!(r.try_pick(&st, &pool, Some(4), false), Some(Pick::Local(_))));
        assert_eq!(r.remotes[0].busy(), 2);
    }

    #[test]
    fn pick_skips_unusable_remotes() {
        let r = pool_router(1, vec![spec("spark", 2)]);
        let st = r.stats["dd"].clone();
        let pool = [0usize];
        let _ = r.try_pick(&st, &pool, Some(1), false); // local now full
        r.remotes[0].healthy.store(false, Ordering::Relaxed);
        assert!(matches!(r.try_pick(&st, &pool, Some(2), false), Some(Pick::Local(_))), "unhealthy: local queue");
        r.remotes[0].healthy.store(true, Ordering::Relaxed);
        let s5 = r.try_pick(&st, &pool, Some(5), false).unwrap(); // a conversation on the remote
        assert!(matches!(s5, Pick::Remote(0, _, _)));
        Router::lock_stats(&r.remotes[0].stats).release(0);
        *r.remotes[0].cooling_until.lock().unwrap() = now() + 60.0;
        assert!(matches!(r.try_pick(&st, &pool, Some(3), false), Some(Pick::Local(_))), "cooling: no new conversations");
        assert!(matches!(r.try_pick(&st, &pool, Some(5), false), Some(Pick::Remote(0, _, "sticky"))), "cooling: its own stay");
        Router::lock_stats(&r.remotes[0].stats).release(0);
        *r.remotes[0].cooling_until.lock().unwrap() = 0.0;
        // a conversation pinned to a remote that went away moves at once (no wait)
        let x = r.try_pick(&st, &pool, Some(9), false).unwrap();
        assert!(matches!(x, Pick::Remote(0, _, _)));
        r.remotes[0].healthy.store(false, Ordering::Relaxed);
        assert!(matches!(r.try_pick(&st, &pool, Some(9), false), Some(Pick::Local(_))));
    }

    #[test]
    fn sticky_key_follows_the_conversation() {
        let h = HeaderMap::new();
        let turn1 = json!({"model": "dd", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "task A"}]});
        let turn2 = json!({"model": "dd", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "task A"},
            {"role": "assistant", "content": "ok"}, {"role": "user", "content": "more"}]});
        let other = json!({"model": "dd", "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "task B"}]});
        let k = |v: &Value| sticky_key(&h, v.as_object().unwrap());
        assert_eq!(k(&turn1), k(&turn2));
        assert_ne!(k(&turn1), k(&other));
        assert_eq!(sticky_key(&h, json!({"prompt": "x"}).as_object().unwrap()), None);
        let mut hs = HeaderMap::new();
        hs.insert("x-fidim-session", HeaderValue::from_static("abc"));
        assert_eq!(sticky_key(&hs, turn1.as_object().unwrap()), sticky_key(&hs, other.as_object().unwrap()));
        let mut lo = HeaderMap::new();
        lo.insert("x-fidim-local-only", HeaderValue::from_static("TRUE"));
        assert!(local_only(&lo) && !local_only(&h));
    }

    #[test]
    fn remote_paths_and_refusals() {
        let u: Uri = "/v1/chat/completions?x=1".parse().unwrap();
        assert_eq!(remote_target("https://s.example/v1", &u), "https://s.example/v1/chat/completions?x=1");
        let ev = br#"data: {"error": {"message": "concurrency limit: 2 in-flight requests", "type": "rate_limited"}}"#;
        assert_eq!(sse_error(ev).as_deref(), Some("concurrency limit: 2 in-flight requests"));
        assert_eq!(sse_error(br#"data: {"choices":[{"delta":{"content":""}}]}"#), None);
        assert_eq!(sse_error(b"data: [DONE]"), None);
    }

    #[test]
    fn remotes_file_validation() {
        let dir = std::env::temp_dir().join(format!("fidim-remotes-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let base = || Opts::parse_args("127.0.0.1", 0, &["dd=http://127.0.0.1:1".into()], &[], "dd").unwrap();
        let write = |name: &str, text: &str| {
            let p = dir.join(name);
            std::fs::write(&p, text).unwrap();
            p
        };
        let good = write("good.json", r#"{"sticky_wait_secs": 5, "remotes": [{"name": "spark", "pool": "dd",
            "base_url": "https://s.example/v1/", "served_model": "m", "max_inflight": 2}]}"#);
        let o = base().with_remotes_file(&good);
        assert_eq!(o.remotes.len(), 1);
        assert_eq!(o.remotes[0].base_url, "https://s.example/v1");
        assert_eq!(o.sticky_wait, 5.0);
        // a bad pool, a clashing name or a bad URL: no remotes, router still starts
        for (n, t) in [
            ("pool.json", r#"{"remotes": [{"name": "s", "pool": "nope", "base_url": "https://s/v1", "served_model": "m"}]}"#),
            ("name.json", r#"{"remotes": [{"name": "dd", "pool": "dd", "base_url": "https://s/v1", "served_model": "m"}]}"#),
            ("url.json", r#"{"remotes": [{"name": "s", "pool": "dd", "base_url": "ftp://s/v1", "served_model": "m"}]}"#),
            ("junk.json", "not json"),
        ] {
            assert!(base().with_remotes_file(&write(n, t)).remotes.is_empty(), "{n}");
        }
        assert!(base().with_remotes_file(&dir.join("missing.json")).remotes.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A fake SGLang (`who` = "local", 1 slot) or a fake gateway in front of a
    /// remote (`who` = "remote": wants the key, the served model name and no
    /// x-fidim headers; refuses with a rate_limited first event while `refuse`).
    async fn fake_member(who: &'static str, refuse: Arc<AtomicBool>, req: Request<Incoming>) -> Result<Response<TestBody>, Infallible> {
        let path = req.uri().path().to_string();
        let json_ct = |b: &'static str| Response::builder().header("content-type", "application/json").body(tb(b)).unwrap();
        Ok(match (req.method().clone(), path.as_str()) {
            (Method::GET, "/get_server_info") => json_ct(r#"{"max_running_requests": 1, "context_length": 4096}"#),
            (Method::GET, "/ready") => Response::new(tb("ok")),
            (Method::GET, "/health") if who == "remote" => json_ct(r#"{"gateway":"ok","inflight":3,"waiting":1}"#),
            (Method::GET, "/v1/models") => json_ct(r#"{"object":"list","data":[{"id":"served-x","object":"model"}]}"#),
            (Method::POST, "/v1/chat/completions") => {
                if who == "remote" {
                    assert_eq!(req.headers().get("authorization").unwrap(), "Bearer testkey");
                    assert!(req.headers().keys().all(|k| !k.as_str().starts_with("x-fidim-")), "x-fidim headers stripped");
                }
                let body = req.into_body().collect().await.unwrap().to_bytes();
                let v: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(v["model"], if who == "remote" { "served-x" } else { "dd" });
                let (tx, rx) = tokio::sync::mpsc::channel::<Bytes>(8);
                let refused = who == "remote" && refuse.load(Ordering::Relaxed);
                tokio::spawn(async move {
                    if refused {
                        let _ = tx.send(Bytes::from("data: {\"error\": {\"message\": \"concurrency limit: 2 in-flight requests\", \"type\": \"rate_limited\"}}\n\ndata: [DONE]\n\n")).await;
                        return;
                    }
                    for _ in 0..3 {
                        let _ = tx.send(Bytes::from(format!("data: {{\"choices\":[{{\"delta\":{{\"content\":\"{who} \"}}}}]}}\n\n"))).await;
                        tokio::time::sleep(Duration::from_millis(400)).await;
                    }
                    let _ = tx.send(Bytes::from("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n")).await;
                });
                Response::builder().header("content-type", "text/event-stream").body(ChannelBody(rx).boxed()).unwrap()
            }
            _ => Response::builder().status(404).body(tb("nope")).unwrap(),
        })
    }

    async fn serve_fake(who: &'static str, refuse: Arc<AtomicBool>) -> String {
        let l = free_listener().await;
        let url = format!("http://{}", l.local_addr().unwrap());
        tokio::spawn(async move {
            loop {
                let (s, _) = l.accept().await.unwrap();
                let refuse = refuse.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req| fake_member(who, refuse.clone(), req));
                    let _ = hyper::server::conn::http1::Builder::new().serve_connection(TokioIo::new(s), svc).await;
                });
            }
        });
        url
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[ignore = "starts local HTTP servers; run with --ignored"]
    async fn pool_end_to_end() {
        let refuse = Arc::new(AtomicBool::new(false));
        let local = serve_fake("local", refuse.clone()).await;
        let remote = serve_fake("remote", refuse.clone()).await;
        let keyfile = std::env::temp_dir().join(format!("fidim-pool-key-{}", std::process::id()));
        std::fs::write(&keyfile, "testkey\n").unwrap();
        let mut opts = Opts::parse_args("127.0.0.1", 0, &[format!("dd={local}")], &[], "dd").unwrap();
        opts.remotes = vec![RemoteSpec {
            name: "spark".into(),
            pool: "dd".into(),
            base_url: format!("{remote}/v1"),
            served_model: "served-x".into(),
            api_key_file: Some(keyfile.to_string_lossy().into_owned()),
            max_inflight: 1,
            context_length: Some(8192),
        }];
        opts.sticky_wait = 0.5;
        let router = Router::new(opts);
        let listener = free_listener().await;
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(serve(router.clone(), listener));
        for _ in 0..50 {
            if router.remotes[0].healthy.load(Ordering::Relaxed) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(router.remotes[0].healthy.load(Ordering::Relaxed), "probe with the key made it healthy");
        let client: Client<HttpConnector, Full<Bytes>> = Client::builder(TokioExecutor::new()).build_http();
        let chat = |conv: &'static str, extra: Option<(&'static str, &'static str)>| {
            let client = client.clone();
            let base = base.clone();
            async move {
                let body = json!({"model": "dd", "stream": true, "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": conv}]});
                let mut req = Request::post(format!("{base}/v1/chat/completions"))
                    .header("content-type", "application/json")
                    .body(Full::new(Bytes::from(body.to_string())))
                    .unwrap();
                if let Some((k, v)) = extra {
                    req.headers_mut().insert(k, HeaderValue::from_static(v));
                }
                let r = client.request(req).await.unwrap();
                String::from_utf8_lossy(&r.into_body().collect().await.unwrap().to_bytes()).into_owned()
            }
        };
        // A holds the only local slot; B spills to the remote; C (local only) queues locally.
        let a = tokio::spawn(chat("conv A", None));
        tokio::time::sleep(Duration::from_millis(150)).await;
        let b = tokio::spawn(chat("conv B", None));
        tokio::time::sleep(Duration::from_millis(150)).await;
        let c = tokio::spawn(chat("conv C", Some(("x-fidim-local-only", "1"))));
        let (a, b, c) = (a.await.unwrap(), b.await.unwrap(), c.await.unwrap());
        assert!(a.contains("local ") && !a.contains("remote"), "{a}");
        assert!(b.contains("remote ") && !b.contains("local"), "{b}");
        assert!(c.contains("local "), "{c}");
        assert!(!b.contains("\"usage\""), "injected usage chunk stripped on the remote path too");
        // B's next turn sticks to the remote though local is free now
        let b2 = chat("conv B", None).await;
        assert!(b2.contains("remote "), "{b2}");
        // the remote refuses (gateway limit): D goes local, nothing of the refusal reaches the client
        refuse.store(true, Ordering::Relaxed);
        let hold = tokio::spawn(chat("conv E", None)); // takes local
        tokio::time::sleep(Duration::from_millis(150)).await;
        let d = chat("conv D", None).await;
        assert!(d.contains("local ") && !d.contains("rate_limited"), "{d}");
        assert!(router.remotes[0].cooling(), "refusal puts the remote out for a while");
        let _ = hold.await;
        let (st, pool) = get(&client, &format!("{base}/fidim/pool")).await;
        assert_eq!(st, StatusCode::OK);
        assert!(pool.contains("concurrency limit") && !pool.contains("testkey"), "{pool}");
        let (_, state) = get(&client, &format!("{base}/fidim/state")).await;
        let state: Value = serde_json::from_str(&state).unwrap();
        assert_eq!(state["dd"]["pool"]["spark"]["max_inflight"], 1);
        // FIDIM's view: listed in /models, its own slots and counters, the remote's queue
        refuse.store(false, Ordering::Relaxed);
        let (_, models) = get(&client, &format!("{base}/models")).await;
        let models: Value = serde_json::from_str(&models).unwrap();
        assert!(models["data"].as_array().unwrap().iter().any(|m| m["id"] == "spark" && m["owned_by"] == "remote"), "{models}");
        let (_, v1) = get(&client, &format!("{base}/v1/models")).await;
        assert!(!v1.contains("spark"), "clients' model list stays as it was: {v1}");
        let (_, slots) = get(&client, &format!("{base}/slots?model=spark")).await;
        let slots: Value = serde_json::from_str(&slots).unwrap();
        assert_eq!(slots.as_array().unwrap().len(), 1);
        assert_eq!(slots[0]["n_ctx"], 8192);
        let mut metrics = String::new();
        for _ in 0..30 {
            metrics = get(&client, &format!("{base}/metrics?model=spark")).await.1;
            if metrics.contains("num_running_reqs") {
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        assert!(metrics.contains("llamacpp:requests_processing") && metrics.contains("sglang:num_running_reqs 3.0") && metrics.contains("sglang:num_queue_reqs 1.0"), "{metrics}");
        // a request for "spark" by name goes there (the cooldown blocks new pool conversations, not this)
        let direct = {
            let body = json!({"model": "spark", "stream": true, "messages": [{"role": "user", "content": "hi"}]});
            let req = Request::post(format!("{base}/v1/chat/completions")).body(Full::new(Bytes::from(body.to_string()))).unwrap();
            String::from_utf8_lossy(&client.request(req).await.unwrap().into_body().collect().await.unwrap().to_bytes()).into_owned()
        };
        assert!(direct.contains("remote "), "{direct}");
        let _ = std::fs::remove_file(&keyfile);
    }
}
