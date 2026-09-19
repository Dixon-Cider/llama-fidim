//! The chat client against scripted servers on real sockets, plus its pure
//! parts. The DiffusionGemma path runs against fidim-dg itself with a fake
//! runner in `diffusion::serve_e2e`.
//!
//! `fixtures/llama-server-chat-stream.sse` is a llama-server stream assembled
//! from the chunk shapes in llama.cpp's `server-task.cpp` and
//! `server-common.cpp` (prompt progress, reasoning and content deltas, a
//! keep-alive ping, the finish chunk, the usage chunk with timings and draft
//! counts, `[DONE]`); no model was loaded to capture it.
//! `fixtures/llama-server-props.json` is `/props` assembled the same way.
//! `fixtures/fidim-dg-models.json` was captured from a running fidim-dg.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::*;

const TRANSCRIPT: &str = include_str!("../../../../fixtures/llama-server-chat-stream.sse");
const PROPS: &str = include_str!("../../../../fixtures/llama-server-props.json");
const DG_MODELS: &str = include_str!("../../../../fixtures/fidim-dg-models.json");

// ---------------------------------------------------------------- helpers ----

/// Read one HTTP request (head and Content-Length body) off `s`.
fn read_request(s: &mut TcpStream) -> String {
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut raw = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        if let Some(i) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..i]).to_ascii_lowercase();
            let len: usize = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .map(|v| v.trim().parse().unwrap())
                .unwrap_or(0);
            if raw.len() >= i + 4 + len {
                return String::from_utf8_lossy(&raw).into_owned();
            }
        }
        let n = s.read(&mut buf).unwrap();
        assert!(n > 0, "client closed before sending its request");
        raw.extend_from_slice(&buf[..n]);
    }
}

/// A one-connection server: `script` gets the socket and the request text.
fn serve(script: impl FnOnce(TcpStream, String) + Send + 'static) -> (u16, JoinHandle<()>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let h = std::thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        let req = read_request(&mut s);
        script(s, req);
    });
    (port, h)
}

const SSE_HEAD: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";

/// Write one chunk. Errors are ignored: the client closes as soon as it has
/// `[DONE]`, so whatever the server still writes may meet a reset.
fn chunk(s: &mut TcpStream, data: &[u8]) {
    let _ = s
        .write_all(format!("{:x}\r\n", data.len()).as_bytes())
        .and_then(|_| s.write_all(data))
        .and_then(|_| s.write_all(b"\r\n"));
}

/// The zero chunk, likewise best effort.
fn end(s: &mut TcpStream) {
    let _ = s.write_all(b"0\r\n\r\n");
}

fn body() -> Value {
    request_body(&json!({ "messages": [{ "role": "user", "content": "hi" }] }), "gemma-4-worker", Engine::LlamaServer)
        .unwrap()
}

/// Run a stream to its end, collecting every event.
fn run(port: u16, cancel: &Cancel, idle: Duration) -> (Vec<ChatEvent>, StreamSummary) {
    let mut evs = Vec::new();
    let sum = stream_chat("127.0.0.1", port, &body(), None, cancel, idle, &mut |e| evs.push(e));
    (evs, sum)
}

fn text_of(evs: &[ChatEvent]) -> (String, String) {
    let (mut c, mut r) = (String::new(), String::new());
    for e in evs {
        if let ChatEvent::Delta { content, reasoning } = e {
            c.push_str(content.as_deref().unwrap_or(""));
            r.push_str(reasoning.as_deref().unwrap_or(""));
        }
    }
    (c, r)
}

fn is_terminal(e: &ChatEvent) -> bool {
    matches!(e, ChatEvent::Done { .. } | ChatEvent::Error { .. } | ChatEvent::Cancelled)
}

/// Exactly one terminal event, and it is the last one.
fn assert_one_terminal(evs: &[ChatEvent]) {
    assert_eq!(evs.iter().filter(|e| is_terminal(e)).count(), 1, "{evs:#?}");
    assert!(is_terminal(evs.last().unwrap()), "{evs:#?}");
}

// ------------------------------------------------------------------- pure ----

#[test]
fn addresses() {
    assert_eq!(connect_host("0.0.0.0"), "127.0.0.1");
    assert_eq!(connect_host("::"), "::1");
    assert_eq!(connect_host("[::]"), "::1");
    assert_eq!(connect_host("127.0.0.1"), "127.0.0.1");
    assert_eq!(connect_host("192.168.1.20"), "192.168.1.20");
    assert_eq!(host_port("0.0.0.0", 9701), "127.0.0.1:9701");
    assert_eq!(host_port("::", 9701), "[::1]:9701");
    assert_eq!(host_port("::1", 80), "[::1]:80");
    assert_eq!(base_url("0.0.0.0", 1234), "http://127.0.0.1:1234/v1");
    assert_eq!(base_url("::1", 1234), "http://[::1]:1234/v1");
    assert_eq!(socket_addr("0.0.0.0", 9).unwrap(), "127.0.0.1:9".parse().unwrap());
    assert_eq!(socket_addr("localhost", 9).unwrap(), "127.0.0.1:9".parse().unwrap());
    assert_eq!(socket_addr("::", 9).unwrap(), "[::1]:9".parse().unwrap());
    assert!(socket_addr("example.com", 9).is_err());
    assert!(is_loopback("127.0.0.1") && is_loopback("::1") && is_loopback("localhost"));
    assert!(!is_loopback("0.0.0.0") && !is_loopback("192.168.1.20"));
}

#[test]
fn request_body_forces_streaming_fields() {
    let b = json!({
        "messages": [{ "role": "user", "content": "hi" }],
        "temperature": 0.7, "model": "ignored", "stream": false,
        "stream_options": { "something": 1 },
    });
    let v = request_body(&b, "gemma-4-worker", Engine::LlamaServer).unwrap();
    assert_eq!(v["model"], "gemma-4-worker");
    assert_eq!(v["stream"], true);
    assert_eq!(v["stream_options"], json!({ "something": 1, "include_usage": true }));
    assert_eq!((v["return_progress"].clone(), v["timings_per_token"].clone()), (json!(true), json!(true)));
    assert_eq!(v["temperature"], 0.7);
    // fidim-dg gets only what it reads.
    let d = request_body(&b, "diffusiongemma", Engine::DiffusionGemma).unwrap();
    assert!(d.get("return_progress").is_none() && d.get("timings_per_token").is_none());
    assert_eq!(d["model"], "diffusiongemma");
    // No messages, no request.
    assert!(request_body(&json!({ "messages": [] }), "m", Engine::LlamaServer).is_err());
    assert!(request_body(&json!([1]), "m", Engine::LlamaServer).is_err());
}

fn profile(v: Value) -> Profile {
    let mut base = json!({
        "schema": 1, "id": "worker-pool", "name": "Worker pool",
        "build": { "path": "C:/b" }, "model": { "path": "D:/m.gguf" },
        "devices": [{ "key": "pci:x:bus08" }],
        "server": { "port": 9701, "alias": "gemma-4-worker" },
        "runtime": { "ctx_total": 32768 },
    });
    for (k, x) in v.as_object().unwrap() {
        base[k] = x.clone();
    }
    serde_json::from_value(base).unwrap()
}

#[test]
fn api_key_from_flags_or_env() {
    assert_eq!(api_key(&profile(json!({}))), None);
    let p = profile(json!({ "runtime": { "ctx_total": 1, "extra_flags": ["--foo", "--api-key", "k1"] } }));
    assert_eq!(api_key(&p).as_deref(), Some("k1"));
    let p = profile(json!({ "runtime": { "ctx_total": 1, "extra_flags": ["--api-key=k2"] } }));
    assert_eq!(api_key(&p).as_deref(), Some("k2"));
    let p = profile(json!({ "env": { "llama_api_key": "k3" } }));
    assert_eq!(api_key(&p).as_deref(), Some("k3"));
    // A trailing flag with no value is no key.
    let p = profile(json!({ "runtime": { "ctx_total": 1, "extra_flags": ["--api-key"] } }));
    assert_eq!(api_key(&p), None);
    // Nor is one that would break the header line.
    let p = profile(json!({ "runtime": { "ctx_total": 1, "extra_flags": ["--api-key=k\r\nX-Evil: 1"] } }));
    assert_eq!(api_key(&p), None);
    // fidim-dg has no key.
    let p = profile(json!({ "engine": "diffusion-gemma", "env": { "LLAMA_API_KEY": "k" } }));
    assert_eq!(api_key(&p), None);
}

#[test]
fn fidim_dg_comments_become_events() {
    assert_eq!(comment_event("queued 3"), Some(ChatEvent::Queued { position: 3 }));
    assert_eq!(comment_event("dg task 17"), Some(ChatEvent::Task { id_task: 17 }));
    assert_eq!(
        comment_event("dg 2/8 17/48"),
        Some(ChatEvent::Progress { stage: "denoise".into(), block: Some(2), n_blocks: Some(8), step: Some(17), total: Some(48) })
    );
    assert!(matches!(comment_event("dg prefill"), Some(ChatEvent::Progress { stage, block: None, .. }) if stage == "prefill"));
    assert!(matches!(comment_event("dg loading"), Some(ChatEvent::Progress { stage, .. }) if stage == "loading"));
    for other in ["", "ping", "queued", "queued x", "dg", "dg 2/8", "dg task x", "dg a/b c/d"] {
        assert_eq!(comment_event(other), None, "{other:?}");
    }
}

/// The whole fixture through the parser and translator, split everywhere.
#[test]
fn transcript_translates_at_any_split() {
    for piece in [1usize, 2, 7, 64, 100_000] {
        let mut p = wire::SseParser::new();
        let mut tr = Translator::default();
        let (mut items, mut evs) = (Vec::new(), Vec::new());
        for c in TRANSCRIPT.as_bytes().chunks(piece) {
            p.push(c, &mut items);
            for it in items.drain(..) {
                tr.feed(it, &mut evs).unwrap();
            }
        }
        p.finish(&mut items);
        for it in items.drain(..) {
            tr.feed(it, &mut evs).unwrap();
        }
        tr.finish(&mut evs).unwrap();
        let prefill: Vec<(u64, u64, u64)> = evs
            .iter()
            .filter_map(|e| match e {
                ChatEvent::Prefill { processed, total, cache, .. } => Some((*processed, *total, *cache)),
                _ => None,
            })
            .collect();
        assert_eq!(prefill, [(1712, 2468, 1200), (2468, 2468, 1200)]);
        assert_eq!(text_of(&evs), ("Silicon hums warm,\ncores think as **one**.".into(), "The user wants a haiku.".into()));
        match evs.last().unwrap() {
            ChatEvent::Done { finish_reason, timings, usage, model } => {
                assert_eq!(finish_reason.as_deref(), Some("stop"));
                assert_eq!(model.as_deref(), Some("gemma-4-worker"));
                assert_eq!(usage.as_ref().unwrap()["total_tokens"], 2477);
                let t = timings.as_ref().unwrap();
                assert_eq!((t["draft_n"].as_u64(), t["draft_n_accepted"].as_u64()), (Some(6), Some(4)));
                assert_eq!(t["predicted_n"], 9);
            }
            other => panic!("{other:?}"),
        }
        assert_one_terminal(&evs);
    }
}

#[test]
fn tool_call_fragments_merge_by_index() {
    let mut tr = Translator::default();
    let mut evs = Vec::new();
    for d in [
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}),
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]}}]}),
        json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}},{"index":1,"function":{"name":"now","arguments":{}}}]}}]}),
        json!({"choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
    ] {
        tr.feed(SseItem::Data(d.to_string()), &mut evs).unwrap();
    }
    tr.finish(&mut evs).unwrap();
    let calls = evs.iter().find_map(|e| match e {
        ChatEvent::ToolCalls { calls } => Some(calls.clone()),
        _ => None,
    });
    let calls = calls.expect("a ToolCalls event");
    assert_eq!(calls[0]["id"], "call_1");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(calls[1]["function"]["name"], "now");
    assert_eq!(calls[1]["function"]["arguments"], "{}");
    assert!(matches!(evs.last(), Some(ChatEvent::Done { finish_reason: Some(f), .. }) if f == "tool_calls"));
}

#[test]
fn error_messages_from_bodies() {
    let body = r#"{"error":{"code":400,"message":"the request exceeds the available context size","type":"exceed_context_size_error"}}"#;
    assert_eq!(error_message(400, body), "the request exceeds the available context size");
    assert_eq!(error_message(401, r#"{"error":"Invalid API Key"}"#), "Invalid API Key");
    assert_eq!(error_message(404, r#"{"message":"model not found"}"#), "model not found");
    assert_eq!(error_message(502, ""), "HTTP 502");
    assert_eq!(error_message(500, &"x".repeat(400)).len(), 300);
}

#[test]
fn props_are_trimmed_to_what_the_chat_uses() {
    let p = parse_props(&serde_json::from_str(PROPS).unwrap());
    assert_eq!(p["engine"], "llama-server");
    assert_eq!(p["n_ctx"], 65536);
    assert_eq!(p["total_slots"], 6);
    assert_eq!(p["params"]["temperature"], 1.0);
    assert_eq!(p["params"]["top_k"], 64);
    assert_eq!(p["params"]["dry_multiplier"], 0.8);
    assert!(p["params"].get("samplers").is_none(), "only the sampler fields the panel shows");
    assert_eq!(p["caps"]["supports_system_role"], true);
    assert_eq!(p["caps"]["supports_preserve_reasoning"], false);
    assert_eq!(p["modalities"]["vision"], false);
    assert!(p.get("chat_template").is_none());

    let d = parse_dg_models(&serde_json::from_str(DG_MODELS).unwrap());
    assert_eq!(d["engine"], "diffusion-gemma");
    assert_eq!(d["n_ctx"], 65536);
    assert_eq!(d["canvas"], 256);
    assert_eq!(d["model_alias"], "diffusiongemma");
}

#[test]
fn targets_describe_runs_and_router_models() {
    let state: RunState = serde_json::from_value(json!({
        "profile_id": "worker-pool", "pid": 1, "port": 9701, "host": "0.0.0.0", "alias": "gemma-4-worker",
        "started_unix": 0, "log_path": "x.log", "command_line": "", "visibility_env": "2",
        "device_keys": [], "free_mib_before": [],
    }))
    .unwrap();
    let p = profile(json!({ "sampling": { "temperature": 1.0 }, "chat": { "enable_thinking": false },
        "runtime": { "ctx_total": 1, "extra_flags": ["--api-key", "k"] } }));
    let sample: LiveSample = LiveSample {
        model: None,
        sampled_unix_ms: 0,
        slots: crate::live::parse_slots(
            r#"[{"id":0,"n_ctx":32768,"is_processing":true},{"id":1,"n_ctx":32768,"is_processing":false}]"#,
        )
        .unwrap(),
        phase: "decode".into(),
        metrics: Default::default(),
        error: None,
    };
    let t = target_for(&state, Some(&p), None, "loaded", Some(&sample));
    assert_eq!((t.key.as_str(), t.run.as_str(), t.model_id.as_str()), ("worker-pool", "worker-pool", "gemma-4-worker"));
    assert_eq!((t.host.as_str(), t.bind_host.as_str()), ("127.0.0.1", "0.0.0.0"));
    assert_eq!(t.base_url, "http://127.0.0.1:9701/v1");
    assert!(!t.loopback, "a 0.0.0.0 bind is reachable from the network");
    assert_eq!((t.slots_busy, t.slots_total, t.n_ctx), (Some(1), Some(2), Some(32768)));
    assert_eq!(t.enable_thinking, Some(false));
    assert_eq!(t.sampling.as_ref().and_then(|s| s.temperature), Some(1.0));
    assert!(t.has_api_key);
    assert_eq!(t.label, "Worker pool");
    // The JSON the GUI reads: engine in kebab case, never the key itself.
    let v = serde_json::to_value(&t).unwrap();
    assert_eq!(v["engine"], "llama-server");
    assert!(!v.to_string().contains("\"k\""));

    // A router model: keyed by run and model, llama-server, no sample when unloaded.
    let router = RunState { profile_id: "router".into(), alias: "router".into(), host: "127.0.0.1".into(), port: 1234, ..state };
    let t = target_for(&router, Some(&p), Some("gemma-4-worker"), "unloaded", None);
    assert_eq!((t.key.as_str(), t.model.as_deref(), t.status.as_str()), ("router/gemma-4-worker", Some("gemma-4-worker"), "unloaded"));
    assert_eq!((t.slots_busy, t.n_ctx), (None, None));
    assert!(t.loopback);
    // An unknown member: labelled by its model id.
    let t = target_for(&router, None, Some("other"), "loaded", None);
    assert_eq!((t.label.as_str(), t.profile_id.as_ref()), ("other", None));
}

// ----------------------------------------------------------------- streams ----

#[test]
fn streams_the_transcript_over_a_socket() {
    for (piece, pause) in [(7usize, 0u64), (61, 2), (100_000, 0)] {
        let (port, h) = serve(move |mut s, req| {
            let lower = req.to_ascii_lowercase();
            assert!(lower.starts_with("post /v1/chat/completions http/1.1"), "{req}");
            assert!(lower.contains("accept: text/event-stream"), "{req}");
            assert!(req.contains("\"stream\":true") && req.contains("\"model\":\"gemma-4-worker\""), "{req}");
            s.write_all(SSE_HEAD).unwrap();
            for c in TRANSCRIPT.as_bytes().chunks(piece) {
                chunk(&mut s, c);
                if pause > 0 {
                    std::thread::sleep(Duration::from_millis(pause));
                }
            }
            end(&mut s);
        });
        let (evs, sum) = run(port, &Cancel::new(), DEFAULT_IDLE);
        h.join().unwrap();
        assert_eq!(evs[0], ChatEvent::Open { status: 200 });
        assert_eq!(text_of(&evs), ("Silicon hums warm,\ncores think as **one**.".into(), "The user wants a haiku.".into()));
        assert_one_terminal(&evs);
        assert_eq!(sum.finish_reason.as_deref(), Some("stop"));
        assert_eq!(sum.usage.as_ref().unwrap()["completion_tokens"], 9);
        assert!(!sum.cancelled && sum.error.is_none());
        // Prefill progress comes before any text.
        let first_prefill = evs.iter().position(|e| matches!(e, ChatEvent::Prefill { .. })).unwrap();
        let first_delta = evs.iter().position(|e| matches!(e, ChatEvent::Delta { .. })).unwrap();
        assert!(first_prefill < first_delta);
    }
}

/// Deltas that arrive in a burst reach the GUI in a few batches, not one
/// event per token, and nothing is lost.
#[test]
fn deltas_are_coalesced() {
    const N: usize = 400;
    let (port, h) = serve(|mut s, _| {
        s.write_all(SSE_HEAD).unwrap();
        for i in 0..N {
            let d = json!({ "choices": [{ "index": 0, "delta": { "content": format!("{} ", i % 10) } }] });
            chunk(&mut s, format!("data: {d}\n\n").as_bytes());
            if i % 50 == 49 {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        chunk(&mut s, b"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"length\"}]}\n\ndata: [DONE]\n\n");
        end(&mut s);
    });
    let t0 = Instant::now();
    let (evs, sum) = run(port, &Cancel::new(), DEFAULT_IDLE);
    let elapsed = t0.elapsed();
    h.join().unwrap();
    let want: String = (0..N).map(|i| format!("{} ", i % 10)).collect();
    assert_eq!(text_of(&evs).0, want);
    let deltas = evs.iter().filter(|e| matches!(e, ChatEvent::Delta { .. })).count();
    // At most one batch per flush interval, plus the final flush.
    let bound = (elapsed.as_millis() / FLUSH.as_millis()) as usize + 2;
    assert!(deltas <= bound && deltas < N / 4, "{deltas} delta events in {elapsed:?} (bound {bound})");
    assert_eq!(sum.finish_reason.as_deref(), Some("length"));
}

/// The fact-check's requirement: a server that accepts and then says
/// nothing must not hold a Stop for longer than about a second.
#[test]
fn cancel_returns_promptly_while_the_server_is_silent() {
    let (closed_tx, closed_rx) = mpsc::channel();
    let (port, h) = serve(move |mut s, _| {
        // Silent. Report when the client's close arrives.
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut b = [0u8; 16];
        let r = s.read(&mut b);
        let _ = closed_tx.send((Instant::now(), r.map_err(|e| e.kind())));
    });
    let cancel = Arc::new(Cancel::new());
    let c = cancel.clone();
    let stopper = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(300));
        let at = Instant::now();
        c.cancel();
        at
    });
    let (evs, sum) = run(port, &cancel, DEFAULT_IDLE);
    let returned = Instant::now();
    let cancelled_at = stopper.join().unwrap();
    assert!(returned.duration_since(cancelled_at) < Duration::from_secs(1), "{:?}", returned - cancelled_at);
    assert_eq!(evs, vec![ChatEvent::Cancelled]);
    assert!(sum.cancelled && sum.error.is_none());
    // The server sees the connection end: that is what frees a llama-server slot.
    let (seen, r) = closed_rx.recv_timeout(Duration::from_secs(5)).expect("the server never saw the close");
    assert!(matches!(r, Ok(0) | Err(_)), "{r:?}");
    assert!(seen.duration_since(cancelled_at) < Duration::from_secs(1));
    h.join().unwrap();
}

#[test]
fn cancel_mid_stream_keeps_what_arrived() {
    let (port, h) = serve(|mut s, _| {
        s.write_all(SSE_HEAD).unwrap();
        chunk(&mut s, b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"}}]}\n\n");
        let mut b = [0u8; 16];
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let _ = s.read(&mut b);
    });
    let cancel = Arc::new(Cancel::new());
    let c = cancel.clone();
    let mut evs = Vec::new();
    let sum = stream_chat("127.0.0.1", port, &body(), None, &cancel, DEFAULT_IDLE, &mut |e| {
        if matches!(e, ChatEvent::Delta { .. }) {
            c.cancel();
        }
        evs.push(e);
    });
    h.join().unwrap();
    assert_eq!(text_of(&evs).0, "partial");
    assert_eq!(evs.last(), Some(&ChatEvent::Cancelled));
    assert_one_terminal(&evs);
    assert!(sum.cancelled);
}

/// The idle limit is its own error, separate from Stop.
#[test]
fn idle_timeout_is_an_error_not_a_cancel() {
    let (port, h) = serve(|mut s, _| {
        s.write_all(SSE_HEAD).unwrap();
        chunk(&mut s, b": queued 1\n\n");
        let mut b = [0u8; 16];
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let _ = s.read(&mut b);
    });
    let t0 = Instant::now();
    let (evs, sum) = run(port, &Cancel::new(), Duration::from_millis(300));
    h.join().unwrap();
    assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
    assert!(evs.contains(&ChatEvent::Queued { position: 1 }));
    match evs.last().unwrap() {
        ChatEvent::Error { status: None, message } => assert!(message.contains("no data from the server for 300 ms"), "{message}"),
        other => panic!("{other:?}"),
    }
    assert!(!sum.cancelled);
    assert_one_terminal(&evs);
}

#[test]
fn http_errors_mid_stream_errors_and_truncation() {
    // A plain error answer: no Open, the envelope's message.
    let (port, h) = serve(|mut s, _| {
        let b = r#"{"error":{"code":400,"message":"the request exceeds the available context size","type":"exceed_context_size_error"}}"#;
        s.write_all(format!("HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len()).as_bytes())
            .unwrap();
    });
    let (evs, sum) = run(port, &Cancel::new(), DEFAULT_IDLE);
    h.join().unwrap();
    assert_eq!(
        evs,
        vec![ChatEvent::Error { status: Some(400), message: "the request exceeds the available context size".into() }]
    );
    assert_eq!(sum.status, Some(400));

    // An error inside the stream (llama-server and fidim-dg both send it as data).
    let (port, h) = serve(|mut s, _| {
        s.write_all(SSE_HEAD).unwrap();
        chunk(&mut s, b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"half\"}}]}\n\n");
        chunk(&mut s, b"data: {\"error\":{\"code\":500,\"message\":\"a denoise step failed (see log)\",\"type\":\"server_error\"}}\n\ndata: [DONE]\n\n");
        end(&mut s);
    });
    let (evs, _) = run(port, &Cancel::new(), DEFAULT_IDLE);
    h.join().unwrap();
    assert_eq!(text_of(&evs).0, "half");
    assert_eq!(evs.last(), Some(&ChatEvent::Error { status: Some(500), message: "a denoise step failed (see log)".into() }));
    assert_one_terminal(&evs);

    // The connection drops mid-reply: what arrived stays, and it is an error.
    let (port, h) = serve(|mut s, _| {
        s.write_all(SSE_HEAD).unwrap();
        chunk(&mut s, b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"cut\"}}]}\n\n");
        s.write_all(b"40\r\ndata: {\"cho").unwrap();
    });
    let (evs, _) = run(port, &Cancel::new(), DEFAULT_IDLE);
    h.join().unwrap();
    assert_eq!(text_of(&evs).0, "cut");
    assert!(matches!(evs.last(), Some(ChatEvent::Error { status: None, message }) if message.contains("before the reply finished")), "{evs:?}");

    // Nobody listening.
    let port = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let (evs, _) = run(port, &Cancel::new(), DEFAULT_IDLE);
    assert!(matches!(&evs[..], [ChatEvent::Error { status: None, message }] if message.contains("cannot connect")), "{evs:?}");
}

#[test]
fn a_plain_json_answer_and_the_api_key() {
    let (port, h) = serve(|mut s, req| {
        assert!(req.contains("Authorization: Bearer sk-local"), "{req}");
        let b = json!({ "object": "chat.completion", "model": "m", "choices": [{ "index": 0, "finish_reason": "stop",
            "message": { "role": "assistant", "content": "hi", "reasoning_content": "r" } }],
            "usage": { "prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4 } })
        .to_string();
        s.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}", b.len()).as_bytes())
            .unwrap();
    });
    let mut evs = Vec::new();
    let sum = stream_chat("127.0.0.1", port, &body(), Some("sk-local"), &Cancel::new(), DEFAULT_IDLE, &mut |e| evs.push(e));
    h.join().unwrap();
    assert_eq!(text_of(&evs), ("hi".into(), "r".into()));
    assert_eq!(sum.finish_reason.as_deref(), Some("stop"));
    assert_eq!(sum.usage.unwrap()["total_tokens"], 4);
    assert_one_terminal(&evs);
}

/// A server bound to the wildcard is reached over loopback.
#[test]
fn wildcard_bind_is_reached_over_loopback() {
    let (port, h) = serve(|mut s, _| {
        s.write_all(SSE_HEAD).unwrap();
        chunk(&mut s, b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
        end(&mut s);
    });
    let mut evs = Vec::new();
    stream_chat("0.0.0.0", port, &body(), None, &Cancel::new(), DEFAULT_IDLE, &mut |e| evs.push(e));
    h.join().unwrap();
    assert_eq!(text_of(&evs).0, "ok");
    // And so is supervise's poll.
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let p = l.local_addr().unwrap().port();
    let t = std::thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        let _ = read_request(&mut s);
        s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}").unwrap();
    });
    assert_eq!(crate::supervise::http_get("0.0.0.0", p, "/v1/models", Duration::from_secs(5)).unwrap().0, 200);
    t.join().unwrap();
}
