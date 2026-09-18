//! End-to-end tests over real sockets: `start()` on 127.0.0.1:0 with a
//! fake runner, driven by FIDIM's own HTTP client and parsers where they
//! exist (`supervise::http_*`, `live::parse_slots`, `live::parse_metrics`).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::engine::fake::{FakeCtx, FakeSpawner, Gate, SpawnRecord};
use super::{start, ModelInfo, ServeConfig, ServerHandle};

pub(crate) struct TestServer {
    handle: Option<ServerHandle>,
    spawns: Arc<Mutex<Vec<SpawnRecord>>>,
    dir: PathBuf,
    port: u16,
}

pub(crate) fn start_fake(
    tweak: impl FnOnce(&mut ServeConfig),
    script: impl Fn(FakeCtx) + Send + Sync + 'static,
) -> TestServer {
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "fidim-dg-e2e-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let mut cfg = ServeConfig {
        runner: "fake-runner.exe".into(),
        model: "fake.gguf".into(),
        host: "127.0.0.1".into(),
        port: 0,
        alias: "dg-e2e".into(),
        req_prefix: dir.join("dg-e2e-0"),
        default_max_tokens: 2048,
        seed: None,
        expect_bus: Some(8),
        build_tag: Some("b11027-mix-3e83366".into()),
        queue_depth: 8,
        load_timeout: Duration::from_secs(10),
        watchdog: Duration::from_secs(10),
        ngl: 99,
        maxtok_env: 0,
    };
    tweak(&mut cfg);
    // vocab 0 = a header without a tokens array: READY's n_vocab must win.
    let info = ModelInfo { canvas: 256, block_count: 30, n_ctx_train: 262_144, vocab: 0 };
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let (spawner, spawns) = FakeSpawner::new(script);
    let handle = start(listener, cfg, info, Box::new(spawner)).unwrap();
    let port = handle.addr.port();
    TestServer { handle: Some(handle), spawns, dir, port }
}

impl TestServer {
    pub(crate) fn port(&self) -> u16 {
        self.port
    }

    pub(crate) fn spawns(&self) -> Vec<SpawnRecord> {
        self.spawns.lock().unwrap().clone()
    }

    pub(crate) fn wait_status(&self, path: &str, status: u16) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if get(self.port, path).0 == status {
                return;
            }
            assert!(Instant::now() < deadline, "{path} never answered {status}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Poll /metrics until `key` has `value`.
    fn wait_metric(&self, key: &str, value: f64) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let m = crate::live::parse_metrics(&get(self.port, "/metrics").1);
            if m.get(key) == Some(&value) {
                return;
            }
            assert!(Instant::now() < deadline, "{key} never reached {value}: {m:?}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub(crate) fn stop(mut self) -> i32 {
        let h = self.handle.take().unwrap();
        h.shutdown();
        h.join()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            h.shutdown();
            let _ = h.join();
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub(crate) fn get(port: u16, path: &str) -> (u16, String) {
    crate::supervise::http_get("127.0.0.1", port, path, Duration::from_secs(10)).unwrap()
}

fn post(port: u16, body: &Value) -> (u16, String) {
    crate::supervise::http_post_json("127.0.0.1", port, "/v1/chat/completions", &body.to_string(), Duration::from_secs(20))
        .unwrap()
}

fn chat_body(content: &str, extra: Value) -> Value {
    let mut b = json!({ "model": "anything", "messages": [{ "role": "user", "content": content }] });
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

fn connect(port: u16) -> TcpStream {
    let s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    s
}

fn post_raw(body: &Value) -> Vec<u8> {
    let b = body.to_string();
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{b}",
        b.len()
    )
    .into_bytes()
}

fn read_all(mut s: TcpStream) -> Vec<u8> {
    let mut out = Vec::new();
    let _ = s.read_to_end(&mut out);
    out
}

/// Read until `needle` has arrived (the rest of the stream stays unread).
fn read_until(s: &mut TcpStream, needle: &str) -> String {
    let mut out = Vec::new();
    let mut buf = [0u8; 4096];
    let deadline = Instant::now() + Duration::from_secs(20);
    while !String::from_utf8_lossy(&out).contains(needle) {
        assert!(Instant::now() < deadline, "never saw {needle:?} in {:?}", String::from_utf8_lossy(&out));
        match s.read(&mut buf) {
            Ok(0) => panic!("EOF before {needle:?}: {:?}", String::from_utf8_lossy(&out)),
            Ok(n) => out.extend_from_slice(&buf[..n]),
            Err(e) => panic!("read: {e}"),
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn split_response(raw: &[u8]) -> (String, Vec<u8>) {
    let i = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("no header end");
    (String::from_utf8_lossy(&raw[..i]).into_owned(), raw[i + 4..].to_vec())
}

fn dechunk(mut b: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let nl = b.windows(2).position(|w| w == b"\r\n").expect("chunk size line");
        let size = usize::from_str_radix(std::str::from_utf8(&b[..nl]).unwrap().trim(), 16).unwrap();
        b = &b[nl + 2..];
        if size == 0 {
            assert_eq!(b, b"\r\n", "the stream ends with the zero chunk");
            return out;
        }
        out.extend_from_slice(&b[..size]);
        assert_eq!(&b[size..size + 2], b"\r\n");
        b = &b[size + 2..];
    }
}

/// `data:` payloads in order; comments are dropped.
fn sse_data(body: &[u8]) -> Vec<String> {
    String::from_utf8_lossy(body)
        .split("\n\n")
        .filter_map(|ev| ev.strip_prefix("data: ").map(str::to_string))
        .collect()
}

fn thinking_reply(cx: &FakeCtx) {
    cx.serve(|_| "<|channel>thought\nthinking<channel|>Hello".into());
}

// ------------------------------------------------------------------ tests ----

#[test]
fn health_and_models_are_503_until_ready() {
    let gate = Gate::default();
    let g = gate.clone();
    let srv = start_fake(|_| {}, move |cx| {
        g.wait();
        cx.ready(12288);
        cx.hang();
    });
    for path in ["/health", "/v1/models"] {
        let (status, body) = get(srv.port(), path);
        assert_eq!(status, 503, "{path}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["message"], "Loading model");
        assert_eq!(v["error"]["type"], "unavailable_error");
        assert_eq!(v["error"]["code"], 503);
    }
    gate.open();
    srv.wait_status("/v1/models", 200);
    let v: Value = serde_json::from_str(&get(srv.port(), "/v1/models").1).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "dg-e2e");
    assert_eq!(v["data"][0]["owned_by"], "llama-fidim");
    assert_eq!(v["data"][0]["meta"]["n_ctx"], 12288);
    assert_eq!(v["data"][0]["meta"]["canvas"], 256);
    assert_eq!(v["data"][0]["meta"]["n_vocab"], 262_144);
    assert_eq!(v["data"][0]["meta"]["n_ctx_train"], 262_144);
    assert_eq!(v["data"][0]["meta"]["diffusion"], true);
    let (status, body) = get(srv.port(), "/health");
    assert_eq!(status, 200);
    let h: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(h["status"], "ok");
    assert_eq!(h["engine"], "diffusion-gemma");
    assert_eq!(h["maxtok"], 12288);
    assert_eq!(h["restarts"], 0);
    assert_eq!(h["child_pid"], 40_000);
    assert_eq!(h["build_tag"], "b11027-mix-3e83366");
    assert_eq!(srv.stop(), 0);
}

#[test]
fn non_stream_chat_through_fidims_client() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        thinking_reply(&cx);
    });
    srv.wait_status("/health", 200);
    let (status, body) = post(srv.port(), &chat_body("hi", json!({})));
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "dg-e2e");
    let id = v["id"].as_str().unwrap();
    assert!(id.starts_with("chatcmpl-") && id.len() == "chatcmpl-".len() + 24, "{id}");
    let choice = &v["choices"][0];
    assert_eq!(choice["message"]["role"], "assistant");
    assert_eq!(choice["message"]["content"], "Hello");
    assert_eq!(choice["message"]["reasoning_content"], "thinking");
    assert_eq!(choice["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], 5);
    assert_eq!(v["timings"]["diffusion"], true);

    // /metrics through FIDIM's parser.
    let m = crate::live::parse_metrics(&get(srv.port(), "/metrics").1);
    assert_eq!(m["prompt_tokens_total"], 5.0);
    assert_eq!(m["tokens_predicted_total"], "<|channel>thought\nthinking<channel|>Hello".len() as f64);
    assert_eq!(m["n_decode_total"], 16.0);
    assert_eq!(m["tokens_predicted_seconds_total"], 0.5);
    assert_eq!(m["requests_processing"], 0.0);
    assert_eq!(m["requests_deferred"], 0.0);
    assert_eq!(m["diffusion_restarts_total"], 0.0);
    assert_eq!(m["diffusion_maxtok"], 12288.0);
    assert!(m["predicted_tokens_seconds"] > 0.0);
    assert_eq!(srv.stop(), 0);
}

#[test]
fn streamed_chat_event_order() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        thinking_reply(&cx);
    });
    srv.wait_status("/health", 200);
    let mut s = connect(srv.port());
    s.write_all(&post_raw(&chat_body(
        "hi",
        json!({"stream": true, "stream_options": {"include_usage": true}}),
    )))
    .unwrap();
    let (head, body) = split_response(&read_all(s));
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    let head_l = head.to_ascii_lowercase();
    for h in [
        "content-type: text/event-stream",
        "transfer-encoding: chunked",
        "cache-control: no-cache",
        "connection: close",
        "x-accel-buffering: no",
    ] {
        assert!(head_l.contains(h), "missing {h} in {head}");
    }
    let events = sse_data(&dechunk(&body));
    assert_eq!(events.len(), 6, "{events:?}");
    let ev: Vec<Value> = events[..5].iter().map(|e| serde_json::from_str(e).unwrap()).collect();
    assert_eq!(ev[0]["choices"][0]["delta"]["role"], "assistant");
    assert!(ev[0]["choices"][0]["delta"]["content"].is_null());
    assert_eq!(ev[1]["choices"][0]["delta"]["reasoning_content"], "thinking");
    assert_eq!(ev[2]["choices"][0]["delta"]["content"], "Hello");
    assert_eq!(ev[3]["choices"][0]["finish_reason"], "stop");
    assert_eq!(ev[3]["choices"][0]["delta"], json!({}));
    assert_eq!(ev[3]["timings"]["diffusion"], true);
    assert_eq!(ev[4]["choices"], json!([]));
    assert_eq!(ev[4]["usage"]["prompt_tokens"], 5);
    assert_eq!(events[5], "[DONE]");
    // Every chunk of one response shares its id.
    assert!(ev.iter().all(|e| e["id"] == ev[0]["id"] && e["object"] == "chat.completion.chunk"));
    assert_eq!(srv.stop(), 0);
}

#[test]
fn slots_show_prefill_then_monotonic_decode() {
    let gate = Gate::default();
    let g = gate.clone();
    let srv = start_fake(|_| {}, move |cx| {
        cx.ready(12288);
        while cx.next_request().is_some() {
            g.wait();
            cx.line("F 0 0 4 \"a\"");
            g.wait();
            cx.line("F 0 1 4 \"ab\"");
            cx.line("C 0 \"ab\"");
            g.wait();
            cx.line("F 1 0 4 \"c\"");
            g.wait();
            cx.line("C 1 \"abc\"");
            cx.stats(9, 300, 2);
            cx.line("DONE");
        }
    });
    srv.wait_status("/health", 200);
    let port = srv.port();
    let client = std::thread::spawn(move || post(port, &chat_body("hi", json!({"max_tokens": 512}))));

    let slot = || crate::live::parse_slots(&get(port, "/slots").1).unwrap().remove(0);
    let wait_for = |pred: &dyn Fn(&crate::live::SlotView) -> bool| {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let s = slot();
            if pred(&s) {
                return s;
            }
            assert!(Instant::now() < deadline, "slot never matched: {s:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    let s0 = wait_for(&|s| s.is_processing);
    assert_eq!(s0.phase, "prefill");
    assert_eq!(s0.n_decoded, 0);
    assert_eq!(s0.n_ctx, 12288);
    assert_eq!(s0.id_task, 1);
    assert!(s0.prompt.as_deref().is_some_and(|p| p.contains("[user]\nhi")), "{s0:?}");
    assert_eq!(s0.generated.as_deref(), Some(""));
    let mut seen = vec![0u64];
    // (n_decoded, generated text, committed chars) after each gate: the
    // draft is live text, the loop detector only ever sees committed text.
    for (want, text, committed) in [(64u64, "a", Some(0)), (128, "ab", None), (320, "abc", Some(2))] {
        gate.open();
        let s = wait_for(&|s| s.n_decoded == want && s.generated.as_deref() == Some(text));
        assert_eq!(s.phase, "decode");
        assert_eq!(s.n_remain, 512 - want as i64);
        if committed.is_some() {
            assert_eq!(s.committed_chars, committed, "{s:?}");
        }
        seen.push(s.n_decoded);
    }
    gate.open();
    let (status, body) = client.join().unwrap();
    assert_eq!(status, 200, "{body}");
    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "{seen:?}");
    let idle = wait_for(&|s| !s.is_processing);
    assert_eq!(idle.phase, "idle");
    assert_eq!(idle.n_prompt_tokens, 9);
    // The last answer stays readable, all of it committed.
    assert_eq!((idle.generated.as_deref(), idle.committed_chars), (Some("abc"), Some(3)));
    // Block/step progress for the canvas view, and every step for its replay.
    let d = idle.diffusion.as_ref().expect("diffusion progress");
    assert_eq!((d.block, d.step, d.total, d.steps_done, d.canvas), (1, 0, 4, 3, 256));
    let (code, body) = get(port, "/frames");
    assert_eq!(code, 200, "{body}");
    let f: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(f["id_task"], 1);
    assert_eq!(f["canvas"], 256);
    assert_eq!(f["dropped"], 0);
    let steps: Vec<(u64, u64, &str)> = f["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| (x["b"].as_u64().unwrap(), x["s"].as_u64().unwrap(), x["x"].as_str().unwrap()))
        .collect();
    assert_eq!(steps, [(0, 0, "a"), (0, 1, "ab"), (1, 0, "c")]);
    assert_eq!(srv.stop(), 0);
}

#[test]
fn queued_stream_gets_headers_and_comments() {
    let gate = Gate::default();
    let g = gate.clone();
    let srv = start_fake(|_| {}, move |cx| {
        cx.ready(12288);
        let mut n = 0;
        while cx.next_request().is_some() {
            if n == 0 {
                g.wait();
            }
            cx.reply(&format!("answer {n}"));
            n += 1;
        }
    });
    srv.wait_status("/health", 200);
    let port = srv.port();
    let first = std::thread::spawn(move || post(port, &chat_body("one", json!({}))));
    srv.wait_metric("requests_processing", 1.0);

    let mut s = connect(port);
    s.write_all(&post_raw(&chat_body("two", json!({"stream": true})))).unwrap();
    let head = read_until(&mut s, ": queued 1");
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    gate.open();
    assert_eq!(first.join().unwrap().0, 200);
    let mut rest = Vec::new();
    s.read_to_end(&mut rest).unwrap();
    let mut raw = head.into_bytes();
    raw.extend_from_slice(&rest);
    let (_, body) = split_response(&raw);
    let events = sse_data(&dechunk(&body));
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    let content: String = events
        .iter()
        .filter_map(|e| serde_json::from_str::<Value>(e).ok())
        .filter_map(|v| v["choices"][0]["delta"]["content"].as_str().map(str::to_string))
        .collect();
    assert_eq!(content, "answer 1");
    assert_eq!(srv.stop(), 0);
}

#[test]
fn queue_depth_refuses_with_503_and_serves_fifo() {
    let gate = Gate::default();
    let g = gate.clone();
    let order = Arc::new(Mutex::new(Vec::new()));
    let o = order.clone();
    let srv = start_fake(|c| c.queue_depth = 2, move |cx| {
        cx.ready(12288);
        while let Some(r) = cx.next_request() {
            let who = r.body["messages"][0]["content"].as_str().unwrap_or("?").to_string();
            o.lock().unwrap().push(who.clone());
            if who == "A" {
                g.wait();
            }
            cx.reply(&who);
        }
    });
    srv.wait_status("/health", 200);
    let port = srv.port();
    let a = std::thread::spawn(move || post(port, &chat_body("A", json!({}))));
    srv.wait_metric("requests_processing", 1.0);
    let b = std::thread::spawn(move || post(port, &chat_body("B", json!({}))));
    srv.wait_metric("requests_deferred", 1.0);
    let c = std::thread::spawn(move || post(port, &chat_body("C", json!({}))));
    srv.wait_metric("requests_deferred", 2.0);

    let mut s = connect(port);
    s.write_all(&post_raw(&chat_body("D", json!({})))).unwrap();
    let (head, body) = split_response(&read_all(s));
    assert!(head.starts_with("HTTP/1.1 503"), "{head}");
    assert!(head.to_ascii_lowercase().contains("retry-after: 5"), "{head}");
    let e: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(e["error"]["type"], "unavailable_error");

    gate.open();
    for (h, want) in [(a, "A"), (b, "B"), (c, "C")] {
        let (status, body) = h.join().unwrap();
        assert_eq!(status, 200);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["choices"][0]["message"]["content"], want);
    }
    assert_eq!(*order.lock().unwrap(), vec!["A", "B", "C"]);
    assert_eq!(srv.stop(), 0);
}

#[test]
fn chunked_request_body_is_accepted() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        cx.serve(|_| "chunked ok".into());
    });
    srv.wait_status("/health", 200);
    let body = chat_body("hi", json!({})).to_string();
    let (p1, p2) = body.split_at(10);
    let req = format!(
        "POST /v1/chat/completions/ HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n\
         {:x};ext=1\r\n{p1}\r\n{:x}\r\n{p2}\r\n0\r\n\r\n",
        p1.len(),
        p2.len()
    );
    let mut s = connect(srv.port());
    s.write_all(req.as_bytes()).unwrap();
    // The client keeps its side open: the body must be decoded without EOF.
    let (head, body) = split_response(&read_all(s));
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "chunked ok");
    assert_eq!(srv.stop(), 0);
}

#[test]
fn expect_100_continue_gets_the_interim_response() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        cx.serve(|_| "continued".into());
    });
    srv.wait_status("/health", 200);
    let body = chat_body("hi", json!({})).to_string();
    let mut s = connect(srv.port());
    s.write_all(
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .as_bytes(),
    )
    .unwrap();
    let interim = read_until(&mut s, "\r\n\r\n");
    assert_eq!(interim, "HTTP/1.1 100 Continue\r\n\r\n");
    s.write_all(body.as_bytes()).unwrap();
    let (head, body) = split_response(&read_all(s));
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    let v: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "continued");
    assert_eq!(srv.stop(), 0);
}

#[test]
fn unknown_path_404_and_wrong_method_405() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        cx.hang();
    });
    srv.wait_status("/health", 200);
    let (status, body) = get(srv.port(), "/v1/completions");
    assert_eq!(status, 404);
    assert_eq!(serde_json::from_str::<Value>(&body).unwrap()["error"]["type"], "not_found_error");
    assert_eq!(get(srv.port(), "/v1/chat/completions").0, 405);
    let (status, _) =
        crate::supervise::http_post_json("127.0.0.1", srv.port(), "/health", "{}", Duration::from_secs(5)).unwrap();
    assert_eq!(status, 405);
    // Query strings and trailing slashes route like the bare path.
    assert_eq!(get(srv.port(), "/v1/models/?x=1").0, 200);
    assert_eq!(srv.stop(), 0);
}

#[test]
fn bad_requests_are_400_before_reaching_the_engine() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        cx.hang();
    });
    srv.wait_status("/health", 200);
    let mut s = connect(srv.port());
    s.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 5\r\n\r\n{nope").unwrap();
    let (head, _) = split_response(&read_all(s));
    assert!(head.starts_with("HTTP/1.1 400"), "{head}");
    let img = json!({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    ]}]});
    let (status, body) = post(srv.port(), &img);
    assert_eq!(status, 400);
    assert!(body.contains("text-only"), "{body}");
    assert_eq!(srv.stop(), 0);
}

#[test]
fn block0_toolong_is_a_plain_400_even_when_streaming() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        while cx.next_request().is_some() {
            cx.line("ERR toolong 12544 12288");
            cx.line("DONE");
        }
    });
    srv.wait_status("/health", 200);
    for stream in [false, true] {
        let (status, body) = post(srv.port(), &chat_body("long", json!({"stream": stream})));
        assert_eq!(status, 400, "stream={stream}: {body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "exceed_context_size_error");
        assert_eq!(v["error"]["n_prompt_tokens"], 12288);
        assert_eq!(v["error"]["n_ctx"], 12032);
    }
    assert_eq!(srv.stop(), 0);
}

#[test]
fn client_disconnecting_mid_stream_does_not_break_the_next_request() {
    let gate = Gate::default();
    let g = gate.clone();
    let srv = start_fake(|_| {}, move |cx| {
        cx.ready(12288);
        let mut n = 0;
        while cx.next_request().is_some() {
            if n == 0 {
                cx.line("F 0 0 48 \"x\"");
                cx.line("C 0 \"first block\"");
                g.wait();
                cx.line("F 1 0 48 \"y\"");
                cx.line("C 1 \"first block, second block\"");
                cx.stats(5, 512, 2);
                cx.line("DONE");
            } else {
                cx.reply("next one");
            }
            n += 1;
        }
    });
    srv.wait_status("/health", 200);
    let mut s = connect(srv.port());
    s.write_all(&post_raw(&chat_body("one", json!({"stream": true})))).unwrap();
    read_until(&mut s, "first block");
    drop(s);
    gate.open();
    let (status, body) = post(srv.port(), &chat_body("two", json!({})));
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "next one");
    assert_eq!(srv.stop(), 0);
}

#[test]
fn stop_strings_end_the_stream_early() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        cx.serve(|_| "alpha STOP beta".into());
    });
    srv.wait_status("/health", 200);
    let (status, body) = post(srv.port(), &chat_body("x", json!({"stop": "STOP"})));
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "alpha ");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");

    let mut s = connect(srv.port());
    s.write_all(&post_raw(&chat_body("x", json!({"stream": true, "stop": ["STOP"]})))).unwrap();
    let (_, body) = split_response(&read_all(s));
    let events = sse_data(&dechunk(&body));
    let content: String = events
        .iter()
        .filter_map(|e| serde_json::from_str::<Value>(e).ok())
        .filter_map(|v| v["choices"][0]["delta"]["content"].as_str().map(str::to_string))
        .collect();
    assert_eq!(content, "alpha ");
    assert_eq!(events.last().map(String::as_str), Some("[DONE]"));
    assert_eq!(srv.stop(), 0);
}

#[test]
fn request_files_are_removed_on_exit() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        cx.hang();
    });
    srv.wait_status("/health", 200);
    let dir = srv.dir.clone();
    let prefix = dir.join("dg-e2e-0");
    let ours = super::req_file(&prefix, 9, 1);
    // Another profile's prefix that merely starts with ours, and junk.
    let foreign = dir.join("dg-e2e-0-5-3-1.req");
    let other = dir.join("dg-e2e-0-x-1.req");
    for p in [&ours, &other, &foreign] {
        std::fs::write(p, b"{}").unwrap();
    }
    let port = srv.port();
    assert_eq!(srv.stop_keep_dir(), 0);
    assert!(!ours.exists(), "our request files go");
    assert!(other.exists() && foreign.exists(), "names that are not ours stay");
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    assert!(
        TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_err(),
        "the listener is closed after join"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn req_prefix_falls_back_to_temp() {
    let dir = std::env::temp_dir().join(format!("fidim-dg-prefix-{}", std::process::id()));
    let ok = dir.join("runs").join("dg-a-1234");
    assert_eq!(super::resolve_req_prefix(&ok, 1234).unwrap(), ok);
    assert!(dir.join("runs").is_dir(), "the runs dir is created");
    // Non-ASCII (the runner's narrow fopen) → %TEMP%\fidim-dg-<port>.
    let bad = dir.join("J\u{00fc}rgen").join("dg-a-1234");
    assert_eq!(super::resolve_req_prefix(&bad, 4321).unwrap(), std::env::temp_dir().join("fidim-dg-4321"));
    assert_eq!(super::req_file(&ok, 7, 2), dir.join("runs").join("dg-a-1234-7-2.req"));
    let _ = std::fs::remove_dir_all(dir);
}

// ------------------------------------------------------- review findings ----
// Regressions found by the adversarial review.

/// A chunk size near usize::MAX used to overflow `body.len() + size` (a
/// panic in debug builds) and `size + 2` (a slice panic in release); in
/// fidim-dg that panic exited the whole helper, killing the runner.
#[test]
fn review_chunk_size_overflow_is_a_400_not_a_panic() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    let server = std::thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        super::http::read_request(&mut s).map(|_| ())
    });
    let mut c = connect(port);
    c.write_all(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n\
          1\r\na\r\nffffffffffffffff\r\nb\r\n",
    )
    .unwrap();
    let r = server.join();
    assert!(r.is_ok(), "read_request panicked on a hostile chunk size");
    assert!(matches!(r.unwrap(), Err((400 | 413, _))));
}

/// Defence in depth for the same class of bug: a panic in a connection
/// thread drops that connection; anywhere else it still ends the helper.
#[test]
fn only_connection_thread_panics_spare_the_helper() {
    assert!(super::panic_is_fatal(), "any other thread (engine, accept loop) is fatal");
    let conn = std::thread::Builder::new().name(super::http::CONN_THREAD.into()).spawn(super::panic_is_fatal).unwrap();
    assert!(!conn.join().unwrap());
}

/// openai-python (DEFAULT_MAX_RETRIES = 2) re-sends any >= 500 unless the
/// response says `x-should-retry: false`. The helper has already retried a
/// crash itself, so one deterministic crash (a pinned seed) costs 2 runner
/// deaths per call, and the client's first automatic retry used to trip the
/// 3-death breaker: the helper exited 6 and the model was gone.
#[test]
fn review_one_openai_client_call_does_not_trip_the_breaker() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        // Every runner dies on the first request it is given.
        if cx.next_request().is_some() {
            cx.stderr("ggml_cuda_compute_forward: MUL_MAT failed");
            cx.stderr("ROCm error: unspecified launch failure");
            cx.eof(-1073740791);
        }
    });
    srv.wait_status("/health", 200);
    // What openai-python does for one `chat.completions.create(...)`.
    let body = chat_body("poison", json!({"seed": 7}));
    let mut sent = 0;
    for _ in 0..3 {
        sent += 1;
        let mut s = connect(srv.port());
        s.write_all(&post_raw(&body)).unwrap();
        let (head, _) = split_response(&read_all(s));
        let status: u16 = head[9..12].parse().unwrap();
        let no_retry = head.to_ascii_lowercase().contains("x-should-retry: false");
        if status < 500 || no_retry {
            break;
        }
    }
    assert_eq!(sent, 1, "the engine failure told the client not to retry");
    // The helper must still be serving after one client call.
    srv.wait_status("/health", 200);
    assert_eq!(srv.stop(), 0);
}

/// Engine 500s say "do not retry"; the queue-full 503 keeps its Retry-After
/// (queue_depth_refuses_with_503_and_serves_fifo).
#[test]
fn engine_500s_tell_clients_not_to_retry() {
    let srv = start_fake(|_| {}, |cx| {
        cx.ready(12288);
        while cx.next_request().is_some() {
            cx.line("ERR gen");
            cx.line("DONE");
        }
    });
    srv.wait_status("/health", 200);
    let mut s = connect(srv.port());
    s.write_all(&post_raw(&chat_body("x", json!({})))).unwrap();
    let (head, _) = split_response(&read_all(s));
    let head = head.to_ascii_lowercase();
    assert!(head.starts_with("http/1.1 500"), "{head}");
    assert!(head.contains("x-should-retry: false"), "{head}");
    assert!(!head.contains("retry-after"), "{head}");
    assert_eq!(srv.stop(), 0);
}

/// A runner that closed stdout but will not die (TerminateProcess pending
/// behind a hung driver call): `kill_and_wait` gives up.
struct Undying;

impl super::engine::EngineChild for Undying {
    fn send_line(&mut self, _l: &str) -> std::io::Result<()> {
        Ok(())
    }
    fn pid(&self) -> Option<u32> {
        Some(1)
    }
    fn kill_and_wait(&mut self, _timeout: Duration) -> Option<i32> {
        None
    }
    fn try_exit_code(&mut self) -> Option<i32> {
        None
    }
    fn stderr_tail(&self) -> Vec<String> {
        Vec::new()
    }
}

struct UndyingSpawner(Arc<AtomicUsize>);

impl super::engine::Spawner for UndyingSpawner {
    fn spawn(
        &mut self,
        gen: u64,
        _maxtok_override: Option<u32>,
        tx: std::sync::mpsc::Sender<super::engine::EngineMsg>,
    ) -> std::io::Result<Box<dyn super::engine::EngineChild>> {
        use super::engine::{ChildOut, EngineMsg};
        let n = self.0.fetch_add(1, Ordering::SeqCst);
        for l in [
            "llama_prepare_model_devices: using device ROCm0 (AMD Radeon AI PRO R9700) (0000:08:00.0) - 32472 MiB free",
            "load_tensors: offloaded 31/31 layers to GPU",
            "diffusion-gemma-visual-server ready (n_vocab=262144, canvas=256, MAXTOK=12288, NGL=99, kv_cache=on)",
        ] {
            let f = super::protocol::parse_stderr_fact(l).unwrap();
            let _ = tx.send(EngineMsg::Child { gen, out: ChildOut::Fact(f) });
        }
        let _ = tx.send(EngineMsg::Child { gen, out: ChildOut::Line(b"READY 262144 12288".to_vec()) });
        if n == 0 {
            // Idle crash: stdout closes, the process lingers.
            let _ = tx.send(EngineMsg::Child { gen, out: ChildOut::Eof });
        }
        Ok(Box::new(Undying))
    }
}

/// Plan §4 crash path: "Never spawn while the old process is alive: two
/// ~17 GB processes would oversubscribe WDDM." `reap()` used to ignore a
/// failed `kill_and_wait`, and `boot()` loaded a second runner next to the
/// first; now the helper exits instead and its job object ends the runner.
#[test]
fn review_no_respawn_while_the_old_runner_is_alive() {
    let spawns = Arc::new(AtomicUsize::new(0));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let dir = std::env::temp_dir().join(format!("fidim-dg-undying-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = ServeConfig {
        runner: "fake-runner.exe".into(),
        model: "fake.gguf".into(),
        host: "127.0.0.1".into(),
        port: 0,
        alias: "dg-undying".into(),
        req_prefix: dir.join("dg-undying-0"),
        default_max_tokens: 2048,
        seed: None,
        expect_bus: Some(8),
        build_tag: None,
        queue_depth: 8,
        load_timeout: Duration::from_secs(10),
        watchdog: Duration::from_secs(10),
        ngl: 99,
        maxtok_env: 0,
    };
    let info = ModelInfo { canvas: 256, block_count: 30, n_ctx_train: 262_144, vocab: 0 };
    let h = start(listener, cfg, info, Box::new(UndyingSpawner(spawns.clone()))).unwrap();
    // reap(): 5 s exit grace, then kill_and_wait gives up and the helper
    // exits on its own (no shutdown is sent). A helper that respawned
    // instead would serve forever, so the wait is bounded.
    let (done_tx, done) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = done_tx.send(h.join());
    });
    let code = done.recv_timeout(Duration::from_secs(30));
    let _ = std::fs::remove_dir_all(dir);
    assert_eq!(spawns.load(Ordering::SeqCst), 1, "a second runner was spawned while the first was still alive");
    assert_eq!(code, Ok(super::EXIT_BREAKER));
}

impl TestServer {
    /// `stop` leaving the directory for the test to inspect.
    fn stop_keep_dir(mut self) -> i32 {
        let h = self.handle.take().unwrap();
        h.shutdown();
        let code = h.join();
        self.dir = PathBuf::new();
        code
    }
}
