//! Pure parsing of the DiffusionGemma runner's output and building of its
//! request file. Ground truth is Unsloth's
//! `examples/diffusion-gemma-server/diffusion-gemma-visual-server.cpp` at
//! f6b9ea7 (identical to release b11027-mix-3e83366), cited as VS:<line>.
//!
//! Stdout carries the line protocol; stderr carries llama.cpp's log, from
//! which only a handful of facts are lifted for the device guard and the
//! crash classifier. Everything here is tolerant of junk (it becomes
//! `Other` / `None`), because the fork changes every few days; the guard
//! that consumes the facts is the part that fails loudly.

use std::path::Path;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

/// Longest runner error text passed on to a client.
const MAX_ERR_CHARS: usize = 300;

/// One stdout record.
#[derive(Debug, Clone, PartialEq)]
pub enum Line {
    /// `READY <n_vocab> <maxtok>`; builds before the auto-sizer print only
    /// `READY <n_vocab>`.
    Ready { n_vocab: u32, maxtok: Option<u32> },
    /// `F <block> <step> <total> <json>`: one denoise step. The canvas text
    /// is discarded in v1 (no frame streaming).
    Frame { block: u32, step: u32, total: u32 },
    /// `C <block> <json>`: the CUMULATIVE committed answer after `block`.
    Commit { block: u32, text: String },
    Stats(Stats),
    Done,
    Err(ErrLine),
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrLine {
    /// The request file was missing or empty (VS:302).
    BadReq,
    /// The chat template or message parser threw (VS:316). The message can
    /// span several lines (tool dumps); only its first line is here.
    Parse(String),
    EmptyPrompt,
    /// `ERR toolong <needed> <budget>`: prompt + one canvas exceeds MAXTOK.
    TooLong { needed: u32, budget: u32 },
    /// Block 0 generated nothing (VS:347).
    Gen,
    Unknown(String),
}

impl ErrLine {
    /// These three end a request with no `DONE` after them (VS:302, 316,
    /// 318: the loop `continue`s straight to the next stdin line).
    pub fn terminal_without_done(&self) -> bool {
        matches!(self, ErrLine::BadReq | ErrLine::Parse(_) | ErrLine::EmptyPrompt)
    }
}

/// `STATS k=v …` (VS:375-378).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stats {
    pub prompt_n: u64,
    pub predicted_n: u64,
    /// Host template + tokenize time, not a GPU prefill.
    pub prompt_prepare_ms: f64,
    /// The generation loop the user waited on, frames included.
    pub wall_ms: f64,
    /// `wall_ms` minus the host visualization overhead.
    pub decode_ms: f64,
    pub blocks: u32,
    pub steps: u32,
    pub canvas: u32,
    pub n_ctx: u32,
}

fn truncate_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Parse one stdout line. Trailing CR/LF is trimmed, invalid UTF-8 is
/// replaced, and anything unrecognised is `Other`.
pub fn parse_line(raw: &[u8]) -> Line {
    let mut end = raw.len();
    while end > 0 && matches!(raw[end - 1], b'\n' | b'\r') {
        end -= 1;
    }
    let text = String::from_utf8_lossy(&raw[..end]);
    parse_text(&text).unwrap_or(Line::Other)
}

fn parse_text(s: &str) -> Option<Line> {
    if s == "DONE" {
        return Some(Line::Done);
    }
    if let Some(rest) = s.strip_prefix("READY ") {
        let mut it = rest.split_whitespace();
        let n_vocab = it.next()?.parse().ok()?;
        let maxtok = it.next().and_then(|t| t.parse().ok());
        return Some(Line::Ready { n_vocab, maxtok });
    }
    if s.starts_with("F ") {
        let p: Vec<&str> = s.splitn(5, ' ').collect();
        if p.len() < 5 {
            return None;
        }
        return Some(Line::Frame {
            block: p[1].parse().ok()?,
            step: p[2].parse().ok()?,
            total: p[3].parse().ok()?,
        });
    }
    if s.starts_with("C ") {
        let p: Vec<&str> = s.splitn(3, ' ').collect();
        if p.len() < 3 {
            return None;
        }
        let block = p[1].parse().ok()?;
        let text = serde_json::from_str::<String>(p[2]).ok()?;
        return Some(Line::Commit { block, text });
    }
    if s == "STATS" || s.starts_with("STATS ") {
        return Some(Line::Stats(parse_stats(&s[5..])));
    }
    if s == "ERR" || s.starts_with("ERR ") {
        return Some(Line::Err(parse_err(s[3..].trim_start())));
    }
    None
}

fn parse_stats(rest: &str) -> Stats {
    let mut st = Stats::default();
    // The canonical names win over the older aliases whatever their order.
    let (mut prep_alias, mut wall_alias) = (None, None);
    let (mut prep_seen, mut wall_seen) = (false, false);
    for tok in rest.split_whitespace() {
        let Some((k, v)) = tok.split_once('=') else { continue };
        let Ok(x) = v.parse::<f64>() else { continue };
        let n = if x.is_finite() && x > 0.0 { x.round() } else { 0.0 };
        match k {
            "prompt_n" => st.prompt_n = n as u64,
            "predicted_n" => st.predicted_n = n as u64,
            "prompt_prepare_ms" => {
                st.prompt_prepare_ms = x;
                prep_seen = true;
            }
            "prompt_ms" => prep_alias = Some(x),
            "wall_ms" => {
                st.wall_ms = x;
                wall_seen = true;
            }
            "predicted_ms" => wall_alias = Some(x),
            "decode_ms" => st.decode_ms = x,
            "blocks" => st.blocks = n as u32,
            "steps" => st.steps = n as u32,
            "canvas" => st.canvas = n as u32,
            "n_ctx" => st.n_ctx = n as u32,
            _ => {}
        }
    }
    if !prep_seen {
        st.prompt_prepare_ms = prep_alias.unwrap_or(0.0);
    }
    if !wall_seen {
        st.wall_ms = wall_alias.unwrap_or(0.0);
    }
    st
}

fn parse_err(rest: &str) -> ErrLine {
    let (word, tail) = rest.split_once(' ').unwrap_or((rest, ""));
    match word {
        "badreq" => ErrLine::BadReq,
        "parse" => ErrLine::Parse(truncate_chars(tail, MAX_ERR_CHARS)),
        "emptyprompt" => ErrLine::EmptyPrompt,
        "gen" => ErrLine::Gen,
        "toolong" => {
            let mut it = tail.split_whitespace().map(|t| t.parse::<u32>().ok());
            match (it.next().flatten(), it.next().flatten()) {
                (Some(needed), Some(budget)) => ErrLine::TooLong { needed, budget },
                _ => ErrLine::Unknown(truncate_chars(rest, MAX_ERR_CHARS)),
            }
        }
        _ => ErrLine::Unknown(truncate_chars(rest, MAX_ERR_CHARS)),
    }
}

// ------------------------------------------------------------ request file ----

/// The runner's request file (VS:297-311): `{seed, n_blocks, messages,
/// tools?}`. The runner applies the GGUF's chat template itself.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineRequest {
    /// Block b is denoised with `seed + b` (VS:339).
    pub seed: i32,
    pub n_blocks: u32,
    pub messages: Value,
    pub tools: Option<Value>,
}

impl EngineRequest {
    /// UTF-8 JSON with no BOM (nlohmann's parser rejects one). An empty
    /// `tools` array is left out: the runner renders any `tools` key it sees
    /// into the template.
    pub fn to_json(&self) -> String {
        let mut o = serde_json::Map::new();
        o.insert("seed".into(), self.seed.into());
        o.insert("n_blocks".into(), self.n_blocks.into());
        o.insert("messages".into(), self.messages.clone());
        if let Some(t) = &self.tools {
            let empty = t.is_null() || t.as_array().is_some_and(|a| a.is_empty());
            if !empty {
                o.insert("tools".into(), t.clone());
            }
        }
        Value::Object(o).to_string()
    }
}

/// The runner reads the request path with a narrow `fgets` into a 4 KiB
/// buffer and opens it with `fopen` (VS:288, 45): ASCII only, one line.
pub fn check_req_prefix(p: &Path) -> Result<(), String> {
    let Some(s) = p.to_str() else {
        return Err(format!("request path {} is not valid Unicode", p.display()));
    };
    if !s.is_ascii() {
        return Err(format!("request path {s} is not ASCII (the runner opens it with a narrow fopen)"));
    }
    if s.len() > 200 {
        return Err(format!("request path is {} characters long (limit 200)", s.len()));
    }
    if s.contains('\r') || s.contains('\n') {
        return Err("request path contains a line break".into());
    }
    Ok(())
}

// ------------------------------------------------------------ stderr facts ----

/// What the helper needs to know from the runner's llama.cpp log.
#[derive(Debug, Clone, PartialEq)]
pub enum Fact {
    /// `llama_prepare_model_devices: using device ROCm0 (…) (0000:08:00.0)`
    /// (src/llama.cpp:306). `bus` is None when the backend has no PCI id.
    UsingDevice { name: String, bus: Option<u32> },
    /// The runner's own `… ready (… NGL=<n> … kv_cache=on)` (VS:281-285).
    RunnerReady { ngl: Option<u32>, kv_cache_on: bool },
    /// `load_tensors: offloaded X/Y layers to GPU` (llama-model.cpp:1860).
    Offloaded { done: u32, total: u32 },
    /// A denoise or prefill decode failed but the runner kept going
    /// (diffusion.cpp:265, 272, 514, 562).
    StepFailure,
    /// A line that explains a crash: `"oom"` or `"utf8"`.
    CrashHint(&'static str),
}

fn regex(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("static regex"))
}

pub fn parse_stderr_fact(line: &str) -> Option<Fact> {
    static USING_BUS: OnceLock<Regex> = OnceLock::new();
    static USING: OnceLock<Regex> = OnceLock::new();
    static NGL: OnceLock<Regex> = OnceLock::new();
    static KV: OnceLock<Regex> = OnceLock::new();
    static OFFLOADED: OnceLock<Regex> = OnceLock::new();

    let line = line.trim_end_matches(['\r', '\n']);
    // `: using` (not `already using`, which the same function prints for a
    // skipped duplicate device, src/llama.cpp:248).
    if line.contains(": using device ") {
        let with_bus = regex(
            &USING_BUS,
            r": using device (\S+) \(.*\) \(([0-9a-fA-F]{4}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.\d\)",
        );
        if let Some(c) = with_bus.captures(line) {
            return Some(Fact::UsingDevice {
                name: c[1].to_string(),
                bus: u32::from_str_radix(&c[3], 16).ok(),
            });
        }
        if let Some(c) = regex(&USING, r": using device (\S+) ").captures(line) {
            return Some(Fact::UsingDevice { name: c[1].to_string(), bus: None });
        }
    }
    if line.starts_with("diffusion-gemma-visual-server ready (") {
        let ngl = regex(&NGL, r"\bNGL=(\d+)").captures(line).and_then(|c| c[1].parse().ok());
        let kv = regex(&KV, r"\bkv_cache=(on|off)\b").captures(line).map(|c| &c[1] == "on");
        return Some(Fact::RunnerReady { ngl, kv_cache_on: kv.unwrap_or(false) });
    }
    if let Some(c) = regex(&OFFLOADED, r"offloaded (\d+)/(\d+) layers to GPU").captures(line) {
        if let (Ok(done), Ok(total)) = (c[1].parse(), c[2].parse()) {
            return Some(Fact::Offloaded { done, total });
        }
    }
    if line.contains("failed to decode at step")
        || line.contains("PREFILL chunk")
        || line.contains("failed to get logits")
    {
        return Some(Fact::StepFailure);
    }
    if line.contains("pkv_buf != nullptr") || line.contains("out of memory") || line.contains("failed to allocate") {
        return Some(Fact::CrashHint("oom"));
    }
    if line.contains("type_error.316") || line.contains("invalid UTF-8") {
        return Some(Fact::CrashHint("utf8"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ready_with_and_without_maxtok() {
        assert_eq!(parse_line(b"READY 262144 12288\n"), Line::Ready { n_vocab: 262144, maxtok: Some(12288) });
        assert_eq!(parse_line(b"READY 262144\r\n"), Line::Ready { n_vocab: 262144, maxtok: None });
        assert_eq!(parse_line(b"READY"), Line::Other);
        assert_eq!(parse_line(b"READY x 1"), Line::Other);
    }

    #[test]
    fn frames_and_commits() {
        // The frame payload (escaped newline, raw UTF-8) is discarded but
        // must not break the header fields.
        let f = "F 1 7 48 \"line one\\nzwei \u{00fc}ber \u{1f600} x y\"\n";
        assert_eq!(parse_line(f.as_bytes()), Line::Frame { block: 1, step: 7, total: 48 });
        assert_eq!(parse_line(b"F 1 7 48"), Line::Other);

        let c = "C 2 \"<|channel>thought\\nhm<channel|>Gr\u{00fc}\u{00df} Gott \\\"x\\\"\"\r\n";
        assert_eq!(
            parse_line(c.as_bytes()),
            Line::Commit { block: 2, text: "<|channel>thought\nhm<channel|>Gr\u{00fc}\u{00df} Gott \"x\"".into() }
        );
        // Not a JSON string → Other, never a half-decoded commit.
        assert_eq!(parse_line(b"C 0 not-json"), Line::Other);
    }

    #[test]
    fn stats_with_aliases_and_unknown_keys() {
        let l = parse_line(
            b"STATS prompt_n=28 predicted_n=347 prompt_prepare_ms=5.799 wall_ms=6904.409 decode_ms=6903.117 blocks=2 steps=32 canvas=256 n_ctx=12288 future_key=1\n",
        );
        let Line::Stats(s) = l else { panic!("{l:?}") };
        assert_eq!((s.prompt_n, s.predicted_n, s.blocks, s.steps, s.canvas, s.n_ctx), (28, 347, 2, 32, 256, 12288));
        assert!((s.wall_ms - 6904.409).abs() < 1e-9);
        assert!((s.prompt_prepare_ms - 5.799).abs() < 1e-9);

        let Line::Stats(old) = parse_line(b"STATS prompt_n=3 predicted_n=4 prompt_ms=1.5 predicted_ms=250") else {
            panic!()
        };
        assert_eq!(old.prompt_prepare_ms, 1.5);
        assert_eq!(old.wall_ms, 250.0);
        // Canonical beats alias regardless of order.
        let Line::Stats(both) = parse_line(b"STATS predicted_ms=1 wall_ms=2") else { panic!() };
        assert_eq!(both.wall_ms, 2.0);
    }

    #[test]
    fn done_and_every_err_variant() {
        assert_eq!(parse_line(b"DONE\r\n"), Line::Done);
        assert_eq!(parse_line(b"ERR badreq"), Line::Err(ErrLine::BadReq));
        assert_eq!(parse_line(b"ERR emptyprompt\n"), Line::Err(ErrLine::EmptyPrompt));
        assert_eq!(parse_line(b"ERR gen"), Line::Err(ErrLine::Gen));
        assert_eq!(
            parse_line(b"ERR toolong 12544 12288\n"),
            Line::Err(ErrLine::TooLong { needed: 12544, budget: 12288 })
        );
        assert_eq!(
            parse_line(b"ERR parse Failed to parse messages: Missing 'role' in message"),
            Line::Err(ErrLine::Parse("Failed to parse messages: Missing 'role' in message".into()))
        );
        let long = format!("ERR parse {}", "x".repeat(1000));
        let Line::Err(ErrLine::Parse(m)) = parse_line(long.as_bytes()) else { panic!() };
        assert_eq!(m.chars().count(), 300);
        assert_eq!(parse_line(b"ERR something new"), Line::Err(ErrLine::Unknown("something new".into())));
        assert_eq!(parse_line(b"ERR toolong x"), Line::Err(ErrLine::Unknown("toolong x".into())));
        assert!(ErrLine::Parse(String::new()).terminal_without_done());
        assert!(ErrLine::BadReq.terminal_without_done());
        assert!(ErrLine::EmptyPrompt.terminal_without_done());
        assert!(!ErrLine::Gen.terminal_without_done());
        assert!(!ErrLine::TooLong { needed: 1, budget: 1 }.terminal_without_done());
    }

    #[test]
    fn crlf_trimmed_and_garbage_is_other() {
        assert_eq!(parse_line(b"DONE\r\r\n"), Line::Done);
        assert_eq!(parse_line(b""), Line::Other);
        assert_eq!(parse_line(b"    \"tools\": ["), Line::Other);
        assert_eq!(parse_line(&[0xff, 0xfe, b'\n']), Line::Other);
        assert_eq!(parse_line(b"DONEX"), Line::Other);
    }

    #[test]
    fn stderr_facts_from_the_run10_log() {
        // run10-shipped-no-hipblaslt.log lines 67, 1336 and 1379, verbatim.
        assert_eq!(
            parse_stderr_fact(
                "llama_prepare_model_devices: using device ROCm0 (AMD Radeon AI PRO R9700) (0000:08:00.0) - 32472 MiB free"
            ),
            Some(Fact::UsingDevice { name: "ROCm0".into(), bus: Some(8) })
        );
        assert_eq!(
            parse_stderr_fact("load_tensors: offloaded 31/31 layers to GPU"),
            Some(Fact::Offloaded { done: 31, total: 31 })
        );
        assert_eq!(
            parse_stderr_fact(
                "diffusion-gemma-visual-server ready (n_vocab=262144, canvas=256, MAXTOK=12288, NGL=99, gpu_sampling=on sample_reduce=on kv_cache=on)\r\n"
            ),
            Some(Fact::RunnerReady { ngl: Some(99), kv_cache_on: true })
        );
    }

    #[test]
    fn stderr_fact_edge_cases() {
        // The bus is hex.
        assert_eq!(
            parse_stderr_fact("f: using device ROCm1 (AMD Radeon (TM) Graphics) (0000:1a:00.0) - 100 MiB free"),
            Some(Fact::UsingDevice { name: "ROCm1".into(), bus: Some(0x1a) })
        );
        assert_eq!(
            parse_stderr_fact("f: using device CPU (AMD Ryzen) (unknown id) - 100 MiB free"),
            Some(Fact::UsingDevice { name: "CPU".into(), bus: None })
        );
        // A skipped duplicate device is not a device in use.
        assert_eq!(
            parse_stderr_fact(
                "f: skipping device ROCm1 (x) with id 0000:08:00.0 - already using device ROCm0 (x) with the same id"
            ),
            None
        );
        assert_eq!(
            parse_stderr_fact("diffusion-gemma-visual-server ready (n_vocab=1, NGL=0, kv_cache=off)"),
            Some(Fact::RunnerReady { ngl: Some(0), kv_cache_on: false })
        );
        assert_eq!(
            parse_stderr_fact("diffusion_generate: failed to decode at step 3"),
            Some(Fact::StepFailure)
        );
        assert_eq!(parse_stderr_fact("x: PREFILL chunk [0,2048) decode failed"), Some(Fact::StepFailure));
        assert_eq!(parse_stderr_fact("x: failed to get logits at step 9"), Some(Fact::StepFailure));
        assert_eq!(
            parse_stderr_fact("D:\\a\\src\\models\\diffusion-gemma.cpp:818: GGML_ASSERT(m.pkv_buf != nullptr) failed"),
            Some(Fact::CrashHint("oom"))
        );
        assert_eq!(parse_stderr_fact("hipMalloc: out of memory"), Some(Fact::CrashHint("oom")));
        assert_eq!(
            parse_stderr_fact("terminate called: [json.exception.type_error.316] invalid UTF-8 byte at index 3"),
            Some(Fact::CrashHint("utf8"))
        );
        assert_eq!(parse_stderr_fact("decode: cannot decode batches with this context (calling encode() instead)"), None);
    }

    #[test]
    fn request_json_has_no_bom_and_omits_empty_tools() {
        let mut r = EngineRequest {
            seed: 7,
            n_blocks: 2,
            messages: serde_json::json!([{"role": "user", "content": "h\u{00e9}"}]),
            tools: Some(serde_json::json!([])),
        };
        let j = r.to_json();
        assert!(!j.starts_with('\u{feff}'));
        assert_eq!(j, "{\"seed\":7,\"n_blocks\":2,\"messages\":[{\"role\":\"user\",\"content\":\"h\u{00e9}\"}]}");
        r.tools = None;
        assert!(!r.to_json().contains("tools"));
        r.tools = Some(serde_json::json!([{"type": "function", "function": {"name": "f"}}]));
        let v: Value = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(v["tools"][0]["function"]["name"], "f");
    }

    #[test]
    fn req_prefix_rules() {
        assert!(check_req_prefix(Path::new("C:\\Users\\x\\.fidim\\runs\\dg-a-1234")).is_ok());
        assert!(check_req_prefix(Path::new("C:\\Users\\J\u{00fc}rgen\\runs\\dg-a-1")).is_err());
        assert!(check_req_prefix(&Path::new("C:\\").join("a".repeat(201))).is_err());
        assert!(check_req_prefix(Path::new("C:\\x\ny")).is_err());
    }
}
