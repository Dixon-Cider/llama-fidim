//! M21 — live visibility into running servers: per-slot phase and progress
//! from `/slots`, throughput counters from `/metrics`, per-card GPU busy
//! from PDH. Polled by the Running view at ~1 Hz; measured 2026-09-05 as
//! having no effect on decode throughput (the server answers between
//! decode steps), ~10 ms per poll on the client side.
//!
//! b10819 `/slots` exposes `n_prompt_tokens` / `n_prompt_tokens_processed`
//! while prefilling and `next_token[0].n_decoded` / `n_remain` while
//! decoding, so both phases get a real progress figure.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::Serialize;

use crate::supervise::http_get;
use crate::{Error, Result};

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SlotView {
    pub id: u64,
    pub n_ctx: u64,
    pub is_processing: bool,
    /// `idle`, `prefill`, `decode`.
    pub phase: String,
    pub n_prompt_tokens: u64,
    pub n_prompt_tokens_processed: u64,
    pub n_prompt_tokens_cache: u64,
    pub n_decoded: u64,
    pub n_remain: i64,
    /// Prompt tokens processed / total, 0..1 (1.0 once prefill is done).
    pub prefill_fraction: f64,
    /// Tokens in the slot's context / n_ctx, 0..1.
    pub ctx_fraction: f64,
    /// Tail of the prompt text the slot last received. Present only when
    /// the server runs with `LLAMA_SERVER_SLOTS_DEBUG=1` (the profile's
    /// "trace tokens" switch); the server then detokenizes it per poll.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    pub prompt_chars: usize,
    /// Tail of the text generated so far (or the last completed generation
    /// while idle). Same gate as `prompt`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated: Option<String>,
    pub generated_chars: usize,
    /// A fragment that repeats back-to-back at the end of `generated`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loop_hint: Option<LoopHint>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LoopHint {
    pub fragment: String,
    pub repeats: usize,
}

/// Keep the last `n` chars of `s` (char-safe).
fn tail(s: &str, n: usize) -> String {
    let count = s.chars().count();
    if count <= n { s.to_string() } else { s.chars().skip(count - n).collect() }
}

/// Endless-loop detector for a generation: the tail of `text` is the same
/// block repeated `repeats` times back-to-back. Period is searched from 6
/// up to 400 chars over the last 3000 chars; the smallest period that
/// repeats at least `min_repeats` times wins, so "the the the the" and a
/// whole repeated paragraph are both caught. Whitespace-only blocks are
/// ignored (a model printing blank lines is a different failure).
pub fn detect_loop(text: &str, min_repeats: usize) -> Option<LoopHint> {
    let chars: Vec<char> = tail(text, 3000).chars().collect();
    let n = chars.len();
    for period in 6..=400usize {
        if period * min_repeats > n { break; }
        let block = &chars[n - period..];
        if block.iter().all(|c| c.is_whitespace()) { continue; }
        let mut repeats = 1;
        while (repeats + 1) * period <= n && chars[n - (repeats + 1) * period..n - repeats * period] == *block {
            repeats += 1;
        }
        if repeats >= min_repeats {
            return Some(LoopHint { fragment: block.iter().collect(), repeats });
        }
    }
    None
}

/// Parse the `/slots` array. Unknown shapes degrade to idle slots rather
/// than failing the whole sample.
pub fn parse_slots(json: &str) -> Result<Vec<SlotView>> {
    let start = json.find('[').unwrap_or(0);
    let end = json.rfind(']').map(|i| i + 1).unwrap_or(json.len());
    let v: serde_json::Value = serde_json::from_str(&json[start..end])?;
    let arr = v.as_array().ok_or_else(|| Error::Platform("/slots is not an array".into()))?;
    Ok(arr
        .iter()
        .map(|s| {
            let g = |k: &str| s.get(k).and_then(|x| x.as_u64()).unwrap_or(0);
            let nt = s.get("next_token").and_then(|x| x.as_array()).and_then(|a| a.first()).cloned().unwrap_or_default();
            let n_decoded = nt.get("n_decoded").and_then(|x| x.as_u64()).unwrap_or(0);
            let n_remain = nt.get("n_remain").and_then(|x| x.as_i64()).unwrap_or(0);
            let is_processing = s.get("is_processing").and_then(|x| x.as_bool()).unwrap_or(false);
            let n_prompt = g("n_prompt_tokens");
            let processed = g("n_prompt_tokens_processed");
            let n_ctx = g("n_ctx");
            let phase = if !is_processing {
                "idle"
            } else if n_decoded == 0 {
                "prefill"
            } else {
                "decode"
            };
            let prefill_fraction = if n_prompt == 0 { 0.0 } else { (processed as f64 / n_prompt as f64).min(1.0) };
            let text = |k: &str| s.get(k).and_then(|x| x.as_str()).map(str::to_string);
            let prompt_full = text("prompt");
            let generated_full = text("generated");
            let loop_hint = generated_full.as_deref().and_then(|g| detect_loop(g, 3));
            SlotView {
                id: g("id"),
                n_ctx,
                is_processing,
                phase: phase.into(),
                n_prompt_tokens: n_prompt,
                n_prompt_tokens_processed: processed,
                n_prompt_tokens_cache: g("n_prompt_tokens_cache"),
                n_decoded,
                n_remain,
                prefill_fraction: if phase == "decode" { 1.0 } else { prefill_fraction },
                ctx_fraction: if n_ctx == 0 { 0.0 } else { (n_prompt as f64 / n_ctx as f64).min(1.0) },
                prompt_chars: prompt_full.as_deref().map(|t| t.chars().count()).unwrap_or(0),
                prompt: prompt_full.as_deref().map(|t| tail(t, 4000)),
                generated_chars: generated_full.as_deref().map(|t| t.chars().count()).unwrap_or(0),
                generated: generated_full.as_deref().map(|t| tail(t, 4000)),
                loop_hint,
            }
        })
        .collect())
}

/// `llamacpp:tokens_predicted_total 329` lines → map of the key after the
/// colon to its value. Comments and other families are ignored.
pub fn parse_metrics(text: &str) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("llamacpp:") else { continue };
        let mut it = rest.split_whitespace();
        if let (Some(k), Some(v)) = (it.next(), it.next()) {
            if let Ok(f) = v.parse::<f64>() {
                out.insert(k.to_string(), f);
            }
        }
    }
    out
}

#[derive(Debug, Clone, Serialize)]
pub struct LiveSample {
    /// Model id on a router, None for a standalone server.
    pub model: Option<String>,
    pub sampled_unix_ms: u128,
    pub slots: Vec<SlotView>,
    /// Aggregate phase: `decode` if any slot decodes, else `prefill` if any
    /// prefills, else `idle`.
    pub phase: String,
    pub metrics: BTreeMap<String, f64>,
    pub error: Option<String>,
}

fn query(path: &str, model: Option<&str>) -> String {
    match model {
        Some(m) => format!("{path}?model={}", urlencode(m)),
        None => path.to_string(),
    }
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// One poll of a server (or one model behind a router).
pub fn sample(host: &str, port: u16, model: Option<&str>) -> LiveSample {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let mut out = LiveSample { model: model.map(str::to_string), sampled_unix_ms: now, slots: vec![], phase: "idle".into(), metrics: BTreeMap::new(), error: None };
    let t = Duration::from_secs(3);
    match http_get(host, port, &query("/slots", model), t) {
        Ok((200, body)) => match parse_slots(&body) {
            Ok(s) => out.slots = s,
            Err(e) => out.error = Some(format!("slots: {e}")),
        },
        Ok((code, _)) => out.error = Some(format!("/slots HTTP {code}")),
        Err(e) => out.error = Some(format!("/slots: {e}")),
    }
    if let Ok((200, body)) = http_get(host, port, &query("/metrics", model), t) {
        out.metrics = parse_metrics(&body);
    }
    out.phase = if out.slots.iter().any(|s| s.phase == "decode") {
        "decode".into()
    } else if out.slots.iter().any(|s| s.phase == "prefill") {
        "prefill".into()
    } else {
        "idle".into()
    };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLOTS: &str = r#"[{"id":5,"n_ctx":32768,"speculative":false,"is_processing":true,"id_task":1847,"n_prompt_tokens":2468,"n_prompt_tokens_processed":1956,"n_prompt_tokens_cache":0,"next_token":[{"has_next_token":false,"has_new_line":false,"n_remain":40,"n_decoded":0}],"prompt":""},
    {"id":6,"n_ctx":32768,"speculative":false,"is_processing":true,"n_prompt_tokens":2494,"n_prompt_tokens_processed":2472,"next_token":[{"has_next_token":true,"n_remain":18,"n_decoded":22}]},
    {"id":7,"n_ctx":32768,"speculative":false,"is_processing":false}]"#;

    #[test]
    fn slots_phases_and_progress() {
        let s = parse_slots(SLOTS).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].phase, "prefill");
        assert!((s[0].prefill_fraction - 1956.0 / 2468.0).abs() < 1e-9);
        assert_eq!(s[1].phase, "decode");
        assert_eq!(s[1].n_decoded, 22);
        assert_eq!(s[1].n_remain, 18);
        assert_eq!(s[1].prefill_fraction, 1.0);
        assert_eq!(s[2].phase, "idle");
        assert_eq!(s[2].n_ctx, 32768);
    }

    #[test]
    fn slot_text_and_loop_detection() {
        let j = r#"[{"id":0,"n_ctx":8192,"is_processing":true,"n_prompt_tokens":10,"n_prompt_tokens_processed":10,"next_token":[{"n_remain":-1,"n_decoded":40}],"prompt":"<|turn>user\nhello","generated":"I will help. I will help. I will help. I will help. "}]"#;
        let s = parse_slots(j).unwrap();
        assert_eq!(s[0].prompt.as_deref(), Some("<|turn>user\nhello"));
        assert_eq!(s[0].generated_chars, 52);
        let hint = s[0].loop_hint.clone().expect("loop");
        assert_eq!(hint.fragment, "I will help. ");
        assert_eq!(hint.repeats, 4);
        // Without slots debug the keys are absent: no text, no hint.
        let plain = parse_slots(SLOTS).unwrap();
        assert!(plain[1].prompt.is_none() && plain[1].generated.is_none() && plain[1].loop_hint.is_none());
    }

    #[test]
    fn loop_detector_ignores_normal_prose_and_blank_runs() {
        assert!(detect_loop("The quick brown fox jumps over the lazy dog and keeps going.", 3).is_none());
        assert!(detect_loop("text\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n", 3).is_none());
        // Two repeats is not yet a loop; three is.
        assert!(detect_loop("abcdefgh abcdefgh ", 3).is_none());
        assert_eq!(detect_loop("abcdefgh abcdefgh abcdefgh ", 3).unwrap().repeats, 3);
        // A repeated paragraph is caught with the paragraph as the fragment.
        let para = "Step 1: open the file. Step 2: read it. Step 3: close it.\n";
        let h = detect_loop(&para.repeat(5), 3).unwrap();
        assert_eq!(h.fragment, para);
        assert_eq!(h.repeats, 5);
    }

    #[test]
    fn metrics_parse() {
        let m = parse_metrics("# HELP x\nllamacpp:tokens_predicted_total 329\nllamacpp:predicted_tokens_seconds 51.1284\nother:thing 1\n");
        assert_eq!(m.get("tokens_predicted_total"), Some(&329.0));
        assert!((m["predicted_tokens_seconds"] - 51.1284).abs() < 1e-9);
        assert!(!m.contains_key("thing"));
    }

    #[test]
    fn router_query_is_encoded() {
        assert_eq!(query("/slots", Some("unsloth/x:Q4")), "/slots?model=unsloth%2Fx%3AQ4");
        assert_eq!(query("/slots", None), "/slots");
    }
}
