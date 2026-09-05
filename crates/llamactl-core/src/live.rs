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
