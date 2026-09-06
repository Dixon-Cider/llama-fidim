//! Benchmark sweep (R-09): warm up, measure serial and N-concurrent
//! throughput against a running server, and attach results to the exact
//! profile revision that produced them. Cold-cache first runs are marked and
//! excluded from saved baselines (R-08).
//!
//! Measurements use llama-server's native `/completion` endpoint because its
//! `timings` block reports server-side decode rate (excludes prompt
//! processing and client overhead); the aggregate rate is wall-clock across
//! all streams, which is what a caller actually experiences.

use std::time::{Duration, Instant};

use serde::Serialize;

use crate::profile::Profile;
use crate::supervise::http_post_json;
use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub warmups: u32,
    /// Decode length per measured request (the historical sweeps used 256).
    pub max_tokens: u32,
    /// Concurrent stream count; 0 or 1 skips the concurrent phase.
    pub concurrency: u32,
    pub request_timeout: Duration,
}

impl Default for BenchOptions {
    fn default() -> Self {
        BenchOptions {
            warmups: 2,
            max_tokens: 256,
            concurrency: 1,
            request_timeout: Duration::from_secs(600),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamMeasure {
    pub tokens: u64,
    pub wall_seconds: f64,
    /// Client wall-clock rate (includes prompt + queueing).
    pub wall_tok_s: f64,
    /// Server-reported decode rate (`timings.predicted_per_second`), when
    /// the endpoint provides it.
    pub decode_tok_s: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConcurrentMeasure {
    pub n: u32,
    /// Mean of per-stream decode rates.
    pub per_stream_tok_s: f64,
    /// Total generated tokens / wall time across all streams — what a
    /// caller actually experiences (includes prompt phases + admission).
    pub aggregate_tok_s: f64,
    /// Sum of per-stream decode rates — decode-phase capacity, the number
    /// the historical batch-file sweep tables called "aggregate"
    /// (e.g. 57.4 x 6 = 344.4).
    pub decode_aggregate_tok_s: f64,
    pub streams: Vec<StreamMeasure>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SweepResult {
    pub warmups: u32,
    pub serial: StreamMeasure,
    pub concurrent: Option<ConcurrentMeasure>,
}

/// Distinct prompts per stream so slot prompt-caching cannot serve one
/// stream's prefill from another's.
fn prompt_for(stream: usize) -> String {
    format!(
        "You are stream {stream}. Write a detailed, numbered walkthrough of how a residential \
         fire alarm system is installed, inspected, and certified, step by step:"
    )
}

fn completion_body(prompt: &str, n_predict: u32) -> String {
    serde_json::json!({
        "prompt": prompt,
        "n_predict": n_predict,
        "cache_prompt": false,
        "stream": false,
    })
    .to_string()
}

/// One `/completion` request, measured. Falls back to `/v1/completions` if
/// the native endpoint is missing (format drift across builds).
fn measure_one(
    host: &str,
    port: u16,
    alias: &str,
    prompt: &str,
    n_predict: u32,
    timeout: Duration,
) -> Result<StreamMeasure> {
    let t0 = Instant::now();
    let (status, body) =
        http_post_json(host, port, "/completion", &completion_body(prompt, n_predict), timeout)?;
    if status == 200 {
        let wall = t0.elapsed().as_secs_f64();
        let v: serde_json::Value = serde_json::from_str(body_json(&body))
            .map_err(|e| Error::Platform(format!("bad /completion response: {e}")))?;
        let timings = v.get("timings");
        let tokens = timings
            .and_then(|t| t.get("predicted_n"))
            .and_then(|x| x.as_u64())
            .unwrap_or(n_predict as u64);
        let decode = timings
            .and_then(|t| t.get("predicted_per_second"))
            .and_then(|x| x.as_f64());
        return Ok(StreamMeasure {
            tokens,
            wall_seconds: wall,
            wall_tok_s: tokens as f64 / wall.max(1e-9),
            decode_tok_s: decode,
        });
    }
    // Fallback: OpenAI-compatible endpoint, wall-clock only.
    let body = serde_json::json!({
        "model": alias, "prompt": prompt, "max_tokens": n_predict, "stream": false,
    })
    .to_string();
    let t0 = Instant::now();
    let (status, resp) = http_post_json(host, port, "/v1/completions", &body, timeout)?;
    if status != 200 {
        return Err(Error::Platform(format!("completion request failed: HTTP {status}")));
    }
    let wall = t0.elapsed().as_secs_f64();
    let v: serde_json::Value = serde_json::from_str(body_json(&resp))
        .map_err(|e| Error::Platform(format!("bad /v1/completions response: {e}")))?;
    let tokens = v
        .pointer("/usage/completion_tokens")
        .and_then(|x| x.as_u64())
        .unwrap_or(n_predict as u64);
    Ok(StreamMeasure {
        tokens,
        wall_seconds: wall,
        wall_tok_s: tokens as f64 / wall.max(1e-9),
        decode_tok_s: None,
    })
}

/// Chunked transfer bodies arrive with chunk-size lines; strip to the JSON
/// object (first `{` to last `}`).
fn body_json(body: &str) -> &str {
    let start = body.find('{').unwrap_or(0);
    let end = body.rfind('}').map(|i| i + 1).unwrap_or(body.len());
    &body[start..end]
}

pub fn run_sweep(
    host: &str,
    port: u16,
    alias: &str,
    opts: &BenchOptions,
) -> Result<SweepResult> {
    // Warmups: short generations to page weights in and settle clocks.
    for i in 0..opts.warmups {
        measure_one(host, port, alias, &prompt_for(1000 + i as usize), 32, opts.request_timeout)?;
    }
    // Serial.
    let serial = measure_one(host, port, alias, &prompt_for(0), opts.max_tokens, opts.request_timeout)?;
    // Concurrent.
    let concurrent = if opts.concurrency > 1 {
        let n = opts.concurrency;
        let t0 = Instant::now();
        let results: Vec<Result<StreamMeasure>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..n)
                .map(|i| {
                    let host = host.to_string();
                    let alias = alias.to_string();
                    let timeout = opts.request_timeout;
                    let max_tokens = opts.max_tokens;
                    scope.spawn(move || {
                        measure_one(&host, port, &alias, &prompt_for(i as usize), max_tokens, timeout)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().expect("bench thread panicked")).collect()
        });
        let wall = t0.elapsed().as_secs_f64();
        let mut streams = Vec::new();
        for r in results {
            streams.push(r?);
        }
        Some(summarize_concurrent(streams, wall))
    } else {
        None
    };
    Ok(SweepResult { warmups: opts.warmups, serial, concurrent })
}

/// Pure math, unit-tested: per-stream mean + wall-clock aggregate.
pub fn summarize_concurrent(streams: Vec<StreamMeasure>, wall_seconds: f64) -> ConcurrentMeasure {
    let n = streams.len() as u32;
    let total_tokens: u64 = streams.iter().map(|s| s.tokens).sum();
    let decode_sum: f64 =
        streams.iter().map(|s| s.decode_tok_s.unwrap_or(s.wall_tok_s)).sum();
    let per_stream: f64 =
        if streams.is_empty() { 0.0 } else { decode_sum / streams.len() as f64 };
    ConcurrentMeasure {
        n,
        per_stream_tok_s: per_stream,
        aggregate_tok_s: total_tokens as f64 / wall_seconds.max(1e-9),
        decode_aggregate_tok_s: decode_sum,
        streams,
    }
}

// ------------------------------------------------------- baseline records ----

/// Fingerprint of everything that affects a launch — baselines are stored
/// against this, so a changed profile visibly orphans old numbers (R-09
/// "the exact profile revision that produced them").
pub fn profile_fingerprint(p: &Profile) -> String {
    let canonical = serde_json::json!([
        p.build.path, p.build.version, p.model.path, p.model.mmproj,
        p.model.draft.as_ref().map(|d| (d.path.clone(), d.enabled)),
        p.devices.iter().map(|d| (d.key.clone(), d.split_fraction)).collect::<Vec<_>>(),
        p.split_mode, p.main_device,
        serde_json::to_value(&p.runtime).unwrap_or_default(),
        p.env,
    ]);
    format!("{:016x}", fnv1a64(canonical.to_string().as_bytes()))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Unix seconds → ISO-8601 UTC (no chrono dependency for one timestamp).
pub fn iso8601_utc(secs: u64) -> String {
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

/// Howard Hinnant's civil-from-days algorithm.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(tokens: u64, wall: f64, decode: Option<f64>) -> StreamMeasure {
        StreamMeasure {
            tokens,
            wall_seconds: wall,
            wall_tok_s: tokens as f64 / wall,
            decode_tok_s: decode,
        }
    }

    #[test]
    fn concurrent_summary_matches_hand_math() {
        // 6 streams x 256 tokens, staggered decode rates, 4.5s wall.
        let streams: Vec<StreamMeasure> =
            (0..6).map(|i| stream(256, 4.0 + i as f64 * 0.1, Some(55.0 + i as f64))).collect();
        let c = summarize_concurrent(streams, 4.5);
        assert_eq!(c.n, 6);
        assert!((c.per_stream_tok_s - 57.5).abs() < 1e-9); // mean of 55..60
        assert!((c.aggregate_tok_s - (6.0 * 256.0 / 4.5)).abs() < 1e-9);
        assert!((c.decode_aggregate_tok_s - 345.0).abs() < 1e-9); // 55+..+60
    }

    #[test]
    fn fingerprint_tracks_launch_relevant_changes_only() {
        let mut p: Profile = serde_json::from_value(serde_json::json!({
            "schema": 1, "id": "t", "name": "t",
            "build": { "path": "C:/b" }, "model": { "path": "E:/m.gguf" },
            "devices": [ { "key": "pci:A:bus08" } ],
            "server": { "port": 9701, "alias": "t" },
            "runtime": { "ctx_total": 8192 }
        }))
        .unwrap();
        let f1 = profile_fingerprint(&p);
        // Rename and notes do not orphan baselines.
        p.name = "renamed".into();
        p.notes = "some notes".into();
        assert_eq!(profile_fingerprint(&p), f1);
        // A runtime change does.
        p.runtime.ctx_total = 16384;
        assert_ne!(profile_fingerprint(&p), f1);
    }

    #[test]
    fn iso8601_matches_known_timestamps() {
        assert_eq!(iso8601_utc(0), "1970-01-01T00:00:00Z");
        // 2026-07-29T14:02:00Z (the spec's example baseline timestamp).
        assert_eq!(iso8601_utc(1_785_333_720), "2026-07-29T14:02:00Z");
    }

    #[test]
    fn body_json_strips_chunked_framing() {
        let chunked = "7f\r\n{\"content\": \"hi\", \"timings\": {\"predicted_n\": 3}}\r\n0\r\n\r\n";
        let v: serde_json::Value = serde_json::from_str(body_json(chunked)).unwrap();
        assert_eq!(v.pointer("/timings/predicted_n").unwrap().as_u64(), Some(3));
    }
}
