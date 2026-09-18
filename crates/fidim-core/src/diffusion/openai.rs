//! Pure OpenAI translation for the diffusion shim: request validation and
//! planning (shim spec §5), response shaping (§6) and the llama-server
//! error envelope (§7). No I/O here; `http.rs` drives it.

use serde_json::{json, Map, Value};

use super::engine::EngineFailure;
use super::protocol::{EngineRequest, Stats};
use super::XorShift64;

// ------------------------------------------------------------------ errors ----

/// An error in llama-server's envelope: `{"error":{"code","message","type",…}}`.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub kind: &'static str,
    pub message: String,
    /// Extra keys inside `error` (e.g. `n_prompt_tokens`, `n_ctx`).
    pub extra: Map<String, Value>,
}

impl ApiError {
    pub fn new(status: u16, kind: &'static str, message: impl Into<String>) -> Self {
        ApiError { status, kind, message: message.into(), extra: Map::new() }
    }
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(400, "invalid_request_error", message)
    }
    pub fn server(message: impl Into<String>) -> Self {
        Self::new(500, "server_error", message)
    }
    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(503, "unavailable_error", message)
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(404, "not_found_error", message)
    }

    pub fn body(&self) -> Value {
        let mut e = Map::new();
        e.insert("code".into(), self.status.into());
        e.insert("message".into(), self.message.clone().into());
        e.insert("type".into(), self.kind.into());
        for (k, v) in &self.extra {
            e.insert(k.clone(), v.clone());
        }
        json!({ "error": e })
    }
}

/// The runner's `ERR toolong <needed> <budget>` counts include the canvas
/// it wanted to append; clients think in prompt tokens, so both drop it.
pub fn exceed_context(needed: u32, budget: u32, canvas: u32) -> ApiError {
    let n_prompt = needed.saturating_sub(canvas);
    let n_ctx = budget.saturating_sub(canvas);
    let mut e = ApiError::new(
        400,
        "exceed_context_size_error",
        format!(
            "request ({n_prompt} tokens) exceeds the available context size ({n_ctx} tokens); \
             DiffusionGemma's context is bounded by this card's VRAM and the runner's ceiling"
        ),
    );
    e.extra.insert("n_prompt_tokens".into(), n_prompt.into());
    e.extra.insert("n_ctx".into(), n_ctx.into());
    e
}

/// A failed engine job as a client-facing error (shim spec §7).
pub fn failure_error(f: &EngineFailure) -> ApiError {
    match f {
        EngineFailure::Parse(m) => ApiError::invalid(format!("the chat template rejected this request: {m}")),
        EngineFailure::EmptyPrompt => ApiError::invalid("the chat template produced an empty prompt"),
        EngineFailure::BadReqFile => ApiError::server("the engine could not read its request file"),
        EngineFailure::Gen => ApiError::server("diffusion prefill or a denoise step failed on block 0 (see log)"),
        EngineFailure::StepFailed => ApiError::server("a denoise step failed (see log)"),
        // 400, not 5xx: a client that retries the same conversation hits
        // the same wall; one that compacts it succeeds.
        EngineFailure::Oom => ApiError::new(
            400,
            "exceed_context_size_error",
            "the prompt-KV store for this conversation did not fit in VRAM; the engine restarted; \
             shorten the conversation",
        ),
        EngineFailure::Crashed { exit, detail } => {
            let code = exit.map(exit_hex).unwrap_or_else(|| "unknown".into());
            if detail.is_empty() {
                ApiError::server(format!("diffusion engine crashed (exit {code})"))
            } else {
                ApiError::server(format!("diffusion engine crashed (exit {code}): {detail}"))
            }
        }
        EngineFailure::Watchdog(s) => {
            ApiError::server(format!("diffusion engine produced no output for {s} s; restarted"))
        }
        EngineFailure::Runner(m) => ApiError::server(format!("diffusion engine error: {m}")),
        EngineFailure::Unavailable => ApiError::unavailable("the diffusion engine is not available"),
    }
}

/// Windows exit codes read best as NTSTATUS hex (0xC0000409 = abort).
pub fn exit_hex(code: i32) -> String {
    format!("0x{:08X}", code as u32)
}

// ----------------------------------------------------------- request plan ----

#[derive(Debug, Clone, PartialEq)]
pub struct ChatPlan {
    pub req: EngineRequest,
    pub stream: bool,
    pub include_usage: bool,
    /// `reasoning_format: "none"`: thought markers stay in `content`.
    pub raw_reasoning: bool,
    pub stop: Vec<String>,
    /// The client (or the profile) chose the seed; retries must keep it.
    pub seed_pinned: bool,
    pub n_blocks: u32,
}

fn ceil_div(n: u64, d: u32) -> u64 {
    let d = d.max(1) as u64;
    n.div_ceil(d)
}

fn to_blocks(n: u64, canvas: u32) -> u32 {
    ceil_div(n, canvas).min(u32::MAX as u64) as u32
}

/// Validate a `/v1/chat/completions` body and plan the engine request.
/// `maxtok` is the runner's context budget (0 = not known yet: no clamp).
pub fn plan_chat(
    body: &Value,
    default_max_tokens: u32,
    profile_seed: Option<i64>,
    canvas: u32,
    maxtok: u32,
    rng: &mut XorShift64,
) -> Result<ChatPlan, ApiError> {
    let obj = body.as_object().ok_or_else(|| ApiError::invalid("the request body must be a JSON object"))?;

    let msgs = obj
        .get("messages")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| ApiError::invalid("'messages' must be a non-empty array"))?;
    let mut messages = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.iter().enumerate() {
        let Some(mo) = m.as_object() else {
            return Err(ApiError::invalid(format!("messages[{i}] must be an object")));
        };
        let role = match mo.get("role").and_then(Value::as_str) {
            Some("system") | Some("developer") => "system",
            Some(r @ ("user" | "assistant" | "tool")) => r,
            Some(other) => return Err(ApiError::invalid(format!("messages[{i}].role '{other}' is not supported"))),
            None => return Err(ApiError::invalid(format!("messages[{i}].role is missing"))),
        };
        match mo.get("content") {
            None | Some(Value::Null) | Some(Value::String(_)) => {}
            Some(Value::Array(parts)) => {
                for p in parts {
                    match p.get("type").and_then(Value::as_str).unwrap_or("") {
                        "text" | "media_marker" => {
                            if !p.get("text").is_some_and(Value::is_string) {
                                return Err(ApiError::invalid(format!(
                                    "messages[{i}].content: every text part needs a string 'text'"
                                )));
                            }
                        }
                        "image_url" | "input_audio" | "file" => {
                            return Err(ApiError::invalid("this profile runs a text-only diffusion model"))
                        }
                        other => {
                            return Err(ApiError::invalid(format!(
                                "messages[{i}].content part type '{other}' is not supported"
                            )))
                        }
                    }
                }
            }
            Some(_) => {
                return Err(ApiError::invalid(format!(
                    "messages[{i}].content must be a string, null or an array of text parts"
                )))
            }
        }
        let mut m = m.clone();
        m["role"] = role.into();
        messages.push(m);
    }
    let mut messages = Value::Array(messages);
    clean_messages(&mut messages);

    if let Some(n) = obj.get("n").filter(|v| !v.is_null()) {
        if n.as_i64().is_none_or(|n| n > 1) {
            return Err(ApiError::invalid("only n = 1 is supported"));
        }
    }

    let mut tools = match obj.get("tools") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) => {
            for (i, t) in a.iter().enumerate() {
                let ok = t.get("type").and_then(Value::as_str) == Some("function")
                    && t.get("function").and_then(|f| f.get("name")).is_some_and(Value::is_string);
                if !ok {
                    return Err(ApiError::invalid(format!(
                        "tools[{i}] must be {{\"type\":\"function\",\"function\":{{\"name\":…}}}}"
                    )));
                }
            }
            Some(a.clone())
        }
        Some(_) => return Err(ApiError::invalid("'tools' must be an array")),
    };
    match obj.get("tool_choice") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => match s.as_str() {
            "none" => tools = None,
            "auto" | "required" => {}
            other => return Err(ApiError::invalid(format!("tool_choice '{other}' is not supported"))),
        },
        Some(Value::Object(tc)) => {
            let Some(name) = tc.get("function").and_then(|f| f.get("name")).and_then(Value::as_str) else {
                return Err(ApiError::invalid("tool_choice.function.name is missing"));
            };
            let kept: Vec<Value> = tools
                .iter()
                .flatten()
                .filter(|t| t["function"]["name"].as_str() == Some(name))
                .cloned()
                .collect();
            if kept.is_empty() {
                return Err(ApiError::invalid(format!("tool_choice names '{name}', which is not in 'tools'")));
            }
            tools = Some(kept);
        }
        Some(_) => return Err(ApiError::invalid("'tool_choice' must be a string or an object")),
    }
    let tools = tools.map(|t| {
        let mut v = Value::Array(t);
        clean_tools(&mut v);
        v
    });

    let max_blocks = if maxtok > 0 { to_blocks(maxtok as u64, canvas).max(1) } else { u32::MAX };
    let requested = ["max_tokens", "max_completion_tokens"]
        .iter()
        .find_map(|k| obj.get(*k).filter(|v| !v.is_null()));
    let n_blocks = match requested {
        None => to_blocks(default_max_tokens as u64, canvas),
        Some(v) => {
            let n = v
                .as_i64()
                .or_else(|| v.as_f64().filter(|f| f.is_finite()).map(|f| f as i64))
                .ok_or_else(|| ApiError::invalid("max_tokens must be an integer"))?;
            if n > 0 {
                to_blocks(n as u64, canvas)
            } else if maxtok > 0 {
                // llama-server reads a non-positive budget as "until the context is full".
                to_blocks(maxtok as u64, canvas)
            } else {
                to_blocks(default_max_tokens as u64, canvas)
            }
        }
    }
    .clamp(1, max_blocks);

    // The runner seeds block b with seed+b and a retry adds one more, so the
    // top of the range stays clear of i32 overflow.
    let seed_max = (i32::MAX as i64 - n_blocks as i64 - 2).max(0);
    let mut fallback = || match profile_seed {
        Some(s) if s >= 0 => (s, true),
        _ => ((rng.next_u64() % (seed_max as u64 + 1)) as i64, false),
    };
    let (seed, seed_pinned) = match obj.get("seed").filter(|v| !v.is_null()) {
        None => fallback(),
        Some(v) => {
            let s = v.as_i64().ok_or_else(|| ApiError::invalid("seed must be an integer"))?;
            if s >= 0 {
                (s, true)
            } else {
                fallback()
            }
        }
    };
    let seed = seed.clamp(0, seed_max) as i32;

    let stream = obj.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = obj
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let stop: Vec<String> = match obj.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) if a.len() <= 4 && a.iter().all(Value::is_string) => {
            a.iter().filter_map(Value::as_str).map(str::to_string).collect()
        }
        Some(_) => return Err(ApiError::invalid("'stop' must be a string or an array of at most 4 strings")),
    }
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect();
    let raw_reasoning = obj.get("reasoning_format").and_then(Value::as_str) == Some("none");

    Ok(ChatPlan {
        req: EngineRequest { seed, n_blocks, messages, tools },
        stream,
        include_usage,
        raw_reasoning,
        stop,
        seed_pinned,
        n_blocks,
    })
}

/// Drop the null-valued keys chat.cpp reads into `std::string` (a null
/// throws `type_error.302` → `ERR parse`): `reasoning_content`, `name` and
/// `tool_call_id` on messages, `id` on tool calls. `content` is NEVER
/// touched: chat.cpp accepts a present `content: null`, and removing it
/// turns a valid tool-call turn into "Expected 'content' or 'tool_calls'".
pub fn clean_messages(msgs: &mut Value) {
    let Some(arr) = msgs.as_array_mut() else { return };
    for m in arr.iter_mut() {
        let Some(mo) = m.as_object_mut() else { continue };
        for k in ["reasoning_content", "name", "tool_call_id"] {
            if mo.get(k).is_some_and(Value::is_null) {
                mo.remove(k);
            }
        }
        if let Some(calls) = mo.get_mut("tool_calls").and_then(Value::as_array_mut) {
            for c in calls.iter_mut().filter_map(Value::as_object_mut) {
                if c.get("id").is_some_and(Value::is_null) {
                    c.remove("id");
                }
            }
        }
    }
}

/// Drop every null key inside `tools[].function`: chat.cpp reads
/// `description` with a string default, and a null there throws.
pub fn clean_tools(tools: &mut Value) {
    let Some(arr) = tools.as_array_mut() else { return };
    for t in arr.iter_mut() {
        if let Some(f) = t.get_mut("function").and_then(Value::as_object_mut) {
            f.retain(|_, v| !v.is_null());
        }
    }
}

// ----------------------------------------------------------- text shaping ----

/// DiffusionGemma's thought markers: the template's native channel format,
/// and the DeepSeek-style tags the model free-runs when thinking is not
/// templated (Unsloth shim.py:88-91).
const THOUGHT_MARKERS: [(&str, &str); 2] = [("<|channel>thought", "<channel|>"), ("<think>", "</think>")];

/// Split committed text into `(reasoning, content)`, dropping the markers.
/// A port of Unsloth's `_split_thought_channels` (shim.py:94-120): earliest
/// start marker wins, an unterminated thought is all reasoning, several
/// thought blocks concatenate, reasoning is trimmed of newlines. Unlike
/// Unsloth, a stray `<channel|>` left in the content is removed too.
pub fn split_thought(text: &str) -> (String, String) {
    let mut reasoning = String::new();
    let mut content = String::new();
    let mut rest = text;
    loop {
        let best = THOUGHT_MARKERS
            .iter()
            .filter_map(|(s, e)| rest.find(s).map(|i| (i, *s, *e)))
            .min_by_key(|(i, _, _)| *i);
        let Some((i, start, end)) = best else {
            content.push_str(rest);
            break;
        };
        content.push_str(&rest[..i]);
        let body = &rest[i + start.len()..];
        match body.find(end) {
            None => {
                reasoning.push_str(body);
                break;
            }
            Some(j) => {
                reasoning.push_str(&body[..j]);
                rest = &body[j + end.len()..];
            }
        }
    }
    (reasoning.trim_matches('\n').to_string(), content.replace(THOUGHT_MARKERS[0].1, ""))
}

/// Bytes at the end of `text` that could be the start of a thought marker
/// or of a stop string: the longest proper prefix of any of them that the
/// text ends with. A block boundary can land mid-marker, and streaming that
/// fragment would leak marker text into a channel.
pub fn holdback_len(text: &str, stop: &[String]) -> usize {
    holdback(text, true, stop)
}

fn holdback(text: &str, markers: bool, stop: &[String]) -> usize {
    let marker_strs = THOUGHT_MARKERS.iter().flat_map(|(s, e)| [*s, *e]).filter(|_| markers);
    let mut longest = 0;
    for m in marker_strs.chain(stop.iter().map(String::as_str)) {
        let upper = m.len().saturating_sub(1).min(text.len());
        for k in (1..=upper).rev() {
            if m.is_char_boundary(k) && text.ends_with(&m[..k]) {
                longest = longest.max(k);
                break;
            }
        }
    }
    longest
}

fn first_stop(text: &str, stop: &[String]) -> Option<usize> {
    stop.iter().filter(|s| !s.is_empty()).filter_map(|s| text.find(s.as_str())).min()
}

/// Apply the stop strings and the thought split to a final text:
/// `(reasoning, content, stop_hit)`.
pub fn shape_final(text: &str, raw_reasoning: bool, stop: &[String]) -> (String, String, bool) {
    let (cut, hit) = match first_stop(text, stop) {
        Some(i) => (&text[..i], true),
        None => (text, false),
    };
    if raw_reasoning {
        (String::new(), cut.to_string(), hit)
    } else {
        let (r, c) = split_thought(cut);
        (r, c, hit)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
}

/// Turns cumulative commits into append-only OpenAI deltas per channel.
/// A channel whose shaped text is not an extension of what was already
/// sent (it can shrink when a marker completes) waits for `finish`.
#[derive(Debug, Clone)]
pub struct StreamShaper {
    raw: bool,
    stop: Vec<String>,
    sent_reasoning: String,
    sent_content: String,
}

impl StreamShaper {
    pub fn new(raw_reasoning: bool, stop: Vec<String>) -> Self {
        StreamShaper { raw: raw_reasoning, stop, sent_reasoning: String::new(), sent_content: String::new() }
    }

    /// Deltas for a new cumulative commit, and whether a stop string was hit
    /// (the caller then finishes the stream with "stop").
    pub fn on_commit(&mut self, cumulative: &str) -> (Vec<Delta>, bool) {
        let (cut, hit) = match first_stop(cumulative, &self.stop) {
            Some(i) => (&cumulative[..i], true),
            None => (cumulative, false),
        };
        let safe = if hit { cut } else { &cut[..cut.len() - holdback(cut, !self.raw, &self.stop)] };
        let (r, c) = if self.raw { (String::new(), safe.to_string()) } else { split_thought(safe) };
        let mut out = Vec::new();
        if let Some(d) = r.strip_prefix(self.sent_reasoning.as_str()).filter(|d| !d.is_empty()) {
            out.push(Delta::Reasoning(d.to_string()));
            self.sent_reasoning = r.clone();
        }
        if let Some(d) = c.strip_prefix(self.sent_content.as_str()).filter(|d| !d.is_empty()) {
            out.push(Delta::Content(d.to_string()));
            self.sent_content = c.clone();
        }
        (out, hit)
    }

    /// A retry restarted the generation from nothing.
    pub fn reset(&mut self) {
        self.sent_reasoning.clear();
        self.sent_content.clear();
    }

    /// The rest of both channels once the job is done. If a channel diverged
    /// from what was streamed, the tail after the common prefix is sent: a
    /// glitch at the seam beats silently dropping the answer.
    pub fn finish(&mut self, final_text: &str) -> Vec<Delta> {
        let (r, c, _) = shape_final(final_text, self.raw, &self.stop);
        let mut out = Vec::new();
        let rest_r = remainder(&self.sent_reasoning, &r);
        if !rest_r.is_empty() {
            out.push(Delta::Reasoning(rest_r.to_string()));
        }
        let rest_c = remainder(&self.sent_content, &c);
        if !rest_c.is_empty() {
            out.push(Delta::Content(rest_c.to_string()));
        }
        self.sent_reasoning = r;
        self.sent_content = c;
        out
    }
}

/// `full` minus its longest common prefix with `sent` (char-aligned).
fn remainder<'a>(sent: &str, full: &'a str) -> &'a str {
    let mut common = 0;
    for ((i, a), b) in full.char_indices().zip(sent.chars()) {
        if a != b {
            break;
        }
        common = i + a.len_utf8();
    }
    &full[common..]
}

// -------------------------------------------------------------- responses ----

#[derive(Debug, Clone, PartialEq)]
pub struct RespMeta {
    pub id: String,
    pub created: u64,
    pub model: String,
}

/// "length" when the budget ran out: TooLong after a commit, or every
/// requested block came back full. Diffusion has no EOS-vs-limit signal
/// beyond that; the runner trims a block at EOS (VS:354).
pub fn finish_reason(
    toolong_after_commit: bool,
    stats: Option<&Stats>,
    canvas: u32,
    n_blocks: u32,
    stop_hit: bool,
) -> &'static str {
    if stop_hit {
        return "stop";
    }
    if toolong_after_commit {
        return "length";
    }
    if let Some(s) = stats {
        if s.blocks == n_blocks && s.blocks > 0 && s.predicted_n == s.blocks as u64 * canvas as u64 {
            return "length";
        }
    }
    "stop"
}

/// llama-server-style `timings` plus the diffusion breakdown.
/// `predicted_per_second` is the honest output rate (G / wall); Unsloth puts
/// the in-step parallel rate there instead, which reads ~10x faster.
pub fn timings(stats: &Stats, seed: i32) -> Value {
    let wall_s = stats.wall_ms / 1000.0;
    let rate = |n: f64| if wall_s > 0.0 { n / wall_s } else { 0.0 };
    let g = stats.predicted_n as f64;
    let canvas = stats.canvas as f64;
    json!({
        "predicted_n": stats.predicted_n,
        "predicted_ms": stats.wall_ms,
        "predicted_per_second": rate(g),
        "prompt_n": stats.prompt_n,
        "prompt_ms": stats.prompt_prepare_ms,
        "cache_n": 0,
        "diffusion": true,
        "diffusion_blocks": stats.blocks,
        "diffusion_steps": stats.steps,
        "diffusion_canvas": stats.canvas,
        "diffusion_wall_ms": stats.wall_ms,
        "diffusion_decode_ms": stats.decode_ms,
        "diffusion_effective_tok_s": rate(canvas * stats.blocks as f64),
        "diffusion_parallel_tok_s": rate(canvas * stats.steps as f64),
        "diffusion_output_tok_s": rate(g),
        "diffusion_steps_per_second": rate(stats.steps as f64),
        "diffusion_seed": seed,
    })
}

pub fn usage(stats: Option<&Stats>) -> Value {
    let (p, g) = stats.map(|s| (s.prompt_n, s.predicted_n)).unwrap_or((0, 0));
    json!({ "prompt_tokens": p, "completion_tokens": g, "total_tokens": p + g })
}

pub fn chat_completion(
    meta: &RespMeta,
    content: &str,
    reasoning: &str,
    finish: &str,
    stats: Option<&Stats>,
    seed: i32,
) -> Value {
    let mut message = json!({ "role": "assistant", "content": content });
    if !reasoning.is_empty() {
        message["reasoning_content"] = reasoning.into();
    }
    let mut v = json!({
        "id": meta.id,
        "object": "chat.completion",
        "created": meta.created,
        "model": meta.model,
        "choices": [{ "index": 0, "finish_reason": finish, "message": message }],
        "usage": usage(stats),
    });
    if let Some(s) = stats {
        v["timings"] = timings(s, seed);
    }
    v
}

pub fn chat_chunk(meta: &RespMeta, delta: Value, finish: Option<&str>, extra: Option<(&str, Value)>) -> Value {
    let mut v = json!({
        "id": meta.id,
        "object": "chat.completion.chunk",
        "created": meta.created,
        "model": meta.model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
    });
    if let Some((k, x)) = extra {
        v[k] = x;
    }
    v
}

/// The `stream_options.include_usage` chunk: empty `choices`.
pub fn usage_chunk(meta: &RespMeta, stats: Option<&Stats>, seed: i32) -> Value {
    let mut v = json!({
        "id": meta.id,
        "object": "chat.completion.chunk",
        "created": meta.created,
        "model": meta.model,
        "choices": [],
        "usage": usage(stats),
    });
    if let Some(s) = stats {
        v["timings"] = timings(s, seed);
    }
    v
}

pub fn delta_json(d: &Delta) -> Value {
    match d {
        Delta::Reasoning(t) => json!({ "reasoning_content": t }),
        Delta::Content(t) => json!({ "content": t }),
    }
}

#[cfg(test)]
mod plan_chat_tests {
    use super::*;

    fn rng() -> XorShift64 {
        XorShift64::new(42)
    }

    fn plan(body: Value) -> Result<ChatPlan, ApiError> {
        plan_chat(&body, 2048, None, 256, 12288, &mut rng())
    }

    fn user(extra: Value) -> Value {
        let mut b = json!({ "messages": [{ "role": "user", "content": "hi" }] });
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b
    }

    #[test]
    fn n_blocks_from_max_tokens() {
        assert_eq!(plan(user(json!({"max_tokens": 1}))).unwrap().n_blocks, 1);
        assert_eq!(plan(user(json!({"max_tokens": 257}))).unwrap().n_blocks, 2);
        assert_eq!(plan(user(json!({}))).unwrap().n_blocks, 8);
        assert_eq!(plan(user(json!({"max_tokens": 0}))).unwrap().n_blocks, 48);
        assert_eq!(plan(user(json!({"max_tokens": -1}))).unwrap().n_blocks, 48);
        assert_eq!(plan(user(json!({"max_tokens": 1_000_000}))).unwrap().n_blocks, 48);
        assert_eq!(plan(user(json!({"max_completion_tokens": 600}))).unwrap().n_blocks, 3);
        // max_tokens wins when both are present.
        assert_eq!(plan(user(json!({"max_tokens": 100, "max_completion_tokens": 600}))).unwrap().n_blocks, 1);
        // Unknown budget (READY without MAXTOK): no upper clamp.
        let p = plan_chat(&user(json!({"max_tokens": 1_000_000})), 2048, None, 256, 0, &mut rng()).unwrap();
        assert_eq!(p.n_blocks, 3907);
        let p = plan_chat(&user(json!({"max_tokens": 0})), 2048, None, 256, 0, &mut rng()).unwrap();
        assert_eq!(p.n_blocks, 8);
        assert!(plan(user(json!({"max_tokens": "lots"}))).is_err());
    }

    #[test]
    fn seeds_pinned_random_and_clamped() {
        let p = plan(user(json!({"seed": 3407}))).unwrap();
        assert_eq!((p.req.seed, p.seed_pinned), (3407, true));
        let p = plan(user(json!({"seed": -1}))).unwrap();
        assert!(!p.seed_pinned);
        let p = plan(user(json!({}))).unwrap();
        assert!(!p.seed_pinned);
        assert!(p.req.seed >= 0 && (p.req.seed as i64) <= i32::MAX as i64 - 8 - 2);
        // Different draws for different requests.
        let mut r = rng();
        let a = plan_chat(&user(json!({})), 2048, None, 256, 12288, &mut r).unwrap().req.seed;
        let b = plan_chat(&user(json!({})), 2048, None, 256, 12288, &mut r).unwrap().req.seed;
        assert_ne!(a, b);
        // The profile seed pins when the request has none.
        let p = plan_chat(&user(json!({})), 2048, Some(11), 256, 12288, &mut rng()).unwrap();
        assert_eq!((p.req.seed, p.seed_pinned), (11, true));
        // A request seed beats the profile seed.
        let p = plan_chat(&user(json!({"seed": 5})), 2048, Some(11), 256, 12288, &mut rng()).unwrap();
        assert_eq!(p.req.seed, 5);
        // Clamped so seed + n_blocks + retry never overflows i32.
        let p = plan(user(json!({"seed": i64::MAX, "max_tokens": 512}))).unwrap();
        assert_eq!(p.req.seed as i64, i32::MAX as i64 - 2 - 2);
        assert!(plan(user(json!({"seed": "x"}))).is_err());
    }

    #[test]
    fn validation_errors() {
        let img = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]}]});
        let e = plan(img).unwrap_err();
        assert_eq!((e.status, e.kind), (400, "invalid_request_error"));
        assert!(e.message.contains("text-only"));
        assert_eq!(plan(user(json!({"n": 2}))).unwrap_err().status, 400);
        assert!(plan(user(json!({"n": 1}))).is_ok());
        assert_eq!(plan(user(json!({"tools": [{"type": "function"}]}))).unwrap_err().status, 400);
        assert_eq!(plan(user(json!({"tools": {"type": "function"}}))).unwrap_err().status, 400);
        assert_eq!(plan(json!({"messages": []})).unwrap_err().status, 400);
        assert_eq!(plan(json!({"messages": [{"role": "robot", "content": "x"}]})).unwrap_err().status, 400);
        assert_eq!(plan(json!({"messages": [{"role": "user", "content": 5}]})).unwrap_err().status, 400);
        assert_eq!(plan(json!([1, 2])).unwrap_err().status, 400);
        assert_eq!(plan(user(json!({"stop": ["a", "b", "c", "d", "e"]}))).unwrap_err().status, 400);
        // Accepted and ignored.
        assert!(plan(user(json!({"temperature": 0.2, "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": false}, "model": "anything"})))
        .is_ok());
    }

    #[test]
    fn tool_choice_rules() {
        let tools = json!([
            {"type": "function", "function": {"name": "a", "description": null, "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "b"}}
        ]);
        let p = plan(user(json!({"tools": tools.clone()}))).unwrap();
        assert_eq!(p.req.tools.as_ref().unwrap().as_array().unwrap().len(), 2);
        // Null keys inside tools[].function are stripped.
        assert!(p.req.tools.as_ref().unwrap()[0]["function"].get("description").is_none());
        assert!(p.req.tools.as_ref().unwrap()[0]["function"].get("parameters").is_some());

        let p = plan(user(json!({"tools": tools.clone(), "tool_choice": "none"}))).unwrap();
        assert!(p.req.tools.is_none());
        for choice in ["auto", "required"] {
            let p = plan(user(json!({"tools": tools.clone(), "tool_choice": choice}))).unwrap();
            assert_eq!(p.req.tools.unwrap().as_array().unwrap().len(), 2);
        }
        let p = plan(user(json!({"tools": tools.clone(),
            "tool_choice": {"type": "function", "function": {"name": "b"}}})))
        .unwrap();
        let kept = p.req.tools.unwrap();
        assert_eq!(kept.as_array().unwrap().len(), 1);
        assert_eq!(kept[0]["function"]["name"], "b");
        let e = plan(user(json!({"tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "zzz"}}})))
        .unwrap_err();
        assert_eq!(e.status, 400);
    }

    #[test]
    fn developer_maps_to_system_and_nulls_are_cleaned() {
        let body = json!({"messages": [
            {"role": "developer", "content": "be brief", "name": null},
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "reasoning_content": null,
             "tool_calls": [{"id": null, "type": "function", "function": {"name": "w", "arguments": "{}"}}]},
            {"role": "tool", "content": "sunny", "tool_call_id": null}
        ]});
        let p = plan(body).unwrap();
        let m = p.req.messages.as_array().unwrap();
        assert_eq!(m[0]["role"], "system");
        assert!(m[0].get("name").is_none());
        // content:null is kept: chat.cpp treats it as present.
        assert!(m[2].get("content").is_some_and(Value::is_null));
        assert!(m[2].get("reasoning_content").is_none());
        assert!(m[2]["tool_calls"][0].get("id").is_none());
        assert!(m[3].get("tool_call_id").is_none());
        assert_eq!(m[3]["content"], "sunny");
    }

    #[test]
    fn stream_options_stop_and_reasoning_format() {
        let p = plan(user(json!({"stream": true, "stream_options": {"include_usage": true},
            "stop": "END", "reasoning_format": "none"})))
        .unwrap();
        assert!(p.stream && p.include_usage && p.raw_reasoning);
        assert_eq!(p.stop, vec!["END".to_string()]);
        let p = plan(user(json!({"stop": ["a", ""]}))).unwrap();
        assert_eq!(p.stop, vec!["a".to_string()]);
        assert!(!p.stream && !p.include_usage && !p.raw_reasoning);
    }
}

#[cfg(test)]
mod shaping_tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn split_thought_dialects() {
        assert_eq!(split_thought("plain answer"), ("".into(), "plain answer".into()));
        assert_eq!(
            split_thought("<|channel>thought\nthink hard\n<channel|>The answer."),
            ("think hard".into(), "The answer.".into())
        );
        assert_eq!(split_thought("pre<think>\nhmm\n</think>post"), ("hmm".into(), "prepost".into()));
        // Unterminated (length-truncated): everything after the marker is reasoning.
        assert_eq!(split_thought("<|channel>thought\nstill going"), ("still going".into(), "".into()));
        // Several blocks, both dialects, earliest first; only the ends of
        // the joined reasoning are trimmed.
        assert_eq!(
            split_thought("<think>a</think>x<|channel>thought\nb<channel|>y"),
            ("a\nb".into(), "xy".into())
        );
        // A stray end marker never reaches the content.
        assert_eq!(split_thought("Hello<channel|> world"), ("".into(), "Hello world".into()));
        // Raw mode keeps the markers.
        let (r, c, hit) = shape_final("<|channel>thought\nx<channel|>y", true, &[]);
        assert_eq!((r.as_str(), c.as_str(), hit), ("", "<|channel>thought\nx<channel|>y", false));
    }

    #[test]
    fn holdback_of_markers_and_stops() {
        assert_eq!(holdback_len("abc<|chan", &[]), "<|chan".len());
        assert_eq!(holdback_len("abc<", &[]), 1);
        assert_eq!(holdback_len("abc</thi", &[]), 5);
        assert_eq!(holdback_len("abc", &[]), 0);
        // A complete marker is not a proper prefix.
        assert_eq!(holdback_len("x<think>", &[]), 0);
        assert_eq!(holdback_len("the EN", &s(&["END"])), 2);
        assert_eq!(holdback_len("the E", &s(&["END", "Eh?"])), 1);
        assert_eq!(holdback_len("caf\u{00e9}", &s(&["\u{00e9}t\u{00e9}"])), "\u{00e9}".len());
    }

    #[test]
    fn stream_shaper_golden_three_blocks_with_split_marker() {
        let mut sh = StreamShaper::new(false, vec![]);
        let (d, hit) = sh.on_commit("<|channel>thought\nThe user wants");
        assert!(!hit);
        assert_eq!(d, vec![Delta::Reasoning("The user wants".into())]);
        // The block boundary lands inside "<channel|>".
        let (d, _) = sh.on_commit("<|channel>thought\nThe user wants a greeting.<chan");
        assert_eq!(d, vec![Delta::Reasoning(" a greeting.".into())]);
        let (d, _) = sh.on_commit("<|channel>thought\nThe user wants a greeting.<channel|>Hello there");
        assert_eq!(d, vec![Delta::Content("Hello there".into())]);
        let d = sh.finish("<|channel>thought\nThe user wants a greeting.<channel|>Hello there");
        assert!(d.is_empty());
    }

    #[test]
    fn stream_shaper_stop_and_finish() {
        let mut sh = StreamShaper::new(false, s(&["END"]));
        assert_eq!(sh.on_commit("abc E").0, vec![Delta::Content("abc ".into())]);
        let (d, hit) = sh.on_commit("abc ENDING");
        assert!(hit);
        assert!(d.is_empty(), "nothing past the stop string: {d:?}");
        // Held-back text is released by finish.
        let mut sh = StreamShaper::new(false, vec![]);
        assert_eq!(sh.on_commit("Hi <").0, vec![Delta::Content("Hi ".into())]);
        assert_eq!(sh.finish("Hi <"), vec![Delta::Content("<".into())]);
        // Raw mode streams markers as content.
        let mut sh = StreamShaper::new(true, vec![]);
        assert_eq!(sh.on_commit("<think>x").0, vec![Delta::Content("<think>x".into())]);
        // reset: a retry starts both channels over.
        let mut sh = StreamShaper::new(false, vec![]);
        sh.on_commit("one");
        sh.reset();
        assert_eq!(sh.on_commit("two").0, vec![Delta::Content("two".into())]);
    }

    #[test]
    fn finish_reasons() {
        let full = Stats { predicted_n: 512, blocks: 2, canvas: 256, ..Default::default() };
        let short = Stats { predicted_n: 300, blocks: 2, canvas: 256, ..Default::default() };
        assert_eq!(finish_reason(false, Some(&short), 256, 2, false), "stop");
        assert_eq!(finish_reason(false, Some(&full), 256, 2, false), "length");
        // Full blocks but fewer than asked for would be a trim, not a budget.
        assert_eq!(finish_reason(false, Some(&full), 256, 4, false), "stop");
        assert_eq!(finish_reason(true, Some(&short), 256, 8, false), "length");
        assert_eq!(finish_reason(true, None, 256, 8, true), "stop");
        assert_eq!(finish_reason(false, None, 256, 8, false), "stop");
    }

    #[test]
    fn timings_use_the_output_rate() {
        let st = Stats {
            prompt_n: 28,
            predicted_n: 347,
            prompt_prepare_ms: 5.8,
            wall_ms: 6904.409,
            decode_ms: 6903.117,
            blocks: 2,
            steps: 32,
            canvas: 256,
            n_ctx: 12288,
        };
        let t = timings(&st, 3407);
        let pps = t["predicted_per_second"].as_f64().unwrap();
        assert!((pps - 347.0 / 6.904409).abs() < 1e-6);
        assert!((t["diffusion_parallel_tok_s"].as_f64().unwrap() - 256.0 * 32.0 / 6.904409).abs() < 1e-6);
        assert!((t["diffusion_effective_tok_s"].as_f64().unwrap() - 512.0 / 6.904409).abs() < 1e-6);
        assert_eq!(t["diffusion_seed"], 3407);
        assert_eq!(t["diffusion"], true);
        assert_eq!(t["cache_n"], 0);
        assert_eq!(usage(Some(&st))["total_tokens"], 375);
        // Zero wall time never divides by zero.
        assert_eq!(timings(&Stats::default(), 0)["predicted_per_second"], 0.0);
    }

    #[test]
    fn exceed_context_shape() {
        let e = exceed_context(12544, 12288, 256);
        assert_eq!((e.status, e.kind), (400, "exceed_context_size_error"));
        assert_eq!(e.extra["n_prompt_tokens"], 12288);
        assert_eq!(e.extra["n_ctx"], 12032);
        assert!(e.message.contains("exceeds the available context size"));
        let b = e.body();
        assert_eq!(b["error"]["code"], 400);
        assert_eq!(b["error"]["type"], "exceed_context_size_error");
        assert_eq!(b["error"]["n_ctx"], 12032);
    }

    #[test]
    fn response_builders() {
        let meta = RespMeta { id: "chatcmpl-x".into(), created: 1, model: "dg".into() };
        let v = chat_completion(&meta, "", "", "stop", None, 1);
        assert_eq!(v["choices"][0]["message"]["content"], "");
        assert!(v["choices"][0]["message"].get("reasoning_content").is_none());
        assert!(v.get("timings").is_none());
        let c = chat_chunk(&meta, json!({"content": "x"}), None, None);
        assert!(c["choices"][0]["finish_reason"].is_null());
        let c = chat_chunk(&meta, json!({}), Some("length"), Some(("timings", json!({"a": 1}))));
        assert_eq!(c["choices"][0]["finish_reason"], "length");
        assert_eq!(c["timings"]["a"], 1);
        assert_eq!(usage_chunk(&meta, None, 0)["choices"], json!([]));
        assert_eq!(exit_hex(-1073740791), "0xC0000409");
        let e = failure_error(&EngineFailure::Crashed { exit: Some(-1073740791), detail: "GGML_ASSERT(x) failed".into() });
        assert_eq!(e.message, "diffusion engine crashed (exit 0xC0000409): GGML_ASSERT(x) failed");
        assert_eq!(failure_error(&EngineFailure::Oom).kind, "exceed_context_size_error");
        assert_eq!(failure_error(&EngineFailure::Watchdog(180)).message, "diffusion engine produced no output for 180 s; restarted");
    }
}
