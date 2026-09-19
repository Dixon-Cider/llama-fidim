//! Hugging Face Hub client for the model wizard: search, repo info with
//! per-file sizes and SHA-256s, file trees, READMEs, access checks, and GGUF
//! headers read with HTTP Range requests before anything is downloaded.
//!
//! Everything is synchronous, like the rest of the core; the UI runs it off
//! its main thread. Callers pin a commit: `model_info` returns the repo's
//! `sha` and the later calls take it, so a push to the repo mid-wizard
//! cannot mix two revisions' files.
//!
//! A token (see `token`) is sent as a bearer to the Hub only. The agent
//! keeps the Authorization header across same-host redirects (the Hub's 307
//! to `/api/resolve-cache/`) and drops it on a redirect to any other host,
//! so it never reaches the CDN a `/resolve/` URL sends the client to.
//!
//! The parsers are pure and tested against responses captured from the live
//! API in `fixtures/hub/` (chat templates trimmed).

use std::ffi::OsString;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::config::Config;
use crate::error::HttpErrorKind;
use crate::gguf::{self, GgufHeader, ReadMode};
use crate::{Error, Result};

pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
const USER_AGENT: &str = concat!("llama-fidim/", env!("CARGO_PKG_VERSION"));
const API_READ_TIMEOUT: Duration = Duration::from_secs(60);
/// First Range read for each mode, and the most a header read fetches.
/// Measured: the estimator's keys end within 2 KB; a whole KV table plus
/// tensor table is 10-16 MB, most of it tokenizer arrays.
const UNTIL_TOKENIZER_FIRST: u64 = 64 * 1024;
const FULL_FIRST: u64 = 1024 * 1024;
const HEADER_CAP: u64 = 32 * 1024 * 1024;
/// Response bodies read into memory are capped: a search with the `gguf`
/// expansion carries every hit's chat template (~40 KB each).
const BODY_CAP: u64 = 32 * 1024 * 1024;
/// A model card longer than this is cut (they run to tens of KB).
const README_CAP: u64 = 2 * 1024 * 1024;
/// Tree pages followed before giving up.
const MAX_PAGES: usize = 100;
const SEARCH_EXPAND: &[&str] =
    &["author", "downloads", "likes", "lastModified", "gated", "gguf", "tags", "pipeline_tag", "library_name"];
const INFO_EXPAND: &[&str] = &[
    "author", "sha", "lastModified", "private", "disabled", "gated", "downloads", "likes", "tags", "pipeline_tag",
    "library_name", "cardData", "gguf", "siblings", "baseModels",
];
/// Relations a `base_model:<relation>:<repo>` tag can name.
const RELATIONS: &[&str] = &["quantized", "finetune", "adapter", "merge"];

/// The Hub's base URL: `HF_ENDPOINT` when set (a mirror, as huggingface_hub
/// honours it), else huggingface.co.
pub fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_ENDPOINT.to_string())
}

// ------------------------------------------------------------------ token ----

/// The Hugging Face token to send, first found of: `HF_TOKEN`; the file
/// `HF_TOKEN_PATH` names; `config.hf_token`; and only when
/// `config.hf_use_cli_token` is on, the file `huggingface-cli login` writes
/// (`%HF_HOME%\token`, else `%USERPROFILE%\.cache\huggingface\token`).
/// A value that could not be an HTTP header (spaces, control or non-ASCII
/// characters) counts as unset.
pub fn token(cfg: &Config) -> Option<String> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from);
    token_with(cfg, &|k| std::env::var_os(k), home.as_deref())
}

fn token_with(cfg: &Config, env: &dyn Fn(&str) -> Option<OsString>, home: Option<&Path>) -> Option<String> {
    fn clean(s: &str) -> Option<String> {
        let t = s.trim();
        (!t.is_empty() && t.bytes().all(|b| b.is_ascii_graphic())).then(|| t.to_string())
    }
    let from_file = |p: PathBuf| std::fs::read_to_string(p).ok().and_then(|s| clean(&s));
    let var = |k: &str| env(k).filter(|v| !v.is_empty());
    if let Some(t) = var("HF_TOKEN").and_then(|v| v.into_string().ok()).and_then(|s| clean(&s)) {
        return Some(t);
    }
    if let Some(t) = var("HF_TOKEN_PATH").and_then(|p| from_file(PathBuf::from(p))) {
        return Some(t);
    }
    if let Some(t) = cfg.hf_token.as_deref().and_then(clean) {
        return Some(t);
    }
    if cfg.hf_use_cli_token {
        let path = match var("HF_HOME") {
            Some(h) => PathBuf::from(h).join("token"),
            None => home?.join(".cache").join("huggingface").join("token"),
        };
        return from_file(path);
    }
    None
}

// ------------------------------------------------------------------ types ----

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sort {
    #[default]
    Downloads,
    Likes,
    Trending,
    LastModified,
    Created,
}

impl Sort {
    fn param(self) -> &'static str {
        match self {
            Sort::Downloads => "downloads",
            Sort::Likes => "likes",
            Sort::Trending => "trendingScore",
            Sort::LastModified => "lastModified",
            Sort::Created => "createdAt",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    /// Substring of the repo id; words are matched in order.
    pub text: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub sort: Sort,
    /// 1-100 (larger values are clamped).
    #[serde(default = "default_limit")]
    pub limit: u32,
    /// Only repos with GGUF files (the Hub's automatic `gguf` tag).
    #[serde(default = "default_true")]
    pub gguf_only: bool,
}

fn default_limit() -> u32 {
    30
}
fn default_true() -> bool {
    true
}

impl Default for SearchQuery {
    fn default() -> Self {
        SearchQuery { text: String::new(), author: None, sort: Sort::default(), limit: default_limit(), gguf_only: true }
    }
}

/// Whether downloading needs the repo's terms accepted on the website.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Gated {
    #[default]
    No,
    /// Accepting the terms grants access at once.
    Auto,
    /// The owner approves each request by hand.
    Manual,
}

/// One search result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoHit {
    pub id: String,
    pub author: Option<String>,
    pub downloads: u64,
    pub likes: u64,
    pub last_modified: Option<String>,
    pub gated: Gated,
    /// From the Hub's `gguf` summary of one representative file.
    pub arch: Option<String>,
    pub context_length: Option<u64>,
    pub total_params: Option<u64>,
    pub tags: Vec<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
}

/// The Hub's `gguf` summary. It describes one representative file of the
/// repo (the unquantized one when there is one), never the repo: take file
/// sizes from `RepoInfo::siblings`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GgufSummary {
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    /// Parameter count.
    pub total: Option<u64>,
}

/// A file in a repo at a pinned commit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoFile {
    /// Path in the repo, `/`-separated (e.g. `BF16/x-00001-of-00002.gguf`).
    pub path: String,
    pub size: u64,
    /// Lowercase hex SHA-256 of an LFS/Xet file; None for small git files.
    pub sha256: Option<String>,
}

impl RepoFile {
    /// The file name, without the repo folders.
    pub fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoInfo {
    pub id: String,
    /// The commit the rest of this describes; pin every later call to it.
    pub sha: String,
    pub gated: Gated,
    /// `cardData.extra_gated_prompt`: what accepting the terms means.
    pub gated_prompt: Option<String>,
    pub card_license: Option<String>,
    /// (relation, repo), e.g. ("quantized", "IFM/K2-Horizon-7B").
    pub base_models: Vec<(String, String)>,
    pub tags: Vec<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub last_modified: Option<String>,
    pub gguf: Option<GgufSummary>,
    pub siblings: Vec<RepoFile>,
}

/// A repo named by the user: `owner/repo`, or a huggingface.co / hf.co URL,
/// possibly of a revision (`/tree/<rev>`) or a file (`/blob/<rev>/<path>`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoRef {
    pub repo: String,
    pub rev: Option<String>,
    pub path: Option<String>,
}

// ---------------------------------------------------------------- parsers ----

fn malformed_response(what: &str, detail: impl std::fmt::Display) -> Error {
    Error::Http {
        kind: HttpErrorKind::Malformed,
        message: format!("{what}: unexpected response from Hugging Face ({detail})"),
    }
}

fn opt_str(v: &Json) -> Option<String> {
    v.as_str().map(str::trim).filter(|s| !s.is_empty()).map(String::from)
}

fn str_list(v: &Json) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn gated_of(v: &Json) -> Gated {
    match v {
        Json::String(s) if s.eq_ignore_ascii_case("manual") => Gated::Manual,
        Json::String(s) if s.eq_ignore_ascii_case("auto") => Gated::Auto,
        // Not a value the Hub sends today, but gated all the same.
        Json::Bool(true) => Gated::Auto,
        _ => Gated::No,
    }
}

fn gguf_of(v: &Json) -> Option<GgufSummary> {
    let g = v.as_object()?;
    Some(GgufSummary {
        architecture: g.get("architecture").and_then(opt_str),
        context_length: g.get("context_length").and_then(Json::as_u64),
        total: g.get("total").and_then(Json::as_u64),
    })
}

fn hit_of(v: &Json) -> Option<RepoHit> {
    let id = opt_str(&v["id"])?;
    let gguf = gguf_of(&v["gguf"]);
    Some(RepoHit {
        author: opt_str(&v["author"]).or_else(|| id.split_once('/').map(|(o, _)| o.to_string())),
        downloads: v["downloads"].as_u64().unwrap_or(0),
        likes: v["likes"].as_u64().unwrap_or(0),
        last_modified: opt_str(&v["lastModified"]),
        gated: gated_of(&v["gated"]),
        arch: gguf.as_ref().and_then(|g| g.architecture.clone()),
        context_length: gguf.as_ref().and_then(|g| g.context_length),
        total_params: gguf.as_ref().and_then(|g| g.total),
        tags: str_list(&v["tags"]),
        pipeline_tag: opt_str(&v["pipeline_tag"]),
        library_name: opt_str(&v["library_name"]),
        id,
    })
}

/// `GET /api/models?...` results.
pub fn parse_search(json: &str) -> Result<Vec<RepoHit>> {
    let v: Json = serde_json::from_str(json).map_err(|e| malformed_response("search", e))?;
    let arr = v.as_array().ok_or_else(|| malformed_response("search", "not a list"))?;
    Ok(arr.iter().filter_map(hit_of).collect())
}

/// (relation, repo) pairs, from the `baseModels` expansion when present,
/// else the card's `base_model` (+ `base_model_relation`), else the
/// `base_model:[<relation>:]<repo>` tags.
fn base_models_of(v: &Json, tags: &[String]) -> Vec<(String, String)> {
    // One entry per repo; a relation, once known, is kept.
    fn push(rel: &str, repo: &str, out: &mut Vec<(String, String)>) {
        let repo = repo.trim();
        if repo.is_empty() {
            return;
        }
        if let Some(existing) = out.iter_mut().find(|(_, r)| r == repo) {
            if existing.0.is_empty() {
                existing.0 = rel.to_string();
            }
        } else {
            out.push((rel.to_string(), repo.to_string()));
        }
    }
    let mut out: Vec<(String, String)> = Vec::new();
    let bm = &v["baseModels"];
    let groups: Vec<&Json> = match bm {
        Json::Array(a) => a.iter().collect(),
        Json::Object(_) => vec![bm],
        _ => vec![],
    };
    for g in groups {
        let rel = g["relation"].as_str().unwrap_or("");
        for m in g["models"].as_array().into_iter().flatten() {
            if let Some(id) = m["id"].as_str() {
                push(rel, id, &mut out);
            }
        }
    }
    if out.is_empty() {
        let card = &v["cardData"];
        let rel = card["base_model_relation"].as_str().unwrap_or("");
        match &card["base_model"] {
            Json::String(s) => push(rel, s, &mut out),
            Json::Array(a) => a.iter().filter_map(Json::as_str).for_each(|s| push(rel, s, &mut out)),
            _ => {}
        }
    }
    if out.is_empty() {
        for t in tags {
            let Some(rest) = t.strip_prefix("base_model:") else { continue };
            match rest.split_once(':') {
                Some((rel, repo)) if RELATIONS.contains(&rel) => push(rel, repo, &mut out),
                _ => push("", rest, &mut out),
            }
        }
    }
    out
}

fn siblings_of(v: &Json) -> Vec<RepoFile> {
    v.as_array()
        .into_iter()
        .flatten()
        .filter_map(|s| {
            Some(RepoFile {
                path: opt_str(&s["rfilename"])?,
                size: s["size"].as_u64().or_else(|| s["lfs"]["size"].as_u64()).unwrap_or(0),
                sha256: opt_str(&s["lfs"]["sha256"]).map(|h| h.to_ascii_lowercase()),
            })
        })
        .collect()
}

/// `GET /api/models/<repo>?blobs=true&expand[]=...` (see `model_info`).
pub fn parse_model_info(json: &str) -> Result<RepoInfo> {
    let v: Json = serde_json::from_str(json).map_err(|e| malformed_response("model info", e))?;
    let id = opt_str(&v["id"]).ok_or_else(|| malformed_response("model info", "no id"))?;
    let sha = opt_str(&v["sha"]).ok_or_else(|| malformed_response(&id, "no commit sha"))?;
    let tags = str_list(&v["tags"]);
    let card = &v["cardData"];
    let card_license = opt_str(&card["license"])
        .or_else(|| tags.iter().find_map(|t| t.strip_prefix("license:").map(String::from)));
    Ok(RepoInfo {
        sha,
        gated: gated_of(&v["gated"]),
        gated_prompt: opt_str(&card["extra_gated_prompt"]),
        card_license,
        base_models: base_models_of(&v, &tags),
        pipeline_tag: opt_str(&v["pipeline_tag"]),
        library_name: opt_str(&v["library_name"]),
        last_modified: opt_str(&v["lastModified"]),
        gguf: gguf_of(&v["gguf"]),
        siblings: siblings_of(&v["siblings"]),
        tags,
        id,
    })
}

/// `GET /api/models/<repo>/tree/<rev>?recursive=true`: the files (not the
/// directories), with the LFS SHA-256 (`lfs.oid` here).
pub fn parse_tree(json: &str) -> Result<Vec<RepoFile>> {
    let v: Json = serde_json::from_str(json).map_err(|e| malformed_response("file list", e))?;
    let arr = v.as_array().ok_or_else(|| malformed_response("file list", "not a list"))?;
    Ok(arr
        .iter()
        .filter(|e| e["type"].as_str() == Some("file"))
        .filter_map(|e| {
            Some(RepoFile {
                path: opt_str(&e["path"])?,
                size: e["size"].as_u64().unwrap_or(0),
                sha256: opt_str(&e["lfs"]["oid"]).map(|h| h.to_ascii_lowercase()),
            })
        })
        .collect())
}

/// The `rel="next"` URL of a `Link` header.
fn next_link(link: &str) -> Option<String> {
    link.split(',').find_map(|part| {
        let (url, params) = part.trim().split_once(';')?;
        let is_next = params.split(';').any(|p| p.trim().replace(' ', "") == "rel=\"next\"");
        is_next.then(|| url.trim().trim_start_matches('<').trim_end_matches('>').to_string())
    })
}

// ------------------------------------------------------------- repo names ----

fn valid_repo_part(p: &str) -> bool {
    !p.is_empty()
        && p.len() <= 96
        && p.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b))
        && !p.starts_with(['-', '.'])
        && !p.ends_with(['-', '.'])
        && !p.contains("--")
        && !p.contains("..")
}

/// `owner/name` as the Hub allows it (ASCII letters, digits, `-_.`; no
/// leading, trailing or doubled `-`/`.`; at most 96 characters each). Every
/// function that puts a repo into a URL or a path checks this first.
pub fn validate_repo(repo: &str) -> Result<()> {
    match repo.split_once('/') {
        Some((o, n)) if valid_repo_part(o) && valid_repo_part(n) => Ok(()),
        _ => Err(Error::InvalidInput(format!("{repo:?} is not a Hugging Face repo id (owner/name)"))),
    }
}

/// Parse what a user pasted: `owner/name`, or a Hub URL of the repo, a
/// revision or a file. None for anything else, including dataset and Space
/// URLs.
pub fn parse_repo_input(input: &str) -> Option<RepoRef> {
    let mut s = input.trim();
    for scheme in ["https://", "http://"] {
        if let Some(rest) = s.strip_prefix(scheme) {
            s = rest;
        }
    }
    let had_host = ["www.huggingface.co/", "huggingface.co/", "hf.co/"].iter().any(|h| {
        if let Some(rest) = s.strip_prefix(h) {
            s = rest;
            true
        } else {
            false
        }
    });
    if !had_host && (input.contains("://") || s.contains(':')) {
        return None;
    }
    let s = s.split(['?', '#']).next().unwrap_or("");
    let parts: Vec<&str> = s.trim_matches('/').split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 || ["datasets", "spaces", "api", "models"].contains(&parts[0]) {
        return None;
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    validate_repo(&repo).ok()?;
    let (mut rev, mut path) = (None, None);
    if parts.len() >= 4 && ["tree", "blob", "resolve"].contains(&parts[2]) {
        rev = Some(parts[3].to_string());
        if parts.len() > 4 {
            path = Some(parts[4..].join("/"));
        }
    }
    Some(RepoRef { repo, rev, path })
}

/// `owner/name` of a Hub `/resolve/` URL, for error messages.
pub fn repo_of_resolve_url(url: &str) -> Option<String> {
    let before = url.split("/resolve/").next().filter(|b| b.len() < url.len())?;
    let mut segs = before.rsplit('/');
    let (name, owner) = (segs.next()?, segs.next()?);
    let repo = format!("{owner}/{name}");
    validate_repo(&repo).ok().map(|_| repo)
}

/// A file path inside a repo: `/`-separated, no empty, `.` or `..`
/// segments, no backslashes or control characters.
pub fn validate_repo_path(path: &str) -> Result<()> {
    let ok = !path.is_empty()
        && !path.contains('\\')
        && !path.chars().any(char::is_control)
        && path.split('/').all(|seg| !seg.is_empty() && seg != "." && seg != "..");
    if ok {
        Ok(())
    } else {
        Err(Error::InvalidInput(format!("{path:?} is not a file path in a repo")))
    }
}

/// Percent-encode one URL path segment (everything but RFC 3986 unreserved).
fn enc_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn enc_path(p: &str) -> String {
    p.split('/').map(enc_segment).collect::<Vec<_>>().join("/")
}

/// The download URL of `path` at commit `sha`. Pinned to the commit, not to
/// a branch, so every file of a download comes from the same revision.
pub fn resolve_url(repo: &str, sha: &str, path: &str) -> String {
    resolve_url_at(&endpoint(), repo, sha, path)
}

fn resolve_url_at(base: &str, repo: &str, sha: &str, path: &str) -> String {
    format!("{base}/{repo}/resolve/{}/{}", enc_segment(sha), enc_path(path.trim_start_matches('/')))
}

// ---------------------------------------------------------------- errors ----

/// The error for a failed HTTP status, by the Hub's `X-Error-Code` header
/// as much as by the status (401 means both "gated" and "no such repo").
/// `what` names the request for the message; `repo` points the gated
/// message at the page where the terms are accepted.
pub(crate) fn status_error(
    status: u16,
    header: &dyn Fn(&str) -> Option<String>,
    body: &str,
    what: &str,
    repo: Option<&str>,
    had_token: bool,
) -> Error {
    let code = header("x-error-code").unwrap_or_default();
    let server_says = header("x-error-message")
        .or_else(|| {
            serde_json::from_str::<Json>(body).ok().and_then(|v| v["error"].as_str().map(String::from))
        })
        .unwrap_or_else(|| body.trim().chars().take(200).collect());
    let page = repo.map(|r| format!("{DEFAULT_ENDPOINT}/{r}"));
    let (kind, message) = match (status, code.as_str()) {
        (401 | 403, "GatedRepo") => {
            let name = repo.unwrap_or("this repo");
            let where_ = page.map(|p| format!(" at {p}")).unwrap_or_default();
            let message = if had_token {
                format!(
                    "{name} is gated and this Hugging Face token has no access yet: accept its terms{where_} \
                     (repos the owner approves by hand can take a while)"
                )
            } else {
                format!(
                    "{name} is gated: accept its terms{where_}, then set a Hugging Face token (HF_TOKEN, or the \
                     token in Settings)"
                )
            };
            (HttpErrorKind::Gated, message)
        }
        (401, _) | (404, "RepoNotFound") => {
            let name = repo.map(String::from).unwrap_or_else(|| what.to_string());
            let message = if had_token {
                format!("{name}: not found on Hugging Face, or private to another account, or the token was rejected")
            } else {
                format!("{name}: not found on Hugging Face (or private: set a token if it is yours)")
            };
            (HttpErrorKind::RepoNotFound, message)
        }
        (404, "RevisionNotFound") => (HttpErrorKind::RevisionNotFound, format!("{what}: revision not found")),
        (404, "EntryNotFound") => (HttpErrorKind::EntryNotFound, format!("{what}: no such file in the repo")),
        (429, _) => {
            let retry = header("retry-after").and_then(|s| s.trim().parse::<u64>().ok()).or_else(|| {
                // `RateLimit: "api";r=0;t=74` — t is the seconds to the reset.
                header("ratelimit").and_then(|s| {
                    s.split(';').find_map(|p| p.trim().strip_prefix("t=").and_then(|t| t.trim().parse().ok()))
                })
            });
            let when = retry.map(|t| format!(" in {t} s")).unwrap_or_else(|| " in a few minutes".into());
            let hint = if had_token { "" } else { " (a token raises the limit)" };
            (
                HttpErrorKind::RateLimited { retry_after_secs: retry },
                format!("{what}: Hugging Face rate limit reached; try again{when}{hint}"),
            )
        }
        _ => (HttpErrorKind::Status { status }, format!("{what}: HTTP {status}: {server_says}")),
    };
    Error::Http { kind, message }
}

/// `status_error` over a ureq error response (reads its headers and body).
pub(crate) fn response_error(
    status: u16,
    resp: ureq::Response,
    what: &str,
    repo: Option<&str>,
    had_token: bool,
) -> Error {
    let headers: Vec<(String, String)> = resp
        .headers_names()
        .into_iter()
        .filter_map(|n| resp.header(&n).map(|v| (n.to_ascii_lowercase(), v.to_string())))
        .collect();
    let mut body = String::new();
    let _ = resp.into_reader().take(64 * 1024).read_to_string(&mut body);
    let get = |name: &str| headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone());
    status_error(status, &get, &body, what, repo, had_token)
}

pub(crate) fn network_error(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Http { kind: HttpErrorKind::Network, message: format!("{what}: {e}") }
}

// ---------------------------------------------------------------- network ----

/// The HTTP agent for the Hub and its CDN: Authorization survives a
/// same-host redirect only.
pub(crate) fn agent(read_timeout: Duration) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(read_timeout)
        .redirect_auth_headers(ureq::RedirectAuthHeaders::SameHost)
        .user_agent(USER_AGENT)
        .build()
}

fn get(agent: &ureq::Agent, url: &str, token: Option<&str>) -> ureq::Request {
    let req = agent.get(url);
    match token {
        Some(t) => req.set("Authorization", &format!("Bearer {t}")),
        None => req,
    }
}

fn call(req: ureq::Request, what: &str, repo: Option<&str>, had_token: bool) -> Result<ureq::Response> {
    match req.call() {
        Ok(r) => Ok(r),
        Err(ureq::Error::Status(status, resp)) => Err(response_error(status, resp, what, repo, had_token)),
        Err(ureq::Error::Transport(t)) => Err(network_error(what, t)),
    }
}

/// The body as text, refusing more than `cap` bytes.
fn read_body(resp: ureq::Response, cap: u64, what: &str) -> Result<String> {
    let mut buf = Vec::new();
    resp.into_reader().take(cap + 1).read_to_end(&mut buf).map_err(|e| network_error(what, e))?;
    if buf.len() as u64 > cap {
        return Err(malformed_response(what, format!("response larger than {} MiB", cap >> 20)));
    }
    String::from_utf8(buf).map_err(|e| malformed_response(what, e))
}

/// Search the Hub. `gguf_only` filters on the Hub's automatic `gguf` tag
/// (the reliable filter: `library_name` is set inconsistently on GGUF repos).
pub fn search(cfg: &Config, q: &SearchQuery) -> Result<Vec<RepoHit>> {
    search_at(&endpoint(), token(cfg).as_deref(), q, &[])
}

/// GGUF quantizations of a (usually safetensors) repo: repos whose card
/// declares it as their base with relation `quantized`. Repos that do not
/// declare a base model are not found this way.
pub fn derivatives(cfg: &Config, base_repo: &str) -> Result<Vec<RepoHit>> {
    validate_repo(base_repo)?;
    let q = SearchQuery { limit: 100, ..SearchQuery::default() };
    search_at(&endpoint(), token(cfg).as_deref(), &q, &[format!("base_model:quantized:{base_repo}")])
}

fn search_at(base: &str, token: Option<&str>, q: &SearchQuery, filters: &[String]) -> Result<Vec<RepoHit>> {
    let agent = agent(API_READ_TIMEOUT);
    let mut req = get(&agent, &format!("{base}/api/models"), token);
    if !q.text.trim().is_empty() {
        req = req.query("search", q.text.trim());
    }
    if let Some(a) = q.author.as_deref().map(str::trim).filter(|a| !a.is_empty()) {
        req = req.query("author", a);
    }
    for f in filters {
        req = req.query("filter", f);
    }
    if q.gguf_only {
        req = req.query("filter", "gguf");
    }
    req = req
        .query("sort", q.sort.param())
        .query("direction", "-1")
        .query("limit", &q.limit.clamp(1, 100).to_string());
    for e in SEARCH_EXPAND {
        req = req.query("expand[]", e);
    }
    let what = "Hugging Face search";
    let text = read_body(call(req, what, None, token.is_some())?, BODY_CAP, what)?;
    parse_search(&text)
}

/// A repo's metadata and files (with sizes and SHA-256s) at `rev` (a
/// branch, tag or commit; None = the default branch). Works without a token
/// on gated repos: only downloads need one.
pub fn model_info(cfg: &Config, repo: &str, rev: Option<&str>) -> Result<RepoInfo> {
    model_info_at(&endpoint(), token(cfg).as_deref(), repo, rev)
}

fn model_info_at(base: &str, token: Option<&str>, repo: &str, rev: Option<&str>) -> Result<RepoInfo> {
    validate_repo(repo)?;
    let url = match rev.map(str::trim).filter(|r| !r.is_empty()) {
        Some(r) => format!("{base}/api/models/{repo}/revision/{}", enc_segment(r)),
        None => format!("{base}/api/models/{repo}"),
    };
    let agent = agent(API_READ_TIMEOUT);
    let mut req = get(&agent, &url, token).query("blobs", "true");
    for e in INFO_EXPAND {
        req = req.query("expand[]", e);
    }
    let text = read_body(call(req, repo, Some(repo), token.is_some())?, BODY_CAP, repo)?;
    parse_model_info(&text)
}

/// Every file of a repo at `rev`, following the tree's pages. `model_info`
/// already lists the files; this is the per-directory view, for repos too
/// large for one response.
pub fn list_files(cfg: &Config, repo: &str, rev: &str) -> Result<Vec<RepoFile>> {
    list_files_at(&endpoint(), token(cfg).as_deref(), repo, rev)
}

fn list_files_at(base: &str, token: Option<&str>, repo: &str, rev: &str) -> Result<Vec<RepoFile>> {
    validate_repo(repo)?;
    let agent = agent(API_READ_TIMEOUT);
    let mut url = format!("{base}/api/models/{repo}/tree/{}?recursive=true", enc_segment(rev));
    let mut files = Vec::new();
    for _ in 0..MAX_PAGES {
        let resp = call(get(&agent, &url, token), repo, Some(repo), token.is_some())?;
        let next = resp.header("link").and_then(next_link);
        files.extend(parse_tree(&read_body(resp, BODY_CAP, repo)?)?);
        match next {
            // Never follow a page link (and send the token) off the Hub.
            Some(n) if n.starts_with(&format!("{base}/")) => url = n,
            _ => return Ok(files),
        }
    }
    Err(malformed_response(repo, format!("file list longer than {MAX_PAGES} pages")))
}

/// The repo's model card at a commit; None when it has none. Untrusted
/// text: callers extract links from it and act on nothing else.
pub fn readme(cfg: &Config, repo: &str, sha: &str) -> Result<Option<String>> {
    readme_at(&endpoint(), token(cfg).as_deref(), repo, sha)
}

fn readme_at(base: &str, token: Option<&str>, repo: &str, sha: &str) -> Result<Option<String>> {
    validate_repo(repo)?;
    let url = format!("{base}/{repo}/raw/{}/README.md", enc_segment(sha));
    let agent = agent(API_READ_TIMEOUT);
    let what = format!("{repo} README");
    let resp = match call(get(&agent, &url, token), &what, Some(repo), token.is_some()) {
        Ok(r) => r,
        Err(Error::Http { kind: HttpErrorKind::EntryNotFound, .. }) => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut buf = Vec::new();
    resp.into_reader().take(README_CAP).read_to_end(&mut buf).map_err(|e| network_error(&what, e))?;
    Ok(Some(String::from_utf8_lossy(&buf).into_owned()))
}

/// Ok when this client may download from `repo`: the cheap check before a
/// big transfer. A gated repo without access fails with
/// `HttpErrorKind::Gated`, a missing or private one with `RepoNotFound`.
pub fn auth_check(cfg: &Config, repo: &str) -> Result<()> {
    auth_check_at(&endpoint(), token(cfg).as_deref(), repo)
}

fn auth_check_at(base: &str, token: Option<&str>, repo: &str) -> Result<()> {
    validate_repo(repo)?;
    let agent = agent(API_READ_TIMEOUT);
    call(get(&agent, &format!("{base}/api/models/{repo}/auth-check"), token), repo, Some(repo), token.is_some())
        .map(|_| ())
}

/// The GGUF header of a remote file, from HTTP Range reads of its start:
/// 64 KiB for `UntilTokenizer` (enough for the architecture keys and the
/// pre-tokenizer name), 1 MiB for `Full`, doubling on `GgufTruncated` up to
/// 32 MiB. `file_size` is the file's size from the repo listing; the header
/// reports it as `file_size` and names the file `hf://<repo>@<sha>/<path>`.
pub fn remote_header(
    cfg: &Config,
    repo: &str,
    sha: &str,
    path: &str,
    file_size: u64,
    mode: ReadMode,
) -> Result<GgufHeader> {
    remote_header_at(&endpoint(), token(cfg).as_deref(), repo, sha, path, file_size, mode)
}

fn remote_header_at(
    base: &str,
    token: Option<&str>,
    repo: &str,
    sha: &str,
    path: &str,
    file_size: u64,
    mode: ReadMode,
) -> Result<GgufHeader> {
    validate_repo(repo)?;
    validate_repo_path(path)?;
    let url = resolve_url_at(base, repo, sha, path);
    let short: String = sha.chars().take(12).collect();
    let label = PathBuf::from(format!("hf://{repo}@{short}/{path}"));
    let cap = if file_size > 0 { HEADER_CAP.min(file_size) } else { HEADER_CAP };
    let mut want = match mode {
        ReadMode::UntilTokenizer => UNTIL_TOKENIZER_FIRST,
        ReadMode::Full => FULL_FIRST,
    }
    .min(cap);
    let agent = agent(API_READ_TIMEOUT);
    let what = format!("{repo}/{path}");
    let mut buf: Vec<u8> = Vec::new();
    loop {
        if (buf.len() as u64) < want {
            let before = buf.len();
            fetch_range(&agent, &url, token, before as u64, want, &mut buf, &what, repo)?;
            if buf.len() == before {
                return Err(malformed_response(&what, format!("no bytes at offset {before}")));
            }
        }
        match gguf::read_header_from(&mut Cursor::new(&buf), file_size, &label, mode) {
            Err(Error::GgufTruncated { at, .. }) if want < cap && (buf.len() as u64) >= want => {
                want = want.saturating_mul(2).max(at.saturating_add(64 * 1024)).min(cap);
            }
            other => return other,
        }
    }
}

/// Append bytes `[start, end)` of `url` to `buf`. A server that ignores
/// Range (200 instead of 206) is read from the start and trimmed.
#[allow(clippy::too_many_arguments)]
fn fetch_range(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    start: u64,
    end: u64,
    buf: &mut Vec<u8>,
    what: &str,
    repo: &str,
) -> Result<()> {
    let req = get(agent, url, token).set("Range", &format!("bytes={start}-{}", end - 1));
    let resp = call(req, what, Some(repo), token.is_some())?;
    // A 206's body starts where its Content-Range says; a 200's at 0.
    let body_start = match resp.status() {
        206 => resp.header("content-range").and_then(content_range_start).unwrap_or(start),
        _ => 0,
    };
    if body_start > start {
        return Err(malformed_response(what, format!("asked for bytes from {start}, got bytes from {body_start}")));
    }
    let mut reader = resp.into_reader();
    // Discard what precedes `start`: nothing for a 206, the prefix already
    // held for a server that ignored the Range header.
    std::io::copy(&mut (&mut reader).take(start - body_start), &mut std::io::sink())
        .map_err(|e| network_error(what, e))?;
    reader.take(end - start).read_to_end(buf).map_err(|e| network_error(what, e))?;
    Ok(())
}

/// First byte of a `Content-Range: bytes <first>-<last>/<size>` header.
pub(crate) fn content_range_start(v: &str) -> Option<u64> {
    content_range(v).map(|(first, _)| first)
}

/// (first byte, total size) of a `Content-Range` header; the size is None
/// when the server writes `*`.
pub(crate) fn content_range(v: &str) -> Option<(u64, Option<u64>)> {
    let rest = v.trim().strip_prefix("bytes")?.trim();
    let (range, total) = rest.split_once('/')?;
    let first = range.split_once('-')?.0.trim().parse().ok()?;
    Some((first, total.trim().parse().ok()))
}

/// The total size in a `Content-Range` header, including the
/// `bytes */<size>` form a 416 answer carries.
pub(crate) fn content_range_total(v: &str) -> Option<u64> {
    v.trim().strip_prefix("bytes")?.rsplit_once('/')?.1.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    //! Fixtures are live responses captured 2026-09-18 from
    //! `/api/models?search=K2-Horizon&filter=gguf&...` (search),
    //! `/api/models?filter=base_model:quantized:IFM/K2-Horizon-MoVA-36B-A4B&filter=gguf`
    //! (derivatives), `/api/models/<repo>?blobs=true&expand[]=...` (model
    //! info), `/api/models/unsloth/gemma-4-26B-A4B-it-GGUF/tree/<sha>?recursive=true`
    //! (tree), and error responses of `auth-check`, `resolve` and `raw` (the
    //! `.http` files: status line, the headers the client reads, body).

    use super::*;
    use crate::test_http::{Resp, Server};
    use std::sync::{Arc, Mutex};

    fn fixture(name: &str) -> String {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/hub").join(name);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    }

    /// (status, header lookup, body) of a captured `.http` fixture.
    fn http_fixture(name: &str) -> (u16, Vec<(String, String)>, String) {
        let text = fixture(name).replace("\r\n", "\n");
        let (head, body) = text.split_once("\n\n").unwrap();
        let mut lines = head.lines();
        let status = lines.next().unwrap().split(' ').nth(1).unwrap().parse().unwrap();
        let headers = lines
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
            .collect();
        (status, headers, body.to_string())
    }

    fn classify(name: &str, repo: Option<&str>, had_token: bool) -> Error {
        let (status, headers, body) = http_fixture(name);
        let get = |k: &str| headers.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        status_error(status, &get, &body, "request", repo, had_token)
    }

    fn cfg() -> Config {
        Config::default_for_machine()
    }

    #[test]
    fn search_fixture() {
        let hits = parse_search(&fixture("search-k2-horizon.json")).unwrap();
        assert_eq!(hits.len(), 25);
        let nani = &hits[0];
        assert_eq!(nani.id, "NANI-Nithin/K2-Horizon-MoVA-36B-A4B-GGUF");
        assert_eq!(nani.author.as_deref(), Some("NANI-Nithin"));
        assert_eq!(nani.arch.as_deref(), Some("k2-horizon"));
        assert_eq!(nani.context_length, Some(524_288));
        assert_eq!(nani.total_params, Some(37_444_792_020));
        assert_eq!(nani.gated, Gated::No);
        assert_eq!(nani.library_name, None, "absent on this repo");
        assert_eq!(nani.pipeline_tag.as_deref(), Some("text-generation"));
        assert!(nani.downloads > 0 && nani.likes > 0);
        assert!(nani.last_modified.as_deref().is_some_and(|d| d.starts_with("2026-")));
        assert!(nani.tags.iter().any(|t| t == "base_model:quantized:IFM/K2-Horizon-MoVA-36B-A4B"));
        let ng = hits.iter().find(|h| h.id == "ngquocvinh/K2-Horizon-7B-GGUF").unwrap();
        assert_eq!(ng.library_name.as_deref(), Some("llama.cpp"));
        assert!(hits.iter().all(|h| h.arch.as_deref() == Some("k2-horizon")));
        // Sorted by downloads, as asked.
        assert!(hits.windows(2).all(|w| w[0].downloads >= w[1].downloads));

        let d = parse_search(&fixture("derivatives-IFM__K2-Horizon-MoVA-36B-A4B.json")).unwrap();
        assert_eq!(d.len(), 10);
        assert!(d.iter().any(|h| h.id == "kingjones777/K2-Horizon-MoVA-36B-A4B-ROCmFP4-GGUF"));

        assert!(matches!(
            parse_search("{\"error\":\"x\"}"),
            Err(Error::Http { kind: HttpErrorKind::Malformed, .. })
        ));
    }

    #[test]
    fn model_info_fixtures() {
        let ifm = parse_model_info(&fixture("model-info-IFM__K2-Horizon-7B-GGUF.json")).unwrap();
        assert_eq!(ifm.id, "IFM/K2-Horizon-7B-GGUF");
        assert_eq!(ifm.sha, "bcb8c25b76112ce96a962f5b8ab624435d1ee0c9");
        assert_eq!(ifm.gated, Gated::No);
        assert_eq!(ifm.gated_prompt, None);
        assert_eq!(ifm.card_license.as_deref(), Some("apache-2.0"));
        assert_eq!(ifm.base_models, vec![("quantized".to_string(), "IFM/K2-Horizon-7B".to_string())]);
        let g = ifm.gguf.as_ref().unwrap();
        assert_eq!(g.architecture.as_deref(), Some("k2-horizon"));
        assert_eq!(g.context_length, Some(524_288));
        let bf16 = ifm.siblings.iter().find(|f| f.path == "K2-Horizon-7B-BF16.gguf").unwrap();
        assert_eq!(bf16.size, 18_010_413_440);
        assert_eq!(
            bf16.sha256.as_deref(),
            Some("088c5d0814ef955d137fd1073ee3a68f6a53411fca44b75dc58a320640000444")
        );
        let readme = ifm.siblings.iter().find(|f| f.path == "README.md").unwrap();
        assert_eq!(readme.sha256, None, "a plain git file has no LFS hash");
        assert!(readme.size > 0);

        let nani = parse_model_info(&fixture("model-info-NANI-Nithin__K2-Horizon-MoVA-36B-A4B-GGUF.json")).unwrap();
        assert_eq!(nani.siblings.iter().filter(|f| f.path.ends_with(".gguf")).count(), 31);
        assert_eq!(nani.library_name, None);
        let q4 = nani.siblings.iter().find(|f| f.path.ends_with("-Q4_K_M.gguf")).unwrap();
        assert_eq!(q4.size, 22_368_011_616);
        assert!(q4.sha256.as_deref().unwrap().starts_with("513dd78590ac"));

        let unsloth = parse_model_info(&fixture("model-info-unsloth__gemma-4-26B-A4B-it-GGUF.json")).unwrap();
        assert_eq!(unsloth.pipeline_tag.as_deref(), Some("image-text-to-text"));
        assert_eq!(unsloth.base_models, vec![("quantized".to_string(), "google/gemma-4-26B-A4B-it".to_string())]);
        let shard = unsloth.siblings.iter().find(|f| f.path.starts_with("BF16/")).unwrap();
        assert_eq!(shard.name(), "gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf");

        let gated = parse_model_info(&fixture("model-info-google__gemma-3-4b-it.json")).unwrap();
        assert_eq!(gated.gated, Gated::Manual);
        assert!(gated.gated_prompt.as_deref().is_some_and(|p| !p.is_empty()));
        assert_eq!(gated.card_license.as_deref(), Some("gemma"));
        assert_eq!(gated.base_models, vec![("finetune".to_string(), "google/gemma-3-4b-pt".to_string())]);
        assert_eq!(gated.gguf, None);

        assert!(matches!(
            parse_model_info("{\"id\":\"a/b\"}"),
            Err(Error::Http { kind: HttpErrorKind::Malformed, .. })
        ));
    }

    /// Without the `baseModels` expansion the card, then the tags, say it.
    #[test]
    fn base_models_fall_back_to_card_and_tags() {
        let card = serde_json::json!({"cardData": {"base_model": ["a/x", "b/y"], "base_model_relation": "merge"}});
        assert_eq!(
            base_models_of(&card, &[]),
            vec![("merge".into(), "a/x".into()), ("merge".into(), "b/y".into())]
        );
        let tags: Vec<String> =
            ["base_model:IFM/K2", "base_model:quantized:IFM/K2", "base_model:adapter:o/r", "license:mit"]
                .map(String::from)
                .to_vec();
        assert_eq!(
            base_models_of(&serde_json::json!({}), &tags),
            vec![("quantized".into(), "IFM/K2".into()), ("adapter".into(), "o/r".into())]
        );
        let list = serde_json::json!({"baseModels": [{"relation": "finetune", "models": [{"id": "a/b"}]}]});
        assert_eq!(base_models_of(&list, &[]), vec![("finetune".into(), "a/b".into())]);
    }

    #[test]
    fn tree_fixture_matches_siblings() {
        let files = parse_tree(&fixture("tree-unsloth__gemma-4-26B-A4B-it-GGUF.json")).unwrap();
        assert_eq!(files.len(), 34, "36 entries less the BF16/ and MTP/ directories");
        let shard = files.iter().find(|f| f.path == "BF16/gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf").unwrap();
        assert_eq!(shard.size, 49_923_215_552);
        assert_eq!(
            shard.sha256.as_deref(),
            Some("56fb42adfb78ab308cd81675bc0f6ce4d4f3d9cccfbff4e9b053713eed976374")
        );
        // The same files, sizes and hashes as the model info's siblings.
        let info = parse_model_info(&fixture("model-info-unsloth__gemma-4-26B-A4B-it-GGUF.json")).unwrap();
        let mut a = files.clone();
        let mut b = info.siblings.clone();
        a.sort_by(|x, y| x.path.cmp(&y.path));
        b.sort_by(|x, y| x.path.cmp(&y.path));
        assert_eq!(a, b);
    }

    #[test]
    fn errors_are_classified_by_header() {
        match classify("gated-auth-check.http", Some("google/gemma-3-4b-it"), false) {
            Error::Http { kind: HttpErrorKind::Gated, message } => {
                assert!(message.contains("https://huggingface.co/google/gemma-3-4b-it"), "{message}");
                assert!(message.contains("token"), "{message}");
            }
            other => panic!("{other:?}"),
        }
        match classify("gated-resolve.http", Some("google/gemma-3-4b-it"), true) {
            Error::Http { kind: HttpErrorKind::Gated, message } => assert!(message.contains("no access yet"), "{message}"),
            other => panic!("{other:?}"),
        }
        // 401 without an error code: missing or private.
        match classify("missing-repo.http", Some("IFM/No-Such-Repo-xyz"), false) {
            Error::Http { kind: HttpErrorKind::RepoNotFound, message } => {
                assert!(message.starts_with("IFM/No-Such-Repo-xyz: not found"), "{message}")
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            classify("missing-entry.http", Some("IFM/K2-Horizon-7B-GGUF"), false),
            Error::Http { kind: HttpErrorKind::EntryNotFound, .. }
        ));

        let headers = [("ratelimit", "\"api\";r=0;t=74")];
        let get = |k: &str| headers.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        match status_error(429, &get, "", "search", None, false) {
            Error::Http { kind: HttpErrorKind::RateLimited { retry_after_secs: Some(74) }, message } => {
                assert!(message.contains("74 s") && message.contains("token raises"), "{message}")
            }
            other => panic!("{other:?}"),
        }
        let headers = [("retry-after", "9"), ("ratelimit", "\"api\";r=0;t=74")];
        let get = |k: &str| headers.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        assert!(matches!(
            status_error(429, &get, "", "search", None, true),
            Error::Http { kind: HttpErrorKind::RateLimited { retry_after_secs: Some(9) }, .. }
        ));
        let none = |_: &str| None;
        match status_error(500, &none, "{\"error\":\"boom\"}", "x/y", None, false) {
            Error::Http { kind: HttpErrorKind::Status { status: 500 }, message } => {
                assert_eq!(message, "x/y: HTTP 500: boom")
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn token_precedence() {
        let dir = std::env::temp_dir().join(format!("fidim-hub-token-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cli = dir.join(".cache").join("huggingface");
        std::fs::create_dir_all(&cli).unwrap();
        std::fs::write(cli.join("token"), "hf_cli\n").unwrap();
        let hf_home = dir.join("hfhome");
        std::fs::create_dir_all(&hf_home).unwrap();
        std::fs::write(hf_home.join("token"), "hf_home").unwrap();
        let path_file = dir.join("explicit-token");
        std::fs::write(&path_file, "  hf_path  \r\n").unwrap();

        let env_of = |pairs: Vec<(&'static str, OsString)>| {
            move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone())
        };
        let mut c = cfg();
        c.hf_token = Some("hf_config".into());
        let all = env_of(vec![("HF_TOKEN", "hf_env".into()), ("HF_TOKEN_PATH", path_file.clone().into())]);
        assert_eq!(token_with(&c, &all, Some(&dir)).as_deref(), Some("hf_env"));
        let path_only = env_of(vec![("HF_TOKEN", "".into()), ("HF_TOKEN_PATH", path_file.clone().into())]);
        assert_eq!(token_with(&c, &path_only, Some(&dir)).as_deref(), Some("hf_path"));
        let nothing = env_of(vec![]);
        assert_eq!(token_with(&c, &nothing, Some(&dir)).as_deref(), Some("hf_config"));

        // The CLI's file only with the opt-in, and HF_HOME moves it.
        c.hf_token = None;
        assert_eq!(token_with(&c, &nothing, Some(&dir)), None);
        c.hf_use_cli_token = true;
        assert_eq!(token_with(&c, &nothing, Some(&dir)).as_deref(), Some("hf_cli"));
        let home_env = env_of(vec![("HF_HOME", hf_home.clone().into())]);
        assert_eq!(token_with(&c, &home_env, Some(&dir)).as_deref(), Some("hf_home"));
        // A missing HF_TOKEN_PATH file falls through to the next source.
        let dangling = env_of(vec![("HF_TOKEN_PATH", dir.join("nope").into())]);
        assert_eq!(token_with(&c, &dangling, Some(&dir)).as_deref(), Some("hf_cli"));
        // A value that cannot be a header is not a token.
        c.hf_token = Some("hf_bad token".into());
        c.hf_use_cli_token = false;
        assert_eq!(token_with(&c, &nothing, Some(&dir)), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn repo_input_and_urls() {
        let r = |s: &str| parse_repo_input(s);
        assert_eq!(r("IFM/K2-Horizon-7B-GGUF").unwrap().repo, "IFM/K2-Horizon-7B-GGUF");
        assert_eq!(
            r("https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/blob/main/BF16/x-00001-of-00002.gguf?download=true"),
            Some(RepoRef {
                repo: "unsloth/gemma-4-26B-A4B-it-GGUF".into(),
                rev: Some("main".into()),
                path: Some("BF16/x-00001-of-00002.gguf".into()),
            })
        );
        assert_eq!(r("hf.co/IFM/K2-Horizon-7B/tree/rl_tool-use").unwrap().rev.as_deref(), Some("rl_tool-use"));
        assert_eq!(r("  huggingface.co/IFM/K2-Horizon-7B/  ").unwrap().rev, None);
        assert_eq!(r("https://huggingface.co/datasets/o/r"), None);
        assert_eq!(r("https://github.com/ifm-ai/llama.cpp"), None);
        assert_eq!(r("owner"), None);
        assert_eq!(r("../etc"), None);
        assert_eq!(r("o/r:Q4_K_M"), None);
        assert!(validate_repo("NANI-Nithin/K2-Horizon-MoVA-36B-A4B-GGUF").is_ok());
        for bad in ["a/..", "../b", "a/b/c", "a/-b", "a/b.", "a--b/c", "a/b c", "", "a/", "/b"] {
            assert!(validate_repo(bad).is_err(), "{bad:?}");
        }
        assert_eq!(
            resolve_url_at("https://huggingface.co", "o/r", "abc123", "BF16/a b#1.gguf"),
            "https://huggingface.co/o/r/resolve/abc123/BF16/a%20b%231.gguf"
        );
        assert_eq!(repo_of_resolve_url("https://huggingface.co/o/r/resolve/abc/x.gguf").as_deref(), Some("o/r"));
        assert_eq!(repo_of_resolve_url("https://example.com/file.zip"), None);
    }

    #[test]
    fn link_and_content_range_headers() {
        let link = "<https://huggingface.co/api/models?search=K2-Horizon&filter=gguf&cursor=eyJ9>; rel=\"next\"";
        assert_eq!(next_link(link).as_deref(), Some("https://huggingface.co/api/models?search=K2-Horizon&filter=gguf&cursor=eyJ9"));
        assert_eq!(next_link("<https://a/b>; rel=\"prev\""), None);
        assert_eq!(next_link("<https://a/p>; rel=\"prev\", <https://a/n>; rel=\"next\"").as_deref(), Some("https://a/n"));
        assert_eq!(content_range("bytes 0-65535/5592219008"), Some((0, Some(5_592_219_008))));
        assert_eq!(content_range("bytes 100-199/*"), Some((100, None)));
        assert_eq!(content_range("items 0-1/2"), None);
        assert_eq!(content_range("bytes */4096"), None);
        assert_eq!(content_range_total("bytes */4096"), Some(4096));
        assert_eq!(content_range_total("bytes 0-1/77"), Some(77));
        assert_eq!(content_range_total("bytes 0-1/*"), None);
    }

    // ------------------------------------------------ against a local server

    /// Serve `file` with Range support at `/o/r/resolve/<rev>/<name>`.
    fn range_handler(file: Arc<Vec<u8>>, honour_range: bool) -> impl Fn(&crate::test_http::Req) -> Resp + Send {
        move |req| {
            if !req.path().contains("/resolve/") {
                return Resp::new(404, "Entry not found").header("X-Error-Code", "EntryNotFound");
            }
            let range = req.header("range").and_then(|r| r.strip_prefix("bytes=")).and_then(|r| r.split_once('-'));
            match range {
                Some((a, b)) if honour_range => {
                    let a: usize = a.parse().unwrap();
                    let b: usize = b.parse::<usize>().unwrap().min(file.len() - 1);
                    Resp::new(206, file[a..=b].to_vec())
                        .header("Content-Range", format!("bytes {a}-{b}/{}", file.len()))
                }
                _ => Resp::new(200, file.to_vec()),
            }
        }
    }

    /// A header whose vocabulary pushes the KV table past 1 MiB.
    fn big_vocab_gguf() -> Vec<u8> {
        let mut out = Vec::new();
        let kv = |out: &mut Vec<u8>, k: &str| {
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
        };
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        out.extend_from_slice(&4u64.to_le_bytes());
        kv(&mut out, "general.architecture");
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&(10u64).to_le_bytes());
        out.extend_from_slice(b"k2-horizon");
        kv(&mut out, "k2-horizon.block_count");
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&36u32.to_le_bytes());
        kv(&mut out, "tokenizer.ggml.pre");
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&(10u64).to_le_bytes());
        out.extend_from_slice(b"k2-horizon");
        kv(&mut out, "tokenizer.ggml.tokens");
        out.extend_from_slice(&9u32.to_le_bytes());
        out.extend_from_slice(&8u32.to_le_bytes());
        let n = 150_000u64;
        out.extend_from_slice(&n.to_le_bytes());
        for i in 0..n {
            let s = format!("t{i:08}");
            out.extend_from_slice(&(s.len() as u64).to_le_bytes());
            out.extend_from_slice(s.as_bytes());
        }
        // tensor info: name, n_dims, dims, type 12 (Q4_K), offset
        out.extend_from_slice(&(5u64).to_le_bytes());
        out.extend_from_slice(b"t.w.0");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&64u64.to_le_bytes());
        out.extend_from_slice(&12u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend(std::iter::repeat_n(0u8, 4096)); // "tensor data"
        out
    }

    #[test]
    fn remote_header_grows_its_range_reads() {
        let file = Arc::new(big_vocab_gguf());
        assert!(file.len() > 2 * 1024 * 1024);
        let srv = Server::start("127.0.0.1", range_handler(file.clone(), true)).unwrap();
        let size = file.len() as u64;

        let early = remote_header_at(&srv.base, None, "o/r", "abc", "m.gguf", size, ReadMode::UntilTokenizer).unwrap();
        assert_eq!(early.architecture.as_deref(), Some("k2-horizon"));
        assert_eq!(early.tokenizer_pre.as_deref(), Some("k2-horizon"));
        assert_eq!(early.vocab_size, Some(150_000));
        assert_eq!(early.file_size, size);
        assert_eq!(early.path, PathBuf::from("hf://o/r@abc/m.gguf"));
        let reqs = srv.requests();
        assert_eq!(reqs.len(), 1, "one 64 KiB read");
        assert_eq!(reqs[0].path(), "/o/r/resolve/abc/m.gguf");
        assert_eq!(reqs[0].header("range"), Some("bytes=0-65535"));

        let full = remote_header_at(&srv.base, Some("hf_x"), "o/r", "abc", "m.gguf", size, ReadMode::Full).unwrap();
        assert_eq!(full.max_tensor_type(), Some(12));
        let ranges: Vec<String> =
            srv.requests()[1..].iter().map(|r| r.header("range").unwrap().to_string()).collect();
        // 1 MiB, then only the missing part of each doubling.
        assert_eq!(ranges[0], "bytes=0-1048575");
        assert_eq!(ranges[1], "bytes=1048576-2097151");
        assert!(ranges.len() >= 3, "{ranges:?}");
        assert!(srv.requests()[1..].iter().all(|r| r.header("authorization") == Some("Bearer hf_x")));

        // A server that ignores Range is read from the start and trimmed.
        let plain = Server::start("127.0.0.1", range_handler(file.clone(), false)).unwrap();
        let h = remote_header_at(&plain.base, None, "o/r", "abc", "m.gguf", size, ReadMode::Full).unwrap();
        assert_eq!(h.vocab_size, Some(150_000));
        assert_eq!(h.tensors.len(), 1);

        // A header larger than the file it claims to be is an error, not a loop.
        let small = remote_header_at(&srv.base, None, "o/r", "abc", "m.gguf", 100_000, ReadMode::Full);
        assert!(matches!(small, Err(Error::GgufTruncated { .. })), "{small:?}");

        // A path that could climb out of the repo is refused before any request.
        let n = srv.requests().len();
        let bad = remote_header_at(&srv.base, None, "o/r", "abc", "../x", size, ReadMode::Full);
        assert!(matches!(bad, Err(Error::InvalidInput(_))), "{bad:?}");
        assert_eq!(srv.requests().len(), n);
        for p in ["a/../b", "a//b", "", "a\\b", "./a"] {
            assert!(validate_repo_path(p).is_err(), "{p:?}");
        }
        assert!(validate_repo_path("BF16/x-00001-of-00002.gguf").is_ok());
    }

    /// The token goes to the Hub and never to the CDN it redirects to.
    #[test]
    fn token_is_not_sent_across_hosts() {
        let file = Arc::new(big_vocab_gguf());
        // A second loopback address stands in for the CDN host.
        let Ok(cdn) = Server::start("127.0.0.2", range_handler(file.clone(), true)) else {
            eprintln!("127.0.0.2 cannot be bound here; skipped");
            return;
        };
        let cdn_base = cdn.base.clone();
        let hub = Server::start("127.0.0.1", move |req| {
            Resp::new(302, "").header("Location", format!("{cdn_base}{}?Signature=s", req.path()))
        })
        .unwrap();
        let h = remote_header_at(&hub.base, Some("hf_secret"), "o/r", "abc", "m.gguf", file.len() as u64, ReadMode::UntilTokenizer)
            .unwrap();
        assert_eq!(h.vocab_size, Some(150_000));
        assert_eq!(hub.requests()[0].header("authorization"), Some("Bearer hf_secret"));
        let at_cdn = cdn.requests();
        assert_eq!(at_cdn.len(), 1);
        assert_eq!(at_cdn[0].header("authorization"), None, "token leaked to the CDN");
        assert_eq!(at_cdn[0].header("range"), Some("bytes=0-65535"), "Range survives the redirect");
    }

    #[test]
    fn api_calls_against_a_local_server() {
        let search = fixture("search-k2-horizon.json");
        let info = fixture("model-info-IFM__K2-Horizon-7B-GGUF.json");
        let pages = Arc::new(Mutex::new(0));
        let base_cell: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let (pages2, base2) = (pages.clone(), base_cell.clone());
        let srv = Server::start("127.0.0.1", move |req| {
            let p = req.path();
            if p == "/api/models" {
                Resp::new(200, search.clone())
            } else if p == "/api/models/IFM/K2-Horizon-7B-GGUF" || p == "/api/models/IFM/K2-Horizon-7B-GGUF/revision/refs%2Fpr%2F1" {
                Resp::new(200, info.clone())
            } else if p.starts_with("/api/models/o/r/tree/") {
                let mut n = pages2.lock().unwrap();
                *n += 1;
                let body = format!("[{{\"type\":\"file\",\"path\":\"f{n}.gguf\",\"size\":{n}}}]");
                if *n == 1 {
                    let next = format!("{}/api/models/o/r/tree/main?recursive=true&cursor=2", base2.lock().unwrap());
                    Resp::new(200, body).header("Link", format!("<{next}>; rel=\"next\""))
                } else {
                    // A next link off the Hub is not followed.
                    Resp::new(200, body).header("Link", "<https://elsewhere.example/p3>; rel=\"next\"")
                }
            } else if p == "/o/r/raw/abc/README.md" {
                Resp::new(200, "See https://github.com/ifm-ai/llama.cpp/tree/model/K2Horizon")
            } else if p == "/o/none/raw/abc/README.md" {
                Resp::new(404, "Entry not found").header("X-Error-Code", "EntryNotFound")
            } else if p == "/api/models/o/r/auth-check" {
                Resp::new(200, "OK")
            } else if p == "/api/models/g/gated/auth-check" {
                Resp::new(401, "{\"error\":\"restricted\"}").header("X-Error-Code", "GatedRepo")
            } else {
                Resp::new(404, "?")
            }
        })
        .unwrap();
        *base_cell.lock().unwrap() = srv.base.clone();

        let q = SearchQuery { text: "K2-Horizon".into(), author: Some("IFM".into()), limit: 500, ..Default::default() };
        let hits = search_at(&srv.base, Some("hf_t"), &q, &["base_model:quantized:a/b".into()]).unwrap();
        assert_eq!(hits.len(), 25);
        let r = &srv.requests()[0];
        assert_eq!(r.header("authorization"), Some("Bearer hf_t"));
        for part in [
            "search=K2-Horizon",
            "author=IFM",
            "filter=base_model%3Aquantized%3Aa%2Fb",
            "filter=gguf",
            "sort=downloads",
            "direction=-1",
            "limit=100",
            "expand%5B%5D=gguf",
            "expand%5B%5D=gated",
        ] {
            assert!(r.target.contains(part), "{part} missing from {}", r.target);
        }
        assert!(r.header("user-agent").is_some_and(|u| u.starts_with("llama-fidim/")));

        let info = model_info_at(&srv.base, None, "IFM/K2-Horizon-7B-GGUF", None).unwrap();
        assert_eq!(info.sha, "bcb8c25b76112ce96a962f5b8ab624435d1ee0c9");
        let r = srv.requests().last().unwrap().clone();
        assert!(r.target.contains("blobs=true") && r.target.contains("expand%5B%5D=siblings"), "{}", r.target);
        assert_eq!(r.header("authorization"), None);
        model_info_at(&srv.base, None, "IFM/K2-Horizon-7B-GGUF", Some("refs/pr/1")).unwrap();
        assert!(matches!(model_info_at(&srv.base, None, "not a repo", None), Err(Error::InvalidInput(_))));

        let files = list_files_at(&srv.base, None, "o/r", "main").unwrap();
        assert_eq!(files.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(), vec!["f1.gguf", "f2.gguf"]);

        assert!(readme_at(&srv.base, None, "o/r", "abc").unwrap().unwrap().contains("ifm-ai/llama.cpp"));
        assert_eq!(readme_at(&srv.base, None, "o/none", "abc").unwrap(), None);

        auth_check_at(&srv.base, None, "o/r").unwrap();
        match auth_check_at(&srv.base, None, "g/gated") {
            Err(Error::Http { kind: HttpErrorKind::Gated, message }) => assert!(message.contains("g/gated"), "{message}"),
            other => panic!("{other:?}"),
        }
    }

    /// Against the real Hub; run with `cargo test -- --ignored live_`.
    #[test]
    #[ignore]
    fn live_hub_k2_horizon() {
        let c = cfg();
        let hits = search(&c, &SearchQuery { text: "K2-Horizon".into(), limit: 10, ..Default::default() }).unwrap();
        assert!(hits.iter().any(|h| h.arch.as_deref() == Some("k2-horizon")));
        let info = model_info(&c, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
        let q4 = info.siblings.iter().find(|f| f.path.ends_with("Q4_K_M.gguf")).unwrap();
        let h = remote_header(&c, &info.id, &info.sha, &q4.path, q4.size, ReadMode::UntilTokenizer).unwrap();
        assert_eq!(h.architecture.as_deref(), Some("k2-horizon"));
        assert_eq!(h.tokenizer_pre.as_deref(), Some("k2-horizon"));
        assert_eq!(h.vocab_size, Some(250_624));
        let derived = derivatives(&c, "IFM/K2-Horizon-MoVA-36B-A4B").unwrap();
        assert!(derived.len() >= 5);
        assert!(readme(&c, "IFM/K2-Horizon-7B-GGUF", "main").unwrap().unwrap().contains("llama.cpp"));
        auth_check(&c, "IFM/K2-Horizon-7B-GGUF").unwrap();
        assert!(matches!(
            auth_check(&c, "google/gemma-3-4b-it"),
            Err(Error::Http { kind: HttpErrorKind::Gated, .. })
        ));
    }
}
