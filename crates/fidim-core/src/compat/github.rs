//! GitHub for the build resolver: llama.cpp refs named in a model card,
//! resolving them to a pinned commit, and upstream pull requests that add
//! an architecture.
//!
//! Unauthenticated, GitHub allows 60 API requests an hour per IP address
//! (shared by every program on the machine) and 10 searches a minute, so
//! every API answer is cached on disk with its ETag: for five minutes no
//! request is made at all, and after that a revalidation that comes back
//! 304 costs nothing against the limit. A token (`GITHUB_TOKEN`, or
//! `config.github_token`) raises the limits to 5000 and 30. File reads go
//! to raw.githubusercontent.com at a pinned commit, which has no quota.
//!
//! Model cards are untrusted text: only URLs are taken from them, through
//! one strict pattern, and nothing they say is acted on.

use std::io::Read;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{compat_err, is_hex, valid_owner, valid_repo, ModelNeeds, Support, UPSTREAM_OWNER, UPSTREAM_REPO};
use crate::config::Config;
use crate::{Error, Result};

const API: &str = "https://api.github.com";
const USER_AGENT: &str = concat!("llama-fidim/", env!("CARGO_PKG_VERSION"));
const JSON: &str = "application/vnd.github+json";
/// `GET /repos/{o}/{r}/commits/{ref}` answers with the bare SHA.
const SHA_ONLY: &str = "application/vnd.github.sha";
/// Upstream's default branch, the base of every comparison.
pub const UPSTREAM_BRANCH: &str = "master";
/// How long an API answer is reused without asking GitHub again.
const FRESH: Duration = Duration::from_secs(300);
/// Largest raw file read (the oldest layouts keep everything in a 1 MB
/// `llama.cpp`).
const MAX_RAW_BYTES: u64 = 16 << 20;

/// The GitHub token: `GITHUB_TOKEN` in the environment, else
/// `config.github_token`. Blank values count as none.
pub fn token(cfg: &Config) -> Option<String> {
    std::env::var("GITHUB_TOKEN")
        .ok()
        .or_else(|| cfg.github_token.clone())
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(60))
        // A renamed repository answers 301 to api.github.com/repositories/<id>:
        // the token may follow on the same host, never elsewhere.
        .redirect_auth_headers(ureq::RedirectAuthHeaders::SameHost)
        .build()
}

fn read_capped(resp: ureq::Response, what: &str) -> Result<String> {
    let mut buf = Vec::new();
    resp.into_reader()
        .take(MAX_RAW_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| compat_err(format!("{what}: reading body: {e}")))?;
    if buf.len() as u64 > MAX_RAW_BYTES {
        return Err(compat_err(format!("{what}: larger than {} MB", MAX_RAW_BYTES >> 20)));
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// One file from raw.githubusercontent.com (no token, no quota). None = 404.
pub fn fetch_raw(url: &str) -> Result<Option<String>> {
    match agent().get(url).set("User-Agent", USER_AGENT).call() {
        Ok(resp) => read_capped(resp, url).map(Some),
        Err(ureq::Error::Status(404, _)) => Ok(None),
        Err(ureq::Error::Status(code, _)) => Err(compat_err(format!("{url}: HTTP {code}"))),
        Err(e) => Err(compat_err(format!("{url}: {e}"))),
    }
}

// ------------------------------------------------------------ API client ----

/// A cached API answer.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Cached {
    url: String,
    accept: String,
    #[serde(default)]
    etag: Option<String>,
    fetched_unix: u64,
    /// None = the resource does not exist (404).
    body: Option<String>,
}

/// GitHub's REST API with an ETag cache on disk.
pub struct Api {
    base: String,
    token: Option<String>,
    cache_dir: Option<PathBuf>,
    fresh: Duration,
}

impl Api {
    pub fn new(cfg: &Config) -> Self {
        Api {
            base: API.to_string(),
            token: token(cfg),
            cache_dir: Some(Config::config_dir().join("cache").join("github")),
            fresh: FRESH,
        }
    }

    /// Against another host (tests run a local fake), with or without a cache.
    pub fn with_base(base: &str, token: Option<String>, cache_dir: Option<PathBuf>, fresh: Duration) -> Self {
        Api { base: base.trim_end_matches('/').to_string(), token, cache_dir, fresh }
    }

    fn cache_path(&self, url: &str, accept: &str) -> Option<PathBuf> {
        use sha2::{Digest, Sha256};
        let dir = self.cache_dir.as_ref()?;
        let h = Sha256::digest(format!("{url}\n{accept}").as_bytes());
        let name: String = h.iter().take(12).map(|b| format!("{b:02x}")).collect();
        Some(dir.join(format!("{name}.json")))
    }

    fn load(&self, url: &str, accept: &str) -> Option<Cached> {
        let p = self.cache_path(url, accept)?;
        let c: Cached = serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
        (c.url == url && c.accept == accept).then_some(c)
    }

    fn store(&self, c: &Cached) {
        let Some(p) = self.cache_path(&c.url, &c.accept) else { return };
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(text) = serde_json::to_string(c) {
            let _ = std::fs::write(p, text);
        }
    }

    /// `GET <base><path>`: the body, or None for 404. Fresh cache first,
    /// then a conditional request; a stale answer beats no answer when
    /// GitHub is unreachable or the hourly limit is spent.
    pub fn get(&self, path: &str, accept: &str) -> Result<Option<String>> {
        let url = format!("{}{path}", self.base);
        let cached = self.load(&url, accept);
        if let Some(c) = &cached {
            if now_unix().saturating_sub(c.fetched_unix) < self.fresh.as_secs() {
                return Ok(c.body.clone());
            }
        }
        let mut req = agent()
            .get(&url)
            .set("User-Agent", USER_AGENT)
            .set("Accept", accept)
            .set("X-GitHub-Api-Version", "2022-11-28");
        if let Some(t) = &self.token {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        if let Some(etag) = cached.as_ref().and_then(|c| c.etag.as_deref()) {
            req = req.set("If-None-Match", etag);
        }
        match req.call() {
            Ok(resp) if resp.status() == 304 => {
                let mut c = cached.ok_or_else(|| compat_err(format!("{url}: 304 without a cached copy")))?;
                c.fetched_unix = now_unix();
                self.store(&c);
                Ok(c.body)
            }
            Ok(resp) => {
                let etag = resp.header("ETag").map(str::to_string);
                let body = read_capped(resp, &url)?;
                self.store(&Cached { url: url.clone(), accept: accept.into(), etag, fetched_unix: now_unix(), body: Some(body.clone()) });
                Ok(Some(body))
            }
            Err(ureq::Error::Status(404, _)) => {
                self.store(&Cached { url: url.clone(), accept: accept.into(), etag: None, fetched_unix: now_unix(), body: None });
                Ok(None)
            }
            Err(ureq::Error::Status(code, resp)) => {
                let err = status_error(&url, code, resp, self.token.is_some());
                match cached {
                    Some(c) if code == 403 || code == 429 || code >= 500 => Ok(c.body),
                    _ => Err(err),
                }
            }
            Err(e) => match cached {
                Some(c) => Ok(c.body),
                None => Err(compat_err(format!("GitHub API {url}: {e}"))),
            },
        }
    }
}

/// A readable error for a failed API call; the rate limit names its fix.
fn status_error(url: &str, code: u16, resp: ureq::Response, authed: bool) -> Error {
    let remaining = resp.header("X-RateLimit-Remaining").map(str::to_string);
    let reset = resp.header("X-RateLimit-Reset").and_then(|s| s.parse::<u64>().ok());
    let retry_after = resp.header("Retry-After").map(str::to_string);
    let message = resp
        .into_string()
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v["message"].as_str().map(str::to_string))
        .unwrap_or_default();
    let limited = code == 429 || (code == 403 && (remaining.as_deref() == Some("0") || message.contains("rate limit")));
    if limited {
        let when = match (reset, retry_after) {
            (Some(r), _) => format!("resets in {} min", r.saturating_sub(now_unix()).div_ceil(60)),
            (None, Some(s)) => format!("retry after {s} s"),
            _ => "try again later".into(),
        };
        let fix = if authed {
            String::new()
        } else {
            "; set GITHUB_TOKEN or config.github_token (any read-only token) for 5000 requests an hour".into()
        };
        return compat_err(format!(
            "GitHub API rate limit reached ({} requests an hour {}); {when}{fix}",
            if authed { 5000 } else { 60 },
            if authed { "for this token" } else { "per IP address without a token" }
        ));
    }
    if code == 401 {
        return compat_err("GitHub rejected the token (401): check GITHUB_TOKEN / config.github_token".to_string());
    }
    compat_err(format!("GitHub API {url}: HTTP {code}{}", if message.is_empty() { String::new() } else { format!(": {message}") }))
}

/// Percent-encode everything but RFC 3986 unreserved characters.
pub(crate) fn pct(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

// ------------------------------------------------------------ card refs ----

/// What a llama.cpp link points at.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "lowercase")]
pub enum RefKind {
    /// The repository itself: its default branch.
    Repo,
    /// `/tree/<branch>` (branch names may contain `/`).
    Branch(String),
    /// `/pull/<n>`.
    Pull(u32),
    /// `/commit/<sha>`, 7 to 40 hex digits, lowercase.
    Commit(String),
}

/// A llama.cpp repository link, as written (before redirects).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitRef {
    pub owner: String,
    pub repo: String,
    pub kind: RefKind,
}

impl GitRef {
    pub fn is_upstream_repo(&self) -> bool {
        self.owner.eq_ignore_ascii_case(UPSTREAM_OWNER) && self.repo.eq_ignore_ascii_case(UPSTREAM_REPO)
    }

    /// `github.com/<o>/<r>/tree/<b>` etc., for display.
    pub fn url(&self) -> String {
        let base = format!("https://github.com/{}/{}", self.owner, self.repo);
        match &self.kind {
            RefKind::Repo => base,
            RefKind::Branch(b) => format!("{base}/tree/{b}"),
            RefKind::Pull(n) => format!("{base}/pull/{n}"),
            RefKind::Commit(c) => format!("{base}/commit/{c}"),
        }
    }
}

/// At most this many links are taken from one card.
const MAX_CARD_REFS: usize = 20;

/// llama.cpp links in a model card: `github.com/<o>/<r>` with an optional
/// `/tree/<branch>`, `/pull/<n>` or `/commit/<sha>`, where the repository
/// name contains `llama.cpp` (any case). The card is untrusted: this only
/// matches URLs against one strict pattern, and returns them deduplicated
/// in order of appearance.
pub fn card_refs(readme: &str) -> Vec<GitRef> {
    let re = regex::Regex::new(
        r"(?i)(?:https?://)?(?:www\.)?github\.com/([A-Za-z0-9][A-Za-z0-9-]{0,38})/([A-Za-z0-9._-]{1,100})(?:/(tree|pull|commit)/([A-Za-z0-9._/-]{1,250}))?",
    )
    .unwrap();
    let mut out: Vec<GitRef> = Vec::new();
    for c in re.captures_iter(readme) {
        let owner = c[1].to_string();
        let mut repo = c[2].trim_end_matches('.').to_string();
        if repo.to_ascii_lowercase().ends_with(".git") {
            repo.truncate(repo.len() - 4);
        }
        if !repo.to_ascii_lowercase().contains("llama.cpp") || !valid_owner(&owner) || !valid_repo(&repo) {
            continue;
        }
        let kind = match (c.get(3).map(|m| m.as_str().to_ascii_lowercase()), c.get(4).map(|m| m.as_str())) {
            (None, _) => RefKind::Repo,
            (Some(k), Some(rest)) if k == "tree" => {
                let b = rest.trim_end_matches(['.', '/']);
                if b.is_empty() || b.starts_with(['-', '/', '.']) || b.contains("..") || b.contains("//") {
                    continue;
                }
                RefKind::Branch(b.to_string())
            }
            (Some(k), Some(rest)) if k == "pull" => {
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                match digits.parse::<u32>() {
                    Ok(n) if n > 0 => RefKind::Pull(n),
                    _ => continue,
                }
            }
            (Some(_), Some(rest)) => {
                let hex: String = rest.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
                if hex.len() < 7 || hex.len() > 40 {
                    continue;
                }
                RefKind::Commit(hex.to_ascii_lowercase())
            }
            (Some(_), None) => continue,
        };
        let r = GitRef { owner, repo, kind };
        let dup = out.iter().any(|o| {
            o.owner.eq_ignore_ascii_case(&r.owner) && o.repo.eq_ignore_ascii_case(&r.repo) && o.kind == r.kind
        });
        if !dup {
            out.push(r);
            if out.len() >= MAX_CARD_REFS {
                break;
            }
        }
    }
    out
}

/// `(owner, repo)` of a GitHub clone URL.
pub fn owner_repo_from_url(url: &str) -> Option<(String, String)> {
    let rest = url.trim().trim_end_matches('/');
    let rest = rest.strip_prefix("https://").or_else(|| rest.strip_prefix("http://")).unwrap_or(rest);
    let rest = rest.strip_prefix("www.").unwrap_or(rest);
    let rest = rest.strip_prefix("github.com/")?;
    let mut parts = rest.split('/');
    let owner = parts.next()?.to_string();
    let mut repo = parts.next()?.to_string();
    if parts.next().is_some() {
        return None;
    }
    if repo.to_ascii_lowercase().ends_with(".git") {
        repo.truncate(repo.len() - 4);
    }
    (valid_owner(&owner) && valid_repo(&repo)).then_some((owner, repo))
}

// -------------------------------------------------------------- parsers ----

/// The fields of `GET /repos/{o}/{r}` the resolver uses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepoMeta {
    /// Canonical `owner/repo` after any rename or transfer.
    pub full_name: String,
    pub fork: bool,
    pub parent: Option<String>,
    /// The root of the fork network.
    pub source: Option<String>,
    pub default_branch: String,
}

pub fn parse_repo(json: &str) -> Result<RepoMeta> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let full_name = v["full_name"].as_str().ok_or_else(|| compat_err("repository JSON has no full_name"))?.to_string();
    Ok(RepoMeta {
        full_name,
        fork: v["fork"].as_bool().unwrap_or(false),
        parent: v["parent"]["full_name"].as_str().map(str::to_string),
        source: v["source"]["full_name"].as_str().map(str::to_string),
        default_branch: v["default_branch"].as_str().unwrap_or(UPSTREAM_BRANCH).to_string(),
    })
}

/// The fields of a compare the resolver uses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompareInfo {
    pub status: String,
    pub ahead_by: u32,
    pub behind_by: u32,
    pub total_commits: u32,
    pub merge_base: Option<String>,
    /// First line of each listed commit's message, oldest first.
    pub subjects: Vec<String>,
}

pub fn parse_compare(json: &str) -> Result<CompareInfo> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let n = |k: &str| v[k].as_u64().map(|x| x as u32);
    let ahead_by = n("ahead_by").ok_or_else(|| compat_err("compare JSON has no ahead_by"))?;
    Ok(CompareInfo {
        status: v["status"].as_str().unwrap_or("").to_string(),
        ahead_by,
        behind_by: n("behind_by").unwrap_or(0),
        total_commits: n("total_commits").unwrap_or(ahead_by),
        merge_base: v["merge_base_commit"]["sha"].as_str().map(str::to_string),
        subjects: v["commits"]
            .as_array()
            .map(|a| a.iter().filter_map(|c| c["commit"]["message"].as_str()).map(subject_line).collect())
            .unwrap_or_default(),
    })
}

fn subject_line(message: &str) -> String {
    let line = message.lines().next().unwrap_or("").trim();
    // Commit subjects are someone else's text; keep them one short line.
    let clean: String = line.chars().filter(|c| !c.is_control()).take(200).collect();
    clean
}

/// A pull request, as the resolver reports it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrInfo {
    pub number: u32,
    pub title: String,
    /// `open` or `closed`.
    pub state: String,
    pub draft: bool,
    pub merged: bool,
    /// GitHub's view of merging it now: `clean`, `dirty` (conflicts),
    /// `blocked`, `unstable`, `unknown`... None when not computed yet.
    pub mergeable_state: Option<String>,
    /// `owner/repo` the branch lives in; None when that fork was deleted.
    pub head_repo: Option<String>,
    pub head_ref: String,
    pub head_sha: String,
    pub base_ref: String,
    pub html_url: String,
}

pub fn parse_pull(json: &str) -> Result<PrInfo> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let number = v["number"].as_u64().ok_or_else(|| compat_err("pull request JSON has no number"))? as u32;
    let head_sha = v["head"]["sha"].as_str().ok_or_else(|| compat_err("pull request JSON has no head.sha"))?.to_string();
    Ok(PrInfo {
        number,
        title: subject_line(v["title"].as_str().unwrap_or("")),
        state: v["state"].as_str().unwrap_or("").to_string(),
        draft: v["draft"].as_bool().unwrap_or(false),
        merged: v["merged"].as_bool().unwrap_or(false) || v["merged_at"].as_str().is_some(),
        mergeable_state: v["mergeable_state"].as_str().map(str::to_string),
        head_repo: v["head"]["repo"]["full_name"].as_str().map(str::to_string),
        head_ref: v["head"]["ref"].as_str().unwrap_or("").to_string(),
        head_sha,
        base_ref: v["base"]["ref"].as_str().unwrap_or(UPSTREAM_BRANCH).to_string(),
        html_url: v["html_url"].as_str().unwrap_or("").to_string(),
    })
}

/// One pull request from a search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrHit {
    pub number: u32,
    pub title: String,
    pub state: String,
    pub draft: bool,
    pub merged: bool,
    pub html_url: String,
}

pub fn parse_search_prs(json: &str) -> Result<Vec<PrHit>> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let items = v["items"].as_array().ok_or_else(|| compat_err("search JSON has no items"))?;
    Ok(items
        .iter()
        .filter(|i| i["pull_request"].is_object())
        .filter_map(|i| {
            Some(PrHit {
                number: i["number"].as_u64()? as u32,
                title: subject_line(i["title"].as_str().unwrap_or("")),
                state: i["state"].as_str().unwrap_or("").to_string(),
                draft: i["draft"].as_bool().unwrap_or(false),
                merged: i["pull_request"]["merged_at"].as_str().is_some(),
                html_url: i["html_url"].as_str().unwrap_or("").to_string(),
            })
        })
        .collect())
}

// ------------------------------------------------------------- resolving ----

/// A ref pinned to one commit, with what a person needs to decide whether
/// to build it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedRef {
    /// The link as written.
    pub requested: GitRef,
    /// Canonical repository (GitHub follows renames and transfers).
    pub owner: String,
    pub repo: String,
    /// The link's repository now lives under another name.
    pub redirected: bool,
    /// What `build_from_ref` fetches from.
    pub remote_url: String,
    /// What it fetches: a branch, `pull/<n>/head` or the commit.
    pub git_ref: String,
    /// The full commit the ref pointed at when resolved.
    pub sha: String,
    /// The repository is ggml-org/llama.cpp itself.
    pub is_upstream: bool,
    /// The repository is in upstream's fork network.
    pub is_fork_of_ggml: bool,
    /// Against upstream master; None outside the fork network.
    pub ahead_by: Option<u32>,
    pub behind_by: Option<u32>,
    pub merge_base: Option<String>,
    /// Subjects of the commits it adds, newest last, at most 10.
    pub head_subjects: Vec<String>,
    pub pr: Option<PrInfo>,
}

impl ResolvedRef {
    /// Where the ref can be viewed.
    pub fn html_url(&self) -> String {
        match &self.pr {
            Some(p) if !p.html_url.is_empty() => p.html_url.clone(),
            _ => format!("https://github.com/{}/{}/tree/{}", self.owner, self.repo, self.git_ref),
        }
    }

    /// The build input for this ref, labelled for the build list.
    pub fn source_ref(&self) -> crate::update::SourceRef {
        crate::update::SourceRef {
            remote_url: self.remote_url.clone(),
            git_ref: self.git_ref.clone(),
            sha: self.sha.clone(),
            label: crate::update::SourceRef::default_label(&self.remote_url, &self.git_ref),
        }
    }
}

/// The full commit of `git_ref` in `<owner>/<repo>`.
fn commit_sha(api: &Api, owner: &str, repo: &str, git_ref: &str) -> Result<String> {
    let body = api
        .get(&format!("/repos/{owner}/{repo}/commits/{}", pct(git_ref)), SHA_ONLY)?
        .ok_or_else(|| compat_err(format!("`{git_ref}` does not exist in github.com/{owner}/{repo}")))?;
    let sha = body.trim().to_ascii_lowercase();
    if sha.len() != 40 || !is_hex(&sha) {
        return Err(compat_err(format!("github.com/{owner}/{repo} answered `{}` for `{git_ref}`", sha.chars().take(60).collect::<String>())));
    }
    Ok(sha)
}

/// Upstream master against `head` (`<sha>` in upstream, or
/// `<owner>:<repo>:<sha>` in a fork), with the newest subjects.
fn compare(api: &Api, head: &str) -> Result<Option<CompareInfo>> {
    let path = |page: u32| format!("/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/compare/{UPSTREAM_BRANCH}...{head}?per_page=10&page={page}");
    let Some(first) = api.get(&path(1), JSON)? else { return Ok(None) };
    let mut info = parse_compare(&first)?;
    // The first page holds the oldest commits; the subjects that describe
    // a branch are its newest.
    if info.total_commits > 10 {
        let last = info.total_commits.div_ceil(10);
        if let Some(body) = api.get(&path(last), JSON)? {
            let tail = parse_compare(&body)?.subjects;
            info.subjects.extend(tail);
        }
    }
    let n = info.subjects.len();
    info.subjects.drain(..n.saturating_sub(10));
    Ok(Some(info))
}

/// Pin a llama.cpp link to a commit through the GitHub API: canonical
/// repository (following renames and transfers), the commit, whether it is
/// in upstream's fork network, how far it is ahead and behind upstream
/// master, and the pull request's state when it is one. Three or four API
/// requests, each cached.
pub fn resolve_ref(cfg: &Config, r: &GitRef) -> Result<ResolvedRef> {
    resolve_ref_with(&Api::new(cfg), r)
}

pub fn resolve_ref_with(api: &Api, r: &GitRef) -> Result<ResolvedRef> {
    if !valid_owner(&r.owner) || !valid_repo(&r.repo) {
        return Err(compat_err(format!("`{}/{}` is not a GitHub repository name", r.owner, r.repo)));
    }
    let meta_json = api
        .get(&format!("/repos/{}/{}", r.owner, r.repo), JSON)?
        .ok_or_else(|| compat_err(format!("github.com/{}/{} does not exist or is private", r.owner, r.repo)))?;
    let meta = parse_repo(&meta_json)?;
    let (owner, repo) = meta
        .full_name
        .split_once('/')
        .map(|(o, n)| (o.to_string(), n.to_string()))
        .filter(|(o, n)| valid_owner(o) && valid_repo(n))
        .ok_or_else(|| compat_err(format!("unexpected repository name `{}`", meta.full_name)))?;
    let redirected = !(owner.eq_ignore_ascii_case(&r.owner) && repo.eq_ignore_ascii_case(&r.repo));
    let upstream = format!("{UPSTREAM_OWNER}/{UPSTREAM_REPO}");
    let is_upstream = meta.full_name.eq_ignore_ascii_case(&upstream);
    let is_fork_of_ggml = !is_upstream
        && [&meta.source, &meta.parent].iter().any(|p| p.as_deref().is_some_and(|p| p.eq_ignore_ascii_case(&upstream)));
    let (git_ref, sha, pr) = match &r.kind {
        RefKind::Pull(n) => {
            let body = api
                .get(&format!("/repos/{owner}/{repo}/pulls/{n}"), JSON)?
                .ok_or_else(|| compat_err(format!("github.com/{owner}/{repo}/pull/{n} does not exist")))?;
            let pr = parse_pull(&body)?;
            let sha = pr.head_sha.to_ascii_lowercase();
            if sha.len() != 40 || !is_hex(&sha) {
                return Err(compat_err(format!("pull request {n} has no usable head commit")));
            }
            (format!("pull/{n}/head"), sha, Some(pr))
        }
        RefKind::Branch(b) => (b.clone(), commit_sha(api, &owner, &repo, b)?, None),
        RefKind::Commit(c) => {
            let full = commit_sha(api, &owner, &repo, c)?;
            (full.clone(), full, None)
        }
        RefKind::Repo => {
            let b = meta.default_branch.clone();
            let sha = commit_sha(api, &owner, &repo, &b)?;
            (b, sha, None)
        }
    };
    let cmp = if is_upstream {
        compare(api, &sha)?
    } else if is_fork_of_ggml {
        compare(api, &format!("{owner}:{repo}:{sha}"))?
    } else {
        None
    };
    Ok(ResolvedRef {
        requested: r.clone(),
        remote_url: format!("https://github.com/{owner}/{repo}"),
        owner,
        repo,
        redirected,
        git_ref,
        sha,
        is_upstream,
        is_fork_of_ggml,
        ahead_by: cmp.as_ref().map(|c| c.ahead_by),
        behind_by: cmp.as_ref().map(|c| c.behind_by),
        merge_base: cmp.as_ref().and_then(|c| c.merge_base.clone()),
        head_subjects: cmp.map(|c| c.subjects).unwrap_or_default(),
        pr,
    })
}

// ----------------------------------------------------------- upstream PRs ----

/// An upstream pull request that may add the model, checked against its
/// own source at its head commit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrCandidate {
    pub pr: PrInfo,
    pub support: Support,
}

/// At most this many pull requests are looked at in detail.
const MAX_PR_CANDIDATES: usize = 5;

/// Upstream pull requests that mention the architecture (and the model's
/// name, when given), each checked by reading its source at the head
/// commit. Closed unmerged ones are dropped; open ones come first. Two
/// searches plus one request per candidate.
pub fn find_upstream_pr(cfg: &Config, needs: &ModelNeeds, model_hint: Option<&str>) -> Result<Vec<PrCandidate>> {
    find_upstream_pr_with(&Api::new(cfg), needs, model_hint, &mut |sha: &str| {
        super::source_caps(UPSTREAM_OWNER, UPSTREAM_REPO, sha).map(|c| c.support(needs))
    })
}

pub fn find_upstream_pr_with(
    api: &Api,
    needs: &ModelNeeds,
    model_hint: Option<&str>,
    probe: &mut dyn FnMut(&str) -> Result<Support>,
) -> Result<Vec<PrCandidate>> {
    let mut terms: Vec<String> = vec![needs.arch.clone()];
    if let Some(h) = model_hint.map(str::trim).filter(|h| !h.is_empty() && !h.eq_ignore_ascii_case(&needs.arch)) {
        terms.push(h.to_string());
    }
    let mut hits: Vec<PrHit> = Vec::new();
    for term in terms {
        let clean: String = term.chars().filter(|c| *c != '"' && !c.is_control()).take(100).collect();
        let q = format!("is:pr repo:{UPSTREAM_OWNER}/{UPSTREAM_REPO} \"{clean}\"");
        let Some(body) = api.get(&format!("/search/issues?q={}&per_page=10", pct(&q)), JSON)? else { continue };
        for h in parse_search_prs(&body)? {
            if !hits.iter().any(|x| x.number == h.number) {
                hits.push(h);
            }
        }
    }
    hits.retain(|h| h.state == "open" || h.merged);
    // Open and ready, open drafts, then merged; newest number first within.
    hits.sort_by_key(|h| (h.state != "open", h.draft, std::cmp::Reverse(h.number)));
    hits.truncate(MAX_PR_CANDIDATES);
    let mut out = Vec::new();
    for h in hits {
        let Some(body) = api.get(&format!("/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/pulls/{}", h.number), JSON)? else {
            continue;
        };
        let pr = parse_pull(&body)?;
        // Upstream serves every commit of its fork network, so the head is
        // read through upstream even when it lives in a contributor's fork.
        let support = probe(&pr.head_sha).unwrap_or_else(|e| Support::Unknown(format!("could not read the source: {e}")));
        out.push(PrCandidate { pr, support });
    }
    Ok(out)
}
