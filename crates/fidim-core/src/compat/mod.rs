//! Build compatibility: can a llama.cpp build load a model, and if none of
//! the installed ones can, which build would.
//!
//! A model file needs three things from the build that loads it, all in
//! the GGUF header:
//! - its `general.architecture` in the build's `LLM_ARCH_NAMES` table
//!   (llama-server: "unknown model architecture");
//! - for BPE vocabularies, its `tokenizer.ggml.pre` among the pre-tokenizer
//!   names the vocab loader accepts ("unknown pre-tokenizer type");
//! - every tensor's ggml type below the build's `GGML_TYPE_COUNT` ("tensor
//!   has invalid ggml type"). Forks number their extra types differently,
//!   so an id is only meaningful against one source tree.
//!
//! Two ways to answer, both GPU-free:
//! - `probe_build` looks inside an installed build: the architecture and
//!   pre-tokenizer names are string literals in `llama.dll`, found as
//!   NUL-terminated strings. A heuristic (the linker may merge a name into
//!   the tail of a longer one, and the compiler may inline a short literal
//!   compare), so an absent pre-tokenizer only makes the answer Unknown;
//!   an absent architecture is a definite No because the name table must
//!   hold a pointer to it.
//! - `probe_source` reads the source at an exact commit from
//!   raw.githubusercontent.com: `src/llama-arch.cpp`, `src/llama-vocab.cpp`
//!   and `ggml/include/ggml.h` (older trees keep the tables in
//!   `src/llama.cpp` or `llama.cpp`, and ggml.h at the root). Cached on disk
//!   per commit, because a commit never changes.
//!
//! `github` finds candidate builds that are not releases (a fork named in
//! the model card, an upstream pull request) and `plan` orders every
//! candidate into one recommendation.

pub mod github;
pub mod plan;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::discovery::{self, Build, Channel, RUNNER_EXE};
use crate::gguf::GgufHeader;
use crate::profile::Engine;
use crate::{Error, Result};

pub use github::{card_refs, find_upstream_pr, resolve_ref, GitRef, PrCandidate, PrInfo, RefKind, ResolvedRef};
pub use plan::{gather_plan_inputs, plan_build, BuildPlan, CardCandidate, PlanAction, PlanInputs, PlanStep, ProbedBuild, UpstreamCandidate};

fn compat_err(msg: impl Into<String>) -> Error {
    Error::Update(msg.into())
}

/// Upstream llama.cpp, the base every fork is compared against.
pub const UPSTREAM_OWNER: &str = "ggml-org";
pub const UPSTREAM_REPO: &str = "llama.cpp";
/// Upstream's owner until 2025; github.com/ggerganov/llama.cpp redirects.
pub const UPSTREAM_OLD_OWNER: &str = "ggerganov";

/// Architectures upstream names in `LLM_ARCH_NAMES` but does not build a
/// graph for: the loader throws "unsupported model architecture" for them,
/// so a name match proves nothing.
pub const NAMED_NOT_IMPLEMENTED: &[&str] = &["gptj"];

/// The pre-tokenizer every build knows (the loader's own fallback). Its
/// compare is short enough to be compiled inline, so it is never looked
/// for in a DLL.
const ALWAYS_KNOWN_PRE: &str = "default";

// ------------------------------------------------------------------ needs ----

/// What a model needs from the build that loads it.
///
/// Stand-in for the wizard's own header type: `from_header` fills it from a
/// local header today; the wizard builds it the same way from a remote one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelNeeds {
    /// `general.architecture`, e.g. `k2-horizon`.
    pub arch: String,
    /// `tokenizer.ggml.pre`, only for BPE (`gpt2`) vocabularies: the loader
    /// ignores it for every other tokenizer model.
    pub tokenizer_pre: Option<String>,
    /// Largest ggml tensor type id among the model's tensors; None when the
    /// tensor table was not read.
    pub max_type_id: Option<u32>,
    pub engine: Engine,
}

impl ModelNeeds {
    pub fn new(arch: impl Into<String>, tokenizer_pre: Option<String>, max_type_id: Option<u32>, engine: Engine) -> Self {
        let tokenizer_pre = tokenizer_pre.map(|p| p.trim().to_string()).filter(|p| !p.is_empty());
        ModelNeeds { arch: arch.into().trim().to_string(), tokenizer_pre, max_type_id, engine }
    }

    /// From a parsed header. None when the header names no architecture.
    /// The pre-tokenizer counts only for a BPE (`gpt2`) vocabulary; an empty
    /// one only makes llama.cpp warn and use its default.
    pub fn from_header(h: &GgufHeader) -> Option<Self> {
        let arch = h.architecture.clone().filter(|a| !a.trim().is_empty())?;
        let meta_str = |k: &str| h.metadata.get(k).and_then(|v| v.as_str()).map(str::to_string);
        let bpe = meta_str("tokenizer.ggml.model").is_some_and(|m| m == "gpt2");
        let pre = if bpe { meta_str("tokenizer.ggml.pre") } else { None };
        Some(ModelNeeds::new(arch, pre, None, h.engine()))
    }
}

/// One thing a build lacks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum Missing {
    /// The model's architecture.
    Arch,
    /// A BPE pre-tokenizer name.
    TokenizerPre(String),
    /// A ggml tensor type id at or past the build's type count.
    TensorType(u32),
}

/// Whether a build (or a source tree) can load a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "support", content = "detail", rename_all = "lowercase")]
pub enum Support {
    Yes,
    No { missing: Vec<Missing> },
    /// Could not tell; the text says which part and why.
    Unknown(String),
}

impl Support {
    pub fn is_yes(&self) -> bool {
        matches!(self, Support::Yes)
    }
    pub fn is_no(&self) -> bool {
        matches!(self, Support::No { .. })
    }
}

/// Human text for what is missing: `architecture 'k2-horizon' and
/// pre-tokenizer 'k2-horizon'`.
pub fn describe_missing(needs: &ModelNeeds, missing: &[Missing]) -> String {
    let parts: Vec<String> = missing
        .iter()
        .map(|m| match m {
            Missing::Arch if NAMED_NOT_IMPLEMENTED.contains(&needs.arch.as_str()) => {
                format!("architecture '{}' (named but not implemented upstream)", needs.arch)
            }
            Missing::Arch => format!("architecture '{}'", needs.arch),
            Missing::TokenizerPre(p) => format!("pre-tokenizer '{p}'"),
            Missing::TensorType(t) => format!("ggml tensor type {t}"),
        })
        .collect();
    match parts.len() {
        0 => "nothing".into(),
        1 => parts[0].clone(),
        _ => format!("{} and {}", parts[..parts.len() - 1].join(", "), parts[parts.len() - 1]),
    }
}

// ------------------------------------------------------------ source caps ----

/// What a source tree can load, parsed from its tables.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SourceCaps {
    /// Every name in `LLM_ARCH_NAMES`, in table order.
    pub arches: Vec<String>,
    /// Every pre-tokenizer name the vocab loader compares against, sorted.
    pub tokenizer_pres: Vec<String>,
    /// `GGML_TYPE_COUNT`; None when ggml.h was not found.
    pub ggml_type_count: Option<u32>,
    /// The files the tables came from, for the record.
    #[serde(default)]
    pub files: Vec<String>,
}

impl SourceCaps {
    /// Exact answer from the tables.
    pub fn support(&self, needs: &ModelNeeds) -> Support {
        let mut missing = Vec::new();
        let named = self.arches.iter().any(|a| a == &needs.arch);
        if !named || NAMED_NOT_IMPLEMENTED.contains(&needs.arch.as_str()) {
            missing.push(Missing::Arch);
        }
        if let Some(pre) = &needs.tokenizer_pre {
            if pre != ALWAYS_KNOWN_PRE && !self.tokenizer_pres.iter().any(|p| p == pre) {
                missing.push(Missing::TokenizerPre(pre.clone()));
            }
        }
        let mut unknown = None;
        if let Some(t) = needs.max_type_id {
            match self.ggml_type_count {
                Some(count) if t >= count => missing.push(Missing::TensorType(t)),
                Some(_) => {}
                None => unknown = Some(format!("tensor type {t} not checked: the source's ggml.h was not found")),
            }
        }
        if !missing.is_empty() {
            Support::No { missing }
        } else if let Some(u) = unknown {
            Support::Unknown(u)
        } else {
            Support::Yes
        }
    }
}

/// Text between `start` and the first line that is exactly `};` after it.
fn table_block<'a>(src: &'a str, marker: &str) -> Option<&'a str> {
    let at = src.find(marker)?;
    let rest = &src[at..];
    let open = rest.find('{')?;
    let body = &rest[open + 1..];
    let end = body.find("\n};").unwrap_or(body.len());
    Some(&body[..end])
}

/// `LLM_ARCH_NAMES` entries (`{ LLM_ARCH_LLAMA, "llama" },`), in order.
/// Works on `src/llama-arch.cpp` and on the older `src/llama.cpp` layout.
pub fn parse_arch_names(src: &str) -> Vec<String> {
    let Some(block) = table_block(src, "LLM_ARCH_NAMES") else { return vec![] };
    let re = regex::Regex::new(r#"\{\s*LLM_ARCH_[A-Z0-9_]+\s*,\s*"([^"]+)"\s*\}"#).unwrap();
    let mut out: Vec<String> = Vec::new();
    for c in re.captures_iter(block) {
        let name = c[1].to_string();
        if !out.contains(&name) {
            out.push(name);
        }
    }
    out
}

/// Every `tokenizer_pre == "name"` the vocab loader tests, sorted. The
/// loader throws for a BPE pre-tokenizer it does not compare against.
pub fn parse_pre_names(src: &str) -> Vec<String> {
    let re = regex::Regex::new(r#"tokenizer_pre\s*==\s*"([^"]+)""#).unwrap();
    let set: std::collections::BTreeSet<String> = re.captures_iter(src).map(|c| c[1].to_string()).collect();
    set.into_iter().collect()
}

/// `GGML_TYPE_COUNT` from ggml.h: its explicit value, or its position in
/// `enum ggml_type` when a tree leaves it implicit.
pub fn parse_ggml_type_count(src: &str) -> Option<u32> {
    let explicit = regex::Regex::new(r"GGML_TYPE_COUNT\s*=\s*(\d+)").unwrap();
    if let Some(c) = explicit.captures(src) {
        return c[1].parse().ok();
    }
    let block = table_block(src, "enum ggml_type")?;
    let no_line_comments = regex::Regex::new(r"//[^\n]*").unwrap().replace_all(block, "");
    let clean = regex::Regex::new(r"(?s)/\*.*?\*/").unwrap().replace_all(&no_line_comments, "");
    let mut next: i64 = 0;
    for entry in clean.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let (name, value) = match entry.split_once('=') {
            Some((n, v)) => (n.trim(), v.trim().parse::<i64>().ok()),
            None => (entry, None),
        };
        let v = value.unwrap_or(next);
        if name == "GGML_TYPE_COUNT" {
            return u32::try_from(v).ok();
        }
        next = v + 1;
    }
    None
}

/// Candidate paths per table, newest layout first (the tables moved out of
/// `llama.cpp` into `src/llama.cpp` in mid-2024, then into their own files
/// in January 2025; ggml moved under `ggml/include` in mid-2024).
const ARCH_FILES: &[&str] = &["src/llama-arch.cpp", "src/llama.cpp", "llama.cpp"];
const PRE_FILES: &[&str] = &["src/llama-vocab.cpp", "src/llama.cpp", "llama.cpp"];
const GGML_H_FILES: &[&str] = &["ggml/include/ggml.h", "ggml.h"];

/// Parse a tree's tables through `fetch(path) -> Ok(None)` for a missing
/// file. Each path is fetched at most once; a later layout is tried only
/// when an earlier file is missing or does not hold the table.
pub fn caps_from_files(fetch: &mut dyn FnMut(&str) -> Result<Option<String>>) -> Result<SourceCaps> {
    let mut seen: HashMap<&'static str, Option<String>> = HashMap::new();
    let mut get = |path: &'static str, fetch: &mut dyn FnMut(&str) -> Result<Option<String>>| -> Result<Option<String>> {
        if let Some(v) = seen.get(path) {
            return Ok(v.clone());
        }
        let v = fetch(path)?;
        seen.insert(path, v.clone());
        Ok(v)
    };
    let mut caps = SourceCaps::default();
    for path in ARCH_FILES {
        if let Some(text) = get(path, fetch)? {
            let names = parse_arch_names(&text);
            if !names.is_empty() {
                caps.arches = names;
                caps.files.push((*path).to_string());
                break;
            }
        }
    }
    if caps.arches.is_empty() {
        return Err(compat_err("no LLM_ARCH_NAMES table found (is this a llama.cpp tree?)"));
    }
    for path in PRE_FILES {
        if let Some(text) = get(path, fetch)? {
            let names = parse_pre_names(&text);
            if !names.is_empty() {
                caps.tokenizer_pres = names;
                if !caps.files.iter().any(|f| f == path) {
                    caps.files.push((*path).to_string());
                }
                break;
            }
        }
    }
    for path in GGML_H_FILES {
        if let Some(text) = get(path, fetch)? {
            if let Some(n) = parse_ggml_type_count(&text) {
                caps.ggml_type_count = Some(n);
                caps.files.push((*path).to_string());
                break;
            }
        }
    }
    Ok(caps)
}

/// Tables of a checked-out tree on disk (a build's worktree).
pub fn caps_from_dir(root: &Path) -> Result<SourceCaps> {
    caps_from_files(&mut |rel: &str| {
        let p = root.join(rel);
        match std::fs::read(&p) {
            Ok(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::io(p, e)),
        }
    })
}

/// GitHub owner / repository names: `[A-Za-z0-9-]` and `[A-Za-z0-9._-]`.
pub(crate) fn valid_owner(s: &str) -> bool {
    !s.is_empty() && s.len() <= 39 && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') && !s.starts_with('-')
}

pub(crate) fn valid_repo(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 100
        && s != "."
        && s != ".."
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

pub(crate) fn is_hex(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// A ref whose tree never changes: a commit (7-40 hex digits) or an
/// upstream-style release tag (`b<n>`). Only those are cached.
fn immutable_ref(r: &str) -> bool {
    (r.len() >= 7 && r.len() <= 40 && is_hex(r)) || r.strip_prefix('b').is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// A ref usable in a raw.githubusercontent.com path.
fn valid_raw_ref(r: &str) -> bool {
    !r.is_empty()
        && r.len() <= 200
        && !r.starts_with('/')
        && !r.contains("..")
        && r.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/'))
}

/// `<config dir>/cache/src`: one JSON per `<owner>/<repo>/<commit>`.
pub fn source_cache_root() -> PathBuf {
    Config::config_dir().join("cache").join("src")
}

fn cache_file(root: &Path, owner: &str, repo: &str, sha: &str) -> PathBuf {
    root.join(owner.to_ascii_lowercase()).join(repo.to_ascii_lowercase()).join(format!("{}.json", sha.to_ascii_lowercase()))
}

/// Cached tables for a commit, by exact name or by prefix (`--version`
/// prints a 9-digit commit). Never touches the network.
pub fn cached_source_caps_in(root: &Path, owner: &str, repo: &str, sha: &str) -> Option<SourceCaps> {
    if !valid_owner(owner) || !valid_repo(repo) || !immutable_ref(sha) {
        return None;
    }
    let exact = cache_file(root, owner, repo, sha);
    let read = |p: &Path| std::fs::read_to_string(p).ok().and_then(|t| serde_json::from_str::<SourceCaps>(&t).ok());
    if let Some(c) = read(&exact) {
        return Some(c);
    }
    if !is_hex(sha) {
        return None;
    }
    let want = sha.to_ascii_lowercase();
    let dir = exact.parent()?;
    let mut hits = std::fs::read_dir(dir).ok()?.flatten().filter_map(|e| {
        let name = e.file_name().to_string_lossy().into_owned();
        let stem = name.strip_suffix(".json")?.to_string();
        (is_hex(&stem) && (stem.starts_with(&want) || want.starts_with(&stem))).then(|| e.path())
    });
    let first = hits.next()?;
    // Two cached commits sharing the prefix: ambiguous, answer nothing.
    if hits.next().is_some() {
        return None;
    }
    read(&first)
}

pub fn cached_source_caps(owner: &str, repo: &str, sha: &str) -> Option<SourceCaps> {
    cached_source_caps_in(&source_cache_root(), owner, repo, sha)
}

/// The tables of `<owner>/<repo>` at `sha` (a commit, or an upstream `b<n>`
/// tag), from the disk cache or raw.githubusercontent.com. `fetch` maps a
/// URL to its body (None = 404) so tests run offline.
pub fn source_caps_with(
    cache_root: &Path,
    owner: &str,
    repo: &str,
    sha: &str,
    fetch: &mut dyn FnMut(&str) -> Result<Option<String>>,
) -> Result<SourceCaps> {
    if !valid_owner(owner) || !valid_repo(repo) {
        return Err(compat_err(format!("`{owner}/{repo}` is not a GitHub repository name")));
    }
    if !valid_raw_ref(sha) {
        return Err(compat_err(format!("`{sha}` is not a commit or tag")));
    }
    let cacheable = immutable_ref(sha);
    if cacheable {
        let exact = cache_file(cache_root, owner, repo, sha);
        if let Some(c) = std::fs::read_to_string(&exact).ok().and_then(|t| serde_json::from_str(&t).ok()) {
            return Ok(c);
        }
    }
    let base = format!("https://raw.githubusercontent.com/{owner}/{repo}/{sha}");
    let caps = caps_from_files(&mut |path: &str| fetch(&format!("{base}/{path}")))
        .map_err(|e| compat_err(format!("{owner}/{repo}@{sha}: {e}")))?;
    if cacheable {
        let p = cache_file(cache_root, owner, repo, sha);
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        // A failed cache write only costs a refetch next time.
        let _ = std::fs::write(&p, serde_json::to_string_pretty(&caps)?);
    }
    Ok(caps)
}

pub fn source_caps(owner: &str, repo: &str, sha: &str) -> Result<SourceCaps> {
    source_caps_with(&source_cache_root(), owner, repo, sha, &mut github::fetch_raw)
}

/// Can the source of `<owner>/<repo>` at `sha` load the model? Reads three
/// files at that commit (cached per commit), no API quota.
pub fn probe_source(owner: &str, repo: &str, sha: &str, needs: &ModelNeeds) -> Result<Support> {
    Ok(source_caps(owner, repo, sha)?.support(needs))
}

// ------------------------------------------------------------ build probe ----

/// NUL-terminated printable strings of a binary, each cut to its last 64
/// characters (enough for any architecture or pre-tokenizer name).
#[derive(Debug, Default)]
struct StringTable {
    strings: HashSet<String>,
}

impl StringTable {
    fn from_bytes(bytes: &[u8]) -> Self {
        const KEEP: usize = 64;
        let mut strings = HashSet::new();
        let mut start = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == 0 {
                if i > start {
                    // Walk back over printable ASCII from the terminator.
                    let mut s = i;
                    while s > start && i - s < KEEP && (0x20..0x7f).contains(&bytes[s - 1]) {
                        s -= 1;
                    }
                    if i - s >= 2 {
                        strings.insert(String::from_utf8_lossy(&bytes[s..i]).into_owned());
                    }
                }
                start = i + 1;
            }
        }
        StringTable { strings }
    }

    /// The literal is there on its own, or as the tail of a longer string
    /// (the linker merges `"bert"` into `"eurobert"`).
    fn has(&self, name: &str) -> bool {
        self.strings.contains(name) || self.strings.iter().any(|s| s.ends_with(name))
    }
}

type TableKey = (PathBuf, u64, u64);

fn table_cache() -> &'static Mutex<HashMap<TableKey, Arc<StringTable>>> {
    static CACHE: OnceLock<Mutex<HashMap<TableKey, Arc<StringTable>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Files larger than this are not scanned (a static build with GPU kernels
/// linked in); the answer is then Unknown.
const MAX_SCAN_BYTES: u64 = 512 << 20;

/// The string table of `path`, cached by path, modified time and size, so
/// the profile editor's live pre-flight reads each DLL once.
fn string_table(path: &Path) -> std::io::Result<Arc<StringTable>> {
    let meta = std::fs::metadata(path)?;
    let mtime = meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as u64);
    let key = (path.to_path_buf(), mtime, meta.len());
    if let Some(t) = table_cache().lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(t.clone());
    }
    if meta.len() > MAX_SCAN_BYTES {
        return Err(std::io::Error::other(format!("{} MB is too large to scan", meta.len() >> 20)));
    }
    let table = Arc::new(StringTable::from_bytes(&std::fs::read(path)?));
    let mut cache = table_cache().lock().unwrap_or_else(|p| p.into_inner());
    // Stale entries for the same path (a rebuilt DLL) go.
    cache.retain(|k, _| k.0 != key.0);
    cache.insert(key, table.clone());
    Ok(table)
}

/// The binary that holds the model tables: `llama.dll` in a shared build,
/// the server executable in a static one.
fn table_binary(bin: &Path) -> Option<PathBuf> {
    ["llama.dll", "libllama.dll", "llama-server.exe"].iter().map(|n| bin.join(n)).find(|p| p.is_file())
}

/// Only what `probe_build` needs from a manifest: exact tables recorded by
/// a source build.
#[derive(Deserialize)]
struct ManifestCaps {
    #[serde(default)]
    caps: Option<SourceCaps>,
}

fn manifest_caps(dir: &Path) -> Option<SourceCaps> {
    let text = std::fs::read_to_string(crate::update::manifest_path(dir)).ok()?;
    serde_json::from_str::<ManifestCaps>(&text).ok()?.caps
}

/// `(owner, repo, commit)` a build's source can be looked up by: the
/// manifest of a git build, or upstream at the commit `--version` printed.
/// Unsloth builds have none: their release tag is not the built source.
pub(crate) fn source_of(dir: &Path, commit: Option<&str>) -> Option<(String, String, String)> {
    let meta = discovery::read_build_meta(dir);
    match meta.channel {
        Channel::Git => {
            let g = meta.git?;
            let (owner, repo) = github::owner_repo_from_url(&g.remote)?;
            Some((owner, repo, g.commit))
        }
        Channel::Upstream => {
            let c = commit?.trim();
            (c.len() >= 7 && is_hex(c)).then(|| (UPSTREAM_OWNER.to_string(), UPSTREAM_REPO.to_string(), c.to_string()))
        }
        _ => None,
    }
}

/// Can the installed build at `dir` load the model? `commit` is what its
/// `--version` printed (used only to find cached source tables for the
/// tensor-type check). Offline; see the module notes for how far the
/// answer can be trusted.
pub fn probe_build_at(dir: &Path, commit: Option<&str>, needs: &ModelNeeds) -> Support {
    probe_build_in(dir, commit, needs, &source_cache_root())
}

pub fn probe_build(build: &Build, needs: &ModelNeeds) -> Support {
    probe_build_at(&build.path, build.commit.as_deref(), needs)
}

fn probe_build_in(dir: &Path, commit: Option<&str>, needs: &ModelNeeds, cache_root: &Path) -> Support {
    let bin = dir.join("bin");
    if needs.engine.is_diffusion() && !bin.join(RUNNER_EXE).is_file() {
        return Support::No { missing: vec![Missing::Arch] };
    }
    // A source build recorded its exact tables.
    if let Some(caps) = manifest_caps(dir) {
        return caps.support(needs);
    }
    let Some(binary) = table_binary(&bin) else {
        return Support::Unknown(format!("{} has no llama.dll or llama-server.exe to inspect", bin.display()));
    };
    let table = match string_table(&binary) {
        Ok(t) => t,
        Err(e) => return Support::Unknown(format!("could not read {}: {e}", binary.display())),
    };
    let mut missing = Vec::new();
    if !table.has(&needs.arch) || NAMED_NOT_IMPLEMENTED.contains(&needs.arch.as_str()) {
        missing.push(Missing::Arch);
    }
    let pre_absent = needs
        .tokenizer_pre
        .as_ref()
        .filter(|p| p.as_str() != ALWAYS_KNOWN_PRE && !table.has(p))
        .cloned();
    let binary_name = binary.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut unknown: Vec<String> = Vec::new();
    if let Some(t) = needs.max_type_id {
        let count = source_of(dir, commit)
            .and_then(|(o, r, c)| cached_source_caps_in(cache_root, &o, &r, &c))
            .and_then(|c| c.ggml_type_count);
        match count {
            Some(n) if t >= n => missing.push(Missing::TensorType(t)),
            Some(_) => {}
            None => unknown.push(format!(
                "tensor type {t} not checked: this build's ggml type table is not known offline"
            )),
        }
    }
    if !missing.is_empty() {
        // Without the architecture, a pre-tokenizer that is not there either
        // is reported with it: a build from the same era lacks both.
        if let (Some(p), Some(Missing::Arch)) = (pre_absent, missing.first()) {
            missing.insert(1, Missing::TokenizerPre(p));
        }
        return Support::No { missing };
    }
    if let Some(p) = pre_absent {
        unknown.insert(
            0,
            format!(
                "pre-tokenizer '{p}' was not found in {binary_name}; the build may not know it (llama-server \
                 would stop with \"unknown pre-tokenizer type\")"
            ),
        );
    }
    if unknown.is_empty() {
        Support::Yes
    } else {
        Support::Unknown(unknown.join("; "))
    }
}

/// Probe every build, keyed by path, for a plan or a table.
pub fn probe_builds(builds: &[Build], needs: &ModelNeeds) -> BTreeMap<PathBuf, Support> {
    builds.iter().map(|b| (b.path.clone(), probe_build(b, needs))).collect()
}

#[cfg(test)]
mod tests;
