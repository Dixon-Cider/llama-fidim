//! Discovery of builds and models by filesystem scan (spec R-01, R-02).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::gguf::{self, GgufHeader};
use crate::{Error, Result};

// ---------------------------------------------------------------- builds ----

/// Where a build came from. Only FIDIM's own manifest sets it: a build is
/// never classed by the files it contains, because upstream may ship the
/// diffusion runner too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    /// ggml-org/llama.cpp, or any build FIDIM did not install.
    #[default]
    Upstream,
    /// unslothai/llama.cpp's fork, which carries the DiffusionGemma runner.
    Unsloth,
    /// Compiled by FIDIM from a git ref that is not a release: a fork's
    /// branch, an upstream pull request or any commit (`update::build_from_ref`).
    /// Its `--version` build number is that ref's own history count, so it
    /// never ranks against upstream releases.
    Git,
    /// A channel a newer FIDIM wrote. Never treated as upstream.
    #[serde(other)]
    Other,
}

impl Channel {
    /// Short name for tables (`upstream`, `unsloth`, `git`, `other`).
    pub fn as_str(self) -> &'static str {
        match self {
            Channel::Upstream => "upstream",
            Channel::Unsloth => "unsloth",
            Channel::Git => "git",
            Channel::Other => "other",
        }
    }
}

/// The git ref a `Channel::Git` build was compiled from, as its manifest
/// records it. `--version` of such a build reports the ref's own history
/// count as a build number; this is what names it instead.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GitSource {
    /// URL the ref was fetched from, e.g. `https://github.com/ifm-ai/llama.cpp`.
    #[serde(default)]
    pub remote: String,
    /// What was fetched: the full name of a branch, tag or ref
    /// (`refs/heads/model/K2Horizon`, `refs/pull/<n>/head`), or the commit.
    /// Builds before full names were recorded hold the short name.
    #[serde(default)]
    pub git_ref: String,
    /// The full commit that was built (checked against what the fetch got).
    #[serde(default)]
    pub commit: String,
    /// Short human name, e.g. `ifm-ai K2Horizon fork`.
    #[serde(default)]
    pub label: String,
}

impl GitSource {
    /// `ifm-ai K2Horizon fork @42adf01`; without a label, the ref's short
    /// name (`model/K2Horizon @42adf01`).
    pub fn display(&self) -> String {
        let short: String = self.commit.chars().take(7).collect();
        let r = self.git_ref.as_str();
        let short_ref = r.strip_prefix("refs/heads/").or_else(|| r.strip_prefix("refs/tags/")).unwrap_or(r);
        let label = if self.label.trim().is_empty() { short_ref } else { self.label.trim() };
        if short.is_empty() { label.to_string() } else { format!("{label} @{short}") }
    }
}

/// serde `deserialize_with` for a manifest's `git`: a malformed block reads
/// as none instead of failing the whole manifest.
pub fn lenient_git<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<GitSource>, D::Error> {
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(v.and_then(|v| serde_json::from_value(v).ok()))
}

/// The DiffusionGemma runner, beside `llama-server.exe` in a build's `bin`.
pub const RUNNER_EXE: &str = "llama-diffusion-gemma-visual-server.exe";

/// Runner features a patched build declares in its manifest (`patch.features`).
/// The estimator and the checks branch on these, never on `patch.name`.
pub mod dg_feature {
    /// The prompt-KV store is F16 with flash attention on or off (stock: F32
    /// unless FA is on).
    pub const PKV_F16: &str = "dg-pkv-f16";
    /// Sliding-window layers keep a ring of `n_swa-1 + n_ubatch` store rows;
    /// only full-attention layers keep every prompt position.
    pub const SWA_RING: &str = "dg-swa-ring";
    /// FA=1 runs the 512-dim heads on the GPU: K/V are padded to the kernel's
    /// 256-key stride (stock: those layers fall back to the CPU).
    pub const FA_PAD: &str = "dg-fa-pad";
    /// Under FA the auto-sizer gates on `llama_diffusion_fa_turn_bytes`
    /// instead of the N² score tensor, up to min(n_ctx_train, 65536).
    pub const FA_TURN_SIZING: &str = "dg-fa-turn-sizing";
    /// A failed denoise step ends block 0 with `ERR gen` instead of
    /// committing a stale canvas.
    pub const STEP_FAIL_ERR: &str = "dg-step-fail-err";
}

/// A build FIDIM did not download as-is: the manifest's `patch` block, set by
/// whoever assembled it (e.g. the DiffusionGemma memory/context patch over
/// Unsloth's source). Every field is optional so an older or hand-written
/// block still parses.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BuildPatch {
    /// Short label the UI appends to the build, e.g. `dgpatch`.
    #[serde(default)]
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_commit: Option<String>,
    /// See `dg_feature`.
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_sha256: Option<String>,
}

impl BuildPatch {
    pub fn has(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }
    /// What the UI and CLI call it: the name, or `patched` when it has none.
    pub fn label(&self) -> &str {
        if self.name.is_empty() { "patched" } else { &self.name }
    }
}

/// serde `deserialize_with` for a manifest's `patch`: a malformed block reads
/// as no patch instead of failing the whole manifest.
pub fn lenient_patch<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<BuildPatch>, D::Error> {
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(v.and_then(|v| serde_json::from_value(v).ok()))
}

/// What FIDIM's install manifest says about a build directory.
#[derive(Debug, Clone, Default)]
pub struct BuildMeta {
    pub channel: Channel,
    /// The build ships its own ROCm DLLs and must run with no PATH prefix:
    /// a runtime on PATH would supply whatever DLL the bundle lacks and mix
    /// ROCm versions.
    pub bundled_runtime: bool,
    pub release_tag: Option<String>,
    pub patch: Option<BuildPatch>,
    /// Set on `Channel::Git` builds.
    pub git: Option<GitSource>,
}

impl BuildMeta {
    /// Whether the build's patch declares this `dg_feature`.
    pub fn has_feature(&self, feature: &str) -> bool {
        self.patch.as_ref().is_some_and(|p| p.has(feature))
    }
}

/// Only the manifest fields discovery needs, all optional, so manifests from
/// every FIDIM version parse (older ones lack the channel fields entirely).
#[derive(Deserialize)]
struct ManifestLite {
    #[serde(default)]
    channel: Option<Channel>,
    #[serde(default)]
    bundled_runtime: bool,
    #[serde(default)]
    release_tag: Option<String>,
    /// Kept loose: a malformed `patch` must not turn the whole build into a
    /// plain upstream one; it just reads as unpatched.
    #[serde(default, deserialize_with = "lenient_patch")]
    patch: Option<BuildPatch>,
    #[serde(default, deserialize_with = "lenient_git")]
    git: Option<GitSource>,
}

/// Read a build's channel metadata. No manifest, or one that does not parse,
/// means a plain upstream build with no bundled runtime.
pub fn read_build_meta(dir: &Path) -> BuildMeta {
    let Ok(text) = std::fs::read_to_string(crate::update::manifest_path(dir)) else {
        return BuildMeta::default();
    };
    match serde_json::from_str::<ManifestLite>(&text) {
        Ok(m) => BuildMeta {
            channel: m.channel.unwrap_or_default(),
            bundled_runtime: m.bundled_runtime,
            release_tag: m.release_tag,
            patch: m.patch,
            git: m.git,
        },
        Err(_) => BuildMeta::default(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Build {
    /// Build directory (contains `bin/llama-server.exe`).
    pub path: PathBuf,
    /// Directory name, e.g. `build-hip-vision` — the human feature tag.
    pub tag: String,
    pub server_exe: PathBuf,
    /// Upstream version, e.g. `b9817`, from `--version`. None if the binary
    /// failed to run (surfaced, not hidden — a broken build is a finding).
    pub version: Option<String>,
    pub commit: Option<String>,
    pub version_error: Option<String>,
    /// From the manifest only (see `Channel`).
    pub channel: Channel,
    pub bundled_runtime: bool,
    /// The release this build was installed from, e.g. `b11027-mix-3e83366`.
    pub release_tag: Option<String>,
    /// From the manifest only. A patched build shares its base's version,
    /// commit and release tag; this is what tells them apart.
    pub patch: Option<BuildPatch>,
    /// `bin/<RUNNER_EXE>` when present: this build can run diffusion profiles.
    pub runner_exe: Option<PathBuf>,
    /// From the manifest only: the git ref a `Channel::Git` build was
    /// compiled from. Its `display()` is the build's name in tables.
    pub git: Option<GitSource>,
}

/// Scan each root and its immediate subdirectories for `bin/llama-server.exe`.
/// Subdirectories named `.…` are skipped: installs stage into `.fidim-tmp-*`
/// and a half-extracted build must never be offered.
pub fn scan_builds(roots: &[PathBuf], rocm_bin: Option<&Path>) -> Vec<Build> {
    let mut found = Vec::new();
    for root in roots {
        let mut candidates = vec![root.clone()];
        if let Ok(entries) = std::fs::read_dir(root) {
            candidates.extend(
                entries
                    .flatten()
                    .filter(|e| !e.file_name().to_string_lossy().starts_with('.'))
                    .map(|e| e.path())
                    .filter(|p| p.is_dir()),
            );
        }
        for dir in candidates {
            let exe = dir.join("bin").join("llama-server.exe");
            if exe.is_file() {
                found.push(probe_build(&dir, &exe, rocm_bin));
            }
        }
    }
    found.sort_by(|a, b| a.tag.cmp(&b.tag));
    found
}

fn probe_build(dir: &Path, exe: &Path, rocm_bin: Option<&Path>) -> Build {
    let tag = dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let meta = read_build_meta(dir);
    let runner = exe.with_file_name(RUNNER_EXE);
    let mut build = Build {
        path: dir.to_path_buf(),
        tag,
        server_exe: exe.to_path_buf(),
        version: None,
        commit: None,
        version_error: None,
        channel: meta.channel,
        bundled_runtime: meta.bundled_runtime,
        release_tag: meta.release_tag,
        patch: meta.patch,
        runner_exe: runner.is_file().then_some(runner),
        git: meta.git,
    };
    // A bundled build is probed exactly as it launches: with nothing on PATH.
    let rocm_bin = if build.bundled_runtime { None } else { rocm_bin };
    match run_version(exe, rocm_bin) {
        Ok(text) => match parse_version_output(&text) {
            Some((ver, commit)) => {
                build.version = Some(ver);
                build.commit = Some(commit);
            }
            None => build.version_error = Some(format!("unrecognised --version output: {text:?}")),
        },
        Err(e) => build.version_error = Some(e.to_string()),
    }
    build
}

/// Invoke `llama-server --version` with the HIP runtime on PATH.
/// llama.cpp prints version info to stderr; capture both streams.
fn run_version(exe: &Path, rocm_bin: Option<&Path>) -> Result<String> {
    let mut cmd = Command::new(exe);
    cmd.arg("--version");
    if let Some(rocm) = rocm_bin {
        let path = std::env::var_os("PATH").unwrap_or_default();
        let mut joined = rocm.as_os_str().to_owned();
        joined.push(";");
        joined.push(path);
        cmd.env("PATH", joined);
    }
    crate::launch::hide_console(&mut cmd);
    let out = cmd.output().map_err(|e| Error::BuildBinary {
        path: exe.to_path_buf(),
        detail: e.to_string(),
    })?;
    let mut text = String::from_utf8_lossy(&out.stderr).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stdout));
    Ok(text)
}

/// Parse the two upstream `--version` formats into (`b<n>`, commit):
/// - pre-b10000: `version: 9817 (5397c3619)`
/// - semver era: `version: 0.3.0-dev (build 10770, commit 9cc33944f)`
pub fn parse_version_output(text: &str) -> Option<(String, String)> {
    let old = regex::Regex::new(r"version:\s*(\d+)\s*\(([0-9a-fA-F]+)\)").unwrap();
    if let Some(caps) = old.captures(text) {
        return Some((format!("b{}", &caps[1]), caps[2].to_string()));
    }
    let new = regex::Regex::new(r"build\s+(\d+),\s*commit\s+([0-9a-fA-F]+)").unwrap();
    let caps = new.captures(text)?;
    Some((format!("b{}", &caps[1]), caps[2].to_string()))
}

// ---------------------------------------------------------------- models ----

#[derive(Debug, Clone, Serialize)]
pub struct Model {
    pub path: PathBuf,
    pub file_size: u64,
    /// Modified time (unix seconds) — feeds the cold-cache heuristic (R-08).
    pub modified_unix: Option<u64>,
    pub header: Option<GgufHeader>,
    pub header_error: Option<String>,
    /// The engine this model needs, from its header (llama-server when the
    /// header is unreadable). The profile editor switches engine on it.
    pub engine: crate::profile::Engine,
    /// Paired multimodal projectors found beside the model (R-02).
    pub mmproj_candidates: Vec<PathBuf>,
    /// Paired speculative-decoding draft models (R-02): `MTP/` subdirectory
    /// contents or `*mtp*.gguf` siblings.
    pub draft_candidates: Vec<PathBuf>,
}

/// Every auxiliary file seen under the roots, wherever it sits. The picker
/// offers these as "elsewhere" so a draft downloaded into its own folder
/// (a different publisher's repo, say) can still be paired by hand.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AuxFiles {
    pub drafts: Vec<PathBuf>,
    pub mmproj: Vec<PathBuf>,
}

/// Recursively scan model roots for `.gguf` files, pairing auxiliary files.
///
/// Pairing rules (from how the real model tree is laid out):
/// - a file named `mmproj-*.gguf` is a projector for the models in its
///   directory, not a model itself;
/// - files under an `MTP/` subdirectory, or with `mtp` in the stem, are draft
///   models for the models in the parent directory;
/// - a draft or projector anywhere under the roots whose name carries the
///   same model name as a model (quant suffix aside) is offered to it too,
///   after the siblings. Publishers ship MTP heads in their own repos.
pub fn scan_models(roots: &[PathBuf]) -> Vec<Model> {
    scan_models_and_aux(roots).0
}

pub fn scan_models_and_aux(roots: &[PathBuf]) -> (Vec<Model>, AuxFiles) {
    let mut models = Vec::new();
    let mut all = AuxFiles::default();
    for root in roots {
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else { continue };
            let mut ggufs: Vec<PathBuf> = Vec::new();
            let mut subdirs: Vec<PathBuf> = Vec::new();
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    subdirs.push(p);
                } else if is_gguf(&p) {
                    ggufs.push(p);
                }
            }
            let (aux, main): (Vec<_>, Vec<_>) = ggufs.into_iter().partition(|p| is_aux(p));
            let mmproj: Vec<PathBuf> =
                aux.iter().filter(|p| stem_lower(p).starts_with("mmproj")).cloned().collect();
            let mut drafts: Vec<PathBuf> =
                aux.iter().filter(|p| stem_lower(p).contains("mtp")).cloned().collect();
            // Draft models shipped in an MTP/ subdirectory (the case that went
            // unnoticed for weeks — spec R-02).
            for sub in &subdirs {
                if sub.file_name().is_some_and(|n| n.eq_ignore_ascii_case("MTP")) {
                    if let Ok(mtp_entries) = std::fs::read_dir(sub) {
                        drafts.extend(mtp_entries.flatten().map(|e| e.path()).filter(|p| is_gguf(p)));
                    }
                }
            }
            all.mmproj.extend(mmproj.iter().cloned());
            all.drafts.extend(drafts.iter().cloned());
            for path in main {
                models.push(load_model(path, &mmproj, &drafts));
            }
            // Recurse, but MTP/ contents are drafts, not standalone models.
            stack.extend(
                subdirs
                    .into_iter()
                    .filter(|s| !s.file_name().is_some_and(|n| n.eq_ignore_ascii_case("MTP"))),
            );
        }
    }
    // Second pass: name-matched drafts and projectors from anywhere.
    for m in &mut models {
        let stem = stem_lower(&m.path);
        for d in &all.drafts {
            if !m.draft_candidates.contains(d) && names_match(&stem, &stem_lower(d)) {
                m.draft_candidates.push(d.clone());
            }
        }
        for p in &all.mmproj {
            if !m.mmproj_candidates.contains(p) && names_match(&stem, &stem_lower(p)) {
                m.mmproj_candidates.push(p.clone());
            }
        }
    }
    models.sort_by(|a, b| a.path.cmp(&b.path));
    all.drafts.sort();
    all.drafts.dedup();
    all.mmproj.sort();
    all.mmproj.dedup();
    (models, all)
}

/// `gemma-4-e4b-it-q8_0` and `mtp-gemma-4-e4b-it-q8_0` name the same model:
/// drop the `mtp-`/`mmproj-` prefix and any trailing quant tokens, then the
/// remainders must be equal. A bare `mmproj-f32` names nothing and matches
/// nothing (the sibling rule covers it).
pub fn names_match(model_stem: &str, aux_stem: &str) -> bool {
    let a = base_name(model_stem);
    let mut b = aux_stem.to_string();
    for prefix in ["mtp-", "mtp_", "mmproj-", "mmproj_", "draft-"] {
        if let Some(rest) = b.strip_prefix(prefix) {
            b = rest.to_string();
            break;
        }
    }
    let b = base_name(&b);
    !a.is_empty() && a.len() >= 6 && a == b
}

fn base_name(stem: &str) -> String {
    let mut s = stem.to_lowercase().replace('_', "-");
    loop {
        let Some(i) = s.rfind('-') else { break };
        let tail = &s[i + 1..];
        let quantish = tail == "ud"
            || tail == "bf16"
            || tail == "f16"
            || tail == "f32"
            || tail == "qat"
            || ((tail.starts_with('q') || tail.starts_with("iq")) && tail.chars().any(|c| c.is_ascii_digit()))
            || tail.chars().all(|c| c.is_ascii_digit())
            || ["xl", "xs", "s", "m", "l", "k", "nl", "xxs"].contains(&tail);
        if quantish && i > 0 {
            s.truncate(i);
        } else {
            break;
        }
    }
    s
}

fn is_gguf(p: &Path) -> bool {
    p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
}

fn stem_lower(p: &Path) -> String {
    p.file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default()
}

fn is_aux(p: &Path) -> bool {
    let stem = stem_lower(p);
    stem.starts_with("mmproj") || stem.contains("mtp")
}

fn load_model(path: PathBuf, mmproj: &[PathBuf], drafts: &[PathBuf]) -> Model {
    let meta = std::fs::metadata(&path).ok();
    let file_size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
    let modified_unix = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs());
    let (header, header_error) = match gguf::read_header(&path) {
        Ok(h) => (Some(h), None),
        Err(e) => (None, Some(e.to_string())),
    };
    Model {
        path,
        file_size,
        modified_unix,
        engine: header.as_ref().map(|h| h.engine()).unwrap_or_default(),
        header,
        header_error,
        mmproj_candidates: mmproj.to_vec(),
        draft_candidates: drafts.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parse_handles_semver_era_output() {
        let text = "version: 0.3.0-dev (build 10770, commit 9cc33944f)\nbuilt with Clang 20.1.8 for Windows x86_64\n";
        assert_eq!(parse_version_output(text), Some(("b10770".into(), "9cc33944f".into())));
    }

    #[test]
    fn version_parse_matches_real_output() {
        // Captured from build-hip-vision on the target machine.
        let text = "version: 9817 (5397c3619)\nbuilt with Clang 21.0.0 for Windows AMD64\n";
        assert_eq!(
            parse_version_output(text),
            Some(("b9817".into(), "5397c3619".into()))
        );
    }

    #[test]
    fn drafts_pair_by_model_name_across_folders() {
        assert!(names_match("gemma-4-e4b-it-q8_0", "mtp-gemma-4-e4b-it-q8_0"));
        assert!(names_match("gemma-4-26b-a4b-it-qat-ud-q4_k_xl", "mtp-gemma-4-26b-a4b-it-q8_0"));
        assert!(names_match("gemma-4-e4b-it-q8_0", "mmproj-gemma-4-e4b-it-bf16"));
        assert!(!names_match("gemma-4-e4b-it-q8_0", "mtp-gemma-4-26b-a4b-it-q8_0"));
        assert!(!names_match("gemma-4-e4b-it-q8_0", "mmproj-f32"));
        assert!(!names_match("qwen3.8-27b-ud-q4_k_xl", "mtp-gemma-4-e4b-it-q8_0"));
    }

    #[test]
    fn pairing_separates_mmproj_and_mtp_from_models() {
        let dir = std::env::temp_dir().join(format!("fidim-disc-{}", std::process::id()));
        let mtp = dir.join("MTP");
        std::fs::create_dir_all(&mtp).unwrap();
        // Minimal valid GGUF: magic, v3, 0 tensors, 0 kvs.
        let mut minimal = Vec::new();
        minimal.extend_from_slice(b"GGUF");
        minimal.extend_from_slice(&3u32.to_le_bytes());
        minimal.extend_from_slice(&0u64.to_le_bytes());
        minimal.extend_from_slice(&0u64.to_le_bytes());
        for name in ["model-q4.gguf", "mmproj-model-f16.gguf"] {
            std::fs::write(dir.join(name), &minimal).unwrap();
        }
        std::fs::write(mtp.join("mtp-draft-Q8_0.gguf"), &minimal).unwrap();

        let models = scan_models(&[dir.clone()]);
        assert_eq!(models.len(), 1, "only the base model is a model");
        let m = &models[0];
        assert!(m.path.ends_with("model-q4.gguf"));
        assert_eq!(m.mmproj_candidates.len(), 1);
        assert_eq!(m.draft_candidates.len(), 1);
        assert!(m.draft_candidates[0].ends_with("mtp-draft-Q8_0.gguf"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn build_meta_and_dotdirs() {
        let root = std::env::temp_dir().join(format!("fidim-disc-builds-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let make = |name: &str, runner: bool| -> PathBuf {
            let bin = root.join(name).join("bin");
            std::fs::create_dir_all(&bin).unwrap();
            // Not a real executable: the probe fails into version_error,
            // which is how a scan reports any broken build.
            std::fs::write(bin.join("llama-server.exe"), b"").unwrap();
            if runner {
                std::fs::write(bin.join(RUNNER_EXE), b"").unwrap();
            }
            root.join(name)
        };
        make(".fidim-tmp-1", true);
        let unsloth = make("b11027-mix-3e83366-unsloth", true);
        let upstream = make("b10819-rocm", false);
        let runner_only = make("hand-built-dg", true);

        // Written the way install_unsloth will write it; the full Manifest
        // fields around the channel ones must not disturb the lite parse.
        std::fs::write(
            unsloth.join(crate::update::MANIFEST_NAME),
            r#"{"tag":"b11027-mix-3e83366","source":"unsloth-prebuilt","installed_at_unix":1,
                "assets":["app-b11027-mix-3e83366-windows-x64-rocm-gfx120X.zip"],
                "verify":{"version":"b11027","commit":null,"devices":[],"hip_ok":true,"detail":"","runner_present":true},
                "channel":"unsloth","bundled_runtime":true,"release_tag":"b11027-mix-3e83366",
                "asset_sha256":"00ff","gfx_target":"gfx120X"}"#,
        )
        .unwrap();
        // A manifest from before the channel fields existed, under the
        // pre-rename file name.
        std::fs::write(
            upstream.join("llamactl-build.json"),
            r#"{"tag":"b10819","source":"prebuilt","installed_at_unix":1,
                "assets":["shim:hipblas.dll <- libhipblas.dll"],
                "verify":{"version":"b10819","commit":"abc","devices":[],"hip_ok":true,"detail":""}}"#,
        )
        .unwrap();

        let m = read_build_meta(&unsloth);
        assert_eq!(m.channel, Channel::Unsloth);
        assert!(m.bundled_runtime);
        assert_eq!(m.release_tag.as_deref(), Some("b11027-mix-3e83366"));
        for dir in [upstream.clone(), runner_only.clone(), root.join("missing")] {
            let m = read_build_meta(&dir);
            assert_eq!(m.channel, Channel::Upstream, "{dir:?}");
            assert!(!m.bundled_runtime, "{dir:?}");
            assert_eq!(m.release_tag, None, "{dir:?}");
        }
        let garbled = root.join("garbled");
        std::fs::create_dir_all(&garbled).unwrap();
        std::fs::write(garbled.join(crate::update::MANIFEST_NAME), r#"{"channel": 5, "bundled_runtime": true}"#)
            .unwrap();
        assert_eq!(read_build_meta(&garbled).channel, Channel::Upstream);
        assert!(!read_build_meta(&garbled).bundled_runtime);

        let builds = scan_builds(&[root.clone()], None);
        let tags: Vec<&str> = builds.iter().map(|b| b.tag.as_str()).collect();
        assert_eq!(tags, vec!["b10819-rocm", "b11027-mix-3e83366-unsloth", "hand-built-dg"], "dot-dir skipped");
        let find = |t: &str| builds.iter().find(|b| b.tag == t).unwrap();

        let b = find("b11027-mix-3e83366-unsloth");
        assert_eq!(b.channel, Channel::Unsloth);
        assert!(b.bundled_runtime);
        assert_eq!(b.release_tag.as_deref(), Some("b11027-mix-3e83366"));
        assert_eq!(b.runner_exe.as_deref(), Some(unsloth.join("bin").join(RUNNER_EXE).as_path()));
        assert!(b.version_error.is_some());

        // The runner alone never makes a build Unsloth: upstream may ship it.
        let b = find("hand-built-dg");
        assert_eq!(b.channel, Channel::Upstream);
        assert!(!b.bundled_runtime);
        assert!(b.runner_exe.is_some());

        let b = find("b10819-rocm");
        assert_eq!(b.channel, Channel::Upstream);
        assert_eq!(b.runner_exe, None);

        let v = serde_json::to_value(find("b11027-mix-3e83366-unsloth")).unwrap();
        assert_eq!(v["channel"], "unsloth");
        assert_eq!(v["bundled_runtime"], true);
        assert_eq!(v["patch"], serde_json::Value::Null);
        std::fs::remove_dir_all(root).ok();
    }

    /// A locally patched build keeps its base's channel and release tag; its
    /// `patch` block is what tells them apart, and a malformed one only drops
    /// the patch, never the channel.
    #[test]
    fn patched_build_meta() {
        let root = std::env::temp_dir().join(format!("fidim-disc-patch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("b11027-mix-3e83366-unsloth-dgpatch");
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        let manifest = |patch: &str| {
            format!(
                r#"{{"tag":"b11027-mix-3e83366","source":"unsloth-local-patch","installed_at_unix":1,"assets":[],
                    "verify":{{"version":"b11027","commit":null,"devices":[],"hip_ok":true,"detail":""}},
                    "channel":"unsloth","bundled_runtime":true,"release_tag":"b11027-mix-3e83366"{patch}}}"#
            )
        };
        std::fs::write(
            dir.join(crate::update::MANIFEST_NAME),
            manifest(r#","patch":{"name":"dgpatch","base_commit":"f6b9ea743","features":["dg-pkv-f16","dg-swa-ring","dg-fa-pad","dg-fa-turn-sizing"]}"#),
        )
        .unwrap();
        let m = read_build_meta(&dir);
        assert_eq!(m.channel, Channel::Unsloth);
        assert!(m.bundled_runtime);
        let patch = m.patch.as_ref().unwrap();
        assert_eq!(patch.name, "dgpatch");
        assert_eq!(patch.base_commit.as_deref(), Some("f6b9ea743"));
        assert!(m.has_feature(dg_feature::FA_PAD) && m.has_feature(dg_feature::SWA_RING));
        assert!(!m.has_feature(dg_feature::STEP_FAIL_ERR));

        // The full manifest round-trips the block (re-verify rewrites it).
        let full: crate::update::Manifest =
            serde_json::from_str(&std::fs::read_to_string(dir.join(crate::update::MANIFEST_NAME)).unwrap()).unwrap();
        assert_eq!(full.patch.as_ref(), Some(patch));
        let back = serde_json::to_string(&full).unwrap();
        assert!(back.contains(r#""patch":{"name":"dgpatch""#), "{back}");

        std::fs::write(dir.join(crate::update::MANIFEST_NAME), manifest(r#","patch":{"features":"not a list"}"#)).unwrap();
        let m = read_build_meta(&dir);
        assert_eq!(m.channel, Channel::Unsloth, "a bad patch block must not demote the build");
        assert!(m.patch.is_none());
        // ...nor make the full manifest unreadable (re-verify, shim retirement).
        let full: crate::update::Manifest =
            serde_json::from_str(&std::fs::read_to_string(dir.join(crate::update::MANIFEST_NAME)).unwrap()).unwrap();
        assert!(full.patch.is_none());
        assert_eq!(full.release_tag.as_deref(), Some("b11027-mix-3e83366"));
        // An unnamed patch still counts, under a label.
        assert_eq!(BuildPatch { features: vec![dg_feature::FA_PAD.into()], ..Default::default() }.label(), "patched");
        std::fs::write(dir.join(crate::update::MANIFEST_NAME), manifest("")).unwrap();
        assert!(read_build_meta(&dir).patch.is_none());
        std::fs::remove_dir_all(root).ok();
    }

    /// A build compiled from a git ref: channel `git` and its source, shown
    /// by its label. A channel from a newer FIDIM reads as `other`, never as
    /// upstream, and a malformed `git` block only drops the label.
    #[test]
    fn git_build_meta() {
        let root = std::env::temp_dir().join(format!("fidim-disc-git-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("ifm-ai-K2Horizon-fork-42adf019-src");
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        std::fs::write(dir.join("bin").join("llama-server.exe"), b"").unwrap();
        let manifest = |channel: &str, git: &str| {
            format!(
                r#"{{"tag":"t","source":"git-ref","installed_at_unix":1,"assets":[],
                    "verify":{{"version":"b10676","commit":null,"devices":[],"hip_ok":true,"detail":""}},
                    "channel":"{channel}","git":{git}}}"#
            )
        };
        let git = r#"{"remote":"https://github.com/ifm-ai/llama.cpp","git_ref":"model/K2Horizon","commit":"42adf019f76013dac873b5b43950d54d5ab27216","label":"ifm-ai K2Horizon fork"}"#;
        std::fs::write(dir.join(crate::update::MANIFEST_NAME), manifest("git", git)).unwrap();
        let m = read_build_meta(&dir);
        assert_eq!(m.channel, Channel::Git);
        assert_eq!(m.git.as_ref().unwrap().display(), "ifm-ai K2Horizon fork @42adf01");
        let b = scan_builds(&[root.clone()], None).remove(0);
        assert_eq!((b.channel, b.git.as_ref().unwrap().git_ref.as_str()), (Channel::Git, "model/K2Horizon"));
        assert_eq!(serde_json::to_value(&b).unwrap()["channel"], "git");

        std::fs::write(dir.join(crate::update::MANIFEST_NAME), manifest("custom", git)).unwrap();
        assert_eq!(read_build_meta(&dir).channel, Channel::Other);
        std::fs::write(dir.join(crate::update::MANIFEST_NAME), manifest("git", "[1,2]")).unwrap();
        let m = read_build_meta(&dir);
        assert_eq!(m.channel, Channel::Git, "a bad git block keeps the channel");
        assert!(m.git.is_none());
        assert_eq!(Channel::Git.as_str(), "git");
        std::fs::remove_dir_all(root).ok();
    }

    /// Minimal GGUF: an architecture and, optionally, the literal
    /// `diffusion.canvas_length` key.
    fn tiny_gguf(arch: &str, canvas: Option<u32>) -> Vec<u8> {
        fn key(out: &mut Vec<u8>, k: &str) {
            out.extend_from_slice(&(k.len() as u64).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        out.extend_from_slice(&(1 + canvas.is_some() as u64).to_le_bytes());
        key(&mut out, "general.architecture");
        out.extend_from_slice(&8u32.to_le_bytes());
        out.extend_from_slice(&(arch.len() as u64).to_le_bytes());
        out.extend_from_slice(arch.as_bytes());
        if let Some(c) = canvas {
            key(&mut out, "diffusion.canvas_length");
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&c.to_le_bytes());
        }
        out
    }

    #[test]
    fn model_engine_comes_from_the_header() {
        use crate::profile::Engine;
        let dir = std::env::temp_dir().join(format!("fidim-disc-engine-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("diffusiongemma-26B-A4B-it-Q4_K_M.gguf"), tiny_gguf("diffusion-gemma", Some(256)))
            .unwrap();
        std::fs::write(dir.join("gemma-4-e4b-it-q8_0.gguf"), tiny_gguf("gemma4", None)).unwrap();
        std::fs::write(dir.join("truncated.gguf"), b"GGUF").unwrap();

        let models = scan_models(&[dir.clone()]);
        let engine = |stem: &str| models.iter().find(|m| m.path.file_stem().unwrap() == stem).unwrap().engine;
        assert_eq!(engine("diffusiongemma-26B-A4B-it-Q4_K_M"), Engine::DiffusionGemma);
        assert_eq!(engine("gemma-4-e4b-it-q8_0"), Engine::LlamaServer);
        assert_eq!(engine("truncated"), Engine::LlamaServer, "unreadable header = the default engine");
        let dg = models.iter().find(|m| m.engine == Engine::DiffusionGemma).unwrap();
        assert_eq!(serde_json::to_value(dg).unwrap()["engine"], "diffusion-gemma");
        std::fs::remove_dir_all(dir).ok();
    }
}
