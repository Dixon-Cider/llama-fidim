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
    /// The file llama.cpp is given: the file, or the first shard of a split
    /// model.
    pub path: PathBuf,
    /// Bytes on disk, every shard included.
    pub file_size: u64,
    /// Modified time (unix seconds) — feeds the cold-cache heuristic (R-08).
    pub modified_unix: Option<u64>,
    /// Read from `path`; for a split model, with every shard's size and
    /// tensor table folded in (`discovery::read_split_header`).
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
    /// Every shard of a split model (`-00001-of-0000N.gguf` ...), in order;
    /// empty for a single file.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub shards: Vec<PathBuf>,
    /// Where FIDIM downloaded it from (its folder's `fidim-source.json`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<ModelSource>,
}

/// A model's origin on the Hugging Face Hub, from the sidecar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSource {
    pub repo: String,
    /// The commit it was downloaded at.
    pub commit: String,
    /// Its path in the repo (the local copy has the last part as its name).
    pub path: String,
    pub sha256: Option<String>,
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
///
/// The shards of a split model (`<name>-00001-of-0000N.gguf` ...) are one
/// model. Importance matrices (`*imatrix*`) are not models, and a download
/// in progress (`*.gguf.part`) is not a `.gguf` at all.
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
                } else if is_gguf(&p) && !is_imatrix(&p) {
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
            let sidecar = if main.is_empty() { None } else { read_source_sidecar(&dir) };
            for files in group_split_sets(main) {
                models.push(load_model(files, &mmproj, &drafts, sidecar.as_ref()));
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
        let stem = model_stem(m);
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

/// The name a model pairs by: its stem, or a split model's name before the
/// shard number.
fn model_stem(m: &Model) -> String {
    if !m.shards.is_empty() {
        if let Some((prefix, _, _)) = m.path.file_name().and_then(|n| n.to_str()).and_then(gguf::split_name) {
            return prefix.to_lowercase();
        }
    }
    stem_lower(&m.path)
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

fn is_imatrix(p: &Path) -> bool {
    p.file_name().is_some_and(|n| n.to_string_lossy().to_lowercase().contains("imatrix"))
}

/// One model's files: a single GGUF, or the shards of a split set found in
/// one folder. `problem` is set when shards are missing.
struct ModelFiles {
    path: PathBuf,
    shards: Vec<PathBuf>,
    problem: Option<String>,
}

/// Group a folder's model files, gathering each split set's shards.
fn group_split_sets(files: Vec<PathBuf>) -> Vec<ModelFiles> {
    let mut out = Vec::new();
    // (lowercase prefix, count) -> (number, path)
    let mut sets: std::collections::BTreeMap<(String, u32), Vec<(u32, PathBuf)>> = Default::default();
    for p in files {
        let split = p.file_name().and_then(|n| n.to_str()).and_then(gguf::split_name).map(|(prefix, no, count)| {
            ((prefix.to_lowercase(), count), no)
        });
        match split {
            Some((key, no)) => sets.entry(key).or_default().push((no, p)),
            None => out.push(ModelFiles { path: p, shards: Vec::new(), problem: None }),
        }
    }
    for ((_, count), mut parts) in sets {
        parts.sort_by_key(|(no, _)| *no);
        let numbers: Vec<u32> = parts.iter().map(|(no, _)| *no).collect();
        let problem = (numbers != (1..=count).collect::<Vec<_>>()).then(|| {
            let present = numbers.iter().map(u32::to_string).collect::<Vec<_>>().join(", ");
            format!("split model incomplete: parts {present} of {count} are here, and llama.cpp needs all of them")
        });
        let shards: Vec<PathBuf> = parts.into_iter().map(|(_, p)| p).collect();
        out.push(ModelFiles { path: shards[0].clone(), shards, problem });
    }
    out
}

/// Every shard of the split set whose first shard is `first`, by name, in
/// order. None when `first` is not a `-00001-of-0000N.gguf` file.
pub fn split_shards(first: &Path) -> Option<Vec<PathBuf>> {
    let name = first.file_name()?.to_str()?;
    let (prefix, no, count) = gguf::split_name(name)?;
    if no != 1 {
        return None;
    }
    let ext = &name[name.len() - ".gguf".len()..];
    Some((1..=count).map(|i| first.with_file_name(format!("{prefix}-{i:05}-of-{count:05}{ext}"))).collect())
}

/// A model's header, covering the whole model: for the first shard of a
/// split set, every shard's folded in (see `read_split_header`). The VRAM
/// estimate takes the weights from `file_size`; a build check takes
/// `max_tensor_type`.
pub fn read_model_header(path: &Path) -> Result<GgufHeader> {
    match split_shards(path) {
        Some(shards) => read_split_header(&shards),
        None => gguf::read_header(path),
    }
}

/// The header of a split model from its shards, in order: the first
/// shard's keys, with every shard's size and tensor table (each shard
/// lists only its own tensors) folded in. A missing shard is an error:
/// llama.cpp cannot load the model, and its size and tensor types would
/// come out short.
pub fn read_split_header(shards: &[PathBuf]) -> Result<GgufHeader> {
    if let Some(missing) = shards.iter().find(|p| !p.is_file()) {
        return Err(Error::io(
            missing,
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "this part of a split model is missing, and llama.cpp needs every part",
            ),
        ));
    }
    let (first, rest) = shards.split_first().ok_or_else(|| Error::InvalidInput("a split model with no parts".into()))?;
    let mut h = gguf::read_header(first)?;
    h.fold_split_shards(rest.iter().map(|p| gguf::read_header(p)).collect::<Result<Vec<_>>>()?);
    Ok(h)
}

fn load_model(files: ModelFiles, mmproj: &[PathBuf], drafts: &[PathBuf], sidecar: Option<&SourceSidecar>) -> Model {
    let all = if files.shards.is_empty() { std::slice::from_ref(&files.path) } else { &files.shards[..] };
    let metas: Vec<std::fs::Metadata> = all.iter().filter_map(|p| std::fs::metadata(p).ok()).collect();
    let file_size = metas.iter().map(|m| m.len()).sum();
    let modified_unix = metas
        .iter()
        .filter_map(|m| m.modified().ok())
        .filter_map(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .max();
    let read = match &files.problem {
        Some(problem) => Err(problem.clone()),
        None if files.shards.is_empty() => gguf::read_header(&files.path).map_err(|e| e.to_string()),
        None => read_split_header(&files.shards).map_err(|e| e.to_string()),
    };
    let (header, header_error) = match read {
        Ok(h) => (Some(h), None),
        Err(e) => (None, Some(e)),
    };
    let source = sidecar.and_then(|s| s.source_of(&files.path));
    Model {
        path: files.path,
        file_size,
        modified_unix,
        engine: header.as_ref().map(|h| h.engine()).unwrap_or_default(),
        header,
        header_error,
        mmproj_candidates: mmproj.to_vec(),
        draft_candidates: drafts.to_vec(),
        shards: files.shards,
        source,
    }
}

// --------------------------------------------------------------- sidecar ----

/// Written beside downloaded models: which repo and commit each file came
/// from and its SHA-256, so provenance survives a folder layout that does
/// not name the repo and the GGUF naming none (K2 Horizon's do not).
pub const SOURCE_SIDECAR: &str = "fidim-source.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceSidecar {
    pub repo: String,
    /// The commit of the latest download into this folder.
    pub sha: String,
    pub files: Vec<SourceFile>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceFile {
    /// Path in the repo; the local file has its last part as its name.
    pub path: String,
    pub size: u64,
    #[serde(default)]
    pub sha256: Option<String>,
    /// The commit this file was downloaded at, when files of one folder
    /// came from different commits (else the sidecar's `sha`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

impl SourceFile {
    fn name(&self) -> &str {
        self.path.rsplit('/').next().unwrap_or(&self.path)
    }
}

impl SourceSidecar {
    /// The entry for a local file, matched by file name.
    pub fn source_of(&self, local: &Path) -> Option<ModelSource> {
        let name = local.file_name()?.to_string_lossy();
        let f = self.files.iter().find(|f| f.name().eq_ignore_ascii_case(&name))?;
        Some(ModelSource {
            repo: self.repo.clone(),
            commit: f.commit.clone().unwrap_or_else(|| self.sha.clone()),
            path: f.path.clone(),
            sha256: f.sha256.clone(),
        })
    }
}

/// The sidecar in `dir`, if there is a readable one.
pub fn read_source_sidecar(dir: &Path) -> Option<SourceSidecar> {
    serde_json::from_str(&std::fs::read_to_string(dir.join(SOURCE_SIDECAR)).ok()?).ok()
}

/// Where the model at `path` came from, from its folder's sidecar.
pub fn model_source(path: &Path) -> Option<ModelSource> {
    read_source_sidecar(path.parent()?)?.source_of(path)
}

/// Serializes sidecar updates within this process (see `SidecarLock` for
/// other processes).
static SIDECAR_WRITERS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Record that `files` of `repo` at commit `sha` were downloaded into
/// `dir`. Entries for other files of the same repo are kept (one folder
/// collects several quants over time); a sidecar naming another repo is
/// replaced. Written to a temporary file and renamed, so a reader never
/// sees half of it.
///
/// Safe to call from parallel downloads into one folder: updates run one
/// at a time (in this process, and on Windows across processes), so none
/// writes the old record over another's file.
pub fn write_source_sidecar(dir: &Path, repo: &str, sha: &str, files: &[crate::hub::RepoFile]) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    // A panic elsewhere while holding it leaves nothing half-done here.
    let _in_process = SIDECAR_WRITERS.lock().unwrap_or_else(|e| e.into_inner());
    let _across = SidecarLock::acquire(dir)?;
    let mut kept: Vec<SourceFile> = match read_source_sidecar(dir) {
        Some(old) if old.repo == repo => {
            let old_sha = old.sha.clone();
            old.files
                .into_iter()
                .filter(|f| !files.iter().any(|n| n.name().eq_ignore_ascii_case(f.name())))
                // Pin each kept entry to the commit it came from.
                .map(|f| SourceFile { commit: f.commit.or_else(|| Some(old_sha.clone())), ..f })
                .collect()
        }
        _ => Vec::new(),
    };
    kept.extend(files.iter().map(|f| SourceFile {
        path: f.path.clone(),
        size: f.size,
        sha256: f.sha256.clone(),
        commit: None,
    }));
    // An entry at the sidecar's own commit does not repeat it.
    for f in &mut kept {
        if f.commit.as_deref() == Some(sha) {
            f.commit = None;
        }
    }
    kept.sort_by(|a, b| a.path.cmp(&b.path));
    let sidecar = SourceSidecar { repo: repo.to_string(), sha: sha.to_string(), files: kept };
    let path = dir.join(SOURCE_SIDECAR);
    // Named for this process: where no lock spans processes, two writers
    // never share (and truncate) one temporary file.
    let tmp = dir.join(format!("{SOURCE_SIDECAR}.{}.tmp", std::process::id()));
    std::fs::write(&tmp, serde_json::to_string_pretty(&sidecar)?).map_err(|e| Error::io(&tmp, e))?;
    crate::fetch::rename_into_place(&tmp, &path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// `fidim-source.json.lock` in the folder, held while this process updates
/// the sidecar. On Windows it is opened with no sharing, so another
/// process's open fails until this one closes it, which also happens if
/// the process dies; it is removed afterwards unless another writer
/// already holds it. Elsewhere this is a no-op (the in-process mutex
/// still applies).
struct SidecarLock {
    #[cfg(windows)]
    file: Option<std::fs::File>,
    #[cfg(windows)]
    path: PathBuf,
}

impl SidecarLock {
    #[cfg(windows)]
    fn acquire(dir: &Path) -> Result<SidecarLock> {
        use std::os::windows::fs::OpenOptionsExt;
        const ERROR_ACCESS_DENIED: i32 = 5;
        const ERROR_SHARING_VIOLATION: i32 = 32;
        let path = dir.join(format!("{SOURCE_SIDECAR}.lock"));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            match std::fs::OpenOptions::new().write(true).create(true).truncate(false).share_mode(0).open(&path) {
                Ok(file) => return Ok(SidecarLock { file: Some(file), path }),
                // Held by another writer, or being deleted by the last one.
                Err(e)
                    if matches!(e.raw_os_error(), Some(ERROR_SHARING_VIOLATION | ERROR_ACCESS_DENIED))
                        && std::time::Instant::now() < deadline =>
                {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                Err(e) => return Err(Error::io(&path, e)),
            }
        }
    }

    #[cfg(not(windows))]
    fn acquire(_dir: &Path) -> Result<SidecarLock> {
        Ok(SidecarLock {})
    }
}

#[cfg(windows)]
impl Drop for SidecarLock {
    fn drop(&mut self) {
        drop(self.file.take());
        // Fails, harmlessly, when another writer opened it in between.
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::testing::split_shard;

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

    /// A split model is one model at its first shard, sized as the whole
    /// set; a set with a part missing says so; importance matrices and
    /// downloads in progress are not models.
    #[test]
    fn split_sets_imatrix_and_partial_downloads() {
        let root = std::env::temp_dir().join(format!("fidim-disc-split-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("unsloth").join("gemma-4-26B-A4B-it-GGUF");
        let elsewhere = root.join("other");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&elsewhere).unwrap();
        let first = tiny_gguf("gemma4", None);
        std::fs::write(dir.join("gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf"), &first).unwrap();
        let mut second = split_shard(None, 1, 2, &[("blk.0.ffn_up.weight", 30)]);
        second.resize(5000, 0); // the tensor data follows the header
        std::fs::write(dir.join("gemma-4-26B-A4B-it-BF16-00002-of-00002.gguf"), &second).unwrap();
        std::fs::write(dir.join("gemma-4-26B-A4B-it-Q8_0-00001-of-00003.gguf"), &first).unwrap();
        std::fs::write(dir.join("gemma-4-26B-A4B-it-Q8_0-00003-of-00003.gguf"), vec![7u8; 10]).unwrap();
        std::fs::write(dir.join("gemma-4-26B-A4B-it-Q4_K_M.gguf"), &first).unwrap();
        std::fs::write(dir.join("gemma-4-26B-A4B-it-Q6_K.gguf.part"), &first).unwrap();
        std::fs::write(dir.join("k2_horizon_7b_combined.imatrix.gguf"), &first).unwrap();
        std::fs::write(elsewhere.join("mmproj-gemma-4-26B-A4B-it-F16.gguf"), &first).unwrap();

        let (models, aux) = scan_models_and_aux(&[root.clone()]);
        let names: Vec<String> =
            models.iter().map(|m| m.path.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(
            names,
            [
                "gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf",
                "gemma-4-26B-A4B-it-Q4_K_M.gguf",
                "gemma-4-26B-A4B-it-Q8_0-00001-of-00003.gguf"
            ]
        );
        let split = &models[0];
        assert_eq!(split.shards.len(), 2);
        assert!(split.shards[1].ends_with("gemma-4-26B-A4B-it-BF16-00002-of-00002.gguf"));
        assert_eq!(split.file_size, first.len() as u64 + 5000);
        let h = split.header.as_ref().unwrap();
        assert_eq!(h.file_size, split.file_size, "the estimate sees every shard");
        assert_eq!(h.architecture.as_deref(), Some("gemma4"));
        assert_eq!(h.max_tensor_type(), Some(30), "the second shard's tensors count");
        assert!(
            split.mmproj_candidates.iter().any(|p| p.ends_with("mmproj-gemma-4-26B-A4B-it-F16.gguf")),
            "a split model pairs by its name before the shard number"
        );
        assert_eq!(aux.mmproj.len(), 1);

        let single = &models[1];
        assert!(single.shards.is_empty());
        let json = serde_json::to_value(single).unwrap();
        assert!(json.get("shards").is_none() && json.get("source").is_none(), "unchanged shape for plain files");

        let broken = &models[2];
        assert!(broken.header.is_none());
        let err = broken.header_error.as_deref().unwrap();
        assert!(err.contains("parts 1, 3 of 3"), "{err}");

        // The launch path sizes a split model the same way.
        let h = read_model_header(&split.path).unwrap();
        assert_eq!(h.file_size, split.file_size);
        assert_eq!(read_model_header(&single.path).unwrap().file_size, first.len() as u64);
        assert_eq!(split_shards(&split.shards[1]), None, "only the first shard names the set");
        std::fs::remove_dir_all(root).ok();
    }

    /// Every shard lists only its own tensors, so a split model's highest
    /// tensor type is the highest over every shard: here a fork-only type
    /// in the last shard, and a first shard written with
    /// `--no-tensor-first-split` that has no tensors at all. A missing
    /// shard is an error, never a smaller model.
    #[test]
    fn split_models_read_every_shard() {
        let root = std::env::temp_dir().join(format!("fidim-disc-shards-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("o").join("r");
        std::fs::create_dir_all(&dir).unwrap();
        let shards = [
            ("M-Q4_K_M-00001-of-00003.gguf", split_shard(Some("k2-horizon"), 0, 3, &[("token_embd.weight", 12)])),
            ("M-Q4_K_M-00002-of-00003.gguf", split_shard(None, 1, 3, &[("blk.0.ffn_up.weight", 14)])),
            ("M-Q4_K_M-00003-of-00003.gguf", split_shard(None, 2, 3, &[("blk.1.ffn_up.weight", 105), ("output.weight", 8)])),
            ("N-00001-of-00002.gguf", split_shard(Some("llama"), 0, 2, &[])),
            ("N-00002-of-00002.gguf", split_shard(None, 1, 2, &[("blk.0.attn_q.weight", 12)])),
        ];
        for (name, bytes) in &shards {
            std::fs::write(dir.join(name), bytes).unwrap();
        }

        let models = scan_models(&[root.clone()]);
        assert_eq!(models.len(), 2);
        let m = &models[0];
        let h = m.header.as_ref().unwrap_or_else(|| panic!("{:?}", m.header_error));
        assert_eq!(h.max_tensor_type(), Some(105), "the last shard's type counts");
        assert_eq!(h.tensor_count, 4);
        assert_eq!(h.architecture.as_deref(), Some("k2-horizon"));
        assert_eq!(h.file_size, shards[..3].iter().map(|(_, b)| b.len() as u64).sum::<u64>());
        let h = models[1].header.as_ref().unwrap();
        assert_eq!(h.max_tensor_type(), Some(12), "a tensor-free first shard");

        // The launch path reads the same.
        let first = dir.join(shards[0].0);
        let h = read_model_header(&first).unwrap();
        assert_eq!((h.max_tensor_type(), h.file_size), (Some(105), m.file_size));
        // The first shard alone does not claim to know the model's types.
        assert_eq!(gguf::read_header(&first).unwrap().max_tensor_type(), None);

        std::fs::remove_file(dir.join(shards[1].0)).unwrap();
        match read_model_header(&first) {
            Err(Error::Io { path, .. }) => assert!(path.ends_with(shards[1].0), "{path:?}"),
            other => panic!("{other:?}"),
        }
        std::fs::remove_dir_all(root).ok();
    }

    /// Downloads finishing together (the shards of one split set, say)
    /// each record their file; none is lost and the record stays whole.
    #[test]
    fn concurrent_sidecar_writes_keep_every_file() {
        use crate::hub::RepoFile;
        let dir = std::env::temp_dir().join(format!("fidim-disc-sidecar-race-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let n = 12u32;
        std::thread::scope(|s| {
            let handles: Vec<_> = (1..=n)
                .map(|i| {
                    let dir = &dir;
                    s.spawn(move || {
                        let f = RepoFile { path: format!("M-{i:05}-of-{n:05}.gguf"), size: i as u64, sha256: None };
                        write_source_sidecar(dir, "o/r", "sha1", &[f])
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap().unwrap();
            }
        });
        let s = read_source_sidecar(&dir).expect("a whole sidecar");
        assert_eq!(s.files.len(), n as usize, "{:?}", s.files.iter().map(|f| &f.path).collect::<Vec<_>>());
        let tmp: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(tmp.is_empty(), "{tmp:?}");
        std::fs::remove_dir_all(dir).ok();
    }

    /// Another process updating the same sidecar (its lock file open with
    /// no sharing) is waited for.
    #[cfg(windows)]
    #[test]
    fn sidecar_writes_wait_for_another_process() {
        use crate::hub::RepoFile;
        use std::os::windows::fs::OpenOptionsExt;
        let dir = std::env::temp_dir().join(format!("fidim-disc-sidecar-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let held = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .share_mode(0)
            .open(dir.join(format!("{SOURCE_SIDECAR}.lock")))
            .unwrap();
        std::thread::scope(|s| {
            let writer = s.spawn(|| {
                let f = RepoFile { path: "m.gguf".into(), size: 1, sha256: None };
                write_source_sidecar(&dir, "o/r", "sha1", &[f])
            });
            std::thread::sleep(std::time::Duration::from_millis(300));
            assert!(!dir.join(SOURCE_SIDECAR).exists(), "written while another process held the lock");
            drop(held);
            writer.join().unwrap().unwrap();
        });
        assert_eq!(read_source_sidecar(&dir).unwrap().files.len(), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn source_sidecar_round_trip() {
        use crate::hub::RepoFile;
        let root = std::env::temp_dir().join(format!("fidim-disc-sidecar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let dir = root.join("IFM").join("K2-Horizon-7B-GGUF");
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("K2-Horizon-7B-Q4_K_M.gguf");
        std::fs::write(&model, tiny_gguf("k2-horizon", None)).unwrap();
        let q4 = RepoFile { path: "K2-Horizon-7B-Q4_K_M.gguf".into(), size: 5, sha256: Some("aa".repeat(32)) };
        write_source_sidecar(&dir, "ngquocvinh/K2-Horizon-7B-GGUF", "sha1", std::slice::from_ref(&q4)).unwrap();
        assert!(!dir.join(format!("{SOURCE_SIDECAR}.{}.tmp", std::process::id())).exists());

        let models = scan_models(&[root.clone()]);
        let src = models[0].source.as_ref().unwrap();
        assert_eq!(src.repo, "ngquocvinh/K2-Horizon-7B-GGUF");
        assert_eq!(src.commit, "sha1");
        assert_eq!(src.sha256.as_deref(), q4.sha256.as_deref());
        assert_eq!(model_source(&model).as_ref(), Some(src));

        // A later download at a newer commit keeps the earlier entry, pinned
        // to the commit it came from; a subfolder path matches by file name.
        let bf16 = RepoFile { path: "BF16/K2-Horizon-7B-BF16.gguf".into(), size: 9, sha256: None };
        write_source_sidecar(&dir, "ngquocvinh/K2-Horizon-7B-GGUF", "sha2", std::slice::from_ref(&bf16)).unwrap();
        let s = read_source_sidecar(&dir).unwrap();
        assert_eq!(s.sha, "sha2");
        assert_eq!(s.files.len(), 2);
        assert_eq!(s.source_of(&model).unwrap().commit, "sha1");
        let b = s.source_of(&dir.join("k2-horizon-7b-bf16.gguf")).unwrap();
        assert_eq!((b.commit.as_str(), b.path.as_str()), ("sha2", "BF16/K2-Horizon-7B-BF16.gguf"));
        // Downloading the same file again at the new commit re-pins it.
        write_source_sidecar(&dir, "ngquocvinh/K2-Horizon-7B-GGUF", "sha2", &[q4.clone()]).unwrap();
        let s = read_source_sidecar(&dir).unwrap();
        assert!(s.files.iter().all(|f| f.commit.is_none()), "{s:?}");
        // Another repo's download replaces the record.
        write_source_sidecar(&dir, "IFM/K2-Horizon-7B-GGUF", "sha3", &[q4]).unwrap();
        let s = read_source_sidecar(&dir).unwrap();
        assert_eq!((s.repo.as_str(), s.files.len()), ("IFM/K2-Horizon-7B-GGUF", 1));
        // A damaged sidecar is ignored, not fatal.
        std::fs::write(dir.join(SOURCE_SIDECAR), "{not json").unwrap();
        assert!(scan_models(&[root.clone()])[0].source.is_none());
        std::fs::remove_dir_all(root).ok();
    }
}
