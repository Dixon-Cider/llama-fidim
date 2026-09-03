//! Discovery of builds and models by filesystem scan (spec R-01, R-02).

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::gguf::{self, GgufHeader};
use crate::{Error, Result};

// ---------------------------------------------------------------- builds ----

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
}

/// Scan each root and its immediate subdirectories for `bin/llama-server.exe`.
pub fn scan_builds(roots: &[PathBuf], rocm_bin: Option<&Path>) -> Vec<Build> {
    let mut found = Vec::new();
    for root in roots {
        let mut candidates = vec![root.clone()];
        if let Ok(entries) = std::fs::read_dir(root) {
            candidates.extend(entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()));
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
    let mut build = Build {
        path: dir.to_path_buf(),
        tag,
        server_exe: exe.to_path_buf(),
        version: None,
        commit: None,
        version_error: None,
    };
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
    /// Paired multimodal projectors found beside the model (R-02).
    pub mmproj_candidates: Vec<PathBuf>,
    /// Paired speculative-decoding draft models (R-02): `MTP/` subdirectory
    /// contents or `*mtp*.gguf` siblings.
    pub draft_candidates: Vec<PathBuf>,
}

/// Recursively scan model roots for `.gguf` files, pairing auxiliary files.
///
/// Pairing rules (from how the real model tree is laid out):
/// - a file named `mmproj-*.gguf` is a projector for the models in its
///   directory, not a model itself;
/// - files under an `MTP/` subdirectory, or with `mtp` in the stem, are draft
///   models for the models in the parent directory.
pub fn scan_models(roots: &[PathBuf]) -> Vec<Model> {
    let mut models = Vec::new();
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
    models.sort_by(|a, b| a.path.cmp(&b.path));
    models
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
    fn pairing_separates_mmproj_and_mtp_from_models() {
        let dir = std::env::temp_dir().join(format!("llamactl-disc-{}", std::process::id()));
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
}
