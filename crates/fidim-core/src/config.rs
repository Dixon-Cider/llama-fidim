//! Tool configuration: where to scan, where profiles/runs live.
//!
//! Stored at `%USERPROFILE%\.fidim\config.json`. Unknown fields are
//! preserved on round-trip so a newer tool version's config survives an older
//! one touching it.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Directories scanned (one level of subdirectories) for `bin/llama-server.exe`.
    pub build_roots: Vec<PathBuf>,
    /// Directories scanned recursively for `*.gguf`.
    pub model_roots: Vec<PathBuf>,
    /// Prepended to PATH when invoking llama-server (HIP runtime DLLs).
    /// Listed as the `default` runtime; profiles that name no runtime use it.
    pub rocm_bin: Option<PathBuf>,
    /// Name of the runtime used when a profile names none (see
    /// `runtime::discover`). None = `default` = `rocm_bin`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_runtime: Option<String>,
    /// GPU family for AMD's nightly ROCm index (`gfx120X-all` = RDNA4).
    /// None = guess from the cards, else `gfx120X-all`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rocm_family: Option<String>,
    /// Extra runtimes declared by hand, on top of the auto-discovered ones.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub runtimes: Vec<crate::runtime::ManualRuntime>,
    /// Device names matching any of these substrings are classified as
    /// integrated graphics. APU marketing names carry no model suffix.
    #[serde(default = "default_igpu_patterns")]
    pub integrated_name_patterns: Vec<String>,
    /// Let a profile bind integrated graphics. Off by default: on a box
    /// with discrete cards the iGPU is a trap (20x slower, no error). On
    /// an APU-only machine, or a Strix Halo with 96 GB of shared memory,
    /// it is the whole point.
    #[serde(default)]
    pub allow_integrated: bool,
    /// Where profile JSON files live.
    pub profile_dir: PathBuf,
    /// Where run state + captured logs live.
    pub runs_dir: PathBuf,
    /// Where `fidim update` installs new builds (`<root>/<tag>-<flavor>`).
    /// Defaults to the first build root so the scan finds them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub install_root: Option<PathBuf>,
    /// llama.cpp git checkout used for source builds (defaults to the first
    /// build root, which on this machine IS the checkout).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub llama_cpp_source: Option<PathBuf>,
    /// Script invoked as `<script> <checkout> <tag> <output dir>` to build a
    /// tag from source with the local HIP toolchain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_build_script: Option<PathBuf>,
    /// Hugging Face token for gated repos when fetching creator defaults
    /// (`HF_TOKEN` in the environment takes precedence).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_token: Option<String>,
    /// Default keep-alive interval for launched servers (seconds; 0 = off).
    /// A 1-token request this often stops WDDM from evicting the model when
    /// the displays power off. OFF by default: the root cause is the PCIe
    /// Link State Power Management setting (pre-flight check 12); turn this
    /// on (5 s measured sufficient) only if that setting cannot be Off.
    #[serde(default = "default_keep_alive")]
    pub keep_alive_seconds: u32,
    /// Preserved unknown fields from newer schema versions.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// `C:\Program Files\AMD\ROCm\<newest>\bin` when a HIP SDK is installed.
fn newest_hip_sdk_bin() -> Option<PathBuf> {
    let root = std::env::var_os("ProgramFiles")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files"))
        .join("AMD")
        .join("ROCm");
    std::fs::read_dir(root)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("bin").join("amdhip64_7.dll").is_file() || p.join("bin").join("amdhip64.dll").is_file())
        .max_by_key(|p| crate::rocm::version_key(&p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()))
        .map(|p| p.join("bin"))
}

/// Names AMD gives integrated graphics. Discrete cards say "Radeon RX",
/// "Radeon PRO" or "Radeon AI PRO"; APUs say "Radeon(TM) Graphics" or a
/// three-digit model with an M/S suffix (780M, 890M, 8060S).
fn default_igpu_patterns() -> Vec<String> {
    ["Radeon(TM) Graphics", "Radeon(TM) 7", "Radeon(TM) 8", "760M", "780M", "880M", "890M", "8040S", "8050S", "8060S"]
        .into_iter()
        .map(String::from)
        .collect()
}

/// LM Studio's model folder when it exists, so a first run has models to
/// pick from. Newer versions keep it under `~/.lmstudio/models`, older
/// ones under `~/.cache/lm-studio/models`.
fn lm_studio_models_dir() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(PathBuf::from)?;
    [home.join(".lmstudio").join("models"), home.join(".cache").join("lm-studio").join("models")]
        .into_iter()
        .find(|p| p.is_dir())
}

fn default_keep_alive() -> u32 {
    0
}

impl Config {
    pub fn config_dir() -> PathBuf {
        // FIDIM_HOME first so tests can redirect; USERPROFILE is the Windows reality.
        let home = std::env::var_os("FIDIM_HOME")
            .or_else(|| std::env::var_os("LLAMACTL_HOME"))
            .or_else(|| std::env::var_os("USERPROFILE"))
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        home.join(".fidim")
    }

    /// The tool was called llamactl until 2026-09-06. A `~/.llamactl` with
    /// no `~/.fidim` beside it is moved over once, and every path inside its
    /// JSON files (config roots, run logs, router preset) is rewritten.
    pub fn migrate_legacy_home() -> Option<PathBuf> {
        let new = Self::config_dir();
        let old = new.parent()?.join(".llamactl");
        if new.exists() || !old.is_dir() {
            return None;
        }
        // A running server keeps its log open under the old folder, and
        // Windows will not rename a folder with open files in it. Copy then.
        let renamed = std::fs::rename(&old, &new).is_ok();
        if !renamed {
            fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
                std::fs::create_dir_all(to)?;
                for e in std::fs::read_dir(from)?.flatten() {
                    let (src, dst) = (e.path(), to.join(e.file_name()));
                    if src.is_dir() { copy_dir(&src, &dst)?; } else { std::fs::copy(&src, &dst)?; }
                }
                Ok(())
            }
            if copy_dir(&old, &new).is_err() {
                let _ = std::fs::remove_dir_all(&new);
                return None;
            }
        }
        let (from, to) = (old.to_string_lossy().into_owned(), new.to_string_lossy().into_owned());
        let from_fwd = from.replace('\\', "/");
        let to_fwd = to.replace('\\', "/");
        let from_json = from.replace('\\', "\\\\");
        let to_json = to.replace('\\', "\\\\");
        fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() { walk(&p, out); } else if p.extension().is_some_and(|x| x == "json" || x == "ini") { out.push(p); }
                }
            }
        }
        let mut files = Vec::new();
        walk(&new, &mut files);
        let runs = new.join("runs");
        for f in files {
            // Run-state files of servers that are still running point at
            // logs the process holds open in the old folder; leave them.
            if !renamed && f.starts_with(&runs) {
                continue;
            }
            if let Ok(t) = std::fs::read_to_string(&f) {
                let u = t.replace(&from_json, &to_json).replace(&from_fwd, &to_fwd).replace(&from, &to);
                if u != t {
                    let _ = std::fs::write(&f, u);
                }
            }
        }
        Some(new)
    }

    /// Where builds are looked for: the configured roots plus wherever
    /// Updates installs to, so an installed build shows up without adding
    /// its folder by hand.
    pub fn build_roots_effective(&self) -> Vec<PathBuf> {
        let mut roots = self.build_roots.clone();
        let install = self.install_root.clone().unwrap_or_else(|| Self::config_dir().join("builds"));
        if !roots.iter().any(|r| r == &install) {
            roots.push(install);
        }
        roots
    }

    pub fn config_path() -> PathBuf {
        Self::config_dir().join("config.json")
    }

    /// First-run defaults: no roots (the Settings tab asks for them), the
    /// newest HIP SDK found under Program Files as the fallback runtime.
    pub fn default_for_machine() -> Self {
        let dir = Self::config_dir();
        Config {
            build_roots: vec![],
            model_roots: lm_studio_models_dir().into_iter().collect(),
            rocm_bin: newest_hip_sdk_bin(),
            default_runtime: None,
            rocm_family: None,
            runtimes: Vec::new(),
            integrated_name_patterns: default_igpu_patterns(),
            allow_integrated: false,
            profile_dir: dir.join("profiles"),
            runs_dir: dir.join("runs"),
            install_root: None,
            llama_cpp_source: None,
            source_build_script: None,
            hf_token: None,
            keep_alive_seconds: default_keep_alive(),
            extra: serde_json::Map::new(),
        }
    }

    /// Load config, creating the default file on first run.
    pub fn load_or_init() -> Result<Self> {
        Self::migrate_legacy_home();
        let path = Self::config_path();
        if path.exists() {
            Self::load(&path)
        } else {
            let cfg = Self::default_for_machine();
            cfg.save(&path)?;
            Ok(cfg)
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        let cfg: Config = serde_json::from_str(&text)
            .map_err(|e| Error::Config(format!("{}: {e}", path.display())))?;
        Ok(cfg)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text).map_err(|e| Error::io(path, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_unknown_fields() {
        let raw = r#"{
            "build_roots": ["C:/b"],
            "model_roots": ["C:/m"],
            "rocm_bin": null,
            "profile_dir": "C:/p",
            "runs_dir": "C:/r",
            "some_future_field": {"x": 1}
        }"#;
        let cfg: Config = serde_json::from_str(raw).unwrap();
        assert!(cfg.extra.contains_key("some_future_field"));
        let out = serde_json::to_string(&cfg).unwrap();
        assert!(out.contains("some_future_field"));
        // Default applied for the missing patterns field.
        assert!(cfg.integrated_name_patterns.iter().any(|p| p == "Radeon(TM) Graphics"));
    }
}
