//! Tool configuration: where to scan, where profiles/runs live.
//!
//! Stored at `%USERPROFILE%\.llamactl\config.json`. Unknown fields are
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
    pub rocm_bin: Option<PathBuf>,
    /// Device names matching any of these substrings are classified as
    /// integrated graphics. APU marketing names carry no model suffix.
    #[serde(default = "default_igpu_patterns")]
    pub integrated_name_patterns: Vec<String>,
    /// Where profile JSON files live.
    pub profile_dir: PathBuf,
    /// Where run state + captured logs live.
    pub runs_dir: PathBuf,
    /// Preserved unknown fields from newer schema versions.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn default_igpu_patterns() -> Vec<String> {
    vec!["Radeon(TM) Graphics".into()]
}

impl Config {
    pub fn config_dir() -> PathBuf {
        // HOME first so tests can redirect; USERPROFILE is the Windows reality.
        let home = std::env::var_os("LLAMACTL_HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        home.join(".llamactl")
    }

    pub fn config_path() -> PathBuf {
        Self::config_dir().join("config.json")
    }

    /// Defaults matched to the machine this tool is being built for; edit the
    /// JSON to point elsewhere.
    pub fn default_for_machine() -> Self {
        let dir = Self::config_dir();
        Config {
            build_roots: vec![PathBuf::from(
                r"C:\Users\Paul\Documents\Claude\Projects\AMD GPU Programming\llama.cpp",
            )],
            model_roots: vec![PathBuf::from(r"E:\models")],
            rocm_bin: Some(PathBuf::from(r"C:\Program Files\AMD\ROCm\7.1\bin")),
            integrated_name_patterns: default_igpu_patterns(),
            profile_dir: dir.join("profiles"),
            runs_dir: dir.join("runs"),
            extra: serde_json::Map::new(),
        }
    }

    /// Load config, creating the default file on first run.
    pub fn load_or_init() -> Result<Self> {
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
        assert_eq!(cfg.integrated_name_patterns, vec!["Radeon(TM) Graphics"]);
    }
}
