//! ROCm runtimes: the directories of HIP/rocBLAS/hipBLAS DLLs a server runs
//! against. Several coexist on this box and are interchangeable for
//! llama.cpp (benched 2026-09-02: no measurable difference), so the tool
//! discovers them, lets a profile pick one by name, and defaults to the
//! system HIP SDK (`config.rocm_bin`, 7.1) when a profile says nothing.
//!
//! Known sources:
//! - **HIP SDK**: `C:\Program Files\AMD\ROCm\<ver>\bin`
//! - **ComfyUI**: AMD's pip-distributed ROCm inside a ComfyUI venv,
//!   `_rocm_sdk_libraries\bin` (+ `_rocm_sdk_core\bin` for the HIP runtime)
//! - **LM Studio**: `~/.lmstudio/extensions/backends/vendor/win-llama-rocm-vendor-*/bin`
//! - **manual**: `config.runtimes` entries
//!
//! A runtime is a list of directories; `path_prepend` joins them with `;`
//! so the existing single-`Option<PathBuf>` plumbing carries it unchanged.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::{Error, Result};

/// The name the default (config.rocm_bin) runtime is listed under.
pub const DEFAULT_NAME: &str = "default";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Runtime {
    pub name: String,
    /// `hip-sdk`, `comfyui`, `lmstudio`, `manual`, or `default`.
    pub source: String,
    pub version: Option<String>,
    /// Search-path directories, first wins. All must exist for `available`.
    pub dirs: Vec<PathBuf>,
    pub available: bool,
    /// Whether this is what profiles get when they name no runtime.
    pub is_default: bool,
}

/// A user-declared runtime in config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManualRuntime {
    pub name: String,
    pub dirs: Vec<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

impl Runtime {
    /// `dir1;dir2` as one PathBuf-compatible value, or None when empty.
    pub fn path_prepend(&self) -> Option<PathBuf> {
        if self.dirs.is_empty() {
            return None;
        }
        let mut s = OsString::new();
        for (i, d) in self.dirs.iter().enumerate() {
            if i > 0 {
                s.push(";");
            }
            s.push(d.as_os_str());
        }
        Some(PathBuf::from(s))
    }

    fn with_availability(mut self) -> Self {
        self.available = !self.dirs.is_empty() && self.dirs.iter().all(|d| d.is_dir());
        self
    }
}

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(PathBuf::from)
}

fn subdirs(root: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(root)
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect())
        .unwrap_or_default()
}

fn dir_name(p: &Path) -> String {
    p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Version from a ComfyUI-style site-packages: `rocm_sdk_core-7.14.0.dist-info`.
fn comfy_rocm_version(site_packages: &Path) -> Option<String> {
    subdirs(site_packages).into_iter().find_map(|d| {
        let n = dir_name(&d);
        n.strip_prefix("rocm_sdk_core-")?.strip_suffix(".dist-info").map(|v| v.to_string())
    })
}

/// Everything discoverable on this machine plus manual config entries,
/// default first. Pure filesystem probing; nothing is executed.
pub fn discover(cfg: &Config) -> Vec<Runtime> {
    let mut out = Vec::new();
    let default_name = cfg.default_runtime.clone().unwrap_or_else(|| DEFAULT_NAME.to_string());

    // The legacy single dir is always listed so existing configs keep working.
    if let Some(bin) = &cfg.rocm_bin {
        let version = bin
            .parent()
            .map(dir_name)
            .filter(|v| v.chars().next().is_some_and(|c| c.is_ascii_digit()));
        out.push(
            Runtime {
                name: DEFAULT_NAME.into(),
                source: "default".into(),
                version,
                dirs: vec![bin.clone()],
                available: false,
                is_default: default_name == DEFAULT_NAME,
            }
            .with_availability(),
        );
    }

    // HIP SDK installs.
    let sdk_root = env_path("ProgramFiles")
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files"))
        .join("AMD")
        .join("ROCm");
    for d in subdirs(&sdk_root) {
        let bin = d.join("bin");
        if !bin.join("amdhip64_7.dll").is_file() && !bin.join("amdhip64.dll").is_file() {
            continue;
        }
        let ver = dir_name(&d);
        out.push(
            Runtime {
                name: format!("hip-sdk-{ver}"),
                source: "hip-sdk".into(),
                version: Some(ver),
                dirs: vec![bin],
                available: false,
                is_default: false,
            }
            .with_availability(),
        );
    }

    // ComfyUI desktop installs carrying AMD's pip ROCm.
    if let Some(local) = env_path("LOCALAPPDATA") {
        let installs = local.join("Comfy-Desktop").join("ComfyUI-Installs");
        for inst in subdirs(&installs) {
            let sp = inst.join("ComfyUI").join(".venv").join("Lib").join("site-packages");
            let libs = sp.join("_rocm_sdk_libraries").join("bin");
            let core = sp.join("_rocm_sdk_core").join("bin");
            if !libs.is_dir() {
                continue;
            }
            let version = comfy_rocm_version(&sp);
            let mut dirs = vec![libs];
            if core.is_dir() {
                dirs.push(core);
            }
            out.push(
                Runtime {
                    name: format!(
                        "comfyui-{}{}",
                        dir_name(&inst).to_lowercase().replace(' ', "-"),
                        version.as_ref().map(|v| format!("-{v}")).unwrap_or_default()
                    ),
                    source: "comfyui".into(),
                    version,
                    dirs,
                    available: false,
                    is_default: false,
                }
                .with_availability(),
            );
        }
    }

    // LM Studio's vendored runtime.
    if let Some(home) = env_path("USERPROFILE").or_else(|| env_path("HOME")) {
        let vendor = home.join(".lmstudio").join("extensions").join("backends").join("vendor");
        for d in subdirs(&vendor) {
            let n = dir_name(&d);
            if !n.contains("rocm") {
                continue;
            }
            let bin = d.join("bin");
            if !bin.is_dir() {
                continue;
            }
            out.push(
                Runtime {
                    name: format!("lmstudio-{n}"),
                    source: "lmstudio".into(),
                    version: None,
                    dirs: vec![bin],
                    available: false,
                    is_default: false,
                }
                .with_availability(),
            );
        }
    }

    for m in &cfg.runtimes {
        out.push(
            Runtime {
                name: m.name.clone(),
                source: "manual".into(),
                version: m.version.clone(),
                dirs: m.dirs.clone(),
                available: false,
                is_default: false,
            }
            .with_availability(),
        );
    }

    // config.default_runtime may name a discovered one.
    if default_name != DEFAULT_NAME {
        for r in &mut out {
            r.is_default = r.name == default_name;
        }
    }
    // Stable order: default first, then by name; drop exact duplicates.
    out.sort_by(|a, b| b.is_default.cmp(&a.is_default).then(a.name.cmp(&b.name)));
    out.dedup_by(|a, b| a.name == b.name);
    out
}

/// The runtime a profile launches with: its named choice, else the config
/// default. Naming a runtime that is missing or unavailable is an error —
/// pre-flight must not silently fall back to a different set of DLLs.
pub fn resolve(cfg: &Config, name: Option<&str>) -> Result<Runtime> {
    let all = discover(cfg);
    let wanted = name
        .map(str::to_string)
        .or_else(|| cfg.default_runtime.clone())
        .unwrap_or_else(|| DEFAULT_NAME.to_string());
    let r = all
        .into_iter()
        .find(|r| r.name == wanted)
        .ok_or_else(|| Error::Config(format!("ROCm runtime `{wanted}` is not known — see `llamactl runtimes`")))?;
    if !r.available {
        return Err(Error::Config(format!(
            "ROCm runtime `{}` is not available: missing {}",
            r.name,
            r.dirs.iter().filter(|d| !d.is_dir()).map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
        )));
    }
    Ok(r)
}

/// Convenience for call sites that only need the PATH prefix.
pub fn path_prepend(cfg: &Config, name: Option<&str>) -> Result<Option<PathBuf>> {
    Ok(resolve(cfg, name)?.path_prepend())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with(rocm_bin: Option<PathBuf>, manual: Vec<ManualRuntime>, default: Option<&str>) -> Config {
        let mut c = Config::default_for_machine();
        c.rocm_bin = rocm_bin;
        c.runtimes = manual;
        c.default_runtime = default.map(str::to_string);
        c
    }

    #[test]
    fn path_prepend_joins_dirs_with_semicolons() {
        let r = Runtime {
            name: "x".into(),
            source: "manual".into(),
            version: None,
            dirs: vec![PathBuf::from(r"C:\a"), PathBuf::from(r"C:\b c")],
            available: true,
            is_default: false,
        };
        assert_eq!(r.path_prepend().unwrap().to_string_lossy(), r"C:\a;C:\b c");
    }

    #[test]
    fn default_runtime_is_the_legacy_rocm_bin_and_listed_first() {
        let tmp = std::env::temp_dir().join(format!("llamactl-rt-{}", std::process::id()));
        let bin = tmp.join("7.1").join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        let cfg = cfg_with(Some(bin.clone()), vec![], None);
        let all = discover(&cfg);
        let d = &all[0];
        assert_eq!(d.name, DEFAULT_NAME);
        assert!(d.is_default && d.available);
        assert_eq!(d.version.as_deref(), Some("7.1"));
        let r = resolve(&cfg, None).unwrap();
        assert_eq!(r.path_prepend().unwrap(), bin);
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn manual_runtime_selectable_by_name_and_unavailable_when_missing() {
        let tmp = std::env::temp_dir().join(format!("llamactl-rt2-{}", std::process::id()));
        let have = tmp.join("have");
        std::fs::create_dir_all(&have).unwrap();
        let cfg = cfg_with(
            None,
            vec![
                ManualRuntime { name: "good".into(), dirs: vec![have.clone()], version: Some("9.9".into()) },
                ManualRuntime { name: "gone".into(), dirs: vec![tmp.join("missing")], version: None },
            ],
            Some("good"),
        );
        let all = discover(&cfg);
        let good = all.iter().find(|r| r.name == "good").unwrap();
        assert!(good.available && good.is_default);
        assert!(!all.iter().find(|r| r.name == "gone").unwrap().available);
        assert!(resolve(&cfg, Some("gone")).is_err());
        assert!(resolve(&cfg, Some("nope")).is_err());
        assert_eq!(resolve(&cfg, None).unwrap().name, "good");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn comfy_version_parsed_from_dist_info() {
        let tmp = std::env::temp_dir().join(format!("llamactl-rt3-{}", std::process::id()));
        std::fs::create_dir_all(tmp.join("rocm_sdk_core-7.14.0.dist-info")).unwrap();
        assert_eq!(comfy_rocm_version(&tmp).as_deref(), Some("7.14.0"));
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
