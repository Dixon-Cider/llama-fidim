//! ROCm runtimes from AMD, installed the way builds are: one folder per
//! version under `<install root>\rocm\`, never modified after install, and
//! selectable by name from any profile.
//!
//! AMD ships ROCm for Windows as pip wheels. Two channels:
//!
//! - **Releases**: `https://repo.radeon.com/rocm/windows/rocm-rel-<ver>/`,
//!   a plain directory with `rocm_sdk_core-<ver>-py3-none-win_amd64.whl`
//!   and `rocm_sdk_libraries_custom-<ver>-py3-none-win_amd64.whl` (all
//!   GPU families in one wheel).
//! - **Nightlies**: `https://rocm.nightlies.amd.com/v2/<family>/`, a PEP 503
//!   index per GPU family (`gfx120X-all` for RDNA4). Core is shared,
//!   libraries are family-specific: `rocm_sdk_libraries_<family>-…`.
//!
//! Both wheels are zips. Everything under `_rocm_sdk_*/bin/` is extracted
//! into one `bin` folder, which is what llama-server needs on PATH (the HIP
//! runtime, rocBLAS/hipBLAS/hipBLASLt and their kernel libraries).

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::{Error, Result};

pub const RELEASE_ROOT: &str = "https://repo.radeon.com/rocm/windows/";
pub const NIGHTLY_ROOT: &str = "https://rocm.nightlies.amd.com/v2/";
pub const MANIFEST_NAME: &str = "fidim-runtime.json";
/// Nightlies pile up (hundreds); the picker shows the newest few.
const NIGHTLIES_SHOWN: usize = 8;

fn err(msg: impl Into<String>) -> Error {
    Error::Update(msg.into())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AvailableRuntime {
    pub version: String,
    /// `release` or `nightly`.
    pub channel: String,
    /// GPU family the libraries wheel targets; None for release wheels
    /// (they carry every family).
    pub family: Option<String>,
    pub core_url: String,
    pub libraries_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeManifest {
    pub version: String,
    pub channel: String,
    pub family: Option<String>,
    pub installed_at_unix: u64,
    pub wheels: Vec<String>,
    pub files: usize,
}

// ------------------------------------------------------------ versions ----

/// Orders AMD versions: `7.2.1` > `7.2` > `7.14.0a20260612`? No: `7.14` is
/// newer than `7.2`, and a release beats a pre-release of the same base.
/// Key = (major, minor, patch, is_release, pre-release date).
pub fn version_key(v: &str) -> (u32, u32, u32, u8, u64) {
    let (base, pre) = match v.find(|c: char| c.is_ascii_alphabetic()) {
        Some(i) => (&v[..i], Some(&v[i..])),
        None => (v, None),
    };
    let mut nums = base.split('.').map(|s| s.parse::<u32>().unwrap_or(0));
    let (a, b, c) = (nums.next().unwrap_or(0), nums.next().unwrap_or(0), nums.next().unwrap_or(0));
    let date = pre
        .map(|p| p.trim_start_matches(|c: char| c.is_ascii_alphabetic()).parse::<u64>().unwrap_or(0))
        .unwrap_or(0);
    (a, b, c, if pre.is_none() { 1 } else { 0 }, date)
}

// ------------------------------------------------------------- listing ----

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(120))
        .build()
}

fn get_text(url: &str) -> Result<String> {
    agent()
        .get(url)
        .set("User-Agent", concat!("llama-fidim/", env!("CARGO_PKG_VERSION")))
        .call()
        .map_err(|e| err(format!("{url}: {e}")))?
        .into_string()
        .map_err(|e| err(format!("{url}: reading body: {e}")))
}

/// `href="…"` targets of an index page, in order, duplicates removed.
pub fn hrefs(html: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut rest = html;
    while let Some(i) = rest.find("href=\"") {
        rest = &rest[i + 6..];
        let Some(j) = rest.find('"') else { break };
        let h = rest[..j].to_string();
        if !out.contains(&h) {
            out.push(h);
        }
        rest = &rest[j + 1..];
    }
    out
}

/// GPU families the nightly index serves (`gfx120X-all`, `gfx110X-all`, …).
pub fn families() -> Result<Vec<String>> {
    Ok(parse_families(&get_text(NIGHTLY_ROOT)?))
}

pub fn parse_families(html: &str) -> Vec<String> {
    hrefs(html)
        .into_iter()
        .filter_map(|h| h.strip_suffix('/').map(str::to_string))
        .filter(|h| h.starts_with("gfx"))
        .collect()
}

/// Version out of `rocm_sdk_core-7.2.1-py3-none-win_amd64.whl`.
fn wheel_version(file: &str, project: &str) -> Option<String> {
    let name = file.rsplit('/').next()?;
    let rest = name.strip_prefix(project)?.strip_prefix('-')?;
    let v = rest.split('-').next()?;
    if !rest.ends_with("win_amd64.whl") || v.is_empty() {
        return None;
    }
    Some(v.to_string())
}

/// Release channel: every `rocm-rel-*/` directory that holds both wheels.
pub fn parse_release_dirs(root_html: &str) -> Vec<String> {
    hrefs(root_html)
        .into_iter()
        .filter(|h| h.starts_with("rocm-rel-") && h.ends_with('/'))
        .collect()
}

pub fn parse_release_dir(dir_url: &str, dir_html: &str) -> Option<AvailableRuntime> {
    let files = hrefs(dir_html);
    let core = files.iter().find(|f| wheel_version(f, "rocm_sdk_core").is_some())?;
    let libs = files.iter().find(|f| wheel_version(f, "rocm_sdk_libraries_custom").is_some())?;
    // The directory carries the real version (`rocm-rel-7.1.1/`); the wheels
    // inside early releases were versioned `0.1.dev0`.
    let from_dir = dir_url
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .and_then(|d| d.strip_prefix("rocm-rel-"))
        .filter(|v| !v.is_empty())
        .map(str::to_string);
    Some(AvailableRuntime {
        version: from_dir.or_else(|| wheel_version(core, "rocm_sdk_core"))?,
        channel: "release".into(),
        family: None,
        core_url: format!("{dir_url}{core}"),
        libraries_url: format!("{dir_url}{libs}"),
    })
}

/// Nightly channel for one family: versions present for both core and the
/// family's libraries wheel, newest first, capped.
pub fn parse_nightlies(family: &str, core_html: &str, libs_html: &str) -> Vec<AvailableRuntime> {
    let family_root = format!("{NIGHTLY_ROOT}{family}/");
    let libs_project = format!("rocm_sdk_libraries_{}", family.to_lowercase().replace('-', "_"));
    let abs = |h: &str| {
        let f = h.trim_start_matches("../");
        format!("{family_root}{}", f.split('#').next().unwrap_or(f))
    };
    let cores: Vec<(String, String)> = hrefs(core_html)
        .iter()
        .filter_map(|h| wheel_version(h.split('#').next().unwrap_or(h), "rocm_sdk_core").map(|v| (v, abs(h))))
        .collect();
    let libs: Vec<(String, String)> = hrefs(libs_html)
        .iter()
        .filter_map(|h| wheel_version(h.split('#').next().unwrap_or(h), &libs_project).map(|v| (v, abs(h))))
        .collect();
    let mut out: Vec<AvailableRuntime> = cores
        .into_iter()
        .filter_map(|(v, core_url)| {
            let (_, libraries_url) = libs.iter().find(|(lv, _)| *lv == v)?;
            Some(AvailableRuntime {
                version: v,
                channel: "nightly".into(),
                family: Some(family.to_string()),
                core_url,
                libraries_url: libraries_url.clone(),
            })
        })
        .collect();
    out.sort_by(|a, b| version_key(&b.version).cmp(&version_key(&a.version)));
    out.dedup_by(|a, b| a.version == b.version);
    out.truncate(NIGHTLIES_SHOWN);
    out
}

/// Everything installable for `family`, newest first: all releases plus the
/// newest nightlies. Network errors on one channel do not hide the other.
pub fn available(family: &str) -> Result<(Vec<AvailableRuntime>, Vec<String>)> {
    let mut out = Vec::new();
    let mut problems = Vec::new();
    match get_text(RELEASE_ROOT) {
        Ok(root) => {
            for d in parse_release_dirs(&root) {
                let url = format!("{RELEASE_ROOT}{d}");
                match get_text(&url) {
                    Ok(html) => {
                        if let Some(a) = parse_release_dir(&url, &html) {
                            out.push(a);
                        }
                    }
                    Err(e) => problems.push(e.to_string()),
                }
            }
        }
        Err(e) => problems.push(format!("release channel: {e}")),
    }
    let core_url = format!("{NIGHTLY_ROOT}{family}/rocm-sdk-core/");
    let libs_url = format!("{NIGHTLY_ROOT}{family}/rocm-sdk-libraries-{}/", family.to_lowercase());
    match (get_text(&core_url), get_text(&libs_url)) {
        (Ok(c), Ok(l)) => out.extend(parse_nightlies(family, &c, &l)),
        (Err(e), _) | (_, Err(e)) => problems.push(format!("nightly channel: {e}")),
    }
    out.sort_by(|a, b| version_key(&b.version).cmp(&version_key(&a.version)));
    if out.is_empty() && !problems.is_empty() {
        return Err(err(problems.join("; ")));
    }
    Ok((out, problems))
}

// ------------------------------------------------------------- install ----

/// `<install root>\rocm`, beside the llama.cpp builds.
pub fn runtimes_root(cfg: &Config) -> PathBuf {
    cfg.install_root
        .clone()
        .or_else(|| cfg.build_roots.first().cloned())
        .unwrap_or_else(|| Config::config_dir().join("builds"))
        .join("rocm")
}

/// Nightly-index family for a card, from its marketing name. RDNA4 (RX
/// 9000, Radeon AI PRO R9700) is gfx120X; RDNA3 (RX 7000, W7000) is
/// gfx110X; RDNA2 (RX 6000) is gfx103X; the Strix APUs have their own.
/// None when nothing matches: the user picks in Settings.
pub fn guess_family(names: &[String]) -> Option<String> {
    let n: Vec<String> = names.iter().map(|s| s.to_ascii_lowercase()).collect();
    let has = |needle: &str| n.iter().any(|s| s.contains(needle));
    if has("r9700") || has("rx 90") || has("rx 9") || has("ai pro r9") {
        return Some("gfx120X-all".into());
    }
    if has("8060s") || has("8050s") || has("8040s") {
        return Some("gfx1151".into());
    }
    if has("880m") || has("890m") {
        return Some("gfx1150".into());
    }
    if has("rx 7") || has("w7") || has("780m") || has("760m") {
        return Some("gfx110X-all".into());
    }
    if has("rx 6") || has("w6") {
        return Some("gfx103X-all".into());
    }
    None
}

pub fn install_dir(cfg: &Config, version: &str) -> PathBuf {
    runtimes_root(cfg).join(version)
}

/// Installed runtimes: (directory, manifest), any order.
pub fn installed(cfg: &Config) -> Vec<(PathBuf, RuntimeManifest)> {
    let root = runtimes_root(cfg);
    let Ok(rd) = std::fs::read_dir(&root) else { return vec![] };
    rd.flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter_map(|p| {
            let mp = [p.join(MANIFEST_NAME), p.join("llamactl-runtime.json")].into_iter().find(|m| m.is_file())?;
            let m: RuntimeManifest = serde_json::from_str(&std::fs::read_to_string(mp).ok()?).ok()?;
            Some((p, m))
        })
        .collect()
}

/// Destination for a wheel entry: `_rocm_sdk_core/bin/x/y.dll` → `x/y.dll`;
/// anything not under a `_rocm_sdk_*/bin/` is skipped (python glue, dist-info).
pub fn bin_relative(entry: &str) -> Option<&str> {
    let e = entry.trim_start_matches('/');
    let (top, rest) = e.split_once('/')?;
    if !top.starts_with("_rocm_sdk") {
        return None;
    }
    let rel = rest.strip_prefix("bin/")?;
    if rel.is_empty() || rel.contains("..") {
        return None;
    }
    Some(rel)
}

fn extract_bin(zip_path: &Path, bin: &Path, progress: &mut dyn FnMut(String)) -> Result<usize> {
    let f = File::open(zip_path).map_err(|e| Error::io(zip_path, e))?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| err(format!("{}: {e}", zip_path.display())))?;
    let mut count = 0;
    let total = z.len();
    for i in 0..total {
        let mut entry = z.by_index(i).map_err(|e| err(format!("{}: entry {i}: {e}", zip_path.display())))?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().replace('\\', "/");
        let Some(rel) = bin_relative(&name) else { continue };
        let out = bin.join(rel);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let mut dst = File::create(&out).map_err(|e| Error::io(&out, e))?;
        std::io::copy(&mut entry, &mut dst).map_err(|e| Error::io(&out, e))?;
        count += 1;
        if count % 200 == 0 {
            progress(format!("{}: {count} files extracted", zip_path.file_name().unwrap_or_default().to_string_lossy()));
        }
    }
    Ok(count)
}

/// Download both wheels and unpack their `bin` trees into one folder. The
/// folder is immutable afterwards; re-installing the same version is a no-op.
pub fn install(cfg: &Config, a: &AvailableRuntime, progress: &mut dyn FnMut(String)) -> Result<PathBuf> {
    let dir = install_dir(cfg, &a.version);
    if dir.join(MANIFEST_NAME).is_file() {
        progress(format!("ROCm {} is already installed at {}", a.version, dir.display()));
        return Ok(dir);
    }
    let tmp = runtimes_root(cfg).join(format!(".tmp-{}", a.version));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| Error::io(&tmp, e))?;
    let bin = tmp.join("bin");
    let mut wheels = Vec::new();
    let mut files = 0;
    for url in [&a.core_url, &a.libraries_url] {
        let name = url.rsplit('/').next().unwrap_or("wheel").to_string();
        let to = tmp.join(&name);
        progress(format!("downloading {name}"));
        crate::update::download_url(url, &name, &to, progress)?;
        progress(format!("unpacking {name}"));
        files += extract_bin(&to, &bin, progress)?;
        let _ = std::fs::remove_file(&to);
        wheels.push(name);
    }
    if !bin.join("amdhip64_7.dll").is_file() && !bin.join("amdhip64.dll").is_file() {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(err("the wheels unpacked without a HIP runtime DLL (amdhip64*.dll); layout changed upstream?"));
    }
    let m = RuntimeManifest {
        version: a.version.clone(),
        channel: a.channel.clone(),
        family: a.family.clone(),
        installed_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        wheels,
        files,
    };
    let mp = tmp.join(MANIFEST_NAME);
    std::fs::write(&mp, serde_json::to_string_pretty(&m)?).map_err(|e| Error::io(&mp, e))?;
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    std::fs::rename(&tmp, &dir).map_err(|e| Error::io(&dir, e))?;
    progress(format!("ROCm {} installed: {files} files at {}", a.version, dir.display()));
    Ok(dir)
}

/// Delete an installed runtime. Refuses when it is the config default.
pub fn remove(cfg: &Config, version: &str) -> Result<()> {
    let dir = install_dir(cfg, version);
    if !dir.join(MANIFEST_NAME).is_file() {
        return Err(err(format!("ROCm {version} is not installed under {}", runtimes_root(cfg).display())));
    }
    if cfg.default_runtime.as_deref() == Some(&format!("rocm-{version}")) {
        return Err(err(format!("ROCm {version} is the default runtime; pick another default first")));
    }
    std::fs::remove_dir_all(&dir).map_err(|e| Error::io(&dir, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_order_like_amd_means_them() {
        let mut v = vec!["7.2", "7.14.0a20260612", "7.2.1", "6.4.4", "7.14.0", "7.14.0a20260101"];
        v.sort_by(|a, b| version_key(b).cmp(&version_key(a)));
        assert_eq!(v, ["7.14.0", "7.14.0a20260612", "7.14.0a20260101", "7.2.1", "7.2", "6.4.4"]);
    }

    #[test]
    fn release_directory_parses() {
        let root = r#"<a href="../">../</a><a href="rocm-rel-7.1.1/">rocm-rel-7.1.1/</a><a href="rocm-rel-7.2.1/">x</a><a href="https://rocm.docs.amd.com/">docs</a>"#;
        assert_eq!(parse_release_dirs(root), ["rocm-rel-7.1.1/", "rocm-rel-7.2.1/"]);
        let dir = r#"<a href="rocm_sdk_core-7.2.1-py3-none-win_amd64.whl">c</a><a href="rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl">d</a><a href="rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl">l</a>"#;
        let a = parse_release_dir("https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/", dir).unwrap();
        assert_eq!(a.version, "7.2.1");
        // Placeholder wheel versions never leak into the picker.
        let early = r#"<a href="rocm_sdk_core-0.1.dev0-py3-none-win_amd64.whl">c</a><a href="rocm_sdk_libraries_custom-0.1.dev0-py3-none-win_amd64.whl">l</a>"#;
        assert_eq!(parse_release_dir("https://repo.radeon.com/rocm/windows/rocm-rel-7.1.1/", early).unwrap().version, "7.1.1");
        assert_eq!(a.channel, "release");
        assert!(a.core_url.ends_with("rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl"));
        assert!(a.libraries_url.contains("libraries_custom-7.2.1"));
    }

    #[test]
    fn nightlies_intersect_core_and_family_libraries() {
        let core = r#"<a href="../rocm_sdk_core-7.14.0a20260611-py3-none-win_amd64.whl#sha256=x">a</a><a href="../rocm_sdk_core-7.14.0a20260612-py3-none-win_amd64.whl">b</a><a href="../rocm_sdk_core-7.14.0a20260612-py3-none-manylinux_2_28_x86_64.whl">lin</a>"#;
        let libs = r#"<a href="../rocm_sdk_libraries_gfx120x_all-7.14.0a20260612-py3-none-win_amd64.whl">l</a>"#;
        let n = parse_nightlies("gfx120X-all", core, libs);
        assert_eq!(n.len(), 1);
        assert_eq!(n[0].version, "7.14.0a20260612");
        assert_eq!(n[0].core_url, "https://rocm.nightlies.amd.com/v2/gfx120X-all/rocm_sdk_core-7.14.0a20260612-py3-none-win_amd64.whl");
        assert_eq!(n[0].family.as_deref(), Some("gfx120X-all"));
    }

    #[test]
    fn family_guess_from_card_names() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(guess_family(&s(&["AMD Radeon AI PRO R9700", "AMD Radeon(TM) Graphics"])).as_deref(), Some("gfx120X-all"));
        assert_eq!(guess_family(&s(&["AMD Radeon RX 7900 XTX"])).as_deref(), Some("gfx110X-all"));
        assert_eq!(guess_family(&s(&["AMD Radeon 8060S Graphics"])).as_deref(), Some("gfx1151"));
        assert_eq!(guess_family(&s(&["AMD Radeon RX 6800"])).as_deref(), Some("gfx103X-all"));
        assert_eq!(guess_family(&s(&["NVIDIA GeForce RTX 3060"])), None);
    }

    #[test]
    fn families_and_bin_mapping() {
        assert_eq!(parse_families(r#"<a href="gfx110X-all/">1</a><a href="gfx120X-all/">2</a><a href="torch/">t</a>"#), ["gfx110X-all", "gfx120X-all"]);
        assert_eq!(bin_relative("_rocm_sdk_core/bin/amdhip64_7.dll"), Some("amdhip64_7.dll"));
        assert_eq!(bin_relative("_rocm_sdk_libraries_gfx120X_all/bin/rocblas/library/x.co"), Some("rocblas/library/x.co"));
        assert_eq!(bin_relative("_rocm_sdk_core/lib/x.lib"), None);
        assert_eq!(bin_relative("rocm_sdk_core-7.2.1.dist-info/METADATA"), None);
        assert_eq!(bin_relative("_rocm_sdk_core/bin/../evil"), None);
    }
}
