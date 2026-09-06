//! M11 — upstream llama.cpp updates.
//!
//! Check the newest GitHub release, install it side-by-side as an immutable
//! per-tag build directory (never overwriting an existing build), verify it
//! WITHOUT loading a model (`--version` + `--list-devices` only), and
//! promote/roll back profiles onto it.
//!
//! Two install paths:
//! - **prebuilt**: upstream's Windows CPU zip (executables + base DLLs) merged
//!   with the Windows ROCm zip (`ggml-hip.dll` + three bundled HIP runtime
//!   DLLs). Upstream builds that against a newer ROCm than this box runs, and
//!   rocBLAS is resolved from PATH, so whether it loads is an empirical
//!   question answered by `verify_build`, never assumed.
//! - **source**: the configured build script, checkout at the tag, HIP for
//!   this GPU with the local toolchain. Slow but proven.
//!
//! Nothing here launches a server or touches VRAM. Benchmarks stay a separate,
//! explicit step because the GPUs are shared.

use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::devices;
use crate::discovery::{self, Build};
use crate::launch::run_capture;
use crate::profile::Profile;
use crate::{Error, Result};

const RELEASES_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases";
const USER_AGENT: &str = concat!("llama-fidim/", env!("CARGO_PKG_VERSION"));
/// Manifest written into every build directory this module creates.
pub const MANIFEST_NAME: &str = "fidim-build.json";

fn upd(msg: impl Into<String>) -> Error {
    Error::Update(msg.into())
}

// ---------------------------------------------------------------- releases ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Release {
    pub tag: String,
    pub published_at: String,
    pub html_url: String,
    pub assets: Vec<Asset>,
    /// Release title (upstream uses the tag) and body; the body's first
    /// line is the commit subject the tag was cut from.
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub body: String,
}

/// One line of the changelog between the installed build and the latest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReleaseNote {
    pub tag: String,
    pub title: String,
    pub published_at: String,
    pub html_url: String,
}

/// The commit subject out of an upstream release body:
/// `<details open>\n\nserver : fix x (#123)\n\n* details…`.
pub fn subject_from_body(body: &str) -> Option<String> {
    body.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('<') && !l.starts_with('*') && !l.starts_with('-') && !l.starts_with('#'))
        .map(|l| l.to_string())
}

/// Releases with `since < b<n> <= until`, newest first, from a release-list
/// JSON. `complete` is false when the installed tag is older than the list
/// covers (the changelog is then a tail, not the whole story).
pub fn release_notes(list_json: &str, since: Option<u32>, until: Option<u32>) -> (Vec<ReleaseNote>, bool) {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(list_json) else { return (vec![], false) };
    let Some(arr) = v.as_array() else { return (vec![], false) };
    let mut notes: Vec<(u32, ReleaseNote)> = Vec::new();
    let mut saw_since = since.is_none();
    for item in arr {
        let Ok(r) = parse_release(&item.to_string()) else { continue };
        let Some(n) = version_number(&r.tag) else { continue };
        if Some(n) == since {
            saw_since = true;
        }
        if since.is_some_and(|s| n <= s) || until.is_some_and(|u| n > u) {
            continue;
        }
        let title = subject_from_body(&r.body).unwrap_or_else(|| r.name.clone());
        notes.push((n, ReleaseNote { tag: r.tag, title, published_at: r.published_at, html_url: r.html_url }));
    }
    notes.sort_by(|a, b| b.0.cmp(&a.0));
    (notes.into_iter().map(|(_, n)| n).collect(), saw_since)
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(20))
        .timeout_read(Duration::from_secs(120))
        .build()
}

fn get_json(url: &str) -> Result<String> {
    agent()
        .get(url)
        .set("User-Agent", USER_AGENT)
        .set("Accept", "application/vnd.github+json")
        .call()
        .map_err(|e| upd(format!("GitHub API {url}: {e}")))?
        .into_string()
        .map_err(|e| upd(format!("GitHub API {url}: reading body: {e}")))
}

/// Parse a GitHub release object. Pure, so the shape is unit-tested offline.
pub fn parse_release(json: &str) -> Result<Release> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    let tag = v["tag_name"]
        .as_str()
        .ok_or_else(|| upd("release JSON has no tag_name"))?
        .to_string();
    let assets = v["assets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    Some(Asset {
                        name: x["name"].as_str()?.to_string(),
                        url: x["browser_download_url"].as_str()?.to_string(),
                        size: x["size"].as_u64().unwrap_or(0),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Release {
        tag,
        published_at: v["published_at"].as_str().unwrap_or("").to_string(),
        html_url: v["html_url"].as_str().unwrap_or("").to_string(),
        assets,
        name: v["name"].as_str().unwrap_or("").to_string(),
        body: v["body"].as_str().unwrap_or("").to_string(),
    })
}

/// Newest binary release. GitHub's `/releases/latest` points at upstream's
/// stable `vX.Y.Z` marker, which ships no binaries; the `b<n>` tags that do
/// are flagged prerelease and excluded from it. So list recent releases and
/// take the highest `b<n>` that actually carries the Windows CPU asset.
pub fn latest_release() -> Result<Release> {
    let json = get_json(&format!("{RELEASES_API}?per_page=30"))?;
    pick_latest_binary_release(&json)
}

pub fn pick_latest_binary_release(list_json: &str) -> Result<Release> {
    let v: serde_json::Value = serde_json::from_str(list_json)?;
    let arr = v.as_array().ok_or_else(|| upd("release list is not an array"))?;
    let mut best: Option<(u32, Release)> = None;
    for item in arr {
        let Ok(r) = parse_release(&item.to_string()) else { continue };
        let Some(n) = version_number(&r.tag) else { continue };
        let cpu = format!("llama-{}-bin-win-cpu-x64.zip", r.tag);
        if !r.assets.iter().any(|a| a.name == cpu) {
            continue;
        }
        if best.as_ref().map_or(true, |(bn, _)| n > *bn) {
            best = Some((n, r));
        }
    }
    best.map(|(_, r)| r)
        .ok_or_else(|| upd("no b<n> release with a Windows CPU asset among the 30 most recent"))
}

pub fn release_by_tag(tag: &str) -> Result<Release> {
    parse_release(&get_json(&format!("{RELEASES_API}/tags/{tag}"))?)
}

/// `b10769` -> 10769. Upstream tags are `b<n>`; anything else is not comparable.
pub fn version_number(tag: &str) -> Option<u32> {
    tag.strip_prefix('b').and_then(|n| n.parse().ok())
}

/// The two Windows assets a ROCm install needs: the CPU zip carries the
/// executables and base DLLs; the ROCm zip carries only the HIP backend.
/// The ROCm asset is matched by prefix/suffix because the SDK version in its
/// name changes (`rocm-10.0` today).
pub fn select_assets(release: &Release) -> Result<(Asset, Asset)> {
    let cpu_name = format!("llama-{}-bin-win-cpu-x64.zip", release.tag);
    let rocm_prefix = format!("llama-{}-bin-win-rocm-", release.tag);
    let cpu = release
        .assets
        .iter()
        .find(|a| a.name == cpu_name)
        .cloned()
        .ok_or_else(|| upd(format!("release {} has no {cpu_name}", release.tag)))?;
    let rocm = release
        .assets
        .iter()
        .find(|a| a.name.starts_with(&rocm_prefix) && a.name.ends_with("-x64.zip"))
        .cloned()
        .ok_or_else(|| {
            upd(format!(
                "release {} has no Windows ROCm asset ({rocm_prefix}*-x64.zip)",
                release.tag
            ))
        })?;
    Ok((cpu, rocm))
}

// ------------------------------------------------------------------- check ----

#[derive(Debug, Clone, Serialize)]
pub struct InstalledRef {
    pub tag: String,
    pub version: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateCheck {
    pub latest: Release,
    pub newest_installed: Option<InstalledRef>,
    /// Release-count distance; None when either side is not a `b<n>` tag.
    pub behind: Option<u32>,
    pub update_available: bool,
    pub install_dir: PathBuf,
    pub already_installed: bool,
    /// The assets a prebuilt install would download, or why it can't.
    pub assets: Vec<Asset>,
    pub asset_error: Option<String>,
    /// What changed between the newest installed build and the latest.
    #[serde(default)]
    pub changes: Vec<ReleaseNote>,
    /// False when the installed build is older than the fetched list covers.
    #[serde(default)]
    pub changes_complete: bool,
}

pub fn newest_installed(builds: &[Build]) -> Option<InstalledRef> {
    builds
        .iter()
        .filter_map(|b| {
            let v = b.version.as_deref()?;
            Some((version_number(v)?, b, v))
        })
        .max_by_key(|(n, _, _)| *n)
        .map(|(_, b, v)| InstalledRef { tag: b.tag.clone(), version: v.to_string(), path: b.path.clone() })
}

pub fn check(cfg: &Config) -> Result<UpdateCheck> {
    let builds = discovery::scan_builds(&cfg.build_roots_effective(), cfg.rocm_bin.as_deref());
    // One list serves both the latest pick and the changelog.
    let json = get_json(&format!("{RELEASES_API}?per_page=100"))?;
    let latest = pick_latest_binary_release(&json)?;
    let mut c = check_against(cfg, &builds, latest)?;
    let since = c.newest_installed.as_ref().and_then(|n| version_number(&n.version));
    let (notes, complete) = release_notes(&json, since, version_number(&c.latest.tag));
    c.changes = notes;
    c.changes_complete = complete;
    Ok(c)
}

pub fn check_against(cfg: &Config, builds: &[Build], latest: Release) -> Result<UpdateCheck> {
    let newest = newest_installed(builds);
    let behind = match (&newest, version_number(&latest.tag)) {
        (Some(n), Some(l)) => version_number(&n.version).map(|i| l.saturating_sub(i)),
        _ => None,
    };
    let update_available = match (&newest, behind) {
        (None, _) => true,
        (Some(_), Some(b)) => b > 0,
        (Some(n), None) => n.version != latest.tag,
    };
    let install_dir = install_dir(cfg, &latest.tag, "rocm")?;
    let already_installed = install_dir.join("bin").join("llama-server.exe").is_file();
    let (assets, asset_error) = match select_assets(&latest) {
        Ok((c, r)) => (vec![c, r], None),
        Err(e) => (vec![], Some(e.to_string())),
    };
    Ok(UpdateCheck {
        latest,
        newest_installed: newest,
        behind,
        update_available,
        install_dir,
        already_installed,
        assets,
        asset_error,
        changes: vec![],
        changes_complete: true,
    })
}

/// `<install_root>/<tag>-<flavor>`. Immutable: a directory is never reused
/// for a different tag, so profiles pinned to it stay reproducible.
pub fn install_dir(cfg: &Config, tag: &str, flavor: &str) -> Result<PathBuf> {
    let root = cfg
        .install_root
        .clone()
        .or_else(|| cfg.build_roots.first().cloned())
        .unwrap_or_else(|| Config::config_dir().join("builds"));
    Ok(root.join(format!("{tag}-{flavor}")))
}

/// Builds installed before the rename carry `llamactl-build.json`.
pub fn manifest_path(dir: &Path) -> PathBuf {
    let new = dir.join(MANIFEST_NAME);
    if new.is_file() {
        return new;
    }
    let old = dir.join("llamactl-build.json");
    if old.is_file() { old } else { new }
}

// ------------------------------------------------------------------ verify ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedDevice {
    pub index: u32,
    pub backend: String,
    pub name: String,
    pub total_mib: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verify {
    pub version: Option<String>,
    pub commit: Option<String>,
    pub devices: Vec<VerifiedDevice>,
    /// At least one ROCm device enumerated = the HIP backend loaded against
    /// this box's runtime. False means "do not promote onto this build".
    pub hip_ok: bool,
    pub detail: String,
}

/// Run the binary with `--version` and `--list-devices`. Loads the HIP
/// runtime and backend DLLs, exactly like every pre-flight does; allocates
/// nothing on the GPU.
pub fn verify_build(exe: &Path, rocm_bin: Option<&Path>) -> Verify {
    let mut v = Verify { version: None, commit: None, devices: vec![], hip_ok: false, detail: String::new() };
    match run_capture(exe, &["--version"], rocm_bin) {
        Ok(text) => match discovery::parse_version_output(&text) {
            Some((ver, commit)) => {
                v.version = Some(ver);
                v.commit = Some(commit);
            }
            None => v.detail.push_str(&format!("unrecognised --version output: {}\n", text.trim())),
        },
        Err(e) => v.detail.push_str(&format!("--version failed: {e}\n")),
    }
    match run_capture(exe, &["--list-devices"], rocm_bin) {
        Ok(text) => match devices::parse_list_devices(&text) {
            Ok(list) => {
                v.devices = list
                    .iter()
                    .map(|d| VerifiedDevice {
                        index: d.index,
                        backend: d.backend.clone(),
                        name: d.name.clone(),
                        total_mib: d.total_mib,
                    })
                    .collect();
                v.hip_ok = v.devices.iter().any(|d| d.backend.starts_with("ROCm"));
                if !v.hip_ok {
                    v.detail.push_str(
                        "--list-devices ran but enumerated no ROCm device: the HIP backend did not load \
                         against this ROCm runtime. Use the source build instead.\n",
                    );
                    v.detail.push_str(text.trim());
                }
            }
            Err(e) => v.detail.push_str(&format!("--list-devices unparseable: {e}\n{}", text.trim())),
        },
        Err(e) => v.detail.push_str(&format!("--list-devices failed: {e}\n")),
    }
    v
}

// ----------------------------------------------------------------- install ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub tag: String,
    /// `prebuilt` or `source`.
    pub source: String,
    pub installed_at_unix: u64,
    pub assets: Vec<String>,
    pub verify: Verify,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstallReport {
    pub tag: String,
    pub dir: PathBuf,
    pub source: String,
    /// True when the directory already held a server and nothing was fetched.
    pub skipped_existing: bool,
    pub verify: Verify,
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn write_manifest(dir: &Path, m: &Manifest) -> Result<()> {
    let p = dir.join(MANIFEST_NAME);
    std::fs::write(&p, serde_json::to_string_pretty(m)?).map_err(|e| Error::io(&p, e))
}

fn download(asset: &Asset, to: &Path, progress: &mut dyn FnMut(String)) -> Result<()> {
    download_url(&asset.url, &asset.name, to, progress).map(|_| ())
}

/// Stream `url` to `to` with progress every 16 MB; returns bytes written.
pub fn download_url(url: &str, name: &str, to: &Path, progress: &mut dyn FnMut(String)) -> Result<u64> {
    let resp = agent()
        .get(url)
        .set("User-Agent", USER_AGENT)
        .call()
        .map_err(|e| upd(format!("download {name}: {e}")))?;
    let total = resp.header("Content-Length").and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    let mut reader = resp.into_reader();
    let mut file = File::create(to).map_err(|e| Error::io(to, e))?;
    let mut buf = vec![0u8; 1 << 20];
    let mut done: u64 = 0;
    let mut last_report: u64 = 0;
    loop {
        let n = reader.read(&mut buf).map_err(|e| upd(format!("download {name}: {e}")))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| Error::io(to, e))?;
        done += n as u64;
        if done - last_report >= 16 << 20 {
            last_report = done;
            if total > 0 {
                progress(format!("{name}: {} / {} MB", done >> 20, total >> 20));
            } else {
                progress(format!("{name}: {} MB", done >> 20));
            }
        }
    }
    file.flush().map_err(|e| Error::io(to, e))?;
    if total > 0 && done != total {
        return Err(upd(format!("download {name}: got {done} of {total} bytes")));
    }
    progress(format!("{name}: {} MB complete", done >> 20));
    Ok(done)
}

/// Shims that older versions of this tool copied into a build's `bin`
/// shadow every runtime (the exe folder wins DLL resolution). Rename them
/// aside; `runtime::shim_dir` provides per-runtime shims instead. A file
/// that is in use cannot be renamed on some systems; that is left alone.
pub fn retire_build_shims(exe: &Path) {
    let Some(bin) = exe.parent() else { return };
    let Some(dir) = bin.parent() else { return };
    let Ok(text) = std::fs::read_to_string(manifest_path(dir)) else { return };
    let Ok(m) = serde_json::from_str::<Manifest>(&text) else { return };
    for a in &m.assets {
        let Some(rest) = a.strip_prefix("shim:") else { continue };
        let name = rest.split(" <-").next().unwrap_or(rest).trim();
        let p = bin.join(name);
        if p.is_file() {
            let _ = std::fs::rename(&p, bin.join(format!("{name}.retired")));
        }
    }
}

/// Extract every file entry of a zip into `bin/`, flattening nothing: upstream
/// zips are already flat (`7z a ... bin\Release\*`). Later archives overwrite
/// earlier ones, which is what merging the CPU and ROCm zips requires.
fn extract_into(zip_path: &Path, bin: &Path) -> Result<usize> {
    let f = File::open(zip_path).map_err(|e| Error::io(zip_path, e))?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| upd(format!("{}: {e}", zip_path.display())))?;
    let mut count = 0;
    for i in 0..z.len() {
        let mut entry = z.by_index(i).map_err(|e| upd(format!("{}: entry {i}: {e}", zip_path.display())))?;
        if entry.is_dir() {
            continue;
        }
        let Some(rel) = entry.enclosed_name() else { continue };
        let out = bin.join(rel);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let mut dst = File::create(&out).map_err(|e| Error::io(&out, e))?;
        std::io::copy(&mut entry, &mut dst).map_err(|e| Error::io(&out, e))?;
        count += 1;
    }
    Ok(count)
}

/// DLL names referenced by a binary's import table (an ASCII scan — good
/// enough to spot `hipblas.dll`; a PE parser would be overkill here).
pub fn imported_dll_names(bytes: &[u8]) -> Vec<String> {
    let re = regex::bytes::Regex::new(r"(?i)([A-Za-z0-9_\-]+\.dll)").unwrap();
    let mut names: Vec<String> = re
        .captures_iter(bytes)
        .map(|c| String::from_utf8_lossy(&c[1]).to_lowercase())
        .collect();
    names.sort();
    names.dedup();
    names
}

/// Upstream builds the ROCm backend against a newer ROCm whose libraries
/// dropped the `lib` prefix on Windows (`hipblas.dll`); ROCm 7.1 ships
/// `libhipblas.dll`. Same library, different file name. For every DLL the
/// backend imports that exists nowhere on the search path but does exist as
/// `lib<name>` in `rocm_bin`, copy it alongside the backend under the
/// expected name. Returns the shims made, for the manifest and the report —
/// this is an ABI assumption, and the bench is where it gets proven.
pub fn shim_renamed_rocm_dlls(bin: &Path, rocm_bin: Option<&Path>) -> Result<Vec<String>> {
    let Some(rocm) = rocm_bin else { return Ok(vec![]) };
    let backend = bin.join("ggml-hip.dll");
    if !backend.is_file() {
        return Ok(vec![]);
    }
    let bytes = std::fs::read(&backend).map_err(|e| Error::io(&backend, e))?;
    let mut shims = Vec::new();
    for name in imported_dll_names(&bytes) {
        if bin.join(&name).is_file() || rocm.join(&name).is_file() {
            continue;
        }
        let alt = rocm.join(format!("lib{name}"));
        if alt.is_file() {
            let dst = bin.join(&name);
            std::fs::copy(&alt, &dst).map_err(|e| Error::io(&dst, e))?;
            shims.push(format!("{name} <- {}", alt.display()));
        }
    }
    Ok(shims)
}

/// Download + merge the prebuilt Windows zips for `release` into an immutable
/// `<tag>-rocm` directory, then verify. A half-finished directory is removed
/// so a failed install can never masquerade as a build.
pub fn install_prebuilt(
    cfg: &Config,
    release: &Release,
    progress: &mut dyn FnMut(String),
) -> Result<InstallReport> {
    let dir = install_dir(cfg, &release.tag, "rocm")?;
    let bin = dir.join("bin");
    let exe = bin.join("llama-server.exe");
    if exe.is_file() {
        progress(format!("{} already installed at {} — verifying only", release.tag, dir.display()));
        retire_build_shims(&exe);
        let shims: Vec<String> = Vec::new();
        let verify = verify_build(&exe, crate::runtime::default_prepend(cfg, &exe).as_deref());
        // Refresh the manifest: a re-verify after a runtime change (or a
        // shim added by a newer tool) is the record that matters.
        let manifest_path = manifest_path(&dir);
        let mut m: Manifest = std::fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Manifest {
                tag: release.tag.clone(),
                source: "prebuilt".into(),
                installed_at_unix: now_unix(),
                assets: vec![],
                verify: verify.clone(),
            });
        let new_shims: Vec<String> = shims.into_iter().filter(|s| !m.assets.contains(s)).collect();
        m.assets.extend(new_shims);
        m.verify = verify.clone();
        write_manifest(&dir, &m)?;
        return Ok(InstallReport {
            tag: release.tag.clone(),
            dir,
            source: "prebuilt".into(),
            skipped_existing: true,
            verify,
        });
    }
    let (cpu, rocm) = select_assets(release)?;
    let parent = dir.parent().ok_or_else(|| upd("install dir has no parent"))?.to_path_buf();
    let tmp = parent.join(format!(".fidim-download-{}", release.tag));
    std::fs::create_dir_all(&tmp).map_err(|e| Error::io(&tmp, e))?;

    let result = (|| -> Result<Vec<String>> {
        let mut names = Vec::new();
        for a in [&cpu, &rocm] {
            let to = tmp.join(&a.name);
            progress(format!("downloading {} ({} MB)", a.name, a.size >> 20));
            download(a, &to, progress)?;
            names.push(a.name.clone());
        }
        std::fs::create_dir_all(&bin).map_err(|e| Error::io(&bin, e))?;
        for a in [&cpu, &rocm] {
            let n = extract_into(&tmp.join(&a.name), &bin)?;
            progress(format!("extracted {n} files from {}", a.name));
        }
        if !exe.is_file() {
            return Err(upd("archives extracted but bin/llama-server.exe is missing — upstream layout changed"));
        }
        Ok(names)
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    let names = match result {
        Ok(n) => n,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&dir);
            return Err(e);
        }
    };

    progress("verifying: --version and --list-devices (no model load)".into());
    let verify = verify_build(&exe, crate::runtime::default_prepend(cfg, &exe).as_deref());
    write_manifest(
        &dir,
        &Manifest {
            tag: release.tag.clone(),
            source: "prebuilt".into(),
            installed_at_unix: now_unix(),
            assets: names,
            verify: verify.clone(),
        },
    )?;
    Ok(InstallReport { tag: release.tag.clone(), dir, source: "prebuilt".into(), skipped_existing: false, verify })
}

/// Build `tag` from source via the configured script:
/// `script <llama.cpp checkout> <tag> <output dir>`. Stdout lines stream to
/// `progress`; a non-zero exit is an error carrying the script's stderr.
pub fn build_from_source(
    cfg: &Config,
    tag: &str,
    progress: &mut dyn FnMut(String),
) -> Result<InstallReport> {
    // Script beside the running exe (installed layout) or in the repo
    // (target/release), unless config names one.
    let script = cfg
        .source_build_script
        .clone()
        .or_else(|| {
            let exe = std::env::current_exe().ok()?;
            let here = exe.parent()?;
            [here.join("scripts"), here.join("..").join("..").join("scripts")]
                .into_iter()
                .map(|d| d.join("build-from-tag.bat"))
                .find(|p| p.is_file())
        })
        .ok_or_else(|| upd("no build script: set source_build_script in Settings (scripts\\build-from-tag.bat)"))?;
    let src = cfg
        .llama_cpp_source
        .clone()
        .or_else(|| cfg.build_roots.first().cloned())
        .ok_or_else(|| upd("config.llama_cpp_source is not set"))?;
    let dir = install_dir(cfg, tag, "src")?;
    let exe = dir.join("bin").join("llama-server.exe");
    if exe.is_file() {
        progress(format!("{tag} already built at {} — verifying only", dir.display()));
        let verify = verify_build(&exe, crate::runtime::default_prepend(cfg, &exe).as_deref());
        return Ok(InstallReport { tag: tag.into(), dir, source: "source".into(), skipped_existing: true, verify });
    }
    progress(format!("running {} {} {} {}", script.display(), src.display(), tag, dir.display()));
    let mut cmd = Command::new("cmd");
    cmd.arg("/c")
        .arg(&script)
        .arg(&src)
        .arg(tag)
        .arg(&dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    crate::launch::hide_console(&mut cmd);
    let mut child = cmd
        .spawn()
        .map_err(|e| upd(format!("spawn {}: {e}", script.display())))?;
    let stderr = child.stderr.take();
    let err_thread = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(mut e) = stderr {
            let _ = e.read_to_string(&mut s);
        }
        s
    });
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(|l| l.ok()) {
            progress(line);
        }
    }
    let status = child.wait().map_err(|e| upd(format!("wait {}: {e}", script.display())))?;
    let err_text = err_thread.join().unwrap_or_default();
    if !status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        return Err(upd(format!(
            "build script exited with {} — stderr tail:\n{}",
            status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into()),
            tail_lines(&err_text, 30)
        )));
    }
    if !exe.is_file() {
        return Err(upd(format!("script succeeded but {} is missing", exe.display())));
    }
    progress("verifying: --version and --list-devices (no model load)".into());
    let verify = verify_build(&exe, crate::runtime::default_prepend(cfg, &exe).as_deref());
    write_manifest(
        &dir,
        &Manifest {
            tag: tag.into(),
            source: "source".into(),
            installed_at_unix: now_unix(),
            assets: vec![],
            verify: verify.clone(),
        },
    )?;
    Ok(InstallReport { tag: tag.into(), dir, source: "source".into(), skipped_existing: false, verify })
}

fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

// ----------------------------------------------------------------- promote ----

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BuildSnap {
    pub path: PathBuf,
    pub version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromoteEntry {
    pub profile_id: String,
    pub from: BuildSnap,
    pub to: BuildSnap,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromoteBatch {
    pub at_unix: u64,
    pub entries: Vec<PromoteEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PromoteReport {
    pub batch: PromoteBatch,
    /// (profile id, reason) for profiles left untouched.
    pub skipped: Vec<(String, String)>,
}

/// Which profiles a promotion may move.
#[derive(Debug, Clone)]
pub enum PromoteScope {
    /// Only profiles currently on this build directory (the safe default:
    /// "everything that was on the previous newest build").
    FromBuild(PathBuf),
    /// Every unpinned profile.
    All,
    /// Explicit ids.
    Ids(Vec<String>),
}

/// A profile opts out of promotion with `"build_pinned": true` (top-level,
/// preserved through the schema's `extra` map). Use it for anything that
/// must stay on a specific tag, like the MTP drafter profile.
pub fn is_pinned(p: &Profile) -> bool {
    p.extra.get("build_pinned").and_then(|v| v.as_bool()).unwrap_or(false)
}

fn history_path() -> PathBuf {
    Config::config_dir().join("update-history.json")
}

pub fn load_history() -> Result<Vec<PromoteBatch>> {
    let p = history_path();
    if !p.exists() {
        return Ok(vec![]);
    }
    let text = std::fs::read_to_string(&p).map_err(|e| Error::io(&p, e))?;
    Ok(serde_json::from_str(&text)?)
}

fn save_history(h: &[PromoteBatch]) -> Result<()> {
    let p = history_path();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    std::fs::write(&p, serde_json::to_string_pretty(h)?).map_err(|e| Error::io(&p, e))
}

fn same_dir(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| p.to_string_lossy().replace('/', "\\").trim_end_matches('\\').to_lowercase();
    norm(a) == norm(b)
}

/// Re-point profiles onto `to_dir`. Old build directories are never touched,
/// so rollback is a metadata operation. Baselines stay on the profile: the
/// fingerprint includes the build version, so the next bench records fresh
/// numbers rather than trusting the old ones.
pub fn promote(
    cfg: &Config,
    to_dir: &Path,
    to_version: Option<String>,
    scope: PromoteScope,
) -> Result<PromoteReport> {
    if !to_dir.join("bin").join("llama-server.exe").is_file() {
        return Err(upd(format!("{} has no bin/llama-server.exe", to_dir.display())));
    }
    let to = BuildSnap { path: to_dir.to_path_buf(), version: to_version };
    let mut entries = Vec::new();
    let mut skipped = Vec::new();
    for mut p in Profile::load_all(&cfg.profile_dir)? {
        let id = p.id.clone();
        if is_pinned(&p) {
            skipped.push((id, "pinned (build_pinned = true)".into()));
            continue;
        }
        if same_dir(&p.build.path, to_dir) {
            skipped.push((id, "already on this build".into()));
            continue;
        }
        let in_scope = match &scope {
            PromoteScope::FromBuild(d) => same_dir(&p.build.path, d),
            PromoteScope::All => true,
            PromoteScope::Ids(ids) => ids.iter().any(|i| i == &id),
        };
        if !in_scope {
            skipped.push((id, "out of scope".into()));
            continue;
        }
        let from = BuildSnap { path: p.build.path.clone(), version: p.build.version.clone() };
        p.build.path = to.path.clone();
        p.build.version = to.version.clone();
        p.save(&cfg.profile_dir.join(format!("{id}.json")))?;
        entries.push(PromoteEntry { profile_id: id, from, to: to.clone() });
    }
    let batch = PromoteBatch { at_unix: now_unix(), entries };
    if !batch.entries.is_empty() {
        let mut h = load_history()?;
        h.push(batch.clone());
        save_history(&h)?;
    }
    Ok(PromoteReport { batch, skipped })
}

#[derive(Debug, Clone, Serialize)]
pub struct RollbackReport {
    pub batch_at_unix: u64,
    pub restored: Vec<PromoteEntry>,
    /// Profiles that were not restored: deleted, or moved again since.
    pub skipped: Vec<(String, String)>,
}

/// Undo the most recent promotion batch. A profile is restored only if it
/// still sits on the build that batch moved it to; otherwise it was changed
/// by hand since and is left alone.
pub fn rollback(cfg: &Config) -> Result<RollbackReport> {
    let mut h = load_history()?;
    let batch = h.pop().ok_or_else(|| upd("no promotion to roll back"))?;
    let mut restored = Vec::new();
    let mut skipped = Vec::new();
    for e in &batch.entries {
        let path = cfg.profile_dir.join(format!("{}.json", e.profile_id));
        if !path.exists() {
            skipped.push((e.profile_id.clone(), "profile no longer exists".into()));
            continue;
        }
        let mut p = Profile::load(&path)?;
        if !same_dir(&p.build.path, &e.to.path) {
            skipped.push((e.profile_id.clone(), format!("now on {}, not the promoted build", p.build.path.display())));
            continue;
        }
        p.build.path = e.from.path.clone();
        p.build.version = e.from.version.clone();
        p.save(&path)?;
        restored.push(e.clone());
    }
    save_history(&h)?;
    Ok(RollbackReport { batch_at_unix: batch.at_unix, restored, skipped })
}

// ------------------------------------------------------------------- tests ----

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASE_JSON: &str = r#"{
      "tag_name": "b10769", "published_at": "2026-09-03T03:37:00Z",
      "html_url": "https://github.com/ggml-org/llama.cpp/releases/tag/b10769",
      "assets": [
        {"name": "llama-b10769-bin-win-cpu-x64.zip", "browser_download_url": "https://x/cpu.zip", "size": 18350080},
        {"name": "llama-b10769-bin-win-rocm-10.0-x64.zip", "browser_download_url": "https://x/rocm.zip", "size": 244318208},
        {"name": "llama-b10769-bin-win-vulkan-x64.zip", "browser_download_url": "https://x/vk.zip", "size": 1}
      ]}"#;

    #[test]
    fn parses_release_and_selects_the_two_windows_assets() {
        let r = parse_release(RELEASE_JSON).unwrap();
        assert_eq!(r.tag, "b10769");
        assert_eq!(r.assets.len(), 3);
        let (cpu, rocm) = select_assets(&r).unwrap();
        assert_eq!(cpu.name, "llama-b10769-bin-win-cpu-x64.zip");
        assert_eq!(rocm.name, "llama-b10769-bin-win-rocm-10.0-x64.zip");
    }

    #[test]
    fn latest_skips_the_stable_marker_and_assetless_tags() {
        // v0.3.0 is newest by date but has no binaries; b10770 has no CPU zip
        // (still uploading); b10769 is the right answer.
        let list = format!(
            r#"[
              {{"tag_name":"v0.3.0","published_at":"2026-09-03T04:00:00Z","html_url":"","assets":[{{"name":"nightly-tag.txt","browser_download_url":"https://x/n","size":7}}]}},
              {{"tag_name":"b10770","published_at":"2026-09-03T03:50:00Z","html_url":"","assets":[]}},
              {},
              {{"tag_name":"b10760","published_at":"2026-09-02T11:14:00Z","html_url":"","assets":[{{"name":"llama-b10760-bin-win-cpu-x64.zip","browser_download_url":"https://x/c","size":1}}]}}
            ]"#,
            RELEASE_JSON
        );
        let r = pick_latest_binary_release(&list).unwrap();
        assert_eq!(r.tag, "b10769");
        assert!(pick_latest_binary_release("[]").is_err());
    }

    #[test]
    fn rocm_asset_missing_is_an_error_not_a_guess() {
        let mut r = parse_release(RELEASE_JSON).unwrap();
        r.assets.retain(|a| !a.name.contains("rocm"));
        assert!(select_assets(&r).is_err());
    }

    #[test]
    fn version_numbers_compare_and_behind_counts() {
        assert_eq!(version_number("b10769"), Some(10769));
        assert_eq!(version_number("v0.3.0"), None);
        // First-run defaults have no roots; the install dir needs one.
        let mut cfg = Config::default_for_machine();
        cfg.build_roots = vec![PathBuf::from(r"C:")];
        let builds = vec![Build {
            path: PathBuf::from(r"C:\b\build-hip-vision"),
            tag: "build-hip-vision".into(),
            server_exe: PathBuf::from(r"C:\b\build-hip-vision\bin\llama-server.exe"),
            version: Some("b9817".into()),
            commit: None,
            version_error: None,
        }];
        let c = check_against(&cfg, &builds, parse_release(RELEASE_JSON).unwrap()).unwrap();
        assert_eq!(c.behind, Some(952));
        assert!(c.update_available);
        assert!(c.install_dir.ends_with("b10769-rocm"));
        assert_eq!(c.assets.len(), 2);
    }

    #[test]
    fn no_installed_build_means_update_available() {
        let mut cfg = Config::default_for_machine();
        cfg.build_roots = vec![PathBuf::from(r"C:")];
        let c = check_against(&cfg, &[], parse_release(RELEASE_JSON).unwrap()).unwrap();
        assert!(c.newest_installed.is_none());
        assert!(c.update_available);
    }

    #[test]
    fn import_scan_finds_dll_names() {
        let blob = b"\x00\x00amdhip64_7.dll\x00junk\x00HIPBLAS.dll\x00KERNEL32.dll\x00hipblas.dll";
        assert_eq!(imported_dll_names(blob), vec!["amdhip64_7.dll", "hipblas.dll", "kernel32.dll"]);
    }

    #[test]
    fn shim_copies_lib_prefixed_dll_under_the_imported_name() {
        let tmp = std::env::temp_dir().join(format!("fidim-shim-{}", std::process::id()));
        let bin = tmp.join("bin");
        let rocm = tmp.join("rocm");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(&rocm).unwrap();
        std::fs::write(bin.join("ggml-hip.dll"), b"xx hipblas.dll xx amdhip64_7.dll xx").unwrap();
        std::fs::write(bin.join("amdhip64_7.dll"), b"present").unwrap();
        std::fs::write(rocm.join("libhipblas.dll"), b"the real one").unwrap();
        let shims = shim_renamed_rocm_dlls(&bin, Some(&rocm)).unwrap();
        assert_eq!(shims.len(), 1);
        assert!(shims[0].starts_with("hipblas.dll <- "));
        assert_eq!(std::fs::read(bin.join("hipblas.dll")).unwrap(), b"the real one");
        // Idempotent: second pass finds it present and does nothing.
        assert!(shim_renamed_rocm_dlls(&bin, Some(&rocm)).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn same_dir_ignores_case_and_slashes() {
        assert!(same_dir(Path::new(r"C:\A\b\"), Path::new("c:/a/B")));
        assert!(!same_dir(Path::new(r"C:\A\b"), Path::new(r"C:\A\c")));
    }

    #[test]
    fn pinned_flag_lives_in_extra() {
        let mut p: Profile = serde_json::from_value(serde_json::json!({
            "schema": 1, "id": "x", "name": "x",
            "build": { "path": "C:/b" }, "model": { "path": "E:/m.gguf" },
            "devices": [], "server": { "port": 1, "alias": "x" },
            "runtime": { "ctx_total": 1 }
        }))
        .unwrap();
        assert!(!is_pinned(&p));
        p.extra.insert("build_pinned".into(), serde_json::Value::Bool(true));
        assert!(is_pinned(&p));
    }
}
