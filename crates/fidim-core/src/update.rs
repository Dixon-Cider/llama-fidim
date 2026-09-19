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
//! A second channel, **unsloth**, installs unslothai/llama.cpp's Windows ROCm
//! zip: one archive with its own ROCm DLLs and the DiffusionGemma runner.
//! Those builds are recorded as such in the manifest, run with nothing on
//! PATH, and are never picked as "newest" for llama-server profiles. The
//! same zip can also be installed with Llama FIDIM's runner patch laid over
//! it (`overlay`).
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
use crate::discovery::{self, Build, BuildPatch, Channel, RUNNER_EXE};
use crate::launch::run_capture;
use crate::profile::{Engine, Profile};
use crate::{Error, Result};

const RELEASES_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases";
const UNSLOTH_RELEASES_API: &str = "https://api.github.com/repos/unslothai/llama.cpp/releases";
const USER_AGENT: &str = concat!("llama-fidim/", env!("CARGO_PKG_VERSION"));
/// Manifest written into every build directory this module creates.
pub const MANIFEST_NAME: &str = "fidim-build.json";

pub(crate) fn upd(msg: impl Into<String>) -> Error {
    Error::Update(msg.into())
}

// ---------------------------------------------------------------- releases ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Asset {
    pub name: String,
    pub url: String,
    pub size: u64,
    /// GitHub's `sha256:<hex>` of the uploaded file; absent on releases
    /// published before GitHub computed digests.
    #[serde(default)]
    pub digest: Option<String>,
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
    get_json_opt(url)?.ok_or_else(|| upd(format!("GitHub API {url}: 404 Not Found")))
}

/// `get_json`, with a 404 as None: a release that does not exist is an
/// answer, not an error.
pub(crate) fn get_json_opt(url: &str) -> Result<Option<String>> {
    let resp = match agent()
        .get(url)
        .set("User-Agent", USER_AGENT)
        .set("Accept", "application/vnd.github+json")
        .call()
    {
        Ok(r) => r,
        Err(ureq::Error::Status(404, _)) => return Ok(None),
        Err(e) => return Err(upd(format!("GitHub API {url}: {e}"))),
    };
    resp.into_string()
        .map(Some)
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
                        digest: x["digest"].as_str().map(str::to_string),
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

// ----------------------------------------------------------------- unsloth ----

/// `b11027-mix-3e83366` -> (11027, "3e83366"): the upstream build the fork
/// was cut from, plus the hash of what was merged into it. Deliberately not
/// part of `version_number`, so a fork tag never ranks as an upstream release.
pub fn unsloth_tag_parts(tag: &str) -> Option<(u32, String)> {
    let (n, sha) = tag.strip_prefix('b')?.split_once("-mix-")?;
    if n.is_empty() || !n.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if sha.len() < 7 || !sha.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')) {
        return None;
    }
    Some((n.parse().ok()?, sha.to_string()))
}

fn unsloth_rocm_prefix(tag: &str) -> String {
    format!("app-{tag}-windows-x64-rocm-")
}

/// The release's Windows ROCm zips, one per GPU target.
fn unsloth_rocm_assets(r: &Release) -> impl Iterator<Item = &Asset> {
    let prefix = unsloth_rocm_prefix(&r.tag);
    r.assets.iter().filter(move |a| a.name.starts_with(&prefix) && a.name.ends_with(".zip"))
}

/// GPU targets the release ships Windows ROCm zips for (`gfx120X`, ...).
pub fn unsloth_gfx_targets(r: &Release) -> Vec<String> {
    let prefix = unsloth_rocm_prefix(&r.tag);
    unsloth_rocm_assets(r)
        .filter_map(|a| a.name.strip_prefix(&prefix)?.strip_suffix(".zip").map(str::to_string))
        .collect()
}

/// Newest fork release that has Windows ROCm zips, by upstream build number
/// and then publish time (the same base is re-cut when the merged branch
/// moves). Pure, so it is tested against a captured list.
pub fn pick_latest_unsloth_release(list_json: &str) -> Result<Release> {
    let v: serde_json::Value = serde_json::from_str(list_json)?;
    let arr = v.as_array().ok_or_else(|| upd("release list is not an array"))?;
    let mut best: Option<((u32, String), Release)> = None;
    for item in arr {
        let Ok(r) = parse_release(&item.to_string()) else { continue };
        let Some((n, _)) = unsloth_tag_parts(&r.tag) else { continue };
        if unsloth_rocm_assets(&r).next().is_none() {
            continue;
        }
        let key = (n, r.published_at.clone());
        if best.as_ref().map_or(true, |(k, _)| key > *k) {
            best = Some((key, r));
        }
    }
    best.map(|(_, r)| r).ok_or_else(|| {
        upd("no b<n>-mix-<sha> release with a Windows ROCm zip among the 30 most recent unslothai/llama.cpp releases")
    })
}

pub fn latest_unsloth_release() -> Result<Release> {
    pick_latest_unsloth_release(&get_json(&format!("{UNSLOTH_RELEASES_API}?per_page=30"))?)
}

pub fn unsloth_release_by_tag(tag: &str) -> Result<Release> {
    if unsloth_tag_parts(tag).is_none() {
        return Err(upd(format!("`{tag}` is not an Unsloth release tag (b<n>-mix-<sha>)")));
    }
    parse_release(&get_json(&format!("{UNSLOTH_RELEASES_API}/tags/{tag}"))?)
}

/// The GPU target in the fork's asset names: the override, else the
/// configured ROCm family, else a guess from the card names, else RDNA4.
/// AMD's family names end in `-all` (`gfx120X-all`); the fork's do not.
pub fn unsloth_gfx(cfg: &Config, device_names: &[String], override_: Option<&str>) -> String {
    let nonempty = |s: &String| !s.trim().is_empty();
    let pick = override_
        .map(str::to_string)
        .filter(nonempty)
        .or_else(|| cfg.rocm_family.clone().filter(nonempty))
        .or_else(|| crate::rocm::guess_family(device_names))
        .unwrap_or_else(|| "gfx120X".to_string());
    let pick = pick.trim();
    pick.strip_suffix("-all").unwrap_or(pick).to_string()
}

/// The Windows ROCm zip for `gfx`, by its whole name: a prefix match could
/// pick another card's kernels. Case is ignored (`gfx120x` from a hand-typed
/// `--gfx`). The error names every Windows ROCm zip the release has.
pub fn select_unsloth_asset(r: &Release, gfx: &str) -> Result<Asset> {
    let want = format!("{}{gfx}.zip", unsloth_rocm_prefix(&r.tag));
    if let Some(a) = r.assets.iter().find(|a| a.name.eq_ignore_ascii_case(&want)) {
        return Ok(a.clone());
    }
    let names: Vec<&str> = unsloth_rocm_assets(r).map(|a| a.name.as_str()).collect();
    Err(upd(format!(
        "release {} has no {want}; its Windows ROCm zips are: {}",
        r.tag,
        if names.is_empty() { "none".to_string() } else { names.join(", ") }
    )))
}

// ------------------------------------------------------------------- check ----

#[derive(Debug, Clone, Serialize)]
pub struct InstalledRef {
    pub tag: String,
    pub version: String,
    pub path: PathBuf,
    /// The build's `patch.name`: a patched build shares its base's version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patch: Option<String>,
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

/// Newest upstream build. Fork builds report the upstream number they were
/// cut from, often higher than anything upstream installed here, and would
/// otherwise become every default (check, promote, router, devices).
pub fn newest_installed(builds: &[Build]) -> Option<InstalledRef> {
    builds
        .iter()
        .filter(|b| b.channel == Channel::Upstream)
        .filter_map(|b| {
            let v = b.version.as_deref()?;
            Some((version_number(v)?, b, v))
        })
        .max_by_key(|(n, _, _)| *n)
        .map(|(_, b, v)| InstalledRef {
            tag: b.tag.clone(),
            version: v.to_string(),
            path: b.path.clone(),
            patch: b.patch.as_ref().map(|x| x.label().to_string()),
        })
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
/// for a different tag, so profiles pinned to it stay reproducible. Every
/// channel (prebuilt, source, Unsloth) and the GUI come through here, so
/// this is where Unsloth Studio's tree is refused: the first build root can
/// be Studio's own llama.cpp folder.
pub fn install_dir(cfg: &Config, tag: &str, flavor: &str) -> Result<PathBuf> {
    let root = cfg
        .install_root
        .clone()
        .or_else(|| cfg.build_roots.first().cloned())
        .unwrap_or_else(|| Config::config_dir().join("builds"));
    let dir = root.join(format!("{tag}-{flavor}"));
    refuse_unsloth_studio_tree(&dir)?;
    Ok(dir)
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Verify {
    pub version: Option<String>,
    pub commit: Option<String>,
    pub devices: Vec<VerifiedDevice>,
    /// At least one ROCm device enumerated = the HIP backend loaded against
    /// this box's runtime. False means "do not promote onto this build".
    pub hip_ok: bool,
    pub detail: String,
    /// `bin/<RUNNER_EXE>` exists. Only checked for fork builds; the runner
    /// is never started here (a probe would load the GPU runtime for nothing).
    #[serde(default)]
    pub runner_present: bool,
}

/// Run the binary with `--version` and `--list-devices`. Loads the HIP
/// runtime and backend DLLs, exactly like every pre-flight does; allocates
/// nothing on the GPU.
pub fn verify_build(exe: &Path, rocm_bin: Option<&Path>) -> Verify {
    retire_build_shims(exe);
    let mut v = Verify {
        version: None,
        commit: None,
        devices: vec![],
        hip_ok: false,
        detail: String::new(),
        runner_present: false,
    };
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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub tag: String,
    /// `prebuilt`, `source`, `unsloth-prebuilt` or `unsloth-overlay`.
    pub source: String,
    pub installed_at_unix: u64,
    pub assets: Vec<String>,
    pub verify: Verify,
    /// Absent = upstream. The only thing that makes a build a fork build
    /// (`discovery::read_build_meta`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub channel: Option<Channel>,
    /// The build carries its own ROCm DLLs and runs with no PATH prefix.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub bundled_runtime: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_tag: Option<String>,
    /// Lowercase hex of the zip that was extracted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gfx_target: Option<String>,
    /// Set only on a build assembled outside FIDIM's installers (see
    /// `discovery::BuildPatch`). Kept through every re-verify rewrite; a
    /// malformed block reads as none.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "discovery::lenient_patch")]
    pub patch: Option<BuildPatch>,
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

pub(crate) fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

pub(crate) fn write_manifest(dir: &Path, m: &Manifest) -> Result<()> {
    let p = dir.join(MANIFEST_NAME);
    std::fs::write(&p, serde_json::to_string_pretty(m)?).map_err(|e| Error::io(&p, e))
}

pub(crate) fn download(asset: &Asset, to: &Path, progress: &mut dyn FnMut(String)) -> Result<()> {
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
                channel: None,
                bundled_runtime: false,
                release_tag: None,
                asset_sha256: None,
                gfx_target: None,
                patch: None,
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
            channel: None,
            bundled_runtime: false,
            release_tag: None,
            asset_sha256: None,
            gfx_target: None,
            patch: None,
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
            channel: None,
            bundled_runtime: false,
            release_tag: None,
            asset_sha256: None,
            gfx_target: None,
            patch: None,
        },
    )?;
    Ok(InstallReport { tag: tag.into(), dir, source: "source".into(), skipped_existing: false, verify })
}

fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

pub(crate) fn read_manifest(dir: &Path) -> Option<Manifest> {
    serde_json::from_str(&std::fs::read_to_string(manifest_path(dir)).ok()?).ok()
}

// --------------------------------------------------------- unsloth install ----

/// Lowercase hex SHA-256 of a file, streamed: the fork's zip is ~500 MB.
pub fn sha256_file(p: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = File::open(p).map_err(|e| Error::io(p, e))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| Error::io(p, e))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// A path as comparable parts: the drive/root anchor, then each component
/// lowercased, with `.` and `..` resolved lexically (the install directory
/// does not exist yet, so it cannot be canonicalized).
fn path_parts(p: &Path) -> (String, Vec<String>) {
    use std::path::{Component, Prefix};
    let mut anchor = String::new();
    let mut parts: Vec<String> = Vec::new();
    for c in p.components() {
        match c {
            Component::Prefix(pre) => {
                anchor.push_str(&match pre.kind() {
                    Prefix::Disk(d) | Prefix::VerbatimDisk(d) => format!("{}:", (d as char).to_ascii_lowercase()),
                    _ => pre.as_os_str().to_string_lossy().to_lowercase(),
                });
            }
            Component::RootDir => anchor.push('\\'),
            Component::CurDir => {}
            Component::ParentDir => {
                parts.pop();
            }
            Component::Normal(s) => parts.push(s.to_string_lossy().to_lowercase()),
        }
    }
    (anchor, parts)
}

/// `dir` is `root` or inside it, by whole components: `.unsloth2` is not
/// inside `.unsloth`.
fn is_under(dir: &Path, root: &Path) -> bool {
    let (da, dp) = path_parts(dir);
    let (ra, rp) = path_parts(root);
    da == ra && !rp.is_empty() && dp.starts_with(&rp)
}

/// Unsloth Studio keeps its own llama.cpp build under `%USERPROFILE%\.unsloth`
/// and updates it itself. Nothing here may write into that tree.
fn refuse_unsloth_studio_tree(dir: &Path) -> Result<()> {
    match std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
        Some(home) => refuse_studio_tree_at(dir, &PathBuf::from(home).join(".unsloth")),
        None => Ok(()),
    }
}

fn refuse_studio_tree_at(dir: &Path, studio: &Path) -> Result<()> {
    if is_under(dir, studio) {
        return Err(upd(format!(
            "{} is inside {}: Llama FIDIM never writes into Unsloth Studio's folder; set a different install_root",
            dir.display(),
            studio.display()
        )));
    }
    Ok(())
}

/// What the manifest records about a fork install.
#[derive(Debug, Clone)]
pub struct UnslothMeta {
    pub tag: String,
    pub asset_name: String,
    pub sha256: Option<String>,
    pub gfx: String,
}

/// Classic Windows MAX_PATH less the terminator. The fork's zip nests
/// rocBLAS/hipBLASLt kernel files ~140 characters deep; a deep install root
/// would otherwise fail part-way through extraction, or later at load.
const MAX_PATH_CHARS: usize = 259;

fn path_chars(p: &Path) -> usize {
    p.as_os_str().to_string_lossy().chars().count()
}

fn longest_entry(zip_path: &Path) -> Result<PathBuf> {
    let f = File::open(zip_path).map_err(|e| Error::io(zip_path, e))?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| upd(format!("{}: {e}", zip_path.display())))?;
    let mut longest = PathBuf::new();
    for i in 0..z.len() {
        let entry = z.by_index(i).map_err(|e| upd(format!("{}: entry {i}: {e}", zip_path.display())))?;
        if let Some(rel) = entry.enclosed_name() {
            if path_chars(&rel) > path_chars(&longest) {
                longest = rel;
            }
        }
    }
    Ok(longest)
}

/// Extract a downloaded fork zip into `tmp/bin`, write the manifest, and
/// move `tmp` into place as `final_dir`. Network-free, so tests drive it
/// directly. Any failure removes `tmp`; `final_dir` only ever appears
/// complete.
pub fn install_unsloth_from_zip(zip: &Path, tmp: &Path, final_dir: &Path, meta: UnslothMeta) -> Result<PathBuf> {
    let result = stage_unsloth(zip, tmp, final_dir, &meta);
    if result.is_err() {
        let _ = std::fs::remove_dir_all(tmp);
    }
    result
}

fn stage_unsloth(zip: &Path, tmp: &Path, final_dir: &Path, meta: &UnslothMeta) -> Result<PathBuf> {
    stage_unsloth_base(zip, tmp, final_dir, &meta.asset_name)?;
    // Before the move, or 500 MB of zip would live on inside the build.
    let _ = std::fs::remove_file(zip);
    finish_unsloth_stage(
        tmp,
        final_dir,
        &Manifest {
            tag: meta.tag.clone(),
            source: "unsloth-prebuilt".into(),
            installed_at_unix: now_unix(),
            // Never a `shim:` entry: retire_build_shims renames exactly those,
            // and a bundle's own hipblas.dll must stay where it is.
            assets: vec![meta.asset_name.clone()],
            verify: Verify::default(),
            channel: Some(Channel::Unsloth),
            bundled_runtime: true,
            release_tag: Some(meta.tag.clone()),
            asset_sha256: meta.sha256.clone(),
            gfx_target: Some(meta.gfx.clone()),
            patch: None,
        },
    )
}

/// The part every fork install shares: refuse an existing `final_dir` or a
/// path too deep for the zip, unpack the zip into `tmp/bin`, and require the
/// runner and llama-server at its top level. Returns `tmp/bin`. The caller
/// removes `tmp` on failure.
pub(crate) fn stage_unsloth_base(zip: &Path, tmp: &Path, final_dir: &Path, asset_name: &str) -> Result<PathBuf> {
    if final_dir.exists() {
        return Err(upd(format!(
            "{} already exists but holds no complete build; remove it and install again",
            final_dir.display()
        )));
    }
    let longest = longest_entry(zip)?;
    // Files land under tmp first, then live under final_dir: both must fit.
    for base in [tmp, final_dir] {
        let p = base.join("bin").join(&longest);
        if path_chars(&p) > MAX_PATH_CHARS {
            return Err(upd(format!(
                "install_root path too long for this build's hipblaslt files ({} characters for {}); \
                 choose a shorter install_root",
                path_chars(&p),
                p.display()
            )));
        }
    }
    let bin = tmp.join("bin");
    extract_into(zip, &bin)?;
    for need in [RUNNER_EXE, "llama-server.exe"] {
        if !bin.join(need).is_file() {
            return Err(upd(format!("{asset_name} has no {need} at its top level: the fork's Windows layout changed")));
        }
    }
    Ok(bin)
}

/// Write the manifest into `tmp` and move `tmp` into place as `final_dir`,
/// which only ever appears complete.
pub(crate) fn finish_unsloth_stage(tmp: &Path, final_dir: &Path, manifest: &Manifest) -> Result<PathBuf> {
    write_manifest(tmp, manifest)?;
    if let Some(parent) = final_dir.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    // Antivirus scanners hold freshly written DLLs open for a moment, and a
    // directory with an open file cannot be renamed.
    let mut retries = 0;
    loop {
        match std::fs::rename(tmp, final_dir) {
            Ok(()) => break,
            Err(_) if retries < 3 => {
                retries += 1;
                std::thread::sleep(Duration::from_millis(700));
            }
            Err(e) => return Err(Error::io(final_dir, e)),
        }
    }
    Ok(final_dir.to_path_buf())
}

/// `verify_build` with nothing on PATH, so the bundle's own ROCm DLLs load
/// exactly as they will at launch, plus whether the runner is there. The
/// runner itself is not started.
pub fn verify_unsloth(dir: &Path) -> Verify {
    let bin = dir.join("bin");
    let mut v = verify_build(&bin.join("llama-server.exe"), None);
    v.runner_present = bin.join(RUNNER_EXE).is_file();
    if !v.runner_present {
        v.detail.push_str(&format!("bin has no {RUNNER_EXE}: this build cannot run diffusion profiles\n"));
    }
    v
}

/// Download one fork release's Windows ROCm zip for `gfx`, check it against
/// GitHub's digest, install it as `<install_root>/<tag>-unsloth`, and verify
/// it. An existing install is re-verified instead. Never starts the runner
/// or loads a model.
pub fn install_unsloth(
    cfg: &Config,
    release: &Release,
    gfx: &str,
    progress: &mut dyn FnMut(String),
) -> Result<InstallReport> {
    // install_dir refuses Unsloth Studio's own tree.
    let dir = install_dir(cfg, &release.tag, "unsloth")?;
    if dir.join("bin").join(RUNNER_EXE).is_file() {
        progress(format!("{} already installed at {} — verifying only", release.tag, dir.display()));
        let verify = verify_unsloth(&dir);
        let mut m = read_manifest(&dir).unwrap_or_else(|| Manifest {
            tag: release.tag.clone(),
            source: "unsloth-prebuilt".into(),
            installed_at_unix: now_unix(),
            assets: vec![],
            verify: Verify::default(),
            channel: Some(Channel::Unsloth),
            bundled_runtime: true,
            release_tag: Some(release.tag.clone()),
            asset_sha256: None,
            gfx_target: None,
            patch: None,
        });
        m.verify = verify.clone();
        write_manifest(&dir, &m)?;
        return Ok(InstallReport {
            tag: release.tag.clone(),
            dir,
            source: "unsloth-prebuilt".into(),
            skipped_existing: true,
            verify,
        });
    }
    // Checked before a 500 MB download that could only end in a failed rename.
    if dir.exists() {
        return Err(upd(format!("{} exists but has no {RUNNER_EXE}; remove it, then install again", dir.display())));
    }
    let (n, _) = unsloth_tag_parts(&release.tag)
        .ok_or_else(|| upd(format!("`{}` is not an Unsloth release tag (b<n>-mix-<sha>)", release.tag)))?;
    let asset = select_unsloth_asset(release, gfx)?;
    let root = dir.parent().ok_or_else(|| upd("install dir has no parent"))?;
    // Dot-prefixed, so a scan never offers a half-extracted build.
    let tmp = root.join(format!(".fidim-tmp-{n}"));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| Error::io(&tmp, e))?;
    let zip = tmp.join(&asset.name);

    let staged = (|| -> Result<PathBuf> {
        progress(format!("downloading {} ({} MB)", asset.name, asset.size >> 20));
        download(&asset, &zip, progress)?;
        let sha = sha256_file(&zip)?;
        match asset.digest.as_deref().and_then(|d| d.strip_prefix("sha256:")) {
            Some(want) if !want.eq_ignore_ascii_case(&sha) => {
                return Err(upd(format!(
                    "sha256 mismatch for {}: GitHub publishes {want}, the download is {sha}",
                    asset.name
                )));
            }
            Some(_) => progress(format!("sha256 verified ({}…)", &sha[..16])),
            None => progress(format!("{}: no digest published; integrity not verified", asset.name)),
        }
        progress(format!("extracting into {}", dir.display()));
        install_unsloth_from_zip(
            &zip,
            &tmp,
            &dir,
            UnslothMeta {
                tag: release.tag.clone(),
                asset_name: asset.name.clone(),
                sha256: Some(sha),
                gfx: gfx.to_string(),
            },
        )
    })();
    if let Err(e) = staged {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(e);
    }

    progress("verifying: --version and --list-devices on the bundled ROCm (no model load)".into());
    let verify = verify_unsloth(&dir);
    let recorded = (|| -> Result<()> {
        let mut m = read_manifest(&dir).ok_or_else(|| upd(format!("{}: manifest unreadable after install", dir.display())))?;
        m.verify = verify.clone();
        write_manifest(&dir, &m)
    })();
    if let Err(e) = recorded {
        // This call created the directory; without its manifest it would scan
        // as an upstream build.
        let _ = std::fs::remove_dir_all(&dir);
        return Err(e);
    }
    Ok(InstallReport { tag: release.tag.clone(), dir, source: "unsloth-prebuilt".into(), skipped_existing: false, verify })
}

#[derive(Debug, Clone, Serialize)]
pub struct UnslothCheck {
    pub latest: Release,
    /// The upstream release the fork build was cut from (`b<n>`).
    pub upstream_tag: Option<String>,
    pub gfx: String,
    /// Every GPU target the release has a Windows ROCm zip for.
    pub gfx_available: Vec<String>,
    pub asset: Option<Asset>,
    pub asset_error: Option<String>,
    pub install_dir: PathBuf,
    pub already_installed: bool,
    /// Fork builds already installed, newest first.
    pub installed: Vec<InstalledRef>,
    /// The runner-patch overlay for this release (see `overlay`): where it
    /// is looked for, whether it is published, and where it installs.
    pub overlay_repo: String,
    pub overlay_patch: String,
    pub overlay_available: bool,
    /// The overlay zip, when published.
    pub overlay_asset: Option<Asset>,
    /// Why the overlay is not available, when it is not a plain "not
    /// published" (the lookup failed, the release lacks an asset).
    pub overlay_error: Option<String>,
    pub overlay_install_dir: PathBuf,
    pub overlay_installed: bool,
}

/// Latest (or `tag`) fork release against what is installed, and whether
/// the runner-patch overlay is published for it. Network and a build scan;
/// the comparison itself is `check_unsloth_against`.
pub fn check_unsloth(
    cfg: &Config,
    device_names: &[String],
    gfx_override: Option<&str>,
    tag: Option<&str>,
) -> Result<UnslothCheck> {
    let release = match tag {
        Some(t) => unsloth_release_by_tag(t)?,
        None => latest_unsloth_release()?,
    };
    let builds = discovery::scan_builds(&cfg.build_roots_effective(), cfg.rocm_bin.as_deref());
    let mut c = check_unsloth_against(cfg, &builds, release, &unsloth_gfx(cfg, device_names, gfx_override))?;
    let lookup = crate::overlay::lookup(cfg, &c.latest.tag);
    crate::overlay::apply_lookup(&mut c, lookup);
    Ok(c)
}

/// The overlay fields say "not looked up" here; `overlay::apply_lookup`
/// fills them from a lookup.
pub fn check_unsloth_against(cfg: &Config, builds: &[Build], latest: Release, gfx: &str) -> Result<UnslothCheck> {
    let install_dir = install_dir(cfg, &latest.tag, "unsloth")?;
    let already_installed = install_dir.join("bin").join(RUNNER_EXE).is_file();
    let overlay_install_dir = crate::overlay::overlay_install_dir(cfg, &latest.tag)?;
    let overlay_installed = overlay_install_dir.join("bin").join(RUNNER_EXE).is_file();
    let (asset, asset_error) = match select_unsloth_asset(&latest, gfx) {
        Ok(a) => (Some(a), None),
        Err(e) => (None, Some(e.to_string())),
    };
    let mut installed: Vec<(u32, InstalledRef)> = builds
        .iter()
        .filter(|b| b.channel == Channel::Unsloth)
        .map(|b| {
            let n = b.release_tag.as_deref().and_then(unsloth_tag_parts).map_or(0, |(n, _)| n);
            let version = b.version.clone().or_else(|| b.release_tag.clone()).unwrap_or_else(|| "?".into());
            (n, InstalledRef { tag: b.tag.clone(), version, path: b.path.clone(), patch: b.patch.as_ref().map(|x| x.label().to_string()) })
        })
        .collect();
    installed.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(UnslothCheck {
        upstream_tag: unsloth_tag_parts(&latest.tag).map(|(n, _)| format!("b{n}")),
        gfx: gfx.to_string(),
        gfx_available: unsloth_gfx_targets(&latest),
        latest,
        asset,
        asset_error,
        install_dir,
        already_installed,
        installed: installed.into_iter().map(|(_, r)| r).collect(),
        overlay_repo: crate::overlay::overlay_repo(cfg),
        overlay_patch: crate::overlay::OVERLAY_PATCH.to_string(),
        overlay_available: false,
        overlay_asset: None,
        overlay_error: None,
        overlay_install_dir,
        overlay_installed,
    })
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

/// What a promotion target can run, from its files and manifest.
#[derive(Debug, Clone)]
pub struct PromoteTarget {
    pub channel: Channel,
    pub has_llama_server: bool,
    pub has_runner: bool,
    /// The runner features the target's manifest `patch` declares (none
    /// for an unpatched build).
    pub features: Vec<String>,
}

pub fn promote_target(dir: &Path) -> PromoteTarget {
    let bin = dir.join("bin");
    let meta = discovery::read_build_meta(dir);
    PromoteTarget {
        channel: meta.channel,
        has_llama_server: bin.join("llama-server.exe").is_file(),
        has_runner: bin.join(RUNNER_EXE).is_file(),
        features: meta.patch.map(|p| p.features).unwrap_or_default(),
    }
}

/// Why `p` stays where it is, or None to move it. Engines never cross: a
/// llama-server profile does not land on a fork build by promotion (it is
/// the same llama-server with unreviewed patches; the editor can still pick
/// it), and a diffusion profile only lands on a build with the runner.
pub fn promote_skip_reason(p: &Profile, to_dir: &Path, t: &PromoteTarget, scope: &PromoteScope) -> Option<String> {
    if is_pinned(p) {
        return Some("pinned (build_pinned = true)".into());
    }
    if same_dir(&p.build.path, to_dir) {
        return Some("already on this build".into());
    }
    let in_scope = match scope {
        PromoteScope::FromBuild(d) => same_dir(&p.build.path, d),
        PromoteScope::All => true,
        PromoteScope::Ids(ids) => ids.iter().any(|i| i == &p.id),
    };
    if !in_scope {
        return Some("out of scope".into());
    }
    match p.engine {
        Engine::Unknown => Some("unknown engine".into()),
        Engine::LlamaServer if !t.has_llama_server => Some("target build has no llama-server.exe".into()),
        Engine::LlamaServer if t.channel == Channel::Unsloth => {
            Some("Unsloth fork build: pick it in the editor if wanted".into())
        }
        Engine::DiffusionGemma if !t.has_runner => Some("target build has no diffusion runner".into()),
        // A patched runner's profile (FA on, a context only the patch can
        // hold) would break on a runner without those features: never swap
        // it onto one by promotion.
        Engine::DiffusionGemma => {
            let patch = discovery::read_build_meta(&p.build.path).patch?;
            let missing: Vec<&str> =
                patch.features.iter().filter(|f| !t.features.contains(f)).map(String::as_str).collect();
            (!missing.is_empty()).then(|| {
                format!(
                    "on a patched runner build ({}) whose features the target lacks ({}); pick the new build in \
                     the editor if wanted",
                    patch.label(),
                    missing.join(", ")
                )
            })
        }
        _ => None,
    }
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
    let target = promote_target(to_dir);
    if !target.has_llama_server && !target.has_runner {
        return Err(upd(format!("{} has neither bin/llama-server.exe nor bin/{RUNNER_EXE}", to_dir.display())));
    }
    let to = BuildSnap { path: to_dir.to_path_buf(), version: to_version };
    let mut entries = Vec::new();
    let mut skipped = Vec::new();
    for mut p in Profile::load_all(&cfg.profile_dir)? {
        let id = p.id.clone();
        if let Some(why) = promote_skip_reason(&p, to_dir, &target, &scope) {
            skipped.push((id, why));
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
            channel: discovery::Channel::Upstream,
            bundled_runtime: false,
            release_tag: None,
            patch: None,
            runner_exe: None,
        }];
        let c = check_against(&cfg, &builds, parse_release(RELEASE_JSON).unwrap()).unwrap();
        assert_eq!(c.behind, Some(952));
        assert!(c.update_available);
        assert!(c.install_dir.ends_with("b10769-rocm"));
        assert_eq!(c.assets.len(), 2);
    }

    /// Studio's llama.cpp folder as the first build root, no install_root:
    /// every channel refuses to install into it, not only the Unsloth one.
    #[test]
    fn install_dir_refuses_the_studio_tree_for_every_channel() {
        let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) else { return };
        let mut cfg = Config::default_for_machine();
        cfg.install_root = None;
        cfg.build_roots = vec![PathBuf::from(home).join(".unsloth").join("llama.cpp")];
        for flavor in ["rocm", "src", "unsloth"] {
            let e = install_dir(&cfg, "b10769", flavor).unwrap_err().to_string();
            assert!(e.contains("never writes into Unsloth Studio's folder"), "{flavor}: {e}");
        }
        assert!(check_against(&cfg, &[], parse_release(RELEASE_JSON).unwrap()).is_err());
        // An install_root elsewhere wins over the build root.
        cfg.install_root = Some(PathBuf::from(r"C:\fidim-builds"));
        assert_eq!(install_dir(&cfg, "b10769", "rocm").unwrap(), PathBuf::from(r"C:\fidim-builds\b10769-rocm"));
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

    // ------------------------------------------------------------ unsloth ----

    /// Trimmed from `gh api repos/unslothai/llama.cpp/releases?per_page=2`
    /// on 2026-09-18 (4 of 43 assets each, body cut to its first line).
    const UNSLOTH_CAPTURED: &str = r#"[
     {"tag_name": "b11027-mix-3e83366", "name": "llama.cpp prebuilt b11027-mix-3e83366",
      "published_at": "2026-09-18T00:20:28Z", "prerelease": false,
      "html_url": "https://github.com/unslothai/llama.cpp/releases/tag/b11027-mix-3e83366",
      "body": "Automated Unsloth llama.cpp CUDA + ROCm + Vulkan + macOS + CPU prebuild for upstream [b11027](https://github.com/ggml-org/llama.cpp/releases/tag/b11027), merged with:",
      "assets": [
       {"name": "app-b11027-mix-3e83366-linux-x64-rocm-gfx120X.tar.gz", "size": 747871241,
        "digest": "sha256:051856f2d29f495f43aee1a65ab80d0addea58c68fdf22e04b0a7720a3acc69c",
        "browser_download_url": "https://github.com/unslothai/llama.cpp/releases/download/b11027-mix-3e83366/app-b11027-mix-3e83366-linux-x64-rocm-gfx120X.tar.gz"},
       {"name": "app-b11027-mix-3e83366-windows-x64-cpu.zip", "size": 19087137,
        "digest": "sha256:05826b60f32cf17705cb2297e7337fdffba53a9c63fa4024526ffa409bf66187",
        "browser_download_url": "https://github.com/unslothai/llama.cpp/releases/download/b11027-mix-3e83366/app-b11027-mix-3e83366-windows-x64-cpu.zip"},
       {"name": "app-b11027-mix-3e83366-windows-x64-rocm-gfx1151.zip", "size": 99967059,
        "digest": "sha256:f49512815050e7c384f5398b9db522dcd0c380d21568212acc9303a7ac201e10",
        "browser_download_url": "https://github.com/unslothai/llama.cpp/releases/download/b11027-mix-3e83366/app-b11027-mix-3e83366-windows-x64-rocm-gfx1151.zip"},
       {"name": "app-b11027-mix-3e83366-windows-x64-rocm-gfx120X.zip", "size": 494370803,
        "digest": "sha256:09135ea01882040460a1eda0f301ca28effd270fdf661c3697d29119e2234011",
        "browser_download_url": "https://github.com/unslothai/llama.cpp/releases/download/b11027-mix-3e83366/app-b11027-mix-3e83366-windows-x64-rocm-gfx120X.zip"}
      ]},
     {"tag_name": "b11007-mix-3e83366", "name": "llama.cpp prebuilt b11007-mix-3e83366",
      "published_at": "2026-09-17T03:23:57Z", "prerelease": false,
      "html_url": "https://github.com/unslothai/llama.cpp/releases/tag/b11007-mix-3e83366",
      "body": "Automated Unsloth llama.cpp CUDA + ROCm + Vulkan + macOS + CPU prebuild for upstream [b11007](https://github.com/ggml-org/llama.cpp/releases/tag/b11007), merged with:",
      "assets": [
       {"name": "app-b11007-mix-3e83366-linux-x64-rocm-gfx120X.tar.gz", "size": 747851953,
        "digest": "sha256:ff7eaed0d3ecf362df32d570377ee7477bce541afc09da4448af65a6ebe60a41",
        "browser_download_url": "https://github.com/unslothai/llama.cpp/releases/download/b11007-mix-3e83366/app-b11007-mix-3e83366-linux-x64-rocm-gfx120X.tar.gz"},
       {"name": "app-b11007-mix-3e83366-windows-x64-cpu.zip", "size": 19085971,
        "digest": "sha256:e725a5ad8981f39a94572a92fa6862275decba2067da94863759ae45f152821d",
        "browser_download_url": "https://github.com/unslothai/llama.cpp/releases/download/b11007-mix-3e83366/app-b11007-mix-3e83366-windows-x64-cpu.zip"},
       {"name": "app-b11007-mix-3e83366-windows-x64-rocm-gfx1151.zip", "size": 99965671,
        "digest": "sha256:6b87947649a00984c61fe16e60b58560cc86b38611e8e010b845878c9bad0d55",
        "browser_download_url": "https://github.com/unslothai/llama.cpp/releases/download/b11007-mix-3e83366/app-b11007-mix-3e83366-windows-x64-rocm-gfx1151.zip"},
       {"name": "app-b11007-mix-3e83366-windows-x64-rocm-gfx120X.zip", "size": 494369385,
        "digest": "sha256:4dcb342f7d04a373eb4927c76202589682bc055750d49514dc2e3a3a11440754",
        "browser_download_url": "https://github.com/unslothai/llama.cpp/releases/download/b11007-mix-3e83366/app-b11007-mix-3e83366-windows-x64-rocm-gfx120X.zip"}
      ]}
    ]"#;

    fn captured(i: usize) -> Release {
        let v: serde_json::Value = serde_json::from_str(UNSLOTH_CAPTURED).unwrap();
        parse_release(&v[i].to_string()).unwrap()
    }

    #[test]
    fn unsloth_parsing() {
        assert_eq!(unsloth_tag_parts("b11027-mix-3e83366"), Some((11027, "3e83366".to_string())));
        assert_eq!(version_number("b11027-mix-3e83366"), None, "a fork tag never ranks as upstream");
        for bad in ["b11027", "v0.3.0", "b-mix-3e83366", "b11027-mix-3e8336", "b11027-mix-3E83366", "b11027-mix-3e83366x", "b+1-mix-3e83366"] {
            assert_eq!(unsloth_tag_parts(bad), None, "{bad}");
        }

        let r = captured(0);
        assert_eq!(r.tag, "b11027-mix-3e83366");
        let win = r.assets.iter().find(|a| a.name.ends_with("gfx120X.zip")).unwrap();
        assert_eq!(win.digest.as_deref(), Some("sha256:09135ea01882040460a1eda0f301ca28effd270fdf661c3697d29119e2234011"));
        assert_eq!(win.size, 494370803);
        // Upstream fixtures predate digests: absent, not an error.
        assert!(parse_release(RELEASE_JSON).unwrap().assets.iter().all(|a| a.digest.is_none()));

        // Newest by upstream number, then publish time; a newer tag without a
        // Windows ROCm zip and non-fork tags are skipped. Upstream's picker
        // must not accept the fork list either.
        let v: serde_json::Value = serde_json::from_str(UNSLOTH_CAPTURED).unwrap();
        let list = serde_json::json!([
            {"tag_name": "b11030-mix-3e83366", "published_at": "2026-09-18T09:00:00Z", "html_url": "",
             "assets": [{"name": "app-b11030-mix-3e83366-linux-x64-rocm-gfx120X.tar.gz", "browser_download_url": "https://x/l", "size": 1}]},
            {"tag_name": "nightly", "published_at": "2026-09-19T00:00:00Z", "html_url": "",
             "assets": [{"name": "app-nightly-windows-x64-rocm-gfx120X.zip", "browser_download_url": "https://x/n", "size": 1}]},
            v[1],
            {"tag_name": "b11027-mix-0aa0aa0", "published_at": "2026-09-17T12:00:00Z", "html_url": "",
             "assets": [{"name": "app-b11027-mix-0aa0aa0-windows-x64-rocm-gfx120X.zip", "browser_download_url": "https://x/o", "size": 1}]},
            v[0],
        ])
        .to_string();
        assert_eq!(pick_latest_unsloth_release(&list).unwrap().tag, "b11027-mix-3e83366");
        assert!(pick_latest_binary_release(&list).is_err());
        assert!(pick_latest_unsloth_release("[]").is_err());
        assert!(pick_latest_unsloth_release(&format!("[{RELEASE_JSON}]")).is_err());

        let a = select_unsloth_asset(&r, "gfx120X").unwrap();
        assert_eq!(a.name, "app-b11027-mix-3e83366-windows-x64-rocm-gfx120X.zip");
        assert_eq!(select_unsloth_asset(&r, "gfx120x").unwrap().name, a.name);
        let e = select_unsloth_asset(&r, "gfx1201").unwrap_err().to_string();
        assert!(e.contains("app-b11027-mix-3e83366-windows-x64-rocm-gfx1201.zip"), "{e}");
        assert!(e.contains("app-b11027-mix-3e83366-windows-x64-rocm-gfx1151.zip"), "{e}");
        assert!(e.contains("app-b11027-mix-3e83366-windows-x64-rocm-gfx120X.zip"), "{e}");
        assert!(!e.contains("linux") && !e.contains("cpu"), "{e}");
        assert_eq!(unsloth_gfx_targets(&r), vec!["gfx1151", "gfx120X"]);

        let mut cfg = Config::default_for_machine();
        cfg.rocm_family = None;
        let r9700 = vec!["AMD Radeon AI PRO R9700".to_string(), "AMD Radeon(TM) Graphics".to_string()];
        assert_eq!(unsloth_gfx(&cfg, &r9700, None), "gfx120X", "guessed gfx120X-all loses its -all");
        assert_eq!(unsloth_gfx(&cfg, &[], None), "gfx120X", "fallback");
        assert_eq!(unsloth_gfx(&cfg, &r9700, Some("gfx1151")), "gfx1151", "override wins");
        assert_eq!(unsloth_gfx(&cfg, &r9700, Some("  ")), "gfx120X", "blank override is no override");
        cfg.rocm_family = Some("gfx110X-all".into());
        assert_eq!(unsloth_gfx(&cfg, &r9700, None), "gfx110X", "config beats the guess");
        assert_eq!(unsloth_gfx(&cfg, &r9700, Some("gfx120X-all")), "gfx120X");
    }

    fn test_build(tag: &str, version: &str, channel: Channel) -> Build {
        let path = PathBuf::from(format!(r"C:\b\{tag}"));
        Build {
            server_exe: path.join("bin").join("llama-server.exe"),
            path,
            tag: tag.into(),
            version: Some(version.into()),
            commit: None,
            version_error: None,
            channel,
            bundled_runtime: channel == Channel::Unsloth,
            release_tag: (channel == Channel::Unsloth).then(|| tag.trim_end_matches("-unsloth").to_string()),
            patch: None,
            runner_exe: None,
        }
    }

    fn test_profile(id: &str, engine: Option<&str>, build: &str) -> Profile {
        let mut v = serde_json::json!({
            "schema": 1, "id": id, "name": id,
            "build": { "path": build }, "model": { "path": "E:/m.gguf" },
            "devices": [], "server": { "port": 1, "alias": id },
            "runtime": { "ctx_total": 0 }
        });
        if let Some(e) = engine {
            v["engine"] = e.into();
        }
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn channel_filters_and_promote_rules() {
        let builds = vec![
            test_build("b10819-rocm", "b10819", Channel::Upstream),
            test_build("b11027-mix-3e83366-unsloth", "b11027", Channel::Unsloth),
            test_build("b9817-src", "b9817", Channel::Upstream),
        ];
        let n = newest_installed(&builds).unwrap();
        assert_eq!(n.tag, "b10819-rocm", "the fork's higher b-number is not an upstream release");
        assert!(newest_installed(&builds[1..2]).is_none());

        let mut cfg = Config::default_for_machine();
        cfg.install_root = Some(std::env::temp_dir().join(format!("fidim-unsloth-check-{}", std::process::id())));
        let c = check_unsloth_against(&cfg, &builds, captured(0), "gfx120X").unwrap();
        assert_eq!(c.upstream_tag.as_deref(), Some("b11027"));
        assert_eq!(c.asset.as_ref().unwrap().name, "app-b11027-mix-3e83366-windows-x64-rocm-gfx120X.zip");
        assert!(c.asset_error.is_none());
        assert!(c.install_dir.ends_with("b11027-mix-3e83366-unsloth"));
        assert!(!c.already_installed);
        let tags: Vec<&str> = c.installed.iter().map(|i| i.tag.as_str()).collect();
        assert_eq!(tags, vec!["b11027-mix-3e83366-unsloth"], "only fork builds are listed");
        let c = check_unsloth_against(&cfg, &builds, captured(0), "gfx1201").unwrap();
        assert!(c.asset.is_none() && c.asset_error.is_some());

        let unsloth_dir = Path::new(r"C:\b\b11027-mix-3e83366-unsloth");
        let upstream_dir = Path::new(r"C:\b\b10819-rocm");
        let fork = PromoteTarget { channel: Channel::Unsloth, has_llama_server: true, has_runner: true, features: vec![] };
        let plain = PromoteTarget { channel: Channel::Upstream, has_llama_server: true, has_runner: false, features: vec![] };
        let upstream_with_runner = PromoteTarget { channel: Channel::Upstream, has_llama_server: true, has_runner: true, features: vec![] };
        let all = PromoteScope::All;

        let llama = test_profile("dd", None, r"C:\b\b9817-src");
        let dg = test_profile("dg-26b", Some("diffusion-gemma"), r"C:\b\b11007-mix-3e83366-unsloth");
        let odd = test_profile("typo", Some("difusion"), r"C:\b\b9817-src");
        assert_eq!(odd.engine, Engine::Unknown);

        let skip = |p: &Profile, to: &Path, t: &PromoteTarget, s: &PromoteScope| promote_skip_reason(p, to, t, s);
        assert_eq!(skip(&llama, unsloth_dir, &fork, &all).as_deref(), Some("Unsloth fork build: pick it in the editor if wanted"));
        assert_eq!(skip(&llama, upstream_dir, &plain, &all), None);
        assert_eq!(skip(&dg, upstream_dir, &plain, &all).as_deref(), Some("target build has no diffusion runner"));
        assert_eq!(skip(&dg, unsloth_dir, &fork, &all), None, "a diffusion profile moves onto a fork build");
        assert_eq!(skip(&dg, upstream_dir, &upstream_with_runner, &all), None, "the runner decides, not the channel");
        assert_eq!(skip(&odd, unsloth_dir, &fork, &all).as_deref(), Some("unknown engine"));
        assert_eq!(skip(&odd, upstream_dir, &plain, &all).as_deref(), Some("unknown engine"));
        let runner_only = PromoteTarget { channel: Channel::Upstream, has_llama_server: false, has_runner: true, features: vec![] };
        assert_eq!(skip(&llama, upstream_dir, &runner_only, &all).as_deref(), Some("target build has no llama-server.exe"));

        let mut pinned = dg.clone();
        pinned.extra.insert("build_pinned".into(), serde_json::Value::Bool(true));
        assert_eq!(skip(&pinned, unsloth_dir, &fork, &all).as_deref(), Some("pinned (build_pinned = true)"));
        let mut there = dg.clone();
        there.build.path = PathBuf::from(r"c:/B/b11027-mix-3e83366-unsloth/");
        assert_eq!(skip(&there, unsloth_dir, &fork, &all).as_deref(), Some("already on this build"));
        let from = PromoteScope::FromBuild(PathBuf::from(r"C:\b\b10819-rocm"));
        assert_eq!(skip(&dg, unsloth_dir, &fork, &from).as_deref(), Some("out of scope"));
        assert_eq!(skip(&dg, unsloth_dir, &fork, &PromoteScope::Ids(vec!["dg-26b".into()])), None);

        // promote_target reads the manifest for the channel, files for the rest.
        let root = std::env::temp_dir().join(format!("fidim-promote-target-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bin = root.join("bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("llama-server.exe"), b"").unwrap();
        let t = promote_target(&root);
        assert_eq!((t.channel, t.has_llama_server, t.has_runner), (Channel::Upstream, true, false));
        std::fs::write(bin.join(RUNNER_EXE), b"").unwrap();
        assert_eq!(promote_target(&root).channel, Channel::Upstream, "the runner alone never makes a fork build");
        std::fs::write(root.join(MANIFEST_NAME), r#"{"tag":"t","source":"unsloth-prebuilt","installed_at_unix":1,"assets":[],"verify":{"version":null,"commit":null,"devices":[],"hip_ok":false,"detail":""},"channel":"unsloth","bundled_runtime":true}"#).unwrap();
        let t = promote_target(&root);
        assert_eq!((t.channel, t.has_llama_server, t.has_runner), (Channel::Unsloth, true, true));
        assert!(t.features.is_empty());
        std::fs::remove_dir_all(root).ok();

        // A diffusion profile on a patched runner build never moves onto an
        // unpatched one by promotion; onto another patched build it may.
        let root = std::env::temp_dir().join(format!("fidim-promote-patched-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let patched_dir = root.join("b11027-mix-3e83366-unsloth-dgpatch");
        std::fs::create_dir_all(patched_dir.join("bin")).unwrap();
        std::fs::write(patched_dir.join("bin").join("llama-server.exe"), b"").unwrap();
        std::fs::write(patched_dir.join("bin").join(RUNNER_EXE), b"").unwrap();
        std::fs::write(
            patched_dir.join(MANIFEST_NAME),
            r#"{"tag":"t","source":"unsloth-local-patch","installed_at_unix":1,"assets":[],"verify":{"version":null,"commit":null,"devices":[],"hip_ok":false,"detail":""},"channel":"unsloth","bundled_runtime":true,"patch":{"name":"dgpatch","features":["dg-fa-pad","dg-fa-turn-sizing"]}}"#,
        )
        .unwrap();
        assert_eq!(promote_target(&patched_dir).features, ["dg-fa-pad", "dg-fa-turn-sizing"]);
        let mut on_patch = dg.clone();
        on_patch.build.path = patched_dir.clone();
        assert_eq!(
            skip(&on_patch, unsloth_dir, &fork, &all).as_deref(),
            Some(
                "on a patched runner build (dgpatch) whose features the target lacks (dg-fa-pad, \
                 dg-fa-turn-sizing); pick the new build in the editor if wanted"
            )
        );
        // A patch block that lacks one of them is not enough either...
        let partial = PromoteTarget { features: vec!["dg-fa-pad".into()], ..fork.clone() };
        assert!(skip(&on_patch, Path::new(r"C:\b\other-dgpatch"), &partial, &all).unwrap().contains("(dg-fa-turn-sizing)"));
        // ...one that carries them all (or more) is.
        let superset = PromoteTarget {
            features: vec!["dg-fa-pad".into(), "dg-fa-turn-sizing".into(), "dg-swa-ring".into()],
            ..fork.clone()
        };
        assert_eq!(skip(&on_patch, Path::new(r"C:\b\other-dgpatch"), &superset, &all), None);
        assert_eq!(skip(&dg, &patched_dir, &promote_target(&patched_dir), &all), None, "moving onto a patch is fine");
        std::fs::remove_dir_all(root).ok();
    }

    /// A zip laid out like the fork's Windows ROCm app zip: flat, with the
    /// ROCm DLLs and kernel folders beside the executables.
    fn write_test_zip(path: &Path, with_runner: bool) {
        use zip::write::SimpleFileOptions;
        let f = File::create(path).unwrap();
        let mut z = zip::ZipWriter::new(f);
        let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        let mut names = vec!["llama-server.exe", "hipblas.dll", "rocblas.dll", "amdhip64_7.dll", "rocblas/library/x"];
        if with_runner {
            names.push(RUNNER_EXE);
        }
        for name in names {
            z.start_file(name, opts).unwrap();
            z.write_all(format!("contents of {name}").as_bytes()).unwrap();
        }
        z.finish().unwrap();
    }

    fn meta(tag: &str) -> UnslothMeta {
        UnslothMeta {
            tag: tag.into(),
            asset_name: format!("app-{tag}-windows-x64-rocm-gfx120X.zip"),
            sha256: Some("00ff".into()),
            gfx: "gfx120X".into(),
        }
    }

    #[test]
    fn install_unsloth_from_zip_offline() {
        let root = std::env::temp_dir().join(format!("fidim-unsloth-install-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let tag = "b11027-mix-3e83366";
        let tmp = root.join(".fidim-tmp-11027");
        let fin = root.join(format!("{tag}-unsloth"));
        std::fs::create_dir_all(&tmp).unwrap();
        let zip = tmp.join(format!("app-{tag}-windows-x64-rocm-gfx120X.zip"));
        write_test_zip(&zip, true);

        let out = install_unsloth_from_zip(&zip, &tmp, &fin, meta(tag)).unwrap();
        assert_eq!(out, fin);
        assert!(!tmp.exists(), "staging dir renamed away");
        let bin = fin.join("bin");
        for f in ["llama-server.exe", RUNNER_EXE, "hipblas.dll", "rocblas.dll", "amdhip64_7.dll"] {
            assert!(bin.join(f).is_file(), "{f}");
        }
        assert_eq!(std::fs::read(bin.join("rocblas").join("library").join("x")).unwrap(), b"contents of rocblas/library/x");
        assert!(!fin.join(zip.file_name().unwrap()).exists(), "the zip does not move into the build");

        let text = std::fs::read_to_string(fin.join(MANIFEST_NAME)).unwrap();
        assert!(!text.starts_with('\u{feff}'));
        let m: Manifest = serde_json::from_str(&text).unwrap();
        assert_eq!(m.channel, Some(Channel::Unsloth));
        assert!(m.bundled_runtime);
        assert_eq!(m.release_tag.as_deref(), Some(tag));
        assert_eq!(m.asset_sha256.as_deref(), Some("00ff"));
        assert_eq!(m.gfx_target.as_deref(), Some("gfx120X"));
        assert_eq!(m.source, "unsloth-prebuilt");
        assert!(m.assets.iter().all(|a| !a.starts_with("shim:")), "{:?}", m.assets);
        let bm = discovery::read_build_meta(&fin);
        assert_eq!((bm.channel, bm.bundled_runtime), (Channel::Unsloth, true));

        // Studio safety: the shim retirer must leave a bundle's own hipblas
        // alone, whether FIDIM installed it or it is Studio's manifest-less
        // tree. Only a manifest-listed shim is renamed.
        retire_build_shims(&bin.join("llama-server.exe"));
        assert!(bin.join("hipblas.dll").is_file());
        assert!(!bin.join("hipblas.dll.retired").exists());
        let studio_bin = root.join("studio").join("build").join("bin");
        std::fs::create_dir_all(&studio_bin).unwrap();
        std::fs::write(studio_bin.join("llama-server.exe"), b"").unwrap();
        std::fs::write(studio_bin.join("hipblas.dll"), b"studio").unwrap();
        retire_build_shims(&studio_bin.join("llama-server.exe"));
        assert!(studio_bin.join("hipblas.dll").is_file());
        let shimmed = root.join("b10771-rocm");
        std::fs::create_dir_all(shimmed.join("bin")).unwrap();
        std::fs::write(shimmed.join("bin").join("hipblas.dll"), b"shim").unwrap();
        std::fs::write(
            shimmed.join(MANIFEST_NAME),
            r#"{"tag":"b10771","source":"prebuilt","installed_at_unix":1,"assets":["shim:hipblas.dll <- C:\\rocm\\libhipblas.dll"],
                "verify":{"version":null,"commit":null,"devices":[],"hip_ok":true,"detail":""}}"#,
        )
        .unwrap();
        retire_build_shims(&shimmed.join("bin").join("llama-server.exe"));
        assert!(shimmed.join("bin").join("hipblas.dll.retired").is_file(), "the retirer does work on listed shims");

        // An existing target is never overwritten.
        let tmp2 = root.join(".fidim-tmp-2");
        std::fs::create_dir_all(&tmp2).unwrap();
        let zip2 = tmp2.join("a.zip");
        write_test_zip(&zip2, true);
        assert!(install_unsloth_from_zip(&zip2, &tmp2, &fin, meta(tag)).is_err());
        assert!(!tmp2.exists());
        assert!(bin.join(RUNNER_EXE).is_file(), "the installed build is untouched");

        // No runner: an error, and nothing left behind.
        let tmp3 = root.join(".fidim-tmp-3");
        let fin3 = root.join("b3-mix-0000000-unsloth");
        std::fs::create_dir_all(&tmp3).unwrap();
        let zip3 = tmp3.join("a.zip");
        write_test_zip(&zip3, false);
        let e = install_unsloth_from_zip(&zip3, &tmp3, &fin3, meta("b3-mix-0000000")).unwrap_err().to_string();
        assert!(e.contains(RUNNER_EXE), "{e}");
        assert!(!tmp3.exists() && !fin3.exists());

        // Too deep for MAX_PATH: refused before anything is extracted.
        let tmp4 = root.join(".fidim-tmp-4");
        let fin4 = root.join("x".repeat(240));
        std::fs::create_dir_all(&tmp4).unwrap();
        let zip4 = tmp4.join("a.zip");
        write_test_zip(&zip4, true);
        let e = install_unsloth_from_zip(&zip4, &tmp4, &fin4, meta(tag)).unwrap_err().to_string();
        assert!(e.contains("install_root path too long"), "{e}");
        assert!(!tmp4.exists() && !fin4.exists());

        // Studio's tree is refused by whole components, any case, after `..`.
        let home = root.join("home");
        let studio = home.join(".unsloth");
        assert!(refuse_studio_tree_at(&studio.join("x"), &studio).is_err());
        assert!(refuse_studio_tree_at(&studio, &studio).is_err());
        assert!(refuse_studio_tree_at(&PathBuf::from(studio.to_string_lossy().to_uppercase()).join("x"), &studio).is_err());
        assert!(refuse_studio_tree_at(&home.join("other").join("..").join(".unsloth").join("x"), &studio).is_err());
        assert!(refuse_studio_tree_at(&home.join(".unsloth2").join("x"), &studio).is_ok());
        assert!(refuse_studio_tree_at(&home.join("builds"), &studio).is_ok());
        assert!(refuse_studio_tree_at(&home.join(".unsloth").join("..").join("builds"), &studio).is_ok());

        let f = root.join("abc.bin");
        std::fs::write(&f, b"abc").unwrap();
        assert_eq!(sha256_file(&f).unwrap(), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        std::fs::write(&f, b"").unwrap();
        assert_eq!(sha256_file(&f).unwrap(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        std::fs::remove_dir_all(root).ok();
    }
}
