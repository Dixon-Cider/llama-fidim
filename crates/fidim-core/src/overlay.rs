//! The runner-patch overlay: an Unsloth Windows ROCm build whose llama-level
//! binaries (llama.dll, llama-common.dll, mtmd.dll, the tools and the
//! DiffusionGemma runner) are replaced by ones built from the same release's
//! source with the dgpatch5 patch (packaging/dg-overlay). Unsloth's ggml,
//! HIP backend and ROCm DLLs stay exactly as shipped.
//!
//! An overlay only fits the release it was built from: llama.dll compiles
//! ggml's enums and struct layouts in from the headers. Its descriptor
//! (`fidim-overlay.json`) pins the release tag, the source commit and the
//! sha256 of every Windows ROCm zip of that release, and lists every file of
//! the overlay zip with its sha256 and size. An install checks all of it,
//! writes only llama-level binaries into `bin` (never ggml, HIP or ROCm
//! files), and records the patch in the manifest: discovery labels the
//! build, and the estimator, pre-flight and promotion follow its features.
//!
//! Published overlays are releases `dgpatch5-<unsloth tag>` of
//! `OVERLAY_REPO` (`overlay_repo` in config.json points elsewhere). A folder
//! built with packaging/dg-overlay/scripts/build-local.ps1 installs the same
//! way, so an overlay can be validated before it is published.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::discovery::{BuildPatch, Channel, RUNNER_EXE};
use crate::update::{self, upd, Asset, InstallReport, Manifest, Release, UnslothCheck, Verify};
use crate::{Error, Result};

/// Where overlay releases are published.
pub const OVERLAY_REPO: &str = "Dixon-Cider/fidim-dg-overlay";
/// config.json key that points FIDIM at another overlay repository.
pub const OVERLAY_REPO_KEY: &str = "overlay_repo";
/// The patch this version of FIDIM installs. Release tags are
/// `<patch>-<unsloth tag>`, install folders `<unsloth tag>-unsloth-<patch>`.
pub const OVERLAY_PATCH: &str = "dgpatch5";
pub const DESCRIPTOR_NAME: &str = "fidim-overlay.json";
/// The only base an overlay is built for.
pub const BASE_REPO: &str = "unslothai/llama.cpp";
/// Where an installed overlay keeps the license texts it ships.
pub const LICENSES_DIR: &str = "overlay-licenses";
/// `Manifest::source` of an overlay install.
pub const SOURCE: &str = "unsloth-overlay";

/// The overlay repository: config.json's `overlay_repo` when it is an
/// `owner/name`, else `OVERLAY_REPO`.
pub fn overlay_repo(cfg: &Config) -> String {
    cfg.extra
        .get(OVERLAY_REPO_KEY)
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| is_repo_name(s))
        .map(str::to_string)
        .unwrap_or_else(|| OVERLAY_REPO.to_string())
}

fn is_repo_name(s: &str) -> bool {
    let ok = |p: &str| !p.is_empty() && p.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c));
    matches!(s.split_once('/'), Some((o, n)) if ok(o) && ok(n) && !n.contains('/'))
}

pub fn overlay_release_tag(tag: &str) -> String {
    format!("{OVERLAY_PATCH}-{tag}")
}

pub fn overlay_zip_name(tag: &str) -> String {
    format!("fidim-dg-overlay-{tag}-windows-x64.zip")
}

/// `<install_root>/<tag>-unsloth-dgpatch5`, beside the plain `-unsloth` one.
pub fn overlay_install_dir(cfg: &Config, tag: &str) -> Result<PathBuf> {
    update::install_dir(cfg, tag, &format!("unsloth-{OVERLAY_PATCH}"))
}

// ------------------------------------------------------------- descriptor ----

/// `fidim-overlay.json`, schema 1 (packaging/dg-overlay/DESCRIPTOR.md).
/// Unknown fields are ignored, so later producers can add some.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Descriptor {
    pub schema: u32,
    /// The patch, e.g. `dgpatch5`.
    pub name: String,
    pub base: DescriptorBase,
    pub patch: DescriptorPatch,
    /// Every file in the overlay zip.
    pub files: Vec<OverlayFile>,
    /// Informational: toolchain, flags, the workflow run.
    #[serde(default)]
    pub build: serde_json::Value,
    /// Informational: what the overlay imports from each ggml DLL.
    #[serde(default)]
    pub ggml_imports: BTreeMap<String, Vec<String>>,
    /// The Authenticode subject the binaries are signed with, or none.
    #[serde(default)]
    pub signer: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DescriptorBase {
    pub repo: String,
    pub release_tag: String,
    /// The full commit the release's source tarball is.
    pub source_commit: String,
    #[serde(default)]
    pub source_asset: Option<String>,
    #[serde(default)]
    pub source_sha256: Option<String>,
    #[serde(default)]
    pub ggml_tree: Option<String>,
    /// GPU target -> sha256 of the release's `app-<tag>-windows-x64-rocm-<gfx>.zip`.
    pub zips: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DescriptorPatch {
    pub file: String,
    pub sha256: String,
    /// `discovery::dg_feature` names.
    #[serde(default)]
    pub features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OverlayFile {
    /// The zip entry: a binary at the top level, or `licenses/<name>`.
    pub name: String,
    pub sha256: String,
    pub size: u64,
}

pub fn parse_descriptor(text: &str) -> Result<Descriptor> {
    serde_json::from_str(text.trim_start_matches('\u{feff}'))
        .map_err(|e| upd(format!("{DESCRIPTOR_NAME} does not parse: {e}")))
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit())
}

/// A file name with nothing that could leave the folder it is written to.
fn is_plain_name(n: &str) -> bool {
    !n.is_empty() && n.len() <= 128 && !n.starts_with('.') && n.bytes().all(|c| c.is_ascii_alphanumeric() || b"._+-".contains(&c))
}

/// A binary the overlay may write into `bin`: llama-level only. ggml, the
/// HIP backend and the ROCm runtime (amd*, hip*, roc*, origami*, lib*) stay
/// Unsloth's.
pub fn overlay_binary_allowed(name: &str) -> bool {
    const KEEP: [&str; 6] = ["ggml", "amd", "hip", "roc", "origami", "lib"];
    let n = name.to_ascii_lowercase();
    if !is_plain_name(name) || KEEP.iter().any(|p| n.starts_with(p)) {
        return false;
    }
    (n.starts_with("llama") && (n.ends_with(".exe") || n.ends_with(".dll"))) || n == "mtmd.dll"
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// Goes into `bin`.
    Binary,
    /// A license text, kept in `LICENSES_DIR`.
    License,
}

/// Where an overlay zip entry may go, or why it may not.
pub fn classify_entry(name: &str) -> Result<EntryKind> {
    if let Some(rest) = name.strip_prefix("licenses/") {
        return if is_plain_name(rest) {
            Ok(EntryKind::License)
        } else {
            Err(upd(format!("overlay entry {name}: licenses/ holds plain files only")))
        };
    }
    if !is_plain_name(name) {
        return Err(upd(format!("overlay entry {name}: not a plain file at the top level")));
    }
    if !overlay_binary_allowed(name) {
        return Err(upd(format!(
            "overlay entry {name} is not a llama-level binary: ggml, HIP and ROCm files stay Unsloth's"
        )));
    }
    Ok(EntryKind::Binary)
}

/// The descriptor's sha256 for the release's Windows ROCm zip of `gfx`.
pub fn base_zip_sha<'a>(d: &'a Descriptor, gfx: &str) -> Result<&'a str> {
    d.base
        .zips
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(gfx))
        .map(|(_, v)| v.as_str())
        .ok_or_else(|| {
            upd(format!(
                "the {} overlay lists no {gfx} zip of {}; it fits {}",
                d.name,
                d.base.release_tag,
                d.base.zips.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })
}

/// Everything about a descriptor that can be checked before a download:
/// the schema, the patch this FIDIM installs, the base release and GPU
/// target, and a file list that only holds files an overlay may carry,
/// including the runner and llama-server.
pub fn check_descriptor(d: &Descriptor, tag: &str, gfx: &str) -> Result<()> {
    if d.schema != 1 {
        return Err(upd(format!("{DESCRIPTOR_NAME} has schema {}; this Llama FIDIM reads schema 1", d.schema)));
    }
    if d.name != OVERLAY_PATCH {
        return Err(upd(format!("the overlay carries {}; this Llama FIDIM installs {OVERLAY_PATCH}", d.name)));
    }
    if d.base.repo != BASE_REPO {
        return Err(upd(format!("the overlay is built on {}, not {BASE_REPO}", d.base.repo)));
    }
    if d.base.release_tag != tag {
        return Err(upd(format!("the overlay was built for {}, not {tag}", d.base.release_tag)));
    }
    let c = &d.base.source_commit;
    if c.len() < 9 || !c.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(upd(format!("the overlay names no source commit ({c:?})")));
    }
    let zip_sha = base_zip_sha(d, gfx)?;
    if !is_sha256(zip_sha) || !is_sha256(&d.patch.sha256) || !is_plain_name(&d.patch.file) {
        return Err(upd(format!("{DESCRIPTOR_NAME} is malformed (base zip or patch entry)")));
    }
    let mut names = BTreeSet::new();
    for f in &d.files {
        classify_entry(&f.name)?;
        if !is_sha256(&f.sha256) {
            return Err(upd(format!("{DESCRIPTOR_NAME}: {} has no sha256", f.name)));
        }
        if !names.insert(f.name.to_ascii_lowercase()) {
            return Err(upd(format!("{DESCRIPTOR_NAME} lists {} twice", f.name)));
        }
    }
    for need in [RUNNER_EXE, "llama-server.exe", "llama.dll"] {
        if !names.contains(&need.to_ascii_lowercase()) {
            return Err(upd(format!("the overlay has no {need}")));
        }
    }
    Ok(())
}

/// `llama-server --version` prints the commit it was built from, shortened:
/// it must be a prefix of the descriptor's source commit.
pub fn commit_matches(printed: &str, source_commit: &str) -> bool {
    printed.len() >= 7 && source_commit.to_ascii_lowercase().starts_with(&printed.to_ascii_lowercase())
}

// ---------------------------------------------------------------- staging ----

/// What an install records about its two zips.
#[derive(Debug, Clone)]
pub struct OverlayMeta {
    pub tag: String,
    pub gfx: String,
    /// The Unsloth zip: its release asset name and the sha256 of the file used.
    pub base_asset: String,
    pub base_sha256: String,
    pub overlay_asset: String,
}

/// Copy `from` into a new file at `to`, returning its sha256 and size.
fn copy_hashed(from: &mut dyn Read, to: &Path) -> Result<(String, u64)> {
    use sha2::{Digest, Sha256};
    let mut out = File::create(to).map_err(|e| Error::io(to, e))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut size = 0u64;
    loop {
        let n = from.read(&mut buf).map_err(|e| Error::io(to, e))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        out.write_all(&buf[..n]).map_err(|e| Error::io(to, e))?;
        size += n as u64;
    }
    out.flush().map_err(|e| Error::io(to, e))?;
    Ok((h.finalize().iter().map(|b| format!("{b:02x}")).collect(), size))
}

fn open_zip(p: &Path) -> Result<zip::ZipArchive<File>> {
    let f = File::open(p).map_err(|e| Error::io(p, e))?;
    zip::ZipArchive::new(f).map_err(|e| upd(format!("{}: {e}", p.display())))
}

/// The overlay zip's entries against the descriptor, before anything is
/// written: every entry an allowed file the descriptor lists, every listed
/// file present once.
fn check_overlay_zip(zip_path: &Path, d: &Descriptor) -> Result<()> {
    let listed: BTreeSet<&str> = d.files.iter().map(|f| f.name.as_str()).collect();
    let mut z = open_zip(zip_path)?;
    let mut seen = BTreeSet::new();
    for i in 0..z.len() {
        let e = z.by_index(i).map_err(|e| upd(format!("{}: entry {i}: {e}", zip_path.display())))?;
        if e.is_dir() {
            continue;
        }
        let name = e.name().to_string();
        classify_entry(&name)?;
        if !listed.contains(name.as_str()) {
            return Err(upd(format!("the overlay zip holds {name}, which {DESCRIPTOR_NAME} does not list")));
        }
        if !seen.insert(name.clone()) {
            return Err(upd(format!("the overlay zip holds {name} twice")));
        }
    }
    if let Some(missing) = listed.iter().find(|n| !seen.contains(**n)) {
        return Err(upd(format!("{DESCRIPTOR_NAME} lists {missing}, which the overlay zip does not hold")));
    }
    Ok(())
}

/// Unpack the overlay zip: binaries over `bin`, license texts into
/// `licenses`, each checked against the descriptor's sha256 and size as it
/// is written. Returns the number of binaries replaced or added.
fn extract_overlay(zip_path: &Path, d: &Descriptor, bin: &Path, licenses: &Path) -> Result<usize> {
    let listed: BTreeMap<&str, &OverlayFile> = d.files.iter().map(|f| (f.name.as_str(), f)).collect();
    let mut z = open_zip(zip_path)?;
    let mut binaries = 0;
    for i in 0..z.len() {
        let mut e = z.by_index(i).map_err(|e| upd(format!("{}: entry {i}: {e}", zip_path.display())))?;
        if e.is_dir() {
            continue;
        }
        let name = e.name().to_string();
        let want = listed.get(name.as_str()).ok_or_else(|| upd(format!("{name} is not in {DESCRIPTOR_NAME}")))?;
        let to = match classify_entry(&name)? {
            EntryKind::Binary => {
                binaries += 1;
                bin.join(&name)
            }
            EntryKind::License => {
                std::fs::create_dir_all(licenses).map_err(|e| Error::io(licenses, e))?;
                licenses.join(&name["licenses/".len()..])
            }
        };
        let (sha, size) = copy_hashed(&mut e, &to)?;
        if size != want.size || !sha.eq_ignore_ascii_case(&want.sha256) {
            return Err(upd(format!(
                "{name} in the overlay zip is {sha} ({size} bytes); {DESCRIPTOR_NAME} says {} ({} bytes)",
                want.sha256, want.size
            )));
        }
    }
    Ok(binaries)
}

/// The manifest of an overlay install: a fork build (bundled runtime, the
/// release tag and base zip) whose `patch` names the overlay's features.
#[allow(clippy::needless_update)]
pub fn overlay_manifest(d: &Descriptor, meta: &OverlayMeta) -> Manifest {
    Manifest {
        tag: meta.tag.clone(),
        source: SOURCE.into(),
        installed_at_unix: update::now_unix(),
        assets: vec![meta.base_asset.clone(), meta.overlay_asset.clone()],
        verify: Verify::default(),
        channel: Some(Channel::Unsloth),
        bundled_runtime: true,
        release_tag: Some(meta.tag.clone()),
        asset_sha256: Some(meta.base_sha256.to_ascii_lowercase()),
        gfx_target: Some(meta.gfx.clone()),
        patch: Some(BuildPatch {
            name: d.name.clone(),
            base_commit: Some(d.base.source_commit.chars().take(9).collect::<String>().to_ascii_lowercase()),
            features: d.patch.features.clone(),
            patch_sha256: Some(d.patch.sha256.to_ascii_lowercase()),
        }),
        // Fields added to Manifest later take their defaults here.
        ..Default::default()
    }
}

/// Unpack the base zip into `tmp/bin`, lay the overlay over it, keep the
/// descriptor (and the patch, when given) beside `bin`, write the manifest
/// and move `tmp` into place as `final_dir`. Network-free, so tests drive
/// it directly. Zips that live directly in `tmp` (downloads) are deleted
/// before the move; any other file is left alone. Any failure removes `tmp`.
#[allow(clippy::too_many_arguments)]
pub fn install_overlay_from_zips(
    base_zip: &Path,
    overlay_zip: &Path,
    d: &Descriptor,
    descriptor_text: &str,
    patch: Option<&Path>,
    tmp: &Path,
    final_dir: &Path,
    meta: &OverlayMeta,
) -> Result<PathBuf> {
    let result = stage_overlay(base_zip, overlay_zip, d, descriptor_text, patch, tmp, final_dir, meta);
    if result.is_err() {
        let _ = std::fs::remove_dir_all(tmp);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn stage_overlay(
    base_zip: &Path,
    overlay_zip: &Path,
    d: &Descriptor,
    descriptor_text: &str,
    patch: Option<&Path>,
    tmp: &Path,
    final_dir: &Path,
    meta: &OverlayMeta,
) -> Result<PathBuf> {
    check_descriptor(d, &meta.tag, &meta.gfx)?;
    let want = base_zip_sha(d, &meta.gfx)?;
    if !want.eq_ignore_ascii_case(&meta.base_sha256) {
        return Err(upd(format!(
            "{} is {}, but the overlay was built against the zip {want}",
            meta.base_asset, meta.base_sha256
        )));
    }
    if let Some(p) = patch {
        let sha = update::sha256_file(p)?;
        if !sha.eq_ignore_ascii_case(&d.patch.sha256) {
            return Err(upd(format!("{} is {sha}; {DESCRIPTOR_NAME} says {}", p.display(), d.patch.sha256)));
        }
    }
    // A bad overlay zip fails here, before 500 MB of base is unpacked.
    check_overlay_zip(overlay_zip, d)?;
    std::fs::create_dir_all(tmp).map_err(|e| Error::io(tmp, e))?;
    let bin = update::stage_unsloth_base(base_zip, tmp, final_dir, &meta.base_asset)?;
    let replaced = extract_overlay(overlay_zip, d, &bin, &tmp.join(LICENSES_DIR))?;
    if replaced == 0 {
        return Err(upd("the overlay zip holds no binaries"));
    }
    let desc_out = tmp.join(DESCRIPTOR_NAME);
    std::fs::write(&desc_out, descriptor_text).map_err(|e| Error::io(&desc_out, e))?;
    if let Some(p) = patch {
        let to = tmp.join(&d.patch.file);
        if p != to {
            std::fs::copy(p, &to).map_err(|e| Error::io(&to, e))?;
        }
    }
    // Downloads staged in tmp must not move into the build.
    for z in [base_zip, overlay_zip] {
        if z.parent() == Some(tmp) {
            let _ = std::fs::remove_file(z);
        }
    }
    update::finish_unsloth_stage(tmp, final_dir, &overlay_manifest(d, meta))
}

// --------------------------------------------------------------- releases ----

/// The published overlay for `tag`: release `dgpatch5-<tag>` of the overlay
/// repository. None when there is no such release.
pub fn overlay_for(cfg: &Config, tag: &str) -> Result<Option<Release>> {
    if update::unsloth_tag_parts(tag).is_none() {
        return Err(upd(format!("`{tag}` is not an Unsloth release tag (b<n>-mix-<sha>)")));
    }
    let url = format!("https://api.github.com/repos/{}/releases/tags/{}", overlay_repo(cfg), overlay_release_tag(tag));
    match update::get_json_opt(&url)? {
        Some(json) => Ok(Some(update::parse_release(&json)?)),
        None => Ok(None),
    }
}

/// An overlay release's assets.
#[derive(Debug, Clone)]
pub struct OverlayAssets {
    pub zip: Asset,
    pub descriptor: Asset,
    /// The patch as published; optional.
    pub patch: Option<Asset>,
}

pub fn select_overlay_assets(r: &Release, tag: &str) -> Result<OverlayAssets> {
    let find = |name: &str| r.assets.iter().find(|a| a.name == name).cloned();
    let zip_name = overlay_zip_name(tag);
    Ok(OverlayAssets {
        zip: find(&zip_name).ok_or_else(|| upd(format!("overlay release {} has no {zip_name}", r.tag)))?,
        descriptor: find(DESCRIPTOR_NAME).ok_or_else(|| upd(format!("overlay release {} has no {DESCRIPTOR_NAME}", r.tag)))?,
        patch: find(&format!("{OVERLAY_PATCH}.diff")),
    })
}

/// What a check found out about the overlay for a release.
#[derive(Debug, Clone)]
pub enum OverlayLookup {
    NotChecked,
    Missing,
    Found(Release),
    Failed(String),
}

pub fn lookup(cfg: &Config, tag: &str) -> OverlayLookup {
    match overlay_for(cfg, tag) {
        Ok(Some(r)) => OverlayLookup::Found(r),
        Ok(None) => OverlayLookup::Missing,
        Err(e) => OverlayLookup::Failed(e.to_string()),
    }
}

/// Fill a check's overlay fields from a lookup. A failed lookup is reported,
/// never fatal: the plain Unsloth build stays installable.
pub fn apply_lookup(c: &mut UnslothCheck, l: OverlayLookup) {
    c.overlay_available = false;
    c.overlay_asset = None;
    c.overlay_error = None;
    match l {
        OverlayLookup::NotChecked | OverlayLookup::Missing => {}
        OverlayLookup::Failed(e) => c.overlay_error = Some(e),
        OverlayLookup::Found(r) => match select_overlay_assets(&r, &c.latest.tag) {
            Ok(a) => {
                c.overlay_available = true;
                c.overlay_asset = Some(a.zip);
            }
            Err(e) => c.overlay_error = Some(e.to_string()),
        },
    }
}

// ---------------------------------------------------------------- install ----

/// Where the overlay comes from.
#[derive(Debug, Clone)]
pub enum OverlaySource {
    /// Release `dgpatch5-<tag>` of the overlay repository.
    Published,
    /// A folder written by build-local.ps1 (the descriptor and the zip), or
    /// the overlay zip itself with the descriptor beside it.
    Local(PathBuf),
}

/// A local overlay's descriptor, zip and (optional) patch.
pub fn local_overlay(path: &Path, tag: &str) -> Result<(PathBuf, PathBuf, Option<PathBuf>)> {
    let zip_name = overlay_zip_name(tag);
    let (dir, zip) = if path.is_dir() {
        let zip = path.join(&zip_name);
        if !zip.is_file() {
            let others: Vec<String> = std::fs::read_dir(path)
                .map(|rd| {
                    rd.flatten()
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .filter(|n| n.starts_with("fidim-dg-overlay-") && n.ends_with(".zip"))
                        .collect()
                })
                .unwrap_or_default();
            return Err(upd(format!(
                "{} holds no {zip_name}{}",
                path.display(),
                if others.is_empty() { String::new() } else { format!(" (it holds {}: an overlay for another release)", others.join(", ")) }
            )));
        }
        (path.to_path_buf(), zip)
    } else if path.is_file() {
        (path.parent().map(Path::to_path_buf).unwrap_or_default(), path.to_path_buf())
    } else {
        return Err(upd(format!("{} does not exist", path.display())));
    };
    let desc = dir.join(DESCRIPTOR_NAME);
    if !desc.is_file() {
        return Err(upd(format!("{} has no {DESCRIPTOR_NAME} beside the overlay zip", dir.display())));
    }
    let patch = dir.join(format!("{OVERLAY_PATCH}.diff"));
    Ok((desc, zip, patch.is_file().then_some(patch)))
}

/// Download `asset` to `to` and require GitHub's digest to match: an
/// overlay file that cannot be checked is not installed.
fn fetch_checked(asset: &Asset, to: &Path, progress: &mut dyn FnMut(String)) -> Result<String> {
    let want = asset
        .digest
        .as_deref()
        .and_then(|d| d.strip_prefix("sha256:"))
        .ok_or_else(|| upd(format!("{}: no digest published; refusing an overlay file that cannot be checked", asset.name)))?
        .to_string();
    progress(format!("downloading {} ({} MB)", asset.name, asset.size >> 20));
    update::download(asset, to, progress)?;
    let got = update::sha256_file(to)?;
    if !got.eq_ignore_ascii_case(&want) {
        return Err(upd(format!("sha256 mismatch for {}: GitHub publishes {want}, the download is {got}", asset.name)));
    }
    Ok(got)
}

/// The installed overlay's descriptor, kept beside `bin`.
pub fn installed_descriptor(dir: &Path) -> Option<Descriptor> {
    parse_descriptor(&std::fs::read_to_string(dir.join(DESCRIPTOR_NAME)).ok()?).ok()
}

/// `verify_unsloth`, plus whether the llama-server that answered is the one
/// the descriptor describes. A mismatch is reported in `detail`.
pub fn verify_overlay(dir: &Path) -> Verify {
    let mut v = update::verify_unsloth(dir);
    if let (Some(printed), Some(d)) = (v.commit.clone(), installed_descriptor(dir)) {
        if !commit_matches(&printed, &d.base.source_commit) {
            v.detail.push_str(&format!(
                "llama-server reports commit {printed}, but the overlay was built from {}\n",
                d.base.source_commit
            ));
        }
    }
    v
}

/// Install `release`'s Windows ROCm zip for `gfx` with the runner-patch
/// overlay laid over it, as `<install_root>/<tag>-unsloth-dgpatch5`, then
/// verify it (`--version` and `--list-devices` on the bundled runtime; no
/// model is loaded). `base_zip` is an already downloaded copy of the Unsloth
/// zip, checked like a download. An existing install is re-verified instead.
pub fn install_unsloth_overlay(
    cfg: &Config,
    release: &Release,
    gfx: &str,
    source: &OverlaySource,
    base_zip: Option<&Path>,
    progress: &mut dyn FnMut(String),
) -> Result<InstallReport> {
    let tag = release.tag.clone();
    // install_dir refuses Unsloth Studio's own tree.
    let dir = overlay_install_dir(cfg, &tag)?;
    if dir.join("bin").join(RUNNER_EXE).is_file() {
        progress(format!("{tag} with {OVERLAY_PATCH} already installed at {} — verifying only", dir.display()));
        let verify = verify_overlay(&dir);
        let mut m = update::read_manifest(&dir)
            .ok_or_else(|| upd(format!("{} has no readable manifest; remove it and install again", dir.display())))?;
        m.verify = verify.clone();
        update::write_manifest(&dir, &m)?;
        return Ok(InstallReport { tag, dir, source: SOURCE.into(), skipped_existing: true, verify });
    }
    if dir.exists() {
        return Err(upd(format!("{} exists but has no {RUNNER_EXE}; remove it, then install again", dir.display())));
    }
    let (n, _) = update::unsloth_tag_parts(&tag)
        .ok_or_else(|| upd(format!("`{tag}` is not an Unsloth release tag (b<n>-mix-<sha>)")))?;
    let base_asset = update::select_unsloth_asset(release, gfx)?;
    let root = dir.parent().ok_or_else(|| upd("install dir has no parent"))?;
    // Dot-prefixed, so a scan never offers a half-extracted build.
    let tmp = root.join(format!(".fidim-tmp-{n}-{OVERLAY_PATCH}"));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| Error::io(&tmp, e))?;

    let staged = (|| -> Result<Descriptor> {
        // The overlay first: it is small, and its descriptor decides whether
        // the base download is worth making.
        let (desc_text, overlay_zip, overlay_name, patch) = match source {
            OverlaySource::Published => {
                let repo = overlay_repo(cfg);
                let rel = overlay_for(cfg, &tag)?.ok_or_else(|| {
                    upd(format!("no {OVERLAY_PATCH} overlay is published for {tag} (no release {} in {repo})", overlay_release_tag(&tag)))
                })?;
                let a = select_overlay_assets(&rel, &tag)?;
                let desc_path = tmp.join(DESCRIPTOR_NAME);
                fetch_checked(&a.descriptor, &desc_path, progress)?;
                let text = std::fs::read_to_string(&desc_path).map_err(|e| Error::io(&desc_path, e))?;
                check_descriptor(&parse_descriptor(&text)?, &tag, gfx)?;
                let zip = tmp.join(&a.zip.name);
                fetch_checked(&a.zip, &zip, progress)?;
                let patch = match &a.patch {
                    Some(p) => {
                        let to = tmp.join(&p.name);
                        fetch_checked(p, &to, progress)?;
                        Some(to)
                    }
                    None => None,
                };
                (text, zip, a.zip.name.clone(), patch)
            }
            OverlaySource::Local(path) => {
                let (desc_path, zip, patch) = local_overlay(path, &tag)?;
                progress(format!("overlay from {}", zip.display()));
                let text = std::fs::read_to_string(&desc_path).map_err(|e| Error::io(&desc_path, e))?;
                check_descriptor(&parse_descriptor(&text)?, &tag, gfx)?;
                let name = zip.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                (text, zip, name, patch)
            }
        };
        let d = parse_descriptor(&desc_text)?;
        progress(format!(
            "overlay {} for {tag}: {} files, {}",
            d.name,
            d.files.len(),
            d.signer.as_deref().map(|s| format!("signed by {s}")).unwrap_or_else(|| "unsigned".into())
        ));

        let digest = base_asset.digest.as_deref().and_then(|x| x.strip_prefix("sha256:")).map(str::to_string);
        let (base_path, base_sha) = match base_zip {
            Some(p) => {
                progress(format!("hashing {}", p.display()));
                (p.to_path_buf(), update::sha256_file(p)?)
            }
            None => {
                let to = tmp.join(&base_asset.name);
                progress(format!("downloading {} ({} MB)", base_asset.name, base_asset.size >> 20));
                update::download(&base_asset, &to, progress)?;
                let sha = update::sha256_file(&to)?;
                (to, sha)
            }
        };
        if let Some(want) = digest {
            if !want.eq_ignore_ascii_case(&base_sha) {
                return Err(upd(format!(
                    "sha256 mismatch for {}: GitHub publishes {want}, {} is {base_sha}",
                    base_asset.name,
                    base_path.display()
                )));
            }
        }
        progress(format!("{} sha256 {}…; extracting into {}", base_asset.name, &base_sha[..16], dir.display()));
        let meta = OverlayMeta {
            tag: tag.clone(),
            gfx: gfx.to_string(),
            base_asset: base_asset.name.clone(),
            base_sha256: base_sha,
            overlay_asset: overlay_name,
        };
        install_overlay_from_zips(&base_path, &overlay_zip, &d, &desc_text, patch.as_deref(), &tmp, &dir, &meta)?;
        Ok(d)
    })();
    let d = match staged {
        Ok(d) => d,
        Err(e) => {
            let _ = std::fs::remove_dir_all(&tmp);
            return Err(e);
        }
    };

    progress("verifying: --version and --list-devices on the bundled ROCm (no model load)".into());
    let verify = update::verify_unsloth(&dir);
    let recorded = (|| -> Result<()> {
        if let Some(printed) = &verify.commit {
            if !commit_matches(printed, &d.base.source_commit) {
                return Err(upd(format!(
                    "the installed llama-server reports commit {printed}, but the overlay was built from {}: \
                     it is not the build its descriptor describes",
                    d.base.source_commit
                )));
            }
        }
        let mut m = update::read_manifest(&dir).ok_or_else(|| upd(format!("{}: manifest unreadable after install", dir.display())))?;
        m.verify = verify.clone();
        update::write_manifest(&dir, &m)
    })();
    if let Err(e) = recorded {
        // This call created the directory; a build that is not what it
        // claims, or has no manifest, must not stay behind.
        let _ = std::fs::remove_dir_all(&dir);
        return Err(e);
    }
    Ok(InstallReport { tag, dir, source: SOURCE.into(), skipped_existing: false, verify })
}

// ------------------------------------------------------------------ tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{self, dg_feature};
    use sha2::{Digest, Sha256};

    const TAG: &str = "b11030-mix-5ff778e";

    fn sha(bytes: &[u8]) -> String {
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn allowlist_and_entry_names() {
        for ok in ["llama.dll", "llama-server.exe", "llama-server-impl.dll", "llama-common.dll", "LLAMA-CLI.EXE", "mtmd.dll",
                   RUNNER_EXE, "llama.exe"] {
            assert!(overlay_binary_allowed(ok), "{ok}");
            assert_eq!(classify_entry(ok).unwrap(), EntryKind::Binary, "{ok}");
        }
        for no in ["ggml.dll", "ggml-hip.dll", "ggml-base.dll", "GGML-CPU.DLL", "ggml-rpc-server.exe", "amdhip64_7.dll",
                   "amd_comgr.dll", "hipblas.dll", "libhipblaslt.dll", "rocblas.dll", "rocm_kpack.dll", "rocsolver.dll",
                   "origami.dll", "llama.pdb", "llama.dll.bak", "mtmd-helper.dll", "vcruntime140.dll", "", ".llama.dll",
                   "../llama.dll", "bin/llama.dll", "rocblas/library/x", "llama dll.dll", "llama.dll:ads"] {
            assert!(!overlay_binary_allowed(no), "{no}");
            assert!(classify_entry(no).is_err(), "{no}");
        }
        assert_eq!(classify_entry("licenses/LICENSE-llama.cpp").unwrap(), EntryKind::License);
        assert_eq!(classify_entry("licenses/THIRD-PARTY-NOTICES.md").unwrap(), EntryKind::License);
        for no in ["licenses/", "licenses/sub/x", "licenses/../llama.dll", "licenses/.x", "Licenses/LICENSE"] {
            assert!(classify_entry(no).is_err(), "{no}");
        }
        let e = classify_entry("amdhip64_7.dll").unwrap_err().to_string();
        assert!(e.contains("stay Unsloth's"), "{e}");
    }

    #[test]
    fn repo_names_and_the_config_override() {
        let mut cfg = Config::default_for_machine();
        assert_eq!(overlay_repo(&cfg), OVERLAY_REPO);
        cfg.extra.insert(OVERLAY_REPO_KEY.into(), "someone/fidim-dg-overlay-test".into());
        assert_eq!(overlay_repo(&cfg), "someone/fidim-dg-overlay-test");
        for bad in ["", "no-slash", "a/b/c", "a/", "/b", "a b/c", "https://github.com/a/b", "a/b?x=1"] {
            cfg.extra.insert(OVERLAY_REPO_KEY.into(), bad.into());
            assert_eq!(overlay_repo(&cfg), OVERLAY_REPO, "{bad:?}");
        }
        assert_eq!(overlay_release_tag(TAG), "dgpatch5-b11030-mix-5ff778e");
        assert_eq!(overlay_zip_name(TAG), "fidim-dg-overlay-b11030-mix-5ff778e-windows-x64.zip");
        cfg.install_root = Some(PathBuf::from(r"C:\fidim-builds"));
        assert_eq!(overlay_install_dir(&cfg, TAG).unwrap(), PathBuf::from(r"C:\fidim-builds\b11030-mix-5ff778e-unsloth-dgpatch5"));
        assert!(commit_matches("6ba30d05b", "6ba30d05b140ebb0baeded27d7d9b843c5b71ff1"));
        assert!(commit_matches("6BA30D05B", "6ba30d05b140ebb0baeded27d7d9b843c5b71ff1"));
        assert!(!commit_matches("f6b9ea743", "6ba30d05b140ebb0baeded27d7d9b843c5b71ff1"));
        assert!(!commit_matches("6ba", "6ba30d05b140ebb0baeded27d7d9b843c5b71ff1"), "too short to mean anything");
    }

    /// The descriptor the local build wrote for b11030 (packaging/dg-overlay,
    /// build-local.ps1, MSVC, BoringSSL off).
    const CAPTURED: &str = include_str!("../../../fixtures/fidim-overlay-b11030.json");

    #[test]
    fn captured_descriptor_parses_and_checks() {
        let d = parse_descriptor(CAPTURED).unwrap();
        assert_eq!(d.name, "dgpatch5");
        assert_eq!(d.base.release_tag, TAG);
        assert_eq!(d.base.source_commit, "6ba30d05b140ebb0baeded27d7d9b843c5b71ff1");
        check_descriptor(&d, TAG, "gfx120X").unwrap();
        check_descriptor(&d, TAG, "gfx120x").unwrap();
        assert_eq!(base_zip_sha(&d, "gfx120X").unwrap(), "c780a9bda3ce315dee42e0a3045a7884dd9e2d46ec1d6d34c76e42bfecaa9557");
        assert_eq!(d.base.zips.len(), 7, "one overlay serves every Windows ROCm zip of the release");
        for f in [dg_feature::PKV_F16, dg_feature::SWA_RING, dg_feature::FA_PAD, dg_feature::FA_TURN_SIZING, dg_feature::STEP_FAIL_ERR] {
            assert!(d.patch.features.iter().any(|x| x == f), "{f}");
        }
        let bins = d.files.iter().filter(|f| classify_entry(&f.name).unwrap() == EntryKind::Binary).count();
        assert_eq!(bins, 60);
        assert!(d.files.iter().any(|f| f.name == "licenses/THIRD-PARTY-NOTICES.md"));
        assert!(!d.ggml_imports.contains_key("ggml-hip.dll"));
        assert_eq!(d.signer, None);
        let e = check_descriptor(&d, "b11027-mix-3e83366", "gfx120X").unwrap_err().to_string();
        assert!(e.contains("built for b11030-mix-5ff778e, not b11027-mix-3e83366"), "{e}");
        let e = check_descriptor(&d, TAG, "gfx1201").unwrap_err().to_string();
        assert!(e.contains("no gfx1201 zip") && e.contains("gfx120X"), "{e}");
        // A BOM (an editor's save) is tolerated.
        assert_eq!(parse_descriptor(&format!("\u{feff}{CAPTURED}")).unwrap(), d);
    }

    /// A zip at `path` holding `files` (name, contents).
    fn write_zip(path: &Path, files: &[(&str, &[u8])]) {
        use zip::write::SimpleFileOptions;
        let mut z = zip::ZipWriter::new(File::create(path).unwrap());
        let opts = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (name, bytes) in files {
            z.start_file(*name, opts).unwrap();
            z.write_all(bytes).unwrap();
        }
        z.finish().unwrap();
    }

    const BASE: &[(&str, &[u8])] = &[
        ("llama-server.exe", b"unsloth server"),
        (RUNNER_EXE, b"unsloth runner"),
        ("llama.dll", b"unsloth llama"),
        ("llama-cvector-generator.exe", b"unsloth cvector"),
        ("ggml.dll", b"unsloth ggml"),
        ("ggml-hip.dll", b"unsloth ggml-hip"),
        ("amdhip64_7.dll", b"amd runtime"),
        ("hipblas.dll", b"amd hipblas"),
        ("rocblas/library/TensileLibrary.dat", b"kernels"),
    ];
    const OVERLAY: &[(&str, &[u8])] = &[
        ("llama-server.exe", b"patched server"),
        (RUNNER_EXE, b"patched runner"),
        ("llama.dll", b"patched llama"),
        ("mtmd.dll", b"patched mtmd"),
        ("licenses/LICENSE-llama.cpp", b"MIT License"),
    ];
    const PATCH: &[u8] = b"dgpatch5 diff";

    fn descriptor_for(files: &[(&str, &[u8])], base_sha: &str) -> Descriptor {
        Descriptor {
            schema: 1,
            name: OVERLAY_PATCH.into(),
            base: DescriptorBase {
                repo: BASE_REPO.into(),
                release_tag: TAG.into(),
                source_commit: "6ba30d05b140ebb0baeded27d7d9b843c5b71ff1".into(),
                source_asset: None,
                source_sha256: None,
                ggml_tree: None,
                zips: [("gfx120X".to_string(), base_sha.to_string()), ("gfx1151".to_string(), "0".repeat(64))].into(),
            },
            patch: DescriptorPatch {
                file: "dgpatch5.diff".into(),
                sha256: sha(PATCH),
                features: vec![dg_feature::FA_PAD.into(), dg_feature::SWA_RING.into(), "dg-sc-splitk".into()],
            },
            files: files.iter().map(|(n, b)| OverlayFile { name: n.to_string(), sha256: sha(b), size: b.len() as u64 }).collect(),
            build: serde_json::Value::Null,
            ggml_imports: BTreeMap::new(),
            signer: None,
        }
    }

    struct Fixture {
        root: PathBuf,
        tmp: PathBuf,
        fin: PathBuf,
        base_zip: PathBuf,
        overlay_zip: PathBuf,
        patch: PathBuf,
        meta: OverlayMeta,
    }

    /// Base zip downloaded into tmp (as an install does), overlay zip and
    /// patch in a separate folder (as `--overlay-from` does).
    fn fixture(name: &str, overlay: &[(&str, &[u8])]) -> Fixture {
        let root = std::env::temp_dir().join(format!("fidim-overlay-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let tmp = root.join(".fidim-tmp-11030-dgpatch5");
        let local = root.join("local");
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::create_dir_all(&local).unwrap();
        let base_asset = format!("app-{TAG}-windows-x64-rocm-gfx120X.zip");
        let base_zip = tmp.join(&base_asset);
        write_zip(&base_zip, BASE);
        let overlay_zip = local.join(overlay_zip_name(TAG));
        write_zip(&overlay_zip, overlay);
        let patch = local.join("dgpatch5.diff");
        std::fs::write(&patch, PATCH).unwrap();
        let base_sha = update::sha256_file(&base_zip).unwrap();
        Fixture {
            fin: root.join(format!("{TAG}-unsloth-dgpatch5")),
            meta: OverlayMeta {
                tag: TAG.into(),
                gfx: "gfx120X".into(),
                base_asset,
                base_sha256: base_sha,
                overlay_asset: overlay_zip_name(TAG),
            },
            root,
            tmp,
            base_zip,
            overlay_zip,
            patch,
        }
    }

    fn install(fx: &Fixture, d: &Descriptor) -> Result<PathBuf> {
        let text = serde_json::to_string_pretty(d).unwrap();
        install_overlay_from_zips(&fx.base_zip, &fx.overlay_zip, d, &text, Some(&fx.patch), &fx.tmp, &fx.fin, &fx.meta)
    }

    #[test]
    fn overlay_lands_on_the_base_and_the_manifest_names_the_patch() {
        let fx = fixture("ok", OVERLAY);
        let d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        assert_eq!(install(&fx, &d).unwrap(), fx.fin);
        assert!(!fx.tmp.exists(), "staging dir renamed away");
        let bin = fx.fin.join("bin");
        let read = |n: &str| std::fs::read(bin.join(n)).unwrap();
        // The overlay replaced the llama level and added mtmd...
        assert_eq!(read("llama-server.exe"), b"patched server");
        assert_eq!(read(RUNNER_EXE), b"patched runner");
        assert_eq!(read("llama.dll"), b"patched llama");
        assert_eq!(read("mtmd.dll"), b"patched mtmd");
        // ...and left ggml, the runtime, the kernels and cvector as Unsloth shipped them.
        assert_eq!(read("ggml.dll"), b"unsloth ggml");
        assert_eq!(read("ggml-hip.dll"), b"unsloth ggml-hip");
        assert_eq!(read("amdhip64_7.dll"), b"amd runtime");
        assert_eq!(read("hipblas.dll"), b"amd hipblas");
        assert_eq!(read("llama-cvector-generator.exe"), b"unsloth cvector");
        assert_eq!(std::fs::read(bin.join("rocblas").join("library").join("TensileLibrary.dat")).unwrap(), b"kernels");
        assert!(!bin.join("licenses").exists(), "license texts never land in bin");
        assert_eq!(std::fs::read(fx.fin.join(LICENSES_DIR).join("LICENSE-llama.cpp")).unwrap(), b"MIT License");
        assert_eq!(installed_descriptor(&fx.fin).unwrap(), d);
        assert_eq!(std::fs::read(fx.fin.join("dgpatch5.diff")).unwrap(), PATCH);
        // The downloaded base zip went; the local overlay zip and patch stay.
        assert!(!fx.fin.join(&fx.meta.base_asset).exists());
        assert!(fx.overlay_zip.is_file() && fx.patch.is_file());

        let text = std::fs::read_to_string(fx.fin.join(update::MANIFEST_NAME)).unwrap();
        let m: Manifest = serde_json::from_str(&text).unwrap();
        assert_eq!(m.source, "unsloth-overlay");
        assert_eq!(m.tag, TAG);
        assert_eq!(m.release_tag.as_deref(), Some(TAG));
        assert_eq!((m.channel, m.bundled_runtime), (Some(Channel::Unsloth), true));
        assert_eq!(m.assets, vec![fx.meta.base_asset.clone(), overlay_zip_name(TAG)]);
        assert_eq!(m.asset_sha256.as_deref(), Some(fx.meta.base_sha256.as_str()));
        assert_eq!(m.gfx_target.as_deref(), Some("gfx120X"));
        let p = m.patch.unwrap();
        assert_eq!(p.name, "dgpatch5");
        assert_eq!(p.base_commit.as_deref(), Some("6ba30d05b"));
        assert_eq!(p.features, d.patch.features);
        assert_eq!(p.patch_sha256.as_deref(), Some(sha(PATCH).as_str()));

        // Discovery, the estimator and promotion read the build as patched.
        let meta = discovery::read_build_meta(&fx.fin);
        assert_eq!(meta.channel, Channel::Unsloth);
        assert!(meta.bundled_runtime && meta.has_feature(dg_feature::FA_PAD) && meta.has_feature(dg_feature::SWA_RING));
        assert_eq!(meta.patch.as_ref().map(|p| p.label()), Some("dgpatch5"));
        let t = update::promote_target(&fx.fin);
        assert!(t.has_runner && t.has_llama_server && t.features.contains(&"dg-sc-splitk".to_string()));

        // An existing target is never overwritten.
        let again = fixture("ok-again", OVERLAY);
        let e = install_overlay_from_zips(
            &again.base_zip, &again.overlay_zip, &d, "{}", None, &again.tmp, &fx.fin,
            &OverlayMeta { base_sha256: fx.meta.base_sha256.clone(), ..again.meta.clone() },
        );
        assert!(e.is_err());
        assert_eq!(std::fs::read(bin.join("llama.dll")).unwrap(), b"patched llama", "the installed build is untouched");
        assert!(!again.tmp.exists());
        std::fs::remove_dir_all(&again.root).ok();
        std::fs::remove_dir_all(&fx.root).ok();
    }

    /// Every refusal leaves neither the staging folder nor the build behind.
    fn refused(fx: &Fixture, d: &Descriptor, expect: &str) {
        let e = install(fx, d).unwrap_err().to_string();
        assert!(e.contains(expect), "expected {expect:?} in: {e}");
        assert!(!fx.tmp.exists(), "{expect}: tmp left behind");
        assert!(!fx.fin.exists(), "{expect}: a build appeared");
        std::fs::remove_dir_all(&fx.root).ok();
    }

    #[test]
    fn a_file_whose_sha256_differs_is_refused() {
        let fx = fixture("sha", OVERLAY);
        let mut d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        d.files.iter_mut().find(|f| f.name == "llama.dll").unwrap().sha256 = sha(b"what the descriptor promised");
        refused(&fx, &d, "llama.dll in the overlay zip is");

        let fx = fixture("size", OVERLAY);
        let mut d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        d.files.iter_mut().find(|f| f.name == "mtmd.dll").unwrap().size += 1;
        refused(&fx, &d, "mtmd.dll in the overlay zip is");

        let fx = fixture("patch", OVERLAY);
        let mut d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        d.patch.sha256 = sha(b"another patch");
        refused(&fx, &d, "dgpatch5.diff is");
    }

    #[test]
    fn a_base_zip_other_than_the_overlay_was_built_against_is_refused() {
        let fx = fixture("base", OVERLAY);
        let d = descriptor_for(OVERLAY, &sha(b"another release's zip"));
        refused(&fx, &d, "but the overlay was built against the zip");

        // The descriptor names another release, or another patch.
        let fx = fixture("tag", OVERLAY);
        let mut d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        d.base.release_tag = "b11027-mix-3e83366".into();
        refused(&fx, &d, "built for b11027-mix-3e83366, not b11030-mix-5ff778e");
        let fx = fixture("name", OVERLAY);
        let mut d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        d.name = "dgpatch6".into();
        refused(&fx, &d, "carries dgpatch6");
        let fx = fixture("schema", OVERLAY);
        let mut d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        d.schema = 2;
        refused(&fx, &d, "schema 2");
        let fx = fixture("repo", OVERLAY);
        let mut d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        d.base.repo = "ggml-org/llama.cpp".into();
        refused(&fx, &d, "built on ggml-org/llama.cpp");
    }

    #[test]
    fn files_outside_the_allowlist_are_refused() {
        // In the zip but not listed.
        let mut with_ggml = OVERLAY.to_vec();
        with_ggml.push(("ggml.dll", b"a rebuilt ggml"));
        let fx = fixture("ggml", &with_ggml);
        let d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        refused(&fx, &d, "ggml.dll is not a llama-level binary");
        // Listed too: the descriptor check refuses it first.
        let fx = fixture("ggml-listed", &with_ggml);
        let d = descriptor_for(&with_ggml, &fx.meta.base_sha256);
        refused(&fx, &d, "ggml.dll is not a llama-level binary");
        for (name, what) in [("amdhip64_7.dll", "stay Unsloth's"), ("rocblas/library/x", "not a plain file"), ("../llama.dll", "not a plain file")] {
            let mut zip = OVERLAY.to_vec();
            zip.push((name, b"x"));
            let fx = fixture("deny", &zip);
            let d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
            refused(&fx, &d, what);
        }
        // An allowed binary the descriptor does not list, and a listed one the zip lacks.
        let mut extra = OVERLAY.to_vec();
        extra.push(("llama-extra.exe", b"x"));
        let fx = fixture("unlisted", &extra);
        let d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        refused(&fx, &d, "llama-extra.exe, which fidim-overlay.json does not list");
        let fx = fixture("missing", &OVERLAY[1..]);
        let d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        refused(&fx, &d, "lists llama-server.exe, which the overlay zip does not hold");
        // No runner at all.
        let no_runner: Vec<_> = OVERLAY.iter().copied().filter(|(n, _)| *n != RUNNER_EXE).collect();
        let fx = fixture("no-runner", &no_runner);
        let d = descriptor_for(&no_runner, &fx.meta.base_sha256);
        refused(&fx, &d, &format!("the overlay has no {RUNNER_EXE}"));
        // A name listed twice, and a malformed hash.
        let fx = fixture("twice", OVERLAY);
        let mut d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        d.files.push(d.files[0].clone());
        refused(&fx, &d, "lists llama-server.exe twice");
        let fx = fixture("hash", OVERLAY);
        let mut d = descriptor_for(OVERLAY, &fx.meta.base_sha256);
        d.files[0].sha256 = "abc".into();
        refused(&fx, &d, "has no sha256");
    }

    #[test]
    fn local_overlay_folder_or_zip() {
        let root = std::env::temp_dir().join(format!("fidim-overlay-local-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let e = local_overlay(&root, TAG).unwrap_err().to_string();
        assert!(e.contains(&overlay_zip_name(TAG)), "{e}");
        std::fs::write(root.join(overlay_zip_name("b11027-mix-3e83366")), b"").unwrap();
        let e = local_overlay(&root, TAG).unwrap_err().to_string();
        assert!(e.contains("an overlay for another release"), "{e}");
        std::fs::write(root.join(overlay_zip_name(TAG)), b"").unwrap();
        let e = local_overlay(&root, TAG).unwrap_err().to_string();
        assert!(e.contains(DESCRIPTOR_NAME), "{e}");
        std::fs::write(root.join(DESCRIPTOR_NAME), b"{}").unwrap();
        let (d, z, p) = local_overlay(&root, TAG).unwrap();
        assert_eq!((d, z.clone(), p), (root.join(DESCRIPTOR_NAME), root.join(overlay_zip_name(TAG)), None));
        std::fs::write(root.join("dgpatch5.diff"), b"").unwrap();
        assert_eq!(local_overlay(&root, TAG).unwrap().2, Some(root.join("dgpatch5.diff")));
        // The zip itself, descriptor beside it.
        assert_eq!(local_overlay(&z, TAG).unwrap().1, z);
        assert!(local_overlay(&root.join("nope"), TAG).is_err());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn check_reports_overlay_availability() {
        let mut cfg = Config::default_for_machine();
        let root = std::env::temp_dir().join(format!("fidim-overlay-check-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        cfg.install_root = Some(root.clone());
        let latest = update::parse_release(&format!(
            r#"{{"tag_name":"{TAG}","published_at":"2026-09-18T15:36:14Z","html_url":"","assets":[
                {{"name":"app-{TAG}-windows-x64-rocm-gfx120X.zip","browser_download_url":"https://x/b","size":494371218,
                  "digest":"sha256:c780a9bda3ce315dee42e0a3045a7884dd9e2d46ec1d6d34c76e42bfecaa9557"}}]}}"#
        ))
        .unwrap();
        let mut c = update::check_unsloth_against(&cfg, &[], latest.clone(), "gfx120X").unwrap();
        assert!(!c.overlay_available && !c.overlay_installed && c.overlay_asset.is_none() && c.overlay_error.is_none());
        assert_eq!(c.overlay_repo, OVERLAY_REPO);
        assert_eq!(c.overlay_patch, "dgpatch5");
        assert_eq!(c.overlay_install_dir, root.join(format!("{TAG}-unsloth-dgpatch5")));
        assert_eq!(c.install_dir, root.join(format!("{TAG}-unsloth")));

        let rel = update::parse_release(&format!(
            r#"{{"tag_name":"dgpatch5-{TAG}","published_at":"2026-09-19T00:00:00Z","html_url":"","assets":[
                {{"name":"fidim-dg-overlay-{TAG}-windows-x64.zip","browser_download_url":"https://x/o","size":9437184,"digest":"sha256:{z}"}},
                {{"name":"fidim-overlay.json","browser_download_url":"https://x/d","size":30000,"digest":"sha256:{z}"}},
                {{"name":"dgpatch5.diff","browser_download_url":"https://x/p","size":40000,"digest":"sha256:{z}"}},
                {{"name":"SHA256SUMS","browser_download_url":"https://x/s","size":300}}]}}"#,
            z = "0".repeat(64)
        ))
        .unwrap();
        apply_lookup(&mut c, OverlayLookup::Found(rel.clone()));
        assert!(c.overlay_available);
        assert_eq!(c.overlay_asset.as_ref().unwrap().name, overlay_zip_name(TAG));
        let a = select_overlay_assets(&rel, TAG).unwrap();
        assert_eq!((a.descriptor.name.as_str(), a.patch.unwrap().name.as_str()), (DESCRIPTOR_NAME, "dgpatch5.diff"));
        apply_lookup(&mut c, OverlayLookup::Missing);
        assert!(!c.overlay_available && c.overlay_asset.is_none() && c.overlay_error.is_none());
        apply_lookup(&mut c, OverlayLookup::Failed("GitHub API: rate limited".into()));
        assert!(!c.overlay_available && c.overlay_error.as_deref() == Some("GitHub API: rate limited"));
        let mut no_zip = rel.clone();
        no_zip.assets.retain(|a| a.name != overlay_zip_name(TAG));
        apply_lookup(&mut c, OverlayLookup::Found(no_zip));
        assert!(!c.overlay_available && c.overlay_error.as_deref().unwrap().contains("has no fidim-dg-overlay-"));

        // Installed = the overlay folder has the runner; the plain one is separate.
        std::fs::create_dir_all(c.overlay_install_dir.join("bin")).unwrap();
        std::fs::write(c.overlay_install_dir.join("bin").join(RUNNER_EXE), b"").unwrap();
        let c = update::check_unsloth_against(&cfg, &[], latest, "gfx120X").unwrap();
        assert!(c.overlay_installed && !c.already_installed);
        std::fs::remove_dir_all(root).ok();
    }
}
