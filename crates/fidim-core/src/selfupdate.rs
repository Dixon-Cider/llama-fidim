//! Updating Llama FIDIM itself: the CLI, the DiffusionGemma server and the
//! desktop app, in the folder they run from.
//!
//! Two sources. A GitHub release of this repository ships
//! `llama-fidim-vX.Y.Z-win-x64.zip` with a `.sha256` beside it; the zip is
//! downloaded, checked against that digest and unpacked. Or a checkout of
//! this repository is built (`cargo build --release -p fidim-cli`, then
//! `pnpm tauri build --no-bundle`), which is what scripts/install.ps1 did by
//! hand. Either way the three executables end up in a staged folder under
//! `~/.fidim/app-updates`, verified by running the staged `fidim.exe
//! --version`.
//!
//! Replacing the installed files cannot be done by the app that runs them:
//! Windows refuses to delete a running image. So the staged `fidim.exe` is
//! started detached with `self-update apply`; it waits for the caller to
//! exit, deletes each installed file or, when one is still in use (a
//! diffusion server holding `fidim-dg.exe`, a terminal's `fidim live`),
//! renames it aside as `<name>.old-<time>` — the process keeps running from
//! the renamed image — copies the staged files in, and reopens the desktop
//! app, breaking out of the caller's job object when that job allows it
//! (an update started from a terminal or an agent must not end with that
//! session). Copies renamed aside by earlier updates are deleted once
//! nothing runs from them.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::update::{self, upd, Asset};
use crate::{build_info, Result};

/// This repository on GitHub.
pub const REPO: &str = "Dixon-Cider/llama-fidim";
const RELEASES_API: &str = "https://api.github.com/repos/Dixon-Cider/llama-fidim/releases";

/// The installed set. `fidim-dg.exe` is the DiffusionGemma server the app
/// starts; an install without it blocks every diffusion launch at pre-flight.
pub const EXES: [&str; 3] = ["fidim.exe", "fidim-dg.exe", "llama-fidim.exe"];
const GUI_EXE: &str = "llama-fidim.exe";
/// Text files the release zip carries beside the executables; kept in the
/// stage so the CHANGELOG is readable there, never installed.
const DOCS: [&str; 3] = ["CHANGELOG.md", "LICENSE", "README.md"];
const STAGED_JSON: &str = "staged.json";
const HISTORY_JSON: &str = "history.json";

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// `v0.2.0` or `0.2.0` as numbers, for ordering. Anything else is None.
pub fn semver(tag: &str) -> Option<(u32, u32, u32)> {
    let s = tag.trim().strip_prefix('v').unwrap_or(tag.trim());
    let mut parts = s.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    let patch = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

/// The Windows zip a release of this repository carries.
pub fn asset_name(tag: &str) -> String {
    format!("llama-fidim-{tag}-win-x64.zip")
}

/// `~/.fidim/app-updates`: downloads, staged sets, the apply log and the
/// history of applied updates.
pub fn updates_dir() -> PathBuf {
    Config::config_path()
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("app-updates")
}

/// Where the per-user installer and install.ps1 put the app.
pub fn default_install_dir() -> Option<PathBuf> {
    std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("Llama FIDIM"))
}

// ---------------------------------------------------------------- current ----

/// The running build and where it lives.
#[derive(Debug, Clone, Serialize)]
pub struct Current {
    pub version: String,
    pub long: String,
    pub commit: Option<String>,
    pub commits_ahead: Option<u32>,
    pub modified: bool,
    /// The folder this executable runs from: what an update replaces.
    pub install_dir: PathBuf,
    pub exe: PathBuf,
    /// The folder is the per-user install folder (not a build's target
    /// directory or a stage).
    pub in_install_folder: bool,
}

pub fn current() -> Result<Current> {
    let exe = std::env::current_exe().map_err(|e| upd(format!("which executable runs: {e}")))?;
    let install_dir = exe
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| upd(format!("{} has no parent folder", exe.display())))?;
    let in_install_folder = default_install_dir().is_some_and(|d| same_dir(&d, &install_dir));
    Ok(Current {
        version: build_info::VERSION.to_string(),
        long: build_info::LONG.to_string(),
        commit: (!build_info::COMMIT.is_empty()).then(|| build_info::COMMIT.to_string()),
        commits_ahead: build_info::commits_ahead(),
        modified: build_info::MODIFIED,
        install_dir,
        exe,
        in_install_folder,
    })
}

fn same_dir(a: &Path, b: &Path) -> bool {
    let norm = |p: &Path| {
        std::fs::canonicalize(p)
            .unwrap_or_else(|_| p.to_path_buf())
            .to_string_lossy()
            .to_lowercase()
    };
    norm(a) == norm(b)
}

// --------------------------------------------------------------- releases ----

/// A release of this repository that carries the Windows zip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppRelease {
    pub tag: String,
    /// `X.Y.Z` from the tag.
    pub version: String,
    pub published_at: String,
    pub html_url: String,
    /// The release notes: that version's CHANGELOG.md section.
    pub notes: String,
    pub zip: Asset,
    /// The `.sha256` published beside the zip; absent on a release that
    /// only has GitHub's own digest.
    pub sha256: Option<Asset>,
}

fn app_release_from(json: &str) -> Result<Option<AppRelease>> {
    let v: serde_json::Value = serde_json::from_str(json)?;
    if v["draft"].as_bool() == Some(true) {
        return Ok(None);
    }
    let r = update::parse_release(json)?;
    let Some((a, b, c)) = semver(&r.tag) else { return Ok(None) };
    let zip_name = asset_name(&r.tag);
    let Some(zip) = r.assets.iter().find(|x| x.name == zip_name).cloned() else { return Ok(None) };
    let sha256 = r.assets.iter().find(|x| x.name == format!("{zip_name}.sha256")).cloned();
    Ok(Some(AppRelease {
        tag: r.tag,
        version: format!("{a}.{b}.{c}"),
        published_at: r.published_at,
        html_url: r.html_url,
        notes: r.body,
        zip,
        sha256,
    }))
}

/// The highest-versioned release in a GitHub release list that carries the
/// Windows zip. Pure, so it is unit-tested offline. A tag's release order
/// on GitHub is by publish date, so the list is not trusted for ordering.
pub fn pick_latest_app_release(list_json: &str) -> Result<Option<AppRelease>> {
    let v: serde_json::Value = serde_json::from_str(list_json)?;
    let arr = v.as_array().ok_or_else(|| upd("release list is not an array"))?;
    let mut best: Option<((u32, u32, u32), AppRelease)> = None;
    for item in arr {
        let Ok(Some(r)) = app_release_from(&item.to_string()) else { continue };
        let Some(sv) = semver(&r.version) else { continue };
        if best.as_ref().map_or(true, |(b, _)| sv > *b) {
            best = Some((sv, r));
        }
    }
    Ok(best.map(|(_, r)| r))
}

/// Newest installable release, or None when the repository has published
/// none with a Windows zip.
pub fn latest_app_release() -> Result<Option<AppRelease>> {
    match update::get_json_opt(&format!("{RELEASES_API}?per_page=20"))? {
        Some(json) => pick_latest_app_release(&json),
        None => Ok(None),
    }
}

/// One release by tag (`v0.2.0`); an error when it has no Windows zip.
pub fn app_release_by_tag(tag: &str) -> Result<AppRelease> {
    let tag = if tag.starts_with('v') { tag.to_string() } else { format!("v{tag}") };
    let json = update::get_json_opt(&format!("{RELEASES_API}/tags/{tag}"))?
        .ok_or_else(|| upd(format!("no release tagged {tag} in {REPO}")))?;
    app_release_from(&json)?.ok_or_else(|| upd(format!("release {tag} has no {}", asset_name(&tag))))
}

// ------------------------------------------------------------------ check ----

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub current: Current,
    pub latest: Option<AppRelease>,
    /// The latest release's version is above this build's.
    pub update_available: bool,
    /// Why there is nothing to install, when there is not.
    pub note: Option<String>,
    /// A checkout of this repository to build from (`config.fidim_source`).
    pub source_dir: Option<PathBuf>,
    /// That folder looks like this repository.
    pub source_ok: bool,
    /// Sets staged earlier and not applied yet, newest first.
    pub staged: Vec<Staged>,
    pub history: Vec<HistoryEntry>,
}

/// A folder is a checkout of this repository.
pub fn is_checkout(dir: &Path) -> bool {
    dir.join("Cargo.toml").is_file()
        && dir.join("crates").join("fidim-cli").join("Cargo.toml").is_file()
        && dir.join("ui").join("src-tauri").join("tauri.conf.json").is_file()
}

pub fn check(cfg: &Config) -> Result<Check> {
    let current = current()?;
    let latest = latest_app_release()?;
    let (update_available, note) = match (&latest, semver(&current.version)) {
        (Some(l), Some(cur)) => match semver(&l.version) {
            Some(lv) if lv > cur => (true, None),
            Some(lv) if lv == cur => {
                let past = current.commits_ahead.is_some_and(|n| n > 0) || current.modified;
                (
                    false,
                    Some(if past {
                        format!("this build is past the {} release; nothing newer is published", l.tag)
                    } else {
                        format!("{} is the newest release", l.tag)
                    }),
                )
            }
            _ => (false, Some(format!("this build ({}) is newer than the newest release, {}", current.version, l.tag))),
        },
        (None, _) => (false, Some(format!("{REPO} has published no release with a Windows zip"))),
        (Some(_), None) => (false, Some(format!("this build's version {} is not X.Y.Z", current.version))),
    };
    let source_dir = cfg.fidim_source.clone();
    let source_ok = source_dir.as_deref().is_some_and(is_checkout);
    Ok(Check {
        current,
        latest,
        update_available,
        note,
        source_dir,
        source_ok,
        staged: staged_sets(),
        history: history(),
    })
}

// ------------------------------------------------------------------ stage ----

/// A verified set of the three executables, ready to apply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Staged {
    /// The folder holding the three executables (and `staged.json`).
    pub dir: PathBuf,
    /// `release` (a GitHub release) or `checkout` (built from source).
    pub source: String,
    /// The release tag, for a release.
    pub tag: Option<String>,
    /// The checkout, for a build.
    pub checkout: Option<PathBuf>,
    /// What the staged `fidim.exe --version` printed.
    pub version: String,
    pub staged_unix: u64,
}

/// Every staged set on disk, newest first.
pub fn staged_sets() -> Vec<Staged> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(updates_dir()) else { return out };
    for e in rd.flatten() {
        let p = e.path().join("stage").join(STAGED_JSON);
        if let Ok(text) = std::fs::read_to_string(&p) {
            if let Ok(s) = serde_json::from_str::<Staged>(&text) {
                if EXES.iter().all(|x| s.dir.join(x).is_file()) {
                    out.push(s);
                }
            }
        }
    }
    out.sort_by(|a, b| b.staged_unix.cmp(&a.staged_unix));
    out
}

/// Download a release's zip and its checksum, verify, unpack the
/// executables into a stage and check they run.
pub fn stage_release(rel: &AppRelease, progress: &mut dyn FnMut(String)) -> Result<Staged> {
    let root = updates_dir().join(&rel.tag);
    let stage = fresh_stage(&root)?;
    let zip_path = root.join(&rel.zip.name);
    progress(format!("downloading {} ({} MB)", rel.zip.name, rel.zip.size >> 20));
    update::download_url(&rel.zip.url, &rel.zip.name, &zip_path, progress)?;

    let expected = match &rel.sha256 {
        Some(sha) => {
            let p = root.join(&sha.name);
            update::download_url(&sha.url, &sha.name, &p, progress)?;
            let text = std::fs::read_to_string(&p).map_err(|e| crate::Error::io(&p, e))?;
            parse_sha256_file(&text)?
        }
        None => match rel.zip.digest.as_deref().and_then(|d| d.strip_prefix("sha256:")) {
            Some(d) => d.to_lowercase(),
            None => return Err(upd(format!("{} publishes no checksum for {}", rel.tag, rel.zip.name))),
        },
    };
    let actual = update::sha256_file(&zip_path)?;
    if actual != expected {
        let _ = std::fs::remove_file(&zip_path);
        return Err(upd(format!("{}: SHA-256 mismatch (published {expected}, downloaded {actual}); the download was deleted", rel.zip.name)));
    }
    progress(format!("{}: checksum OK", rel.zip.name));

    let n = extract_exes(&zip_path, &stage)?;
    progress(format!("unpacked {n} files into {}", stage.display()));
    finish_stage(stage, "release", Some(rel.tag.clone()), None, progress)
}

/// The first token of a `sha256sum`-style file (`<hex>  <name>`), lowercase.
pub fn parse_sha256_file(text: &str) -> Result<String> {
    let tok = text.split_whitespace().next().ok_or_else(|| upd("the .sha256 file is empty"))?;
    // sha256sum marks binary mode with a leading backslash on some builds.
    let tok = tok.trim_start_matches('\\');
    if tok.len() == 64 && tok.chars().all(|c| c.is_ascii_hexdigit()) {
        Ok(tok.to_lowercase())
    } else {
        Err(upd(format!("the .sha256 file does not start with a SHA-256: {tok}")))
    }
}

/// Unpack the executables (and the text files) out of a release zip into a
/// flat stage, whatever folder the zip nests them in. Every executable must
/// be present.
pub fn extract_exes(zip_path: &Path, stage: &Path) -> Result<usize> {
    let f = File::open(zip_path).map_err(|e| crate::Error::io(zip_path, e))?;
    let mut z = zip::ZipArchive::new(f).map_err(|e| upd(format!("{}: {e}", zip_path.display())))?;
    let mut count = 0;
    for i in 0..z.len() {
        let mut entry = z.by_index(i).map_err(|e| upd(format!("{}: entry {i}: {e}", zip_path.display())))?;
        if entry.is_dir() {
            continue;
        }
        let Some(rel) = entry.enclosed_name() else { continue };
        let Some(name) = rel.file_name().and_then(|n| n.to_str()) else { continue };
        if !EXES.contains(&name) && !DOCS.contains(&name) {
            continue;
        }
        let out = stage.join(name);
        let mut dst = File::create(&out).map_err(|e| crate::Error::io(&out, e))?;
        std::io::copy(&mut entry, &mut dst).map_err(|e| crate::Error::io(&out, e))?;
        count += 1;
    }
    for exe in EXES {
        if !stage.join(exe).is_file() {
            return Err(upd(format!("{} does not contain {exe}", zip_path.display())));
        }
    }
    Ok(count)
}

/// Build a checkout of this repository the way install.ps1 did and stage
/// what it built. The build names the commit it came from and whether the
/// tree had uncommitted changes, so the installed copy never passes for a
/// commit it does not match.
pub fn stage_from_checkout(src: &Path, progress: &mut dyn FnMut(String)) -> Result<Staged> {
    if !is_checkout(src) {
        return Err(upd(format!("{} is not a checkout of {REPO} (no Cargo.toml, crates/fidim-cli and ui/src-tauri)", src.display())));
    }
    let head = git(src, &["rev-parse", "HEAD"]);
    let dirty = git(src, &["status", "--porcelain", "--untracked-files=no"]).map(|s| !s.trim().is_empty());
    let mut env: Vec<(String, String)> = Vec::new();
    if let Some(h) = &head {
        env.push(("FIDIM_BUILD_ID".into(), format!("{}@{h}", src.display())));
        env.push(("FIDIM_BUILD_MODIFIED".into(), if dirty == Some(true) { "1" } else { "0" }.into()));
    }
    if dirty == Some(true) {
        progress("building uncommitted changes: the version will read <commit>-modified".into());
    }
    run_streamed(src, "cargo", &["build", "--release", "-p", "fidim-cli"], &env, progress)?;
    let ui = src.join("ui");
    if !ui.join("node_modules").is_dir() {
        run_streamed(&ui, "pnpm", &["install"], &env, progress)?;
    }
    run_streamed(&ui, "pnpm", &["tauri", "build", "--no-bundle"], &env, progress)?;

    let release = src.join("target").join("release");
    let short = head.as_deref().map(|h| h.chars().take(7).collect::<String>()).unwrap_or_else(|| "nogit".into());
    let root = updates_dir().join(format!("checkout-{short}"));
    let stage = fresh_stage(&root)?;
    for exe in EXES {
        let from = release.join(exe);
        if !from.is_file() {
            return Err(upd(format!("the build left no {}", from.display())));
        }
        std::fs::copy(&from, stage.join(exe)).map_err(|e| crate::Error::io(&from, e))?;
    }
    finish_stage(stage, "checkout", None, Some(src.to_path_buf()), progress)
}

fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git").env("GIT_OPTIONAL_LOCKS", "0").arg("-C").arg(dir).args(args).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// An empty `<root>/stage`.
fn fresh_stage(root: &Path) -> Result<PathBuf> {
    let stage = root.join("stage");
    let _ = std::fs::remove_dir_all(&stage);
    std::fs::create_dir_all(&stage).map_err(|e| crate::Error::io(&stage, e))?;
    Ok(stage)
}

/// The stage holds every executable and its `fidim.exe` runs; record it.
fn finish_stage(
    stage: PathBuf,
    source: &str,
    tag: Option<String>,
    checkout: Option<PathBuf>,
    progress: &mut dyn FnMut(String),
) -> Result<Staged> {
    for exe in EXES {
        if !stage.join(exe).is_file() {
            return Err(upd(format!("the staged files lack {exe}")));
        }
    }
    let version = crate::launch::run_capture(&stage.join("fidim.exe"), &["--version"], None)
        .map(|s| s.trim().to_string())
        .map_err(|e| upd(format!("the staged fidim.exe does not run: {e}")))?;
    let s = Staged { dir: stage.clone(), source: source.into(), tag, checkout, version: version.clone(), staged_unix: now_unix() };
    let p = stage.join(STAGED_JSON);
    std::fs::write(&p, serde_json::to_string_pretty(&s)?).map_err(|e| crate::Error::io(&p, e))?;
    progress(format!("staged: {version}"));
    Ok(s)
}

/// Run a build step with its output streamed line by line. `cmd /C` so
/// `pnpm` (a .cmd shim) and `cargo` are found the same way a shell finds
/// them.
fn run_streamed(cwd: &Path, program: &str, args: &[&str], env: &[(String, String)], progress: &mut dyn FnMut(String)) -> Result<()> {
    progress(format!("$ {program} {}   (in {})", args.join(" "), cwd.display()));
    let mut cmd = Command::new("cmd");
    cmd.arg("/C").arg(program).args(args).current_dir(cwd);
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    crate::launch::hide_console(&mut cmd);
    let mut child = cmd.spawn().map_err(|e| upd(format!("start {program}: {e}")))?;
    let (tx, rx) = mpsc::channel::<String>();
    let mut readers = Vec::new();
    if let Some(out) = child.stdout.take() {
        let tx = tx.clone();
        readers.push(std::thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(|l| l.ok()) {
                let _ = tx.send(line);
            }
        }));
    }
    if let Some(err) = child.stderr.take() {
        readers.push(std::thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(|l| l.ok()) {
                let _ = tx.send(line);
            }
        }));
    }
    for line in rx {
        let t = line.trim_end();
        if !t.is_empty() {
            progress(t.to_string());
        }
    }
    for r in readers {
        let _ = r.join();
    }
    let status = child.wait().map_err(|e| upd(format!("wait for {program}: {e}")))?;
    if !status.success() {
        return Err(upd(format!("{program} {} exited {status}", args.join(" "))));
    }
    Ok(())
}

// ------------------------------------------------------------------ apply ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub at_unix: u64,
    /// `fidim --version` of what was installed before, or `none`.
    pub from: String,
    pub to: String,
    pub source: String,
    pub tag: Option<String>,
    pub install_dir: PathBuf,
    pub ok: bool,
    pub detail: String,
}

pub fn history() -> Vec<HistoryEntry> {
    history_in(&updates_dir())
}

fn history_in(dir: &Path) -> Vec<HistoryEntry> {
    let p = dir.join(HISTORY_JSON);
    std::fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str::<Vec<HistoryEntry>>(&t).ok())
        .map(|mut v| {
            v.sort_by(|a, b| b.at_unix.cmp(&a.at_unix));
            v
        })
        .unwrap_or_default()
}

fn push_history(dir: &Path, e: HistoryEntry) {
    let _ = std::fs::create_dir_all(dir);
    let mut all = history_in(dir);
    all.insert(0, e);
    all.truncate(50);
    if let Ok(text) = serde_json::to_string_pretty(&all) {
        let _ = std::fs::write(dir.join(HISTORY_JSON), text);
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ApplyReport {
    pub install_dir: PathBuf,
    pub from: String,
    pub to: String,
    pub replaced: Vec<String>,
    /// Files that were in use and now run from a `.old-<time>` name.
    pub moved_aside: Vec<String>,
    pub relaunched: bool,
    /// Why the app did not reopen, when it was asked to and did not. The
    /// files are installed regardless.
    pub relaunch_error: Option<String>,
}

/// Start the staged `fidim.exe` detached to do the replacing once the
/// process `wait_pid` (normally the caller) has exited. Returns the
/// updater's pid; its output goes to `<updates_dir>/apply.log`.
pub fn spawn_apply(stage: &Path, install_dir: &Path, wait_pid: Option<u32>, relaunch: bool) -> Result<u32> {
    let updater = stage.join("fidim.exe");
    if !updater.is_file() {
        return Err(upd(format!("{} is not a staged set (no fidim.exe)", stage.display())));
    }
    let dir = updates_dir();
    std::fs::create_dir_all(&dir).map_err(|e| crate::Error::io(&dir, e))?;
    let log_path = dir.join("apply.log");
    let log = File::create(&log_path).map_err(|e| crate::Error::io(&log_path, e))?;
    let log_err = log.try_clone().map_err(|e| crate::Error::io(&log_path, e))?;
    let mut cmd = Command::new(&updater);
    cmd.arg("self-update").arg("apply").arg("--stage").arg(stage).arg("--install-dir").arg(install_dir);
    if let Some(pid) = wait_pid {
        cmd.arg("--wait-pid").arg(pid.to_string());
    }
    if relaunch {
        cmd.arg("--relaunch");
    }
    cmd.stdin(Stdio::null()).stdout(Stdio::from(log)).stderr(Stdio::from(log_err));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    let child = cmd.spawn().map_err(|e| upd(format!("start the updater {}: {e}", updater.display())))?;
    Ok(child.id())
}

/// Replace the installed executables with the staged ones. Runs in the
/// staged `fidim.exe` (see `spawn_apply`); refused from an executable inside
/// `install_dir`, which could not replace itself. With `relaunch`, any
/// desktop app still open from `install_dir` is closed first and the app is
/// reopened at the end.
pub fn apply(
    stage: &Path,
    install_dir: &Path,
    wait_pid: Option<u32>,
    relaunch: bool,
    log: &mut dyn FnMut(String),
) -> Result<ApplyReport> {
    apply_in(stage, install_dir, wait_pid, relaunch, log, &updates_dir())
}

/// `apply` with the folder that keeps the history named, for tests.
fn apply_in(
    stage: &Path,
    install_dir: &Path,
    wait_pid: Option<u32>,
    relaunch: bool,
    log: &mut dyn FnMut(String),
    updates: &Path,
) -> Result<ApplyReport> {
    for exe in EXES {
        if !stage.join(exe).is_file() {
            return Err(upd(format!("{} lacks {exe}", stage.display())));
        }
    }
    if let Ok(me) = std::env::current_exe() {
        if me.parent().is_some_and(|p| same_dir(p, install_dir)) {
            return Err(upd(format!("{} runs from {}; run the apply step from the staged copy", me.display(), install_dir.display())));
        }
    }
    let staged: Option<Staged> =
        std::fs::read_to_string(stage.join(STAGED_JSON)).ok().and_then(|t| serde_json::from_str(&t).ok());

    if let Some(pid) = wait_pid {
        wait_for_exit(pid, Duration::from_secs(60), log);
    }
    if relaunch {
        // The app is about to be reopened on the new files; a copy still
        // open from this folder (an update started from the terminal, or a
        // second window) would otherwise stay on the old one beside it.
        close_gui_instances(install_dir, log);
    }
    std::fs::create_dir_all(install_dir).map_err(|e| crate::Error::io(install_dir, e))?;
    let from = crate::launch::run_capture(&install_dir.join("fidim.exe"), &["--version"], None)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "none".into());
    let to = staged.as_ref().map(|s| s.version.clone()).unwrap_or_else(|| "?".into());
    log(format!("installing {to} over {from} in {}", install_dir.display()));

    let stamp = {
        let t = now_unix();
        format!("{t}")
    };
    let mut replaced = Vec::new();
    let mut moved_aside = Vec::new();
    let mut result: Result<()> = Ok(());
    for exe in EXES {
        remove_old_copies(install_dir, exe, log);
        let dest = install_dir.join(exe);
        if dest.exists() {
            if let Err(e) = std::fs::remove_file(&dest) {
                // In use: Windows lets a running image be renamed but not
                // deleted or overwritten. It keeps running from the new name.
                let aside = install_dir.join(format!("{exe}.old-{stamp}"));
                match std::fs::rename(&dest, &aside) {
                    Ok(()) => {
                        log(format!("{exe} is in use (delete: {e}); moved aside as {}", aside.display()));
                        moved_aside.push(exe.to_string());
                    }
                    Err(e2) => {
                        result = Err(upd(format!("{exe} is in use and could not be moved aside: {e2}")));
                        break;
                    }
                }
            }
        }
        if let Err(e) = std::fs::copy(stage.join(exe), &dest) {
            result = Err(upd(format!("copy {exe} into {}: {e}", install_dir.display())));
            break;
        }
        log(format!("{exe}: installed"));
        replaced.push(exe.to_string());
    }

    let (source, tag) = staged.as_ref().map(|s| (s.source.clone(), s.tag.clone())).unwrap_or_else(|| ("?".into(), None));
    let entry = |ok: bool, detail: String| HistoryEntry {
        at_unix: now_unix(),
        from: from.clone(),
        to: to.clone(),
        source: source.clone(),
        tag: tag.clone(),
        install_dir: install_dir.to_path_buf(),
        ok,
        detail,
    };
    if let Err(e) = &result {
        push_history(updates, entry(false, e.to_string()));
    }
    result?;

    // The files are in place from here; a relaunch that fails is reported,
    // not an update that failed.
    let mut notes: Vec<String> = Vec::new();
    if !moved_aside.is_empty() {
        notes.push(format!("in use, moved aside: {}", moved_aside.join(", ")));
    }
    let mut relaunched = false;
    let mut relaunch_error = None;
    if relaunch {
        match relaunch_gui(install_dir) {
            Ok(pid) => {
                log(format!("reopened the desktop app (pid {pid})"));
                relaunched = true;
            }
            Err(e) => {
                log(format!("the desktop app did not reopen: {e}"));
                notes.push(format!("relaunch failed: {e}"));
                relaunch_error = Some(e.to_string());
            }
        }
    }
    push_history(updates, entry(true, notes.join("; ")));
    Ok(ApplyReport { install_dir: install_dir.to_path_buf(), from, to, replaced, moved_aside, relaunched, relaunch_error })
}

/// Delete `<name>.old-*` copies earlier updates renamed aside, where
/// nothing runs from them any more (a copy still in use refuses).
fn remove_old_copies(dir: &Path, name: &str, log: &mut dyn FnMut(String)) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let prefix = format!("{name}.old-");
    for e in rd.flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if n.starts_with(&prefix) && std::fs::remove_file(e.path()).is_ok() {
            log(format!("removed {n}"));
        }
    }
}

/// Start the desktop app from `install_dir` and confirm it stays up.
/// First out of the caller's job object, so an update started from a
/// terminal or an agent does not end with that session; when the job
/// forbids breaking away, as a plain child. (Handing the path to
/// explorer.exe, the usual trick, starts nothing from a windowless
/// process.) Returns the new pid.
pub fn relaunch_gui(install_dir: &Path) -> Result<u32> {
    let exe = install_dir.join(GUI_EXE);
    if !exe.is_file() {
        return Err(upd(format!("{} is missing", exe.display())));
    }
    let spawn = |breakaway: bool| {
        let mut cmd = Command::new(&exe);
        cmd.current_dir(install_dir).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
            const CREATE_BREAKAWAY_FROM_JOB: u32 = 0x0100_0000;
            cmd.creation_flags(if breakaway { CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB } else { CREATE_NEW_PROCESS_GROUP });
        }
        #[cfg(not(windows))]
        let _ = breakaway;
        cmd.spawn()
    };
    let mut child = match spawn(true) {
        Ok(c) => c,
        Err(_) => spawn(false).map_err(|e| upd(format!("start {}: {e}", exe.display())))?,
    };
    std::thread::sleep(Duration::from_millis(1500));
    match child.try_wait() {
        Ok(Some(status)) => Err(upd(format!("{GUI_EXE} exited right after starting ({status})"))),
        _ => Ok(child.id()),
    }
}

/// End every `llama-fidim.exe` running from `install_dir`, other than this
/// process, and give them a moment to go. Only the desktop app: servers
/// (`fidim-dg.exe`) and helpers (`fidim.exe keepalive`) are never touched.
#[cfg(windows)]
fn close_gui_instances(install_dir: &Path, log: &mut dyn FnMut(String)) {
    use windows::Win32::Foundation::{CloseHandle, HANDLE, MAX_PATH, WAIT_OBJECT_0};
    use windows::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Process32FirstW, Process32NextW, PROCESSENTRY32W, TH32CS_SNAPPROCESS,
    };
    use windows::Win32::System::Threading::{
        OpenProcess, QueryFullProcessImageNameW, TerminateProcess, WaitForSingleObject, PROCESS_NAME_WIN32,
        PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE, PROCESS_TERMINATE,
    };
    let me = std::process::id();
    let want = GUI_EXE.to_lowercase();
    let mut closed: Vec<HANDLE> = Vec::new();
    unsafe {
        let Ok(snap) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else { return };
        let mut e = PROCESSENTRY32W { dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32, ..Default::default() };
        let mut ok = Process32FirstW(snap, &mut e).is_ok();
        while ok {
            let name = String::from_utf16_lossy(&e.szExeFile).trim_end_matches('\0').to_lowercase();
            if name == want && e.th32ProcessID != me {
                let pid = e.th32ProcessID;
                if let Ok(h) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE | PROCESS_SYNCHRONIZE, false, pid) {
                    let mut buf = [0u16; MAX_PATH as usize * 2];
                    let mut len = buf.len() as u32;
                    let path = QueryFullProcessImageNameW(h, PROCESS_NAME_WIN32, windows::core::PWSTR(buf.as_mut_ptr()), &mut len)
                        .ok()
                        .map(|_| PathBuf::from(String::from_utf16_lossy(&buf[..len as usize])));
                    let here = path.as_deref().and_then(Path::parent).is_some_and(|d| same_dir(d, install_dir));
                    if here && TerminateProcess(h, 0).is_ok() {
                        log(format!("closed the running app (pid {pid})"));
                        closed.push(h);
                    } else {
                        let _ = CloseHandle(h);
                    }
                }
            }
            ok = Process32NextW(snap, &mut e).is_ok();
        }
        let _ = CloseHandle(snap);
        for h in closed {
            if WaitForSingleObject(h, 5_000) != WAIT_OBJECT_0 {
                log("a closed app instance has not exited after 5 s; continuing".into());
            }
            let _ = CloseHandle(h);
        }
    }
}

#[cfg(not(windows))]
fn close_gui_instances(_install_dir: &Path, _log: &mut dyn FnMut(String)) {}

/// Block until process `pid` has exited, or `timeout` has passed (the
/// caller proceeds either way and the copy step reports what is in use).
#[cfg(windows)]
fn wait_for_exit(pid: u32, timeout: Duration, log: &mut dyn FnMut(String)) {
    use windows::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
    use windows::Win32::System::Threading::{OpenProcess, WaitForSingleObject, PROCESS_SYNCHRONIZE};
    let Ok(h) = (unsafe { OpenProcess(PROCESS_SYNCHRONIZE, false, pid) }) else {
        log(format!("pid {pid} has already exited"));
        return;
    };
    let started = Instant::now();
    log(format!("waiting for pid {pid} to exit"));
    let r = unsafe { WaitForSingleObject(h, timeout.as_millis() as u32) };
    let _ = unsafe { CloseHandle(h) };
    if r == WAIT_OBJECT_0 {
        log(format!("pid {pid} exited after {:.1} s", started.elapsed().as_secs_f32()));
    } else {
        log(format!("pid {pid} still runs after {} s; continuing", timeout.as_secs()));
    }
}

#[cfg(not(windows))]
fn wait_for_exit(_pid: u32, _timeout: Duration, _log: &mut dyn FnMut(String)) {}

/// Append one line to the apply log kept beside the stages (the updater's
/// stdout also goes there; this is for a caller that has no console).
pub fn append_log(line: &str) {
    let p = updates_dir().join("apply.log");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
        let _ = writeln!(f, "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semver_parses_tags_and_orders() {
        assert_eq!(semver("v0.2.0"), Some((0, 2, 0)));
        assert_eq!(semver("0.10.3"), Some((0, 10, 3)));
        assert_eq!(semver("b10984"), None);
        assert_eq!(semver("v1.2"), None);
        assert_eq!(semver("v1.2.3.4"), None);
        assert!(semver("v0.10.0") > semver("v0.9.9"));
        assert_eq!(asset_name("v0.2.0"), "llama-fidim-v0.2.0-win-x64.zip");
    }

    fn release(tag: &str, with_zip: bool, draft: bool) -> serde_json::Value {
        let mut assets = vec![serde_json::json!({ "name": "notes.txt", "browser_download_url": "https://x/notes.txt", "size": 1 })];
        if with_zip {
            let z = asset_name(tag);
            assets.push(serde_json::json!({ "name": z, "browser_download_url": format!("https://x/{z}"), "size": 9_000_000 }));
            assets.push(serde_json::json!({ "name": format!("{z}.sha256"), "browser_download_url": format!("https://x/{z}.sha256"), "size": 97 }));
        }
        serde_json::json!({ "tag_name": tag, "draft": draft, "published_at": "2026-09-18T20:31:47Z", "html_url": format!("https://github.com/{REPO}/releases/tag/{tag}"), "name": tag, "body": "notes", "assets": assets })
    }

    #[test]
    fn picks_the_highest_version_that_ships_the_zip() {
        // Listed by publish date, not version; 0.3.1 has no zip; 0.4.0 is a draft.
        let list = serde_json::json!([release("v0.3.1", false, false), release("v0.2.0", true, false), release("v0.3.0", true, false), release("v0.4.0", true, true), release("b123", true, false)]);
        let r = pick_latest_app_release(&list.to_string()).unwrap().unwrap();
        assert_eq!(r.tag, "v0.3.0");
        assert_eq!(r.version, "0.3.0");
        assert_eq!(r.zip.name, "llama-fidim-v0.3.0-win-x64.zip");
        assert_eq!(r.sha256.as_ref().map(|a| a.name.as_str()), Some("llama-fidim-v0.3.0-win-x64.zip.sha256"));
        assert!(pick_latest_app_release("[]").unwrap().is_none());
    }

    #[test]
    fn sha256_file_first_token() {
        let hex = "e263f7a82940a00beba1fccb6dc93ed66055d3c3349747e57110f92077b55637";
        assert_eq!(parse_sha256_file(&format!("{hex}  llama-fidim-v0.2.0-win-x64.zip\n")).unwrap(), hex);
        assert_eq!(parse_sha256_file(&format!("\\{} *x.zip", hex.to_uppercase())).unwrap(), hex);
        assert!(parse_sha256_file("").is_err());
        assert!(parse_sha256_file("deadbeef  x.zip").is_err());
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fidim-selfupdate-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn extracts_the_executables_out_of_the_nested_folder() {
        let d = tmp("zip");
        let zip_path = d.join("r.zip");
        {
            let f = File::create(&zip_path).unwrap();
            let mut w = zip::ZipWriter::new(f);
            let o = zip::write::SimpleFileOptions::default();
            for (name, body) in [("llama-fidim-v9.9.9-win-x64/fidim.exe", "A"), ("llama-fidim-v9.9.9-win-x64/fidim-dg.exe", "B"), ("llama-fidim-v9.9.9-win-x64/llama-fidim.exe", "C"), ("llama-fidim-v9.9.9-win-x64/README.md", "R"), ("llama-fidim-v9.9.9-win-x64/other.dll", "X")] {
                w.start_file(name, o).unwrap();
                w.write_all(body.as_bytes()).unwrap();
            }
            w.finish().unwrap();
        }
        let stage = d.join("stage");
        std::fs::create_dir_all(&stage).unwrap();
        assert_eq!(extract_exes(&zip_path, &stage).unwrap(), 4);
        assert_eq!(std::fs::read_to_string(stage.join("fidim-dg.exe")).unwrap(), "B");
        assert!(!stage.join("other.dll").exists());
        assert!(!stage.join("llama-fidim-v9.9.9-win-x64").exists());

        // A zip missing one executable is refused.
        let zip2 = d.join("short.zip");
        {
            let mut w = zip::ZipWriter::new(File::create(&zip2).unwrap());
            w.start_file("fidim.exe", zip::write::SimpleFileOptions::default()).unwrap();
            w.write_all(b"A").unwrap();
            w.finish().unwrap();
        }
        let stage2 = d.join("stage2");
        std::fs::create_dir_all(&stage2).unwrap();
        assert!(extract_exes(&zip2, &stage2).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_checkout_is_recognised_by_its_shape() {
        let d = tmp("checkout");
        assert!(!is_checkout(&d));
        std::fs::write(d.join("Cargo.toml"), "[workspace]").unwrap();
        std::fs::create_dir_all(d.join("crates/fidim-cli")).unwrap();
        std::fs::write(d.join("crates/fidim-cli/Cargo.toml"), "").unwrap();
        std::fs::create_dir_all(d.join("ui/src-tauri")).unwrap();
        std::fs::write(d.join("ui/src-tauri/tauri.conf.json"), "{}").unwrap();
        assert!(is_checkout(&d));
        // This very repository qualifies.
        assert!(is_checkout(Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").as_path()));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn apply_refuses_to_run_from_inside_the_install_folder() {
        // The test binary's own folder as the install dir: apply must refuse
        // before touching anything, whatever the stage holds.
        let d = tmp("refuse");
        for exe in EXES {
            std::fs::write(d.join(exe), "x").unwrap();
        }
        let me = std::env::current_exe().unwrap();
        let mut lines = Vec::new();
        let r = apply_in(&d, me.parent().unwrap(), None, false, &mut |l| lines.push(l), &d.join("updates"));
        let msg = r.err().map(|e| e.to_string()).unwrap_or_default();
        assert!(msg.contains("staged copy"), "{msg}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn apply_copies_every_executable_and_clears_old_copies() {
        let d = tmp("apply");
        let stage = d.join("stage");
        let dest = d.join("install");
        std::fs::create_dir_all(&stage).unwrap();
        std::fs::create_dir_all(&dest).unwrap();
        for exe in EXES {
            std::fs::write(stage.join(exe), format!("new {exe}")).unwrap();
            std::fs::write(dest.join(exe), format!("old {exe}")).unwrap();
        }
        std::fs::write(dest.join("fidim.exe.old-1"), "stale").unwrap();
        let mut lines = Vec::new();
        let updates = d.join("updates");
        let r = apply_in(&stage, &dest, None, false, &mut |l| lines.push(l), &updates).unwrap();
        let h = history_in(&updates);
        assert_eq!(h.len(), 1);
        assert!(h[0].ok && h[0].from == "none", "{h:?}");
        assert_eq!(r.replaced.len(), 3);
        assert!(r.moved_aside.is_empty());
        assert!(!r.relaunched && r.relaunch_error.is_none());
        for exe in EXES {
            assert_eq!(std::fs::read_to_string(dest.join(exe)).unwrap(), format!("new {exe}"));
        }
        assert!(!dest.join("fidim.exe.old-1").exists(), "stale copy should be gone");
        assert!(lines.iter().any(|l| l.contains("removed fidim.exe.old-1")), "{lines:?}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
