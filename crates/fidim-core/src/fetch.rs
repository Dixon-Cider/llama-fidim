//! Resumable, verified downloads: model files now, build zips later.
//!
//! A file is written to `<dest>.part` and renamed into place only once its
//! size and SHA-256 check out, so a partial download is never mistaken for a
//! model (discovery skips `*.part`). An interrupted transfer resumes with a
//! Range request after re-hashing what is already on disk. Every attempt
//! requests the original URL again, so a redirect to a signed CDN URL (the
//! Hub's expire after an hour) is resolved fresh each time; a token is sent
//! to the URL's own host only (see `hub::agent`). Failed attempts back off
//! and retry; a cancel leaves the `.part` for a later resume.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::HttpErrorKind;
use crate::{hub, Error, Result};

/// Progress callbacks are at most this frequent (plus one per stage change
/// and one at the end).
const REPORT_EVERY: Duration = Duration::from_millis(250);
const CHUNK: usize = 1 << 20;
/// A stalled read fails after this long and the attempt is retried.
const READ_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_BACKOFF: Duration = Duration::from_secs(60);
const MAX_RATE_LIMIT_WAIT: Duration = Duration::from_secs(300);

/// What to download.
#[derive(Debug, Clone)]
pub struct FetchReq {
    /// A stable URL, such as a Hub `/resolve/<sha>/<path>` URL. It is
    /// requested again on every attempt, so the signed URL it redirects to
    /// is never reused after it expires.
    pub url: String,
    /// Bearer token for the URL's host; never sent across a redirect to
    /// another host.
    pub token: Option<String>,
    pub expected_size: Option<u64>,
    /// Hex SHA-256 (the Hub's LFS oid).
    pub expected_sha256: Option<String>,
    /// Consecutive failed attempts allowed; an attempt that transfers
    /// anything resets the count.
    pub attempts: u32,
    /// Delay before the first retry, doubling per failure up to a minute.
    pub backoff: Duration,
}

impl FetchReq {
    pub fn new(url: impl Into<String>) -> Self {
        FetchReq {
            url: url.into(),
            token: None,
            expected_size: None,
            expected_sha256: None,
            attempts: 5,
            backoff: Duration::from_secs(2),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stage {
    /// Re-reading bytes already on disk (a resumed `.part`, or a finished
    /// file being verified) to seed the SHA-256.
    Hashing,
    Downloading,
}

#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    /// The destination's file name.
    pub file: String,
    pub stage: Stage,
    pub done: u64,
    pub total: Option<u64>,
    /// Bytes per second, smoothed; 0 while hashing.
    pub bps: f64,
}

/// `<dest>.part`: where `dest` is written until it is verified.
pub fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".part");
    dest.with_file_name(name)
}

/// Download `req.url` to `dest`, resuming `<dest>.part` if one is there, and
/// return the file's size. On success `dest` holds exactly the expected
/// bytes and the `.part` is gone. On a mismatch nothing is deleted: the
/// error names the `.part` to remove before trying again. When `dest`
/// already exists it is verified instead of downloaded.
pub fn download_resumable(
    req: &FetchReq,
    dest: &Path,
    progress: &mut dyn FnMut(&Progress),
    cancel: &AtomicBool,
) -> Result<u64> {
    let expected_sha = match req.expected_sha256.as_deref().map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()) => Some(s),
        Some(s) => return Err(Error::InvalidInput(format!("{s:?} is not a hex SHA-256"))),
        None => None,
    };
    let mut reporter = Reporter::new(dest, req.expected_size, progress);
    if dest.exists() {
        return verify_existing(dest, req.expected_size, expected_sha.as_deref(), &mut reporter, cancel);
    }
    if let Some(parent) = dest.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let mut part = Part::open(part_path(dest))?;
    if req.expected_size.is_some_and(|size| part.have > size) {
        // Longer than the file can be: not a prefix of it.
        part.restart()?;
    }
    if part.have > 0 {
        part.rehash(&mut reporter, cancel)?;
    }

    let ctx = Ctx {
        agent: hub::agent(READ_TIMEOUT),
        req,
        what: reporter.file.clone(),
        repo: hub::repo_of_resolve_url(&req.url),
    };
    let mut total = req.expected_size;
    let mut failures = 0u32;
    let outcome = loop {
        if req.expected_size.is_some_and(|size| part.have == size) {
            break Ok(());
        }
        if cancel.load(Ordering::Relaxed) {
            break Err(Error::Cancelled);
        }
        let received_before = part.received;
        match attempt(&ctx, &mut part, &mut total, &mut reporter, cancel) {
            Ok(Attempt::Complete) => break Ok(()),
            Ok(Attempt::Cancelled) => break Err(Error::Cancelled),
            Ok(Attempt::Retry { wait, error }) => {
                if part.received > received_before {
                    failures = 0;
                }
                failures += 1;
                if failures >= req.attempts.max(1) {
                    break Err(error);
                }
                let delay = wait.unwrap_or_else(|| {
                    let doublings = (failures - 1).min(16);
                    req.backoff.saturating_mul(1u32 << doublings).min(MAX_BACKOFF)
                });
                if !sleep_unless_cancelled(delay, cancel) {
                    break Err(Error::Cancelled);
                }
            }
            Err(e) => break Err(e),
        }
    };
    // Whatever happened, what arrived is on disk for the next attempt.
    part.finish()?;
    outcome?;
    reporter.report(Stage::Downloading, part.have, true);

    let (have, path) = (part.have, part.path.clone());
    if let Some(size) = req.expected_size.filter(|size| *size != have) {
        return Err(Error::Integrity {
            path,
            detail: format!("downloaded {have} bytes, expected {size}; delete this file to download it again"),
        });
    }
    let got = part.digest();
    if let Some(want) = expected_sha.filter(|want| *want != got) {
        return Err(Error::Integrity {
            path,
            detail: format!("SHA-256 is {got}, expected {want}; delete this file to download it again"),
        });
    }
    rename_into_place(&path, dest)?;
    Ok(have)
}

/// The `.part` being written: its bytes so far and their running hash.
struct Part {
    file: File,
    path: PathBuf,
    hasher: Sha256,
    /// Bytes in the file.
    have: u64,
    /// Bytes received over the network since it was opened (a restart
    /// lowers `have`, never this).
    received: u64,
}

impl Part {
    fn open(path: PathBuf) -> Result<Part> {
        let mut opts = OpenOptions::new();
        opts.read(true).write(true).create(true).truncate(false);
        // One writer at a time: a second download of the same file fails
        // to open it (sharing violation) instead of interleaving bytes.
        // Readers, such as a virus scanner, are still let in.
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            const FILE_SHARE_READ: u32 = 0x1;
            opts.share_mode(FILE_SHARE_READ);
        }
        let file = opts.open(&path).map_err(|e| Error::io(&path, e))?;
        let have = file.metadata().map_err(|e| Error::io(&path, e))?.len();
        Ok(Part { file, path, hasher: Sha256::new(), have, received: 0 })
    }

    /// Hash the bytes already in the file, leaving it positioned to append.
    fn rehash(&mut self, reporter: &mut Reporter<'_>, cancel: &AtomicBool) -> Result<()> {
        self.file.seek(SeekFrom::Start(0)).map_err(|e| Error::io(&self.path, e))?;
        hash_into(&mut self.file, &self.path, self.have, &mut self.hasher, reporter, cancel)?;
        self.file.seek(SeekFrom::End(0)).map_err(|e| Error::io(&self.path, e))?;
        Ok(())
    }

    /// Empty the file and the hash: the next bytes start the file.
    fn restart(&mut self) -> Result<()> {
        self.file.set_len(0).map_err(|e| Error::io(&self.path, e))?;
        self.file.seek(SeekFrom::Start(0)).map_err(|e| Error::io(&self.path, e))?;
        self.hasher = Sha256::new();
        self.have = 0;
        Ok(())
    }

    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        self.file.write_all(bytes).map_err(|e| Error::io(&self.path, e))?;
        self.hasher.update(bytes);
        self.have += bytes.len() as u64;
        self.received += bytes.len() as u64;
        Ok(())
    }

    /// Flush to disk before the file is left for a resume or renamed.
    fn finish(&mut self) -> Result<()> {
        self.file.flush().map_err(|e| Error::io(&self.path, e))?;
        self.file.sync_all().map_err(|e| Error::io(&self.path, e))
    }

    /// Hex SHA-256 of the file; closes it.
    fn digest(self) -> String {
        hex(&self.hasher.finalize())
    }
}

/// What every attempt of one download shares.
struct Ctx<'a> {
    agent: ureq::Agent,
    req: &'a FetchReq,
    /// The file name, for messages.
    what: String,
    /// The Hub repo of a `/resolve/` URL, for the gated-repo message.
    repo: Option<String>,
}

enum Attempt {
    Complete,
    Cancelled,
    /// Try again after `wait` (None = the backoff), failing with `error`
    /// when out of attempts.
    Retry { wait: Option<Duration>, error: Error },
}

/// One request: resume from what the `.part` holds, append what arrives.
fn attempt(
    ctx: &Ctx<'_>,
    part: &mut Part,
    total: &mut Option<u64>,
    reporter: &mut Reporter<'_>,
    cancel: &AtomicBool,
) -> Result<Attempt> {
    let (req, what) = (ctx.req, ctx.what.as_str());
    let mut request = ctx.agent.get(&req.url);
    if let Some(t) = &req.token {
        request = request.set("Authorization", &format!("Bearer {t}"));
    }
    if part.have > 0 {
        request = request.set("Range", &format!("bytes={}-", part.have));
    }
    let resp = match request.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(416, resp)) => {
            // Asked for bytes past the end: the .part is the whole file, or
            // longer than it.
            let size = resp.header("content-range").and_then(hub::content_range_total).or(req.expected_size);
            if size == Some(part.have) {
                return Ok(Attempt::Complete);
            }
            part.restart()?;
            return Ok(Attempt::Retry {
                wait: Some(Duration::ZERO),
                error: Error::Integrity {
                    path: part.path.clone(),
                    detail: "the server refused to resume this partial file".into(),
                },
            });
        }
        Err(ureq::Error::Status(status, resp)) => {
            // A redirect took the request to another host (a CDN): its 403
            // is an expired signature, not a refusal.
            let redirected = host_of(resp.get_url()) != host_of(&req.url);
            let error = hub::response_error(status, resp, what, ctx.repo.as_deref(), req.token.is_some());
            return retry_or_fail(error, redirected);
        }
        Err(ureq::Error::Transport(t)) => {
            return Ok(Attempt::Retry { wait: None, error: hub::network_error(what, t) });
        }
    };

    match resp.status() {
        206 => match resp.header("content-range").and_then(hub::content_range) {
            Some((first, size)) if first == part.have => {
                if let Some(size) = size {
                    check_size(req, size, &part.path)?;
                    *total = Some(size);
                }
            }
            _ => {
                // Not the bytes that were asked for: start over.
                part.restart()?;
                return Ok(Attempt::Retry {
                    wait: Some(Duration::ZERO),
                    error: Error::Http {
                        kind: HttpErrorKind::Malformed,
                        message: format!("{what}: the server answered a resume with the wrong byte range"),
                    },
                });
            }
        },
        200 => {
            // The whole file, whatever was asked for.
            if part.have > 0 {
                part.restart()?;
            }
            if let Some(size) = resp.header("content-length").and_then(|v| v.trim().parse::<u64>().ok()) {
                check_size(req, size, &part.path)?;
                *total = Some(size);
            }
        }
        status => {
            return Err(Error::Http {
                kind: HttpErrorKind::Status { status },
                message: format!("{what}: unexpected HTTP {status} for a download"),
            })
        }
    }
    reporter.total = *total;

    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; CHUNK];
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Ok(Attempt::Cancelled);
        }
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => return Ok(Attempt::Retry { wait: None, error: hub::network_error(what, e) }),
        };
        part.append(&buf[..n])?;
        reporter.report(Stage::Downloading, part.have, false);
    }
    let have = part.have;
    match *total {
        Some(size) if have < size => Ok(Attempt::Retry {
            wait: None,
            error: hub::network_error(what, format!("the connection closed at {have} of {size} bytes")),
        }),
        Some(size) if have > size => Err(Error::Integrity {
            path: part.path.clone(),
            detail: format!("received {have} bytes of a {size}-byte file; delete this file to download it again"),
        }),
        _ => Ok(Attempt::Complete),
    }
}

/// Which failures are worth another attempt: the network, rate limits,
/// server errors, timeouts, and a 403 from the host a redirect led to (an
/// expired signed CDN URL, which the next attempt re-resolves). A gated or
/// missing repo or file, or a 403 from the URL's own host, is not.
fn retry_or_fail(error: Error, redirected: bool) -> Result<Attempt> {
    let wait = match &error {
        Error::Http { kind: HttpErrorKind::Network, .. } => None,
        Error::Http { kind: HttpErrorKind::RateLimited { retry_after_secs }, .. } => {
            Some(retry_after_secs.map(Duration::from_secs).unwrap_or(MAX_BACKOFF).min(MAX_RATE_LIMIT_WAIT))
        }
        Error::Http { kind: HttpErrorKind::Status { status }, .. }
            if *status >= 500 || *status == 408 || (*status == 403 && redirected) =>
        {
            None
        }
        _ => return Err(error),
    };
    Ok(Attempt::Retry { wait, error })
}

/// `host[:port]` of a URL, lowercased.
fn host_of(url: &str) -> String {
    let rest = url.split_once("://").map_or(url, |(_, r)| r);
    rest.split(['/', '?', '#']).next().unwrap_or("").to_ascii_lowercase()
}

fn check_size(req: &FetchReq, size: u64, part: &Path) -> Result<()> {
    match req.expected_size {
        Some(want) if want != size => Err(Error::Integrity {
            path: part.to_path_buf(),
            detail: format!("the server has {size} bytes, the repo listing says {want}; the file changed"),
        }),
        _ => Ok(()),
    }
}

/// Hash the first `len` bytes of `file` into `hasher`, reporting progress.
fn hash_into(
    file: &mut File,
    path: &Path,
    len: u64,
    hasher: &mut Sha256,
    reporter: &mut Reporter<'_>,
    cancel: &AtomicBool,
) -> Result<()> {
    let mut buf = vec![0u8; CHUNK];
    let mut done = 0u64;
    reporter.report(Stage::Hashing, 0, true);
    while done < len {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let want = (len - done).min(CHUNK as u64) as usize;
        let n = file.read(&mut buf[..want]).map_err(|e| Error::io(path, e))?;
        if n == 0 {
            return Err(Error::io(path, std::io::ErrorKind::UnexpectedEof.into()));
        }
        hasher.update(&buf[..n]);
        done += n as u64;
        reporter.report(Stage::Hashing, done, false);
    }
    reporter.report(Stage::Hashing, done, true);
    Ok(())
}

/// A file already at `dest`: check it rather than download over it.
fn verify_existing(
    dest: &Path,
    expected_size: Option<u64>,
    expected_sha: Option<&str>,
    reporter: &mut Reporter<'_>,
    cancel: &AtomicBool,
) -> Result<u64> {
    let len = std::fs::metadata(dest).map_err(|e| Error::io(dest, e))?.len();
    if let Some(size) = expected_size.filter(|size| *size != len) {
        return Err(Error::Integrity {
            path: dest.to_path_buf(),
            detail: format!("already exists with {len} bytes, expected {size}; move it away to download again"),
        });
    }
    if let Some(want) = expected_sha {
        let mut file = File::open(dest).map_err(|e| Error::io(dest, e))?;
        let mut hasher = Sha256::new();
        reporter.total = Some(len);
        hash_into(&mut file, dest, len, &mut hasher, reporter, cancel)?;
        let got = hex(&hasher.finalize());
        if got != want {
            return Err(Error::Integrity {
                path: dest.to_path_buf(),
                detail: format!("already exists with SHA-256 {got}, expected {want}; move it away to download again"),
            });
        }
    }
    Ok(len)
}

/// Rename the verified `.part` over `dest`. An antivirus scanner that
/// opened the fresh file can hold it for a moment: retry briefly.
fn rename_into_place(part: &Path, dest: &Path) -> Result<()> {
    let mut last = None;
    for i in 0..5 {
        match std::fs::rename(part, dest) {
            Ok(()) => return Ok(()),
            Err(e) => last = Some(e),
        }
        std::thread::sleep(Duration::from_millis(200 * (i + 1)));
    }
    Err(Error::io(dest, last.unwrap_or_else(|| std::io::ErrorKind::Other.into())))
}

/// Sleep for `d` in short steps; false when `cancel` was raised.
fn sleep_unless_cancelled(d: Duration, cancel: &AtomicBool) -> bool {
    let until = Instant::now() + d;
    while Instant::now() < until {
        if cancel.load(Ordering::Relaxed) {
            return false;
        }
        std::thread::sleep((until - Instant::now()).min(Duration::from_millis(100)));
    }
    !cancel.load(Ordering::Relaxed)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Throttled progress with a smoothed transfer rate.
struct Reporter<'a> {
    file: String,
    total: Option<u64>,
    sink: &'a mut dyn FnMut(&Progress),
    last_at: Option<Instant>,
    last_done: u64,
    last_stage: Option<Stage>,
    bps: f64,
}

impl<'a> Reporter<'a> {
    fn new(dest: &Path, total: Option<u64>, sink: &'a mut dyn FnMut(&Progress)) -> Self {
        let file = dest.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        Reporter { file, total, sink, last_at: None, last_done: 0, last_stage: None, bps: 0.0 }
    }

    fn report(&mut self, stage: Stage, done: u64, force: bool) {
        let now = Instant::now();
        let stage_changed = self.last_stage != Some(stage);
        if stage_changed {
            self.last_at = Some(now);
            self.last_done = done;
            self.bps = 0.0;
        } else if !force && self.last_at.is_some_and(|t| now - t < REPORT_EVERY) {
            return;
        }
        if stage == Stage::Downloading && !stage_changed {
            if let Some(t) = self.last_at {
                let dt = (now - t).as_secs_f64();
                if dt > 0.0 && done >= self.last_done {
                    let rate = (done - self.last_done) as f64 / dt;
                    self.bps = if self.bps == 0.0 { rate } else { 0.7 * self.bps + 0.3 * rate };
                }
            }
            self.last_at = Some(now);
            self.last_done = done;
        } else if !stage_changed {
            self.last_at = Some(now);
        }
        self.last_stage = Some(stage);
        let bps = if stage == Stage::Downloading { self.bps } else { 0.0 };
        (self.sink)(&Progress { file: self.file.clone(), stage, done, total: self.total, bps });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_http::{Req, Resp, Server};
    use std::sync::{Arc, Mutex};

    fn sha(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    /// Deterministic bytes, varied enough that an offset error changes the hash.
    fn payload(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect()
    }

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fidim-fetch-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Serves `data` honouring `Range: bytes=N-`, like the Hub's CDN.
    fn serve_ranges(data: Arc<Vec<u8>>, req: &Req) -> Resp {
        match req.header("range").and_then(|r| r.strip_prefix("bytes=")).and_then(|r| r.strip_suffix('-')) {
            Some(from) => {
                let from: usize = from.parse().unwrap();
                if from >= data.len() {
                    return Resp::new(416, "").header("Content-Range", format!("bytes */{}", data.len()));
                }
                Resp::new(206, data[from..].to_vec())
                    .header("Content-Range", format!("bytes {from}-{}/{}", data.len() - 1, data.len()))
            }
            None => Resp::new(200, data.to_vec()),
        }
    }

    fn req_for(url: String, data: &[u8]) -> FetchReq {
        FetchReq {
            expected_size: Some(data.len() as u64),
            expected_sha256: Some(sha(data)),
            backoff: Duration::from_millis(5),
            ..FetchReq::new(url)
        }
    }

    fn quiet() -> impl FnMut(&Progress) {
        |_| {}
    }

    #[test]
    fn downloads_verifies_and_renames() {
        let data = Arc::new(payload(3 * CHUNK + 123));
        let d = data.clone();
        let srv = Server::start("127.0.0.1", move |r| serve_ranges(d.clone(), r)).unwrap();
        let dir = tmp("fresh");
        let dest = dir.join("sub").join("model-Q4_K_M.gguf");
        let mut seen: Vec<Progress> = Vec::new();
        let n = download_resumable(
            &req_for(format!("{}/f", srv.base), &data),
            &dest,
            &mut |p| seen.push(p.clone()),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(n, data.len() as u64);
        assert_eq!(std::fs::read(&dest).unwrap(), *data);
        assert!(!part_path(&dest).exists());
        let last = seen.last().unwrap();
        assert_eq!((last.stage, last.done, last.total), (Stage::Downloading, n, Some(n)));
        assert_eq!(last.file, "model-Q4_K_M.gguf");
        assert!(seen.iter().all(|p| p.stage == Stage::Downloading), "nothing to re-hash");
        assert_eq!(srv.requests().len(), 1);
        assert_eq!(srv.requests()[0].header("range"), None);

        // Asked again: the file is verified, not downloaded.
        let again = download_resumable(
            &req_for(format!("{}/f", srv.base), &data),
            &dest,
            &mut quiet(),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(again, n);
        assert_eq!(srv.requests().len(), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn resumes_a_partial_file() {
        let data = Arc::new(payload(2 * CHUNK + 777));
        let d = data.clone();
        let srv = Server::start("127.0.0.1", move |r| serve_ranges(d.clone(), r)).unwrap();
        let dir = tmp("resume");
        let dest = dir.join("m.gguf");
        let have = CHUNK + 5;
        std::fs::write(part_path(&dest), &data[..have]).unwrap();
        let mut stages = Vec::new();
        download_resumable(
            &req_for(format!("{}/f", srv.base), &data),
            &dest,
            &mut |p| stages.push((p.stage, p.done)),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), *data, "resumed bytes appended at the right offset");
        assert_eq!(srv.requests()[0].header("range"), Some(format!("bytes={have}-").as_str()));
        assert!(stages.contains(&(Stage::Hashing, have as u64)), "the existing part was re-hashed");
        std::fs::remove_dir_all(dir).ok();
    }

    /// A server that ignores Range sends the whole file: start over.
    #[test]
    fn restarts_when_range_is_ignored() {
        let data = Arc::new(payload(CHUNK + 999));
        let d = data.clone();
        let srv = Server::start("127.0.0.1", move |_| Resp::new(200, d.to_vec())).unwrap();
        let dir = tmp("norange");
        let dest = dir.join("m.gguf");
        // Garbage in the .part: kept, it would corrupt the result.
        std::fs::write(part_path(&dest), vec![0xAAu8; 5000]).unwrap();
        download_resumable(&req_for(format!("{}/f", srv.base), &data), &dest, &mut quiet(), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), *data);
        assert_eq!(srv.requests()[0].header("range"), Some("bytes=5000-"));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn hash_mismatch_errors_and_deletes_nothing() {
        let data = Arc::new(payload(10_000));
        let d = data.clone();
        let srv = Server::start("127.0.0.1", move |r| serve_ranges(d.clone(), r)).unwrap();
        let dir = tmp("badsha");
        let dest = dir.join("m.gguf");
        let mut req = req_for(format!("{}/f", srv.base), &data);
        req.expected_sha256 = Some("0".repeat(64));
        match download_resumable(&req, &dest, &mut quiet(), &AtomicBool::new(false)) {
            Err(Error::Integrity { path, detail }) => {
                assert_eq!(path, part_path(&dest));
                assert!(detail.contains(&sha(&data)) && detail.contains("delete this file"), "{detail}");
            }
            other => panic!("{other:?}"),
        }
        assert!(!dest.exists());
        assert_eq!(std::fs::read(part_path(&dest)).unwrap(), *data, "the .part is left alone");
        // A wrong hash on a finished file is refused the same way.
        std::fs::write(&dest, &*data).unwrap();
        assert!(matches!(
            download_resumable(&req, &dest, &mut quiet(), &AtomicBool::new(false)),
            Err(Error::Integrity { .. })
        ));
        assert!(dest.exists());
        let bad = FetchReq { expected_sha256: Some("xyz".into()), ..req };
        assert!(matches!(
            download_resumable(&bad, &dest, &mut quiet(), &AtomicBool::new(false)),
            Err(Error::InvalidInput(_))
        ));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn cancel_keeps_the_part_for_a_resume() {
        let data = Arc::new(payload(6 * CHUNK));
        let d = data.clone();
        let srv = Server::start("127.0.0.1", move |r| serve_ranges(d.clone(), r)).unwrap();
        let dir = tmp("cancel");
        let dest = dir.join("m.gguf");
        let cancel = AtomicBool::new(false);
        let req = req_for(format!("{}/f", srv.base), &data);
        let r = download_resumable(
            &req,
            &dest,
            &mut |p| {
                if p.done > 0 {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
            &cancel,
        );
        assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
        assert!(!dest.exists());
        let kept = std::fs::metadata(part_path(&dest)).unwrap().len();
        assert!(kept > 0 && kept < data.len() as u64, "{kept}");
        assert_eq!(std::fs::read(part_path(&dest)).unwrap(), data[..kept as usize]);

        download_resumable(&req, &dest, &mut quiet(), &AtomicBool::new(false)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), *data);
        assert_eq!(srv.requests().last().unwrap().header("range"), Some(format!("bytes={kept}-").as_str()));
        std::fs::remove_dir_all(dir).ok();
    }

    /// Each attempt goes back to the stable URL, whose redirect hands out a
    /// fresh signed URL; the old one has expired. The token reaches the
    /// first host only.
    #[test]
    fn every_attempt_re_resolves_the_redirect() {
        let data = Arc::new(payload(3 * CHUNK));
        let signed = Arc::new(Mutex::new(0u32));
        let (d, s) = (data.clone(), signed.clone());
        let Ok(cdn) = Server::start("127.0.0.2", move |req| {
            let current = *s.lock().unwrap();
            let asked: u32 = req.target.split("sig=").nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
            if asked != current {
                return Resp::new(403, "Request has expired");
            }
            let resp = serve_ranges(d.clone(), req);
            // The first signed URL's transfer drops half way.
            if current == 1 { resp.cut_after(CHUNK + 10) } else { resp }
        }) else {
            eprintln!("127.0.0.2 cannot be bound here; skipped");
            return;
        };
        let (cdn_base, s) = (cdn.base.clone(), signed.clone());
        let hub = Server::start("127.0.0.1", move |_| {
            let mut n = s.lock().unwrap();
            *n += 1;
            Resp::new(302, "").header("Location", format!("{cdn_base}/blob?sig={n}"))
        })
        .unwrap();
        let dir = tmp("redirect");
        let dest = dir.join("m.gguf");
        let mut req = req_for(format!("{}/o/r/resolve/abc/m.gguf", hub.base), &data);
        req.token = Some("hf_secret".into());
        download_resumable(&req, &dest, &mut quiet(), &AtomicBool::new(false)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), *data);

        let at_hub = hub.requests();
        assert_eq!(at_hub.len(), 2, "one resolve per attempt");
        assert!(at_hub.iter().all(|r| r.header("authorization") == Some("Bearer hf_secret")));
        assert_eq!(at_hub[1].header("range"), Some(format!("bytes={}-", CHUNK + 10).as_str()));
        let at_cdn = cdn.requests();
        assert_eq!(at_cdn.iter().map(|r| r.target.as_str()).collect::<Vec<_>>(), ["/blob?sig=1", "/blob?sig=2"]);
        assert!(at_cdn.iter().all(|r| r.header("authorization").is_none()), "token leaked to the CDN");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn retries_transient_failures_but_not_gated_repos() {
        let data = Arc::new(payload(50_000));
        let calls = Arc::new(Mutex::new(0u32));
        let (d, c) = (data.clone(), calls.clone());
        let srv = Server::start("127.0.0.1", move |req| {
            let mut n = c.lock().unwrap();
            *n += 1;
            match *n {
                1 => Resp::new(503, "busy"),
                2 => Resp::new(429, "slow down").header("Retry-After", "0"),
                3 => serve_ranges(d.clone(), req).cut_after(20_000),
                _ => serve_ranges(d.clone(), req),
            }
        })
        .unwrap();
        let dir = tmp("retry");
        let dest = dir.join("m.gguf");
        let mut req = req_for(format!("{}/f", srv.base), &data);
        req.attempts = 3; // three failures in a row, but the cut one made progress
        download_resumable(&req, &dest, &mut quiet(), &AtomicBool::new(false)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), *data);
        assert_eq!(*calls.lock().unwrap(), 4);

        // Out of attempts: the last error comes back and the .part stays.
        let srv = Server::start("127.0.0.1", |_| Resp::new(500, "down")).unwrap();
        let dest2 = dir.join("m2.gguf");
        match download_resumable(&req_for(format!("{}/f", srv.base), &data), &dest2, &mut quiet(), &AtomicBool::new(false)) {
            Err(Error::Http { kind: HttpErrorKind::Status { status: 500 }, .. }) => {}
            other => panic!("{other:?}"),
        }
        assert_eq!(srv.requests().len(), 5);

        let srv = Server::start("127.0.0.1", |_| {
            Resp::new(401, "Access to model g/gated is restricted.").header("X-Error-Code", "GatedRepo")
        })
        .unwrap();
        let url = format!("{}/g/gated/resolve/abc/m.gguf", srv.base);
        match download_resumable(&req_for(url, &data), &dir.join("m3.gguf"), &mut quiet(), &AtomicBool::new(false)) {
            Err(Error::Http { kind: HttpErrorKind::Gated, message }) => {
                assert!(message.contains("https://huggingface.co/g/gated"), "{message}")
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(srv.requests().len(), 1, "a gated repo is not retried");
        std::fs::remove_dir_all(dir).ok();
    }

    /// A 403 from the URL's own host is a refusal; only one from the host a
    /// redirect led to (an expired signed URL) is retried.
    #[test]
    fn a_403_from_the_origin_is_not_retried() {
        let data = payload(100);
        let srv = Server::start("127.0.0.1", |_| Resp::new(403, "Forbidden")).unwrap();
        let dir = tmp("forbidden");
        match download_resumable(&req_for(format!("{}/f", srv.base), &data), &dir.join("m.gguf"), &mut quiet(), &AtomicBool::new(false)) {
            Err(Error::Http { kind: HttpErrorKind::Status { status: 403 }, .. }) => {}
            other => panic!("{other:?}"),
        }
        assert_eq!(srv.requests().len(), 1);
        assert_eq!(host_of("https://us.aws.cdn.hf.co/x?y=1"), "us.aws.cdn.hf.co");
        assert_eq!(host_of("http://127.0.0.1:8080/o/r"), "127.0.0.1:8080");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn complete_part_and_size_changes() {
        let data = Arc::new(payload(4096));
        let d = data.clone();
        let srv = Server::start("127.0.0.1", move |r| serve_ranges(d.clone(), r)).unwrap();
        let dir = tmp("sizes");
        // A .part that is already whole: verified and renamed, no request.
        let dest = dir.join("whole.gguf");
        std::fs::write(part_path(&dest), &*data).unwrap();
        download_resumable(&req_for(format!("{}/f", srv.base), &data), &dest, &mut quiet(), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), *data);
        assert_eq!(srv.requests().len(), 0);

        // Without an expected size the server's 416 says the part is whole.
        let dest = dir.join("nosize.gguf");
        std::fs::write(part_path(&dest), &*data).unwrap();
        let req = FetchReq { expected_size: None, ..req_for(format!("{}/f", srv.base), &data) };
        download_resumable(&req, &dest, &mut quiet(), &AtomicBool::new(false)).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), *data);

        // The server's file is not the size the listing promised.
        let dest = dir.join("changed.gguf");
        let req = FetchReq { expected_size: Some(5000), expected_sha256: None, ..FetchReq::new(format!("{}/f", srv.base)) };
        assert!(matches!(
            download_resumable(&req, &dest, &mut quiet(), &AtomicBool::new(false)),
            Err(Error::Integrity { .. })
        ));
        std::fs::remove_dir_all(dir).ok();
    }

    /// Two downloads of one file do not interleave: the second cannot open
    /// the `.part` while the first holds it.
    #[cfg(windows)]
    #[test]
    fn one_writer_per_part() {
        let dir = tmp("writers");
        let dest = dir.join("m.gguf");
        let held = Part::open(part_path(&dest)).unwrap();
        let req = FetchReq { backoff: Duration::from_millis(1), ..FetchReq::new("http://127.0.0.1:9/never") };
        match download_resumable(&req, &dest, &mut quiet(), &AtomicBool::new(false)) {
            Err(Error::Io { path, .. }) => assert_eq!(path, part_path(&dest)),
            other => panic!("{other:?}"),
        }
        // Readers are let in.
        assert!(File::open(part_path(&dest)).is_ok());
        drop(held);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn part_path_appends_to_the_name() {
        assert_eq!(part_path(Path::new(r"E:\m\o\r\x.gguf")), PathBuf::from(r"E:\m\o\r\x.gguf.part"));
    }
}
