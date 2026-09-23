use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("not a GGUF file (bad magic) at {0}")]
    GgufBadMagic(PathBuf),

    #[error("unsupported GGUF version {version} at {path}")]
    GgufVersion { path: PathBuf, version: u32 },

    #[error("malformed GGUF metadata at {path}: {detail}")]
    GgufMalformed { path: PathBuf, detail: String },

    /// The bytes ran out before the header did. On a local file that is a
    /// damaged or partial file; on a Range-read prefix it means "fetch at
    /// least `at` bytes and parse again".
    #[error("GGUF header at {path} ends early: the parser needed the first {at} bytes")]
    GgufTruncated { path: PathBuf, at: u64 },

    #[error("could not parse `--list-devices` output: {0}")]
    ListDevicesUnparseable(String),

    #[error("device key {key} did not resolve to any current device")]
    DeviceKeyUnresolved { key: String },

    #[error("device key {key} is ambiguous: {candidates} identical candidates")]
    DeviceKeyAmbiguous { key: String, candidates: usize },

    #[error("platform query failed: {0}")]
    Platform(String),

    #[error("build binary failed to run: {path}: {detail}")]
    BuildBinary { path: PathBuf, detail: String },

    #[error("config error: {0}")]
    Config(String),

    #[error("update: {0}")]
    Update(String),

    /// A remote service (the Hugging Face Hub, a CDN) refused or failed a
    /// request. `kind` is what a caller branches on; `message` is written
    /// for the user and already says what to do.
    #[error("{message}")]
    Http { kind: HttpErrorKind, message: String },

    /// A download finished but is not the file that was asked for.
    #[error("{path}: {detail}")]
    Integrity { path: PathBuf, detail: String },

    /// The caller's cancel flag was raised. Partial work is kept for a resume.
    #[error("cancelled")]
    Cancelled,

    /// An argument that cannot name anything real (a malformed repo id).
    #[error("{0}")]
    InvalidInput(String),

    #[error("user PATH: {0}")]
    UserPath(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Why a remote request failed, from the status code and the Hub's
/// `X-Error-Code` header (the status alone is ambiguous: the Hub answers 401
/// both for a gated repo and for one that does not exist).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HttpErrorKind {
    /// The repo is gated: its terms must be accepted on the website and a
    /// token sent (401 without one, 403 with one that has no access yet).
    Gated,
    /// The token was rejected, or the repo is private or does not exist
    /// (the Hub does not say which).
    RepoNotFound,
    RevisionNotFound,
    /// The file is not in the repo at that revision.
    EntryNotFound,
    RateLimited { retry_after_secs: Option<u64> },
    /// Any other HTTP status.
    Status { status: u16 },
    /// DNS, connect, TLS, reset, timeout.
    Network,
    /// A success status with a body that is not what the endpoint returns.
    Malformed,
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io { path: path.into(), source }
    }
}
