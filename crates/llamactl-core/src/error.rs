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

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io { path: path.into(), source }
    }
}
