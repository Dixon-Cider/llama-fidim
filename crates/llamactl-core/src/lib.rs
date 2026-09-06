//! llamactl-core — all logic for discovering llama.cpp builds/models, enumerating
//! GPUs by stable key, validating launch configurations, and supervising servers.
//!
//! The UI layers (CLI now, Tauri later) are thin wrappers over this crate.

pub mod bench;
pub mod config;
pub mod devices;
pub mod discovery;
pub mod error;
pub mod estimate;
pub mod export;
pub mod gguf;
pub mod hf;
pub mod launch;
pub mod live;
pub mod platform;
pub mod preflight;
pub mod profile;
pub mod rocm;
pub mod router;
pub mod runtime;
pub mod supervise;
pub mod update;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;
