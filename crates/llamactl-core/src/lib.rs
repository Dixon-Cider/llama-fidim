//! llamactl-core — all logic for discovering llama.cpp builds/models, enumerating
//! GPUs by stable key, validating launch configurations, and supervising servers.
//!
//! The UI layers (CLI now, Tauri later) are thin wrappers over this crate.

pub mod config;
pub mod devices;
pub mod discovery;
pub mod error;
pub mod estimate;
pub mod gguf;
pub mod platform;
pub mod preflight;
pub mod profile;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;
