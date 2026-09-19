//! fidim-core — all logic for discovering llama.cpp builds/models, enumerating
//! GPUs by stable key, validating launch configurations, and supervising servers.
//!
//! The UI layers (CLI now, Tauri later) are thin wrappers over this crate.

pub mod bench;
pub mod build_info;
pub mod catalog;
pub mod chat;
pub mod compat;
pub mod config;
pub mod devices;
pub mod diffusion;
pub mod discovery;
pub mod disk;
pub mod error;
pub mod estimate;
pub mod export;
pub mod fetch;
pub mod gguf;
pub mod hf;
pub mod hub;
pub mod launch;
pub mod live;
pub mod overlay;
pub mod platform;
pub mod preflight;
pub mod profile;
pub mod rocm;
pub mod router;
pub mod runtime;
pub mod supervise;
pub mod toolchain;
pub mod update;
pub mod user_path;

#[cfg(test)]
mod test_http;

pub use error::{Error, HttpErrorKind};
pub type Result<T> = std::result::Result<T, Error>;
