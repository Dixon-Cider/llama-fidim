//! OS queries behind a trait so check logic is testable with fakes and a
//! non-Windows port is additive (spec §07: the Windows checks are the point).

use serde::Serialize;

use crate::devices::OsAdapter;
use crate::Result;

#[cfg(windows)]
mod windows_impl;
#[cfg(windows)]
pub use windows_impl::WindowsPlatform;

/// System commit state (R-05): a GPU allocation of N bytes consumes ~N bytes
/// of commit; exhausting commit silently evicts models from VRAM.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SystemCommit {
    pub limit_bytes: u64,
    pub charge_bytes: u64,
}

impl SystemCommit {
    pub fn charge_fraction(&self) -> f64 {
        if self.limit_bytes == 0 {
            return 1.0;
        }
        self.charge_bytes as f64 / self.limit_bytes as f64
    }
}

pub trait Platform {
    /// Video adapters with PNP identity, bus number, and attached display.
    fn video_adapters(&self) -> Result<Vec<OsAdapter>>;
    /// Current commit limit and charge.
    fn system_commit(&self) -> Result<SystemCommit>;
}

/// Deterministic fake for tests: scripted responses, no OS access.
pub struct FakePlatform {
    pub adapters: Vec<OsAdapter>,
    pub commit: SystemCommit,
}

impl Platform for FakePlatform {
    fn video_adapters(&self) -> Result<Vec<OsAdapter>> {
        Ok(self.adapters.clone())
    }
    fn system_commit(&self) -> Result<SystemCommit> {
        Ok(self.commit)
    }
}
