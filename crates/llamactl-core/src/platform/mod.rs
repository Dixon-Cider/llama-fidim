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

/// Dedicated GPU memory one process holds on one adapter, from the PDH
/// "GPU Process Memory" counters. This is the residency ground truth on
/// Windows: WDDM virtualizes VRAM, so `--list-devices` free-memory deltas
/// do NOT see other processes' allocations (verified on the target machine).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct GpuProcessMem {
    pub luid_low: u64,
    /// Resident in dedicated VRAM. On a layer-split launch the non-main
    /// card stays near zero until the first forward pass touches its layers
    /// (WDDM residency is on-demand) — verified on the target machine.
    pub dedicated_bytes: u64,
    /// Committed against the adapter — proves placement immediately, before
    /// first inference makes it resident.
    pub committed_bytes: u64,
}

pub trait Platform {
    /// Video adapters with PNP identity, bus number, LUID, and display.
    fn video_adapters(&self) -> Result<Vec<OsAdapter>>;
    /// Current commit limit and charge.
    fn system_commit(&self) -> Result<SystemCommit>;
    /// Per-adapter dedicated GPU memory held by `pid`.
    fn gpu_process_memory(&self, pid: u32) -> Result<Vec<GpuProcessMem>>;
}

/// Deterministic fake for tests: scripted responses, no OS access.
#[derive(Default)]
pub struct FakePlatform {
    pub adapters: Vec<OsAdapter>,
    pub commit: Option<SystemCommit>,
    pub gpu_mem: Vec<GpuProcessMem>,
}

impl Platform for FakePlatform {
    fn video_adapters(&self) -> Result<Vec<OsAdapter>> {
        Ok(self.adapters.clone())
    }
    fn system_commit(&self) -> Result<SystemCommit> {
        Ok(self.commit.unwrap_or(SystemCommit {
            limit_bytes: 100 * 1024 * 1024 * 1024,
            charge_bytes: 10 * 1024 * 1024 * 1024,
        }))
    }
    fn gpu_process_memory(&self, _pid: u32) -> Result<Vec<GpuProcessMem>> {
        Ok(self.gpu_mem.clone())
    }
}
