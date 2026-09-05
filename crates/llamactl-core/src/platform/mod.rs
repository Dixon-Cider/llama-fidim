//! OS queries behind a trait so check logic is testable with fakes and a
//! non-Windows port is additive (spec §07: the Windows checks are the point).

use serde::Serialize;

use crate::devices::OsAdapter;
use crate::Result;

#[cfg(windows)]
mod windows_impl;
#[cfg(windows)]
pub use windows_impl::WindowsPlatform;

/// System memory state (R-05).
///
/// **Commit is the ceiling that matters**, and `limit_bytes` = physical RAM +
/// allocated pagefile. WDDM makes every GPU allocation commit-backed, because
/// the driver reserves the right to evict VRAM to system memory. Critically,
/// that charge is a *reservation, not a transfer*: measured on the target
/// machine 2026-08-01, loading a 20 GB model charged +19.6 GiB of commit,
/// cost only ~8 GiB of physical RAM (mostly evictable file cache from
/// reading the GGUF), and left the pagefile holding 1.5 GiB of its 84 GiB.
///
/// That measurement is why there is NO physical-RAM gate here. A model far
/// larger than RAM loads fine as long as it fits in VRAM and commit can
/// promise it — which is exactly how other loaders behave on this box. An
/// earlier revision of this file gated on RAM 1:1 and would have blocked
/// working configurations; it was wrong and was removed.
///
/// The real hazard near the limit is `pagefile_allocated_bytes`: when
/// projected commit approaches the limit, Windows must EXTEND the pagefile,
/// a synchronous disk operation that can stall the whole machine for
/// minutes and leaves no trace afterwards but a larger limit.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SystemCommit {
    pub limit_bytes: u64,
    pub charge_bytes: u64,
    /// Informational only — surfaced in views, never a launch gate.
    pub physical_total_bytes: u64,
    /// Informational only — surfaced in views, never a launch gate.
    pub physical_available_bytes: u64,
    /// Currently allocated pagefile size (sum across pagefiles).
    pub pagefile_allocated_bytes: u64,
    /// True when at least one pagefile is system-managed and may grow.
    /// A fixed-size pagefile cannot stall on growth, but also cannot
    /// rescue an over-limit allocation — it fails outright instead.
    pub pagefile_can_grow: bool,
}

impl SystemCommit {
    pub fn charge_fraction(&self) -> f64 {
        if self.limit_bytes == 0 {
            return 1.0;
        }
        self.charge_bytes as f64 / self.limit_bytes as f64
    }

    /// Physical RAM in use right now (informational).
    pub fn physical_used_bytes(&self) -> u64 {
        self.physical_total_bytes.saturating_sub(self.physical_available_bytes)
    }
}

/// Dedicated GPU memory one process holds on one adapter, from the PDH
/// "GPU Process Memory" counters. This is the residency ground truth on
/// Windows: WDDM virtualizes VRAM, so `--list-devices` free-memory deltas
/// do NOT see other processes' allocations (verified on the target machine).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct GpuEngineUtil {
    pub pid: u32,
    pub luid_low: u64,
    pub percent: f64,
}

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
    /// GPU engine busy percentage per (pid, adapter), sampled over a short
    /// window. Sum over a pid's instances on one adapter; Windows reports
    /// compute work under several engine types.
    fn gpu_utilization(&self) -> Result<Vec<GpuEngineUtil>> {
        Ok(Vec::new())
    }
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
            physical_total_bytes: 64 * 1024 * 1024 * 1024,
            physical_available_bytes: 48 * 1024 * 1024 * 1024,
            pagefile_allocated_bytes: 36 * 1024 * 1024 * 1024,
            pagefile_can_grow: true,
        }))
    }
    fn gpu_process_memory(&self, _pid: u32) -> Result<Vec<GpuProcessMem>> {
        Ok(self.gpu_mem.clone())
    }
}
