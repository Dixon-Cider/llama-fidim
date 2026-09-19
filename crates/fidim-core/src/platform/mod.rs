//! OS queries behind a trait so check logic is testable with fakes and a
//! non-Windows port is additive (spec §07: the Windows checks are the point).

use serde::Serialize;

use crate::devices::OsAdapter;
use crate::Result;

#[cfg(windows)]
mod windows_impl;
#[cfg(windows)]
pub use windows_impl::{process_descendants, WindowsPlatform};

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

/// A run's process and everything below it. What a run holds on the GPU
/// is the sum over these: a diffusion run's helper holds nothing itself
/// (its runner child does), and a router's model instances are its children.
pub fn run_pids(root: u32) -> Vec<u32> {
    let mut pids = vec![root];
    #[cfg(windows)]
    pids.extend(process_descendants(root));
    pids
}

/// Every process below `root`, from (pid, parent pid) pairs and a creation
/// time lookup. Windows reuses pids and never rewrites a child's parent pid:
/// once the CLI that launched a server exits, a later process can get its
/// pid and would "adopt" that server, whose VRAM would then be counted
/// against the wrong run. A child created before its parent cannot be its
/// child, so it is skipped; a time that cannot be read is given the benefit
/// of the doubt.
pub fn descendants_in(root: u32, pairs: &[(u32, u32)], created: &dyn Fn(u32) -> Option<u64>) -> Vec<u32> {
    let mut times: std::collections::HashMap<u32, Option<u64>> = std::collections::HashMap::new();
    let mut time = |pid: u32| *times.entry(pid).or_insert_with(|| created(pid));
    let mut out = Vec::new();
    let mut frontier = vec![root];
    while let Some(p) = frontier.pop() {
        for &(pid, parent) in pairs {
            if parent != p || pid == p || pid == root || out.contains(&pid) {
                continue;
            }
            if let (Some(child), Some(parent)) = (time(pid), time(p)) {
                if child < parent {
                    continue;
                }
            }
            out.push(pid);
            frontier.push(pid);
        }
    }
    out
}

/// Raise `flag` on the first Ctrl+C (or Ctrl+Break) in this console instead
/// of ending the process, so a long job (a download, a build) stops at its
/// next check and keeps what it has; a second press ends the process as
/// usual. False when the handler could not be installed (no console).
pub fn cancel_on_ctrl_c(flag: &'static std::sync::atomic::AtomicBool) -> bool {
    #[cfg(windows)]
    {
        windows_impl::cancel_on_ctrl_c(flag)
    }
    #[cfg(not(windows))]
    {
        let _ = flag;
        false
    }
}

/// Per-adapter GPU memory summed over `pids`, one entry per LUID. An error
/// for the first pid (the run's own) means the counters are unavailable and
/// is returned; a descendant that exits mid-query is skipped.
pub fn sum_gpu_memory(p: &dyn Platform, pids: &[u32]) -> Result<Vec<GpuProcessMem>> {
    let mut out: Vec<GpuProcessMem> = Vec::new();
    for (i, pid) in pids.iter().enumerate() {
        let mem = match p.gpu_process_memory(*pid) {
            Ok(m) => m,
            Err(e) if i == 0 => return Err(e),
            Err(_) => continue,
        };
        for m in mem {
            match out.iter_mut().find(|o| o.luid_low == m.luid_low) {
                Some(o) => {
                    o.dedicated_bytes += m.dedicated_bytes;
                    o.committed_bytes += m.committed_bytes;
                }
                None => out.push(m),
            }
        }
    }
    Ok(out)
}

/// Deterministic fake for tests: scripted responses, no OS access.
#[derive(Default)]
pub struct FakePlatform {
    pub adapters: Vec<OsAdapter>,
    pub commit: Option<SystemCommit>,
    pub gpu_mem: Vec<GpuProcessMem>,
    /// Per-pid answers; a pid not listed gets `gpu_mem`.
    pub gpu_mem_by_pid: std::collections::HashMap<u32, Vec<GpuProcessMem>>,
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
    fn gpu_process_memory(&self, pid: u32) -> Result<Vec<GpuProcessMem>> {
        Ok(self.gpu_mem_by_pid.get(&pid).cloned().unwrap_or_else(|| self.gpu_mem.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(luid_low: u64, dedicated_bytes: u64, committed_bytes: u64) -> GpuProcessMem {
        GpuProcessMem { luid_low, dedicated_bytes, committed_bytes }
    }

    #[test]
    fn sum_gpu_memory() {
        let mut p = FakePlatform::default();
        // The helper holds a sliver; its runner child holds the model.
        p.gpu_mem_by_pid.insert(10, vec![mem(0x1BAAD, 1 << 20, 2 << 20)]);
        p.gpu_mem_by_pid.insert(11, vec![mem(0x1BAAD, 16 << 30, 17 << 30), mem(0x14427, 3 << 20, 4 << 20)]);
        let sum = super::sum_gpu_memory(&p, &[10, 11]).unwrap();
        assert_eq!(sum.len(), 2);
        let card = sum.iter().find(|m| m.luid_low == 0x1BAAD).unwrap();
        assert_eq!(card.dedicated_bytes, (16 << 30) + (1 << 20));
        assert_eq!(card.committed_bytes, (17 << 30) + (2 << 20));
        let other = sum.iter().find(|m| m.luid_low == 0x14427).unwrap();
        assert_eq!((other.dedicated_bytes, other.committed_bytes), (3 << 20, 4 << 20));
        // The root is always first.
        assert_eq!(run_pids(std::process::id())[0], std::process::id());
    }

    #[test]
    fn descendants_skip_children_of_a_reused_pid() {
        // CLI pid 100 launched server 200 at t=10 and exited; pid 100 was then
        // reused by a later server (t=50), which started 300, which started 400.
        let pairs = [(0, 0), (1, 0), (200, 100), (100, 1), (300, 100), (400, 300)];
        let t = |pid: u32| match pid {
            200 => Some(10),
            100 => Some(50),
            300 => Some(60),
            400 => Some(70),
            _ => None,
        };
        assert_eq!(descendants_in(100, &pairs, &t), vec![300, 400]);
        // A time that cannot be read is trusted.
        assert_eq!(descendants_in(100, &pairs, &|_| None), vec![200, 300, 400]);
        // pid 0 names itself as its parent; no loops.
        assert_eq!(descendants_in(0, &pairs, &|_| None), vec![1, 100, 200, 300, 400]);
    }

    /// The creation-time filter must keep a real child: it starts after us.
    #[cfg(windows)]
    #[test]
    fn a_real_child_is_a_descendant() {
        let mut cmd = std::process::Command::new("cmd.exe");
        cmd.args(["/d", "/c", "pause"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        crate::launch::hide_console(&mut cmd);
        let mut child = cmd.spawn().unwrap();
        let found = process_descendants(std::process::id()).contains(&child.id());
        let _ = child.kill();
        let _ = child.wait();
        assert!(found);
    }
}
