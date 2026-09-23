//! Linux implementation of [`Platform`]: sysfs for adapters, procfs for
//! commit and per-process GPU memory / busy. The Windows build reads WMI,
//! DEVPROPKEYs and PDH; here the same facts come from
//! `/sys/class/drm/card<N>/device` (vendor, device, PCI slot, busy, VRAM),
//! `/sys/class/drm/card<N>-<connector>/status` (a display attached) and
//! `/proc/<pid>/fdinfo/*` (the amdgpu `drm-*` client lines: VRAM held and
//! engine time per process and per card).
//!
//! Adapter identity: `luid_low` is the PCI bus number, which is what
//! fdinfo's `drm-pdev` (`0000:03:00.0`) reports too, so per-process memory
//! and busy link to the card the same way the LUID does on Windows.
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use super::{GpuEngineUtil, GpuProcessMem, Platform, SystemCommit};
use crate::devices::{DisplayMode, OsAdapter};
use crate::{Error, Result};

pub struct LinuxPlatform;

fn read_trim(p: &Path) -> Option<String> {
    fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

fn read_u64(p: &Path) -> Option<u64> {
    read_trim(p).and_then(|s| s.parse().ok())
}

fn read_hex(p: &Path) -> Option<u32> {
    read_trim(p).and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
}

/// `0000:03:00.0` -> bus 3.
pub(crate) fn pci_bus(slot: &str) -> Option<u32> {
    let mut it = slot.split(':');
    let _domain = it.next()?;
    u32::from_str_radix(it.next()?, 16).ok()
}

/// Every `/sys/class/drm/cardN` (not the `cardN-<connector>` entries) that
/// is a PCI device, with its device directory.
fn drm_cards() -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir("/sys/class/drm") else { return out };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let dev = e.path().join("device");
        if dev.join("vendor").is_file() {
            out.push((name, dev));
        }
    }
    out.sort();
    out
}

/// Marketing name for a PCI device id from hwdata's pci.ids when present,
/// else a vendor/device fallback. AMD's sysfs `product_name` is empty on
/// consumer parts.
fn pci_device_name(vendor: u32, device: u32) -> String {
    for ids in ["/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids"] {
        if let Ok(text) = fs::read_to_string(ids) {
            let mut in_vendor = false;
            for line in text.lines() {
                if line.starts_with('#') || line.is_empty() {
                    continue;
                }
                if !line.starts_with('\t') {
                    in_vendor = line
                        .get(..4)
                        .and_then(|h| u32::from_str_radix(h, 16).ok())
                        == Some(vendor);
                    continue;
                }
                if in_vendor && !line.starts_with("\t\t") {
                    let l = line.trim_start();
                    if l.get(..4).and_then(|h| u32::from_str_radix(h, 16).ok()) == Some(device) {
                        let name = l[4..].trim();
                        // "Navi 48 [Radeon RX 9070 XT / AI PRO R9700]" -> keep the bracket part.
                        let name = name
                            .rsplit_once('[')
                            .map(|(_, b)| b.trim_end_matches(']').trim())
                            .unwrap_or(name);
                        return name.to_string();
                    }
                }
            }
        }
    }
    let vendor_name = match vendor {
        0x1002 => "AMD",
        0x10de => "NVIDIA",
        0x8086 => "Intel",
        _ => "PCI",
    };
    format!("{vendor_name} GPU {device:04x}")
}

fn connected_display(card: &str) -> Option<DisplayMode> {
    let Ok(rd) = fs::read_dir("/sys/class/drm") else { return None };
    let prefix = format!("{card}-");
    let mut best: Option<DisplayMode> = None;
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with(&prefix) || name.contains("Writeback") {
            continue;
        }
        if read_trim(&e.path().join("status")).as_deref() != Some("connected") {
            continue;
        }
        let mode = read_trim(&e.path().join("modes"))
            .and_then(|m| m.lines().next().map(str::to_string))
            .and_then(|m| {
                let (w, h) = m.split_once('x')?;
                Some(DisplayMode {
                    width: w.parse().ok()?,
                    height: h.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()?,
                    refresh_hz: 0,
                })
            })
            .unwrap_or(DisplayMode { width: 0, height: 0, refresh_hz: 0 });
        if best.is_none() || mode.width > best.as_ref().map_or(0, |b| b.width) {
            best = Some(mode);
        }
    }
    best
}

fn meminfo() -> BTreeMap<String, u64> {
    let mut m = BTreeMap::new();
    if let Ok(text) = fs::read_to_string("/proc/meminfo") {
        for line in text.lines() {
            if let Some((k, v)) = line.split_once(':') {
                let kb: u64 = v.trim().trim_end_matches(" kB").trim().parse().unwrap_or(0);
                m.insert(k.to_string(), kb * 1024);
            }
        }
    }
    m
}

/// One amdgpu client of `pid`: which card, VRAM/GTT held, engine time so far.
#[derive(Debug, Clone, Default)]
struct DrmClient {
    bus: u32,
    vram_bytes: u64,
    gtt_bytes: u64,
    /// Sum of `drm-engine-*` nanoseconds (gfx + compute + sdma).
    engine_ns: u64,
}

/// amdgpu fdinfo lines for every drm client of `pid`, one entry per
/// `drm-client-id` (a process holds several fds on the same client and they
/// all report the same numbers, so they must not be summed twice).
fn drm_clients(pid: u32) -> Vec<DrmClient> {
    let mut by_client: HashMap<String, DrmClient> = HashMap::new();
    let Ok(rd) = fs::read_dir(format!("/proc/{pid}/fdinfo")) else { return vec![] };
    for e in rd.flatten() {
        let Ok(text) = fs::read_to_string(e.path()) else { continue };
        if !text.contains("drm-driver:") {
            continue;
        }
        let mut c = DrmClient::default();
        let mut id = String::new();
        for line in text.lines() {
            let Some((k, v)) = line.split_once(':') else { continue };
            let v = v.trim();
            match k.trim() {
                "drm-client-id" => id = v.to_string(),
                "drm-pdev" => c.bus = pci_bus(v).unwrap_or(u32::MAX),
                "drm-memory-vram" | "drm-total-vram" => c.vram_bytes = kib(v),
                "drm-memory-gtt" | "drm-total-gtt" => c.gtt_bytes = kib(v),
                k if k.starts_with("drm-engine-") => {
                    c.engine_ns += v.trim_end_matches(" ns").trim().parse::<u64>().unwrap_or(0)
                }
                _ => {}
            }
        }
        if !id.is_empty() {
            by_client.entry(format!("{}@{}", id, c.bus)).or_insert(c);
        }
    }
    by_client.into_values().collect()
}

fn kib(v: &str) -> u64 {
    let n: u64 = v.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(0);
    if v.ends_with("MiB") {
        n * 1024 * 1024
    } else {
        n * 1024
    }
}

/// (pid, ppid, starttime ticks) for every process.
fn proc_table() -> Vec<(u32, u32, u64)> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir("/proc") else { return out };
    for e in rd.flatten() {
        let Ok(pid) = e.file_name().to_string_lossy().parse::<u32>() else { continue };
        let Ok(stat) = fs::read_to_string(e.path().join("stat")) else { continue };
        // comm may contain spaces/parens: fields resume after the last ')'.
        let Some(rest) = stat.rsplit_once(')').map(|(_, r)| r) else { continue };
        let f: Vec<&str> = rest.split_whitespace().collect();
        // rest[0]=state, [1]=ppid, ... [19]=starttime (field 22 overall)
        let ppid = f.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        let start = f.get(19).and_then(|s| s.parse().ok()).unwrap_or(0);
        out.push((pid, ppid, start));
    }
    out
}

/// Every process below `root` (see [`super::descendants_in`]). Linux does
/// reparent orphans to init, so the creation-time guard is mostly moot, but
/// the same rule keeps a recycled pid from adopting a stranger.
pub fn process_descendants(root: u32) -> Vec<u32> {
    let table = proc_table();
    let pairs: Vec<(u32, u32)> = table.iter().map(|(p, pp, _)| (*p, *pp)).collect();
    let times: HashMap<u32, u64> = table.iter().map(|(p, _, t)| (*p, *t)).collect();
    super::descendants_in(root, &pairs, &|pid| times.get(&pid).copied())
}

/// Per-card busy for every process with an amdgpu client, from the delta of
/// fdinfo engine time over a short window (the PDH "GPU Engine" analogue).
fn fdinfo_utilization(window: std::time::Duration) -> Vec<GpuEngineUtil> {
    let pids: Vec<u32> = proc_table().iter().map(|(p, _, _)| *p).collect();
    let snap = |pids: &[u32]| -> HashMap<(u32, u32), u64> {
        let mut m = HashMap::new();
        for &pid in pids {
            for c in drm_clients(pid) {
                *m.entry((pid, c.bus)).or_insert(0) += c.engine_ns;
            }
        }
        m
    };
    let a = snap(&pids);
    if a.is_empty() {
        return vec![];
    }
    std::thread::sleep(window);
    let b = snap(&pids);
    let win_ns = window.as_nanos() as f64;
    b.into_iter()
        .filter_map(|((pid, bus), ns)| {
            let before = *a.get(&(pid, bus))?;
            let pct = (ns.saturating_sub(before) as f64 / win_ns * 100.0).min(100.0);
            (pct > 0.0).then_some(GpuEngineUtil { pid, luid_low: bus as u64, percent: pct })
        })
        .collect()
}

/// KFD (ROCm compute) allocations never appear in DRM fdinfo: they are
/// accounted under `/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`. The gpu_id
/// maps to a PCI bus through the topology node's `location_id` (bus << 8 | devfn).
fn kfd_gpu_buses() -> HashMap<String, u32> {
    let mut m = HashMap::new();
    let Ok(rd) = fs::read_dir("/sys/class/kfd/kfd/topology/nodes") else { return m };
    for e in rd.flatten() {
        let Some(gpu_id) = read_trim(&e.path().join("gpu_id")) else { continue };
        if gpu_id == "0" {
            continue;
        }
        let Ok(props) = fs::read_to_string(e.path().join("properties")) else { continue };
        let loc = props.lines().find_map(|l| l.strip_prefix("location_id ")).and_then(|v| v.trim().parse::<u32>().ok());
        if let Some(loc) = loc {
            m.insert(gpu_id, loc >> 8);
        }
    }
    m
}

/// (bus, vram bytes) held by `pid` through KFD.
fn kfd_vram(pid: u32, buses: &HashMap<String, u32>) -> Vec<(u32, u64)> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(format!("/sys/class/kfd/kfd/proc/{pid}")) else { return out };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let Some(gpu_id) = name.strip_prefix("vram_") else { continue };
        let Some(bus) = buses.get(gpu_id) else { continue };
        let bytes = read_u64(&e.path()).unwrap_or(0);
        if bytes > 0 {
            out.push((*bus, bytes));
        }
    }
    out
}

/// Milliseconds this process has spent evicted from the GPU (KFD
/// `stats_<gpu>/evicted_ms`, summed over the cards it touches). Climbing
/// while serving means the desktop is forcing it out of VRAM.
pub fn kfd_evicted_ms(pid: u32) -> u64 {
    let mut total = 0;
    if let Ok(rd) = fs::read_dir(format!("/sys/class/kfd/kfd/proc/{pid}")) {
        for e in rd.flatten() {
            if e.file_name().to_string_lossy().starts_with("stats_") {
                total += read_u64(&e.path().join("evicted_ms")).unwrap_or(0);
            }
        }
    }
    total
}

/// Every pid with a KFD process entry.
fn kfd_pids() -> Vec<u32> {
    fs::read_dir("/sys/class/kfd/kfd/proc")
        .map(|rd| rd.flatten().filter_map(|e| e.file_name().to_string_lossy().parse().ok()).collect())
        .unwrap_or_default()
}

/// Per-card busy from sysfs (all processes together), for callers that
/// only need the card total. Keyed by PCI bus number.
pub fn card_busy_percent() -> BTreeMap<u32, u64> {
    drm_cards()
        .into_iter()
        .filter_map(|(_, dev)| {
            let slot = read_trim(&dev.join("uevent"))?
                .lines()
                .find_map(|l| l.strip_prefix("PCI_SLOT_NAME=").map(str::to_string))?;
            Some((pci_bus(&slot)?, read_u64(&dev.join("gpu_busy_percent"))?))
        })
        .collect()
}

/// (total, used) VRAM bytes of the card on PCI bus `bus`, from sysfs.
pub fn card_vram_bytes(bus: u32) -> Option<(u64, u64)> {
    drm_cards().into_iter().find_map(|(_, dev)| {
        let slot = read_trim(&dev.join("uevent"))?
            .lines()
            .find_map(|l| l.strip_prefix("PCI_SLOT_NAME=").map(str::to_string))?;
        (pci_bus(&slot)? == bus).then(|| {
            (read_u64(&dev.join("mem_info_vram_total")).unwrap_or(0), read_u64(&dev.join("mem_info_vram_used")).unwrap_or(0))
        })
    })
}

impl Platform for LinuxPlatform {
    fn video_adapters(&self) -> Result<Vec<OsAdapter>> {
        let driver_version = read_trim(Path::new("/sys/module/amdgpu/version"))
            .or_else(|| read_trim(Path::new("/proc/sys/kernel/osrelease")))
            .unwrap_or_default();
        let mut out = Vec::new();
        for (card, dev) in drm_cards() {
            let vendor = read_hex(&dev.join("vendor")).unwrap_or(0);
            let device = read_hex(&dev.join("device")).unwrap_or(0);
            let sub_vendor = read_hex(&dev.join("subsystem_vendor")).unwrap_or(0);
            let sub_device = read_hex(&dev.join("subsystem_device")).unwrap_or(0);
            let slot = read_trim(&dev.join("uevent"))
                .and_then(|u| u.lines().find_map(|l| l.strip_prefix("PCI_SLOT_NAME=").map(str::to_string)))
                .unwrap_or_default();
            let bus_number = pci_bus(&slot);
            let name = read_trim(&dev.join("product_name"))
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| pci_device_name(vendor, device));
            out.push(OsAdapter {
                name,
                // Same shape as the Windows PNP id so `stable_key` yields
                // `pci:VEN_1002&DEV_7551&SUBSYS_...:bus03`.
                pnp_device_id: format!(
                    "PCI\\VEN_{vendor:04X}&DEV_{device:04X}&SUBSYS_{sub_device:04X}{sub_vendor:04X}\\{slot}"
                ),
                driver_version: driver_version.clone(),
                bus_number,
                display: connected_display(&card),
                luid_low: bus_number.map(u64::from),
            });
        }
        if out.is_empty() {
            return Err(Error::Platform("no PCI video adapters under /sys/class/drm".into()));
        }
        Ok(out)
    }

    fn system_commit(&self) -> Result<SystemCommit> {
        let m = meminfo();
        let g = |k: &str| m.get(k).copied().unwrap_or(0);
        if g("MemTotal") == 0 {
            return Err(Error::Platform("/proc/meminfo unreadable".into()));
        }
        // Linux overcommits by default (Committed_AS routinely exceeds
        // CommitLimit); the fraction is informational here, not a gate.
        Ok(SystemCommit {
            limit_bytes: g("CommitLimit"),
            charge_bytes: g("Committed_AS"),
            physical_total_bytes: g("MemTotal"),
            physical_available_bytes: g("MemAvailable"),
            pagefile_allocated_bytes: g("SwapTotal").saturating_sub(g("SwapFree")),
            pagefile_can_grow: false,
        })
    }

    fn gpu_utilization(&self) -> Result<Vec<GpuEngineUtil>> {
        // ROCm compute runs on user-mode KFD queues the DRM scheduler never
        // sees, so fdinfo engine time stays 0 for a serving process. Use the
        // card's own busy figure and charge it to the KFD process holding the
        // most VRAM on that card (one server per card here); DRM clients
        // (compositor, browsers) still come from fdinfo deltas.
        let mut out = fdinfo_utilization(std::time::Duration::from_millis(200));
        let buses = kfd_gpu_buses();
        let busy = card_busy_percent();
        let mut top: HashMap<u32, (u32, u64)> = HashMap::new();
        for pid in kfd_pids() {
            for (bus, bytes) in kfd_vram(pid, &buses) {
                let e = top.entry(bus).or_insert((pid, 0));
                if bytes > e.1 {
                    *e = (pid, bytes);
                }
            }
        }
        for (bus, (pid, _)) in top {
            if let Some(pct) = busy.get(&bus) {
                out.retain(|u| !(u.pid == pid && u.luid_low == bus as u64));
                out.push(GpuEngineUtil { pid, luid_low: bus as u64, percent: *pct as f64 });
            }
        }
        Ok(out)
    }

    fn gpu_process_memory(&self, pid: u32) -> Result<Vec<GpuProcessMem>> {
        // amdgpu's DRM fdinfo already includes the KFD allocations of the same
        // VM (a 33 GB model shows under both drm-memory-vram and the kfd
        // vram_<gpu> file), so per card take the larger of the two, not the sum.
        let mut by_bus: BTreeMap<u32, GpuProcessMem> = BTreeMap::new();
        for (bus, bytes) in kfd_vram(pid, &kfd_gpu_buses()) {
            let e = by_bus.entry(bus).or_insert(GpuProcessMem { luid_low: bus as u64, dedicated_bytes: 0, committed_bytes: 0 });
            e.dedicated_bytes = e.dedicated_bytes.max(bytes);
            e.committed_bytes = e.committed_bytes.max(bytes);
        }
        let mut drm: BTreeMap<u32, (u64, u64)> = BTreeMap::new();
        for c in drm_clients(pid) {
            let e = drm.entry(c.bus).or_insert((0, 0));
            e.0 += c.vram_bytes;
            e.1 += c.vram_bytes + c.gtt_bytes;
        }
        for (bus, (vram, committed)) in drm {
            let e = by_bus.entry(bus).or_insert(GpuProcessMem { luid_low: bus as u64, dedicated_bytes: 0, committed_bytes: 0 });
            e.dedicated_bytes = e.dedicated_bytes.max(vram);
            e.committed_bytes = e.committed_bytes.max(committed);
        }
        Ok(by_bus.into_values().collect())
    }
}

#[allow(dead_code)]
fn _unused(_: HashSet<u32>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_bus_parses_slot() {
        assert_eq!(pci_bus("0000:03:00.0"), Some(3));
        assert_eq!(pci_bus("0000:13:00.0"), Some(0x13));
        assert_eq!(pci_bus("garbage"), None);
    }

    #[test]
    fn kib_units() {
        assert_eq!(kib("960 KiB"), 960 * 1024);
        assert_eq!(kib("12 MiB"), 12 * 1024 * 1024);
    }

    #[test]
    fn adapters_enumerate_on_this_box() {
        let a = LinuxPlatform.video_adapters().unwrap_or_default();
        for ad in &a {
            assert!(ad.pnp_device_id.starts_with("PCI\\VEN_"));
        }
    }

    #[test]
    fn commit_reads() {
        let c = LinuxPlatform.system_commit().unwrap();
        assert!(c.physical_total_bytes > 0);
    }
}
