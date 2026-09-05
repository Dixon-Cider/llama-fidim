//! Device enumeration, stable keys, and profile-key resolution (spec R-03,
//! R-13; pre-flight checks 3–5).
//!
//! The canonical device index space is llama-server's own (`--list-devices`),
//! because it is by definition the enumeration the launched server will use.
//! On the target machine that space is: 0 = R9700 (bus 3), 1 = iGPU (bus 19),
//! 2 = R9700 (bus 8) — NOT bus-ascending, and NOT the same space hipInfo
//! reports (hipInfo filters the iGPU out entirely). Nothing here may assume
//! any ordering beyond one, explicitly-marked case: discrete devices appear in
//! the same relative order in both spaces (verified against hipInfo bus data,
//! and re-verified at runtime by per-process residency checks).

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

/// One line of `llama-server --list-devices`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ListedDevice {
    /// Runtime index within the backend (the N of `ROCmN`).
    pub index: u32,
    /// Backend prefix, e.g. `ROCm`.
    pub backend: String,
    pub name: String,
    pub total_mib: u64,
    pub free_mib: u64,
}

/// Parse `--list-devices` output. Unparseable output is a hard error — this
/// feeds launch decisions, so guessing is worse than failing (spec §07).
pub fn parse_list_devices(text: &str) -> Result<Vec<ListedDevice>> {
    let re = regex::Regex::new(
        r"(?m)^\s*([A-Za-z]+)(\d+):\s+(.+?)\s+\((\d+)\s+MiB,\s+(\d+)\s+MiB free\)\s*$",
    )
    .unwrap();
    let mut devices = Vec::new();
    for caps in re.captures_iter(text) {
        devices.push(ListedDevice {
            backend: caps[1].to_string(),
            index: caps[2].parse().map_err(|_| bad_output(text))?,
            name: caps[3].trim().to_string(),
            total_mib: caps[4].parse().map_err(|_| bad_output(text))?,
            free_mib: caps[5].parse().map_err(|_| bad_output(text))?,
        });
    }
    if devices.is_empty() {
        // Distinguish "no devices" (header present) from format drift.
        if text.contains("Available devices:") {
            return Ok(devices);
        }
        return Err(bad_output(text));
    }
    Ok(devices)
}

fn bad_output(text: &str) -> Error {
    let sample: String = text.chars().take(300).collect();
    Error::ListDevicesUnparseable(sample)
}

/// One device block of ROCm's `hipInfo.exe` output. Only used as a source of
/// PCI bus numbers and integrated flags for discrete GPUs — its index space
/// is NOT llama-server's.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HipInfoDevice {
    pub index: u32,
    pub name: String,
    pub pci_bus: u32,
    pub is_integrated: bool,
    pub gcn_arch: Option<String>,
}

pub fn parse_hipinfo(text: &str) -> Vec<HipInfoDevice> {
    let mut devices: Vec<HipInfoDevice> = Vec::new();
    let mut current: Option<HipInfoDevice> = None;
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("device#") {
            if let Some(d) = current.take() {
                devices.push(d);
            }
            if let Ok(idx) = rest.trim().parse() {
                current = Some(HipInfoDevice {
                    index: idx,
                    name: String::new(),
                    pci_bus: 0,
                    is_integrated: false,
                    gcn_arch: None,
                });
            }
        } else if let Some(d) = current.as_mut() {
            if let Some(v) = field(line, "Name:") {
                d.name = v;
            } else if let Some(v) = field(line, "pciBusID:") {
                d.pci_bus = v.parse().unwrap_or(0);
            } else if let Some(v) = field(line, "isIntegrated:") {
                d.is_integrated = v.trim() != "0";
            } else if let Some(v) = field(line, "gcnArchName:") {
                d.gcn_arch = Some(v);
            }
        }
    }
    if let Some(d) = current.take() {
        devices.push(d);
    }
    devices
}

fn field(line: &str, prefix: &str) -> Option<String> {
    line.strip_prefix(prefix).map(|v| v.trim().to_string())
}

/// A display adapter as reported by the OS (WMI Win32_VideoController +
/// DEVPKEY_Device_BusNumber). Produced by the platform layer; consumed here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OsAdapter {
    pub name: String,
    /// PNP instance path, e.g. `PCI\VEN_1002&DEV_7551&SUBSYS_...\...`.
    pub pnp_device_id: String,
    pub driver_version: String,
    pub bus_number: Option<u32>,
    /// Present when the adapter is currently driving a display (R-06).
    pub display: Option<DisplayMode>,
    /// Low dword of the adapter LUID — links PDH "GPU Process Memory"
    /// counter instances (`pid_N_luid_0x.._0x<low>_phys_0`) to this card.
    /// Volatile across reboots/driver resets; never persisted in profiles.
    #[serde(default)]
    pub luid_low: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayMode {
    pub width: u32,
    pub height: u32,
    pub refresh_hz: u32,
}

/// The fully-correlated device the rest of the tool operates on.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct Device {
    /// Stable identity persisted in profiles: `pci:<VEN&DEV&SUBSYS>:busNN`.
    pub stable_key: String,
    pub name: String,
    /// Index in llama-server's enumeration — resolved fresh, never persisted
    /// as authoritative.
    pub hip_index: u32,
    pub backend: String,
    pub total_mib: u64,
    pub free_mib: u64,
    pub integrated: bool,
    pub bus_number: Option<u32>,
    pub driver_version: Option<String>,
    pub display: Option<DisplayMode>,
    /// LUID low dword for PDH counter attribution (residency verification).
    pub luid_low: Option<u64>,
    /// True when the OS-adapter correlation relied on the discrete-order
    /// assumption rather than a unique name match. Surfaced in the Devices
    /// view; runtime residency checks are the backstop.
    pub correlation_assumed: bool,
}

/// Derive the stable key from a PNP instance path plus bus number.
/// `PCI\VEN_1002&DEV_7551&SUBSYS_54131849&REV_C0\<instance>` + bus 8
/// → `pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08`.
/// The volatile instance suffix (renumbered by driver updates) is dropped;
/// the bus number distinguishes identical cards.
pub fn stable_key(pnp_device_id: &str, bus_number: Option<u32>) -> String {
    let hw = pnp_device_id
        .split('\\')
        .nth(1)
        .unwrap_or(pnp_device_id)
        .split('&')
        .filter(|part| !part.starts_with("REV_"))
        .collect::<Vec<_>>()
        .join("&");
    match bus_number {
        Some(bus) => format!("pci:{hw}:bus{bus:02}"),
        None => format!("pci:{hw}:bus??"),
    }
}

/// Correlate llama-server's enumeration with OS adapters.
///
/// Classification: a listed device is integrated when its name contains any
/// configured integrated pattern (APU marketing names — `Radeon(TM) Graphics`
/// — carry no model suffix), or when hipInfo (if provided) knows the name
/// only as an integrated device.
///
/// Matching: adapters whose name matches exactly one listed device match by
/// name. Groups of identically-named devices (the two R9700s) are matched in
/// order: listed devices by ascending hip index ↔ adapters by ascending bus.
/// That order assumption is marked on the result (`correlation_assumed`) and
/// verified at runtime by the per-process residency check.
pub fn correlate(
    listed: &[ListedDevice],
    adapters: &[OsAdapter],
    hipinfo: &[HipInfoDevice],
    integrated_patterns: &[String],
) -> Vec<Device> {
    let mut result = Vec::new();
    // Group listed devices and adapters by name.
    let names: std::collections::BTreeSet<&str> =
        listed.iter().map(|d| d.name.as_str()).collect();
    for name in names {
        let group: Vec<&ListedDevice> =
            listed.iter().filter(|d| d.name == name).collect();
        let mut adapter_group: Vec<&OsAdapter> =
            adapters.iter().filter(|a| a.name == name).collect();
        adapter_group.sort_by_key(|a| a.bus_number.unwrap_or(u32::MAX));

        let integrated = integrated_patterns.iter().any(|p| name.contains(p.as_str()))
            || hipinfo.iter().any(|h| h.name == name && h.is_integrated);
        let assumed = group.len() > 1;

        for (i, dev) in group.iter().enumerate() {
            let adapter = adapter_group.get(i).copied();
            let key = match adapter {
                Some(a) => stable_key(&a.pnp_device_id, a.bus_number),
                // No OS adapter match — key on name+index as a last resort,
                // clearly marked non-PCI so it never silently masquerades.
                None => format!("name:{}:{}", name.replace(' ', "_"), dev.index),
            };
            result.push(Device {
                stable_key: key,
                name: dev.name.clone(),
                hip_index: dev.index,
                backend: dev.backend.clone(),
                total_mib: dev.total_mib,
                free_mib: dev.free_mib,
                integrated,
                bus_number: adapter.and_then(|a| a.bus_number),
                driver_version: adapter.map(|a| a.driver_version.clone()),
                display: adapter.and_then(|a| a.display.clone()),
                luid_low: adapter.and_then(|a| a.luid_low),
                correlation_assumed: assumed && adapter.is_some(),
            });
        }
    }
    result.sort_by_key(|d| d.hip_index);
    result
}

/// Outcome of resolving a persisted profile key against current devices.
#[derive(Debug, Clone, Serialize)]
pub struct Resolution {
    pub device: Device,
    /// Set when the key did not match exactly but re-bound to the only
    /// unclaimed device of the same model (a renumbering event) — the caller
    /// must surface this as a warning and update the stored key.
    pub rebound_from: Option<String>,
}

/// Resolve a stable key to a current device (pre-flight check 3).
///
/// Exact key match wins. If the key's hardware part matches devices but the
/// bus moved (card reseated, enumeration change), re-bind only when exactly
/// one same-model candidate is not claimed by `taken_keys`; ambiguity is an
/// error, never a guess.
pub fn resolve_key<'d>(
    key: &str,
    devices: &'d [Device],
    taken_keys: &[String],
) -> Result<Resolution> {
    if let Some(d) = devices.iter().find(|d| d.stable_key == key) {
        return Ok(Resolution { device: d.clone(), rebound_from: None });
    }
    // Hardware part of `pci:<hw>:busNN`.
    let hw_part = key.split(':').nth(1).unwrap_or(key);
    let candidates: Vec<&'d Device> = devices
        .iter()
        .filter(|d| {
            d.stable_key.split(':').nth(1) == Some(hw_part)
                && !taken_keys.contains(&d.stable_key)
        })
        .collect();
    match candidates.len() {
        1 => Ok(Resolution {
            device: candidates[0].clone(),
            rebound_from: Some(key.to_string()),
        }),
        0 => Err(Error::DeviceKeyUnresolved { key: key.to_string() }),
        n => Err(Error::DeviceKeyAmbiguous { key: key.to_string(), candidates: n }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real output captured from the target machine (fixtures/).
    const REAL_LIST: &str = "Available devices:\n  ROCm0: AMD Radeon AI PRO R9700 (32624 MiB, 32472 MiB free)\n  ROCm1: AMD Radeon(TM) Graphics (12381 MiB, 12099 MiB free)\n  ROCm2: AMD Radeon AI PRO R9700 (32624 MiB, 32472 MiB free)\n";

    fn real_adapters() -> Vec<OsAdapter> {
        vec![
            OsAdapter {
                name: "AMD Radeon(TM) Graphics".into(),
                pnp_device_id: r"PCI\VEN_1002&DEV_13C0&SUBSYS_7D781462&REV_CB\4&27E89230&0&0041".into(),
                driver_version: "32.0.21045.1000".into(),
                bus_number: Some(19),
                display: Some(DisplayMode { width: 1920, height: 1080, refresh_hz: 59 }),
                luid_low: Some(0x1DCEC),
            },
            OsAdapter {
                name: "AMD Radeon AI PRO R9700".into(),
                pnp_device_id: r"PCI\VEN_1002&DEV_7551&SUBSYS_54131849&REV_C0\6&3305601B&0&00000009".into(),
                driver_version: "32.0.31035.1003".into(),
                bus_number: Some(3),
                display: Some(DisplayMode { width: 2560, height: 1440, refresh_hz: 144 }),
                luid_low: Some(0x1621C),
            },
            OsAdapter {
                name: "AMD Radeon AI PRO R9700".into(),
                pnp_device_id: r"PCI\VEN_1002&DEV_7551&SUBSYS_54131849&REV_C0\8&1A11ECB9&0&000000000011".into(),
                driver_version: "32.0.31035.1003".into(),
                bus_number: Some(8),
                display: Some(DisplayMode { width: 1920, height: 1080, refresh_hz: 59 }),
                luid_low: Some(0x1B592),
            },
        ]
    }

    #[test]
    fn parses_real_list_devices() {
        let devs = parse_list_devices(REAL_LIST).unwrap();
        assert_eq!(devs.len(), 3);
        assert_eq!(devs[0].index, 0);
        assert_eq!(devs[1].name, "AMD Radeon(TM) Graphics");
        assert_eq!(devs[2].index, 2);
        assert_eq!(devs[2].total_mib, 32624);
        assert_eq!(devs[1].free_mib, 12099);
    }

    #[test]
    fn format_drift_is_an_error_not_a_guess() {
        let drifted = "Devices found:\n  gpu 0 -> R9700 [32 GB]\n";
        assert!(parse_list_devices(drifted).is_err());
    }

    #[test]
    fn stable_key_drops_volatile_instance_and_rev() {
        let key = stable_key(
            r"PCI\VEN_1002&DEV_7551&SUBSYS_54131849&REV_C0\6&3305601B&0&00000009",
            Some(3),
        );
        assert_eq!(key, "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus03");
    }

    #[test]
    fn correlates_real_topology() {
        let listed = parse_list_devices(REAL_LIST).unwrap();
        let devices = correlate(
            &listed,
            &real_adapters(),
            &[],
            &["Radeon(TM) Graphics".to_string()],
        );
        assert_eq!(devices.len(), 3);
        // The iGPU at index 1, classified integrated, on bus 19.
        assert!(devices[1].integrated);
        assert_eq!(devices[1].bus_number, Some(19));
        assert!(!devices[1].correlation_assumed, "unique name match is not assumed");
        // The two R9700s: index 0 ↔ bus 3, index 2 ↔ bus 8, marked assumed.
        assert!(!devices[0].integrated);
        assert_eq!(devices[0].bus_number, Some(3));
        assert_eq!(devices[2].bus_number, Some(8));
        assert!(devices[0].correlation_assumed);
        assert_eq!(devices[0].stable_key, "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus03");
        assert_eq!(devices[2].stable_key, "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08");
        // Both R9700s are currently driving displays — the live R-06 case.
        assert!(devices[0].display.is_some());
        assert!(devices[2].display.is_some());
    }

    #[test]
    fn resolve_exact_then_rebind_then_ambiguous() {
        let listed = parse_list_devices(REAL_LIST).unwrap();
        let devices = correlate(
            &listed,
            &real_adapters(),
            &[],
            &["Radeon(TM) Graphics".to_string()],
        );
        // Exact.
        let r = resolve_key("pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08", &devices, &[]).unwrap();
        assert_eq!(r.device.hip_index, 2);
        assert!(r.rebound_from.is_none());

        // Bus moved: old bus 5 no longer exists; both R9700s free → ambiguous.
        let stale = "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus05";
        assert!(matches!(
            resolve_key(stale, &devices, &[]),
            Err(Error::DeviceKeyAmbiguous { .. })
        ));

        // With one R9700 claimed, the stale key re-binds to the other.
        let taken = vec!["pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus03".to_string()];
        let r = resolve_key(stale, &devices, &taken).unwrap();
        assert_eq!(r.device.hip_index, 2);
        assert!(r.rebound_from.is_some());

        // Unknown hardware entirely.
        assert!(matches!(
            resolve_key("pci:VEN_10DE&DEV_2684:bus01", &devices, &[]),
            Err(Error::DeviceKeyUnresolved { .. })
        ));
    }

    #[test]
    fn parses_real_hipinfo_shape() {
        let text = "device#                           0\nName:                             AMD Radeon AI PRO R9700\npciBusID:                         3\ntotalGlobalMem:                   31.86 GB\nisIntegrated:                     0\ngcnArchName:                      gfx1201\ndevice#                           1\nName:                             AMD Radeon AI PRO R9700\npciBusID:                         8\nisIntegrated:                     0\ngcnArchName:                      gfx1201\n";
        let devs = parse_hipinfo(text);
        assert_eq!(devs.len(), 2);
        assert_eq!(devs[0].pci_bus, 3);
        assert_eq!(devs[1].pci_bus, 8);
        assert!(!devs[0].is_integrated);
        assert_eq!(devs[0].gcn_arch.as_deref(), Some("gfx1201"));
    }
}
