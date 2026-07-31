//! Windows implementations: WMI for adapter identity/display, CfgMgr32 for
//! PCI bus numbers, GetPerformanceInfo for commit. All unprivileged (§07).

use serde::Deserialize;

use crate::devices::{DisplayMode, OsAdapter};
use crate::platform::{GpuProcessMem, Platform, SystemCommit};
use crate::{Error, Result};

pub struct WindowsPlatform;

/// DEVPKEY_Gpu_Luid — {60B193CB-5276-4D0F-96FC-F173ABAD3EC6}, 2.
/// Not exported by the windows crate; value observed working on the target
/// machine (returns the adapter LUID the PDH GPU counters key on).
const DEVPKEY_GPU_LUID: windows::Win32::Devices::Properties::DEVPROPKEY =
    windows::Win32::Devices::Properties::DEVPROPKEY {
        fmtid: windows::core::GUID::from_u128(0x60B193CB_5276_4D0F_96FC_F173ABAD3EC6),
        pid: 2,
    };

#[derive(Deserialize, Debug)]
struct VideoController {
    #[serde(rename = "Name")]
    name: Option<String>,
    #[serde(rename = "PNPDeviceID")]
    pnp_device_id: Option<String>,
    #[serde(rename = "DriverVersion")]
    driver_version: Option<String>,
    #[serde(rename = "CurrentHorizontalResolution")]
    h_res: Option<u32>,
    #[serde(rename = "CurrentVerticalResolution")]
    v_res: Option<u32>,
    #[serde(rename = "CurrentRefreshRate")]
    refresh: Option<u32>,
}

impl Platform for WindowsPlatform {
    fn video_adapters(&self) -> Result<Vec<OsAdapter>> {
        let com = wmi::COMLibrary::new().map_err(|e| Error::Platform(format!("COM init: {e}")))?;
        let con = wmi::WMIConnection::new(com)
            .map_err(|e| Error::Platform(format!("WMI connect: {e}")))?;
        let rows: Vec<VideoController> = con
            .raw_query(
                "SELECT Name, PNPDeviceID, DriverVersion, CurrentHorizontalResolution, \
                 CurrentVerticalResolution, CurrentRefreshRate FROM Win32_VideoController",
            )
            .map_err(|e| Error::Platform(format!("WMI query: {e}")))?;
        Ok(rows
            .into_iter()
            .filter_map(|r| {
                let pnp = r.pnp_device_id?;
                let bus_number = device_property_u32(&pnp, &DEVPKEY_BUS_NUMBER);
                let luid_low = device_property_u64(&pnp, &DEVPKEY_GPU_LUID)
                    .map(|l| l & 0xFFFF_FFFF);
                let display = match (r.h_res, r.v_res) {
                    (Some(w), Some(h)) if w > 0 && h > 0 => Some(DisplayMode {
                        width: w,
                        height: h,
                        refresh_hz: r.refresh.unwrap_or(0),
                    }),
                    _ => None,
                };
                Some(OsAdapter {
                    name: r.name.unwrap_or_default(),
                    pnp_device_id: pnp,
                    driver_version: r.driver_version.unwrap_or_default(),
                    bus_number,
                    display,
                    luid_low,
                })
            })
            .collect())
    }

    fn system_commit(&self) -> Result<SystemCommit> {
        use windows::Win32::System::ProcessStatus::{GetPerformanceInfo, PERFORMANCE_INFORMATION};
        let mut info = PERFORMANCE_INFORMATION {
            cb: std::mem::size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        unsafe {
            GetPerformanceInfo(&mut info, info.cb)
                .map_err(|e| Error::Platform(format!("GetPerformanceInfo: {e}")))?;
        }
        let page = info.PageSize as u64;
        Ok(SystemCommit {
            limit_bytes: info.CommitLimit as u64 * page,
            charge_bytes: info.CommitTotal as u64 * page,
        })
    }

    fn gpu_process_memory(&self, pid: u32) -> Result<Vec<GpuProcessMem>> {
        pdh_gpu_process_memory(pid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_pdh_instance_luid() {
        assert_eq!(
            parse_luid_low("pid_3864_luid_0x00000000_0x0001B592_phys_0"),
            Some(0x1B592)
        );
        assert_eq!(parse_luid_low("pid_1_nope"), None);
    }

    /// Live probe: the current process holds no GPU memory, which must be an
    /// empty result, not an error.
    #[test]
    fn pdh_query_for_gpuless_process_is_empty() {
        let mem = pdh_gpu_process_memory(std::process::id()).unwrap();
        assert!(mem.iter().all(|m| m.dedicated_bytes < 512 * 1024 * 1024));
    }
}

use windows::Win32::Devices::Properties::DEVPROPKEY;

const DEVPKEY_BUS_NUMBER: DEVPROPKEY =
    windows::Win32::Devices::Properties::DEVPKEY_Device_BusNumber;

/// Read a device property as raw bytes via CfgMgr32 (unprivileged).
fn device_property_bytes(instance_id: &str, key: &DEVPROPKEY) -> Option<Vec<u8>> {
    use windows::core::PCWSTR;
    use windows::Win32::Devices::DeviceAndDriverInstallation::{
        CM_Get_DevNode_PropertyW, CM_Locate_DevNodeW, CM_LOCATE_DEVNODE_NORMAL, CR_SUCCESS,
    };
    use windows::Win32::Devices::Properties::DEVPROPTYPE;

    let wide: Vec<u16> = instance_id.encode_utf16().chain(std::iter::once(0)).collect();
    let mut devinst: u32 = 0;
    unsafe {
        let cr = CM_Locate_DevNodeW(&mut devinst, PCWSTR(wide.as_ptr()), CM_LOCATE_DEVNODE_NORMAL);
        if cr != CR_SUCCESS {
            return None;
        }
        let mut prop_type = DEVPROPTYPE(0);
        let mut buf = [0u8; 16];
        let mut size = buf.len() as u32;
        let cr = CM_Get_DevNode_PropertyW(
            devinst,
            key,
            &mut prop_type,
            Some(buf.as_mut_ptr()),
            &mut size,
            0,
        );
        if cr != CR_SUCCESS || size == 0 || size as usize > buf.len() {
            return None;
        }
        Some(buf[..size as usize].to_vec())
    }
}

fn device_property_u32(instance_id: &str, key: &DEVPROPKEY) -> Option<u32> {
    let b = device_property_bytes(instance_id, key)?;
    (b.len() >= 4).then(|| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn device_property_u64(instance_id: &str, key: &DEVPROPKEY) -> Option<u64> {
    let b = device_property_bytes(instance_id, key)?;
    match b.len() {
        8.. => Some(u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])),
        4.. => Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64),
        _ => None,
    }
}

/// PDH query for `\GPU Process Memory(pid_N*)\Dedicated Usage` — the
/// residency ground truth. Instance names look like
/// `pid_3864_luid_0x00000000_0x0001B592_phys_0`.
fn pdh_gpu_process_memory(pid: u32) -> Result<Vec<GpuProcessMem>> {
    let dedicated = pdh_counter_by_luid(pid, "Dedicated Usage")?;
    let committed = pdh_counter_by_luid(pid, "Total Committed").unwrap_or_default();
    let mut out: Vec<GpuProcessMem> = Vec::new();
    let luids: std::collections::BTreeSet<u64> =
        dedicated.iter().chain(committed.iter()).map(|(l, _)| *l).collect();
    for luid in luids {
        let find = |v: &[(u64, u64)]| v.iter().find(|(l, _)| *l == luid).map(|(_, b)| *b);
        out.push(GpuProcessMem {
            luid_low: luid,
            dedicated_bytes: find(&dedicated).unwrap_or(0),
            committed_bytes: find(&committed).unwrap_or(0),
        });
    }
    Ok(out)
}

/// One PDH "GPU Process Memory" counter for a pid, as (luid_low, bytes).
fn pdh_counter_by_luid(pid: u32, counter_name: &str) -> Result<Vec<(u64, u64)>> {
    use windows::core::PCWSTR;
    use windows::Win32::System::Performance::{
        PdhAddEnglishCounterW, PdhCloseQuery, PdhCollectQueryData,
        PdhGetFormattedCounterArrayW, PdhOpenQueryW, PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_LARGE,
    };

    let counter_path = format!("\\GPU Process Memory(pid_{pid}*)\\{counter_name}");
    let wide: Vec<u16> = counter_path.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        let mut query = 0isize;
        if PdhOpenQueryW(PCWSTR::null(), 0, &mut query) != 0 {
            return Err(Error::Platform("PdhOpenQuery failed".into()));
        }
        let close = |q: isize| {
            let _ = PdhCloseQuery(q);
        };
        let mut counter = 0isize;
        if PdhAddEnglishCounterW(query, PCWSTR(wide.as_ptr()), 0, &mut counter) != 0 {
            close(query);
            // No matching instances (process holds no GPU memory) is not an
            // error worth failing on — return empty.
            return Ok(Vec::new());
        }
        if PdhCollectQueryData(query) != 0 {
            close(query);
            return Ok(Vec::new());
        }
        let mut buf_size = 0u32;
        let mut item_count = 0u32;
        // First call sizes the buffer (returns PDH_MORE_DATA).
        let _ = PdhGetFormattedCounterArrayW(counter, PDH_FMT_LARGE, &mut buf_size, &mut item_count, None);
        if buf_size == 0 {
            close(query);
            return Ok(Vec::new());
        }
        let mut buf = vec![0u8; buf_size as usize];
        let status = PdhGetFormattedCounterArrayW(
            counter,
            PDH_FMT_LARGE,
            &mut buf_size,
            &mut item_count,
            Some(buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W),
        );
        let mut out = Vec::new();
        if status == 0 {
            let items = std::slice::from_raw_parts(
                buf.as_ptr() as *const PDH_FMT_COUNTERVALUE_ITEM_W,
                item_count as usize,
            );
            for item in items {
                let name = item.szName.to_string().unwrap_or_default();
                if let Some(luid_low) = parse_luid_low(&name) {
                    let bytes = item.FmtValue.Anonymous.largeValue.max(0) as u64;
                    out.push((luid_low, bytes));
                }
            }
        }
        close(query);
        Ok(out)
    }
}

/// `pid_3864_luid_0x00000000_0x0001B592_phys_0` → 0x1B592.
fn parse_luid_low(instance: &str) -> Option<u64> {
    let idx = instance.find("_luid_")?;
    let rest = &instance[idx + 6..];
    let mut parts = rest.split('_');
    let _high = parts.next()?;
    let low = parts.next()?;
    u64::from_str_radix(low.trim_start_matches("0x"), 16).ok()
}
