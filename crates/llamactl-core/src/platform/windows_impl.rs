//! Windows implementations: WMI for adapter identity/display, CfgMgr32 for
//! PCI bus numbers, GetPerformanceInfo for commit. All unprivileged (§07).

use serde::Deserialize;

use crate::devices::{DisplayMode, OsAdapter};
use crate::platform::{Platform, SystemCommit};
use crate::{Error, Result};

pub struct WindowsPlatform;

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
                let bus_number = bus_number_for_instance(&pnp);
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
}

/// PCI bus number for a PNP instance path via CfgMgr32 (unprivileged).
fn bus_number_for_instance(instance_id: &str) -> Option<u32> {
    use windows::core::PCWSTR;
    use windows::Win32::Devices::DeviceAndDriverInstallation::{
        CM_Get_DevNode_PropertyW, CM_Locate_DevNodeW, CM_LOCATE_DEVNODE_NORMAL, CR_SUCCESS,
    };
    use windows::Win32::Devices::Properties::{DEVPKEY_Device_BusNumber, DEVPROPTYPE};

    let wide: Vec<u16> = instance_id.encode_utf16().chain(std::iter::once(0)).collect();
    let mut devinst: u32 = 0;
    unsafe {
        let cr = CM_Locate_DevNodeW(&mut devinst, PCWSTR(wide.as_ptr()), CM_LOCATE_DEVNODE_NORMAL);
        if cr != CR_SUCCESS {
            return None;
        }
        let mut prop_type = DEVPROPTYPE(0);
        let mut buf = [0u8; 4];
        let mut size = buf.len() as u32;
        let cr = CM_Get_DevNode_PropertyW(
            devinst,
            &DEVPKEY_Device_BusNumber,
            &mut prop_type,
            Some(buf.as_mut_ptr()),
            &mut size,
            0,
        );
        if cr != CR_SUCCESS || size != 4 {
            return None;
        }
        Some(u32::from_le_bytes(buf))
    }
}
