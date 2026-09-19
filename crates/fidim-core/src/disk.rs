//! Checks before a download: is there room on the drive, will every path
//! fit Windows' classic MAX_PATH (with the `.part` suffix it is written
//! under first), and can every tool open it.

use std::path::{Path, PathBuf};

use crate::profile::{Finding, Severity};

/// Room asked for beyond the download itself: the drive is not filled to
/// the last byte.
const FREE_MARGIN: f64 = 0.05;
const GIB: f64 = (1u64 << 30) as f64;

/// Bytes free to this user on the volume holding `path`, or on the
/// volume of its nearest existing ancestor (a destination folder is often
/// not created yet). None when it cannot be told.
pub fn free_bytes(path: &Path) -> Option<u64> {
    let dir = path.ancestors().find(|p| !p.as_os_str().is_empty() && p.exists())?;
    free_bytes_at(dir)
}

#[cfg(windows)]
fn free_bytes_at(dir: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    use windows::core::PCWSTR;
    use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let mut available = 0u64;
    // "Available to the caller" honours disk quotas, unlike the volume total.
    unsafe { GetDiskFreeSpaceExW(PCWSTR(wide.as_ptr()), Some(&mut available), None, None) }.ok()?;
    Some(available)
}

#[cfg(not(windows))]
fn free_bytes_at(_dir: &Path) -> Option<u64> {
    None
}

/// What a download of `files` (destination, size) still has to write:
/// nothing for a file already in place, the rest of a `.part` being resumed.
pub fn bytes_to_fetch(files: &[(PathBuf, u64)]) -> u64 {
    files
        .iter()
        .map(|(dest, size)| {
            if dest.exists() {
                0
            } else {
                let part = std::fs::metadata(crate::fetch::part_path(dest)).map(|m| m.len()).unwrap_or(0);
                size.saturating_sub(part)
            }
        })
        .sum()
}

/// Problems with downloading to `rel_paths` under `root` (paths relative to
/// it, e.g. from `catalog::dest_path`, or absolute), `total_bytes` being
/// what the download still has to write (see `bytes_to_fetch`).
/// `diffusion`: the files are for the DiffusionGemma runner, which opens
/// its model through narrow (ANSI) file APIs and needs an ASCII path.
pub fn check_destination(root: &Path, rel_paths: &[PathBuf], total_bytes: u64, diffusion: bool) -> Vec<Finding> {
    let mut out = Vec::new();
    let full: Vec<PathBuf> = rel_paths.iter().map(|p| root.join(p)).collect();

    // The .part name is the longest the file ever has.
    let longest = full.iter().map(|p| crate::fetch::part_path(p)).max_by_key(|p| crate::update::path_chars(p));
    if let Some(longest) = longest {
        let n = crate::update::path_chars(&longest);
        if n > crate::update::MAX_PATH_CHARS {
            out.push(Finding {
                severity: Severity::Error,
                code: "dest-path-long",
                message: format!(
                    "{} is {n} characters, past Windows' {}-character path limit; choose a model folder with a \
                     shorter path",
                    longest.display(),
                    crate::update::MAX_PATH_CHARS
                ),
            });
        }
    }

    let mut seen: Vec<String> = Vec::new();
    for p in &full {
        // Windows paths compare without case, either separator.
        let key = p.to_string_lossy().to_lowercase().replace('/', "\\");
        if seen.contains(&key) {
            out.push(Finding {
                severity: Severity::Error,
                code: "dest-duplicate",
                message: format!("two files would be saved as {}", p.display()),
            });
        } else {
            seen.push(key);
        }
    }

    let need = (total_bytes as f64 * (1.0 + FREE_MARGIN)).ceil() as u64;
    match free_bytes(root) {
        Some(free) if free < need => out.push(Finding {
            severity: Severity::Error,
            code: "dest-space",
            message: format!(
                "the download needs {:.1} GiB (plus 5%) and the drive holding {} has {:.1} GiB free",
                total_bytes as f64 / GIB,
                root.display(),
                free as f64 / GIB
            ),
        }),
        Some(_) => {}
        None => out.push(Finding {
            severity: Severity::Warning,
            code: "dest-space-unknown",
            message: format!("could not tell how much space is free for {}", root.display()),
        }),
    }

    if let Some(p) = full.iter().find(|p| !p.to_string_lossy().is_ascii()) {
        out.push(if diffusion {
            Finding {
                severity: Severity::Error,
                code: "dest-path-ascii",
                message: format!(
                    "{} is not ASCII: the diffusion runner cannot open a model there; choose a model folder \
                     with an ASCII path",
                    p.display()
                ),
            }
        } else {
            Finding {
                severity: Severity::Warning,
                code: "dest-path-ascii",
                message: format!(
                    "{} is not ASCII: llama-server opens it, but some tools and scripts that take the path \
                     may not",
                    p.display()
                ),
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codes(f: &[Finding]) -> Vec<(&'static str, Severity)> {
        f.iter().map(|x| (x.code, x.severity)).collect()
    }

    #[test]
    fn free_space_of_a_folder_not_made_yet() {
        let tmp = std::env::temp_dir();
        let free = free_bytes(&tmp.join("fidim-not-created").join("deeper"));
        #[cfg(windows)]
        assert!(free.is_some_and(|f| f > 0), "{free:?}");
        #[cfg(not(windows))]
        assert_eq!(free, None);
    }

    #[test]
    fn destination_findings() {
        let root = std::env::temp_dir().join(format!("fidim-disk-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let ok = [PathBuf::from("IFM").join("K2-Horizon-7B-GGUF").join("K2-Horizon-7B-Q4_K_M.gguf")];
        let f = check_destination(&root, &ok, 1 << 20, false);
        #[cfg(windows)]
        assert!(f.is_empty(), "{f:?}");

        // Too long once ".part" is added.
        let base = crate::update::path_chars(&root.join("o").join("r")) + 1;
        let room = crate::update::MAX_PATH_CHARS - base - ".part".len() - ".gguf".len();
        let name = format!("{}.gguf", "x".repeat(room));
        let fits = [PathBuf::from("o").join("r").join(&name)];
        assert_eq!(crate::update::path_chars(&root.join(&fits[0])), crate::update::MAX_PATH_CHARS - 5);
        assert!(!codes(&check_destination(&root, &fits, 0, false)).iter().any(|c| c.0 == "dest-path-long"));
        let long = [PathBuf::from("o").join("r").join(format!("x{name}"))];
        let f = check_destination(&root, &long, 0, false);
        assert!(codes(&f).contains(&("dest-path-long", Severity::Error)), "{f:?}");

        let twice = [PathBuf::from("o/r/a.gguf"), PathBuf::from("o").join("r").join("A.gguf")];
        assert!(codes(&check_destination(&root, &twice, 0, false)).contains(&("dest-duplicate", Severity::Error)));

        #[cfg(windows)]
        {
            let f = check_destination(&root, &ok, u64::MAX / 4, false);
            assert!(codes(&f).contains(&("dest-space", Severity::Error)), "{f:?}");
        }

        let odd = [PathBuf::from("o").join("r\u{e9}").join("m.gguf")];
        assert!(codes(&check_destination(&root, &odd, 0, false)).contains(&("dest-path-ascii", Severity::Warning)));
        assert!(codes(&check_destination(&root, &odd, 0, true)).contains(&("dest-path-ascii", Severity::Error)));
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn bytes_still_to_fetch() {
        let root = std::env::temp_dir().join(format!("fidim-disk-fetch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let done = root.join("done.gguf");
        std::fs::write(&done, b"1234").unwrap();
        let half = root.join("half.gguf");
        std::fs::write(crate::fetch::part_path(&half), vec![0u8; 40]).unwrap();
        let files = [(done, 4), (half, 100), (root.join("new.gguf"), 1000)];
        assert_eq!(bytes_to_fetch(&files), 60 + 1000);
        std::fs::remove_dir_all(&root).ok();
    }
}
