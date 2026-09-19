//! The user PATH, for `fidim path add|remove|status`: put the folder that
//! holds `fidim.exe` on it so `fidim` works in every new terminal, take it
//! off again, or say where things stand.
//!
//! The installer leaves PATH alone on purpose. NSIS string handling cuts a
//! value at 1024 (or 8192) characters, `setx` cuts at 1024, and .NET's
//! `SetEnvironmentVariable` rewrites the value as REG_SZ with every
//! `%VARIABLE%` expanded. So the value is read and written here, whole:
//! `HKCU\Environment\Path`, its registry type kept, every other entry kept
//! exactly as written. Afterwards a WM_SETTINGCHANGE broadcast tells
//! Explorer, so terminals started from then on see the change; terminals
//! already open keep the PATH they started with.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::{Error, Result};

/// The most characters one Windows environment variable can hold.
pub const MAX_CHARS: usize = 32_767;

/// Where the user PATH lives, for messages.
pub const USER_PATH_KEY: &str = r"HKCU\Environment\Path";

/// What `add` or `remove` did to the user PATH.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Change {
    Added,
    AlreadyThere,
    /// How many entries naming the folder were dropped (duplicates count).
    Removed(usize),
    NotThere,
}

impl Change {
    /// The user PATH value was rewritten.
    pub fn changed(self) -> bool {
        matches!(self, Change::Added | Change::Removed(_))
    }
}

/// Another PATH folder that holds a `fidim.exe`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OtherFidim {
    pub dir: PathBuf,
    /// `system` or `user`: which PATH lists it.
    pub from: &'static str,
    /// A new terminal finds this one before the folder asked about, so
    /// `fidim` runs this copy (always true when that folder is not on PATH).
    pub first: bool,
}

/// Where a folder stands on PATH.
#[derive(Debug, Clone, Serialize)]
pub struct Status {
    pub dir: PathBuf,
    /// On the user PATH (`HKCU\Environment\Path`).
    pub user: bool,
    /// On the system PATH (read only here; changing it needs an administrator).
    pub system: bool,
    /// On the PATH of this process, i.e. of the terminal `fidim` runs in.
    pub this_shell: bool,
    /// Length of the user PATH value, in characters.
    pub user_chars: usize,
    /// Other folders on PATH that hold a `fidim.exe`, in the order a new
    /// terminal searches them (the system PATH, then the user PATH).
    pub others: Vec<OtherFidim>,
}

/// The folder holding the running executable: the folder `fidim path` manages.
pub fn this_dir() -> Result<PathBuf> {
    let exe = std::env::current_exe().map_err(|e| Error::UserPath(format!("cannot locate this executable: {e}")))?;
    exe.parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| Error::UserPath(format!("{} has no parent folder", exe.display())))
}

// ------------------------------------------------------------ PATH text ----

/// A PATH entry reduced for comparison: variables expanded, surrounding
/// quotes and trailing separators dropped, `/` read as `\`, lower case
/// (Windows paths ignore case). Empty for an empty entry.
fn comparable(entry: &str, expand: &dyn Fn(&str) -> String) -> String {
    let e = entry.trim().trim_matches('"');
    let e = if e.contains('%') { expand(e) } else { e.to_string() };
    let mut s = e.trim().replace('/', "\\");
    while s.len() > 1 && s.ends_with('\\') {
        s.pop();
    }
    s.to_lowercase()
}

/// Whether any of `value`'s entries names `dir`.
pub fn contains(value: &str, dir: &Path, expand: &dyn Fn(&str) -> String) -> bool {
    let want = comparable(&dir.to_string_lossy(), expand);
    !want.is_empty() && value.split(';').any(|e| comparable(e, expand) == want)
}

/// `value` with `dir` added as its last entry (searched last, as Windows'
/// own editor adds them), or None when an entry already names it.
pub fn appended(value: &str, dir: &Path, expand: &dyn Fn(&str) -> String) -> Option<String> {
    if contains(value, dir, expand) {
        return None;
    }
    let d = dir.to_string_lossy();
    Some(if value.is_empty() {
        d.into_owned()
    } else if value.ends_with(';') {
        format!("{value}{d}")
    } else {
        format!("{value};{d}")
    })
}

/// `value` without the entries that name `dir`, and how many there were.
/// Every other entry, empty ones included, stays exactly as written.
pub fn removed(value: &str, dir: &Path, expand: &dyn Fn(&str) -> String) -> (String, usize) {
    let want = comparable(&dir.to_string_lossy(), expand);
    let mut dropped = 0;
    let kept: Vec<&str> = value
        .split(';')
        .filter(|e| {
            let hit = !want.is_empty() && comparable(e, expand) == want;
            dropped += usize::from(hit);
            !hit
        })
        .collect();
    (kept.join(";"), dropped)
}

/// Folders other than `dir` that hold a `fidim.exe`, in search order: the
/// system PATH's entries, then the user PATH's. Each folder once.
fn other_fidims(
    system: &str,
    user: &str,
    dir: &Path,
    expand: &dyn Fn(&str) -> String,
    has_fidim: &dyn Fn(&Path) -> bool,
) -> Vec<OtherFidim> {
    let want = comparable(&dir.to_string_lossy(), expand);
    let mut seen_ours = false;
    let mut seen: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for (from, value) in [("system", system), ("user", user)] {
        for entry in value.split(';') {
            let key = comparable(entry, expand);
            if key.is_empty() || seen.contains(&key) {
                continue;
            }
            seen.push(key.clone());
            if key == want {
                seen_ours = true;
                continue;
            }
            let raw = entry.trim().trim_matches('"');
            let folder = PathBuf::from(if raw.contains('%') { expand(raw) } else { raw.to_string() });
            if has_fidim(&folder) {
                out.push(OtherFidim { dir: folder, from, first: !seen_ours });
            }
        }
    }
    out
}

/// Why `dir` cannot be a PATH entry as it stands, if it cannot.
fn unusable(dir: &Path) -> Option<String> {
    let d = dir.to_string_lossy();
    if !dir.is_absolute() {
        Some(format!("{d} is not an absolute path"))
    } else if d.contains(';') || d.contains('"') {
        Some(format!("{d} contains ';' or '\"', which a PATH entry cannot hold unquoted"))
    } else {
        None
    }
}

// --------------------------------------------------------------- public ----

/// Where `dir` stands: on the user PATH, the system PATH, this terminal's
/// PATH, and which other `fidim.exe` a new terminal could find.
#[cfg(windows)]
pub fn status(dir: &Path) -> Result<Status> {
    let user = reg::read(reg::Hive::User, reg::USER_ENV, "Path")?.map(|v| v.data).unwrap_or_default();
    let system = reg::read(reg::Hive::Machine, reg::SYSTEM_ENV, "Path")?.map(|v| v.data).unwrap_or_default();
    let shell = std::env::var("PATH").unwrap_or_default();
    let expand = |s: &str| reg::expand(s);
    Ok(Status {
        dir: dir.to_path_buf(),
        user: contains(&user, dir, &expand),
        system: contains(&system, dir, &expand),
        this_shell: contains(&shell, dir, &expand),
        user_chars: user.encode_utf16().count(),
        others: other_fidims(&system, &user, dir, &expand, &|d: &Path| d.join("fidim.exe").is_file()),
    })
}

/// Add `dir` to the end of the user PATH. Call `notify_changed` afterwards
/// when this returns `Added`.
#[cfg(windows)]
pub fn add(dir: &Path) -> Result<Change> {
    reg::add_in(reg::Hive::User, reg::USER_ENV, dir)
}

/// Take every entry naming `dir` off the user PATH. Call `notify_changed`
/// afterwards when this returns `Removed`.
#[cfg(windows)]
pub fn remove(dir: &Path) -> Result<Change> {
    reg::remove_in(reg::Hive::User, reg::USER_ENV, dir)
}

/// Tell running programs (Explorer above all) that the environment changed,
/// so terminals they start from now on see the new PATH. False when the
/// broadcast failed or timed out; then only a new sign-in picks it up.
#[cfg(windows)]
pub fn notify_changed() -> bool {
    reg::broadcast_environment_change()
}

#[cfg(not(windows))]
pub fn status(_dir: &Path) -> Result<Status> {
    Err(Error::UserPath("the user PATH is managed here on Windows only".into()))
}

#[cfg(not(windows))]
pub fn add(_dir: &Path) -> Result<Change> {
    Err(Error::UserPath("the user PATH is managed here on Windows only".into()))
}

#[cfg(not(windows))]
pub fn remove(_dir: &Path) -> Result<Change> {
    Err(Error::UserPath("the user PATH is managed here on Windows only".into()))
}

#[cfg(not(windows))]
pub fn notify_changed() -> bool {
    false
}

// ------------------------------------------------------------- registry ----

#[cfg(windows)]
mod reg {
    use std::path::Path;

    use windows::core::PCWSTR;
    use windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_MORE_DATA, LPARAM, WPARAM};
    use windows::Win32::System::Environment::ExpandEnvironmentStringsW;
    use windows::Win32::System::Registry::{
        RegCloseKey, RegCreateKeyExW, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW, HKEY, HKEY_CURRENT_USER,
        HKEY_LOCAL_MACHINE, KEY_QUERY_VALUE, KEY_SET_VALUE, REG_EXPAND_SZ, REG_OPTION_NON_VOLATILE, REG_SAM_FLAGS,
        REG_SZ, REG_VALUE_TYPE,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        SendMessageTimeoutW, HWND_BROADCAST, SMTO_ABORTIFHUNG, WM_SETTINGCHANGE,
    };

    use super::{appended, removed, unusable, Change, MAX_CHARS};
    use crate::{Error, Result};

    pub const USER_ENV: &str = "Environment";
    pub const SYSTEM_ENV: &str = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment";

    #[derive(Clone, Copy)]
    pub enum Hive {
        User,
        Machine,
    }

    impl Hive {
        fn hkey(self) -> HKEY {
            match self {
                Hive::User => HKEY_CURRENT_USER,
                Hive::Machine => HKEY_LOCAL_MACHINE,
            }
        }
        fn name(self) -> &'static str {
            match self {
                Hive::User => "HKCU",
                Hive::Machine => "HKLM",
            }
        }
    }

    /// A string value and whether it is REG_EXPAND_SZ (its `%VARIABLES%`
    /// expand when Windows builds an environment) rather than REG_SZ.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct StrValue {
        pub data: String,
        pub expand: bool,
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(Some(0)).collect()
    }

    /// An open key, closed on drop.
    struct Key(HKEY);

    impl Drop for Key {
        fn drop(&mut self) {
            let _ = unsafe { RegCloseKey(self.0) };
        }
    }

    /// The key, or None when it does not exist.
    fn open(hive: Hive, subkey: &str, access: REG_SAM_FLAGS) -> Result<Option<Key>> {
        let name = wide(subkey);
        let mut h = HKEY::default();
        let rc = unsafe { RegOpenKeyExW(hive.hkey(), PCWSTR(name.as_ptr()), 0, access, &mut h) };
        if rc == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        rc.ok().map_err(|e| Error::UserPath(format!("opening {}\\{subkey}: {e}", hive.name())))?;
        Ok(Some(Key(h)))
    }

    /// The key for writing, created when missing.
    fn create(hive: Hive, subkey: &str) -> Result<Key> {
        let name = wide(subkey);
        let mut h = HKEY::default();
        let rc = unsafe {
            RegCreateKeyExW(
                hive.hkey(),
                PCWSTR(name.as_ptr()),
                0,
                PCWSTR::null(),
                REG_OPTION_NON_VOLATILE,
                KEY_QUERY_VALUE | KEY_SET_VALUE,
                None,
                &mut h,
                None,
            )
        };
        rc.ok().map_err(|e| Error::UserPath(format!("opening {}\\{subkey} for writing: {e}", hive.name())))?;
        Ok(Key(h))
    }

    /// A string value, whole however long it is; None when the key or the
    /// value does not exist. Any other type is an error: this code must
    /// never rewrite a value it could not read back exactly.
    pub fn read(hive: Hive, subkey: &str, value: &str) -> Result<Option<StrValue>> {
        let Some(key) = open(hive, subkey, KEY_QUERY_VALUE)? else { return Ok(None) };
        let name = wide(value);
        let mut buf: Vec<u16> = vec![0; 2048];
        loop {
            let mut ty = REG_VALUE_TYPE::default();
            let mut bytes = (buf.len() * 2) as u32;
            let rc = unsafe {
                RegQueryValueExW(
                    key.0,
                    PCWSTR(name.as_ptr()),
                    None,
                    Some(&mut ty),
                    Some(buf.as_mut_ptr().cast()),
                    Some(&mut bytes),
                )
            };
            if rc == ERROR_MORE_DATA {
                // `bytes` is the size needed now; the value may grow again
                // before the next read, so this loops.
                buf.resize((bytes as usize).div_ceil(2) + 1, 0);
                continue;
            }
            if rc == ERROR_FILE_NOT_FOUND {
                return Ok(None);
            }
            rc.ok().map_err(|e| Error::UserPath(format!("reading {}\\{subkey}\\{value}: {e}", hive.name())))?;
            if ty != REG_SZ && ty != REG_EXPAND_SZ {
                return Err(Error::UserPath(format!(
                    "{}\\{subkey}\\{value} is not a string (registry type {}); leaving it alone",
                    hive.name(),
                    ty.0
                )));
            }
            // The stored data may or may not end in a terminating NUL.
            let mut n = bytes as usize / 2;
            while n > 0 && buf[n - 1] == 0 {
                n -= 1;
            }
            let data = String::from_utf16(&buf[..n]).map_err(|_| {
                Error::UserPath(format!("{}\\{subkey}\\{value} is not valid UTF-16; leaving it alone", hive.name()))
            })?;
            return Ok(Some(StrValue { data, expand: ty == REG_EXPAND_SZ }));
        }
    }

    pub fn write(hive: Hive, subkey: &str, value: &str, v: &StrValue) -> Result<()> {
        let key = create(hive, subkey)?;
        let name = wide(value);
        let data = wide(&v.data);
        // SAFETY: a u16 slice viewed as its bytes, NUL terminator included.
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 2) };
        let ty = if v.expand { REG_EXPAND_SZ } else { REG_SZ };
        let rc = unsafe { RegSetValueExW(key.0, PCWSTR(name.as_ptr()), 0, ty, Some(bytes)) };
        rc.ok().map_err(|e| Error::UserPath(format!("writing {}\\{subkey}\\{value}: {e}", hive.name())))
    }

    /// `%VARIABLES%` expanded against this process's environment; the text
    /// unchanged when expansion fails.
    pub fn expand(s: &str) -> String {
        let src = wide(s);
        let mut buf: Vec<u16> = vec![0; 1024];
        loop {
            let n = unsafe { ExpandEnvironmentStringsW(PCWSTR(src.as_ptr()), Some(&mut buf)) } as usize;
            if n == 0 {
                return s.to_string();
            }
            if n > buf.len() {
                buf.resize(n, 0);
                continue;
            }
            // n counts the terminating NUL.
            return String::from_utf16_lossy(&buf[..n - 1]);
        }
    }

    /// Add `dir` to the `Path` value under `subkey`, keeping its type; a
    /// missing value is created as REG_EXPAND_SZ, the type Windows uses.
    pub fn add_in(hive: Hive, subkey: &str, dir: &Path) -> Result<Change> {
        if let Some(why) = unusable(dir) {
            return Err(Error::UserPath(why));
        }
        let old = read(hive, subkey, "Path")?.unwrap_or(StrValue { data: String::new(), expand: true });
        let Some(data) = appended(&old.data, dir, &expand) else { return Ok(Change::AlreadyThere) };
        let chars = data.encode_utf16().count();
        if chars > MAX_CHARS {
            return Err(Error::UserPath(format!(
                "the user PATH would be {chars} characters, over Windows' limit of {MAX_CHARS}; \
                 remove entries you no longer need first"
            )));
        }
        write(hive, subkey, "Path", &StrValue { data, expand: old.expand })?;
        Ok(Change::Added)
    }

    /// Drop every entry naming `dir` from the `Path` value under `subkey`.
    pub fn remove_in(hive: Hive, subkey: &str, dir: &Path) -> Result<Change> {
        let Some(old) = read(hive, subkey, "Path")? else { return Ok(Change::NotThere) };
        let (data, dropped) = removed(&old.data, dir, &expand);
        if dropped == 0 {
            return Ok(Change::NotThere);
        }
        write(hive, subkey, "Path", &StrValue { data, expand: old.expand })?;
        Ok(Change::Removed(dropped))
    }

    /// WM_SETTINGCHANGE with "Environment" to every top-level window, the
    /// way the System Properties dialog announces an edit. A hung window is
    /// skipped after 5 s instead of blocking.
    pub fn broadcast_environment_change() -> bool {
        let what = wide("Environment");
        let mut result = 0usize;
        let sent = unsafe {
            SendMessageTimeoutW(
                HWND_BROADCAST,
                WM_SETTINGCHANGE,
                WPARAM(0),
                LPARAM(what.as_ptr() as isize),
                SMTO_ABORTIFHUNG,
                5000,
                Some(&mut result),
            )
        };
        sent.0 != 0
    }

    #[cfg(test)]
    mod tests {
        //! Against a scratch key under HKCU\Software that each test deletes
        //! again; the real user PATH is never read or written here.
        use std::path::Path;
        use std::sync::atomic::{AtomicU32, Ordering};

        use windows::core::PCWSTR;
        use windows::Win32::System::Registry::{RegDeleteKeyW, HKEY_CURRENT_USER};

        use super::*;

        struct Scratch(String);

        impl Scratch {
            fn new() -> Self {
                static N: AtomicU32 = AtomicU32::new(0);
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.subsec_nanos())
                    .unwrap_or(0);
                Scratch(format!(
                    r"Software\fidim-test-user-path-{}-{nanos}-{}",
                    std::process::id(),
                    N.fetch_add(1, Ordering::SeqCst)
                ))
            }
            fn get(&self) -> Option<StrValue> {
                read(Hive::User, &self.0, "Path").unwrap()
            }
            fn set(&self, data: &str, expand: bool) {
                write(Hive::User, &self.0, "Path", &StrValue { data: data.into(), expand }).unwrap();
            }
        }

        impl Drop for Scratch {
            fn drop(&mut self) {
                let name = wide(&self.0);
                let _ = unsafe { RegDeleteKeyW(HKEY_CURRENT_USER, PCWSTR(name.as_ptr())) };
            }
        }

        #[test]
        fn missing_key_or_value_reads_as_none() {
            let s = Scratch::new();
            assert_eq!(s.get(), None);
            write(Hive::User, &s.0, "Other", &StrValue { data: "x".into(), expand: false }).unwrap();
            assert_eq!(s.get(), None);
        }

        #[test]
        fn round_trips_long_values_and_keeps_the_type() {
            let s = Scratch::new();
            // Past every limit that bites elsewhere: NSIS (1024/8192), setx
            // (1024), the first read buffer (2048 UTF-16 units) and even the
            // 32,767 a variable may hold. The registry stores it; the read
            // must return it whole.
            let long: String = (0..4000).map(|i| format!(r"C:\dir{i:05};")).collect();
            assert!(long.len() > 40_000);
            s.set(&long, true);
            assert_eq!(s.get(), Some(StrValue { data: long.clone(), expand: true }));
            s.set("%SystemRoot%\\x;C:\\é ü", false);
            assert_eq!(s.get(), Some(StrValue { data: "%SystemRoot%\\x;C:\\é ü".into(), expand: false }));
        }

        #[test]
        fn add_then_remove_restores_the_value_exactly() {
            let s = Scratch::new();
            let before = r"%SystemRoot%\keep;C:\Tools;;D:\x\";
            s.set(before, true);
            let dir = Path::new(r"C:\Users\me\AppData\Local\Llama FIDIM");

            assert_eq!(add_in(Hive::User, &s.0, dir).unwrap(), Change::Added);
            let after = s.get().unwrap();
            assert_eq!(after.data, format!("{before};{}", dir.display()));
            assert!(after.expand, "a REG_EXPAND_SZ PATH stays REG_EXPAND_SZ");
            // Case and a trailing separator do not make a second entry.
            let shouted = Path::new(r"c:\users\ME\appdata\local\llama fidim\");
            assert_eq!(add_in(Hive::User, &s.0, shouted).unwrap(), Change::AlreadyThere);

            assert_eq!(remove_in(Hive::User, &s.0, dir).unwrap(), Change::Removed(1));
            assert_eq!(s.get().unwrap(), StrValue { data: before.into(), expand: true });
            assert_eq!(remove_in(Hive::User, &s.0, dir).unwrap(), Change::NotThere);
        }

        #[test]
        fn a_missing_path_is_created_as_expandable() {
            let s = Scratch::new();
            let dir = Path::new(r"C:\Llama FIDIM");
            assert_eq!(remove_in(Hive::User, &s.0, dir).unwrap(), Change::NotThere);
            assert_eq!(add_in(Hive::User, &s.0, dir).unwrap(), Change::Added);
            assert_eq!(s.get().unwrap(), StrValue { data: r"C:\Llama FIDIM".into(), expand: true });
        }

        #[test]
        fn a_reg_sz_path_stays_reg_sz() {
            let s = Scratch::new();
            s.set(r"C:\a", false);
            add_in(Hive::User, &s.0, Path::new(r"C:\b")).unwrap();
            assert_eq!(s.get().unwrap(), StrValue { data: r"C:\a;C:\b".into(), expand: false });
        }

        #[test]
        fn refuses_to_grow_past_the_variable_limit() {
            let s = Scratch::new();
            let full = "x".repeat(MAX_CHARS - 5);
            s.set(&full, true);
            let e = add_in(Hive::User, &s.0, Path::new(r"C:\Llama FIDIM")).unwrap_err().to_string();
            assert!(e.contains("32767"), "{e}");
            assert_eq!(s.get().unwrap().data, full, "left untouched");
        }

        #[test]
        fn refuses_relative_or_unquotable_folders() {
            let s = Scratch::new();
            assert!(add_in(Hive::User, &s.0, Path::new(r"bin")).is_err());
            assert!(add_in(Hive::User, &s.0, Path::new(r"C:\a;b")).is_err());
            assert_eq!(s.get(), None, "nothing written");
        }

        #[test]
        fn expands_variables_from_this_process() {
            let windir = std::env::var("SystemRoot").unwrap();
            assert_eq!(expand(r"%SystemRoot%\System32"), format!(r"{windir}\System32"));
            assert_eq!(expand("no variables"), "no variables");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_expand(s: &str) -> String {
        s.replace("%LOCALAPPDATA%", r"C:\Users\me\AppData\Local").replace("%SystemRoot%", r"C:\Windows")
    }

    const DIR: &str = r"C:\Users\me\AppData\Local\Llama FIDIM";

    #[test]
    fn contains_ignores_case_quotes_slashes_and_trailing_separators() {
        let dir = Path::new(DIR);
        for v in [
            r"C:\a;C:\Users\me\AppData\Local\Llama FIDIM",
            r"c:\users\ME\appdata\local\llama fidim\",
            r#""C:\Users\me\AppData\Local\Llama FIDIM";C:\a"#,
            "C:/Users/me/AppData/Local/Llama FIDIM/",
            r"C:\a;%LOCALAPPDATA%\Llama FIDIM;C:\b",
        ] {
            assert!(contains(v, dir, &fake_expand), "{v}");
        }
        for v in ["", ";;", r"C:\Users\me\AppData\Local\Llama FIDIM2", r"C:\Users\me\AppData\Local"] {
            assert!(!contains(v, dir, &fake_expand), "{v}");
        }
    }

    #[test]
    fn appended_adds_one_entry_at_the_end() {
        let dir = Path::new(DIR);
        assert_eq!(appended("", dir, &fake_expand).as_deref(), Some(DIR));
        assert_eq!(appended(r"C:\a", dir, &fake_expand), Some(format!(r"C:\a;{DIR}")));
        assert_eq!(appended(r"C:\a;", dir, &fake_expand), Some(format!(r"C:\a;{DIR}")));
        assert_eq!(appended(r"%LOCALAPPDATA%\Llama FIDIM", dir, &fake_expand), None);
    }

    #[test]
    fn removed_drops_every_copy_and_keeps_the_rest_verbatim() {
        let dir = Path::new(DIR);
        let v = format!(r"%SystemRoot%\x;{DIR};;C:\keep\;%LOCALAPPDATA%\Llama FIDIM\;C:\z");
        let (out, n) = removed(&v, dir, &fake_expand);
        assert_eq!(n, 2);
        assert_eq!(out, r"%SystemRoot%\x;;C:\keep\;C:\z");
        let (out, n) = removed(r"C:\a;C:\b;", dir, &fake_expand);
        assert_eq!((out.as_str(), n), (r"C:\a;C:\b;", 0));
        assert_eq!(removed(DIR, dir, &fake_expand), (String::new(), 1));
    }

    #[test]
    fn other_fidims_in_search_order() {
        let dir = Path::new(DIR);
        let has = |d: &Path| {
            let d = d.to_string_lossy().to_lowercase();
            d.contains("programs\\llamafidim") || d.contains("dev\\target")
        };
        let system = r"C:\Windows;C:\dev\target\release";
        let user = format!(r"C:\Users\me\AppData\Local\Programs\LlamaFIDIM;{DIR};C:\dev\target\release\;C:\later\dev\target\x");
        let found = other_fidims(system, &user, dir, &fake_expand, &has);
        assert_eq!(
            found,
            vec![
                OtherFidim { dir: r"C:\dev\target\release".into(), from: "system", first: true },
                OtherFidim { dir: r"C:\Users\me\AppData\Local\Programs\LlamaFIDIM".into(), from: "user", first: true },
                OtherFidim { dir: r"C:\later\dev\target\x".into(), from: "user", first: false },
            ]
        );
        // Not on PATH at all: every other copy is the one a terminal finds.
        let found = other_fidims("", r"%LOCALAPPDATA%\Programs\LlamaFIDIM", dir, &fake_expand, &has);
        assert_eq!(found.len(), 1);
        assert!(found[0].first);
        assert_eq!(found[0].dir, PathBuf::from(r"C:\Users\me\AppData\Local\Programs\LlamaFIDIM"));
    }

    #[test]
    fn unusable_folders() {
        assert!(unusable(Path::new(DIR)).is_none());
        assert!(unusable(Path::new("relative")).is_some());
        assert!(unusable(Path::new(r"C:\a;b")).is_some());
        assert!(unusable(Path::new(r#"C:\a"b"#)).is_some());
    }

    #[test]
    fn change_reports_whether_the_value_was_rewritten() {
        assert!(Change::Added.changed());
        assert!(Change::Removed(2).changed());
        assert!(!Change::AlreadyThere.changed());
        assert!(!Change::NotThere.changed());
    }
}
