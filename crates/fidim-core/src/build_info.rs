//! Which build this is: the release version and, for a build from a git
//! checkout, the commit, its date, how many commits past the release tag it
//! is, and whether the tree had uncommitted changes. build.rs fills these in.

use serde_json::{json, Value};

/// The release version, `X.Y.Z`: the workspace version in Cargo.toml.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// What `--version` prints after the program name, in rustc's style:
/// `0.2.0 (4f2a1c9 2026-09-20)` is the v0.2.0 release commit; `0.2.0+3 (…)`
/// is three commits past it; `0.2.0+? (…)` is a commit whose distance from
/// the v0.2.0 tag is unknown (a clone without tags); `-modified` after the
/// hash marks a build with uncommitted changes. Just `0.2.0` when built
/// without git.
pub const LONG: &str = env!("FIDIM_VERSION_LONG");

/// Short commit hash; empty without git.
pub const COMMIT: &str = env!("FIDIM_GIT_COMMIT");

/// The commit's date, `YYYY-MM-DD`; empty without git.
pub const COMMIT_DATE: &str = env!("FIDIM_GIT_DATE");

/// The tree had uncommitted changes when this was built.
pub const MODIFIED: bool = matches!(env!("FIDIM_GIT_MODIFIED").as_bytes(), b"1");

/// Commits past this version's `vX.Y.Z` tag; `None` when that tag was not
/// found (not cut yet, or built without tags).
pub fn commits_ahead() -> Option<u32> {
    env!("FIDIM_GIT_AHEAD").parse().ok()
}

/// All of it, for the GUI.
pub fn json() -> Value {
    let opt = |s: &'static str| (!s.is_empty()).then_some(s);
    json!({
        "version": VERSION,
        "long": LONG,
        "commit": opt(COMMIT),
        "commit_date": opt(COMMIT_DATE),
        "commits_ahead": commits_ahead(),
        "modified": MODIFIED,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn repo_file(rel: &str) -> String {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
        std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    }

    /// The version lives in Cargo.toml, ui/package.json, tauri.conf.json and
    /// Cargo.lock, and every release has a changelog section; a release
    /// (scripts/release.ps1) updates them together.
    #[test]
    fn every_copy_of_the_version_agrees() {
        for f in ["ui/package.json", "ui/src-tauri/tauri.conf.json"] {
            let v: Value = serde_json::from_str(&repo_file(f)).unwrap();
            assert_eq!(v["version"], VERSION, "{f} disagrees with Cargo.toml");
        }
        let lock = repo_file("Cargo.lock");
        let lines: Vec<&str> = lock.lines().collect();
        for name in ["fidim-core", "fidim-cli", "llama-fidim"] {
            let at = lines
                .iter()
                .position(|l| *l == format!("name = \"{name}\""))
                .unwrap_or_else(|| panic!("{name} is not in Cargo.lock"));
            assert_eq!(lines[at + 1], format!("version = \"{VERSION}\""), "Cargo.lock's {name}: run cargo update --workspace");
        }
        assert!(
            repo_file("CHANGELOG.md").lines().any(|l| l.starts_with(&format!("## [{VERSION}]"))),
            "CHANGELOG.md has no section for {VERSION}"
        );
    }

    #[test]
    fn long_version_names_the_release_and_commit() {
        assert!(LONG.starts_with(VERSION), "{LONG}");
        if !COMMIT.is_empty() {
            assert!(LONG.contains(COMMIT), "{LONG}");
            assert_eq!(LONG.contains("-modified"), MODIFIED, "{LONG}");
            let suffix = match commits_ahead() {
                Some(0) => String::new(),
                Some(n) => format!("+{n}"),
                None => "+?".into(),
            };
            assert!(LONG.starts_with(&format!("{VERSION}{suffix} (")), "{LONG}");
        } else {
            assert_eq!(LONG, VERSION);
        }
        assert_eq!(json()["version"], VERSION);
    }
}
