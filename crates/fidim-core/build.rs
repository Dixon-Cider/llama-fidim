//! Build identity for `fidim_core::build_info`: the commit a build came from,
//! how many commits past its release tag it is, and whether the tree had
//! uncommitted changes, so every binary can say which code it runs. Git is
//! optional; without it the version is the bare release number.
//!
//! A plain `cargo build` re-runs this when HEAD moves or a tag appears, not
//! on every edit, and a target dir shared between two checkouts can keep the
//! other one's values. install.ps1 and the release workflow set
//! FIDIM_BUILD_ID and FIDIM_BUILD_MODIFIED, so the builds people run are
//! exact.

use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    // No optional locks: `git status` must not take the index lock a
    // concurrent `git add` or commit needs.
    let out = Command::new("git").env("GIT_OPTIONAL_LOCKS", "0").args(args).output().ok()?;
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    (out.status.success() && !s.is_empty()).then(|| s.to_string())
}

/// Re-run when this git path changes. Only paths that exist: cargo re-runs a
/// script on every build when a watched path is missing.
fn watch(git_path: &str) {
    if let Some(p) = git(&["rev-parse", "--git-path", git_path]) {
        if Path::new(&p).exists() {
            println!("cargo:rerun-if-changed={p}");
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // "<checkout>@<HEAD>" and "1"/"0" for uncommitted changes, from
    // install.ps1 and the release workflow.
    println!("cargo:rerun-if-env-changed=FIDIM_BUILD_ID");
    println!("cargo:rerun-if-env-changed=FIDIM_BUILD_MODIFIED");

    let version = std::env::var("CARGO_PKG_VERSION").expect("cargo sets CARGO_PKG_VERSION");
    // Only this workspace's own repository: a source archive unpacked inside
    // some other repo must not report that repo's commit.
    let ours = git(&["rev-parse", "--show-prefix"]).is_some_and(|p| p == "crates/fidim-core/");
    let commit = if ours { git(&["rev-parse", "--short=7", "HEAD"]) } else { None };
    let mut date = None;
    let mut ahead = None;
    let mut modified = false;
    if commit.is_some() {
        // A commit, reset or checkout appends to HEAD's reflog even when the
        // branch ref lives in packed-refs (after git gc) and its loose file is
        // written fresh; a release adds a tag.
        watch("HEAD");
        watch("logs/HEAD");
        if let Some(branch) = git(&["symbolic-ref", "-q", "HEAD"]) {
            watch(&branch);
        }
        watch("packed-refs");
        watch("refs/tags");

        date = git(&["log", "-1", "--format=%cd", "--date=short"]);
        // Commits since this version's own tag: `v0.2.0-3-g4f2a1c9` is 3.
        // None when that tag is not an ancestor (not cut yet, or a clone
        // without tags).
        ahead = git(&["describe", "--tags", "--long", "--match", &format!("v{version}")])
            .and_then(|d| d.rsplitn(3, '-').nth(1).and_then(|n| n.parse::<u32>().ok()));
        modified = match std::env::var("FIDIM_BUILD_MODIFIED") {
            Ok(v) => v.trim() == "1",
            Err(_) => git(&["status", "--porcelain", "--untracked-files=no"]).is_some(),
        };
    }

    // rustc's style: `0.2.0 (4f2a1c9 2026-09-20)` is the release commit,
    // `0.2.0+3` three commits past it, `0.2.0+?` a commit whose distance
    // from the release is unknown.
    let mut long = version;
    if commit.is_some() {
        match ahead {
            Some(0) => {}
            Some(n) => long.push_str(&format!("+{n}")),
            None => long.push_str("+?"),
        }
    }
    if let Some(c) = &commit {
        long.push_str(&format!(" ({c}{}", if modified { "-modified" } else { "" }));
        if let Some(d) = &date {
            long.push_str(&format!(" {d}"));
        }
        long.push(')');
    }

    println!("cargo:rustc-env=FIDIM_VERSION_LONG={long}");
    println!("cargo:rustc-env=FIDIM_GIT_COMMIT={}", commit.as_deref().unwrap_or_default());
    println!("cargo:rustc-env=FIDIM_GIT_DATE={}", date.unwrap_or_default());
    println!("cargo:rustc-env=FIDIM_GIT_AHEAD={}", ahead.map(|n| n.to_string()).unwrap_or_default());
    println!("cargo:rustc-env=FIDIM_GIT_MODIFIED={}", if modified { "1" } else { "0" });
}
