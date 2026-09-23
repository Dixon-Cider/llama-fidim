//! Windows version resources for fidim.exe and fidim-dg.exe: the product,
//! the publisher, a description per executable, the version and the app
//! icon. Explorer's Properties dialog, Task Manager and the Authenticode
//! prompts show these; without them a signed file names nothing but its
//! file name.
//!
//! Each executable gets its own resource, linked into that binary only
//! (`embed_resource::compile_for`): a single resource for the package would
//! be linked into both, and one executable cannot hold two version
//! resources.

use std::path::{Path, PathBuf};

const PRODUCT: &str = "Llama FIDIM";
const COMPANY: &str = "Dixon-Cider";
const COPYRIGHT: &str = "Copyright (c) 2026 Dixon-Cider";

/// (binary, FileDescription)
const BINS: [(&str, &str); 2] = [
    ("fidim", "Llama FIDIM command line"),
    ("fidim-dg", "Llama FIDIM DiffusionGemma server"),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    // The GUI's icon; a source tree without the ui folder still builds.
    let icon = manifest
        .ancestors()
        .nth(2)
        .map(|root| root.join("ui").join("src-tauri").join("icons").join("icon.ico"))
        .filter(|p| p.is_file());
    if let Some(icon) = &icon {
        println!("cargo:rerun-if-changed={}", icon.display());
    }

    let version = std::env::var("CARGO_PKG_VERSION").expect("cargo sets CARGO_PKG_VERSION");
    let part = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<u16>().ok()).unwrap_or(0);
    let numbers = [part("CARGO_PKG_VERSION_MAJOR"), part("CARGO_PKG_VERSION_MINOR"), part("CARGO_PKG_VERSION_PATCH"), 0];

    for (bin, description) in BINS {
        let rc = out.join(format!("{bin}.rc"));
        std::fs::write(&rc, version_rc(bin, description, &version, numbers, icon.as_deref()))
            .unwrap_or_else(|e| panic!("writing {}: {e}", rc.display()));
        match embed_resource::compile_for(&rc, [bin], embed_resource::NONE) {
            embed_resource::CompilationResult::Ok | embed_resource::CompilationResult::NotWindows => {}
            // No resource compiler (rc.exe comes with the Windows SDK): the
            // binaries work, they just carry no version details.
            embed_resource::CompilationResult::NotAttempted(why) => {
                println!("cargo:warning={bin}.exe gets no version resource: {why}");
            }
            embed_resource::CompilationResult::Failed(why) => panic!("compiling {}: {why}", rc.display()),
        }
    }
}

/// An .rc string literal: quotes doubled, backslashes escaped.
fn rc_str(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\"\""))
}

fn version_rc(bin: &str, description: &str, version: &str, n: [u16; 4], icon: Option<&Path>) -> String {
    let nums = format!("{},{},{},{}", n[0], n[1], n[2], n[3]);
    let strings = [
        ("CompanyName", COMPANY.to_string()),
        ("FileDescription", description.to_string()),
        ("FileVersion", version.to_string()),
        ("InternalName", bin.to_string()),
        ("LegalCopyright", COPYRIGHT.to_string()),
        ("OriginalFilename", format!("{bin}.exe")),
        ("ProductName", PRODUCT.to_string()),
        ("ProductVersion", version.to_string()),
    ];
    let mut rc = String::from("#pragma code_page(65001)\n");
    rc.push_str(&format!(
        "1 VERSIONINFO\nFILEVERSION {nums}\nPRODUCTVERSION {nums}\nFILEFLAGSMASK 0x3F\nFILEFLAGS 0x0\n\
         FILEOS 0x40004\nFILETYPE 0x1\nFILESUBTYPE 0x0\nBEGIN\n  BLOCK \"StringFileInfo\"\n  BEGIN\n    \
         BLOCK \"000004B0\"\n    BEGIN\n"
    ));
    for (k, v) in strings {
        rc.push_str(&format!("      VALUE \"{k}\", {}\n", rc_str(&v)));
    }
    // Language neutral, Unicode: the same block the GUI's resource uses.
    rc.push_str("    END\n  END\n  BLOCK \"VarFileInfo\"\n  BEGIN\n    VALUE \"Translation\", 0x0, 1200\n  END\nEND\n");
    if let Some(icon) = icon {
        rc.push_str(&format!("1 ICON {}\n", rc_str(&icon.display().to_string())));
    }
    rc
}
