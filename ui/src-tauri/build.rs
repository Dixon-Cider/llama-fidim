fn main() {
    // The front end is embedded at compile time from `frontendDist` (../dist).
    // Without this, a rebuilt dist does not recompile the crate and the app
    // keeps shipping the previous assets (seen 2026-09-21: a swept UI that
    // never reached the binary).
    println!("cargo:rerun-if-changed=../dist");
    tauri_build::build()
}
