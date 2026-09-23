//! The toolchain a source build needs, and a doctor that says what is
//! missing before a build spends minutes finding out.
//!
//! A llama.cpp HIP build on Windows needs Visual Studio's C++ tools (found
//! with vswhere; `vcvars64.bat` sets up the MSVC headers and libraries),
//! git, CMake and Ninja (on PATH, or the copies Visual Studio bundles), and
//! the HIP SDK's clang. The doctor also compiles one tiny `.hip` file that
//! includes `<cmath>` for the card's GPU target: HIP SDK clang up to 21
//! declares the math comparisons `__device__` while MSVC 14.40 and newer
//! declare them `constexpr`, which clang treats as `__host__ __device__`,
//! so every HIP file fails with "cannot overload" (llama.cpp#22570; fixed
//! in LLVM #201563). That takes seconds to see here and minutes into a
//! build.
//!
//! Nothing here changes the system: a blocker comes back with the fix to
//! apply by hand.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::config::Config;
use crate::preflight::Outcome;

/// One Visual Studio installation with the C++ tools.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VsInstall {
    pub path: PathBuf,
    pub version: String,
    pub name: String,
    /// `VC\Tools\MSVC\<version>` directories, oldest first.
    pub msvc_toolsets: Vec<String>,
}

impl VsInstall {
    pub fn vcvars64(&self) -> PathBuf {
        self.path.join("VC").join("Auxiliary").join("Build").join("vcvars64.bat")
    }
    fn bundled_cmake(&self) -> PathBuf {
        self.path.join(r"Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")
    }
    fn bundled_ninja(&self) -> PathBuf {
        self.path.join(r"Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe")
    }
}

/// What was found. `build_from_ref` hands the chosen parts to the build
/// script, so the doctor and the build use the same tools.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Toolchain {
    pub vswhere: Option<PathBuf>,
    /// Installations with the C++ tools, newest first.
    pub vs: Vec<VsInstall>,
    /// The one builds use: `FIDIM_VS`, else the newest with vcvars64.bat.
    pub chosen_vs: Option<VsInstall>,
    /// `FIDIM_VCVARS_VER`: an MSVC toolset to select (`-vcvars_ver=`),
    /// e.g. 14.39 to sidestep the `<cmath>` clash.
    pub vcvars_ver: Option<String>,
    pub git: Option<PathBuf>,
    pub git_version: Option<String>,
    pub cmake: Option<PathBuf>,
    pub ninja: Option<PathBuf>,
    /// HIP SDK root (`HIP_PATH`), e.g. `C:\Program Files\AMD\ROCm\7.1`.
    pub rocm: Option<PathBuf>,
    /// Its clang++ (`bin\` in the HIP SDK, `lib\llvm\bin\` in TheRock's layout).
    pub clangxx: Option<PathBuf>,
}

impl Toolchain {
    /// Environment for `scripts/build-from-ref.bat`.
    pub fn apply_env(&self, cmd: &mut Command) {
        if let Some(vs) = &self.chosen_vs {
            cmd.env("FIDIM_VS", &vs.path);
        }
        if let Some(v) = &self.vcvars_ver {
            cmd.env("FIDIM_VCVARS_VER", v);
        }
        if let Some(r) = &self.rocm {
            cmd.env("FIDIM_ROCM", r);
        }
        if let Some(c) = self.clangxx.as_ref().and_then(|c| c.parent()) {
            cmd.env("FIDIM_CLANG_DIR", c);
        }
        if let Some(c) = &self.cmake {
            cmd.env("FIDIM_CMAKE", c);
        }
        if let Some(n) = self.ninja.as_ref().and_then(|n| n.parent()) {
            cmd.env("FIDIM_NINJA_DIR", n);
        }
    }
}

/// A doctor finding: `Pass`, `Note`, `Warn` or `Block`, with the fix for
/// anything that is not a pass.
#[derive(Debug, Clone, Serialize)]
pub struct Finding {
    pub id: &'static str,
    pub title: &'static str,
    pub outcome: Outcome,
    /// What was found, for a passing check (`git version 2.53.0`).
    pub detail: String,
    pub fix: Option<String>,
}

impl Finding {
    fn new(id: &'static str, title: &'static str, outcome: Outcome, fix: Option<String>) -> Self {
        Finding { id, title, outcome, detail: String::new(), fix }
    }
    fn pass(id: &'static str, title: &'static str, detail: impl Into<String>) -> Self {
        Finding { id, title, outcome: Outcome::Pass, detail: detail.into(), fix: None }
    }
    pub fn blocks(&self) -> bool {
        matches!(self.outcome, Outcome::Block(_))
    }
    /// `ok    git: git version 2.53.0`.
    pub fn summary(&self) -> String {
        let (tag, msg) = match &self.outcome {
            Outcome::Pass => ("ok", self.detail.clone()),
            Outcome::Note(m) => ("note", m.clone()),
            Outcome::Warn(m) => ("WARN", m.clone()),
            Outcome::Block(m) => ("BLOCK", m.clone()),
        };
        if msg.is_empty() { format!("{tag:<5} {}", self.title) } else { format!("{tag:<5} {}: {msg}", self.title) }
    }
    /// The summary, then the fix on the next line when there is one.
    pub fn with_fix(&self) -> String {
        match &self.fix {
            Some(f) => format!("{}\n      fix: {f}", self.summary()),
            None => self.summary(),
        }
    }
}

fn hidden(cmd: &mut Command) -> &mut Command {
    crate::launch::hide_console(cmd);
    cmd.stdin(Stdio::null())
}

/// `name.exe` in a PATH directory.
fn on_path(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(name)).find(|p| p.is_file())
}

fn version_key(v: &str) -> Vec<u64> {
    v.split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).map(|s| s.parse().unwrap_or(0)).collect()
}

/// `vswhere -format json` output -> installations, newest first (toolsets
/// are filled in by the caller, which can look at the disk).
pub fn parse_vswhere(json: &str) -> Vec<VsInstall> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else { return vec![] };
    let mut out: Vec<VsInstall> = v
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|i| {
                    Some(VsInstall {
                        path: PathBuf::from(i["installationPath"].as_str()?),
                        version: i["installationVersion"].as_str().unwrap_or("").to_string(),
                        name: i["displayName"].as_str().unwrap_or("Visual Studio").to_string(),
                        msvc_toolsets: vec![],
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort_by_key(|v| std::cmp::Reverse(version_key(&v.version)));
    out
}

fn msvc_toolsets(vs: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(vs.join("VC").join("Tools").join("MSVC"))
        .map(|rd| rd.flatten().filter(|e| e.path().is_dir()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    v.sort_by_key(|a| version_key(a));
    v
}

/// GPU targets for CMake's `GPU_TARGETS`, comma-separated (`;` would split
/// the batch argument): `gfx1201`, `gfx1100,gfx1201`.
pub fn normalize_gpu_targets(s: &str) -> crate::Result<String> {
    let mut out: Vec<String> = Vec::new();
    for t in s.split([',', ';', ' ']).map(str::trim).filter(|t| !t.is_empty()) {
        let t = t.to_ascii_lowercase();
        let ok = t.strip_prefix("gfx").is_some_and(|rest| {
            (3..=5).contains(&rest.len()) && rest.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        });
        if !ok {
            return Err(crate::Error::Update(format!("`{t}` is not a GPU target (gfx1201, gfx1100, gfx90a...)")));
        }
        if !out.contains(&t) {
            out.push(t);
        }
    }
    if out.is_empty() {
        return Err(crate::Error::Update("no GPU target given (e.g. gfx1201)".into()));
    }
    Ok(out.join(","))
}

/// The HIP SDK root: `FIDIM_ROCM`, `HIP_PATH`, the configured runtime's
/// parent, then the newest under `%ProgramFiles%\AMD\ROCm` — the first that
/// has a clang++.
fn find_rocm(cfg: &Config) -> (Option<PathBuf>, Option<PathBuf>) {
    let clang_in = |root: &Path| -> Option<PathBuf> {
        [root.join("bin").join("clang++.exe"), root.join("lib").join("llvm").join("bin").join("clang++.exe")]
            .into_iter()
            .find(|p| p.is_file())
    };
    let mut candidates: Vec<PathBuf> = Vec::new();
    for var in ["FIDIM_ROCM", "HIP_PATH"] {
        if let Some(v) = std::env::var_os(var).filter(|v| !v.is_empty()) {
            candidates.push(PathBuf::from(v));
        }
    }
    if let Some(bin) = &cfg.rocm_bin {
        if let Some(parent) = bin.parent() {
            candidates.push(parent.to_path_buf());
        }
    }
    let pf = std::env::var_os("ProgramFiles").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Program Files"));
    let mut sdks: Vec<PathBuf> = std::fs::read_dir(pf.join("AMD").join("ROCm"))
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    sdks.sort_by_key(|p| std::cmp::Reverse(version_key(&p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())));
    candidates.extend(sdks);
    for c in candidates {
        // HIP_PATH ends in a backslash; the components are what matter.
        let c: PathBuf = c.components().collect();
        if let Some(clang) = clang_in(&c) {
            return (Some(c), Some(clang));
        }
    }
    (None, None)
}

/// Look for everything a source build needs. Runs vswhere and
/// `git --version`; compiles nothing.
pub fn detect(cfg: &Config) -> Toolchain {
    let mut tc = Toolchain::default();
    let pf86 = std::env::var_os("ProgramFiles(x86)").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Program Files (x86)"));
    let vswhere = pf86.join(r"Microsoft Visual Studio\Installer\vswhere.exe");
    if vswhere.is_file() {
        let mut cmd = Command::new(&vswhere);
        cmd.args(["-all", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json", "-utf8"]);
        if let Ok(out) = hidden(&mut cmd).output() {
            tc.vs = parse_vswhere(&String::from_utf8_lossy(&out.stdout));
        }
        tc.vswhere = Some(vswhere);
    }
    for vs in &mut tc.vs {
        vs.msvc_toolsets = msvc_toolsets(&vs.path);
    }
    let forced = std::env::var_os("FIDIM_VS").filter(|v| !v.is_empty()).map(PathBuf::from);
    tc.chosen_vs = match forced {
        Some(p) => Some(tc.vs.iter().find(|v| v.path == p).cloned().unwrap_or_else(|| VsInstall {
            msvc_toolsets: msvc_toolsets(&p),
            path: p,
            version: String::new(),
            name: "FIDIM_VS".into(),
        })),
        None => tc.vs.iter().find(|v| v.vcvars64().is_file()).cloned(),
    };
    tc.vcvars_ver = std::env::var("FIDIM_VCVARS_VER").ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    tc.git = on_path("git.exe");
    if let Some(git) = &tc.git {
        let mut cmd = Command::new(git);
        cmd.arg("--version");
        if let Ok(out) = hidden(&mut cmd).output() {
            tc.git_version = Some(String::from_utf8_lossy(&out.stdout).trim().to_string()).filter(|s| !s.is_empty());
        }
    }
    // Visual Studio's own CMake and Ninja are the tested pair; PATH is the
    // fallback.
    tc.cmake = tc.chosen_vs.as_ref().map(|v| v.bundled_cmake()).filter(|p| p.is_file()).or_else(|| on_path("cmake.exe"));
    tc.ninja = tc.chosen_vs.as_ref().map(|v| v.bundled_ninja()).filter(|p| p.is_file()).or_else(|| on_path("ninja.exe"));
    (tc.rocm, tc.clangxx) = find_rocm(cfg);
    tc
}

/// The doctor: what `detect` found, judged, plus the `<cmath>` test
/// compile for `gfx`. About five seconds (vcvars64.bat is most of it).
pub fn doctor(cfg: &Config, gfx: &str) -> Vec<Finding> {
    doctor_with(&detect(cfg), gfx)
}

const VS_FIX: &str = "install Visual Studio Build Tools (2022 or 2026) with the \"Desktop development with C++\" workload";

pub fn doctor_with(tc: &Toolchain, gfx: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    let vs_ok = tc.chosen_vs.as_ref().is_some_and(|v| v.vcvars64().is_file());
    out.push(match &tc.chosen_vs {
        Some(vs) if vs_ok => {
            let others: Vec<String> = tc.vs.iter().filter(|v| v.path != vs.path).map(|v| v.name.clone()).collect();
            Finding::pass(
                "vs",
                "Visual Studio C++ tools",
                format!(
                    "{} {} (MSVC {}){}{}",
                    vs.name,
                    vs.version,
                    if vs.msvc_toolsets.is_empty() { "?".into() } else { vs.msvc_toolsets.join(", ") },
                    tc.vcvars_ver.as_ref().map(|v| format!(", toolset {v} selected")).unwrap_or_default(),
                    if others.is_empty() { String::new() } else { format!("; also installed: {}", others.join(", ")) }
                ),
            )
        }
        Some(vs) => Finding::new(
            "vs",
            "Visual Studio C++ tools",
            Outcome::Block(format!("{} has no VC\\Auxiliary\\Build\\vcvars64.bat", vs.path.display())),
            Some(VS_FIX.into()),
        ),
        None => Finding::new(
            "vs",
            "Visual Studio C++ tools",
            Outcome::Block(if tc.vswhere.is_none() {
                "no Visual Studio installer (vswhere.exe) found".into()
            } else {
                "no Visual Studio installation has the C++ build tools".into()
            }),
            Some(VS_FIX.into()),
        ),
    });
    out.push(match (&tc.git, &tc.git_version) {
        (Some(_), Some(v)) => Finding::pass("git", "git", v.clone()),
        (Some(p), None) => Finding::new(
            "git",
            "git",
            Outcome::Block(format!("{} does not run", p.display())),
            Some("reinstall Git for Windows (git-scm.com)".into()),
        ),
        (None, _) => Finding::new(
            "git",
            "git",
            Outcome::Block("git is not on PATH".into()),
            Some("install Git for Windows (git-scm.com) and keep \"git on PATH\" selected".into()),
        ),
    });
    let cmake_fix = "add the \"C++ CMake tools for Windows\" component in the Visual Studio Installer (it bundles CMake and Ninja), or install CMake and Ninja on PATH";
    out.push(match &tc.cmake {
        Some(p) => Finding::pass("cmake", "CMake", p.display().to_string()),
        None => Finding::new("cmake", "CMake", Outcome::Block("not found".into()), Some(cmake_fix.into())),
    });
    out.push(match &tc.ninja {
        Some(p) => Finding::pass("ninja", "Ninja", p.display().to_string()),
        None => Finding::new("ninja", "Ninja", Outcome::Block("not found".into()), Some(cmake_fix.into())),
    });
    out.push(match (&tc.rocm, &tc.clangxx) {
        (Some(r), Some(c)) => Finding::pass("hip-sdk", "HIP SDK clang", format!("{} ({})", c.display(), r.display())),
        _ => Finding::new(
            "hip-sdk",
            "HIP SDK clang",
            Outcome::Block("no HIP SDK with clang++ found (HIP_PATH is unset or points nowhere)".into()),
            Some("install AMD's HIP SDK for Windows (it sets HIP_PATH), or point FIDIM_ROCM at one".into()),
        ),
    });
    let gfx_ok = normalize_gpu_targets(gfx).is_ok_and(|g| !g.contains(','));
    if !gfx_ok {
        out.push(Finding::new(
            "gpu-target",
            "GPU target",
            Outcome::Block(format!("`{gfx}` is not one GPU target")),
            Some("pass the card's target, e.g. gfx1201 (Radeon AI PRO R9700, RX 9070), gfx1100 (RX 7900)".into()),
        ));
    }
    if let (true, true, Some(vs), Some(clangxx), Some(rocm)) = (vs_ok, gfx_ok, &tc.chosen_vs, &tc.clangxx, &tc.rocm) {
        out.push(cmath_probe(vs, tc.vcvars_ver.as_deref(), clangxx, rocm, gfx));
    }
    out
}

/// What a failed test compile means.
#[derive(Debug, Clone, PartialEq)]
pub enum CompileFailure {
    /// llama.cpp#22570: MSVC's `constexpr` comparisons vs HIP clang's
    /// `__device__` declarations.
    CmathClash { function: String },
    Other(String),
}

/// Classify the test compile's output.
pub fn classify_compile_output(text: &str) -> CompileFailure {
    let re = regex::Regex::new(r"__device__ function '(\w+)' cannot overload __host__ __device__ function").unwrap();
    if let Some(c) = re.captures(text) {
        return CompileFailure::CmathClash { function: c[1].to_string() };
    }
    let errors: Vec<&str> = text.lines().filter(|l| l.contains("error")).take(5).collect();
    CompileFailure::Other(if errors.is_empty() { text.lines().take(5).collect::<Vec<_>>().join("\n") } else { errors.join("\n") })
}

/// The HIP SDK's runtime wrapper already includes the forward declarations
/// before `<cmath>` (LLVM #201563, or the same edit made by hand).
pub fn wrapper_has_cmath_fix(wrapper: &str) -> bool {
    let fwd = wrapper.find("#include <__clang_cuda_math_forward_declares.h>");
    let cmath = wrapper.lines().position(|l| l.trim_start().starts_with("#include <cmath>"));
    match (fwd, cmath) {
        (Some(f), Some(line)) => {
            let fwd_line = wrapper[..f].lines().count().saturating_sub(1);
            fwd_line < line
        }
        _ => false,
    }
}

fn wrapper_path(rocm: &Path) -> Option<PathBuf> {
    let clang_root = rocm.join("lib").join("clang");
    let mut vers: Vec<PathBuf> = std::fs::read_dir(&clang_root).ok()?.flatten().map(|e| e.path()).collect();
    vers.sort_by_key(|p| std::cmp::Reverse(version_key(&p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default())));
    vers.into_iter().map(|v| v.join("include").join("__clang_hip_runtime_wrapper.h")).find(|p| p.is_file())
}

/// The environment `vcvars64.bat` sets up, as `NAME=value` lines from `set`.
pub fn parse_env_dump(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|l| {
            let (k, v) = l.split_once('=')?;
            (!k.is_empty() && !k.contains(' ')).then(|| (k.to_string(), v.to_string()))
        })
        .collect()
}

const PROBE_HIP: &str = "#include <hip/hip_runtime.h>\n#include <cmath>\n\
__global__ void fidim_probe(float * x) { x[0] = std::isless(x[0], 1.0f) ? std::sqrt(x[0]) : 0.0f; }\n";
/// `vcvars-env.bat <vcvars64.bat> [toolset]`: the environment vcvars64
/// sets up, printed by `set`. The toolset arrives bare and is joined to
/// `-vcvars_ver=` here, because cmd splits arguments at `=`.
const ENV_BAT: &str = concat!(
    "@echo off\r\n",
    "set \"PATH=%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer;%PATH%\"\r\n",
    "if \"%~2\"==\"\" (\r\n",
    "  call \"%~1\" >nul 2>&1\r\n",
    ") else (\r\n",
    "  call \"%~1\" -vcvars_ver=%~2 >nul 2>&1\r\n",
    ")\r\n",
    "if errorlevel 1 exit /b 90\r\n",
    "set\r\n",
);

/// Compile one `<cmath>`-using HIP kernel for `gfx` under the chosen MSVC,
/// in a scratch directory under %TEMP%.
fn cmath_probe(vs: &VsInstall, vcvars_ver: Option<&str>, clangxx: &Path, rocm: &Path, gfx: &str) -> Finding {
    const ID: &str = "hip-cmath";
    const TITLE: &str = "HIP compile with MSVC's <cmath>";
    let dir = std::env::temp_dir().join(format!("fidim-doctor-{}", std::process::id()));
    let result = (|| -> std::result::Result<Finding, String> {
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let bat = dir.join("vcvars-env.bat");
        let src = dir.join("probe.hip");
        std::fs::write(&bat, ENV_BAT).map_err(|e| e.to_string())?;
        std::fs::write(&src, PROBE_HIP).map_err(|e| e.to_string())?;
        let mut env_cmd = Command::new(&bat);
        env_cmd.arg(vs.vcvars64()).arg(vcvars_ver.unwrap_or(""));
        let env_out = hidden(&mut env_cmd).output().map_err(|e| format!("running vcvars64.bat: {e}"))?;
        if !env_out.status.success() {
            return Ok(Finding::new(
                ID,
                TITLE,
                Outcome::Block(format!("{} failed (exit {:?})", vs.vcvars64().display(), env_out.status.code())),
                Some(if vcvars_ver.is_some() {
                    "FIDIM_VCVARS_VER names a toolset that is not installed; install it or unset the variable".into()
                } else {
                    "repair the Visual Studio installation".into()
                }),
            ));
        }
        let env = parse_env_dump(&String::from_utf8_lossy(&env_out.stdout));
        let toolset = env.iter().find(|(k, _)| k.eq_ignore_ascii_case("VCToolsVersion")).map(|(_, v)| v.clone()).unwrap_or_default();
        let mut cc = Command::new(clangxx);
        cc.env_clear()
            .envs(env)
            .env("HIP_PATH", rocm)
            .args(["-x", "hip", &format!("--offload-arch={gfx}"), "-c"])
            .arg(&src)
            .arg("-o")
            .arg(dir.join("probe.o"))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let start = Instant::now();
        let child = hidden(&mut cc).spawn().map_err(|e| format!("{}: {e}", clangxx.display()))?;
        let out = wait_with_timeout(child, Duration::from_secs(90)).map_err(|e| format!("{}: {e}", clangxx.display()))?;
        let ms = start.elapsed().as_millis();
        let text = format!("{}{}", String::from_utf8_lossy(&out.stderr), String::from_utf8_lossy(&out.stdout));
        let wrapper = wrapper_path(rocm);
        if out.status.success() {
            let patched = wrapper
                .as_ref()
                .and_then(|w| std::fs::read_to_string(w).ok())
                .is_some_and(|t| wrapper_has_cmath_fix(&t));
            return Ok(Finding::pass(
                ID,
                TITLE,
                format!(
                    "compiled for {gfx} with MSVC {toolset} in {ms} ms{}",
                    if patched { "; this HIP SDK has the <cmath> declaration-order fix" } else { "" }
                ),
            ));
        }
        Ok(match classify_compile_output(&text) {
            CompileFailure::CmathClash { function } => Finding::new(
                ID,
                TITLE,
                Outcome::Block(format!(
                    "HIP clang rejects MSVC {toolset}'s <cmath> (`{function}` cannot overload, llama.cpp#22570): \
                     every HIP file of the build would fail"
                )),
                Some(cmath_fix(wrapper.as_deref(), &toolset)),
            ),
            CompileFailure::Other(first) => Finding::new(
                ID,
                TITLE,
                Outcome::Block(format!("the test compile for {gfx} failed:\n{first}")),
                Some(format!(
                    "check that {} supports --offload-arch={gfx} and that the HIP SDK install is complete",
                    clangxx.display()
                )),
            ),
        })
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result.unwrap_or_else(|e| {
        Finding::new(ID, TITLE, Outcome::Warn(format!("could not run the test compile: {e}")), None)
    })
}

/// How to get past the `<cmath>` clash, in the order to try.
pub fn cmath_fix(wrapper: Option<&Path>, toolset: &str) -> String {
    let wrapper = wrapper
        .map(|w| w.display().to_string())
        .unwrap_or_else(|| r"<HIP SDK>\lib\clang\<N>\include\__clang_hip_runtime_wrapper.h".into());
    format!(
        "one of: (1) as administrator, back up {wrapper} and add the line \
         `#include <__clang_cuda_math_forward_declares.h>` directly above its first `#include <cmath>` \
         (the change LLVM #201563 made upstream), then run the build again; (2) install the MSVC v14.39 toolset \
         (Visual Studio Installer > Individual components > \"MSVC v143 - VS 2022 C++ x64/x86 build tools \
         (v14.39)\") and set FIDIM_VCVARS_VER=14.39 (toolsets from 14.40 on, like {toolset}, clash; 14.29 is too old \
         for llama.cpp); (3) install a HIP SDK whose clang includes LLVM #201563"
    )
}

/// `wait_with_output` with a deadline; the child is killed at the deadline.
fn wait_with_timeout(mut child: std::process::Child, limit: Duration) -> std::io::Result<std::process::Output> {
    use std::io::Read;
    let mut stdout = child.stdout.take();
    let mut stderr = child.stderr.take();
    let out_t = std::thread::spawn(move || {
        let mut b = Vec::new();
        if let Some(s) = stdout.as_mut() {
            let _ = s.read_to_end(&mut b);
        }
        b
    });
    let err_t = std::thread::spawn(move || {
        let mut b = Vec::new();
        if let Some(s) = stderr.as_mut() {
            let _ = s.read_to_end(&mut b);
        }
        b
    });
    let start = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break s;
        }
        if start.elapsed() > limit {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::other(format!("no answer after {} s", limit.as_secs())));
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    Ok(std::process::Output { status, stdout: out_t.join().unwrap_or_default(), stderr: err_t.join().unwrap_or_default() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn vswhere_installs_newest_first() {
        let json = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/toolchain/vswhere.json")).unwrap();
        let v = parse_vswhere(&json);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].name, "Visual Studio Build Tools 2026");
        assert_eq!(v[0].version, "18.9.12128.139");
        assert!(v[0].vcvars64().ends_with(r"VC\Auxiliary\Build\vcvars64.bat"));
        assert!(v[1].version.starts_with("17.14"));
        assert!(parse_vswhere("not json").is_empty());
        assert!(parse_vswhere("[]").is_empty());
    }

    #[test]
    fn classifies_the_cmath_clash() {
        let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/toolchain/hip-cmath-clash.txt")).unwrap();
        assert_eq!(classify_compile_output(&text), CompileFailure::CmathClash { function: "isgreater".into() });
        match classify_compile_output("probe.hip:1:10: fatal error: 'hip/hip_runtime.h' file not found\n1 error generated.") {
            CompileFailure::Other(t) => assert!(t.contains("hip_runtime.h"), "{t}"),
            o => panic!("{o:?}"),
        }
        let fix = cmath_fix(Some(Path::new(r"C:\ROCm\7.1\lib\clang\21\include\__clang_hip_runtime_wrapper.h")), "14.51.36231");
        assert!(fix.contains("__clang_cuda_math_forward_declares.h") && fix.contains("FIDIM_VCVARS_VER=14.39"), "{fix}");
        assert!(fix.contains(r"C:\ROCm\7.1\lib\clang\21\include\__clang_hip_runtime_wrapper.h"));
    }

    #[test]
    fn detects_the_wrapper_reorder() {
        let pristine = "#if !defined(__HIPCC_RTC__)\n#include <cmath>\n#include <cstdlib>\n#endif\n#include <__clang_cuda_math_forward_declares.h>\n";
        assert!(!wrapper_has_cmath_fix(pristine), "declares after <cmath>: the clash");
        let fixed = "#if !defined(__HIPCC_RTC__)\n// HIPFIX\n#include <__clang_cuda_math_forward_declares.h>\n#include <cmath>\n#endif\n#include <__clang_cuda_math_forward_declares.h>\n";
        assert!(wrapper_has_cmath_fix(fixed));
        assert!(!wrapper_has_cmath_fix("#include <cstdlib>\n"));
    }

    #[test]
    fn env_dump_and_findings() {
        let env = parse_env_dump("Path=C:\\a;C:\\b\nVCToolsVersion=14.51.36231\nnot a pair\n=C:=C:\\x\nINCLUDE=C:\\inc\n");
        assert_eq!(env.len(), 3);
        assert!(env.iter().any(|(k, v)| k == "VCToolsVersion" && v == "14.51.36231"));
        let f = Finding::new("git", "git", Outcome::Block("git is not on PATH".into()), Some("install git".into()));
        assert!(f.blocks());
        assert_eq!(f.summary(), "BLOCK git: git is not on PATH");
        assert_eq!(f.with_fix(), "BLOCK git: git is not on PATH\n      fix: install git");
        let ok = Finding::pass("cmake", "CMake", r"C:\VS\cmake.exe");
        assert!(!ok.blocks());
        assert_eq!(ok.summary(), r"ok    CMake: C:\VS\cmake.exe");
        assert_eq!(ok.with_fix(), ok.summary());
    }

    /// An empty toolchain blocks on every part, and the GPU target is checked
    /// before anything is compiled.
    #[test]
    fn doctor_on_nothing_blocks_with_fixes() {
        let f = doctor_with(&Toolchain::default(), "gfx1201");
        let ids: Vec<&str> = f.iter().map(|x| x.id).collect();
        assert_eq!(ids, ["vs", "git", "cmake", "ninja", "hip-sdk"]);
        assert!(f.iter().all(|x| x.blocks() && x.fix.is_some()), "{f:?}");
        let f = doctor_with(&Toolchain::default(), "gfx1201;gfx1100");
        assert!(f.iter().any(|x| x.id == "gpu-target" && x.blocks()));
        let f = doctor_with(&Toolchain::default(), "rm -rf");
        assert!(f.iter().any(|x| x.id == "gpu-target" && x.blocks()));
    }
}
