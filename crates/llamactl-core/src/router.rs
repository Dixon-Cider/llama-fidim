//! M20 — router mode: one llama-server on a well-known port serving every
//! member profile behind the OpenAI-compatible API, selected by the `model`
//! field. Backed by llama-server's native router (`--models-preset`, b10819+):
//! the router process loads per-model child instances on demand and forwards
//! requests; llamactl generates the preset INI from its profiles.
//!
//! Instances inherit the router's arguments and environment, so per-model GPU
//! placement is expressed with `device = ROCm0[,ROCm2]` (llama.cpp's own
//! device names) rather than `HIP_VISIBLE_DEVICES`, and process-level knobs
//! (`GPU_MAX_HW_QUEUES` and friends) are set once on the router.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::devices::Device;
use crate::launch::{self, LaunchPlan};
use crate::preflight::ResolvedDevice;
use crate::profile::Profile;
use crate::supervise::{self, http_get, http_post_json, RunState};
use crate::{Error, Result};

/// The run id the router is supervised under (`llamactl stop router`).
pub const ROUTER_ID: &str = "router";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouterMember {
    pub profile_id: String,
    /// Load this model when the router starts (else on first request).
    #[serde(default)]
    pub load_on_startup: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouterConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Build directory; None = the newest scanned build.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<PathBuf>,
    /// `--models-max`: instances kept loaded at once (0 = unlimited). Two
    /// cards, so two is the sane default.
    #[serde(default = "default_models_max")]
    pub models_max: u32,
    /// Load a model on first request (`--models-autoload`).
    #[serde(default = "default_true")]
    pub autoload: bool,
    /// Named ROCm runtime for the router process; None = config default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rocm_runtime: Option<String>,
    #[serde(default)]
    pub members: Vec<RouterMember>,
}

fn default_host() -> String { "127.0.0.1".into() }
fn default_port() -> u16 { 1234 }
fn default_models_max() -> u32 { 2 }
fn default_true() -> bool { true }

impl Default for RouterConfig {
    fn default() -> Self {
        RouterConfig {
            host: default_host(),
            port: default_port(),
            build: None,
            models_max: default_models_max(),
            autoload: true,
            rocm_runtime: None,
            members: Vec::new(),
        }
    }
}

pub fn config_path() -> PathBuf {
    Config::config_dir().join("router.json")
}

pub fn ini_path() -> PathBuf {
    Config::config_dir().join("router.ini")
}

pub fn load_config() -> Result<RouterConfig> {
    let p = config_path();
    if !p.exists() {
        return Ok(RouterConfig::default());
    }
    let text = std::fs::read_to_string(&p).map_err(|e| Error::io(&p, e))?;
    Ok(serde_json::from_str(&text)?)
}

pub fn save_config(rc: &RouterConfig) -> Result<()> {
    let p = config_path();
    if let Some(parent) = p.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    std::fs::write(&p, serde_json::to_string_pretty(rc)?).map_err(|e| Error::io(&p, e))
}

/// The model id clients send: the profile's alias (what they already use
/// against the single-model server), falling back to the profile id.
pub fn model_id(p: &Profile) -> String {
    if p.server.alias.trim().is_empty() { p.id.clone() } else { p.server.alias.clone() }
}

/// Resolve a profile's device keys against an enumeration (exact stable-key
/// match; the launch path's rebinding is not repeated here) and turn its
/// split fractions into the shares `compose` expects.
fn resolve_members(p: &Profile, devices: &[Device]) -> Result<Vec<ResolvedDevice>> {
    let mut out = Vec::new();
    let n = p.devices.len().max(1) as f64;
    let explicit: f64 = p.devices.iter().filter_map(|d| d.split_fraction).sum();
    for d in &p.devices {
        let dev = devices
            .iter()
            .find(|x| x.stable_key == d.key)
            .ok_or_else(|| Error::DeviceKeyUnresolved { key: d.key.clone() })?;
        let fraction = match d.split_fraction {
            Some(f) if explicit > 0.0 => f / explicit,
            _ => 1.0 / n,
        };
        out.push(ResolvedDevice { profile_key: d.key.clone(), device: dev.clone(), fraction, rebound: false });
    }
    Ok(out)
}

/// One preset section: the profile's composed argument list translated into
/// INI keys. Router-controlled keys (host, port, alias) are dropped, the
/// visibility pin becomes `device = ...`, `enable_thinking` becomes the
/// `chat-template-kwargs` argument, and the load policy is appended.
pub fn render_section(p: &Profile, resolved: &[ResolvedDevice], load_on_startup: bool) -> String {
    let plan = launch::compose(p, resolved);
    let mut lines = vec![format!("[{}]", model_id(p))];
    let args = &plan.args;
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let key = a.trim_start_matches('-');
        let has_value = i + 1 < args.len() && !args[i + 1].starts_with('-');
        match key {
            "host" | "port" | "alias" => {
                i += if has_value { 2 } else { 1 };
                continue;
            }
            _ => {}
        }
        if has_value {
            let v = &args[i + 1];
            let v = if v.contains(' ') && !v.starts_with('"') { format!("\"{v}\"") } else { v.clone() };
            lines.push(format!("{key} = {v}"));
            i += 2;
        } else {
            lines.push(format!("{key} = true"));
            i += 1;
        }
    }
    let devs: Vec<String> = resolved.iter().map(|r| format!("{}{}", r.device.backend, r.device.hip_index)).collect();
    if !devs.is_empty() {
        lines.push(format!("device = {}", devs.join(",")));
    }
    if p.chat.enable_thinking == Some(false) {
        lines.push("chat-template-kwargs = {\"enable_thinking\": false}".into());
    }
    lines.push(format!("load-on-startup = {}", if load_on_startup { "true" } else { "false" }));
    lines.join("\n")
}

#[derive(Debug, Clone, Serialize)]
pub struct RenderedIni {
    pub text: String,
    /// Process environment the router itself must carry (union of member
    /// envs minus the per-process visibility pin).
    pub env: BTreeMap<String, String>,
    /// (key, a, b) where two members disagree; the first value wins.
    pub env_conflicts: Vec<(String, String, String)>,
    pub device_keys: Vec<String>,
    pub sections: Vec<String>,
}

pub fn render_ini(rc: &RouterConfig, profiles: &[Profile], devices: &[Device]) -> Result<RenderedIni> {
    let mut text = String::from("version = 1\n\n[*]\njinja = true\nmetrics = true\nslots = true\n");
    let mut env: BTreeMap<String, String> = BTreeMap::new();
    let mut conflicts = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    let mut sections = Vec::new();
    for m in &rc.members {
        let p = profiles
            .iter()
            .find(|p| p.id == m.profile_id)
            .ok_or_else(|| Error::Config(format!("router member `{}` is not a saved profile", m.profile_id)))?;
        let resolved = resolve_members(p, devices)?;
        text.push('\n');
        text.push_str(&render_section(p, &resolved, m.load_on_startup));
        text.push('\n');
        sections.push(model_id(p));
        for (k, v) in &p.env {
            if k == "HIP_VISIBLE_DEVICES" {
                continue;
            }
            match env.get(k) {
                Some(existing) if existing != v => conflicts.push((k.clone(), existing.clone(), v.clone())),
                Some(_) => {}
                None => {
                    env.insert(k.clone(), v.clone());
                }
            }
        }
        for d in &p.devices {
            if !keys.contains(&d.key) {
                keys.push(d.key.clone());
            }
        }
    }
    Ok(RenderedIni { text, env, env_conflicts: conflicts, device_keys: keys, sections })
}

/// The router's own command line.
pub fn router_plan(rc: &RouterConfig, exe: PathBuf, ini: &Path, env: &BTreeMap<String, String>, path_prepend: Option<PathBuf>) -> LaunchPlan {
    let mut args: Vec<String> = vec![
        "--models-preset".into(),
        ini.to_string_lossy().into_owned(),
        "--models-max".into(),
        rc.models_max.to_string(),
    ];
    if !rc.autoload {
        args.push("--no-models-autoload".into());
    }
    args.extend(["--host".into(), rc.host.clone(), "--port".into(), rc.port.to_string()]);
    args.extend(["--jinja".into(), "--metrics".into(), "--slots".into()]);
    LaunchPlan {
        exe,
        args,
        env: env.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        path_prepend,
        visibility_env: String::new(),
    }
}

/// A Profile stand-in so the supervisor can name the run's log and state.
fn synthetic_profile(rc: &RouterConfig, build: &Path) -> Result<Profile> {
    Ok(serde_json::from_value(serde_json::json!({
        "schema": 1, "id": ROUTER_ID, "name": "router",
        "build": { "path": build },
        "model": { "path": "" },
        "devices": [],
        "server": { "port": rc.port, "alias": ROUTER_ID, "host": rc.host },
        "runtime": { "ctx_total": 0 },
    }))?)
}

#[derive(Debug, Clone, Serialize)]
pub struct RouterLaunch {
    pub state: RunState,
    pub ini_path: PathBuf,
    pub replaced: Vec<String>,
    pub env_conflicts: Vec<(String, String, String)>,
    pub sections: Vec<String>,
}

/// Write the INI, take the port over from any llamactl server on it, spawn
/// the router, wait for it to answer. Foreign port holders are an error.
pub fn launch(cfg: &Config, rc: &RouterConfig, profiles: &[Profile], devices: &[Device], build_dir: &Path, ready_timeout: Duration) -> Result<RouterLaunch> {
    let exe = build_dir.join("bin").join("llama-server.exe");
    if !exe.is_file() {
        return Err(Error::BuildBinary { path: exe, detail: "not found".into() });
    }
    if rc.members.is_empty() {
        return Err(Error::Config("router has no members — add profiles first".into()));
    }
    let rendered = render_ini(rc, profiles, devices)?;
    let ini = ini_path();
    std::fs::write(&ini, &rendered.text).map_err(|e| Error::io(&ini, e))?;

    let mut replaced = Vec::new();
    if let Some(h) = launch::port_holder(&rc.host, rc.port, &cfg.runs_dir) {
        if h.profile_id.is_none() {
            return Err(Error::Platform(format!(
                "port {} is in use by {}{} — not a llamactl server",
                rc.port,
                h.process_name.as_deref().unwrap_or("an unknown process"),
                h.pid.map(|p| format!(" (pid {p})")).unwrap_or_default()
            )));
        }
        for run in supervise::reattach(&cfg.runs_dir).into_iter().filter(|r| r.alive && r.state.port == rc.port) {
            supervise::stop(&run.state, &cfg.runs_dir)?;
            replaced.push(run.state.profile_id.clone());
        }
        launch::wait_port_free(&rc.host, rc.port, Duration::from_secs(20))?;
    }

    let path_prepend = crate::runtime::path_prepend(cfg, rc.rocm_runtime.as_deref())?;
    let plan = router_plan(rc, exe, &ini, &rendered.env, path_prepend);
    let profile = synthetic_profile(rc, build_dir)?;
    let state = supervise::spawn(&plan, &profile, &cfg.runs_dir, rendered.device_keys.clone(), vec![], false, 0)?;
    supervise::wait_ready_in(&state, ready_timeout, Some(&cfg.runs_dir))?;
    Ok(RouterLaunch { state, ini_path: ini, replaced, env_conflicts: rendered.env_conflicts, sections: rendered.sections })
}

// ------------------------------------------------------------ router API ----

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterModel {
    pub id: String,
    /// `unloaded`, `loading`, `loaded`, `sleeping`, `downloading`.
    pub status: String,
    pub failed: bool,
    pub exit_code: Option<i64>,
}

pub fn models(host: &str, port: u16) -> Result<Vec<RouterModel>> {
    let (code, body) = http_get(host, port, "/models", Duration::from_secs(5))?;
    if code != 200 {
        return Err(Error::Platform(format!("router /models answered {code}")));
    }
    let v: serde_json::Value = serde_json::from_str(&body)?;
    Ok(v["data"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|m| RouterModel {
                    id: m["id"].as_str().unwrap_or("?").to_string(),
                    status: m["status"]["value"].as_str().unwrap_or("?").to_string(),
                    failed: m["status"]["failed"].as_bool().unwrap_or(false),
                    exit_code: m["status"]["exit_code"].as_i64(),
                })
                .collect()
        })
        .unwrap_or_default())
}

fn post_model(host: &str, port: u16, path: &str, id: &str) -> Result<()> {
    let body = serde_json::json!({ "model": id }).to_string();
    let (code, resp) = http_post_json(host, port, path, &body, Duration::from_secs(600))?;
    if code != 200 {
        return Err(Error::Platform(format!("router {path} answered {code}: {resp}")));
    }
    Ok(())
}

pub fn load_model(host: &str, port: u16, id: &str) -> Result<()> {
    post_model(host, port, "/models/load", id)
}

pub fn unload_model(host: &str, port: u16, id: &str) -> Result<()> {
    post_model(host, port, "/models/unload", id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::Device;

    fn dev(idx: u32, key: &str) -> Device {
        serde_json::from_value(serde_json::json!({
            "stable_key": key, "name": "AMD Radeon AI PRO R9700", "hip_index": idx, "backend": "ROCm",
            "total_mib": 32624, "free_mib": 32472, "integrated": false, "bus_number": 3,
            "driver_version": null, "display": null, "luid_low": 1, "correlation_assumed": false
        }))
        .unwrap()
    }

    fn profile() -> Profile {
        serde_json::from_value(serde_json::json!({
            "schema": 1, "id": "dd", "name": "x",
            "build": { "path": "C:/b" }, "model": { "path": "E:/m.gguf", "mmproj": "E:/mm.gguf" },
            "devices": [{ "key": "pci:x:bus03" }],
            "server": { "port": 1234, "alias": "dd" },
            "runtime": { "ctx_total": 131072, "slots": 2, "kv_type_k": "q4_0", "kv_type_v": "q4_0", "cont_batching": true, "kv_unified": true },
            "sampling": { "temperature": 1.0 },
            "speculative": { "mode": "mtp", "n_max": 3 },
            "chat": { "enable_thinking": false },
            "env": { "GPU_MAX_HW_QUEUES": "1" }
        }))
        .unwrap()
    }

    #[test]
    fn section_translates_args_and_drops_router_owned_keys() {
        let p = profile();
        let resolved = resolve_members(&p, &[dev(0, "pci:x:bus03")]).unwrap();
        let s = render_section(&p, &resolved, true);
        assert!(s.starts_with("[dd]\n"));
        for want in ["m = E:/m.gguf", "mmproj = E:/mm.gguf", "c = 131072", "np = 2", "ctk = q4_0", "cb = true", "kv-unified = true",
                     "temp = 1.0", "spec-type = draft-mtp", "spec-draft-n-max = 3", "device = ROCm0",
                     "chat-template-kwargs = {\"enable_thinking\": false}", "load-on-startup = true"] {
            assert!(s.contains(want), "missing {want:?} in:\n{s}");
        }
        for banned in ["host =", "port =", "alias ="] {
            assert!(!s.contains(banned), "router-owned key leaked: {banned}\n{s}");
        }
    }

    #[test]
    fn ini_collects_env_and_device_keys() {
        let rc = RouterConfig { members: vec![RouterMember { profile_id: "dd".into(), load_on_startup: false }], ..Default::default() };
        let r = render_ini(&rc, &[profile()], &[dev(0, "pci:x:bus03")]).unwrap();
        assert!(r.text.starts_with("version = 1\n"));
        assert_eq!(r.env.get("GPU_MAX_HW_QUEUES").map(String::as_str), Some("1"));
        assert_eq!(r.device_keys, vec!["pci:x:bus03"]);
        assert_eq!(r.sections, vec!["dd"]);
        assert!(r.text.contains("load-on-startup = false"));
    }

    #[test]
    fn unknown_member_is_an_error() {
        let rc = RouterConfig { members: vec![RouterMember { profile_id: "nope".into(), load_on_startup: false }], ..Default::default() };
        assert!(render_ini(&rc, &[profile()], &[]).is_err());
    }

    #[test]
    fn router_plan_args() {
        let rc = RouterConfig { autoload: false, models_max: 3, ..Default::default() };
        let plan = router_plan(&rc, PathBuf::from("x.exe"), Path::new("r.ini"), &BTreeMap::new(), None);
        let s = plan.args.join(" ");
        assert!(s.contains("--models-preset r.ini"));
        assert!(s.contains("--models-max 3"));
        assert!(s.contains("--no-models-autoload"));
        assert!(s.contains("--port 1234"));
    }
}
