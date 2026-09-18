//! Script export (R-12): any profile emits a `.bat` or `.ps1` that
//! reproduces the LaunchPlan exactly, and the export doubles as an audit of
//! what the tool would run. A llama-server profile's script needs no Llama
//! FIDIM; a diffusion profile's runs FIDIM's `fidim-dg.exe` helper, which
//! owns the runner, so the header says so.

use crate::launch::LaunchPlan;
use crate::profile::Profile;

fn standalone_note(profile: &Profile) -> &'static str {
    if profile.engine.is_diffusion() {
        "Needs Llama FIDIM's fidim-dg.exe (the path below), which owns the DiffusionGemma runner."
    } else {
        "Runs standalone."
    }
}

pub fn to_bat(profile: &Profile, plan: &LaunchPlan) -> String {
    let mut s = String::new();
    s.push_str("@echo off\r\n");
    s.push_str(&format!(
        "REM Exported by Llama FIDIM from profile {:?} ({}). {}\r\n",
        profile.id,
        profile.name,
        standalone_note(profile)
    ));
    s.push_str(&format!(
        "REM Devices: {} (HIP_VISIBLE_DEVICES={} pins visibility; indices remap to 0..n in-process).\r\n",
        profile
            .devices
            .iter()
            .map(|d| d.key.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        plan.visibility_env
    ));
    s.push_str("REM Regenerate with: fidim export ");
    s.push_str(&profile.id);
    s.push_str("\r\n\r\n");
    if let Some(rocm) = &plan.path_prepend {
        s.push_str(&format!("set \"PATH={};%PATH%\"\r\n", rocm.display()));
    }
    for (k, v) in &plan.env {
        // cmd: unquoted `set K=rest of line` keeps JSON braces/spaces intact.
        if v.contains('"') {
            s.push_str(&format!("set {k}={v}\r\n"));
        } else {
            s.push_str(&format!("set \"{k}={v}\"\r\n"));
        }
    }
    s.push_str("\r\n");
    let args = plan
        .args
        .iter()
        .map(|a| bat_quote(a))
        .collect::<Vec<_>>()
        .join(" ");
    s.push_str(&format!("\"{}\" {}\r\n", plan.exe.display(), args));
    s
}

pub fn to_ps1(profile: &Profile, plan: &LaunchPlan) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# Exported by Llama FIDIM from profile '{}' ({}). {}\r\n",
        profile.id,
        profile.name,
        standalone_note(profile)
    ));
    s.push_str(&format!(
        "# Devices: {} (HIP_VISIBLE_DEVICES={}).\r\n\r\n",
        profile.devices.iter().map(|d| d.key.as_str()).collect::<Vec<_>>().join(", "),
        plan.visibility_env
    ));
    if let Some(rocm) = &plan.path_prepend {
        s.push_str(&format!("$env:PATH = \"{};\" + $env:PATH\r\n", rocm.display()));
    }
    for (k, v) in &plan.env {
        s.push_str(&format!("$env:{k} = '{}'\r\n", v.replace('\'', "''")));
    }
    s.push_str("\r\n& ");
    s.push_str(&format!("\"{}\"", plan.exe.display()));
    for a in &plan.args {
        s.push_str(" ");
        s.push_str(&ps1_quote(a));
    }
    s.push_str("\r\n");
    s
}

fn bat_quote(a: &str) -> String {
    if a.contains(' ') || a.contains('&') || a.contains('^') {
        format!("\"{a}\"")
    } else {
        a.to_string()
    }
}

fn ps1_quote(a: &str) -> String {
    if a.chars().all(|c| c.is_ascii_alphanumeric() || "-_./:\\=,".contains(c)) {
        a.to_string()
    } else {
        format!("'{}'", a.replace('\'', "''"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflight::ResolvedDevice;

    fn plan_and_profile() -> (Profile, LaunchPlan) {
        let profile: Profile = serde_json::from_value(serde_json::json!({
            "schema": 1, "id": "worker-pool", "name": "Worker pool",
            "build": { "path": "C:/llama.cpp/build-hip-vision" },
            "model": { "path": "E:/models/my model.gguf" },
            "devices": [ { "key": "pci:A:bus08" } ],
            "server": { "port": 9701, "alias": "worker" },
            "runtime": { "ctx_total": 8192 },
            "chat": { "enable_thinking": false },
            "env": { "GPU_MAX_HW_QUEUES": "1" }
        }))
        .unwrap();
        let mut plan = crate::launch::compose(&profile, &resolved_bus08());
        plan.path_prepend = Some("C:/Program Files/AMD/ROCm/7.1/bin".into());
        (profile, plan)
    }

    fn resolved_bus08() -> Vec<ResolvedDevice> {
        vec![ResolvedDevice {
            profile_key: "pci:A:bus08".into(),
            device: crate::devices::Device {
                stable_key: "pci:A:bus08".into(),
                name: "R9700".into(),
                hip_index: 2,
                backend: "ROCm".into(),
                total_mib: 32624,
                free_mib: 32472,
                integrated: false,
                bus_number: Some(8),
                driver_version: None,
                display: None,
                luid_low: None,
                correlation_assumed: false,
            },
            fraction: 1.0,
            rebound: false,
        }]
    }

    #[test]
    fn bat_export_pins_visibility_and_quotes_paths() {
        let (profile, plan) = plan_and_profile();
        let bat = to_bat(&profile, &plan);
        assert!(bat.contains("set \"HIP_VISIBLE_DEVICES=2\""));
        assert!(bat.contains("set \"GPU_MAX_HW_QUEUES=1\""));
        // JSON env var emitted unquoted so cmd keeps the braces literal.
        assert!(bat.contains(r#"set LLAMA_ARG_CHAT_TEMPLATE_KWARGS={"enable_thinking": false}"#));
        // Path with a space is quoted in the command line.
        assert!(bat.contains("\"E:/models/my model.gguf\""));
        assert!(bat.contains("set \"PATH=C:/Program Files/AMD/ROCm/7.1/bin;%PATH%\""));
    }

    #[test]
    fn diffusion_header() {
        // llama-server output, byte for byte as it was before the diffusion
        // engine existed.
        let (profile, plan) = plan_and_profile();
        assert_eq!(
            to_bat(&profile, &plan),
            "@echo off\r\nREM Exported by Llama FIDIM from profile \"worker-pool\" (Worker pool). Runs standalone.\r\nREM Devices: pci:A:bus08 (HIP_VISIBLE_DEVICES=2 pins visibility; indices remap to 0..n in-process).\r\nREM Regenerate with: fidim export worker-pool\r\n\r\nset \"PATH=C:/Program Files/AMD/ROCm/7.1/bin;%PATH%\"\r\nset \"HIP_VISIBLE_DEVICES=2\"\r\nset \"GPU_MAX_HW_QUEUES=1\"\r\nset LLAMA_ARG_CHAT_TEMPLATE_KWARGS={\"enable_thinking\": false}\r\n\r\n\"C:/llama.cpp/build-hip-vision\\bin\\llama-server.exe\" -m \"E:/models/my model.gguf\" -ngl 99 -c 8192 -np 1 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -cb --jinja --metrics --slots --alias worker --host 127.0.0.1 --port 9701\r\n"
        );
        assert_eq!(
            to_ps1(&profile, &plan),
            "# Exported by Llama FIDIM from profile \'worker-pool\' (Worker pool). Runs standalone.\r\n# Devices: pci:A:bus08 (HIP_VISIBLE_DEVICES=2).\r\n\r\n$env:PATH = \"C:/Program Files/AMD/ROCm/7.1/bin;\" + $env:PATH\r\n$env:HIP_VISIBLE_DEVICES = \'2\'\r\n$env:GPU_MAX_HW_QUEUES = \'1\'\r\n$env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = \'{\"enable_thinking\": false}\'\r\n\r\n& \"C:/llama.cpp/build-hip-vision\\bin\\llama-server.exe\" -m \'E:/models/my model.gguf\' -ngl 99 -c 8192 -np 1 -fa on -ctk f16 -ctv f16 -b 2048 -ub 512 -cb --jinja --metrics --slots --alias worker --host 127.0.0.1 --port 9701\r\n"
        );

        // A diffusion profile's script runs the helper, so it says it needs FIDIM.
        let mut dg = profile.clone();
        dg.engine = crate::profile::Engine::DiffusionGemma;
        let paths = crate::launch::DiffusionPaths {
            helper_exe: "C:/Programs/LlamaFIDIM/fidim-dg.exe".into(),
            runner_exe: "C:/b/bin/llama-diffusion-gemma-visual-server.exe".into(),
            req_prefix: "C:/runs/dg-worker-pool-9701".into(),
            expect_bus: Some(8),
            build_tag: None,
            full_offload: true,
        };
        let dplan = crate::launch::compose_diffusion(&dg, &resolved_bus08(), &paths);
        let bat = to_bat(&dg, &dplan);
        let ps1 = to_ps1(&dg, &dplan);
        for text in [&bat, &ps1] {
            assert!(text.contains("Needs Llama FIDIM's fidim-dg.exe"), "{text}");
            assert!(!text.contains("Runs standalone"), "{text}");
            assert!(text.contains("C:/Programs/LlamaFIDIM/fidim-dg.exe"), "{text}");
        }
        assert!(bat.contains("set \"NGL=99\""), "{bat}");
        assert!(ps1.contains("$env:MAXTOK = '8192'"), "{ps1}");
    }

    #[test]
    fn ps1_export_escapes_and_pins() {
        let (profile, plan) = plan_and_profile();
        let ps1 = to_ps1(&profile, &plan);
        assert!(ps1.contains("$env:HIP_VISIBLE_DEVICES = '2'"));
        assert!(ps1.contains(r#"$env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = '{"enable_thinking": false}'"#));
        assert!(ps1.contains("'E:/models/my model.gguf'"));
    }
}
