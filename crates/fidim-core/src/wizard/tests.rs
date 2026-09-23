//! The wizard's orchestration against a fake machine: no network, no GPU,
//! files only under a temporary folder. The repo listings are the captured
//! Hub fixtures; the K2 Horizon fork plan uses the captured GitHub compare
//! and the fork's source tables.

use super::*;
use crate::compat::github::{parse_compare, ResolvedRef};
use crate::compat::{CardCandidate, ProbedBuild, UpstreamCandidate};
use std::cell::RefCell;
use std::sync::atomic::AtomicBool;

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
const IFM_SHA: &str = "42adf019f76013dac873b5b43950d54d5ab27216";

fn fixture(rel: &str) -> String {
    std::fs::read_to_string(format!("{FIX}/{rel}")).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("fidim-wizard-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn r9700(key: &str, free_mib: u64) -> Device {
    serde_json::from_value(serde_json::json!({
        "stable_key": key, "name": "AMD Radeon AI PRO R9700", "hip_index": 0, "backend": "ROCm",
        "total_mib": 32624, "free_mib": free_mib, "integrated": false, "bus_number": 3,
        "driver_version": null, "display": null, "luid_low": null, "correlation_assumed": false
    }))
    .unwrap()
}

fn igpu() -> Device {
    let mut d = r9700("pci:igpu:bus19", 12099);
    d.name = "AMD Radeon(TM) Graphics".into();
    d.total_mib = 12381;
    d.integrated = true;
    d
}

fn build(root: &Path, tag: &str, version: Option<&str>, channel: Channel) -> Build {
    let path = root.join(tag);
    Build {
        server_exe: path.join("bin").join("llama-server.exe"),
        path,
        tag: tag.into(),
        version: version.map(String::from),
        commit: None,
        version_error: None,
        channel,
        bundled_runtime: channel == Channel::Unsloth,
        release_tag: None,
        patch: None,
        runner_exe: None,
        git: None,
    }
}

/// K2-Horizon-7B's architecture keys (36 layers, 8 KV heads of 128 dims,
/// no sliding window), as a 64 KiB read gets them.
fn k2_7b_header(repo: &str, path: &str, size: u64) -> GgufHeader {
    serde_json::from_value(serde_json::json!({
        "path": format!("hf://{repo}@x/{path}"), "file_size": size, "gguf_version": 3, "tensor_count": 0,
        "architecture": "k2-horizon", "block_count": 36, "context_length": 524288, "embedding_length": 4096,
        "head_count": 32, "head_count_kv": 8, "key_length": 128, "value_length": 128,
        "tokenizer_model": "gpt2", "tokenizer_pre": "k2-horizon", "vocab_size": 250624, "partial": true,
        "metadata": {}
    }))
    .unwrap()
}

fn gemma_header() -> GgufHeader {
    let mut h = k2_7b_header("x/y", "m.gguf", 0);
    h.architecture = Some("gemma4".into());
    h.tokenizer_model = Some("llama".into());
    h.tokenizer_pre = None;
    h
}

fn diffusion_header() -> GgufHeader {
    let kv: Vec<u64> = (0..30).map(|i| if i % 6 == 5 { 2 } else { 8 }).collect();
    let swa: Vec<bool> = (0..30).map(|i| i % 6 != 5).collect();
    serde_json::from_value(serde_json::json!({
        "path": "hf://unsloth/dg@x/dg-Q4_K_M.gguf", "file_size": 16_806_810_208u64, "gguf_version": 3, "tensor_count": 0,
        "architecture": "diffusion-gemma", "block_count": 30, "context_length": 262144,
        "embedding_length": 2816, "head_count": 16, "head_count_kv_per_layer": kv,
        "key_length": 512, "value_length": 512, "key_length_swa": 256, "value_length_swa": 256,
        "sliding_window": 1024, "swa_layer_flags": swa, "diffusion_canvas_length": 256, "vocab_size": 262144,
        "metadata": {}
    }))
    .unwrap()
}

/// The fork a K2 Horizon card links, resolved the way GitHub answered on
/// 2026-09-18, with its source tables read from the captured fork files.
fn k2_card() -> CardCandidate {
    let r = compat::card_refs("https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon").remove(0);
    let compare = parse_compare(&fixture("github/compare-master-ifm-ai-42adf019.json")).unwrap();
    let caps = compat::caps_from_dir(Path::new(&format!("{FIX}/llama-src/ifm-ai-42adf019"))).unwrap();
    let needs = ModelNeeds::new("k2-horizon", Some("k2-horizon".into()), None, Engine::LlamaServer);
    CardCandidate {
        git_ref: r.clone(),
        resolved: Some(ResolvedRef {
            requested: r,
            owner: "ifm-ai".into(),
            repo: "llama.cpp".into(),
            redirected: true,
            remote_url: "https://github.com/ifm-ai/llama.cpp".into(),
            git_ref: "model/K2Horizon".into(),
            sha: IFM_SHA.into(),
            is_upstream: false,
            is_fork_of_ggml: true,
            ahead_by: Some(compare.ahead_by),
            behind_by: Some(compare.behind_by),
            merge_base: compare.merge_base.clone(),
            head_subjects: compare.subjects,
            pr: None,
        }),
        error: None,
        support: Some(caps.support(&needs)),
    }
}

fn upstream_release() -> update::Release {
    let v: serde_json::Value = serde_json::from_str(&fixture("github/releases-upstream-b11046.json")).unwrap();
    update::parse_release(&v[0].to_string()).unwrap()
}

/// The machine and the network, faked; every side effect is logged.
struct Fake {
    root: PathBuf,
    info: RepoInfo,
    header: Option<GgufHeader>,
    full_fails: bool,
    builds: Vec<Build>,
    support: HashMap<PathBuf, Support>,
    inputs: PlanInputs,
    readme: Option<String>,
    devices: Vec<Device>,
    derivatives: Vec<RepoHit>,
    doctor_blocks: bool,
    auth_fails: bool,
    /// Free space on every drive; None = unknown.
    free: std::cell::Cell<Option<u64>>,
    calls: RefCell<Vec<String>>,
    saved: RefCell<Vec<Profile>>,
}

impl Fake {
    fn new(name: &str, info_fixture: &str) -> Self {
        let root = tmp(name);
        let info = hub::parse_model_info(&fixture(&format!("hub/{info_fixture}"))).unwrap();
        let builds = vec![
            build(&root.join("builds"), "b10984-rocm", Some("b10984"), Channel::Upstream),
            build(&root.join("builds"), "b11027-mix-unsloth", Some("b11027"), Channel::Unsloth),
        ];
        Fake {
            root,
            info,
            header: None,
            full_fails: false,
            builds,
            support: HashMap::new(),
            inputs: PlanInputs::default(),
            readme: None,
            devices: vec![r9700("pci:a:bus03", 32472), igpu(), r9700("pci:a:bus08", 32472)],
            derivatives: Vec::new(),
            doctor_blocks: false,
            auth_fails: false,
            free: std::cell::Cell::new(None),
            calls: RefCell::new(Vec::new()),
            saved: RefCell::new(Vec::new()),
        }
    }

    fn cfg(&self) -> Config {
        let mut c = Config::default_for_machine();
        c.model_roots = vec![self.root.join("models")];
        c.install_root = Some(self.root.join("builds"));
        c.profile_dir = self.root.join("profiles");
        c.runs_dir = self.root.join("runs");
        c
    }

    fn call(&self, s: impl Into<String>) {
        self.calls.borrow_mut().push(s.into());
    }

    fn calls(&self) -> Vec<String> {
        self.calls.borrow().clone()
    }

    /// K2 Horizon: no installed build knows it; the card links the fork.
    fn k2(name: &str) -> Self {
        let mut f = Fake::new(name, "model-info-ngquocvinh__K2-Horizon-7B-GGUF.json");
        f.header = Some(k2_7b_header("ngquocvinh/K2-Horizon-7B-GGUF", "K2-Horizon-7B-Q4_K_M.gguf", 0));
        let no = Support::No { missing: vec![compat::Missing::Arch, compat::Missing::TokenizerPre("k2-horizon".into())] };
        for b in &f.builds {
            f.support.insert(b.path.clone(), no.clone());
        }
        f.readme = Some("Needs [the fork](https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon).".into());
        let up = compat::caps_from_dir(Path::new(&format!("{FIX}/llama-src/b11046"))).unwrap();
        let needs = ModelNeeds::new("k2-horizon", Some("k2-horizon".into()), None, Engine::LlamaServer);
        f.inputs = PlanInputs {
            installed: f.builds.iter().map(|b| ProbedBuild { build: b.clone(), support: no.clone() }).collect(),
            upstream_latest: Some(UpstreamCandidate { release: upstream_release(), support: up.support(&needs) }),
            card_refs: vec![k2_card()],
            gfx: "gfx1201".into(),
            ..Default::default()
        };
        f
    }

    /// A model every installed upstream build knows.
    fn supported(name: &str) -> Self {
        let mut f = Fake::new(name, "model-info-ngquocvinh__K2-Horizon-7B-GGUF.json");
        f.header = Some(gemma_header());
        for b in &f.builds {
            f.support.insert(b.path.clone(), Support::Yes);
        }
        f
    }
}

impl Env for Fake {
    fn search(&self, q: &SearchQuery) -> Result<Vec<RepoHit>> {
        self.call(format!("search {}", q.text));
        hub::parse_search(&fixture("hub/search-k2-horizon.json"))
    }
    fn model_info(&self, repo: &str, _rev: Option<&str>) -> Result<RepoInfo> {
        self.call(format!("model_info {repo}"));
        Ok(self.info.clone())
    }
    fn list_files(&self, _repo: &str, _sha: &str) -> Result<Vec<RepoFile>> {
        Ok(self.info.siblings.clone())
    }
    fn readme(&self, repo: &str, _rev: &str) -> Result<Option<String>> {
        self.call(format!("readme {repo}"));
        Ok(self.readme.clone())
    }
    fn header(&self, _repo: &str, _sha: &str, files: &[RepoFile], mode: ReadMode) -> Result<GgufHeader> {
        self.call(format!("header {mode:?} {}", files[0].path));
        if mode == ReadMode::Full && self.full_fails {
            return Err(Error::Http { kind: crate::HttpErrorKind::Network, message: "reset".into() });
        }
        let mut h = self.header.clone().ok_or_else(|| Error::InvalidInput("no header".into()))?;
        h.file_size = files.iter().map(|f| f.size).sum();
        if mode == ReadMode::Full {
            h.partial = false;
            h.max_tensor_type_id = Some(41);
        }
        Ok(h)
    }
    fn derivatives(&self, base: &str) -> Result<Vec<RepoHit>> {
        self.call(format!("derivatives {base}"));
        Ok(self.derivatives.clone())
    }
    fn has_hf_token(&self) -> bool {
        false
    }
    fn auth_check(&self, repo: &str) -> Result<()> {
        self.call(format!("auth_check {repo}"));
        if self.auth_fails {
            return Err(Error::Http { kind: crate::HttpErrorKind::Gated, message: format!("{repo} is gated") });
        }
        Ok(())
    }
    fn builds(&self) -> Vec<Build> {
        self.builds.clone()
    }
    fn devices(&self) -> Result<Vec<Device>> {
        Ok(self.devices.clone())
    }
    fn gpu_targets(&self) -> Option<String> {
        Some("gfx1201".into())
    }
    fn busy_cards(&self) -> Vec<String> {
        vec!["pci:a:bus03".into()]
    }
    fn taken_ports(&self) -> Vec<u16> {
        vec![9710]
    }
    fn profiles(&self) -> Vec<Profile> {
        self.saved.borrow().clone()
    }
    fn probe(&self, b: &Build, _needs: &ModelNeeds) -> Support {
        self.support.get(&b.path).cloned().unwrap_or(Support::Yes)
    }
    fn gather(&self, needs: &ModelNeeds, _installed: &[Build], card_refs: &[GitRef], gfx: &str, hint: Option<&str>) -> PlanInputs {
        self.call(format!("gather {} refs={} gfx={gfx} hint={}", needs.arch, card_refs.len(), hint.unwrap_or("-")));
        self.inputs.clone()
    }
    fn doctor(&self, gfx: &str) -> Vec<crate::toolchain::Finding> {
        self.call(format!("doctor {gfx}"));
        let outcome = if self.doctor_blocks {
            crate::preflight::Outcome::Block("no HIP SDK clang found".into())
        } else {
            crate::preflight::Outcome::Pass
        };
        vec![crate::toolchain::Finding { id: "hip", title: "HIP SDK", outcome, detail: "7.1".into(), fix: None }]
    }
    fn download(
        &self,
        repo: &str,
        _sha: &str,
        file: &RepoFile,
        dest: &Path,
        progress: &mut dyn FnMut(&fetch::Progress),
        cancel: &AtomicBool,
    ) -> Result<u64> {
        self.call(format!("download {repo}/{}", file.path));
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        let name = dest.file_name().unwrap().to_string_lossy().into_owned();
        progress(&fetch::Progress { file: name, stage: fetch::Stage::Downloading, done: file.size / 2, total: Some(file.size), bps: 1e6 });
        std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
        std::fs::write(dest, b"not a real model").unwrap();
        Ok(file.size)
    }
    fn install_upstream(&self, tag: &str, progress: &mut dyn FnMut(String), cancel: &AtomicBool) -> Result<InstallReport> {
        self.call(format!("install_upstream {tag}"));
        progress(format!("{tag}: 10 / 20 MB"));
        // As install_prebuilt_cancellable: the flag, raised while its zips
        // download, ends it.
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        Ok(report(&self.root.join("builds").join(format!("{tag}-rocm")), tag))
    }
    fn install_unsloth(
        &self,
        tag: &str,
        gfx: &str,
        _progress: &mut dyn FnMut(String),
        cancel: &AtomicBool,
    ) -> Result<InstallReport> {
        self.call(format!("install_unsloth {tag} {gfx}"));
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        Ok(report(&self.root.join("builds").join(format!("{tag}-unsloth")), tag))
    }
    fn build_source(
        &self,
        src: &SourceRef,
        gpu_targets: &str,
        progress: &mut dyn FnMut(BuildProgress),
        cancel: &AtomicBool,
    ) -> Result<InstallReport> {
        self.call(format!("build_source {} {} {} {gpu_targets}", src.remote_url, src.git_ref, src.sha));
        progress(BuildProgress { step: "build".into(), done: Some(239), total: Some(478), line: "[239/478] Building HIP object".into() });
        // As build_from_ref: a stop ends the build script and reports its cleanup.
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Update(
                "build of ifm-ai K2Horizon fork @42adf01 cancelled; its worktree and staging files were removed".into(),
            ));
        }
        let dir = update::source_install_dir(&self.cfg(), src).unwrap();
        Ok(report(&dir, "ifm-ai K2Horizon fork @42adf01"))
    }
    fn save_profile(&self, p: &Profile) -> Result<PathBuf> {
        self.call(format!("save_profile {}", p.id));
        let path = save_new_profile(&self.cfg().profile_dir, p)?;
        self.saved.borrow_mut().push(p.clone());
        Ok(path)
    }
    fn free_bytes(&self, _path: &Path) -> Option<u64> {
        self.free.get()
    }
}

fn report(dir: &Path, tag: &str) -> InstallReport {
    InstallReport {
        tag: tag.into(),
        dir: dir.to_path_buf(),
        source: "fake".into(),
        skipped_existing: false,
        verify: update::Verify { version: Some("b6000".into()), hip_ok: true, ..Default::default() },
    }
}

// ------------------------------------------------------------------ pure ----

#[test]
fn needs_come_from_the_header() {
    let h = k2_7b_header("a/b", "c.gguf", 1);
    let n = needs_of(&h).unwrap();
    assert_eq!((n.arch.as_str(), n.tokenizer_pre.as_deref(), n.max_type_id), ("k2-horizon", Some("k2-horizon"), None));
    // A SentencePiece vocabulary: the pre-tokenizer name is not needed.
    let n = needs_of(&gemma_header()).unwrap();
    assert_eq!(n.tokenizer_pre, None);
    let n = needs_of(&diffusion_header()).unwrap();
    assert_eq!(n.engine, Engine::DiffusionGemma);
    let mut none = h.clone();
    none.architecture = None;
    assert!(needs_of(&none).is_none());
    // Tensor types every build knows are not a need; newer ones are.
    let mut typed = h.clone();
    typed.partial = false;
    typed.max_tensor_type_id = Some(12);
    assert_eq!(needs_of(&typed).unwrap().max_type_id, None, "Q4_K");
    typed.max_tensor_type_id = Some(39);
    assert_eq!(needs_of(&typed).unwrap().max_type_id, Some(39), "MXFP4");
}

/// The view and the plan cross the Tauri boundary twice (to the web view
/// and back): a header value JSON cannot hold must not break the trip.
#[test]
fn headers_travel_without_their_raw_metadata() {
    let mut f = Fake::supported("slim");
    let mut h = gemma_header();
    h.metadata.insert("general.sampling.temp".into(), crate::gguf::Value::F64(f64::NAN));
    h.metadata.insert("tokenizer.chat_template".into(), crate::gguf::Value::Str("x".repeat(40_000)));
    f.header = Some(h);
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    assert!(view.header.as_ref().unwrap().metadata.is_empty());
    let view: RepoView = serde_json::from_str(&serde_json::to_string(&view).unwrap()).unwrap();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest::default()).unwrap();
    assert!(plan.header.as_ref().unwrap().metadata.is_empty());
    assert!(serde_json::to_string(&plan).unwrap().len() < 40_000);
    let back: WizardPlan = serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
    assert_eq!(back.header.unwrap().block_count, Some(36));
}

#[test]
fn sizes_read_in_their_unit() {
    assert_eq!(human_size(1_185_376), "1.1 MiB");
    assert_eq!(human_size(5_592_219_008), "5.2 GiB");
    assert_eq!(human_size(2598), "3 KiB");
    assert_eq!(human_size(0), "0 KiB");
}

#[test]
fn hints_for_the_pull_request_search() {
    assert_eq!(model_hint("IFM/K2-Horizon-MoVA-36B-A4B", None).as_deref(), Some("K2 Horizon MoVA"));
    assert_eq!(model_hint("ngquocvinh/K2-Horizon-7B-GGUF", None).as_deref(), Some("K2 Horizon"));
    assert_eq!(model_hint("unsloth/gemma-4-26B-A4B-it-GGUF", None).as_deref(), Some("gemma 4"));
    assert_eq!(model_hint("a/7B", None), None);
    let info = hub::parse_model_info(&fixture("hub/model-info-ngquocvinh__K2-Horizon-7B-GGUF.json")).unwrap();
    assert_eq!(model_hint(&info.id, Some(&info)).as_deref(), Some("K2 Horizon"), "from the base model");
}

fn info_with(id: &str, files: &[&str], library: Option<&str>, tags: &[&str]) -> RepoInfo {
    RepoInfo {
        id: id.into(),
        sha: "a".repeat(40),
        gated: Gated::No,
        gated_prompt: None,
        card_license: None,
        base_models: vec![("finetune".into(), "IFM/K2-Horizon-7B".into())],
        tags: tags.iter().map(|t| t.to_string()).collect(),
        pipeline_tag: None,
        library_name: library.map(String::from),
        last_modified: None,
        gguf: None,
        siblings: files.iter().map(|p| RepoFile { path: p.to_string(), size: 10, sha256: None }).collect(),
    }
}

#[test]
fn what_a_repo_holds() {
    let kind = |i: &RepoInfo| repo_kind(i, &catalog::classify(&i.siblings));
    assert_eq!(kind(&info_with("a/b", &["m-Q4_K_M.gguf", "model.safetensors"], None, &[])), RepoKind::Gguf);
    assert_eq!(kind(&info_with("IFM/K2-Horizon-7B", &["model-00001-of-00002.safetensors", "config.json"], Some("transformers"), &[])), RepoKind::Safetensors);
    assert_eq!(kind(&info_with("IFM/K2-Horizon-7B-Uno", &["adapter_model.safetensors", "adapter_config.json"], Some("peft"), &[])), RepoKind::Adapter);
    assert_eq!(kind(&info_with("IFM/K2-Horizon-7B-FP8", &["model.safetensors"], Some("transformers"), &[])), RepoKind::OtherFormat("FP8".into()));
    assert_eq!(kind(&info_with("x/Model-AWQ", &["model.safetensors"], None, &[])), RepoKind::OtherFormat("AWQ".into()));
    assert_eq!(kind(&info_with("x/model-4bit", &["model.safetensors"], Some("mlx"), &[])), RepoKind::OtherFormat("MLX".into()));
    assert_eq!(kind(&info_with("x/model", &["model.safetensors"], None, &["nvfp4"])), RepoKind::OtherFormat("NVFP4".into()));
    assert_eq!(kind(&info_with("x/empty", &["README.md"], None, &[])), RepoKind::Empty);
    assert_eq!(kind(&info_with("x/proj", &["mmproj-F16.gguf"], None, &[])), RepoKind::Empty);
}

#[test]
fn job_progress_follows_the_events() {
    let f = Fake::k2("progress");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest { choice: Some("Q4_K_M".into()), ..Default::default() }).unwrap();
    let mut jp = JobProgress::new(&plan);
    assert_eq!(jp.steps.len(), plan.steps.len());
    assert!(jp.steps.iter().all(|s| s.status == StepStatus::Pending));
    jp.apply(&WizardEvent::status(0, StepStatus::Running));
    jp.apply(&WizardEvent { stage: Some("build".into()), done: Some(10), total: Some(478), line: Some("[10/478] x".into()), ..WizardEvent::status(0, StepStatus::Running) });
    assert_eq!((jp.steps[0].done, jp.steps[0].total, jp.steps[0].stage.as_deref()), (Some(10), Some(478), Some("build")));
    assert_eq!(jp.log, ["[10/478] x"]);
    jp.apply(&WizardEvent::status(0, StepStatus::Done));
    assert_eq!(jp.steps[0].status, StepStatus::Done);
    assert_eq!(jp.steps[0].done, Some(10), "a status change keeps the last numbers");
    jp.apply(&WizardEvent { bps: Some(5e7), file: Some("a.gguf".into()), done: Some(1), total: Some(2), ..WizardEvent::status(1, StepStatus::Running) });
    jp.apply(&WizardEvent { line: Some("disk full".into()), ..WizardEvent::status(1, StepStatus::Failed) });
    assert_eq!(jp.steps[1].error.as_deref(), Some("disk full"));
    assert_eq!(jp.steps[1].bps, None, "a stopped step has no speed");
    assert_eq!(jp.log.len(), 1, "errors are not build output");
    for n in 0..500 {
        jp.apply(&WizardEvent { line: Some(format!("l{n}")), ..WizardEvent::status(0, StepStatus::Running) });
    }
    assert_eq!(jp.log.len(), 400);
    assert_eq!(jp.log.last().map(String::as_str), Some("l499"));
    jp.apply(&WizardEvent::status(99, StepStatus::Done)); // out of range: ignored
}

// --------------------------------------------------------------- inspect ----

#[test]
fn inspect_k2_finds_the_fork() {
    let f = Fake::k2("inspect-k2");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "https://huggingface.co/ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    assert_eq!(view.kind, RepoKind::Gguf);
    assert_eq!(view.repo, "ngquocvinh/K2-Horizon-7B-GGUF");
    assert!(view.catalog.choices.iter().any(|c| c.label == "Q4_K_M"));
    assert!(view.catalog.ignored.iter().any(|(f, _)| f.path.contains("imatrix")));
    // One 64 KiB read, never the whole header, at inspect.
    let headers: Vec<String> = f.calls().into_iter().filter(|c| c.starts_with("header")).collect();
    assert_eq!(headers.len(), 1, "{headers:?}");
    assert!(headers[0].starts_with("header UntilTokenizer"));
    let needs = view.needs.as_ref().unwrap();
    assert_eq!((needs.arch.as_str(), needs.tokenizer_pre.as_deref()), ("k2-horizon", Some("k2-horizon")));
    // Every choice estimated on the cards; a 7B fits one R9700.
    assert_eq!(view.fits.len(), view.catalog.choices.len());
    let rec = view.recommended.clone().unwrap();
    let fit = view.fits.iter().find(|x| x.label == rec).unwrap();
    assert_eq!(fit.one_card.fit, Fit::Fits);
    assert!(view.devices.iter().any(|d| d.integrated), "the view keeps every device; the estimate skips the iGPU");
    // No installed build: the plan is the card's fork, with consent.
    assert_eq!(view.usable_build, None);
    assert!(view.builds.iter().all(|b| b.support.is_no()));
    let bp = view.build_plan.as_ref().unwrap();
    assert!(matches!(&bp.step.action, PlanAction::BuildFork { owner, .. } if owner == "ifm-ai"), "{:?}", bp.step.action);
    assert!(bp.step.needs_consent);
    assert!(f.calls().iter().any(|c| c == "gather k2-horizon refs=1 gfx=gfx1201 hint=K2 Horizon"), "{:?}", f.calls());
    // What the fork is, for the person asked to build it.
    let src = &view.build_sources[0];
    assert_eq!((src.owner.as_str(), src.repo.as_str(), src.sha.as_str()), ("ifm-ai", "llama.cpp", IFM_SHA));
    assert_eq!(src.linked_as.as_deref(), Some("MBZUAI-IFM/llama.cpp"));
    assert_eq!(src.ahead_by, Some(10));
    assert!(src.behind_by.is_some_and(|b| b > 300));
    assert!(!src.subjects.is_empty() && src.is_fork_of_upstream && src.support.as_ref().is_some_and(|s| s.is_yes()));
    // The view crosses the Tauri boundary and comes back for wizard_plan.
    let back: RepoView = serde_json::from_str(&serde_json::to_string(&view).unwrap()).unwrap();
    assert_eq!(back.build_plan, view.build_plan);
    assert_eq!(back.fits, view.fits);
}

#[test]
fn inspect_a_supported_model_asks_github_nothing() {
    let f = Fake::supported("inspect-ok");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    assert_eq!(view.usable_build.as_deref(), Some(f.builds[0].path.as_path()), "the upstream build first");
    assert!(view.build_plan.is_none());
    assert!(!f.calls().iter().any(|c| c.starts_with("gather") || c.starts_with("readme")), "{:?}", f.calls());
}

#[test]
fn inspect_a_pasted_file_link_preselects_it() {
    let f = Fake::supported("inspect-link");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "https://huggingface.co/ngquocvinh/K2-Horizon-7B-GGUF/blob/main/K2-Horizon-7B-Q8_0.gguf", None).unwrap();
    assert_eq!(view.preselect.as_deref(), Some("Q8_0"));
    assert_eq!(view.recommended.as_deref(), Some("Q8_0"));
    assert_eq!(view.header_of.as_deref(), Some("Q8_0"));
    assert!(matches!(inspect_with(&f, &cfg, "not a repo at all", None), Err(Error::InvalidInput(_))));
    assert!(matches!(inspect_with(&f, &cfg, "https://huggingface.co/datasets/a/b", None), Err(Error::InvalidInput(_))));
}

#[test]
fn inspect_explains_repos_llama_cpp_cannot_run() {
    let mut f = Fake::new("inspect-st", "model-info-ngquocvinh__K2-Horizon-7B-GGUF.json");
    f.info = info_with("IFM/K2-Horizon-7B", &["model-00001-of-00002.safetensors", "config.json"], Some("transformers"), &[]);
    f.derivatives = hub::parse_search(&fixture("hub/search-k2-horizon.json")).unwrap();
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "IFM/K2-Horizon-7B", None).unwrap();
    assert_eq!(view.kind, RepoKind::Safetensors);
    assert_eq!(view.derivatives_of.as_deref(), Some("IFM/K2-Horizon-7B"));
    assert_eq!(view.derivatives.len(), 25);
    assert!(view.notes.iter().any(|n| n.code == "not-gguf" && n.message.contains("transformers")), "{:?}", view.notes);
    assert!(!f.calls().iter().any(|c| c.starts_with("header")), "nothing to read");
    assert!(plan_with(&f, &cfg, &view, &PlanRequest::default()).is_err());

    // An adapter lists its base model's quantizations.
    f.info = info_with("IFM/K2-Horizon-7B-Uno", &["adapter_config.json", "adapter_model.safetensors"], Some("peft"), &[]);
    let view = inspect_with(&f, &cfg, "IFM/K2-Horizon-7B-Uno", None).unwrap();
    assert_eq!(view.kind, RepoKind::Adapter);
    assert_eq!(view.derivatives_of.as_deref(), Some("IFM/K2-Horizon-7B"));
    assert!(f.calls().iter().any(|c| c == "derivatives IFM/K2-Horizon-7B"));

    f.info = info_with("IFM/K2-Horizon-7B-FP8", &["model.safetensors"], None, &[]);
    let view = inspect_with(&f, &cfg, "IFM/K2-Horizon-7B-FP8", None).unwrap();
    assert_eq!(view.kind, RepoKind::OtherFormat("FP8".into()));
    assert!(view.notes[0].message.contains("FP8"), "{:?}", view.notes);
}

#[test]
fn inspect_notes_a_gated_repo_without_a_token() {
    let mut f = Fake::supported("inspect-gated");
    f.info.gated = Gated::Manual;
    f.info.gated_prompt = Some("Agree to share your contact information.".into());
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let gated = view.notes.iter().find(|n| n.code == "gated").unwrap();
    assert!(gated.message.contains("https://huggingface.co/ngquocvinh/K2-Horizon-7B-GGUF") && gated.message.contains("by hand"), "{}", gated.message);
    assert!(view.notes.iter().any(|n| n.code == "no-token" && n.level == Level::Warning));

    // The card's terms are someone else's text: one line, no control
    // characters, at most 400 of them.
    f.info.gated_prompt = Some("Agree\r\n\x1b[2K\x1b[1Ato  the\tterms.\u{9b}8m\n\n".to_string() + &"x".repeat(1000));
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let gated = &view.notes.iter().find(|n| n.code == "gated").unwrap().message;
    let terms = gated.split(". The terms: ").nth(1).unwrap();
    assert!(!gated.chars().any(char::is_control), "{gated:?}");
    assert!(terms.starts_with("Agree [2K [1Ato the terms. 8m xxx"), "{terms:?}");
    assert_eq!(terms.chars().count(), 400);
    f.info.gated_prompt = Some(" \r\n ".into());
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    assert!(!view.notes.iter().find(|n| n.code == "gated").unwrap().message.contains("The terms"));
}

#[test]
fn search_marks_architectures_no_build_knows() {
    let mut f = Fake::supported("search");
    let no = Support::No { missing: vec![compat::Missing::Arch] };
    for b in f.builds.clone() {
        f.support.insert(b.path, no.clone());
    }
    let hits = search_with(&f, &SearchQuery { text: "K2-Horizon".into(), ..Default::default() }).unwrap();
    assert_eq!(hits.len(), 25);
    assert!(hits.iter().all(|h| h.arch_known == Some(false)));
    f.builds.clear();
    let hits = search_with(&f, &SearchQuery { text: "K2-Horizon".into(), ..Default::default() }).unwrap();
    assert!(hits.iter().all(|h| h.arch_known.is_none()), "no build: unknown");
}

// ------------------------------------------------------------------ plan ----

#[test]
fn plan_for_a_supported_model_downloads_and_makes_a_profile() {
    let f = Fake::supported("plan-ok");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest { choice: Some("Q4_K_M".into()), ..Default::default() }).unwrap();
    assert!(!plan.blocked, "{:?}", plan.notes);
    assert!(!plan.needs_consent && plan.consent.is_none());
    assert_eq!(plan.steps.len(), 2, "{:?}", plan.steps.iter().map(|s| &s.title).collect::<Vec<_>>());
    match &plan.steps[0].action {
        StepAction::Download { role, file, dest, have, present } => {
            assert_eq!(*role, FileRole::Model);
            assert_eq!(file.path, "K2-Horizon-7B-Q4_K_M.gguf");
            assert_eq!(dest, &cfg.model_roots[0].join("ngquocvinh").join("K2-Horizon-7B-GGUF").join("K2-Horizon-7B-Q4_K_M.gguf"));
            assert_eq!((*have, *present), (0, false));
        }
        o => panic!("{o:?}"),
    }
    assert_eq!(plan.download_bytes, plan.total_bytes);
    assert_eq!(plan.download_bytes, plan.choice.total_size);
    // The whole header of the chosen file was read for its tensor types.
    assert!(f.calls().iter().any(|c| c == "header Full K2-Horizon-7B-Q4_K_M.gguf"), "{:?}", f.calls());
    assert_eq!(plan.needs.as_ref().unwrap().max_type_id, Some(41));
    // The profile: on the upstream build, an idle card, the first free port.
    let p = plan.profile.as_ref().unwrap();
    assert_eq!(p.build.path, f.builds[0].path);
    assert_eq!(p.id, "k2-horizon-7b-q4-k-m");
    assert_eq!(p.devices.len(), 1);
    assert_eq!(p.devices[0].key, "pci:a:bus08", "bus03 is busy");
    assert_eq!(p.server.port, 9711, "9710 is taken by a running server");
    assert_eq!(p.runtime.ctx_total, 32768);
    assert_eq!(p.model.path, match &plan.steps[0].action { StepAction::Download { dest, .. } => dest.clone(), _ => unreachable!() });
    assert!(matches!(&plan.steps[1].action, StepAction::Profile { id, .. } if id == &p.id));
    assert!(p.notes.contains("ngquocvinh/K2-Horizon-7B-GGUF"));
    // The plan crosses the Tauri boundary and comes back for wizard_start.
    let back: WizardPlan = serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
    assert_eq!(back.steps, plan.steps);
    assert!(check_runnable(&back, false).is_ok());
}

#[test]
fn plan_for_k2_builds_the_fork_with_consent() {
    let f = Fake::k2("plan-k2");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest { choice: Some("Q4_K_M".into()), ..Default::default() }).unwrap();
    assert!(!plan.blocked, "{:?}", plan.notes);
    assert!(plan.needs_consent);
    let consent = plan.consent.as_deref().unwrap();
    assert!(consent.contains("ifm-ai/llama.cpp") && consent.contains("nobody reviewed for your machine"), "{consent}");
    let s0 = &plan.steps[0];
    assert!(s0.needs_consent);
    match &s0.action {
        StepAction::Build { plan: ps } => match &ps.action {
            PlanAction::BuildFork { source, gpu_targets, install_dir, .. } => {
                assert_eq!(source.sha, IFM_SHA);
                assert_eq!(gpu_targets, "gfx1201");
                assert_eq!(install_dir, &cfg.install_root.clone().unwrap().join("ifm-ai-K2Horizon-fork-42adf019-src"));
            }
            o => panic!("{o:?}"),
        },
        o => panic!("{o:?}"),
    }
    assert!(s0.title.contains("ifm-ai K2Horizon fork @42adf01"), "{}", s0.title);
    assert_eq!(plan.build_sources.len(), 1);
    assert_eq!(plan.build_sources[0].sha, IFM_SHA);
    // The toolchain doctor ran for the source build.
    assert!(f.calls().iter().any(|c| c == "doctor gfx1201"));
    assert_eq!(plan.toolchain.len(), 1);
    assert_eq!(plan.toolchain[0].outcome, "pass");
    // The new profile goes on the build the plan makes, labelled as a git build.
    let pb = plan.build.as_ref().unwrap();
    assert!(!pb.installed);
    assert_eq!(plan.profile.as_ref().unwrap().build.path, pb.path);
    assert!(matches!(plan.steps.last().unwrap().action, StepAction::Profile { .. }));
    // The fork is far behind upstream: the step says so.
    assert!(plan.notes.iter().any(|n| n.code == "build-warning" && n.message.contains("redirects to ifm-ai/llama.cpp")), "{:?}", plan.notes);

    // A toolchain blocker blocks the plan.
    let mut f2 = Fake::k2("plan-k2-doctor");
    f2.doctor_blocks = true;
    let cfg2 = f2.cfg();
    let view2 = inspect_with(&f2, &cfg2, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let blocked = plan_with(&f2, &cfg2, &view2, &PlanRequest::default()).unwrap();
    assert!(blocked.blocked);
    assert!(blocked.notes.iter().any(|n| n.level == Level::Error && n.message.contains("HIP SDK")));
}

#[test]
fn a_source_build_without_a_gpu_target_says_how_to_give_one() {
    let mut f = Fake::k2("plan-no-gfx");
    f.inputs.gfx = String::new();
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest::default()).unwrap();
    assert!(plan.blocked);
    let note = plan.notes.iter().find(|n| n.code == "no-gpu-target").unwrap();
    assert!(note.message.contains("fidim models get <repo> --gfx gfx1201") && note.message.contains("Settings"), "{}", note.message);
    assert!(!f.calls().iter().any(|c| c.starts_with("doctor")), "no toolchain check without a target");
}

#[test]
fn plan_choices_skip_install_and_alternatives() {
    let f = Fake::k2("plan-choices");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    // Skip: no build step; the profile goes on the best build that runs,
    // with a warning that it cannot load the model.
    let skip = plan_with(&f, &cfg, &view, &PlanRequest { build: BuildChoice::Skip, ..Default::default() }).unwrap();
    assert!(!skip.steps.iter().any(|s| matches!(s.action, StepAction::Build { .. })));
    assert!(!skip.needs_consent);
    assert!(skip.notes.iter().any(|n| n.code == "build-skipped"), "{:?}", skip.notes);
    assert_eq!(skip.profile.as_ref().unwrap().build.path, f.builds[0].path);
    // An installed build that lacks the architecture: an error.
    let bad = plan_with(&f, &cfg, &view, &PlanRequest { build: BuildChoice::Installed(f.builds[0].path.clone()), ..Default::default() }).unwrap();
    assert!(bad.blocked);
    assert!(bad.notes.iter().any(|n| n.code == "build-cannot-load" && n.message.contains("architecture 'k2-horizon'")), "{:?}", bad.notes);
    assert!(plan_with(&f, &cfg, &view, &PlanRequest { build: BuildChoice::Installed(PathBuf::from(r"C:\nowhere")), ..Default::default() }).is_err());
    // The plan has no alternative: asking for one is an error.
    assert!(plan_with(&f, &cfg, &view, &PlanRequest { build: BuildChoice::Plan(1), ..Default::default() }).is_err());
    // Unknown choice, projector or draft names.
    assert!(plan_with(&f, &cfg, &view, &PlanRequest { choice: Some("Q9_K".into()), ..Default::default() }).is_err());
    assert!(plan_with(&f, &cfg, &view, &PlanRequest { mmproj: Some("mmproj.gguf".into()), ..Default::default() }).is_err());
    // No model folder and no destination.
    let mut no_roots = cfg.clone();
    no_roots.model_roots.clear();
    assert!(plan_with(&f, &no_roots, &view, &PlanRequest::default()).is_err());
    // A destination that is not a model folder: the scan will not list it.
    let other = f.root.join("elsewhere");
    let p = plan_with(&f, &cfg, &view, &PlanRequest { dest_root: Some(other.clone()), build: BuildChoice::Skip, ..Default::default() }).unwrap();
    assert!(p.notes.iter().any(|n| n.code == "dest-not-scanned"));
    assert!(p.dest_dir.starts_with(&other));
}

#[test]
fn plan_without_any_build_downloads_but_makes_no_profile() {
    let mut f = Fake::k2("plan-unsupported");
    f.inputs.card_refs.clear();
    f.readme = None;
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    assert!(matches!(view.build_plan.as_ref().unwrap().step.action, PlanAction::Unsupported { .. }));
    let plan = plan_with(&f, &cfg, &view, &PlanRequest::default()).unwrap();
    assert!(!plan.blocked);
    assert!(plan.steps.iter().all(|s| matches!(s.action, StepAction::Download { .. })), "downloads only");
    assert!(plan.profile.is_none());
    assert!(plan.notes.iter().any(|n| n.code == "no-build-found"));
}

#[test]
fn plan_with_projector_draft_and_partial_downloads() {
    let mut f = Fake::supported("plan-aux");
    f.info = hub::parse_model_info(&fixture("hub/model-info-unsloth__gemma-4-26B-A4B-it-GGUF.json")).unwrap();
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "unsloth/gemma-4-26B-A4B-it-GGUF", None).unwrap();
    let mmproj = view.catalog.mmproj[0].path.clone();
    let draft = view.catalog.drafts[0].path.clone();
    // A part of the BF16 split set was fetched earlier.
    let bf16 = view.catalog.choices.iter().find(|c| c.label == "BF16").unwrap().clone();
    let dest0 = catalog::dest_path(&cfg.model_roots[0], &view.repo, &bf16.files[0].path);
    std::fs::create_dir_all(dest0.parent().unwrap()).unwrap();
    std::fs::write(fetch::part_path(&dest0), vec![0u8; 1000]).unwrap();
    let req = PlanRequest { choice: Some("BF16".into()), mmproj: Some(mmproj.clone()), draft: Some(draft.clone()), ctx: Some(8192), ..Default::default() };
    let plan = plan_with(&f, &cfg, &view, &req).unwrap();
    let downloads: Vec<(&FileRole, &PathBuf, u64)> = plan
        .steps
        .iter()
        .filter_map(|s| match &s.action {
            StepAction::Download { role, dest, have, .. } => Some((role, dest, *have)),
            _ => None,
        })
        .collect();
    assert_eq!(downloads.len(), 4, "two shards, the projector, the draft");
    assert_eq!(downloads[0].2, 1000, "resumes the .part");
    assert!(plan.steps[0].detail.starts_with("resumes at"));
    // Shards are flattened into one folder beside each other.
    assert_eq!(downloads[0].1.parent(), downloads[1].1.parent());
    assert_eq!(plan.download_bytes, plan.total_bytes - 1000);
    // BF16 of a 26B needs both cards: the profile layer-splits.
    let fit = plan.fit.as_ref().unwrap();
    assert_eq!(fit.ctx, 8192);
    let p = plan.profile.as_ref().unwrap();
    assert_eq!(p.devices.len(), 2, "{:?} / {:?}", fit.one_card, fit.two_card_split);
    assert_eq!(p.split_mode, Some(crate::profile::SplitMode::Layer));
    assert_eq!(p.runtime.ctx_total, 8192, "the context asked for");
    assert_eq!(p.model.mmproj.as_ref(), Some(downloads[2].1));
    let d = p.model.draft.as_ref().unwrap();
    assert!(d.enabled && d.path == *downloads[3].1);
    assert_eq!(p.speculative.as_ref().unwrap().mode, "mtp");
    std::fs::remove_dir_all(&f.root).ok();
}

/// ggml-org/gpt-oss-20b-GGUF's layout: the model with an EAGLE3 head
/// beside it (and, here, a DFlash draft and an MTP/ folder too).
#[test]
fn draft_heads_get_their_own_mode_or_are_refused() {
    let mut f = Fake::supported("plan-heads");
    f.info = info_with(
        "ggml-org/gpt-oss-20b-GGUF",
        &["gpt-oss-20b-mxfp4.gguf", "eagle3-gpt-oss-20b-Q8_0.gguf", "dflash-gpt-oss-20b-Q8_0.gguf", "MTP/gpt-oss-20b-head-Q8_0.gguf"],
        None,
        &[],
    );
    for s in &mut f.info.siblings {
        s.size = if s.path.starts_with("gpt-oss") { 12_109_566_560 } else { 100 << 20 };
    }
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ggml-org/gpt-oss-20b-GGUF", None).unwrap();
    assert_eq!(view.catalog.drafts.len(), 3, "{:?}", view.catalog.drafts);
    assert_eq!(view.draft_modes.get("eagle3-gpt-oss-20b-Q8_0.gguf"), Some(&None));
    assert_eq!(view.draft_modes.get("dflash-gpt-oss-20b-Q8_0.gguf"), Some(&Some("dflash".to_string())));
    assert_eq!(view.draft_modes.get("MTP/gpt-oss-20b-head-Q8_0.gguf"), Some(&Some("mtp".to_string())));
    // `--draft` with no name never picks the EAGLE3 head.
    let choice = &view.catalog.choices[0];
    assert_ne!(default_draft(&view.catalog, choice).map(|d| d.path.as_str()), Some("eagle3-gpt-oss-20b-Q8_0.gguf"));
    // Asked for by name, it is refused.
    let req = |d: &str| PlanRequest { draft: Some(d.into()), ..Default::default() };
    let err = plan_with(&f, &cfg, &view, &req("eagle3-gpt-oss-20b-Q8_0.gguf")).unwrap_err().to_string();
    assert!(err.contains("an EAGLE3 head") && err.contains("pick another draft"), "{err}");
    // A DFlash draft gets draft-dflash; the head in MTP/ is MTP though its
    // name, flattened on disk, does not say so.
    let p = plan_with(&f, &cfg, &view, &req("dflash-gpt-oss-20b-Q8_0.gguf")).unwrap().profile.unwrap();
    assert_eq!(p.speculative_effective().spec_type(), Some("draft-dflash"));
    let p = plan_with(&f, &cfg, &view, &req("MTP/gpt-oss-20b-head-Q8_0.gguf")).unwrap().profile.unwrap();
    assert_eq!(p.model.draft.as_ref().unwrap().path.file_name().unwrap(), "gpt-oss-20b-head-Q8_0.gguf");
    assert_eq!(p.speculative.as_ref().unwrap().mode, "mtp");
}

#[test]
fn plan_keeps_going_when_the_whole_header_cannot_be_read() {
    let mut f = Fake::supported("plan-nofull");
    f.full_fails = true;
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest::default()).unwrap();
    assert!(!plan.blocked);
    assert!(plan.notes.iter().any(|n| n.code == "types-unchecked"));
    assert_eq!(plan.needs.as_ref().unwrap().max_type_id, None);
    assert_eq!(plan.header.as_ref().unwrap().file_size, plan.choice.total_size);
}

#[test]
fn a_diffusion_model_gets_the_diffusion_profile() {
    let mut f = Fake::supported("plan-dg");
    f.header = Some(diffusion_header());
    let mut runner = build(&f.root.join("builds"), "b11027-mix-unsloth-dgpatch", Some("b11027"), Channel::Unsloth);
    runner.patch = Some(discovery::BuildPatch { name: "dgpatch".into(), features: vec!["dg-fa-pad".into()], ..Default::default() });
    f.builds = vec![runner.clone()];
    f.support = HashMap::from([(runner.path.clone(), Support::Yes)]);
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    assert_eq!(view.needs.as_ref().unwrap().engine, Engine::DiffusionGemma);
    let plan = plan_with(&f, &cfg, &view, &PlanRequest::default()).unwrap();
    assert_eq!(plan.engine, Engine::DiffusionGemma);
    let p = plan.profile.as_ref().unwrap();
    assert_eq!(p.engine, Engine::DiffusionGemma);
    assert_eq!(p.runtime.ctx_total, 0, "the runner sizes itself");
    assert_eq!(p.devices.len(), 1);
    assert!(p.diffusion.as_ref().unwrap().flash_attn, "the build pads keys for flash attention");
    assert!(p.speculative.is_none() && p.model.mmproj.is_none());
}

// ------------------------------------------------------------------- run ----

#[test]
fn a_fork_build_never_runs_without_consent() {
    let f = Fake::k2("run-consent");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest::default()).unwrap();
    let before = f.calls().len();
    let mut events = Vec::new();
    let err = run_with(&f, &cfg, &plan, false, &mut |e| events.push(e.clone()), &AtomicBool::new(false)).unwrap_err();
    assert!(err.to_string().contains("ifm-ai/llama.cpp") && err.to_string().contains("Nothing was done"), "{err}");
    assert_eq!(f.calls().len(), before, "nothing ran: {:?}", &f.calls()[before..]);
    assert!(events.is_empty());
    assert!(f.saved.borrow().is_empty());

    // A plan edited to drop every consent flag (the step's, its build
    // plan's, the plan's) and its sentence is refused too: consent follows
    // what the step builds, not the flags.
    let mut edited = plan.clone();
    edited.steps[0].needs_consent = false;
    if let StepAction::Build { plan: ps } = &mut edited.steps[0].action {
        ps.needs_consent = false;
    }
    edited.needs_consent = false;
    edited.consent = None;
    assert!(edited.requires_consent());
    let err = run_with(&f, &cfg, &edited, false, &mut |_| {}, &AtomicBool::new(false)).unwrap_err();
    assert!(err.to_string().contains("runs code from ifm-ai/llama.cpp"), "the sentence comes from the step: {err}");
    assert!(check_runnable(&edited, false).is_err());
    assert_eq!(f.calls().len(), before);
    // The same for a pull request's build.
    let mut pr = edited.clone();
    if let StepAction::Build { plan: ps } = &mut pr.steps[0].action {
        let PlanAction::BuildFork { source, gpu_targets, install_dir, .. } = ps.action.clone() else { unreachable!() };
        ps.action = PlanAction::BuildPr { number: 17000, source, gpu_targets, install_dir };
    }
    assert!(check_runnable(&pr, false).unwrap_err().to_string().contains("pull request #17000"));
    // With consent it is runnable; a plan with no build needs none.
    assert!(check_runnable(&edited, true).is_ok());
    let mut downloads_only = plan.clone();
    downloads_only.steps.remove(0);
    assert!(!downloads_only.requires_consent() && check_runnable(&downloads_only, false).is_ok());
}

#[test]
fn an_unsloth_mix_for_a_pull_request_needs_consent() {
    let f = Fake::k2("run-unsloth");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let mut plan = plan_with(&f, &cfg, &view, &PlanRequest::default()).unwrap();
    let dir = cfg.install_root.clone().unwrap().join("b11027-mix-3e83366-unsloth");
    let mix = PlanStep {
        action: PlanAction::InstallUnsloth { tag: "b11027-mix-3e83366".into(), gfx: "gfx120X".into(), install_dir: dir },
        explanation: "merges pull request #17000".into(),
        needs_consent: true,
        verified: true,
        warnings: vec![],
    };
    plan.steps[0] = Step { title: step_title(&mix), detail: String::new(), needs_consent: true, action: StepAction::Build { plan: mix.clone() } };
    assert!(check_runnable(&plan, false).unwrap_err().to_string().contains("Unsloth's b11027-mix-3e83366"));
    // Without its flag, an Unsloth install (a published release) runs as
    // any prebuilt does.
    let mut plain = plan.clone();
    plain.steps[0].needs_consent = false;
    if let StepAction::Build { plan: ps } = &mut plain.steps[0].action {
        ps.needs_consent = false;
    }
    assert!(check_runnable(&plain, false).is_ok());
    // With its flag on the build plan only, the step still needs consent.
    let mut half = plain.clone();
    if let StepAction::Build { plan: ps } = &mut half.steps[0].action {
        ps.needs_consent = true;
    }
    assert!(check_runnable(&half, false).is_err());
}

#[test]
fn run_builds_downloads_then_creates_the_profile() {
    let f = Fake::k2("run-all");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest { choice: Some("Q4_K_M".into()), ..Default::default() }).unwrap();
    let before = f.calls().len();
    let mut jp = JobProgress::new(&plan);
    let mut events = Vec::new();
    let r = run_with(
        &f,
        &cfg,
        &plan,
        true,
        &mut |e| {
            jp.apply(e);
            events.push(e.clone())
        },
        &AtomicBool::new(false),
    )
    .unwrap();
    let ran: Vec<String> = f.calls()[before..].to_vec();
    assert_eq!(ran[0], "auth_check ngquocvinh/K2-Horizon-7B-GGUF", "access first, before a long build");
    assert_eq!(ran[1], format!("build_source https://github.com/ifm-ai/llama.cpp model/K2Horizon {IFM_SHA} gfx1201"));
    assert_eq!(ran[2], "download ngquocvinh/K2-Horizon-7B-GGUF/K2-Horizon-7B-Q4_K_M.gguf");
    assert!(ran.last().unwrap().starts_with("save_profile k2-horizon-7b-q4-k-m"), "{ran:?}");
    assert!(!ran.iter().any(|c| c.contains("launch")), "never launches");
    // Events: every step ran and finished, with the build's ninja progress
    // and the download's bytes.
    assert!(jp.steps.iter().all(|s| s.status == StepStatus::Done), "{:?}", jp.steps);
    assert_eq!(jp.steps[0].total, Some(478));
    assert_eq!(jp.steps[1].file.as_deref(), Some("K2-Horizon-7B-Q4_K_M.gguf"));
    assert!(events.iter().any(|e| e.step == 1 && e.bps == Some(1e6)));
    // The profile sits on what was built, and points at the download.
    let p = r.profile.as_ref().unwrap();
    assert_eq!(Some(&p.build.path), r.build_path.as_ref());
    assert_eq!(p.build.version.as_deref(), Some("b6000"), "from the build's own --version");
    assert_eq!(p.model.path, r.model_path);
    assert!(r.profile_path.as_ref().unwrap().is_file());
    // The download's provenance is beside it.
    let side = discovery::read_source_sidecar(r.model_path.parent().unwrap()).unwrap();
    assert_eq!(side.repo, "ngquocvinh/K2-Horizon-7B-GGUF");
    assert_eq!(side.sha, plan.sha);
    assert_eq!(side.files[0].path, "K2-Horizon-7B-Q4_K_M.gguf");

    // Again: the profile id is taken now, so the second gets -2 and the next port.
    let r2 = run_with(&f, &cfg, &plan, true, &mut |_| {}, &AtomicBool::new(false)).unwrap();
    let p2 = r2.profile.unwrap();
    assert_eq!(p2.id, "k2-horizon-7b-q4-k-m-2");
    assert_ne!(p2.server.port, p.server.port);
    std::fs::remove_dir_all(&f.root).ok();
}

#[test]
fn run_stops_at_a_failure_and_on_cancel() {
    let mut f = Fake::supported("run-fail");
    f.auth_fails = true;
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest::default()).unwrap();
    let err = run_with(&f, &cfg, &plan, false, &mut |_| {}, &AtomicBool::new(false)).unwrap_err();
    assert!(err.to_string().contains("gated"));
    assert!(!f.calls().iter().any(|c| c.starts_with("download")));

    f.auth_fails = false;
    let cancel = AtomicBool::new(true);
    let mut events = Vec::new();
    assert!(matches!(run_with(&f, &cfg, &plan, false, &mut |e| events.push(e.clone()), &cancel), Err(Error::Cancelled)));
    assert!(!f.calls().iter().any(|c| c.starts_with("download")));

    // A blocked plan, and one whose destination was changed by hand.
    let mut blocked = plan.clone();
    blocked.blocked = true;
    blocked.notes.push(Note::new(Level::Error, "x", "no room"));
    assert!(run_with(&f, &cfg, &blocked, true, &mut |_| {}, &AtomicBool::new(false)).unwrap_err().to_string().contains("no room"));
    let mut moved = plan.clone();
    if let StepAction::Download { dest, .. } = &mut moved.steps[0].action {
        *dest = PathBuf::from(r"C:\Windows\System32\evil.gguf");
    }
    assert!(run_with(&f, &cfg, &moved, true, &mut |_| {}, &AtomicBool::new(false)).is_err());
    let mut bad_sha = plan.clone();
    bad_sha.sha = "main".into();
    assert!(check_runnable(&bad_sha, true).is_err());
    std::fs::remove_dir_all(&f.root).ok();
}

#[test]
fn run_checks_the_room_on_the_drive_again() {
    const GB: u64 = 1 << 30;
    let f = Fake::k2("run-room");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    // Planned with room: 50 GiB free for a 5.2 GiB file and the build.
    f.free.set(Some(50 * GB));
    let plan = plan_with(&f, &cfg, &view, &PlanRequest { choice: Some("Q4_K_M".into()), ..Default::default() }).unwrap();
    assert!(!plan.notes.iter().any(|n| n.code == "build-space"), "{:?}", plan.notes);
    // Then something else filled the drive: nothing starts, not even the build.
    f.free.set(Some(3 * GB));
    let before = f.calls().len();
    let err = run_with(&f, &cfg, &plan, true, &mut |_| {}, &AtomicBool::new(false)).unwrap_err();
    assert!(err.to_string().contains("5.2 GiB (plus 5%)") && err.to_string().contains("3.0 GiB free now"), "{err}");
    let ran = &f.calls()[before..];
    assert!(!ran.iter().any(|c| c.starts_with("build_source") || c.starts_with("download")), "{ran:?}");
    // A drive too full for a source build's scratch space: the build step refuses.
    let mut downloads_done = plan.clone();
    for s in &mut downloads_done.steps {
        if let StepAction::Download { dest, .. } = &s.action {
            std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
            std::fs::write(dest, b"already here").unwrap();
        }
    }
    f.free.set(Some(100 << 20));
    let mut events = Vec::new();
    let err = run_with(&f, &cfg, &downloads_done, true, &mut |e| events.push(e.clone()), &AtomicBool::new(false)).unwrap_err();
    assert!(err.to_string().contains("scratch space"), "{err}");
    assert!(events.iter().any(|e| e.step == 0 && e.status == StepStatus::Failed));
    assert!(!f.calls()[before..].iter().any(|c| c.starts_with("build_source")));
    // Unknown free space: nothing to refuse on.
    f.free.set(None);
    assert!(run_with(&f, &cfg, &downloads_done, true, &mut |_| {}, &AtomicBool::new(false)).is_ok());
    std::fs::remove_dir_all(&f.root).ok();
}

#[test]
fn a_download_that_no_longer_fits_stops_before_it_starts() {
    let mut f = Fake::supported("run-room-files");
    f.info = hub::parse_model_info(&fixture("hub/model-info-unsloth__gemma-4-26B-A4B-it-GGUF.json")).unwrap();
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "unsloth/gemma-4-26B-A4B-it-GGUF", None).unwrap();
    let mmproj = view.catalog.mmproj[0].path.clone();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest { choice: Some("UD-Q4_K_XL".into()), mmproj: Some(mmproj), ..Default::default() }).unwrap();
    let sizes: Vec<u64> = plan.download_dests().iter().map(|(_, s)| *s).collect();
    // Room for both at the start; once the model is in place, what is left
    // to fetch is the projector alone.
    f.free.set(Some(sizes.iter().sum::<u64>() * 2));
    let mut events = Vec::new();
    let shrink = std::cell::Cell::new(false);
    let r = run_with(
        &f,
        &cfg,
        &plan,
        false,
        &mut |e| {
            // The drive fills while the model downloads.
            if e.step == 0 && e.status == StepStatus::Done && !shrink.get() {
                shrink.set(true);
                f.free.set(Some(sizes[1] / 2));
            }
            events.push(e.clone())
        },
        &AtomicBool::new(false),
    );
    let err = r.unwrap_err();
    assert!(err.to_string().contains("free now"), "{err}");
    let failed: Vec<usize> = events.iter().filter(|e| e.status == StepStatus::Failed).map(|e| e.step).collect();
    assert_eq!(failed, vec![1], "the projector's step fails before it downloads; {events:?}");
    assert_eq!(f.calls().iter().filter(|c| c.starts_with("download")).count(), 1, "the model only");
    std::fs::remove_dir_all(&f.root).ok();
}

#[cfg(windows)]
#[test]
fn a_source_build_counts_its_scratch_drive() {
    let f = Fake::k2("plan-scratch");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    f.free.set(Some(700 << 20));
    let plan = plan_with(&f, &cfg, &view, &PlanRequest::default()).unwrap();
    let space: Vec<&str> = plan.notes.iter().filter(|n| n.code == "build-space").map(|n| n.message.as_str()).collect();
    let src = update::source_checkout_dir();
    assert!(space.iter().any(|m| m.contains(&src.display().to_string()) && m.contains("source checkout")), "{space:?}");
    // The parts of a build on one drive are counted together.
    let ps = match &plan.steps[0].action {
        StepAction::Build { plan } => plan.clone(),
        o => panic!("{o:?}"),
    };
    let on_src_drive = build_space(&ps, &src.parent().unwrap().join("builds").join("x"));
    assert_eq!(on_src_drive.len(), 1);
    assert_eq!(on_src_drive[0].1, SOURCE_SCRATCH_BYTES + SOURCE_INSTALL_BYTES);
}

/// Stop pressed during a source build or an install: the job ends as
/// `cancelled` (the view's "stopped"), not with the build's own words.
#[test]
fn a_stopped_build_or_install_reads_cancelled() {
    let f = Fake::k2("run-stop-build");
    let cfg = f.cfg();
    let view = inspect_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF", None).unwrap();
    let plan = plan_with(&f, &cfg, &view, &PlanRequest::default()).unwrap();
    let cancel = AtomicBool::new(false);
    let mut events = Vec::new();
    let r = run_with(
        &f,
        &cfg,
        &plan,
        true,
        &mut |e| {
            // Stop while ninja runs.
            if e.stage.as_deref() == Some("build") {
                cancel.store(true, Ordering::Relaxed);
            }
            events.push(e.clone())
        },
        &cancel,
    );
    assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
    let failed = events.iter().find(|e| e.status == StepStatus::Failed).unwrap();
    assert_eq!((failed.step, failed.line.as_deref()), (0, Some("cancelled")));
    let mut jp = JobProgress::new(&plan);
    events.iter().for_each(|e| jp.apply(e));
    assert_eq!(jp.steps[0].error.as_deref(), Some("cancelled"));
    assert!(!f.calls().iter().any(|c| c.starts_with("download")), "nothing after the stop");

    // An upstream install stops too (its zips' download sees the flag).
    let mut up = plan.clone();
    let dir = cfg.install_root.clone().unwrap().join("b11046-rocm");
    let ps = PlanStep {
        action: PlanAction::InstallUpstream { tag: "b11046".into(), install_dir: dir },
        explanation: String::new(),
        needs_consent: false,
        verified: true,
        warnings: vec![],
    };
    up.steps[0] = Step { title: step_title(&ps), detail: String::new(), needs_consent: false, action: StepAction::Build { plan: ps } };
    let cancel = AtomicBool::new(false);
    let r = run_with(
        &f,
        &cfg,
        &up,
        false,
        &mut |e| {
            if e.stage.as_deref() == Some("install") {
                cancel.store(true, Ordering::Relaxed);
            }
        },
        &cancel,
    );
    assert!(matches!(r, Err(Error::Cancelled)), "{r:?}");
    assert!(f.calls().iter().any(|c| c == "install_upstream b11046"));
}

#[test]
fn needs_of_a_local_file_and_a_repo() {
    let f = Fake::k2("needs");
    let cfg = f.cfg();
    let r = needs_with(&f, &cfg, "ngquocvinh/K2-Horizon-7B-GGUF").unwrap();
    assert_eq!(r.needs.arch, "k2-horizon");
    assert_eq!(r.max_tensor_type, Some(41), "the whole header was read");
    assert!(r.usable_build.is_none());
    assert!(matches!(r.build_plan.as_ref().unwrap().step.action, PlanAction::BuildFork { .. }));

    // A local file: its own header, read from disk.
    let path = f.root.join("m.gguf");
    std::fs::write(&path, crate::gguf::testing::split_shard(Some("gemma4"), 0, 1, &[("t", 1), ("u", 40)])).unwrap();
    let mut ok = Fake::supported("needs-local");
    ok.builds.truncate(1);
    let r = needs_with(&ok, &cfg, path.to_str().unwrap()).unwrap();
    assert_eq!(r.needs.arch, "gemma4");
    assert_eq!(r.needs.max_type_id, Some(40));
    assert_eq!(r.usable_build.as_deref(), Some(ok.builds[0].path.as_path()));
    assert!(needs_with(&ok, &cfg, "no such thing").is_err());
    std::fs::remove_dir_all(&f.root).ok();
}

#[test]
fn folders_compare_as_windows_does() {
    assert!(under(Path::new(r"E:\Models\sub"), Path::new(r"e:\models")));
    assert!(under(Path::new("E:/models"), Path::new(r"E:\models\")));
    assert!(!under(Path::new(r"E:\models2"), Path::new(r"E:\models")));
}

#[test]
fn model_folders_can_be_added() {
    let root = tmp("roots");
    let mut cfg = Config::default_for_machine();
    cfg.model_roots = vec![root.join("a")];
    assert!(add_model_root(&mut cfg, &root.join("b")).unwrap());
    assert!(root.join("b").is_dir(), "created");
    assert!(!add_model_root(&mut cfg, &root.join("B\\")).unwrap(), "the same folder, spelled otherwise");
    assert!(add_model_root(&mut cfg, Path::new("relative")).is_err());
    std::fs::write(root.join("f"), b"x").unwrap();
    assert!(add_model_root(&mut cfg, &root.join("f")).is_err());
    assert_eq!(cfg.model_roots.len(), 2);
    let infos = model_roots(&cfg);
    assert!(!infos[0].exists && infos[1].exists);
    std::fs::remove_dir_all(&root).ok();
}

/// A drive dedicated to models: `E:\` is a folder like any other (the
/// separator was trimmed off, leaving `E:`, which is not absolute).
#[cfg(windows)]
#[test]
fn a_drive_root_can_be_a_model_folder() {
    // The temp folder's drive: one that exists here.
    let drive = std::env::temp_dir().to_string_lossy()[..2].to_ascii_lowercase();
    let mut cfg = Config::default_for_machine();
    cfg.model_roots.clear();
    assert!(add_model_root(&mut cfg, Path::new(&format!("{drive}\\"))).unwrap());
    let upper = drive.to_ascii_uppercase();
    assert_eq!(cfg.model_roots, vec![PathBuf::from(format!("{upper}\\"))]);
    assert!(cfg.model_roots[0].is_absolute());
    // The same drive, spelled otherwise: already there.
    for spelled in [format!("{upper}/"), upper.clone(), format!(" {upper}\\\\ ")] {
        assert!(!add_model_root(&mut cfg, Path::new(&spelled)).unwrap(), "{spelled}");
    }
    assert_eq!(cfg.model_roots.len(), 1);
    // Downloads land in <drive>\<owner>\<repo>, which the scan counts as inside it.
    let dest = catalog::dest_path(&cfg.model_roots[0], "IFM/K2-Horizon-7B-GGUF", "K2-Horizon-7B-Q4_K_M.gguf");
    assert_eq!(dest, PathBuf::from(format!(r"{upper}\IFM\K2-Horizon-7B-GGUF\K2-Horizon-7B-Q4_K_M.gguf")));
    assert!(under(&dest, &cfg.model_roots[0]));
    assert!(add_model_root(&mut cfg, Path::new(r"\no-drive")).is_err(), "a path without its drive");
}

#[test]
fn default_projector_and_draft() {
    let info = hub::parse_model_info(&fixture("hub/model-info-unsloth__gemma-4-26B-A4B-it-GGUF.json")).unwrap();
    let cat = catalog::classify(&info.siblings);
    assert_eq!(default_mmproj(&cat).map(|f| f.name()), Some("mmproj-F16.gguf"));
    let q8 = cat.choices.iter().find(|c| c.label == "Q8_0").unwrap();
    assert_eq!(default_draft(&cat, q8).map(|f| f.path.as_str()), Some("MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"));
    let bf16 = cat.choices.iter().find(|c| c.label == "BF16").unwrap();
    assert_eq!(default_draft(&cat, bf16).map(|f| f.path.as_str()), Some("MTP/mtp-gemma-4-26B-A4B-it-BF16.gguf"));
    let ud = cat.choices.iter().find(|c| c.label == "UD-Q4_K_XL").unwrap();
    assert!(default_draft(&cat, ud).unwrap().name().contains("Q8_0"), "no draft for this quant: the Q8_0 one");
    assert_eq!(default_mmproj(&Catalog::default()), None);
}
