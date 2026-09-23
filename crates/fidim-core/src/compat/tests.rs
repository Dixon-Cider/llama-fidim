use super::github::*;
use super::plan::*;
use super::*;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");
const IFM_SHA: &str = "42adf019f76013dac873b5b43950d54d5ab27216";

fn fixture(rel: &str) -> String {
    std::fs::read_to_string(format!("{FIX}/{rel}")).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn src_caps(dir: &str) -> SourceCaps {
    caps_from_dir(Path::new(&format!("{FIX}/llama-src/{dir}"))).unwrap()
}

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("fidim-compat-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn k2() -> ModelNeeds {
    ModelNeeds::new("k2-horizon", Some("k2-horizon".into()), None, Engine::LlamaServer)
}

// ------------------------------------------------------------- parsers ----

#[test]
fn arch_tables_at_three_layouts() {
    let up = parse_arch_names(&fixture("llama-src/b11046/src/llama-arch.cpp"));
    assert_eq!(up.len(), 153, "b11046 names 153 architectures, the (unknown) sentinel included");
    assert_eq!(&up[..3], ["clip", "llama", "llama4"]);
    for a in ["gemma4", "qwen35", "bert", "gptj", "hrm_text"] {
        assert!(up.iter().any(|x| x == a), "{a}");
    }
    assert!(!up.iter().any(|x| x == "k2-horizon"), "upstream has no K2 Horizon");
    assert!(!up.iter().any(|x| x == "diffusion-gemma"), "only Unsloth's mix merges the DiffusionGemma PR");

    let ifm = parse_arch_names(&fixture("llama-src/ifm-ai-42adf019/src/llama-arch.cpp"));
    assert!(ifm.iter().any(|x| x == "k2-horizon"));
    assert!(ifm.len() < up.len(), "the fork is behind upstream: fewer names");

    let old = parse_arch_names(&fixture("llama-src/b4400/src/llama.cpp"));
    assert_eq!(old.len(), 55);
    assert!(old.iter().any(|x| x == "llama") && !old.iter().any(|x| x == "gemma4"));

    assert!(parse_arch_names("int main() {}").is_empty());
    assert!(parse_arch_names("LLM_ARCH_NAMES = {\n};").is_empty());
}

#[test]
fn pre_tokenizer_tables() {
    let up = parse_pre_names(&fixture("llama-src/b11046/src/llama-vocab.cpp"));
    assert_eq!(up.len(), 90);
    for p in ["default", "llama3", "llama-bpe", "qwen35", "kimi-k2", "gpt-4o"] {
        assert!(up.iter().any(|x| x == p), "{p}");
    }
    assert!(!up.iter().any(|x| x == "k2-horizon"));
    assert!(up.windows(2).all(|w| w[0] < w[1]), "sorted and deduplicated");
    let ifm = parse_pre_names(&fixture("llama-src/ifm-ai-42adf019/src/llama-vocab.cpp"));
    assert!(ifm.iter().any(|x| x == "k2-horizon"));
    let old = parse_pre_names(&fixture("llama-src/b4400/src/llama.cpp"));
    assert!(old.iter().any(|x| x == "llama3") && old.len() >= 35, "{}", old.len());
    assert!(parse_pre_names(&fixture("llama-src/b4400/src/llama-vocab.cpp")).is_empty());
}

#[test]
fn ggml_type_counts() {
    assert_eq!(parse_ggml_type_count(&fixture("llama-src/b11046/ggml/include/ggml.h")), Some(43));
    assert_eq!(parse_ggml_type_count(&fixture("llama-src/ifm-ai-42adf019/ggml/include/ggml.h")), Some(43));
    assert_eq!(parse_ggml_type_count(&fixture("llama-src/b4400/ggml/include/ggml.h")), Some(39));
    // A tree that leaves the count implicit: its position in the enum.
    let implicit = "enum ggml_type {\n    GGML_TYPE_F32 = 0,\n    GGML_TYPE_F16 = 1, // half\n    /* gap */ GGML_TYPE_Q4_0 = 2,\n    \
                    GGML_TYPE_Q8_0 = 8,\n    GGML_TYPE_X,\n    GGML_TYPE_COUNT,\n};\n";
    assert_eq!(parse_ggml_type_count(implicit), Some(10));
    assert_eq!(parse_ggml_type_count("no enum here"), None);
}

#[test]
fn source_tables_answer_exactly() {
    let up = src_caps("b11046");
    let ifm = src_caps("ifm-ai-42adf019");
    assert_eq!(up.files, ["src/llama-arch.cpp", "src/llama-vocab.cpp", "ggml/include/ggml.h"]);
    assert_eq!(up.support(&k2()), Support::No { missing: vec![Missing::Arch, Missing::TokenizerPre("k2-horizon".into())] });
    assert_eq!(ifm.support(&k2()), Support::Yes);
    // kingjones777's ROCmFP4 files: the fork's architecture, ROCmFPX's types.
    let fp4 = ModelNeeds { max_type_id: Some(101), ..k2() };
    assert_eq!(ifm.support(&fp4), Support::No { missing: vec![Missing::TensorType(101)] });
    assert_eq!(ifm.support(&ModelNeeds { max_type_id: Some(42), ..k2() }), Support::Yes, "Q2_0 = 42 < 43");
    // Named in the table but no graph.
    let gptj = ModelNeeds::new("gptj", None, None, Engine::LlamaServer);
    assert_eq!(up.support(&gptj), Support::No { missing: vec![Missing::Arch] });
    // "default" is every build's fallback, listed or not.
    let q = ModelNeeds::new("qwen35", Some("default".into()), None, Engine::LlamaServer);
    assert_eq!(SourceCaps { tokenizer_pres: vec![], ..up.clone() }.support(&q), Support::Yes);
    // No ggml.h: the type check is Unknown, the rest still decides.
    let no_h = SourceCaps { ggml_type_count: None, ..ifm.clone() };
    assert!(matches!(no_h.support(&fp4), Support::Unknown(_)));
    assert!(no_h.support(&ModelNeeds { max_type_id: Some(1), ..gptj.clone() }).is_no());

    // The older layout: tables in src/llama.cpp, a src/llama-vocab.cpp
    // without them.
    let old = src_caps("b4400");
    assert_eq!(old.files, ["src/llama.cpp", "ggml/include/ggml.h"]);
    assert_eq!(old.ggml_type_count, Some(39));
    let llama3 = ModelNeeds::new("llama", Some("llama3".into()), Some(30), Engine::LlamaServer);
    assert_eq!(old.support(&llama3), Support::Yes);

    assert_eq!(describe_missing(&k2(), &[Missing::Arch, Missing::TokenizerPre("k2-horizon".into())]), "architecture 'k2-horizon' and pre-tokenizer 'k2-horizon'");
    assert_eq!(
        describe_missing(&k2(), &[Missing::Arch, Missing::TokenizerPre("x".into()), Missing::TensorType(101)]),
        "architecture 'k2-horizon', pre-tokenizer 'x' and ggml tensor type 101"
    );
    assert!(describe_missing(&gptj, &[Missing::Arch]).contains("named but not implemented"));
    let json = serde_json::to_value(Support::No { missing: vec![Missing::Arch, Missing::TensorType(101)] }).unwrap();
    assert_eq!(json, serde_json::json!({"support": "no", "detail": {"missing": [{"kind": "arch"}, {"kind": "tensor_type", "value": 101}]}}));
    assert_eq!(serde_json::to_value(Support::Unknown("x".into())).unwrap(), serde_json::json!({"support": "unknown", "detail": "x"}));
}

/// A fetcher over the fixture trees, counting requests.
fn fixture_fetch<'a>(dir: &'a str, calls: &'a AtomicUsize, missing: &'a [&'a str]) -> impl FnMut(&str) -> Result<Option<String>> + 'a {
    move |url: &str| {
        calls.fetch_add(1, Ordering::SeqCst);
        assert!(url.starts_with("https://raw.githubusercontent.com/"), "{url}");
        // The ref may contain `/` (a branch): match the file by its tail,
        // longest names first.
        let files = ["src/llama-arch.cpp", "src/llama-vocab.cpp", "src/llama.cpp", "ggml/include/ggml.h", "llama.cpp", "ggml.h"];
        let rel = files.iter().find(|f| url.ends_with(&format!("/{f}"))).unwrap().to_string();
        if missing.contains(&rel.as_str()) {
            return Ok(None);
        }
        Ok(std::fs::read_to_string(format!("{FIX}/llama-src/{dir}/{rel}")).ok())
    }
}

#[test]
fn source_caps_are_fetched_once_per_commit() {
    let root = tmp("srccache");
    let calls = AtomicUsize::new(0);
    let mut fetch = fixture_fetch("ifm-ai-42adf019", &calls, &[]);
    let c = source_caps_with(&root, "ifm-ai", "llama.cpp", IFM_SHA, &mut fetch).unwrap();
    assert!(c.arches.iter().any(|a| a == "k2-horizon"));
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert!(root.join("ifm-ai").join("llama.cpp").join(format!("{IFM_SHA}.json")).is_file());
    // Cached: no fetch at all, and any case of the names hits the same file.
    let c2 = source_caps_with(&root, "IFM-AI", "llama.cpp", IFM_SHA, &mut fetch).unwrap();
    assert_eq!(c, c2);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    // `--version` prints nine digits: the cache answers a prefix, offline.
    assert_eq!(cached_source_caps_in(&root, "ifm-ai", "llama.cpp", "42adf019f"), Some(c.clone()));
    assert_eq!(cached_source_caps_in(&root, "ifm-ai", "llama.cpp", "42adf01"), Some(c.clone()));
    assert_eq!(cached_source_caps_in(&root, "ifm-ai", "llama.cpp", "42adf0"), None, "too short");
    assert_eq!(cached_source_caps_in(&root, "ifm-ai", "llama.cpp", "deadbeef"), None);
    // Two cached commits sharing the prefix: no answer rather than a guess.
    std::fs::write(root.join("ifm-ai").join("llama.cpp").join("42adf01900000000000000000000000000000000.json"), "{}").unwrap();
    assert_eq!(cached_source_caps_in(&root, "ifm-ai", "llama.cpp", "42adf01"), None);

    // A branch moves: never cached.
    let before = calls.load(Ordering::SeqCst);
    source_caps_with(&root, "ifm-ai", "llama.cpp", "model/K2Horizon", &mut fetch).unwrap();
    source_caps_with(&root, "ifm-ai", "llama.cpp", "model/K2Horizon", &mut fetch).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), before + 6);

    // The old layout falls back file by file, each file fetched once.
    let calls_old = AtomicUsize::new(0);
    let mut old = fixture_fetch("b4400", &calls_old, &["src/llama-arch.cpp"]);
    let c = source_caps_with(&root, "ggml-org", "llama.cpp", "b4400", &mut old).unwrap();
    assert_eq!(c.arches.len(), 55);
    assert!(c.tokenizer_pres.iter().any(|p| p == "llama3"));
    assert_eq!(calls_old.load(Ordering::SeqCst), 4, "arch 404, src/llama.cpp, src/llama-vocab.cpp, ggml.h");
    assert!(root.join("ggml-org").join("llama.cpp").join("b4400.json").is_file(), "b<n> tags are immutable");

    // Untrusted names never reach a URL.
    for (o, r, s) in [("a/b", "llama.cpp", IFM_SHA), ("ok", "..", IFM_SHA), ("ok", "llama.cpp", "../x"), ("ok", "llama.cpp", "a b")] {
        assert!(source_caps_with(&root, o, r, s, &mut |_: &str| panic!("fetched")).is_err(), "{o} {r} {s}");
    }
    // Not a llama.cpp tree at all.
    let e = source_caps_with(&root, "someone", "llama.cpp-notes", "0123456789abcdef0123456789abcdef01234567", &mut |_: &str| Ok(None))
        .unwrap_err()
        .to_string();
    assert!(e.contains("no LLM_ARCH_NAMES"), "{e}");
    std::fs::remove_dir_all(root).ok();
}

// ---------------------------------------------------------- build probe ----

/// A build directory whose `bin/<binary>` holds these NUL-separated strings.
fn fake_build(name: &str, binary: &str, strings: &[&str]) -> PathBuf {
    let dir = tmp(name);
    let bin = dir.join("bin");
    std::fs::create_dir_all(&bin).unwrap();
    let mut bytes = b"MZ\x90\x00\x03\x00\x00\x00".to_vec();
    for s in strings {
        bytes.extend_from_slice(s.as_bytes());
        bytes.push(0);
    }
    bytes.extend_from_slice(b"\xff\xfe\x00");
    std::fs::write(bin.join(binary), bytes).unwrap();
    dir
}

#[test]
fn dll_probe_reads_the_name_tables() {
    let cache = tmp("probe-cache");
    // b10984's shape: the arch table, "bert" merged into the tail of
    // "eurobert", "qwen2" into "rwkv6qwen2", no "default" literal.
    let up = fake_build("probe-up", "llama.dll", &["llama", "eurobert", "rwkv6qwen2", "gemma4", "qwen35", "llama-bpe", "gptj"]);
    let yes = |arch: &str, pre: Option<&str>| {
        probe_build_in(&up, None, &ModelNeeds::new(arch, pre.map(String::from), None, Engine::LlamaServer), &cache)
    };
    assert_eq!(yes("gemma4", None), Support::Yes);
    assert_eq!(yes("bert", None), Support::Yes, "a tail-merged name is there");
    assert_eq!(yes("qwen2", Some("llama-bpe")), Support::Yes);
    assert_eq!(yes("qwen35", Some("default")), Support::Yes, "default is never looked for");
    assert_eq!(yes("gptj", None), Support::No { missing: vec![Missing::Arch] }, "named, not implemented");
    assert_eq!(yes("k2-horizon", Some("k2-horizon")), Support::No { missing: vec![Missing::Arch, Missing::TokenizerPre("k2-horizon".into())] });
    assert_eq!(yes("k2-horizon", Some("llama-bpe")), Support::No { missing: vec![Missing::Arch] });
    assert_eq!(yes("gemm", None), Support::No { missing: vec![Missing::Arch] }, "a prefix is not a name");
    // Architecture there, pre-tokenizer not found: a compare may have been
    // inlined, so only Unknown.
    match yes("gemma4", Some("brand-new-pre")) {
        Support::Unknown(m) => assert!(m.contains("'brand-new-pre'") && m.contains("llama.dll"), "{m}"),
        o => panic!("{o:?}"),
    }
    // A tensor type with no source table known offline: Unknown.
    let typed = ModelNeeds::new("gemma4", None, Some(30), Engine::LlamaServer);
    assert!(matches!(probe_build_in(&up, Some("9cc33944f"), &typed, &cache), Support::Unknown(m) if m.contains("tensor type 30")));

    // A static build: the server executable holds the tables.
    let st = fake_build("probe-static", "llama-server.exe", &["k2-horizon"]);
    assert_eq!(probe_build_in(&st, None, &k2(), &cache), Support::Yes);
    let none = tmp("probe-none");
    assert!(matches!(probe_build_in(&none, None, &k2(), &cache), Support::Unknown(m) if m.contains("no llama.dll")));

    // Diffusion needs the runner first.
    let dg = fake_build("probe-dg", "llama.dll", &["diffusion-gemma", "gemma4"]);
    let dg_needs = ModelNeeds::new("diffusion-gemma", None, None, Engine::DiffusionGemma);
    assert_eq!(probe_build_in(&dg, None, &dg_needs, &cache), Support::No { missing: vec![Missing::Arch] });
    std::fs::write(dg.join("bin").join(RUNNER_EXE), b"").unwrap();
    assert_eq!(probe_build_in(&dg, None, &dg_needs, &cache), Support::Yes);

    // A source build's manifest tables beat the DLL heuristic.
    let git = fake_build("probe-git", "llama.dll", &["gemma4"]);
    let caps = src_caps("ifm-ai-42adf019");
    std::fs::write(
        git.join(crate::update::MANIFEST_NAME),
        serde_json::json!({"tag": "t", "source": "git-ref", "installed_at_unix": 1, "assets": [],
            "verify": {"version": null, "commit": null, "devices": [], "hip_ok": false, "detail": ""},
            "channel": "git", "caps": caps}).to_string(),
    )
    .unwrap();
    assert_eq!(probe_build_in(&git, None, &k2(), &cache), Support::Yes);
    assert_eq!(
        probe_build_in(&git, None, &ModelNeeds { max_type_id: Some(101), ..k2() }, &cache),
        Support::No { missing: vec![Missing::TensorType(101)] }
    );

    // Types from the cached source tables of the build's commit: a git
    // build by its manifest, an upstream one by what --version printed.
    let fork = fake_build("probe-fork", "llama.dll", &["k2-horizon"]);
    std::fs::write(
        fork.join(crate::update::MANIFEST_NAME),
        serde_json::json!({"tag": "t", "source": "git-ref", "installed_at_unix": 1, "assets": [],
            "verify": {"version": null, "commit": null, "devices": [], "hip_ok": false, "detail": ""},
            "channel": "git", "git": {"remote": "https://github.com/ifm-ai/llama.cpp", "git_ref": "model/K2Horizon", "commit": IFM_SHA, "label": "ifm-ai K2Horizon fork"}})
        .to_string(),
    )
    .unwrap();
    let t101 = ModelNeeds { max_type_id: Some(101), ..k2() };
    assert!(matches!(probe_build_in(&fork, None, &t101, &cache), Support::Unknown(_)), "nothing cached yet");
    let calls = AtomicUsize::new(0);
    source_caps_with(&cache, "ifm-ai", "llama.cpp", IFM_SHA, &mut fixture_fetch("ifm-ai-42adf019", &calls, &[])).unwrap();
    assert_eq!(probe_build_in(&fork, None, &t101, &cache), Support::No { missing: vec![Missing::TensorType(101)] });
    assert_eq!(probe_build_in(&fork, None, &ModelNeeds { max_type_id: Some(12), ..k2() }, &cache), Support::Yes);
    let upstream_typed = fake_build("probe-up-typed", "llama.dll", &["gemma4"]);
    source_caps_with(&cache, "ggml-org", "llama.cpp", "60081bb2b5b3294165a4d67c5cbeebe74c868014", &mut fixture_fetch("b11046", &calls, &[])).unwrap();
    assert_eq!(probe_build_in(&upstream_typed, Some("60081bb2b"), &typed, &cache), Support::Yes);
    for d in [cache, up, st, none, dg, git, fork, upstream_typed] {
        std::fs::remove_dir_all(d).ok();
    }
}

#[test]
fn needs_from_a_header() {
    let header = |meta: serde_json::Value, arch: Option<&str>| -> GgufHeader {
        serde_json::from_value(serde_json::json!({
            "path": "E:/m.gguf", "file_size": 1, "gguf_version": 3, "tensor_count": 0,
            "architecture": arch, "metadata": meta
        }))
        .unwrap()
    };
    let bpe = header(serde_json::json!({"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "k2-horizon"}), Some("k2-horizon"));
    assert_eq!(ModelNeeds::from_header(&bpe), Some(k2()));
    // Only BPE vocabularies have pre-tokenizers.
    let spm = header(serde_json::json!({"tokenizer.ggml.model": "llama", "tokenizer.ggml.pre": "default"}), Some("gemma4"));
    assert_eq!(ModelNeeds::from_header(&spm).unwrap().tokenizer_pre, None);
    let empty = header(serde_json::json!({"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": ""}), Some("qwen35"));
    assert_eq!(ModelNeeds::from_header(&empty).unwrap().tokenizer_pre, None, "empty only warns in llama.cpp");
    assert_eq!(ModelNeeds::from_header(&header(serde_json::json!({}), None)), None);
    let dg = header(serde_json::json!({}), Some("diffusion-gemma"));
    assert_eq!(ModelNeeds::from_header(&dg).unwrap().engine, Engine::DiffusionGemma);
}

// ------------------------------------------------------------ card refs ----

#[test]
fn card_refs_take_only_llama_cpp_links() {
    // IFM's K2 Horizon cards write the link bare at the end of a sentence.
    let k2_card = "> **Compatibility:** These models require a version of `llama.cpp` containing K2 Horizon \
                   architecture support. PR to llama.cpp is in progress. MBZUAI-IFM fork of llama.cpp is in \
                   https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon\n";
    assert_eq!(
        card_refs(k2_card),
        vec![GitRef { owner: "MBZUAI-IFM".into(), repo: "llama.cpp".into(), kind: RefKind::Branch("model/K2Horizon".into()) }]
    );
    let card = "Use [llama.cpp](https://github.com/ggml-org/llama.cpp) or the [fork](https://github.com/Some-Org/llama.cpp/tree/feat/new-arch).\n\
        Needs https://github.com/ggml-org/llama.cpp/pull/27752/files and github.com/x/ik_Llama.CPP.git.\n\
        Pinned: https://www.github.com/y/llama.cpp/commit/42ADF019F76013DAC. Also https://github.com/ggml-org/llama.cpp again.\n\
        Not these: https://github.com/huggingface/transformers/tree/main https://github.com/a/llama.cpp/blob/master/x.md \
        https://github.com/a/llama.cpp/tree/../../etc https://github.com/a/llama.cpp/pull/0 https://github.com/a/llama.cpp/commit/abc12 \
        https://github.com/b/llama.cpp/tree/x;rm%20-rf https://github.com/c/llama.cpp/tree/-evil\n";
    let refs = card_refs(card);
    let want = vec![
        GitRef { owner: "ggml-org".into(), repo: "llama.cpp".into(), kind: RefKind::Repo },
        GitRef { owner: "Some-Org".into(), repo: "llama.cpp".into(), kind: RefKind::Branch("feat/new-arch".into()) },
        GitRef { owner: "ggml-org".into(), repo: "llama.cpp".into(), kind: RefKind::Pull(27752) },
        GitRef { owner: "x".into(), repo: "ik_Llama.CPP".into(), kind: RefKind::Repo },
        GitRef { owner: "y".into(), repo: "llama.cpp".into(), kind: RefKind::Commit("42adf019f76013dac".into()) },
        // blob/ is not a ref kind: the repository itself.
        GitRef { owner: "a".into(), repo: "llama.cpp".into(), kind: RefKind::Repo },
        GitRef { owner: "b".into(), repo: "llama.cpp".into(), kind: RefKind::Branch("x".into()) },
    ];
    assert_eq!(refs, want);
    assert!(refs[0].is_upstream_repo() && !refs[1].is_upstream_repo());
    assert_eq!(refs[1].url(), "https://github.com/Some-Org/llama.cpp/tree/feat/new-arch");
    assert_eq!(refs[2].url(), "https://github.com/ggml-org/llama.cpp/pull/27752");
    let many: String = (0..40).map(|i| format!("https://github.com/o{i}/llama.cpp ")).collect();
    assert_eq!(card_refs(&many).len(), 20, "capped");
    assert!(card_refs("").is_empty());

    assert_eq!(owner_repo_from_url("https://github.com/ifm-ai/llama.cpp"), Some(("ifm-ai".into(), "llama.cpp".into())));
    assert_eq!(owner_repo_from_url("https://github.com/ifm-ai/llama.cpp.git/"), Some(("ifm-ai".into(), "llama.cpp".into())));
    assert_eq!(owner_repo_from_url("https://gitlab.com/ifm-ai/llama.cpp"), None);
    assert_eq!(owner_repo_from_url("https://github.com/ifm-ai/llama.cpp/tree/x"), None);
}

// -------------------------------------------------------- GitHub parsers ----

#[test]
fn github_parsers_on_captured_answers() {
    let repo = parse_repo(&fixture("github/repo-mbzuai-ifm-llama.cpp.json")).unwrap();
    assert_eq!(repo.full_name, "ifm-ai/llama.cpp", "the transferred repository's new name");
    assert!(repo.fork);
    assert_eq!(repo.source.as_deref(), Some("ggml-org/llama.cpp"));
    assert_eq!(repo.default_branch, "master");

    let c = parse_compare(&fixture("github/compare-master-ifm-ai-42adf019.json")).unwrap();
    assert_eq!((c.ahead_by, c.behind_by, c.total_commits), (10, 380, 10));
    assert_eq!(c.status, "diverged");
    assert_eq!(c.merge_base.as_deref(), Some("4e97ac86ebe2c4cb8212d98d2641ad6768810896"));
    assert_eq!(c.subjects.len(), 10);
    assert_eq!(c.subjects[0], "model: K2 Horizon gguf conversion code");
    assert_eq!(c.subjects[9], "Merge pull request #1 from another-contributor/k2-horizon-msvc-pretokenizer");

    let pr = parse_pull(&fixture("github/pull-27752.json")).unwrap();
    assert_eq!(pr.number, 27752);
    assert_eq!(pr.title, "model : add GLM-5.3-Flash (glm5next)");
    assert_eq!((pr.state.as_str(), pr.draft, pr.merged), ("open", false, false));
    assert_eq!(pr.mergeable_state.as_deref(), Some("dirty"));
    assert_eq!(pr.head_sha, "1d0c76f3c6d030fdfc269aa27db6334ea2834cec");
    assert_eq!(pr.head_repo.as_deref(), Some("a-contributor/llama.cpp"));
    assert_eq!(pr.base_ref, "master");

    let hits = parse_search_prs(&fixture("github/search-prs-inkling.json")).unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!((hits[0].number, hits[0].draft, hits[0].state.as_str()), (25731, true, "open"));
    assert_eq!((hits[1].number, hits[1].merged), (24523, false));
    assert!(parse_search_prs(&fixture("github/search-prs-k2-horizon.json")).unwrap().is_empty());
    assert!(parse_search_prs("{}").is_err());
    assert_eq!(
        crate::compat::plan::unsloth_merged_prs(&serde_json::from_str::<serde_json::Value>(&fixture("github/release-unsloth-b11030.json")).unwrap()["body"].as_str().unwrap().to_string()),
        vec![24423, 25731, 27754]
    );
}

// ------------------------------------------------------ fake GitHub API ----

#[derive(Clone)]
struct Resp {
    status: u16,
    headers: Vec<(String, String)>,
    body: String,
}

fn ok(body: &str) -> Resp {
    Resp { status: 200, headers: vec![], body: body.into() }
}

/// What the fake saw: path (with query) and lowercase headers.
type Seen = Arc<Mutex<Vec<(String, HashMap<String, String>)>>>;

/// A one-thread HTTP/1.1 server on 127.0.0.1: each path answers from its
/// queue (the last answer repeats), anything else is 404. `{base}` in a
/// header is replaced with the server's own address.
fn fake_github(routes: Vec<(&str, Vec<Resp>)>) -> (String, Seen) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let mut routes: HashMap<String, Vec<Resp>> = routes.into_iter().map(|(p, r)| (p.to_string(), r)).collect();
    let (seen2, base2) = (seen.clone(), base.clone());
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                match s.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => buf.extend_from_slice(&chunk[..n]),
                }
            }
            let text = String::from_utf8_lossy(&buf).into_owned();
            let mut lines = text.split("\r\n");
            let path = lines.next().unwrap_or("").split(' ').nth(1).unwrap_or("").to_string();
            let headers: HashMap<String, String> = lines
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
                .collect();
            seen2.lock().unwrap().push((path.clone(), headers));
            let resp = match routes.get_mut(&path) {
                Some(q) if q.len() > 1 => q.remove(0),
                Some(q) => q[0].clone(),
                None => Resp { status: 404, headers: vec![], body: r#"{"message":"Not Found"}"#.into() },
            };
            let mut head = format!("HTTP/1.1 {} X\r\nContent-Length: {}\r\nConnection: close\r\n", resp.status, resp.body.len());
            for (k, v) in &resp.headers {
                head.push_str(&format!("{k}: {}\r\n", v.replace("{base}", &base2)));
            }
            head.push_str("\r\n");
            let _ = s.write_all(head.as_bytes());
            let _ = s.write_all(resp.body.as_bytes());
        }
    });
    (base, seen)
}

fn api(base: &str, cache: Option<PathBuf>, fresh: u64) -> Api {
    Api::with_base(base, None, cache, Duration::from_secs(fresh))
}

#[test]
fn api_cache_revalidates_with_etags() {
    let cache = tmp("api-cache");
    let reset = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() + 600).to_string();
    let limited = Resp {
        status: 403,
        headers: vec![("X-RateLimit-Remaining".into(), "0".into()), ("X-RateLimit-Reset".into(), reset)],
        body: r#"{"message":"API rate limit exceeded for 127.0.0.1."}"#.into(),
    };
    let (base, seen) = fake_github(vec![
        ("/r", vec![Resp { status: 200, headers: vec![("ETag".into(), "\"e1\"".into())], body: "one".into() }, Resp { status: 304, headers: vec![], body: String::new() }]),
        ("/limited", vec![limited.clone()]),
        ("/stale", vec![Resp { status: 200, headers: vec![("ETag".into(), "\"s\"".into())], body: "old".into() }, limited]),
    ]);
    let count = || seen.lock().unwrap().len();
    let a = api(&base, Some(cache.clone()), 0);
    assert_eq!(a.get("/r", "application/json").unwrap().as_deref(), Some("one"));
    assert!(!seen.lock().unwrap()[0].1.contains_key("if-none-match"));
    assert_eq!(a.get("/r", "application/json").unwrap().as_deref(), Some("one"), "304: the cached body");
    assert_eq!(seen.lock().unwrap()[1].1.get("if-none-match").map(String::as_str), Some("\"e1\""));
    // Fresh answers cost no request; a different Accept is another entry.
    let fresh = api(&base, Some(cache.clone()), 300);
    assert_eq!(fresh.get("/r", "application/json").unwrap().as_deref(), Some("one"));
    assert_eq!(count(), 2);
    // 404 is an answer (None), cached like any other.
    assert_eq!(fresh.get("/missing", "application/json").unwrap(), None);
    assert_eq!(fresh.get("/missing", "application/json").unwrap(), None);
    assert_eq!(count(), 3);
    // The rate limit names its fix; a stale copy beats the error.
    let e = a.get("/limited", "application/json").unwrap_err().to_string();
    assert!(e.contains("rate limit") && e.contains("GITHUB_TOKEN") && e.contains("resets in 10 min"), "{e}");
    assert_eq!(a.get("/stale", "application/json").unwrap().as_deref(), Some("old"));
    assert_eq!(a.get("/stale", "application/json").unwrap().as_deref(), Some("old"), "403 with a cached copy");
    // A token goes along as a bearer.
    let t = Api::with_base(&base, Some("tok".into()), None, Duration::ZERO);
    let _ = t.get("/missing", "application/json");
    assert_eq!(seen.lock().unwrap().last().unwrap().1.get("authorization").map(String::as_str), Some("Bearer tok"));
    std::fs::remove_dir_all(cache).ok();
}

#[test]
fn resolve_the_k2_card_link() {
    let head = format!("/repos/ggml-org/llama.cpp/compare/master...ifm-ai:llama.cpp:{IFM_SHA}?per_page=10&page=1");
    let (base, seen) = fake_github(vec![
        // The old name answers with a redirect to the repository id, as GitHub does.
        ("/repos/MBZUAI-IFM/llama.cpp", vec![Resp { status: 301, headers: vec![("Location".into(), "{base}/repositories/1349334965".into())], body: String::new() }]),
        ("/repositories/1349334965", vec![ok(&fixture("github/repo-mbzuai-ifm-llama.cpp.json"))]),
        ("/repos/ifm-ai/llama.cpp/commits/model%2FK2Horizon", vec![ok(IFM_SHA)]),
        (head.as_str(), vec![ok(&fixture("github/compare-master-ifm-ai-42adf019.json"))]),
    ]);
    let r = card_refs("https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon").remove(0);
    let res = resolve_ref_with(&api(&base, None, 0), &r).unwrap();
    assert_eq!((res.owner.as_str(), res.repo.as_str()), ("ifm-ai", "llama.cpp"));
    assert!(res.redirected && res.is_fork_of_ggml && !res.is_upstream);
    assert_eq!(res.remote_url, "https://github.com/ifm-ai/llama.cpp");
    assert_eq!((res.git_ref.as_str(), res.sha.as_str()), ("model/K2Horizon", IFM_SHA));
    assert_eq!((res.ahead_by, res.behind_by), (Some(10), Some(380)));
    assert_eq!(res.head_subjects.len(), 10);
    assert!(res.pr.is_none());
    let src = res.source_ref();
    assert_eq!(src.label, "ifm-ai K2Horizon fork");
    assert_eq!(src.git_source().display(), "ifm-ai K2Horizon fork @42adf01");
    let seen = seen.lock().unwrap();
    let sha_req = seen.iter().find(|(p, _)| p.contains("/commits/")).unwrap();
    assert_eq!(sha_req.1.get("accept").map(String::as_str), Some("application/vnd.github.sha"));
}

#[test]
fn resolve_an_upstream_pull_request() {
    let sha = "1d0c76f3c6d030fdfc269aa27db6334ea2834cec";
    let cmp = format!("/repos/ggml-org/llama.cpp/compare/master...{sha}?per_page=10&page=1");
    let last = format!("/repos/ggml-org/llama.cpp/compare/master...{sha}?per_page=10&page=2");
    let commits = |n: usize, from: usize| -> Vec<serde_json::Value> {
        (from..from + n).map(|i| serde_json::json!({"sha": format!("{i:040}"), "commit": {"message": format!("step {i}\n\nbody")}})).collect()
    };
    let page = |c: Vec<serde_json::Value>| {
        serde_json::json!({"status": "diverged", "ahead_by": 11, "behind_by": 90, "total_commits": 11,
            "merge_base_commit": {"sha": "e107984bcffcfd701e82738092a2b000b6fda7a2"}, "commits": c})
        .to_string()
    };
    let (base, _) = fake_github(vec![
        ("/repos/ggml-org/llama.cpp", vec![ok(r#"{"full_name":"ggml-org/llama.cpp","fork":false,"default_branch":"master"}"#)]),
        ("/repos/ggml-org/llama.cpp/pulls/27752", vec![ok(&fixture("github/pull-27752.json"))]),
        (cmp.as_str(), vec![ok(&page(commits(10, 1)))]),
        (last.as_str(), vec![ok(&page(commits(1, 11)))]),
    ]);
    let r = GitRef { owner: "ggml-org".into(), repo: "llama.cpp".into(), kind: RefKind::Pull(27752) };
    let res = resolve_ref_with(&api(&base, None, 0), &r).unwrap();
    assert!(res.is_upstream && !res.is_fork_of_ggml && !res.redirected);
    assert_eq!((res.git_ref.as_str(), res.sha.as_str()), ("pull/27752/head", sha));
    assert_eq!(res.pr.as_ref().unwrap().mergeable_state.as_deref(), Some("dirty"));
    // Eleven commits ahead: the newest ten subjects, oldest first.
    assert_eq!(res.head_subjects.first().map(String::as_str), Some("step 2"));
    assert_eq!(res.head_subjects.last().map(String::as_str), Some("step 11"));
    assert_eq!(res.source_ref().label, "PR #27752");
    assert_eq!(res.html_url(), "https://github.com/ggml-org/llama.cpp/pull/27752");

    // A repository that is not there says so.
    let gone = GitRef { owner: "nobody".into(), repo: "llama.cpp".into(), kind: RefKind::Repo };
    let e = resolve_ref_with(&api(&base, None, 0), &gone).unwrap_err().to_string();
    assert!(e.contains("does not exist or is private"), "{e}");
}

#[test]
fn upstream_pr_search_checks_each_head() {
    let q = |term: &str| format!("/search/issues?q={}&per_page=10", pct(&format!("is:pr repo:ggml-org/llama.cpp \"{term}\"")));
    let head = "946fc11d1afb1e6cd316e853f1b23487754a74b9";
    let pr = serde_json::json!({"number": 25731, "title": "Add TML Inkling architecture", "state": "open", "draft": true,
        "merged": false, "mergeable_state": "clean", "html_url": "https://github.com/ggml-org/llama.cpp/pull/25731",
        "head": {"ref": "inkling", "sha": head, "repo": {"full_name": "a-contributor/llama.cpp"}}, "base": {"ref": "master"}});
    let (base, seen) = fake_github(vec![
        (q("inkling").as_str(), vec![ok(&fixture("github/search-prs-inkling.json"))]),
        (q("TML Inkling").as_str(), vec![ok(r#"{"total_count":0,"items":[]}"#)]),
        ("/repos/ggml-org/llama.cpp/pulls/25731", vec![ok(&pr.to_string())]),
    ]);
    let needs = ModelNeeds::new("inkling", None, None, Engine::LlamaServer);
    let mut probed = Vec::new();
    let found = find_upstream_pr_with(&api(&base, None, 0), &needs, Some("TML Inkling"), &mut |sha: &str| {
        probed.push(sha.to_string());
        Ok(Support::Yes)
    })
    .unwrap();
    assert_eq!(found.len(), 1, "the closed, unmerged #24523 is dropped");
    assert_eq!(found[0].pr.number, 25731);
    assert!(found[0].pr.draft && found[0].support.is_yes());
    assert_eq!(probed, [head]);
    assert!(!seen.lock().unwrap().iter().any(|(p, _)| p.contains("/pulls/24523")), "never looked up");
}

// ------------------------------------------------------------------ plan ----

fn build(tag: &str, version: Option<&str>, channel: Channel) -> Build {
    let path = PathBuf::from(format!(r"C:\b\{tag}"));
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

fn cfg() -> Config {
    let mut c = Config::default_for_machine();
    c.install_root = Some(PathBuf::from(r"C:\fidim-builds"));
    c
}

fn upstream_release() -> crate::update::Release {
    let v: serde_json::Value = serde_json::from_str(&fixture("github/releases-upstream-b11046.json")).unwrap();
    crate::update::parse_release(&v[0].to_string()).unwrap()
}

fn unsloth_release() -> crate::update::Release {
    crate::update::parse_release(&fixture("github/release-unsloth-b11030.json")).unwrap()
}

fn k2_card() -> CardCandidate {
    let r = card_refs("https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon").remove(0);
    let compare = parse_compare(&fixture("github/compare-master-ifm-ai-42adf019.json")).unwrap();
    let resolved = ResolvedRef {
        requested: r.clone(),
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
    };
    CardCandidate { git_ref: r, resolved: Some(resolved), error: None, support: Some(src_caps("ifm-ai-42adf019").support(&k2())) }
}

/// K2 Horizon today: no installed build, no upstream release and no
/// upstream pull request knows it; the card's fork does.
fn k2_inputs() -> PlanInputs {
    let up = src_caps("b11046");
    PlanInputs {
        installed: vec![
            ProbedBuild { build: build("b10984-rocm", Some("b10984"), Channel::Upstream), support: up.support(&k2()) },
            ProbedBuild { build: build("b11027-mix-3e83366-unsloth", Some("b11027"), Channel::Unsloth), support: up.support(&k2()) },
        ],
        upstream_latest: Some(UpstreamCandidate { release: upstream_release(), support: up.support(&k2()) }),
        unsloth_latest: None,
        prs: vec![],
        card_refs: vec![k2_card()],
        gfx: "gfx1201".into(),
        errors: vec![],
    }
}

#[cfg(windows)]
#[test]
fn plan_for_k2_horizon_is_the_card_fork() {
    let plan = plan_build(&cfg(), &k2(), &k2_inputs());
    let step = &plan.step;
    match &step.action {
        PlanAction::BuildFork { owner, repo, source, gpu_targets, install_dir } => {
            assert_eq!((owner.as_str(), repo.as_str()), ("ifm-ai", "llama.cpp"));
            assert_eq!(source.remote_url, "https://github.com/ifm-ai/llama.cpp");
            assert_eq!((source.git_ref.as_str(), source.sha.as_str()), ("model/K2Horizon", IFM_SHA));
            assert_eq!(source.label, "ifm-ai K2Horizon fork");
            assert_eq!(gpu_targets, "gfx1201");
            assert_eq!(install_dir, &PathBuf::from(r"C:\fidim-builds\ifm-ai-K2Horizon-fork-42adf019-src"));
        }
        o => panic!("{o:?}"),
    }
    assert!(step.needs_consent && step.verified);
    assert!(step.explanation.contains("10 commits ahead of upstream master and 380 behind"), "{}", step.explanation);
    assert!(step.explanation.contains("\"Merge pull request #1 from another-contributor/k2-horizon-msvc-pretokenizer\""), "{}", step.explanation);
    assert!(step.warnings.iter().any(|w| w.contains("now redirects to ifm-ai/llama.cpp")), "{:?}", step.warnings);
    assert!(plan.alternatives.is_empty());
    let r = plan.rejected.join("\n");
    assert!(r.contains("installed builds lack architecture 'k2-horizon' and pre-tokenizer 'k2-horizon': b10984-rocm, b11027-mix-3e83366-unsloth"), "{r}");
    assert!(r.contains("upstream b11046 (the newest release) lacks architecture 'k2-horizon'"), "{r}");
    assert!(r.contains("no open upstream pull request mentions 'k2-horizon'"), "{r}");
    // The plan crosses the Tauri boundary as JSON and back.
    let back: BuildPlan = serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
    assert_eq!(back, plan);
}

#[cfg(windows)]
#[test]
fn plan_prefers_what_needs_no_build() {
    let c = cfg();
    let gemma = ModelNeeds::new("gemma4", None, None, Engine::LlamaServer);
    let up = src_caps("b11046");
    // Installed and known: use it, the newest upstream first.
    let mut inputs = PlanInputs {
        installed: vec![
            ProbedBuild { build: build("b10771-rocm", Some("b10771"), Channel::Upstream), support: Support::Yes },
            ProbedBuild { build: build("b11027-mix-unsloth", Some("b11027"), Channel::Unsloth), support: Support::Yes },
            ProbedBuild { build: build("b10984-rocm", Some("b10984"), Channel::Upstream), support: Support::Yes },
        ],
        gfx: "gfx1201".into(),
        ..Default::default()
    };
    let plan = plan_build(&c, &gemma, &inputs);
    assert_eq!(plan.step.action, PlanAction::UseInstalled { path: PathBuf::from(r"C:\b\b10984-rocm"), name: "b10984-rocm".into() });
    assert!(!plan.step.needs_consent && plan.step.verified);

    // Nothing installed knows it, the newest release does: install it.
    inputs.installed = vec![ProbedBuild { build: build("b9553-src", Some("b9553"), Channel::Upstream), support: Support::No { missing: vec![Missing::Arch] } }];
    inputs.upstream_latest = Some(UpstreamCandidate { release: upstream_release(), support: up.support(&gemma) });
    let plan = plan_build(&c, &gemma, &inputs);
    assert_eq!(plan.step.action, PlanAction::InstallUpstream { tag: "b11046".into(), install_dir: PathBuf::from(r"C:\fidim-builds\b11046-rocm") });
    assert!(!plan.step.needs_consent);
    // ...unless the release has no Windows ROCm zip.
    let mut no_zip = inputs.clone();
    no_zip.upstream_latest.as_mut().unwrap().release.assets.retain(|a| !a.name.contains("rocm"));
    let plan = plan_build(&c, &gemma, &no_zip);
    assert!(matches!(plan.step.action, PlanAction::Unsupported { .. }));
    assert!(plan.rejected.iter().any(|r| r.contains("no Windows ROCm asset")), "{:?}", plan.rejected);

    // An installed build that may know it comes after a verified download,
    // as an alternative with its doubt spelled out.
    inputs.installed[0].support = Support::Unknown("pre-tokenizer 'x' was not found in llama.dll".into());
    let plan = plan_build(&c, &gemma, &inputs);
    assert!(matches!(plan.step.action, PlanAction::InstallUpstream { .. }));
    assert_eq!(plan.alternatives.len(), 1);
    assert!(!plan.alternatives[0].verified);
    assert!(plan.alternatives[0].warnings[0].starts_with("not verified: pre-tokenizer 'x'"));
    // With nothing verified it becomes the recommendation.
    inputs.upstream_latest = None;
    let plan = plan_build(&c, &gemma, &inputs);
    assert!(matches!(plan.step.action, PlanAction::UseInstalled { .. }) && !plan.step.verified);
}

#[test]
fn plan_for_pull_requests_and_unsloth_mixes() {
    let c = cfg();
    let inkling = ModelNeeds::new("inkling", None, None, Engine::LlamaServer);
    let pr = |number: u32, draft: bool, dirty: bool, merged: bool, state: &str| PrCandidate {
        pr: PrInfo {
            number,
            title: format!("model : add thing {number}"),
            state: state.into(),
            draft,
            merged,
            mergeable_state: Some(if dirty { "dirty" } else { "clean" }.into()),
            head_repo: Some("a-contributor/llama.cpp".into()),
            head_ref: "b".into(),
            head_sha: format!("{number:0>40}"),
            base_ref: "master".into(),
            html_url: String::new(),
        },
        support: Support::Yes,
    };
    let mut inputs = PlanInputs {
        installed: vec![ProbedBuild { build: build("b10984-rocm", Some("b10984"), Channel::Upstream), support: Support::No { missing: vec![Missing::Arch] } }],
        upstream_latest: Some(UpstreamCandidate { release: upstream_release(), support: Support::No { missing: vec![Missing::Arch] } }),
        prs: vec![pr(25000, true, false, false, "open"), pr(25731, false, true, false, "open"), pr(24000, false, false, false, "closed")],
        gfx: "gfx1201".into(),
        ..Default::default()
    };
    let plan = plan_build(&c, &inkling, &inputs);
    match &plan.step.action {
        PlanAction::BuildPr { number, source, install_dir, .. } => {
            assert_eq!(*number, 25731, "a ready PR, even a conflicting one, before a draft");
            assert_eq!(source.git_ref, "pull/25731/head");
            assert_eq!(source.remote_url, "https://github.com/ggml-org/llama.cpp");
            assert_eq!(source.label, "PR #25731");
            assert!(install_dir.ends_with("PR-25731-00000000-src"), "{install_dir:?}");
        }
        o => panic!("{o:?}"),
    }
    assert!(plan.step.needs_consent);
    assert!(plan.step.warnings.iter().any(|w| w.contains("conflicts with upstream master")));
    assert_eq!(plan.alternatives.len(), 1);
    assert!(plan.alternatives[0].warnings.iter().any(|w| w.contains("draft")));
    assert!(plan.rejected.iter().any(|r| r.contains("#24000 was closed without merging")));

    // Unsloth's newest mix merges #25731: a prebuilt beats compiling it.
    inputs.unsloth_latest = Some(unsloth_release());
    let plan = plan_build(&c, &inkling, &inputs);
    match &plan.step.action {
        PlanAction::InstallUnsloth { tag, gfx, install_dir } => {
            assert_eq!(tag, "b11030-mix-5ff778e");
            assert_eq!(gfx, "gfx120X", "gfx1201 falls back to its family zip");
            assert!(install_dir.ends_with("b11030-mix-5ff778e-unsloth"));
        }
        o => panic!("{o:?}"),
    }
    assert!(plan.step.needs_consent, "an unmerged pull request's code");
    assert!(plan.step.explanation.contains("#25731"));

    // Merged upstream but not released yet: still buildable, with the note.
    inputs.unsloth_latest = None;
    inputs.prs = vec![pr(26000, false, false, true, "closed")];
    let plan = plan_build(&c, &inkling, &inputs);
    assert!(matches!(plan.step.action, PlanAction::BuildPr { number: 26000, .. }));
    assert!(plan.step.warnings.iter().any(|w| w.contains("already merged")));

    // No GPU target: the build step says so instead of guessing one.
    inputs.gfx = String::new();
    let plan = plan_build(&c, &inkling, &inputs);
    assert!(plan.step.warnings.iter().any(|w| w.contains("no GPU target")), "{:?}", plan.step.warnings);
}

#[test]
fn plan_says_why_nothing_fits() {
    let c = cfg();
    // kingjones777's ROCmFP4 K2 files: the fork has the architecture but
    // not ROCmFPX's tensor type 101.
    let fp4 = ModelNeeds { max_type_id: Some(101), ..k2() };
    let mut inputs = k2_inputs();
    inputs.card_refs[0].support = Some(src_caps("ifm-ai-42adf019").support(&fp4));
    let plan = plan_build(&c, &fp4, &inputs);
    match &plan.step.action {
        PlanAction::Unsupported { reason } => {
            assert!(reason.contains("ggml tensor type 101"), "{reason}");
            assert!(reason.contains("lacks ggml tensor type 101"), "{reason}");
            assert!(reason.contains("pre-tokenizer 'k2-horizon'"), "{reason}");
        }
        o => panic!("{o:?}"),
    }
    assert!(!plan.step.needs_consent && plan.alternatives.is_empty());

    // A card that only says "use llama.cpp", and lookups that failed.
    let mut inputs = k2_inputs();
    inputs.card_refs = vec![CardCandidate {
        git_ref: GitRef { owner: "ggml-org".into(), repo: "llama.cpp".into(), kind: RefKind::Repo },
        resolved: None,
        error: None,
        support: None,
    }];
    inputs.errors = vec!["upstream pull requests: GitHub API rate limit reached".into()];
    let plan = plan_build(&c, &k2(), &inputs);
    let PlanAction::Unsupported { reason } = &plan.step.action else { panic!("{:?}", plan.step) };
    assert!(reason.contains("the model card links no llama.cpp fork"), "{reason}");
    assert!(reason.contains("could not check: upstream pull requests: GitHub API rate limit reached"), "{reason}");

    // A link that could not be resolved is reported with its reason.
    let mut inputs = k2_inputs();
    inputs.card_refs[0].resolved = None;
    inputs.card_refs[0].support = None;
    inputs.card_refs[0].error = Some("github.com/MBZUAI-IFM/llama.cpp does not exist or is private".into());
    let plan = plan_build(&c, &k2(), &inputs);
    assert!(plan.rejected.iter().any(|r| r.starts_with("https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon: github.com")));

    // A fork outside upstream's network, whose source was not read.
    let mut inputs = k2_inputs();
    let r = inputs.card_refs[0].resolved.as_mut().unwrap();
    r.is_fork_of_ggml = false;
    r.ahead_by = None;
    r.behind_by = None;
    inputs.card_refs[0].support = None;
    let plan = plan_build(&c, &k2(), &inputs);
    assert!(!plan.step.verified && plan.step.needs_consent);
    assert!(plan.step.warnings.iter().any(|w| w.contains("not in ggml-org/llama.cpp's fork network")));
    assert!(plan.step.warnings.iter().any(|w| w.contains("its source was not read")));
}

#[test]
fn plan_for_diffusion_gemma() {
    let c = cfg();
    let dg = ModelNeeds::new("diffusion-gemma", None, None, Engine::DiffusionGemma);
    let inputs = PlanInputs {
        installed: vec![ProbedBuild { build: build("b10984-rocm", Some("b10984"), Channel::Upstream), support: Support::No { missing: vec![Missing::Arch] } }],
        unsloth_latest: Some(unsloth_release()),
        gfx: "gfx1201".into(),
        ..Default::default()
    };
    let plan = plan_build(&c, &dg, &inputs);
    assert!(matches!(&plan.step.action, PlanAction::InstallUnsloth { gfx, .. } if gfx == "gfx120X"));
    // An exact target the release ships wins over the family; junk never panics.
    let exact = PlanInputs { gfx: "gfx1151".into(), ..inputs.clone() };
    assert!(matches!(&plan_build(&c, &dg, &exact).step.action, PlanAction::InstallUnsloth { gfx, .. } if gfx == "gfx1151"));
    for junk in ["", "g", "gfx\u{e9}", "\u{1F600}"] {
        let odd = PlanInputs { gfx: junk.into(), ..inputs.clone() };
        assert!(matches!(plan_build(&c, &dg, &odd).step.action, PlanAction::Unsupported { .. }), "{junk:?}");
    }
    assert!(!plan.step.needs_consent, "FIDIM's diffusion channel");
    assert!(plan.rejected.iter().any(|r| r.contains("no DiffusionGemma runner")));
}

// ------------------------------------------------ upstream under any name ----

/// Cards link upstream by its old name and link folders inside master;
/// neither is a fork, and neither may use up the lookups a real fork needs.
#[test]
fn card_links_to_upstream_are_not_forks() {
    let card = "Quantized with [llama.cpp](https://github.com/ggerganov/llama.cpp); serve with \
                https://github.com/ggml-org/llama.cpp/tree/master/tools/server; upstream PR \
                https://github.com/ggerganov/llama.cpp/pull/12345; until then use \
                https://github.com/example-org/llama.cpp/tree/add-new-arch\n";
    let refs = card_refs(card);
    let gref = |o: &str, kind: RefKind| GitRef { owner: o.into(), repo: "llama.cpp".into(), kind };
    assert_eq!(
        refs,
        vec![
            gref("ggerganov", RefKind::Repo),
            gref("ggml-org", RefKind::Branch("master".into())),
            gref("ggerganov", RefKind::Pull(12345)),
            gref("example-org", RefKind::Branch("add-new-arch".into())),
        ],
        "a folder in master is master"
    );
    assert!(refs[0].is_upstream_repo() && refs[0].is_upstream_master());
    assert!(refs[1].is_upstream_master());
    assert!(refs[2].is_upstream_repo() && !refs[2].is_upstream_master());
    assert!(!refs[3].is_upstream_repo());

    // The fork is looked up first; upstream itself never.
    let lookups = card_lookups(&refs, &[]);
    assert_eq!(lookups, vec![&refs[3], &refs[2]]);
    // A pull request the search already read is not read again.
    assert_eq!(card_lookups(&refs, &[12345]), vec![&refs[3]]);
    // The same pull request under both names is one lookup; forks come
    // first however late the card names them, three at most.
    let mut more = refs.clone();
    more.insert(0, gref("ggml-org", RefKind::Pull(12345)));
    for o in ["a", "b", "c"] {
        more.push(gref(o, RefKind::Repo));
    }
    let picked: Vec<String> = card_lookups(&more, &[]).iter().map(|r| r.url()).collect();
    assert_eq!(
        picked,
        [
            "https://github.com/example-org/llama.cpp/tree/add-new-arch",
            "https://github.com/a/llama.cpp",
            "https://github.com/b/llama.cpp"
        ]
    );
    let few = vec![gref("ggml-org", RefKind::Pull(12345)), gref("ggerganov", RefKind::Pull(12345))];
    assert_eq!(card_lookups(&few, &[]).len(), 1);
    // Branch names with a slash stay whole unless they start in master.
    assert_eq!(
        card_refs("https://github.com/ggml-org/llama.cpp/tree/gg/new-arch https://github.com/x/llama.cpp/tree/main/docs"),
        vec![gref("ggml-org", RefKind::Branch("gg/new-arch".into())), gref("x", RefKind::Branch("main".into()))]
    );
}

/// A card link that resolves to upstream master (its old name redirects
/// there) is never offered as a fork build.
#[test]
fn plan_never_builds_upstream_master_as_a_fork() {
    let mut inputs = k2_inputs();
    let old_name = GitRef { owner: "ggerganov".into(), repo: "llama.cpp".into(), kind: RefKind::Repo };
    let mut upstream = inputs.card_refs[0].clone();
    upstream.git_ref = old_name.clone();
    let r = upstream.resolved.as_mut().unwrap();
    r.requested = old_name;
    (r.owner, r.repo, r.remote_url, r.git_ref) =
        ("ggml-org".into(), "llama.cpp".into(), "https://github.com/ggml-org/llama.cpp".into(), "master".into());
    r.is_upstream = true;
    r.is_fork_of_ggml = false;
    let mut folder = upstream.clone();
    folder.git_ref = GitRef { owner: "ggml-org".into(), repo: "llama.cpp".into(), kind: RefKind::Branch("master".into()) };
    folder.resolved = None;
    folder.error = Some("no such ref".into());
    inputs.card_refs.insert(0, upstream);
    inputs.card_refs.insert(0, folder);
    let plan = plan_build(&cfg(), &k2(), &inputs);
    assert!(matches!(&plan.step.action, PlanAction::BuildFork { owner, .. } if owner == "ifm-ai"), "{:?}", plan.step);
    assert!(plan.alternatives.is_empty(), "{:?}", plan.alternatives);
    assert!(!plan.rejected.iter().any(|r| r.contains("ggerganov") || r.contains("tree/master")), "{:?}", plan.rejected);
}

/// A fork based on upstream from before ROCm 7's hipBLAS change cannot be
/// compiled with a ROCm 7 HIP SDK: it is still offered, last, and says so.
#[test]
fn plan_puts_forks_too_old_for_rocm7_last() {
    let mut inputs = k2_inputs();
    let mut old = inputs.card_refs[0].clone();
    old.git_ref = GitRef { owner: "old-org".into(), repo: "llama.cpp".into(), kind: RefKind::Branch("k2".into()) };
    let r = old.resolved.as_mut().unwrap();
    r.requested = old.git_ref.clone();
    (r.owner, r.remote_url, r.git_ref, r.sha) =
        ("old-org".into(), "https://github.com/old-org/llama.cpp".into(), "k2".into(), "1".repeat(40));
    r.redirected = false;
    // Upstream's newest release is b11046: 6000 behind is about b5046.
    r.behind_by = Some(6000);
    inputs.card_refs.insert(0, old.clone());
    let plan = plan_build(&cfg(), &k2(), &inputs);
    assert!(matches!(&plan.step.action, PlanAction::BuildFork { owner, .. } if owner == "ifm-ai"), "{:?}", plan.step);
    assert!(!plan.step.warnings.iter().any(|w| w.contains("b5872")), "380 behind is recent: {:?}", plan.step.warnings);
    assert_eq!(plan.alternatives.len(), 1);
    let alt = &plan.alternatives[0];
    assert!(matches!(&alt.action, PlanAction::BuildFork { owner, .. } if owner == "old-org"));
    assert!(alt.verified, "the source knows the model; only the build is in doubt");
    let w = alt.warnings.join("\n");
    assert!(w.contains("around b5046") && w.contains("older than b5872") && w.contains("ROCm 7"), "{w}");

    // Alone, it is still the plan, with the warning.
    inputs.card_refs = vec![old];
    let plan = plan_build(&cfg(), &k2(), &inputs);
    assert!(matches!(&plan.step.action, PlanAction::BuildFork { owner, .. } if owner == "old-org"));
    assert!(plan.step.warnings.iter().any(|w| w.contains("older than b5872")));
    // Without upstream's newest release number there is no estimate.
    inputs.upstream_latest = None;
    let plan = plan_build(&cfg(), &k2(), &inputs);
    assert!(!plan.step.warnings.iter().any(|w| w.contains("b5872")), "{:?}", plan.step.warnings);
}

/// A token GitHub rejects (expired, revoked, from another tool's
/// environment) costs one request, not the answer.
#[test]
fn api_drops_a_rejected_token() {
    let unauthorized = Resp { status: 401, headers: vec![], body: r#"{"message":"Bad credentials"}"#.into() };
    let reset = (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() + 600).to_string();
    let limited = Resp {
        status: 403,
        headers: vec![("X-RateLimit-Remaining".into(), "0".into()), ("X-RateLimit-Reset".into(), reset)],
        body: r#"{"message":"API rate limit exceeded for 127.0.0.1."}"#.into(),
    };
    let (base, seen) = fake_github(vec![
        ("/r", vec![unauthorized.clone(), ok("fine")]),
        ("/r2", vec![ok("two")]),
        ("/limited", vec![limited]),
        ("/private", vec![unauthorized]),
    ]);
    let auth = |i: usize| seen.lock().unwrap()[i].1.get("authorization").cloned();
    let a = Api::with_base(&base, Some("stale".into()), None, Duration::ZERO);
    assert_eq!(a.get("/r", "application/json").unwrap().as_deref(), Some("fine"));
    assert_eq!(auth(0).as_deref(), Some("Bearer stale"));
    assert_eq!(auth(1), None, "asked again without the token");
    assert!(a.token_rejected());
    assert_eq!(a.get("/r2", "application/json").unwrap().as_deref(), Some("two"));
    assert_eq!(auth(2), None, "and never sent again");
    let e = a.get("/limited", "application/json").unwrap_err().to_string();
    assert!(e.contains("rate limit") && e.contains("rejected the token"), "{e}");
    // Without a token, a 401 is not about one.
    let plain = Api::with_base(&base, None, None, Duration::ZERO);
    let e = plain.get("/private", "application/json").unwrap_err().to_string();
    assert!(e.contains("HTTP 401") && !e.contains("token"), "{e}");
    assert!(!plain.token_rejected());

    // FIDIM's own setting beats the environment other tools share.
    let mut c = Config::default_for_machine();
    c.github_token = Some(" from-config ".into());
    assert_eq!(token(&c).as_deref(), Some("from-config"));
    c.github_token = Some("  ".into());
    assert_ne!(token(&c).as_deref(), Some(""), "blank is none");
}
