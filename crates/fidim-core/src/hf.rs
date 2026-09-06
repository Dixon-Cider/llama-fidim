//! Creator defaults: the sampling parameters the model's author published in
//! `generation_config.json` on Hugging Face. Shown next to the profile's
//! sampling controls so "what did the creator intend" is one click away.
//!
//! The GGUF header names the base repo (`general.base_model.0.*`); the
//! quantizer's repo is tried second. Gated repos (Gemma, Llama) return 401
//! without a token — `HF_TOKEN` in the environment or `config.hf_token`
//! is sent as a bearer when present. Results are cached for a week under
//! `~/.fidim/hf-cache/`, because the answer does not change.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::gguf;
use crate::{Error, Result};

const CACHE_TTL: Duration = Duration::from_secs(7 * 24 * 3600);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatorDefaults {
    /// Repo the values came from.
    pub repo: String,
    pub url: String,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub min_p: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub fetched_at_unix: u64,
    pub from_cache: bool,
    /// The whole file, for anything the fixed fields miss.
    pub raw: serde_json::Value,
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn cache_path(repo: &str) -> PathBuf {
    Config::config_dir().join("hf-cache").join(format!("{}.generation_config.json", repo.replace('/', "__")))
}

/// Parse a generation_config.json into the fields we surface.
pub fn parse_generation_config(repo: &str, url: &str, text: &str, from_cache: bool) -> Result<CreatorDefaults> {
    let raw: serde_json::Value = serde_json::from_str(text)?;
    let f = |k: &str| raw.get(k).and_then(|v| v.as_f64());
    Ok(CreatorDefaults {
        repo: repo.into(),
        url: url.into(),
        temperature: f("temperature"),
        top_p: f("top_p"),
        top_k: raw.get("top_k").and_then(|v| v.as_u64()).map(|k| k as u32),
        min_p: f("min_p"),
        repetition_penalty: f("repetition_penalty"),
        fetched_at_unix: now_unix(),
        from_cache,
        raw,
    })
}

fn token(cfg: &Config) -> Option<String> {
    std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty()).or_else(|| cfg.hf_token.clone())
}

fn fetch_generation_config(cfg: &Config, repo: &str) -> Result<CreatorDefaults> {
    let url = format!("https://huggingface.co/{repo}/raw/main/generation_config.json");
    let cache = cache_path(repo);
    if let Ok(meta) = std::fs::metadata(&cache) {
        let fresh = meta.modified().ok().and_then(|m| m.elapsed().ok()).is_some_and(|age| age < CACHE_TTL);
        if fresh {
            if let Ok(text) = std::fs::read_to_string(&cache) {
                return parse_generation_config(repo, &url, &text, true);
            }
        }
    }
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(30))
        .build();
    let mut req = agent.get(&url).set("User-Agent", concat!("llama-fidim/", env!("CARGO_PKG_VERSION")));
    if let Some(t) = token(cfg) {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let text = match req.call() {
        Ok(resp) => resp.into_string().map_err(|e| Error::Update(format!("{url}: {e}")))?,
        Err(ureq::Error::Status(401 | 403, _)) => {
            return Err(Error::Update(format!(
                "{repo} is gated on Hugging Face — accept its terms and set HF_TOKEN (or config.hf_token)"
            )))
        }
        Err(ureq::Error::Status(404, _)) => {
            return Err(Error::Update(format!("{repo} has no generation_config.json")))
        }
        Err(e) => return Err(Error::Update(format!("{url}: {e}"))),
    };
    let parsed = parse_generation_config(repo, &url, &text, false)?;
    if let Some(parent) = cache.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&cache, &text);
    Ok(parsed)
}

/// `owner/repo` implied by the download layout `<model root>/<owner>/<repo>/<file>.gguf`
/// (how LM Studio and `huggingface-cli` lay files out). None when the file
/// sits directly under a root or the layout is something else.
pub fn repo_from_layout(model_roots: &[PathBuf], model_path: &Path) -> Option<String> {
    let norm = |p: &Path| p.to_string_lossy().replace('/', "\\").trim_end_matches('\\').to_lowercase();
    let repo_dir = model_path.parent()?;
    let owner_dir = repo_dir.parent()?;
    let root = owner_dir.parent()?;
    let is_root = model_roots.iter().any(|r| norm(r) == norm(root));
    if !is_root {
        return None;
    }
    let name = |p: &Path| p.file_name().map(|s| s.to_string_lossy().into_owned());
    Some(format!("{}/{}", name(owner_dir)?, name(repo_dir)?))
}

/// `cardData.base_model` of a repo via the HF API — the way a quant repo
/// points back at the creator's model when the GGUF header does not.
fn base_model_via_api(cfg: &Config, repo: &str) -> Result<Vec<String>> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(30))
        .build();
    let mut req = agent.get(&url).set("User-Agent", concat!("llama-fidim/", env!("CARGO_PKG_VERSION")));
    if let Some(t) = token(cfg) {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let text = req
        .call()
        .map_err(|e| Error::Update(format!("{url}: {e}")))?
        .into_string()
        .map_err(|e| Error::Update(format!("{url}: {e}")))?;
    let v: serde_json::Value = serde_json::from_str(&text)?;
    let bm = &v["cardData"]["base_model"];
    let mut out = Vec::new();
    match bm {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(a) => out.extend(a.iter().filter_map(|x| x.as_str().map(String::from))),
        _ => {}
    }
    Ok(out)
}

/// Creator defaults for a GGUF on disk. Candidate repos, in order: the base
/// model named in the GGUF header, the quant repo named in the header, the
/// repo implied by the `<root>/<owner>/<repo>/` folder layout, and finally
/// whatever `base_model` those quant repos declare on their model card.
pub fn creator_defaults(cfg: &Config, model_path: &Path) -> Result<CreatorDefaults> {
    let header = gguf::read_header(model_path)?;
    // Converters since mid-2026 embed generation_config values as
    // `general.sampling.*` — authoritative and offline, so they win.
    if header.sampling_temp.is_some() || header.sampling_top_k.is_some() || header.sampling_top_p.is_some() {
        return Ok(CreatorDefaults {
            repo: header.source_repo.clone().or_else(|| header.quant_repo.clone()).unwrap_or_else(|| "embedded".into()),
            url: format!("embedded in GGUF: {}", model_path.display()),
            temperature: header.sampling_temp,
            top_p: header.sampling_top_p,
            top_k: header.sampling_top_k,
            min_p: header.sampling_min_p,
            repetition_penalty: header.sampling_repeat_penalty,
            fetched_at_unix: now_unix(),
            from_cache: false,
            raw: serde_json::json!({ "source": "general.sampling.* in the GGUF header" }),
        });
    }
    let mut candidates: Vec<String> = Vec::new();
    let push = |r: Option<String>, into: &mut Vec<String>| {
        if let Some(r) = r {
            if !into.contains(&r) {
                into.push(r);
            }
        }
    };
    push(header.source_repo.clone(), &mut candidates);
    push(header.quant_repo.clone(), &mut candidates);
    push(repo_from_layout(&cfg.model_roots, model_path), &mut candidates);
    if candidates.is_empty() {
        return Err(Error::Update(
            "no Hugging Face repo could be inferred: the GGUF carries no general.base_model / general.source keys \
             and the file is not under <model root>/<owner>/<repo>/"
                .into(),
        ));
    }
    let mut errors = Vec::new();
    let mut tried: Vec<String> = Vec::new();
    let mut queue = candidates.clone();
    while let Some(repo) = queue.first().cloned() {
        queue.remove(0);
        if tried.contains(&repo) {
            continue;
        }
        tried.push(repo.clone());
        match fetch_generation_config(cfg, &repo) {
            Ok(d) => return Ok(d),
            Err(e) => {
                errors.push(e.to_string());
                // A quant repo without the file usually names its base model.
                if let Ok(bases) = base_model_via_api(cfg, &repo) {
                    for b in bases {
                        if !tried.contains(&b) && !queue.contains(&b) {
                            queue.push(b);
                        }
                    }
                }
            }
        }
    }
    Err(Error::Update(format!("tried {}: {}", tried.join(", "), errors.join("; "))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_common_fields_and_keeps_raw() {
        let text = r#"{"bos_token_id":2,"temperature":1.0,"top_p":0.95,"top_k":64,"repetition_penalty":1.0,"do_sample":true}"#;
        let d = parse_generation_config("google/x", "u", text, false).unwrap();
        assert_eq!(d.temperature, Some(1.0));
        assert_eq!(d.top_k, Some(64));
        assert_eq!(d.min_p, None);
        assert_eq!(d.raw["bos_token_id"], 2);
    }

    #[test]
    fn layout_repo_needs_root_owner_repo_file() {
        let roots = vec![PathBuf::from(r"E:\models")];
        assert_eq!(
            repo_from_layout(&roots, Path::new(r"E:\models\unsloth\gemma-4-26B-A4B-it-qat-GGUF\x.gguf")).as_deref(),
            Some("unsloth/gemma-4-26B-A4B-it-qat-GGUF")
        );
        assert_eq!(repo_from_layout(&roots, Path::new(r"E:\models\loose.gguf")), None);
        assert_eq!(repo_from_layout(&roots, Path::new(r"D:\elsewhere\a\b\x.gguf")), None);
    }

    #[test]
    fn cache_path_is_flat_and_per_repo() {
        let p = cache_path("google/gemma-4-26B-A4B-it");
        assert!(p.to_string_lossy().ends_with("google__gemma-4-26B-A4B-it.generation_config.json"));
    }
}
