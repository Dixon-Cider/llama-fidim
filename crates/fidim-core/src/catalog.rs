//! What a Hugging Face repo offers, and what of it fits: the repo's GGUF
//! files grouped into downloadable choices (one file, or every shard of a
//! split set, per quant), the vision projectors and draft models beside
//! them, the VRAM estimate of each choice on one card and split over two,
//! a recommendation, and where the files go on disk.
//!
//! Pure: file lists, headers and devices come in, nothing is fetched.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::devices::Device;
use crate::estimate::{self, DgRunner, EstimateInput};
use crate::gguf::{self, GgufHeader};
use crate::hub::RepoFile;
use crate::profile::{Runtime, SplitMode};

const MIB: u64 = 1024 * 1024;
/// The context a choice is judged at when none is asked for, as a new
/// profile gets (or the model's own maximum when that is smaller).
pub const DEFAULT_CTX: u64 = 32_768;
/// A verdict over this fraction of the card is `Tight`: no headroom for a
/// display or another server (pre-flight check 6 warns at the same point).
const TIGHT: f64 = 0.90;

/// One thing to download: a single GGUF, or every shard of a split set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// The quant (e.g. `Q4_K_M`, `UD-Q4_K_XL`, `IQ4_XS`, `BF16`,
    /// `MXFP4_MOE`), or the file name when none is recognised. Unique
    /// within a catalog.
    pub label: String,
    /// The recognised quant, uppercase; None when the name has none.
    pub quant: Option<String>,
    /// The file, or the shards in order (`-00001-of-` first).
    pub files: Vec<RepoFile>,
    pub total_size: u64,
    /// The path llama.cpp is given: the file, or the first shard.
    pub first_file: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    /// Smallest first.
    pub choices: Vec<Choice>,
    /// Vision projectors (`*mmproj*.gguf`).
    pub mmproj: Vec<RepoFile>,
    /// Speculative-decoding drafts: MTP heads (`*mtp*` or under `MTP/`)
    /// and the `draft-`, `eagle3-`, `dflash-`, `dspark-` sidecars.
    pub drafts: Vec<RepoFile>,
    /// Everything else, with the reason.
    pub ignored: Vec<(RepoFile, String)>,
}

/// Sort a repo's files into choices, projectors, drafts and the rest.
pub fn classify(files: &[RepoFile]) -> Catalog {
    let mut cat = Catalog::default();
    // (folder, lowercase prefix, count) -> (number, file)
    let mut sets: BTreeMap<(String, String, u32), Vec<(u32, RepoFile)>> = BTreeMap::new();
    let mut singles: Vec<RepoFile> = Vec::new();
    for f in files {
        let name = f.name();
        let lower = name.to_ascii_lowercase();
        if !lower.ends_with(".gguf") {
            let why = if lower.contains("imatrix") {
                "importance matrix (quantizer input, not a model)"
            } else {
                "not a GGUF file"
            };
            cat.ignored.push((f.clone(), why.into()));
        } else if lower.contains("imatrix") {
            cat.ignored.push((f.clone(), "importance matrix (quantizer input, not a model)".into()));
        } else if lower.contains("mmproj") {
            cat.mmproj.push(f.clone());
        } else if is_draft(&f.path, &lower) {
            cat.drafts.push(f.clone());
        } else if let Some((prefix, no, count)) = gguf::split_name(name) {
            let dir = folder(&f.path).to_ascii_lowercase();
            sets.entry((dir, prefix.to_ascii_lowercase(), count)).or_default().push((no, f.clone()));
        } else {
            singles.push(f.clone());
        }
    }

    let mut choices: Vec<Choice> = singles
        .into_iter()
        .map(|f| {
            let stem = &f.name()[..f.name().len() - ".gguf".len()];
            let quant = quant_label(stem);
            Choice {
                label: quant.clone().unwrap_or_else(|| stem.to_string()),
                quant,
                total_size: f.size,
                first_file: f.path.clone(),
                files: vec![f],
            }
        })
        .collect();
    for ((_, _, count), mut parts) in sets {
        parts.sort_by_key(|(no, _)| *no);
        let numbers: Vec<u32> = parts.iter().map(|(no, _)| *no).collect();
        if numbers != (1..=count).collect::<Vec<_>>() {
            let present = numbers.iter().map(u32::to_string).collect::<Vec<_>>().join(", ");
            for (_, f) in parts {
                cat.ignored
                    .push((f, format!("incomplete split set: parts {present} of {count} (every part is needed)")));
            }
            continue;
        }
        let files: Vec<RepoFile> = parts.into_iter().map(|(_, f)| f).collect();
        let first = &files[0];
        let prefix = gguf::split_name(first.name()).map(|(p, _, _)| p.to_string()).unwrap_or_default();
        let quant = quant_label(&prefix);
        choices.push(Choice {
            label: quant.clone().unwrap_or_else(|| prefix.clone()),
            quant,
            total_size: files.iter().map(|f| f.size).sum(),
            first_file: first.path.clone(),
            files,
        });
    }

    // Two choices with one quant (the same quant of two models, or a single
    // file beside a split set) keep their file names apart.
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for c in &choices {
        *seen.entry(c.label.clone()).or_default() += 1;
    }
    for c in &mut choices {
        if seen[&c.label] > 1 {
            c.label = format!("{} ({})", c.label, c.first_file);
        }
    }
    choices.sort_by(|a, b| a.total_size.cmp(&b.total_size).then_with(|| a.label.cmp(&b.label)));
    cat.choices = choices;
    cat
}

/// A speculative-decoding sidecar rather than a model: FIDIM's discovery
/// rule (`mtp` in the name, or an `MTP/` folder) plus llama.cpp's sidecar
/// prefixes.
fn is_draft(path: &str, lower_name: &str) -> bool {
    let in_mtp_dir = path.split('/').rev().skip(1).any(|d| d.eq_ignore_ascii_case("mtp"));
    in_mtp_dir
        || lower_name.contains("mtp")
        || ["draft-", "eagle3-", "dflash-", "dspark-"].iter().any(|p| lower_name.starts_with(p))
}

/// The folder part of a repo path (empty at the root).
fn folder(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

/// A quant token: `Q4_K_M`, `IQ4_XS`, `Q4_0_ROCMFP4_FAST`, `TQ1_0`,
/// `BF16`, `F16`, `F32`, `MXFP4_MOE`, `NVFP4` (case-insensitive).
fn is_quant_token(t: &str) -> bool {
    let u = t.to_ascii_uppercase();
    if ["BF16", "F16", "FP16", "F32", "FP32", "MXFP4", "MXFP4_MOE", "NVFP4", "TQ1_0", "TQ2_0"].contains(&u.as_str()) {
        return true;
    }
    let rest = u.strip_prefix("IQ").or_else(|| u.strip_prefix('Q'));
    let Some(rest) = rest else { return false };
    let mut segs = rest.split('_');
    let head = segs.next().unwrap_or("");
    matches!(head.as_bytes(), [b'1'..=b'8'])
        && segs.all(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_alphanumeric()))
}

/// The quant in a file stem, e.g. `gemma-4-26B-A4B-it-UD-Q4_K_XL` ->
/// `UD-Q4_K_XL`, `Model.Q8_0` -> `Q8_0`, `model_q4_k_m` -> `Q4_K_M`. The last
/// match wins (the quant ends the name; a model name may contain `Q3`).
pub fn quant_label(stem: &str) -> Option<String> {
    let tokens: Vec<&str> = stem.split(['-', '.']).collect();
    for i in (0..tokens.len()).rev() {
        let tok = tokens[i];
        // The token itself, or its longest `_`-joined tail (`model_q4_k_m`).
        let segs: Vec<&str> = tok.split('_').collect();
        let found = (0..segs.len()).map(|j| segs[j..].join("_")).find(|cand| is_quant_token(cand));
        if let Some(q) = found {
            let q = q.to_ascii_uppercase();
            let ud = i > 0 && tokens[i - 1].eq_ignore_ascii_case("ud") && q.len() == tok.len();
            return Some(if ud { format!("UD-{q}") } else { q });
        }
    }
    None
}

// ------------------------------------------------------------------ fits ----

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fit {
    Fits,
    /// Fits, but over 90% of a card.
    Tight,
    NoFit,
    /// No such configuration here (one GPU, no GPU, or a single-card engine).
    NotApplicable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitVerdict {
    pub fit: Fit,
    /// Estimated bytes on the card that is fullest relative to its size.
    pub need_bytes: u64,
    /// That card's total VRAM.
    pub capacity_bytes: u64,
    /// Whether every card also has that much free right now; other servers
    /// may be holding VRAM, and the verdict is against the whole card.
    pub fits_free_now: bool,
    /// The arithmetic, or why the verdict does not apply.
    pub detail: String,
}

impl FitVerdict {
    fn not_applicable(detail: impl Into<String>) -> Self {
        FitVerdict { fit: Fit::NotApplicable, need_bytes: 0, capacity_bytes: 0, fits_free_now: false, detail: detail.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceFit {
    pub label: String,
    pub quant: Option<String>,
    pub total_size: u64,
    /// The context the verdicts are judged at.
    pub ctx: u64,
    pub one_card: FitVerdict,
    pub two_card_split: FitVerdict,
    /// Largest context (a multiple of 1024, at most the model's trained
    /// context) that fits the largest card with 10% to spare; for a
    /// diffusion model, the budget its runner would size itself to. None
    /// when nothing fits.
    pub max_ctx_one_card: Option<u64>,
    /// What the estimate had to assume (from `estimate`).
    pub assumptions: Vec<String>,
}

/// The launch settings the estimates assume: a new profile's defaults.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FitOptions {
    /// None = `DEFAULT_CTX`, or the model's maximum when smaller.
    #[serde(default)]
    pub ctx: Option<u64>,
    #[serde(default = "f16")]
    pub kv_type_k: String,
    #[serde(default = "f16")]
    pub kv_type_v: String,
    #[serde(default = "default_ub")]
    pub batch_physical: u32,
    #[serde(default = "one")]
    pub slots: u32,
    /// A projector and a draft model sit on the main card beside the model.
    #[serde(default)]
    pub mmproj_bytes: u64,
    #[serde(default)]
    pub draft_bytes: u64,
}

fn f16() -> String {
    "f16".into()
}
fn default_ub() -> u32 {
    512
}
fn one() -> u32 {
    1
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            ctx: None,
            kv_type_k: f16(),
            kv_type_v: f16(),
            batch_physical: default_ub(),
            slots: one(),
            mmproj_bytes: 0,
            draft_bytes: 0,
        }
    }
}

impl FitOptions {
    fn runtime(&self, ctx: u64) -> Runtime {
        // Through serde so fields added to Runtime later keep their defaults.
        serde_json::from_value(serde_json::json!({
            "ctx_total": ctx,
            "slots": self.slots.max(1),
            "kv_type_k": self.kv_type_k,
            "kv_type_v": self.kv_type_v,
            "batch_physical": self.batch_physical.max(1),
            "n_gpu_layers": 999,
        }))
        .expect("a Runtime from its own fields")
    }
}

/// The cards a model can go on: the discrete ones, largest first (every
/// device on a machine with no discrete card, e.g. an APU). Among cards of
/// one size the one with the most VRAM free comes first: the verdicts are
/// against the card's size either way, but `fits_free_now` must not say no
/// because the card judged is busy while an identical one is idle, and the
/// split's main card, which also holds the compute buffers, is the freer.
fn cards(devices: &[Device]) -> Vec<&Device> {
    let mut v: Vec<&Device> = devices.iter().filter(|d| !d.integrated).collect();
    if v.is_empty() {
        v = devices.iter().collect();
    }
    v.sort_by(|a, b| {
        b.total_mib
            .cmp(&a.total_mib)
            .then_with(|| b.free_mib.cmp(&a.free_mib))
            .then_with(|| a.stable_key.cmp(&b.stable_key))
    });
    v
}

fn verdict_for(ratio: f64) -> Fit {
    if ratio > 1.0 {
        Fit::NoFit
    } else if ratio > TIGHT {
        Fit::Tight
    } else {
        Fit::Fits
    }
}

const GIB: f64 = (1u64 << 30) as f64;

/// A llama-server estimate over `on` with the split fractions given.
fn server_verdict(h: &GgufHeader, rt: &Runtime, on: &[(&Device, f64)], opts: &FitOptions) -> (FitVerdict, Vec<String>) {
    let est = estimate::estimate(&EstimateInput {
        header: h,
        runtime: rt,
        devices: on.iter().map(|(d, f)| (d.stable_key.clone(), *f)).collect(),
        split_mode: (on.len() > 1).then_some(SplitMode::Layer),
        main_index: 0,
        mmproj_bytes: opts.mmproj_bytes,
        draft_bytes: opts.draft_bytes,
    });
    let mut worst: Option<(f64, u64, u64, String)> = None;
    let mut fits_free_now = true;
    for (de, (d, _)) in est.per_device.iter().zip(on) {
        let cap = d.total_mib * MIB;
        fits_free_now &= de.total_bytes <= d.free_mib * MIB;
        let ratio = if cap == 0 { f64::INFINITY } else { de.total_bytes as f64 / cap as f64 };
        if worst.as_ref().is_none_or(|w| ratio > w.0) {
            let detail = format!("{}: {} of {:.2} GiB", d.name, de.breakdown(), cap as f64 / GIB);
            worst = Some((ratio, de.total_bytes, cap, detail));
        }
    }
    let (ratio, need_bytes, capacity_bytes, detail) = worst.unwrap_or((f64::INFINITY, 0, 0, String::new()));
    (FitVerdict { fit: verdict_for(ratio), need_bytes, capacity_bytes, fits_free_now, detail }, est.assumptions)
}

/// Largest multiple of 1024 up to `max_ctx` whose one-card estimate stays
/// within 90% of the card (the estimate grows with context).
fn max_ctx_on(h: &GgufHeader, card: &Device, max_ctx: u64, opts: &FitOptions) -> Option<u64> {
    let fits = |ctx: u64| {
        let (v, _) = server_verdict(h, &opts.runtime(ctx), &[(card, 1.0)], opts);
        v.fit == Fit::Fits
    };
    if max_ctx < 1024 {
        return fits(max_ctx.max(1)).then_some(max_ctx);
    }
    let (mut lo, mut hi) = (0u64, max_ctx / 1024); // lo fits (0 = none yet), hi is the ceiling
    if fits(hi * 1024) {
        return Some(hi * 1024);
    }
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if fits(mid * 1024) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo > 0).then_some(lo * 1024)
}

/// Estimate every choice with `header` (read from any one of them: the
/// architecture keys are the same for every quant of a model) sized to the
/// choice, on the largest card alone and layer-split over the two largest.
/// Verdicts are against each card's total VRAM; `fits_free_now` says
/// whether the VRAM free right now would do.
pub fn estimate_choices(header: &GgufHeader, choices: &[Choice], devices: &[Device], opts: &FitOptions) -> Vec<ChoiceFit> {
    let cards = cards(devices);
    let model_ctx = header.context_length.filter(|c| *c > 0);
    let ctx = opts.ctx.filter(|c| *c > 0).unwrap_or_else(|| model_ctx.map_or(DEFAULT_CTX, |m| m.min(DEFAULT_CTX)));
    let mut h = header.clone();
    choices
        .iter()
        .map(|c| {
            h.file_size = c.total_size;
            let mut fit = ChoiceFit {
                label: c.label.clone(),
                quant: c.quant.clone(),
                total_size: c.total_size,
                ctx,
                one_card: FitVerdict::not_applicable("no GPU found"),
                two_card_split: FitVerdict::not_applicable("fewer than two GPUs"),
                max_ctx_one_card: None,
                assumptions: Vec::new(),
            };
            if h.is_diffusion() {
                diffusion_fit(&h, &cards, &mut fit);
                return fit;
            }
            let rt = opts.runtime(ctx);
            if let Some(card) = cards.first() {
                let (v, assumptions) = server_verdict(&h, &rt, &[(card, 1.0)], opts);
                fit.one_card = v;
                fit.assumptions = assumptions;
                fit.max_ctx_one_card = max_ctx_on(&h, card, model_ctx.unwrap_or(1 << 20), opts);
            }
            if let [a, b, ..] = cards.as_slice() {
                let f = estimate::auto_fractions(&[a.total_mib, b.total_mib]);
                let (v, _) = server_verdict(&h, &rt, &[(a, f[0]), (b, f[1])], opts);
                fit.two_card_split = v;
            }
            fit
        })
        .collect()
}

/// The DiffusionGemma runner: one card, with a context budget it sizes to
/// the VRAM left after the weights (stock runner, flash attention off).
fn diffusion_fit(h: &GgufHeader, cards: &[&Device], fit: &mut ChoiceFit) {
    fit.two_card_split = FitVerdict::not_applicable("the diffusion runner uses one card");
    let Some(card) = cards.first() else { return };
    let (est, sizing) = estimate::estimate_diffusion(h, 999, 0, false, DgRunner::STOCK, &card.stable_key, card.total_mib);
    let d = &est.per_device[0];
    let cap = card.total_mib * MIB;
    let load = d.total_bytes.saturating_sub(d.kv_bytes);
    let fit_now = if sizing.predicted_auto_maxtok.is_none() || load > cap {
        Fit::NoFit
    } else if d.total_bytes as f64 > cap as f64 * TIGHT {
        Fit::Tight
    } else {
        Fit::Fits
    };
    fit.ctx = sizing.maxtok_used as u64;
    fit.max_ctx_one_card = sizing.predicted_auto_maxtok.map(u64::from);
    fit.one_card = FitVerdict {
        fit: fit_now,
        need_bytes: d.total_bytes,
        capacity_bytes: cap,
        fits_free_now: d.total_bytes <= card.free_mib * MIB,
        detail: format!("{}: {} of {:.2} GiB", card.name, d.breakdown(), cap as f64 / GIB),
    };
    fit.assumptions = est.assumptions;
}

/// The label to suggest: the highest-quality choice (the largest file, F32
/// counting below everything else since it carries no more than BF16/F16)
/// that fits one card, else the best two-card split, else the best `Tight`
/// one-card and then two-card fit. None when nothing fits.
pub fn recommend(fits: &[ChoiceFit]) -> Option<String> {
    let is_f32 = |f: &ChoiceFit| matches!(f.quant.as_deref(), Some("F32" | "FP32"));
    let best = |pick: &dyn Fn(&ChoiceFit) -> bool| {
        fits.iter().filter(|f| pick(f)).max_by_key(|f| (!is_f32(f), f.total_size)).map(|f| f.label.clone())
    };
    best(&|f| f.one_card.fit == Fit::Fits)
        .or_else(|| best(&|f| f.two_card_split.fit == Fit::Fits))
        .or_else(|| best(&|f| f.one_card.fit == Fit::Tight))
        .or_else(|| best(&|f| f.two_card_split.fit == Fit::Tight))
}

// ------------------------------------------------------------ destination ----

/// Where a repo file is saved: `<root>\<owner>\<repo>\<file name>`. Repo
/// folders are flattened, so a split set's shards sit side by side (as
/// llama.cpp needs) and `hf::repo_from_layout` still finds the repo.
/// Characters Windows forbids in a name, and reserved device names, are
/// replaced; ids the Hub accepts never contain them.
pub fn dest_path(model_root: &Path, repo: &str, file_path: &str) -> PathBuf {
    let (owner, name) = repo.split_once('/').unwrap_or(("", repo));
    let file = file_path.rsplit(['/', '\\']).next().unwrap_or(file_path);
    model_root.join(safe_component(owner)).join(safe_component(name)).join(safe_component(file))
}

fn safe_component(s: &str) -> String {
    if s.chars().all(|c| c == '.') {
        return "_".into(); // "", "." and ".."
    }
    let mut out: String =
        s.chars().map(|c| if c.is_control() || "<>:\"/\\|?*".contains(c) { '_' } else { c }).collect();
    // Windows drops trailing dots and spaces from a name.
    let kept = out.trim_end_matches(['.', ' ']).len();
    if kept < out.len() {
        out.truncate(kept);
        out.push('_');
    }
    let stem = out.split('.').next().unwrap_or("").to_ascii_uppercase();
    let reserved = ["CON", "PRN", "AUX", "NUL"].contains(&stem.as_str())
        || ((stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.len() == 4
            && stem.as_bytes()[3].is_ascii_digit());
    if reserved {
        out.insert(0, '_');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::parse_model_info;

    fn fixture_files(name: &str) -> Vec<RepoFile> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/hub").join(name);
        parse_model_info(&std::fs::read_to_string(p).unwrap()).unwrap().siblings
    }

    fn file(path: &str, size: u64) -> RepoFile {
        RepoFile { path: path.into(), size, sha256: None }
    }

    fn labels(c: &Catalog) -> Vec<&str> {
        c.choices.iter().map(|x| x.label.as_str()).collect()
    }

    #[test]
    fn quant_labels() {
        let q = |s: &str| quant_label(s);
        assert_eq!(q("gemma-4-26B-A4B-it-UD-Q4_K_XL").as_deref(), Some("UD-Q4_K_XL"));
        assert_eq!(q("K2-Horizon-7B-Q4_K_M").as_deref(), Some("Q4_K_M"));
        assert_eq!(q("K2-Horizon-MoVA-36B-A4B-IQ4_XS").as_deref(), Some("IQ4_XS"));
        assert_eq!(q("K2-Horizon-MoVA-36B-A4B-MXFP4_MOE").as_deref(), Some("MXFP4_MOE"));
        assert_eq!(q("K2-Horizon-7B-BF16").as_deref(), Some("BF16"));
        assert_eq!(q("K2-Horizon-MoVA-36B-A4B-Q4_0_ROCMFP4_FAST").as_deref(), Some("Q4_0_ROCMFP4_FAST"));
        assert_eq!(q("gemma-3-4b-it-qat-q4_0").as_deref(), Some("Q4_0"));
        assert_eq!(q("Qwen3.8-27B.Q8_0").as_deref(), Some("Q8_0"), "mradermacher's dots");
        assert_eq!(q("model_q4_k_m").as_deref(), Some("Q4_K_M"));
        assert_eq!(q("K2-Horizon-7B-Q4_K_M-imat").as_deref(), Some("Q4_K_M"));
        assert_eq!(q("ggml-model-f16").as_deref(), Some("F16"));
        assert_eq!(q("K2-Horizon-MoVA-36B-A4B-TQ1_0").as_deref(), Some("TQ1_0"));
        assert_eq!(q("Qwen3-30B-A3B"), None, "Qwen3 and A3B are not quants");
        assert_eq!(q("K2-Horizon-7B"), None);
        assert_eq!(q("model-Q9_0"), None);
    }

    /// unsloth's layout: UD quants at the root, a split BF16 in a folder,
    /// projectors at the root, MTP heads in MTP/ and at the root, and an
    /// importance matrix with a disguised extension.
    #[test]
    fn classifies_an_unsloth_repo() {
        let files = fixture_files("model-info-unsloth__gemma-4-26B-A4B-it-GGUF.json");
        let c = classify(&files);
        assert_eq!(c.choices.len(), 21, "{:?}", labels(&c));
        let bf16 = c.choices.iter().find(|x| x.label == "BF16").unwrap();
        assert_eq!(bf16.files.len(), 2);
        assert_eq!(bf16.first_file, "BF16/gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf");
        assert!(bf16.files[1].path.ends_with("-00002-of-00002.gguf"));
        assert_eq!(bf16.total_size, 49_923_215_552 + 581_922_272);
        for l in ["UD-Q4_K_XL", "UD-IQ2_XXS", "MXFP4_MOE", "Q8_0", "UD-Q8_K_XL", "UD-Q6_K"] {
            assert!(c.choices.iter().any(|x| x.label == l), "{l} missing from {:?}", labels(&c));
        }
        assert!(c.choices.windows(2).all(|w| w[0].total_size <= w[1].total_size), "smallest first");
        assert_eq!(c.mmproj.iter().map(|f| f.path.as_str()).collect::<Vec<_>>(), ["mmproj-BF16.gguf", "mmproj-F16.gguf", "mmproj-F32.gguf"]);
        assert_eq!(c.drafts.len(), 4, "{:?}", c.drafts);
        let ignored: Vec<&str> = c.ignored.iter().map(|(f, _)| f.path.as_str()).collect();
        assert_eq!(ignored, [".gitattributes", "MTP/README.md", "README.md", "config.json", "imatrix_unsloth.gguf_file"]);
        let (_, why) = c.ignored.iter().find(|(f, _)| f.path.starts_with("imatrix")).unwrap();
        assert!(why.contains("importance matrix"));
        let accounted = c.choices.iter().map(|x| x.files.len()).sum::<usize>() + c.mmproj.len() + c.drafts.len() + c.ignored.len();
        assert_eq!(accounted, files.len());
    }

    #[test]
    fn classifies_k2_quant_ladders() {
        let c = classify(&fixture_files("model-info-ngquocvinh__K2-Horizon-7B-GGUF.json"));
        assert_eq!(c.choices.len(), 15, "{:?}", labels(&c));
        assert!(c.choices.iter().all(|x| x.quant.is_some() && x.files.len() == 1));
        assert!(c.mmproj.is_empty() && c.drafts.is_empty());
        let (f, why) = c.ignored.iter().find(|(f, _)| f.path.ends_with(".imatrix.gguf")).unwrap();
        assert_eq!(f.path, "reproducibility/k2_horizon_7b_combined.imatrix.gguf");
        assert!(why.contains("importance matrix"));

        let c = classify(&fixture_files("model-info-NANI-Nithin__K2-Horizon-MoVA-36B-A4B-GGUF.json"));
        assert_eq!(c.choices.len(), 31);
        assert_eq!(c.choices.first().unwrap().label, "Q1_0");
        assert_eq!(c.choices.last().unwrap().label, "BF16");
        let q4 = c.choices.iter().find(|x| x.label == "Q4_K_M").unwrap();
        assert_eq!(q4.total_size, 22_368_011_616);
        assert!(q4.files[0].sha256.is_some());

        let c = classify(&fixture_files("model-info-IFM__K2-Horizon-7B-GGUF.json"));
        assert_eq!(labels(&c), ["BF16"]);
    }

    #[test]
    fn split_sets_labels_and_leftovers() {
        let files = vec![
            file("Q8_0/m-Q8_0-00002-of-00002.gguf", 20),
            file("Q8_0/m-Q8_0-00001-of-00002.gguf", 10),
            file("big/m-F16-00001-of-00003.gguf", 1),
            file("big/m-F16-00003-of-00003.gguf", 1),
            file("m-Q4_K_M.gguf", 5),
            file("other-Q4_K_M.gguf", 6),
            file("mystery.gguf", 7),
            file("mtp-m-Q8_0.gguf", 1),
            file("m-Q6_K-draft.gguf", 1),
            file("model.gguf.part", 1),
        ];
        let c = classify(&files);
        let q8 = c.choices.iter().find(|x| x.label == "Q8_0").unwrap();
        assert_eq!(q8.first_file, "Q8_0/m-Q8_0-00001-of-00002.gguf", "shards ordered by number");
        assert_eq!(q8.total_size, 30);
        assert!(c.choices.iter().any(|x| x.label == "Q4_K_M (m-Q4_K_M.gguf)"));
        assert!(c.choices.iter().any(|x| x.label == "Q4_K_M (other-Q4_K_M.gguf)"));
        let mystery = c.choices.iter().find(|x| x.label == "mystery").unwrap();
        assert_eq!(mystery.quant, None);
        let incomplete: Vec<&String> =
            c.ignored.iter().filter(|(f, _)| f.path.starts_with("big/")).map(|(_, why)| why).collect();
        assert_eq!(incomplete.len(), 2);
        assert!(incomplete[0].contains("parts 1, 3 of 3"), "{}", incomplete[0]);
        assert_eq!(c.drafts.len(), 1, "mtp-: {:?}", c.drafts);
        assert!(c.choices.iter().any(|x| x.first_file == "m-Q6_K-draft.gguf"), "only a draft- prefix marks a sidecar");
        assert!(c.ignored.iter().any(|(f, why)| f.path == "model.gguf.part" && why == "not a GGUF file"));
    }

    // ---------------------------------------------------------------- fits

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

    /// The architecture keys of IFM's K2-Horizon-MoVA-36B-A4B: 48 layers,
    /// 8 KV heads of 128 dims, no sliding window (192 KiB of f16 KV per
    /// token).
    fn mova_36b() -> GgufHeader {
        serde_json::from_value(serde_json::json!({
            "path": "hf://NANI-Nithin/K2-Horizon-MoVA-36B-A4B-GGUF@x/K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf",
            "file_size": 0, "gguf_version": 3, "tensor_count": 0, "architecture": "k2-horizon",
            "block_count": 48, "context_length": 524288, "embedding_length": 2560, "head_count": 20,
            "head_count_kv": 8, "key_length": 128, "value_length": 128, "metadata": {}
        }))
        .unwrap()
    }

    #[test]
    fn k2_mova_fits_and_recommendation() {
        let c = classify(&fixture_files("model-info-NANI-Nithin__K2-Horizon-MoVA-36B-A4B-GGUF.json"));
        let devices = [r9700("pci:a:bus03", 32472), igpu(), r9700("pci:a:bus08", 1000)];
        let fits = estimate_choices(&mova_36b(), &c.choices, &devices, &FitOptions::default());
        assert_eq!(fits.len(), 31);
        let get = |l: &str| fits.iter().find(|f| f.label == l).unwrap();
        assert!(fits.iter().all(|f| f.ctx == DEFAULT_CTX));

        // By hand at 32K, f16 KV, ub 512: Q4_K_M = 20.83 GiB weights + 6.00 KV
        // + 1.25 compute + 0.40 overhead = 28.48 GiB of 31.86: 89%.
        let q4 = get("Q4_K_M");
        assert_eq!(q4.one_card.fit, Fit::Fits, "{}", q4.one_card.detail);
        let gib = q4.one_card.need_bytes as f64 / GIB;
        assert!((gib - 28.48).abs() < 0.02, "{gib}");
        assert_eq!(q4.one_card.capacity_bytes, 32624 * MIB);
        assert!(q4.one_card.fits_free_now, "judged on the idle card");
        assert!(q4.one_card.detail.contains("R9700"));
        // Q5_K_M needs a second card; layer split over both fits.
        let q5 = get("Q5_K_M");
        assert_eq!(q5.one_card.fit, Fit::NoFit);
        assert_eq!(q5.two_card_split.fit, Fit::Fits, "{}", q5.two_card_split.detail);
        assert!(!q5.two_card_split.fits_free_now, "the second card is busy right now");
        assert_eq!(get("BF16").two_card_split.fit, Fit::NoFit);
        // Context: Q4_K_M has ~3.3 GiB to spare at 90% of the card, about
        // 17K more tokens of 192 KiB.
        let max = q4.max_ctx_one_card.unwrap();
        assert!((32_768..65_536).contains(&max) && max % 1024 == 0, "{max}");
        assert_eq!(get("Q1_0").max_ctx_one_card, fits.iter().map(|f| f.max_ctx_one_card).max().unwrap());
        assert_eq!(recommend(&fits).as_deref(), Some("Q4_K_M"));

        // Asking for 128K moves the recommendation down the ladder.
        let opts = FitOptions { ctx: Some(131_072), kv_type_k: "q8_0".into(), kv_type_v: "q8_0".into(), ..Default::default() };
        let fits = estimate_choices(&mova_36b(), &c.choices, &devices, &opts);
        assert!(fits.iter().all(|f| f.ctx == 131_072));
        let pick = recommend(&fits).unwrap();
        let picked = fits.iter().find(|f| f.label == pick).unwrap();
        assert_eq!(picked.one_card.fit, Fit::Fits);
        assert!(picked.total_size < 22_368_011_616, "{pick}");
    }

    /// Two cards of one size, the one that sorts first busy with another
    /// model: a choice that fits the idle one fits one card now, and the
    /// split puts its main (compute) share on the idle card.
    #[test]
    fn equal_cards_are_judged_on_the_idle_one() {
        let c = classify(&fixture_files("model-info-NANI-Nithin__K2-Horizon-MoVA-36B-A4B-GGUF.json"));
        for devices in [
            [r9700("pci:a:bus03", 1000), r9700("pci:a:bus08", 32472)],
            [r9700("pci:a:bus08", 32472), r9700("pci:a:bus03", 1000)],
        ] {
            assert_eq!(cards(&devices)[0].stable_key, "pci:a:bus08");
            let fits = estimate_choices(&mova_36b(), &c.choices, &devices, &FitOptions::default());
            let q4 = fits.iter().find(|f| f.label == "Q4_K_M").unwrap();
            assert_eq!(q4.one_card.fit, Fit::Fits);
            assert!(q4.one_card.fits_free_now, "bus08 is idle: {}", q4.one_card.detail);
        }
        // Equal free VRAM too: the order stays stable.
        let devices = [r9700("pci:a:bus08", 32472), r9700("pci:a:bus03", 32472)];
        assert_eq!(cards(&devices)[0].stable_key, "pci:a:bus03");
    }

    /// K2-Horizon-7B's own keys, from the 64 KiB prefix of a real file: BF16
    /// fits one card at 32K (16.77 + 4.50 KV + 1.25 + 0.40 = 22.92 GiB) and
    /// is the recommendation, F32 aside.
    #[test]
    fn k2_7b_from_a_real_header() {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gguf/k2-horizon-7b-q4_k_m.head64k.bin"),
        )
        .unwrap();
        let h = gguf::read_header_from(&mut std::io::Cursor::new(bytes), 5_592_219_008, Path::new("k2"), gguf::ReadMode::UntilTokenizer)
            .unwrap();
        let mut files = fixture_files("model-info-ngquocvinh__K2-Horizon-7B-GGUF.json");
        files.push(file("K2-Horizon-7B-BF16.gguf", 18_010_413_440));
        files.push(file("K2-Horizon-7B-F32.gguf", 36_020_826_880));
        let c = classify(&files);
        let one = [r9700("pci:a:bus03", 32472)];
        let fits = estimate_choices(&h, &c.choices, &one, &FitOptions::default());
        let bf16 = fits.iter().find(|f| f.label == "BF16").unwrap();
        let gib = bf16.one_card.need_bytes as f64 / GIB;
        assert!((gib - 22.92).abs() < 0.02, "{gib}: {}", bf16.one_card.detail);
        assert_eq!(bf16.one_card.fit, Fit::Fits);
        assert_eq!(bf16.two_card_split.fit, Fit::NotApplicable, "one GPU");
        assert_eq!(fits.iter().find(|f| f.label == "F32").unwrap().one_card.fit, Fit::NoFit);
        assert_eq!(recommend(&fits).as_deref(), Some("BF16"));

        // No GPU: nothing applies and nothing is recommended.
        let none = estimate_choices(&h, &c.choices, &[], &FitOptions::default());
        assert!(none.iter().all(|f| f.one_card.fit == Fit::NotApplicable && f.max_ctx_one_card.is_none()));
        assert_eq!(recommend(&none), None);
    }

    #[test]
    fn recommendation_order() {
        let v = |fit: Fit| FitVerdict { fit, need_bytes: 0, capacity_bytes: 0, fits_free_now: true, detail: String::new() };
        let f = |label: &str, size: u64, one: Fit, two: Fit| ChoiceFit {
            label: label.into(),
            quant: Some(label.into()),
            total_size: size,
            ctx: 1,
            one_card: v(one),
            two_card_split: v(two),
            max_ctx_one_card: None,
            assumptions: vec![],
        };
        let fits = [
            f("Q4_K_M", 10, Fit::Fits, Fit::Fits),
            f("F32", 99, Fit::Fits, Fit::Fits),
            f("Q8_0", 20, Fit::NoFit, Fit::Fits),
        ];
        assert_eq!(recommend(&fits).as_deref(), Some("Q4_K_M"), "one card first, F32 last");
        let fits = [f("Q4_K_M", 10, Fit::Tight, Fit::Fits), f("Q8_0", 20, Fit::NoFit, Fit::Fits)];
        assert_eq!(recommend(&fits).as_deref(), Some("Q8_0"), "a comfortable split beats a tight card");
        let fits = [f("Q4_K_M", 10, Fit::Tight, Fit::NotApplicable), f("Q8_0", 20, Fit::NoFit, Fit::NotApplicable)];
        assert_eq!(recommend(&fits).as_deref(), Some("Q4_K_M"));
        let fits = [f("BF16", 10, Fit::NoFit, Fit::NoFit)];
        assert_eq!(recommend(&fits), None);
    }

    #[test]
    fn diffusion_models_fit_one_card() {
        let swa: Vec<bool> = (0..30).map(|l| (l + 1) % 6 != 0).collect();
        let kv: Vec<u64> = swa.iter().map(|s| if *s { 8 } else { 2 }).collect();
        let h: GgufHeader = serde_json::from_value(serde_json::json!({
            "path": "dg", "file_size": 0, "gguf_version": 3, "tensor_count": 0,
            "architecture": "diffusion-gemma", "block_count": 30, "context_length": 262144,
            "embedding_length": 2816, "head_count": 16, "head_count_kv_per_layer": kv,
            "key_length": 512, "value_length": 512, "key_length_swa": 256, "value_length_swa": 256,
            "sliding_window": 1024, "swa_layer_flags": swa, "diffusion_canvas_length": 256,
            "vocab_size": 262144, "metadata": {}
        }))
        .unwrap();
        let c = classify(&[file("diffusiongemma-26B-A4B-it-Q4_K_M.gguf", 16_806_810_208)]);
        let devices = [r9700("pci:a:bus03", 32472), r9700("pci:a:bus08", 32472)];
        let fits = estimate_choices(&h, &c.choices, &devices, &FitOptions::default());
        assert_eq!(fits[0].two_card_split.fit, Fit::NotApplicable);
        assert_ne!(fits[0].one_card.fit, Fit::NoFit, "{}", fits[0].one_card.detail);
        // The stock runner sizes itself to 12,288 on an idle R9700 (see estimate.rs).
        assert_eq!(fits[0].max_ctx_one_card, Some(12_288));
        assert_eq!(fits[0].ctx, 12_288);
    }

    #[test]
    fn destinations_flatten_repo_folders() {
        let root = Path::new(r"E:\models");
        assert_eq!(
            dest_path(root, "unsloth/gemma-4-26B-A4B-it-GGUF", "BF16/gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf"),
            PathBuf::from(r"E:\models\unsloth\gemma-4-26B-A4B-it-GGUF\gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf")
        );
        let p = dest_path(root, "IFM/K2-Horizon-7B-GGUF", "K2-Horizon-7B-BF16.gguf");
        assert_eq!(
            crate::hf::repo_from_layout(&[root.to_path_buf()], &p).as_deref(),
            Some("IFM/K2-Horizon-7B-GGUF"),
            "the layout still names the repo"
        );
        assert_eq!(safe_component("a:b?.gguf"), "a_b_.gguf");
        assert_eq!(safe_component(".."), "_");
        assert_eq!(safe_component(""), "_");
        assert_eq!(safe_component("nul"), "_nul");
        assert_eq!(safe_component("COM1.gguf"), "_COM1.gguf");
        assert_eq!(safe_component("COMPUTE.gguf"), "COMPUTE.gguf");
        assert_eq!(safe_component("x."), "x_");
        assert_eq!(safe_component("x. ."), "x_");
        assert_eq!(safe_component(" "), "_");
        assert_eq!(safe_component("..."), "_");
        assert_eq!(dest_path(root, "o/r", "../../evil.gguf"), PathBuf::from(r"E:\models\o\r\evil.gguf"));
    }
}
