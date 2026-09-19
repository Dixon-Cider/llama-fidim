//! Which build to use for a model, in order of preference:
//!
//! 1. an installed build that knows the model;
//! 2. upstream's newest release, a prebuilt download, when its source at
//!    that tag knows it;
//! 3. Unsloth's newest mix, a prebuilt download: for DiffusionGemma, or
//!    when the mix merges an upstream pull request that adds the model;
//! 4. an upstream pull request that adds it, built from source;
//! 5. a llama.cpp fork linked from the model card, built from source;
//! 6. nothing: say why.
//!
//! Building a pull request or a fork runs code nobody reviewed for this
//! machine, so those steps carry `needs_consent`; so does an Unsloth mix
//! chosen for an unmerged pull request. Within that order a verified step
//! (the source was read) beats an unverified one (only a heuristic or
//! nothing could be checked), which is offered as an alternative.
//!
//! `plan_build` is pure over what `gather_plan_inputs` collected, so every
//! decision is tested offline.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::github::{self, Api, GitRef, PrCandidate, RefKind, ResolvedRef};
use super::{
    compat_err, describe_missing, probe_build, probe_source, source_caps, source_of, ModelNeeds, Support, UPSTREAM_OWNER,
    UPSTREAM_REPO,
};
use crate::config::Config;
use crate::discovery::{Build, Channel};
use crate::update::{self, Release, SourceRef};

/// An installed build with the probe's answer.
#[derive(Debug, Clone, Serialize)]
pub struct ProbedBuild {
    pub build: Build,
    pub support: Support,
}

/// Upstream's newest release with its source's answer at that tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamCandidate {
    pub release: Release,
    pub support: Support,
}

/// A link from the model card, resolved and checked (or why not).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CardCandidate {
    pub git_ref: GitRef,
    pub resolved: Option<ResolvedRef>,
    /// Why it could not be resolved.
    pub error: Option<String>,
    /// The source's answer at the resolved commit; None when unresolved.
    pub support: Option<Support>,
}

/// Everything `plan_build` decides from.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PlanInputs {
    pub installed: Vec<ProbedBuild>,
    pub upstream_latest: Option<UpstreamCandidate>,
    pub unsloth_latest: Option<Release>,
    pub prs: Vec<PrCandidate>,
    pub card_refs: Vec<CardCandidate>,
    /// GPU target(s) for source builds, e.g. `gfx1201`.
    pub gfx: String,
    /// Lookups that failed (network, rate limit): reported, never fatal.
    pub errors: Vec<String>,
}

/// One way to get a build that loads the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum PlanAction {
    UseInstalled { path: PathBuf, name: String },
    InstallUpstream { tag: String, install_dir: PathBuf },
    InstallUnsloth { tag: String, gfx: String, install_dir: PathBuf },
    BuildPr { number: u32, source: SourceRef, gpu_targets: String, install_dir: PathBuf },
    BuildFork { owner: String, repo: String, source: SourceRef, gpu_targets: String, install_dir: PathBuf },
    Unsupported { reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStep {
    pub action: PlanAction,
    /// What this step is and why it works, in a sentence or three.
    pub explanation: String,
    /// Code from a pull request or a fork: ask before running it.
    pub needs_consent: bool,
    /// The support was read from source (or an exact table), not guessed.
    pub verified: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildPlan {
    pub needs: ModelNeeds,
    /// The recommendation; `Unsupported` when nothing fits.
    pub step: PlanStep,
    /// Other ways that would also work, best first.
    pub alternatives: Vec<PlanStep>,
    /// One line per candidate that was ruled out, in the order considered.
    pub rejected: Vec<String>,
}

fn short(sha: &str) -> &str {
    sha.get(..7).unwrap_or(sha)
}

fn build_name(b: &Build) -> String {
    b.git.as_ref().map(|g| g.display()).unwrap_or_else(|| b.tag.clone())
}

/// Installed builds best first: upstream (newest release first), git
/// builds, Unsloth, anything else.
fn installed_rank(b: &Build) -> (u8, std::cmp::Reverse<u32>) {
    let ch = match b.channel {
        Channel::Upstream => 0,
        Channel::Git => 1,
        Channel::Unsloth => 2,
        Channel::Other => 3,
    };
    let n = b.version.as_deref().and_then(update::version_number).unwrap_or(0);
    (ch, std::cmp::Reverse(n))
}

/// Unsloth's zip for this card: its exact target, else its family
/// (`gfx1201` -> `gfx120X`).
fn unsloth_target(release: &Release, gfx: &str) -> Option<String> {
    let first = gfx.split(',').next().unwrap_or("").trim().to_string();
    let family = (first.is_ascii() && first.len() > 1).then(|| format!("{}X", &first[..first.len() - 1]));
    [Some(first), family].into_iter().flatten().find(|g| update::select_unsloth_asset(release, g).is_ok())
}

/// Upstream pull request numbers an Unsloth release body says it merged.
pub fn unsloth_merged_prs(body: &str) -> Vec<u32> {
    let re = regex::Regex::new(r"github\.com/ggml-org/llama\.cpp/pull/(\d+)").unwrap();
    let mut out: Vec<u32> = re.captures_iter(body).filter_map(|c| c[1].parse().ok()).collect();
    out.sort_unstable();
    out.dedup();
    out
}

struct Collector {
    verified: Vec<PlanStep>,
    unverified: Vec<PlanStep>,
    rejected: Vec<String>,
}

impl Collector {
    /// File a candidate by its support: a step, an unverified step, or a
    /// rejection that names what is missing.
    fn offer(&mut self, needs: &ModelNeeds, support: &Support, what: &str, mut step: PlanStep) {
        match support {
            Support::Yes => {
                step.verified = true;
                self.verified.push(step);
            }
            Support::Unknown(why) => {
                step.verified = false;
                step.warnings.insert(0, format!("not verified: {why}"));
                self.unverified.push(step);
            }
            Support::No { missing } => self.rejected.push(format!("{what} lacks {}", describe_missing(needs, missing))),
        }
    }
}

/// Order every candidate into one recommendation (see the module notes).
pub fn plan_build(cfg: &Config, needs: &ModelNeeds, inputs: &PlanInputs) -> BuildPlan {
    let mut c = Collector { verified: vec![], unverified: vec![], rejected: vec![] };
    let diffusion = needs.engine.is_diffusion();
    let gpu_targets = update::normalize_gpu_targets(&inputs.gfx).unwrap_or_default();
    let gfx_warning = gpu_targets.is_empty().then(|| {
        format!("no GPU target is known for this machine (`{}`); pass one (e.g. gfx1201) before building", inputs.gfx)
    });

    // 1. Installed builds: the best that works, and the best that might.
    let mut installed: Vec<&ProbedBuild> = inputs.installed.iter().collect();
    installed.sort_by_key(|p| installed_rank(&p.build));
    let mut lacking: Vec<String> = Vec::new();
    let (mut took_yes, mut took_unknown) = (false, false);
    for p in installed {
        let name = build_name(&p.build);
        let step = PlanStep {
            action: PlanAction::UseInstalled { path: p.build.path.clone(), name: name.clone() },
            explanation: format!("{name} is installed and knows architecture '{}'.", needs.arch),
            needs_consent: false,
            verified: false,
            warnings: vec![],
        };
        match &p.support {
            Support::Yes if !took_yes => {
                took_yes = true;
                c.offer(needs, &p.support, &name, step);
            }
            Support::Unknown(_) if !took_unknown => {
                took_unknown = true;
                c.offer(needs, &p.support, &name, step);
            }
            Support::No { .. } => lacking.push(name),
            _ => {}
        }
    }
    if !lacking.is_empty() {
        let missing = inputs
            .installed
            .iter()
            .find_map(|p| match &p.support {
                Support::No { missing } => Some(describe_missing(needs, missing)),
                _ => None,
            })
            .unwrap_or_default();
        c.rejected.push(format!("installed builds lack {missing}: {}", lacking.join(", ")));
    }
    if inputs.installed.is_empty() {
        c.rejected.push("no llama.cpp build is installed".into());
    }

    // 2. Upstream's newest release.
    if diffusion {
        c.rejected.push("upstream releases carry no DiffusionGemma runner".into());
    } else if let Some(u) = &inputs.upstream_latest {
        let tag = &u.release.tag;
        let what = format!("upstream {tag} (the newest release)");
        if let Support::No { missing } = &u.support {
            c.rejected.push(format!("{what} lacks {}", describe_missing(needs, missing)));
        } else {
            match (update::select_assets(&u.release), update::install_dir(cfg, tag, "rocm")) {
                (Ok(_), Ok(dir)) => c.offer(
                    needs,
                    &u.support,
                    &what,
                    PlanStep {
                        action: PlanAction::InstallUpstream { tag: tag.clone(), install_dir: dir },
                        explanation: format!(
                            "Upstream {tag} knows architecture '{}': install its prebuilt Windows ROCm build \
                             (a download of about 260 MB, nothing to compile).",
                            needs.arch
                        ),
                        needs_consent: false,
                        verified: false,
                        warnings: vec![],
                    },
                ),
                (Err(e), _) | (_, Err(e)) => c.rejected.push(format!("{what}: {e}")),
            }
        }
    } else {
        c.rejected.push("upstream's newest release was not checked".into());
    }

    // 3. Unsloth's mix.
    let verified_prs: Vec<&PrCandidate> = inputs.prs.iter().filter(|p| p.support.is_yes()).collect();
    if let Some(rel) = &inputs.unsloth_latest {
        let merged = unsloth_merged_prs(&rel.body);
        let via_pr = verified_prs.iter().find(|p| merged.contains(&p.pr.number));
        let runner_model = diffusion && needs.arch == "diffusion-gemma";
        if runner_model || via_pr.is_some() {
            match (unsloth_target(rel, &inputs.gfx), update::install_dir(cfg, &rel.tag, "unsloth")) {
                (Some(g), Ok(dir)) => {
                    let (explanation, consent, warnings) = match via_pr {
                        Some(p) if !runner_model => (
                            format!(
                                "Unsloth's {} merges upstream pull request #{} ({}), which adds architecture '{}': \
                                 install its prebuilt Windows ROCm zip ({g}).",
                                rel.tag, p.pr.number, p.pr.title, needs.arch
                            ),
                            true,
                            vec![
                                "Unsloth's mix is upstream plus pull requests upstream has not merged".into(),
                                "profiles are never promoted onto it; pick it in the profile editor".into(),
                            ],
                        ),
                        _ => (
                            format!("Unsloth's {} carries the DiffusionGemma runner: install its prebuilt zip ({g}).", rel.tag),
                            false,
                            vec![],
                        ),
                    };
                    c.verified.push(PlanStep {
                        action: PlanAction::InstallUnsloth { tag: rel.tag.clone(), gfx: g, install_dir: dir },
                        explanation,
                        needs_consent: consent,
                        verified: true,
                        warnings,
                    });
                }
                (None, _) => c.rejected.push(format!("Unsloth {} has no Windows ROCm zip for {}", rel.tag, inputs.gfx)),
                (_, Err(e)) => c.rejected.push(format!("Unsloth {}: {e}", rel.tag)),
            }
        }
    } else if diffusion {
        c.rejected.push("Unsloth's newest release was not checked".into());
    }

    // 4. Upstream pull requests, from the search and from the card.
    let mut prs: Vec<PrCandidate> = inputs.prs.clone();
    for cc in &inputs.card_refs {
        if let (Some(r), Some(s)) = (&cc.resolved, &cc.support) {
            if r.is_upstream {
                if let Some(pr) = &r.pr {
                    if !prs.iter().any(|p| p.pr.number == pr.number) {
                        prs.push(PrCandidate { pr: pr.clone(), support: s.clone() });
                    }
                }
            }
        }
    }
    prs.sort_by_key(|p| {
        let pr = &p.pr;
        (pr.merged, pr.draft, pr.mergeable_state.as_deref() == Some("dirty"), std::cmp::Reverse(pr.number))
    });
    for p in &prs {
        let pr = &p.pr;
        if pr.state != "open" && !pr.merged {
            c.rejected.push(format!("pull request #{} was closed without merging", pr.number));
            continue;
        }
        let source = SourceRef {
            remote_url: format!("https://github.com/{UPSTREAM_OWNER}/{UPSTREAM_REPO}"),
            git_ref: format!("pull/{}/head", pr.number),
            sha: pr.head_sha.to_ascii_lowercase(),
            label: format!("PR #{}", pr.number),
        };
        let dir = match update::source_install_dir(cfg, &source) {
            Ok(d) => d,
            Err(e) => {
                c.rejected.push(format!("pull request #{}: {e}", pr.number));
                continue;
            }
        };
        let mut warnings = Vec::new();
        if pr.draft {
            warnings.push("a draft: its author does not consider it finished".to_string());
        }
        if pr.mergeable_state.as_deref() == Some("dirty") {
            warnings.push("it conflicts with upstream master; its head commit is built as it is".to_string());
        }
        if pr.merged {
            warnings.push(
                "already merged: the next upstream release carries it (usually 2-4 hours after the merge), and \
                 installing that needs no compiler"
                    .to_string(),
            );
        }
        warnings.extend(gfx_warning.clone());
        c.offer(
            needs,
            &p.support,
            &format!("pull request #{} at {}", pr.number, short(&pr.head_sha)),
            PlanStep {
                action: PlanAction::BuildPr { number: pr.number, source, gpu_targets: gpu_targets.clone(), install_dir: dir },
                explanation: format!(
                    "Upstream pull request #{} \"{}\" adds architecture '{}' (read at its head {}): build it from \
                     source for {} ({BUILD_COST}).",
                    pr.number,
                    pr.title,
                    needs.arch,
                    short(&pr.head_sha),
                    if gpu_targets.is_empty() { "this GPU" } else { gpu_targets.as_str() }
                ),
                needs_consent: true,
                verified: false,
                warnings,
            },
        );
    }
    if prs.is_empty() && !diffusion {
        c.rejected.push(format!("no open upstream pull request mentions '{}'", needs.arch));
    }

    // 5. Forks linked from the model card.
    let mut forks = 0;
    for cc in &inputs.card_refs {
        let r = &cc.git_ref;
        // A bare link to upstream is how cards say "use llama.cpp"; its
        // releases were rung 2. Upstream pull requests were rung 4.
        if r.is_upstream_repo() && matches!(r.kind, RefKind::Repo | RefKind::Pull(_)) {
            continue;
        }
        forks += 1;
        let Some(res) = &cc.resolved else {
            c.rejected.push(format!("{}: {}", r.url(), cc.error.as_deref().unwrap_or("not resolved")));
            continue;
        };
        if res.is_upstream && res.pr.is_some() {
            continue;
        }
        let support = cc.support.clone().unwrap_or_else(|| Support::Unknown("its source was not read".into()));
        let source = res.source_ref();
        let dir = match update::source_install_dir(cfg, &source) {
            Ok(d) => d,
            Err(e) => {
                c.rejected.push(format!("{}: {e}", r.url()));
                continue;
            }
        };
        let mut warnings = Vec::new();
        if res.redirected {
            warnings.push(format!(
                "github.com/{}/{} now redirects to {}/{} (the repository was renamed or transferred)",
                r.owner, r.repo, res.owner, res.repo
            ));
        }
        if !res.is_upstream && !res.is_fork_of_ggml {
            warnings.push(
                "not in ggml-org/llama.cpp's fork network: unrelated code, review it before building".to_string(),
            );
        }
        if let Some(b) = res.behind_by.filter(|b| *b >= 500) {
            warnings.push(format!("{b} commits behind upstream master: fixes since then are missing"));
        }
        if let Some(pr) = &res.pr {
            if pr.draft {
                warnings.push("a draft pull request".to_string());
            }
            if pr.state != "open" && !pr.merged {
                warnings.push("a closed pull request".to_string());
            }
        }
        warnings.extend(gfx_warning.clone());
        let distance = match (res.ahead_by, res.behind_by) {
            (Some(a), Some(b)) => format!(", {a} commits ahead of upstream master and {b} behind"),
            _ => String::new(),
        };
        let subjects = if res.head_subjects.is_empty() {
            String::new()
        } else {
            let last: Vec<String> = res.head_subjects.iter().rev().take(3).map(|s| format!("\"{s}\"")).collect();
            format!(" Its newest commits: {}.", last.join(", "))
        };
        let label = source.git_source().display();
        c.offer(
            needs,
            &support,
            &format!("{} ({})", r.url(), label),
            PlanStep {
                action: PlanAction::BuildFork {
                    owner: res.owner.clone(),
                    repo: res.repo.clone(),
                    source,
                    gpu_targets: gpu_targets.clone(),
                    install_dir: dir,
                },
                explanation: format!(
                    "The model card links {}; {}/{} {} at {}{distance} knows architecture '{}': build it from \
                     source ({BUILD_COST}).{subjects}",
                    r.url(),
                    res.owner,
                    res.repo,
                    res.git_ref,
                    short(&res.sha),
                    needs.arch
                ),
                needs_consent: true,
                verified: false,
                warnings,
            },
        );
    }
    if forks == 0 {
        c.rejected.push("the model card links no llama.cpp fork".into());
    }

    let mut steps = c.verified;
    steps.extend(c.unverified);
    let mut rejected = c.rejected;
    rejected.extend(inputs.errors.iter().map(|e| format!("could not check: {e}")));
    let step = if steps.is_empty() {
        let type_note = needs
            .max_type_id
            .map(|t| {
                format!(
                    " The file uses ggml tensor type {t}; a build defines only the types its own ggml knows, and \
                     forks number theirs differently."
                )
            })
            .unwrap_or_default();
        PlanStep {
            action: PlanAction::Unsupported {
                reason: format!(
                    "No build found that loads architecture '{}'{}.{type_note} Checked: {}.",
                    needs.arch,
                    needs.tokenizer_pre.as_ref().map(|p| format!(" with pre-tokenizer '{p}'")).unwrap_or_default(),
                    rejected.join("; ")
                ),
            },
            explanation: "Nothing installed, released, proposed upstream or linked from the model card can load it."
                .into(),
            needs_consent: false,
            verified: false,
            warnings: vec![],
        }
    } else {
        steps.remove(0)
    };
    BuildPlan { needs: needs.clone(), step, alternatives: steps, rejected }
}

/// What a source build costs, as measured for the K2 Horizon fork (one GPU
/// target, 478 Ninja edges, 16 threads): 6.2 minutes end to end, 65 MB of
/// git objects fetched the first time, about 0.5 GB of scratch space at the
/// peak, 91 MB installed.
const BUILD_COST: &str = "several minutes of compiling, about 0.6 GB of scratch space and 0.1 GB installed";

/// At most this many card links are resolved (three or four API requests
/// each).
const MAX_CARD_LOOKUPS: usize = 3;

/// Collect what `plan_build` needs, cheapest first, stopping as soon as a
/// verified answer makes the rest moot: installed builds (offline), then
/// upstream's newest release (one API request, source read at the tag),
/// then upstream pull requests (two searches plus one request each), then
/// the card's links. `upstream_latest` saves the release lookup when the
/// caller already has it; `model_hint` (e.g. "K2 Horizon") widens the pull
/// request search.
pub fn gather_plan_inputs(
    cfg: &Config,
    needs: &ModelNeeds,
    installed: &[Build],
    card_refs: &[GitRef],
    gfx: &str,
    model_hint: Option<&str>,
    upstream_latest: Option<Release>,
) -> PlanInputs {
    let api = Api::new(cfg);
    let mut inputs = PlanInputs { gfx: gfx.to_string(), ..Default::default() };
    inputs.installed = installed.iter().map(|b| ProbedBuild { build: b.clone(), support: probe_build(b, needs) }).collect();
    // A tensor type the probe could not check offline: read the build's
    // source tables once (cached per commit), then ask again.
    if needs.max_type_id.is_some() {
        for p in inputs.installed.iter_mut().filter(|p| matches!(p.support, Support::Unknown(_))) {
            if let Some((o, r, sha)) = source_of(&p.build.path, p.build.commit.as_deref()) {
                if source_caps(&o, &r, &sha).is_ok() {
                    p.support = probe_build(&p.build, needs);
                }
            }
        }
    }
    if inputs.installed.iter().any(|p| p.support.is_yes()) {
        return inputs;
    }
    let diffusion = needs.engine.is_diffusion();
    if diffusion {
        match latest_unsloth(&api) {
            Ok(r) => inputs.unsloth_latest = Some(r),
            Err(e) => inputs.errors.push(format!("Unsloth releases: {e}")),
        }
        return inputs;
    }
    let latest = match upstream_latest {
        Some(r) => Ok(r),
        None => latest_upstream(&api),
    };
    match latest {
        Ok(release) => {
            let support = probe_source(UPSTREAM_OWNER, UPSTREAM_REPO, &release.tag, needs)
                .unwrap_or_else(|e| Support::Unknown(format!("its source could not be read: {e}")));
            let done = support.is_yes();
            inputs.upstream_latest = Some(UpstreamCandidate { release, support });
            if done {
                return inputs;
            }
        }
        Err(e) => inputs.errors.push(format!("upstream releases: {e}")),
    }
    match github::find_upstream_pr_with(&api, needs, model_hint, &mut |sha: &str| {
        source_caps(UPSTREAM_OWNER, UPSTREAM_REPO, sha).map(|c| c.support(needs))
    }) {
        Ok(p) => inputs.prs = p,
        Err(e) => inputs.errors.push(format!("upstream pull requests: {e}")),
    }
    if inputs.prs.iter().any(|p| p.support.is_yes()) {
        match latest_unsloth(&api) {
            Ok(r) => inputs.unsloth_latest = Some(r),
            Err(e) => inputs.errors.push(format!("Unsloth releases: {e}")),
        }
        return inputs;
    }
    let lookups = card_refs.iter().filter(|r| !(r.is_upstream_repo() && r.kind == RefKind::Repo)).take(MAX_CARD_LOOKUPS);
    for r in lookups {
        inputs.card_refs.push(match github::resolve_ref_with(&api, r) {
            Ok(res) => {
                let support = probe_source(&res.owner, &res.repo, &res.sha, needs)
                    .unwrap_or_else(|e| Support::Unknown(format!("its source could not be read: {e}")));
                CardCandidate { git_ref: r.clone(), resolved: Some(res), error: None, support: Some(support) }
            }
            Err(e) => CardCandidate { git_ref: r.clone(), resolved: None, error: Some(e.to_string()), support: None },
        });
    }
    inputs
}

fn latest_upstream(api: &Api) -> crate::Result<Release> {
    let body = api
        .get(&format!("/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/releases?per_page=30"), "application/vnd.github+json")?
        .ok_or_else(|| compat_err("upstream has no releases"))?;
    update::pick_latest_binary_release(&body)
}

fn latest_unsloth(api: &Api) -> crate::Result<Release> {
    let body = api
        .get("/repos/unslothai/llama.cpp/releases?per_page=30", "application/vnd.github+json")?
        .ok_or_else(|| compat_err("unslothai/llama.cpp has no releases"))?;
    update::pick_latest_unsloth_release(&body)
}
