//! `fidim models`: the model wizard ("Get a model") on the command line.
//! Thin over `fidim_core::wizard`, as the app's Models tab is.

use std::borrow::Cow;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::{bail, Context};
use clap::Subcommand;
use fidim_core::catalog::{ChoiceFit, Fit, FitVerdict};
use fidim_core::compat::{BuildPlan, PlanAction, PlanStep, Support};
use fidim_core::config::Config;
use fidim_core::hub::{Gated, SearchQuery, Sort};
use fidim_core::wizard::{
    self, BuildChoice, BuildSource, BuildVerdict, Level, Note, PlanRequest, RepoKind, StepAction, StepStatus, WizardEvent,
    WizardPlan,
};

#[derive(Subcommand, Debug)]
pub enum ModelsCmd {
    /// Search Hugging Face for GGUF models.
    Search {
        /// Words of the repo name, e.g. K2-Horizon.
        query: String,
        #[arg(long, default_value = "20")]
        limit: u32,
        /// Every repo, not only those with GGUF files.
        #[arg(long)]
        all: bool,
        /// downloads, likes, trending, modified or created.
        #[arg(long, default_value = "downloads")]
        sort: String,
    },
    /// A repo's files, what fits this machine's cards, and which build can
    /// load it (or how to get one). Reads only file headers.
    Show {
        /// owner/name, or a huggingface.co link.
        repo: String,
        /// A branch, tag or commit (default: the main branch).
        #[arg(long)]
        rev: Option<String>,
        /// GPU target(s) a source build in the plan is for, e.g. gfx1201
        /// (default: what ROCm's hipInfo or the card names say).
        #[arg(long)]
        gfx: Option<String>,
    },
    /// What a model needs from a llama.cpp build (architecture,
    /// pre-tokenizer, tensor types), which installed builds have it, and
    /// where to get one when none does.
    Needs {
        /// A local .gguf file, or owner/name, or a huggingface.co link.
        target: String,
        /// GPU target(s) a source build in the plan is for, e.g. gfx1201.
        #[arg(long)]
        gfx: Option<String>,
    },
    /// Download a model, and the build it needs, then create a profile.
    /// Prints the plan and asks first; nothing is launched.
    Get {
        /// owner/name, or a huggingface.co link.
        repo: String,
        #[arg(long)]
        rev: Option<String>,
        /// The quant to download, e.g. Q4_K_M or UD-Q4_K_XL (default: the
        /// best that fits one card).
        #[arg(long, conflicts_with = "file")]
        quant: Option<String>,
        /// A file of the repo by its path, as `show` lists it.
        #[arg(long)]
        file: Option<String>,
        /// Also the vision projector: this file, or with no value the best one.
        #[arg(long, num_args = 0..=1, default_missing_value = "auto", value_name = "FILE")]
        mmproj: Option<String>,
        /// Also a draft / MTP head: this file, or with no value the best one.
        #[arg(long, num_args = 0..=1, default_missing_value = "auto", value_name = "FILE")]
        draft: Option<String>,
        /// Model folder to save into (default: the first model folder).
        #[arg(long)]
        dest: Option<PathBuf>,
        /// `auto` (an installed build that loads it, else the build plan),
        /// `none` (no install or build), or an installed build's folder.
        #[arg(long, default_value = "auto")]
        build: String,
        /// Allow building code from a pull request or a fork, which nobody
        /// reviewed for your machine.
        #[arg(long)]
        allow_fork: bool,
        /// Start without asking.
        #[arg(long)]
        yes: bool,
        /// Download only; create no profile.
        #[arg(long)]
        no_profile: bool,
        /// The profile's context (default: what fits one card, at most 32768).
        #[arg(long)]
        ctx: Option<u64>,
        /// GPU target(s) to compile for when the plan builds a pull request
        /// or a fork, e.g. gfx1201 (default: what ROCm's hipInfo or the card
        /// names say).
        #[arg(long)]
        gfx: Option<String>,
    },
}

pub fn cmd_models(cfg: &Config, json: bool, cmd: ModelsCmd) -> anyhow::Result<()> {
    match cmd {
        ModelsCmd::Search { query, limit, all, sort } => search(cfg, json, &query, limit, all, &sort),
        ModelsCmd::Show { repo, rev, gfx } => show(cfg, json, &repo, rev.as_deref(), gfx),
        ModelsCmd::Needs { target, gfx } => needs(cfg, json, &target, gfx),
        ModelsCmd::Get { repo, rev, quant, file, mmproj, draft, dest, build, allow_fork, yes, no_profile, ctx, gfx } => {
            let opts = GetOpts { quant, file, mmproj, draft, dest, build, allow_fork, yes, no_profile, ctx, gfx };
            get(cfg, json, &repo, rev.as_deref(), opts)
        }
    }
}

/// The machine for the wizard, with the GPU target the user gave (checked).
fn env(cfg: &Config, gfx: Option<String>) -> anyhow::Result<wizard::LiveEnv> {
    let gfx = gfx.map(|g| fidim_core::update::normalize_gpu_targets(&g)).transpose()?;
    Ok(wizard::LiveEnv { gfx, ..wizard::LiveEnv::new(cfg.clone()) })
}

// ---------------------------------------------------------------- format ----

const GIB: f64 = (1u64 << 30) as f64;

/// Text from a model, a model card, the Hub or GitHub, made safe to print:
/// control characters (ESC starts the terminal's cursor and colour
/// sequences, CR and LF start lines) and the bidirectional overrides that
/// reorder what is shown come out as `\u{..}` escapes. A GGUF's metadata
/// and a card's YAML are whatever their authors wrote, and a terminal
/// that obeyed them could hide or fake a line of the plan or of the
/// consent question.
fn safe(s: &str) -> Cow<'_, str> {
    let bad = |c: char| {
        c.is_control()
            || matches!(c, '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
    };
    if !s.chars().any(bad) {
        return Cow::Borrowed(s);
    }
    Cow::Owned(s.chars().map(|c| if bad(c) { c.escape_unicode().to_string() } else { c.to_string() }).collect())
}

/// A path, made safe to print.
fn safe_path(p: &std::path::Path) -> String {
    safe(&p.to_string_lossy()).into_owned()
}

fn gib(b: u64) -> String {
    wizard::human_size(b)
}

fn short(s: &str) -> &str {
    s.get(..7).unwrap_or(s)
}

fn params(n: Option<u64>) -> String {
    match n {
        Some(n) if n >= 1_000_000_000 => format!("{:.1}B", n as f64 / 1e9),
        Some(n) if n >= 1_000_000 => format!("{:.0}M", n as f64 / 1e6),
        Some(n) => n.to_string(),
        None => "-".into(),
    }
}

fn verdict(v: &FitVerdict) -> String {
    let cap = v.capacity_bytes as f64 / GIB;
    let need = v.need_bytes as f64 / GIB;
    match v.fit {
        Fit::Fits => format!("fits {need:.1}/{cap:.0}"),
        Fit::Tight => format!("tight {need:.1}/{cap:.0}"),
        Fit::NoFit => format!("no {need:.1}/{cap:.0}"),
        Fit::NotApplicable => "-".into(),
    }
}

fn support(s: &Support, needs: &fidim_core::compat::ModelNeeds) -> String {
    match s {
        Support::Yes => "yes".into(),
        Support::No { missing } => format!("no: lacks {}", safe(&fidim_core::compat::describe_missing(needs, missing))),
        Support::Unknown(why) => format!("maybe: {}", safe(why)),
    }
}

fn print_notes(notes: &[Note]) {
    for n in notes {
        let tag = match n.level {
            Level::Error => "ERROR",
            Level::Warning => "WARN ",
            Level::Info => "note ",
        };
        println!("  {tag} {}", safe(&n.message));
    }
}

fn print_builds(builds: &[BuildVerdict], needs: Option<&fidim_core::compat::ModelNeeds>) {
    let Some(needs) = needs else { return };
    println!("\ninstalled builds");
    if builds.is_empty() {
        println!("  none");
    }
    let w = builds.iter().map(|b| safe(&b.name).chars().count()).max().unwrap_or(0).max(10);
    for b in builds {
        println!(
            "  {:<w$} {:<9} {:<8} {}{}",
            safe(&b.name),
            b.channel.as_str(),
            safe(b.version.as_deref().unwrap_or("?")),
            support(&b.support, needs),
            if b.broken { " (does not run)" } else { "" }
        );
    }
}

fn print_step(s: &PlanStep, prefix: &str) {
    println!("{prefix}{}{}", safe(&s.explanation), if s.needs_consent { "  [needs your consent]" } else { "" });
    for w in &s.warnings {
        println!("{prefix}  ! {}", safe(w));
    }
}

fn print_build_plan(bp: &BuildPlan) {
    println!("\nbuild plan");
    print_step(&bp.step, "  -> ");
    for (i, a) in bp.alternatives.iter().enumerate() {
        print_step(a, &format!("  alt {}: ", i + 1));
    }
    if !bp.rejected.is_empty() {
        println!("  ruled out:");
        for r in &bp.rejected {
            println!("    - {}", safe(r));
        }
    }
}

// ---------------------------------------------------------------- search ----

fn search(cfg: &Config, json: bool, query: &str, limit: u32, all: bool, sort: &str) -> anyhow::Result<()> {
    let sort = match sort {
        "downloads" => Sort::Downloads,
        "likes" => Sort::Likes,
        "trending" => Sort::Trending,
        "modified" => Sort::LastModified,
        "created" => Sort::Created,
        other => bail!("unknown --sort `{other}`: use downloads, likes, trending, modified or created"),
    };
    let q = SearchQuery { text: query.to_string(), author: None, sort, limit, gguf_only: !all };
    let hits = wizard::search(cfg, &q)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&hits)?);
        return Ok(());
    }
    if hits.is_empty() {
        println!("nothing on Hugging Face matches {:?}{}", safe(query), if all { "" } else { " with GGUF files (--all for every format)" });
        return Ok(());
    }
    println!("{:<52} {:<16} {:>7} {:>9} {:>6} {:<7} {:<10} BUILD", "REPO", "ARCH", "PARAMS", "DOWNLOADS", "LIKES", "GATED", "MODIFIED");
    for h in &hits {
        let r = &h.hit;
        let gated = match r.gated {
            Gated::No => "-",
            Gated::Auto => "yes",
            Gated::Manual => "manual",
        };
        let build = match h.arch_known {
            Some(true) => "installed",
            Some(false) => "needs one",
            None => "-",
        };
        println!(
            "{:<52} {:<16} {:>7} {:>9} {:>6} {:<7} {:<10} {}",
            safe(&r.id),
            safe(r.arch.as_deref().unwrap_or("-")),
            params(r.total_params),
            r.downloads,
            r.likes,
            gated,
            safe(r.last_modified.as_deref().map(|d| d.get(..10).unwrap_or(d)).unwrap_or("-")),
            build
        );
    }
    println!("\n`fidim models show <repo>` for its files, what fits and which build loads it.");
    Ok(())
}

// ------------------------------------------------------------------ show ----

fn fits_table(fits: &[ChoiceFit], recommended: Option<&str>, labels: &[(String, String)]) {
    println!(
        "\n  {:<2}{:<24} {:>9}  {:<16} {:<16} {:>9}  FILE",
        "", "CHOICE", "SIZE", "ONE CARD", "TWO-CARD SPLIT", "MAX CTX"
    );
    for (label, file) in labels {
        let f = fits.iter().find(|f| &f.label == label);
        let mark = if recommended == Some(label.as_str()) { "* " } else { "  " };
        let (label, file) = (safe(label), safe(file));
        match f {
            Some(f) => println!(
                "  {mark}{:<24} {:>9}  {:<16} {:<16} {:>9}  {file}",
                label,
                gib(f.total_size),
                verdict(&f.one_card),
                verdict(&f.two_card_split),
                f.max_ctx_one_card.map(|c| c.to_string()).unwrap_or_else(|| "-".into())
            ),
            None => println!("  {mark}{label:<24} {:>9}  {:<16} {:<16} {:>9}  {file}", "", "-", "-", "-"),
        }
    }
}

fn show(cfg: &Config, json: bool, repo: &str, rev: Option<&str>, gfx: Option<String>) -> anyhow::Result<()> {
    let view = wizard::inspect_with(&env(cfg, gfx)?, cfg, repo, rev)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&view)?);
        return Ok(());
    }
    let kind = match &view.kind {
        RepoKind::Gguf => "GGUF".to_string(),
        RepoKind::Safetensors => "safetensors (no GGUF)".into(),
        RepoKind::Adapter => "LoRA adapter".into(),
        RepoKind::OtherFormat(f) => format!("{f} (not for llama.cpp)"),
        RepoKind::Empty => "no model files".into(),
    };
    println!("{} @ {}   {}", safe(&view.repo), safe(short(&view.sha)), safe(&kind));
    if let Some(n) = &view.needs {
        println!(
            "architecture {}{}{}",
            safe(&n.arch),
            n.tokenizer_pre.as_deref().map(|p| format!(", pre-tokenizer {}", safe(p))).unwrap_or_default(),
            view.header.as_ref().and_then(|h| h.context_length).map(|c| format!(", trained context {c}")).unwrap_or_default()
        );
    }
    print_notes(&view.notes);
    if !view.derivatives.is_empty() {
        println!("\nGGUF quantizations of {} on the Hub:", safe(view.derivatives_of.as_deref().unwrap_or(&view.repo)));
        for d in view.derivatives.iter().take(20) {
            println!("  {:<56} {:>9} downloads", safe(&d.id), d.downloads);
        }
    }
    if view.kind != RepoKind::Gguf {
        return Ok(());
    }
    let labels: Vec<(String, String)> =
        view.catalog.choices.iter().map(|c| (c.label.clone(), format!("{}{}", c.first_file, if c.files.len() > 1 { format!(" (+{} parts)", c.files.len() - 1) } else { String::new() }))).collect();
    let cards: Vec<String> = view.devices.iter().filter(|d| !d.integrated).map(|d| format!("{} ({:.1} GiB)", safe(&d.name), d.total_mib as f64 / 1024.0)).collect();
    println!("\ncards: {}", if cards.is_empty() { "none found".to_string() } else { cards.join(", ") });
    let ctx = view.fits.first().map(|f| f.ctx).unwrap_or(0);
    println!("estimates at context {ctx}, f16 KV; * = recommended");
    fits_table(&view.fits, view.recommended.as_deref(), &labels);
    if !view.catalog.mmproj.is_empty() {
        let names: Vec<String> = view.catalog.mmproj.iter().map(|f| format!("{} ({})", safe(&f.path), gib(f.size))).collect();
        println!("  projectors: {}", names.join(", "));
    }
    if !view.catalog.drafts.is_empty() {
        // Each with the speculative mode a profile would run it in.
        let names: Vec<String> = view
            .catalog
            .drafts
            .iter()
            .map(|f| {
                let mode = match view.draft_modes.get(&f.path) {
                    Some(Some(m)) => m.clone(),
                    _ => format!("{}, which profiles cannot run", fidim_core::profile::unsupported_draft_kind(&f.path)),
                };
                format!("{} ({}, {mode})", safe(&f.path), gib(f.size))
            })
            .collect();
        println!("  drafts: {}", names.join(", "));
    }
    print_builds(&view.builds, view.needs.as_ref());
    match (&view.usable_build, &view.build_plan) {
        (Some(p), _) => println!("\n{} can load it.", safe(view.builds.iter().find(|b| &b.path == p).map(|b| b.name.as_str()).unwrap_or("an installed build"))),
        (None, Some(bp)) => {
            print_build_plan(bp);
            // The sources of the steps offered, not the ones ruled out.
            let offered: Vec<&str> = std::iter::once(&bp.step)
                .chain(&bp.alternatives)
                .filter_map(|s| match &s.action {
                    PlanAction::BuildFork { source, .. } | PlanAction::BuildPr { source, .. } => Some(source.sha.as_str()),
                    _ => None,
                })
                .collect();
            for src in view.build_sources.iter().filter(|s| offered.iter().any(|o| o.eq_ignore_ascii_case(&s.sha))) {
                println!();
                print_source(src);
            }
        }
        _ => {}
    }
    println!(
        "\n`fidim models get {}{}` downloads{}, then creates a profile.",
        safe(&view.repo),
        view.recommended.as_deref().map(|r| format!(" --quant {}", safe(r))).unwrap_or_default(),
        if view.usable_build.is_none() { " (and gets a build)" } else { "" }
    );
    Ok(())
}

// ----------------------------------------------------------------- needs ----

fn needs(cfg: &Config, json: bool, target: &str, gfx: Option<String>) -> anyhow::Result<()> {
    let r = wizard::needs_with(&env(cfg, gfx)?, cfg, target)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
        return Ok(());
    }
    let n = &r.needs;
    println!("{}{}", safe(&r.target), r.repo.as_deref().filter(|rp| *rp != r.target).map(|rp| format!("  ({})", safe(rp))).unwrap_or_default());
    println!("  architecture   {}", safe(&n.arch));
    println!("  pre-tokenizer  {}", safe(n.tokenizer_pre.as_deref().unwrap_or("(not needed: not a BPE vocabulary)")));
    println!(
        "  tensor types   {}",
        match (n.max_type_id, r.max_tensor_type) {
            (Some(t), _) => format!("up to {} (id {t}), checked against each build", fidim_core::gguf::ggml_type_name(t)),
            (None, Some(t)) => format!("up to {} (id {t}), which every build knows", fidim_core::gguf::ggml_type_name(t)),
            (None, None) => "not read".into(),
        }
    );
    println!("  engine         {}", n.engine.label());
    print_notes(&r.notes);
    print_builds(&r.builds, Some(n));
    if let Some(bp) = &r.build_plan {
        print_build_plan(bp);
    }
    Ok(())
}

// ------------------------------------------------------------------- get ----

#[derive(Debug)]
struct GetOpts {
    quant: Option<String>,
    file: Option<String>,
    mmproj: Option<String>,
    draft: Option<String>,
    dest: Option<PathBuf>,
    build: String,
    allow_fork: bool,
    yes: bool,
    no_profile: bool,
    ctx: Option<u64>,
    gfx: Option<String>,
}

/// `--build auto|none|<dir>`.
fn build_choice(s: &str) -> BuildChoice {
    match s.trim() {
        "" | "auto" => BuildChoice::Auto,
        "none" | "skip" => BuildChoice::Skip,
        dir => BuildChoice::Installed(PathBuf::from(dir)),
    }
}

fn print_plan(p: &WizardPlan) {
    println!("{} @ {}", safe(&p.repo), safe(short(&p.sha)));
    let fit = p
        .fit
        .as_ref()
        .map(|f| format!("one card: {}, two-card split: {}", verdict(&f.one_card), verdict(&f.two_card_split)))
        .unwrap_or_else(|| "not estimated".into());
    println!("file       {} ({}), {fit}", safe(&p.choice.label), gib(p.choice.total_size));
    if let Some(b) = &p.build {
        println!("build      {}{}", safe(&b.name), if b.installed { "" } else { " (installed by this plan)" });
    }
    if let Some(StepAction::Build { plan: ps }) = p.steps.first().map(|s| &s.action) {
        if let PlanAction::BuildFork { source, .. } | PlanAction::BuildPr { source, .. } = &ps.action {
            if let Some(src) = p.build_sources.iter().find(|s| s.sha.eq_ignore_ascii_case(&source.sha)) {
                print_source(src);
            }
        }
    }
    println!("folder     {}", safe_path(&p.dest_dir));
    println!("to fetch   {} of {}", gib(p.download_bytes), gib(p.total_bytes));
    println!("steps");
    for (i, s) in p.steps.iter().enumerate() {
        println!("  {}. {}{}", i + 1, safe(&s.title), if wizard::step_needs_consent(s) { "  [needs your consent]" } else { "" });
        if !s.detail.is_empty() {
            for line in s.detail.lines() {
                println!("     {}", safe(line));
            }
        }
    }
    let tool_problems: Vec<_> = p.toolchain.iter().filter(|t| t.outcome != "pass").collect();
    if !tool_problems.is_empty() {
        println!("toolchain");
        for t in tool_problems {
            println!(
                "  {:<5} {}: {}{}",
                safe(&t.outcome),
                safe(&t.title),
                safe(&t.message),
                t.fix.as_deref().map(|f| format!(" (fix: {})", safe(f))).unwrap_or_default()
            );
        }
    }
    if !p.notes.is_empty() {
        println!("notes");
        print_notes(&p.notes);
    }
}

fn print_source(src: &BuildSource) {
    let (owner, repo) = (safe(&src.owner), safe(&src.repo));
    println!("source     github.com/{owner}/{repo} {} @ {}", safe(&src.git_ref), safe(&src.sha));
    if let Some(l) = &src.linked_as {
        println!("           (the card links {}, which GitHub now sends to {owner}/{repo})", safe(l));
    }
    if let (Some(a), Some(b)) = (src.ahead_by, src.behind_by) {
        println!("           {a} commits ahead of upstream master, {b} behind");
    }
    if let Some(pr) = &src.pr {
        println!("           pull request #{} \"{}\" ({}{})", pr.number, safe(&pr.title), safe(&pr.state), if pr.draft { ", draft" } else { "" });
    }
    for subj in src.subjects.iter().rev().take(5) {
        println!("           - {}", safe(subj));
    }
}

fn ask(question: &str) -> anyhow::Result<bool> {
    print!("{question} [y/N] ");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).context("reading the answer")?;
    Ok(matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes"))
}

/// The first Ctrl+C raises this; downloads keep their `.part` for a resume.
static CANCEL: AtomicBool = AtomicBool::new(false);

fn get(cfg: &Config, json: bool, repo: &str, rev: Option<&str>, o: GetOpts) -> anyhow::Result<()> {
    let mut env = env(cfg, o.gfx.clone())?;
    let view = wizard::inspect_with(&env, cfg, repo, rev)?;
    if view.kind != RepoKind::Gguf {
        let why = view.notes.iter().find(|n| n.level == Level::Error).map(|n| safe(&n.message).into_owned()).unwrap_or_default();
        let alts: Vec<String> = view.derivatives.iter().take(5).map(|d| safe(&d.id).into_owned()).collect();
        bail!("{why}{}", if alts.is_empty() { String::new() } else { format!("\n  try: {}", alts.join(", ")) });
    }
    let choice = match (&o.quant, &o.file) {
        (_, Some(f)) => Some(
            view.catalog
                .choices
                .iter()
                .find(|c| c.files.iter().any(|x| x.path == *f || x.name().eq_ignore_ascii_case(f)))
                .map(|c| c.label.clone())
                .with_context(|| format!("{} has no model file {:?} (see `fidim models show {}`)", safe(&view.repo), safe(f), safe(&view.repo)))?,
        ),
        (Some(q), None) => Some(q.clone()),
        _ => None,
    };
    let label = choice.clone().or_else(|| view.recommended.clone());
    let chosen = label.as_ref().and_then(|l| view.catalog.choices.iter().find(|c| &c.label == l || c.quant.as_deref().is_some_and(|q| q.eq_ignore_ascii_case(l))));
    let pick_aux = |arg: &Option<String>, auto: Option<String>, what: &str| -> anyhow::Result<Option<String>> {
        match arg.as_deref() {
            None => Ok(None),
            Some("auto") => auto.map(Some).with_context(|| format!("{} has no {what}", safe(&view.repo))),
            Some(name) => Ok(Some(name.to_string())),
        }
    };
    let mmproj = pick_aux(&o.mmproj, wizard::default_mmproj(&view.catalog).map(|f| f.path.clone()), "vision projector")?;
    let draft = pick_aux(
        &o.draft,
        chosen.and_then(|c| wizard::default_draft(&view.catalog, c)).map(|f| f.path.clone()),
        "draft model",
    )?;
    let req = PlanRequest {
        choice,
        mmproj,
        draft,
        dest_root: o.dest.clone(),
        build: build_choice(&o.build),
        profile: !o.no_profile,
        ctx: o.ctx,
    };
    // The cards inspect saw, so the plan's estimate is the view's.
    env.devices = Some(view.devices.clone()).filter(|d| !d.is_empty());
    let plan = wizard::plan_with(&env, cfg, &view, &req)?;
    if json && !o.yes {
        // A dry run: the plan, nothing done.
        println!("{}", serde_json::to_string_pretty(&plan)?);
        return Ok(());
    }
    if !json {
        print_plan(&plan);
    }
    if plan.blocked {
        bail!("the plan has errors (above); nothing was done");
    }
    if plan.requires_consent() {
        let text = safe(plan.consent.as_deref().unwrap_or("A step runs code nobody reviewed for your machine.")).into_owned();
        if !o.allow_fork {
            bail!("{text}\nPass --allow-fork to build it, or --build none to download the files only; nothing was done.");
        }
        if !json {
            println!("\n{text} (--allow-fork given)");
        }
    }
    if plan.steps.is_empty() {
        if json {
            println!("{}", serde_json::to_string_pretty(&plan)?);
        } else {
            println!("nothing to do");
        }
        return Ok(());
    }
    if !o.yes && !ask("\nGo ahead?")? {
        println!("nothing was done");
        return Ok(());
    }
    fidim_core::platform::cancel_on_ctrl_c(&CANCEL);
    let started = Instant::now();
    let mut printer = Printer::new(&plan, json);
    let r = wizard::run(cfg, None, &plan, o.allow_fork, &mut |e| printer.event(e), &CANCEL);
    printer.finish_line();
    let r = match r {
        Ok(r) => r,
        Err(fidim_core::Error::Cancelled) => {
            bail!("stopped: finished downloads are kept and a partial one resumes from its .part file the next time")
        }
        Err(e) => return Err(e.into()),
    };
    if json {
        println!("{}", serde_json::to_string_pretty(&r)?);
        return Ok(());
    }
    println!("\ndone in {:.1} min", started.elapsed().as_secs_f64() / 60.0);
    for f in &r.files {
        println!("  file     {}", safe_path(f));
    }
    if let Some(b) = &r.build {
        println!(
            "  build    {} at {} ({}; HIP {})",
            safe(&b.tag),
            safe_path(&b.dir),
            safe(b.verify.version.as_deref().unwrap_or("?")),
            if b.verify.hip_ok { "OK" } else { "NOT LOADED" }
        );
    }
    for w in &r.warnings {
        println!("  WARN     {}", safe(w));
    }
    match (&r.profile, &r.profile_path) {
        (Some(p), Some(path)) => {
            println!("  profile  {} ({})", p.id, safe_path(path));
            println!("\nnothing was launched: `fidim check {id}` runs the pre-flight, `fidim launch {id}` loads it.", id = p.id);
        }
        _ => println!("\nno profile was created"),
    }
    Ok(())
}

/// Progress on the terminal: one line per step, a download's bytes and
/// speed rewritten in place (a line per tenth when not a terminal), a
/// build's Ninja count every 5%. With --json, nothing: stdout is one JSON
/// document, the result, as with `fidim update --source`.
struct Printer<'a> {
    out: Box<dyn Write + 'a>,
    plan: &'a WizardPlan,
    json: bool,
    tty: bool,
    last_step: Option<usize>,
    last_print: Instant,
    last_tenth: u64,
    last_done: u64,
    stage: String,
    open_line: bool,
}

impl<'a> Printer<'a> {
    fn new(plan: &'a WizardPlan, json: bool) -> Self {
        Printer::to(Box::new(std::io::stdout()), std::io::stdout().is_terminal(), plan, json)
    }

    fn to(out: Box<dyn Write + 'a>, tty: bool, plan: &'a WizardPlan, json: bool) -> Self {
        Printer {
            out,
            plan,
            json,
            tty,
            last_step: None,
            last_print: Instant::now(),
            last_tenth: 0,
            last_done: 0,
            stage: String::new(),
            open_line: false,
        }
    }

    fn finish_line(&mut self) {
        if self.open_line {
            let _ = writeln!(self.out);
            self.open_line = false;
        }
    }

    fn event(&mut self, e: &WizardEvent) {
        if self.json {
            return;
        }
        let n = self.plan.steps.len();
        if self.last_step != Some(e.step) {
            self.finish_line();
            self.last_step = Some(e.step);
            self.last_tenth = 0;
            self.last_done = 0;
            self.stage.clear();
            if let Some(s) = self.plan.steps.get(e.step) {
                let _ = writeln!(self.out, "== [{}/{n}] {}", e.step + 1, safe(&s.title));
            }
        }
        match e.status {
            StepStatus::Failed => {
                self.finish_line();
                let _ = writeln!(self.out, "   failed: {}", safe(e.line.as_deref().unwrap_or("?")));
                return;
            }
            StepStatus::Done => {
                self.finish_line();
                let _ = writeln!(self.out, "   done");
                return;
            }
            StepStatus::Skipped => {
                self.finish_line();
                let _ = writeln!(self.out, "   skipped");
                return;
            }
            _ => {}
        }
        let is_download = matches!(self.plan.steps.get(e.step).map(|s| &s.action), Some(StepAction::Download { .. }));
        if is_download {
            let (Some(done), Some(total)) = (e.done, e.total) else { return };
            let stage = safe(e.stage.as_deref().unwrap_or_default()).into_owned();
            let tenth = (done * 10).checked_div(total).unwrap_or(0);
            let now = Instant::now();
            let due = now.duration_since(self.last_print).as_millis() >= 500 || done == total || stage != self.stage;
            if self.tty && due {
                let bps = e.bps.unwrap_or(0.0);
                let eta = if bps > 0.0 && total > done { format!("  {}", eta((total - done) as f64 / bps)) } else { String::new() };
                let _ = write!(self.out, "\r   {:<11} {} / {}  {:>6.1} MB/s{eta}   ", stage, gib(done), gib(total), bps / 1e6);
                self.out.flush().ok();
                self.open_line = true;
                self.last_print = now;
            } else if !self.tty && (tenth > self.last_tenth || stage != self.stage) {
                let _ = writeln!(self.out, "   {stage} {} / {}", gib(done), gib(total));
                self.last_tenth = tenth;
            }
            self.stage = stage;
            return;
        }
        // A build or an install.
        if let Some(stage) = e.stage.as_deref().map(safe) {
            if *stage != self.stage {
                self.finish_line();
                let _ = writeln!(self.out, "   -- {stage}");
                self.stage = stage.into_owned();
            }
        }
        match (e.done, e.total) {
            (Some(d), Some(t)) if t > 0 => {
                if d == t || d < self.last_done || (d - self.last_done) * 20 >= t {
                    self.last_done = d;
                    let _ = writeln!(self.out, "   [{d}/{t}] {}%", d * 100 / t);
                }
            }
            // Compiler warnings run to thousands of lines; a failure's last
            // lines come back in the error anyway.
            _ if self.stage == "build" => {
                if let Some(l) = e.line.as_deref().filter(|l| l.contains("error") || l.contains("FAILED")) {
                    let _ = writeln!(self.out, "   {}", safe(l));
                }
            }
            _ => {
                if let Some(l) = &e.line {
                    let _ = writeln!(self.out, "   {}", safe(l));
                }
            }
        }
    }
}

fn eta(secs: f64) -> String {
    let s = secs.round() as u64;
    if s >= 3600 {
        format!("ETA {}h{:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("ETA {}m{:02}s", s / 60, s % 60)
    } else {
        format!("ETA {s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_choices() {
        assert_eq!(build_choice("auto"), BuildChoice::Auto);
        assert_eq!(build_choice(""), BuildChoice::Auto);
        assert_eq!(build_choice("none"), BuildChoice::Skip);
        assert_eq!(build_choice(r"C:\b\b10984-rocm"), BuildChoice::Installed(PathBuf::from(r"C:\b\b10984-rocm")));
    }

    /// A plan of one download, as `plan` makes it.
    fn one_download(title: &str) -> WizardPlan {
        serde_json::from_value(serde_json::json!({
            "repo": "a/b", "sha": "0".repeat(40), "dest_root": r"C:\m", "dest_dir": r"C:\m",
            "choice": { "label": "Q8_0", "quant": "Q8_0", "files": [{ "path": "m.gguf", "size": 100, "sha256": null }],
                        "total_size": 100, "first_file": "m.gguf" },
            "steps": [{ "title": title, "detail": "", "needs_consent": false,
                        "action": { "kind": "download", "role": "model", "file": { "path": "m.gguf", "size": 100, "sha256": null },
                                    "dest": r"C:\m\m.gguf", "have": 0, "present": false } }],
            "notes": [], "blocked": false, "needs_consent": false, "download_bytes": 100, "total_bytes": 100,
            "engine": "llama-server", "gated": "no", "build_sources": [], "build_choice": { "kind": "auto" },
            "toolchain": [], "profile_opts": {}, "devices": []
        }))
        .unwrap()
    }

    fn progress(plan: &WizardPlan, json: bool, events: &[WizardEvent]) -> String {
        let buf = std::rc::Rc::new(std::cell::RefCell::new(Vec::<u8>::new()));
        struct Shared(std::rc::Rc<std::cell::RefCell<Vec<u8>>>);
        impl Write for Shared {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.borrow_mut().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut p = Printer::to(Box::new(Shared(buf.clone())), false, plan, json);
        for e in events {
            p.event(e);
        }
        p.finish_line();
        drop(p);
        let out = buf.borrow().clone();
        String::from_utf8(out).unwrap()
    }

    fn download_events() -> Vec<WizardEvent> {
        let at = |done| WizardEvent {
            step: 0, status: StepStatus::Running, stage: Some("downloading".into()), file: Some("m.gguf".into()),
            done: Some(done), total: Some(100), bps: Some(1e6), line: None,
        };
        let done = WizardEvent { step: 0, status: StepStatus::Done, stage: None, file: None, done: None, total: None, bps: None, line: None };
        vec![at(0), at(50), at(100), done]
    }

    /// With --json, stdout is the one result document: no progress lines.
    #[test]
    fn json_progress_prints_nothing() {
        let plan = one_download("Download m.gguf (the model)");
        assert_eq!(progress(&plan, true, &download_events()), "");
        let text = progress(&plan, false, &download_events());
        assert!(text.starts_with("== [1/1] Download m.gguf (the model)
"), "{text}");
        assert!(text.contains("
   downloading 1 KiB / 1 KiB
"), "{text}");
        assert!(text.ends_with("   done
"), "{text}");
    }

    /// A GGUF's architecture or a card's text with terminal sequences in it
    /// prints as escapes: no line of the plan can be erased, moved or hidden.
    #[test]
    fn untrusted_text_prints_without_control_characters() {
        let evil = "k2\x1b[2K\x1b[1A\x1b[2K\rSPOOFED-LINE\npre\x1b[8mHIDDEN\u{9b}2J\u{202e}txt.exe";
        let shown = safe(evil);
        assert!(!shown.chars().any(|c| c.is_control() || c == '\u{202e}'), "{shown}");
        assert_eq!(
            shown,
            r"k2\u{1b}[2K\u{1b}[1A\u{1b}[2K\u{d}SPOOFED-LINE\u{a}pre\u{1b}[8mHIDDEN\u{9b}2J\u{202e}txt.exe"
        );
        assert!(matches!(safe("gemma4 · modèle"), Cow::Borrowed(_)), "ordinary text, accents included, is untouched");
        // Through the progress printer too: a step title and a build line.
        let plan = one_download("Build \x1b[8mfork");
        let events = [WizardEvent {
            step: 0,
            status: StepStatus::Failed,
            stage: None,
            file: None,
            done: None,
            total: None,
            bps: None,
            line: Some("error: \x1b[31mred\x1b[0m".into()),
        }];
        let out = progress(&plan, false, &events);
        assert!(!out.contains('\x1b'), "{out}");
        assert!(out.contains(r"Build \u{1b}[8mfork") && out.contains(r"failed: error: \u{1b}[31mred"), "{out}");
    }

    #[test]
    fn gpu_targets_given_are_checked() {
        let cfg = Config::default_for_machine();
        assert_eq!(env(&cfg, Some("GFX1201, gfx1100".into())).unwrap().gfx.as_deref(), Some("gfx1201,gfx1100"));
        assert!(env(&cfg, Some("gfx1201 && calc".into())).is_err());
        assert_eq!(env(&cfg, None).unwrap().gfx, None);
    }

    #[test]
    fn small_formats() {
        assert_eq!(params(Some(37_444_792_020)), "37.4B");
        assert_eq!(params(Some(270_000_000)), "270M");
        assert_eq!(params(None), "-");
        assert_eq!(eta(59.4), "ETA 59s");
        assert_eq!(eta(125.0), "ETA 2m05s");
        assert_eq!(eta(3725.0), "ETA 1h02m");
        assert_eq!(short("223e6f683d88b82b"), "223e6f6");
    }
}
