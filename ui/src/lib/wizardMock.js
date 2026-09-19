// The model wizard in the browser preview: a few Hugging Face repos as the
// real commands describe them, and jobs that play the whole flow on timers.
// ngquocvinh/K2-Horizon-7B-GGUF needs the K2 Horizon fork built (the
// consent step), unsloth/gemma-4-26B-A4B-it-GGUF loads on an installed
// build, IFM/K2-Horizon-7B is safetensors only. Numbers follow core's
// estimate closely enough to look right; nothing here touches a disk.

import { newProgress, applyWizardEvent } from "./wizard.js";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;
const K2_SHA = "223e6f683d88b82b08a33f4ecad1237afa7ef486";
const MOVA_SHA = "8d0b6e3f1ac54c2e9d7a07cf7a6c1b2e3d4f5a61";
const IFM7_SHA = "bcb8c25b76112ce96a962f5b8ab624435d1ee0c9";
const GEMMA_SHA = "c099eb48e663fd284577b04978a94ffccb261841";
const IFM_SHA = "42adf019f76013dac873b5b43950d54d5ab27216";
const FORK_DIR = "C:\\llama.cpp\\ifm-ai-K2Horizon-fork-42adf019-src";
const FORK_SUBJECTS = [
  "model: K2 Horizon gguf conversion code",
  "model: loading hparams and tensors in k2-horizon.cpp",
  "model: K2 Horizon compute graph",
  "model: K2 Horizon compute graph adjustment and registering tokenizers",
  "model: K2 Horizon chat template and accomodate safetensors naming",
  "unicode : add the K2-Horizon pre-tokenizer splitter",
  "tests: expand K2 Horizon unicode splitter coverage",
  "Merge pull request #1 from a-contributor/test/k2-pr1-expanded-tests",
  "unicode: handle K2 Horizon case folding and empty input",
  "Merge pull request #1 from another-contributor/k2-horizon-msvc-pretokenizer",
];

const hex = (seed, n = 64) => {
  let x = seed * 2654435761 >>> 0, out = "";
  while (out.length < n) { x = (x * 1103515245 + 12345) >>> 0; out += x.toString(16).padStart(8, "0"); }
  return out.slice(0, n);
};
const file = (path, size, i = 0) => ({ path, size, sha256: hex(size % 9973 + i) });

// Model facts the estimate needs: KV bytes per token of context (f16),
// trained context, layers.
const REPOS = {
  "ngquocvinh/K2-Horizon-7B-GGUF": {
    sha: K2_SHA, arch: "k2-horizon", pre: "k2-horizon", ctx: 524288, layers: 36, kv: 147456, params: 8999178240,
    downloads: 3120, likes: 41, modified: "2026-09-17T21:40:11.000Z", license: "apache-2.0",
    base: [["quantized", "IFM/K2-Horizon-7B"]], library: "llama.cpp", pipeline: "text-generation",
    files: [["Q1_0", 1975860608], ["IQ2_XS", 3112164736], ["Q2_K", 3751427456], ["IQ3_M", 4407369088], ["Q3_K_M", 4634910080],
      ["IQ4_XS", 5113642368], ["Q4_K_M", 5592219008], ["Q5_K_M", 6466076032], ["Q6_K", 7394549120], ["Q8_0", 9573964896]]
      .map(([q, s], i) => ({ label: q, files: [file(`K2-Horizon-7B-${q}.gguf`, s, i)] })),
    ignored: [[file("reproducibility/k2_horizon_7b_combined.imatrix.gguf", 5347264), "importance matrix (quantizer input, not a model)"]],
    supported: false,
  },
  "NANI-Nithin/K2-Horizon-MoVA-36B-A4B-GGUF": {
    sha: MOVA_SHA, arch: "k2-horizon", pre: "k2-horizon", ctx: 524288, layers: 48, kv: 196608, params: 37444792020,
    downloads: 8412, likes: 96, modified: "2026-09-16T09:02:47.000Z", license: "apache-2.0",
    base: [["quantized", "IFM/K2-Horizon-MoVA-36B-A4B"]], library: null, pipeline: "text-generation",
    files: [["Q2_K", 13806174208], ["Q3_K_M", 17882124288], ["IQ4_XS", 20115372032], ["Q4_K_M", 22368011616], ["Q5_K_M", 26386247680],
      ["Q6_K", 30770438144], ["Q8_0", 39834918912], ["BF16", 74924627296]]
      .map(([q, s], i) => ({ label: q, files: [file(`K2-Horizon-MoVA-36B-A4B-${q}.gguf`, s, i)] })),
    ignored: [],
    supported: false,
  },
  "IFM/K2-Horizon-7B-GGUF": {
    sha: IFM7_SHA, arch: "k2-horizon", pre: "k2-horizon", ctx: 524288, layers: 36, kv: 147456, params: 8999178240,
    downloads: 1904, likes: 58, modified: "2026-09-03T12:11:05.000Z", license: "apache-2.0",
    base: [["quantized", "IFM/K2-Horizon-7B"]], library: null, pipeline: "text-generation",
    files: [{ label: "BF16", files: [file("K2-Horizon-7B-BF16.gguf", 18010413440)] }],
    ignored: [],
    supported: false,
  },
  "unsloth/gemma-4-26B-A4B-it-GGUF": {
    sha: GEMMA_SHA, arch: "gemma4", pre: null, ctx: 262144, layers: 30, kv: 20480, fixedKv: 0.62 * GIB, params: 25233775360,
    downloads: 412580, likes: 812, modified: "2026-09-11T16:20:31.000Z", license: "gemma", pipeline: "image-text-to-text",
    base: [["quantized", "google/gemma-4-26B-A4B-it"]], library: null,
    files: [
      ...[["UD-IQ2_XXS", 9922480608], ["UD-Q3_K_XL", 12907280096], ["UD-Q4_K_XL", 17010980576], ["MXFP4_MOE", 16551048928],
        ["UD-Q5_K_XL", 21217769184], ["UD-Q6_K_XL", 23295391456], ["Q8_0", 26859861728]]
        .map(([q, s], i) => ({ label: q, files: [file(`gemma-4-26B-A4B-it-${q}.gguf`, s, i)] })),
      { label: "BF16", files: [file("BF16/gemma-4-26B-A4B-it-BF16-00001-of-00002.gguf", 49923215552, 7), file("BF16/gemma-4-26B-A4B-it-BF16-00002-of-00002.gguf", 581922272, 8)] },
    ],
    mmproj: [file("mmproj-BF16.gguf", 1194828256), file("mmproj-F16.gguf", 1193058784), file("mmproj-F32.gguf", 2291200480)],
    drafts: [file("MTP/mtp-gemma-4-26B-A4B-it-BF16.gguf", 855228576), file("MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf", 461766816), file("mtp-gemma-4-26B-A4B-it.gguf", 461766816, 3)],
    ignored: [[file("imatrix_unsloth.gguf_file", 56941536), "importance matrix (quantizer input, not a model)"]],
    supported: true,
  },
  "ggml-org/gpt-oss-20b-GGUF": {
    sha: hex(9, 40), arch: "gpt-oss", pre: null, ctx: 131072, layers: 24, kv: 49152, params: 20914757184,
    downloads: 98110, likes: 160, modified: "2026-08-05T10:12:40.000Z", license: "apache-2.0", pipeline: "text-generation",
    base: [["quantized", "openai/gpt-oss-20b"]], library: null,
    files: [{ label: "MXFP4", files: [file("gpt-oss-20b-mxfp4.gguf", 12109566560)] }],
    // An EAGLE3 head: llama-server's draft-eagle3, which profiles cannot name yet.
    drafts: [file("eagle3-gpt-oss-20b-Q8_0.gguf", 937560064, 5)],
    ignored: [],
    supported: true,
  },
  "IFM/K2-Horizon-7B": { kind: "safetensors", sha: hex(7, 40), downloads: 2210, likes: 77, license: "apache-2.0", library: "transformers" },
  "IFM/K2-Horizon-7B-Uno": { kind: "adapter", sha: hex(8, 40), downloads: 120, likes: 9, license: "apache-2.0", library: "peft", base: [["adapter", "IFM/K2-Horizon-7B"]] },
};

/// A draft's speculative mode, as core's profile::draft_mode_of: null for
/// the heads profiles cannot run (EAGLE3, DSpark).
function draftMode(path) {
  const segs = String(path).split(/[\\/]/).filter(Boolean);
  const name = (segs.at(-1) ?? "").toLowerCase();
  if (name.startsWith("eagle") || name.includes("eagle3") || name.includes("dspark")) return null;
  if (name.includes("dflash")) return "dflash";
  if (segs.slice(0, -1).some((d) => d.toLowerCase() === "mtp") || name.includes("mtp")) return "mtp";
  return "draft";
}

const EXTRA_HITS = [
  ["IFM/K2-Horizon-32B-GGUF", "k2-horizon", 34779304960, 612, 31], ["IFM/K2-Horizon-0.9B-GGUF", "k2-horizon", 1078285824, 1480, 22],
  ["kingjones777/K2-Horizon-MoVA-36B-A4B-ROCmFP4-GGUF", "k2-horizon", 37444792020, 208, 12],
  ["ggml-org/gemma-4-26B-A4B-it-GGUF", "gemma4", 25233775360, 99120, 201], ["bartowski/google_gemma-4-26B-A4B-it-GGUF", "gemma4", 25233775360, 188012, 344],
];

export function createWizardMock(ctx) {
  // ctx: { devices(), builds(), models, profiles, emit(name, payload), sleep }
  const jobs = new Map();       // id -> { plan, progress, cancel, finished, consent, started_unix }
  // Views and plans handed out, by id: as in the app, the view names them
  // and the mock plans and runs its own copies.
  const madeViews = new Map();
  const madePlans = new Map();
  let nextMade = 1;
  const partial = new Map();    // dest -> bytes fetched earlier (a cancelled download resumes)
  const present = new Set();    // dests downloaded in this preview
  const roots = [
    { path: "D:\\models", exists: true, free_bytes: 412.6 * GIB },
    { path: "E:\\models", exists: true, free_bytes: 11.8 * GIB },
  ];
  let forkBuilt = false;
  let nextJob = 1;

  const cards = () => ctx.devices().filter((d) => !d.integrated);
  const busy = () => ctx.devicesRaw().filter((d) => d.occupied_by.length).map((d) => d.device.stable_key);
  const cap = () => (cards()[0]?.total_mib ?? 32624) * MIB;

  function verdict(need, capacity, free) {
    const ratio = need / capacity;
    return {
      fit: ratio > 1 ? "no_fit" : ratio > 0.9 ? "tight" : "fits",
      need_bytes: Math.round(need), capacity_bytes: capacity, fits_free_now: need <= free,
      detail: `AMD Radeon AI PRO R9700: ${(need / GIB).toFixed(2)} GiB of ${(capacity / GIB).toFixed(2)} GiB`,
    };
  }

  function fitOf(r, c, context = null, extras = 0) {
    const ctxUsed = context ?? Math.min(32768, r.ctx);
    const fixed = 1.25 * GIB + 0.4 * GIB + (r.fixedKv ?? 0);
    const kv = r.kv * ctxUsed;
    const capacity = cap();
    const idleFree = (cards().find((d) => !busy().includes(d.stable_key)) ?? cards()[0])?.free_mib * MIB ?? capacity;
    const one = verdict(c.total_size + kv + fixed + extras, capacity, idleFree);
    const two = cards().length >= 2 ? verdict(c.total_size / 2 + kv / 2 + fixed + extras, capacity, idleFree) : { fit: "not_applicable", need_bytes: 0, capacity_bytes: 0, fits_free_now: false, detail: "fewer than two GPUs" };
    const room = 0.9 * capacity - c.total_size - fixed - extras;
    const maxCtx = room > r.kv * 1024 ? Math.min(r.ctx, Math.floor(room / r.kv / 1024) * 1024) : null;
    return { label: c.label, quant: c.quant, total_size: c.total_size, ctx: ctxUsed, one_card: one, two_card_split: two, max_ctx_one_card: maxCtx, assumptions: ["compute buffer heuristic: 0.75 GiB + 0.5 GiB x (batch_physical / 512)"] };
  }

  function recommend(fits) {
    const best = (ok) => fits.filter(ok).sort((a, b) => b.total_size - a.total_size)[0]?.label ?? null;
    return best((f) => f.one_card.fit === "fits") ?? best((f) => f.two_card_split.fit === "fits") ?? best((f) => f.one_card.fit === "tight") ?? null;
  }

  function needsOf(r) {
    return { arch: r.arch, tokenizer_pre: r.pre, max_type_id: null, engine: "llama-server" };
  }

  function buildVerdicts(r) {
    const noArch = { support: "no", detail: { missing: [{ kind: "arch" }, ...(r.pre ? [{ kind: "tokenizer_pre", value: r.pre }] : [])] } };
    return ctx.builds().map((b) => {
      const git = b.channel === "git";
      const yes = r.supported ? b.channel === "upstream" || git : git && r.arch === "k2-horizon";
      return { path: b.path, name: b.git ? `${b.git.label} @${b.git.commit.slice(0, 7)}` : b.tag, channel: b.channel, version: b.version, broken: !!b.version_error, support: yes ? { support: "yes" } : noArch };
    }).sort((a, b) => (a.channel === "upstream" ? 0 : a.channel === "git" ? 1 : 2) - (b.channel === "upstream" ? 0 : b.channel === "git" ? 1 : 2));
  }

  function forkPlan(r) {
    const lacking = ctx.builds().filter((b) => b.channel !== "git").map((b) => b.tag).join(", ");
    return {
      needs: needsOf(r),
      step: {
        action: {
          action: "build_fork", owner: "ifm-ai", repo: "llama.cpp",
          source: { remote_url: "https://github.com/ifm-ai/llama.cpp", git_ref: "model/K2Horizon", sha: IFM_SHA, label: "ifm-ai K2Horizon fork" },
          gpu_targets: "gfx1201", install_dir: FORK_DIR,
        },
        explanation: `The model card links https://github.com/MBZUAI-IFM/llama.cpp/tree/model/K2Horizon; ifm-ai/llama.cpp model/K2Horizon at 42adf01, 10 commits ahead of upstream master and 380 behind knows architecture 'k2-horizon': build it from source (several minutes of compiling, about 0.6 GB of scratch space and 0.1 GB installed). Its newest commits: "${FORK_SUBJECTS[9]}", "${FORK_SUBJECTS[8]}", "${FORK_SUBJECTS[7]}".`,
        needs_consent: true, verified: true,
        warnings: ["github.com/MBZUAI-IFM/llama.cpp now redirects to ifm-ai/llama.cpp (the repository was renamed or transferred)"],
      },
      alternatives: [],
      rejected: [
        `installed builds lack architecture 'k2-horizon' and pre-tokenizer 'k2-horizon': ${lacking}`,
        "upstream b11046 (the newest release) lacks architecture 'k2-horizon' and pre-tokenizer 'k2-horizon'",
        "no open upstream pull request mentions 'k2-horizon'",
      ],
    };
  }
  const forkSources = () => [{
    owner: "ifm-ai", repo: "llama.cpp", linked_as: "MBZUAI-IFM/llama.cpp", git_ref: "model/K2Horizon", sha: IFM_SHA,
    url: "https://github.com/ifm-ai/llama.cpp/tree/model/K2Horizon", is_upstream: false, is_fork_of_upstream: true,
    ahead_by: 10, behind_by: 380, subjects: FORK_SUBJECTS, pr: null, support: { support: "yes" },
  }];

  function info(id, r) {
    const siblings = [...(r.files ?? []).flatMap((c) => c.files), ...(r.mmproj ?? []), ...(r.drafts ?? []), ...(r.ignored ?? []).map(([f]) => f)];
    return {
      id, sha: r.sha, gated: "no", gated_prompt: null, card_license: r.license, base_models: r.base ?? [],
      tags: [], pipeline_tag: r.pipeline ?? null, library_name: r.library ?? null, last_modified: r.modified ?? null,
      gguf: r.arch ? { architecture: r.arch, context_length: r.ctx, total: r.params } : null, siblings,
    };
  }

  function catalog(r) {
    const choices = (r.files ?? []).map((c) => ({
      label: c.label, quant: c.label, files: c.files, total_size: c.files.reduce((a, f) => a + f.size, 0), first_file: c.files[0].path,
    })).sort((a, b) => a.total_size - b.total_size);
    return { choices, mmproj: r.mmproj ?? [], drafts: r.drafts ?? [], ignored: r.ignored ?? [] };
  }

  function view(input, id, r) {
    const base = {
      input, repo: id, sha: r.sha, rev: null, info: info(id, r), catalog: catalog(r), notes: [], derivatives: [], derivatives_of: null,
      header: null, header_of: null, header_error: null, needs: null, preselect: null, devices: ctx.devices(), fits: [], recommended: null,
      draft_modes: Object.fromEntries((r.drafts ?? []).map((f) => [f.path, draftMode(f.path)])),
      builds: [], usable_build: null, build_plan: null, build_sources: [], model_roots: roots.map((x) => ({ ...x })),
    };
    if (r.kind === "safetensors" || r.kind === "adapter") {
      const of = r.kind === "adapter" ? r.base[0][1] : id;
      base.kind = { kind: r.kind };
      base.info.siblings = r.kind === "adapter" ? [file("adapter_config.json", 820), file("adapter_model.safetensors", 167772160)] : [file("model-00001-of-00004.safetensors", 4.98e9), file("model-00002-of-00004.safetensors", 4.95e9), file("config.json", 1411)];
      base.notes = [{ level: "error", code: "not-gguf", message: r.kind === "adapter"
        ? `${id} is a LoRA / PEFT adapter, not a model: llama.cpp cannot run it on its own, and profiles do not apply adapters. GGUF quantizations of its base model ${of} are listed instead.`
        : `${id} holds weights for transformers (safetensors), not GGUF. llama.cpp loads only GGUF; converting takes Python and the matching build's convert script. GGUF quantizations of it published on the Hub are listed instead.` }];
      base.derivatives = searchHits("K2-Horizon-7B").map((h) => h).filter((h) => /7B/.test(h.id));
      base.derivatives_of = of;
      return base;
    }
    base.kind = { kind: "gguf" };
    const cat = base.catalog;
    base.header_of = cat.choices[0]?.label ?? null;
    base.header = { path: `hf://${id}@${r.sha.slice(0, 12)}/${cat.choices[0]?.first_file}`, file_size: cat.choices[0]?.total_size ?? 0, architecture: r.arch, block_count: r.layers, context_length: r.ctx, tokenizer_model: r.pre ? "gpt2" : "llama", tokenizer_pre: r.pre, partial: true, metadata: {} };
    base.needs = needsOf(r);
    base.fits = cat.choices.map((c) => fitOf(r, c));
    base.recommended = recommend(base.fits);
    base.builds = buildVerdicts(r);
    base.usable_build = base.builds.find((b) => b.support.support === "yes" && !b.broken)?.path ?? null;
    if (!base.usable_build) {
      base.build_plan = forkPlan(r);
      base.build_sources = forkSources();
    }
    if (cat.ignored.length === 0 && id.startsWith("NANI")) {
      base.notes.push({ level: "info", code: "card-of-base", message: "llama.cpp links taken from the base model's card (IFM/K2-Horizon-MoVA-36B-A4B)" });
    }
    return base;
  }

  function searchHits(q) {
    const words = String(q ?? "").toLowerCase().split(/\s+/).filter(Boolean);
    const hits = [
      ...Object.entries(REPOS).filter(([, r]) => r.arch).map(([id, r]) => [id, r.arch, r.params, r.downloads, r.likes, r.modified]),
      ...EXTRA_HITS.map(([id, arch, params, downloads, likes]) => [id, arch, params, downloads, likes, "2026-09-10T10:00:00.000Z"]),
    ];
    return hits
      .filter(([id]) => words.every((w) => id.toLowerCase().includes(w)))
      .sort((a, b) => b[3] - a[3])
      .map(([id, arch, params, downloads, likes, modified]) => ({
        id, author: id.split("/")[0], downloads, likes, last_modified: modified, gated: id.startsWith("google/") ? "manual" : "no",
        arch, context_length: arch === "gemma4" ? 262144 : 524288, total_params: params, tags: ["gguf"], pipeline_tag: "text-generation", library_name: null,
        arch_known: arch === "k2-horizon" ? forkBuilt : true,
      }));
  }

  function inspect(input) {
    const s = String(input ?? "").trim().replace(/^https?:\/\/(www\.)?(huggingface\.co|hf\.co)\//, "");
    const parts = s.split(/[/?#]/).filter(Boolean);
    if (parts.length < 2 || /\s/.test(s)) throw `${JSON.stringify(input)} is not a Hugging Face model: paste owner/name or a huggingface.co link`;
    const id = `${parts[0]}/${parts[1]}`;
    const key = Object.keys(REPOS).find((k) => k.toLowerCase() === id.toLowerCase());
    if (!key) throw `${id}: not found on Hugging Face (or private: set a token if it is yours)`;
    const v = view(input, key, REPOS[key]);
    // A pasted file link names the choice.
    if (parts[2] === "blob" || parts[2] === "resolve") {
      const path = parts.slice(4).join("/");
      const c = v.catalog.choices.find((x) => x.files.some((f) => f.path === path));
      if (c) { v.preselect = c.label; v.recommended = c.label; }
    }
    return v;
  }

  // ---- plan ----------------------------------------------------------------
  const slug = (s) => (String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48) || "model");

  function plan(v, req) {
    if (v.kind?.kind !== "gguf") throw `${v.repo} has no GGUF file to download`;
    const r = REPOS[v.repo];
    const label = req.choice ?? v.recommended ?? v.catalog.choices[0]?.label;
    const choice = v.catalog.choices.find((c) => c.label === label);
    if (!choice) throw `${v.repo} has no file ${JSON.stringify(label)}`;
    const mmproj = req.mmproj ? v.catalog.mmproj.find((f) => f.path === req.mmproj) : null;
    const draft = req.draft ? v.catalog.drafts.find((f) => f.path === req.draft) : null;
    if (draft && !draftMode(draft.path)) throw `${draft.path} is an EAGLE3 head: llama-server runs it with a speculative type profiles do not have yet, and as a plain draft model it would not load; pick another draft, or none`;
    const rootPath = req.dest_root ?? roots[0].path;
    const root = roots.find((x) => x.path.toLowerCase() === String(rootPath).toLowerCase()) ?? { path: rootPath, free_bytes: null };
    const [owner, name] = v.repo.split("/");
    const destDir = `${root.path}\\${owner}\\${name}`;
    const dest = (f) => `${destDir}\\${f.path.split("/").pop()}`;
    const notes = [];
    const steps = [];
    const extras = (mmproj?.size ?? 0) + (draft?.size ?? 0);
    const fit = fitOf(r, choice, req.ctx ?? null, extras);
    const fitsWord = (f) => f === "fits" || f === "tight";
    const oneCard = fitsWord(fit.one_card.fit);
    const split = !oneCard && fitsWord(fit.two_card_split.fit);
    if (!oneCard && !split) notes.push({ level: "warning", code: "does-not-fit", message: `${choice.label} is estimated not to fit at context ${fit.ctx}: ${fit.one_card.detail}. Pick a smaller file or a shorter context.` });

    // The build.
    const kind = req.build?.kind ?? "auto";
    let build = null, buildStep = null, bp = null, toolchain = [], consent = null, sources = [];
    const verdicts = buildVerdicts(r);
    const usable = verdicts.find((b) => b.support.support === "yes" && !b.broken);
    if (kind === "skip") {
      const b = usable ?? verdicts.find((x) => !x.broken);
      if (b && b.support.support !== "yes") notes.push({ level: "warning", code: "build-skipped", message: `no installed build is known to load it; the profile goes on ${b.name} and pre-flight will say what is missing` });
      if (b) build = { path: b.path, name: b.name, installed: true, support: b.support };
    } else if (kind === "installed") {
      const b = verdicts.find((x) => x.path === req.build.value);
      if (!b) throw `no installed build at ${req.build.value}`;
      if (b.support.support === "no") notes.push({ level: "error", code: "build-cannot-load", message: `${b.name} cannot load this model: it lacks architecture '${r.arch}'${r.pre ? ` and pre-tokenizer '${r.pre}'` : ""}` });
      build = { path: b.path, name: b.name, installed: true, support: b.support };
    } else if (kind === "auto" && usable) {
      build = { path: usable.path, name: usable.name, installed: true, support: usable.support };
    } else {
      bp = forkPlan(r);
      sources = forkSources();
      const idx = kind === "plan" ? req.build.value : 0;
      const ps = idx === 0 ? bp.step : bp.alternatives[idx - 1];
      if (!ps) throw `the build plan has no alternative ${idx}`;
      for (const w of ps.warnings) notes.push({ level: "warning", code: "build-warning", message: w });
      buildStep = ps;
      consent = `Building ifm-ai K2Horizon fork @42adf01 runs code from ifm-ai/llama.cpp (commit 42adf01) that nobody reviewed for your machine.`;
      build = { path: ps.action.install_dir, name: "ifm-ai K2Horizon fork @42adf01", installed: forkBuilt, support: { support: "yes" } };
      toolchain = [
        { id: "vs", title: "Visual Studio C++ tools", outcome: "pass", message: "Visual Studio Build Tools 2026 (18.0), MSVC 14.50", fix: null },
        { id: "git", title: "git", outcome: "pass", message: "git version 2.53.0.windows.1", fix: null },
        { id: "cmake", title: "CMake and Ninja", outcome: "pass", message: "cmake 4.1.2 and ninja 1.12.1 (Visual Studio's)", fix: null },
        { id: "hip", title: "HIP SDK clang", outcome: "pass", message: "C:\\Program Files\\AMD\\ROCm\\7.1\\bin\\clang++.exe", fix: null },
        { id: "cmath", title: "HIP test compile for gfx1201", outcome: "pass", message: "compiled a .hip file including <cmath> in 1.9 s", fix: null },
      ];
      steps.push({
        title: `Build ifm-ai K2Horizon fork @42adf01 for gfx1201`, detail: ps.explanation, needs_consent: true,
        action: { kind: "build", plan: ps },
      });
    }

    // Downloads.
    const files = [...choice.files.map((f) => ["model", f]), ...(mmproj ? [["mmproj", mmproj]] : []), ...(draft ? [["draft", draft]] : [])];
    let fetch = 0, total = 0;
    for (const [role, f] of files) {
      const d = dest(f);
      const isThere = present.has(d);
      const have = isThere ? f.size : Math.min(partial.get(d) ?? 0, f.size);
      fetch += f.size - have; total += f.size;
      const what = role === "model" ? (choice.files.length > 1 ? "part of the model" : "the model") : role === "mmproj" ? "the vision projector" : "the draft model";
      steps.push({
        title: `Download ${f.path.split("/").pop()} (${what})`,
        detail: isThere ? `${d} is already there; its size and SHA-256 are checked instead` : have ? `resumes at ${(have / GIB).toFixed(1)} GiB of ${(f.size / GIB).toFixed(1)} GiB into ${d}` : `${(f.size / GIB).toFixed(1)} GiB into ${d}`,
        needs_consent: false, action: { kind: "download", role, file: f, dest: d, have, present: isThere },
      });
    }
    if (root.free_bytes != null && root.free_bytes < fetch * 1.05) {
      notes.push({ level: "error", code: "dest-space", message: `the download needs ${(fetch / GIB).toFixed(1)} GiB (plus 5%) and the drive holding ${root.path} has ${(root.free_bytes / GIB).toFixed(1)} GiB free` });
    }
    if (!roots.some((x) => x.path.toLowerCase() === String(root.path).toLowerCase())) {
      notes.push({ level: "warning", code: "dest-not-scanned", message: `${root.path} is not one of your model folders: the Profiles picker will not list the model (the new profile still points at it). Add the folder in Settings to have it scanned.` });
    }

    // The profile.
    let profile = null;
    const stem = choice.files.length > 1 ? choice.first_file.split("/").pop().replace(/-\d{5}-of-\d{5}\.gguf$/i, "") : choice.first_file.split("/").pop().replace(/\.gguf$/i, "");
    if (req.profile !== false && build) {
      const taken = new Set(ctx.profiles().map((p) => p.id.toLowerCase()));
      let id = slug(stem), n = 2;
      while (taken.has(id)) id = `${slug(stem)}-${n++}`;
      const ports = new Set(ctx.profiles().map((p) => p.server.port));
      let port = 9710;
      while (ports.has(port)) port++;
      const idle = cards().filter((d) => !busy().includes(d.stable_key));
      const devs = split ? cards().slice(0, 2) : [idle[0] ?? cards()[0]].filter(Boolean);
      const ctxWant = req.ctx ?? Math.min(32768, fit.max_ctx_one_card ?? 32768);
      profile = {
        schema: 1, id, name: stem,
        build: { path: build.path, version: build.installed ? (ctx.builds().find((b) => b.path === build.path)?.version ?? null) : null },
        model: { path: dest(choice.files[0]), mmproj: mmproj ? dest(mmproj) : null, draft: draft ? { path: dest(draft), enabled: true } : null },
        devices: devs.map((d) => ({ key: d.stable_key, split_fraction: null, resolved_index_last_launch: null })),
        split_mode: devs.length > 1 ? "layer" : null, main_device: 0,
        server: { port, alias: id, host: "127.0.0.1" },
        runtime: { n_gpu_layers: 99, ctx_total: Math.floor(Math.min(ctxWant, r.ctx) / 256) * 256, slots: 1, kv_type_k: "f16", kv_type_v: "f16", flash_attn: "on", batch_logical: 2048, batch_physical: 512, cont_batching: true, kv_unified: false, cache_reuse: null },
        sampling: {}, speculative: draft ? { mode: draftMode(draft.path) } : undefined, chat: {}, env: {},
        notes: `From ${v.repo} at ${v.sha.slice(0, 7)} (${choice.label}), by the model wizard.`,
      };
      steps.push({
        title: `Create profile ${id}`,
        detail: `on ${build.name} with ${devs.length > 1 ? "a layer split over two cards" : "one card"}, port ${port}, context ${profile.runtime.ctx_total}; nothing is launched`,
        needs_consent: false, action: { kind: "profile", id, path: `C:\\Users\\me\\.fidim\\profiles\\${id}.json` },
      });
    }
    return {
      repo: v.repo, sha: v.sha, choice, mmproj, draft, dest_root: root.path, dest_dir: destDir, steps, notes,
      blocked: notes.some((n) => n.level === "error"), needs_consent: steps.some((s) => s.needs_consent), consent,
      download_bytes: fetch, total_bytes: total, engine: "llama-server", gated: "no", needs: needsOf(r),
      header: v.header, fit, build, build_plan: bp, build_sources: sources, build_choice: req.build ?? { kind: "auto" }, toolchain,
      profile, profile_opts: {}, devices: v.devices,
    };
  }

  // ---- jobs ------------------------------------------------------------------
  function snapshot(id) {
    const j = jobs.get(id);
    return { job: id, plan: j.plan, progress: j.progress, started_unix: j.started_unix, consent: j.consent, cancelling: j.cancel && !j.finished, finished: j.finished };
  }

  function emit(id, ev) {
    const j = jobs.get(id);
    applyWizardEvent(j.progress, ev);
    ctx.emit("wizard-progress", { ...ev, job: id });
  }

  async function runJob(id) {
    const j = jobs.get(id);
    const p = j.plan;
    const tick = async (ms) => {
      for (let left = ms; left > 0; left -= 50) {
        if (j.cancel) throw "cancelled";
        await ctx.sleep(Math.min(50, left));
      }
    };
    let buildReport = null;
    let current = 0;
    try {
      for (let i = 0; i < p.steps.length; i++) {
        current = i;
        const s = p.steps[i];
        emit(id, { step: i, status: "running" });
        if (s.action.kind === "build") {
          if (forkBuilt) {
            emit(id, { step: i, status: "running", stage: "verify", line: `ifm-ai K2Horizon fork @42adf01 already built at ${FORK_DIR} — verifying only` });
            await tick(600);
          } else {
            const say = async (stage, line, ms) => { emit(id, { step: i, status: "running", stage, line }); await tick(ms); };
            await say("doctor", "checking the toolchain for gfx1201", 300);
            for (const t of p.toolchain) await say("doctor", `ok    ${t.title}: ${t.message}`, 80);
            await say("fetch", `checking that model/K2Horizon still points at ${IFM_SHA.slice(0, 12)}`, 500);
            await say("clone", "building ifm-ai K2Horizon fork @42adf01 for gfx1201", 300);
            await say("fetch", "From https://github.com/ifm-ai/llama.cpp\n * branch  refs/heads/model/K2Horizon -> FETCH_HEAD", 700);
            await say("worktree", `Preparing worktree (detached HEAD ${IFM_SHA.slice(0, 7)})`, 300);
            await say("configure", "-- The HIP compiler identification is Clang 20.0.0", 400);
            await say("configure", "-- Configuring done (6.2s)", 400);
            const total = 478;
            for (let n = 1; n <= total; n += 7) {
              const f = ["ggml-hip/fattn-tile-f16.cu", "ggml-hip/mmq-instance-q4_k.cu", "src/models/k2-horizon.cpp", "src/llama-vocab.cpp", "ggml-hip/mmvq.cu"][n % 5];
              emit(id, { step: i, status: "running", stage: "build", done: n, total, line: `[${n}/${total}] Building HIP object ggml/src/${f}.obj` });
              await tick(40);
            }
            emit(id, { step: i, status: "running", stage: "build", done: total, total, line: `[${total}/${total}] Linking CXX executable bin\\llama-server.exe` });
            await say("copy", "copying bin out of the build tree", 300);
            await say("install", `moving the build into ${FORK_DIR}`, 300);
            await say("verify", "verifying: --version and --list-devices (no model load)", 700);
            forkBuilt = true;
            ctx.addBuild({
              path: FORK_DIR, tag: "ifm-ai-K2Horizon-fork-42adf019-src", server_exe: `${FORK_DIR}\\bin\\llama-server.exe`,
              version: "b10673", commit: IFM_SHA.slice(0, 9), version_error: null, channel: "git", bundled_runtime: false, release_tag: null, runner_exe: null,
              git: { remote: "https://github.com/ifm-ai/llama.cpp", git_ref: "refs/heads/model/K2Horizon", commit: IFM_SHA, label: "ifm-ai K2Horizon fork" },
            });
          }
          buildReport = { tag: "ifm-ai K2Horizon fork @42adf01", dir: FORK_DIR, source: "git-ref", skipped_existing: false, verify: { version: "b10673", commit: IFM_SHA.slice(0, 9), hip_ok: true, detail: "", runner_present: false, devices: [] } };
        } else if (s.action.kind === "download") {
          const { file: f, dest } = s.action;
          const name = f.path.split("/").pop();
          const secs = Math.min(12, Math.max(3, (f.size / GIB) * 0.9));
          const bps = f.size / secs;
          let done = present.has(dest) ? f.size : (partial.get(dest) ?? 0);
          if (done > 0 && done < f.size) {
            emit(id, { step: i, status: "running", stage: "hashing", file: name, done: 0, total: done, bps: 0 });
            await tick(500);
          }
          while (done < f.size) {
            done = Math.min(f.size, done + bps * 0.25);
            partial.set(dest, done);
            emit(id, { step: i, status: "running", stage: present.has(dest) ? "hashing" : "downloading", file: name, done: Math.round(done), total: f.size, bps: bps * (0.92 + 0.16 * Math.random()) });
            await tick(250);
          }
          present.add(dest);
          partial.delete(dest);
        } else if (s.action.kind === "profile") {
          await tick(300);
          const prof = structuredClone(p.profile);
          prof.build.version ??= buildReport?.verify.version ?? null;
          ctx.addProfile(prof, p);
        }
        emit(id, { step: i, status: "done" });
      }
      const result = {
        repo: p.repo, sha: p.sha, model_path: p.steps.find((s) => s.action.kind === "download")?.action.dest ?? null,
        files: p.steps.filter((s) => s.action.kind === "download").map((s) => s.action.dest),
        build: buildReport, build_path: p.build?.path ?? null, profile: p.profile ? ctx.profiles().find((x) => x.id === p.profile.id) ?? null : null,
        profile_path: p.profile ? `C:\\Users\\me\\.fidim\\profiles\\${p.profile.id}.json` : null, warnings: [],
      };
      j.finished = { job: id, ok: true, result };
    } catch (e) {
      const error = String(e);
      emit(id, { step: current, status: "failed", line: error });
      j.finished = { job: id, ok: false, error };
    }
    ctx.emit("wizard-done", j.finished);
  }

  async function handle(cmd, args) {
    switch (cmd) {
      case "hub_search":
        return searchHits(args.query).slice(0, args.limit ?? 30);
      case "wizard_inspect": {
        await ctx.sleep(700);
        const v = inspect(args.input);
        v.view_id = `v${nextMade++}`;
        madeViews.set(v.view_id, structuredClone(v));
        return v;
      }
      case "wizard_plan": {
        await ctx.sleep(400);
        const v = madeViews.get(args.viewId);
        if (!v) throw "this repo's page is out of date: open the repo again";
        const p = plan(v, args.request ?? {});
        p.plan_id = `p${nextMade++}`;
        madePlans.set(p.plan_id, structuredClone(p));
        return p;
      }
      case "wizard_start": {
        const p = madePlans.get(args.planId) ?? [...jobs.values()].find((j) => j.plan.plan_id === args.planId)?.plan;
        if (!p) throw "this plan is out of date: plan it again";
        if (p.blocked) throw `the plan cannot run: ${p.notes.filter((n) => n.level === "error").map((n) => n.message).join("; ")}`;
        // As core: a fork or pull request build needs consent whatever the flags say.
        const needsConsent = (s) => s.needs_consent || (s.action.kind === "build" && (s.action.plan.needs_consent || ["build_fork", "build_pr"].includes(s.action.plan.action.action)));
        if (p.steps.some(needsConsent) && !args.consent) throw `${p.consent} Nothing was done: confirm it first (the consent box in the app, --allow-fork in the CLI).`;
        const mine = p.steps.filter((s) => s.action.kind === "download").map((s) => s.action.dest);
        for (const [other, j] of jobs) {
          if (j.finished) continue;
          const d = j.plan.steps.find((s) => s.action.kind === "download" && mine.includes(s.action.dest));
          if (d) throw `job ${other} is already downloading ${d.action.dest}`;
        }
        const id = `w${Math.floor(Date.now() / 1000)}-${nextJob++}`;
        jobs.set(id, { plan: structuredClone(p), progress: newProgress(p), cancel: false, finished: null, consent: !!args.consent, started_unix: Math.floor(Date.now() / 1000) });
        runJob(id);
        return id;
      }
      case "wizard_cancel": {
        const j = jobs.get(args.job);
        if (!j || j.finished) return false;
        j.cancel = true;
        return true;
      }
      case "wizard_jobs":
        return [...jobs.keys()].map(snapshot);
      case "wizard_forget": {
        const j = jobs.get(args.job);
        if (!j?.finished) return false;
        jobs.delete(args.job);
        return true;
      }
      case "wizard_roots":
        return roots.map((x) => ({ ...x }));
      case "wizard_add_root": {
        const path = String(args.path ?? "").trim().replace(/[\\/]+$/, "");
        if (!/^[A-Za-z]:\\/.test(path)) throw `${args.path} is not an absolute folder path`;
        if (!roots.some((x) => x.path.toLowerCase() === path.toLowerCase())) roots.push({ path, exists: true, free_bytes: 96.2 * GIB });
        return roots.map((x) => ({ ...x }));
      }
      default:
        return undefined;
    }
  }

  return { handle };
}
