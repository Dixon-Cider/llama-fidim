<script>
  import { api, log } from "../api.js";
  import Range from "../components/Range.svelte";
  import { fly, fade, slide, scale } from "svelte/transition";
  import { flip } from "svelte/animate";
  import { arrive, leave, flipParams, toastFly, stagger, LAYOUT } from "../motion.js";
  import { takeProfileId, takeModelPath } from "../lib/handoff.js";

  // App passes `go(viewId)` so a missing build can link straight to Updates.
  let { go = () => {} } = $props();

  let profiles = $state([]);
  let devices = $state([]);
  let occupancy = $state({}); // stable_key -> ids of live runs on that card
  let runtimes = $state([]);
  let builds = $state([]);
  let models = $state([]);
  let allDrafts = $state([]);   // every draft/MTP file under the roots
  let allMmproj = $state([]);   // every projector under the roots
  let selectedId = $state(null);
  let draft = $state(null); // deep-copied profile being edited
  let check = $state(null); // live_check result
  let checking = $state(false);
  let busy = $state("");
  let toast = $state(null);
  let creator = $state(null); // creator_defaults result or { error }
  let creatorBusy = $state(false);
  let checkTimer = null;
  // Rows whose pre-flight outcome changed on the last check pulse once.
  let changed = $state(new Set());
  let prevKinds = new Map();
  let changedTimer = null;
  let savedFlash = $state(false);
  let savedTimer = null;
  function flashSaved() {
    savedFlash = true;
    if (savedTimer) clearTimeout(savedTimer);
    savedTimer = setTimeout(() => (savedFlash = false), 2200);
  }

  const GIB = 1024 * 1024 * 1024;
  const fmtInt = (v) => Number(v).toLocaleString();
  const fmtGib = (b) => (b / GIB).toFixed(1) + " GiB";

  let loadErrors = $state([]); // [name, message] for anything that failed to load
  let loading = $state(true);
  let keepAliveDefault = $state(5);
  api("get_config").then((c) => (keepAliveDefault = c.config?.keep_alive_seconds ?? 5)).catch(() => {});

  async function load() {
    loading = true;
    const errs = [];
    const grab = async (name, fn, fallback) => {
      try { return await fn(); } catch (e) { errs.push([name, String(e)]); return fallback; }
    };
    // Independent calls, each with its own failure — one broken scan must
    // not blank the whole editor.
    const [p, d, r, s] = await Promise.all([
      grab("profiles", () => api("list_profiles"), []),
      grab("devices", () => api("devices", { refresh: false }), []),
      grab("runtimes", () => api("list_runtimes"), []),
      grab("scan", () => api("scan"), { builds: [], models: [] }),
    ]);
    profiles = p;
    devices = d.map((r) => r.device);
    occupancy = Object.fromEntries(d.map((r) => [r.device?.stable_key, r.occupied_by ?? []]));
    runtimes = r;
    builds = s.builds ?? [];
    models = (s.models ?? []).slice().sort((a, b) => a.path.localeCompare(b.path));
    allDrafts = s.drafts ?? [];
    allMmproj = s.mmproj ?? [];
    loadErrors = errs;
    loading = false;
    // A profile the Models view just made opens here; a checkpoint path it
    // handed over starts a new SGLang profile on it.
    const handed = takeProfileId();
    const handedPath = takeModelPath();
    if (handed && profiles.some((r) => r.profile.id === handed)) select(handed);
    else if (handedPath) newSgLangProfile(handedPath);
    else if (!selectedId && profiles.length) select(profiles[0].profile.id);
  }
  load();

  async function rescan() {
    busy = "scan";
    try {
      const s = await api("scan", { refresh: true });
      builds = s.builds ?? [];
      models = (s.models ?? []).slice().sort((a, b) => a.path.localeCompare(b.path));
      allDrafts = s.drafts ?? [];
      allMmproj = s.mmproj ?? [];
      toastMsg(`Found ${builds.length} builds, ${models.length} models`);
    } catch (e) { toastMsg(String(e), true); }
    busy = "";
  }

  // ---- model-derived limits ------------------------------------------------
  // The scan lists GGUF files and Hugging Face checkpoint folders
  // (format "safetensors", hf facts instead of a header); only SGLang loads
  // the latter, so the llama-server picker never shows them.
  const isHf = (m) => m?.format === "safetensors";
  const ggufModels = $derived(models.filter((m) => !isHf(m)));
  const hfModels = $derived(models.filter(isHf));
  const selectedModel = $derived(models.find((m) => samePath(m.path, draft?.model?.path)));
  const header = $derived(selectedModel?.header ?? null);
  const hf = $derived(isHf(selectedModel) ? selectedModel.hf ?? {} : null);
  const ctxMax = $derived(header?.context_length ?? hf?.max_position_embeddings ?? 262144);
  const layerMax = $derived(header?.block_count ?? 99);
  const selectedBuild = $derived(builds.find((b) => samePath(b.path, draft?.build?.path)));

  // ---- engine ----------------------------------------------------------------
  // A profile carries `engine` only when it is not llama-server; the model's
  // header decides which one it needs (discovery sets Model.engine).
  const isDiffusion = $derived(draft?.engine === "diffusion-gemma");
  const isSgLang = $derived(draft?.engine === "sglang");
  const modelEngine = $derived(selectedModel?.engine ?? "llama-server");
  // A GGUF header never names SGLang (it serves GGUF and safetensors alike),
  // so only a diffusion header contradicts an SGLang profile.
  const engineMismatch = $derived(!!selectedModel && modelEngine !== (draft?.engine ?? "llama-server") &&
    !(draft?.engine === "sglang" && modelEngine === "llama-server"));

  // ---- sglang ----------------------------------------------------------------
  // Mirrors SgLangCfg (crates/fidim-core/src/sglang.rs): serde fills every
  // absent field with these, so a profile saved with them is byte-stable.
  const SG_DEFAULTS = {
    mem_fraction: 0.85, chunked_prefill: 8192, max_running_requests: 2, mamba_slots: null,
    kv_cache_dtype: "fp8_e4m3", attention_backend: "triton", dtype: "bfloat16", spec: null,
    reasoning_parser: "", tool_call_parser: "", memguard_gb: 12, enable_metrics: true,
    sleep_on_idle: true, compile_threads: 2, aliases: [], extra_args: [],
  };
  const SG_SPEC_DEFAULTS = { algorithm: "NEXTN", steps: 3, topk: 1, draft_tokens: 4, token_map: "" };
  const SG_KV_DTYPES = ["fp8_e4m3", "fp4_mx_block16", "fp8_e5m2", "bf16", "auto"];
  const SG_ATTN_BACKENDS = ["triton", "flashinfer", "fa3", "trtllm_mha", "torch_native"];
  const SG_DTYPES = ["bfloat16", "float16", "auto"];
  const SG_SPEC_ALGOS = ["NEXTN", "EAGLE", "EAGLE3", "NGRAM"];
  const SG_REASONING_PARSERS = ["qwen3-thinking", "qwen3", "deepseek-r1", "deepseek-v3", "glm45", "kimi", "gpt-oss", "step3"];
  const SG_TOOL_PARSERS = ["qwen3_coder", "qwen25", "hermes", "llama3", "mistral", "deepseekv3", "glm45", "kimi_k2", "pythonic", "gpt-oss"];
  // The one card an SGLang profile runs on, from the device list.
  const sgDevice = $derived(isSgLang ? devices.find((d) => d.stable_key === draft?.devices?.[0]?.key) ?? null : null);
  // Text of the env map while it is being edited (parsing every keystroke
  // back into the map would drop the line being typed).
  let sgEnvText = $state("");
  // Parser selects: "other…" shows a free-text input.
  let sgCustomParser = $state({ reasoning_parser: false, tool_call_parser: false });

  function sgFill(s) {
    s ??= {};
    for (const [k, v] of Object.entries(SG_DEFAULTS)) s[k] ??= Array.isArray(v) ? [] : v;
    if (s.spec) for (const [k, v] of Object.entries(SG_SPEC_DEFAULTS)) s.spec[k] ??= v;
    return s;
  }
  function envToText(env) {
    return Object.entries(env ?? {}).map(([k, v]) => `${k}=${v}`).join("\n");
  }
  function parseEnvText(text) {
    const out = {};
    for (const line of String(text ?? "").split(/\r?\n/)) {
      const s = line.trim();
      if (!s || s.startsWith("#")) continue;
      const i = s.indexOf("=");
      if (i <= 0) continue;
      out[s.slice(0, i).trim()] = s.slice(i + 1).trim();
    }
    return out;
  }
  function onEnvText(text) {
    sgEnvText = text;
    draft.env = parseEnvText(text);
    scheduleCheck();
  }
  const listOf = (v, sep) => (Array.isArray(v) ? v : String(v ?? "").split(sep)).map((x) => String(x).trim()).filter(Boolean);
  const intOr = (v, d) => (v === "" || v === null || v === undefined || !Number.isFinite(Number(v)) ? d : Math.max(0, Math.round(Number(v))));
  function parserSel(k, list) {
    const v = draft?.sglang?.[k] ?? "";
    return sgCustomParser[k] || (v && !list.includes(v)) ? "__custom" : v;
  }
  function onParserPick(k, v, list) {
    if (v === "__custom") {
      sgCustomParser[k] = true;
      if (list.includes(draft.sglang[k])) draft.sglang[k] = "";
    } else {
      sgCustomParser[k] = false;
      draft.sglang[k] = v;
    }
    scheduleCheck();
  }
  function onSgSpecToggle(on) {
    draft.sglang.spec = on ? { ...SG_SPEC_DEFAULTS } : null;
    scheduleCheck();
  }

  function enterSgLang() {
    draft.engine = "sglang";
    delete draft.diffusion;
    // No llama.cpp build: the venv from Settings launches it.
    draft.build = { path: "", version: null };
    draft.rocm_runtime = null;
    draft.sglang = sgFill(draft.sglang);
    // One card: keep the profile's first discrete one, else an empty one, else the first.
    const discrete = devices.filter((d) => !d.integrated);
    const keep = draft.devices.find((x) => discrete.some((d) => d.stable_key === x.key));
    const pick = discrete.find((d) => !occupancy[d.stable_key]?.length) ?? discrete[0];
    draft.devices = keep ? [keep] : pick ? [{ key: pick.stable_key, split_fraction: null, resolved_index_last_launch: null }] : [];
    draft.split_mode = null;
    draft.main_device = 0;
    // Kept as an "off" object so nothing hidden reads null; normalized() drops it.
    draft.speculative = { mode: "off", n_max: null, n_min: null, p_min: null };
    draft.model.mmproj = null;
    draft.model.draft = null;
    draft.keep_alive_seconds = null;
    draft.runtime.extra_flags = [];
    if (!draft.runtime.ctx_total) draft.runtime.ctx_total = 32768;
    sgEnvText = envToText(draft.env);
  }

  function leaveSgLang(h) {
    delete draft.engine;
    delete draft.sglang;
    if (!draft.runtime.ctx_total) draft.runtime.ctx_total = Math.min(32768, h?.context_length ?? 262144);
    if (!draft.build?.path) {
      const b = newestUpstreamBuild();
      if (b) onBuildPick(b.path);
      else draft.build = { path: "", version: null };
    }
  }

  // The engine select (shown unless a diffusion header decides it).
  function onEnginePick(v) {
    const cur = draft.engine ?? "llama-server";
    if (v === cur) return;
    if (cur === "diffusion-gemma") leaveDiffusion(header);
    if (cur === "sglang") leaveSgLang(header);
    if (v === "sglang") enterSgLang();
    else if (v === "diffusion-gemma") enterDiffusion();
    creator = null;
    scheduleCheck();
  }
  const DG_DEFAULTS = { hipblaslt_safeguard: true, default_max_tokens: 2048 };
  const canvas = $derived(header?.diffusion_canvas_length ?? 256);
  const dgSizing = $derived(check?.diffusion?.sizing ?? null);
  const runnerBuilds = $derived(builds.filter((b) => b.runner_exe));
  // A locally patched runner build declares what it changes in its manifest
  // (discovery::dg_feature). dg-fa-pad: FA runs the 512-dim heads on the GPU.
  const hasFeature = (b, f) => !!b?.patch?.features?.includes(f);
  const patchLabel = (b) => (b?.patch ? ` · ${b.patch.name || "patched"}` : "");
  const faWorks = $derived(hasFeature(selectedBuild, "dg-fa-pad"));
  const faTurnSizing = $derived(faWorks && hasFeature(selectedBuild, "dg-fa-turn-sizing"));
  // A patched runner's test hooks never reach it (the helper strips them).
  const DG_TEST_HOOK_ENV = ["DG_PKV_TYPE", "DG_SWA_WINDOW", "DG_POISON", "DG_RING_POISON", "DG_DUMP_LOGITS", "DG_EXIT_AFTER_DUMP", "DG_PROFILE", "DG_SC_SPLITK", "DG_SC_SPLITK_CHECK"];
  const faSized = $derived(hasFeature(selectedBuild, "dg-fa-pad") && hasFeature(selectedBuild, "dg-fa-turn-sizing") && !!draft?.diffusion?.flash_attn);
  // Env keys FIDIM composes for every diffusion run (profile::DG_OWNED_ENV):
  // validate refuses them, and the editor has no env field to clear them.
  const DG_OWNED_ENV = ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "NGL", "MAXTOK", "FA", "DG_FREE_VRAM_MB", "DG_FREE_RAM_MB", "GGML_BACKEND_PATH", "GGML_CUDA_DEVICES", "GGML_CUDA_ENABLE_UNIFIED_MEMORY"];
  // With the safeguard on, a ROCBLAS_USE_HIPBLASLT* entry in the env would
  // override its value (validate warns the same way).
  const dgEnvConflicts = $derived.by(() => {
    if (!isDiffusion) return [];
    const upper = (k) => String(k).toUpperCase();
    return Object.keys(draft?.env ?? {}).filter((k) => DG_OWNED_ENV.includes(upper(k)) ||
      DG_TEST_HOOK_ENV.includes(upper(k)) ||
      (upper(k) === "GPU_RESOURCE_CACHE_SIZE" && String(draft.env[k]).trim() !== "0") ||
      (draft?.diffusion?.hipblaslt_safeguard !== false && upper(k).startsWith("ROCBLAS_USE_HIPBLASLT")));
  });
  function dropEnvConflicts() {
    for (const k of [...dgEnvConflicts]) delete draft.env[k];
    scheduleCheck();
  }

  const buildNum = (b) => Number(/^b(\d+)/.exec(String(b?.release_tag ?? b?.version ?? ""))?.[1] ?? 0);
  // Newest upstream build by release NUMBER (a string sort ranks b9817 above
  // b10771). A fork build reports a higher number but is picked per profile,
  // never by default.
  function newestUpstreamBuild() {
    const newestOf = (list) => list.filter((b) => b.version).sort((a, b) => buildNum(b) - buildNum(a))[0];
    return newestOf(builds.filter((b) => (b.channel ?? "upstream") === "upstream")) ?? newestOf(builds);
  }
  // Newest runner build; at the same release a patched one wins (it shares
  // its base's release tag and version).
  function newestRunnerBuild() {
    const runs = runnerBuilds.filter((b) => !b.version_error);
    return (runs.length ? runs : runnerBuilds).slice()
      .sort((a, b) => buildNum(b) - buildNum(a) || (b.patch ? 1 : 0) - (a.patch ? 1 : 0))[0];
  }

  function enterDiffusion() {
    draft.engine = "diffusion-gemma";
    draft.diffusion ??= { ...DG_DEFAULTS };
    draft.diffusion.seed ??= null;
    draft.diffusion.flash_attn ??= false;
    // One card: an empty one if any (the runner sizes its context to the
    // whole card and cannot see other processes' VRAM), else the first.
    const discrete = devices.filter((d) => !d.integrated);
    const pick = discrete.find((d) => !occupancy[d.stable_key]?.length) ?? discrete[0];
    draft.devices = pick ? [{ key: pick.stable_key, split_fraction: null, resolved_index_last_launch: null }] : [];
    draft.split_mode = null;
    draft.main_device = 0;
    // Kept as an "off" object so the (hidden) speculative card never reads
    // null; normalized() drops it from the saved diffusion profile.
    draft.speculative = { mode: "off", n_max: null, n_min: null, p_min: null };
    draft.model.mmproj = null;
    draft.model.draft = null;
    draft.runtime.slots = 1;
    draft.runtime.n_gpu_layers = 99;
    draft.runtime.ctx_total = 0;
    draft.keep_alive_seconds = null;
    // What validate reports as ignored lives on cards hidden for this
    // engine; clear it so no finding is left that cannot be fixed here.
    for (const k of Object.keys(draft.sampling ?? {})) draft.sampling[k] = null;
    if (draft.chat) draft.chat.enable_thinking = null;
    draft.runtime.extra_flags = [];
    draft.runtime.cache_reuse = null;
    // A runtime still applies to a runner build without its own ROCm.
    if (selectedBuild?.bundled_runtime) draft.rocm_runtime = null;
    if (!selectedBuild?.runner_exe) {
      const b = newestRunnerBuild();
      if (b) onBuildPick(b.path);
    } else {
      // Kept build: flash attention follows it, as a build pick would set it.
      draft.diffusion.flash_attn = hasFeature(selectedBuild, "dg-fa-pad");
    }
  }

  function leaveDiffusion(h) {
    delete draft.engine;
    delete draft.diffusion;
    if (!draft.runtime.ctx_total) draft.runtime.ctx_total = Math.min(32768, h?.context_length ?? 262144);
    // A fork build is chosen per profile, never inherited by a llama-server one.
    if (selectedBuild && (selectedBuild.channel ?? "upstream") !== "upstream") {
      const b = newestUpstreamBuild();
      if (b && (b.channel ?? "upstream") === "upstream") onBuildPick(b.path);
    }
  }

  function samePath(a, b) {
    if (!a || !b) return false;
    return String(a).replace(/\//g, "\\").replace(/\\+$/, "").toLowerCase() ===
           String(b).replace(/\//g, "\\").replace(/\\+$/, "").toLowerCase();
  }
  function base(p) { return String(p ?? "").split(/[\\/]/).pop(); }
  function modelLabel(m) {
    const bits = [base(m.path)];
    if (isHf(m)) {
      // A checkpoint folder: name · quantization or dtype · weights size.
      bits.push(m.hf?.quantization || m.hf?.torch_dtype || "safetensors");
      bits.push(fmtGib(m.hf?.weights_bytes ?? m.file_size ?? 0));
      return bits.join(" · ");
    }
    const h = m.header;
    if (h?.architecture) bits.push(h.architecture);
    if (h?.file_type != null) bits.push(quantName(h.file_type));
    bits.push(fmtGib(m.file_size));
    return bits.join(" · ");
  }
  function quantName(ft) {
    const names = { 0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16" };
    return names[ft] ?? `ft${ft}`;
  }

  function onModelPick(path) {
    draft.model.path = path;
    const m = models.find((x) => samePath(x.path, path));
    // Auto-pair what discovery found beside the file; keep explicit choices otherwise.
    draft.model.mmproj = m?.mmproj_candidates?.[0] ?? null;
    draft.model.draft = null;
    const h = m?.header;
    // The header decides the engine; switching resets what the other one
    // cannot use.
    const engine = m?.engine ?? "llama-server";
    if (engine === "sglang") {
      // A safetensors checkpoint: only SGLang loads it.
      if (!isSgLang) {
        if (isDiffusion) leaveDiffusion(h);
        enterSgLang();
      }
    } else if (isSgLang) {
      // SGLang serves this GGUF as it is; only a diffusion header pulls the
      // profile off it.
      if (engine === "diffusion-gemma") { leaveSgLang(h); enterDiffusion(); }
    } else if (engine !== (draft.engine ?? "llama-server")) {
      if (engine === "diffusion-gemma") enterDiffusion();
      else leaveDiffusion(h);
    }
    if (engine === "diffusion-gemma") {
      // MAXTOK is capped far below the trained context, and NGL must reach
      // block_count + 1 (the output layer) for a full offload: no clamps.
      draft.model.mmproj = null;
    } else {
      const trained = h?.context_length ?? m?.hf?.max_position_embeddings;
      if (trained && draft.runtime.ctx_total > trained) draft.runtime.ctx_total = trained;
      if (h?.block_count && draft.runtime.n_gpu_layers > h.block_count) draft.runtime.n_gpu_layers = h.block_count;
    }
    creator = null;
    scheduleCheck();
  }
  function onBuildPick(path) {
    const b = builds.find((x) => samePath(x.path, path));
    draft.build.path = path;
    draft.build.version = b?.version ?? null;
    // A bundled build runs on its own ROCm and replaces the runtime picker,
    // so a runtime left set would be a hidden setting that does nothing.
    if (isDiffusion && b?.bundled_runtime) draft.rocm_runtime = null;
    // Flash attention helps only a runner that pads keys for it; on the stock
    // runner it puts the 512-dim heads on the CPU. Follow the build.
    if (isDiffusion && draft.diffusion) draft.diffusion.flash_attn = hasFeature(b, "dg-fa-pad");
    scheduleCheck();
  }
  function onDraftPick(path) {
    draft.model.draft = path ? { path, enabled: false } : null;
    if (!path && draft.speculative && (draft.speculative.mode === "draft" || draft.speculative.mode === "dflash" || (draft.speculative.mode === "mtp" && !mtpBuiltIn)))
      draft.speculative.mode = "off";
    scheduleCheck();
  }

  // ---- selection / new / duplicate ---------------------------------------
  function select(id) {
    try {
      selectedId = id;
      const row = profiles.find((r) => r.profile.id === id);
      draft = row ? withDefaults(JSON.parse(JSON.stringify(row.profile))) : null;
      check = null;
      creator = null;
      sgEnvText = envToText(draft?.env);
      sgCustomParser = { reasoning_parser: false, tool_call_parser: false };
      log(`select ${id}: engine=${draft?.engine ?? "llama-server"} model=${base(draft?.model?.path)} build=${draft?.build?.version} ctx=${draft?.runtime?.ctx_total} slots=${draft?.runtime?.slots} kv=${draft?.runtime?.kv_type_k} spec=${draft?.speculative?.mode} port=${draft?.server?.port}`);
      scheduleCheck();
    } catch (e) {
      log(`select ${id} FAILED: ${e?.stack ?? String(e)}`);
      toastMsg(`could not load profile ${id}: ${String(e)}`, true);
    }
  }

  function withDefaults(p) {
    p.sampling ??= {};
    p.chat ??= {};
    p.env ??= {};
    p.build ??= { path: "", version: null };
    p.server.host ??= "127.0.0.1";
    p.runtime.extra_flags ??= [];
    p.runtime.threads ??= null;
    p.runtime.cache_reuse ??= null;
    p.rocm_runtime ??= null;
    p.keep_alive_seconds ??= null;
    for (const k of ["temperature", "top_p", "top_k", "min_p", "dry_multiplier", "repeat_penalty", "presence_penalty"])
      p.sampling[k] ??= null;
    p.chat.enable_thinking ??= null;
    // Legacy profiles: an enabled draft file means speculative was on.
    if (!p.speculative) {
      const d = p.model.draft;
      p.speculative = {
        mode: d?.enabled ? (String(d.path).toLowerCase().includes("mtp") ? "mtp" : "draft") : "off",
        n_max: null, n_min: null, p_min: null,
      };
    }
    p.speculative.n_max ??= null; p.speculative.n_min ??= null; p.speculative.p_min ??= null;
    // Serde fills an absent diffusion section with these same defaults.
    if (p.engine === "diffusion-gemma") {
      p.diffusion ??= {};
      p.diffusion.hipblaslt_safeguard ??= DG_DEFAULTS.hipblaslt_safeguard;
      p.diffusion.default_max_tokens ??= DG_DEFAULTS.default_max_tokens;
      p.diffusion.flash_attn ??= false;
      p.diffusion.seed ??= null;
    }
    // Same for an absent sglang section (cfg_of() = SgLangCfg::default()).
    if (p.engine === "sglang") p.sglang = sgFill(p.sglang);
    return p;
  }

  // ---- speculative decoding -------------------------------------------------
  const mtpBuiltIn = $derived((header?.nextn_predict_layers ?? 0) > 0);
  const draftName = $derived(base(draft?.model?.draft?.path ?? ""));
  const specModes = $derived.by(() => {
    const m = [{ v: "off", l: "off" }];
    if (mtpBuiltIn) m.push({ v: "mtp", l: `MTP — built into this model (${header.nextn_predict_layers} predict layer${header.nextn_predict_layers === 1 ? "" : "s"})` });
    else if (draft?.model?.draft?.path && draftName.toLowerCase().includes("mtp")) m.push({ v: "mtp", l: `MTP — external head ${draftName}` });
    if (draft?.model?.draft?.path) {
      m.push({ v: "draft", l: `draft model — ${draftName}` });
      if (draftName.toLowerCase().includes("dflash")) m.push({ v: "dflash", l: `DFlash — ${draftName}` });
    }
    m.push({ v: "ngram", l: "n-gram (model-free, prompt-lookup)" });
    return m;
  });
  const SPEC_ENGINE_DEFAULTS = { n_max: 3, n_min: 0, p_min: 0 };
  function onSpecMode(mode) {
    draft.speculative.mode = mode;
    // Keep the legacy flag in step so `-md` is emitted exactly when a file is in use.
    if (draft.model.draft) draft.model.draft.enabled = mode !== "off" && mode !== "ngram" && !(mode === "mtp" && mtpBuiltIn);
    scheduleCheck();
  }
  function applySpecDefaults() {
    draft.speculative.n_max = SPEC_ENGINE_DEFAULTS.n_max;
    draft.speculative.n_min = SPEC_ENGINE_DEFAULTS.n_min;
    draft.speculative.p_min = SPEC_ENGINE_DEFAULTS.p_min;
    scheduleCheck();
  }

  function newProfile() {
    const first = devices.find((d) => !d.integrated);
    const newest = newestUpstreamBuild();
    draft = withDefaults({
      schema: 1, id: "new-profile", name: "New profile",
      build: { path: newest?.path ?? "", version: newest?.version ?? null },
      model: { path: "", mmproj: null, draft: null },
      devices: first ? [{ key: first.stable_key, split_fraction: null, resolved_index_last_launch: null }] : [],
      split_mode: null, main_device: 0, rocm_runtime: null,
      server: { port: 9710, alias: "new-profile", host: "127.0.0.1" },
      runtime: {
        n_gpu_layers: 99, ctx_total: 32768, slots: 1, kv_type_k: "f16", kv_type_v: "f16",
        flash_attn: "on", batch_logical: 2048, batch_physical: 512, cont_batching: true,
        kv_unified: false, cache_reuse: null, threads: null, extra_flags: [],
      },
      sampling: {}, chat: {}, env: {}, baseline: null, notes: "",
    });
    selectedId = null;
    check = null;
    creator = null;
    sgEnvText = "";
    sgCustomParser = { reasoning_parser: false, tool_call_parser: false };
    scheduleCheck();
  }

  // A new SGLang profile on a checkpoint the Models view handed over: id and
  // name from the folder (or file stem), alias = id.
  function newSgLangProfile(path) {
    newProfile();
    enterSgLang();
    const stem = base(String(path).replace(/[\\/]+$/, "")).replace(/\.gguf$/i, "") || "sglang";
    draft.id = stem.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "") || "sglang";
    draft.name = stem;
    draft.server.alias = draft.id;
    draft.server.port = 30000;
    onModelPick(path);
  }

  function duplicate() {
    if (!draft) return;
    draft = { ...JSON.parse(JSON.stringify(draft)), id: draft.id + "-copy", baseline: null };
    draft.server.port += 1;
    draft.server.alias = draft.id;
    selectedId = null;
    sgEnvText = envToText(draft.env);
    scheduleCheck();
  }

  // ---- live pre-flight (V-2): debounced, same check objects as launch ----
  function scheduleCheck() {
    if (checkTimer) clearTimeout(checkTimer);
    checkTimer = setTimeout(runCheck, 700);
  }

  async function runCheck() {
    if (!draft) return;
    checking = true;
    try {
      check = await api("live_check", { p: normalized(draft) });
      const now = new Map((check?.results ?? []).map((r) => [r.id, outcomeKind(r.outcome)]));
      const diff = new Set();
      for (const [id, kind] of now) if (prevKinds.has(id) && prevKinds.get(id) !== kind) diff.add(id);
      prevKinds = now;
      if (diff.size) {
        changed = diff;
        if (changedTimer) clearTimeout(changedTimer);
        changedTimer = setTimeout(() => (changed = new Set()), 900);
      }
    } catch (e) {
      check = { error: String(e) };
    }
    checking = false;
  }

  const numOrNull = (v) => (v === "" || v === null || v === undefined ? null : Number(v));

  function normalized(p) {
    const copy = JSON.parse(JSON.stringify(p));
    copy.server.port = Number(copy.server.port) || 0;
    const r = copy.runtime;
    for (const k of ["n_gpu_layers", "ctx_total", "slots", "batch_logical", "batch_physical"])
      r[k] = Number(r[k]) || 0;
    r.cache_reuse = numOrNull(r.cache_reuse);
    r.threads = numOrNull(r.threads);
    copy.keep_alive_seconds = numOrNull(copy.keep_alive_seconds);
    if (copy.keep_alive_seconds === null) delete copy.keep_alive_seconds;
    else copy.keep_alive_seconds = Math.round(copy.keep_alive_seconds);
    if (copy.speculative) {
      const sp = copy.speculative;
      sp.n_max = numOrNull(sp.n_max); sp.n_min = numOrNull(sp.n_min); sp.p_min = numOrNull(sp.p_min);
      if (sp.n_max !== null) sp.n_max = Math.round(sp.n_max);
      if (sp.n_min !== null) sp.n_min = Math.round(sp.n_min);
      for (const k of ["n_max", "n_min", "p_min"]) if (sp[k] === null) delete sp[k];
    }
    if (typeof r.extra_flags === "string")
      r.extra_flags = r.extra_flags.split(/\s+/).filter(Boolean);
    copy.main_device = Number(copy.main_device) || 0;
    for (const d of copy.devices)
      d.split_fraction = d.split_fraction === null || d.split_fraction === "" ? null : Number(d.split_fraction);
    if (copy.devices.length < 2) copy.split_mode = null;
    const s = copy.sampling ?? {};
    for (const k of Object.keys(s)) s[k] = numOrNull(s[k]);
    if (s.top_k !== null && s.top_k !== undefined) s.top_k = Math.round(s.top_k);
    for (const k of Object.keys(s)) if (s[k] === null) delete s[k];
    copy.sampling = s;
    if (copy.chat.enable_thinking === null) delete copy.chat.enable_thinking;
    if (!copy.model.mmproj) copy.model.mmproj = null;
    if (copy.model.draft && !copy.model.draft.path) copy.model.draft = null;
    if ((copy.engine ?? "llama-server") === "llama-server") {
      // Absent means llama-server, so these profiles save byte-identically.
      // Never send `engine: null`: it does not parse.
      delete copy.engine;
      delete copy.diffusion;
      delete copy.sglang;
    } else if (copy.engine === "sglang") {
      // No llama.cpp build or runtime; one card; nothing of the hidden
      // llama-server cards is kept, so validate has nothing to report on them.
      copy.build = { path: "", version: null };
      copy.rocm_runtime = null;
      copy.devices = copy.devices.slice(0, 1);
      copy.split_mode = null;
      copy.main_device = 0;
      copy.model.mmproj = null;
      copy.model.draft = null;
      r.extra_flags = [];
      delete copy.speculative;
      delete copy.keep_alive_seconds;
      delete copy.diffusion;
      const s = (copy.sglang = sgFill(copy.sglang));
      s.mem_fraction = Number(s.mem_fraction);
      if (!Number.isFinite(s.mem_fraction)) s.mem_fraction = SG_DEFAULTS.mem_fraction;
      for (const k of ["chunked_prefill", "max_running_requests", "memguard_gb", "compile_threads"]) s[k] = intOr(s[k], SG_DEFAULTS[k]);
      // Optionals are absent, never null: that is how serde writes them back.
      s.mamba_slots = numOrNull(s.mamba_slots);
      if (s.mamba_slots === null || !(s.mamba_slots > 0)) delete s.mamba_slots;
      else s.mamba_slots = Math.round(s.mamba_slots);
      if (s.spec) {
        const sp = s.spec;
        sp.algorithm = String(sp.algorithm || SG_SPEC_DEFAULTS.algorithm);
        for (const k of ["steps", "topk", "draft_tokens"]) sp[k] = Math.max(1, intOr(sp[k], SG_SPEC_DEFAULTS[k]));
        sp.token_map = String(sp.token_map ?? "").trim();
        if (!sp.token_map) delete sp.token_map;
      } else {
        delete s.spec;
      }
      for (const k of ["reasoning_parser", "tool_call_parser"]) {
        s[k] = String(s[k] ?? "").trim();
        if (!s[k]) delete s[k];
      }
      s.aliases = listOf(s.aliases, /[,\s]+/);
      s.extra_args = listOf(s.extra_args, /\r?\n/);
      s.enable_metrics = !!s.enable_metrics;
      s.sleep_on_idle = !!s.sleep_on_idle;
    } else if (copy.engine === "diffusion-gemma") {
      delete copy.sglang;
      copy.devices = copy.devices.slice(0, 1);
      copy.split_mode = null;
      // e.g. a profile promoted onto a bundled build keeps its old runtime,
      // and the picker that could clear it is hidden for that build.
      if (builds.find((b) => samePath(b.path, copy.build?.path))?.bundled_runtime) copy.rocm_runtime = null;
      r.slots = 1;
      delete copy.speculative;
      delete copy.keep_alive_seconds;
      const dg = (copy.diffusion ??= { ...DG_DEFAULTS });
      dg.default_max_tokens = Math.round(Number(dg.default_max_tokens)) || 0;
      dg.seed = numOrNull(dg.seed);
      if (dg.seed === null || !Number.isFinite(dg.seed)) delete dg.seed;
      else dg.seed = Math.round(dg.seed);
      if (!dg.flash_attn) delete dg.flash_attn;
    }
    return copy;
  }

  function toggleDevice(dev) {
    if (isDiffusion || isSgLang) {
      // Radio: a second visible device makes the runner abort every prompt,
      // and the bundle has no kernels for the iGPU. SGLang runs one card per
      // server (a tensor-parallel pair would be a second profile's job).
      if (dev.integrated || (draft.devices.length === 1 && draft.devices[0].key === dev.stable_key)) return;
      draft.devices = [{ key: dev.stable_key, split_fraction: null, resolved_index_last_launch: null }];
      draft.split_mode = null;
      scheduleCheck();
      return;
    }
    const i = draft.devices.findIndex((d) => d.key === dev.stable_key);
    if (i >= 0) draft.devices.splice(i, 1);
    else draft.devices.push({ key: dev.stable_key, split_fraction: null, resolved_index_last_launch: null });
    if (draft.devices.length > 1 && !draft.split_mode) draft.split_mode = "layer";
    if (draft.devices.length <= 1) draft.split_mode = null;
    scheduleCheck();
  }

  // Outcome comes from Rust as `{ outcome: "pass" | "note" | "warn" | "block", message? }`.
  // Never default to "pass": an unrecognised shape must look wrong, not green.
  function outcomeKind(o) {
    if (o === "pass") return "pass";
    if (o && typeof o === "object") {
      if (typeof o.outcome === "string") return o.outcome;
      for (const k of ["block", "warn", "note", "pass"]) if (o[k] !== undefined) return k;
    }
    return "block";
  }
  function outcomeMsg(o) {
    if (o === "pass") return "";
    if (o && typeof o === "object") {
      if (typeof o.outcome === "string") return o.message ?? "";
      return o.warn ?? o.block ?? o.note ?? "";
    }
    return `unrecognised pre-flight outcome: ${JSON.stringify(o)}`;
  }

  const anyBlock = $derived(check?.results?.some((r) => outcomeKind(r.outcome) === "block") ?? false);

  // Per-device budget bars: estimate vs free VRAM on the resolved device.
  const budgets = $derived.by(() => {
    if (!check?.estimate?.per_device || !check?.resolved) return [];
    return check.estimate.per_device.map((est, i) => {
      const res = check.resolved[i];
      const freeBytes = (res?.device?.free_mib ?? 0) * 1024 * 1024;
      const frac = freeBytes > 0 ? est.total_bytes / freeBytes : 0;
      return {
        key: est.key,
        name: res?.device?.name ?? est.key,
        estGib: est.total_bytes / GIB,
        freeGib: freeBytes / GIB,
        frac,
        kind: frac > 1 ? "block" : frac > 0.9 ? "warn" : "pass",
        detail: `weights ${(est.weights_bytes / GIB).toFixed(2)} + kv ${(est.kv_bytes / GIB).toFixed(2)} + compute ${(est.compute_bytes / GIB).toFixed(2)} + overhead ${(est.overhead_bytes / GIB).toFixed(2)}`,
      };
    });
  });

  // SGLang's estimate is one number for the one card (per_device is empty):
  // the static pool = mem_fraction x the card's VRAM, against what is free now.
  const sgBudget = $derived.by(() => {
    if (!isSgLang || !check?.estimate) return null;
    const dev = check.resolved?.[0]?.device;
    const totalBytes = (dev?.total_mib ?? 0) * 1024 * 1024;
    const freeBytes = (dev?.free_mib ?? 0) * 1024 * 1024;
    const est = check.estimate.total_bytes ?? 0;
    const frac = totalBytes > 0 ? est / totalBytes : 0;
    return {
      name: dev?.name ?? draft?.devices?.[0]?.key ?? "?",
      key: dev?.stable_key ?? draft?.devices?.[0]?.key ?? "",
      estGib: est / GIB, totalGib: totalBytes / GIB, freeGib: freeBytes / GIB,
      frac, freeFrac: totalBytes > 0 ? freeBytes / totalBytes : 0,
      kind: est > freeBytes ? "block" : frac > 0.9 ? "warn" : "pass",
      assumptions: check.estimate.assumptions ?? [],
    };
  });

  // ---- creator defaults ----------------------------------------------------
  // The GGUF header often embeds them (general.sampling.*); Hugging Face's
  // generation_config.json is the fallback behind the button.
  const embedded = $derived.by(() => {
    if (!header) return null;
    const has = header.sampling_temp != null || header.sampling_top_k != null || header.sampling_top_p != null;
    return has ? {
      repo: header.source_repo ?? "GGUF header",
      temperature: header.sampling_temp ?? null,
      top_p: header.sampling_top_p ?? null,
      top_k: header.sampling_top_k ?? null,
      min_p: header.sampling_min_p ?? null,
      repetition_penalty: header.sampling_repeat_penalty ?? null,
      embedded: true,
    } : null;
  });
  const creatorShown = $derived(creator && !creator.error ? creator : embedded);

  async function fetchCreator() {
    if (!draft?.model?.path) return;
    creatorBusy = true;
    try {
      creator = await api("creator_defaults", { modelPath: draft.model.path });
    } catch (e) {
      creator = { error: String(e) };
    }
    creatorBusy = false;
  }
  function applyCreator() {
    const c = creatorShown;
    if (!c) return;
    const s = draft.sampling;
    if (c.temperature != null) s.temperature = c.temperature;
    if (c.top_p != null) s.top_p = c.top_p;
    if (c.top_k != null) s.top_k = c.top_k;
    if (c.min_p != null) s.min_p = c.min_p;
    if (c.repetition_penalty != null) s.repeat_penalty = c.repetition_penalty;
    scheduleCheck();
  }

  // ---- actions --------------------------------------------------------------
  async function save() {
    busy = "save";
    try {
      await api("save_profile", { p: normalized(draft) });
      flashSaved();
      await load();
      selectedId = draft.id;
    } catch (e) {
      toastMsg(String(e), true);
    }
    busy = "";
  }

  async function remove() {
    if (!selectedId) return;
    busy = "delete";
    try {
      await api("delete_profile", { id: selectedId });
      toastMsg(`Deleted ${selectedId}`);
      selectedId = null;
      draft = null;
      await load();
    } catch (e) {
      toastMsg(String(e), true);
    }
    busy = "";
  }

  async function launch(overrideBlocks) {
    busy = "launch";
    try {
      await api("save_profile", { p: normalized(draft) });
      const r = await api("launch_profile", { id: draft.id, overrideBlocks });
      if (r.blocked) {
        toastMsg("Blocked by pre-flight. See the list below, or Override to accept the named risks.", true);
        check = { ...check, results: r.results };
      } else {
        const place = (r.placement ?? [])
          .map((p) => `${(Math.max(p.dedicated_bytes ?? 0, p.committed_bytes ?? 0) / GIB).toFixed(1)} GiB on ${p.key.split(":").pop()}`)
          .join(", ");
        const ka = r.keepalive?.pid ? ` · keep-alive ${r.keepalive.interval_s}s` : r.keepalive?.error ? ` · keep-alive FAILED: ${r.keepalive.error}` : " · keep-alive off";
        const rep = r.replaced ? ` · replaced ${r.replaced} on this port` : "";
        toastMsg(`Launched pid ${r.state.pid} on port ${r.state.port}${r.cold_start ? " (cold cache)" : ""} — ${place}${ka}${rep}`);
      }
    } catch (e) {
      toastMsg(String(e), true);
    }
    busy = "";
  }

  async function exportScript(format) {
    busy = "export";
    try {
      const r = await api("export_profile", { id: draft.id, format });
      toastMsg(`Exported to ${r.path}`);
    } catch (e) {
      toastMsg(String(e), true);
    }
    busy = "";
  }

  let toastTimer = null;
  function toastMsg(text, isError = false) {
    toast = { text, isError };
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (toast = null), 6000);
  }
</script>
<h1>
  Profiles
  <span class="sub" title="Pre-flight here is the same check a launch runs">One saved launch: model, build, GPUs and flags.</span>
</h1>
{#if loadErrors.length}
  <div class="card">
    {#each loadErrors as [name, msg]}
      <div class="notice"><span class="chip block">{name} failed</span> <span class="mono">{msg}</span></div>
    {/each}
    <div class="faint small" style="margin-top: 6px;">Check the roots on the Settings tab, then Rescan.</div>
  </div>
{:else if !loading && (!models.length || !builds.length || !devices.length)}
  <div class="card notice">
    <span class="chip warn" title="Nothing to pick from: no models, builds or GPUs were found under the roots">empty</span>
    <span>{models.length} models · {builds.length} builds · {devices.length} GPUs — set the roots in <b>Settings</b>, then Rescan.</span>
  </div>
{/if}

<div class="pf">
  <!-- list -->
  <aside class="list">
    <div class="toolbar" style="margin-bottom: 10px;">
      <button class="btn" onclick={newProfile}>New</button>
      <button class="btn" onclick={duplicate} disabled={!draft}>Duplicate</button>
      <div class="grow"></div>
      <button class="btn small" onclick={rescan} disabled={busy === "scan"} title="Scan the build and model roots again">{busy === "scan" ? "…" : "Rescan"}</button>
    </div>
    <div class="rows">
      {#each profiles as row, i (row.profile.id)}
        {@const p = row.profile}
        <button class="row" class:active={selectedId === p.id} onclick={() => select(p.id)} in:fly={arrive(stagger(i, 30))} out:slide={leave} animate:flip={flipParams}>
          <div class="r1"><span class="id">{p.id}</span><span class="mono faint">:{p.server.port}</span></div>
          <div class="r2">{base(p.model.path) || "no model"}</div>
          <div class="r3">
            <span class="mono">{p.engine === "sglang" ? "venv" : p.build.version ?? "?"}</span>
            <span>{p.devices.length} GPU{p.devices.length === 1 ? "" : "s"}{p.split_mode ? `, ${p.split_mode} split` : ""}</span>
            {#if p.engine === "diffusion-gemma"}<span class="chip note" title="Runs on the DiffusionGemma runner, not llama-server">diffusion</span>
            {:else if p.engine === "sglang"}<span class="chip note" title="Runs on SGLang from the venv in Settings, behind model_router.py">sglang</span>
            {:else if p.engine && p.engine !== "llama-server"}<span class="chip block" title="This file names an engine this version does not know; fix the engine field by hand">unknown</span>{/if}
            {#if p.baseline}<span class="tok num">{p.baseline.serial_tok_s} tok/s</span>{/if}
            {#if row.findings.length}<span class="chip warn" title="Problems in the profile file; open it to see them">{row.findings.length}</span>{/if}
          </div>
        </button>
      {:else}
        <div class="empty">No profiles yet — New, or <button class="link" onclick={() => go("models")}>Models</button> → Get a model</div>
      {/each}
    </div>
  </aside>

  <!-- editor -->
  {#if draft}
    {@const kinds = (check?.results ?? []).map((r) => outcomeKind(r.outcome))}
    {@const nPass = kinds.filter((k) => k === "pass").length}
    {@const nWarn = kinds.filter((k) => k === "warn" || k === "note").length}
    {@const nBlock = kinds.filter((k) => k === "block").length}
    <div class="editor">
      <div class="editbar">
        <div class="who">
          <span class="name">{draft.name || draft.id}</span>
          <span class="meta"><span class="mono muted">{draft.id} · :{draft.server.port} · {draft.server.alias}</span>{#if savedFlash}<span class="chip pass" in:scale={{ duration: LAYOUT, start: 0.7 }} out:fade={{ duration: LAYOUT }}>saved</span>{:else if !selectedId}<span class="chip accent" in:scale={{ duration: LAYOUT, start: 0.7 }}>unsaved</span>{/if}
          <span class="pfsum" title="Pre-flight, re-run on every edit">
          {#if checking}<span class="chip plain live">checking</span>
          {:else if check?.error}<span class="chip block">check failed</span>
          {:else if kinds.length}
            <span class="chip pass">{nPass} pass</span>
            {#if nWarn}<span class="chip warn">{nWarn} warn</span>{/if}
            {#if nBlock}<span class="chip block">{nBlock} block</span>{/if}
          {/if}
          </span></span>
        </div>
        <div class="actions">
          <button class="btn primary" onclick={save} disabled={!!busy}>Save</button>
          <button class="btn" onclick={() => launch(false)} disabled={!!busy || anyBlock}>{#if busy === "launch"}<span class="spinner"></span> Loading…{:else}Save & load{/if}</button>
          {#if anyBlock}
            <button class="btn danger" onclick={() => launch(true)} disabled={!!busy}>Override blocks &amp; load</button>
          {/if}
          <button class="btn" onclick={() => exportScript("bat")} disabled={!!busy} title="Write a standalone .bat that launches this profile">Export .bat</button>
          <button class="btn" onclick={() => exportScript("ps1")} disabled={!!busy} title="Write a standalone .ps1 that launches this profile">Export .ps1</button>
          <button class="btn danger" onclick={remove} disabled={!!busy || !selectedId}>Delete</button>
        </div>
      </div>

      <!-- identity -->
      <section class="card">
        <div class="sec">Identity</div>
        <div class="formgrid">
          <label class="field" title="File name of the profile and what the CLI launches; letters, digits, dashes (fidim launch <id>)"><span class="k">id</span><input bind:value={draft.id} oninput={scheduleCheck} /></label>
          <label class="field" style="grid-column: span 2;" title="Display name, free text"><span class="k">name</span><input bind:value={draft.name} /></label>
          <label class="field" title="Port the server listens on; pre-flight checks it is free (--port)"><span class="k">port</span><input type="number" bind:value={draft.server.port} oninput={scheduleCheck} /></label>
          <label class="field" style="grid-column: span 2;" title={`Name clients pass as 'model', unique across running servers (${isSgLang ? "--served-model-name" : "--alias"})`}><span class="k">served name</span><input bind:value={draft.server.alias} oninput={scheduleCheck} /></label>
          {#if isSgLang}
            <label class="field" style="grid-column: span 2;" title="Interface to bind; 127.0.0.1 stays local, 0.0.0.0 is every interface (--host)"><span class="k">host</span><input bind:value={draft.server.host} oninput={scheduleCheck} placeholder="127.0.0.1" /></label>
          {/if}
        </div>
      </section>

      <!-- model + build -->
      <section class="card">
        <div class="sec">Model <span class="faint">{isSgLang ? "a .gguf file or a Hugging Face safetensors folder" : `${ggufModels.length} GGUF files in the roots`}</span></div>
        <div class="formgrid">
          {#if modelEngine !== "diffusion-gemma"}
            <label class="field" style="grid-column: span 2;" title="Which server runs this profile: a llama.cpp build, or SGLang from the venv in Settings">
              <span class="k">engine</span>
              <select value={draft.engine ?? "llama-server"} onchange={(e) => onEnginePick(e.target.value)}>
                <option value="llama-server">llama-server (llama.cpp build)</option>
                <option value="sglang">SGLang (Python venv)</option>
                {#if isDiffusion}<option value="diffusion-gemma">DiffusionGemma</option>{/if}
              </select>
            </label>
          {/if}
          {#if isSgLang}
            <label class="field" style="grid-column: span 4;" title="A .gguf file or a Hugging Face checkpoint folder, inside the roots or not (--model-path)">
              <span class="k">model path</span>
              <input bind:value={draft.model.path} oninput={() => { creator = null; scheduleCheck(); }} placeholder="/models/Qwen3-27B-GGUF/model.gguf, or /models/Qwen3-27B (safetensors)" />
            </label>
            {#if models.length}
              <label class="field" style="grid-column: 1 / -1;" title="Fills the path from the scan: GGUF files and Hugging Face checkpoint folders">
                <span class="k">scanned models</span>
                <select value={selectedModel?.path ?? ""} onchange={(e) => { if (e.target.value) onModelPick(e.target.value); }}>
                  <option value="">— pick one —</option>
                  {#if hfModels.length}
                    <optgroup label="Hugging Face checkpoints (safetensors)">
                      {#each hfModels as m}<option value={m.path}>{modelLabel(m)}</option>{/each}
                    </optgroup>
                  {/if}
                  {#if ggufModels.length}
                    <optgroup label="GGUF files">
                      {#each ggufModels as m}<option value={m.path}>{modelLabel(m)}</option>{/each}
                    </optgroup>
                  {/if}
                </select>
              </label>
            {/if}
          {:else}
          <label class="field" style="grid-column: {modelEngine !== 'diffusion-gemma' ? 'span 4' : '1 / -1'};" title="GGUF file the server loads (-m)">
            <span class="k">weights</span>
            <select value={ggufModels.find((m) => samePath(m.path, draft.model.path))?.path ?? ""} onchange={(e) => onModelPick(e.target.value)}>
              <option value="" disabled>choose a model…</option>
              {#each ggufModels as m}
                <option value={m.path}>{modelLabel(m)}</option>
              {/each}
            </select>
          </label>
          {/if}
          {#if header}
            <div class="facts" style="grid-column: 1 / -1;">
              <span><b>{header.model_name ?? base(draft.model.path)}</b></span>
              <span>{header.architecture}</span>
              {#if header.size_label}<span>{header.size_label}</span>{/if}
              <span>{header.block_count} layers</span>
              <span>trained context {fmtInt(header.context_length ?? 0)}</span>
              {#if header.source_repo}<span title="Source repo from the GGUF header (general.base_model)">{header.source_repo}</span>{/if}
              {#if mtpBuiltIn}<span class="chip pass" title="The header has next-token predict layers; MTP needs no draft file">MTP</span>{/if}
              {#if isDiffusion}<span class="chip note" title="A diffusion LM: each reply is denoised in whole blocks of this many tokens">canvas {canvas}</span>{/if}
              <span class="path" style="flex-basis: 100%;">{draft.model.path}</span>
            </div>
          {:else if hf}
            <div class="facts" style="grid-column: 1 / -1;">
              <span><b>{base(String(draft.model.path).replace(/[\\/]+$/, ""))}</b></span>
              {#if hf.architecture || hf.model_type}<span>{hf.architecture ?? hf.model_type}{hf.architecture && hf.model_type && hf.architecture !== hf.model_type ? ` (${hf.model_type})` : ""}</span>{/if}
              <span title="Quantization from config.json, else the checkpoint's torch_dtype">{hf.quantization || hf.torch_dtype || "safetensors"}</span>
              {#if hf.num_layers != null}<span title="Transformer layers; on a hybrid model only the attention count carries a KV cache">{hf.num_layers} layers{hf.attention_layers != null && hf.attention_layers !== hf.num_layers ? ` (${hf.attention_layers} attention)` : ""}</span>{/if}
              {#if hf.num_experts}<span>{hf.num_experts} experts</span>{/if}
              {#if hf.max_position_embeddings}<span>trained context {fmtInt(hf.max_position_embeddings)}</span>{/if}
              {#if hf.weights_bytes}<span>{fmtGib(hf.weights_bytes)} weights</span>{/if}
              {#if hf.has_vision}<span class="chip note" title="The checkpoint carries a vision tower; SGLang serves images through it">vision</span>{/if}
              {#if !isSgLang}<span class="chip warn" title="A safetensors checkpoint loads only on SGLang; pick that engine above">needs sglang</span>{/if}
              <span class="path" style="flex-basis: 100%;">{draft.model.path}</span>
            </div>
          {:else if draft.model.path && isSgLang}
            <div class="facts" style="grid-column: 1 / -1;"><span class="chip note" title="Not under the roots or not scanned yet; pre-flight checks that it exists">not in scan</span><span class="path">{draft.model.path}</span></div>
          {:else if draft.model.path}
            <div class="facts" style="grid-column: 1 / -1;"><span class="chip warn" title="Not under the roots or not scanned yet; pre-flight checks that it exists">not in scan</span><span class="path">{draft.model.path}</span></div>
          {/if}
          {#if engineMismatch}
            <div class="notice" style="grid-column: 1 / -1;" title="The model needs the other engine; pre-flight blocks the launch until they agree">
              <span class="chip block">needs {modelEngine}</span>
              <span>runs {draft.engine ?? "llama-server"} · <button class="link" onclick={() => onModelPick(draft.model.path)}>Switch</button></span>
            </div>
          {/if}
          {#if !isDiffusion && !isSgLang}
          <label class="field" style="grid-column: span 3;" title="Multimodal projector paired with the weights so the server can read images (--mmproj)">
            <span class="k">vision projector</span>
            <select bind:value={draft.model.mmproj} onchange={scheduleCheck}>
              <option value={null}>none</option>
              {#each selectedModel?.mmproj_candidates ?? [] as c}<option value={c}>{base(c)}</option>{/each}
              {#if allMmproj.some((p) => !(selectedModel?.mmproj_candidates ?? []).includes(p))}
                <optgroup label="elsewhere in the roots">
                  {#each allMmproj.filter((p) => !(selectedModel?.mmproj_candidates ?? []).includes(p)) as c}<option value={c}>{base(c)} — {c.split(/[\\/]/).slice(-2, -1)[0]}</option>{/each}
                </optgroup>
              {/if}
              {#if draft.model.mmproj && !(selectedModel?.mmproj_candidates ?? []).includes(draft.model.mmproj) && !allMmproj.includes(draft.model.mmproj)}
                <option value={draft.model.mmproj}>{base(draft.model.mmproj)}</option>
              {/if}
            </select>
          </label>
          <label class="field" style="grid-column: span 3;" title="Small draft model or MTP sidecar for speculative decoding, turned on below (-md)">
            <span class="k">draft file</span>
            <select value={draft.model.draft?.path ?? ""} onchange={(e) => onDraftPick(e.target.value)}>
              <option value="">none</option>
              {#each selectedModel?.draft_candidates ?? [] as c}<option value={c}>{base(c)}</option>{/each}
              {#if allDrafts.some((p) => !(selectedModel?.draft_candidates ?? []).includes(p))}
                <optgroup label="elsewhere in the roots">
                  {#each allDrafts.filter((p) => !(selectedModel?.draft_candidates ?? []).includes(p)) as c}<option value={c}>{base(c)} — {c.split(/[\\/]/).slice(-2, -1)[0]}</option>{/each}
                </optgroup>
              {/if}
              {#if draft.model.draft?.path && !(selectedModel?.draft_candidates ?? []).includes(draft.model.draft.path) && !allDrafts.includes(draft.model.draft.path)}
                <option value={draft.model.draft.path}>{base(draft.model.draft.path)}</option>
              {/if}
            </select>
          </label>
          {/if}
          {#if !isSgLang}
          <label class="field" style="grid-column: span 3;" title={isDiffusion
            ? "Build that carries the DiffusionGemma runner; Unsloth's builds and patched builds of them do"
            : "llama.cpp build that launches this profile; Updates installs more side by side"}>
            <span class="k">build</span>
            <select value={selectedBuild?.path ?? ""} onchange={(e) => onBuildPick(e.target.value)}>
              <option value="" disabled>choose a build…</option>
              {#each builds as b}
                {@const noRunner = isDiffusion && !b.runner_exe}
                <option value={b.path} disabled={!!b.version_error || noRunner}>{b.tag} · {b.version ?? "broken"}{patchLabel(b)}{b.version_error ? " (does not run)" : ""}{noRunner ? " (no diffusion runner)" : ""}</option>
              {/each}
              {#if draft.build.path && !selectedBuild}<option value={draft.build.path}>{draft.build.path} (not in scan)</option>{/if}
            </select>
          </label>
          {#if selectedBuild?.bundled_runtime}
            <label class="field" style="grid-column: span 3;" title="This build ships its own ROCm, so no runtime applies">
              <span class="k">ROCm runtime</span>
              <select disabled><option>bundled with this build</option></select>
            </label>
          {:else}
          <label class="field" style="grid-column: span 3;" title="ROCm whose libraries go first on PATH; install more from Updates">
            <span class="k">ROCm runtime</span>
            <select bind:value={draft.rocm_runtime} onchange={scheduleCheck}>
              <option value={null}>config default ({runtimes.find((r) => r.is_default)?.name ?? "default"}{runtimes.find((r) => r.is_default)?.version ? ` · ${runtimes.find((r) => r.is_default).version}` : ""})</option>
              {#each runtimes.filter((r) => !r.is_default) as r}
                <option value={r.name} disabled={!r.available}>{r.name}{r.version ? ` · ${r.version}` : ""}{r.is_latest ? " (latest)" : ""}{r.available ? "" : " (missing)"}</option>
              {/each}
            </select>
          </label>
          {/if}
          {#if isDiffusion && !runnerBuilds.length}
            <div class="notice" style="grid-column: 1 / -1;">
              <span class="chip block">no runner</span>
              <span>Save, then <button class="link" onclick={() => go("updates")} title="Opens Updates; unsaved edits here are dropped">install an Unsloth build from Updates</button></span>
            </div>
          {/if}
          {/if}
        </div>
      </section>

      <!-- devices -->
      <section class="card">
        <div class="sec">GPU placement <span class="faint">{isDiffusion || isSgLang ? "one card" : "one card, or a layer split across two"}</span></div>
        <div class="devrows">
          {#each devices.filter((d) => !d.integrated) as dev}
            {@const entry = isDiffusion || isSgLang
              ? (draft.devices[0]?.key === dev.stable_key ? draft.devices[0] : undefined)
              : draft.devices.find((x) => x.key === dev.stable_key)}
            {@const used = dev.total_mib - dev.free_mib}
            {@const users = occupancy[dev.stable_key] ?? []}
            <label class="devrow" class:on={!!entry}>
              {#if isDiffusion}
                <input type="radio" name="dg-device" checked={!!entry} onchange={() => toggleDevice(dev)} title="The runner gets this one card; picking another moves it" />
              {:else if isSgLang}
                <input type="radio" name="sg-device" checked={!!entry} onchange={() => toggleDevice(dev)} title="SGLang gets this one card; picking another moves it (HIP_VISIBLE_DEVICES)" />
              {:else}
                <input type="checkbox" checked={!!entry} onchange={() => toggleDevice(dev)} />
              {/if}
              <span class="dname">
                <b>{dev.name.replace(/^AMD /, "")}</b>
                <span class="mono faint">{dev.stable_key.split(":").pop()} · {dev.backend}{dev.hip_index}</span>
                {#if dev.display}<span class="chip warn" title="This card drives a display; the desktop needs about 1.5 GB of it">display</span>{/if}
                {#if isDiffusion && users.length}<span class="chip warn" title="Running here: {users.join(", ")}. The runner cannot see their VRAM and may spill into shared memory; prefer an empty card">in use</span>
                {:else if isSgLang && users.length}<span class="chip warn" title="Running here: {users.join(", ")}. SGLang takes its VRAM share up front; they must fit in the rest">in use</span>{/if}
              </span>
              <span class="dmeter">
                <span class="meter"><span class="fill {used / dev.total_mib > 0.9 ? 'block' : used / dev.total_mib > 0.7 ? 'warn' : 'pass'}" style="width: {Math.round(100 * used / Math.max(1, dev.total_mib))}%;"></span></span>
                <span class="mono faint num">{(dev.free_mib / 1024).toFixed(1)} of {(dev.total_mib / 1024).toFixed(0)} GiB free</span>
              </span>
              {#if entry && !isDiffusion && draft.devices.length > 1}
                <span class="frac" title="Share of the layers on this card; blank splits evenly (--tensor-split)">
                  <span class="k">share</span>
                  <input type="number" step="0.05" min="0" max="1" placeholder="auto" bind:value={entry.split_fraction} oninput={scheduleCheck} />
                </span>
              {/if}
            </label>
          {/each}
        </div>
        {#if draft.devices.length > 1 && !isDiffusion && !isSgLang}
          <div class="formgrid" style="margin-top: 14px;">
            <label class="field" style="grid-column: span 2;" title="layer puts whole layers on each card; row splits every tensor, experimental (--split-mode)"><span class="k">split mode</span>
              <select bind:value={draft.split_mode} onchange={scheduleCheck}>
                <option value="layer">layer (supported)</option>
                <option value="row">row (experimental)</option>
              </select>
            </label>
            <label class="field" style="grid-column: span 2;" title="Index, in order, of the checked card that holds the KV cache and small tensors (--main-gpu)"><span class="k">main device</span>
              <input type="number" min="0" bind:value={draft.main_device} oninput={scheduleCheck} />
            </label>
          </div>
        {/if}
        {#if budgets.length}
          <div class="budget">
            {#each budgets as b}
              <div>
                <div class="cap">
                  <span>estimate on {b.name.replace(/^AMD /, "")} <span class="faint">({b.key.split(":").pop()})</span></span>
                  <span><span class="num">{b.estGib.toFixed(2)}</span> of <span class="num">{b.freeGib.toFixed(2)}</span> GiB free · {Math.round(b.frac * 100)}%</span>
                </div>
                <div class="bar" title={b.detail}>
                  <div class="fill {b.kind}" style="width: {Math.min(100, b.frac * 100)}%;"></div>
                  <div class="mark" style="left: 90%;"></div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
        {#if sgBudget}
          <div class="budget">
            <div>
              <div class="cap">
                <span>static pool on {sgBudget.name.replace(/^AMD /, "")} <span class="faint">({sgBudget.key.split(":").pop()})</span></span>
                <span><span class="num">{sgBudget.estGib.toFixed(2)}</span> of <span class="num">{sgBudget.totalGib.toFixed(2)}</span> GiB · {Math.round(sgBudget.frac * 100)}% · <span class="num">{sgBudget.freeGib.toFixed(2)}</span> GiB free now</span>
              </div>
              <div class="bar" title="The bar is the card, the mark what is free now; the pool must fit in the free part">
                <div class="fill {sgBudget.kind}" style="width: {Math.min(100, sgBudget.frac * 100)}%;"></div>
                <div class="mark" style="left: {Math.min(100, sgBudget.freeFrac * 100)}%;"></div>
              </div>
              {#each sgBudget.assumptions as a}
                <div class="faint small" style="margin-top: 4px;">{a}</div>
              {/each}
            </div>
          </div>
        {/if}
      </section>

      <!-- context & offload -->
      <section class="card">
        <div class="sec">Context and offload <span class="faint">allocated when the server starts</span></div>
        <div class="formgrid">
          {#if isDiffusion}
            {@const predicted = dgSizing?.predicted_auto_maxtok ?? null}
            <Range bind:value={() => draft.runtime.ctx_total || null, (v) => (draft.runtime.ctx_total = v ?? 0)}
              label="context budget" min={2048} max={65536} step={256} nullable offLabel="auto"
              placeholder={predicted ?? (faSized ? 65536 : 12288)} format={fmtInt} onchange={scheduleCheck} span={3}
              hint={`auto = largest that fits VRAM at load${predicted ? ` (≈ ${fmtInt(predicted)})` : ""}; one reply uses ceil(max_tokens/${canvas}) blocks`}
              title={`Prompt plus reply in tokens; auto is the largest that fits the card at load${faSized ? ", up to 65,536 with flash attention" : ""} (MAXTOK)`} />
            <Range bind:value={draft.runtime.n_gpu_layers} label="layers on GPU" min={0} max={layerMax + 1} step={1}
              hint={header ? `≥ ${layerMax + 1} = all, incl. the output layer; runner default is 0 = CPU` : "runner default is 0 = CPU; FIDIM always sends NGL"} onchange={scheduleCheck} span={3}
              title={`Full offload needs all ${header ? layerMax : "the model's"} blocks plus the output layer${header ? ` (${layerMax + 1})` : ""}; the runner's own default is CPU (NGL)`} />
          {:else if isSgLang && draft.sglang}
            <Range bind:value={draft.runtime.ctx_total} label="context length" min={512} max={ctxMax} step={256}
              hint={header || hf ? `model supports up to ${fmtInt(ctxMax)} tokens` : "not in scan — default cap"} format={fmtInt} onchange={scheduleCheck} span={3}
              title="Longest prompt plus reply for one request; the KV pool itself is shared by all (--context-length)" />
            <Range bind:value={draft.sglang.max_running_requests} label="concurrent requests" min={1} max={64} step={1}
              hint="the router reports them as slots" onchange={scheduleCheck} span={3}
              title="Requests decoded at once, each holding pool KV and a state slot (--max-running-requests)" />
          {:else}
          <Range bind:value={draft.runtime.ctx_total} label="context length" title="Total context tokens, divided across the slots (-c)" min={512} max={ctxMax} step={256}
            hint={header ? `model supports up to ${fmtInt(ctxMax)} tokens` : "not in scan — default cap"} format={fmtInt} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.runtime.n_gpu_layers} label="layers on GPU" title="Layers held on the GPU; at or above the model's count is a full offload (-ngl)" min={0} max={layerMax} step={1}
            hint={header ? `${layerMax} layers; ≥ ${layerMax} = all` : "99 = all"} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.runtime.slots} label="request slots" title="Parallel request slots; the context divides evenly across them (-np)" min={1} max={16} step={1}
            hint={`per-slot context = ${fmtInt(Math.floor(draft.runtime.ctx_total / Math.max(1, draft.runtime.slots)))}`} onchange={scheduleCheck} span={3} />
          <label class="field" style="grid-column: span 3; justify-content: end;" title="One KV pool shared by all slots instead of a fixed share each; experimental (-kvu)">
            <span class="k">unified cache</span>
            <span><input type="checkbox" bind:checked={draft.runtime.kv_unified} onchange={scheduleCheck} /> on</span>
          </label>
          {/if}
        </div>
      </section>

      {#if isDiffusion && draft.diffusion}
        {@const maxTok = Number(draft.diffusion.default_max_tokens) || 0}
        <!-- diffusion -->
        <section class="card">
          <div class="sec">Diffusion <span class="faint">runner settings; sampling and batching do not apply</span></div>
          <div class="formgrid">
            <Range bind:value={draft.diffusion.default_max_tokens} label="reply budget" min={256} max={8192} step={256}
              hint={`= ${Math.ceil(maxTok / canvas)} blocks; a client's max_tokens overrides; the thought channel uses blocks too`}
              format={fmtInt} onchange={scheduleCheck} span={3}
              title={`Reply tokens when a request sends none, spent in whole ${canvas}-token blocks together with the thinking (max_tokens)`} />
            <label class="field" style="grid-column: span 3;" title="Seed for requests that send none; empty is random per request">
              <span class="k">seed</span>
              <input type="number" step="1" placeholder="random per request" value={draft.diffusion.seed ?? ""}
                oninput={(e) => { draft.diffusion.seed = e.target.value === "" ? null : e.target.value; scheduleCheck(); }} />
            </label>
            <label class="field" style="grid-column: span 3;" title="Keeps rocBLAS off hipBLASLt at no cost (ROCBLAS_USE_HIPBLASLT=0). Off, the first denoise step sometimes fails.">
              <span class="k">hipBLASLt safeguard</span>
              <span>
                <input type="checkbox" bind:checked={draft.diffusion.hipblaslt_safeguard} onchange={scheduleCheck} /> on
                {#if !draft.diffusion.hipblaslt_safeguard}<span class="chip warn" title="Without the safeguard the first denoise step intermittently fails with MUL_MAT / invalid argument">unsafe</span>{/if}
              </span>
            </label>
            <label class="field" style="grid-column: span 3;" title={faWorks
              ? `This patched build runs the 512-dim heads on the GPU${faTurnSizing ? " and sizes the context by working set" : ""} (FA=1)`
              : "On this build the 512-dim heads fall back to the CPU; a patched runner from Updates fixes that (FA=1)"}>
              <span class="k">flash attention</span>
              <span>
                <input type="checkbox" bind:checked={draft.diffusion.flash_attn} onchange={scheduleCheck} /> on
                {#if draft.diffusion.flash_attn && !faWorks}<span class="chip warn" title="The 512-dim heads have no flash kernel on this build and fall back to the CPU">slower</span>
                {:else if !draft.diffusion.flash_attn && faWorks}<span class="chip accent" title="This build pads keys so flash attention runs the heads on the GPU">recommended</span>{/if}
              </span>
            </label>
          </div>
          {#if dgEnvConflicts.length}
            <div class="notice" style="margin-top: 12px;" title="Llama FIDIM sets these itself for every diffusion run; a profile env entry would override them">
              <span class="chip block">env conflict</span>
              <span class="mono">{dgEnvConflicts.join(", ")}</span>
              <button class="btn small" onclick={dropEnvConflicts}>Remove</button>
            </div>
          {/if}
          <div class="faint small" style="margin-top: 10px;" title="Also sends GPU_RESOURCE_CACHE_SIZE=0 so freed memory is returned; a profile env entry overrides it">
            Reasoning arrives as <span class="mono">reasoning_content</span>, tool calls as text; no keep-alive, router or benchmarks.
          </div>
        </section>
      {/if}

      {#if isSgLang && draft.sglang}
        <!-- sglang -->
        <section class="card">
          <div class="sec">SGLang <span class="faint">server flags; sampling comes from each request</span></div>
          <div class="formgrid">
            <Range bind:value={draft.sglang.mem_fraction} label="VRAM share" min={0.3} max={0.98} step={0.01}
              hint={sgDevice?.display ? "display on this card: keep ≥ 1.5 GB free (0.85 on 32 GB)" : "weights + KV pool"}
              format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3}
              title="Share of the card SGLang reserves for weights and cache; leave 1.5 GB when the card drives a display (--mem-fraction-static)" />
            <Range bind:value={draft.sglang.chunked_prefill} label="prompt chunk" min={512} max={32768} step={512}
              hint="tokens per forward pass" format={fmtInt} onchange={scheduleCheck} span={3}
              title="Prompt tokens per prefill pass; larger is faster but takes more VRAM (--chunked-prefill-size)" />
            <Range bind:value={draft.sglang.mamba_slots} label="state slots" min={1} max={256} step={1} nullable offLabel="auto"
              placeholder={Math.max(1, Number(draft.sglang.max_running_requests) || 1) * 5}
              hint="hybrid (GatedDeltaNet) models; auto = 5 × concurrent requests" onchange={scheduleCheck} span={3}
              title="Recurrent-state slots on a hybrid model, one per running or queued request (--max-mamba-cache-size)" />
            <Range bind:value={draft.sglang.memguard_gb} label="host RAM guard" min={0} max={128} step={1}
              hint="GB per process" onchange={scheduleCheck} span={3}
              title="Aborts the server when its host RSS passes this GB, ahead of the OOM killer (memguard). A GGUF needs file size plus margin." />
            <label class="field" style="grid-column: span 2;" title="KV pool storage type; fp8_e4m3 halves it against bf16, fp4_mx_block16 halves it again and needs the gfx1201 kernel patch (--kv-cache-dtype)">
              <span class="k">cache precision</span>
              <select bind:value={draft.sglang.kv_cache_dtype} onchange={scheduleCheck}>
                {#each SG_KV_DTYPES as t}<option value={t}>{t}</option>{/each}
                {#if !SG_KV_DTYPES.includes(draft.sglang.kv_cache_dtype)}<option value={draft.sglang.kv_cache_dtype}>{draft.sglang.kv_cache_dtype}</option>{/if}
              </select>
            </label>
            <label class="field" style="grid-column: span 2;" title="triton is the one that runs on gfx1201; the others are for other cards (--attention-backend)">
              <span class="k">attention kernel</span>
              <select bind:value={draft.sglang.attention_backend} onchange={scheduleCheck}>
                {#each SG_ATTN_BACKENDS as t}<option value={t}>{t}</option>{/each}
                {#if !SG_ATTN_BACKENDS.includes(draft.sglang.attention_backend)}<option value={draft.sglang.attention_backend}>{draft.sglang.attention_backend}</option>{/if}
              </select>
            </label>
            <label class="field" style="grid-column: span 2;" title="Type for unquantised weights and activations; bfloat16 is safe, float16 can overflow (--dtype)">
              <span class="k">compute precision</span>
              <select bind:value={draft.sglang.dtype} onchange={scheduleCheck}>
                {#each SG_DTYPES as t}<option value={t}>{t}</option>{/each}
                {#if !SG_DTYPES.includes(draft.sglang.dtype)}<option value={draft.sglang.dtype}>{draft.sglang.dtype}</option>{/if}
              </select>
            </label>
            <label class="field" style="grid-column: span 3;" title="Splits thinking into reasoning_content; other… takes any name SGLang knows (--reasoning-parser)">
              <span class="k">reasoning parser</span>
              <select value={parserSel("reasoning_parser", SG_REASONING_PARSERS)} onchange={(e) => onParserPick("reasoning_parser", e.target.value, SG_REASONING_PARSERS)}>
                <option value="">none</option>
                {#each SG_REASONING_PARSERS as v}<option value={v}>{v}</option>{/each}
                <option value="__custom">other…</option>
              </select>
              {#if parserSel("reasoning_parser", SG_REASONING_PARSERS) === "__custom"}
                <input bind:value={draft.sglang.reasoning_parser} oninput={scheduleCheck} placeholder="parser name as SGLang spells it" />
              {/if}
            </label>
            <label class="field" style="grid-column: span 3;" title="Turns tool-call markup into tool_calls; other… takes any name SGLang knows (--tool-call-parser)">
              <span class="k">tool call parser</span>
              <select value={parserSel("tool_call_parser", SG_TOOL_PARSERS)} onchange={(e) => onParserPick("tool_call_parser", e.target.value, SG_TOOL_PARSERS)}>
                <option value="">none</option>
                {#each SG_TOOL_PARSERS as v}<option value={v}>{v}</option>{/each}
                <option value="__custom">other…</option>
              </select>
              {#if parserSel("tool_call_parser", SG_TOOL_PARSERS) === "__custom"}
                <input bind:value={draft.sglang.tool_call_parser} oninput={scheduleCheck} placeholder="parser name as SGLang spells it" />
              {/if}
            </label>
            <Range bind:value={draft.sglang.compile_threads} label="compile workers" min={1} max={32} step={1}
              hint="a small pool only slows the first launch" onchange={scheduleCheck} span={3}
              title="Kernel compile processes, each a torch import in host RAM (TORCHINDUCTOR_COMPILE_THREADS)" />
            <div style="grid-column: span 3;"></div>
            <label class="field" style="grid-column: span 3;" title="Prometheus counters on /metrics for the Running tab and the router; leave on (--enable-metrics)">
              <span class="k">metrics endpoint</span>
              <span><input type="checkbox" bind:checked={draft.sglang.enable_metrics} onchange={scheduleCheck} /> on</span>
            </label>
            <label class="field" style="grid-column: span 3;" title="Blocks in a socket poll between requests instead of spinning a full CPU core (--sleep-on-idle)">
              <span class="k">sleep when idle</span>
              <span><input type="checkbox" bind:checked={draft.sglang.sleep_on_idle} onchange={scheduleCheck} /> on</span>
            </label>
            <label class="field" style="grid-column: 1 / -1;" title="More model names the router maps to this server, comma-separated">
              <span class="k">extra names</span>
              <input value={Array.isArray(draft.sglang.aliases) ? draft.sglang.aliases.join(", ") : draft.sglang.aliases}
                oninput={(e) => { draft.sglang.aliases = e.target.value; scheduleCheck(); }} placeholder="ddg, qwen-daily" />
            </label>
            <label class="field" style="grid-column: 1 / -1;" title="Passed to sglang.launch_server unchanged, one per line; a flag and its value are two lines">
              <span class="k">extra arguments</span>
              <textarea rows="3" value={Array.isArray(draft.sglang.extra_args) ? draft.sglang.extra_args.join("\n") : draft.sglang.extra_args}
                oninput={(e) => { draft.sglang.extra_args = e.target.value; scheduleCheck(); }} placeholder={"--disable-radix-cache\n--schedule-policy\nfcfs"}></textarea>
            </label>
            <label class="field" style="grid-column: 1 / -1;" title="KEY=VALUE per line for the server process, on top of what Settings sets for every run">
              <span class="k">environment</span>
              <textarea rows="3" value={sgEnvText} oninput={(e) => onEnvText(e.target.value)} placeholder={"SGLANG_TORCH_PROFILER_DIR=/tmp/prof\nHSA_OVERRIDE_GFX_VERSION=12.0.1"}></textarea>
            </label>
          </div>
        </section>

        <!-- sglang speculative decoding -->
        <section class="card">
          <div class="sec">
            Speculative decoding
            {#if mtpBuiltIn}<span class="chip pass" title="The checkpoint has its own draft head (next-token predict layers)">MTP</span>{/if}
          </div>
          <div class="formgrid">
            <label class="field" style="grid-column: span 3;" title="On sends --speculative-algorithm with the settings below">
              <span class="k">enabled</span>
              <span><input type="checkbox" checked={!!draft.sglang.spec} onchange={(e) => onSgSpecToggle(e.target.checked)} /> {draft.sglang.spec ? "on" : "off"}</span>
            </label>
            {#if draft.sglang.spec}
              <label class="field" style="grid-column: span 3;" title="NEXTN uses the model's own draft head, EAGLE a draft model from extra arguments, NGRAM prompt lookup">
                <span class="k">algorithm</span>
                <select bind:value={draft.sglang.spec.algorithm} onchange={scheduleCheck}>
                  {#each SG_SPEC_ALGOS as a}<option value={a}>{a === "NEXTN" ? "the model's own draft head (NEXTN)" : a === "NGRAM" ? "prompt lookup, no model (NGRAM)" : a}</option>{/each}
                  {#if !SG_SPEC_ALGOS.includes(draft.sglang.spec.algorithm)}<option value={draft.sglang.spec.algorithm}>{draft.sglang.spec.algorithm}</option>{/if}
                </select>
              </label>
              <Range bind:value={draft.sglang.spec.steps} label="draft steps" min={1} max={8} step={1} hint="SGLang default 3" onchange={scheduleCheck} span={2}
                title="Draft head runs per verify pass (--speculative-num-steps)" />
              <Range bind:value={draft.sglang.spec.topk} label="draft top-k" min={1} max={8} step={1} hint="SGLang default 1" onchange={scheduleCheck} span={2}
                title="Branches kept per step; 1 is what an MTP head is trained for (--speculative-eagle-topk)" />
              <Range bind:value={draft.sglang.spec.draft_tokens} label="draft tokens" min={1} max={16} step={1} hint="SGLang default 4" onchange={scheduleCheck} span={2}
                title="Tokens verified per pass; more is faster if accepted, wasted if not (--speculative-num-draft-tokens)" />
              <label class="field" style="grid-column: 1 / -1;" title="Torch-saved id tensor limiting the draft head to hot tokens; empty is full (--speculative-token-map)">
                <span class="k">hot-token map</span>
                <input bind:value={draft.sglang.spec.token_map} oninput={scheduleCheck} placeholder="/models/hot_tokens.pt" />
              </label>
            {/if}
          </div>
          {#if draft.sglang.spec?.algorithm === "NEXTN" && header && !mtpBuiltIn}
            <div class="notice" style="margin-top: 10px;"><span class="chip warn" title="NEXTN needs a checkpoint with next-token predict layers; this header reports none">no head</span><span>pick another algorithm</span></div>
          {/if}
        </section>
      {/if}

      {#if !isDiffusion && !isSgLang}
      <!-- speculative decoding -->
      <section class="card">
        <div class="sec">
          Speculative decoding
          {#if mtpBuiltIn}<span class="chip pass" title="The model has its own draft head (next-token predict layers)">MTP</span>{/if}
          <span style="margin-left: auto;"><button class="btn small" onclick={applySpecDefaults} disabled={draft.speculative.mode === "off"} title="Sets llama-server's own defaults: 3 max, 0 min, 0.00 confidence">Engine defaults</button></span>
        </div>
        <div class="formgrid">
          <label class="field" style="grid-column: span 3;" title="MTP uses the model's own head, draft a small model, DFlash a DFlash file, n-gram prompt lookup">
            <span class="k">mode</span>
            <select value={draft.speculative.mode} onchange={(e) => onSpecMode(e.target.value)}>
              {#each specModes as m}<option value={m.v}>{m.l}</option>{/each}
            </select>
          </label>
          {#if draft.speculative.mode !== "off"}
            <div style="grid-column: span 3;"></div>
            <Range bind:value={draft.speculative.n_max} label="max draft tokens" min={1} max={16} step={1} nullable placeholder={3}
              hint="engine default 3" onchange={scheduleCheck} span={3}
              title="Tokens proposed per step; more is faster when accepted, wasted when rejected (--spec-draft-n-max)" />
            <Range bind:value={draft.speculative.n_min} label="min draft tokens" min={0} max={16} step={1} nullable placeholder={0}
              hint="engine default 0" onchange={scheduleCheck} span={3}
              title="Skip speculation when fewer draft tokens than this are available (--spec-draft-n-min)" />
            <Range bind:value={draft.speculative.p_min} label="draft confidence" min={0} max={1} step={0.01} nullable placeholder={0}
              hint="engine default 0.00" format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3}
              title="Keep only draft tokens at least this likely; 0 keeps all (--spec-draft-p-min)" />
          {/if}
        </div>
        {#if draft.speculative.mode === "mtp" && !mtpBuiltIn && !draft.model.draft?.path}
          <div class="notice" style="margin-top: 10px;"><span class="chip block" title="MTP needs a built-in head or an MTP sidecar in the draft file above">no head</span><span>pick an MTP sidecar as the draft file</span></div>
        {/if}
      </section>

      <!-- inference -->
      <section class="card">
        <div class="sec">
          Inference
          <span class="faint">unchecked = the engine's own default</span>
          <span style="margin-left: auto; display: flex; gap: 8px;">
            <button class="btn small" onclick={fetchCreator} disabled={creatorBusy || !draft.model.path} title="Looks up generation_config.json on Hugging Face">
              {creatorBusy ? "Fetching…" : embedded ? "Re-check on Hugging Face" : "Creator defaults"}
            </button>
            {#if creatorShown}
              <button class="btn small primary" onclick={applyCreator} title="Copies the creator defaults into the fields below">Apply</button>
            {/if}
          </span>
        </div>
        {#if creatorShown || creator?.error}
          <div class="creator">
            {#if creator?.error}
              <div class="notice"><span class="chip block">hugging face</span><span class="mono">{creator.error}</span></div>
            {/if}
            {#if creatorShown}
              <div class="facts">
                <span class="chip accent">{creatorShown.repo}</span>
                <span>temperature <b class="num">{creatorShown.temperature ?? "—"}</b></span>
                <span>top_p <b class="num">{creatorShown.top_p ?? "—"}</b></span>
                <span>top_k <b class="num">{creatorShown.top_k ?? "—"}</b></span>
                <span>min_p <b class="num">{creatorShown.min_p ?? "—"}</b></span>
                <span>repetition_penalty <b class="num">{creatorShown.repetition_penalty ?? "—"}</b></span>
                <span class="faint">{creatorShown.embedded ? "from the GGUF header" : (creatorShown.from_cache ? "cached" : "fetched") + " from generation_config.json"}</span>
              </div>
            {/if}
          </div>
        {/if}
        <div class="formgrid">
          <label class="field" style="grid-column: span 3;" title="Sets enable_thinking in the chat template; off avoids the agentic loops some models fall into"><span class="k">thinking</span>
            <select bind:value={draft.chat.enable_thinking} onchange={scheduleCheck}>
              <option value={null}>model default</option>
              <option value={true}>on</option>
              <option value={false}>off</option>
            </select>
          </label>
          <div style="grid-column: span 3;"></div>
          <Range bind:value={draft.sampling.temperature} label="temperature" title="Randomness; 0 is greedy, 1 the model's raw distribution (--temp)" min={0} max={2} step={0.05} nullable placeholder={creatorShown?.temperature ?? 0.8}
            hint={creatorShown?.temperature != null ? `creator: ${creatorShown.temperature}` : ""} format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.top_k} label="top K" title="Keep only the K most likely tokens; 0 is off (--top-k)" min={0} max={200} step={1} nullable placeholder={creatorShown?.top_k ?? 40}
            hint={creatorShown?.top_k != null ? `creator: ${creatorShown.top_k}` : "0 = off"} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.top_p} label="top P" title="Keep the smallest set of tokens whose probabilities sum to P (--top-p)" min={0} max={1} step={0.01} nullable placeholder={creatorShown?.top_p ?? 0.95}
            hint={creatorShown?.top_p != null ? `creator: ${creatorShown.top_p}` : ""} format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.min_p} label="min P" title="Drop tokens below P × the top token's probability; steadier than top P when hot (--min-p)" min={0} max={1} step={0.01} nullable placeholder={creatorShown?.min_p ?? 0.05}
            hint={creatorShown?.min_p != null ? `creator: ${creatorShown.min_p}` : ""} format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.repeat_penalty} label="repeat penalty" title="Penalty on tokens already in the context; 1.0 is off (--repeat-penalty)" min={1} max={2} step={0.01} nullable placeholder={creatorShown?.repetition_penalty ?? 1.0}
            hint={creatorShown?.repetition_penalty != null ? `creator: ${creatorShown.repetition_penalty}` : "1.0 = off"} format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.presence_penalty} label="presence penalty" title="Flat penalty on any token seen before; 0 is off (--presence-penalty)" min={0} max={2} step={0.05} nullable placeholder={0}
            hint="0 = off" format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.dry_multiplier} label="DRY multiplier" title="Penalises repeating whole sequences, not single tokens; 0 is off (--dry-multiplier)" min={0} max={2} step={0.05} nullable placeholder={0}
            hint="0 = off" format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
        </div>
      </section>

      <!-- advanced -->
      <section class="card">
        <div class="sec">Advanced <span class="faint">batching, KV storage and pass-through flags</span></div>
        <div class="formgrid">
          <Range bind:value={draft.runtime.batch_logical} label="batch" title="Logical batch, the per-iteration token budget shared by prefill and decode (-b)" min={64} max={8192} step={64}
            hint="tokens per iteration" format={fmtInt} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.runtime.batch_physical} label="micro-batch" title="Tokens per GPU pass, which sizes the compute buffer; 256 measured best at long context (-ub)" min={32} max={2048} step={32}
            hint="sizes the compute buffer" format={fmtInt} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.runtime.cache_reuse} label="prompt cache reuse" title="Reuse KV for a shared prompt prefix, in chunks at least this long; 0 is off (--cache-reuse)" min={0} max={2048} step={32} nullable placeholder={256}
            hint="min chunk, tokens" onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.runtime.threads} label="CPU threads" title="Threads for layers left on the CPU and for tokenising (-t)" min={1} max={32} step={1} nullable placeholder={8}
            hint="only matters for layers left on CPU" onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.keep_alive_seconds} label="keep-alive" min={0} max={30} step={1} nullable placeholder={keepAliveDefault}
            hint={`seconds · config default ${keepAliveDefault} · 0 = off`} onchange={scheduleCheck} span={3}
            title="Seconds between 1-token pings keeping the GPU awake when PCIe power saving cannot be off (≈70 ms each)" />
          <div style="grid-column: span 3;"></div>
          <label class="field" style="grid-column: span 2;" title="Fused attention kernel: less VRAM, faster prefill, required for V-cache quantisation (-fa)"><span class="k">flash attention</span>
            <select bind:value={draft.runtime.flash_attn} onchange={scheduleCheck}>
              <option value="on">on</option><option value="off">off</option><option value="auto">auto</option>
            </select>
          </label>
          <label class="field" title="Storage type of the key cache; q8_0 halves its VRAM at little cost (-ctk)"><span class="k">K cache</span>
            <select bind:value={draft.runtime.kv_type_k} onchange={scheduleCheck}>
              {#each ["f16", "q8_0", "q4_0"] as t}<option value={t}>{t}</option>{/each}
            </select>
          </label>
          <label class="field" title="Storage type of the value cache; anything but f16 needs flash attention on (-ctv)"><span class="k">V cache</span>
            <select bind:value={draft.runtime.kv_type_v} onchange={scheduleCheck}>
              {#each ["f16", "q8_0", "q4_0"] as t}<option value={t}>{t}</option>{/each}
            </select>
          </label>
          <label class="field" style="grid-column: span 2;" title="Serves concurrent requests in one forward pass instead of queueing them; leave on (-cb)"><span class="k">continuous batching</span>
            <span><input type="checkbox" bind:checked={draft.runtime.cont_batching} onchange={scheduleCheck} /> on</span>
          </label>
          <label class="field" style="grid-column: span 4;" title="Puts each slot's prompt and text on the Running tab, with loop detection (LLAMA_SERVER_SLOTS_DEBUG=1). One detokenize per poll."><span class="k">trace tokens</span>
            <span><input type="checkbox" checked={draft.env?.LLAMA_SERVER_SLOTS_DEBUG === "1"} onchange={(e) => { if (e.target.checked) draft.env.LLAMA_SERVER_SLOTS_DEBUG = "1"; else delete draft.env.LLAMA_SERVER_SLOTS_DEBUG; scheduleCheck(); }} /> on</span>
          </label>
          <label class="field" style="grid-column: 1 / -1;" title="Passed to llama-server unchanged, space-separated, e.g. --no-mmap"><span class="k">extra flags</span>
            <input value={Array.isArray(draft.runtime.extra_flags) ? draft.runtime.extra_flags.join(" ") : draft.runtime.extra_flags}
              oninput={(e) => { draft.runtime.extra_flags = e.target.value; scheduleCheck(); }} placeholder="--no-mmap" />
          </label>
        </div>
      </section>
      {/if}

      <!-- notes -->
      <section class="card">
        <div class="sec">Notes</div>
        <textarea rows="3" bind:value={draft.notes} placeholder="What was measured, what to remember."></textarea>
      </section>

      {#if check?.findings?.length}
        <section class="card" transition:slide={leave}>
          <div class="sec">Findings <span class="faint">in the profile file</span></div>
          {#each check.findings as f}
            <div class="notice" style="padding: 4px 0;">
              <span class="chip {f.severity === 'error' ? 'block' : 'warn'}">{f.severity}</span>
              <span>{f.message}</span>
            </div>
          {/each}
        </section>
      {/if}

      <!-- live pre-flight -->
      <section class="card">
        <div class="sec">
          Pre-flight
          <span class="faint">re-run on every edit</span>
          {#if checking}<span class="chip plain live" style="margin-left: auto;">re-checking</span>{/if}
          {#if check?.error}<span class="chip block">{check.error}</span>{/if}
        </div>
        {#if check?.results}
          <div class="preflight">
            {#each check.results as r (r.id)}
              {@const kind = outcomeKind(r.outcome)}
              {@const msg = outcomeMsg(r.outcome)}
              <div class="row" class:pulse-once={changed.has(r.id)}>
                <span class="n">{r.spec_number}</span>
                <span class="t">{r.title}</span>
                <span class="chip {kind}">{kind}</span>
                {#if msg}<span class="msg">{msg}</span>{/if}
              </div>
            {/each}
          </div>
          {#if check.command_line}
            <div class="cmd">
              <div class="k">command</div>
              <div class="path">{check.command_line}</div>
              <div class="k" style="margin-top: 8px;">environment</div>
              <div class="path">{(check.env ?? []).map(([k, v]) => `${k}=${v}`).join("  ")}</div>
            </div>
          {/if}
        {:else if !checking}
          <div class="empty">Edit any field to run pre-flight</div>
        {/if}
      </section>

      {#if draft.baseline}
        <section class="card">
          <div class="sec">Last baseline <span class="faint">{draft.baseline.measured_at} · driver {draft.baseline.driver} · {draft.baseline.sdk}</span>{#if draft.baseline.cold_cache}<span class="chip warn">cold cache</span>{/if}</div>
          <div class="stats">
            <div class="stat"><span class="v">{draft.baseline.serial_tok_s}<small>tok/s</small></span><span class="l">serial decode</span></div>
            {#if draft.baseline.concurrent}
              <div class="stat"><span class="v">{draft.baseline.concurrent.decode_aggregate_tok_s ?? draft.baseline.concurrent.aggregate_tok_s}<small>tok/s</small></span><span class="l">aggregate · n = {draft.baseline.concurrent.n}</span></div>
              <div class="stat"><span class="v">{draft.baseline.concurrent.per_stream_tok_s}<small>tok/s</small></span><span class="l">per stream</span></div>
            {/if}
            <div class="stat"><span class="v">{draft.baseline.vram_gb}<small>GiB</small></span><span class="l">resident</span></div>
          </div>
        </section>
      {/if}
    </div>
  {:else}
    <div class="card" style="flex: 1;"><div class="empty">Pick a profile on the left, or New</div></div>
  {/if}
</div>

{#if toast}
  <div class="toast" class:error={toast.isError} transition:fly={toastFly}>{toast.text}</div>
{/if}

<style>
  .pf { display: flex; gap: 20px; align-items: flex-start; }
  .list { width: 280px; flex: none; position: sticky; top: 0; max-height: calc(100vh - 32px); overflow: auto; }
  .rows { display: flex; flex-direction: column; gap: 6px; }
  .row {
    all: unset; cursor: pointer; display: flex; flex-direction: column; gap: 4px; padding: 10px 12px;
    background: var(--ground-raised); border: 1px solid var(--rule); border-radius: var(--radius-sm); font-family: var(--sans);
  }
  .row:hover { border-color: var(--rule-strong); background: var(--ground-inset); }
  .row.active { border-color: var(--accent-line); background: var(--accent-soft); }
  .row:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
  .row .r1 { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
  .row .id { font-family: var(--mono); font-weight: 700; font-size: 13.5px; color: var(--ink); }
  .row.active .id { color: var(--accent); }
  .row .r2 { font-size: 12px; color: var(--ink-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .row .r3 { display: flex; gap: 10px; align-items: center; font-size: 11.5px; color: var(--ink-faint); flex-wrap: wrap; }
  .row .tok { color: var(--ink-muted); }

  .editor { flex: 1; min-width: 0; }
  .editbar {
    position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    background: var(--ground); padding: 6px 0 12px; margin-bottom: 4px; border-bottom: 1px solid var(--rule);
  }
  .editbar .who { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .editbar .name { font-family: var(--display); font-weight: 700; font-size: 17px; letter-spacing: -0.01em; }
  .editbar .who .mono { font-size: 11.5px; }
  .editbar .meta { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .pfsum { display: inline-flex; gap: 6px; align-items: center; }
  .actions { display: flex; gap: 8px; margin-left: auto; flex-wrap: wrap; }
  .editor section.card { padding: 18px 20px; }
  .editor :global(.formgrid) { gap: 16px 20px; }

  .facts { display: flex; gap: 6px 18px; flex-wrap: wrap; font-size: 12.5px; color: var(--ink-muted); align-items: center; }
  .facts b { color: var(--ink); font-weight: 600; }
  .creator { margin-bottom: 14px; padding: 10px 12px; background: var(--ground-inset); border-radius: 6px; }

  .devrows { display: flex; flex-direction: column; gap: 8px; }
  .devrow {
    display: grid; grid-template-columns: auto 1fr 240px auto; gap: 16px; align-items: center;
    padding: 10px 14px; border: 1px solid var(--rule); border-radius: var(--radius-sm); background: var(--ground-inset); cursor: pointer;
  }
  .devrow.on { border-color: var(--accent-line); background: var(--accent-soft); }
  .devrow .dname { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; font-size: 13px; }
  .devrow .dmeter { display: flex; flex-direction: column; gap: 4px; }
  .devrow .meter { display: block; }
  .devrow .meter .fill { display: block; }
  .devrow .frac { display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: var(--ink-muted); }
  .devrow .frac input { width: 84px; }

  .cmd { margin-top: 12px; }
  .cmd .k { font-size: 11.5px; font-weight: 600; color: var(--ink-muted); margin-bottom: 2px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px 20px; }
</style>
