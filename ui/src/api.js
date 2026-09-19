// Command bridge. Inside Tauri every call hits the Rust command layer
// (fidim-core). In a plain browser (vite dev without Tauri) a mock layer
// answers with the real target machine's topology so the UI can be exercised
// visually without hardware access.

let invoke = null;
export const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
if (inTauri) {
  const mod = await import("@tauri-apps/api/core");
  invoke = mod.invoke;
}


// A DiffusionGemma run for the browser preview: a haiku block that resolves
// out of noise over 16 steps (one step per 200 ms), after a committed
// thought. mockDgFrames() is the same reply as the helper's /frames.
const DG_THOUGHT = "<|channel>thought\nThe user wants a haiku about GPUs. Five, seven, five syllables.<channel|>";
const DG_ANSWER = "Silicon hums warm,\nthousand small cores think as one,\nthe fan sings them home.";
const DG_STEPS = 16;
function dgNoisy(text, step, total) {
  const junk = [",", ",", " ", "\n", "*", ":", "the", "the", "."];
  let out = "";
  for (let i = 0; i < text.length; i++) {
    const settled = ((i * 7919) % 97) / 97 < (step + 1) / total;
    out += settled || text[i] === "\n" ? text[i] : junk[(i * 31 + step * 7) % junk.length];
  }
  return out;
}
function mockDgFrames() {
  const frames = [];
  for (let st = 0; st < DG_STEPS; st++) frames.push({ b: 0, s: st, t: 48, x: dgNoisy(DG_THOUGHT, st, DG_STEPS) });
  for (let st = 0; st < DG_STEPS; st++) frames.push({ b: 1, s: st, t: 48, x: dgNoisy(DG_ANSWER, st, DG_STEPS) });
  return { id_task: 12, canvas: 256, dropped: 0, frames };
}
function mockDiffusionSample(t) {
  const step = Math.floor(t / 200) % (DG_STEPS + 4);   // a short pause between replies
  const st = Math.min(step, DG_STEPS - 1);
  const draft = step < DG_STEPS ? dgNoisy(DG_ANSWER, st, DG_STEPS) : "";
  const processing = step < DG_STEPS;
  const generated = DG_THOUGHT + (processing ? draft : DG_ANSWER);
  return {
    model: null, sampled_unix_ms: t, phase: processing ? "decode" : "idle",
    slots: [{
      id: 0, id_task: 12, n_ctx: 65536, is_processing: processing, phase: processing ? "decode" : "idle",
      n_prompt_tokens: 18, n_prompt_tokens_processed: 18, n_prompt_tokens_cache: 0,
      n_decoded: processing ? 256 + Math.round(256 * (st + 1) / 48) : 0, n_remain: processing ? 256 - Math.round(256 * (st + 1) / 48) : 0,
      prefill_fraction: 1, ctx_fraction: 0.01,
      prompt: "[user]\nWrite a haiku about GPUs.", prompt_chars: 33,
      generated, generated_chars: generated.length, committed_chars: DG_THOUGHT.length + (processing ? 0 : DG_ANSWER.length), loop_hint: null,
      diffusion: { block: processing ? 1 : 1, n_blocks: 2, step: st, total: 48, steps_done: DG_STEPS + st + 1, canvas: 256, state: processing ? "denoise" : "idle" },
    }],
    metrics: { tokens_predicted_total: 512, prompt_tokens_total: 18, requests_processing: processing ? 1 : 0, requests_deferred: 0, predicted_tokens_seconds: 38.4, diffusion_canvas_tokens_seconds: 1114 },
    error: null,
  };
}

export async function api(cmd, args = {}) {
  if (!invoke) return mock(cmd, args);
  try {
    return await invoke(cmd, args);
  } catch (e) {
    // Surface every failed command in ~/.fidim/ui.log so problems inside
    // the web view can be diagnosed from outside it.
    if (cmd !== "ui_log") log(`command ${cmd} failed: ${String(e)}`);
    throw e;
  }
}

export function log(line) {
  if (!invoke) { console.log(line); return; }
  invoke("ui_log", { line }).catch(() => {});
}

/// Subscribe to a Tauri event; returns an unlisten function. No-op outside Tauri.
export async function onEvent(name, cb) {
  if (!inTauri) return () => {};
  const { listen } = await import("@tauri-apps/api/event");
  return listen(name, (e) => cb(e.payload));
}

// ------------------------------------------------------------------ mocks ----

const MOCK_DEVICES = [
  {
    device: {
      stable_key: "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus03",
      name: "AMD Radeon AI PRO R9700", hip_index: 0, backend: "ROCm",
      total_mib: 32624, free_mib: 32472, integrated: false, bus_number: 3,
      driver_version: "32.0.31035.1003",
      display: { width: 2560, height: 1440, refresh_hz: 144 },
      luid_low: 90652, correlation_assumed: true,
    },
    occupied_by: [],
  },
  {
    device: {
      stable_key: "pci:VEN_1002&DEV_13C0&SUBSYS_7D781462:bus19",
      name: "AMD Radeon(TM) Graphics", hip_index: 1, backend: "ROCm",
      total_mib: 12381, free_mib: 12099, integrated: true, bus_number: 19,
      driver_version: "32.0.21045.1000",
      display: { width: 1920, height: 1080, refresh_hz: 59 },
      luid_low: 122348, correlation_assumed: false,
    },
    occupied_by: [],
  },
  {
    device: {
      stable_key: "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08",
      name: "AMD Radeon AI PRO R9700", hip_index: 2, backend: "ROCm",
      total_mib: 32624, free_mib: 32472, integrated: false, bus_number: 8,
      driver_version: "32.0.31035.1003",
      display: { width: 1920, height: 1080, refresh_hz: 59 },
      luid_low: 112018, correlation_assumed: true,
    },
    occupied_by: ["worker-pool"],
  },
];

const MOCK_PROFILE = {
  schema: 1, id: "worker-pool", name: "Subagent worker pool",
  build: { path: "D:\\llama.cpp\\build-hip-vision", version: "b9817" },
  model: {
    path: "D:\\models\\unsloth\\gemma-4-26B-A4B-it-qat-GGUF\\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    mmproj: null,
    draft: { path: "D:\\models\\...\\MTP\\mtp-gemma-4-26B-A4B-it-Q8_0.gguf", enabled: false },
  },
  devices: [{ key: "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08", split_fraction: null, resolved_index_last_launch: 2 }],
  split_mode: null, main_device: 0,
  server: { port: 9701, alias: "gemma-4-worker", host: "127.0.0.1" },
  runtime: {
    n_gpu_layers: 99, ctx_total: 393216, slots: 6, kv_type_k: "q8_0", kv_type_v: "q8_0",
    flash_attn: "on", batch_logical: 2048, batch_physical: 256, cont_batching: true,
    kv_unified: false, cache_reuse: null, extra_flags: [],
  },
  sampling: { temperature: 1.0, top_p: 0.95, top_k: 64, dry_multiplier: 0.8 },
  chat: { enable_thinking: false },
  env: { GPU_MAX_HW_QUEUES: "1", ROCBLAS_USE_HIPBLASLT: "0" },
  baseline: {
    measured_at: "2026-07-31T15:49:16Z", driver: "32.0.31035.1003", sdk: "7.1.51803-d3a86bd04",
    split: null, vram_gb: 19.87, serial_tok_s: 90.1,
    concurrent: { n: 6, aggregate_tok_s: 261.5, decode_aggregate_tok_s: 293.8, per_stream_tok_s: 49.0 },
    cold_cache: false,
  },
  notes: "Knee is at 6 slots; 6->8 buys +2.4% aggregate for -23% per-stream.",
};

const MOCK_SPLIT_PROFILE = {
  ...MOCK_PROFILE,
  id: "qwen-split", name: "Qwen3.6-35B Q4 split across both R9700s",
  model: { path: "D:\\models\\lmstudio-community\\Qwen3.6-35B-A3B-GGUF\\Qwen3.6-35B-A3B-Q4_K_M.gguf", mmproj: null, draft: null },
  devices: [
    { key: "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus03", split_fraction: 0.5, resolved_index_last_launch: 0 },
    { key: "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08", split_fraction: 0.5, resolved_index_last_launch: 2 },
  ],
  split_mode: "layer",
  server: { port: 9702, alias: "qwen-split", host: "127.0.0.1" },
  runtime: { ...MOCK_PROFILE.runtime, ctx_total: 65536, slots: 1, batch_physical: 512 },
  baseline: null,
  notes: "R-14 acceptance profile.",
};

function mockCheckResults(p) {
  const multi = (p?.devices?.length ?? 1) > 1;
  const results = [
    { id: "build-runs", spec_number: 1, title: "Build binary exists and runs", outcome: "pass" },
    { id: "files-exist", spec_number: 2, title: "Model and paired mmproj / draft files exist", outcome: "pass" },
    { id: "devices-resolve", spec_number: 3, title: "Every target device resolves from its stable key", outcome: "pass" },
    { id: "discrete-only", spec_number: 4, title: "Every resolved device is a discrete compute GPU", outcome: "pass" },
    { id: "visibility-pinned", spec_number: 5, title: "Composed device visibility matches the resolved set exactly", outcome: "pass" },
    { id: "vram-fits", spec_number: 6, title: "Estimated VRAM fits in free VRAM per device", outcome: "pass" },
    { id: "commit-headroom", spec_number: 7, title: "Projected system commit stays under 90% of the limit", outcome: "pass" },
    { id: "port-free", spec_number: 8, title: "Requested port is free", outcome: "pass" },
    { id: "alias-unique", spec_number: 9, title: "Alias is unique across running servers", outcome: "pass" },
    {
      id: "no-display", spec_number: 10, title: "No target device is driving a display",
      outcome: { warn: "target device(s) driving a display: AMD Radeon AI PRO R9700 (1920x1080@59) — desktop compositing has consumed 2.68 GB and preempted compute in the past" },
    },
    { id: "versions-match", spec_number: 11, title: "Driver and SDK match the profile's baselines", outcome: "pass" },
    { id: "pcie-aspm-off", spec_number: 12, title: "PCIe link-state power management is Off", outcome: "pass" },
  ].map((r) => (multi && r.id === "visibility-pinned" ? r : r));
  if (!p) return results;
  // Checks 13-15 as core's run_all appends them: 13 whenever the profile or
  // its model is diffusion, 14 when a diffusion run shares the card, 15 for
  // diffusion profiles.
  const dg = p.engine === "diffusion-gemma";
  const model = MOCK_MODELS.find((m) => m.path === p.model?.path);
  const modelDg = model?.engine === "diffusion-gemma";
  if (dg) {
    const build = MOCK_BUILDS.find((b) => b.path === p.build?.path);
    if (!build?.runner_exe) {
      results[0].outcome = { outcome: "block", message: `build ${build?.tag ?? p.build?.path} has no llama-diffusion-gemma-visual-server.exe; install one with \`fidim update --channel unsloth --install\` (or the Updates tab)` };
    }
    if ((p.devices?.length ?? 0) !== 1) {
      results[4].outcome = { outcome: "block", message: `the diffusion engine needs exactly one device; ${p.devices?.length ?? 0} resolved` };
    }
  }
  if (dg || modelDg) {
    const outcome = !dg && modelDg
      ? { outcome: "block", message: `llama-server cannot load ${model.header.architecture}; pick this model again in the editor so it switches to the diffusion engine` }
      : dg && model && !modelDg ? { outcome: "block", message: 'the diffusion runner exits with "not a diffusion model"' } : "pass";
    results.push({ id: "engine-matches-model", spec_number: 13, title: "Engine can load this model", outcome });
  }
  if (dg) {
    const others = (p.devices ?? []).flatMap((d) => MOCK_DEVICES.find((m) => m.device.stable_key === d.key)?.occupied_by ?? []);
    if (others.length) {
      results.push({
        id: "diffusion-card-sharing", spec_number: 14, title: "Card is not shared with a diffusion run",
        outcome: { outcome: "warn", message: p.runtime?.ctx_total
          ? `card also hosts ${others.join(", ")}; this profile allocates its prompt-KV store (up to ${(mockDiffusion(p).sizing.request_bytes / 2 ** 30).toFixed(1)} GiB) per request, after load; sharing can push either model into shared memory`
          : `card also hosts ${others.join(", ")}; the runner auto-sizes its context to the VRAM it believes is free, and WDDM hides other processes' allocations, so it will oversubscribe. Pick an empty card` },
      });
    }
    const s = mockDiffusion(p).sizing;
    const ctx = p.runtime?.ctx_total ?? 0;
    const warns = [];
    const fa = !!p.diffusion?.flash_attn, faPad = !!mockDiffusion(p).runner.fa_pad;
    if (fa && !faPad) warns.push("diffusion.flash_attn is on, but this build's runner does not pad keys for flash attention: DiffusionGemma's 512-dim heads fall back to the CPU, which is slower. Turn it off, or pick a patched runner build");
    if (s.explicit_capped) warns.push(`explicit budget ${ctx} is above the runner's flash-attention ceiling ${s.ctx_ceiling} (training context, and the pad kernel's 65,536-row grid limit); it auto-sizes instead (≈ ${s.predicted_auto_maxtok})`);
    if (s.explicit_fits === false) warns.push(`explicit budget ${ctx} is above what fits (auto MAXTOK predicted ≈ ${s.predicted_auto_maxtok}); the runner will degrade it at load`);
    if (!s.full_offload) warns.push("NGL < block_count+1: partial offload is slow and the context is sized against system RAM");
    results.push({
      id: "diffusion-context", spec_number: 15, title: "Diffusion context budget",
      outcome: warns.length ? { outcome: "warn", message: warns.join("; ") }
        : ctx ? { outcome: "note", message: `explicit MAXTOK ${ctx} (largest prompt ≈ ${s.max_prompt_tokens} tokens); auto would pick ≈ ${s.predicted_auto_maxtok}` }
        : { outcome: "note", message: `auto MAXTOK predicted ≈ ${s.predicted_auto_maxtok} (largest prompt ≈ ${s.predicted_auto_maxtok - s.canvas} tokens); the runner decides at load` },
    });
  }
  return results;
}

// live_check's `diffusion` block (preflight::DiffusionPreflight) for the
// mock DiffusionGemma 26B-A4B header on a 32 GB card.
function mockDiffusion(p) {
  const build = MOCK_BUILDS.find((b) => b.path === p?.build?.path);
  const has = (f) => !!build?.patch?.features?.includes(f);
  const runner = { pkv_f16: has("dg-pkv-f16"), swa_ring: has("dg-swa-ring"), fa_pad: has("dg-fa-pad"), fa_turn_sizing: has("dg-fa-pad") && has("dg-fa-turn-sizing"), runtime_cache_off: true };
  const faSized = runner.fa_turn_sizing && !!p?.diffusion?.flash_attn;
  const ctx = p?.runtime?.ctx_total ?? 0;
  const predicted = faSized ? 65536 : 12288;
  const capped = faSized && ctx > 65536;
  const used = ctx > 0 && !capped ? ctx : predicted;
  // estimate::estimate_diffusion for the 26B-A4B header: the patched runner's
  // FA turn, or the stock store + prefill term (runtime cache off).
  const perTok = runner.swa_ring ? 20480 : 450560, p2 = used - 256;
  const request = faSized ? 45056 * used + 628940800 + 268435456 : 400 * 2 ** 20 + perTok * p2 + (runner.swa_ring ? 628940800 : 0) + 14 * p2 * p2;
  return {
    helper_exe: "C:\\Users\\me\\AppData\\Local\\Programs\\LlamaFIDIM\\fidim-dg.exe",
    helper_dir: "C:\\Users\\me\\AppData\\Local\\Programs\\LlamaFIDIM",
    runner_exe: `${p?.build?.path ?? ""}\\bin\\llama-diffusion-gemma-visual-server.exe`,
    runner_present: !!build?.runner_exe,
    vulkan_backend: false,
    runner,
    sizing: {
      canvas: 256, predicted_auto_maxtok: predicted, maxtok_used: used, max_prompt_tokens: used - 256,
      pkv_bytes_per_token: perTok, request_bytes: request, fa_sized: faSized, ctx_ceiling: 65536,
      explicit_fits: ctx > 0 && !capped ? (faSized || ctx <= 13824) : null, explicit_capped: capped,
      full_offload: (p?.runtime?.n_gpu_layers ?? 0) >= 31,
    },
  };
}

// A subset of profile::validate_diffusion, so the Findings card shows.
function mockFindings(p) {
  if (p?.engine !== "diffusion-gemma") return [];
  const f = [];
  const dg = p.diffusion ?? {};
  const ctx = p.runtime?.ctx_total ?? 0;
  if ((p.devices?.length ?? 0) !== 1) f.push({ severity: "error", code: "dg-single-device", message: `the diffusion engine needs exactly one device (profile lists ${p.devices?.length ?? 0})` });
  if (!dg.default_max_tokens) f.push({ severity: "error", code: "dg-max-tokens", message: "diffusion.default_max_tokens must be at least 1" });
  if (dg.hipblaslt_safeguard === false) f.push({ severity: "warning", code: "dg-safeguard-off", message: "diffusion.hipblaslt_safeguard is off: the first denoise step intermittently fails with 'MUL_MAT failed / ROCm error: invalid argument' when rocBLAS routes through hipBLASLt" });
  if (ctx !== 0 && (ctx < 2048 || ctx % 256 !== 0 || ctx > 65536)) f.push({ severity: "warning", code: "dg-context", message: `runtime.ctx_total ${ctx} is the diffusion context budget (MAXTOK, 0 = auto-size): expected a multiple of 256 between 2048 and 65536` });
  return f;
}

function mockEstimate(p) {
  if (p?.engine === "diffusion-gemma") {
    const s = mockDiffusion(p).sizing;
    const w = 15.65 * 2 ** 30, kv = s.request_bytes, c = 1.93 * 2 ** 30, o = 0.4 * 2 ** 30;
    return {
      per_device: [{ key: p.devices?.[0]?.key ?? "?", fraction: 1.0, weights_bytes: w, kv_bytes: kv, compute_bytes: c, overhead_bytes: o, total_bytes: w + kv + c + o }],
      total_bytes: w + kv + c + o,
      assumptions: ["prompt-KV store is allocated lazily per request (inferred)", "sc_embT buffer inferred", "auto-size cannot see other processes' VRAM (WDDM)"],
    };
  }
  const n = p?.devices?.length ?? 1;
  const per = n === 2
    ? [
        { key: p.devices[0].key, fraction: 0.5, weights_bytes: 10.2e9, kv_bytes: 0.7e9, compute_bytes: 1.34e9, overhead_bytes: 0.43e9, total_bytes: 12.7e9 },
        { key: p.devices[1].key, fraction: 0.5, weights_bytes: 10.2e9, kv_bytes: 0.7e9, compute_bytes: 0.33e9, overhead_bytes: 0.43e9, total_bytes: 11.7e9 },
      ]
    : [{ key: p?.devices?.[0]?.key ?? "?", fraction: 1.0, weights_bytes: 14.25e9, kv_bytes: 4.95e9, compute_bytes: 1.07e9, overhead_bytes: 0.43e9, total_bytes: 20.7e9 }];
  return { per_device: per, total_bytes: per.reduce((a, d) => a + d.total_bytes, 0), assumptions: ["compute buffer heuristic: 0.75 GiB + 0.5 GiB x (batch_physical / 512)"] };
}

const MOCK_RUN = {
  state: {
    profile_id: "worker-pool", pid: 26388, port: 9701, host: "127.0.0.1",
    alias: "gemma-4-worker", started_unix: Math.floor(Date.now() / 1000) - 1830,
    log_path: "C:\\Users\\me\\.fidim\\runs\\worker-pool-9701.log",
    command_line: "llama-server.exe -m ... --port 9701",
    visibility_env: "2",
    device_keys: ["pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08"],
    free_mib_before: [32472], cold_start: false,
  },
  alive: true, health: "healthy", crashed: false,
};

// An upstream build plus an Unsloth fork build (bundled ROCm, carries the
// DiffusionGemma runner). The fork reports the higher upstream number: every
// "newest build" default must still land on the upstream one.
const MOCK_BUILDS = [
  {
    path: "C:\\llama.cpp\\b10819-rocm", tag: "b10819-rocm",
    server_exe: "C:\\llama.cpp\\b10819-rocm\\bin\\llama-server.exe",
    version: "b10819", commit: "8d2c5a1f0", version_error: null,
    channel: "upstream", bundled_runtime: false, release_tag: null, runner_exe: null,
  },
  {
    path: "C:\\llama.cpp\\b11027-mix-3e83366-unsloth", tag: "b11027-mix-3e83366-unsloth",
    server_exe: "C:\\llama.cpp\\b11027-mix-3e83366-unsloth\\bin\\llama-server.exe",
    version: "b11027", commit: "f6b9ea743", version_error: null,
    channel: "unsloth", bundled_runtime: true, release_tag: "b11027-mix-3e83366",
    runner_exe: "C:\\llama.cpp\\b11027-mix-3e83366-unsloth\\bin\\llama-diffusion-gemma-visual-server.exe",
  },
  {
    path: "C:\\llama.cpp\\b11027-mix-3e83366-unsloth-dgpatch", tag: "b11027-mix-3e83366-unsloth-dgpatch",
    server_exe: "C:\\llama.cpp\\b11027-mix-3e83366-unsloth-dgpatch\\bin\\llama-server.exe",
    version: "b11027", commit: "f6b9ea743", version_error: null,
    channel: "unsloth", bundled_runtime: true, release_tag: "b11027-mix-3e83366",
    patch: { name: "dgpatch", base_commit: "f6b9ea743", features: ["dg-pkv-f16", "dg-swa-ring", "dg-fa-pad", "dg-fa-turn-sizing", "dg-step-fail-err"] },
    runner_exe: "C:\\llama.cpp\\b11027-mix-3e83366-unsloth-dgpatch\\bin\\llama-diffusion-gemma-visual-server.exe",
  },
];

// A llama-server model and a DiffusionGemma model; discovery sets `engine`
// from the header and the editor switches the profile's engine on it.
const MOCK_MODELS = [
  {
    path: MOCK_PROFILE.model.path, file_size: 17.0e9, modified_unix: 1785000000, header_error: null,
    engine: "llama-server", mmproj_candidates: [], draft_candidates: [MOCK_PROFILE.model.draft.path],
    header: {
      architecture: "gemma4", model_name: "Gemma 4 26B A4B It", size_label: "26B-A4B", file_type: 15,
      block_count: 30, context_length: 262144, embedding_length: 2816, head_count: 16,
    },
  },
  {
    path: "D:\\models\\unsloth\\diffusiongemma-26B-A4B-it-GGUF\\diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
    file_size: 16806810208, modified_unix: 1789000000, header_error: null,
    engine: "diffusion-gemma", mmproj_candidates: [], draft_candidates: [],
    header: {
      architecture: "diffusion-gemma", model_name: "DiffusionGemma 26B A4B It", size_label: "26B-A4B", file_type: 15,
      block_count: 30, context_length: 262144, embedding_length: 2816, head_count: 16,
      diffusion_canvas_length: 256, attention_causal: false, vocab_size: 262144,
    },
  },
];

const MOCK_DG_PROFILE = {
  schema: 1, engine: "diffusion-gemma", id: "dg-26b", name: "DiffusionGemma 26B-A4B",
  build: { path: MOCK_BUILDS[1].path, version: "b11027" },
  model: { path: MOCK_MODELS[1].path, mmproj: null, draft: null },
  devices: [{ key: "pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08", split_fraction: null, resolved_index_last_launch: null }],
  split_mode: null, main_device: 0,
  server: { port: 9760, alias: "diffusiongemma", host: "127.0.0.1" },
  runtime: {
    n_gpu_layers: 99, ctx_total: 0, slots: 1, kv_type_k: "f16", kv_type_v: "f16",
    flash_attn: "on", batch_logical: 2048, batch_physical: 512, cont_batching: true,
    kv_unified: false, cache_reuse: null,
  },
  sampling: {}, chat: {}, env: {},
  diffusion: { hipblaslt_safeguard: true, default_max_tokens: 2048 },
};

// The mock's newest Unsloth release is not installed until unsloth_install runs.
const MOCK_UNSLOTH_TAG = "b11031-mix-3e83366";
const MOCK_UNSLOTH_GFX = ["gfx103X", "gfx110X", "gfx1150", "gfx1151", "gfx120X", "gfx908", "gfx90a"];
let mockUnslothInstalled = false;
// The runner-patch overlay is published for the mock release; installed on demand.
let mockOverlayInstalled = false;

const MOCK_LOG_LINES = [
  "0.00.339 I   - ROCm0   : AMD Radeon AI PRO R9700 (32624 MiB, 32472 MiB free)",
  "0.00.640 I   - ROCm1   : AMD Radeon AI PRO R9700 (32624 MiB, 32472 MiB free)",
  "1.13.460 I srv          init: init: chat template, thinking = 1",
  "1.13.460 I srv  llama_server: model loaded",
  "1.13.460 I srv  llama_server: server is listening on http://127.0.0.1:9701",
  "1.13.461 I srv  update_slots: all slots are idle",
];

async function mock(cmd, args) {
  await new Promise((r) => setTimeout(r, 120));
  switch (cmd) {
    case "devices":
      return MOCK_DEVICES;
    case "list_profiles":
      return [
        { profile: MOCK_PROFILE, findings: [] },
        { profile: MOCK_SPLIT_PROFILE, findings: [] },
        { profile: MOCK_DG_PROFILE, findings: [] },
      ];
    case "live_check": {
      const p = args.p;
      if (p?.engine === "diffusion-gemma") {
        const dev = MOCK_DEVICES.find((m) => m.device.stable_key === p.devices?.[0]?.key)?.device ?? MOCK_DEVICES[2].device;
        const idx = dev.hip_index;
        const d = mockDiffusion(p);
        const dg = p.diffusion ?? {};
        const env = [...Object.entries(p.env ?? {}),
          ["HIP_VISIBLE_DEVICES", String(idx)], ["CUDA_VISIBLE_DEVICES", String(idx)],
          ["NGL", String(p.runtime?.n_gpu_layers ?? 0)], ["MAXTOK", String(p.runtime?.ctx_total ?? 0)], ["FA", dg.flash_attn ? "1" : "0"]];
        if (d.sizing.full_offload) env.push(["DG_FREE_RAM_MB", "0"]);
        if (dg.hipblaslt_safeguard !== false) env.push(["ROCBLAS_USE_HIPBLASLT", "0"], ["ROCBLAS_USE_HIPBLASLT_BATCHED", "0"]);
        return {
          findings: mockFindings(p),
          results: mockCheckResults(p),
          estimate: mockEstimate(p),
          resolved: (p.devices ?? []).map((x) => ({
            profile_key: x.key,
            device: MOCK_DEVICES.find((m) => m.device.stable_key === x.key)?.device ?? MOCK_DEVICES[2].device,
            fraction: 1.0, rebound: false,
          })),
          command_line: `"${d.helper_exe}" --runner "${d.runner_exe}" --model "${p.model?.path}" --host ${p.server?.host ?? "127.0.0.1"} --port ${p.server?.port} --alias ${p.server?.alias} --req-prefix "C:\\Users\\me\\.fidim\\runs\\dg-${p.id}-${p.server?.port}" --default-max-tokens ${dg.default_max_tokens ?? 2048}${dg.seed != null ? ` --seed ${dg.seed}` : ""} --expect-bus ${dev.bus_number} --build-tag ${MOCK_BUILDS.find((b) => b.path === p.build?.path)?.tag ?? "?"}`,
          env,
          commit: { limit_bytes: 95.4e9, charge_bytes: 48.7e9 },
          diffusion: d,
        };
      }
      return {
        findings: [],
        results: mockCheckResults(p),
        estimate: mockEstimate(p),
        resolved: (p?.devices ?? []).map((d, i) => ({
          profile_key: d.key,
          device: MOCK_DEVICES.find((m) => m.device.stable_key === d.key)?.device ?? MOCK_DEVICES[2].device,
          fraction: d.split_fraction ?? 1.0 / (p.devices.length || 1),
          rebound: false,
        })),
        command_line: '"llama-server.exe" -m model.gguf -ngl 99 -c ' + (p?.runtime?.ctx_total ?? 0) + " -np " + (p?.runtime?.slots ?? 1) + " --port " + (p?.server?.port ?? 0),
        env: [["HIP_VISIBLE_DEVICES", (p?.devices?.length ?? 1) > 1 ? "0,2" : "2"], ["GPU_MAX_HW_QUEUES", "1"]],
        commit: { limit_bytes: 95.4e9, charge_bytes: 48.7e9 },
        diffusion: null,
      };
    }
    case "status":
      return [MOCK_RUN];
    case "slots":
      return [
        { id: 0, state: 1, prompt: "..." }, { id: 1, state: 0 }, { id: 2, state: 1 },
        { id: 3, state: 0 }, { id: 4, state: 0 }, { id: 5, state: 0 },
      ];
    case "log_sources":
      return [{ profile_id: "worker-pool", port: 9701, alive: true, log_path: MOCK_RUN.state.log_path }];
    case "read_log":
      return { path: MOCK_RUN.state.log_path, lines: MOCK_LOG_LINES.filter((l) => !args.filter || l.includes(args.filter)), total_matching: MOCK_LOG_LINES.length };
    case "bench_history":
      return args.id === "worker-pool"
        ? [
            { measured_at: "2026-07-31T15:36:02Z", cold_cache: true, serial_tok_s: 96.7, vram_gb: 19.87, driver: "32.0.31035.1003", sdk: "7.1.51803", concurrent: { n: 6, aggregate_tok_s: 138.1, decode_aggregate_tok_s: 300.5, per_stream_tok_s: 50.1 } },
            { measured_at: "2026-07-31T15:49:16Z", cold_cache: false, serial_tok_s: 90.1, vram_gb: 19.87, driver: "32.0.31035.1003", sdk: "7.1.51803", concurrent: { n: 6, aggregate_tok_s: 261.5, decode_aggregate_tok_s: 293.8, per_stream_tok_s: 49.0 } },
          ]
        : [];
    case "bench_profile":
      await new Promise((r) => setTimeout(r, 1500));
      return { saved: true, cold: false, baseline: { serial_tok_s: 90.1, vram_gb: 19.87, concurrent: { n: 6, aggregate_tok_s: 261.5, decode_aggregate_tok_s: 293.8, per_stream_tok_s: 49.0 } } };
    case "launch_profile":
      await new Promise((r) => setTimeout(r, 2000));
      return { blocked: false, results: mockCheckResults(null), state: MOCK_RUN.state, cold_start: false, placement: [{ key: MOCK_RUN.state.device_keys[0], expected_bytes: 20.7e9, dedicated_bytes: 21.3e9, committed_bytes: 19.3e9 }] };
    case "live_one":
      return mockDiffusionSample(Date.now());
    case "dg_frames":
      return mockDgFrames();
    case "live": {
      const t = Date.now();
      const tick = Math.floor(t / 1000);
      const LOOP = "I have completed the task. ".repeat(6);
      const slot = (id, phase, extra = {}) => ({ id, id_task: 4000 + id, n_ctx: 32768, is_processing: phase !== "idle", phase, prompt: phase === "idle" && id > 1 ? null : "<|turn>system\nYou are a helpful assistant.<|turn>user\nSummarise the router log and say whether gemma4 is healthy.", prompt_chars: 110, generated: phase === "idle" && id > 1 ? null : (id === 1 ? LOOP : "The router is serving three models. gemma4 answered the probe with OK and has no errors since the last init; the only warning is the missing API key, which is expected for a keyless router. ".slice(0, 40 + (tick * 9) % 150)), generated_chars: id === 1 ? LOOP.length : 190, loop_hint: id === 1 ? { fragment: "I have completed the task. ", repeats: 6 } : null, n_prompt_tokens: 2468, n_prompt_tokens_processed: phase === "prefill" ? 900 + (tick * 137) % 1500 : 2468, n_prompt_tokens_cache: 0, n_decoded: phase === "decode" ? 40 + (tick * 7) % 300 : 0, n_remain: phase === "decode" ? 400 : 0, prefill_fraction: phase === "prefill" ? ((900 + (tick * 137) % 1500) / 2468) : phase === "decode" ? 1 : 0, ctx_fraction: phase === "idle" ? 0.08 : 0.21 + (id % 3) * 0.2, ...extra });
      const metrics = (gen, prompt, drafts) => ({ tokens_predicted_total: gen, prompt_tokens_total: prompt, requests_processing: 1, requests_deferred: 0, spec_decode_num_draft_tokens_total: drafts, spec_decode_num_accepted_tokens_total: Math.round(drafts * 0.62) });
      const runRouter = {
        run: { state: { profile_id: "router", pid: 24304, port: 1234, host: "127.0.0.1", alias: "router", started_unix: Math.floor(t / 1000) - 4560, log_path: "", command_line: "", visibility_env: null, device_keys: MOCK_DEVICES.map((d) => d.device.stable_key), free_mib_before: [], cold_start: false }, alive: true, health: "healthy", crashed: false },
        samples: [
          { model: "bonsai", sampled_unix_ms: t, phase: "idle", slots: [0, 1, 2, 3].map((i) => slot(i, "idle")), metrics: metrics(120000, 400000, 0), error: null },
          { model: "dd", sampled_unix_ms: t, phase: "decode", slots: [slot(0, "decode"), slot(1, "idle")], metrics: metrics(30000 + tick * 29, 90000 + tick * 3, 8000 + tick * 40), error: null },
          { model: "gemma4", sampled_unix_ms: t, phase: "prefill", slots: [slot(0, "prefill"), slot(1, "decode"), ...[2, 3, 4, 5, 6, 7].map((i) => slot(i, "idle"))], metrics: metrics(50000 + tick * 50, 200000 + tick * 230, 0), error: null },
        ],
        resident: [{ card: MOCK_DEVICES[0].device.stable_key, dedicated_bytes: 26.2e9, committed_bytes: 26.2e9 }, { card: MOCK_DEVICES[2].device.stable_key, dedicated_bytes: 13.1e9, committed_bytes: 13.1e9 }],
        gpu_busy_percent: 67,
      };
      // A standalone DiffusionGemma run (fidim-dg): committed blocks, then the
      // current block's draft sharpening step by step.
      const dgSample = mockDiffusionSample(t);
      const runDiffusion = {
        run: { state: { profile_id: "dg-26b", pid: 32400, port: 9760, host: "127.0.0.1", alias: "diffusiongemma", started_unix: Math.floor(t / 1000) - 600, log_path: "", command_line: "", visibility_env: "1", device_keys: [MOCK_DEVICES[2].device.stable_key], free_mib_before: [], cold_start: false }, alive: true, health: "healthy", crashed: false },
        samples: [dgSample],
        resident: [{ card: MOCK_DEVICES[2].device.stable_key, dedicated_bytes: 19.7e9, committed_bytes: 19.7e9 }],
        gpu_busy_percent: 88,
      };
      const runCrashed = { run: { ...MOCK_RUN, alive: false, health: "dead", crashed: true }, samples: [], resident: [], gpu_busy_percent: 0 };
      return { runs: [runRouter, runDiffusion, runCrashed], cards: MOCK_DEVICES.filter((d) => !d.device.integrated).map((d, i) => ({ key: d.device.stable_key, name: d.device.name, busy_percent: i ? 24 : 95, total_mib: d.device.total_mib })) };
    }
    case "app_version":
      return { version: "0.2.0", long: "0.2.0+3 (4f2a1c9 2026-09-20)", commit: "4f2a1c9", commit_date: "2026-09-20", commits_ahead: 3, modified: false };
    case "get_config":
      return { path: "C:\\Users\\me\\.fidim\\config.json", config: { build_roots: ["C:\\llama.cpp"], model_roots: ["D:\\models"], rocm_bin: "C:\\Program Files\\AMD\\ROCm\\7.1\\bin", default_runtime: null, install_root: null, llama_cpp_source: null, source_build_script: "scripts\\build-from-tag.bat", hf_token: null, integrated_name_patterns: ["Radeon(TM) Graphics"], profile_dir: "C:\\Users\\me\\.fidim\\profiles", runs_dir: "C:\\Users\\me\\.fidim\\runs", keep_alive_seconds: 0, runtimes: [] } };
    case "list_runtimes":
      return [{ name: "default", source: "config", version: "7.1", available: true, is_default: true, dirs: ["C:\\Program Files\\AMD\\ROCm\\7.1\\bin"] }, { name: "rocm-7.14.0a20260612", source: "amd-nightly", version: "7.14.0a20260612", available: true, is_default: false, is_latest: true, dirs: ["D:\\llama.cpp\\rocm\\7.14.0a20260612\\bin"] }];
    case "router_get":
      return { host: "127.0.0.1", port: 1234, models_max: 3, autoload: true, build: null, rocm_runtime: null, members: [{ profile_id: "worker-pool", load_on_startup: true }] };
    case "router_ini":
      return { text: "[gemma-4-worker]\nmodel = D:\\models\\...\\gemma-4-26B-A4B.gguf\ndevice = ROCm2\nload-on-startup = true\n" };
    case "router_status":
      return { alive: true, state: { pid: 24304, port: 1234 } };
    case "router_models":
      return [{ id: "gemma-4-worker", status: "loaded" }, { id: "qwen-split", status: "unloaded" }];
    case "update_check":
      return { latest: { tag: "b10819", published_at: "2026-09-05" }, update_available: false, behind: 0, already_installed: true, newest_installed: { version: "b10819", path: "C:\\llama.cpp\\b10819-rocm" }, install_dir: "C:\\llama.cpp\\b10819-rocm", assets: [{ name: "llama-b10819-bin-win-cpu-x64.zip", size: 21e6 }], asset_error: null };
    case "update_history":
      return [];
    case "unsloth_check": {
      const tag = args.tag ?? MOCK_UNSLOTH_TAG;
      const gfx = args.gfx || "gfx120X";
      const name = `app-${tag}-windows-x64-rocm-${gfx}.zip`;
      const known = MOCK_UNSLOTH_GFX.includes(gfx);
      const dir = `C:\\llama.cpp\\${tag}-unsloth`;
      const installed = MOCK_BUILDS.filter((b) => b.channel === "unsloth").map((b) => ({ tag: b.tag, version: b.version, path: b.path, patch: b.patch?.name }));
      if (mockUnslothInstalled) installed.unshift({ tag: `${tag}-unsloth`, version: "b11031", path: dir });
      if (mockOverlayInstalled) installed.unshift({ tag: `${tag}-unsloth-dgpatch5`, version: "b11031", path: `${dir}-dgpatch5`, patch: "dgpatch5" });
      const overlayPublished = tag === MOCK_UNSLOTH_TAG;
      return {
        latest: { tag, published_at: "2026-09-18T21:04:11Z", html_url: `https://github.com/unslothai/llama.cpp/releases/tag/${tag}`, assets: [], name: `llama.cpp prebuilt ${tag}`, body: "" },
        upstream_tag: "b11031", gfx, gfx_available: MOCK_UNSLOTH_GFX,
        asset: known ? { name, url: "", size: 494370803, digest: "sha256:09135ea01882040460a1eda0f301ca28effd270fdf661c3697d29119e2234011" } : null,
        asset_error: known ? null : `release ${tag} has no ${name}; its Windows ROCm zips are: ${MOCK_UNSLOTH_GFX.map((g) => `app-${tag}-windows-x64-rocm-${g}.zip`).join(", ")}`,
        install_dir: dir, already_installed: mockUnslothInstalled, installed,
        overlay_repo: "Dixon-Cider/fidim-dg-overlay", overlay_patch: "dgpatch5", overlay_available: overlayPublished,
        overlay_asset: overlayPublished ? { name: `fidim-dg-overlay-${tag}-windows-x64.zip`, url: "", size: 9437184, digest: "sha256:0ce183b190a9b2e678e9bd6e31c40c42b878834233f75529e6d785fa3106eeb8" } : null,
        overlay_error: null, overlay_install_dir: `${dir}-dgpatch5`, overlay_installed: mockOverlayInstalled,
      };
    }
    case "unsloth_install": {
      await new Promise((r) => setTimeout(r, 1500));
      const tag = args.tag ?? MOCK_UNSLOTH_TAG;
      const overlay = !!args.overlay;
      const skipped = overlay ? mockOverlayInstalled : mockUnslothInstalled;
      if (overlay) mockOverlayInstalled = true; else mockUnslothInstalled = true;
      return {
        tag, dir: `C:\\llama.cpp\\${tag}-unsloth${overlay ? "-dgpatch5" : ""}`,
        source: overlay ? "unsloth-overlay" : "unsloth-prebuilt", skipped_existing: skipped,
        verify: {
          version: "b11031", commit: "a41c7e2d0", hip_ok: true, detail: "", runner_present: true,
          devices: MOCK_DEVICES.map((d) => ({ index: d.device.hip_index, backend: "ROCm", name: d.device.name, total_mib: d.device.total_mib })),
        },
      };
    }
    case "unsloth_promote_preview": {
      // The diffusion profile sits on the locally patched build: onto the
      // dgpatch5 overlay it moves (every feature is there), onto a plain
      // build it stays.
      const patched = MOCK_BUILDS.find((b) => b.patch);
      const fork = [["worker-pool", "Unsloth fork build: pick it in the editor if wanted"], ["qwen-split", "Unsloth fork build: pick it in the editor if wanted"]];
      if (/-dgpatch5$/.test(args.toPath)) {
        return { moves: [{ profile_id: "dg-26b", from: { path: patched.path, version: patched.version }, from_patch: patched.patch.name }], skipped: fork };
      }
      return { moves: [], skipped: [["dg-26b", `on a patched runner build (${patched.patch.name}) whose features the target lacks (${patched.patch.features.join(", ")}); pick the new build in the editor if wanted`], ...fork] };
    }
    case "unsloth_promote": {
      const ids = args.ids ?? ["dg-26b"];
      return {
        batch: { at_unix: Math.floor(Date.now() / 1000), entries: ids.map((id) => ({ profile_id: id, from: { path: MOCK_BUILDS[1].path, version: "b11027" }, to: { path: args.toPath, version: args.toVersion } })) },
        skipped: [["worker-pool", "Unsloth fork build: pick it in the editor if wanted"], ["qwen-split", "Unsloth fork build: pick it in the editor if wanted"]],
      };
    }
    case "rocm_families":
      return ["gfx103X-all", "gfx110X-all", "gfx1151", "gfx120X-all"];
    case "rocm_available":
      return { runtimes: [
        { version: "7.14.0a20260612", channel: "nightly", family: args.family, core_url: "", libraries_url: "" },
        { version: "7.2.1", channel: "release", family: null, core_url: "", libraries_url: "" },
        { version: "7.1.1", channel: "release", family: null, core_url: "", libraries_url: "" },
      ], problems: [] };
    case "rocm_install":
      await new Promise((r) => setTimeout(r, 1500));
      return { dir: "D:\\llama.cpp\\rocm\\" + args.runtime.version, name: "rocm-" + args.runtime.version };
    case "rocm_remove":
      return null;
    case "stop_run":
    case "save_profile":
    case "delete_profile":
      return null;
    case "export_profile":
      return { path: "C:\\Users\\me\\.fidim\\profiles\\worker-pool.bat", text: "@echo off\r\nset \"HIP_VISIBLE_DEVICES=2\"\r\n\"llama-server.exe\" -m ..." };
    case "scan":
      return { builds: MOCK_BUILDS, models: MOCK_MODELS, drafts: [MOCK_PROFILE.model.draft.path], mmproj: [] };
    default:
      throw new Error("unmocked command: " + cmd);
  }
}
