// Command bridge. Inside Tauri every call hits the Rust command layer
// (llamactl-core). In a plain browser (vite dev without Tauri) a mock layer
// answers with the real target machine's topology so the UI can be exercised
// visually without hardware access.

let invoke = null;
export const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
if (inTauri) {
  const mod = await import("@tauri-apps/api/core");
  invoke = mod.invoke;
}

export async function api(cmd, args = {}) {
  if (invoke) return invoke(cmd, args);
  return mock(cmd, args);
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
  build: { path: "C:\\...\\llama.cpp\\build-hip-vision", version: "b9817" },
  model: {
    path: "E:\\models\\unsloth\\gemma-4-26B-A4B-it-qat-GGUF\\gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    mmproj: null,
    draft: { path: "E:\\models\\...\\MTP\\mtp-gemma-4-26B-A4B-it-Q8_0.gguf", enabled: false },
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
  model: { path: "E:\\models\\lmstudio-community\\Qwen3.6-35B-A3B-GGUF\\Qwen3.6-35B-A3B-Q4_K_M.gguf", mmproj: null, draft: null },
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
  return [
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
  ].map((r) => (multi && r.id === "visibility-pinned" ? r : r));
}

function mockEstimate(p) {
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
    log_path: "C:\\Users\\me\\.llamactl\\runs\\worker-pool-9701.log",
    command_line: "llama-server.exe -m ... --port 9701",
    visibility_env: "2",
    device_keys: ["pci:VEN_1002&DEV_7551&SUBSYS_54131849:bus08"],
    free_mib_before: [32472], cold_start: false,
  },
  alive: true, health: "healthy", crashed: false,
};

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
      ];
    case "live_check": {
      const p = args.p;
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
    case "stop_run":
    case "save_profile":
    case "delete_profile":
      return null;
    case "export_profile":
      return { path: "C:\\Users\\me\\.llamactl\\profiles\\worker-pool.bat", text: "@echo off\r\nset \"HIP_VISIBLE_DEVICES=2\"\r\n\"llama-server.exe\" -m ..." };
    case "scan":
      return { builds: [], models: [] };
    default:
      throw new Error("unmocked command: " + cmd);
  }
}
