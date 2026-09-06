<script>
  import { api, log } from "../api.js";
  import PathList from "../components/PathList.svelte";
  import Range from "../components/Range.svelte";
  import { scale, fade } from "svelte/transition";
  import { LAYOUT } from "../motion.js";

  let cfg = $state(null);       // editable copy of config.json
  let path = $state("");
  let runtimes = $state([]);
  let devices = $state([]);
  let saving = $state(false);
  let error = $state("");
  let saved = $state("");
  let dirty = $state(false);
  let diag = $state(null);      // { builds, models, devices, runtimes, errors, ms }
  let diagBusy = $state(false);

  let manualRuntimesText = $state("");

  const HINTS = {
    model_roots: "Folders scanned for *.gguf, recursively. LM Studio's download folder works as one and is picked up on first run when it exists.",
    build_roots: "Folders scanned one level deep for <dir>\\bin\\llama-server.exe. The install root below is always scanned too.",
    install_root: "Where Updates puts llama.cpp builds and ROCm runtimes. Empty = the first build root, or the tool's own folder when there is none.",
    llama_cpp_source: "git checkout used by Build from source. Empty = the first build root.",
    source_build_script: "Script run as <script> <checkout> <tag> <out dir>. Empty = scripts\\build-from-tag.bat next to the exe or in the repo.",
    profile_dir: "Where profile JSON files live.",
    runs_dir: "Run state and captured server logs.",
    rocm_bin: "Fallback ROCm runtime: a DLL folder prepended to PATH when a profile names no runtime and no default is picked below. Usually a HIP SDK bin folder.",
    default_runtime: "Runtime used when a profile names none. Install more on the Updates tab.",
    rocm_family: "GPU family for AMD's nightly ROCm index. Empty = guessed from your cards. RDNA4 (Radeon AI PRO R9700, RX 9000) is gfx120X-all; RDNA3 (RX 7000) is gfx110X-all; RDNA2 (RX 6000) is gfx103X-all; Strix Halo is gfx1151.",
    runtimes: "Runtimes added by hand, JSON list: [{\"name\":\"x\",\"dirs\":[\"C:\\\\a\\\\bin\"],\"version\":\"9.9\"}]",
    integrated_name_patterns: "Device names containing any of these count as integrated graphics.",
    allow_integrated: "Let profiles bind integrated graphics. Off, pre-flight blocks it, because next to a discrete card an iGPU is a trap: it runs at a fraction of the speed and never errors. On an APU-only machine, or a Strix Halo with up to 96 GB of shared memory, turn this on.",
    keep_alive: "Off by default. VRAM eviction on idle comes from the PCIe Link State Power Management power setting; pre-flight check 12 warns when it is not Off. If it cannot be Off, a 1-token request every N seconds keeps the GPU awake. Profiles can override.",
    hf_token: "Hugging Face token, used only to read generation_config.json from gated repos. HF_TOKEN in the environment wins.",
  };

  async function load() {
    error = "";
    try {
      const [c, r, d] = await Promise.all([api("get_config"), api("list_runtimes").catch(() => []), api("devices", { refresh: false }).catch(() => [])]);
      cfg = c.config;
      cfg.build_roots ??= []; cfg.model_roots ??= []; cfg.integrated_name_patterns ??= [];
      path = c.path;
      runtimes = r;
      devices = d.map((x) => x.device);
      manualRuntimesText = JSON.stringify(cfg.runtimes ?? [], null, 2);
      dirty = false;
    } catch (e) { error = String(e); }
  }
  load();

  const touch = () => (dirty = true);

  function assembled() {
    const out = JSON.parse(JSON.stringify(cfg));
    const clean = (arr) => (arr ?? []).map((s) => String(s).trim()).filter(Boolean);
    out.build_roots = clean(out.build_roots);
    out.model_roots = clean(out.model_roots);
    out.integrated_name_patterns = clean(out.integrated_name_patterns);
    for (const k of ["rocm_bin", "install_root", "llama_cpp_source", "source_build_script", "hf_token", "default_runtime", "rocm_family"])
      if (out[k] === "" || out[k] === undefined) out[k] = null;
    out.keep_alive_seconds = Math.max(0, Math.round(Number(out.keep_alive_seconds) || 0));
    out.allow_integrated = !!out.allow_integrated;
    if (out.default_runtime === "default") out.default_runtime = null;
    let manual = [];
    if (manualRuntimesText.trim()) manual = JSON.parse(manualRuntimesText);
    if (!Array.isArray(manual)) throw new Error("manual runtimes must be a JSON list");
    out.runtimes = manual;
    return out;
  }

  async function save() {
    saving = true; error = ""; saved = "";
    try {
      const c = assembled();
      await api("save_config", { config: c });
      saved = "Saved. Models, builds and devices will rescan.";
      log("settings saved");
      await load();
      setTimeout(() => (saved = ""), 2500);
    } catch (e) { error = String(e); }
    saving = false;
  }

  async function diagnose() {
    diagBusy = true; diag = null;
    const t = performance.now();
    const errors = [];
    const grab = async (name, fn) => { try { return await fn(); } catch (e) { errors.push(`${name}: ${String(e)}`); return null; } };
    const s = await grab("scan", () => api("scan", { refresh: true }));
    const d = await grab("devices", () => api("devices", { refresh: true }));
    const r = await grab("list_runtimes", () => api("list_runtimes"));
    diag = {
      builds: s?.builds ?? [], models: s?.models ?? [], devices: (d ?? []).map((x) => x.device), runtimes: r ?? [],
      errors, ms: Math.round(performance.now() - t),
    };
    log(`diagnostics: builds=${diag.builds.length} models=${diag.models.length} devices=${diag.devices.length} runtimes=${diag.runtimes.length} errors=${errors.length} in ${diag.ms}ms`);
    diagBusy = false;
  }

  const defaultRt = $derived(runtimes.find((r) => r.is_default));
  const igpus = $derived(devices.filter((d) => d.integrated));
</script>

<h1>
  Settings
  <span class="sub">Llama FIDIM's own settings. Hover a label for help.</span>
</h1>

{#if cfg}
  <div class="savebar">
    <span class="path">{path}</span>
    {#if saved}<span class="chip pass" in:scale={{ duration: LAYOUT, start: 0.7 }} out:fade={{ duration: LAYOUT }}>{saved}</span>
    {:else if dirty}<span class="chip warn" in:scale={{ duration: LAYOUT, start: 0.7 }}>unsaved changes</span>{/if}
    {#if error}<span class="chip block shake">{error}</span>{/if}
    <span style="margin-left: auto; display: flex; gap: 8px;">
      <button class="btn" onclick={load} disabled={saving}>Reload</button>
      <button class="btn primary" onclick={save} disabled={saving}>{saving ? "Saving…" : "Save settings"}</button>
    </span>
  </div>

  <section class="card">
    <div class="sec">Folders <span class="faint">where models and builds are found, and where new ones go</span></div>
    <div class="grid2">
      <label class="field" title={HINTS.model_roots}>
        <span class="k">model folders <span class="faint">({cfg.model_roots.length})</span></span>
        <PathList bind:value={cfg.model_roots} placeholder="D:\models" onchange={touch} />
      </label>
      <label class="field" title={HINTS.build_roots}>
        <span class="k">llama.cpp build folders <span class="faint">({cfg.build_roots.length})</span></span>
        <PathList bind:value={cfg.build_roots} placeholder="D:\llama.cpp" />
      </label>
    </div>
    <div class="formgrid" style="margin-top: 16px;">
      <label class="field" style="grid-column: span 3;" title={HINTS.install_root}>
        <span class="k">install root for Updates</span>
        <input bind:value={cfg.install_root} oninput={touch} placeholder="(first build folder, or the tool's own folder)" />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.profile_dir}>
        <span class="k">profiles</span>
        <input bind:value={cfg.profile_dir} oninput={touch} />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.runs_dir}>
        <span class="k">run state and server logs</span>
        <input bind:value={cfg.runs_dir} oninput={touch} />
      </label>
    </div>
  </section>

  <section class="card">
    <div class="sec">ROCm <span class="faint">which DLLs a server runs against</span></div>
    <div class="formgrid">
      <label class="field" style="grid-column: span 3;" title={HINTS.default_runtime}>
        <span class="k">default runtime</span>
        <select bind:value={cfg.default_runtime} onchange={touch}>
          <option value={null}>fallback folder below{defaultRt && defaultRt.name === "default" && defaultRt.version ? ` (${defaultRt.version})` : ""}</option>
          {#each runtimes.filter((r) => r.name !== "default") as r}
            <option value={r.name} disabled={!r.available}>{r.name}{r.version ? ` · ${r.version}` : ""}{r.is_latest ? " (latest)" : ""}{r.available ? "" : " (missing)"}</option>
          {/each}
        </select>
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.rocm_bin}>
        <span class="k">fallback runtime folder</span>
        <input bind:value={cfg.rocm_bin} oninput={touch} placeholder="C:\Program Files\AMD\ROCm\7.1\bin" />
      </label>
      <label class="field" style="grid-column: span 2;" title={HINTS.rocm_family}>
        <span class="k">GPU family for AMD downloads</span>
        <input bind:value={cfg.rocm_family} oninput={touch} placeholder="guessed from your cards" list="rocm-families" />
        <datalist id="rocm-families"><option value="gfx120X-all"></option><option value="gfx110X-all"></option><option value="gfx103X-all"></option><option value="gfx1151"></option><option value="gfx1150"></option></datalist>
      </label>
      <label class="field" style="grid-column: span 4;" title={HINTS.runtimes}>
        <span class="k">runtimes added by hand (JSON)</span>
        <textarea rows="2" bind:value={manualRuntimesText} oninput={touch} spellcheck="false"></textarea>
      </label>
    </div>
    <div class="faint small" style="margin-top: 10px;">
      Found now, newest first: {runtimes.map((r) => r.name + (r.is_latest ? " (latest)" : "") + (r.available ? "" : " (missing)")).join(" · ") || "none"}. Install more on the Updates tab.
    </div>
  </section>

  <section class="card">
    <div class="sec">GPUs <span class="faint">{devices.length ? `${devices.length} adapters seen, ${igpus.length} integrated` : "no devices enumerated yet"}</span></div>
    <div class="grid2">
      <label class="field" title={HINTS.integrated_name_patterns}>
        <span class="k">integrated graphics name patterns</span>
        <PathList bind:value={cfg.integrated_name_patterns} placeholder="Radeon(TM) Graphics" addLabel="Add pattern" mono={false} />
      </label>
      <div style="display: flex; flex-direction: column; gap: 14px;">
        <label class="field" title={HINTS.allow_integrated}>
          <span class="k">integrated graphics</span>
          <span><input type="checkbox" checked={!!cfg.allow_integrated} onchange={(e) => { cfg.allow_integrated = e.target.checked; touch(); }} /> allow profiles to bind it (pre-flight warns instead of blocking)</span>
        </label>
        <div class="formgrid" style="grid-template-columns: 1fr;">
          <Range bind:value={cfg.keep_alive_seconds} label="keep-alive interval (seconds, 0 = off)" title={HINTS.keep_alive} min={0} max={60} step={1} span={1} onchange={touch} />
        </div>
      </div>
    </div>
    {#if igpus.length}
      <div class="faint small" style="margin-top: 10px;">Classified as integrated right now: {igpus.map((d) => d.name).join(", ")}.</div>
    {/if}
  </section>

  <section class="card">
    <div class="sec">Source builds and Hugging Face</div>
    <div class="formgrid">
      <label class="field" style="grid-column: span 3;" title={HINTS.llama_cpp_source}>
        <span class="k">llama.cpp checkout</span>
        <input bind:value={cfg.llama_cpp_source} oninput={touch} placeholder="(first build folder)" />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.source_build_script}>
        <span class="k">build script</span>
        <input bind:value={cfg.source_build_script} oninput={touch} placeholder="(scripts\build-from-tag.bat)" />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.hf_token}>
        <span class="k">Hugging Face token (gated repos only)</span>
        <input type="password" bind:value={cfg.hf_token} oninput={touch} placeholder="hf_…" autocomplete="off" />
      </label>
    </div>
  </section>

  <section class="card">
    <div class="sec">
      Diagnostics
      <span class="faint">rescan everything and report what the tool can see</span>
      <span style="margin-left: auto;"><button class="btn" onclick={diagnose} disabled={diagBusy}>{diagBusy ? "Running…" : "Run diagnostics"}</button></span>
    </div>
    {#if diag}
      {#each diag.errors as e}<div class="notice" style="margin-bottom: 6px;"><span class="chip block">failed</span> <span class="mono">{e}</span></div>{/each}
      <div class="stats">
        <div class="stat" class:dim={!diag.models.length}><span class="v">{diag.models.length}</span><span class="l">models{diag.models.filter((m) => m.header_error).length ? `, ${diag.models.filter((m) => m.header_error).length} with header errors` : ""}</span></div>
        <div class="stat" class:dim={!diag.builds.length}><span class="v">{diag.builds.length}</span><span class="l">builds{diag.builds.length ? ": " + diag.builds.map((b) => b.version ?? "broken").join(", ") : ""}</span></div>
        <div class="stat" class:dim={!diag.devices.length}><span class="v">{diag.devices.length}</span><span class="l">devices{diag.devices.length ? ": " + diag.devices.map((d) => `${d.backend}${d.hip_index}`).join(", ") : ""}</span></div>
        <div class="stat" class:dim={!diag.runtimes.length}><span class="v">{diag.runtimes.filter((r) => r.available).length}</span><span class="l">runtimes usable</span></div>
      </div>
      <div class="faint small" style="margin-top: 10px;">{diag.ms} ms · also written to ui.log in the tool's folder</div>
    {:else}
      <div class="empty" style="padding: 12px 0;">Not run yet</div>
    {/if}
  </section>
{:else if error}
  <div class="card notice"><span class="chip block">error</span> <span class="mono">{error}</span></div>
{/if}

<style>
  .savebar {
    position: sticky; top: 0; z-index: 5; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
    background: var(--ground); padding: 6px 0 12px; margin-bottom: 4px; border-bottom: 1px solid var(--rule);
  }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px 24px; }
  @media (max-width: 1100px) { .grid2 { grid-template-columns: 1fr; } }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px 20px; }
  section.card { padding: 18px 20px; }
</style>
