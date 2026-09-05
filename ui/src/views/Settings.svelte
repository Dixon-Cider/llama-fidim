<script>
  import { api, log } from "../api.js";

  let cfg = $state(null);       // editable copy of config.json
  let path = $state("");
  let runtimes = $state([]);
  let saving = $state(false);
  let error = $state("");
  let saved = $state("");
  let diag = $state(null);      // { builds, models, devices, runtimes, errors, ms }
  let diagBusy = $state(false);

  // Multi-line text <-> string arrays.
  const lines = (arr) => (arr ?? []).join("\n");
  const unlines = (text) => String(text ?? "").split(/\r?\n/).map((s) => s.trim()).filter(Boolean);

  let buildRootsText = $state("");
  let modelRootsText = $state("");
  let igpuText = $state("");
  let manualRuntimesText = $state("");

  const HINTS = {
    build_roots: "Folders scanned one level deep for <dir>\\bin\\llama-server.exe. Your llama.cpp checkout is one; Updates installs new builds under the first root.",
    model_roots: "Folders scanned recursively for *.gguf. Point one at LM Studio's download folder (Settings → Models) to serve what it downloads.",
    rocm_bin: "DLL folder of the default ROCm runtime, prepended to PATH for every server and probe. HIP SDK 7.1 today.",
    default_runtime: "Which discovered runtime profiles use when they name none. Leave on 'default' to keep rocm_bin.",
    install_root: "Where Updates puts new builds (<tag>-rocm). Empty = the first build root.",
    llama_cpp_source: "git checkout used by 'Build from source'. Empty = the first build root.",
    source_build_script: "Script run as <script> <checkout> <tag> <out dir> for source builds.",
    hf_token: "Hugging Face token, only used to read generation_config.json from gated repos. HF_TOKEN in the environment overrides it.",
    integrated_name_patterns: "Device names containing any of these are classified as the iGPU and never bound (one per line).",
    profile_dir: "Where profile JSON files live.",
    runs_dir: "Run state and captured server logs.",
    runtimes: "Extra runtimes by hand, JSON list: [{\"name\":\"x\",\"dirs\":[\"C:\\\\a\\\\bin\"],\"version\":\"9.9\"}]",
  };

  async function load() {
    error = "";
    try {
      const [c, r] = await Promise.all([api("get_config"), api("list_runtimes").catch(() => [])]);
      cfg = c.config;
      path = c.path;
      runtimes = r;
      buildRootsText = lines(cfg.build_roots);
      modelRootsText = lines(cfg.model_roots);
      igpuText = lines(cfg.integrated_name_patterns);
      manualRuntimesText = JSON.stringify(cfg.runtimes ?? [], null, 2);
    } catch (e) { error = String(e); }
  }
  load();

  function assembled() {
    const out = JSON.parse(JSON.stringify(cfg));
    out.build_roots = unlines(buildRootsText);
    out.model_roots = unlines(modelRootsText);
    out.integrated_name_patterns = unlines(igpuText);
    for (const k of ["rocm_bin", "install_root", "llama_cpp_source", "source_build_script", "hf_token", "default_runtime"])
      if (out[k] === "" || out[k] === undefined) out[k] = null;
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
      saved = "Saved. Caches cleared — Profiles and Devices will rescan.";
      log("settings saved");
      await load();
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
</script>

<h1>Settings <span class="sub">the tool's own configuration — not a model's, not a server's</span></h1>
<p class="lede">Stored at <span class="mono">{path}</span>. Hover any label for what it does. Saving clears the scan and device caches.</p>

{#if error}<div class="card"><span class="chip block">error</span> <span class="mono" style="font-size: 12px;">{error}</span></div>{/if}
{#if saved}<div class="card"><span class="chip pass">ok</span> {saved}</div>{/if}

{#if cfg}
  <div class="card">
    <div style="font-weight: 700; font-size: 12.5px; margin-bottom: 8px;">Where things are</div>
    <div class="formgrid">
      <label class="field" style="grid-column: span 3;" title={HINTS.model_roots}>
        <span class="k">model roots (one per line)</span>
        <textarea rows="3" bind:value={modelRootsText}></textarea>
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.build_roots}>
        <span class="k">llama.cpp build roots (one per line)</span>
        <textarea rows="3" bind:value={buildRootsText}></textarea>
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.install_root}>
        <span class="k">install root for updates</span>
        <input bind:value={cfg.install_root} placeholder="(first build root)" />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.llama_cpp_source}>
        <span class="k">llama.cpp source checkout</span>
        <input bind:value={cfg.llama_cpp_source} placeholder="(first build root)" />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.source_build_script}>
        <span class="k">source build script</span>
        <input bind:value={cfg.source_build_script} />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.profile_dir}>
        <span class="k">profile directory</span>
        <input bind:value={cfg.profile_dir} />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.runs_dir}>
        <span class="k">runs directory (state + logs)</span>
        <input bind:value={cfg.runs_dir} />
      </label>
    </div>
  </div>

  <div class="card">
    <div style="font-weight: 700; font-size: 12.5px; margin-bottom: 8px;">ROCm runtime</div>
    <div class="formgrid">
      <label class="field" style="grid-column: span 3;" title={HINTS.rocm_bin}>
        <span class="k">default runtime DLL folder (rocm_bin)</span>
        <input bind:value={cfg.rocm_bin} />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.default_runtime}>
        <span class="k">default runtime by name</span>
        <select bind:value={cfg.default_runtime}>
          <option value={null}>default (rocm_bin above)</option>
          {#each runtimes.filter((r) => r.name !== "default") as r}
            <option value={r.name} disabled={!r.available}>{r.name}{r.version ? ` · ${r.version}` : ""}{r.available ? "" : " (missing)"}</option>
          {/each}
        </select>
      </label>
      <label class="field" style="grid-column: 1 / -1;" title={HINTS.runtimes}>
        <span class="k">manual runtimes (JSON)</span>
        <textarea rows="3" bind:value={manualRuntimesText} spellcheck="false"></textarea>
      </label>
    </div>
    <div class="faint" style="font-size: 11px; margin-top: 6px;">Discovered right now: {runtimes.map((r) => r.name + (r.available ? "" : " (missing)")).join(" · ") || "none"}</div>
  </div>

  <div class="card">
    <div style="font-weight: 700; font-size: 12.5px; margin-bottom: 8px;">Devices and Hugging Face</div>
    <div class="formgrid">
      <label class="field" style="grid-column: span 3;" title={HINTS.integrated_name_patterns}>
        <span class="k">iGPU name patterns (one per line)</span>
        <textarea rows="2" bind:value={igpuText}></textarea>
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.hf_token}>
        <span class="k">hugging face token (gated repos only)</span>
        <input type="password" bind:value={cfg.hf_token} placeholder="hf_…" autocomplete="off" />
      </label>
    </div>
  </div>

  <div class="toolbar">
    <button class="btn primary" onclick={save} disabled={saving}>{saving ? "Saving…" : "Save settings"}</button>
    <button class="btn" onclick={load} disabled={saving}>Reload</button>
    <div class="grow"></div>
    <button class="btn" onclick={diagnose} disabled={diagBusy}>{diagBusy ? "Running…" : "Run diagnostics"}</button>
  </div>

  {#if diag}
    <div class="card">
      <div style="font-weight: 700; font-size: 12.5px; margin-bottom: 8px;">Diagnostics <span class="faint mono" style="font-weight: 400; font-size: 10.5px;">{diag.ms} ms · also written to ~/.llamactl/ui.log</span></div>
      {#each diag.errors as e}<div><span class="chip block">failed</span> <span class="mono" style="font-size: 11.5px;">{e}</span></div>{/each}
      <table class="grid" style="margin-top: 6px;">
        <tbody>
          <tr><th>builds</th><td class="mono">{diag.builds.length}</td><td class="mono faint" style="font-size: 10.5px;">{diag.builds.map((b) => `${b.tag} ${b.version ?? "broken"}`).join(" · ")}</td></tr>
          <tr><th>models</th><td class="mono">{diag.models.length}</td><td class="mono faint" style="font-size: 10.5px;">{diag.models.filter((m) => m.header_error).length} with header errors</td></tr>
          <tr><th>devices</th><td class="mono">{diag.devices.length}</td><td class="mono faint" style="font-size: 10.5px;">{diag.devices.map((d) => `${d.backend}${d.hip_index} ${d.name}`).join(" · ")}</td></tr>
          <tr><th>runtimes</th><td class="mono">{diag.runtimes.length}</td><td class="mono faint" style="font-size: 10.5px;">{diag.runtimes.map((r) => r.name).join(" · ")}</td></tr>
        </tbody>
      </table>
    </div>
  {/if}
{/if}
