<script>
  import { api, log } from "../api.js";
  import PathList from "../components/PathList.svelte";
  import Range from "../components/Range.svelte";
  import { scale, fade } from "svelte/transition";
  import { LAYOUT } from "../motion.js";
  import { chat, deleteAllSaved } from "../lib/chat.svelte.js";

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

  // One sentence per hover; the config.json key or flag last, in parentheses.
  const HINTS = {
    model_roots: "Folders scanned recursively for *.gguf; LM Studio's download folder is found on first run (model_roots)",
    build_roots: "Folders scanned one level deep for bin\\llama-server.exe; the install root is scanned too (build_roots)",
    install_root: "Where Updates puts builds and runtimes; empty = the first build folder, or the tool's own (install_root)",
    llama_cpp_source: "git checkout used by Build from source; empty = the first build folder (llama_cpp_source)",
    source_build_script: "Run as <script> <checkout> <tag> <out dir>; empty = scripts\\build-from-tag.bat beside the exe",
    profile_dir: "Where profile JSON files live (profile_dir)",
    runs_dir: "Run state and captured server logs (runs_dir)",
    rocm_bin: "DLL folder put on PATH when a profile names no runtime and no default is picked; a HIP SDK bin folder (rocm_bin)",
    default_runtime: "Runtime used when a profile names none; install more in Updates (default_runtime)",
    rocm_family: "GPU family for AMD's nightly ROCm index; empty = guessed from your cards, RDNA4 is gfx120X-all (rocm_family)",
    runtimes: "Runtimes added by hand, a JSON list of {name, dirs, version} (runtimes)",
    integrated_name_patterns: "Device names containing any of these count as integrated graphics (integrated_name_patterns)",
    allow_integrated: "Let profiles bind integrated graphics, for an APU-only box or Strix Halo. Off, pre-flight blocks it: beside a discrete card an iGPU is slow and never errors",
    keep_alive: "1-token request every N seconds, keeping the GPU awake when PCIe link power saving cannot be Off; profiles can override",
    hf_token: "Read token for gated repos (Gemma, Llama) in Models and the creator defaults; HF_TOKEN in the environment wins",
    hf_use_cli_token: "When no token is set here, use the one huggingface-cli login saved; off because that file belongs to another tool",
    github_token: "Read-only token for the model wizard's build lookups; without one GitHub allows 60 an hour (GITHUB_TOKEN)",
    save_chats: "Keep each conversation as a JSON file in the tool's chats folder; off, they last until the app closes (save_chats)",
    sglang_venv: "Its bin/python launches every SGLang server and the router; pick one below with Use, or type its path (sglang.venv)",
    sglang_tools_dir: "Where model_router.py and the memory guard live; empty = the copies bundled with the tool (sglang.tools_dir)",
    sglang_env: "Environment every SGLang server and the router get, edited in config.json (sglang.env)",
    sglang_install: "Creates the venv and pip-installs SGLang with torch for that flavor; minutes and gigabytes, pip output in Logs",
  };

  // SGLang engine: the venvs found that import sglang, and an installer.
  let sgInstalls = $state([]);
  let sgScanning = $state(false);
  let sgError = $state("");
  let sgUsing = $state("");         // venv a Use click is switching to
  let sgDir = $state("");
  let sgFlavor = $state("rocm");
  let sgInstalling = $state(false);
  let sgResult = $state(null);      // the row sglang_install returned
  const sgEnvKeys = $derived(Object.keys(cfg?.sglang?.env ?? {}));

  async function sgRescan() {
    sgScanning = true; sgError = "";
    try { sgInstalls = await api("sglang_installs", { roots: null }); }
    catch (e) { sgError = String(e); }
    sgScanning = false;
  }

  async function sgUse(venv) {
    sgUsing = venv; sgError = "";
    try {
      await api("sglang_use", { venv });
      log(`settings: sglang venv -> ${venv}`);
      // The command saved config.json itself: take its sglang block so a
      // later Save here does not write the old venv back (other unsaved
      // edits on this page are kept).
      const c = await api("get_config");
      if (cfg) cfg.sglang = c.config.sglang ?? null;
      await sgRescan();
    } catch (e) { sgError = String(e); }
    sgUsing = "";
  }

  async function sgInstall() {
    const dir = sgDir.trim();
    if (!dir) { sgError = "Give the venv a directory first."; return; }
    sgInstalling = true; sgError = ""; sgResult = null;
    log(`settings: sglang install ${sgFlavor} into ${dir}`);
    try {
      sgResult = await api("sglang_install", { dir, flavor: sgFlavor });
      await sgRescan();
    } catch (e) { sgError = String(e); }
    sgInstalling = false;
  }

  function sgEdit() {
    // Typing a venv by hand: the config object exists once there is one.
    cfg.sglang ??= { venv: "", pythonpath: [], env: {} };
    touch();
  }

  // Delete every saved conversation: two clicks, the first arms it.
  let chatsArmed = $state(false);
  let chatsNote = $state("");
  async function deleteChats() {
    if (!chatsArmed) {
      chatsArmed = true;
      setTimeout(() => (chatsArmed = false), 3000);
      return;
    }
    chatsArmed = false;
    try {
      const n = await deleteAllSaved();
      chatsNote = n === 1 ? "Deleted 1 saved conversation." : "Deleted " + n + " saved conversations.";
      log("settings: deleted saved chats");
    } catch (e) {
      chatsNote = String(e);
    }
    setTimeout(() => (chatsNote = ""), 4000);
  }

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
  sgRescan();

  const touch = () => (dirty = true);

  function assembled() {
    const out = JSON.parse(JSON.stringify(cfg));
    const clean = (arr) => (arr ?? []).map((s) => String(s).trim()).filter(Boolean);
    out.build_roots = clean(out.build_roots);
    out.model_roots = clean(out.model_roots);
    out.integrated_name_patterns = clean(out.integrated_name_patterns);
    for (const k of ["rocm_bin", "install_root", "llama_cpp_source", "source_build_script", "hf_token", "github_token", "default_runtime", "rocm_family"])
      if (out[k] === "" || out[k] === undefined) out[k] = null;
    out.keep_alive_seconds = Math.max(0, Math.round(Number(out.keep_alive_seconds) || 0));
    out.allow_integrated = !!out.allow_integrated;
    out.save_chats = out.save_chats !== false;
    out.hf_use_cli_token = !!out.hf_use_cli_token;
    if (out.default_runtime === "default") out.default_runtime = null;
    // SGLang host: no venv = no engine configured (the field is optional in
    // config.json); tools_dir and cwd empty = the defaults.
    if (out.sglang) {
      const venv = String(out.sglang.venv ?? "").trim();
      if (!venv) out.sglang = null;
      else {
        out.sglang.venv = venv;
        for (const k of ["tools_dir", "cwd"]) if (!String(out.sglang[k] ?? "").trim()) delete out.sglang[k]; else out.sglang[k] = String(out.sglang[k]).trim();
        out.sglang.pythonpath = (out.sglang.pythonpath ?? []).map((s) => String(s).trim()).filter(Boolean);
        out.sglang.env ??= {};
      }
    }
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
      chat.saving = c.save_chats;
      saved = "saved";
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
  <span class="sub">Hover a label for help.</span>
</h1>

{#if cfg}
  <div class="savebar">
    <span class="path">{path}</span>
    {#if saved}<span class="chip pass" title="Models, builds and devices rescan now" in:scale={{ duration: LAYOUT, start: 0.7 }} out:fade={{ duration: LAYOUT }}>{saved}</span>
    {:else if dirty}<span class="chip warn" title="Edits not yet written to config.json" in:scale={{ duration: LAYOUT, start: 0.7 }}>unsaved</span>{/if}
    {#if error}<span class="chip block shake">{error}</span>{/if}
    <span style="margin-left: auto; display: flex; gap: 8px;">
      <button class="btn" onclick={load} disabled={saving}>Reload</button>
      <button class="btn primary" onclick={save} disabled={saving}>{saving ? "Saving…" : "Save settings"}</button>
    </span>
  </div>

  <section class="card">
    <div class="sec">Folders <span class="faint">where models and builds are found</span></div>
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
        <span class="k">install root</span>
        <input bind:value={cfg.install_root} oninput={touch} placeholder="(first build folder, or the tool's own folder)" />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.profile_dir}>
        <span class="k">profile folder</span>
        <input bind:value={cfg.profile_dir} oninput={touch} />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.runs_dir}>
        <span class="k">runs folder</span>
        <input bind:value={cfg.runs_dir} oninput={touch} />
      </label>
    </div>
  </section>

  <section class="card">
    <div class="sec">ROCm <span class="faint">which runtime a server loads</span></div>
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
        <span class="k">GPU family</span>
        <input bind:value={cfg.rocm_family} oninput={touch} placeholder="guessed from your cards" list="rocm-families" />
        <datalist id="rocm-families"><option value="gfx120X-all"></option><option value="gfx110X-all"></option><option value="gfx103X-all"></option><option value="gfx1151"></option><option value="gfx1150"></option></datalist>
      </label>
      <label class="field" style="grid-column: span 4;" title={HINTS.runtimes}>
        <span class="k">manual runtimes</span>
        <textarea rows="2" bind:value={manualRuntimesText} oninput={touch} spellcheck="false"></textarea>
      </label>
    </div>
    <div class="faint small" style="margin-top: 10px;" title="Runtimes found now, newest first; install more in Updates">
      found: {runtimes.map((r) => r.name + (r.is_latest ? " (latest)" : "") + (r.available ? "" : " (missing)")).join(" · ") || "none"}
    </div>
  </section>

  <section class="card">
    <div class="sec">GPUs <span class="faint">{devices.length ? `${devices.length} adapters, ${igpus.length} integrated` : "none seen yet"}</span></div>
    <div class="grid2">
      <label class="field" title={HINTS.integrated_name_patterns}>
        <span class="k">integrated name patterns</span>
        <PathList bind:value={cfg.integrated_name_patterns} placeholder="Radeon(TM) Graphics" addLabel="Add pattern" mono={false} />
      </label>
      <div style="display: flex; flex-direction: column; gap: 14px;">
        <label class="field" title={HINTS.allow_integrated}>
          <span class="k">integrated graphics</span>
          <span><input type="checkbox" checked={!!cfg.allow_integrated} onchange={(e) => { cfg.allow_integrated = e.target.checked; touch(); }} /> allow profiles to bind it</span>
        </label>
        <div class="formgrid" style="grid-template-columns: 1fr;">
          <Range bind:value={cfg.keep_alive_seconds} label="keep-alive, seconds (0 = off)" title={HINTS.keep_alive} min={0} max={60} step={1} span={1} onchange={touch} />
        </div>
      </div>
    </div>
    {#if igpus.length}
      <div class="faint small" style="margin-top: 10px;" title="Devices the patterns above classify as integrated">integrated now: {igpus.map((d) => d.name).join(", ")}</div>
    {/if}
  </section>

  <section class="card">
    <div class="sec">Sources <span class="faint">source builds, Hugging Face, GitHub</span></div>
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
        <span class="k">Hugging Face token</span>
        <input type="password" bind:value={cfg.hf_token} oninput={touch} placeholder="hf_…" autocomplete="off" />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.hf_use_cli_token}>
        <span class="k">Hugging Face CLI login</span>
        <span><input type="checkbox" checked={!!cfg.hf_use_cli_token} onchange={(e) => { cfg.hf_use_cli_token = e.target.checked; touch(); }} /> use the CLI's saved token</span>
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.github_token}>
        <span class="k">GitHub token</span>
        <input type="password" bind:value={cfg.github_token} oninput={touch} placeholder="github_pat_… or ghp_…" autocomplete="off" />
      </label>
    </div>
  </section>

  <section class="card">
    <div class="sec">
      SGLang
      <span class="faint">the Python engine behind SGLang profiles (Linux)</span>
      <span style="margin-left: auto;"><button class="btn" onclick={sgRescan} disabled={sgScanning || sgInstalling}>{sgScanning ? "Scanning…" : "Rescan"}</button></span>
    </div>
    <div class="formgrid">
      <label class="field" style="grid-column: span 3;" title={HINTS.sglang_venv}>
        <span class="k">Python environment</span>
        <input value={cfg.sglang?.venv ?? ""} oninput={(e) => { sgEdit(); cfg.sglang.venv = e.target.value; }} placeholder="/home/me/venvs/sglang (none configured)" spellcheck="false" />
      </label>
      <label class="field" style="grid-column: span 3;" title={HINTS.sglang_tools_dir}>
        <span class="k">Router and guard scripts</span>
        <input value={cfg.sglang?.tools_dir ?? ""} oninput={(e) => { sgEdit(); cfg.sglang.tools_dir = e.target.value; }} placeholder="(bundled copies)" spellcheck="false" />
      </label>
      <div class="field" style="grid-column: 1 / -1;" title={HINTS.sglang_env}>
        <span class="k">environment</span>
        <span class="faint small">
          {#if sgEnvKeys.length}<span class="mono">{sgEnvKeys.join(", ")}</span>
          {:else}none{/if}
          {#if cfg.sglang?.pythonpath?.length} · PYTHONPATH +{cfg.sglang.pythonpath.length}{/if}
          {#if cfg.sglang?.cwd} · cwd <span class="mono">{cfg.sglang.cwd}</span>{/if}
        </span>
      </div>
    </div>

    {#if sgError}<div class="notice" style="margin-top: 12px;" transition:fade={{ duration: LAYOUT }}><span class="chip block">error</span> <span class="mono">{sgError}</span></div>{/if}

    <div class="k small" style="margin: 14px 0 6px;" title="Environments that import sglang, under the config dir, model and build folders, and ~/venvs">installs found</div>
    <table class="grid sgtab">
      <thead><tr><th>path</th><th>version</th><th>torch</th><th>device</th><th></th></tr></thead>
      <tbody>
        {#each sgInstalls as i (i.venv)}
          <tr class:on={i.configured}>
            <td><span class="mono">{i.venv}</span>{#if i.configured} <span class="chip pass">configured</span>{/if}</td>
            <td class="mono">{i.sglang_version ?? "—"}</td>
            <td class="mono">{i.torch_version ?? "—"}</td>
            <td class="mono">{i.device ?? "cpu"}</td>
            <td class="r">
              {#if !i.configured}
                <button class="btn small" onclick={() => sgUse(i.venv)} disabled={!!sgUsing || sgInstalling} title="Point profiles at this environment; saves config.json">{sgUsing === i.venv ? "…" : "Use"}</button>
              {/if}
            </td>
          </tr>
        {:else}
          <tr><td colspan="5"><div class="empty" style="padding: 8px 0;">{sgScanning ? "Scanning…" : "None found — install one below, or type a path above"}</div></td></tr>
        {/each}
      </tbody>
    </table>

    <div class="k small" style="margin: 14px 0 6px;">install <span class="faint">a new environment</span></div>
    <div class="sginstall">
      <input class="mono" bind:value={sgDir} placeholder="/home/me/venvs/sglang-rocm" spellcheck="false" disabled={sgInstalling} title="Directory the new environment is created in"
        onkeydown={(e) => { if (e.key === "Enter" && !sgInstalling) sgInstall(); }} />
      <select bind:value={sgFlavor} disabled={sgInstalling} title="Which torch build pip installs">
        <option value="rocm">AMD ROCm</option>
        <option value="cuda">NVIDIA CUDA</option>
        <option value="cpu">CPU only</option>
      </select>
      <button class="btn primary" onclick={sgInstall} disabled={sgInstalling || !sgDir.trim()} title={HINTS.sglang_install}>
        {#if sgInstalling}<span class="spinner"></span> installing…{:else}Install{/if}
      </button>
    </div>
    {#if sgResult}
      <div class="notice" style="margin-top: 10px;" in:scale={{ duration: LAYOUT, start: 0.95 }}>
        <span class="chip pass">installed</span>
        <span><span class="mono">{sgResult.venv}</span> · sglang {sgResult.sglang_version ?? "?"} · torch {sgResult.torch_version ?? "?"} · {sgResult.device ?? "cpu"}
          {#if !sgResult.configured}<button class="link" onclick={() => sgUse(sgResult.venv)} disabled={!!sgUsing}>Use it</button>{/if}</span>
      </div>
    {/if}
  </section>

  <section class="card">
    <div class="sec">Chat <span class="faint">conversations in the Chat tab</span></div>
    <div class="formgrid" style="align-items: end;">
      <label class="field" style="grid-column: span 4;" title={HINTS.save_chats}>
        <span class="k">save chats</span>
        <span><input type="checkbox" checked={cfg.save_chats !== false} onchange={(e) => { cfg.save_chats = e.target.checked; touch(); }} /> keep conversations on this PC</span>
      </label>
      <div style="grid-column: span 2; display: flex; gap: 10px; align-items: center; justify-content: flex-end; flex-wrap: wrap;">
        {#if chatsNote}<span class="faint small">{chatsNote}</span>{/if}
        <button class="btn danger" onclick={deleteChats}>{chatsArmed ? "Click again to delete them all" : "Delete saved chats"}</button>
      </div>
    </div>
  </section>

  <section class="card">
    <div class="sec">
      Diagnostics
      <span class="faint">rescan and report what the tool sees</span>
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
      <div class="faint small" style="margin-top: 10px;" title="The same numbers went to ui.log in the tool's folder">{diag.ms} ms</div>
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
  .k { font-weight: 600; color: var(--ink-muted); }
  .sgtab td.r { text-align: right; }
  .sgtab tr.on td { background: var(--accent-soft); }
  .sginstall { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .sginstall input { flex: 1; min-width: 260px; }
</style>
