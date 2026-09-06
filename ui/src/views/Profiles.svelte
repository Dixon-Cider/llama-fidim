<script>
  import { api, log } from "../api.js";
  import Range from "../components/Range.svelte";

  let profiles = $state([]);
  let devices = $state([]);
  let runtimes = $state([]);
  let builds = $state([]);
  let models = $state([]);
  let selectedId = $state(null);
  let draft = $state(null); // deep-copied profile being edited
  let check = $state(null); // live_check result
  let checking = $state(false);
  let busy = $state("");
  let toast = $state(null);
  let creator = $state(null); // creator_defaults result or { error }
  let creatorBusy = $state(false);
  let checkTimer = null;

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
    runtimes = r;
    builds = s.builds ?? [];
    models = (s.models ?? []).slice().sort((a, b) => a.path.localeCompare(b.path));
    loadErrors = errs;
    loading = false;
    if (!selectedId && profiles.length) select(profiles[0].profile.id);
  }
  load();

  async function rescan() {
    busy = "scan";
    try {
      const s = await api("scan", { refresh: true });
      builds = s.builds ?? [];
      models = (s.models ?? []).slice().sort((a, b) => a.path.localeCompare(b.path));
      toastMsg(`Found ${builds.length} builds, ${models.length} models`);
    } catch (e) { toastMsg(String(e), true); }
    busy = "";
  }

  // ---- model-derived limits ------------------------------------------------
  const selectedModel = $derived(models.find((m) => samePath(m.path, draft?.model?.path)));
  const header = $derived(selectedModel?.header ?? null);
  const ctxMax = $derived(header?.context_length ?? 262144);
  const layerMax = $derived(header?.block_count ?? 99);
  const selectedBuild = $derived(builds.find((b) => samePath(b.path, draft?.build?.path)));

  function samePath(a, b) {
    if (!a || !b) return false;
    return String(a).replace(/\//g, "\\").replace(/\\+$/, "").toLowerCase() ===
           String(b).replace(/\//g, "\\").replace(/\\+$/, "").toLowerCase();
  }
  function base(p) { return String(p ?? "").split(/[\\/]/).pop(); }
  function modelLabel(m) {
    const h = m.header;
    const bits = [base(m.path)];
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
    if (h?.context_length && draft.runtime.ctx_total > h.context_length) draft.runtime.ctx_total = h.context_length;
    if (h?.block_count && draft.runtime.n_gpu_layers > h.block_count) draft.runtime.n_gpu_layers = h.block_count;
    creator = null;
    scheduleCheck();
  }
  function onBuildPick(path) {
    draft.build.path = path;
    draft.build.version = builds.find((b) => samePath(b.path, path))?.version ?? null;
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
      log(`select ${id}: model=${base(draft?.model?.path)} build=${draft?.build?.version} ctx=${draft?.runtime?.ctx_total} slots=${draft?.runtime?.slots} kv=${draft?.runtime?.kv_type_k} spec=${draft?.speculative?.mode} port=${draft?.server?.port}`);
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
    const newest = builds.filter((b) => b.version).sort((a, b) => (b.version > a.version ? 1 : -1))[0];
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
    scheduleCheck();
  }

  function duplicate() {
    if (!draft) return;
    draft = { ...JSON.parse(JSON.stringify(draft)), id: draft.id + "-copy", baseline: null };
    draft.server.port += 1;
    draft.server.alias = draft.id;
    selectedId = null;
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
    return copy;
  }

  function toggleDevice(dev) {
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
      toastMsg(`Saved ${draft.id}`);
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
  <span class="sub">A saved launch: model, build, GPU placement and server flags. Pre-flight here is the same check launch runs.</span>
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
    <span class="chip warn">nothing to pick from</span>
    <span>
      {models.length} models · {builds.length} builds · {devices.length} GPUs.
      Set the model and build roots on the <b>Settings</b> tab{models.length ? "" : " (no GGUF files found)"}{!builds.length ? (models.length ? "" : "; ") + " no bin\\llama-server.exe found" : ""}.
    </span>
  </div>
{/if}

<div class="pf">
  <!-- list -->
  <aside class="list">
    <div class="toolbar" style="margin-bottom: 10px;">
      <button class="btn" onclick={newProfile}>New</button>
      <button class="btn" onclick={duplicate} disabled={!draft}>Duplicate</button>
      <div class="grow"></div>
      <button class="btn small" onclick={rescan} disabled={busy === "scan"} title="rescan build and model roots">{busy === "scan" ? "…" : "Rescan"}</button>
    </div>
    <div class="rows">
      {#each profiles as row}
        {@const p = row.profile}
        <button class="row" class:active={selectedId === p.id} onclick={() => select(p.id)}>
          <div class="r1"><span class="id">{p.id}</span><span class="mono faint">:{p.server.port}</span></div>
          <div class="r2">{base(p.model.path) || "no model"}</div>
          <div class="r3">
            <span class="mono">{p.build.version ?? "?"}</span>
            <span>{p.devices.length} GPU{p.devices.length === 1 ? "" : "s"}{p.split_mode ? `, ${p.split_mode} split` : ""}</span>
            {#if p.baseline}<span class="tok num">{p.baseline.serial_tok_s} tok/s</span>{/if}
            {#if row.findings.length}<span class="chip warn">{row.findings.length}</span>{/if}
          </div>
        </button>
      {:else}
        <div class="empty">No profiles yet. New, or <span class="mono">fidim seed</span>.</div>
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
          <span class="meta"><span class="mono muted">{draft.id} · :{draft.server.port} · {draft.server.alias}</span>{#if !selectedId}<span class="chip accent">unsaved</span>{/if}
          <span class="pfsum" title="pre-flight, re-run on every edit">
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
          <button class="btn" onclick={() => launch(false)} disabled={!!busy || anyBlock}>{busy === "launch" ? "Loading…" : "Save & load"}</button>
          {#if anyBlock}
            <button class="btn danger" onclick={() => launch(true)} disabled={!!busy}>Override blocks &amp; load</button>
          {/if}
          <button class="btn" onclick={() => exportScript("bat")} disabled={!!busy} title="write a standalone .bat that launches this profile">Export .bat</button>
          <button class="btn" onclick={() => exportScript("ps1")} disabled={!!busy} title="write a standalone .ps1 that launches this profile">Export .ps1</button>
          <button class="btn danger" onclick={remove} disabled={!!busy || !selectedId}>Delete</button>
        </div>
      </div>

      <!-- identity -->
      <section class="card">
        <div class="sec">Identity <span class="faint">names the CLI, the router and clients use</span></div>
        <div class="formgrid">
          <label class="field" title="Profile identifier: the file name under the profile directory and what the CLI uses (fidim launch <id>). Letters, digits, dashes."><span class="k">id</span><input bind:value={draft.id} oninput={scheduleCheck} /></label>
          <label class="field" style="grid-column: span 2;" title="Free-text display name."><span class="k">name</span><input bind:value={draft.name} /></label>
          <label class="field" title="TCP port the server listens on. Each running profile needs its own; pre-flight checks it is free."><span class="k">port</span><input type="number" bind:value={draft.server.port} oninput={scheduleCheck} /></label>
          <label class="field" style="grid-column: span 2;" title="Model name the server reports on /v1/models and what clients pass as 'model'. Unique across running servers; the router uses it as the model id."><span class="k">alias</span><input bind:value={draft.server.alias} oninput={scheduleCheck} /></label>
        </div>
      </section>

      <!-- model + build -->
      <section class="card">
        <div class="sec">Model <span class="faint">{models.length} GGUF files under the model roots</span></div>
        <div class="formgrid">
          <label class="field" style="grid-column: 1 / -1;">
            <span class="k">weights</span>
            <select value={models.find((m) => samePath(m.path, draft.model.path))?.path ?? ""} onchange={(e) => onModelPick(e.target.value)}>
              <option value="" disabled>choose a model…</option>
              {#each models as m}
                <option value={m.path}>{modelLabel(m)}</option>
              {/each}
            </select>
          </label>
          {#if header}
            <div class="facts" style="grid-column: 1 / -1;">
              <span><b>{header.model_name ?? base(draft.model.path)}</b></span>
              <span>{header.architecture}</span>
              {#if header.size_label}<span>{header.size_label}</span>{/if}
              <span>{header.block_count} layers</span>
              <span>trained context {fmtInt(header.context_length ?? 0)}</span>
              {#if header.source_repo}<span title="from GGUF general.base_model">{header.source_repo}</span>{/if}
              {#if mtpBuiltIn}<span class="chip pass">MTP built in</span>{/if}
              <span class="path" style="flex-basis: 100%;">{draft.model.path}</span>
            </div>
          {:else if draft.model.path}
            <div class="facts" style="grid-column: 1 / -1;"><span class="chip warn">not in scan</span><span class="path">{draft.model.path}</span></div>
          {/if}
          <label class="field" style="grid-column: span 3;" title="Multimodal projector paired with the weights; lets the server read images.">
            <span class="k">vision projector (mmproj)</span>
            <select bind:value={draft.model.mmproj} onchange={scheduleCheck}>
              <option value={null}>none</option>
              {#each selectedModel?.mmproj_candidates ?? [] as c}<option value={c}>{base(c)}</option>{/each}
              {#if draft.model.mmproj && !(selectedModel?.mmproj_candidates ?? []).includes(draft.model.mmproj)}
                <option value={draft.model.mmproj}>{base(draft.model.mmproj)}</option>
              {/if}
            </select>
          </label>
          <label class="field" style="grid-column: span 3;" title="A small draft model or MTP sidecar file for speculative decoding. Turned on in the Speculative decoding section.">
            <span class="k">speculative draft file</span>
            <select value={draft.model.draft?.path ?? ""} onchange={(e) => onDraftPick(e.target.value)}>
              <option value="">none</option>
              {#each selectedModel?.draft_candidates ?? [] as c}<option value={c}>{base(c)}</option>{/each}
              {#if draft.model.draft?.path && !(selectedModel?.draft_candidates ?? []).includes(draft.model.draft.path)}
                <option value={draft.model.draft.path}>{base(draft.model.draft.path)}</option>
              {/if}
            </select>
          </label>
          <label class="field" style="grid-column: span 3;" title="Which llama.cpp build launches this profile. Updates installs new builds side by side and can promote profiles to them.">
            <span class="k">llama.cpp build</span>
            <select value={selectedBuild?.path ?? ""} onchange={(e) => onBuildPick(e.target.value)}>
              <option value="" disabled>choose a build…</option>
              {#each builds as b}
                <option value={b.path} disabled={!!b.version_error}>{b.tag} · {b.version ?? "broken"}{b.version_error ? " (does not run)" : ""}</option>
              {/each}
              {#if draft.build.path && !selectedBuild}<option value={draft.build.path}>{draft.build.path} (not in scan)</option>{/if}
            </select>
          </label>
          <label class="field" style="grid-column: span 3;" title="ROCm runtime this server runs against; its DLL folders go first on PATH. Newest first; install more from the Updates tab.">
            <span class="k">ROCm runtime</span>
            <select bind:value={draft.rocm_runtime} onchange={scheduleCheck}>
              <option value={null}>config default ({runtimes.find((r) => r.is_default)?.name ?? "default"}{runtimes.find((r) => r.is_default)?.version ? ` · ${runtimes.find((r) => r.is_default).version}` : ""})</option>
              {#each runtimes.filter((r) => !r.is_default) as r}
                <option value={r.name} disabled={!r.available}>{r.name}{r.version ? ` · ${r.version}` : ""}{r.is_latest ? " (latest)" : ""}{r.available ? "" : " (missing)"}</option>
              {/each}
            </select>
          </label>
        </div>
      </section>

      <!-- devices -->
      <section class="card">
        <div class="sec">GPU placement <span class="faint">one card, or a layer split across two</span></div>
        <div class="devrows">
          {#each devices.filter((d) => !d.integrated) as dev}
            {@const entry = draft.devices.find((x) => x.key === dev.stable_key)}
            {@const used = dev.total_mib - dev.free_mib}
            <label class="devrow" class:on={!!entry}>
              <input type="checkbox" checked={!!entry} onchange={() => toggleDevice(dev)} />
              <span class="dname">
                <b>{dev.name.replace(/^AMD /, "")}</b>
                <span class="mono faint">{dev.stable_key.split(":").pop()} · {dev.backend}{dev.hip_index}</span>
                {#if dev.display}<span class="chip warn">display attached</span>{/if}
              </span>
              <span class="dmeter">
                <span class="meter"><span class="fill {used / dev.total_mib > 0.9 ? 'block' : used / dev.total_mib > 0.7 ? 'warn' : 'pass'}" style="width: {Math.round(100 * used / Math.max(1, dev.total_mib))}%;"></span></span>
                <span class="mono faint num">{(dev.free_mib / 1024).toFixed(1)} of {(dev.total_mib / 1024).toFixed(0)} GiB free</span>
              </span>
              {#if entry && draft.devices.length > 1}
                <span class="frac" title="Share of the layers placed on this card. Blank = split evenly.">
                  <span class="k">fraction</span>
                  <input type="number" step="0.05" min="0" max="1" placeholder="auto" bind:value={entry.split_fraction} oninput={scheduleCheck} />
                </span>
              {/if}
            </label>
          {/each}
        </div>
        {#if draft.devices.length > 1}
          <div class="formgrid" style="margin-top: 14px;">
            <label class="field" style="grid-column: span 2;" title="layer: whole layers per card (supported, no cross-GPU collectives). row: split each tensor across cards (experimental on this stack)."><span class="k">split mode</span>
              <select bind:value={draft.split_mode} onchange={scheduleCheck}>
                <option value="layer">layer (supported)</option>
                <option value="row">row (experimental)</option>
              </select>
            </label>
            <label class="field" style="grid-column: span 2;" title="Which selected card holds the KV cache and small tensors — index into the checked cards above, in order."><span class="k">main device (list index)</span>
              <input type="number" min="0" bind:value={draft.main_device} oninput={scheduleCheck} />
            </label>
          </div>
        {/if}
        {#if budgets.length}
          <div class="budget">
            {#each budgets as b}
              <div>
                <div class="cap">
                  <span>estimated need on {b.name.replace(/^AMD /, "")} <span class="faint">({b.key.split(":").pop()})</span></span>
                  <span><span class="num">{b.estGib.toFixed(2)}</span> of <span class="num">{b.freeGib.toFixed(2)}</span> GiB free · {Math.round(b.frac * 100)}%</span>
                </div>
                <div class="bar" title={b.detail}>
                  <div class="fill {b.kind}" style="width: {Math.min(100, b.frac * 100)}%;"></div>
                  <div class="mark" style="left: 90%;"></div>
                </div>
                <div class="faint small" style="margin-top: 4px;">{b.detail}</div>
              </div>
            {/each}
          </div>
        {/if}
      </section>

      <!-- context & offload -->
      <section class="card">
        <div class="sec">Context and offload <span class="faint">allocated when the server starts</span></div>
        <div class="formgrid">
          <Range bind:value={draft.runtime.ctx_total} label="context length" title="Total tokens of context, split across slots. On sliding-window models VRAM barely grows with context; on dense models it grows linearly." min={512} max={ctxMax} step={256}
            hint={header ? `model supports up to ${fmtInt(ctxMax)} tokens` : "no model header — default cap"} format={fmtInt} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.runtime.n_gpu_layers} label="GPU offload (layers)" title="How many transformer layers live on the GPU. Anything at or above the model's layer count = everything on GPU (fastest). Lower it only when the model does not fit." min={0} max={layerMax} step={1}
            hint={header ? `${layerMax} layers; ≥ ${layerMax} = all` : "99 = all"} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.runtime.slots} label="max concurrent requests (slots)" title="-np: parallel request slots. Context divides evenly across them." min={1} max={16} step={1}
            hint={`per-slot context = ${fmtInt(Math.floor(draft.runtime.ctx_total / Math.max(1, draft.runtime.slots)))}`} onchange={scheduleCheck} span={3} />
          <label class="field" style="grid-column: span 3; justify-content: end;" title="One shared KV pool across slots instead of a fixed share per slot. Experimental.">
            <span class="k">unified KV cache</span>
            <span><input type="checkbox" bind:checked={draft.runtime.kv_unified} onchange={scheduleCheck} /> one shared KV pool across slots</span>
          </label>
        </div>
      </section>

      <!-- speculative decoding -->
      <section class="card">
        <div class="sec">
          Speculative decoding
          {#if mtpBuiltIn}<span class="chip pass">model supports MTP</span>{/if}
          <span class="faint">guess several tokens per step, verify them in one pass</span>
          <span style="margin-left: auto;"><button class="btn small" onclick={applySpecDefaults} disabled={draft.speculative.mode === "off"} title="engine defaults: 3 max, 0 min, 0.0 probability">Engine defaults</button></span>
        </div>
        <div class="formgrid">
          <label class="field" style="grid-column: span 3;" title="How drafts are produced. MTP uses the model's own multi-token head (built in for Qwen 3.5+/3.8, a sidecar file for Gemma 4). draft = a separate small model. DFlash = a DFlash draft file. n-gram = prompt lookup, no model.">
            <span class="k">mode</span>
            <select value={draft.speculative.mode} onchange={(e) => onSpecMode(e.target.value)}>
              {#each specModes as m}<option value={m.v}>{m.l}</option>{/each}
            </select>
          </label>
          {#if draft.speculative.mode !== "off"}
            <div style="grid-column: span 3;"></div>
            <Range bind:value={draft.speculative.n_max} label="max draft tokens" min={1} max={16} step={1} nullable placeholder={3}
              hint="engine default 3" onchange={scheduleCheck} span={3}
              title="--spec-draft-n-max: how many tokens the draft proposes per step. Higher = more speed when accepted, more waste when rejected." />
            <Range bind:value={draft.speculative.n_min} label="min draft tokens" min={0} max={16} step={1} nullable placeholder={0}
              hint="engine default 0" onchange={scheduleCheck} span={3}
              title="--spec-draft-n-min: skip speculation entirely when fewer than this many draft tokens are available." />
            <Range bind:value={draft.speculative.p_min} label="draft probability" min={0} max={1} step={0.01} nullable placeholder={0}
              hint="engine default 0.00" format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3}
              title="--spec-draft-p-min: only keep draft tokens the draft itself is at least this confident in (greedy). 0 = keep all." />
          {/if}
        </div>
        {#if draft.speculative.mode === "mtp" && !mtpBuiltIn && !draft.model.draft?.path}
          <div class="notice" style="margin-top: 10px;"><span class="chip block">needs a head</span><span>MTP needs either a model with a built-in head or an MTP sidecar chosen above.</span></div>
        {/if}
      </section>

      <!-- inference -->
      <section class="card">
        <div class="sec">
          Inference
          <span class="faint">unchecked = the engine's own default</span>
          <span style="margin-left: auto; display: flex; gap: 8px;">
            <button class="btn small" onclick={fetchCreator} disabled={creatorBusy || !draft.model.path} title="look up generation_config.json on Hugging Face">
              {creatorBusy ? "Fetching…" : embedded ? "Re-check on Hugging Face" : "Creator defaults"}
            </button>
            {#if creatorShown}
              <button class="btn small primary" onclick={applyCreator}>Apply creator defaults</button>
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
                <span class="faint">{creatorShown.embedded ? "embedded in the GGUF header by the converter" : (creatorShown.from_cache ? "cached" : "fetched") + " from generation_config.json"}</span>
              </div>
            {/if}
          </div>
        {/if}
        <div class="formgrid">
          <label class="field" style="grid-column: span 3;" title="Sets enable_thinking in the chat template. Off avoids the agentic loops some models fall into with thinking on; on gives reasoning traces."><span class="k">thinking</span>
            <select bind:value={draft.chat.enable_thinking} onchange={scheduleCheck}>
              <option value={null}>model default</option>
              <option value={true}>on</option>
              <option value={false}>off (recommended for agentic use)</option>
            </select>
          </label>
          <div style="grid-column: span 3;"></div>
          <Range bind:value={draft.sampling.temperature} label="temperature" title="Randomness of sampling. 0 = greedy, 1 = the model's raw distribution. Creator default shown as the hint when known." min={0} max={2} step={0.05} nullable placeholder={creatorShown?.temperature ?? 0.8}
            hint={creatorShown?.temperature != null ? `creator: ${creatorShown.temperature}` : ""} format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.top_k} label="top K sampling" title="Keep only the K most likely tokens before sampling. 0 = disabled." min={0} max={200} step={1} nullable placeholder={creatorShown?.top_k ?? 40}
            hint={creatorShown?.top_k != null ? `creator: ${creatorShown.top_k}` : "0 = off"} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.top_p} label="top P sampling" title="Nucleus sampling: keep the smallest set of tokens whose probabilities sum to P." min={0} max={1} step={0.01} nullable placeholder={creatorShown?.top_p ?? 0.95}
            hint={creatorShown?.top_p != null ? `creator: ${creatorShown.top_p}` : ""} format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.min_p} label="min P sampling" title="Drop tokens whose probability is below P × the top token's probability. Stronger and more stable than top-p at high temperature." min={0} max={1} step={0.01} nullable placeholder={creatorShown?.min_p ?? 0.05}
            hint={creatorShown?.min_p != null ? `creator: ${creatorShown.min_p}` : ""} format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.repeat_penalty} label="repeat penalty" title="Multiplicative penalty on tokens already seen in the context. 1.0 = off; 1.1 is a common mild setting." min={1} max={2} step={0.01} nullable placeholder={creatorShown?.repetition_penalty ?? 1.0}
            hint={creatorShown?.repetition_penalty != null ? `creator: ${creatorShown.repetition_penalty}` : "1.0 = off"} format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.presence_penalty} label="presence penalty" title="Flat penalty on any token that has appeared at all. 0 = off." min={0} max={2} step={0.05} nullable placeholder={0}
            hint="0 = off" format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.sampling.dry_multiplier} label="DRY multiplier" title="DRY (Don't Repeat Yourself) sampler strength: penalises repeating whole sequences, not single tokens. 0 = off." min={0} max={2} step={0.05} nullable placeholder={0}
            hint="0 = off; repetition suppression" format={(v) => Number(v).toFixed(2)} onchange={scheduleCheck} span={3} />
        </div>
      </section>

      <!-- advanced -->
      <section class="card">
        <div class="sec">Advanced <span class="faint">batching, KV storage and pass-through flags</span></div>
        <div class="formgrid">
          <Range bind:value={draft.runtime.batch_logical} label="evaluation batch size (-b)" title="-b: logical batch, the per-iteration token budget shared by prefill and decode." min={64} max={8192} step={64}
            hint="shared per-iteration budget" format={fmtInt} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.runtime.batch_physical} label="physical batch size (-ub)" title="-ub: micro-batch pushed through the GPU; sizes the compute buffer. 256 measured best on an R9700 at long context; larger buys prefill speed at short context." min={32} max={2048} step={32}
            hint="sizes the compute buffer" format={fmtInt} onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.runtime.cache_reuse} label="prompt cache reuse (min chunk)" title="--cache-reuse: reuse KV cache for a prompt that shares a prefix with a previous one, in chunks of at least this many tokens. 0 = off." min={0} max={2048} step={32} nullable placeholder={256}
            hint="--cache-reuse" onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.runtime.threads} label="CPU thread pool size" title="-t: CPU threads for layers not offloaded and for tokenisation. Irrelevant when everything is on the GPU." min={1} max={32} step={1} nullable placeholder={8}
            hint="only matters for layers left on CPU" onchange={scheduleCheck} span={3} />
          <Range bind:value={draft.keep_alive_seconds} label="keep model resident (keep-alive, seconds)" min={0} max={30} step={1} nullable placeholder={keepAliveDefault}
            hint={`config default ${keepAliveDefault} s · 0 = off`} onchange={scheduleCheck} span={3}
            title="Off by default. VRAM eviction on idle comes from the PCIe Link State Power Management power setting (pre-flight check 12). If it cannot be Off, a 1-token request this often keeps the GPU awake; about 70 ms of GPU time per ping." />
          <div style="grid-column: span 3;"></div>
          <label class="field" style="grid-column: span 2;" title="Fused attention kernel: less VRAM, faster prefill. Required for V-cache quantisation. auto lets llama.cpp decide."><span class="k">flash attention</span>
            <select bind:value={draft.runtime.flash_attn} onchange={scheduleCheck}>
              <option value="on">on</option><option value="off">off</option><option value="auto">auto</option>
            </select>
          </label>
          <label class="field" title="Storage type of the attention key cache. q8_0 halves KV VRAM with little visible cost; q4_0 quarters it with some quality cost."><span class="k">K cache quant</span>
            <select bind:value={draft.runtime.kv_type_k} onchange={scheduleCheck}>
              {#each ["f16", "q8_0", "q4_0"] as t}<option value={t}>{t}</option>{/each}
            </select>
          </label>
          <label class="field" title="Storage type of the attention value cache. Needs flash attention on for anything but f16."><span class="k">V cache quant</span>
            <select bind:value={draft.runtime.kv_type_v} onchange={scheduleCheck}>
              {#each ["f16", "q8_0", "q4_0"] as t}<option value={t}>{t}</option>{/each}
            </select>
          </label>
          <label class="field" style="grid-column: span 2;" title="Serve concurrent requests inside one forward pass instead of queueing them. Leave on."><span class="k">continuous batching</span>
            <span><input type="checkbox" bind:checked={draft.runtime.cont_batching} onchange={scheduleCheck} /> on (-cb)</span>
          </label>
          <label class="field" style="grid-column: span 4;" title="Sets LLAMA_SERVER_SLOTS_DEBUG=1 for the server, so /slots carries each slot's last prompt and the text generated so far. The Running tab shows both per slot and flags endless loops. Costs the server one detokenize per poll; the router inherits it from any member. Takes effect on the next load."><span class="k">trace tokens</span>
            <span><input type="checkbox" checked={draft.env?.LLAMA_SERVER_SLOTS_DEBUG === "1"} onchange={(e) => { if (e.target.checked) draft.env.LLAMA_SERVER_SLOTS_DEBUG = "1"; else delete draft.env.LLAMA_SERVER_SLOTS_DEBUG; scheduleCheck(); }} /> show each slot's last prompt and generated text on the Running tab, with loop detection</span>
          </label>
          <label class="field" style="grid-column: 1 / -1;" title="Anything this editor does not model, e.g. --no-mmap. Passed to llama-server unchanged."><span class="k">extra llama-server flags (space-separated, passed through verbatim)</span>
            <input value={Array.isArray(draft.runtime.extra_flags) ? draft.runtime.extra_flags.join(" ") : draft.runtime.extra_flags}
              oninput={(e) => { draft.runtime.extra_flags = e.target.value; scheduleCheck(); }} placeholder="--no-mmap" />
          </label>
        </div>
        <div class="faint small" style="margin-top: 10px;">
          KV quantisation is the big VRAM lever (q8_0 is as good as f16 in practice); flash attention must be on for V-cache quant.
        </div>
      </section>

      <!-- notes -->
      <section class="card">
        <div class="sec">Notes <span class="faint">free text, saved with the profile</span></div>
        <textarea rows="3" bind:value={draft.notes} placeholder="What was measured, what to remember."></textarea>
      </section>

      {#if check?.findings?.length}
        <section class="card">
          <div class="sec">Findings <span class="faint">problems with the profile itself</span></div>
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
            {#each check.results as r}
              {@const kind = outcomeKind(r.outcome)}
              {@const msg = outcomeMsg(r.outcome)}
              <div class="row">
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
              <div class="stat"><span class="v">{draft.baseline.concurrent.decode_aggregate_tok_s ?? draft.baseline.concurrent.aggregate_tok_s}<small>tok/s</small></span><span class="l">aggregate at n = {draft.baseline.concurrent.n}</span></div>
              <div class="stat"><span class="v">{draft.baseline.concurrent.per_stream_tok_s}<small>tok/s</small></span><span class="l">per stream</span></div>
            {/if}
            <div class="stat"><span class="v">{draft.baseline.vram_gb}<small>GiB</small></span><span class="l">resident</span></div>
          </div>
        </section>
      {/if}
    </div>
  {:else}
    <div class="card" style="flex: 1;"><div class="empty">Select a profile on the left, or create one.</div></div>
  {/if}
</div>

{#if toast}
  <div class="toast" class:error={toast.isError}>{toast.text}</div>
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
