<script>
  import { api } from "../api.js";

  let profiles = $state([]);
  let devices = $state([]);
  let runtimes = $state([]);
  let selectedId = $state(null);
  let draft = $state(null); // deep-copied profile being edited
  let check = $state(null); // live_check result
  let checking = $state(false);
  let busy = $state("");
  let toast = $state(null);
  let checkTimer = null;

  const GIB = 1024 * 1024 * 1024;

  async function load() {
    const [p, d, r] = await Promise.all([
      api("list_profiles"),
      api("devices", { refresh: false }),
      api("list_runtimes").catch(() => []),
    ]);
    profiles = p;
    devices = d.map((r) => r.device);
    runtimes = r;
    if (!selectedId && profiles.length) select(profiles[0].profile.id);
  }
  load();

  function select(id) {
    selectedId = id;
    const row = profiles.find((r) => r.profile.id === id);
    draft = row ? JSON.parse(JSON.stringify(row.profile)) : null;
    check = null;
    scheduleCheck();
  }

  function newProfile() {
    const first = devices.find((d) => !d.integrated);
    draft = {
      schema: 1, id: "new-profile", name: "New profile",
      build: { path: "", version: null },
      model: { path: "", mmproj: null, draft: null },
      devices: first ? [{ key: first.stable_key, split_fraction: null, resolved_index_last_launch: null }] : [],
      split_mode: null, main_device: 0, rocm_runtime: null,
      server: { port: 9710, alias: "new-profile", host: "127.0.0.1" },
      runtime: {
        n_gpu_layers: 99, ctx_total: 32768, slots: 1, kv_type_k: "f16", kv_type_v: "f16",
        flash_attn: "on", batch_logical: 2048, batch_physical: 512, cont_batching: true,
        kv_unified: false, cache_reuse: null, extra_flags: [],
      },
      sampling: {}, chat: {}, env: {}, baseline: null, notes: "",
    };
    selectedId = null;
    check = null;
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

  function normalized(p) {
    const copy = JSON.parse(JSON.stringify(p));
    copy.server.port = Number(copy.server.port) || 0;
    const r = copy.runtime;
    for (const k of ["n_gpu_layers", "ctx_total", "slots", "batch_logical", "batch_physical"])
      r[k] = Number(r[k]) || 0;
    if (r.cache_reuse === "" || r.cache_reuse === null) r.cache_reuse = null;
    else r.cache_reuse = Number(r.cache_reuse);
    copy.main_device = Number(copy.main_device) || 0;
    for (const d of copy.devices)
      d.split_fraction = d.split_fraction === null || d.split_fraction === "" ? null : Number(d.split_fraction);
    if (copy.devices.length < 2) copy.split_mode = null;
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

  function outcomeKind(o) {
    if (o === "pass") return "pass";
    if (o.warn !== undefined) return "warn";
    if (o.block !== undefined) return "block";
    if (o.note !== undefined) return "note";
    return "pass";
  }
  function outcomeMsg(o) {
    return o === "pass" ? "" : (o.warn ?? o.block ?? o.note ?? "");
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
        toastMsg("Launch blocked by pre-flight — see the check list; use Override to accept the named risks", true);
        check = { ...check, results: r.results };
      } else {
        const place = (r.placement ?? [])
          .map((p) => `${(Math.max(p.dedicated_bytes ?? 0, p.committed_bytes ?? 0) / GIB).toFixed(1)} GiB on ${p.key.split(":").pop()}`)
          .join(", ");
        toastMsg(`Launched pid ${r.state.pid} on port ${r.state.port}${r.cold_start ? " (cold cache)" : ""} — ${place}`);
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

<h1>Profiles <span class="sub">saved launch configurations — the editor runs the same pre-flight as launch</span></h1>

<div style="display: flex; gap: 16px; align-items: flex-start;">
  <!-- list -->
  <div style="width: 250px; flex: none;">
    <div class="toolbar">
      <button class="btn" onclick={newProfile}>New</button>
      <button class="btn" onclick={duplicate} disabled={!draft}>Duplicate</button>
    </div>
    <div class="card" style="padding: 6px;">
      {#each profiles as row}
        <button
          class="btn"
          style="width: 100%; text-align: left; margin: 2px 0; border-color: {selectedId === row.profile.id ? 'var(--accent)' : 'var(--rule)'};"
          onclick={() => select(row.profile.id)}
        >
          <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <span>{row.profile.id}</span>
            <span class="mono faint" style="font-size: 10px;">:{row.profile.server.port}</span>
          </div>
          <div class="faint" style="font-size: 10.5px; font-weight: 400;">
            {row.profile.devices.length} device{row.profile.devices.length === 1 ? "" : "s"}{row.profile.split_mode ? ` · ${row.profile.split_mode} split` : ""}
            {#if row.profile.baseline}
              · {row.profile.baseline.serial_tok_s} tok/s
            {/if}
          </div>
          {#if row.findings.length}
            <span class="chip warn" style="margin-top: 3px;">{row.findings.length} finding{row.findings.length === 1 ? "" : "s"}</span>
          {/if}
        </button>
      {:else}
        <div class="empty">No profiles — New, or `llamactl seed`</div>
      {/each}
    </div>
  </div>

  <!-- editor -->
  {#if draft}
    <div style="flex: 1; min-width: 0;">
      <div class="card">
        <div class="formgrid">
          <label class="field"><span class="k">id</span><input bind:value={draft.id} oninput={scheduleCheck} /></label>
          <label class="field" style="grid-column: span 2;"><span class="k">name</span><input bind:value={draft.name} /></label>
          <label class="field"><span class="k">port</span><input type="number" bind:value={draft.server.port} oninput={scheduleCheck} /></label>
          <label class="field"><span class="k">alias</span><input bind:value={draft.server.alias} oninput={scheduleCheck} /></label>
        </div>
        <div class="formgrid" style="margin-top: 10px;">
          <label class="field" style="grid-column: span 3;"><span class="k">build path</span><input bind:value={draft.build.path} oninput={scheduleCheck} /></label>
          <label class="field" style="grid-column: span 3;"><span class="k">model path</span><input bind:value={draft.model.path} oninput={scheduleCheck} /></label>
        </div>
        <div class="formgrid" style="margin-top: 10px;">
          <label class="field" style="grid-column: span 3;">
            <span class="k">ROCm runtime</span>
            <select bind:value={draft.rocm_runtime} onchange={scheduleCheck}>
              <option value={null}>config default ({runtimes.find((r) => r.is_default)?.name ?? "default"}{runtimes.find((r) => r.is_default)?.version ? ` · ${runtimes.find((r) => r.is_default).version}` : ""})</option>
              {#each runtimes.filter((r) => !r.is_default) as r}
                <option value={r.name} disabled={!r.available}>{r.name}{r.version ? ` · ${r.version}` : ""}{r.available ? "" : " (missing)"}</option>
              {/each}
            </select>
          </label>
          <div class="faint" style="grid-column: span 3; font-size: 11px; align-self: end;">
            DLL search path the server launches with. Benched 2026-09-02: 7.1 vs 7.14 identical for llama.cpp; leave on default unless a build needs otherwise.
          </div>
        </div>
      </div>

      <!-- device picker (v1.1: multi-select + fractions) -->
      <div class="card">
        <div style="font-weight: 700; font-size: 12.5px; margin-bottom: 8px;">
          Devices
          <span class="faint" style="font-weight: 400;">— multi-select spans one model across cards (layer split)</span>
        </div>
        {#each devices.filter((d) => !d.integrated) as dev}
          {@const entry = draft.devices.find((x) => x.key === dev.stable_key)}
          <div style="display: flex; gap: 10px; align-items: center; padding: 5px 0;">
            <input type="checkbox" style="width: auto;" checked={!!entry} onchange={() => toggleDevice(dev)} />
            <span class="mono" style="flex: 1;">{dev.stable_key}</span>
            <span class="faint mono" style="font-size: 10.5px;">{dev.backend}{dev.hip_index} · {(dev.free_mib / 1024).toFixed(1)} GiB free</span>
            {#if entry && draft.devices.length > 1}
              <label class="field" style="width: 90px;">
                <span class="k">fraction</span>
                <input type="number" step="0.05" min="0" max="1" placeholder="auto"
                  bind:value={entry.split_fraction} oninput={scheduleCheck} />
              </label>
            {/if}
          </div>
        {/each}
        {#if draft.devices.length > 1}
          <div class="formgrid" style="margin-top: 6px;">
            <label class="field"><span class="k">split mode</span>
              <select bind:value={draft.split_mode} onchange={scheduleCheck}>
                <option value="layer">layer (supported)</option>
                <option value="row">row (experimental)</option>
              </select>
            </label>
            <label class="field"><span class="k">main device (list index)</span>
              <input type="number" min="0" bind:value={draft.main_device} oninput={scheduleCheck} />
            </label>
          </div>
        {/if}

        <!-- per-device budget bars -->
        {#if budgets.length}
          <div class="budget">
            {#each budgets as b}
              <div>
                <div class="cap">
                  <span>{b.name} <span class="faint">({b.key.split(":").pop()})</span></span>
                  <span>{b.estGib.toFixed(2)} / {b.freeGib.toFixed(2)} GiB free ({Math.round(b.frac * 100)}%)</span>
                </div>
                <div class="bar" title={b.detail}>
                  <div class="fill {b.kind}" style="width: {Math.min(100, b.frac * 100)}%;"></div>
                  <div class="mark" style="left: 90%;"></div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- runtime -->
      <div class="card">
        <div style="font-weight: 700; font-size: 12.5px; margin-bottom: 8px;">Runtime</div>
        <div class="formgrid">
          <label class="field"><span class="k">ctx total</span><input type="number" bind:value={draft.runtime.ctx_total} oninput={scheduleCheck} /></label>
          <label class="field"><span class="k">slots (-np)</span><input type="number" bind:value={draft.runtime.slots} oninput={scheduleCheck} /></label>
          <label class="field"><span class="k">gpu layers</span><input type="number" bind:value={draft.runtime.n_gpu_layers} oninput={scheduleCheck} /></label>
          <label class="field"><span class="k">kv type K</span>
            <select bind:value={draft.runtime.kv_type_k} onchange={scheduleCheck}>
              {#each ["f16", "q8_0", "q4_0"] as t}<option value={t}>{t}</option>{/each}
            </select>
          </label>
          <label class="field"><span class="k">kv type V</span>
            <select bind:value={draft.runtime.kv_type_v} onchange={scheduleCheck}>
              {#each ["f16", "q8_0", "q4_0"] as t}<option value={t}>{t}</option>{/each}
            </select>
          </label>
          <label class="field"><span class="k">flash attn</span>
            <select bind:value={draft.runtime.flash_attn} onchange={scheduleCheck}>
              <option value="on">on</option><option value="off">off</option><option value="auto">auto</option>
            </select>
          </label>
          <label class="field"><span class="k">batch logical (-b)</span><input type="number" bind:value={draft.runtime.batch_logical} oninput={scheduleCheck} /></label>
          <label class="field"><span class="k">batch physical (-ub)</span><input type="number" bind:value={draft.runtime.batch_physical} oninput={scheduleCheck} /></label>
        </div>
        {#if draft.runtime.slots > 0}
          <div class="faint mono" style="font-size: 10.5px; margin-top: 6px;">
            per-slot context = {Math.floor(draft.runtime.ctx_total / Math.max(1, draft.runtime.slots)).toLocaleString()} tokens
            (context divides across slots — R-10)
          </div>
        {/if}
      </div>

      {#if check?.findings?.length}
        <div class="card">
          {#each check.findings as f}
            <div style="display: flex; gap: 8px; align-items: baseline; padding: 3px 0;">
              <span class="chip {f.severity === 'error' ? 'block' : 'warn'}">{f.severity}</span>
              <span style="font-size: 12px;">{f.message}</span>
            </div>
          {/each}
        </div>
      {/if}

      <!-- live pre-flight -->
      <div class="card">
        <div style="display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px;">
          <span style="font-weight: 700; font-size: 12.5px;">Pre-flight</span>
          {#if checking}<span class="faint mono" style="font-size: 10.5px;">re-checking…</span>{/if}
          {#if check?.error}<span class="chip block">{check.error}</span>{/if}
        </div>
        {#if check?.results}
          <div class="preflight">
            {#each check.results as r}
              {@const kind = outcomeKind(r.outcome)}
              {@const msg = outcomeMsg(r.outcome)}
              <div class="row" class:has-msg={!!msg}>
                <span class="n">{r.spec_number}</span>
                <span class="t">{r.title}</span>
                <span class="chip {kind}">{kind}</span>
                {#if msg}<span class="msg">{msg}</span>{/if}
              </div>
            {/each}
          </div>
          {#if check.command_line}
            <div class="mono faint" style="font-size: 10.5px; margin-top: 8px; word-break: break-all;">
              {check.command_line}
            </div>
            <div class="mono faint" style="font-size: 10.5px; word-break: break-all;">
              env: {(check.env ?? []).map(([k, v]) => `${k}=${v}`).join("  ")}
            </div>
          {/if}
        {:else if !checking}
          <div class="empty">Edit any field to run pre-flight</div>
        {/if}
      </div>

      <div class="toolbar">
        <button class="btn primary" onclick={save} disabled={!!busy}>Save</button>
        <button class="btn" onclick={() => launch(false)} disabled={!!busy || anyBlock}>
          {busy === "launch" ? "Launching…" : "Launch"}
        </button>
        {#if anyBlock}
          <button class="btn danger" onclick={() => launch(true)} disabled={!!busy}>
            Override blocks &amp; launch
          </button>
        {/if}
        <button class="btn" onclick={() => exportScript("bat")} disabled={!!busy}>Export .bat</button>
        <button class="btn" onclick={() => exportScript("ps1")} disabled={!!busy}>Export .ps1</button>
        <div class="grow"></div>
        <button class="btn danger" onclick={remove} disabled={!!busy || !selectedId}>Delete</button>
      </div>

      {#if draft.baseline}
        <div class="card">
          <div style="font-weight: 700; font-size: 12.5px; margin-bottom: 6px;">
            Last baseline <span class="faint mono" style="font-weight: 400; font-size: 10.5px;">{draft.baseline.measured_at} · driver {draft.baseline.driver} · {draft.baseline.sdk}</span>
          </div>
          <div class="mono" style="font-size: 12px;">
            serial {draft.baseline.serial_tok_s} tok/s
            {#if draft.baseline.concurrent}
              · n={draft.baseline.concurrent.n}: {draft.baseline.concurrent.decode_aggregate_tok_s ?? draft.baseline.concurrent.aggregate_tok_s} tok/s decode-agg,
              {draft.baseline.concurrent.per_stream_tok_s} per-stream
            {/if}
            · {draft.baseline.vram_gb} GiB resident
            {#if draft.baseline.cold_cache}<span class="chip warn">cold cache</span>{/if}
          </div>
        </div>
      {/if}
    </div>
  {:else}
    <div class="empty" style="flex: 1;">Select or create a profile</div>
  {/if}
</div>

{#if toast}
  <div class="toast" class:error={toast.isError}>{toast.text}</div>
{/if}
