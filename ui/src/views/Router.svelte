<script>
  import { onDestroy } from "svelte";
  import { api } from "../api.js";

  let rc = $state(null);          // router config (editable)
  let profiles = $state([]);
  let builds = $state([]);
  let runtimes = $state([]);
  let ini = $state("");           // rendered preview
  let status = $state(null);      // { alive, state } from Llama FIDIM run state
  let models = $state([]);        // router /models
  let busy = $state("");
  let error = $state("");
  let toast = $state(null);

  async function load() {
    error = "";
    try {
      const [r, p, s, rt, st] = await Promise.all([
        api("router_get"), api("list_profiles"), api("scan"), api("list_runtimes").catch(() => []), api("router_status").catch(() => null),
      ]);
      rc = r;
      profiles = p.map((x) => x.profile);
      builds = (s.builds ?? []).filter((b) => b.version);
      runtimes = rt;
      status = st;
      await preview();
      await refreshModels();
    } catch (e) { error = String(e); }
  }
  load();

  const poll = setInterval(() => { if (status?.alive) refreshModels(); }, 5000);
  onDestroy(() => clearInterval(poll));

  function isMember(id) { return rc?.members?.some((m) => m.profile_id === id) ?? false; }
  function member(id) { return rc?.members?.find((m) => m.profile_id === id); }
  function toggle(id) {
    if (isMember(id)) rc.members = rc.members.filter((m) => m.profile_id !== id);
    else rc.members = [...rc.members, { profile_id: id, load_on_startup: false }];
    preview();
  }
  async function preview() {
    try { ini = (await api("router_ini", { rc: normalized() })).text; } catch (e) { ini = `(cannot render: ${String(e)})`; }
  }
  function normalized() {
    const c = JSON.parse(JSON.stringify(rc));
    c.port = Number(c.port) || 1234;
    c.models_max = Math.max(0, Math.round(Number(c.models_max) || 0));
    if (!c.build) c.build = null;
    if (!c.rocm_runtime) c.rocm_runtime = null;
    return c;
  }
  async function save() {
    busy = "save";
    try { await api("router_save", { rc: normalized() }); toastMsg("Router config saved"); await preview(); } catch (e) { toastMsg(String(e), true); }
    busy = "";
  }
  async function launch() {
    busy = "launch";
    try {
      await api("router_save", { rc: normalized() });
      const r = await api("router_launch");
      const rep = r.replaced?.length ? ` · replaced ${r.replaced.join(", ")} on port ${rc.port}` : "";
      const conf = r.env_conflicts?.length ? ` · env conflicts: ${r.env_conflicts.map((c) => c[0]).join(", ")} (first value used)` : "";
      toastMsg(`Router up on port ${rc.port} (pid ${r.state.pid}) serving ${r.sections.join(", ")}${rep}${conf}`);
      status = await api("router_status");
      await refreshModels();
    } catch (e) { toastMsg(String(e), true); }
    busy = "";
  }
  async function stop() {
    busy = "stop";
    try { await api("stop_run", { target: "router" }); toastMsg("Router stopped"); status = await api("router_status"); models = []; } catch (e) { toastMsg(String(e), true); }
    busy = "";
  }
  async function refreshModels() {
    if (!status?.alive) { models = []; return; }
    try { models = await api("router_models"); } catch (e) { models = []; }
  }
  async function loadModel(id) { busy = id; try { await api("router_load", { id }); await refreshModels(); } catch (e) { toastMsg(String(e), true); } busy = ""; }
  async function unloadModel(id) { busy = id; try { await api("router_unload", { id }); await refreshModels(); } catch (e) { toastMsg(String(e), true); } busy = ""; }

  let toastTimer = null;
  function toastMsg(text, isError = false) {
    toast = { text, isError };
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (toast = null), 7000);
  }
  const base = (p) => String(p ?? "").split(/[\\/]/).pop();
</script>

<h1>Router <span class="sub">One port for every model. The request's <span class="mono">model</span> field picks the profile.</span></h1>
<p class="lede">
  llama-server's router mode. One process owns the port and starts a child per model from a preset file
  written from your profiles. Clients keep one base URL and pick a model by name. Past
  <span class="mono">models-max</span> loaded instances, the least recently used is evicted.
</p>

{#if error}<div class="card"><span class="chip block">error</span> <span class="mono">{error}</span></div>{/if}

{#if rc}
  <div class="card">
    <div style="display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px;">
      <span class="sec-title">Router</span>
      {#if status?.alive}
        <span class="chip pass">running · pid {status.state.pid} · port {status.state.port}</span>
      {:else}
        <span class="chip plain">not running</span>
      {/if}
    </div>
    <div class="formgrid">
      <label class="field" title="TCP port clients connect to. Any Llama FIDIM server already on it is stopped when the router launches; a foreign process blocks.">
        <span class="k">port</span><input type="number" bind:value={rc.port} />
      </label>
      <label class="field" title="--models-max: how many model instances may be loaded at the same time. 0 = unlimited. With two cards, 2 keeps one per card; VRAM is NOT pre-checked per load, so keep this honest.">
        <span class="k">models loaded at once</span><input type="number" min="0" max="8" bind:value={rc.models_max} />
      </label>
      <label class="field" title="Load a model automatically on the first request that names it. Off = only explicit loads (the Load buttons below, or POST /models/load).">
        <span class="k">autoload on request</span>
        <span><input type="checkbox" style="width: auto;" bind:checked={rc.autoload} /> enabled</span>
      </label>
      <label class="field" style="grid-column: span 2;" title="llama-server build that runs the router and its child instances. Needs router mode (b10819+).">
        <span class="k">build</span>
        <select bind:value={rc.build}>
          <option value={null}>newest ({builds.map((b) => b.version).sort().reverse()[0] ?? "?"})</option>
          {#each builds as b}<option value={b.path}>{b.tag} · {b.version}</option>{/each}
        </select>
      </label>
      <label class="field" title="ROCm runtime DLL search path for the router process (inherited by every instance).">
        <span class="k">ROCm runtime</span>
        <select bind:value={rc.rocm_runtime}>
          <option value={null}>default ({runtimes.find((r) => r.is_default)?.name ?? "default"}{runtimes.find((r) => r.is_default)?.version ? ` · ${runtimes.find((r) => r.is_default).version}` : ""})</option>
          {#each runtimes.filter((r) => !r.is_default) as r}<option value={r.name} disabled={!r.available}>{r.name}{r.version ? ` · ${r.version}` : ""}{r.is_latest ? " (latest)" : ""}{r.available ? "" : " (missing)"}</option>{/each}
        </select>
      </label>
    </div>
  </div>

  <div class="card">
    <div class="sec">Members <span class="faint">profiles this router serves; the model id is the profile's alias</span></div>
    <table class="grid">
      <thead><tr><th></th><th>Profile</th><th>Model id</th><th>Model</th><th>GPU</th><th>Load on startup</th></tr></thead>
      <tbody>
        {#each profiles as p}
          <tr>
            <td><input type="checkbox" style="width: auto;" checked={isMember(p.id)} onchange={() => toggle(p.id)} /></td>
            <td class="mono">{p.id}</td>
            <td class="mono">{p.server.alias || p.id}</td>
            <td class="faint">{base(p.model.path)}</td>
            <td class="path">{p.devices.map((d) => d.key.split(":").pop()).join(", ")}</td>
            <td>{#if isMember(p.id)}<input type="checkbox" style="width: auto;" checked={member(p.id)?.load_on_startup ?? false} onchange={(e) => { member(p.id).load_on_startup = e.target.checked; preview(); }} />{/if}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    <div class="faint small" style=" margin-top: 6px;">
      The router replaces each member's port and visibility pin; GPU placement becomes
      <span class="mono">device = ROCmN</span>. VRAM is not checked per load, so keep <span class="mono">models loaded at once</span> honest for your cards.
    </div>
  </div>

  <div class="toolbar">
    <button class="btn" onclick={save} disabled={!!busy}>Save</button>
    <button class="btn primary" onclick={launch} disabled={!!busy || !rc.members.length}>{busy === "launch" ? "Launching…" : status?.alive ? "Relaunch router" : "Launch router"}</button>
    <button class="btn danger" onclick={stop} disabled={!!busy || !status?.alive}>Stop router</button>
    <div class="grow"></div>
    <span class="path">clients: http://{rc.host}:{rc.port}/v1 · model = alias</span>
  </div>

  {#if status?.alive}
    <div class="card">
      <div class="sec">Models on the router</div>
      <table class="grid">
        <thead><tr><th>Model id</th><th>Status</th><th></th></tr></thead>
        <tbody>
          {#each models as m}
            <tr>
              <td class="mono">{m.id}</td>
              <td>
                {#if m.status === "loaded"}<span class="chip pass">loaded</span>
                {:else if m.status === "loading"}<span class="chip warn">loading</span>
                {:else if m.failed}<span class="chip block">failed (exit {m.exit_code ?? "?"})</span>
                {:else}<span class="chip plain">{m.status}</span>{/if}
              </td>
              <td>
                {#if m.status === "loaded"}
                  <button class="btn" onclick={() => unloadModel(m.id)} disabled={!!busy}>Unload</button>
                {:else if m.status !== "loading"}
                  <button class="btn" onclick={() => loadModel(m.id)} disabled={!!busy}>{busy === m.id ? "Loading…" : "Load"}</button>
                {/if}
              </td>
            </tr>
          {:else}
            <tr><td colspan="3"><div class="empty">no models reported yet</div></td></tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}

  <div class="card">
    <div class="sec">Preset file <span class="faint">written to ~/.fidim/router.ini on launch</span></div>
    <pre class="logbox" style="max-height: 320px;">{ini}</pre>
  </div>
{/if}

{#if toast}<div class="toast" class:error={toast.isError}>{toast.text}</div>{/if}
