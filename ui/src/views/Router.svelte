<script>
  import { onDestroy } from "svelte";
  import { fade, fly, slide } from "svelte/transition";
  import { flip } from "svelte/animate";
  import { api } from "../api.js";
  import { leave, flipParams, toastFly, LAYOUT } from "../motion.js";

  let rc = $state(null);          // router config (editable)
  let defaultKind = $state("llama"); // what an empty router becomes (router_get)
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
      const { kind: _k, default_kind, ...cfg } = r;
      rc = cfg;
      defaultKind = default_kind ?? "llama";
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

  // What the router runs when no build is chosen: the newest UPSTREAM build
  // by number, as core's newest_installed picks it.
  const newestUpstream = $derived(
    builds
      .filter((b) => (b.channel ?? "upstream") === "upstream")
      .map((b) => b.version)
      .sort((a, b) => (Number(b.replace(/^b/, "")) || 0) - (Number(a.replace(/^b/, "")) || 0))[0],
  );

  // Two routers share router.json. Every member SGLang (core's
  // sglang::is_sglang_router): FIDIM's router in front of SGLang servers.
  // Otherwise llama-server's router mode, whose members must be
  // llama-server profiles (router::check_member). An empty router takes
  // the kind router_get reports and lists what could start either kind.
  const isServerProfile = (p) => (p.engine ?? "llama-server") === "llama-server";
  const isSglang = (p) => p.engine === "sglang";
  const kind = $derived(
    rc?.members?.length
      ? rc.members.every((m) => profiles.some((p) => p.id === m.profile_id && isSglang(p))) ? "sglang" : "llama"
      : defaultKind,
  );
  const sg = $derived(kind === "sglang");
  const fits = (p) =>
    rc?.members?.length ? (sg ? isSglang(p) : isServerProfile(p)) : isSglang(p) || (isServerProfile(p) && defaultKind !== "sglang");
  // A member that no longer fits stays listed so it can be removed.
  const listed = $derived(profiles.filter((p) => fits(p) || isMember(p.id)));
  const hiddenCount = $derived(profiles.length - listed.length);

  function isMember(id) { return rc?.members?.some((m) => m.profile_id === id) ?? false; }
  function member(id) { return rc?.members?.find((m) => m.profile_id === id); }
  function toggle(id) {
    if (isMember(id)) rc.members = rc.members.filter((m) => m.profile_id !== id);
    else rc.members = [...rc.members, { profile_id: id, load_on_startup: false }];
    preview();
  }
  async function preview() {
    if (sg && !rc.members.length) { ini = ""; return; }
    try { ini = (await api("router_ini", { rc: normalized() })).text; } catch (e) { ini = `(cannot render: ${String(e)})`; }
  }
  function normalized() {
    const c = JSON.parse(JSON.stringify(rc));
    delete c.kind;
    delete c.default_kind;
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
      const started = r.started?.length ? ` · started ${r.started.join(", ")}` : "";
      toastMsg(`Router up on port ${rc.port} (pid ${r.state.pid}) serving ${r.sections.join(", ")}${started}${rep}${conf}`);
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
  // Behind the SGLang router, load/unload start and stop the member's own
  // server (router_load / router_unload branch on the saved config).
  let busyVerb = $state("");
  async function loadModel(id) {
    busy = id; busyVerb = sg ? "Starting…" : "Loading…";
    try { await api("router_load", { id }); if (sg) toastMsg(`${id} started`); await refreshModels(); } catch (e) { toastMsg(String(e), true); }
    busy = ""; busyVerb = "";
  }
  async function unloadModel(id) {
    busy = id; busyVerb = sg ? "Stopping…" : "Unloading…";
    try { await api("router_unload", { id }); if (sg) toastMsg(`${id} stopped`); await refreshModels(); } catch (e) { toastMsg(String(e), true); }
    busy = ""; busyVerb = "";
  }

  let toastTimer = null;
  function toastMsg(text, isError = false) {
    toast = { text, isError };
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (toast = null), 7000);
  }
  const base = (p) => String(p ?? "").split(/[\\/]/).pop();
</script>

<h1>Router <span class="sub">One port for every model. The request's <span class="mono">model</span> field picks the profile.</span></h1>
{#if sg}
  <p class="lede">
    Llama FIDIM's router in front of your SGLang servers. Each member keeps its own port and process;
    the router forwards every request to the member its <span class="mono">model</span> names.
  </p>
{:else}
  <p class="lede">
    llama-server's router mode. One process owns the port and starts a child per model from a preset file
    written from your profiles. Clients keep one base URL and pick a model by name. Past
    <span class="mono">models-max</span> loaded instances, the least recently used is evicted.
  </p>
{/if}

{#if error}<div class="card" transition:slide={leave}><span class="chip block shake">error</span> <span class="mono">{error}</span></div>{/if}

{#if rc}
  <div class="card">
    <div style="display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px;">
      <span class="sec-title">Router</span>
      <span class="chip plain" title={sg ? "Every member is an SGLang profile" : "Members are llama-server profiles"}>{sg ? "sglang" : "llama-server"}</span>
      {#if status?.alive}
        <span class="chip pass">running · pid {status.state.pid} · port {status.state.port}</span>
      {:else}
        <span class="chip plain">not running</span>
      {/if}
    </div>
    <div class="formgrid">
      <label class="field" title="TCP port clients connect to; a Llama FIDIM server already on it is stopped at launch">
        <span class="k">port</span><input type="number" bind:value={rc.port} />
      </label>
      {#if !sg}
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
          <option value={null}>newest ({newestUpstream ?? "?"})</option>
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
      {/if}
    </div>
  </div>

  <div class="card">
    <div class="sec">Members <span class="faint">profiles this router serves; the model id is the profile's alias</span></div>
    <table class="grid">
      <thead><tr><th></th><th>Profile</th><th>Model id</th><th>Model</th><th>GPU</th>
        {#if sg}<th title="Start this member's SGLang server at router launch if it is not running">Start with router</th>
        {:else}<th title="Load this model when the router starts, else on its first request">Load on startup</th>{/if}
      </tr></thead>
      <tbody>
        {#each listed as p}
          <tr>
            <td><input type="checkbox" style="width: auto;" checked={isMember(p.id)} onchange={() => toggle(p.id)} /></td>
            <td class="mono">{p.id}{#if !fits(p)}
              {#if isSglang(p) || (sg && isServerProfile(p))} <span class="chip block" title="SGLang and llama-server profiles cannot share a router; uncheck one kind">mixed</span>
              {:else} <span class="chip block" title="This engine runs standalone only; uncheck it to render the router again">standalone only</span>{/if}
            {/if}</td>
            <td class="mono">{p.server.alias || p.id}</td>
            <td class="faint">{base(p.model.path)}</td>
            <td class="path">{p.devices.map((d) => d.key.split(":").pop()).join(", ")}</td>
            <td>{#if isMember(p.id)}<input type="checkbox" style="width: auto;" checked={member(p.id)?.load_on_startup ?? false} onchange={(e) => { member(p.id).load_on_startup = e.target.checked; preview(); }} />{/if}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    {#if sg}
      {#if hiddenCount}<div class="faint small" style="margin-top: 6px;">{`${hiddenCount === 1 ? "One profile is" : `${hiddenCount} profiles are`} not listed: the SGLang router fronts SGLang profiles only.`}</div>{/if}
    {:else}
      <div class="faint small" style=" margin-top: 6px;">
        The router replaces each member's port and visibility pin; GPU placement becomes
        <span class="mono">device = ROCmN</span>. VRAM is not checked per load, so keep <span class="mono">models loaded at once</span> honest for your cards.
      </div>
      <div class="faint small" style="margin-top: 4px;">
        Diffusion profiles run standalone only{hiddenCount ? `; ${hiddenCount === 1 ? "one is" : `${hiddenCount} are`} not listed here` : ""}.
      </div>
    {/if}
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
          {#each models as m (m.id)}
            <tr in:fade={{ duration: LAYOUT }} animate:flip={flipParams}>
              <td class="mono">{m.id}</td>
              <td>
                {#if sg}
                  {#if m.status === "loaded"}<span class="chip pass" title="The member's SGLang server answers the router's health check">running</span>
                  {:else if m.status === "loading"}<span class="chip warn live" title="The member's SGLang server is up but has not opened its port yet">loading</span>
                  {:else}<span class="chip plain" title="The member's SGLang server is not answering; Start launches it from its profile">stopped</span>{/if}
                {:else if m.status === "loaded"}<span class="chip pass">loaded</span>
                {:else if m.status === "loading"}<span class="chip warn">loading</span>
                {:else if m.failed}<span class="chip block">failed (exit {m.exit_code ?? "?"})</span>
                {:else}<span class="chip plain">{m.status}</span>{/if}
              </td>
              <td>
                {#if sg}
                  {#if m.status === "loaded" || m.status === "loading"}
                    <button class="btn" onclick={() => unloadModel(m.id)} disabled={!!busy} title="Stop this member's SGLang server; the router stays up">{busy === m.id ? busyVerb : "Stop"}</button>
                  {:else}
                    <button class="btn" onclick={() => loadModel(m.id)} disabled={!!busy} title="Launch this member's SGLang server from its profile (can take minutes)">{busy === m.id ? busyVerb : "Start"}</button>
                  {/if}
                {:else if m.status === "loaded"}
                  <button class="btn" onclick={() => unloadModel(m.id)} disabled={!!busy}>Unload</button>
                {:else if m.status !== "loading"}
                  <button class="btn" onclick={() => loadModel(m.id)} disabled={!!busy}>{busy === m.id ? busyVerb : "Load"}</button>
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
    {#if sg}
      <div class="sec">Router command <span class="faint">what Launch runs; members start from their own profiles</span></div>
      <pre class="logbox" style="max-height: 320px;">{ini || "(add a member)"}</pre>
    {:else}
      <div class="sec">Preset file <span class="faint">written to ~/.fidim/router.ini on launch</span></div>
      <pre class="logbox" style="max-height: 320px;">{ini}</pre>
    {/if}
  </div>
{/if}

{#if toast}<div class="toast" class:error={toast.isError} transition:fly={toastFly}>{toast.text}</div>{/if}
