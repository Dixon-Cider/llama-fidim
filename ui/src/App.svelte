<script>
  import { onDestroy } from "svelte";
  import { inTauri } from "./api.js";
  import Devices from "./views/Devices.svelte";
  import Profiles from "./views/Profiles.svelte";
  import Models from "./views/Models.svelte";
  import Running from "./views/Running.svelte";
  import Benchmarks from "./views/Benchmarks.svelte";
  import Logs from "./views/Logs.svelte";
  import Updates from "./views/Updates.svelte";
  import Settings from "./views/Settings.svelte";
  import Router from "./views/Router.svelte";
  import Chat from "./views/Chat.svelte";
  import { api, log } from "./api.js";
  import { fly } from "svelte/transition";
  import { arrive } from "./motion.js";

  // Grouped by what the person is doing: operating servers, inspecting the
  // box, maintaining the tool.
  const groups = [
    { label: "operate", views: [
      { id: "running", label: "Running", component: Running },
      { id: "chat", label: "Chat", component: Chat },
      { id: "profiles", label: "Profiles", component: Profiles },
      { id: "models", label: "Models", component: Models },
      { id: "router", label: "Router", component: Router },
    ]},
    { label: "inspect", views: [
      { id: "devices", label: "Devices", component: Devices },
      { id: "benchmarks", label: "Benchmarks", component: Benchmarks },
      { id: "logs", label: "Logs", component: Logs },
    ]},
    { label: "maintain", views: [
      { id: "updates", label: "Updates", component: Updates },
      { id: "settings", label: "Settings", component: Settings },
    ]},
  ];
  const views = groups.flatMap((g) => g.views);
  let active = $state("running");
  let alive = $state(0);
  let setup = $state(null); // { models, builds, sglang, profiles, sglangHost } from boot; drives the first-run banner
  // Where a llama.cpp build cannot exist (Linux) or the config has an
  // sglang section, the engine to set up is SGLang, never a build. No
  // command reports the platform, so the web view's own is used.
  const linux = /linux/i.test(navigator.platform ?? "");
  // The active pill slides between nav items instead of appearing.
  let btns = $state({});
  let pill = $state(null);
  $effect(() => {
    const b = btns[active];
    if (b) pill = { top: b.offsetTop, height: b.offsetHeight };
  });
  // Any uncaught error in the web view goes to ~/.fidim/ui.log; a thrown
  // template expression otherwise just leaves the view half-updated.
  window.addEventListener("error", (e) => log(`uncaught: ${e.message} @ ${e.filename}:${e.lineno} ${e.error?.stack ?? ""}`));
  window.addEventListener("unhandledrejection", (e) => log(`unhandled rejection: ${e.reason?.stack ?? String(e.reason)}`));

  // A reload of this web view leaves any chat reply streaming to nobody:
  // stop them all once at boot.
  api("chat_cancel_all").catch(() => {});

  // Live server count for the nav badge (every 15 s, skipped while the
  // window is hidden: each call probes every run over HTTP).
  async function count() {
    if (document.hidden) return;
    try { alive = (await api("status", { deep: false })).filter((r) => r.alive).length; } catch { /* keep last */ }
  }
  count();
  const counter = setInterval(count, 15000);
  onDestroy(() => clearInterval(counter));

  // Boot diagnostics: one line in ~/.fidim/ui.log saying what the GUI can
  // see, so an empty picker can be diagnosed without a debugger.
  (async () => {
    const t = performance.now();
    const r = {};
    for (const [name, call] of [
      ["scan", () => api("scan")],
      ["devices", () => api("devices", { refresh: false })],
      ["runtimes", () => api("list_runtimes")],
    ]) {
      try { r[name] = await call(); } catch (e) { r[name] = { error: String(e) }; }
    }
    const n = (x, k) => (x?.error ? `ERR(${x.error})` : k ? (x?.[k]?.length ?? 0) : (x?.length ?? 0));
    log(`boot: builds=${n(r.scan, "builds")} models=${n(r.scan, "models")} devices=${n(r.devices)} runtimes=${n(r.runtimes)} in ${Math.round(performance.now() - t)}ms`);
    const models = r.scan?.models?.length ?? 0, builds = r.scan?.builds?.length ?? 0;
    // The rest of the first-run picture: is SGLang configured (a venv in
    // config.json, or a found install marked configured), and are there
    // profiles at all.
    const [cfgRes, installs, profiles] = await Promise.all([
      api("get_config").catch(() => null),
      api("sglang_installs", { roots: null }).catch(() => []),
      api("list_profiles").catch(() => []),
    ]);
    const cfg = cfgRes?.config ?? {};
    const sglang = !!String(cfg.sglang?.venv ?? "").trim() || (installs ?? []).some((i) => i.configured);
    setup = { models, builds, sglang, profiles: profiles.length, sglangHost: linux || !!cfg.sglang };
    // Also run the editor's live check on one saved profile at boot; the
    // command logs its inputs, so a check/launch disagreement is diagnosable.
    try {
      const pick = profiles.find((p) => p.profile.id === "daily-driver") ?? profiles[0];
      if (pick) {
        const c = await api("live_check", { p: pick.profile });
        const kinds = (c.results ?? []).map((x) => (x.outcome === "pass" ? "P" : (x.outcome?.outcome ?? Object.keys(x.outcome ?? {})[0] ?? "?")[0].toUpperCase())).join("");
        log(`boot live_check ${pick.profile.id}: ${kinds} sample=${JSON.stringify((c.results ?? [])[5] ?? null)}`);
      }
    } catch (e) { log(`boot live_check failed: ${String(e)}`); }
  })();
  const ActiveComponent = $derived(views.find((v) => v.id === active).component);
  // The one next action for the first-run banner, or null once set up.
  const firstRun = $derived(
    !setup ? null
    : setup.sglangHost && !setup.sglang ? "sglang"
    : !setup.sglangHost && !setup.builds ? "build"
    : !setup.models ? "models"
    : !setup.profiles ? "profiles"
    : null
  );

  // Which build this is, under the tagline: v0.2.0+3 is three commits past
  // the 0.2.0 release, +? an unknown distance from it; the tooltip has the
  // full `fidim --version` text.
  let version = $state(null);
  api("app_version").then((v) => (version = v)).catch(() => {});
  const versionLabel = $derived(
    version
      ? "v" + version.version +
        (!version.commit ? "" : version.commits_ahead == null ? "+?" : version.commits_ahead ? "+" + version.commits_ahead : "") +
        (version.commit ? " · " + version.commit : "") + (version.modified ? " · modified" : "")
      : ""
  );

  // ---- hover hints ---------------------------------------------------------
  // Every `title=` in the views is the control's one-sentence explanation.
  // WebKitGTK does not surface them reliably inside a Tauri window, so this
  // renders them itself: the title moves to data-tip on first hover (which
  // also suppresses a second, native tooltip where the webview does show one)
  // and a small panel follows the cursor after a short delay.
  let tip = $state(null);            // { text, x, y }
  let tipTimer = null;
  function tipShow(el, x, y) {
    if (el.getAttribute("title")) {
      el.dataset.tip = el.getAttribute("title");
      el.removeAttribute("title");
    }
    const text = el.dataset.tip;
    if (!text) return;
    clearTimeout(tipTimer);
    tipTimer = setTimeout(() => (tip = { text, x, y }), 220);
  }
  function tipHide() {
    clearTimeout(tipTimer);
    tip = null;
  }
  function onTipOver(e) {
    const el = e.target?.closest?.("[title], [data-tip]");
    if (!el) return tipHide();
    tipShow(el, e.clientX, e.clientY);
  }
  function onTipMove(e) {
    if (tip) tip = { ...tip, x: e.clientX, y: e.clientY };
  }
  const tipStyle = $derived.by(() => {
    if (!tip) return "";
    const w = 340, pad = 14;
    const left = Math.min(tip.x + pad, (window.innerWidth || 1200) - w - pad);
    const below = tip.y + 22;
    const top = below + 80 > (window.innerHeight || 800) ? tip.y - 60 : below;
    return `left:${Math.max(8, left)}px; top:${top}px; max-width:${w}px;`;
  });
</script>
<svelte:window onmouseover={onTipOver} onmousemove={onTipMove} onmouseout={(e) => { if (!e.relatedTarget?.closest?.("[data-tip], [title]")) tipHide(); }} onscroll={tipHide} onmousedown={tipHide} />
{#if tip}<div class="tip" role="tooltip" style={tipStyle}>{tip.text}</div>{/if}


<div class="shell">
  <nav class="nav">
    <div class="brand">Llama <em>FIDIM</em><span class="dot" class:off={!inTauri} title={inTauri ? "connected to the Rust core" : "browser preview: mock data"}></span></div>
    {#if pill}<div class="pill" style="top: {pill.top}px; height: {pill.height}px;"></div>{/if}
    {#each groups as g}
      <div class="group">{g.label}</div>
      {#each g.views as v}
        <button class:active={active === v.id} onclick={() => (active = v.id)} bind:this={btns[v.id]}>
          {v.label}
          {#if v.id === "running" && alive > 0}{#key alive}<span class="count pop">{alive}</span>{/key}{/if}
        </button>
      {/each}
    {/each}
    <div class="spacer"></div>
    <div class="foot">
      {inTauri ? "Fine, I'll do it myself." : "MOCK DATA\nbrowser preview"}
      {#if versionLabel}<div class="ver" title={"Llama FIDIM " + version.long}>{versionLabel}</div>{/if}
    </div>
  </nav>
  <main class="view">
    {#if firstRun && active !== "settings" && active !== "updates" && active !== "models"}
      <div class="card notice" style="border-color: var(--accent-line);">
        <span class="chip accent" title={firstRun === "sglang" ? "No SGLang environment is configured yet" : firstRun === "build" ? "No llama.cpp build found yet" : firstRun === "models" ? "No models found yet" : "No profiles yet"}>first run</span>
        <span>
          {#if firstRun === "sglang"}Set up SGLang in <button class="link" onclick={() => (active = "settings")}>Settings</button>
          {:else if firstRun === "build"}Install a llama.cpp build in <button class="link" onclick={() => (active = "updates")}>Updates</button>
          {:else if firstRun === "models"}Add a model folder in <button class="link" onclick={() => (active = "settings")}>Settings</button> or get one in <button class="link" onclick={() => (active = "models")}>Models</button>
          {:else}Make a profile from a model in <button class="link" onclick={() => (active = "models")}>Models</button>{/if}
        </span>
      </div>
    {/if}
    {#key active}
      <div in:fly={arrive()}>
        <ActiveComponent go={(id) => (active = id)} />
      </div>
    {/key}
  </main>
</div>

<style>
  .tip {
    position: fixed; z-index: 1000; pointer-events: none;
    padding: 6px 9px; border-radius: 6px;
    background: var(--ground-raised, #23272f); color: var(--ink, #e6e6e6);
    border: 1px solid var(--rule-strong, rgba(255,255,255,0.12));
    box-shadow: 0 6px 20px rgba(0,0,0,0.45);
    font-size: 12px; line-height: 1.35; white-space: normal;
  }
</style>
