<script>
  import { onDestroy } from "svelte";
  import { inTauri } from "./api.js";
  import Devices from "./views/Devices.svelte";
  import Profiles from "./views/Profiles.svelte";
  import Running from "./views/Running.svelte";
  import Benchmarks from "./views/Benchmarks.svelte";
  import Logs from "./views/Logs.svelte";
  import Updates from "./views/Updates.svelte";
  import Settings from "./views/Settings.svelte";
  import Router from "./views/Router.svelte";
  import { api, log } from "./api.js";
  import { fly } from "svelte/transition";
  import { arrive } from "./motion.js";

  // Grouped by what the person is doing: operating servers, inspecting the
  // box, maintaining the tool.
  const groups = [
    { label: "operate", views: [
      { id: "running", label: "Running", component: Running },
      { id: "profiles", label: "Profiles", component: Profiles },
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
  let setup = $state(null); // { models, builds } from the boot scan; drives the first-run banner
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

  // Live server count for the nav badge.
  async function count() {
    try { alive = (await api("status", { deep: false })).filter((r) => r.alive).length; } catch { /* keep last */ }
  }
  count();
  const counter = setInterval(count, 5000);
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
    setup = { models: r.scan?.models?.length ?? 0, builds: r.scan?.builds?.length ?? 0 };
    // Also run the editor's live check on one saved profile at boot; the
    // command logs its inputs, so a check/launch disagreement is diagnosable.
    try {
      const profiles = await api("list_profiles");
      const pick = profiles.find((p) => p.profile.id === "daily-driver") ?? profiles[0];
      if (pick) {
        const c = await api("live_check", { p: pick.profile });
        const kinds = (c.results ?? []).map((x) => (x.outcome === "pass" ? "P" : (x.outcome?.outcome ?? Object.keys(x.outcome ?? {})[0] ?? "?")[0].toUpperCase())).join("");
        log(`boot live_check ${pick.profile.id}: ${kinds} sample=${JSON.stringify((c.results ?? [])[5] ?? null)}`);
      }
    } catch (e) { log(`boot live_check failed: ${String(e)}`); }
  })();
  const ActiveComponent = $derived(views.find((v) => v.id === active).component);
</script>

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
    </div>
  </nav>
  <main class="view">
    {#if setup && (!setup.models || !setup.builds) && active !== "settings" && active !== "updates"}
      <div class="card notice" style="border-color: var(--accent-line);">
        <span class="chip accent">first run</span>
        <span>
          {#if !setup.models && !setup.builds}No models or llama.cpp builds found yet.
          {:else if !setup.models}No models found yet.
          {:else}No llama.cpp build found yet.{/if}
          {#if !setup.models}Add a model folder in <button class="link" onclick={() => (active = "settings")}>Settings</button>.{/if}
          {#if !setup.builds}Install a build from <button class="link" onclick={() => (active = "updates")}>Updates</button>.{/if}
        </span>
      </div>
    {/if}
    {#key active}
      <div in:fly={arrive()}>
        <ActiveComponent />
      </div>
    {/key}
  </main>
</div>
