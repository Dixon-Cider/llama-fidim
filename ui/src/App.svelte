<script>
  import { inTauri } from "./api.js";
  import Devices from "./views/Devices.svelte";
  import Profiles from "./views/Profiles.svelte";
  import Running from "./views/Running.svelte";
  import Benchmarks from "./views/Benchmarks.svelte";
  import Logs from "./views/Logs.svelte";
  import Updates from "./views/Updates.svelte";
  import Settings from "./views/Settings.svelte";
  import { api, log } from "./api.js";

  const views = [
    { id: "devices", label: "Devices", component: Devices },
    { id: "profiles", label: "Profiles", component: Profiles },
    { id: "running", label: "Running", component: Running },
    { id: "benchmarks", label: "Benchmarks", component: Benchmarks },
    { id: "logs", label: "Logs", component: Logs },
    { id: "updates", label: "Updates", component: Updates },
    { id: "settings", label: "Settings", component: Settings },
  ];
  let active = $state("devices");

  // Boot diagnostics: one line in ~/.llamactl/ui.log saying what the GUI can
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
    // Also run the editor's live check on one saved profile at boot; the
    // command logs its inputs, so a check/launch disagreement is diagnosable.
    try {
      const profiles = await api("list_profiles");
      const pick = profiles.find((p) => p.profile.id === "daily-driver") ?? profiles[0];
      if (pick) {
        const c = await api("live_check", { p: pick.profile });
        const kinds = (c.results ?? []).map((x) => (x.outcome === "pass" ? "P" : Object.keys(x.outcome)[0][0].toUpperCase())).join("");
        log(`boot live_check ${pick.profile.id}: ${kinds} sample=${JSON.stringify((c.results ?? [])[5] ?? null)}`);
      }
    } catch (e) { log(`boot live_check failed: ${String(e)}`); }
  })();
  const ActiveComponent = $derived(views.find((v) => v.id === active).component);
</script>

<div class="shell">
  <nav class="nav">
    <div class="brand">llama<em>ctl</em></div>
    {#each views as v}
      <button class:active={active === v.id} onclick={() => (active = v.id)}>
        {v.label}
      </button>
    {/each}
    <div class="spacer"></div>
    <div class="foot">
      {inTauri ? "connected" : "MOCK DATA (browser preview)"}
    </div>
  </nav>
  <main class="view">
    <ActiveComponent />
  </main>
</div>
