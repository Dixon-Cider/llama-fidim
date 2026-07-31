<script>
  import { inTauri } from "./api.js";
  import Devices from "./views/Devices.svelte";
  import Profiles from "./views/Profiles.svelte";
  import Running from "./views/Running.svelte";
  import Benchmarks from "./views/Benchmarks.svelte";
  import Logs from "./views/Logs.svelte";

  const views = [
    { id: "devices", label: "Devices", component: Devices },
    { id: "profiles", label: "Profiles", component: Profiles },
    { id: "running", label: "Running", component: Running },
    { id: "benchmarks", label: "Benchmarks", component: Benchmarks },
    { id: "logs", label: "Logs", component: Logs },
  ];
  let active = $state("devices");
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
