<script>
  import { onDestroy } from "svelte";
  import { api } from "../api.js";

  let sources = $state([]);
  let selected = $state("");
  let filter = $state("");
  let lines = $state([]);
  let path = $state("");
  let totalMatching = $state(0);
  let follow = $state(true);

  async function loadSources() {
    sources = await api("log_sources");
    if (!selected && sources.length) selected = sources[0].profile_id;
    await tail();
  }
  async function tail() {
    if (!selected) return;
    try {
      const r = await api("read_log", { target: selected, tail: 400, filter });
      lines = r.lines;
      path = r.path;
      totalMatching = r.total_matching;
    } catch {
      lines = [];
    }
  }
  loadSources();

  const poll = setInterval(() => {
    if (follow) tail();
  }, 2000);
  onDestroy(() => clearInterval(poll));
</script>

<h1>Logs <span class="sub">live tail per server, plus logs from stopped and crashed runs</span></h1>

<div class="toolbar">
  <select style="width: 220px;" bind:value={selected} onchange={tail}>
    {#each sources as s}
      <option value={s.profile_id}>{s.profile_id} (:{s.port}){s.alive ? "" : " — stopped"}</option>
    {/each}
  </select>
  <input style="width: 260px;" placeholder="filter…" bind:value={filter} oninput={tail} />
  <label style="display: flex; gap: 6px; align-items: center; font-size: 12px; color: var(--ink-muted);">
    <input type="checkbox" style="width: auto;" bind:checked={follow} /> follow
  </label>
  <div class="grow"></div>
  <span class="path">{totalMatching} matching · {path}</span>
</div>

<div class="logbox">
  {#if lines.length}
    {lines.join("\n")}
  {:else}
    <span class="faint">no log lines{filter ? " matching filter" : ""}</span>
  {/if}
</div>
