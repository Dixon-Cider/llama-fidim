<script>
  import { api } from "../api.js";

  let profiles = $state([]);
  let selected = $state("");
  let history = $state([]);
  let running = $state(false);
  let liveRunIds = $state([]);
  let message = $state("");

  async function load() {
    const [p, s] = await Promise.all([api("list_profiles"), api("status", { deep: false })]);
    profiles = p.map((r) => r.profile);
    liveRunIds = s.filter((r) => r.alive).map((r) => r.state.profile_id);
    if (!selected && profiles.length) selected = profiles[0].id;
    await loadHistory();
  }
  async function loadHistory() {
    history = selected ? await api("bench_history", { id: selected }) : [];
  }
  load();

  async function sweep() {
    running = true;
    message = "";
    try {
      const r = await api("bench_profile", { id: selected, concurrency: null, tokens: 256 });
      message = r.cold
        ? "Cold-cache run recorded to history but NOT saved as the baseline (R-08) — relaunch and re-bench for a warm number."
        : r.saved
          ? "Baseline saved to the profile."
          : "Sweep recorded.";
      await loadHistory();
    } catch (e) {
      message = String(e);
    }
    running = false;
  }
</script>

<h1>Benchmarks <span class="sub">sweeps against a running server; results stored per profile revision</span></h1>
<p class="lede">
  Warm up twice, measure serial then N-concurrent decode. "Decode agg" matches the old batch-file
  sweep tables (sum of per-stream rates); "wall agg" is what a caller actually experiences.
</p>

<div class="toolbar">
  <select style="width: 240px;" bind:value={selected} onchange={loadHistory}>
    {#each profiles as p}<option value={p.id}>{p.id}</option>{/each}
  </select>
  <button class="btn primary" onclick={sweep} disabled={running || !liveRunIds.includes(selected)}>
    {running ? "Sweeping…" : "Run sweep"}
  </button>
  {#if !liveRunIds.includes(selected)}
    <span class="chip plain">server not running — launch it first</span>
  {/if}
  {#if message}<span class="muted">{message}</span>{/if}
</div>

<div class="card flush" style="overflow-x: auto;">
  <table class="grid">
    <thead>
      <tr>
        <th>Measured</th><th>Serial</th><th>N</th><th>Decode agg</th><th>Wall agg</th>
        <th>Per-stream</th><th>VRAM</th><th>Driver / SDK</th><th>Split</th><th></th>
      </tr>
    </thead>
    <tbody>
      {#each [...history].reverse() as h}
        <tr>
          <td class="mono faint">{h.measured_at}</td>
          <td class="num">{h.serial_tok_s} tok/s</td>
          <td class="num">{h.concurrent?.n ?? "—"}</td>
          <td class="num">{h.concurrent?.decode_aggregate_tok_s ?? "—"}</td>
          <td class="num">{h.concurrent?.aggregate_tok_s ?? "—"}</td>
          <td class="num">{h.concurrent?.per_stream_tok_s ?? "—"}</td>
          <td class="num">{h.vram_gb} GiB</td>
          <td class="path">{h.driver}<br />{h.sdk}</td>
          <td class="mono faint">{h.split ? `${h.split.mode} ${JSON.stringify(h.split.fractions)}` : "—"}</td>
          <td>{#if h.cold_cache}<span class="chip warn">cold</span>{/if}</td>
        </tr>
      {:else}
        <tr><td colspan="10"><div class="empty">No sweeps recorded for this profile yet</div></td></tr>
      {/each}
    </tbody>
  </table>
</div>
