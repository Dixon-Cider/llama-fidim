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

  // Any engine that answers /v1/completions on the profile's port can be
  // swept: llama-server and SGLang (behind model_router.py). The diffusion
  // engine cannot (core's ensure_benchable says why).
  const current = $derived(profiles.find((p) => p.id === selected) ?? null);
  const engineOf = (p) => p?.engine ?? "llama-server";
  const benchable = (p) => engineOf(p) !== "diffusion-gemma";
  const live = $derived(liveRunIds.includes(selected));

  async function sweep() {
    running = true;
    message = "";
    try {
      const r = await api("bench_profile", { id: selected, concurrency: null, tokens: 256 });
      message = r.cold
        ? "Cold-cache run recorded, but not saved as the baseline. Run it again warm."
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

<h1>Benchmarks <span class="sub">Throughput sweeps against a running server.</span></h1>
<p class="lede">
  Two warm-ups, then serial and N-concurrent decode. Decode agg is the sum of per-stream rates; wall agg is what a caller sees.
</p>

<div class="toolbar">
  <select style="width: 280px;" bind:value={selected} onchange={loadHistory}>
    {#each profiles as p}<option value={p.id}>{p.id}{engineOf(p) !== "llama-server" ? ` · ${engineOf(p)}` : ""}</option>{/each}
  </select>
  <button class="btn primary" onclick={sweep} disabled={running || !live || !benchable(current)}>
    {running ? "Sweeping…" : "Run sweep"}
  </button>
  {#if current && !benchable(current)}
    <span class="chip warn" title="Every diffusion request denoises whole 256-token blocks serially; use the timings in the chat response instead.">diffusion engine: not benchmarked</span>
  {:else if !live}
    <span class="chip plain">server not running</span>
  {:else if engineOf(current) === "sglang"}
    <span class="chip note" title="Swept through model_router.py on the profile's port, the same /v1/completions calls as llama-server.">sglang</span>
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
