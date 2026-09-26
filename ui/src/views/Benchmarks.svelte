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

  // Any engine that answers on the profile's own port can be swept:
  // llama-server over /completion, SGLang over a streamed /v1/completions.
  // The diffusion engine cannot (core's ensure_benchable says why).
  const current = $derived(profiles.find((p) => p.id === selected) ?? null);
  const engineOf = (p) => p?.engine ?? "llama-server";
  const benchable = (p) => engineOf(p) !== "diffusion-gemma";
  const live = $derived(liveRunIds.includes(selected));

  // What ran the sweep: the build a llama-server sweep recorded, or for an
  // SGLang profile the configured install's version (sweeps do not record
  // one). A source install reports 0.0.0.
  let sgVersion = $state(null);
  let sgAsked = false;
  $effect(() => {
    if (engineOf(current) !== "sglang" || sgAsked) return;
    sgAsked = true;
    api("sglang_installs", { roots: null })
      .then((l) => { const v = (l ?? []).find((i) => i.configured)?.sglang_version; sgVersion = v ? (/^0\.0\.0/.test(v) ? "source" : v) : ""; })
      .catch(() => (sgVersion = ""));
  });
  const engineCell = (h) => engineOf(current) === "sglang" ? `sglang${sgVersion ? " " + sgVersion : ""}` : h.build_version ?? "—";
  // Driver and HIP SDK are read on Windows only; hide the column when no sweep has them.
  const hasDriver = $derived(history.some((h) => h.driver || h.sdk));

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

<h1>Benchmarks <span class="sub" title="Two warm-ups, then one request alone and N at once, 256 tokens each">Throughput sweeps against a running server.</span></h1>

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
    <span class="chip note" title="Streamed /v1/completions on the profile's own port; decode is timed from the first token">sglang</span>
  {/if}
  {#if message}<span class="muted">{message}</span>{/if}
</div>

<div class="card flush" style="overflow-x: auto;">
  <table class="grid">
    <thead>
      <tr>
        <th>Measured</th>
        <th title="Decode rate of one request alone, prompt processing excluded">Serial</th>
        <th title="Requests run at once in the concurrent phase">N</th>
        <th title="Sum of the concurrent streams' decode rates, prompt processing excluded">Decode agg</th>
        <th title="All tokens generated over the phase's wall time: what callers see together">Wall agg</th>
        <th title="Mean decode rate of one concurrent stream">Per-stream</th>
        <th title="GPU memory the server held after the sweep">VRAM</th>
        <th title={engineOf(current) === "sglang" ? "The configured SGLang install now; sweeps do not record it" : "llama.cpp build the sweep ran on"}>Engine</th>
        {#if hasDriver}<th>Driver / SDK</th>{/if}
        <th>Split</th><th></th>
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
          <td class="num">{#if h.vram_gb}{h.vram_gb} GiB{:else}<span class="faint" title="Not measured: per-process GPU memory is read on Windows only">—</span>{/if}</td>
          <td class="mono">{engineCell(h)}</td>
          {#if hasDriver}<td class="path">{#if h.driver || h.sdk}{h.driver ?? "—"}<br />{h.sdk ?? "—"}{:else}<span class="faint" title="No driver or HIP SDK version was found for this sweep">—</span>{/if}</td>{/if}
          <td class="mono faint">{h.split ? `${h.split.mode} ${JSON.stringify(h.split.fractions)}` : "—"}</td>
          <td>{#if h.cold_cache}<span class="chip warn">cold</span>{/if}</td>
        </tr>
      {:else}
        <tr><td colspan={hasDriver ? 11 : 10}><div class="empty">No sweeps recorded for this profile yet</div></td></tr>
      {/each}
    </tbody>
  </table>
</div>
