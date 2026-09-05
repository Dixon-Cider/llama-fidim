<script>
  import { onDestroy } from "svelte";
  import { api } from "../api.js";

  let data = $state({ runs: [], cards: [] });
  let error = $state("");
  let paused = $state(false);
  let now = $state(Math.floor(Date.now() / 1000));
  // Per (run, model) history of counter samples for rates and sparklines.
  const hist = new Map();
  let rates = $state({});   // key -> { decode, prompt, accept }
  let spark = $state({});   // key -> [decode tok/s ...] last 60

  const GIB = 1024 * 1024 * 1024;
  const key = (r, s) => `${r.state.profile_id}:${s?.model ?? ""}`;

  async function poll() {
    if (paused) return;
    try {
      const d = await api("live");
      // Rates from counter deltas between polls: what is happening now,
      // not the last request's average.
      for (const run of d.runs) {
        for (const s of run.samples) {
          const k = key(run.run, s);
          const m = s.metrics ?? {};
          const prev = hist.get(k);
          const cur = { t: s.sampled_unix_ms, gen: m.tokens_predicted_total ?? 0, prompt: m.prompt_tokens_total ?? 0, dn: m.spec_decode_num_draft_tokens_total ?? 0, da: m.spec_decode_num_accepted_tokens_total ?? 0 };
          if (prev && cur.t > prev.t) {
            const dt = (cur.t - prev.t) / 1000;
            const decode = Math.max(0, (cur.gen - prev.gen) / dt);
            const prompt = Math.max(0, (cur.prompt - prev.prompt) / dt);
            const dd = cur.dn - prev.dn;
            const accept = dd > 0 ? (cur.da - prev.da) / dd : null;
            rates = { ...rates, [k]: { decode, prompt, accept } };
            spark = { ...spark, [k]: [...(spark[k] ?? []), decode].slice(-60) };
          }
          hist.set(k, cur);
        }
      }
      data = d;
      error = "";
    } catch (e) {
      error = String(e);
    }
  }
  poll();
  const timer = setInterval(poll, 1000);
  const tick = setInterval(() => (now = Math.floor(Date.now() / 1000)), 1000);
  onDestroy(() => { clearInterval(timer); clearInterval(tick); });

  async function stop(target) {
    try { await api("stop_run", { target }); await poll(); } catch (e) { error = String(e); }
  }
  function uptime(started) {
    const s = Math.max(0, now - started);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`;
  }
  function healthChip(r) {
    if (r.crashed) return { kind: "block", label: "crashed" };
    if (r.health === "healthy") return { kind: "pass", label: "healthy" };
    if (r.health === "responding-not-generating") return { kind: "warn", label: "responding, NOT generating" };
    return { kind: "block", label: "dead" };
  }
  function phaseChip(p) {
    return p === "decode" ? "accent" : p === "prefill" ? "warn" : "plain";
  }
  function sparkPath(vals, w = 160, h = 28) {
    if (!vals?.length) return "";
    const max = Math.max(1, ...vals);
    return vals.map((v, i) => `${(i / Math.max(1, vals.length - 1)) * w},${h - (v / max) * (h - 2) - 1}`).join(" ");
  }
  const fmt1 = (v) => (v == null ? "—" : Number(v).toFixed(1));
  const short = (k) => k.split(":").pop();
</script>

<h1>Running <span class="sub">live: phase, progress and throughput per server, polled every second</span></h1>
<p class="lede">
  Prefill and decode progress come from each server's slot state; rates are counter deltas between polls,
  so they show what is happening now. Card busy is the GPU engine utilization across every process on that card.
</p>

<div class="toolbar">
  <button class="btn" onclick={() => (paused = !paused)}>{paused ? "Resume polling" : "Pause polling"}</button>
  {#each data.cards as c}
    <span class="chip {c.busy_percent > 50 ? 'accent' : 'plain'}" title="{c.name}">{short(c.key)} busy {Math.round(c.busy_percent)}%</span>
  {/each}
  {#if error}<span class="chip block">{error}</span>{/if}
</div>

{#each data.runs as row}
  {@const r = row.run}
  {@const chip = healthChip(r)}
  {@const vram = row.resident.reduce((a, m) => a + Math.max(m.dedicated_bytes, m.committed_bytes), 0)}
  <div class="card">
    <div style="display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;">
      <span style="font-weight: 700; font-size: 14px;">{r.state.profile_id}</span>
      <span class="chip {chip.kind}">{chip.label}</span>
      {#if r.alive}
        {@const phase = row.samples.some((s) => s.phase === "decode") ? "decode" : row.samples.some((s) => s.phase === "prefill") ? "prefill" : "idle"}
        <span class="chip {phaseChip(phase)}">{phase}</span>
        <span class="mono muted" title="GPU engine utilization of this process">gpu {Math.round(row.gpu_busy_percent)}%</span>
        <span class="mono muted" title="resident VRAM (dedicated, or committed while paging in)">{(vram / GIB).toFixed(1)} GiB{#if row.resident.length} on {row.resident.map((m) => short(m.card ?? "?")).join("+")}{/if}</span>
      {/if}
      <span class="mono muted">pid {r.state.pid} · :{r.state.port} · {r.state.alias}</span>
      <span class="num muted">up {uptime(r.state.started_unix)}</span>
      <div style="flex: 1;"></div>
      {#if r.alive}<button class="btn danger" onclick={() => stop(r.state.profile_id)}>Stop</button>{/if}
    </div>

    {#each row.samples as s}
      {@const k = key(r, s)}
      {@const rt = rates[k]}
      {@const m = s.metrics ?? {}}
      <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--rule);">
        <div style="display: flex; gap: 14px; align-items: center; flex-wrap: wrap;">
          {#if s.model}<span class="mono" style="font-weight: 600;">{s.model}</span>{/if}
          <span class="chip {phaseChip(s.phase)}">{s.phase}</span>
          <span class="num" title="generated tokens per second, from counter deltas"><b>{fmt1(rt?.decode)}</b> <span class="faint">tok/s decode</span></span>
          <span class="num" title="prompt tokens per second, from counter deltas"><b>{fmt1(rt?.prompt)}</b> <span class="faint">tok/s prefill</span></span>
          <span class="num" title="requests processing / waiting for a slot">{m.requests_processing ?? 0} <span class="faint">in flight</span> · {m.requests_deferred ?? 0} <span class="faint">queued</span></span>
          {#if (m.spec_decode_num_draft_tokens_total ?? 0) > 0}
            <span class="num" title="speculative draft tokens accepted (this poll window / lifetime)">{rt?.accept != null ? Math.round(rt.accept * 100) + "%" : "—"} <span class="faint">draft accept</span> <span class="faint">(life {Math.round(100 * m.spec_decode_num_accepted_tokens_total / m.spec_decode_num_draft_tokens_total)}%)</span></span>
          {/if}
          <svg width="160" height="28" style="margin-left: auto;" aria-label="decode tok/s, last 60 s">
            <polyline points={sparkPath(spark[k])} fill="none" stroke="var(--accent)" stroke-width="1.5" />
          </svg>
        </div>
        {#if s.error}<div class="mono faint" style="font-size: 10.5px; margin-top: 4px;">{s.error}</div>{/if}
        <div style="display: flex; gap: 4px; align-items: center; margin-top: 8px; flex-wrap: wrap;">
          <span class="mono faint" style="font-size: 10.5px; width: 70px;">slots {s.slots.filter((x) => x.is_processing).length}/{s.slots.length}</span>
          {#each s.slots as x}
            <div title="slot {x.id}: {x.phase}{x.is_processing ? (x.phase === 'prefill' ? ` ${x.n_prompt_tokens_processed}/${x.n_prompt_tokens} prompt tokens` : ` ${x.n_decoded} generated, ${x.n_remain} remain`) : ''} · ctx {Math.round(100 * x.ctx_fraction)}% of {x.n_ctx.toLocaleString()}"
               style="width: 84px; height: 12px; border-radius: 2px; background: var(--ground-inset); border: 1px solid var(--rule-strong); position: relative; overflow: hidden;">
              {#if x.phase === "prefill"}
                <div style="position: absolute; inset: 0 auto 0 0; width: {Math.round(100 * x.prefill_fraction)}%; background: var(--warn);"></div>
              {:else if x.phase === "decode"}
                <div style="position: absolute; inset: 0 auto 0 0; width: {Math.round(100 * (x.n_decoded / Math.max(1, x.n_decoded + Math.max(0, x.n_remain))))}%; background: var(--accent);"></div>
              {/if}
              <div style="position: absolute; bottom: 0; left: 0; height: 3px; width: {Math.round(100 * x.ctx_fraction)}%; background: var(--note); opacity: .9;"></div>
            </div>
          {/each}
          <span class="faint" style="font-size: 10px;">bar = prefill (amber) / decode (orange) progress · bottom line = context used</span>
        </div>
      </div>
    {:else}
      {#if r.alive}<div class="mono faint" style="font-size: 10.5px; margin-top: 6px;">no loaded models to sample</div>{/if}
    {/each}

    <div class="mono faint" style="font-size: 10.5px; margin-top: 6px;">
      devices {r.state.device_keys.join(", ")}{r.state.visibility_env ? ` · HIP_VISIBLE_DEVICES=${r.state.visibility_env}` : ""}
    </div>
    {#if r.crashed}
      <div class="mono" style="font-size: 11px; margin-top: 6px; color: var(--block);">Process gone — log kept for diagnosis: {r.state.log_path}</div>
    {/if}
  </div>
{:else}
  <div class="empty">No servers running — launch one from Profiles or the Router</div>
{/each}
