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
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
    if (d) return `${d}d ${h}h`;
    return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`;
  }
  function health(r) {
    if (r.crashed) return { kind: "block", label: "crashed" };
    if (r.health === "healthy") return { kind: "pass", label: "healthy" };
    if (r.health === "responding-not-generating") return { kind: "warn", label: "responding, not generating" };
    return { kind: "block", label: "not responding" };
  }
  function phaseOf(samples) {
    return samples.some((s) => s.phase === "decode") ? "decode" : samples.some((s) => s.phase === "prefill") ? "prefill" : "idle";
  }
  const phaseChip = (p) => (p === "decode" ? "accent live" : p === "prefill" ? "warn live" : "plain");
  const heldOn = (cardKey) => data.runs.reduce((a, row) => a + row.resident.filter((m) => m.card === cardKey).reduce((b, m) => b + Math.max(m.dedicated_bytes, m.committed_bytes), 0), 0);
  const short = (k) => String(k ?? "").split(":").pop();
  const fmt1 = (v) => (v == null ? "—" : Number(v).toFixed(1));
  const fmt0 = (v) => (v == null ? "—" : Math.round(Number(v)).toLocaleString());

  // Sparkline: area + line + endpoint over the last 60 samples.
  const SW = 260, SH = 48;
  function sparkPts(vals) {
    if (!vals?.length) return [];
    const max = Math.max(1, ...vals);
    const n = Math.max(1, vals.length - 1);
    return vals.map((v, i) => [(i / n) * SW, SH - 3 - (v / max) * (SH - 8)]);
  }
  const sparkLine = (pts) => pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const sparkArea = (pts) => (pts.length ? `M0,${SH} L${sparkLine(pts)} L${SW},${SH} Z` : "");

  function slotTitle(x) {
    if (!x.is_processing) return `slot ${x.id}: idle · ${Math.round(100 * x.ctx_fraction)}% of ${x.n_ctx.toLocaleString()} ctx used`;
    const p = x.phase === "prefill"
      ? `prefill ${x.n_prompt_tokens_processed.toLocaleString()} / ${x.n_prompt_tokens.toLocaleString()} prompt tokens`
      : `decode ${x.n_decoded.toLocaleString()} generated, ${Math.max(0, x.n_remain).toLocaleString()} remaining`;
    return `slot ${x.id}: ${p} · ${Math.round(100 * x.ctx_fraction)}% of ${x.n_ctx.toLocaleString()} ctx used`;
  }
  function slotProgress(x) {
    if (x.phase === "prefill") return x.prefill_fraction;
    if (x.phase === "decode") return x.n_decoded / Math.max(1, x.n_decoded + Math.max(0, x.n_remain));
    return 0;
  }

  // Slot drawers: which (model, slot) text panels are open.
  let open = $state({});
  const slotKey = (k, id) => `${k}#${id}`;
  function toggleSlot(k, id) {
    const sk = slotKey(k, id);
    open = { ...open, [sk]: !open[sk] };
  }
  const trimFrag = (f) => (f.length > 48 ? f.slice(0, 48) + "…" : f).replace(/\n/g, "⏎");
  // Keep a text pane pinned to its end as new tokens arrive.
  function stick(node) {
    const toEnd = () => { node.scrollTop = node.scrollHeight; };
    const o = new MutationObserver(toEnd);
    o.observe(node, { childList: true, characterData: true, subtree: true });
    toEnd();
    return { destroy() { o.disconnect(); } };
  }
</script>

<h1>
  Running
  <span class="sub">Each server llamactl started, sampled once a second.</span>
</h1>

<div class="strip">
  {#each data.cards as c}
    {@const held = heldOn(c.key)}
    {@const total = c.total_mib * 1024 * 1024}
    <div class="tile card-tile" class:hot={c.busy_percent > 50}>
      <div class="head">
        <span class="name">{c.name.replace(/^AMD /, "")}</span>
        <span class="mono faint">{short(c.key)}</span>
      </div>
      <div class="two">
        <div class="stat">
          <span class="v">{Math.round(c.busy_percent)}<small>% busy</small></span>
          <span class="l">GPU engine, all processes</span>
        </div>
        <div class="stat">
          <span class="v">{(held / GIB).toFixed(1)}<small>/ {(total / GIB).toFixed(0)} GiB</small></span>
          <span class="l">VRAM held by servers</span>
        </div>
      </div>
      <div class="meter" title="GPU busy"><div class="fill" style="width: {Math.round(c.busy_percent)}%;"></div></div>
    </div>
  {/each}
  <div class="strip-tools">
    <button class="btn" onclick={() => (paused = !paused)}>{paused ? "Resume" : "Pause"}</button>
    <span class="chip {paused ? 'plain' : 'pass live'}">{paused ? "paused" : "live · 1 Hz"}</span>
    {#if error}<span class="chip block">{error}</span>{/if}
  </div>
</div>

{#each data.runs as row}
  {@const r = row.run}
  {@const h = health(r)}
  {@const vram = row.resident.reduce((a, m) => a + Math.max(m.dedicated_bytes, m.committed_bytes), 0)}
  {@const phase = phaseOf(row.samples)}
  <section class="server" class:crashed={r.crashed}>
    <header class="server-head">
      <div class="who">
        <span class="title">{r.state.profile_id}</span>
        <span class="chip {h.kind}">{h.label}</span>
        {#if r.alive}<span class="chip {phaseChip(phase)}">{phase}</span>{/if}
      </div>
      <div class="facts">
        {#if r.alive}
          <span title="GPU engine utilization of this process"><b class="num">{Math.round(row.gpu_busy_percent)}%</b> gpu</span>
          <span title="resident VRAM (dedicated, or committed while paging in)"><b class="num">{(vram / GIB).toFixed(1)} GiB</b>{#if row.resident.length}&nbsp;on {row.resident.map((m) => short(m.card ?? "?")).join(" + ")}{/if}</span>
        {/if}
        <span><b class="num">{uptime(r.state.started_unix)}</b> up</span>
        <span class="mono">pid {r.state.pid} · :{r.state.port} · {r.state.alias}</span>
      </div>
      {#if r.alive}<button class="btn danger" onclick={() => stop(r.state.profile_id)}>Stop</button>
      {:else}<button class="btn" onclick={() => stop(r.state.profile_id)} title="forget this run; the log file stays">Dismiss</button>{/if}
    </header>

    {#each row.samples as s}
      {@const k = key(r, s)}
      {@const rt = rates[k]}
      {@const m = s.metrics ?? {}}
      {@const pts = sparkPts(spark[k])}
      {@const busySlots = s.slots.filter((x) => x.is_processing).length}
      {@const hasDraft = (m.spec_decode_num_draft_tokens_total ?? 0) > 0}
      <div class="model" class:active={s.phase !== "idle"}>
        <div class="ident">
          <div class="model-name">{s.model ?? r.state.alias}</div>
          <span class="chip {phaseChip(s.phase)}">{s.phase}</span>
          {#if s.slots.some((x) => x.loop_hint)}<span class="chip block live" title="a slot is repeating the same fragment back-to-back; click it to see the text">looping</span>{/if}
          <div class="slot-count"><b class="num">{busySlots}</b><span class="faint"> of {s.slots.length} slots busy</span></div>
          {#if s.error}<div class="err">{s.error}</div>{/if}
        </div>

        <div class="stats">
          <div class="stat" class:dim={!rt?.decode}>
            <span class="v">{fmt1(rt?.decode)}<small>tok/s</small></span>
            <span class="l">decode, now</span>
          </div>
          <div class="stat" class:dim={!rt?.prompt}>
            <span class="v">{fmt0(rt?.prompt)}<small>tok/s</small></span>
            <span class="l">prefill, now</span>
          </div>
          <div class="stat" class:dim={!(m.requests_processing || m.requests_deferred)}>
            <span class="v">{m.requests_processing ?? 0}<small>+ {m.requests_deferred ?? 0} queued</small></span>
            <span class="l">requests in flight</span>
          </div>
          {#if hasDraft}
            <div class="stat">
              <span class="v">{rt?.accept != null ? Math.round(rt.accept * 100) : Math.round(100 * m.spec_decode_num_accepted_tokens_total / m.spec_decode_num_draft_tokens_total)}<small>%</small></span>
              <span class="l">draft accepted{rt?.accept != null ? "" : ", lifetime"}</span>
            </div>
          {/if}
        </div>

        <div class="spark" aria-label="decode tok/s, last 60 seconds">
          <svg viewBox="0 0 {SW} {SH}" preserveAspectRatio="none">
            <line x1="0" y1={SH - 3} x2={SW} y2={SH - 3} class="base" />
            {#if pts.length > 1}
              <path d={sparkArea(pts)} class="area" />
              <polyline points={sparkLine(pts)} class="line" />
              <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r="2.5" class="end" />
            {/if}
          </svg>
          <span class="l">decode tok/s · last 60 s{#if pts.length > 1} · peak {fmt1(Math.max(...spark[k]))}{/if}</span>
        </div>

        <div class="slots">
          {#each s.slots as x}
            {@const prog = slotProgress(x)}
            <button type="button" class="slot {x.phase}" class:loop={!!x.loop_hint} class:open={!!open[slotKey(k, x.id)]} title="{slotTitle(x)} · click for prompt and generated text" onclick={() => toggleSlot(k, x.id)}>
              <div class="fill" style="width: {Math.round(100 * prog)}%;"></div>
              <span class="n">{x.id}</span>
              <span class="p">
                {#if x.phase === "prefill"}{Math.round(100 * prog)}%
                {:else if x.phase === "decode"}{x.n_decoded.toLocaleString()}
                {:else}idle{/if}
              </span>
              <span class="what">
                {#if x.loop_hint}loop ×{x.loop_hint.repeats}
                {:else if x.phase === "prefill"}prefill
                {:else if x.phase === "decode"}decoding
                {:else}&nbsp;{/if}
              </span>
              <div class="ctx" style="width: {Math.round(100 * x.ctx_fraction)}%;" title="context used"></div>
            </button>
          {/each}
        </div>

        {#each s.slots.filter((x) => open[slotKey(k, x.id)]) as x (x.id)}
          <div class="drawer" class:loop={!!x.loop_hint}>
            <div class="dhead">
              <span class="mono" style="font-weight: 700;">slot {x.id}</span>
              <span class="chip {phaseChip(x.phase)}">{x.phase}</span>
              {#if x.loop_hint}<span class="chip block live">looping × {x.loop_hint.repeats} <span style="text-transform: none; letter-spacing: 0; font-weight: 500;">“{trimFrag(x.loop_hint.fragment)}”</span></span>{/if}
              {#if x.generated != null}<span class="faint small">{x.generated_chars.toLocaleString()} chars generated · prompt {x.prompt_chars.toLocaleString()} chars{#if x.prompt_chars > 4000 || x.generated_chars > 4000} · showing the last 4,000 of each{/if}</span>{/if}
              <button class="btn small" style="margin-left: auto;" onclick={() => toggleSlot(k, x.id)}>Close</button>
            </div>
            {#if x.prompt == null && x.generated == null}
              <div class="faint small">This server does not expose slot text. Turn on <b>trace tokens</b> in the profile's Advanced section and reload it. For the router, turn it on for any member and relaunch the router.</div>
            {:else}
              <div class="panes">
                <div class="pane">
                  <div class="pl">last prompt received</div>
                  <pre use:stick>{x.prompt ?? ""}</pre>
                </div>
                <div class="pane">
                  <div class="pl">generated {x.is_processing ? "so far" : "in the last request"}</div>
                  <pre use:stick>{x.generated ?? ""}</pre>
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {:else}
      {#if r.alive}<div class="empty small">No loaded models to sample</div>{/if}
    {/each}

    <footer class="server-foot">
      <span>devices {r.state.device_keys.map(short).join(", ")}{r.state.visibility_env ? ` · HIP_VISIBLE_DEVICES=${r.state.visibility_env}` : ""}</span>
      {#if r.crashed}<span class="gone">Process gone. Log kept for diagnosis: <span class="mono">{r.state.log_path}</span></span>{/if}
    </footer>
  </section>
{:else}
  <div class="card"><div class="empty">Nothing is running. Launch a profile, or start the router.</div></div>
{/each}

{#if data.runs.some((x) => x.samples.length)}
  <div class="legend">
    <span><i class="sw prefill"></i> prefill, fill = prompt tokens read</span>
    <span><i class="sw decode"></i> decoding, fill = tokens generated so far</span>
    <span><i class="sw ctx"></i> thin bar = context used in that slot</span>
  </div>
{/if}

<style>
  .strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 20px; align-items: stretch; }
  .strip-tools { display: flex; gap: 10px; align-items: center; justify-content: flex-end; flex-wrap: wrap; }
  .card-tile .two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .card-tile.hot .meter .fill { background: var(--accent); }
  .card-tile .meter .fill { background: var(--ink-faint); }

  .server { background: var(--ground-raised); border: 1px solid var(--rule); border-radius: var(--radius); margin-bottom: 16px; overflow: hidden; }
  .server.crashed { border-color: rgba(232, 125, 110, 0.35); }
  .server-head { display: flex; align-items: center; gap: 18px; padding: 14px 18px; border-bottom: 1px solid var(--rule); flex-wrap: wrap; }
  .server.crashed .server-head { background: var(--block-bg); }
  .who { display: flex; align-items: center; gap: 10px; }
  .who .title { font-family: var(--display); font-weight: 700; font-size: 16px; letter-spacing: -0.01em; }
  .facts { display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; color: var(--ink-muted); font-size: 12.5px; margin-left: auto; }
  .facts b { color: var(--ink); font-weight: 600; }
  .facts .mono { font-size: 11.5px; color: var(--ink-faint); }

  .model {
    display: grid; grid-template-columns: 200px 1fr 280px; grid-template-areas: "ident stats spark" "slots slots slots";
    gap: 14px 22px; padding: 16px 18px; border-bottom: 1px solid var(--rule);
  }
  .model:last-of-type { border-bottom: none; }
  .model.active { background: linear-gradient(90deg, var(--accent-soft), transparent 45%); }
  .ident { grid-area: ident; display: flex; flex-direction: column; gap: 6px; align-items: flex-start; }
  .model-name { font-family: var(--mono); font-weight: 700; font-size: 15px; letter-spacing: -0.01em; }
  .slot-count { font-size: 12.5px; }
  .slot-count b { font-size: 14px; }
  .err { font-family: var(--mono); font-size: 11.5px; color: var(--block); word-break: break-word; }
  .stats { grid-area: stats; display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px 18px; align-content: start; }
  .spark { grid-area: spark; display: flex; flex-direction: column; gap: 4px; min-width: 0; }
  .spark svg { width: 100%; height: 48px; display: block; }
  .spark .base { stroke: var(--rule-strong); stroke-width: 1; }
  .spark .area { fill: var(--accent-soft); }
  .spark .line { fill: none; stroke: var(--accent); stroke-width: 1.5; vector-effect: non-scaling-stroke; }
  .spark .end { fill: var(--accent); }
  .spark .l { font-size: 11.5px; color: var(--ink-faint); }

  .slots { grid-area: slots; display: grid; grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); gap: 8px; }
  .slot {
    position: relative; height: 54px; border-radius: 6px; overflow: hidden; cursor: pointer;
    appearance: none; text-align: left; color: inherit; font: inherit; width: 100%;
    background: var(--ground-inset); border: 1px solid var(--rule-strong);
    display: grid; grid-template-columns: auto 1fr; grid-template-rows: 1fr auto; align-items: center; padding: 6px 10px 8px;
    font-family: var(--mono); font-variant-numeric: tabular-nums;
  }
  .slot:hover { border-color: var(--ink-faint); }
  .slot.open { box-shadow: 0 0 0 2px var(--accent-soft); border-color: var(--accent); }
  .slot.loop { border-color: var(--block); }
  .slot.loop .what { color: var(--block); font-weight: 600; }
  .slot.loop .fill { background: var(--block-bg); box-shadow: inset -1px 0 0 var(--block); }

  .drawer { grid-column: 1 / -1; border: 1px solid var(--rule-strong); border-radius: 6px; background: var(--ground-inset); padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 10px; }
  .drawer.loop { border-color: rgba(232, 125, 110, 0.5); }
  .dhead { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .panes { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .pane { min-width: 0; display: flex; flex-direction: column; gap: 4px; }
  .pane .pl { font-size: 11.5px; font-weight: 600; color: var(--ink-muted); }
  .pane pre {
    margin: 0; height: 220px; overflow: auto; white-space: pre-wrap; word-break: break-word;
    background: #0b0f13; border: 1px solid var(--rule); border-radius: 5px; padding: 10px 12px;
    font-family: var(--mono); font-size: 12px; line-height: 1.55; color: var(--ink-muted);
  }
  .drawer.loop .pane:last-child pre { border-color: rgba(232, 125, 110, 0.5); color: var(--ink); }
  @media (max-width: 1100px) { .panes { grid-template-columns: 1fr; } }
  .slot .fill { position: absolute; inset: 0 auto 0 0; transition: width .3s ease; }
  .slot.prefill .fill { background: var(--warn-bg); box-shadow: inset -1px 0 0 var(--warn); }
  .slot.decode .fill { background: var(--accent-soft); box-shadow: inset -1px 0 0 var(--accent); }
  .slot.prefill { border-color: rgba(220, 176, 74, 0.55); }
  .slot.decode { border-color: rgba(224, 140, 76, 0.6); }
  .slot .n { position: relative; font-size: 11px; color: var(--ink-faint); align-self: start; }
  .slot .p { position: relative; font-size: 15px; font-weight: 600; text-align: right; color: var(--ink); align-self: start; }
  .slot.idle .p { color: var(--ink-faint); font-weight: 500; font-size: 12px; }
  .slot .what { position: relative; grid-column: 1 / -1; font-family: var(--sans); font-size: 11px; color: var(--ink-muted); }
  .slot.prefill .what { color: var(--warn); }
  .slot.decode .what { color: var(--accent); }
  .slot .ctx { position: absolute; left: 0; bottom: 0; height: 3px; background: var(--note); opacity: 0.9; }

  .server-foot { display: flex; gap: 18px; flex-wrap: wrap; padding: 8px 18px 10px; font-family: var(--mono); font-size: 11px; color: var(--ink-faint); border-top: 1px solid var(--rule); }
  .server-foot .gone { color: var(--block); font-family: var(--sans); font-size: 12px; }
  .server-foot .gone .mono { color: var(--block); font-size: 11px; }
  .legend { display: flex; gap: 22px; flex-wrap: wrap; font-size: 12px; color: var(--ink-faint); margin-top: 4px; }
  .legend .sw { display: inline-block; width: 14px; height: 10px; border-radius: 2px; vertical-align: -1px; margin-right: 6px; border: 1px solid var(--rule-strong); }
  .legend .sw.prefill { background: var(--warn-bg); border-color: var(--warn); }
  .legend .sw.decode { background: var(--accent-soft); border-color: var(--accent); }
  .legend .sw.ctx { height: 3px; background: var(--note); border: none; vertical-align: 2px; }
  @media (prefers-reduced-motion: reduce) { .slot .fill { transition: none; } }
  @media (max-width: 1100px) {
    .model { grid-template-columns: 1fr 1fr; grid-template-areas: "ident stats" "spark spark" "slots slots"; }
  }
</style>
