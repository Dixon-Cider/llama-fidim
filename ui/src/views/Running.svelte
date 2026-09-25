<script>
  import { onDestroy } from "svelte";
  import { fly, slide, fade } from "svelte/transition";
  import { flip } from "svelte/animate";
  import { api } from "../api.js";
  import { arrive, leave, flipParams, stagger, LAYOUT } from "../motion.js";
  import CountUp from "../components/CountUp.svelte";
  import Skeleton from "../components/Skeleton.svelte";
  import DiffusionCanvas from "../components/DiffusionCanvas.svelte";
  import EndpointCard from "../components/chat/EndpointCard.svelte";
  import { endpointFor } from "../lib/endpoint.js";
  import { copyText } from "../lib/clipboard.js";
  import { openTarget } from "../lib/chat.svelte.js";

  let { go = () => {} } = $props();

  let data = $state({ runs: [], cards: [] });

  // Chat with a server, or hand its endpoint to another program. Copy
  // endpoint copies the base URL at once and shows the model id and
  // snippets beside it.
  function chatWith(run, model = null) {
    openTarget(run, model);
    go("chat");
  }
  let endpointOpen = $state(null);   // { key, copied }
  async function copyEndpoint(key, state, model = null) {
    if (endpointOpen?.key === key) {
      endpointOpen = null;
      return;
    }
    const ok = await copyText(endpointFor(state, model).baseUrl);
    endpointOpen = { key, copied: ok ? "url" : "failed" };
  }
  let loaded = $state(false);   // first poll landed: skeletons give way to content
  let error = $state("");
  // Dismiss with a 3-second undo: the row leaves at once, the state file
  // only goes when the chip times out.
  let pending = $state(null);   // { id, timer }
  function dismiss(id) {
    if (pending) { clearTimeout(pending.timer); stop(pending.id); }
    pending = { id, timer: setTimeout(async () => { const p = pending; pending = null; if (p) await stop(p.id); }, 3000) };
  }
  function undo() {
    if (!pending) return;
    clearTimeout(pending.timer);
    pending = null;
  }
  let paused = $state(false);
  let now = $state(Math.floor(Date.now() / 1000));
  // Per (run, model) history of counter samples for rates and sparklines.
  const hist = new Map();
  let rates = $state({});   // key -> { decode, prompt, accept, canvas }
  let spark = $state({});   // key -> [decode tok/s ...] last 60

  const GIB = 1024 * 1024 * 1024;
  const key = (r, s) => `${r.state.profile_id}:${s?.model ?? ""}`;

  async function poll() {
    if (paused) return;
    try {
      const d = await api("live");
      // Rates from per-slot progress between polls. llama-server's
      // /metrics counters only move when a request finishes, which made
      // the old rate spike once per request; slot n_decoded and
      // n_prompt_tokens_processed move with every token and batch.
      for (const run of d.runs) {
        for (const s of run.samples) {
          const k = key(run.run, s);
          const m = s.metrics ?? {};
          const prev = hist.get(k);
          const slots = new Map();
          let gen = 0, prompt = 0, canvas = 0;
          for (const x of s.slots) {
            // DiffusionGemma: every denoise step predicts the whole canvas
            // (Unsloth Studio's "Speed" counts those tokens).
            const steps = x.diffusion?.steps_done ?? 0, cv = x.diffusion?.canvas ?? 0;
            slots.set(x.id, { task: x.id_task, dec: x.n_decoded, proc: x.n_prompt_tokens_processed, steps });
            const ps = prev?.slots?.get(x.id);
            if (ps && ps.task === x.id_task) {
              if (x.n_decoded > ps.dec) gen += x.n_decoded - ps.dec;
              if (x.n_prompt_tokens_processed > ps.proc) prompt += x.n_prompt_tokens_processed - ps.proc;
              if (steps > ps.steps) canvas += (steps - ps.steps) * cv;
            } else if (ps && x.is_processing) {
              // A new request started since the last poll: what it has
              // done so far is this interval's work.
              gen += x.n_decoded;
              prompt += x.n_prompt_tokens_processed;
              canvas += steps * cv;
            }
          }
          const cur = { t: s.sampled_unix_ms, slots, dn: m.spec_decode_num_draft_tokens_total ?? 0, da: m.spec_decode_num_accepted_tokens_total ?? 0 };
          if (prev && cur.t > prev.t) {
            const dt = (cur.t - prev.t) / 1000;
            const decode = gen / dt;
            const promptRate = prompt / dt;
            const dd = cur.dn - prev.dn;
            const accept = dd > 0 ? (cur.da - prev.da) / dd : null;
            const canvasRate = canvas / dt;
            const dg = s.slots.some((x) => x.diffusion);
            rates = { ...rates, [k]: { decode, prompt: promptRate, accept, canvas: canvasRate } };
            spark = { ...spark, [k]: [...(spark[k] ?? []), dg ? canvasRate : decode].slice(-60) };
          }
          hist.set(k, cur);
        }
      }
      data = d;
      loaded = true;
      error = "";
      if (d.runs.some((r) => r.samples.some((s) => s.slots.some((x) => x.is_processing)))) lastBusy = Date.now();
    } catch (e) {
      error = String(e);
      loaded = true;
    }
  }
  // Poll once a second while any slot is working (and for 30 s after, so the
  // tail of a request and the next one in a conversation stay smooth); every
  // 10 s when every model is idle; not at all while the window is hidden. A
  // sample costs a /proc scan plus a round trip to every server, which at
  // 1 Hz kept the app and its webview at ~14% CPU with nothing running.
  let lastBusy = 0;
  let active = $state(false);   // a model worked within BUSY_TAIL_MS: fast polling, animated chip
  let timer = null;
  const BUSY_MS = 1000, IDLE_MS = 10000, BUSY_TAIL_MS = 30000;
  function busy() { return Date.now() - lastBusy < BUSY_TAIL_MS; }
  async function loop() {
    timer = null;
    if (document.hidden) return;   // visibilitychange restarts the loop
    await poll();
    active = busy();
    if (!document.hidden && timer === null) timer = setTimeout(loop, active ? BUSY_MS : IDLE_MS);
  }
  function onVisibility() {
    if (document.hidden) { if (timer !== null) { clearTimeout(timer); timer = null; } }
    else if (timer === null) loop();
  }
  document.addEventListener("visibilitychange", onVisibility);
  loop();
  // uptime shows minutes: a coarse tick is enough
  const tick = setInterval(() => (now = Math.floor(Date.now() / 1000)), 15000);
  onDestroy(() => { if (timer !== null) clearTimeout(timer); clearInterval(tick); document.removeEventListener("visibilitychange", onVisibility); });

  async function stop(target) {
    try { await api("stop_run", { target }); lastBusy = Date.now(); await poll(); } catch (e) { error = String(e); }
  }
  function uptime(started) {
    const s = Math.max(0, now - started);
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
    if (d) return `${d}d ${h}h`;
    return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`;
  }
  function health(r) {
    if (r.crashed) return { kind: "block", label: "crashed", hint: "The process exited; its log is kept below" };
    if (r.health === "healthy") return { kind: "pass", label: "healthy", hint: "Answers /v1/models and generated a token when probed" };
    if (r.health === "responding-not-generating") return { kind: "warn", label: "stalled", hint: "Answers /v1/models but a 1-token completion fails" };
    return { kind: "block", label: "unreachable", hint: "No answer on /v1/models within 3 s" };
  }
  function phaseOf(samples) {
    return samples.some((s) => s.phase === "decode") ? "decode" : samples.some((s) => s.phase === "prefill") ? "prefill" : "idle";
  }
  const phaseChip = (p) => (p === "decode" ? "accent live" : p === "prefill" ? "warn live" : "plain");
  const phaseHint = (p) => (p === "decode" ? "Generating tokens on at least one slot" : p === "prefill" ? "Reading a prompt on at least one slot" : "No request in progress");
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

  // DiffusionGemma slots mark how much of `generated` is committed; the rest
  // is the current block's draft, rewritten every denoise step.
  const isDiffusion = (x) => x.committed_chars != null;
  function draftChars(x) {
    if (!isDiffusion(x)) return 0;
    const shown = [...(x.generated ?? "")].length;
    return Math.min(shown, Math.max(0, x.generated_chars - x.committed_chars));
  }
  // generated is a tail; the draft is its last draftChars(x) chars.
  function splitDraft(x) {
    const chars = [...(x.generated ?? "")];
    const n = draftChars(x);
    return [chars.slice(0, chars.length - n).join(""), chars.slice(chars.length - n).join("")];
  }
  function slotTitle(x) {
    if (!x.is_processing) return `slot ${x.id}: idle · ${Math.round(100 * x.ctx_fraction)}% of ${x.n_ctx.toLocaleString()} ctx used`;
    const p = x.phase === "prefill"
      ? `prefill ${x.n_prompt_tokens_processed.toLocaleString()} / ${x.n_prompt_tokens.toLocaleString()} prompt tokens`
      : isDiffusion(x)
        ? `denoise: ${x.n_decoded.toLocaleString()} of ${(x.n_decoded + Math.max(0, x.n_remain)).toLocaleString()} canvas tokens`
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
  <span class="sub">Every server started here.</span>
</h1>

<div class="strip">
  {#if !loaded}
    <Skeleton height="118px" /><Skeleton height="118px" />
  {/if}
  {#each data.cards as c, i (c.key)}
    {@const held = heldOn(c.key)}
    {@const total = c.total_mib * 1024 * 1024}
    <div class="tile card-tile" class:hot={c.busy_percent > 50} in:fly={arrive(stagger(i, 60))}>
      <div class="head">
        <span class="name">{c.name.replace(/^AMD /, "")}</span>
        <span class="mono faint">{short(c.key)}</span>
      </div>
      <div class="two">
        <div class="stat">
          <span class="v" title="GPU engine busy across every process, not only servers"><CountUp value={Math.round(c.busy_percent)} /><small>%</small></span>
          <span class="l">busy</span>
        </div>
        <div class="stat">
          <span class="v" title="VRAM held by servers started here, of the card's total"><CountUp value={held / GIB} format={(v) => v.toFixed(1)} /><small>/ {(total / GIB).toFixed(0)} GiB</small></span>
          <span class="l">held</span>
        </div>
        {#if c.free_mib != null}
          {@const freeGib = c.free_mib * 1024 * 1024 / GIB}
          {@const tight = c.display && freeGib < 1.5}
          <div class="stat">
            <span class="v" class:warn-text={tight} title={c.display ? "Free VRAM; this card drives a display, so under 1.5 GiB the desktop evicts the server" : "Free VRAM on the card right now"}>{freeGib.toFixed(1)}<small>GiB</small></span>
            <span class="l">headroom</span>
          </div>
        {/if}
      </div>
      <div class="meter" title="GPU engine busy"><div class="fill" style="width: {Math.round(c.busy_percent)}%;"></div></div>
    </div>
  {/each}
  <div class="strip-tools">
    <button class="btn" onclick={() => (paused = !paused)}>{paused ? "Resume" : "Pause"}</button>
    <span class="chip {paused ? 'plain' : active ? 'pass live' : 'pass'}" title={paused ? "Polling stopped; Resume to sample again" : "Sampled once a second while a model is working, every 10 s when idle"}>{paused ? "paused" : "live"}</span>
    {#if error}<span class="chip block shake">{error}</span>{/if}
    {#if pending}
      <span class="chip warn" transition:fade={{ duration: LAYOUT }}>dismissed {pending.id} · <button class="link" onclick={undo}>undo</button></span>
    {/if}
  </div>
</div>

{#if !loaded}
  <Skeleton height="220px" />
{/if}

{#each data.runs.filter((x) => x.run.state.profile_id !== pending?.id) as row, ri (row.run.state.profile_id + ":" + row.run.state.port)}
  {@const r = row.run}
  {@const h = health(r)}
  {@const vram = row.resident.reduce((a, m) => a + Math.max(m.dedicated_bytes, m.committed_bytes), 0)}
  {@const phase = phaseOf(row.samples)}
  <section class="server" class:crashed={r.crashed} in:fly={arrive(stagger(ri, 60))} out:slide={leave} animate:flip={flipParams}>
    <header class="server-head">
      <div class="who">
        <span class="title">{r.state.profile_id}</span>
        <span class="chip {h.kind}" title={h.hint}>{h.label}</span>
        {#if r.alive}<span class="chip {phaseChip(phase)}" title={phaseHint(phase)}>{phase}</span>{/if}
        {#if (row.evicting_ms_per_s ?? 0) > 20}<span class="chip block live" title="The driver is evicting this process from VRAM ({Math.round(row.evicting_ms_per_s)} ms per second): lower the VRAM share or move a display">evicting</span>{/if}
      </div>
      <div class="facts">
        {#if r.alive}
          <span title="GPU engine busy for this process alone"><b class="num">{Math.round(row.gpu_busy_percent)}%</b> gpu</span>
          {#if (row.evicted_ms ?? 0) > 0}<span title="Time this process has spent evicted from VRAM since it started"><b class="num">{(row.evicted_ms / 1000).toFixed(0)}s</b> evicted</span>{/if}
          <span title="VRAM held by this server and its children (dedicated, or committed while paging in)"><b class="num">{(vram / GIB).toFixed(1)} GiB</b>{#if row.resident.length}&nbsp;on {[...new Set(row.resident.map((m) => short(m.card ?? "?")))].join(" + ")}{/if}</span>
        {/if}
        <span title="Time since the process started"><b class="num">{uptime(r.state.started_unix)}</b> up</span>
        <span class="mono" title="Listens on port {r.state.port} as “{r.state.alias}”; Copy endpoint shows the URL">pid {r.state.pid}</span>
      </div>
      {#if r.alive && r.state.profile_id !== "router"}
        <button class="btn" onclick={() => chatWith(r.state.profile_id)} title="Open a chat with this server">Chat</button>
        <button class="btn" class:primary={endpointOpen?.key === r.state.profile_id} onclick={() => copyEndpoint(r.state.profile_id, r.state)} title="Copy the base URL; the model id and snippets show below">Copy endpoint</button>
      {/if}
      {#if r.alive}<button class="btn danger" onclick={() => stop(r.state.profile_id)}>Stop</button>
      {:else}<button class="btn" onclick={() => dismiss(r.state.profile_id)} title="Forget this run; the log file stays">Dismiss</button>{/if}
    </header>
    {#if endpointOpen?.key === r.state.profile_id}
      <div class="endpoint-pop" transition:slide={leave}>
        <EndpointCard endpoint={endpointFor(r.state, null, !!row.has_api_key)} copiedFirst={endpointOpen.copied} onclose={() => (endpointOpen = null)} />
      </div>
    {/if}

    {#each row.samples as s, si (key(r, s))}
      {@const k = key(r, s)}
      {@const rt = rates[k]}
      {@const m = s.metrics ?? {}}
      {@const pts = sparkPts(spark[k])}
      {@const busySlots = s.slots.filter((x) => x.is_processing).length}
      {@const hasDraft = (m.spec_decode_num_draft_tokens_total ?? 0) > 0}
      {@const dg = s.slots.some((x) => x.diffusion)}
      <div class="model" class:active={s.phase !== "idle"} in:fly={arrive(stagger(si, 40))} out:slide={leave} animate:flip={flipParams}>
        <div class="ident">
          <div class="model-name">{s.model ?? r.state.alias}</div>
          <span class="chip {phaseChip(s.phase)}" title={phaseHint(s.phase)}>{s.phase}</span>
          {#if s.slots.some((x) => x.loop_hint)}<span class="chip block live" title="A slot is repeating the same fragment; click it to see the text">looping</span>{/if}
          <div class="slot-count" title="Slots serving a request, of the {s.slots.length} this model has (-np)"><b class="num">{busySlots}</b><span class="faint"> / {s.slots.length} busy</span></div>
          {#if s.error}<div class="err">{s.error}</div>{/if}
          {#if s.model && r.alive}
            <div class="row-actions">
              <button class="btn small" onclick={() => chatWith(r.state.profile_id, s.model)} title="Open a chat with this model through the router">Chat</button>
              <button class="btn small" class:primary={endpointOpen?.key === k} onclick={() => copyEndpoint(k, r.state, s.model)} title="Copy the router's base URL; the model id and snippets show below">Copy endpoint</button>
            </div>
          {/if}
        </div>
        {#if endpointOpen?.key === k}
          <div class="endpoint-pop in-model" transition:slide={leave}>
            <EndpointCard endpoint={endpointFor(r.state, s.model, !!row.keyed_models?.includes(s.model))} copiedFirst={endpointOpen.copied} onclose={() => (endpointOpen = null)} />
          </div>
        {/if}

        <div class="stats">
          {#if dg}
            <!-- Diffusion: text delivered (compare with autoregressive decode) vs.
                 canvas tokens predicted per step (Unsloth Studio's "Speed"). -->
            <div class="stat" class:dim={!m.predicted_tokens_seconds}>
              <span class="v" title="Answer tokens per second over the last reply, prefill included; compare with a decode rate">{fmt1(m.predicted_tokens_seconds)}<small>tok/s</small></span>
              <span class="l">last reply</span>
            </div>
            <div class="stat" class:dim={!rt?.canvas}>
              <span class="v" title="Canvas tokens predicted per second right now; each denoise step redoes the whole 256-token block">{fmt0(rt?.canvas)}<small>tok/s</small></span>
              <span class="l">canvas</span>
            </div>
            <div class="stat" class:dim={!m.diffusion_canvas_tokens_seconds}>
              <span class="v" title="Unsloth Studio's Speed for the last reply: 256 × denoise steps ÷ time">{fmt0(m.diffusion_canvas_tokens_seconds)}<small>tok/s</small></span>
              <span class="l">last canvas</span>
            </div>
          {:else}
          <div class="stat" class:dim={!(rt?.decode || (busySlots && m.gen_throughput))}>
            <span class="v" title="Tokens generated per second right now, over the last poll">{fmt1(rt?.decode || (busySlots && m.gen_throughput) || 0)}<small>tok/s</small></span>
            <span class="l">decode</span>
          </div>
          <div class="stat" class:dim={!rt?.prompt}>
            <span class="v" title="Prompt tokens read per second right now, over the last poll">{fmt0(rt?.prompt)}<small>tok/s</small></span>
            <span class="l">prefill</span>
          </div>
          {/if}
          <div class="stat" class:dim={!(m.requests_processing || m.requests_deferred)}>
            <span class="v" title="Requests being served now, plus those waiting in the queue">{m.requests_processing ?? 0}<small>+ {m.requests_deferred ?? 0} queued</small></span>
            <span class="l">in flight</span>
          </div>
          {#if hasDraft}
            <div class="stat">
              <span class="v" title={rt?.accept != null ? "Share of draft tokens the main model accepted since the last poll" : "Share of draft tokens the main model accepted over this server's lifetime"}>{rt?.accept != null ? Math.round(rt.accept * 100) : Math.round(100 * m.spec_decode_num_accepted_tokens_total / m.spec_decode_num_draft_tokens_total)}<small>%</small></span>
              <span class="l">accepted</span>
            </div>
          {:else if m.spec_accept_rate != null}
            <div class="stat" class:dim={!m.spec_accept_rate}>
              <span class="v" title="Share of draft tokens accepted in the last batch, and mean accepted length per verify step">{Math.round(m.spec_accept_rate * 100)}<small>% · {(m.spec_accept_length ?? 0).toFixed(1)} tok</small></span>
              <span class="l">accepted</span>
            </div>
          {/if}
          {#if m.cache_hit_rate != null}
            <div class="stat" class:dim={!m.cache_hit_rate}>
              <span class="v" title="Prompt tokens served from the prefix cache, last batch; low mid-chat means the prefix was evicted">{Math.round(m.cache_hit_rate * 100)}<small>%</small></span>
              <span class="l">cache hits</span>
            </div>
          {/if}
          {#if m.token_usage != null}
            <div class="stat" class:dim={!m.token_usage}>
              <span class="v" title="KV cache tokens held of the pool ({(m.kv_used_tokens ?? 0).toLocaleString()} of {(m.max_total_num_tokens ?? 0).toLocaleString()}){m.mamba_usage != null ? `; state slots are GatedDeltaNet's (mamba)` : ""}">{Math.round(m.token_usage * 100)}<small>%{m.mamba_usage != null ? ` · state ${Math.round(m.mamba_usage * 100)}%` : ""}</small></span>
              <span class="l">cache used</span>
            </div>
          {/if}
          {#if m.num_retracted_reqs != null && m.num_retracted_reqs > 0}
            <div class="stat">
              <span class="v warn-text" title="Requests pulled back to the queue because the KV cache filled; the context budget is too tight">{m.num_retracted_reqs}</span>
              <span class="l">retracted</span>
            </div>
          {/if}
          {#if m.time_to_first_token_seconds_count > 0}
            <div class="stat">
              <span class="v" title="Mean time to first token over {m.time_to_first_token_seconds_count} requests; inter-token latency {m.inter_token_latency_seconds_count > 0 ? (1000 * m.inter_token_latency_seconds_sum / m.inter_token_latency_seconds_count).toFixed(0) : "?"} ms mean">{(m.time_to_first_token_seconds_sum / m.time_to_first_token_seconds_count).toFixed(1)}<small>s</small></span>
              <span class="l">first token</span>
            </div>
          {/if}
        </div>

        <div class="spark" aria-label="{dg ? "canvas" : "decode"} tok/s, last 60 seconds">
          <svg viewBox="0 0 {SW} {SH}" preserveAspectRatio="none">
            <line x1="0" y1={SH - 3} x2={SW} y2={SH - 3} class="base" />
            {#if pts.length > 1}
              <path d={sparkArea(pts)} class="area" />
              <polyline points={sparkLine(pts)} class="line" />
              <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r="2.5" class="end" />
            {/if}
          </svg>
          <span class="l" title="{dg ? "Canvas" : "Decode"} tokens per second, one sample a second for the last minute">{dg ? "canvas" : "decode"} tok/s · 60 s{pts.length > 1 ? ` · peak ${dg ? fmt0(Math.max(...spark[k])) : fmt1(Math.max(...spark[k]))}` : ""}</span>
        </div>

        <div class="slots">
          {#each s.slots as x}
            {@const prog = slotProgress(x)}
            <button type="button" class="slot {x.phase}" class:loop={!!x.loop_hint} class:open={!!open[slotKey(k, x.id)]} title="{slotTitle(x)}; click for its text" onclick={() => toggleSlot(k, x.id)}>
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
                {:else if x.phase === "decode"}{isDiffusion(x) ? "denoising" : "decoding"}
                {:else}&nbsp;{/if}
              </span>
              <div class="ctx" style="width: {Math.round(100 * x.ctx_fraction)}%;" title="Context used in this slot"></div>
            </button>
          {/each}
        </div>

        {#each s.slots.filter((x) => open[slotKey(k, x.id)]) as x (x.id)}
          <div class="drawer" class:loop={!!x.loop_hint} transition:slide={leave}>
            <div class="dhead">
              <span class="mono" style="font-weight: 700;">slot {x.id}</span>
              <span class="chip {phaseChip(x.phase)}">{x.phase}</span>
              {#if x.loop_hint}<span class="chip block live" title="The same fragment came back {x.loop_hint.repeats} times: “{trimFrag(x.loop_hint.fragment)}”">looping ×{x.loop_hint.repeats}</span>{/if}
              {#if x.generated != null}<span class="faint small" title="Characters generated, then in the prompt{x.prompt_chars > 4000 || x.generated_chars > 4000 ? "; the panes show the last 4,000 of each" : ""}">{x.generated_chars.toLocaleString()} out · {x.prompt_chars.toLocaleString()} in</span>{/if}
              {#if draftChars(x) > 0}<span class="chip plain" title="Each denoise step rewrites the block's draft (dimmed) until it settles and is committed">draft {draftChars(x).toLocaleString()}</span>{/if}
              <button class="btn small" style="margin-left: auto;" onclick={() => toggleSlot(k, x.id)}>Close</button>
            </div>
            {#if x.diffusion}
              <DiffusionCanvas host={r.state.host} port={r.state.port} slot={x} />
            {/if}
            {#if x.prompt == null && x.generated == null}
              <div class="faint small" title="For the router, turn it on for any member profile and relaunch the router">No slot text — turn on <b>trace tokens</b> in the profile's Advanced section and reload it.</div>
            {:else}
              <div class="panes">
                <div class="pane">
                  <div class="pl" title="The last prompt this slot received">prompt</div>
                  <pre use:stick>{x.prompt ?? ""}</pre>
                </div>
                <div class="pane">
                  <div class="pl" title={x.is_processing ? (isDiffusion(x) ? "Generated so far: committed text, then the current block's draft dimmed" : "Generated so far in this request") : "Generated in the last request"}>generated</div>
                  {#if isDiffusion(x)}
                    {@const parts = splitDraft(x)}
                    <pre use:stick>{parts[0]}<span class="draft">{parts[1]}</span></pre>
                  {:else}
                    <pre use:stick>{x.generated ?? ""}</pre>
                  {/if}
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    {:else}
      {#if r.alive && row.fronting?.length}
        <div class="empty small" title="The router forwards to these models; each has its own card below">fronting {row.fronting.join(" · ")}</div>
      {:else if r.alive}<div class="empty small">No models loaded</div>{/if}
    {/each}

    <footer class="server-foot">
      <span>devices {r.state.device_keys.map(short).join(", ")}{r.state.visibility_env ? ` · HIP_VISIBLE_DEVICES=${r.state.visibility_env}` : ""}</span>
      {#if r.crashed}<span class="gone" title="The process exited; its log is kept for diagnosis">log <span class="mono">{r.state.log_path}</span></span>{/if}
    </footer>
  </section>
{:else}
  {#if loaded}<div class="card" in:fade={{ duration: LAYOUT }}><div class="empty">Nothing running — Profiles → Launch, or Router → Start</div></div>{/if}
{/each}

{#if data.runs.some((x) => x.samples.length)}
  <div class="legend">
    <span title="Fill grows as the prompt is read"><i class="sw prefill"></i> prefill</span>
    <span title="Fill grows with tokens generated so far"><i class="sw decode"></i> decode</span>
    <span title="The thin bar is how much of the slot's context is used"><i class="sw ctx"></i> context used</span>
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
  .endpoint-pop { padding: 14px 18px; border-bottom: 1px solid var(--rule); background: var(--ground-inset); }
  .endpoint-pop.in-model { grid-column: 1 / -1; border: 1px solid var(--rule-strong); border-radius: 6px; padding: 12px 14px; }
  .row-actions { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 2px; }

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
  .pane pre .draft { opacity: 0.55; font-style: italic; }
  .warn-text { color: var(--warn, #e8a33d); }
  @media (max-width: 1100px) { .panes { grid-template-columns: 1fr; } }
  .slot { transition: border-color var(--t-layout), background-color var(--t-layout), box-shadow var(--t-fast); }
  .slot .fill { position: absolute; inset: 0 auto 0 0; transition: width .3s ease, background-color var(--t-layout); }
  .slot .what { transition: color var(--t-layout); }
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
