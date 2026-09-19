<script>
  // The DiffusionGemma block as it denoises, the way Unsloth Studio shows it:
  // block/step header, a progress bar for the step, and the 256-token canvas
  // repainted every step, fading in as it settles. Live follows the running
  // request (~5 polls/s while this is open); Replay plays back every step of
  // the current or last reply from the helper's /frames.
  //
  // Running passes the slot it already polls. The chat passes `follow`
  // (poll whatever the slot is doing), `idTask` (draw only its own request;
  // `queued` says where it waits meanwhile) and, after a reply, `snapshot`:
  // that reply's frames, kept because another client's request would
  // replace them on the server.
  import { onDestroy, onMount } from "svelte";
  import { api } from "../api.js";

  let { host, port, slot = null, follow = false, idTask = null, queued = null, snapshot = null, start = "live" } = $props();

  let mode = $state("live");      // "live" | "replay"
  let fresh = $state(null);       // latest slot from the fast poll
  let pollErr = $state("");
  let frames = $state([]);
  let dropped = $state(0);
  let idx = $state(0);
  let playing = $state(false);
  let stepMs = $state(90);
  let replayErr = $state("");
  let timer = null;
  let holding = false;

  const seen = $derived(fresh ?? slot);
  // With an idTask, a slot busy with any other request is not ours to draw;
  // nor is anything while our request still waits in the queue.
  const theirs = $derived(
    seen != null && (idTask != null ? seen.id_task !== idTask : follow && queued != null)
  );
  const s = $derived(theirs ? null : seen);
  const d = $derived(s?.diffusion ?? null);
  const processing = $derived(!!s?.is_processing);

  // The draft is the part of `generated` past the committed text; generated
  // may be a tail, so count from its end.
  const draft = $derived.by(() => {
    if (!s || s.committed_chars == null) return "";
    const chars = [...(s.generated ?? "")];
    const n = Math.min(chars.length, Math.max(0, (s.generated_chars ?? 0) - s.committed_chars));
    return chars.slice(chars.length - n).join("");
  });

  // Fast poll while live, open and running (or always, when following).
  let inflight = false;
  $effect(() => {
    if (mode !== "live" || !(follow || slot?.is_processing)) { fresh = null; return; }
    const id = setInterval(async () => {
      if (inflight) return;
      inflight = true;
      try {
        const r = await api("live_one", { host, port });
        fresh = r?.slots?.[0] ?? null;
        pollErr = "";
      } catch (e) {
        pollErr = String(e);
      } finally {
        inflight = false;
      }
    }, 200);
    return () => clearInterval(id);
  });

  // Replay: Studio's player loop (hold on the settled canvas, then start over).
  function schedule(ms) { clearTimeout(timer); timer = setTimeout(tick, ms); }
  function tick() {
    if (!playing || !frames.length) return;
    if (idx >= frames.length - 1) {
      if (!holding) { holding = true; schedule(1400); return; }
      holding = false; idx = 0; schedule(700); return;
    }
    idx += 1;
    schedule(stepMs);
  }
  function setPlaying(p) {
    playing = p;
    if (p) { holding = false; schedule(stepMs); } else clearTimeout(timer);
  }
  async function startReplay() {
    mode = "replay";
    replayErr = "";
    try {
      const r = snapshot ?? (await api("dg_frames", { host, port }));
      if (idTask != null && r?.id_task !== idTask) {
        throw new Error(`the server's replay now holds another request (#${r?.id_task ?? "?"}), not this reply (#${idTask})`);
      }
      frames = r?.frames ?? [];
      dropped = r?.dropped ?? 0;
      idx = 0;
      setPlaying(frames.length > 1);
    } catch (e) {
      frames = [];
      replayErr = String(e?.message ?? e);
      setPlaying(false);
    }
  }
  function goLive() { setPlaying(false); mode = "live"; }
  function scrub(e) { setPlaying(false); idx = parseInt(e.target.value, 10) || 0; }
  function setSpeed(ms) { stepMs = ms; if (playing) schedule(stepMs); }
  onMount(() => { if (start === "replay") startReplay(); });
  onDestroy(() => clearTimeout(timer));

  const f = $derived(mode === "replay" ? frames[idx] : null);
  const frac = $derived(mode === "replay"
    ? (f?.t ? Math.min(1, f.s / f.t) : 1)
    : (d && d.state === "denoise" && d.total ? Math.min(1, d.step / d.total) : processing ? 0 : 1));
  const text = $derived(mode === "replay" ? (f?.x ?? "") : draft);
  const meta = $derived(mode === "replay"
    ? (f ? `replay · block ${f.b + 1} · step ${f.s + 1}/${f.t} · frame ${idx + 1}/${frames.length}` : "replay")
    : theirs ? (queued ? `queued #${queued}` : "waiting")
    : !processing ? "idle"
    : !d || d.state === "prefill" ? "prefilling the prompt"
    : `block ${d.block + 1} of ${d.n_blocks} · step ${d.step + 1}/${d.total}`);
  const placeholder = $derived(mode === "replay"
    ? (replayErr || (frames.length ? "" : "No steps captured yet: the helper keeps the current or last reply's steps."))
    : theirs ? (queued ? `Another request is denoising; you're #${queued} in the queue.` : "Another request is denoising; this one starts after it.")
    : !processing ? (follow ? "Waiting for the engine to start this reply…" : "Idle. Replay plays the last reply back, step by step.")
    : !draft ? (d?.state === "denoise" ? "Block committed; starting the next one…" : "Prefilling the prompt…")
    : "");
</script>

<div class="dgc">
  <div class="head">
    <div class="title"><span class="dot">&#9632;</span> DiffusionGemma <span class="sub">· block diffusion</span></div>
    <div class="meta">{meta}</div>
  </div>
  <div class="bar"><i style="width: {(100 * frac).toFixed(1)}%;"></i></div>
  <div class="canvas">
    {#if placeholder}<div class="ph">{placeholder}</div>{/if}
    {#if text}<pre class="cv" style="opacity: {(0.55 + 0.45 * frac).toFixed(2)};">{text}</pre>{/if}
  </div>
  <div class="controls">
    {#if !snapshot}
      <button class="btn small" class:primary={mode === "live"} onclick={goLive} title="Follow the running request">Live</button>
    {/if}
    <button class="btn small" class:primary={mode === "replay"} onclick={startReplay} title="Play back every denoise step of the current or last reply">{snapshot ? "Replay this reply" : "Replay last reply"}</button>
    {#if mode === "replay" && frames.length}
      <button class="btn small" onclick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
      <button class="btn small" onclick={() => { idx = 0; setPlaying(true); }}>Restart</button>
      <input type="range" min="0" max={frames.length - 1} value={idx} oninput={scrub} aria-label="replay position" />
      <span class="speed">speed
        {#each [[180, "0.5x"], [90, "1x"], [45, "2x"]] as [ms, label]}
          <button class="btn small" class:primary={stepMs === ms} onclick={() => setSpeed(ms)}>{label}</button>
        {/each}
      </span>
    {/if}
  </div>
  <div class="foot">
    {#if mode === "replay"}
      {frames.length.toLocaleString()} steps captured{dropped ? ` (the oldest ${dropped.toLocaleString()} dropped)` : ""}.
    {:else}
      The model's best guess for the whole {d?.canvas ?? 256}-token block, repainted every denoise step until it settles and commits{d && processing ? ` · ${d.steps_done.toLocaleString()} steps so far` : ""}.
    {/if}
    {#if pollErr}<span class="err"> {pollErr}</span>{/if}
  </div>
</div>

<style>
  .dgc { --dg-panel: #121826; --dg-line: #1e2738; --dg-accent: #a78bfa; --dg-accent2: #34d399; margin-bottom: 12px; }
  .head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 6px; }
  .title { font-weight: 600; font-size: 13px; letter-spacing: .2px; }
  .title .dot { color: var(--dg-accent); }
  .title .sub { color: var(--ink-muted); font-weight: 400; }
  .meta { font-variant-numeric: tabular-nums; font-size: 12px; color: var(--ink-muted); white-space: nowrap; }
  .bar { height: 6px; border-radius: 6px; background: var(--dg-line); overflow: hidden; margin: 4px 0 10px; }
  .bar > i { display: block; height: 100%; width: 0; background: linear-gradient(90deg, var(--dg-accent), var(--dg-accent2)); transition: width .08s linear; }
  .canvas { background: var(--dg-panel); border: 1px solid var(--dg-line); border-radius: 10px; padding: 14px 16px; min-height: 150px; max-height: 360px; overflow: auto; }
  .cv { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 13px; line-height: 1.5; font-family: var(--mono); color: var(--ink); transition: opacity .12s ease; }
  .ph { color: var(--ink-muted); font-size: 12px; font-style: italic; }
  .controls { display: flex; align-items: center; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
  .controls input[type=range] { flex: 1; min-width: 120px; accent-color: var(--dg-accent); }
  .speed { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; color: var(--ink-muted); }
  .foot { color: var(--ink-muted); font-size: 11px; margin-top: 8px; line-height: 1.4; }
  .foot .err { color: var(--block, #e87d6e); }
</style>
