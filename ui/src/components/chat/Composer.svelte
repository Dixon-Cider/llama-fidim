<script>
  // Enter sends, Shift+Enter starts a new line. While a reply streams, Send
  // becomes Stop and the status line says where the reply is.
  import { chat, send, stop } from "../../lib/chat.svelte.js";

  let { conv, target = null, stream = null } = $props();

  let ta = $state(null);
  const text = $derived(conv ? chat.drafts[conv.id] ?? "" : "");
  const canSend = $derived(!!conv && !!target && !stream && text.trim().length > 0);

  function setText(v) {
    if (conv) chat.drafts[conv.id] = v;
  }
  function grow() {
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight + 2, 240) + "px";
  }
  $effect(() => {
    text;
    queueMicrotask(grow);
  });
  function onKey(e) {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      if (canSend) submit();
    }
  }
  async function submit() {
    if (!canSend) return;
    await send(text);
    ta?.focus();
  }

  const n = (v) => Math.round(Number(v) || 0).toLocaleString();
  const status = $derived.by(() => {
    const s = stream;
    if (!s) {
      if (!target) return conv?.target ? "that server is not running; pick another one above" : "";
      if (target.status !== "loaded") return "the router loads this model when you send (that can take a minute)";
      return "";
    }
    if (s.stopping) return "stopping…";
    switch (s.phase) {
      case "connecting": return "connecting…";
      case "loading": return "loading the model…";
      case "waiting": return "waiting for the first token…";
      case "queued": return "queued #" + (s.queue ?? "?") + " behind another request";
      case "prefill": {
        const p = s.prefill;
        if (p?.total) return "prefill " + Math.round((100 * p.processed) / p.total) + "% (" + n(p.processed) + " / " + n(p.total) + ")";
        return "reading the prompt…";
      }
      case "denoise": {
        const p = s.progress;
        return p?.block ? "denoising block " + p.block + " of " + p.n_blocks + " · step " + p.step + "/" + p.total : "denoising…";
      }
      case "thinking": return "thinking…";
      case "writing": {
        const r = s.live?.predicted_per_second;
        return r ? "writing · " + Number(r).toFixed(1) + " tok/s" : "writing…";
      }
      default: return "";
    }
  });

  // Context gauge: the last reply's prompt plus output against the slot's
  // context (a DiffusionGemma run's MAXTOK).
  const nCtx = $derived((target && chat.props[target.key]?.n_ctx) || target?.n_ctx || null);
  const used = $derived.by(() => {
    const last = conv?.messages.findLast((m) => m.role === "assistant" && (m.usage || m.timings));
    if (!last) return null;
    if (last.usage) return (last.usage.prompt_tokens ?? 0) + (last.usage.completion_tokens ?? 0);
    const t = last.timings;
    return (t.cache_n ?? 0) + (t.prompt_n ?? 0) + (t.predicted_n ?? 0);
  });
  const frac = $derived(used != null && nCtx ? Math.min(1, used / nCtx) : null);
</script>

<div class="composer">
  <div class="statusline">
    <span class="status" class:live={!!stream}>{status}</span>
    {#if frac != null}
      <span class="gauge" title="the last reply used this much of the slot's context">
        <span class="num">{n(used) + " / " + n(nCtx) + " ctx"}</span>
        <span class="meter"><span class="fill" class:warn={frac > 0.8} class:block={frac > 0.95} style="width: {(100 * frac).toFixed(1)}%;"></span></span>
      </span>
    {/if}
  </div>
  <div class="row">
    <textarea
      bind:this={ta}
      rows="1"
      value={text}
      oninput={(e) => setText(e.target.value)}
      onkeydown={onKey}
      placeholder={target ? "Message " + (target.model_id ?? target.label) + " (Enter to send, Shift+Enter for a new line)" : "No server to talk to"}
      disabled={!conv}
      spellcheck="true"
    ></textarea>
    {#if stream}
      <button class="btn danger stop" onclick={() => stop(conv.id)} disabled={stream.stopping}>Stop</button>
    {:else}
      <button class="btn primary" onclick={submit} disabled={!canSend}>Send</button>
    {/if}
  </div>
</div>

<style>
  .composer { border-top: 1px solid var(--rule); padding: 8px 12px 12px; display: flex; flex-direction: column; gap: 6px; background: var(--ground-raised); }
  .statusline { display: flex; align-items: center; gap: 12px; min-height: 18px; }
  .status { font-family: var(--mono); font-size: 11.5px; color: var(--ink-faint); }
  .status.live { color: var(--accent); }
  .gauge { margin-left: auto; display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--ink-faint); }
  .gauge .meter { width: 90px; height: 6px; }
  .gauge .fill { background: var(--note); }
  .gauge .fill.warn { background: var(--warn); }
  .gauge .fill.block { background: var(--block); }
  .row { display: flex; gap: 10px; align-items: flex-end; }
  textarea { flex: 1; font-family: var(--sans); font-size: 13.5px; line-height: 1.5; resize: none; min-height: 40px; max-height: 240px; padding: 9px 12px; }
  .row .btn { height: 40px; min-width: 76px; justify-content: center; }
  .stop { border-color: var(--block); color: var(--block); }
</style>
