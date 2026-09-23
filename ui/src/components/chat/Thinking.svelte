<script>
  // The model's reasoning: plain text, never HTML. Open while it streams
  // and nothing else has arrived yet; it folds away once the answer starts,
  // unless the person opened or closed it themselves.
  let { text = "", streaming = false, ms = null, startedAt = null } = $props();

  let pinned = $state(false);
  let open = $state(false);
  let now = $state(performance.now());

  $effect(() => {
    if (!pinned) open = streaming;
  });
  // A live clock while it thinks.
  $effect(() => {
    if (!streaming) return;
    const id = setInterval(() => (now = performance.now()), 250);
    return () => clearInterval(id);
  });

  function toggle() {
    pinned = true;
    open = !open;
  }
  const secs = $derived(ms != null ? ms / 1000 : streaming && startedAt != null ? (now - startedAt) / 1000 : null);
  const label = $derived(
    (streaming ? "Thinking" : "Thought") +
      (secs != null ? (streaming ? " · " : " for ") + secs.toFixed(1) + " s" : "") +
      " · " + text.length.toLocaleString() + " chars"
  );
</script>

<div class="think" class:open>
  <button type="button" class="head" onclick={toggle} aria-expanded={open}>
    <span class="caret" aria-hidden="true">{open ? "▾" : "▸"}</span>
    <span class="lbl">{label}</span>
    {#if streaming}<span class="dot" aria-hidden="true"></span>{/if}
  </button>
  {#if open}
    <pre class="body">{text}</pre>
  {/if}
</div>

<style>
  .think { border: 1px solid var(--rule); border-radius: var(--radius-sm); background: var(--ground-inset); margin: 2px 0 10px; }
  .head { all: unset; cursor: pointer; display: flex; align-items: center; gap: 8px; width: 100%; box-sizing: border-box; padding: 6px 10px; font-size: 12px; color: var(--ink-muted); }
  .head:hover { color: var(--ink); }
  .head:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; border-radius: var(--radius-sm); }
  .caret { font-size: 10px; width: 10px; }
  .lbl { font-family: var(--mono); font-size: 11.5px; }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--note); animation: pulse 1.4s ease-in-out infinite; }
  .body { margin: 0; padding: 4px 12px 10px 28px; white-space: pre-wrap; word-break: break-word; font-family: var(--sans); font-size: 12.5px; line-height: 1.55; color: var(--ink-faint); max-height: 320px; overflow-y: auto; }
  @media (prefers-reduced-motion: reduce) { .dot { animation: none; } }
</style>
