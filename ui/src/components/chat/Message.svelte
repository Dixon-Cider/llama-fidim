<script>
  // One turn. A reply renders as sanitized markdown (lib/markdown.js),
  // re-rendered at most once per animation frame while it streams; its
  // reasoning stays plain text. Links never navigate the app: a click
  // offers Open (in the browser, http and https only) or Copy.
  import { onDestroy } from "svelte";
  import { fade } from "svelte/transition";
  import { renderMarkdown, openable } from "../../lib/markdown.js";
  import { copyText } from "../../lib/clipboard.js";
  import { openUrl } from "../../api.js";
  import { chat, regenerate, editAndResend, deleteMessage } from "../../lib/chat.svelte.js";
  import { FAST } from "../../motion.js";
  import Thinking from "./Thinking.svelte";
  import DiffusionCanvas from "../DiffusionCanvas.svelte";

  let { m, target = null, stream = null, busy = false } = $props();

  const assistant = $derived(m.role === "assistant");
  const live = $derived(!!stream && !stream.finished);
  const dgLive = $derived(live && stream.engine === "diffusion-gemma");
  const dg = $derived(!!m.dg || !!m.timings?.diffusion || dgLive);

  // ---- markdown, throttled to the frame rate while streaming ----
  let html = $state("");
  let raf = 0;
  $effect(() => {
    const text = m.content;
    if (!assistant) return;
    if (!live) {
      cancelAnimationFrame(raf);
      raf = 0;
      html = renderMarkdown(text);
      return;
    }
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = 0;
      html = renderMarkdown(m.content);
    });
  });
  onDestroy(() => cancelAnimationFrame(raf));

  // ---- links and code blocks inside the rendered reply ----
  let mdEl = $state(null);
  let menu = $state(null);   // { href, x, y }
  function onMdClick(e) {
    const btn = e.target.closest?.("[data-copy-code]");
    if (btn) {
      e.preventDefault();
      const code = btn.closest(".code")?.querySelector("code")?.textContent ?? "";
      copyText(code).then((ok) => {
        btn.textContent = ok ? "Copied" : "Copy failed";
        setTimeout(() => (btn.textContent = "Copy"), 1200);
      });
      return;
    }
    const a = e.target.closest?.("a");
    if (a && mdEl?.contains(a)) {
      e.preventDefault();
      e.stopPropagation();
      const r = a.getBoundingClientRect();
      const box = mdEl.getBoundingClientRect();
      menu = { href: a.getAttribute("href") ?? "", x: Math.max(0, r.left - box.left), y: r.bottom - box.top + 4 };
    }
  }
  let linkErr = $state("");
  async function openLink() {
    const href = menu?.href;
    menu = null;
    try {
      await openUrl(href);
    } catch (e) {
      linkErr = String(e?.message ?? e);
      setTimeout(() => (linkErr = ""), 4000);
    }
  }
  async function copyLink() {
    const href = menu?.href;
    menu = null;
    await copyText(href);
  }

  // ---- actions ----
  let copied = $state(false);
  async function copyMessage() {
    copied = await copyText(m.content);
    setTimeout(() => (copied = false), 1200);
  }
  let editing = $state(false);
  let editText = $state("");
  function startEdit() {
    editText = m.content;
    editing = true;
  }
  function resend() {
    editing = false;
    editAndResend(m.id, editText);
  }
  let confirmDelete = $state(false);
  let confirmTimer = null;
  function del() {
    if (!confirmDelete) {
      confirmDelete = true;
      clearTimeout(confirmTimer);
      confirmTimer = setTimeout(() => (confirmDelete = false), 3000);
      return;
    }
    deleteMessage(m.id);
  }
  let showReplay = $state(false);

  // ---- stats ----
  const t = $derived(m.timings ?? (live ? stream.live : null));
  const f1 = (v) => (v == null || !Number.isFinite(Number(v)) ? "—" : Number(v).toFixed(1));
  const f0 = (v) => (v == null || !Number.isFinite(Number(v)) ? "—" : Math.round(Number(v)).toLocaleString());
  const stats = $derived.by(() => {
    if (!assistant || !t) return [];
    if (t.diffusion) {
      return [
        [f1(t.diffusion_output_tok_s ?? t.predicted_per_second) + " tok/s", "output: text delivered per second, prefill included"],
        [f0(t.diffusion_parallel_tok_s) + " canvas tok/s", "tokens re-predicted per second across every denoise step (Unsloth Studio's Speed)"],
        [f0(t.diffusion_blocks) + " blocks · " + f0(t.diffusion_steps) + " steps", "256-token blocks and denoise steps"],
        ...(t.diffusion_seed != null ? [["seed " + t.diffusion_seed, "send this seed (Settings) to reproduce the reply"]] : []),
      ];
    }
    const out = [[f1(t.predicted_per_second) + " tok/s", "decode speed"]];
    if (t.prompt_per_second != null) out.push([f0(t.prompt_per_second) + " prompt tok/s", "prefill speed"]);
    if (t.cache_n) out.push([f0(t.cache_n) + " cached", "prompt tokens reused from the slot's cache"]);
    if (t.draft_n) out.push([Math.round((100 * (t.draft_n_accepted ?? 0)) / t.draft_n) + "% draft accepted", "speculative decoding: " + (t.draft_n_accepted ?? 0) + " of " + t.draft_n + " drafted tokens accepted"]);
    if (t.predicted_n != null) out.push([f0(t.predicted_n) + " tokens", "tokens generated"]);
    return out;
  });
  const when = $derived(m.created_unix ? new Date(m.created_unix * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "");
  const who = $derived(assistant ? (m.model ?? target?.model_id ?? "assistant") : "You");
  const waiting = $derived(live && !m.content && !m.reasoning && !dgLive);
</script>

<svelte:window onclick={() => (menu = null)} onkeydown={(e) => e.key === "Escape" && (menu = null)} />

<article class="msg" class:user={!assistant} class:live>
  <header>
    <span class="who" class:mono={assistant}>{who}</span>
    {#if when}<span class="faint small">{when}</span>{/if}
    {#if m.stopped}<span class="chip warn">stopped</span>{/if}
    {#if m.error}<span class="chip block">{m.error_status ? "error " + m.error_status : "error"}</span>{/if}
    {#if m.finish === "length"}<span class="chip warn" title="the reply hit its token budget">cut at max tokens</span>{/if}
    {#if m.finish === "tool_calls"}<span class="chip note">tool call</span>{/if}
  </header>

  {#if assistant && m.reasoning}
    <Thinking text={m.reasoning} streaming={live && !m.content} ms={m.reasoning_ms ?? null} startedAt={stream?.reasoningAt ?? null} />
  {/if}

  {#if editing}
    <div class="edit">
      <textarea rows="4" bind:value={editText} onkeydown={(e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); resend(); } else if (e.key === "Escape") editing = false; }}></textarea>
      <div class="edit-actions">
        <button class="btn small primary" onclick={resend} disabled={!editText.trim() || busy}>Send</button>
        <button class="btn small" onclick={() => (editing = false)}>Cancel</button>
        <span class="faint small">Replies after this message are replaced.</span>
      </div>
    </div>
  {:else if !assistant}
    <div class="plain">{m.content}</div>
  {:else}
    <div class="body" class:split={dgLive}>
      <div class="md-wrap">
        {#if waiting}
          <div class="typing" aria-label="waiting for the reply"><i></i><i></i><i></i></div>
        {/if}
        <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
        <div class="md" bind:this={mdEl} onclick={onMdClick}>{@html html}</div>
        {#if menu}
          <div class="linkmenu" style="left: {menu.x}px; top: {menu.y}px;" transition:fade={{ duration: FAST }} role="menu">
            <div class="href mono">{menu.href || "(no address)"}</div>
            <button class="btn small" role="menuitem" disabled={!openable(menu.href)} onclick={(e) => { e.stopPropagation(); openLink(); }} title={openable(menu.href) ? "open in your browser" : "only http and https links open"}>Open in browser</button>
            <button class="btn small" role="menuitem" disabled={!menu.href} onclick={(e) => { e.stopPropagation(); copyLink(); }}>Copy link</button>
          </div>
        {/if}
        {#if linkErr}<div class="err small">{linkErr}</div>{/if}
      </div>
      {#if dgLive}
        <div class="canvas-col">
          <DiffusionCanvas host={stream.host} port={stream.port} follow idTask={stream.idTask} queued={stream.queue} />
        </div>
      {/if}
    </div>
    {#if m.tool_calls?.length}
      <div class="tools">
        {#each m.tool_calls as c}
          <div class="tool"><span class="chip note">call</span> <span class="mono">{(c.function?.name || "?") + "(" + (c.function?.arguments ?? "") + ")"}</span></div>
        {/each}
      </div>
    {/if}
    {#if m.error}<div class="err">{m.error}</div>{/if}
    {#if m.stopped && dg}
      <div class="note">Stopped. DiffusionGemma finishes the current reply in the background (the runner has no mid-request cancel); your next message will queue behind it.</div>
    {/if}
    {#if showReplay && chat.frames[m.id]}
      <div class="replay">
        <DiffusionCanvas host={target?.host ?? "127.0.0.1"} port={target?.port ?? 0} snapshot={chat.frames[m.id]} idTask={m.dg?.id_task ?? null} start="replay" />
      </div>
    {/if}
  {/if}

  {#if !editing}
    <footer>
      {#if stats.length}
        <div class="stats">
          {#each stats as [v, why]}<span class="num" title={why}>{v}</span>{/each}
        </div>
      {/if}
      <div class="actions">
        {#if assistant && chat.frames[m.id]}
          <button class="btn small" class:primary={showReplay} onclick={() => (showReplay = !showReplay)} title="Play this reply's denoise steps back">{showReplay ? "Hide denoise" : "Replay denoise"}</button>
        {/if}
        {#if m.content}<button class="btn small" onclick={copyMessage}>{copied ? "Copied" : "Copy"}</button>{/if}
        {#if assistant}
          <button class="btn small" onclick={() => regenerate(m.id)} disabled={busy} title="Ask again from here">Regenerate</button>
        {:else}
          <button class="btn small" onclick={startEdit} disabled={busy} title="Change this message and ask again">Edit</button>
        {/if}
        <button class="btn small danger" onclick={del} disabled={live}>{confirmDelete ? "Delete?" : "Delete"}</button>
      </div>
    </footer>
  {/if}
</article>

<style>
  .msg { padding: 14px 16px 10px; border-bottom: 1px solid var(--rule); container-type: inline-size; }
  .msg.user { background: rgba(255, 255, 255, 0.015); }
  header { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; flex-wrap: wrap; }
  .who { font-weight: 650; font-size: 12.5px; color: var(--ink); }
  .who.mono { font-family: var(--mono); color: var(--accent); }
  .plain { white-space: pre-wrap; word-break: break-word; font-size: 13.5px; line-height: 1.6; }

  .body.split { display: grid; grid-template-columns: minmax(0, 1fr) minmax(360px, 42%); gap: 16px; align-items: start; }
  @container (max-width: 760px) { .body.split { grid-template-columns: 1fr; } }
  .md-wrap { position: relative; min-width: 0; }
  .canvas-col { min-width: 0; }
  .replay { margin-top: 10px; }

  .md { font-size: 13.5px; line-height: 1.62; word-break: break-word; }
  .md :global(p) { margin: 0 0 10px; }
  .md :global(p:last-child) { margin-bottom: 0; }
  .md :global(h1), .md :global(h2), .md :global(h3), .md :global(h4) { font-family: var(--display); font-weight: 650; line-height: 1.3; margin: 14px 0 8px; }
  .md :global(h1) { font-size: 18px; } .md :global(h2) { font-size: 16px; } .md :global(h3), .md :global(h4) { font-size: 14px; }
  .md :global(ul), .md :global(ol) { margin: 0 0 10px; padding-left: 22px; }
  .md :global(li) { margin: 2px 0; }
  .md :global(a) { color: var(--accent); text-decoration: underline; text-underline-offset: 2px; cursor: pointer; }
  .md :global(code) { font-family: var(--mono); font-size: 12px; background: var(--ground-inset); border: 1px solid var(--rule); border-radius: 4px; padding: 1px 5px; }
  .md :global(blockquote) { margin: 0 0 10px; padding: 2px 12px; border-left: 3px solid var(--rule-strong); color: var(--ink-muted); }
  .md :global(hr) { border: none; border-top: 1px solid var(--rule); margin: 14px 0; }
  .md :global(table) { border-collapse: collapse; margin: 0 0 12px; font-size: 12.5px; display: block; overflow-x: auto; max-width: 100%; }
  .md :global(th), .md :global(td) { border: 1px solid var(--rule-strong); padding: 5px 10px; text-align: left; vertical-align: top; }
  .md :global(th) { background: var(--ground-inset); font-weight: 650; }
  .md :global(.code) { margin: 0 0 12px; border: 1px solid var(--rule); border-radius: var(--radius-sm); overflow: hidden; background: #0b0f13; }
  .md :global(.code-head) { display: flex; align-items: center; justify-content: space-between; padding: 3px 6px 3px 10px; border-bottom: 1px solid var(--rule); background: var(--ground-inset); }
  .md :global(.code-head .lang) { font-family: var(--mono); font-size: 11px; color: var(--ink-faint); }
  .md :global(.code-head .copy) { all: unset; cursor: pointer; font-size: 11px; font-weight: 600; color: var(--ink-muted); padding: 2px 8px; border-radius: 4px; }
  .md :global(.code-head .copy:hover) { color: var(--accent); background: var(--ground-lift); }
  .md :global(.code-head .copy:focus-visible) { outline: 2px solid var(--accent); }
  .md :global(.code pre) { margin: 0; padding: 10px 12px; overflow-x: auto; }
  .md :global(.code pre code) { background: none; border: none; padding: 0; font-size: 12px; line-height: 1.55; color: var(--ink); white-space: pre; }

  .typing { display: inline-flex; gap: 4px; padding: 6px 0; }
  .typing i { width: 6px; height: 6px; border-radius: 50%; background: var(--ink-faint); animation: blink 1.2s infinite ease-in-out; }
  .typing i:nth-child(2) { animation-delay: .15s; } .typing i:nth-child(3) { animation-delay: .3s; }
  @keyframes blink { 0%, 80%, 100% { opacity: .25; } 40% { opacity: 1; } }

  .linkmenu { position: absolute; z-index: 30; display: flex; flex-direction: column; gap: 6px; align-items: stretch; min-width: 220px; max-width: 420px;
    background: var(--ground-lift); border: 1px solid var(--rule-strong); border-radius: var(--radius); padding: 8px; box-shadow: var(--shadow); }
  .linkmenu .href { font-size: 11px; color: var(--ink-muted); word-break: break-all; }

  .tools { display: flex; flex-direction: column; gap: 4px; margin-top: 8px; }
  .tool .mono { font-size: 11.5px; word-break: break-all; }
  .err { color: var(--block); font-size: 12.5px; margin-top: 8px; font-family: var(--mono); word-break: break-word; }
  .err.small { font-size: 11.5px; margin-top: 4px; }
  .note { color: var(--ink-muted); font-size: 12px; margin-top: 8px; border-left: 2px solid var(--warn); padding-left: 10px; }

  .edit { display: flex; flex-direction: column; gap: 8px; }
  .edit textarea { font-family: var(--sans); font-size: 13.5px; }
  .edit-actions { display: flex; align-items: center; gap: 8px; }

  footer { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 8px; min-height: 26px; }
  .stats { display: flex; gap: 14px; flex-wrap: wrap; font-size: 11.5px; color: var(--ink-faint); }
  .stats span { cursor: default; }
  .actions { display: flex; gap: 6px; margin-left: auto; opacity: 0.35; transition: opacity var(--t-fast); }
  .msg:hover .actions, .msg:focus-within .actions { opacity: 1; }
  @media (prefers-reduced-motion: reduce) { .typing i { animation: none; opacity: .6; } }
</style>
