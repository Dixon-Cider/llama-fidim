<script>
  // Chat with any server Llama FIDIM started: a standalone llama-server, a
  // model behind the router, or a DiffusionGemma run. Replies stream from
  // Rust (never a fetch from this page) over a per-request channel.
  // Everything lives in lib/chat.svelte.js, so it survives switching tabs.
  import { onDestroy } from "svelte";
  import { fade } from "svelte/transition";
  import { chat, init, refreshTargets, rail, newChat, openChat, deleteChat, targetKey } from "../lib/chat.svelte.js";
  import { LAYOUT, FAST } from "../motion.js";
  import Skeleton from "../components/Skeleton.svelte";
  import TargetBar from "../components/chat/TargetBar.svelte";
  import Thread from "../components/chat/Thread.svelte";
  import Composer from "../components/chat/Composer.svelte";
  import SamplerPanel from "../components/chat/SamplerPanel.svelte";

  let { go = () => {} } = $props();

  let inspector = $state(false);   // the slide-over below 1280 px wide

  init();
  // Targets carry live slot counts and router load states.
  const timer = setInterval(refreshTargets, 5000);
  onDestroy(() => clearInterval(timer));

  const conv = $derived(chat.currentId ? chat.convs[chat.currentId] ?? null : null);
  const target = $derived(conv?.target ? chat.targets.find((t) => t.key === targetKey(conv.target)) ?? null : null);
  const stream = $derived(conv ? chat.streams[conv.id] ?? null : null);
  const items = $derived(rail());

  function startNew() {
    newChat(target ?? chat.targets[0] ?? null);
  }

  // Two clicks to delete from the rail: the first arms it for three seconds.
  let armed = $state(null);
  let armTimer = null;
  function del(e, id) {
    e.stopPropagation();
    if (armed !== id) {
      armed = id;
      clearTimeout(armTimer);
      armTimer = setTimeout(() => (armed = null), 3000);
      return;
    }
    armed = null;
    deleteChat(id);
  }

  function ago(unix) {
    const s = Math.max(0, Math.floor(Date.now() / 1000) - (unix ?? 0));
    if (s < 60) return "just now";
    if (s < 3600) return Math.floor(s / 60) + " min ago";
    if (s < 86400) return Math.floor(s / 3600) + " h ago";
    if (s < 7 * 86400) return Math.floor(s / 86400) + " d ago";
    return new Date(unix * 1000).toLocaleDateString();
  }
  const targetName = (t) => (t ? t.model ?? t.run : "");
</script>

<div class="chat-view">
  <h1>
    Chat
    <span class="sub">Talk to any server Llama FIDIM started. Replies stream from the app itself; nothing leaves this PC.</span>
  </h1>

  {#if !chat.targetsLoaded}
    <Skeleton height="420px" />
  {:else if !chat.targets.length && !items.length}
    <div class="card" in:fade={{ duration: LAYOUT }}>
      <div class="empty">
        Nothing is running to chat with.
        <div class="gap"></div>
        <button class="btn" onclick={() => go("profiles")}>Launch a profile</button>
        <button class="btn" onclick={() => go("router")}>Start the router</button>
      </div>
      {#if chat.targetsError}<div class="err mono">{chat.targetsError}</div>{/if}
    </div>
  {:else}
    <div class="grid" class:inspector-open={inspector}>
      <aside class="rail">
        <button class="btn primary new" onclick={startNew} disabled={!chat.targets.length}>New chat</button>
        <div class="list" role="list">
          {#each items as c (c.id)}
            <div class="item" class:active={c.id === chat.currentId} role="listitem">
              <button class="open" onclick={() => openChat(c.id)} title={c.title || "New chat"}>
                <span class="title">{c.title || "New chat"}</span>
                <span class="meta">
                  {#if chat.streams[c.id]}<span class="dot" title="a reply is streaming"></span>{/if}
                  <span class="mono">{targetName(c.target)}</span>
                  <span>{" · " + ago(c.updated_unix)}</span>
                </span>
              </button>
              <button class="del" class:armed={armed === c.id} onclick={(e) => del(e, c.id)} title={armed === c.id ? "click again to delete" : "delete this conversation"} aria-label="delete">{armed === c.id ? "delete?" : "×"}</button>
            </div>
          {/each}
        </div>
        <div class="saved faint small" title={chat.saving ? "~/.fidim/chats, one JSON file per conversation" : "Turn saving on in Settings"}>
          {chat.saving ? "Saved on this PC" : "Not saved: gone when the app closes"}
          {#if chat.saveError}<div class="err">{chat.saveError}</div>{/if}
        </div>
      </aside>

      <section class="main">
        <TargetBar {conv} {target} oninspector={() => (inspector = !inspector)} />
        <Thread {conv} {target} {stream} />
        <Composer {conv} {target} {stream} />
      </section>

      {#if inspector}
        <button class="scrim" aria-label="close settings" onclick={() => (inspector = false)} transition:fade={{ duration: FAST }}></button>
      {/if}
      <aside class="inspector">
        <div class="insp-head">
          <span class="sec-title">Conversation settings</span>
          <button class="btn small close" onclick={() => (inspector = false)}>Close</button>
        </div>
        <SamplerPanel {conv} {target} />
      </aside>
    </div>
  {/if}
</div>

<style>
  /* The view fills the window; the thread scrolls, the composer stays. */
  .chat-view { display: flex; flex-direction: column; height: calc(100vh - 82px); min-height: 520px; }
  .grid {
    flex: 1; min-height: 0; margin-top: 14px;
    display: grid; grid-template-columns: 220px minmax(0, 1fr) 320px; gap: 14px;
  }
  .rail, .main, .inspector { background: var(--ground-raised); border: 1px solid var(--rule); border-radius: var(--radius); min-height: 0; }
  .rail { display: flex; flex-direction: column; padding: 10px; gap: 10px; }
  .rail .new { justify-content: center; }
  .list { flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 2px; margin: 0 -4px; padding: 0 4px; }
  .item { position: relative; display: flex; align-items: stretch; border-radius: 6px; }
  .item:hover { background: var(--ground-inset); }
  .item.active { background: var(--accent-soft); }
  .item .open { all: unset; cursor: pointer; flex: 1; min-width: 0; padding: 7px 8px; display: flex; flex-direction: column; gap: 2px; }
  .item .open:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; border-radius: 6px; }
  .item .title { font-size: 12.5px; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .item.active .title { color: var(--accent); font-weight: 600; }
  .item .meta { display: flex; align-items: center; gap: 4px; font-size: 11px; color: var(--ink-faint); white-space: nowrap; overflow: hidden; }
  .item .meta .mono { font-size: 11px; overflow: hidden; text-overflow: ellipsis; }
  .item .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); flex: none; animation: pulse 1.4s ease-in-out infinite; }
  .item .del { all: unset; cursor: pointer; opacity: 0; padding: 0 8px; font-size: 13px; color: var(--ink-faint); border-radius: 6px; }
  .item:hover .del, .item .del:focus-visible, .item .del.armed { opacity: 1; }
  .item .del:hover { color: var(--block); }
  .item .del.armed { color: var(--block); font-size: 11px; font-weight: 600; }
  .saved { border-top: 1px solid var(--rule); padding-top: 8px; }
  .err { color: var(--block); word-break: break-word; }

  .main { position: relative; display: flex; flex-direction: column; overflow: hidden; }
  .inspector { overflow-y: auto; }
  .insp-head { display: none; align-items: center; justify-content: space-between; padding: 12px 14px 0; }
  .scrim { display: none; }

  .empty .gap { height: 12px; }
  .empty .btn { margin: 0 4px; }

  /* Narrower than 1280 px: the inspector becomes a slide-over. */
  @media (max-width: 1279px) {
    .grid { grid-template-columns: 200px minmax(0, 1fr); }
    .inspector {
      position: fixed; top: 0; right: 0; bottom: 0; width: 340px; z-index: 40; border-radius: 0;
      border-width: 0 0 0 1px; box-shadow: var(--shadow);
      transform: translateX(100%); transition: transform var(--t-layout) cubic-bezier(.2, .8, .2, 1); visibility: hidden;
    }
    .inspector-open .inspector { transform: none; visibility: visible; }
    .insp-head { display: flex; }
    .scrim { all: unset; display: block; position: fixed; inset: 0; z-index: 39; background: rgba(0, 0, 0, 0.35); }
  }
  @media (prefers-reduced-motion: reduce) { .item .dot { animation: none; } }
</style>
