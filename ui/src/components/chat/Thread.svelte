<script>
  // The conversation. It follows a streaming reply only while the reader
  // is already at the bottom; scrolling up to reread stops the follow.
  import { fade } from "svelte/transition";
  import { FAST } from "../../motion.js";
  import Message from "./Message.svelte";

  let { conv, target = null, stream = null } = $props();

  let box = $state(null);
  let atBottom = $state(true);
  const NEAR = 48;

  function onScroll() {
    if (box) atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < NEAR;
  }
  function toBottom() {
    if (box) box.scrollTop = box.scrollHeight;
    atBottom = true;
  }
  // Keep pinned to the end as content grows, if the reader was there.
  function follow(node) {
    const o = new MutationObserver(() => {
      if (atBottom) node.scrollTop = node.scrollHeight;
    });
    o.observe(node, { childList: true, characterData: true, subtree: true });
    return { destroy() { o.disconnect(); } };
  }
  // A different conversation starts at its end.
  $effect(() => {
    conv?.id;
    queueMicrotask(toBottom);
  });
</script>

<div class="thread" bind:this={box} onscroll={onScroll} use:follow>
  {#if conv}
    {#each conv.messages as m (m.id)}
      <Message {m} {target} stream={stream?.msgId === m.id ? stream : null} busy={!!stream} />
    {:else}
      <div class="empty">
        {target ? "Ask " + (target.model_id ?? target.label) + " anything. The reply streams in as it is written." : "Pick a running server above to start."}
      </div>
    {/each}
  {/if}
</div>
{#if !atBottom && stream}
  <button class="btn small jump" onclick={toBottom} transition:fade={{ duration: FAST }}>Jump to the latest</button>
{/if}

<style>
  .thread { flex: 1; min-height: 0; overflow-y: auto; overscroll-behavior: contain; }
  .empty { padding: 60px 24px; }
  .jump { position: absolute; left: 50%; transform: translateX(-50%); bottom: 150px; z-index: 5; box-shadow: var(--shadow); }
</style>
