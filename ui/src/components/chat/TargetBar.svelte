<script>
  // Which server this conversation talks to, how busy it is, and how any
  // other program would reach it.
  import { fade } from "svelte/transition";
  import { chat, targetKey, setTarget } from "../../lib/chat.svelte.js";
  import { endpointFor } from "../../lib/endpoint.js";
  import { LAYOUT } from "../../motion.js";
  import EndpointCard from "./EndpointCard.svelte";

  let { conv, target, oninspector = null } = $props();

  let showEndpoint = $state(false);

  const standalone = $derived(chat.targets.filter((t) => !t.model));
  const routed = $derived(chat.targets.filter((t) => t.model));
  const key = $derived(conv?.target ? targetKey(conv.target) : "");

  function pick(e) {
    const t = chat.targets.find((x) => x.key === e.target.value);
    if (t) setTarget(t);
  }
  const optLabel = (t) => (t.model ? t.model_id + (t.label !== t.model_id ? " · " + t.label : "") : t.label + " · :" + t.port)
    + (t.status === "loaded" ? "" : " (" + t.status + ")");
  const statusChip = $derived(
    !target ? { kind: "block", text: "not running" }
      : target.status === "loaded" ? { kind: "pass", text: "loaded" }
      : target.status === "loading" ? { kind: "warn live", text: "loading" }
      : { kind: "plain", text: "loads on first message" }
  );
  const busy = $derived(
    target?.slots_total ? target.slots_busy + " of " + target.slots_total + " slots busy" : ""
  );
  const endpoint = $derived(target ? endpointFor({ host: target.bind_host, port: target.port, alias: target.model_id }, null, target.has_api_key) : null);
</script>

<div class="bar">
  <label class="pick" title="The server this conversation talks to. Switching keeps the history: the next reply comes from the new one.">
    <span class="sr">target</span>
    <select value={key} onchange={pick} disabled={!chat.targets.length}>
      {#if !target}
        <option value={key}>{conv?.target ? (conv.target.model ?? conv.target.run) + " (not running)" : "pick a server"}</option>
      {/if}
      {#if standalone.length}
        <optgroup label="Servers">
          {#each standalone as t (t.key)}<option value={t.key}>{optLabel(t)}</option>{/each}
        </optgroup>
      {/if}
      {#if routed.length}
        <optgroup label="Router">
          {#each routed as t (t.key)}<option value={t.key}>{optLabel(t)}</option>{/each}
        </optgroup>
      {/if}
    </select>
  </label>
  {#if target}
    <span class="chip {target.engine === 'diffusion-gemma' ? 'note' : 'plain'}">{target.engine === "diffusion-gemma" ? "DiffusionGemma" : "llama-server"}</span>
  {/if}
  <span class="chip {statusChip.kind}">{statusChip.text}</span>
  {#if busy}<span class="faint small num" title="from the server's /slots">{busy}</span>{/if}
  <span class="grow"></span>
  {#if endpoint}
    <button class="btn small" class:primary={showEndpoint} onclick={() => (showEndpoint = !showEndpoint)} title="Base URL, model id and snippets for other programs">Endpoint</button>
  {/if}
  {#if oninspector}
    <button class="btn small inspector-toggle" onclick={oninspector} title="System prompt, thinking and sampler for this conversation">Options</button>
  {/if}
  {#if showEndpoint && endpoint}
    <div class="pop card" transition:fade={{ duration: LAYOUT }}>
      <EndpointCard {endpoint} onclose={() => (showEndpoint = false)} />
    </div>
  {/if}
</div>

<style>
  .bar { position: relative; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; padding: 10px 12px; border-bottom: 1px solid var(--rule); }
  .pick { min-width: 180px; max-width: 360px; flex: 1 1 200px; }
  .pick select { font-family: var(--sans); font-weight: 600; }
  .sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
  .grow { flex: 1; }
  .pop { position: absolute; top: calc(100% + 6px); right: 12px; z-index: 20; width: min(560px, calc(100% - 24px)); box-shadow: var(--shadow); border-color: var(--rule-strong); margin: 0; }
  .inspector-toggle { display: none; }
  @media (max-width: 1279px) { .inspector-toggle { display: inline-flex; } }
</style>
