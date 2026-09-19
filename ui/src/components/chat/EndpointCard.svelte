<script>
  // How another program talks to this server: the OpenAI base URL first
  // (what most clients ask for), then the model id and snippets to paste.
  import { copyText } from "../../lib/clipboard.js";

  // `copiedFirst`: the opener already copied this item (Running's Copy
  // endpoint copies the base URL, then shows the rest here).
  let { endpoint, onclose = null, copiedFirst = "" } = $props();

  let copied = $state("");
  let timer = null;
  $effect(() => {
    if (!copiedFirst) return;
    copied = copiedFirst;
    timer = setTimeout(() => (copied = ""), 1400);
    return () => clearTimeout(timer);
  });
  async function copy(what, text) {
    const ok = await copyText(text);
    copied = ok ? what : "failed";
    clearTimeout(timer);
    timer = setTimeout(() => (copied = ""), 1400);
  }
  const reach = $derived(
    endpoint.loopback ? "this PC only"
      : endpoint.wildcard ? "every network interface: other machines can connect"
      : "bound to " + endpoint.bindHost
  );
</script>

<div class="ep" role="dialog" aria-label="Endpoint">
  <div class="head">
    <span class="sec-title">Endpoint</span>
    <span class="chip {endpoint.loopback ? 'pass' : 'warn'}" title="where the server listens">{reach}</span>
    {#if onclose}<button class="btn small close" onclick={onclose} aria-label="close">Close</button>{/if}
  </div>

  <div class="row main">
    <div class="k">base URL</div>
    <code class="v">{endpoint.baseUrl}</code>
    <button class="btn small primary" onclick={() => copy("url", endpoint.baseUrl)}>{copied === "url" ? "Copied" : "Copy"}</button>
  </div>
  <div class="row">
    <div class="k">model</div>
    <code class="v">{endpoint.modelId}</code>
    <button class="btn small" onclick={() => copy("model", endpoint.modelId)}>{copied === "model" ? "Copied" : "Copy"}</button>
  </div>
  {#if endpoint.hasKey}
    <div class="note">This profile sets <span class="mono">--api-key</span>: clients send it as the bearer token.</div>
  {/if}

  {#each [["curl", "curl (streaming)", endpoint.curl], ["python", "Python (openai)", endpoint.python], ["env", "environment", endpoint.env]] as [id, label, text]}
    <div class="snip">
      <div class="snip-head">
        <span class="k">{label}</span>
        <button class="btn small" onclick={() => copy(id, text)}>{copied === id ? "Copied" : "Copy"}</button>
      </div>
      <pre>{text}</pre>
    </div>
  {/each}
  {#if copied === "failed"}<div class="note err">The clipboard refused; select the text and copy it by hand.</div>{/if}
</div>

<style>
  .ep { display: flex; flex-direction: column; gap: 10px; min-width: 0; }
  .head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .head .close { margin-left: auto; }
  .row { display: grid; grid-template-columns: 64px minmax(0, 1fr) auto; align-items: center; gap: 10px; }
  .k { font-size: 11.5px; font-weight: 600; color: var(--ink-muted); }
  .v { font-family: var(--mono); font-size: 12px; color: var(--ink); background: var(--ground-inset); border: 1px solid var(--rule); border-radius: var(--radius-sm); padding: 5px 8px; overflow-x: auto; white-space: nowrap; }
  .row.main .v { color: var(--accent); }
  .snip { display: flex; flex-direction: column; gap: 4px; }
  .snip-head { display: flex; align-items: center; justify-content: space-between; }
  pre { margin: 0; font-family: var(--mono); font-size: 11.5px; line-height: 1.5; color: var(--ink-muted); background: #0b0f13; border: 1px solid var(--rule); border-radius: var(--radius-sm); padding: 8px 10px; overflow-x: auto; white-space: pre; }
  .note { font-size: 12px; color: var(--ink-muted); }
  .note.err { color: var(--block); }
</style>
