<script>
  // The conversation's system prompt and sampler. Every field starts at
  // "inherit": the request then leaves it out and the server's own default
  // applies (the profile's flags included, shown from /props). Only the
  // fields switched on here go into the request.
  import { chat, setParam, setSystem, setThinking, setPreserveReasoning, savePreset, resetParams, loadCreator, presetKey } from "../../lib/chat.svelte.js";
  import Range from "../Range.svelte";

  let { conv, target = null } = $props();

  const dg = $derived(target?.engine === "diffusion-gemma" || conv?.target?.engine === "diffusion-gemma");
  const props = $derived(target ? chat.props[target.key] ?? null : null);
  const server = $derived(props?.params ?? {});
  const caps = $derived(props?.caps ?? {});
  const creator = $derived(target?.model_path ? chat.creator[target.model_path] ?? null : null);

  const FIELDS = [
    { k: "temperature", min: 0, max: 2, step: 0.05, title: "Randomness: 0 is greedy, 1 the model's own distribution.", creator: "temperature" },
    { k: "top_p", min: 0, max: 1, step: 0.01, title: "Keep the smallest set of tokens whose probability adds up to this.", creator: "top_p" },
    { k: "top_k", min: 0, max: 200, step: 1, title: "Keep only the k most likely tokens (0 = off).", creator: "top_k" },
    { k: "min_p", min: 0, max: 1, step: 0.01, title: "Drop tokens less likely than this fraction of the top token.", creator: "min_p" },
    { k: "repeat_penalty", min: 0.8, max: 2, step: 0.01, title: "Penalise recently used tokens (1 = off).", creator: "repetition_penalty" },
    { k: "presence_penalty", min: -2, max: 2, step: 0.05, title: "Penalise any token already present (0 = off)." },
    { k: "dry_multiplier", min: 0, max: 2, step: 0.05, title: "DRY repetition penalty strength (0 = off)." },
  ];
  const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(2).replace(/0$/, "")) : String(v));
  const serverLabel = (k) => (server[k] != null ? "server: " + fmt(server[k]) : "inherit");
  const creatorHint = (f) => (f.creator && creator && creator[f.creator] != null ? "creator " + fmt(creator[f.creator]) : "");

  function applyCreator() {
    for (const f of FIELDS) {
      const v = f.creator && creator?.[f.creator];
      if (v != null) setParam(f.k, v);
    }
  }
  const hasCreator = $derived(!!creator && !creator.error && !creator.busy && FIELDS.some((f) => f.creator && creator[f.creator] != null));

  const thinkingDefault = $derived(
    target?.enable_thinking === false ? "off" : target?.enable_thinking === true ? "on" : "template default"
  );
  const stopText = $derived(Array.isArray(conv?.params?.stop) ? conv.params.stop.join("\n") : conv?.params?.stop ?? "");
  const stopCount = $derived(stopText.split("\n").filter((s) => s.length).length);
  const dgDefaultMax = $derived(target?.diffusion?.default_max_tokens ?? 2048);
  const blocks = (n) => Math.ceil(Number(n) / 256);

  let savedNote = $state("");
  async function save() {
    try {
      await savePreset();
      savedNote = "Saved: new chats on " + (target?.label ?? "this profile") + " start with these.";
    } catch (e) {
      savedNote = "Not saved: " + String(e);
    }
    setTimeout(() => (savedNote = ""), 3500);
  }
</script>

{#if conv}
  <div class="panel">
    <section>
      <label class="field">
        <span class="k">system prompt</span>
        <textarea rows="5" value={conv.system_prompt} oninput={(e) => setSystem(e.target.value)} placeholder="(none)" spellcheck="true"></textarea>
      </label>
      {#if caps.supports_system_role === false}
        <div class="warnline">This model's chat template has no system role: llama-server folds the system prompt into the first message.</div>
      {/if}
    </section>

    {#if dg}
      <section>
        <div class="sec-title">DiffusionGemma</div>
        <div class="faint small">It ignores temperature and the other samplers, and always thinks. Replies come in 256-token blocks.</div>
        <Range bind:value={conv.params.max_tokens} label="max tokens" nullable span={1}
          min={256} max={Math.max(512, props?.n_ctx ?? 16384)} step={256}
          placeholder={dgDefaultMax} offLabel={"profile: " + dgDefaultMax.toLocaleString()}
          hint={blocks(conv.params.max_tokens ?? dgDefaultMax) + " blocks"}
          format={(v) => Number(v).toLocaleString()}
          title="Reply budget, spent in whole 256-token blocks and clamped to the context budget."
          onchange={() => setParam("max_tokens", conv.params.max_tokens)} />
      </section>
    {:else}
      <section>
        <div class="sec-title">Thinking</div>
        <div class="seg" role="radiogroup" aria-label="thinking">
          {#each [[null, "inherit"], [true, "on"], [false, "off"]] as [v, label]}
            <button type="button" role="radio" aria-checked={conv.thinking === v} class:on={conv.thinking === v} onclick={() => setThinking(v)}>{label}</button>
          {/each}
        </div>
        <div class="faint small">{"inherit = the profile's setting: " + thinkingDefault}</div>
        {#if caps.supports_preserve_reasoning}
          <label class="check"><input type="checkbox" checked={!!conv.preserve_reasoning} onchange={(e) => setPreserveReasoning(e.target.checked)} /> send earlier reasoning back (costs context)</label>
        {/if}
      </section>

      <section class="fields">
        <div class="sec-title">
          Sampler
          <span class="faint small">{props?.error ? "server defaults unavailable" : props?.loaded === false ? "server defaults appear once the model loads" : "off = the server's own"}</span>
        </div>
        {#each FIELDS as f (f.k)}
          <Range bind:value={conv.params[f.k]} label={f.k} min={f.min} max={f.max} step={f.step} nullable span={1}
            placeholder={server[f.k] ?? null} offLabel={serverLabel(f.k)} hint={creatorHint(f)} title={f.title}
            onchange={() => setParam(f.k, conv.params[f.k])} />
        {/each}
        <div class="two">
          <label class="field" title="Reply budget in tokens. Empty = the server's (-1 = until the context is full).">
            <span class="k">max_tokens</span>
            <input type="number" min="1" step="1" value={conv.params.max_tokens ?? ""} placeholder={server.max_tokens != null ? fmt(server.max_tokens) : "inherit"}
              oninput={(e) => setParam("max_tokens", e.target.value === "" ? null : Math.max(1, Math.round(Number(e.target.value))))} />
          </label>
          <label class="field" title="Fixed seed, to reproduce a reply. Empty = random.">
            <span class="k">seed</span>
            <input type="number" step="1" value={conv.params.seed ?? ""} placeholder="random"
              oninput={(e) => setParam("seed", e.target.value === "" ? null : Math.round(Number(e.target.value)))} />
          </label>
        </div>
      </section>
    {/if}

    <section>
      <label class="field" title={dg ? "Up to 4 stop strings, one per line." : "Stop strings, one per line."}>
        <span class="k">{"stop strings" + (dg ? " (" + stopCount + " of 4)" : "")}</span>
        <textarea rows="2" value={stopText} placeholder="one per line" spellcheck="false"
          oninput={(e) => setParam("stop", e.target.value)}></textarea>
      </label>
      {#if dg}
        <label class="field" title="Fixed seed, to reproduce a reply. Empty = the profile's (random unless the profile pins one).">
          <span class="k">seed</span>
          <input type="number" step="1" min="0" value={conv.params.seed ?? ""} placeholder={target?.diffusion?.seed != null ? "profile: " + target.diffusion.seed : "random"}
            oninput={(e) => setParam("seed", e.target.value === "" ? null : Math.max(0, Math.round(Number(e.target.value))))} />
        </label>
        {#if stopCount > 4}<div class="warnline">DiffusionGemma takes 4 stop strings; the first 4 are sent.</div>{/if}
      {/if}
    </section>

    {#if !dg && target?.model_path}
      <section>
        <div class="row">
          <button class="btn small" onclick={() => loadCreator(target)} disabled={creator?.busy} title="generation_config.json from the model's Hugging Face repo">
            {creator?.busy ? "Fetching…" : "Creator defaults"}
          </button>
          {#if hasCreator}<button class="btn small primary" onclick={applyCreator}>Apply</button>{/if}
        </div>
        {#if creator?.error}<div class="faint small err">{creator.error}</div>
        {:else if hasCreator}<div class="faint small">{"from " + creator.repo}</div>{/if}
      </section>
    {/if}

    <section class="foot">
      <div class="row">
        <button class="btn small" onclick={save} disabled={!target} title={"New chats on " + (target ? presetKey(target) : "this profile") + " start with this system prompt and these settings"}>Save as default for this profile</button>
        <button class="btn small" onclick={resetParams} title="Back to inherit for every field">Reset</button>
      </div>
      {#if savedNote}<div class="faint small">{savedNote}</div>{/if}
      {#if props?.n_ctx}
        <div class="faint small num">{"context " + props.n_ctx.toLocaleString() + " tokens per slot" + (props.total_slots ? " · " + props.total_slots + (props.total_slots === 1 ? " slot" : " slots") : "") + (props.canvas ? " · canvas " + props.canvas : "")}</div>
      {/if}
    </section>
  </div>
{/if}

<style>
  .panel { display: flex; flex-direction: column; gap: 14px; padding: 12px 14px 16px; }
  section { display: flex; flex-direction: column; gap: 10px; }
  .fields :global(.field.range) { gap: 2px; }
  .sec-title { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
  .sec-title .small { font-family: var(--sans); font-weight: 400; }
  textarea { font-family: var(--sans); font-size: 12.5px; }
  .seg { display: flex; border: 1px solid var(--rule-strong); border-radius: 6px; overflow: hidden; }
  .seg button { all: unset; cursor: pointer; flex: 1; text-align: center; padding: 5px 6px; font-size: 12px; color: var(--ink-muted); }
  .seg button + button { border-left: 1px solid var(--rule-strong); }
  .seg button.on { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
  .seg button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  .check { display: flex; gap: 8px; align-items: center; font-size: 12px; color: var(--ink-muted); }
  .two { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; }
  .warnline { font-size: 12px; color: var(--warn); }
  .err { color: var(--block); }
  .foot { border-top: 1px solid var(--rule); padding-top: 12px; }
</style>
