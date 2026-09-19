<script>
  // Get a model: find it on Hugging Face, pick the file that fits the cards,
  // get the build that loads it, download, and end at a saved profile.
  // Every decision is fidim-core's (the same as `fidim models`); state
  // lives in lib/models.svelte.js so a download keeps going, and shows,
  // while the person looks at other tabs. Nothing here loads a model: the
  // Done step has a Launch button for that.
  import { api, openUrl, log } from "../api.js";
  import { fly, fade, slide, scale } from "svelte/transition";
  import { arrive, leave, stagger, toastFly, LAYOUT } from "../motion.js";
  import { handoff } from "../lib/handoff.js";
  import {
    wz, init, submit, open, looksLikeRepo, toBuild, setBuild, setDest, addRoot, start, cancel, resume, runCheck, startOver, makePlan,
  } from "../lib/models.svelte.js";
  import {
    gib, mbps, eta, params, shortSha, day, levelChip, fitChip, fitWord, actionTitle, sourceOf, supportText, supportChip,
  } from "../lib/wizard.js";

  let { go = () => {} } = $props();

  init();

  const STEPS = [
    { n: 1, label: "Find" },
    { n: 2, label: "Choose file" },
    { n: 3, label: "Build" },
    { n: 4, label: "Get" },
    { n: 5, label: "Done" },
  ];
  const running = $derived(!!wz.job && !wz.job.finished);
  const reached = $derived(
    wz.job?.finished?.ok ? 5 : wz.job ? 4 : wz.plan ? 4 : wz.view?.kind?.kind === "gguf" ? 2 : wz.view ? 2 : 1
  );
  function canGo(n) {
    if (running) return n === 4;
    if (n === 5) return !!wz.job?.finished?.ok;
    if (n >= 2 && !wz.view) return n === 4 && !!wz.job;
    if (n >= 3 && !wz.plan) return false;
    return n <= Math.max(reached, wz.step);
  }
  async function goStep(n) {
    if (!canGo(n)) return;
    if ((n === 3 || n === 4) && !wz.plan) await makePlan();
    wz.step = n;
  }

  const v = $derived(wz.view);
  const plan = $derived(wz.plan);
  const isGguf = $derived(v?.kind?.kind === "gguf");
  const fitOf = (label) => v?.fits?.find((f) => f.label === label) ?? null;
  const buildStep = $derived(plan?.steps?.find((s) => s.action.kind === "build")?.action.plan ?? null);
  const source = $derived(buildStep ? sourceOf(buildStep, plan?.build_sources) : null);
  const usable = $derived(v?.builds?.find((b) => b.path === v?.usable_build) ?? null);
  const cards = $derived((v?.devices ?? []).filter((d) => !d.integrated));
  const blockedByConsent = $derived(!!plan?.needs_consent && !wz.consent);
  const progressOf = (i) => wz.job?.progress?.steps?.[i] ?? null;
  const jobPlan = $derived(wz.job?.plan ?? plan);
  const result = $derived(wz.job?.finished?.ok ? wz.job.finished.result : null);

  const kindLabel = (k) => ({ gguf: "GGUF", safetensors: "safetensors", adapter: "LoRA adapter", other_format: k?.format, empty: "no model files" }[k?.kind] ?? "?");
  const gatedLabel = (g) => (g === "manual" ? "gated, approved by hand" : g === "auto" ? "gated" : "");
  const base = (p) => String(p ?? "").split(/[\\/]/).pop();

  // Build options: the plan's recommendation, its alternatives, an
  // installed build, or no build at all.
  const planSteps = $derived(plan?.build_plan ? [plan.build_plan.step, ...(plan.build_plan.alternatives ?? [])] : []);
  // Which option the plan now follows: auto means an installed build when
  // one loads the model, else the plan's recommendation.
  const selected = $derived(
    wz.build.kind === "plan" ? `plan:${wz.build.value}`
      : wz.build.kind === "auto" ? (buildStep || !plan?.build?.installed ? "plan:0" : "auto")
      : wz.build.kind
  );
  let installedPick = $state(wz.build.kind === "installed" ? wz.build.value : "");

  let newRoot = $state("");
  let toast = $state(null);
  let toastTimer = null;
  function toastMsg(text, isError = false) {
    toast = { text, isError };
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (toast = null), 6000);
  }

  // Done step.
  let launching = $state(false);
  let launched = $state(null);   // { id, pid, port } once Launch succeeded
  function openInProfiles() {
    const id = result?.profile?.id;
    if (id) handoff.profileId = id;
    go("profiles");
  }
  async function launch() {
    const id = result?.profile?.id;
    if (!id) return;
    launching = true;
    try {
      const r = await api("launch_profile", { id, overrideBlocks: false });
      if (r.blocked) {
        toastMsg("Blocked by pre-flight: see the checks below. Open it in Profiles to change it or override.", true);
        wz.check = { ...(wz.check ?? {}), results: r.results };
      } else {
        launched = { id, pid: r.state.pid, port: r.state.port };
        toastMsg(`Launched pid ${r.state.pid} on port ${r.state.port}; the Running tab follows it.`);
        log(`models: launched ${id}`);
      }
    } catch (e) {
      toastMsg(String(e), true);
    }
    launching = false;
  }

  function outcomeKind(o) {
    if (o === "pass") return "pass";
    if (o && typeof o === "object") {
      if (typeof o.outcome === "string") return o.outcome;
      for (const k of ["block", "warn", "note", "pass"]) if (o[k] !== undefined) return k;
    }
    return "block";
  }
  function outcomeMsg(o) {
    if (o && typeof o === "object") return typeof o.outcome === "string" ? o.message ?? "" : o.warn ?? o.block ?? o.note ?? "";
    return "";
  }

  // The build log follows its newest line unless scrolled up to read.
  function stick(node) {
    const follow = () => {
      if (node.scrollHeight - node.scrollTop - node.clientHeight < 60) node.scrollTop = node.scrollHeight;
    };
    const obs = new MutationObserver(follow);
    obs.observe(node, { childList: true, subtree: true });
    node.scrollTop = node.scrollHeight;
    return { destroy: () => obs.disconnect() };
  }

  const statusChip = (s) => ({ pending: "plain", running: "accent", done: "pass", failed: "block", stopped: "warn", skipped: "plain" }[s] ?? "plain");
  const stepIcon = (a) => (a.kind === "build" ? "build" : a.kind === "download" ? a.role : "profile");
</script>

<h1>
  Get a model
  <span class="sub">Find a model on Hugging Face, see which file fits your cards and which llama.cpp build loads it, then download it and make a profile. Nothing here loads a model.</span>
</h1>

<nav class="stepper" aria-label="steps">
  {#each STEPS as s (s.n)}
    <button class="st" class:on={wz.step === s.n} class:past={s.n < wz.step} disabled={!canGo(s.n) && wz.step !== s.n} onclick={() => goStep(s.n)}>
      <span class="num">{s.n}</span><span>{s.label}</span>
    </button>
    {#if s.n < 5}<span class="sep"></span>{/if}
  {/each}
</nav>

{#key wz.step}
<div in:fly={arrive()}>

<!-- 1 Find -------------------------------------------------------------- -->
{#if wz.step === 1}
  <section class="card">
    <div class="searchbar">
      <input class="q" bind:value={wz.query} spellcheck="false" autocomplete="off"
        placeholder="Search Hugging Face, or paste owner/name or a huggingface.co link"
        onkeydown={(e) => e.key === "Enter" && submit()} />
      <label class="opt" title="Only repos with .gguf files, the format llama.cpp loads. Off also lists safetensors and other formats (the wizard then points to their GGUF versions).">
        <input type="checkbox" checked={!wz.allFormats} onchange={(e) => (wz.allFormats = !e.target.checked)} /> GGUF only
      </label>
      <button class="btn primary" onclick={submit} disabled={wz.searching || !!wz.opening || !wz.query.trim()}>
        {#if wz.searching || wz.opening}<span class="spinner"></span>{/if}
        {looksLikeRepo(wz.query) ? "Open repo" : "Search"}
      </button>
    </div>
    <div class="faint small" style="margin-top: 8px;">
      Search matches words of the repo name. A repo id (<span class="mono">IFM/K2-Horizon-7B-GGUF</span>) or a link to a repo or a file opens it directly.
    </div>
    {#if wz.findError}<div class="notice" style="margin-top: 10px;" transition:slide={leave}><span class="chip block">error</span><span class="mono">{wz.findError}</span></div>{/if}
  </section>

  {#if wz.hits}
    <section class="card flush">
      {#if wz.hits.length}
        <table class="grid hits">
          <thead><tr><th>Repo</th><th>Architecture</th><th class="r">Params</th><th class="r">Downloads</th><th class="r">Likes</th><th>Updated</th><th>Build</th></tr></thead>
          <tbody>
            {#each wz.hits as h, i (h.id)}
              <tr class="hit" in:fade={{ duration: LAYOUT, delay: stagger(i, 15) }} onclick={() => open(h.id)} title="Open {h.id}">
                <td>
                  <span class="mono repo">{h.id}</span>
                  {#if h.gated !== "no"}<span class="chip warn" title="Downloads need the repo's terms accepted on Hugging Face and a token in Settings.">{gatedLabel(h.gated)}</span>{/if}
                  {#if wz.opening === h.id}<span class="spinner"></span>{/if}
                </td>
                <td class="mono">{h.arch ?? "—"}</td>
                <td class="r num">{params(h.total_params)}</td>
                <td class="r num">{Number(h.downloads).toLocaleString()}</td>
                <td class="r num">{h.likes}</td>
                <td class="mono faint">{day(h.last_modified)}</td>
                <td>
                  {#if h.arch_known === true}<span class="chip pass" title="An installed build names this architecture.">installed</span>
                  {:else if h.arch_known === false}<span class="chip warn" title="No installed build names this architecture: the wizard will look for one to install or build.">needs a build</span>
                  {:else}<span class="faint">—</span>{/if}
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else}
        <div class="empty">Nothing on Hugging Face matches “{wz.searched}”{wz.allFormats ? "" : " with GGUF files"}.</div>
      {/if}
    </section>
  {/if}

<!-- 2 Choose file ------------------------------------------------------- -->
{:else if wz.step === 2 && v}
  <section class="card">
    <div class="repohead">
      <div class="who">
        <button class="link mono big" onclick={() => openUrl(`https://huggingface.co/${v.repo}`)} title="Open on huggingface.co">{v.repo}</button>
        <span class="mono faint" title="Every file is downloaded from this commit: {v.sha}">@{shortSha(v.sha)}</span>
      </div>
      <div class="facts">
        <span class="chip {isGguf ? 'accent' : 'block'}">{kindLabel(v.kind)}</span>
        {#if v.needs}<span class="mono">{v.needs.arch}</span>{/if}
        {#if v.info?.gguf?.total}<span>{params(v.info.gguf.total)} parameters</span>{/if}
        {#if v.header?.context_length}<span>trained context {Number(v.header.context_length).toLocaleString()}</span>{/if}
        {#if v.info?.card_license}<span>license {v.info.card_license}</span>{/if}
        {#if v.info?.gated !== "no"}<span class="chip warn">{gatedLabel(v.info.gated)}</span>{/if}
        {#if v.info?.base_models?.length}<span class="faint">{v.info.base_models[0][0] || "from"} {v.info.base_models[0][1]}</span>{/if}
      </div>
    </div>
    {#each v.notes as n}
      <div class="notice note-row" transition:slide={leave}><span class="chip {levelChip(n.level)}">{n.level}</span><span>{n.message}</span></div>
    {/each}
  </section>

  {#if !isGguf}
    <section class="card">
      <div class="sec">GGUF versions <span class="faint">{v.derivatives_of ? `quantizations of ${v.derivatives_of} on the Hub` : ""}</span></div>
      {#if v.derivatives.length}
        <table class="grid">
          <tbody>
            {#each v.derivatives as d (d.id)}
              <tr class="hit" onclick={() => open(d.id)}>
                <td class="mono">{d.id}</td><td class="mono faint">{d.arch ?? ""}</td>
                <td class="r num">{Number(d.downloads).toLocaleString()} downloads</td>
                <td class="r"><button class="btn small" onclick={(e) => { e.stopPropagation(); open(d.id); }}>{wz.opening === d.id ? "…" : "Open"}</button></td>
              </tr>
            {/each}
          </tbody>
        </table>
      {:else}
        <div class="empty">None found. Search for the model's name instead.</div>
      {/if}
      <div class="toolbar" style="margin: 12px 0 0;"><button class="btn" onclick={() => (wz.step = 1)}>Back to search</button></div>
    </section>
  {:else}
    <section class="card flush">
      <div class="sec pad">
        Files
        <span class="faint">
          {v.catalog.choices.length} choices · estimated at context {Number(v.fits?.[0]?.ctx ?? 32768).toLocaleString()}, f16 KV, on
          {cards.length ? `${cards.length} × ${cards[0].name.replace(/^AMD /, "")} (${(cards[0].total_mib / 1024).toFixed(1)} GiB)` : "no GPU found"}
        </span>
      </div>
      <table class="grid quants">
        <thead>
          <tr>
            <th></th><th>File</th><th class="r">Size</th>
            <th class="r" title="The estimate on one card: weights + KV cache + compute buffers + overhead.">Needs</th>
            <th title="On one card, against its whole VRAM. Tight = over 90%: no headroom for a display or another model.">One card</th>
            <th title="Layer-split over the two largest cards.">Two-card split</th>
            <th class="r" title="The longest context that fits one card with 10% to spare.">Max context</th>
          </tr>
        </thead>
        <tbody>
          {#each v.catalog.choices as c (c.label)}
            {@const f = fitOf(c.label)}
            <tr class="choice" class:sel={wz.choice === c.label} class:rec={v.recommended === c.label} onclick={() => (wz.choice = c.label)}>
              <td><input type="radio" name="quant" checked={wz.choice === c.label} onchange={() => (wz.choice = c.label)} /></td>
              <td>
                <span class="mono lbl">{c.label}</span>
                {#if v.recommended === c.label}<span class="chip accent" title="The largest file that fits one card at this context.">recommended</span>{/if}
                {#if c.files.length > 1}<span class="chip plain" title={c.files.map((x) => x.path).join("\n")}>{c.files.length} parts</span>{/if}
                <div class="path">{c.first_file}</div>
              </td>
              <td class="r num">{gib(c.total_size)}</td>
              <td class="r num">{f?.one_card?.need_bytes ? gib(f.one_card.need_bytes) : "—"}</td>
              <td>{#if f}<span class="chip {fitChip(f.one_card.fit)}" title={f.one_card.detail}>{fitWord(f.one_card.fit)}</span>{#if f.one_card.fit !== "not_applicable" && !f.one_card.fits_free_now && f.one_card.fit !== "no_fit"}<span class="faint small" title="The cards' VRAM free right now is less: another server holds some."> busy now</span>{/if}{:else}<span class="faint">—</span>{/if}</td>
              <td>{#if f}<span class="chip {fitChip(f.two_card_split.fit)}" title={f.two_card_split.detail}>{fitWord(f.two_card_split.fit)}</span>{:else}<span class="faint">—</span>{/if}</td>
              <td class="r num">{f?.max_ctx_one_card ? Number(f.max_ctx_one_card).toLocaleString() : "—"}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>

    <section class="card">
      <div class="sec">Extras <span class="faint">downloaded beside the model and paired in the profile</span></div>
      <div class="formgrid">
        <label class="field" style="grid-column: span 2;" title="A multimodal projector lets the server read images. Its VRAM counts on the main card.">
          <span class="k">vision projector</span>
          <select bind:value={wz.mmproj} disabled={!v.catalog.mmproj.length}>
            <option value="">{v.catalog.mmproj.length ? "none" : "none in this repo"}</option>
            {#each v.catalog.mmproj as m}<option value={m.path}>{m.path} · {gib(m.size)}</option>{/each}
          </select>
        </label>
        <label class="field" style="grid-column: span 2;" title="A draft model or MTP head for speculative decoding; the profile turns it on.">
          <span class="k">draft / MTP head</span>
          <select bind:value={wz.draft} disabled={!v.catalog.drafts.length}>
            <option value="">{v.catalog.drafts.length ? "none" : "none in this repo"}</option>
            {#each v.catalog.drafts as d}<option value={d.path}>{d.path} · {gib(d.size)}</option>{/each}
          </select>
        </label>
        <label class="field" style="grid-column: span 2;" title="The new profile's context in tokens. Empty = the largest that fits one card, at most 32,768. Change it later in Profiles.">
          <span class="k">context</span>
          <input type="number" min="512" step="1024" bind:value={wz.ctx} placeholder={`auto (${Number(Math.min(32768, fitOf(wz.choice)?.max_ctx_one_card ?? 32768)).toLocaleString()})`} />
        </label>
      </div>
    </section>

    <section class="card">
      <div class="notice">
        {#if usable}
          <span class="chip pass">build</span><span><b>{usable.name}</b> can load it.</span>
        {:else if v.needs}
          <span class="chip warn">build</span><span>No installed build knows <span class="mono">{v.needs.arch}</span>{v.needs.tokenizer_pre ? ` (pre-tokenizer ${v.needs.tokenizer_pre})` : ""}. The next step gets one.</span>
        {:else}
          <span class="chip plain">build</span><span>The architecture could not be read, so no build was checked.</span>
        {/if}
      </div>
      {#if wz.planError}<div class="notice" style="margin-top: 10px;"><span class="chip block">error</span><span class="mono">{wz.planError}</span></div>{/if}
      <div class="toolbar" style="margin: 14px 0 0;">
        <button class="btn" onclick={() => (wz.step = 1)}>Back</button>
        <div class="grow"></div>
        <button class="btn primary" onclick={toBuild} disabled={!wz.choice || wz.planning}>{#if wz.planning}<span class="spinner"></span> Planning…{:else}Continue{/if}</button>
      </div>
    </section>
  {/if}

<!-- 3 Build ------------------------------------------------------------- -->
{:else if wz.step === 3 && plan}
  <section class="card">
    <div class="sec">
      Build
      <span class="faint">a llama.cpp build that knows <span class="mono">{plan.needs?.arch ?? "?"}</span>{plan.needs?.tokenizer_pre ? `, pre-tokenizer ${plan.needs.tokenizer_pre}` : ""}{plan.needs?.max_type_id != null ? `, tensor types up to ${plan.needs.max_type_id}` : ""}</span>
      {#if wz.planning}<span class="chip plain live" style="margin-left: auto;">planning</span>{/if}
    </div>
    {#if plan.build?.installed && !buildStep}
      <div class="verdict pass">
        <span class="chip pass">ready</span>
        <span><b>{plan.build.name}</b> is installed and can load this model. Nothing to install or build.</span>
      </div>
    {:else if buildStep}
      <div class="verdict {buildStep.needs_consent ? 'warn' : 'accent'}">
        <div class="vt">
          <b>{actionTitle(buildStep)}</b>
          {#if buildStep.verified}<span class="chip pass" title="Read from the source at that commit.">verified</span>{:else}<span class="chip warn">not verified</span>{/if}
          {#if buildStep.needs_consent}<span class="chip warn">needs your consent</span>{/if}
        </div>
        <div class="expl">{buildStep.explanation}</div>
        {#each buildStep.warnings as w}<div class="notice small"><span class="chip warn">note</span><span>{w}</span></div>{/each}
      </div>
      {#if source}
        <div class="source">
          <dl class="kv">
            <dt>repository</dt><dd><button class="link mono" onclick={() => openUrl(source.url)}>{source.owner}/{source.repo}</button>
              {#if source.linked_as}<span class="faint"> · the card links {source.linked_as}, which GitHub now sends here</span>{/if}
              {#if !source.is_upstream && !source.is_fork_of_upstream}<span class="chip block">not a fork of ggml-org/llama.cpp</span>{/if}</dd>
            <dt>ref</dt><dd class="mono">{source.git_ref}</dd>
            <dt>commit</dt><dd class="mono">{source.sha}</dd>
            {#if source.ahead_by != null}<dt>distance</dt><dd>{source.ahead_by} commits ahead of upstream master, {source.behind_by} behind</dd>{/if}
            {#if source.pr}<dt>pull request</dt><dd>#{source.pr.number} “{source.pr.title}” · {source.pr.state}{source.pr.draft ? ", draft" : ""}{source.pr.mergeable_state ? `, ${source.pr.mergeable_state}` : ""}</dd>{/if}
          </dl>
          {#if source.subjects?.length}
            <div class="k small" style="margin: 10px 0 4px;">its commits, newest first</div>
            <ol class="subjects">
              {#each [...source.subjects].reverse() as subj}<li class="mono">{subj}</li>{/each}
            </ol>
          {/if}
        </div>
      {/if}
    {:else}
      <div class="verdict warn">
        <span class="chip warn">no build</span>
        <span>{plan.build_plan?.step?.action?.reason ?? "No build is planned."} The files can still be downloaded.</span>
      </div>
    {/if}

    <div class="k small" style="margin: 16px 0 6px;">how to get the build</div>
    <div class="options">
      {#each planSteps as ps, i}
        <label class="opt-row" class:on={selected === `plan:${i}`}>
          <input type="radio" name="build" checked={selected === `plan:${i}`} onchange={() => setBuild({ kind: "plan", value: i })} disabled={wz.planning} />
          <span><b>{actionTitle(ps)}</b>{i === 0 ? " (recommended)" : ""}{ps.needs_consent ? " · needs consent" : ""}</span>
        </label>
      {/each}
      {#if usable}
        <label class="opt-row" class:on={selected === "auto"}>
          <input type="radio" name="build" checked={selected === "auto"} onchange={() => setBuild({ kind: "auto" })} disabled={wz.planning} />
          <span><b>The best installed build that loads it</b> ({usable.name})</span>
        </label>
      {/if}
      {#if v?.builds?.length}
        <label class="opt-row" class:on={selected === "installed"}>
          <input type="radio" name="build" checked={selected === "installed"} disabled={wz.planning}
            onchange={() => { installedPick ||= v.builds[0]?.path ?? ""; setBuild({ kind: "installed", value: installedPick }); }} />
          <span><b>An installed build</b></span>
          <select style="max-width: 460px;" bind:value={installedPick} disabled={wz.planning}
            onchange={() => setBuild({ kind: "installed", value: installedPick })}>
            <option value="" disabled>pick one…</option>
            {#each v.builds as b}<option value={b.path}>{b.name} · {b.channel} · {supportText(b.support)}</option>{/each}
          </select>
        </label>
      {/if}
      <label class="opt-row" class:on={selected === "skip"}>
        <input type="radio" name="build" checked={selected === "skip"} onchange={() => setBuild({ kind: "skip" })} disabled={wz.planning} />
        <span><b>No build</b>: download the files only; the profile goes on the best installed build</span>
      </label>
    </div>

    {#if plan.build_plan?.rejected?.length}
      <details class="ruled">
        <summary class="faint small">ruled out ({plan.build_plan.rejected.length})</summary>
        <ul>{#each plan.build_plan.rejected as r}<li class="small">{r}</li>{/each}</ul>
      </details>
    {/if}
  </section>

  {#if plan.toolchain?.length}
    <section class="card">
      <div class="sec">Toolchain <span class="faint">what a source build needs on this PC, checked with a test compile</span></div>
      <table class="grid">
        <tbody>
          {#each plan.toolchain as t}
            <tr>
              <td style="width: 70px;"><span class="chip {t.outcome === 'pass' ? 'pass' : t.outcome === 'block' ? 'block' : t.outcome === 'warn' ? 'warn' : 'note'}">{t.outcome}</span></td>
              <td>{t.title}</td>
              <td class="mono faint">{t.message}{#if t.fix}<div class="fix">fix: {t.fix}</div>{/if}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </section>
  {/if}

  <section class="card">
    {#if plan.needs_consent}
      <label class="consent" class:ok={wz.consent}>
        <input type="checkbox" bind:checked={wz.consent} />
        <span><b>I want to build it.</b> {plan.consent}</span>
      </label>
    {/if}
    {#if wz.planError}<div class="notice" style="margin-top: 10px;"><span class="chip block">error</span><span class="mono">{wz.planError}</span></div>{/if}
    <div class="toolbar" style="margin: 14px 0 0;">
      <button class="btn" onclick={() => (wz.step = 2)}>Back</button>
      <div class="grow"></div>
      {#if blockedByConsent}<span class="faint small">Tick the box to build code nobody reviewed, or pick another way above.</span>{/if}
      <button class="btn primary" onclick={() => (wz.step = 4)} disabled={blockedByConsent || wz.planning}>Continue</button>
    </div>
  </section>

<!-- 4 Get --------------------------------------------------------------- -->
{:else if wz.step === 4 && jobPlan}
  {#if !wz.job}
    <section class="card">
      <div class="sec">Save to <span class="faint">model folders; the scan finds what lands in them</span></div>
      <div class="roots">
        {#each wz.roots as r (r.path)}
          <label class="opt-row" class:on={wz.destRoot === r.path}>
            <input type="radio" name="root" checked={wz.destRoot === r.path} onchange={() => setDest(r.path)} disabled={wz.planning} />
            <span class="mono">{r.path}</span>
            <span class="faint small">{r.free_bytes != null ? `${gib(r.free_bytes)} free` : "free space unknown"}{r.exists ? "" : " · created on first download"}</span>
          </label>
        {:else}
          <div class="faint small">No model folder yet: add one.</div>
        {/each}
      </div>
      <div class="addroot">
        <input class="mono" bind:value={newRoot} placeholder="D:\models" spellcheck="false"
          onkeydown={async (e) => { if (e.key === "Enter" && (await addRoot(newRoot))) newRoot = ""; }} />
        <button class="btn" disabled={!newRoot.trim() || wz.rootBusy} onclick={async () => { if (await addRoot(newRoot)) newRoot = ""; }}
          title="Adds the folder to Settings > model folders (and creates it), then saves there.">{wz.rootBusy ? "Adding…" : "Add a folder"}</button>
        {#if wz.rootError}<span class="chip block">{wz.rootError}</span>{/if}
      </div>
    </section>
  {/if}

  <section class="card">
    <div class="sec">
      {wz.job ? (wz.job.finished ? (wz.job.finished.ok ? "Done" : "Stopped") : wz.job.cancelling ? "Stopping…" : "Getting it") : "Plan"}
      <span class="faint">{jobPlan.repo} · {jobPlan.choice?.label} · {gib(jobPlan.download_bytes)} to fetch{jobPlan.total_bytes !== jobPlan.download_bytes ? ` of ${gib(jobPlan.total_bytes)}` : ""} into {jobPlan.dest_dir}</span>
      {#if wz.planning}<span class="chip plain live" style="margin-left: auto;">planning</span>{/if}
    </div>
    {#if !wz.job}
      {#each jobPlan.notes as n}
        <div class="notice note-row"><span class="chip {levelChip(n.level)}">{n.level}</span><span>{n.message}</span></div>
      {/each}
    {/if}
    <ol class="steps">
      {#each jobPlan.steps as s, i}
        {@const pg = progressOf(i)}
        {@const stopped = pg?.status === "failed" && pg?.error === "cancelled"}
        {@const status = stopped ? "stopped" : pg?.status ?? "pending"}
        <li class="stp {status}">
          <div class="head">
            <span class="chip {statusChip(status)}" class:live={status === "running"}>{status === "pending" ? stepIcon(s.action) : status}</span>
            <span class="title">{s.title}</span>
            {#if s.needs_consent}<span class="chip warn" title={jobPlan.consent}>consented</span>{/if}
          </div>
          {#if s.action.kind === "download"}
            {@const total = pg?.total ?? s.action.file.size}
            {@const done = pg?.done ?? s.action.have}
            {@const hashing = pg?.stage === "hashing"}
            {#if status === "running" || (status === "pending" && s.action.have > 0) || status === "failed" || status === "stopped"}
              <div class="meter"><div class="fill {hashing ? 'warn' : ''}" style="width: {Math.min(100, (100 * done) / Math.max(1, total))}%;"></div></div>
              <div class="prog faint small num">
                {#if hashing}checking SHA-256 of what is on disk: {gib(done, 2)} of {gib(total, 2)}
                {:else}{gib(done, 2)} of {gib(total, 2)}{#if status === "running" && pg?.bps} · {mbps(pg.bps)} · {eta((total - done) / pg.bps)} left{/if}{/if}
              </div>
            {:else if status === "done"}
              <div class="faint small path">{s.action.dest}</div>
            {:else}
              <div class="faint small">{s.detail}</div>
            {/if}
          {:else if s.action.kind === "build"}
            {#if pg?.total}
              <div class="meter"><div class="fill" style="width: {Math.min(100, (100 * (pg.done ?? 0)) / pg.total)}%;"></div></div>
              <div class="prog faint small num">{pg.stage ?? ""} · [{pg.done}/{pg.total}]</div>
            {:else if status === "running"}
              <div class="prog faint small">{pg?.stage ?? "starting"}…</div>
            {:else if status === "pending"}
              <div class="faint small">{s.detail}</div>
            {/if}
            {#if wz.job && (wz.job.progress?.log?.length ?? 0) > 0 && status !== "pending"}
              <div class="logbox buildlog" use:stick>{#each wz.job.progress.log.slice(-80) as line}<div class="mono">{line}</div>{/each}</div>
            {/if}
          {:else}
            <div class="faint small">{s.detail}</div>
          {/if}
          {#if pg?.error && !stopped}<div class="notice small" style="margin-top: 6px;"><span class="chip block">failed</span><span class="mono">{pg.error}</span></div>{/if}
        </li>
      {/each}
    </ol>

    {#if wz.job?.finished && !wz.job.finished.ok}
      <div class="notice" style="margin-top: 12px;">
        <span class="chip {wz.job.finished.error === 'cancelled' ? 'warn' : 'block'}">{wz.job.finished.error === "cancelled" ? "stopped" : "failed"}</span>
        <span>{wz.job.finished.error === "cancelled" ? "Stopped. Finished downloads are kept, and a partial one continues from where it stopped." : wz.job.finished.error}</span>
      </div>
    {/if}
    {#if wz.jobError}<div class="notice" style="margin-top: 10px;"><span class="chip block">error</span><span class="mono">{wz.jobError}</span></div>{/if}

    <div class="toolbar" style="margin: 14px 0 0;">
      {#if !wz.job}
        <button class="btn" onclick={() => (wz.step = 3)} disabled={!wz.view}>Back</button>
        <div class="grow"></div>
        {#if jobPlan.blocked}<span class="faint small">Fix the errors above first.</span>
        {:else if blockedByConsent}<span class="faint small">The build needs your consent on the Build step.</span>{/if}
        <button class="btn primary" onclick={start} disabled={wz.starting || wz.planning || jobPlan.blocked || blockedByConsent || !jobPlan.steps.length}>
          {#if wz.starting}<span class="spinner"></span>{/if} Start
        </button>
      {:else if running}
        <span class="faint small">Runs in the app: switch tabs freely. Stopping keeps what is downloaded.</span>
        <div class="grow"></div>
        <button class="btn danger" onclick={cancel} disabled={wz.job.cancelling}>{wz.job.cancelling ? "Stopping…" : "Stop"}</button>
      {:else}
        <button class="btn" onclick={startOver}>Start over</button>
        <div class="grow"></div>
        {#if !wz.job.finished.ok}
          <button class="btn primary" onclick={resume} disabled={wz.starting || wz.planning}>Resume</button>
        {/if}
      {/if}
    </div>
  </section>

<!-- 5 Done -------------------------------------------------------------- -->
{:else if wz.step === 5 && result}
  {@const p = result.profile}
  <section class="card">
    <div class="sec">
      {p ? "Profile ready" : "Downloaded"}
      <span class="faint">{result.repo} @{shortSha(result.sha)}</span>
    </div>
    {#if p}
      <div class="done-head">
        <span class="mono big">{p.id}</span>
        <span class="chip pass" in:scale={{ duration: LAYOUT, start: 0.7 }}>saved</span>
        {#if launched?.id === p.id}<span class="chip accent">running on port {launched.port}</span> <button class="link small" onclick={() => go("running")}>Running</button>
        {:else}<span class="faint small">nothing was launched</span>{/if}
      </div>
      <dl class="kv" style="margin-top: 10px;">
        <dt>model</dt><dd class="path">{p.model.path}</dd>
        {#if p.model.mmproj}<dt>projector</dt><dd class="path">{p.model.mmproj}</dd>{/if}
        {#if p.model.draft}<dt>draft</dt><dd class="path">{p.model.draft.path} ({p.speculative?.mode ?? "draft"})</dd>{/if}
        <dt>build</dt><dd class="path">{p.build.path}{p.build.version ? ` (${p.build.version})` : ""}</dd>
        <dt>GPU</dt><dd class="mono">{p.devices.map((d) => d.key.split(":").pop()).join(" + ") || "none"}{p.split_mode ? ` · ${p.split_mode} split` : ""}</dd>
        <dt>server</dt><dd class="mono">port {p.server.port} · alias {p.server.alias}</dd>
        <dt>context</dt><dd class="mono">{p.runtime.ctx_total ? Number(p.runtime.ctx_total).toLocaleString() : "auto"}</dd>
        {#if result.profile_path}<dt>file</dt><dd class="path">{result.profile_path}</dd>{/if}
      </dl>
    {:else}
      {#each result.files as f}<div class="path">{f}</div>{/each}
      <div class="faint small" style="margin-top: 8px;">No profile was made: no build can load this model yet. Make one in Profiles once a build does.</div>
    {/if}
    {#each result.warnings as w}<div class="notice small" style="margin-top: 6px;"><span class="chip warn">note</span><span>{w}</span></div>{/each}
    {#if result.build}
      <div class="notice small" style="margin-top: 8px;">
        <span class="chip {result.build.verify?.hip_ok ? 'pass' : 'block'}">{result.build.verify?.hip_ok ? "HIP backend loaded" : "HIP backend did not load"}</span>
        <span>{result.build.tag} {result.build.skipped_existing ? "was already built" : "built and installed"} at <span class="mono">{result.build.dir}</span></span>
      </div>
    {/if}
    <div class="toolbar" style="margin: 16px 0 0;">
      {#if p}
        <button class="btn primary" onclick={openInProfiles}>Open in Profiles</button>
        <button class="btn" onclick={launch} disabled={launching || !!wz.check?.results?.some((r) => outcomeKind(r.outcome) === "block")}
          title="Loads the model onto the GPU now, after the same pre-flight as Profiles. The wizard never does this by itself.">
          {#if launching}<span class="spinner"></span> Loading…{:else}Launch{/if}
        </button>
      {:else}
        <button class="btn" onclick={() => go("profiles")}>Open Profiles</button>
      {/if}
      <div class="grow"></div>
      <button class="btn" onclick={startOver}>Get another model</button>
    </div>
  </section>

  {#if p}
    <section class="card">
      <div class="sec">
        Pre-flight <span class="faint">the checks a launch runs, now</span>
        {#if wz.checking}<span class="chip plain live" style="margin-left: auto;">checking</span>{/if}
        <span style="margin-left: {wz.checking ? '8px' : 'auto'};"><button class="btn small" onclick={runCheck} disabled={wz.checking}>Re-check</button></span>
      </div>
      {#if wz.check?.error}<div class="notice"><span class="chip block">check failed</span><span class="mono">{wz.check.error}</span></div>{/if}
      {#if wz.check?.results}
        <div class="preflight">
          {#each wz.check.results as r (r.id)}
            {@const kind = outcomeKind(r.outcome)}
            {@const msg = outcomeMsg(r.outcome)}
            <div class="row">
              <span class="n">{r.spec_number}</span>
              <span class="t">{r.title}</span>
              <span class="chip {kind}">{kind}</span>
              {#if msg}<span class="msg">{msg}</span>{/if}
            </div>
          {/each}
        </div>
      {:else if !wz.checking && !wz.check?.error}
        <div class="empty">Not checked yet</div>
      {/if}
    </section>
  {/if}
{:else}
  <section class="card"><div class="empty">Nothing here yet. <button class="link" onclick={startOver}>Start from the search</button>.</div></section>
{/if}

</div>
{/key}

{#if toast}
  <div class="toast" class:error={toast.isError} transition:fly={toastFly}>{toast.text}</div>
{/if}

<style>
  .stepper { display: flex; align-items: center; gap: 6px; margin: 0 0 18px; flex-wrap: wrap; }
  .stepper .st {
    all: unset; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px 6px 6px;
    border: 1px solid var(--rule); border-radius: 999px; font-size: 13px; color: var(--ink-muted); background: var(--ground-raised);
  }
  .stepper .st .num {
    width: 20px; height: 20px; border-radius: 50%; display: inline-grid; place-items: center; font-family: var(--mono);
    font-size: 11px; background: var(--ground-inset); color: var(--ink-faint);
  }
  .stepper .st.past { color: var(--ink); }
  .stepper .st.past .num { background: var(--pass-bg); color: var(--pass); }
  .stepper .st.on { border-color: var(--accent-line); background: var(--accent-soft); color: var(--accent); font-weight: 600; }
  .stepper .st.on .num { background: var(--accent); color: var(--accent-ink); }
  .stepper .st:disabled { cursor: default; opacity: 0.5; }
  .stepper .st:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .stepper .sep { width: 18px; height: 1px; background: var(--rule-strong); }

  .searchbar { display: flex; gap: 10px; align-items: center; }
  .searchbar .q { flex: 1; font-size: 14px; height: 38px; }
  .opt { display: inline-flex; gap: 6px; align-items: center; font-size: 12.5px; color: var(--ink-muted); white-space: nowrap; }

  .card.flush .pad { padding: 14px 18px 0; }
  table.grid th.r, table.grid td.r { text-align: right; }
  tr.hit { cursor: pointer; }
  tr.hit .repo { font-size: 12.5px; color: var(--ink); margin-right: 6px; }
  tr.choice { cursor: pointer; }
  tr.choice.rec td { background: var(--accent-soft); }
  tr.choice.sel td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
  tr.choice .lbl { font-weight: 700; font-size: 12.5px; margin-right: 6px; }
  tr.choice .path { margin-top: 2px; }

  .repohead { display: flex; flex-direction: column; gap: 8px; margin-bottom: 8px; }
  .repohead .who { display: flex; gap: 10px; align-items: baseline; }
  .big { font-size: 15px; font-weight: 700; }
  .facts { display: flex; gap: 6px 16px; flex-wrap: wrap; font-size: 12.5px; color: var(--ink-muted); align-items: center; }
  .note-row { padding: 4px 0; align-items: flex-start; }
  .note-row .chip { flex: none; }

  .verdict { border: 1px solid var(--rule); border-radius: var(--radius-sm); padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; background: var(--ground-inset); }
  .verdict.pass { flex-direction: row; align-items: center; gap: 10px; border-color: var(--pass-bg); }
  .verdict.warn { border-color: var(--warn-bg); }
  .verdict.accent { border-color: var(--accent-line); }
  .verdict .vt { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 14px; }
  .verdict .expl { font-size: 12.5px; color: var(--ink-muted); line-height: 1.5; }
  .source { margin-top: 12px; padding: 12px 14px; border: 1px dashed var(--rule-strong); border-radius: var(--radius-sm); }
  .subjects { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 2px; }
  .subjects li { font-size: 11.5px; color: var(--ink-muted); }
  .k { font-weight: 600; color: var(--ink-muted); }

  .options, .roots { display: flex; flex-direction: column; gap: 6px; }
  .opt-row {
    display: flex; gap: 10px; align-items: center; padding: 8px 12px; border: 1px solid var(--rule); border-radius: var(--radius-sm);
    background: var(--ground-inset); cursor: pointer; font-size: 13px; flex-wrap: wrap;
  }
  .opt-row.on { border-color: var(--accent-line); background: var(--accent-soft); }
  .ruled { margin-top: 12px; }
  .ruled summary { cursor: pointer; }
  .ruled ul { margin: 6px 0 0; padding-left: 18px; color: var(--ink-muted); }
  .fix { color: var(--ink-muted); margin-top: 2px; }

  .consent {
    display: flex; gap: 10px; align-items: flex-start; padding: 12px 14px; border: 1px solid var(--warn-bg);
    border-radius: var(--radius-sm); background: var(--warn-bg); font-size: 13px; line-height: 1.5; cursor: pointer;
  }
  .consent.ok { border-color: var(--accent-line); background: var(--accent-soft); }
  .consent input { margin-top: 3px; }

  .addroot { display: flex; gap: 8px; align-items: center; margin-top: 10px; }
  .addroot input { width: 320px; }

  .steps { list-style: none; margin: 10px 0 0; padding: 0; display: flex; flex-direction: column; gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: var(--radius-sm); overflow: hidden; }
  .stp { background: var(--ground-raised); padding: 10px 14px; display: flex; flex-direction: column; gap: 6px; }
  .stp .head { display: flex; gap: 10px; align-items: center; }
  .stp .title { font-size: 13px; }
  .stp.done .title { color: var(--ink-muted); }
  .stp .chip { min-width: 64px; justify-content: center; }
  .stp .meter { margin-left: 74px; }
  .stp .prog, .stp > .faint { margin-left: 74px; }
  .buildlog { margin-left: 74px; max-height: 190px; overflow: auto; font-size: 11px; }

  .done-head { display: flex; gap: 10px; align-items: center; }
</style>
