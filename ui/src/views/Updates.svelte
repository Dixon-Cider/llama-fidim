<script>
  import { onDestroy } from "svelte";
  import { api, onEvent } from "../api.js";
  import { fly, fade, slide } from "svelte/transition";
  import { arrive, leave, stagger, LAYOUT } from "../motion.js";

  let check = $state(null);      // update_check result
  let checking = $state(false);
  let installing = $state(false);
  let install = $state(null);    // InstallReport of the last install this session
  let promote = $state(null);    // PromoteReport
  let rollback = $state(null);   // RollbackReport
  let history = $state([]);
  let log = $state([]);
  let error = $state("");
  let unlisten = null;

  // ROCm runtimes
  let cfg = $state(null);
  let runtimes = $state([]);
  let families = $state([]);
  let avail = $state(null);      // { runtimes, problems }
  let availBusy = $state(false);
  let rocmBusy = $state("");
  let rocmError = $state("");

  // Unsloth fork builds (the DiffusionGemma engine)
  let dg = $state(null);         // unsloth_check result
  let dgChecking = $state(false);
  let dgInstalling = $state(false);
  let dgInstall = $state(null);  // InstallReport of the last fork install this session
  let dgPromote = $state(null);  // PromoteReport
  let dgPreview = $state(null);  // PromotePreview, shown for confirmation before anything moves
  let dgError = $state("");
  let dgGfx = $state("");        // "" = automatic (config family, else a guess from the cards)
  // Install with the FIDIM runner patch laid over the zip (the overlay
  // published for the release). Offered only when it is.
  let dgOverlay = $state(false);
  const dgWithPatch = $derived(dgOverlay && !!dg?.overlay_available);
  const dgTargetInstalled = $derived(dgWithPatch ? !!dg?.overlay_installed : !!dg?.already_installed);
  // Which section's action the shared progress log belongs to, so it
  // renders under the button that started it.
  let logOwner = $state("upstream");

  onEvent("update-progress", (line) => {
    log = [...log.slice(-400), line];
  }).then((u) => (unlisten = u));
  onDestroy(() => unlisten && unlisten());

  async function loadHistory() {
    try { history = await api("update_history"); } catch (e) { /* non-fatal */ }
  }
  async function loadRuntimes() {
    try {
      const [c, r] = await Promise.all([api("get_config"), api("list_runtimes")]);
      cfg = c.config;
      runtimes = r;
    } catch (e) { rocmError = String(e); }
  }

  async function doCheck() {
    checking = true; error = ""; promote = null; rollback = null;
    try {
      check = await api("update_check");
    } catch (e) { error = String(e); }
    checking = false;
  }

  async function doInstall(source) {
    installing = true; error = ""; log = []; logOwner = "upstream"; install = null; promote = null;
    try {
      install = await api("update_install", { tag: check?.latest?.tag ?? null, source });
      await doCheck();
    } catch (e) { error = String(e); }
    installing = false;
  }

  async function doPromote(all) {
    error = ""; rollback = null;
    try {
      promote = await api("update_promote", {
        toPath: install.dir,
        toVersion: install.verify.version,
        fromPath: all ? null : (check?.newest_installed?.path ?? null),
        all,
      });
      await loadHistory();
    } catch (e) { error = String(e); }
  }

  async function doRollback() {
    error = ""; promote = null;
    try {
      rollback = await api("update_rollback");
      await loadHistory();
    } catch (e) { error = String(e); }
  }

  const family = $derived(cfg?.rocm_family || "gfx120X-all");
  const installedVersions = $derived(new Set(runtimes.filter((r) => r.source.startsWith("amd-")).map((r) => r.version)));

  async function checkRocm() {
    availBusy = true; rocmError = ""; avail = null;
    try {
      if (!families.length) families = await api("rocm_families").catch(() => []);
      avail = await api("rocm_available", { family });
    } catch (e) { rocmError = String(e); }
    availBusy = false;
  }
  async function installRocm(rt) {
    rocmBusy = rt.version; rocmError = ""; log = []; logOwner = "rocm";
    try {
      await api("rocm_install", { runtime: rt });
      await loadRuntimes();
    } catch (e) { rocmError = String(e); }
    rocmBusy = "";
  }
  async function removeRocm(version) {
    rocmBusy = version; rocmError = "";
    try { await api("rocm_remove", { version }); await loadRuntimes(); } catch (e) { rocmError = String(e); }
    rocmBusy = "";
  }
  async function makeDefault(name) {
    rocmBusy = name; rocmError = "";
    try {
      const c = { ...cfg, default_runtime: name === "default" ? null : name };
      await api("save_config", { config: c });
      await loadRuntimes();
    } catch (e) { rocmError = String(e); }
    rocmBusy = "";
  }
  // Not run on load: the unauthenticated GitHub API allows 60 calls an hour,
  // shared with the upstream check.
  async function dgCheck() {
    dgChecking = true; dgError = ""; dgPromote = null; dgPreview = null;
    try {
      dg = await api("unsloth_check", { tag: null, gfx: dgGfx || null });
    } catch (e) { dgError = String(e); }
    dgChecking = false;
  }

  async function dgDoInstall() {
    dgInstalling = true; dgError = ""; log = []; logOwner = "unsloth"; dgInstall = null; dgPromote = null; dgPreview = null;
    try {
      dgInstall = await api("unsloth_install", { tag: dg?.latest?.tag ?? null, gfx: dg?.gfx ?? (dgGfx || null), overlay: dgWithPatch });
      await dgCheck();
    } catch (e) { dgError = String(e); }
    dgInstalling = false;
  }

  // Which profiles would move, and off which runner patch, is shown first;
  // only the ones listed there move.
  async function dgPreviewPromote() {
    dgError = ""; dgPromote = null;
    try {
      dgPreview = await api("unsloth_promote_preview", { toPath: dgInstall.dir });
    } catch (e) { dgError = String(e); }
  }

  async function dgDoPromote() {
    dgError = "";
    try {
      const ids = dgPreview.moves.map((m) => m.profile_id);
      dgPromote = await api("unsloth_promote", { toPath: dgInstall.dir, toVersion: dgInstall.verify.version, ids });
      dgPreview = null;
      await loadHistory();
    } catch (e) { dgError = String(e); }
  }

  async function setFamily(f) {
    try {
      const c = { ...cfg, rocm_family: f || null };
      await api("save_config", { config: c });
      cfg = c;
      avail = null;
    } catch (e) { rocmError = String(e); }
  }

  // Download progress out of the log lines the installers emit
  // ("name: 123 / 745 MB"); the bar fills between files.
  const progress = $derived.by(() => {
    for (let i = log.length - 1; i >= 0; i--) {
      const m = /: (\d+) \/ (\d+) MB/.exec(log[i]);
      if (m) return Math.round((100 * Number(m[1])) / Math.max(1, Number(m[2])));
      if (/MB complete|unpacking|extracted/.test(log[i])) return 100;
    }
    return null;
  });

  function when(unix) {
    return unix ? new Date(unix * 1000).toLocaleString() : "—";
  }
  const day = (iso) => (iso ? String(iso).slice(0, 10) : "");

  doCheck();
  loadHistory();
  loadRuntimes();
</script>

<h1>
  Updates
  <span class="sub">llama.cpp releases and ROCm runtimes, installed side by side. Nothing here starts a server.</span>
</h1>
<p class="lede">
  Each build gets its own folder and is never modified. Promote moves profiles onto it and remembers
  the old build, so rollback is one click.
</p>

<div class="toolbar">
  <button class="btn" onclick={doCheck} disabled={checking || installing}>
    {checking ? "Checking…" : "Check for updates"}
  </button>
  {#if check}
    <button class="btn primary" onclick={() => doInstall(false)}
      disabled={installing || !!check.asset_error || (!check.update_available && !check.already_installed)}>
      {installing ? "Installing…" : check.already_installed ? "Re-verify prebuilt" : "Install prebuilt"}
    </button>
    <button class="btn" onclick={() => doInstall(true)} disabled={installing}>
      Build from source
    </button>
  {/if}
  {#if error}<span class="chip block shake">{error}</span>{/if}
</div>

{#if check}
  <div class="card">
    <table class="grid">
      <tbody>
        <tr><th>Upstream latest</th>
          <td class="mono">{check.latest.tag}</td>
          <td class="faint">{check.latest.published_at}</td>
          <td>{#if check.update_available}<span class="chip warn">update available{check.behind != null ? ` · ${check.behind} releases behind` : ""}</span>
              {:else}<span class="chip pass">up to date</span>{/if}</td></tr>
        <tr><th>Newest installed</th>
          <td class="mono">{check.newest_installed?.version ?? "none"}</td>
          <td class="path" colspan="2">{check.newest_installed?.path ?? "—"}</td></tr>
        <tr><th>Install target</th>
          <td class="path" colspan="2">{check.install_dir}</td>
          <td>{#if check.already_installed}<span class="chip pass">present</span>{/if}</td></tr>
        <tr><th>Prebuilt assets</th>
          <td colspan="3" class="path">
            {#if check.asset_error}<span class="chip block">{check.asset_error}</span>
            {:else}{#each check.assets as a}{a.name} ({(a.size / 1048576).toFixed(0)} MB)<br />{/each}{/if}
          </td></tr>
      </tbody>
    </table>
    <p class="faint small" style="margin: 10px 0 0;">
      Prebuilt = upstream's CPU zip merged with its ROCm zip. Whether the HIP backend loads is checked with
      --list-devices. If it does not, Build from source compiles the same tag with your toolchain.
    </p>
  </div>

  {#if check.changes?.length}
    <div class="card">
      <div class="sec">
        What changed
        <span class="faint">{check.changes.length} releases since {check.newest_installed?.version ?? "the start"}{check.changes_complete ? "" : ", newest 100 only"}</span>
      </div>
      <div class="changes">
        {#each check.changes as c, i (c.tag)}
          <div class="change" in:fly={arrive(stagger(i, 20))}>
            <span class="mono tag">{c.tag}</span>
            <span class="faint mono">{day(c.published_at)}</span>
            <span class="title">{c.title}</span>
          </div>
        {/each}
      </div>
    </div>
  {/if}
{/if}

{#snippet logCard(show, busy)}
  {#if show}
    <div class="card" transition:slide={leave}>
      {#if busy && progress != null}
        <div class="meter" style="margin-bottom: 10px;"><div class="fill" style="width: {progress}%;"></div></div>
      {/if}
      <div class="logbox" style="max-height: 220px; overflow: auto;">
        {#each log as line}<div class="mono">{line}</div>{/each}
        {#if busy && !log.length}<div class="faint">starting…</div>{/if}
      </div>
    </div>
  {/if}
{/snippet}

{@render logCard(logOwner !== "unsloth" && !!(installing || rocmBusy || log.length), !!(installing || rocmBusy))}

{#if install}
  <div class="card">
    <h2 style="margin-top: 0;">
      {install.tag} <span class="chip plain">{install.source}</span>
      {#if install.verify.hip_ok}<span class="chip pass">HIP backend loaded</span>
      {:else}<span class="chip block">HIP backend did not load</span>{/if}
      {#if install.skipped_existing}<span class="chip plain">already present, verified only</span>{/if}
    </h2>
    <div class="path">{install.dir}</div>
    <div class="mono">binary reports {install.verify.version ?? "?"} {install.verify.commit ? `(${install.verify.commit})` : ""}</div>
    {#if install.verify.devices.length}
      <table class="grid" style="margin-top: 6px;">
        <thead><tr><th>Index</th><th>Name</th><th>VRAM</th></tr></thead>
        <tbody>
          {#each install.verify.devices as d}
            <tr><td class="mono">{d.backend}{d.index}</td><td>{d.name}</td><td class="num">{(d.total_mib / 1024).toFixed(1)} GiB</td></tr>
          {/each}
        </tbody>
      </table>
    {/if}
    {#if install.verify.detail}<pre class="path" style="white-space: pre-wrap;">{install.verify.detail}</pre>{/if}
    <div class="toolbar" style="margin-top: 10px;">
      <button class="btn primary" onclick={() => doPromote(false)} disabled={!install.verify.hip_ok || !check?.newest_installed}>
        Promote profiles on {check?.newest_installed?.version ?? "previous build"}
      </button>
      <button class="btn" onclick={() => doPromote(true)} disabled={!install.verify.hip_ok}>
        Promote all unpinned profiles
      </button>
    </div>
    <p class="faint small">
      Profiles with <span class="mono">"build_pinned": true</span> stay where they are.
    </p>
  </div>
{/if}

{#if promote}
  <div class="card">
    <h2 style="margin-top: 0;">Promoted {promote.batch.entries.length} profile(s)</h2>
    <table class="grid">
      <tbody>
        {#each promote.batch.entries as e}
          <tr><td class="mono">{e.profile_id}</td><td class="mono faint">{e.from.version ?? "?"} → {e.to.version ?? "?"}</td></tr>
        {/each}
        {#each promote.skipped as [id, why]}
          <tr><td class="mono">{id}</td><td class="faint">skipped: {why}</td></tr>
        {/each}
      </tbody>
    </table>
    <p class="muted">Nothing was launched. Rollback undoes this batch.</p>
  </div>
{/if}

{#if rollback}
  <div class="card">
    <h2 style="margin-top: 0;">Rolled back {rollback.restored.length} profile(s)</h2>
    {#each rollback.restored as e}<div class="mono">{e.profile_id} → {e.from.version ?? "?"}</div>{/each}
    {#each rollback.skipped as [id, why]}<div class="faint">{id}: {why}</div>{/each}
  </div>
{/if}

<div class="card">
  <div class="toolbar" style="margin-bottom: 6px;">
    <strong>Promotion history</strong>
    <button class="btn" onclick={doRollback} disabled={!history.length}>Roll back most recent</button>
  </div>
  <table class="grid">
    <thead><tr><th>When</th><th>Profiles</th><th>To</th></tr></thead>
    <tbody>
      {#each [...history].reverse() as b}
        <tr>
          <td class="mono faint">{when(b.at_unix)}</td>
          <td class="mono">{b.entries.map((e) => e.profile_id).join(", ")}</td>
          <td class="mono faint">{b.entries[0]?.to.version ?? "?"}</td>
        </tr>
      {:else}
        <tr><td colspan="3"><div class="empty">No promotions yet</div></td></tr>
      {/each}
    </tbody>
  </table>
</div>

<h2>DiffusionGemma engine <span class="sub">Unsloth's llama.cpp builds, which carry the DiffusionGemma runner. Each brings its own ROCm and runs with nothing added to PATH.</span></h2>

<div class="card">
  <div class="sec">
    Unsloth builds
    <span class="faint">unslothai/llama.cpp releases, one zip per GPU target</span>
    {#if dgError}<span class="chip block shake">{dgError}</span>{/if}
    <span style="margin-left: auto; display: flex; gap: 8px; align-items: center;">
      <label class="field" style="flex-direction: row; align-items: center; gap: 8px;" title="GPU target of the zip. Automatic uses the ROCm family from Settings, else a guess from your cards: Radeon AI PRO R9700 and RX 9000 are gfx120X.">
        <span class="k">gpu target</span>
        <select style="width: 200px;" value={dgGfx} onchange={(e) => { dgGfx = e.target.value; dgCheck(); }} disabled={dgChecking || dgInstalling}>
          <option value="">automatic{dg && !dgGfx ? ` (${dg.gfx})` : ""}</option>
          {#each dg?.gfx_available ?? [] as g}<option value={g}>{g}</option>{/each}
        </select>
      </label>
      <button class="btn" onclick={dgCheck} disabled={dgChecking || dgInstalling} title="Ask GitHub for the newest Unsloth release. Nothing is downloaded.">
        {dgChecking ? "Checking…" : "Check Unsloth"}
      </button>
      {#if dg}
        <button class="btn primary" onclick={dgDoInstall} disabled={dgInstalling || !dg.asset}
          title={dgWithPatch
            ? "Download the zip and the runner patch, check both against GitHub's sha256 digests and the patch's descriptor, unpack the zip into its own folder with the patched binaries laid over it, then run --version and --list-devices on it. No model is loaded."
            : "Download the zip, check it against GitHub's sha256 digest, unpack it into its own folder, then run --version and --list-devices on it. No model is loaded."}>
          {dgInstalling ? "Installing…" : dgTargetInstalled ? "Re-verify" : dgWithPatch ? "Install with patch" : "Install"}
        </button>
      {/if}
    </span>
  </div>
  {#if dg}
    <table class="grid">
      <tbody>
        <tr><th>Latest release</th>
          <td class="mono">{dg.latest.tag}</td>
          <td class="faint">{day(dg.latest.published_at)}</td>
          <td>{#if dg.already_installed}<span class="chip pass">installed</span>{/if}</td></tr>
        <tr><th title="The upstream llama.cpp release this build was cut from.">Upstream base</th>
          <td class="mono" colspan="3">{dg.upstream_tag ?? "?"}</td></tr>
        <tr><th>GPU target</th>
          <td class="mono" colspan="3">{dg.gfx}</td></tr>
        <tr><th>Asset</th>
          <td colspan="3" class="path">
            {#if dg.asset}
              {dg.asset.name} ({(dg.asset.size / 1048576).toFixed(0)} MB)
              {#if dg.asset.digest}<span class="mono faint" title="Checked against the download before anything is unpacked: {dg.asset.digest}">{dg.asset.digest.slice(0, 19)}…</span>
              {:else}<span class="chip warn" title="The download cannot be checked against a published hash.">no digest published</span>{/if}
            {:else}<span class="chip block">{dg.asset_error}</span>{/if}
          </td></tr>
        <tr><th title="Llama FIDIM's DiffusionGemma runner patch ({dg.overlay_patch}), built for this exact release and laid over Unsloth's zip: F16 prompt-KV store with a sliding-window ring, flash attention on the GPU, longer context, prefill reuse across blocks. Unsloth's ggml and ROCm files stay as shipped.">Runner patch</th>
          <td colspan="3">
            {#if dg.overlay_available}
              <label style="display: inline-flex; gap: 6px; align-items: center; margin-right: 10px;"
                title="Installs into {dg.overlay_install_dir}, beside the plain build. Every file is checked against the patch's descriptor, which also names the exact Unsloth zip it was built for.">
                <input type="checkbox" bind:checked={dgOverlay} disabled={dgInstalling} />
                Install with FIDIM runner patch
              </label>
              <span class="mono faint">{dg.overlay_patch} · {dg.overlay_asset.name} ({(dg.overlay_asset.size / 1048576).toFixed(0)} MB)</span>
              {#if dg.overlay_installed}<span class="chip pass">installed</span>{/if}
            {:else if dg.overlay_error}<span class="chip warn" title={dg.overlay_error}>{dg.overlay_patch} unavailable</span>
              <span class="faint small">{dg.overlay_error}</span>
            {:else}<span class="faint">{dg.overlay_patch} is not published for this release ({dg.overlay_repo})</span>{/if}
          </td></tr>
        <tr><th>Install target</th>
          <td class="path" colspan="3">{dgWithPatch ? dg.overlay_install_dir : dg.install_dir}</td></tr>
        <tr><th>Installed</th>
          <td colspan="3">
            {#each dg.installed as b}
              <div><span class="mono">{b.tag}</span> <span class="faint mono">{b.version}</span>
                <span class="chip accent" title="Runs on the ROCm DLLs inside the build, never a runtime from the list below.">bundled ROCm</span>
                {#if b.patch}<span class="chip plain" title="A build with a runner patch. Promotion moves a diffusion profile off it only onto a build whose patch has every feature this one declares.">patch: {b.patch}</span>{/if}</div>
            {:else}<span class="faint">none yet</span>{/each}
          </td></tr>
      </tbody>
    </table>
  {:else}
    <div class="empty">Check Unsloth to see the newest build for your GPU</div>
  {/if}
  <p class="faint small" style="margin: 10px 0 0;">
    Installed into Llama FIDIM's build folder; Unsloth Studio's own copy is never touched.
    Only diffusion profiles move onto these builds. llama-server profiles stay on upstream builds.
  </p>
</div>

{@render logCard(logOwner === "unsloth" && !!(dgInstalling || log.length), dgInstalling)}

{#if dgInstall}
  <div class="card">
    <h2 style="margin-top: 0;">
      {dgInstall.tag} <span class="chip plain">unsloth</span>
      {#if dgInstall.source === "unsloth-overlay"}<span class="chip accent" title="Unsloth's zip with the runner patch laid over it">patch: {dg?.overlay_patch ?? "dgpatch5"}</span>{/if}
      {#if dgInstall.verify.hip_ok}<span class="chip pass">HIP backend loaded</span>
      {:else}<span class="chip block">HIP backend did not load</span>{/if}
      {#if dgInstall.verify.runner_present}<span class="chip pass">runner present</span>
      {:else}<span class="chip block">runner missing</span>{/if}
      {#if dgInstall.skipped_existing}<span class="chip plain">already present, verified only</span>{/if}
    </h2>
    <div class="path">{dgInstall.dir}</div>
    <div class="mono">bundled llama-server reports {dgInstall.verify.version ?? "?"} {dgInstall.verify.commit ? `(${dgInstall.verify.commit})` : ""}</div>
    {#if dgInstall.verify.devices.length}
      <table class="grid" style="margin-top: 6px;">
        <thead><tr><th title="Indices as this build's own ROCm enumerates them.">Index</th><th>Name</th><th>VRAM</th></tr></thead>
        <tbody>
          {#each dgInstall.verify.devices as d}
            <tr><td class="mono">{d.backend}{d.index}</td><td>{d.name}</td><td class="num">{(d.total_mib / 1024).toFixed(1)} GiB</td></tr>
          {/each}
        </tbody>
      </table>
    {/if}
    {#if dgInstall.verify.detail}<pre class="path" style="white-space: pre-wrap;">{dgInstall.verify.detail}</pre>{/if}
    {#if dgInstall.verify.hip_ok && dgInstall.verify.runner_present}
      <div class="toolbar" style="margin-top: 10px;">
        <button class="btn primary" onclick={dgPreviewPromote} disabled={!!dgPreview}
          title="Lists the diffusion profiles that would move onto this build, and why the others stay, before anything moves. llama-server profiles never move here and pinned ones stay. A profile on a patched runner build moves only onto a build whose patch has every feature of its own: a dgpatch4 profile moves onto dgpatch5. Roll back most recent undoes a move.">
          Move diffusion profiles onto it…
        </button>
      </div>
    {/if}
  </div>
{/if}

{#if dgPreview && dgInstall}
  <div class="card" transition:slide={leave}>
    <h2 style="margin-top: 0;">
      {dgPreview.moves.length ? `Move ${dgPreview.moves.length} diffusion profile(s) onto ${dgInstall.tag}?` : `No diffusion profile would move onto ${dgInstall.tag}`}
    </h2>
    <table class="grid">
      <tbody>
        {#each dgPreview.moves as m}
          <tr><td class="mono">{m.profile_id}</td>
            <td class="mono faint">{m.from.version ?? "?"} → {dgInstall.verify.version ?? "?"}</td>
            <td>{#if m.from_patch}<span class="chip warn" title="On a {m.from_patch} runner build today. It moves because this build's runner patch has every feature {m.from_patch} declares; it then runs on the runtime bundled here.">leaves {m.from_patch}</span>{/if}</td>
            <td class="path">{m.from.path}</td></tr>
        {/each}
        {#each dgPreview.skipped as [id, why]}
          <tr><td class="mono">{id}</td><td class="faint" colspan="3">stays: {why}</td></tr>
        {/each}
      </tbody>
    </table>
    {#if dgPreview.moves.some((m) => m.from_patch)}
      <p class="faint small">
        Profiles marked <span class="mono">leaves …</span> run on another patched runner today. To keep one where it is,
        cancel and set <span class="mono">"build_pinned": true</span> on it.
      </p>
    {/if}
    <div class="toolbar" style="margin-top: 10px;">
      <button class="btn primary" onclick={dgDoPromote} disabled={!dgPreview.moves.length}>
        Move {dgPreview.moves.length} profile(s)
      </button>
      <button class="btn" onclick={() => (dgPreview = null)}>Cancel</button>
    </div>
  </div>
{/if}

{#if dgPromote}
  <div class="card">
    <h2 style="margin-top: 0;">Moved {dgPromote.batch.entries.length} diffusion profile(s)</h2>
    <table class="grid">
      <tbody>
        {#each dgPromote.batch.entries as e}
          <tr><td class="mono">{e.profile_id}</td><td class="mono faint">{e.from.version ?? "?"} → {e.to.version ?? "?"}</td></tr>
        {/each}
        {#each dgPromote.skipped as [id, why]}
          <tr><td class="mono">{id}</td><td class="faint">skipped: {why}</td></tr>
        {/each}
      </tbody>
    </table>
    <p class="muted">Nothing was launched. Rollback undoes this batch.</p>
  </div>
{/if}

<h2>ROCm runtimes <span class="sub">AMD's Windows ROCm, one folder per version next to the builds. Profiles pick one by name; the default applies when a profile names none.</span></h2>

<div class="card">
  <div class="sec">
    Installed
    <span class="faint">{runtimes.filter((r) => r.available).length} usable</span>
    {#if rocmError}<span class="chip block shake">{rocmError}</span>{/if}
  </div>
  <table class="grid">
    <thead><tr><th>Runtime</th><th>Version</th><th>Source</th><th>State</th><th>Folder</th><th></th></tr></thead>
    <tbody>
      {#each runtimes as r}
        <tr>
          <td class="mono">{r.name}{#if r.is_default} <span class="chip accent">default</span>{/if}{#if r.is_latest} <span class="chip pass">latest</span>{/if}</td>
          <td class="mono">{r.version ?? "—"}</td>
          <td>{r.source}</td>
          <td>{#if r.available}<span class="chip pass">ok</span>{:else}<span class="chip block">missing</span>{/if}</td>
          <td class="path">{r.dirs.join(" ; ")}</td>
          <td style="white-space: nowrap;">
            {#if !r.is_default && r.available}<button class="btn small" onclick={() => makeDefault(r.name)} disabled={!!rocmBusy}>Make default</button>{/if}
            {#if r.source.startsWith("amd-") && !r.is_default}<button class="btn small danger" onclick={() => removeRocm(r.version)} disabled={!!rocmBusy}>Remove</button>{/if}
          </td>
        </tr>
      {:else}
        <tr><td colspan="6"><div class="empty">No runtimes found. Install one below, or point rocm_bin at a HIP SDK in Settings.</div></td></tr>
      {/each}
    </tbody>
  </table>
</div>

<div class="card">
  <div class="sec">
    Available from AMD
    <span class="faint">releases from repo.radeon.com, nightlies from rocm.nightlies.amd.com</span>
    <span style="margin-left: auto; display: flex; gap: 8px; align-items: center;">
      <label class="field" style="flex-direction: row; align-items: center; gap: 8px;" title="GPU family the nightly index is built for. RDNA4 (Radeon AI PRO R9700, RX 9000) is gfx120X-all; RDNA3 is gfx110X-all.">
        <span class="k">family</span>
        <select style="width: 160px;" value={family} onchange={(e) => setFamily(e.target.value)}>
          {#each [...new Set([family, ...families])] as f}<option value={f}>{f}</option>{/each}
        </select>
      </label>
      <button class="btn" onclick={checkRocm} disabled={availBusy || !!rocmBusy}>{availBusy ? "Checking…" : "Check AMD"}</button>
    </span>
  </div>
  {#if avail}
    <table class="grid">
      <thead><tr><th>Version</th><th>Channel</th><th>Family</th><th></th></tr></thead>
      <tbody>
        {#each avail.runtimes as a, i (a.channel + a.version)}
          <tr in:fade={{ duration: LAYOUT, delay: stagger(i, 20) }}>
            <td class="mono">{a.version}{#if i === 0} <span class="chip pass">latest</span>{/if}</td>
            <td><span class="chip {a.channel === 'release' ? 'accent' : 'plain'}">{a.channel}</span></td>
            <td class="mono faint">{a.family ?? "all"}</td>
            <td>
              {#if installedVersions.has(a.version)}<span class="chip pass">installed</span>
              {:else}<button class="btn small" onclick={() => installRocm(a)} disabled={!!rocmBusy}>{rocmBusy === a.version ? "Installing…" : "Install"}</button>{/if}
            </td>
          </tr>
        {:else}
          <tr><td colspan="4"><div class="empty">Nothing found for {family}</div></td></tr>
        {/each}
      </tbody>
    </table>
    {#each avail.problems as p}<div class="faint small" style="margin-top: 6px;">{p}</div>{/each}
    <p class="faint small" style="margin: 10px 0 0;">
      Two wheels per version, roughly 1.5 GB to download and a few GB unpacked. Only the <span class="mono">bin</span> trees are kept.
      A new runtime takes effect the next time a server that names it starts.
    </p>
  {:else}
    <div class="empty">Check AMD to list versions for {family}</div>
  {/if}
</div>

<style>
  .changes { display: flex; flex-direction: column; }
  .change { display: grid; grid-template-columns: 80px 92px 1fr; gap: 12px; padding: 5px 0; border-bottom: 1px solid var(--rule); font-size: 13px; align-items: baseline; }
  .change:last-child { border-bottom: none; }
  .change .tag { font-weight: 600; }
</style>
