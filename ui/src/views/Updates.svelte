<script>
  import { onDestroy } from "svelte";
  import { api, onEvent } from "../api.js";

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

  onEvent("update-progress", (line) => {
    log = [...log.slice(-400), line];
  }).then((u) => (unlisten = u));
  onDestroy(() => unlisten && unlisten());

  async function loadHistory() {
    try { history = await api("update_history"); } catch (e) { /* non-fatal */ }
  }

  async function doCheck() {
    checking = true; error = ""; promote = null; rollback = null;
    try {
      check = await api("update_check");
    } catch (e) { error = String(e); }
    checking = false;
  }

  async function doInstall(source) {
    installing = true; error = ""; log = []; install = null; promote = null;
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

  function when(unix) {
    return unix ? new Date(unix * 1000).toLocaleString() : "—";
  }

  doCheck();
  loadHistory();
</script>

<h1>Updates <span class="sub">upstream llama.cpp releases, installed side-by-side; nothing here launches a server</span></h1>
<p class="lede">
  Each tag lands in its own immutable directory. Promotion re-points profiles and records the old
  build so rollback is one click. Bench a promoted profile from the Benchmarks view when the GPUs are free.
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
  {#if error}<span class="chip block">{error}</span>{/if}
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
          <td class="mono faint" colspan="2">{check.newest_installed?.path ?? "—"}</td></tr>
        <tr><th>Install target</th>
          <td class="mono faint" colspan="2">{check.install_dir}</td>
          <td>{#if check.already_installed}<span class="chip pass">present</span>{/if}</td></tr>
        <tr><th>Prebuilt assets</th>
          <td colspan="3" class="mono faint">
            {#if check.asset_error}<span class="chip block">{check.asset_error}</span>
            {:else}{#each check.assets as a}{a.name} ({(a.size / 1048576).toFixed(0)} MB)<br />{/each}{/if}
          </td></tr>
      </tbody>
    </table>
    <p class="faint small" style=" margin: 8px 0 0;">
      Prebuilt = upstream's CPU zip merged with its ROCm zip (built against a newer ROCm than this box runs;
      rocBLAS resolves from PATH). Whether the HIP backend loads is verified with --list-devices, never assumed.
      If it doesn't, "Build from source" compiles the same tag with the local toolchain.
    </p>
  </div>
{/if}

{#if installing || log.length}
  <div class="card">
    <div class="logbox" style="max-height: 220px; overflow: auto;">
      {#each log as line}<div class="mono">{line}</div>{/each}
      {#if installing && !log.length}<div class="faint">starting…</div>{/if}
    </div>
  </div>
{/if}

{#if install}
  <div class="card">
    <h2 style="margin-top: 0;">
      {install.tag} <span class="chip plain">{install.source}</span>
      {#if install.verify.hip_ok}<span class="chip pass">HIP backend loaded</span>
      {:else}<span class="chip block">HIP backend did not load</span>{/if}
      {#if install.skipped_existing}<span class="chip plain">already present, verified only</span>{/if}
    </h2>
    <div class="mono faint">{install.dir}</div>
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
    <p class="faint small" style="">
      Profiles with <span class="mono">"build_pinned": true</span> are never moved (use it for the MTP drafter profile).
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
    <p class="muted">Nothing was launched. Bench when the GPUs are free; Rollback undoes this batch.</p>
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
