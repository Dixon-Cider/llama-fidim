<script>
  import { api } from "../api.js";

  let rows = $state([]);
  let runtimes = $state([]);
  let loading = $state(false);
  let error = $state("");
  api("list_runtimes").then((r) => (runtimes = r)).catch(() => {});

  async function load(refresh) {
    loading = true;
    error = "";
    try {
      rows = await api("devices", { refresh });
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }
  load(false);

  const short = (k) => String(k ?? "").split(":").pop();
  const gib = (mib) => (mib / 1024).toFixed(1);
  function usedKind(d) {
    const used = 1 - d.free_mib / Math.max(1, d.total_mib);
    return used > 0.9 ? "block" : used > 0.7 ? "warn" : "pass";
  }
</script>

<h1>
  Devices
  <span class="sub">Fresh from <span class="mono">llama-server --list-devices</span>. Profiles bind to the stable key; the index is resolved at every launch.</span>
</h1>

<div class="toolbar">
  <button class="btn" onclick={() => load(true)} disabled={loading}>{loading ? "Enumerating…" : "Re-enumerate"}</button>
  {#if error}<span class="chip block">{error}</span>{/if}
</div>

<div class="tiles">
  {#each rows as row}
    {@const d = row.device}
    {@const used = d.total_mib - d.free_mib}
    <div class="tile" class:dim={d.integrated}>
      <div class="head">
        <span class="mono chip {d.integrated ? 'plain' : 'accent'}">{d.backend}{d.hip_index}{d.correlation_assumed ? " ~" : ""}</span>
        <span class="name">{d.name}</span>
        <span style="margin-left: auto;">
          {#if d.integrated}<span class="chip plain">iGPU · never bound</span>{:else}<span class="chip pass">discrete</span>{/if}
        </span>
      </div>
      <div>
        <div class="cap"><span class="num" style="color: var(--ink);">{gib(d.free_mib)} GiB free</span><span class="faint">{gib(used)} used of {gib(d.total_mib)} GiB</span></div>
        <div class="meter"><div class="fill {usedKind(d)}" style="width: {Math.round(100 * used / Math.max(1, d.total_mib))}%;"></div></div>
      </div>
      <dl class="kv">
        <dt>display</dt>
        <dd>{#if d.display}<span class="chip warn">{d.display.width}×{d.display.height} @ {d.display.refresh_hz} Hz</span>{:else}<span class="faint">none attached</span>{/if}</dd>
        <dt>occupied by</dt>
        <dd>{#if row.occupied_by.length}{#each row.occupied_by as p}<span class="chip accent">{p}</span> {/each}{:else}<span class="faint">no Llama FIDIM server</span>{/if}</dd>
        <dt>driver</dt>
        <dd class="mono">{d.driver_version ?? "—"}</dd>
        <dt>stable key</dt>
        <dd class="mono faint" style="font-size: 11px; word-break: break-all;">{d.stable_key}</dd>
      </dl>
    </div>
  {:else}
    <div class="card" style="grid-column: 1 / -1;"><div class="empty">{loading ? "Enumerating devices…" : "No devices found"}</div></div>
  {/each}
</div>
<p class="faint small" style="margin: 10px 0 0;">
  ~ means the index was matched by bus order between identically named cards; launch verifies it by per-process residency.
  A display on a compute card costs VRAM and can pre-empt compute.
</p>

<h2>ROCm runtimes <span class="sub">DLL folders a profile can launch with: HIP SDK installs, versions installed from the Updates tab, LM Studio's, and any you add in Settings. Newest first.</span></h2>
<div class="card flush" style="overflow-x: auto;">
  <table class="grid">
    <thead><tr><th>Name</th><th>Source</th><th>Version</th><th>State</th><th>Directories</th></tr></thead>
    <tbody>
      {#each runtimes as r}
        <tr>
          <td class="mono">{r.name}{#if r.is_default} <span class="chip accent">default</span>{/if}{#if r.is_latest} <span class="chip pass">latest</span>{/if}</td>
          <td>{r.source}</td>
          <td class="mono">{r.version ?? "—"}</td>
          <td>{#if r.available}<span class="chip pass">available</span>{:else}<span class="chip block">missing</span>{/if}</td>
          <td class="path">{r.dirs.join(" ; ")}</td>
        </tr>
      {:else}
        <tr><td colspan="5"><div class="empty">No runtimes discovered</div></td></tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .cap { display: flex; justify-content: space-between; font-size: 12.5px; margin-bottom: 6px; }
</style>
