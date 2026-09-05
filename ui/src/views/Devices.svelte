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
</script>

<h1>
  Devices
  <span class="sub">enumerated fresh via llama-server --list-devices — the canonical index space</span>
</h1>
<p class="lede">
  Stable keys are what profiles bind to; indices are resolved at every launch and never trusted from disk.
</p>

<div class="toolbar">
  <button class="btn" onclick={() => load(true)} disabled={loading}>
    {loading ? "Enumerating…" : "Re-enumerate"}
  </button>
  {#if error}<span class="chip block">{error}</span>{/if}
</div>

<div class="card" style="padding: 0; overflow-x: auto;">
  <table class="grid">
    <thead>
      <tr>
        <th>Index</th><th>Name</th><th>Stable key</th><th>VRAM free / total</th>
        <th>Class</th><th>Display</th><th>Driver</th><th>Occupied by</th>
      </tr>
    </thead>
    <tbody>
      {#each rows as row}
        {@const d = row.device}
        <tr>
          <td class="mono">{d.backend}{d.hip_index}{d.correlation_assumed ? " ~" : ""}</td>
          <td>{d.name}</td>
          <td class="mono faint">{d.stable_key}</td>
          <td class="num">{(d.free_mib / 1024).toFixed(1)} / {(d.total_mib / 1024).toFixed(1)} GiB</td>
          <td>
            {#if d.integrated}<span class="chip block">iGPU</span>
            {:else}<span class="chip pass">discrete</span>{/if}
          </td>
          <td>
            {#if d.display}
              <span class="chip warn">{d.display.width}x{d.display.height}@{d.display.refresh_hz}</span>
            {:else}<span class="faint">—</span>{/if}
          </td>
          <td class="mono faint">{d.driver_version ?? "—"}</td>
          <td>
            {#if row.occupied_by.length}
              {#each row.occupied_by as p}<span class="chip accent">{p}</span>{/each}
            {:else}<span class="faint">—</span>{/if}
          </td>
        </tr>
      {:else}
        <tr><td colspan="8"><div class="empty">{loading ? "Enumerating devices…" : "No devices found"}</div></td></tr>
      {/each}
    </tbody>
  </table>
</div>
<p class="faint mono" style="font-size: 10.5px;">
  ~ identical-name correlation by bus order — verified at launch by per-process residency ·
  display attached = compositing can consume VRAM and preempt compute (R-06)
</p>

<h2 style="margin-top: 18px;">ROCm runtimes <span class="sub">DLL search paths a profile can launch against; discovered from the HIP SDK, ComfyUI, LM Studio, and config</span></h2>
<div class="card" style="padding: 0; overflow-x: auto;">
  <table class="grid">
    <thead><tr><th>Name</th><th>Source</th><th>Version</th><th>Available</th><th>Directories</th></tr></thead>
    <tbody>
      {#each runtimes as r}
        <tr>
          <td class="mono">{r.name}{#if r.is_default} <span class="chip accent">default</span>{/if}</td>
          <td>{r.source}</td>
          <td class="mono">{r.version ?? "—"}</td>
          <td>{#if r.available}<span class="chip pass">yes</span>{:else}<span class="chip block">missing</span>{/if}</td>
          <td class="mono faint" style="font-size: 10.5px;">{r.dirs.join(" ; ")}</td>
        </tr>
      {:else}
        <tr><td colspan="5"><div class="empty">No runtimes discovered</div></td></tr>
      {/each}
    </tbody>
  </table>
</div>
