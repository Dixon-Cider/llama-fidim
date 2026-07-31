<script>
  import { onDestroy } from "svelte";
  import { api } from "../api.js";

  let runs = $state([]);
  let slotsByPort = $state({});
  let loading = $state(false);
  let deepBusy = $state(false);
  let error = $state("");
  let now = $state(Math.floor(Date.now() / 1000));

  async function load(deep = false) {
    loading = true;
    error = "";
    try {
      runs = await api("status", { deep });
      const bySlots = {};
      for (const r of runs) {
        if (!r.alive) continue;
        try {
          bySlots[r.state.port] = await api("slots", { port: r.state.port, host: r.state.host });
        } catch {
          bySlots[r.state.port] = null; // endpoint absent on this build
        }
      }
      slotsByPort = bySlots;
    } catch (e) {
      error = String(e);
    }
    loading = false;
  }
  load();

  const tick = setInterval(() => (now = Math.floor(Date.now() / 1000)), 1000);
  const poll = setInterval(() => load(false), 5000);
  onDestroy(() => {
    clearInterval(tick);
    clearInterval(poll);
  });

  async function deepProbe() {
    deepBusy = true;
    await load(true);
    deepBusy = false;
  }

  async function stop(target) {
    try {
      await api("stop_run", { target });
      await load(false);
    } catch (e) {
      error = String(e);
    }
  }

  function uptime(started) {
    const s = Math.max(0, now - started);
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? `${h}h ${m}m` : `${m}m ${s % 60}s`;
  }

  function healthChip(r) {
    if (r.crashed) return { kind: "block", label: "crashed" };
    if (r.health === "healthy") return { kind: "pass", label: "healthy" };
    if (r.health === "responding-not-generating") return { kind: "warn", label: "responding, NOT generating" };
    return { kind: "block", label: "dead" };
  }
</script>

<h1>Running <span class="sub">re-attached from run state — servers outlive this window</span></h1>
<p class="lede">
  Health has three states, not two: a server that answers /v1/models but fails generation is a real
  observed condition. Deep probe sends a 1-token completion to distinguish it.
</p>

<div class="toolbar">
  <button class="btn" onclick={() => load(false)} disabled={loading}>Refresh</button>
  <button class="btn" onclick={deepProbe} disabled={deepBusy}>
    {deepBusy ? "Probing generation…" : "Deep health probe"}
  </button>
  {#if error}<span class="chip block">{error}</span>{/if}
</div>

{#each runs as r}
  {@const chip = healthChip(r)}
  {@const slots = slotsByPort[r.state.port]}
  <div class="card">
    <div style="display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;">
      <span style="font-weight: 700; font-size: 14px;">{r.state.profile_id}</span>
      <span class="chip {chip.kind}">{chip.label}</span>
      {#if r.state.cold_start}<span class="chip warn">cold cache</span>{/if}
      <span class="mono muted">pid {r.state.pid} · :{r.state.port} · alias {r.state.alias}</span>
      <span class="num muted">up {uptime(r.state.started_unix)}</span>
      <div class="grow" style="flex: 1;"></div>
      {#if r.alive}
        <button class="btn danger" onclick={() => stop(r.state.profile_id)}>Stop</button>
      {/if}
    </div>
    <div class="mono faint" style="font-size: 10.5px; margin-top: 6px;">
      devices {r.state.device_keys.join(", ")} · HIP_VISIBLE_DEVICES={r.state.visibility_env}
    </div>
    {#if slots}
      {@const busySlots = slots.filter((s) => s.state !== 0).length}
      <div style="display: flex; gap: 4px; align-items: center; margin-top: 8px;">
        <span class="mono faint" style="font-size: 10.5px;">slots {busySlots}/{slots.length}</span>
        {#each slots as s}
          <div
            title="slot {s.id}: {s.state === 0 ? 'idle' : 'processing'}"
            style="width: 18px; height: 10px; border-radius: 2px; background: {s.state === 0 ? 'var(--ground-inset)' : 'var(--accent)'}; border: 1px solid var(--rule-strong);"
          ></div>
        {/each}
      </div>
    {:else if r.alive}
      <div class="mono faint" style="font-size: 10.5px; margin-top: 6px;">/slots endpoint unavailable on this build</div>
    {/if}
    {#if r.crashed}
      <div class="mono" style="font-size: 11px; margin-top: 6px; color: var(--block);">
        Process gone — log kept for diagnosis: {r.state.log_path}
      </div>
    {/if}
  </div>
{:else}
  <div class="empty">{loading ? "Checking run state…" : "No servers running — launch one from Profiles"}</div>
{/each}
