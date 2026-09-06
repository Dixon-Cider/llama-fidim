<script>
  // A list of folders or patterns, one row each. Rows are plain inputs so
  // paths can be pasted; empty rows are dropped on save by the caller.
  let { value = $bindable([]), placeholder = "", addLabel = "Add folder", mono = true, onchange = () => {} } = $props();
  function set(i, v) { value = value.map((x, j) => (j === i ? v : x)); onchange(); }
  function remove(i) { value = value.filter((_, j) => j !== i); onchange(); }
  function add() { value = [...value, ""]; onchange(); }
</script>

<div class="plist">
  {#each value as v, i}
    <div class="prow">
      <input class:mono value={v} {placeholder} oninput={(e) => set(i, e.target.value)} spellcheck="false" />
      <button type="button" class="btn small" onclick={() => remove(i)} title="remove">×</button>
    </div>
  {:else}
    <div class="faint small" style="padding: 4px 0;">none yet</div>
  {/each}
  <button type="button" class="btn small" onclick={add}>+ {addLabel}</button>
</div>

<style>
  .plist { display: flex; flex-direction: column; gap: 6px; align-items: flex-start; }
  .prow { display: flex; gap: 6px; width: 100%; }
  .prow input { flex: 1; }
</style>
