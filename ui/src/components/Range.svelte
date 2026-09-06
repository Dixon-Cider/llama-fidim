<script>
  // Number + slider pair. Every bounded numeric setting uses this so the
  // user always sees where a value sits in its range.
  //
  // `nullable`: the value may be null = "use the engine default"; a checkbox
  // gates the control and `placeholder` shows what the default is.
  let {
    value = $bindable(),
    label,
    min = 0,
    max = 100,
    step = 1,
    hint = "",
    nullable = false,
    placeholder = null,
    format = (v) => v,
    onchange = () => {},
    span = 2,
    title = "",
  } = $props();

  const disabled = $derived(nullable && (value === null || value === undefined));
  const pct = $derived(
    disabled ? 0 : Math.max(0, Math.min(100, ((Number(value) - min) / (max - min)) * 100)),
  );

  function fromInput(e) {
    const n = Number(e.target.value);
    value = Number.isFinite(n) ? n : min;
    onchange();
  }
  function toggle(e) {
    value = e.target.checked ? (placeholder ?? min) : null;
    onchange();
  }
</script>

<label class="field range" style="grid-column: span {span};" {title}>
  <span class="k" style="display: flex; justify-content: space-between; align-items: center;">
    <span style="display: flex; gap: 6px; align-items: center;">
      {#if nullable}
        <input type="checkbox" style="width: auto; margin: 0;" checked={!disabled} onchange={toggle} title="off = engine default" />
      {/if}
      {label}
    </span>
    {#if hint}<span class="faint" style="text-transform: none; letter-spacing: 0;">{hint}</span>{/if}
  </span>
  <!-- The slider and its min/value/max labels share one column so the
       labels sit under the track, not under the track + number box. -->
  <span style="display: flex; gap: 8px; align-items: flex-start;">
    <span style="flex: 1; display: flex; flex-direction: column; gap: 2px; min-width: 0;">
      <input
        type="range"
        {min} {max} {step}
        value={disabled ? min : value}
        {disabled}
        oninput={fromInput}
        style="width: 100%; --pct: {pct}%;"
      />
      <span class="faint mono" style="font-size: 11px; display: flex; justify-content: space-between;">
        <span>{format(min)}</span>
        <span>{disabled ? "engine default" : format(value)}</span>
        <span>{format(max)}</span>
      </span>
    </span>
    <input
      type="number"
      {min} {max} {step}
      value={disabled ? "" : value}
      placeholder={disabled ? (placeholder ?? "default") : ""}
      {disabled}
      oninput={fromInput}
      style="width: 104px; text-align: right; flex: none;"
    />
  </span>
</label>

<style>
  input[type="range"] {
    -webkit-appearance: none; appearance: none; height: 4px; padding: 0; border: none; margin: 8px 0 4px;
    background: linear-gradient(to right, var(--accent) var(--pct), var(--rule-strong) var(--pct));
    border-radius: 2px; cursor: pointer;
  }
  input[type="range"]:disabled { background: var(--rule); cursor: default; }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
    background: var(--ink); border: 2px solid var(--accent); cursor: pointer;
  }
  input[type="range"]:disabled::-webkit-slider-thumb { border-color: var(--rule-strong); background: var(--ink-faint); }
  input[type="range"]:focus { outline: none; }
  input[type="range"]:focus-visible::-webkit-slider-thumb { outline: 2px solid var(--accent); outline-offset: 2px; }
</style>
