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

  const THUMB = 14; // px, must match the CSS below
  const disabled = $derived(nullable && (value === null || value === undefined));
  const pct = $derived(
    disabled ? 0 : Math.max(0, Math.min(100, ((Number(value) - min) / (max - min)) * 100)),
  );
  // The thumb's centre travels from THUMB/2 to width - THUMB/2, not from 0
  // to width, so the value label is offset by that much to sit under it.
  const thumbShift = $derived((50 - pct) * (THUMB / 100));
  // Hide the end labels when the value label would sit on top of them.
  const nearMin = $derived(pct < 12);
  const nearMax = $derived(pct > 88);

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
  <span class="k" style="display: flex; justify-content: space-between; align-items: center; gap: 10px;">
    <span style="display: flex; gap: 6px; align-items: center;">
      {#if nullable}
        <input type="checkbox" style="width: auto; margin: 0;" checked={!disabled} onchange={toggle} title="off = engine default" />
      {/if}
      {label}
    </span>
    {#if hint}<span class="faint" style="font-weight: 400; text-align: right;">{hint}</span>{/if}
  </span>
  <span class="row">
    <span class="track">
      <input
        type="range"
        {min} {max} {step}
        value={disabled ? min : value}
        {disabled}
        oninput={fromInput}
        style="--pct: {pct}%;"
      />
      <span class="scale">
        <span class="end" class:hide={!disabled && nearMin}>{format(min)}</span>
        <span class="end" class:hide={!disabled && nearMax}>{format(max)}</span>
        <span class="cur" class:off={disabled} style="left: calc({pct}% + {thumbShift}px);">{disabled ? "engine default" : format(value)}</span>
      </span>
    </span>
    <input
      type="number"
      {min} {max} {step}
      value={disabled ? "" : value}
      placeholder={disabled ? (placeholder ?? "default") : ""}
      {disabled}
      oninput={fromInput}
      class="num"
    />
  </span>
</label>

<style>
  .row { display: flex; gap: 10px; align-items: flex-start; }
  .track { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .num { width: 104px; text-align: right; flex: none; }
  input[type="range"] {
    -webkit-appearance: none; appearance: none; width: 100%; height: 4px; min-height: 0;
    padding: 0; border: none; margin: 14px 0 6px; box-shadow: none;
    background: linear-gradient(to right, var(--accent) var(--pct), var(--rule-strong) var(--pct));
    border-radius: 2px; cursor: pointer;
  }
  input[type="range"]:disabled { background: var(--rule); cursor: default; }
  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%;
    background: var(--ink); border: 2px solid var(--accent); cursor: pointer;
    transition: transform .12s ease;
  }
  input[type="range"]:hover::-webkit-slider-thumb { transform: scale(1.15); }
  input[type="range"]:disabled::-webkit-slider-thumb { border-color: var(--rule-strong); background: var(--ink-faint); }
  input[type="range"]:focus { outline: none; box-shadow: none; }
  input[type="range"]:focus-visible::-webkit-slider-thumb { outline: 2px solid var(--accent); outline-offset: 2px; }
  .scale { position: relative; height: 16px; font-family: var(--mono); font-size: 11px; color: var(--ink-faint); }
  .scale .end { position: absolute; top: 0; transition: opacity .12s; }
  .scale .end:first-child { left: 0; }
  .scale .end:nth-child(2) { right: 0; }
  .scale .end.hide { opacity: 0; }
  .scale .cur { position: absolute; top: 0; transform: translateX(-50%); color: var(--ink); font-weight: 600; white-space: nowrap; transition: transform var(--t-fast), color var(--t-fast); }
  input[type="range"]:active + .scale .cur { transform: translate(-50%, -3px) scale(1.1); color: var(--accent); }
  .scale .cur.off { left: 0 !important; transform: none; color: var(--ink-faint); font-weight: 400; }
  @media (prefers-reduced-motion: reduce) { input[type="range"]::-webkit-slider-thumb, .scale .end { transition: none; } }
</style>
