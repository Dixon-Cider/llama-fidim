<script>
  // A number that counts up from zero the first time it appears, then steps
  // without animation, so live data does not jitter on every poll.
  import { onMount } from "svelte";
  import { ARRIVE } from "../motion.js";
  let { value = 0, format = (v) => v } = $props();
  let shown = $state(0);
  let settled = $state(false);
  onMount(() => {
    const target = Number(value) || 0;
    if (ARRIVE === 0 || !target) { shown = target; settled = true; return; }
    const t0 = performance.now();
    let raf;
    const tick = (t) => {
      const p = Math.min(1, (t - t0) / (ARRIVE + 100));
      shown = target * (1 - Math.pow(1 - p, 3));
      if (p < 1) raf = requestAnimationFrame(tick);
      else settled = true;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  });
  const display = $derived(settled ? value : shown);
</script>

{format(display)}
