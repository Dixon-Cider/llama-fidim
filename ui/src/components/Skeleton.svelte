<script>
  // A shimmering placeholder the size of the content that is coming.
  let { height = "72px", width = "100%", radius = "8px" } = $props();
</script>

<div class="skel" style="height: {height}; width: {width}; border-radius: {radius};" aria-hidden="true"></div>

<style>
  .skel {
    background: linear-gradient(90deg, var(--ground-raised) 0%, var(--ground-lift) 50%, var(--ground-raised) 100%);
    background-size: 200% 100%;
    animation: shimmer 1.4s ease-in-out infinite;
    border: 1px solid var(--rule);
  }
  @keyframes shimmer { from { background-position: 200% 0; } to { background-position: -200% 0; } }
  @media (prefers-reduced-motion: reduce) { .skel { animation: none; background: var(--ground-raised); } }
</style>
