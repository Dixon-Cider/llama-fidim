// One duration scale for the whole app, and one switch that turns all of
// it off when the OS asks for reduced motion. Svelte transitions take these
// as their duration; CSS gets the same numbers through custom properties.
import { cubicOut } from "svelte/easing";

export const reduced =
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/// 120 ms feedback, 200 ms layout, 300 ms arrival.
export const dur = (ms) => (reduced ? 0 : ms);
export const FAST = dur(120);
export const LAYOUT = dur(200);
export const ARRIVE = dur(300);

// Presets for svelte/transition parameters.
export const arrive = (delay = 0) => ({ y: 6, duration: ARRIVE, delay: reduced ? 0 : delay, easing: cubicOut });
export const leave = { duration: LAYOUT, easing: cubicOut };
export const flipParams = { duration: LAYOUT, easing: cubicOut };
export const toastFly = { y: 12, duration: LAYOUT, easing: cubicOut };

/// Stagger delay for the i-th item in a list, capped so long lists don't crawl.
export const stagger = (i, step = 25, cap = 200) => (reduced ? 0 : Math.min(i * step, cap));
