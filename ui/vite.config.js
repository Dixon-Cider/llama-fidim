import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

// Tauri expects a fixed dev port; clearScreen off keeps cargo output visible.
export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  build: {
    // api.js uses top-level await; the only runtime is WebView2 (evergreen
    // Chromium) or a modern browser in mock mode.
    target: "es2022",
  },
  server: {
    port: 1420,
    strictPort: true,
  },
});
