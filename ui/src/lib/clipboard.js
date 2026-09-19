// Copy text: the async clipboard API first (WebView2 treats tauri.localhost
// as a secure context), then the old hidden-textarea copy, which works from a
// click handler even where the API is refused. Resolves to whether it worked.
export async function copyText(text) {
  const s = String(text ?? "");
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(s);
      return true;
    }
  } catch {
    // fall through to the textarea
  }
  const ta = document.createElement("textarea");
  ta.value = s;
  ta.setAttribute("readonly", "");
  ta.setAttribute("aria-hidden", "true");
  Object.assign(ta.style, { position: "fixed", top: "0", left: "0", opacity: "0", pointerEvents: "none" });
  document.body.appendChild(ta);
  const prev = document.activeElement;
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  ta.remove();
  prev?.focus?.();
  return ok;
}
