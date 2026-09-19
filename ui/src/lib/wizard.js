// Shared by the Models view and its browser-preview mock: how a job's
// progress is rebuilt from its events (the same rules as core's
// wizard::JobProgress), and the small formats the view prints.

const LOG_KEEP = 400;

/// A job's progress before any event: every step pending.
export function newProgress(plan) {
  return {
    steps: (plan?.steps ?? []).map(() => ({ status: "pending", stage: null, file: null, done: null, total: null, bps: null, error: null })),
    log: [],
  };
}

/// Fold one `wizard-progress` event into a progress (mutates and returns it).
export function applyWizardEvent(progress, ev) {
  if (ev.line != null && ev.status !== "failed") {
    progress.log.push(ev.line);
    if (progress.log.length > LOG_KEEP) progress.log.splice(0, progress.log.length - LOG_KEEP);
  }
  const s = progress.steps[ev.step];
  if (!s) return progress;
  s.status = ev.status;
  if (ev.stage != null) s.stage = ev.stage;
  if (ev.file != null) s.file = ev.file;
  if (ev.done != null || ev.total != null) {
    s.done = ev.done ?? null;
    s.total = ev.total ?? null;
  }
  s.bps = ev.bps ?? (ev.status === "running" ? s.bps : null);
  if (ev.status === "failed") s.error = ev.line ?? null;
  return progress;
}

const GIB = 1024 ** 3;
/// A size in the unit that reads best (as core's wizard::human_size).
export function gib(b, digits = 1) {
  const n = Number(b ?? 0);
  if (n >= GIB) return (n / GIB).toFixed(digits) + " GiB";
  if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + " MiB";
  return Math.ceil(n / 1024) + " KiB";
}
export const mbps = (bps) => (Number(bps ?? 0) / 1e6).toFixed(1) + " MB/s";
export function eta(secs) {
  const s = Math.max(0, Math.round(secs));
  if (s >= 3600) return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${s}s`;
}
export function params(n) {
  if (n == null) return "—";
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return Math.round(n / 1e6) + "M";
  return String(n);
}
export const shortSha = (s) => String(s ?? "").slice(0, 7);
export const day = (iso) => (iso ? String(iso).slice(0, 10) : "—");

/// A note's level as a chip class.
export const levelChip = (l) => (l === "error" ? "block" : l === "warning" ? "warn" : "note");
/// A fit verdict as a chip class and a word.
export const fitChip = (f) => ({ fits: "pass", tight: "warn", no_fit: "block", not_applicable: "plain" }[f] ?? "plain");
export const fitWord = (f) => ({ fits: "fits", tight: "tight", no_fit: "no", not_applicable: "—" }[f] ?? "—");

/// A build plan step's action, in a few words.
export function actionTitle(step) {
  const a = step?.action ?? {};
  switch (a.action) {
    case "use_installed": return `Use ${a.name}`;
    case "install_upstream": return `Install upstream ${a.tag} (prebuilt)`;
    case "install_unsloth": return `Install Unsloth ${a.tag} (${a.gfx})`;
    case "build_pr": return `Build pull request #${a.number} at ${shortSha(a.source?.sha)}`;
    case "build_fork": return `Build ${a.source?.label ?? `${a.owner}/${a.repo}`} @${shortSha(a.source?.sha)}`;
    case "unsupported": return "No build can load it";
    default: return a.action ?? "?";
  }
}

/// The source a fork or pull-request step builds, from the plan's sources.
export function sourceOf(step, sources) {
  const sha = step?.action?.source?.sha;
  if (!sha) return null;
  return (sources ?? []).find((s) => s.sha?.toLowerCase() === sha.toLowerCase()) ?? null;
}

/// Support of a build, in words.
export function supportText(s) {
  if (!s) return "not checked";
  if (s.support === "yes") return "can load it";
  if (s.support === "no") return "lacks " + (s.detail?.missing ?? []).map((m) => (m.kind === "arch" ? "the architecture" : m.kind === "tokenizer_pre" ? `pre-tokenizer '${m.value}'` : `tensor type ${m.value}`)).join(" and ");
  return "may not load it: " + (s.detail ?? "");
}
export const supportChip = (s) => (s?.support === "yes" ? "pass" : s?.support === "no" ? "block" : "warn");
