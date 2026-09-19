// The Models view's state and actions ("Get a model"). Module level, not
// component state: the app remounts the active view on every nav switch,
// and a search, a choice or a running download must still be there when
// the person comes back. The job itself runs in Rust; its events are
// heard here even while the view is not shown, and a slow poll of
// wizard_jobs keeps the steps right if an event was missed.
import { api, onEvent, log } from "../api.js";
import { applyWizardEvent } from "./wizard.js";

export const wz = $state({
  step: 1,
  // 1 Find
  query: "",
  allFormats: false,
  hits: null,           // null = not searched yet
  searched: "",         // the query the hits are for
  searching: false,
  findError: "",
  opening: "",          // the repo being inspected
  // 2 Choose file
  view: null,           // wizard_inspect result
  choice: null,         // a choice label
  mmproj: "",           // repo path, "" = none
  draft: "",
  ctx: null,            // null = the estimate's, at most 32768
  // 3 Build, 4 Get
  plan: null,           // wizard_plan result
  planKey: "",          // the picks (request) the plan was made for
  planning: false,
  planError: "",
  build: { kind: "auto" },
  consent: false,
  destRoot: null,
  roots: [],
  rootError: "",
  rootBusy: false,
  // the job
  job: null,            // wizard_jobs snapshot, kept current by events
  starting: false,
  jobError: "",
  // 5 Done
  check: null,          // live_check of the new profile
  checking: false,
  inited: false,
});

let poll = null;

export function init() {
  if (wz.inited) return;
  wz.inited = true;
  onEvent("wizard-progress", (ev) => {
    if (wz.job && ev.job === wz.job.job && !wz.job.finished) applyWizardEvent(wz.job.progress, ev);
  });
  onEvent("wizard-done", (d) => {
    if (wz.job && d.job === wz.job.job) finish(d);
  });
  reattach();
  loadRoots();
}

/// Pick up the latest job of this session: running, or finished and not
/// dismissed yet.
async function reattach() {
  try {
    const jobs = await api("wizard_jobs");
    const last = jobs.at(-1);
    if (!last || (wz.job && wz.job.job === last.job)) return;
    wz.job = last;
    wz.plan ??= last.plan;
    wz.consent = last.consent;
    if (last.finished) finish(last.finished, true);
    else {
      wz.step = 4;
      watch();
    }
  } catch (e) {
    log(`models: wizard_jobs failed: ${String(e)}`);
  }
}

function watch() {
  if (poll) return;
  poll = setInterval(sync, 2000);
}

function unwatch() {
  if (poll) clearInterval(poll);
  poll = null;
}

async function sync() {
  if (!wz.job || wz.job.finished) return unwatch();
  try {
    const j = (await api("wizard_jobs")).find((x) => x.job === wz.job?.job);
    if (!j) return unwatch();
    wz.job.progress = j.progress;
    wz.job.cancelling = j.cancelling;
    if (j.finished) finish(j.finished);
  } catch {
    /* the next tick tries again */
  }
}

function finish(done, quiet = false) {
  if (!wz.job) return;
  unwatch();
  wz.job.finished = done;
  wz.job.cancelling = false;
  if (done.ok) {
    wz.step = 5;
    runCheck();
  } else {
    wz.step = 4;
  }
  if (!quiet) loadRoots();
}

export async function loadRoots() {
  try {
    wz.roots = await api("wizard_roots");
    if (!wz.destRoot || !wz.roots.some((r) => r.path === wz.destRoot)) wz.destRoot = wz.roots[0]?.path ?? null;
  } catch (e) {
    wz.rootError = String(e);
  }
}

/// A repo id or a Hub link opens directly; anything else is a search.
export const looksLikeRepo = (q) =>
  /^\s*(https?:\/\/)?(www\.)?(huggingface\.co|hf\.co)\//i.test(q) || /^\s*[\w.-]+\/[\w.-]+\s*$/.test(q);

export async function submit() {
  const q = wz.query.trim();
  if (!q) return;
  if (looksLikeRepo(q)) return open(q);
  wz.searching = true;
  wz.findError = "";
  try {
    wz.hits = await api("hub_search", { query: q, limit: 40, all: wz.allFormats });
    wz.searched = q;
  } catch (e) {
    wz.findError = String(e);
  }
  wz.searching = false;
}

export async function open(input) {
  wz.opening = input;
  wz.findError = "";
  try {
    const v = await api("wizard_inspect", { input, rev: null });
    // Another repo: a finished job (done, failed or stopped) is dismissed,
    // so Get shows this repo's plan and its Start, not the old job and a
    // Resume that would run something else. A running job is never
    // dismissed (the steps cannot be left while one runs).
    if (wz.job?.finished) {
      await api("wizard_forget", { job: wz.job.job }).catch(() => {});
      wz.job = null;
      wz.jobError = "";
      wz.check = null;
    }
    wz.view = v;
    wz.choice = v.preselect ?? v.recommended ?? v.catalog?.choices?.[0]?.label ?? null;
    wz.mmproj = "";
    wz.draft = "";
    wz.ctx = null;
    wz.plan = null;
    wz.planKey = "";
    wz.planError = "";
    wz.build = { kind: "auto" };
    wz.consent = false;
    if (v.model_roots?.length) wz.roots = v.model_roots;
    if (!wz.destRoot || !wz.roots.some((r) => r.path === wz.destRoot)) wz.destRoot = wz.roots[0]?.path ?? null;
    wz.step = 2;
    log(`models: opened ${v.repo} (${v.kind?.kind}, ${v.catalog?.choices?.length ?? 0} choices)`);
  } catch (e) {
    wz.findError = String(e);
  }
  wz.opening = "";
}

function request() {
  return {
    choice: wz.choice,
    mmproj: wz.mmproj || null,
    draft: wz.draft || null,
    dest_root: wz.destRoot,
    build: $state.snapshot(wz.build),
    profile: true,
    ctx: wz.ctx ? Math.round(Number(wz.ctx)) : null,
  };
}

const requestKey = () => JSON.stringify(request());

/// The plan was made for the picks as they are now (the file, extras,
/// context, build and folder): a change on Choose file makes it stale.
export const planIsCurrent = () => !!wz.plan && !!wz.view && wz.plan.repo === wz.view.repo && wz.planKey === requestKey();

/// Plan again for the current picks. The consent given stays only while
/// it is consent to the same thing.
export async function makePlan() {
  if (!wz.view) return false;
  wz.planning = true;
  wz.planError = "";
  const before = wz.plan?.consent ?? null;
  const key = requestKey();
  try {
    // The view by its id: core plans from its own copy of it.
    const p = await api("wizard_plan", { viewId: wz.view.view_id, request: request() });
    if (p.consent !== before) wz.consent = false;
    wz.plan = p;
    wz.planKey = key;
  } catch (e) {
    wz.planError = String(e);
  }
  wz.planning = false;
  return !wz.planError;
}

/// The plan for the current picks: the one there is when it is current,
/// else a new one. False when planning failed (planError says why).
export async function ensurePlan() {
  return planIsCurrent() || (await makePlan());
}

export async function toBuild() {
  if (!wz.destRoot) {
    wz.planError = "Add a model folder to save into first (Save to, above).";
    return;
  }
  if (await makePlan()) wz.step = 3;
}

export function setBuild(choice) {
  wz.build = choice;
  makePlan();
}

export function setDest(path) {
  wz.destRoot = path;
  if (wz.planError.startsWith("Add a model folder")) wz.planError = "";
  // On Choose file there is no plan yet: Continue makes it.
  if (wz.plan) makePlan();
}

export async function addRoot(path) {
  const p = String(path ?? "").trim();
  if (!p) return false;
  wz.rootBusy = true;
  wz.rootError = "";
  try {
    wz.roots = await api("wizard_add_root", { path: p });
    // Compared without case or trailing separators: core keeps `E:\` for a
    // drive's root and trims them from any other folder.
    const key = (x) => String(x).replace(/[\\/]+$/, "").toLowerCase();
    const added = wz.roots.find((r) => key(r.path) === key(p));
    wz.rootBusy = false;
    if (added) setDest(added.path);
    return true;
  } catch (e) {
    wz.rootError = String(e);
  }
  wz.rootBusy = false;
  return false;
}

/// Start the plan. `asIs`: a finished job's own plan (resumed with no repo
/// view to plan again from); otherwise a plan made for other picks than
/// the ones shown is made again first and not started, so what starts is
/// always what the person looked at.
export async function start({ asIs = false } = {}) {
  if (!wz.plan) return;
  if (!asIs && !planIsCurrent()) {
    await makePlan();
    return;
  }
  wz.starting = true;
  wz.jobError = "";
  wz.check = null;
  try {
    // The plan by its id: core runs its own copy, never one sent back.
    const id = await api("wizard_start", { planId: wz.plan.plan_id, consent: !!wz.consent });
    const snap = (await api("wizard_jobs")).find((j) => j.job === id);
    wz.job = snap ?? { job: id, plan: $state.snapshot(wz.plan), progress: { steps: [], log: [] }, finished: null, cancelling: false, consent: wz.consent };
    wz.step = 4;
    if (wz.job.finished) finish(wz.job.finished);
    else watch();
    log(`models: started ${id} for ${wz.plan.repo} ${wz.plan.choice?.label}`);
  } catch (e) {
    wz.jobError = String(e);
  }
  wz.starting = false;
}

export async function cancel() {
  if (!wz.job || wz.job.finished) return;
  try {
    if (await api("wizard_cancel", { job: wz.job.job })) wz.job.cancelling = true;
  } catch (e) {
    wz.jobError = String(e);
  }
}

/// Try a failed or stopped job again: plan afresh (what is on disk now)
/// when the repo shown is the job's, then start. Downloads resume from
/// their .part files.
export async function resume() {
  const old = wz.job;
  if (!old?.finished) return;
  const same = !!wz.view && wz.view.repo === old.plan?.repo && wz.view.sha === old.plan?.sha;
  if (same) {
    if (!(await makePlan())) return;
    await api("wizard_forget", { job: old.job }).catch(() => {});
    wz.job = null;
    await start();
  } else if (old.plan) {
    // No view of the job's repo (the page reloaded): the job's own plan,
    // which core still holds by its id.
    await api("wizard_forget", { job: old.job }).catch(() => {});
    wz.job = null;
    wz.plan = old.plan;
    await start({ asIs: true });
  }
}

export async function runCheck() {
  const p = wz.job?.finished?.result?.profile;
  if (!p) return;
  wz.checking = true;
  try {
    wz.check = await api("live_check", { p });
  } catch (e) {
    wz.check = { error: String(e) };
  }
  wz.checking = false;
}

/// Back to a clean Find step; a finished job is dismissed.
export async function startOver() {
  if (wz.job && !wz.job.finished) return;
  if (wz.job) await api("wizard_forget", { job: wz.job.job }).catch(() => {});
  Object.assign(wz, {
    step: 1, view: null, choice: null, mmproj: "", draft: "", ctx: null, plan: null, planKey: "", planError: "",
    build: { kind: "auto" }, consent: false, job: null, jobError: "", check: null,
  });
}
