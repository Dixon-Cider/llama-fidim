// The chat's state and actions. Module level, not component state: the
// app remounts the active view on every nav switch ({#key active} in
// App.svelte), and a reply has to keep streaming, and a half-typed message
// keep its text, while the person looks at another tab.
//
// Conversations are saved to <config-dir>/chats by chat_save unless
// Settings turns saving off; then they live only here, until the app
// closes. A DiffusionGemma reply's denoise frames are memory-only either way.
import { api, stream, log } from "../api.js";

/// Sampler fields sent to llama-server when overridden, and the few a
/// DiffusionGemma run reads (it ignores sampling; its thinking is always on).
export const LLAMA_PARAMS = ["temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty", "dry_multiplier", "max_tokens", "seed", "stop"];
export const DG_PARAMS = ["max_tokens", "seed", "stop"];

export const chat = $state({
  targets: [],          // chat_targets: every run and router model the chat can use
  targetsError: "",
  targetsLoaded: false,
  list: [],             // saved conversations (chat_list summaries)
  convs: {},            // id -> conversation, for each one opened this session
  currentId: null,
  streams: {},          // conversation id -> the reply streaming into it
  props: {},            // target key -> chat_props (or { error })
  creator: {},          // model path -> creator_defaults (or { error })
  presets: {},          // profile id (or target key) -> { system_prompt, params, thinking }
  saving: true,         // Settings > Save chats
  saveError: "",
  frames: {},           // message id -> denoise frames for replay (never saved)
  drafts: {},           // conversation id -> unsent text
  pending: null,        // { run, model } chosen in Running before switching here
  inited: false,
});

const nowS = () => Math.floor(Date.now() / 1000);

/// Ids become file names in Rust: letters, digits, `-` and `_` only.
export function uid(prefix) {
  const r = new Uint8Array(6);
  crypto.getRandomValues(r);
  return `${prefix}-${Date.now().toString(36)}-${[...r].map((b) => b.toString(16).padStart(2, "0")).join("")}`;
}

export const targetKey = (t) => (t?.model ? `${t.run}/${t.model}` : t?.run ?? "");
export const presetKey = (t) => t?.profile_id ?? t?.key ?? "";
export const current = () => (chat.currentId ? chat.convs[chat.currentId] ?? null : null);
export const targetOf = (conv) => (conv ? chat.targets.find((t) => t.key === targetKey(conv.target)) ?? null : null);
export const isDiffusion = (conv) => (targetOf(conv)?.engine ?? conv?.target?.engine) === "diffusion-gemma";

function titleFrom(text) {
  const t = String(text).replace(/\s+/g, " ").trim();
  return t.length > 60 ? t.slice(0, 60) + "…" : t;
}

// ------------------------------------------------------------------ loading ----

/// Once per session: saving flag, presets, saved list. Every mount: targets.
export async function init() {
  if (!chat.inited) {
    chat.inited = true;
    try {
      const c = await api("get_config");
      chat.saving = c?.config?.save_chats !== false;
    } catch {
      // keep the default
    }
    try {
      chat.presets = (await api("chat_presets_get")) ?? {};
    } catch {
      chat.presets = {};
    }
    await refreshList();
  } else {
    // Settings may have changed the saving switch since.
    api("get_config").then((c) => (chat.saving = c?.config?.save_chats !== false)).catch(() => {});
  }
  await refreshTargets();
  consumePending();
}

// A slow server can make one poll outlast the refresh interval: never
// stack them.
let targetsBusy = null;
export function refreshTargets() {
  targetsBusy ??= (async () => {
    try {
      chat.targets = (await api("chat_targets")) ?? [];
      chat.targetsError = "";
    } catch (e) {
      chat.targetsError = String(e);
    }
    chat.targetsLoaded = true;
  })().finally(() => (targetsBusy = null));
  return targetsBusy;
}

export async function refreshList() {
  try {
    chat.list = (await api("chat_list")) ?? [];
  } catch (e) {
    chat.list = [];
    chat.saveError = String(e);
  }
}

/// The rail: saved conversations plus the ones only in memory (new, or
/// saving off), newest first.
export function rail() {
  const byId = new Map(chat.list.map((s) => [s.id, s]));
  for (const c of Object.values(chat.convs)) {
    byId.set(c.id, { id: c.id, title: c.title, created_unix: c.created_unix, updated_unix: c.updated_unix, target: c.target, n_messages: c.messages.length });
  }
  return [...byId.values()].sort((a, b) => b.updated_unix - a.updated_unix || (a.id < b.id ? -1 : 1));
}

/// Server facts for a target (context, default sampler, template caps).
/// A router model that is not loaded is skipped: asking could load it.
export async function loadProps(target, force = false) {
  if (!target || (!force && chat.props[target.key] && !chat.props[target.key].error)) return;
  if (target.status !== "loaded") return;
  try {
    chat.props[target.key] = await api("chat_props", { target: { run: target.run, model: target.model } });
  } catch (e) {
    chat.props[target.key] = { error: String(e) };
  }
}

export async function loadCreator(target) {
  const p = target?.model_path;
  if (!p) return;
  chat.creator[p] = { busy: true };
  try {
    chat.creator[p] = await api("creator_defaults", { modelPath: p });
  } catch (e) {
    chat.creator[p] = { error: String(e) };
  }
}

// ------------------------------------------------------------ conversations ----

/// Running's Chat button: remember the target, the view picks it up.
export function openTarget(run, model = null) {
  chat.pending = { run, model };
}

function consumePending() {
  const p = chat.pending;
  if (!p) {
    // First visit: continue the last conversation, else start one.
    if (!current()) {
      const first = rail()[0];
      if (first) openChat(first.id);
      else if (chat.targets.length) newChat(chat.targets[0]);
    }
    return;
  }
  chat.pending = null;
  const t = chat.targets.find((x) => x.key === targetKey(p));
  if (!t) return;
  const c = current();
  if (c && !c.messages.length) {
    setTarget(t);
    return;
  }
  newChat(t);
}

/// A conversation with no messages and no unsent text is a scratch pad:
/// only the one in view is kept, so New chat does not pile them up.
function pruneEmpty(keep) {
  for (const [id, c] of Object.entries(chat.convs)) {
    if (id === keep || c.messages.length || chat.streams[id] || chat.drafts[id]?.trim()) continue;
    if (chat.list.some((s) => s.id === id)) continue;
    delete chat.convs[id];
  }
}

export function newChat(target) {
  const preset = chat.presets[presetKey(target)] ?? {};
  const id = uid("c");
  chat.convs[id] = {
    schema: 1,
    id,
    title: "",
    created_unix: nowS(),
    updated_unix: nowS(),
    target: target ? { run: target.run, model: target.model ?? null, engine: target.engine } : null,
    system_prompt: preset.system_prompt ?? "",
    params: { ...(preset.params ?? {}) },
    thinking: preset.thinking ?? null,
    preserve_reasoning: false,
    messages: [],
  };
  chat.currentId = id;
  pruneEmpty(id);
  if (target) loadProps(target);
  return id;
}

export async function openChat(id) {
  if (!chat.convs[id]) {
    try {
      const c = await api("chat_load", { id });
      if (!c || typeof c !== "object" || !Array.isArray(c.messages)) throw new Error("not a conversation");
      c.params ??= {};
      c.thinking ??= null;
      c.system_prompt ??= "";
      c.title ??= "";
      chat.convs[id] = c;
    } catch (e) {
      chat.saveError = `could not open that conversation: ${String(e)}`;
      return;
    }
  }
  chat.currentId = id;
  pruneEmpty(id);
  const t = targetOf(chat.convs[id]);
  if (t) loadProps(t);
}

/// Point the current conversation at another server. The history goes
/// along: the next reply comes from the new target.
export function setTarget(target) {
  const c = current();
  if (!c || !target) return;
  c.target = { run: target.run, model: target.model ?? null, engine: target.engine };
  // An empty conversation takes the new target's preset.
  if (!c.messages.length) {
    const preset = chat.presets[presetKey(target)] ?? {};
    c.system_prompt = preset.system_prompt ?? "";
    c.params = { ...(preset.params ?? {}) };
    c.thinking = preset.thinking ?? null;
  }
  touch(c);
  loadProps(target);
}

export async function deleteChat(id) {
  if (chat.streams[id]) await stop(id);
  const wasSaved = chat.list.some((s) => s.id === id);
  delete chat.convs[id];
  delete chat.drafts[id];
  chat.list = chat.list.filter((s) => s.id !== id);
  if (wasSaved) {
    try {
      await api("chat_delete", { id });
    } catch (e) {
      chat.saveError = String(e);
    }
  }
  if (chat.currentId === id) {
    chat.currentId = null;
    const next = rail()[0];
    if (next) openChat(next.id);
  }
}

/// Settings' "Delete saved chats": every file goes; conversations still
/// streaming stay open in memory.
export async function deleteAllSaved() {
  const n = await api("chat_delete_all");
  for (const s of chat.list) {
    if (chat.streams[s.id]) continue;
    delete chat.convs[s.id];
    delete chat.drafts[s.id];
    if (chat.currentId === s.id) chat.currentId = null;
  }
  chat.list = [];
  return n;
}

// ------------------------------------------------------------------- saving ----

const saveTimers = {};

/// Something changed: save in about a second (and at once when a reply
/// ends). Settings do not move a conversation up the list; messages do.
function touch(conv, moved = false) {
  if (moved) conv.updated_unix = nowS();
  clearTimeout(saveTimers[conv.id]);
  saveTimers[conv.id] = setTimeout(() => saveNow(conv.id), 1000);
}

async function saveNow(id) {
  clearTimeout(saveTimers[id]);
  const conv = chat.convs[id];
  if (!conv || !conv.messages.length) return;
  const snap = $state.snapshot(conv);
  for (const m of snap.messages) delete m.pending;
  if (!chat.saving) return;
  try {
    const saved = await api("chat_save", { conv: snap });
    if (saved === false) {
      chat.saving = false;
      return;
    }
    const s = { id: snap.id, title: snap.title, created_unix: snap.created_unix, updated_unix: snap.updated_unix, target: snap.target, n_messages: snap.messages.length };
    chat.list = [s, ...chat.list.filter((x) => x.id !== snap.id)];
    chat.saveError = "";
  } catch (e) {
    chat.saveError = `not saved: ${String(e)}`;
  }
}

// ----------------------------------------------------------------- settings ----

export function setParam(k, v) {
  const c = current();
  if (!c) return;
  if (v === null || v === undefined || v === "" || (typeof v === "number" && !Number.isFinite(v))) delete c.params[k];
  else c.params[k] = v;
  touch(c);
}

export function setSystem(s) {
  const c = current();
  if (!c) return;
  c.system_prompt = s;
  touch(c);
}

/// null = the profile's setting, true/false = this conversation's.
export function setThinking(v) {
  const c = current();
  if (!c) return;
  c.thinking = v;
  touch(c);
}

export function setPreserveReasoning(v) {
  const c = current();
  if (!c) return;
  c.preserve_reasoning = !!v;
  touch(c);
}

/// Save the conversation's system prompt, overrides and thinking as the
/// defaults for new chats on this profile (kept apart from the profile).
export async function savePreset() {
  const c = current();
  const t = targetOf(c);
  if (!c || !t) return;
  chat.presets[presetKey(t)] = { system_prompt: c.system_prompt, params: $state.snapshot(c.params), thinking: c.thinking };
  await api("chat_presets_save", { presets: $state.snapshot(chat.presets) });
}

export function resetParams() {
  const c = current();
  if (!c) return;
  c.params = {};
  c.thinking = null;
  touch(c);
}

// --------------------------------------------------------------- streaming ----

/// What the server receives: the system prompt, the turns so far and only
/// the sampler fields this conversation overrides.
export function buildBody(conv, target) {
  const dg = target.engine === "diffusion-gemma";
  const preserve = conv.preserve_reasoning && !!chat.props[target.key]?.caps?.supports_preserve_reasoning;
  const messages = [];
  if (conv.system_prompt?.trim()) messages.push({ role: "system", content: conv.system_prompt });
  for (const m of conv.messages) {
    if (m.pending) continue;
    // A failed or empty reply is not a turn.
    if (m.role === "assistant" && !m.content) continue;
    const last = messages.at(-1);
    // Two user turns in a row (the reply between them failed) become one:
    // many chat templates insist that roles alternate.
    if (m.role === "user" && last?.role === "user") {
      last.content += "\n\n" + m.content;
      continue;
    }
    const x = { role: m.role, content: m.content };
    if (preserve && m.role === "assistant" && m.reasoning) x.reasoning_content = m.reasoning;
    messages.push(x);
  }
  const body = { messages };
  const p = conv.params ?? {};
  for (const k of dg ? DG_PARAMS : LLAMA_PARAMS) {
    const v = p[k];
    if (v === null || v === undefined || v === "") continue;
    if (k === "stop") {
      const list = (Array.isArray(v) ? v : String(v).split("\n")).map((s) => String(s)).filter((s) => s.length);
      if (list.length) body.stop = dg ? list.slice(0, 4) : list;
      continue;
    }
    body[k] = v;
  }
  if (!dg && conv.thinking !== null && conv.thinking !== undefined) body.chat_template_kwargs = { enable_thinking: !!conv.thinking };
  return body;
}

export async function send(text) {
  const conv = current();
  const t = String(text ?? "").trim();
  if (!conv || !t || chat.streams[conv.id]) return false;
  conv.messages.push({ id: uid("m"), role: "user", content: t, created_unix: nowS() });
  if (!conv.title) conv.title = titleFrom(t);
  conv.updated_unix = nowS();
  delete chat.drafts[conv.id];
  reply(conv);
  return true;
}

/// Run a reply into `conv` from its current history.
async function reply(conv) {
  const target = targetOf(conv);
  const convId = conv.id;
  conv.messages.push({ id: uid("m"), role: "assistant", content: "", reasoning: "", created_unix: nowS(), model: target?.model_id ?? null, pending: true });
  const m = conv.messages.at(-1);
  if (!target) {
    finalize(convId, m, null, { error: `${conv.target?.model ?? conv.target?.run ?? "that server"} is not running. Pick another target above.` });
    return;
  }
  const streamId = uid("s");
  chat.streams[convId] = {
    streamId,
    msgId: m.id,
    engine: target.engine,
    host: target.host,
    port: target.port,
    phase: target.status === "loaded" ? "connecting" : "loading",
    queue: null,
    progress: null,
    idTask: null,
    prefill: null,
    live: null,
    startedAt: performance.now(),
    reasoningAt: null,
    contentAt: null,
    finished: false,
  };
  const st = chat.streams[convId];
  const body = buildBody(conv, target);
  try {
    const sum = await stream("chat_send", { streamId, target: { run: target.run, model: target.model ?? null }, body }, (ev) => onEvent(convId, m, st, ev));
    // The command's answer and the channel's events travel separately:
    // give the final event a moment, then settle from the summary.
    setTimeout(() => finalize(convId, m, st, {
      finish: sum?.finish_reason, timings: sum?.timings, usage: sum?.usage,
      stopped: sum?.cancelled, error: sum?.error, status: sum?.status,
    }), 1500);
  } catch (e) {
    finalize(convId, m, st, { error: String(e) });
  }
}

function onEvent(convId, m, st, ev) {
  if (st.finished) return;
  const now = performance.now();
  switch (ev.kind) {
    case "open":
      if (st.phase === "connecting" || st.phase === "loading") st.phase = "waiting";
      break;
    case "queued":
      st.phase = "queued";
      st.queue = ev.position;
      break;
    case "task":
      st.idTask = ev.id_task;
      st.queue = null;
      if (st.phase === "queued" || st.phase === "waiting") st.phase = "prefill";
      m.dg = { ...(m.dg ?? {}), id_task: ev.id_task };
      break;
    case "progress":
      st.progress = ev;
      st.queue = null;
      if (st.phase !== "writing" || ev.stage === "denoise") st.phase = ev.stage;
      break;
    case "prefill":
      st.phase = "prefill";
      st.prefill = { processed: ev.processed, total: ev.total, cache: ev.cache };
      break;
    case "delta":
      if (ev.reasoning) {
        m.reasoning = (m.reasoning ?? "") + ev.reasoning;
        st.reasoningAt ??= now;
        if (st.phase !== "writing" && st.phase !== "denoise") st.phase = "thinking";
      }
      if (ev.content) {
        m.content += ev.content;
        if (st.contentAt === null) {
          st.contentAt = now;
          if (st.reasoningAt !== null) m.reasoning_ms = Math.round(now - st.reasoningAt);
        }
        if (st.phase !== "denoise") st.phase = "writing";
      }
      break;
    case "tool_calls":
      m.tool_calls = ev.calls;
      break;
    case "timings":
      st.live = ev.timings;
      break;
    case "done":
      finalize(convId, m, st, { finish: ev.finish_reason, timings: ev.timings, usage: ev.usage, model: ev.model });
      break;
    case "error":
      finalize(convId, m, st, { error: ev.message, status: ev.status });
      break;
    case "cancelled":
      finalize(convId, m, st, { stopped: true });
      break;
  }
}

/// End a reply once: record how it ended, free the conversation, save.
function finalize(convId, m, st, r) {
  if (st?.finished || (!st && !m.pending)) return;
  if (st) st.finished = true;
  delete m.pending;
  if (r.finish) m.finish = r.finish;
  if (r.timings) m.timings = r.timings;
  if (r.usage) m.usage = r.usage;
  if (r.model) m.model = r.model;
  if (r.stopped) m.stopped = true;
  if (r.error) {
    m.error = r.error;
    if (r.status) m.error_status = r.status;
  }
  if (!m.reasoning) delete m.reasoning;
  if (m.reasoning && st?.reasoningAt != null && m.reasoning_ms == null) m.reasoning_ms = Math.round(performance.now() - st.reasoningAt);
  if (st?.engine === "diffusion-gemma") {
    const seed = r.timings?.diffusion_seed;
    m.dg = { id_task: st.idTask ?? m.dg?.id_task ?? null, ...(seed != null ? { seed } : {}) };
    if (!r.error && !r.stopped && st.idTask != null) keepFrames(st, m.id, st.idTask);
  }
  if (chat.streams[convId] === st || (st && chat.streams[convId]?.streamId === st.streamId)) delete chat.streams[convId];
  const conv = chat.convs[convId];
  if (conv) {
    conv.updated_unix = nowS();
    saveNow(convId);
    // A router model that just loaded now has props to show.
    const t = targetOf(conv);
    if (t && !chat.props[t.key]?.loaded) {
      refreshTargets().then(() => loadProps(targetOf(conv), true));
    }
  }
}

/// A DiffusionGemma reply's denoise steps, fetched right after it ends and
/// kept only if they are this reply's (another client's request would
/// replace them on the server).
async function keepFrames(st, msgId, idTask) {
  try {
    const f = await api("dg_frames", { host: st.host, port: st.port });
    if (f?.id_task === idTask && f.frames?.length) chat.frames[msgId] = f;
  } catch (e) {
    log(`chat: could not keep the denoise frames: ${String(e)}`);
  }
}

export async function stop(convId = chat.currentId) {
  const st = chat.streams[convId];
  if (!st) return;
  st.stopping = true;
  try {
    await api("chat_cancel", { streamId: st.streamId });
  } catch {
    // the stream ends by itself
  }
}

/// Drop an assistant reply (and anything after it) and ask again.
export function regenerate(msgId) {
  const conv = current();
  if (!conv || chat.streams[conv.id]) return;
  const i = conv.messages.findIndex((m) => m.id === msgId);
  if (i < 0) return;
  const cut = conv.messages[i].role === "assistant" ? i : i + 1;
  for (const m of conv.messages.slice(cut)) delete chat.frames[m.id];
  conv.messages.splice(cut);
  reply(conv);
}

/// Change a user message, drop everything after it and ask again.
export function editAndResend(msgId, text) {
  const conv = current();
  const t = String(text ?? "").trim();
  if (!conv || !t || chat.streams[conv.id]) return;
  const i = conv.messages.findIndex((m) => m.id === msgId);
  if (i < 0 || conv.messages[i].role !== "user") return;
  conv.messages[i].content = t;
  for (const m of conv.messages.slice(i + 1)) delete chat.frames[m.id];
  conv.messages.splice(i + 1);
  if (i === 0) conv.title = titleFrom(t);
  reply(conv);
}

export function deleteMessage(msgId) {
  const conv = current();
  if (!conv) return;
  const st = chat.streams[conv.id];
  if (st?.msgId === msgId) return;
  conv.messages = conv.messages.filter((m) => m.id !== msgId);
  delete chat.frames[msgId];
  touch(conv, true);
}
