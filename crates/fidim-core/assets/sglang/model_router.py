#!/usr/bin/env python3
"""Model-name router: one OpenAI-compatible port in front of several SGLang servers,
with a llama-server-shaped live view for Llama FIDIM.

Routes every request by the JSON `model` field (falls back to DEFAULT), streams
responses through (SSE included), merges /v1/models, and answers /health only
when every backend is healthy. Aliases let "dd" and "ddg" hit the same backend.

Because every request and every streamed chunk pass through here, the router
also tracks per-model "slots" the way llama-server reports them, so FIDIM's
Running tab works unchanged against SGLang:

  GET /models                      llama-server router shape (status.value = loaded)
  GET /slots?model=X               per-slot phase, prompt/decode progress, text tails
  GET /metrics?model=X             llamacpp:tokens_predicted_total, llamacpp:predicted_tokens_seconds
  GET /fidim/state                 everything above as one JSON (for a custom page)

  python model_router.py --port 1234 --route dd=http://127.0.0.1:30000 \
      --route qwenMoE=http://127.0.0.1:30001 --alias ddg=dd --default dd
"""
import argparse, asyncio, json, logging, time
from aiohttp import web, ClientSession, ClientTimeout

log = logging.getLogger("router")
HOP = {"host", "content-length", "transfer-encoding", "connection", "keep-alive"}
TRACKED = ("/v1/chat/completions", "/v1/completions", "/generate")
TEXT_TAIL = 6000


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text")
    return ""


def _prompt_text(body):
    """Last user turn (or the raw prompt) and a token estimate for the whole request."""
    msgs = body.get("messages")
    if msgs:
        total = sum(len(_text_of(m.get("content"))) + len(json.dumps(m.get("tool_calls") or "")) for m in msgs)
        total += len(json.dumps(body.get("tools") or ""))
        last = next((_text_of(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"), "")
        return last, max(1, total // 4)
    p = body.get("prompt") or body.get("text") or ""
    p = p if isinstance(p, str) else json.dumps(p)
    return p, max(1, len(p) // 4)


class Slot:
    __slots__ = ("id", "id_task", "busy", "t0", "t_first", "t_end", "n_prompt", "n_prompt_exact",
                 "processed", "prompt", "generated", "n_decoded", "n_decoded_exact", "max_tokens", "gen_chars")

    def __init__(self, i):
        self.id = i; self.id_task = -1; self.busy = False; self.t0 = 0.0; self.t_first = 0.0; self.t_end = 0.0
        self.n_prompt = 0; self.n_prompt_exact = False; self.processed = 0; self.prompt = ""; self.generated = ""
        self.n_decoded = 0; self.n_decoded_exact = False; self.max_tokens = -1; self.gen_chars = 0

    def view(self, n_ctx):
        decoded = self.n_decoded if (self.n_decoded_exact or not self.busy) else max(self.n_decoded, self.gen_chars // 4)
        return {
            "id": self.id, "id_task": self.id_task, "n_ctx": n_ctx, "is_processing": self.busy,
            "speculative": True,
            "n_prompt_tokens": self.n_prompt, "n_prompt_tokens_processed": self.processed if self.busy and decoded == 0 else self.n_prompt,
            "n_prompt_tokens_cache": 0,
            "next_token": [{"has_next_token": self.busy, "n_decoded": decoded,
                            "n_remain": (self.max_tokens - decoded) if self.max_tokens > 0 else -1}],
            "prompt": self.prompt[-TEXT_TAIL:], "generated": self.generated[-TEXT_TAIL:],
        }


class ModelStats:
    def __init__(self, name, url):
        self.name = name; self.url = url; self.slots = []; self.n_ctx = 0; self.n_slots = 0
        self.tokens_predicted_total = 0; self.predicted_seconds = 0.0
        self.prompt_tokens_total = 0; self.prompt_seconds = 0.0; self.requests_total = 0
        self.task_counter = 0; self.info_at = 0.0

    async def ensure_info(self, session):
        if self.slots and time.time() - self.info_at < 60:
            return
        try:
            async with session.get(f"{self.url}/get_server_info", timeout=ClientTimeout(total=3)) as r:
                d = await r.json()
            # What one request can hold: the KV pool bounds it below the model's
            # nominal window (SGLang: max_req_input_len < context_length).
            self.n_ctx = int(d.get("max_req_input_len") or d.get("context_length") or d.get("max_total_num_tokens") or 0)
            want = int(d.get("max_running_requests") or 1)
        except Exception:  # noqa: BLE001
            want = max(1, self.n_slots or 1)
        while len(self.slots) < want:
            self.slots.append(Slot(len(self.slots)))
        self.n_slots = want; self.info_at = time.time()

    def take(self):
        free = [s for s in self.slots if not s.busy]
        s = free[0] if free else Slot(len(self.slots))
        if s not in self.slots:
            self.slots.append(s)
        self.task_counter += 1; self.requests_total += 1
        s.busy = True; s.id_task = self.task_counter; s.t0 = time.time(); s.t_first = 0.0; s.t_end = 0.0
        s.n_prompt = 0; s.n_prompt_exact = False; s.processed = 0; s.prompt = ""; s.generated = ""
        s.n_decoded = 0; s.n_decoded_exact = False; s.max_tokens = -1; s.gen_chars = 0
        return s

    def release(self, s):
        now = time.time()
        s.t_end = now; s.busy = False
        if not s.n_decoded_exact:
            s.n_decoded = max(s.n_decoded, s.gen_chars // 4)
        self.tokens_predicted_total += s.n_decoded
        if s.t_first:
            self.predicted_seconds += max(0.0, now - s.t_first)
            self.prompt_seconds += max(0.0, s.t_first - s.t0)
        self.prompt_tokens_total += s.n_prompt

    async def prefill_progress(self, session):
        """SGLang has no per-request prefill counter; with one request prefilling,
        the backend's pending-token count is that request's remaining prefill."""
        pre = [s for s in self.slots if s.busy and not s.t_first]
        if not pre:
            return
        try:
            async with session.get(f"{self.url}/get_load", timeout=ClientTimeout(total=2)) as r:
                load = await r.json()
            pending = sum(int(x.get("num_pending_tokens") or 0) for x in load)
        except Exception:  # noqa: BLE001
            return
        if len(pre) == 1:
            s = pre[0]; s.processed = max(s.processed, max(0, s.n_prompt - pending))
        else:
            for s in pre:
                s.processed = max(s.processed, int(s.n_prompt * 0.5))

    def metrics_text(self):
        # llama-server counts live: include what in-flight slots have produced so far,
        # so a 1 Hz sampler sees the rate while a request is still decoding.
        now = time.time()
        live_tokens = 0; live_secs = 0.0
        for s in self.slots:
            if s.busy and s.t_first:
                live_tokens += s.n_decoded if s.n_decoded_exact else max(s.n_decoded, s.gen_chars // 4)
                live_secs += now - s.t_first
        return "\n".join([
            "# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.",
            "# TYPE llamacpp:tokens_predicted_total counter",
            f"llamacpp:tokens_predicted_total {self.tokens_predicted_total + live_tokens}",
            "# TYPE llamacpp:predicted_tokens_seconds counter",
            f"llamacpp:predicted_tokens_seconds {self.predicted_seconds + live_secs:.3f}",
            "# TYPE llamacpp:prompt_tokens_total counter",
            f"llamacpp:prompt_tokens_total {self.prompt_tokens_total + sum(s.n_prompt for s in self.slots if s.busy and s.t_first)}",
            "# TYPE llamacpp:prompt_seconds_total counter",
            f"llamacpp:prompt_seconds_total {self.prompt_seconds:.3f}",
            "# TYPE llamacpp:requests_processing gauge",
            f"llamacpp:requests_processing {sum(1 for s in self.slots if s.busy)}",
            "# TYPE llamacpp:n_decode_total counter",
            f"llamacpp:n_decode_total {self.requests_total}",
            "",
        ])


class SSETracker:
    """Feed raw SSE bytes; updates the slot from OpenAI-style chunks and strips a
    usage-only chunk the client did not ask for."""

    def __init__(self, slot, want_usage):
        self.slot = slot; self.want_usage = want_usage; self.buf = b""

    def feed(self, chunk):
        self.buf += chunk
        out = []
        while b"\n\n" in self.buf:
            event, self.buf = self.buf.split(b"\n\n", 1)
            if self._consume(event):
                out.append(event + b"\n\n")
        return b"".join(out)

    def flush(self):
        rest, self.buf = self.buf, b""
        if rest and self._consume(rest):
            return rest
        return b""

    def _consume(self, event):
        s = self.slot
        for line in event.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                return True
            try:
                d = json.loads(payload)
            except Exception:  # noqa: BLE001
                return True
            usage = d.get("usage")
            choices = d.get("choices") or []
            for c in choices:
                delta = c.get("delta") or {}
                text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if not text and c.get("text"):
                    text = c["text"]
                if text:
                    if not s.t_first:
                        s.t_first = time.time(); s.processed = s.n_prompt
                    s.generated += text; s.gen_chars += len(text); s.n_decoded += 1
            if usage:
                if usage.get("prompt_tokens"):
                    s.n_prompt = int(usage["prompt_tokens"]); s.n_prompt_exact = True
                if usage.get("completion_tokens") is not None:
                    s.n_decoded = int(usage["completion_tokens"]); s.n_decoded_exact = True
                if not choices and not self.want_usage:
                    return False  # our injected usage chunk: keep it from the client
        return True


def make_app(routes: dict, aliases: dict, default: str):
    stats = {name: ModelStats(name, url) for name, url in routes.items()}
    healthy = {name: False for name in routes}

    async def health_poller(app):
        """Backend /health every 2 s into a cache: SGLang's /health can take ~2 s
        while a request is in flight, and FIDIM asks /models and /health at 1 Hz."""
        async with ClientSession(timeout=ClientTimeout(total=4)) as s:
            while True:
                for name, url in routes.items():
                    try:
                        async with s.get(f"{url}/health") as r:
                            healthy[name] = r.status == 200
                    except Exception:  # noqa: BLE001
                        healthy[name] = False
                await asyncio.sleep(2)

    async def start_poller(app):
        app["poller"] = asyncio.create_task(health_poller(app))

    async def stop_poller(app):
        app["poller"].cancel()

    def backend_for(model):
        m = aliases.get(model, model)
        return routes.get(m, routes[default]), (m if m in routes else default)

    async def models(request):
        data = []
        async with ClientSession(timeout=ClientTimeout(total=10)) as s:
            for name, url in routes.items():
                try:
                    async with s.get(f"{url}/v1/models") as r:
                        await stats[name].ensure_info(s)
                        # Real per-request limit (KV pool), not the model's nominal window;
                        # clients such as Hermes read n_ctx to size compaction.
                        ctx = {"n_ctx": stats[name].n_ctx, "context_length": stats[name].n_ctx} if stats[name].n_ctx else {}
                        for m in (await r.json()).get("data", []):
                            data.append({**m, **ctx, "id": name, "status": {"value": "loaded", "failed": False, "exit_code": None}})
                except Exception as e:  # noqa: BLE001
                    log.warning("backend %s /v1/models failed: %s", name, e)
                    data.append({"id": name, "object": "model", "owned_by": "router",
                                 "status": {"value": "unloaded", "failed": True, "exit_code": None}})
        for alias, target in aliases.items():
            ctx = {"n_ctx": stats[target].n_ctx, "context_length": stats[target].n_ctx} if stats[target].n_ctx else {}
            data.append({"id": alias, "object": "model", "owned_by": "router", "routed_to": target, **ctx,
                         "status": {"value": "loaded", "failed": False, "exit_code": None}})
        return web.json_response({"object": "list", "data": data})

    async def model(request):
        """/v1/models/{name}: the backend's entry (alias resolved) with the real per-request
        limit, since clients probe this path first."""
        name = request.match_info["name"]
        target = aliases.get(name, name)
        if target not in routes:
            return web.json_response({"error": {"message": f"model `{name}` is not served here", "type": "not_found"}}, status=404)
        out = {}
        async with ClientSession(timeout=ClientTimeout(total=10)) as s:
            try:
                async with s.get(f"{routes[target]}/v1/models/{target}") as r:
                    if r.status == 200:
                        out = await r.json()
            except Exception as e:  # noqa: BLE001
                log.warning("backend %s /v1/models/%s failed: %s", target, target, e)
            await stats[target].ensure_info(s)
        out = {**(out if isinstance(out, dict) else {}), "id": name, "object": out.get("object", "model") if isinstance(out, dict) else "model"}
        if target != name:
            out["routed_to"] = target
        if stats[target].n_ctx:
            out["n_ctx"] = stats[target].n_ctx; out["context_length"] = stats[target].n_ctx
        return web.json_response(out)

    async def router_models(request):
        """llama-server router-mode /models: only real members, with status.value (cached health)."""
        data = [{"id": name, "object": "model", "owned_by": "sglang",
                 "status": {"value": "loaded" if healthy[name] else "unloaded", "failed": not healthy[name], "exit_code": None}}
                for name in routes]
        return web.json_response({"object": "list", "data": data})

    async def health(request):
        bad = [name for name in routes if not healthy[name]]
        if bad:
            return web.Response(status=503, text=f"{', '.join(bad)} unhealthy")
        return web.Response(text="ok")

    def _stats_for(request):
        m = request.query.get("model")
        if m is None:
            return list(stats.values())
        m = aliases.get(m, m)
        return [stats[m]] if m in stats else []

    async def slots(request):
        out = []
        async with ClientSession() as s:
            for st in _stats_for(request):
                await st.ensure_info(s); await st.prefill_progress(s)
                out.extend(sl.view(st.n_ctx) for sl in st.slots)
        return web.json_response(out)

    backend_metrics = {}  # name -> (t, text)

    async def backend_metrics_text(st):
        """The backend's own Prometheus metrics (--enable-metrics), labels stripped and
        label groups aggregated (sums/counts/totals added, gauges last-wins), cached 1 s."""
        t, text = backend_metrics.get(st.name, (0.0, ""))
        if time.time() - t < 1.0:
            return text
        agg = {}
        try:
            async with ClientSession(timeout=ClientTimeout(total=1.5)) as s:
                async with s.get(f"{st.url}/metrics") as r:
                    raw = await r.text() if r.status == 200 else ""
        except Exception:  # noqa: BLE001
            raw = ""
        for line in raw.splitlines():
            if not line.startswith("sglang:") or "_bucket{" in line:
                continue
            name, _, rest = line.partition("{") if "{" in line.split(" ")[0] else (line.split(" ")[0], "", line)
            value = line.rsplit(" ", 1)[-1]
            try:
                v = float(value)
            except ValueError:
                continue
            if name.endswith(("_sum", "_count", "_total")):
                agg[name] = agg.get(name, 0.0) + v
            else:
                agg[name] = v
        text = "".join(f"{k} {v}\n" for k, v in agg.items())
        backend_metrics[st.name] = (time.time(), text)
        return text

    async def metrics(request):
        parts = []
        for st in _stats_for(request):
            parts.append(st.metrics_text())
            parts.append(await backend_metrics_text(st))
        return web.Response(text="".join(parts), content_type="text/plain")

    async def fidim_state(request):
        async with ClientSession() as s:
            for st in stats.values():
                await st.ensure_info(s); await st.prefill_progress(s)
        return web.json_response({name: {
            "url": st.url, "n_ctx": st.n_ctx, "slots": [sl.view(st.n_ctx) for sl in st.slots],
            "tokens_predicted_total": st.tokens_predicted_total, "predicted_seconds": st.predicted_seconds,
            "prompt_tokens_total": st.prompt_tokens_total, "prompt_seconds": st.prompt_seconds,
            "requests_total": st.requests_total} for name, st in stats.items()})

    async def proxy(request):
        body = await request.read()
        model = default; parsed = None
        if body:
            try:
                parsed = json.loads(body); model = parsed.get("model") or default
            except Exception:  # noqa: BLE001
                parsed = None
        url, resolved = backend_for(model)
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
        tracked = request.method == "POST" and request.path in TRACKED and isinstance(parsed, dict)
        slot = tracker = None
        if tracked:
            st = stats[resolved]
            async with ClientSession() as s0:
                await st.ensure_info(s0)
            slot = st.take()
            slot.prompt, slot.n_prompt = _prompt_text(parsed)
            slot.max_tokens = int(parsed.get("max_tokens") or parsed.get("max_completion_tokens") or -1)
            if parsed.get("stream"):
                so = dict(parsed.get("stream_options") or {})
                want_usage = bool(so.get("include_usage"))
                so["include_usage"] = True
                parsed = {**parsed, "stream_options": so}
                body = json.dumps(parsed).encode()
                tracker = SSETracker(slot, want_usage)
        try:
            async with ClientSession(timeout=ClientTimeout(total=None, sock_read=None)) as s:
                async with s.request(request.method, f"{url}{request.rel_url}", data=body, headers=headers) as r:
                    resp = web.StreamResponse(status=r.status, headers={k: v for k, v in r.headers.items() if k.lower() not in HOP})
                    await resp.prepare(request)
                    if tracker is not None and r.status == 200 and "text/event-stream" in (r.headers.get("content-type") or ""):
                        async for chunk in r.content.iter_any():
                            out = tracker.feed(chunk)
                            if out:
                                await resp.write(out)
                        tail = tracker.flush()
                        if tail:
                            await resp.write(tail)
                    else:
                        collected = bytearray()
                        async for chunk in r.content.iter_any():
                            if slot is not None and len(collected) < 4 * 1024 * 1024:
                                collected += chunk
                            await resp.write(chunk)
                        if slot is not None and r.status == 200:
                            try:
                                d = json.loads(bytes(collected))
                                ch = (d.get("choices") or [{}])[0]
                                msg = ch.get("message") or {}
                                slot.generated = (msg.get("reasoning_content") or "") + (msg.get("content") or ch.get("text") or d.get("text") or "")
                                slot.gen_chars = len(slot.generated)
                                # no stream: the whole wall time counts as decode (no prefill split)
                                slot.t_first = slot.t_first or slot.t0
                                u = d.get("usage") or {}
                                if u.get("completion_tokens") is not None:
                                    slot.n_decoded = int(u["completion_tokens"]); slot.n_decoded_exact = True
                                if u.get("prompt_tokens"):
                                    slot.n_prompt = int(u["prompt_tokens"]); slot.n_prompt_exact = True
                            except Exception:  # noqa: BLE001
                                pass
                    await resp.write_eof()
                    return resp
        finally:
            if slot is not None:
                stats[resolved].release(slot)

    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/v1/models/{name}", model)
    app.router.add_get("/models", router_models)
    app.router.add_get("/health", health)
    app.router.add_get("/slots", slots)
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/fidim/state", fidim_state)
    app.router.add_route("*", "/{tail:.*}", proxy)
    app.on_startup.append(start_poller)
    app.on_cleanup.append(stop_poller)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--route", action="append", default=[], help="model=http://host:port")
    ap.add_argument("--alias", action="append", default=[], help="alias=model")
    ap.add_argument("--default", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    routes = dict(r.split("=", 1) for r in a.route)
    aliases = dict(x.split("=", 1) for x in a.alias)
    web.run_app(make_app(routes, aliases, a.default), host=a.host, port=a.port, access_log=None)
