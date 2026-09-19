// How another program reaches a running server: the OpenAI base URL, the
// model id to send, and ready-to-paste snippets. Used by Running's Endpoint
// button and the chat's target bar. Mirrors fidim-core's chat::base_url: a
// server bound to 0.0.0.0 (every interface) is reached over loopback here.

export function connectHost(host) {
  const h = String(host ?? "").trim();
  if (h === "" || h === "0.0.0.0") return "127.0.0.1";
  if (h === "::" || h === "[::]") return "::1";
  return h;
}

export function hostPort(host, port) {
  const h = connectHost(host).replace(/^\[|\]$/g, "");
  return h.includes(":") ? `[${h}]:${port}` : `${h}:${port}`;
}

export function isLoopback(host) {
  const h = String(host ?? "").trim().replace(/^\[|\]$/g, "").toLowerCase();
  return h === "localhost" || h === "::1" || /^127\./.test(h);
}

/// `run` is a run state ({host, port, alias}); `model` a router model id.
/// `hasKey`: the profile sets --api-key (the key itself never reaches here).
export function endpointFor(run, model = null, hasKey = false) {
  const baseUrl = `http://${hostPort(run.host, run.port)}/v1`;
  const modelId = model ?? run.alias;
  const key = hasKey ? "<the profile's --api-key>" : "sk-no-key";
  const body = JSON.stringify({ model: modelId, stream: true, messages: [{ role: "user", content: "Hello" }] });
  const auth = hasKey ? ` \\\n  -H "Authorization: Bearer ${key}"` : "";
  return {
    baseUrl,
    modelId,
    bindHost: run.host,
    loopback: isLoopback(run.host),
    wildcard: ["0.0.0.0", "::", "[::]"].includes(String(run.host ?? "").trim()),
    hasKey,
    curl: `curl -N ${baseUrl}/chat/completions \\\n  -H "Content-Type: application/json"${auth} \\\n  -d '${body}'`,
    python: [
      "from openai import OpenAI",
      "",
      `client = OpenAI(base_url="${baseUrl}", api_key="${key}")`,
      `stream = client.chat.completions.create(`,
      `    model="${modelId}",`,
      `    messages=[{"role": "user", "content": "Hello"}],`,
      `    stream=True,`,
      `)`,
      `for chunk in stream:`,
      `    if chunk.choices:`,
      `        print(chunk.choices[0].delta.content or "", end="", flush=True)`,
    ].join("\n"),
    env: `OPENAI_BASE_URL=${baseUrl}\nOPENAI_API_KEY=${key}`,
  };
}
