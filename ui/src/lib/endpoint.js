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

/// The shells the snippets are written for, in the order the card offers
/// them. On Windows `curl` in PowerShell 5.1 is Invoke-WebRequest, and cmd
/// has neither `\` continuations nor single quotes, so each gets its own.
export const SHELLS = [
  ["powershell", "PowerShell"],
  ["cmd", "cmd"],
  ["bash", "Git Bash / WSL"],
];

// Quoting. PowerShell and bash: single quotes, nothing inside expands.
// cmd: double quotes, and curl.exe reads `\"` as a quote (the C runtime's
// rule, which also doubles the backslashes before one).
const psq = (s) => `'${String(s).replace(/'/g, "''")}'`;
const shq = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`;
const cmdq = (s) => `"${String(s).replace(/(\\*)"/g, '$1$1\\"').replace(/(\\+)$/, "$1$1")}"`;

/// `run` is a run state ({host, port, alias}); `model` a router model id.
/// `hasKey`: the profile requires an API key (the key itself never reaches here).
export function endpointFor(run, model = null, hasKey = false) {
  const baseUrl = `http://${hostPort(run.host, run.port)}/v1`;
  const url = `${baseUrl}/chat/completions`;
  const modelId = model ?? run.alias;
  const key = hasKey ? "<the profile's API key>" : "sk-no-key";
  const body = JSON.stringify({ model: modelId, stream: true, messages: [{ role: "user", content: "Hello" }] });
  const hs = ["Content-Type: application/json", ...(hasKey ? [`Authorization: Bearer ${key}`] : [])];
  // -g: curl would read an IPv6 host's brackets as a URL pattern.
  const args = (q) => [`-N${url.includes("[") ? " -g" : ""} ${q(url)}`, ...hs.map((h) => `-H ${q(h)}`)];
  return {
    baseUrl,
    modelId,
    bindHost: run.host,
    loopback: isLoopback(run.host),
    wildcard: ["0.0.0.0", "::", "[::]"].includes(String(run.host ?? "").trim()),
    hasKey,
    // curl.exe, never PowerShell's curl alias; the body goes in on stdin,
    // since PowerShell 5.1 mangles quotes in a native command's arguments.
    curl: {
      powershell: `${psq(body)} | curl.exe ${args(psq).join(" ")} -d '@-'`,
      cmd: `curl.exe ${args(cmdq).join(" ")} -d ${cmdq(body)}`,
      bash: [`curl ${args(shq)[0]}`, ...args(shq).slice(1), `-d ${shq(body)}`].join(" \\\n  "),
    },
    env: {
      powershell: `$env:OPENAI_BASE_URL = ${psq(baseUrl)}\n$env:OPENAI_API_KEY = ${psq(key)}`,
      cmd: `set "OPENAI_BASE_URL=${baseUrl}"\nset "OPENAI_API_KEY=${key}"`,
      bash: `export OPENAI_BASE_URL=${shq(baseUrl)}\nexport OPENAI_API_KEY=${shq(key)}`,
    },
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
  };
}
