// Model output is untrusted, and this web view can call every FIDIM command
// (save a config, stop a server, install a build). A reply therefore goes
// through two independent filters before it reaches {@html}: markdown-it
// with raw HTML off (tags in the text come out escaped, and javascript:,
// vbscript:, file: and non-image data: links are refused), then DOMPurify
// with images, styles, forms and frames forbidden. Images never load (a
// reply could beacon out, and the CSP blocks them anyway): they render as
// links. Reasoning text never comes through here; it stays plain text.
import MarkdownIt from "markdown-it";
import DOMPurify from "dompurify";

const md = new MarkdownIt({ html: false, linkify: true, breaks: true, typographer: false });
const esc = md.utils.escapeHtml;

// ![alt](src): a link to the image, never an <img>.
md.renderer.rules.image = (tokens, idx, options, env, self) => {
  const t = tokens[idx];
  const alt = self.renderInlineAsText(t.children ?? [], options, env) || "image";
  return `<a href="${esc(t.attrGet("src") ?? "")}">${esc("[image: " + alt + "]")}</a>`;
};

// Fenced code gets a header with its language and a copy button (the
// click is handled by delegation in the message view).
md.renderer.rules.fence = (tokens, idx) => {
  const t = tokens[idx];
  const lang = (t.info || "").trim().split(/\s+/)[0];
  return `<div class="code"><div class="code-head"><span class="lang">${esc(lang || "text")}</span>` +
    `<button type="button" class="copy" data-copy-code>Copy</button></div><pre><code>${esc(t.content)}</code></pre></div>`;
};

const PURIFY = {
  FORBID_TAGS: ["img", "style", "form", "input", "iframe", "svg", "math"],
  FORBID_ATTR: ["style"],
};

// Links never open inside the app window: no target, no referrer.
DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") {
    node.removeAttribute("target");
    node.setAttribute("rel", "noopener noreferrer");
  }
});

/// Markdown text to sanitized HTML.
export function renderMarkdown(text) {
  return DOMPurify.sanitize(md.render(String(text ?? "")), PURIFY);
}

/// Whether the link can be opened in the browser (the opener plugin allows
/// http and https only).
export const openable = (href) => /^https?:\/\//i.test(String(href ?? ""));
