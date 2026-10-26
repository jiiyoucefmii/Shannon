// Markdown subset + LaTeX (KaTeX) + code blocks (highlight.js) for Shannon's answers.
// Both libraries load from a CDN; offline, math shows as code and code is not coloured.
const esc = s => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function renderMath(tex, display) {
  if (window.katex) {
    try { return katex.renderToString(tex, { displayMode: display, throwOnError: false }); } catch {}
  }
  return display ? `<pre><code>${esc(tex)}</code></pre>` : `<code>${esc(tex)}</code>`;
}

function inline(s) {
  return s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?!\s)(.+?)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/(^|[^\w])_(?!\s)(.+?)_(?!\w)/g, "$1<em>$2</em>")
    .replace(/<em>\((not from the course material)\)<\/em>|\((not from the course material)\)/g,
             '<span class="unsourced" title="No course passage supports this sentence">not from the course material</span>')
    .replace(/\[(\d+)\]/g, '<button class="cite" data-n="$1">$1</button>');
}

function codeBlock(lang, code) {
  return `<div class="codeblock"><button class="copy">Copy</button><pre><code class="${lang ? "language-" + esc(lang) : ""}">${esc(code)}</code></pre></div>`;
}

function markdown(src, opts = {}) {
  const keep = [];
  const hold = html => `\u0000${keep.push(html) - 1}\u0000`;
  let s = String(src || "").replace(/```([\w+-]*)[^\n]*\n?([\s\S]*?)```/g, (_, l, c) => hold(codeBlock(l, c.replace(/\n$/, ""))))
    .replace(/\$\$([\s\S]+?)\$\$/g, (_, t) => hold(renderMath(t.trim(), true)))
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, t) => hold(renderMath(t.trim(), true)))
    .replace(/\\\((.+?)\\\)/g, (_, t) => hold(renderMath(t, false)))
    .replace(/\$([^\s$](?:[^$\n]*?[^\s$])?)\$/g, (_, t) => hold(renderMath(t, false)))
    .replace(/`([^`\n]+)`/g, (_, c) => hold(`<code>${esc(c)}</code>`));
  s = esc(s);
  const inl = opts.noCites ? x => inline(x).replace(/<button class="cite" data-n="(\d+)">\d+<\/button>/g, "[$1]") : inline;
  const out = [];
  let list = null;
  const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const block of s.split(/\n{2,}/)) {
    const lines = block.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i];
      let m;
      if ((m = l.match(/^\s*[-*•]\s+(.*)/))) { if (list !== "ul") { close(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inl(m[1])}</li>`); continue; }
      if ((m = l.match(/^\s*\d+[.)]\s+(.*)/))) { if (list !== "ol") { close(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inl(m[1])}</li>`); continue; }
      close();
      if ((m = l.match(/^#{1,6}\s+(.*)/))) { out.push(`<h4>${inl(m[1])}</h4>`); continue; }
      const para = [l];
      while (i + 1 < lines.length && !/^\s*([-*•]|\d+[.)]|#{1,6})\s+/.test(lines[i + 1])) para.push(lines[++i]);
      const text = para.join("<br>").trim();
      if (text) out.push(/^\u0000\d+\u0000$/.test(text) ? text : `<p>${inl(text)}</p>`);
    }
    close();
  }
  return out.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => keep[+i]);
}

// colour code blocks and wire their copy buttons; call after inserting markdown() output
function enhance(root) {
  root.querySelectorAll(".codeblock").forEach(cb => {
    const code = cb.querySelector("code");
    if (window.hljs && !code.dataset.hl) {
      try { hljs.highlightElement(code); } catch {}
      code.dataset.hl = 1;
    }
    cb.querySelector(".copy").onclick = () => {
      navigator.clipboard?.writeText(code.textContent);
      cb.querySelector(".copy").textContent = "Copied";
      setTimeout(() => cb.querySelector(".copy").textContent = "Copy", 1200);
    };
  });
}
