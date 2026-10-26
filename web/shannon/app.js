// Shannon core: helpers, navigation, student name, and the chat.
const $ = s => document.querySelector(s);
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el[k] = v;
    else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) el.append(k instanceof Node ? k : document.createTextNode(k));
  return el;
};
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};
const fmtDate = t => new Date(t * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
const LOGO = "logo.svg";
const spinner = text => h("div", { class: "spinner" }, h("img", { src: LOGO, alt: "" }), text);

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json" };
  const pin = sessionStorage.getItem("shannon.pin");
  if (pin) headers["X-Instructor-Pin"] = pin;
  const r = await fetch(path, { method: opts.body ? "POST" : "GET", headers,
                                body: opts.body ? JSON.stringify(opts.body) : undefined });
  const isJson = (r.headers.get("Content-Type") || "").includes("json");
  const d = isJson ? await r.json() : await r.blob();
  if (!r.ok) { const e = new Error((d && d.error) || `server error ${r.status}`); e.status = r.status; throw e; }
  return d;
}

/* ---------- student name ---------- */
let studentName = store.get("shannon.student", "");
function setWho() { $("#who").textContent = studentName || "guest"; }
function askName(force = false) {
  if (studentName && !force) return Promise.resolve(studentName);
  return new Promise(resolve => {
    const inp = h("input", { class: "inp", placeholder: "Your name", maxlength: 40, value: studentName });
    const bg = h("div", { class: "modal-bg" });
    const done = () => {
      const v = inp.value.trim();
      if (!v) { inp.focus(); return; }
      studentName = v; store.set("shannon.student", v); setWho(); bg.remove(); resolve(v);
    };
    inp.onkeydown = e => { if (e.key === "Enter") done(); };
    bg.append(h("div", { class: "modal" }, h("h3", {}, "What should Shannon call you?"),
      h("div", { class: "muted small" }, "Your name is attached to your answers so you (and your instructor) can follow your progress. It stays on this PC."),
      inp, h("div", { class: "row" }, h("span", { class: "spacer" }), h("button", { class: "btn primary", onclick: done }, "Continue"))));
    document.body.append(bg); inp.focus();
  });
}
$("#change-name").onclick = () => askName(true);
setWho();

/* ---------- navigation ---------- */
const VIEWS = { chat: null, practice: () => Learn.practice(), assign: () => Learn.assignments(),
                progress: () => Learn.progress(), instructor: () => Instructor.open() };
let view = "chat";
function go(v) {
  view = v;
  for (const k of Object.keys(VIEWS)) $("#view-" + k).hidden = k !== v;
  document.querySelectorAll(".side-btn[data-view]").forEach(b => b.classList.toggle("on", b.dataset.view === v));
  $("#top-title").textContent = v === "chat" ? (current && current.messages.length ? current.title : "") :
    { practice: "Practice", assign: "Assignments", progress: "My progress", instructor: "Instructor" }[v];
  if (VIEWS[v]) VIEWS[v]();
  if (v === "chat") renderChat();
  if (innerWidth < 760) $("#sidebar").classList.add("hidden");
}
document.querySelectorAll(".side-btn[data-view]").forEach(b => b.onclick = () => go(b.dataset.view));
$("#toggle-side").onclick = () => $("#sidebar").classList.toggle("hidden");
if (innerWidth < 760) $("#sidebar").classList.add("hidden");

/* ---------- chat ---------- */
const SUGGEST = ["Why do we divide by √d_k in attention?", "How does byte-pair encoding build its vocabulary?",
  "What is the difference between top-k and top-p sampling?", "Why do n-gram models need smoothing?",
  "Show me Python code for edit distance", "Quiz me on word embeddings"];
let chats = store.get("shannon.chats", []);
let current = null, busy = false, ready = false;
let style = store.get("shannon.style", "hint"), model = store.get("shannon.model", "tuned");
const saveChats = () => store.set("shannon.chats", chats.slice(0, 50));

function newChat() { current = null; go("chat"); $("#input").focus(); }
function openChat(id) { current = chats.find(c => c.id === id) || null; go("chat"); }
function deleteChat(id) { chats = chats.filter(c => c.id !== id); if (current && current.id === id) current = null; saveChats(); renderChat(); }

function renderSidebar() {
  const box = $("#chats");
  box.innerHTML = "";
  for (const c of chats) {
    box.append(h("div", { class: "chat-item" + (view === "chat" && current && c.id === current.id ? " active" : "") },
      h("button", { class: "open", onclick: () => openChat(c.id) }, c.title),
      h("button", { class: "del", title: "Delete chat", "aria-label": "Delete chat", onclick: () => deleteChat(c.id) }, "✕")));
  }
}

function attachCodeChecks(el, checks) {
  const blocks = el.querySelectorAll(".answer .codeblock");
  (checks || []).forEach((c, i) => {
    if (!c || !blocks[i]) return;
    const box = h("div", { class: "codecheck" });
    if (!c.syntax_ok) box.append(h("span", { class: "badge fail" }, "⚠ " + (c.warnings[0]?.text || "syntax error")));
    else if (!c.warnings.length) box.append(h("span", { class: "badge ok", title: "Checked for syntax errors, unsafe calls (eval, shell, pickle, secrets in code) and common style problems" }, "✓ Code checked: no security or style problems found"));
    else {
      box.append(h("span", { class: "badge warn" }, `⚠ Code check: ${c.warnings.length} note${c.warnings.length > 1 ? "s" : ""}`));
      for (const w of c.warnings) box.append(h("div", { class: "muted" }, `line ${w.line}: ${w.text}`));
    }
    blocks[i].after(box);
  });
}

function rateControls(m, meta, question) {
  const up = h("button", { class: "icon-btn" + (m.rating > 0 ? " on" : ""), title: "Helpful", "aria-label": "Helpful",
    html: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 10v11H3V10zM7 10l4-8a3 3 0 0 1 3 3v4h6a2 2 0 0 1 2 2l-2 8a2 2 0 0 1-2 2H7"/></svg>' });
  const down = h("button", { class: "icon-btn" + (m.rating < 0 ? " on" : ""), title: "Not helpful", "aria-label": "Not helpful",
    html: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 14V3h4v11zM17 14l-4 8a3 3 0 0 1-3-3v-4H4a2 2 0 0 1-2-2l2-8a2 2 0 0 1 2-2h11"/></svg>' });
  const send = (rating, comment = "") => api("/api/feedback", { body: { student: studentName, question, answer: m.content, rating, comment,
    style: m.stats?.style, model: m.stats?.model } }).catch(() => {});
  up.onclick = () => { if (m.rating) return; m.rating = 1; saveChats(); up.classList.add("on"); send(1); };
  down.onclick = () => {
    if (m.rating) return;
    m.rating = -1; saveChats(); down.classList.add("on");
    const inp = h("input", { class: "inp", placeholder: "What was unhelpful? (optional)" });
    const box = h("div", { class: "rate-box" }, inp, h("button", { class: "btn", onclick: () => { send(-1, inp.value.trim()); box.replaceWith(h("span", { class: "label" }, "Thanks for the feedback")); } }, "Send"));
    meta.append(box); inp.focus();
  };
  return [up, down];
}

function botMessage(m, question) {
  const el = h("div", { class: "msg bot" }, h("img", { class: "avatar", src: LOGO, alt: "" }));
  const body = h("div", { class: "body" });
  el.append(body);
  if (m.error) { body.append(h("div", { class: "error" }, m.error)); return el; }
  const ans = h("div", { class: "answer", html: markdown(m.content) });
  body.append(ans);
  enhance(ans);
  attachCodeChecks(el, m.code_checks);
  const cited = new Set([...m.content.matchAll(/\[(\d+)\]/g)].map(x => +x[1]));
  const srcs = (m.sources || []).filter(s => cited.has(s.n));
  if (srcs.length) {
    const box = h("div", { class: "sources" });
    const ex = h("div", { class: "excerpt", hidden: true });
    const show = n => {
      const s = srcs.find(x => x.n === n);
      const already = !ex.hidden && ex.dataset.n == n;
      el.querySelectorAll(".cite, .src").forEach(b => b.classList.toggle("on", !already && +b.dataset.n === n));
      if (already || !s) { ex.hidden = true; return; }
      ex.dataset.n = n; ex.hidden = false; ex.innerHTML = "";
      ex.append(h("div", { class: "h" }, `[${n}] ${s.title}${s.loc ? " · " + s.loc : ""}`), h("div", {}, s.text || ""));
    };
    for (const s of srcs)
      box.append(h("button", { class: "src", "data-n": s.n, title: s.title + (s.loc ? " · " + s.loc : ""), onclick: () => show(s.n) },
        h("span", { class: "n" }, s.n), h("span", { class: "t" }, s.title)));
    body.append(box, ex);
    ans.querySelectorAll(".cite").forEach(b => {
      const s = srcs.find(x => x.n === +b.dataset.n);
      if (s) b.title = s.title;
      b.onclick = () => show(+b.dataset.n);
    });
  } else ans.querySelectorAll(".cite").forEach(b => b.replaceWith(document.createTextNode(`[${b.dataset.n}]`)));
  const st = m.stats || {};
  const meta = h("div", { class: "meta" });
  const copy = h("button", { class: "icon-btn", title: "Copy answer", "aria-label": "Copy answer", onclick: () => navigator.clipboard?.writeText(m.content),
    html: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>' });
  meta.append(copy, ...rateControls(m, meta, question),
    h("span", { class: "label" }, [st.style === "explain" ? "Explain" : "Hint", st.model === "base" ? "base model" : "fine-tuned",
      st.seconds ? `${st.seconds}s` : ""].filter(Boolean).join(" · ")));
  body.append(meta);
  return el;
}

function renderChat() {
  renderSidebar();
  const has = current && current.messages.length;
  $("#welcome").hidden = !!has;
  $("#scroll").hidden = !has;
  if (view === "chat") $("#top-title").textContent = has ? current.title : "";
  (has ? $("#view-chat") : $("#composer-slot-welcome")).appendChild($("#composer-wrap"));
  const t = $("#thread");
  t.innerHTML = "";
  if (!has) return;
  let lastQ = "";
  for (const m of current.messages) {
    if (m.role === "user") { lastQ = m.content; t.append(h("div", { class: "msg user" }, h("div", { class: "bubble" }, m.content))); }
    else t.append(botMessage(m, lastQ));
  }
  if (busy) t.append(h("div", { class: "msg bot" }, h("div", { class: "thinking" }, h("img", { class: "avatar", src: LOGO, alt: "", width: 28, height: 28 }), "Shannon is thinking…")));
  $("#scroll").scrollTop = $("#scroll").scrollHeight;
}

async function send(text) {
  text = (text ?? $("#input").value).trim();
  if (!text || busy || !ready) return;
  if (view !== "chat") go("chat");
  if (!current) {
    current = { id: Date.now().toString(36), title: text.length > 48 ? text.slice(0, 46) + "…" : text, messages: [] };
    chats.unshift(current);
  } else chats = [current, ...chats.filter(c => c !== current)];
  current.messages.push({ role: "user", content: text });
  $("#input").value = ""; grow(); busy = true; updateSend(); renderChat(); saveChats();
  const chat = current;
  try {
    const d = await api("/api/chat", { body: { style, base: model === "base",
      messages: chat.messages.filter(m => !m.error).map(m => ({ role: m.role, content: m.content })) } });
    chat.messages.push({ role: "assistant", content: d.answer, sources: d.sources, stats: d.stats, code_checks: d.code_checks });
  } catch (e) {
    chat.messages.push({ role: "assistant", content: "", error: "Shannon couldn't answer: " + e.message + ". Is the Shannon window (the server) still running?" });
  }
  busy = false; updateSend(); saveChats(); renderChat(); $("#input").focus();
}
// open a new chat about something (used by "Discuss with Shannon" on results)
function discuss(text) { current = null; go("chat"); send(text); }

function grow() { const i = $("#input"); i.style.height = "auto"; i.style.height = Math.min(i.scrollHeight, 220) + "px"; updateSend(); }
function updateSend() { $("#send").disabled = busy || !ready || !$("#input").value.trim(); }
$("#input").addEventListener("input", grow);
$("#input").addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
$("#send").onclick = () => send();
$("#new-chat").onclick = newChat;
document.querySelectorAll("#style button").forEach(b => {
  b.classList.toggle("on", b.dataset.v === style);
  b.onclick = () => { style = b.dataset.v; store.set("shannon.style", style);
    document.querySelectorAll("#style button").forEach(x => x.classList.toggle("on", x === b)); };
});
$("#model").value = model;
$("#model").onchange = e => { model = e.target.value; store.set("shannon.model", model); };
for (const q of SUGGEST) $("#chips").append(h("button", { class: "chip", onclick: () => send(q) }, q));
const hr = new Date().getHours();
$("#greet").textContent = hr < 5 ? "Studying late? Let's learn." : hr < 12 ? "Good morning. What are we learning?" :
  hr < 18 ? "Good afternoon. What are we learning?" : "Good evening. What are we learning?";

let status = {};
async function poll() {
  try {
    status = await api("/api/status");
    if (status.error) { $("#loading-msg").textContent = "Shannon couldn't start: " + status.error; return; }
    if (status.ready) {
      ready = true; $("#loading").classList.add("gone"); updateSend();
      if (!status.adapter) { $("#model").value = "base"; $("#model").disabled = true; model = "base"; }
      $("#model-name").textContent = "Qwen3.5-4B" + (status.adapter ? ` + ${status.adapter}` : "");
      return;
    }
  } catch { $("#loading-msg").textContent = "Can't reach the Shannon server. Start it with Shannon.bat."; }
  setTimeout(poll, 1500);
}
document.addEventListener("DOMContentLoaded", () => { go("chat"); poll(); });
addEventListener("load", () => { if (view === "chat") renderChat(); });  // KaTeX/highlight.js arrive after this script
