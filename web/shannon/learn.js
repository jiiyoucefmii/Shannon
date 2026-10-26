// Student pages: practice (questions generated on demand), assignments (instructor-published), progress.
const KIND_LABEL = { mcq: "Multiple choice", short: "Short answer", code: "Coding" };
const LEVEL_LABEL = { easy: "Easy", medium: "Medium", hard: "Hard" };
const TOPICS = ["n-gram language models", "smoothing and Kneser-Ney", "perplexity", "minimum edit distance", "byte-pair encoding",
  "word embeddings and word2vec", "TF-IDF and cosine similarity", "logistic regression for text", "naive Bayes classification",
  "self-attention", "transformers", "decoding: greedy, beam, top-k, top-p", "fine-tuning and LoRA", "RLHF and DPO",
  "retrieval-augmented generation", "named entity recognition", "part-of-speech tagging and HMMs", "evaluation metrics (BLEU, F1)"];

function seg(options, value, onChange) {
  const el = h("div", { class: "seg" });
  for (const [v, label] of Object.entries(options)) {
    const b = h("button", { class: v === value ? "on" : "", onclick: () => {
      el.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b)); onChange(v); } }, label);
    el.append(b);
  }
  return el;
}

function codeEditor(value) {
  const ta = h("textarea", { class: "editor", spellcheck: "false", autocapitalize: "off" });
  ta.value = value || "";
  ta.addEventListener("keydown", e => {  // Tab indents instead of leaving the box
    if (e.key === "Tab") {
      e.preventDefault();
      const s = ta.selectionStart, en = ta.selectionEnd;
      ta.value = ta.value.slice(0, s) + "    " + ta.value.slice(en);
      ta.selectionStart = ta.selectionEnd = s + 4;
    }
  });
  return ta;
}

function feedbackBox(text) {
  return h("div", { class: "feedback" }, h("div", { class: "who" }, h("img", { src: LOGO, alt: "" }), "Shannon's feedback"),
           h("div", { html: markdown((text || "").replace(/\s*\[\d+\](\[\d+\])*/g, ""), { noCites: true }) }));
}

// marks breakdown for any graded answer; also used by the instructor's submission view
function renderResult(kind, score, max, r, opts = {}) {
  const box = h("div", { class: "result" });
  const pc = max ? score / max : 0;
  box.append(h("div", { class: "score" }, `${+score.toFixed(2)}`, h("small", {}, `/ ${max} marks`),
    h("span", { class: "badge " + (pc >= 0.8 ? "ok" : pc >= 0.5 ? "warn" : "fail") }, `${Math.round(pc * 100)}%`)));
  if (opts.exam) box.append(h("div", { class: "muted small" }, "This is an exam: the detailed breakdown is shown to your instructor only."));
  if (kind === "mcq" && r.correct !== undefined)
    box.append(h("div", { class: "small" }, r.correct ? "Correct." : `Not quite: the answer is ${"ABCD"[r.key]}.`));
  if (r.tests && r.tests.length) {
    const passed = r.tests.filter(t => t.ok).length;
    box.append(h("h3", { class: "small", style: "margin:10px 0 0" }, `Tests: ${passed} of ${r.tests.length} passed`));
    box.append(h("ul", { class: "checklist" }, r.tests.map(t => h("li", {}, h("span", { class: "tick " + (t.ok ? "y" : "n") }, t.ok ? "✓" : "✗"),
      h("div", {}, h("code", {}, t.test), t.ok ? null : h("div", { class: "err" }, t.error))))));
  }
  if (r.static && (r.static.blocked.length || r.static.warnings.length)) {
    box.append(h("div", { class: "notes" }, r.static.blocked.map(([l, t]) => h("div", { class: "note fail" }, `Line ${l}: ${t}`)),
      r.static.warnings.map(([l, t]) => h("div", { class: "note warn" }, `Line ${l}: ${t}`))));
  }
  if (r.points && r.points.length) {
    box.append(h("h3", { class: "small", style: "margin:10px 0 0" }, kind === "code" ? "Your explanation" : "Marking points"));
    box.append(h("ul", { class: "checklist" }, r.points.map(p => h("li", {}, h("span", { class: "tick " + (p.ok ? "y" : "n") }, p.ok ? "✓" : "✗"),
      h("div", {}, p.point, h("span", { class: "muted small" }, `  (${p.marks} mark${p.marks === 1 ? "" : "s"})`))))));
  }
  if (r.feedback) box.append(feedbackBox(r.feedback));
  return box;
}

// a question the student can answer; onSubmitted(result) after marking
function questionCard(q, { assessmentId = null, done = null, extraButtons = [] } = {}) {
  const card = h("div", { class: "qcard" });
  card.append(h("div", { class: "qhead" }, h("span", { class: "badge accent" }, KIND_LABEL[q.kind]),
    h("span", { class: "badge" }, LEVEL_LABEL[q.difficulty] || q.difficulty), h("span", { class: "badge" }, q.topic),
    h("span", { class: "marks" }, `${q.max_score} mark${q.max_score === 1 ? "" : "s"}`)));
  const qt = h("div", { class: "q", html: markdown(q.question, { noCites: true }) });
  card.append(qt); enhance(qt);
  let answer = () => ({});
  if (q.kind === "mcq") {
    let choice = null;
    const opts = h("div", { class: "options" }, q.options.map((o, i) => {
      const b = h("button", { class: "opt", onclick: () => { choice = i; opts.querySelectorAll(".opt").forEach((x, j) => x.classList.toggle("sel", j === i)); } },
        h("span", { class: "l" }, "ABCD"[i]), h("span", { html: markdown(o, { noCites: true }).replace(/^<p>|<\/p>$/g, "") }));
      return b;
    }));
    card.append(opts);
    answer = () => ({ choice });
  } else if (q.kind === "short") {
    const ta = h("textarea", { class: "inp", placeholder: "Write your answer in your own words…", rows: 6 });
    card.append(ta);
    answer = () => ({ text: ta.value });
  } else {
    const ed = codeEditor(q.starter);
    const ex = h("textarea", { class: "inp", rows: 3, placeholder: "Explain your approach in 2-3 sentences (this is marked too)…" });
    card.append(ed, h("div", { class: "muted small", style: "margin:8px 0 4px" },
      `${q.n_tests} tests will run on your function. Only the standard library is allowed; no files, input() or eval().`), ex);
    if (q.hint) card.append(h("details", { class: "more" }, h("summary", {}, "Show a hint"), h("div", { class: "small", style: "margin-top:4px" }, q.hint)));
    answer = () => ({ code: ed.value, explanation: ex.value });
  }
  if (q.sources && q.sources.length)
    card.append(h("details", { class: "more" }, h("summary", {}, "Written from these course sources"),
      h("ul", { class: "small muted" }, q.sources.map(s => h("li", {}, s.title + (s.loc ? " · " + s.loc : ""))))));
  const status = h("div");
  const submit = h("button", { class: "btn primary" }, "Submit answer");
  const bar = h("div", { class: "row", style: "margin-top:12px" }, submit, ...extraButtons, status);
  card.append(bar);
  submit.onclick = async () => {
    const a = answer();
    if (q.kind === "mcq" && a.choice == null) { status.replaceChildren(h("span", { class: "muted small" }, "Pick an option first.")); return; }
    if (q.kind === "short" && !a.text.trim()) { status.replaceChildren(h("span", { class: "muted small" }, "Write an answer first.")); return; }
    const name = await askName();
    submit.disabled = true;
    status.replaceChildren(spinner(q.kind === "mcq" ? "Marking…" : "Shannon is marking your answer (about 20 seconds)…"));
    try {
      const d = await api("/api/submit", { body: { student: name, item_id: q.id, assessment_id: assessmentId, answer: a } });
      status.replaceChildren();
      if (q.kind === "mcq" && d.result.key !== undefined)
        card.querySelectorAll(".opt").forEach((x, j) => { x.disabled = true; x.classList.toggle("right", j === d.result.key);
          x.classList.toggle("wrong", j === a.choice && j !== d.result.key); });
      card.querySelector(".result")?.remove();
      bar.before(renderResult(q.kind, d.score, d.max_score, d.result, { exam: d.exam }));
      submit.textContent = d.exam ? "Submitted" : "Submit again";
      submit.disabled = d.exam || q.kind === "mcq";
      done && done(d, a);
    } catch (e) {
      submit.disabled = false;
      status.replaceChildren(h("span", { class: "error small" }, e.message));
    }
  };
  return card;
}

function discussButton(q, getAnswer) {
  return h("button", { class: "btn", onclick: () => {
    const a = getAnswer ? getAnswer() : "";
    discuss(`I'm practising this question: "${q.question}"${a ? `\n\nMy answer was: ${a}` : ""}\n\nCan you help me understand it?`);
  } }, "Discuss with Shannon");
}

const Learn = {
  settings: store.get("shannon.practice", { topic: "", kind: "mcq", difficulty: "medium" }),

  practice() {
    const page = $("#practice-page");
    if (page.dataset.built) return;
    page.dataset.built = 1;
    const s = this.settings;
    const save = () => store.set("shannon.practice", s);
    const topic = h("input", { class: "inp grow", list: "topic-list", placeholder: "Topic, e.g. Kneser-Ney smoothing", value: s.topic });
    topic.oninput = () => { s.topic = topic.value; save(); };
    const area = h("div");
    const gen = h("button", { class: "btn primary" }, "New question");
    const run = async () => {
      if (!topic.value.trim()) { topic.focus(); return; }
      const name = await askName();
      gen.disabled = true;
      area.replaceChildren(h("div", { class: "card" }, spinner(`Writing a ${LEVEL_LABEL[s.difficulty].toLowerCase()} ${KIND_LABEL[s.kind].toLowerCase()} question from the course material and checking it… (${s.kind === "code" ? "about 2 minutes" : "about 1-2 minutes"})`)));
      try {
        const q = await api("/api/practice", { body: { student: name, topic: topic.value.trim(), kind: s.kind, difficulty: s.difficulty } });
        let lastAnswer = "";
        const card = questionCard(q, { done: (d, a) => { lastAnswer = a.text || a.code || (a.choice != null ? "ABCD"[a.choice] : ""); },
          extraButtons: [h("button", { class: "btn", onclick: run }, "Next question"), discussButton(q, () => lastAnswer)] });
        area.replaceChildren(card);
      } catch (e) { area.replaceChildren(h("div", { class: "error" }, e.message)); }
      gen.disabled = false;
    };
    gen.onclick = run;
    topic.onkeydown = e => { if (e.key === "Enter") run(); };
    page.append(h("h2", {}, "Practice"),
      h("p", { class: "lead" }, "Shannon writes a fresh question from the course material, checks it, and marks your answer straight away."),
      h("div", { class: "card" },
        h("div", { class: "row" }, topic, gen),
        h("div", { class: "row", style: "margin-top:10px" },
          seg(KIND_LABEL, s.kind, v => { s.kind = v; save(); }), seg(LEVEL_LABEL, s.difficulty, v => { s.difficulty = v; save(); }))),
      h("datalist", { id: "topic-list" }, TOPICS.map(t => h("option", { value: t }))), area);
  },

  async assignments() {
    const page = $("#assign-page");
    page.replaceChildren(h("h2", {}, "Assignments"), h("p", { class: "lead" }, "Quizzes and exams published by your instructor."), spinner("Loading…"));
    try {
      const [list, hist] = await Promise.all([api("/api/assessments"), studentName ? api("/api/history?student=" + encodeURIComponent(studentName)) : []]);
      page.lastChild.remove();
      if (!list.length) { page.append(h("div", { class: "card muted" }, "Nothing has been published yet.")); return; }
      for (const a of list) {
        const mine = hist.filter(s => s.assessment_id === a.id);
        const done = new Set(mine.map(s => s.item_id));
        const scored = a.items.map(i => mine.filter(s => s.item_id === i.id).sort((x, y) => y.score - x.score)[0]).filter(Boolean);
        const total = a.items.reduce((t, i) => t + i.max_score, 0);
        const got = scored.reduce((t, s) => t + (s.instructor_score ?? s.score), 0);
        const body = h("div", { hidden: true });
        const head = h("div", { class: "card", style: "cursor:pointer" },
          h("div", { class: "row" }, h("h3", { style: "margin:0" }, a.title), h("span", { class: "badge " + (a.kind === "exam" ? "warn" : "accent") }, a.kind === "exam" ? "Exam" : "Quiz"),
            h("span", { class: "spacer" }), h("span", { class: "muted small" }, `${done.size}/${a.items.length} answered` + (done.size ? ` · ${+got.toFixed(1)}/${total} marks` : ""))),
          a.kind === "exam" ? h("div", { class: "muted small" }, "One attempt per question. Results are reviewed by your instructor.") : null);
        head.onclick = () => {
          body.hidden = !body.hidden;
          if (!body.childElementCount)
            a.items.forEach((q, i) => {
              const prev = mine.find(s => s.item_id === q.id);
              if (a.kind === "exam" && prev) {
                body.append(h("div", { class: "qcard" }, h("div", { class: "qhead" }, h("span", { class: "badge accent" }, `Question ${i + 1}`),
                  h("span", { class: "badge ok" }, "Submitted"), h("span", { class: "marks" }, `${prev.instructor_score ?? prev.score}/${prev.max_score} marks` + (prev.instructor_score != null ? " (reviewed)" : ""))),
                  h("div", { class: "q", html: markdown(q.question, { noCites: true }) }),
                  prev.instructor_comment ? feedbackBox(prev.instructor_comment) : null));
              } else body.append(questionCard(q, { assessmentId: a.id }));
            });
        };
        page.append(head, body);
      }
    } catch (e) { page.lastChild.replaceWith(h("div", { class: "error" }, e.message)); }
  },

  async progress() {
    const page = $("#progress-page");
    page.replaceChildren(h("h2", {}, "My progress"));
    if (!studentName) { page.append(h("div", { class: "card" }, "Answer a practice question first, or ", h("button", { class: "linkish", onclick: async () => { await askName(); this.progress(); } }, "set your name"), ".")); return; }
    page.append(spinner("Loading…"));
    try {
      const hist = await api("/api/history?student=" + encodeURIComponent(studentName));
      page.lastChild.remove();
      if (!hist.length) { page.append(h("div", { class: "card muted" }, "No answers yet. Try the Practice page.")); return; }
      const pc = s => s.max_score ? 100 * (s.instructor_score ?? s.score) / s.max_score : 0;
      const avg = hist.reduce((t, s) => t + pc(s), 0) / hist.length;
      const topics = {};
      for (const s of [...hist].reverse()) (topics[s.topic.toLowerCase()] ||= []).push(s);
      page.append(h("div", { class: "stats" },
        h("div", { class: "stat" }, h("div", { class: "v" }, hist.length), h("div", { class: "l" }, "questions answered")),
        h("div", { class: "stat" }, h("div", { class: "v" }, Math.round(avg) + "%"), h("div", { class: "l" }, "average mark")),
        h("div", { class: "stat" }, h("div", { class: "v" }, Object.keys(topics).length), h("div", { class: "l" }, "topics practised"))));
      page.append(h("div", { class: "card" }, h("h3", {}, "By topic"), h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" },
        h("tr", {}, h("th", {}, "Topic"), h("th", {}, "Answers"), h("th", {}, "First"), h("th", {}, "Latest"), h("th", {}, "Average")),
        Object.entries(topics).map(([t, ss]) => {
          const a = ss.reduce((x, s) => x + pc(s), 0) / ss.length;
          return h("tr", {}, h("td", {}, t), h("td", {}, ss.length), h("td", {}, Math.round(pc(ss[0])) + "%"), h("td", {}, Math.round(pc(ss.at(-1))) + "%"),
            h("td", {}, h("div", { class: "row" }, h("div", { class: "bar grow", style: "min-width:80px" }, h("i", { style: `width:${a}%` })), Math.round(a) + "%")));
        })))));
      page.append(h("div", { class: "card" }, h("h3", {}, "Recent answers"), h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" },
        h("tr", {}, h("th", {}, "When"), h("th", {}, "Question"), h("th", {}, "Type"), h("th", {}, "Mark")),
        hist.slice(0, 40).map(s => h("tr", {}, h("td", { class: "muted" }, fmtDate(s.created)), h("td", {}, s.question.slice(0, 110) + (s.question.length > 110 ? "…" : ""),
          s.instructor_comment ? h("div", { class: "muted small" }, "Instructor: " + s.instructor_comment) : null),
          h("td", {}, KIND_LABEL[s.kind] || s.kind), h("td", {}, `${+(s.instructor_score ?? s.score).toFixed(2)}/${s.max_score}`, s.instructor_score != null ? h("span", { class: "badge ok", style: "margin-left:6px" }, "reviewed") : null)))))));
    } catch (e) { page.lastChild.replaceWith(h("div", { class: "error" }, e.message)); }
  },
};
