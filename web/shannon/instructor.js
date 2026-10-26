// Instructor dashboard: question bank (generate, verify, edit, approve), assessments, submissions review, insights.
const Instructor = {
  tab: "bank", filter: "pending", selected: new Set(),

  async open() {
    const page = $("#instructor-page");
    if (status.pin_required && !sessionStorage.getItem("shannon.pin")) return this.pinGate(page);
    page.replaceChildren(h("h2", {}, "Instructor"),
      h("p", { class: "lead" }, "Generate and approve questions, publish quizzes and exams, review marks and see how students are doing."),
      h("div", { class: "subtabs" }, seg({ bank: "Question bank", assess: "Assessments", subs: "Submissions", insights: "Insights" }, this.tab,
        v => { this.tab = v; this.render(); })),
      h("div", { id: "ins-body" }));
    this.render();
  },

  pinGate(page) {
    const inp = h("input", { class: "inp", type: "password", placeholder: "Instructor PIN", autocomplete: "off" });
    const msg = h("div");
    const go = async () => {
      sessionStorage.setItem("shannon.pin", inp.value);
      try { await api("/api/instructor/check"); this.open(); }
      catch (e) { sessionStorage.removeItem("shannon.pin"); msg.replaceChildren(h("div", { class: "error small" }, e.message)); }
    };
    inp.onkeydown = e => { if (e.key === "Enter") go(); };
    page.replaceChildren(h("h2", {}, "Instructor"), h("div", { class: "card", style: "max-width:380px" },
      h("p", { class: "muted small", style: "margin-top:0" }, "This area is protected with a PIN."), inp,
      h("div", { class: "row", style: "margin-top:10px" }, h("button", { class: "btn primary", onclick: go }, "Unlock")), msg));
    inp.focus();
  },

  body() { return $("#ins-body"); },
  fail(e) { this.body().replaceChildren(h("div", { class: "error" }, e.message)); },
  render() { ({ bank: () => this.bank(), assess: () => this.assessments(), subs: () => this.submissions(), insights: () => this.insights() })[this.tab](); },

  /* ---------- question bank ---------- */
  async bank() {
    const b = this.body();
    const s = store.get("shannon.gen", { topic: "", kind: "mcq", difficulty: "medium", n: 2 });
    const save = () => store.set("shannon.gen", s);
    const topic = h("input", { class: "inp grow", list: "topic-list-i", placeholder: "Topic, e.g. perplexity", value: s.topic });
    topic.oninput = () => { s.topic = topic.value; save(); };
    const n = h("select", {}, [1, 2, 3, 4, 5].map(k => h("option", { value: k, selected: k === s.n }, `${k} question${k > 1 ? "s" : ""}`)));
    n.onchange = () => { s.n = +n.value; save(); };
    const msg = h("div");
    const gen = h("button", { class: "btn primary" }, "Generate");
    gen.onclick = async () => {
      if (!topic.value.trim()) { topic.focus(); return; }
      gen.disabled = true;
      msg.replaceChildren(spinner(`Generating and verifying ${s.n} question${s.n > 1 ? "s" : ""} (about ${s.kind === "code" ? "2 minutes" : "1-2 minutes"} each)…`));
      try {
        const d = await api("/api/instructor/generate", { body: { topic: topic.value.trim(), kind: s.kind, difficulty: s.difficulty, n: s.n } });
        msg.replaceChildren(h("div", { class: "note ok" }, `${d.items.length} new question${d.items.length === 1 ? "" : "s"} waiting for review.`),
          ...d.errors.map(e => h("div", { class: "note warn" }, e)));
        this.filter = "pending"; this.list(list);
      } catch (e) { msg.replaceChildren(h("div", { class: "error small" }, e.message)); }
      gen.disabled = false;
    };
    const list = h("div");
    b.replaceChildren(
      h("div", { class: "card" }, h("h3", {}, "Generate questions from the course material"),
        h("div", { class: "row" }, topic, n, gen),
        h("div", { class: "row", style: "margin-top:10px" }, seg(KIND_LABEL, s.kind, v => { s.kind = v; save(); }),
          seg(LEVEL_LABEL, s.difficulty, v => { s.difficulty = v; save(); })),
        h("div", { class: "muted small", style: "margin-top:8px" }, "Every question is checked automatically: multiple-choice keys are re-solved blind, marking points must be supported by the course text, and coding exercises are run against their tests. You still approve each one."),
        msg, h("datalist", { id: "topic-list-i" }, TOPICS.map(t => h("option", { value: t })))),
      h("div", { class: "row", style: "margin-bottom:10px" }, seg({ pending: "Waiting for review", approved: "Approved", rejected: "Rejected", all: "All" }, this.filter,
        v => { this.filter = v; this.list(list); })),
      list);
    this.list(list);
  },

  async list(box) {
    box.replaceChildren(spinner("Loading…"));
    try {
      const items = await api("/api/instructor/items" + (this.filter === "all" ? "" : "?status=" + this.filter));
      box.replaceChildren();
      if (!items.length) { box.append(h("div", { class: "card muted" }, this.filter === "pending" ? "Nothing is waiting for review. Generate some questions above." : "No questions here.")); }
      for (const it of items) box.append(this.itemCard(it, box));
      const approved = items.filter(i => i.status === "approved");
      if (approved.length) box.append(this.selectionBar());
    } catch (e) { box.replaceChildren(h("div", { class: "error" }, e.message)); }
  },

  schemeText(scheme) { return scheme.map(p => `[${p.marks}] ${p.point}`).join("\n"); },
  parseScheme(text) {
    return text.split("\n").map(l => l.trim()).filter(Boolean).map(l => {
      const m = l.match(/^\[(\d+(?:\.\d+)?)\]\s*(.+)$/);
      if (!m) throw new Error(`Each marking line must look like "[2] the point": ${l}`);
      return { point: m[2], marks: +m[1] };
    });
  },

  itemCard(it, listBox) {
    const b = it.body;
    const card = h("div", { class: "qcard" });
    const sel = it.status === "approved" ? h("input", { type: "checkbox", checked: this.selected.has(it.id), title: "Select for a new assessment",
      onchange: e => { e.target.checked ? this.selected.add(it.id) : this.selected.delete(it.id); this.updateBar(); } }) : null;
    card.append(h("div", { class: "qhead" }, sel, h("span", { class: "badge accent" }, KIND_LABEL[it.kind]), h("span", { class: "badge" }, LEVEL_LABEL[it.difficulty]),
      h("span", { class: "badge" }, it.topic), h("span", { class: "badge " + ({ approved: "ok", rejected: "fail" }[it.status] || "warn") }, it.status),
      h("span", { class: "marks" }, `#${it.id} · ${fmtDate(it.created)}`)));
    card.append(h("div", { class: "notes" }, (it.checks?.notes || []).map(n => h("div", { class: "note " + n.level }, (n.level === "ok" ? "✓ " : "⚠ ") + n.text))));
    const view = h("div");
    const showView = () => {
      view.replaceChildren(h("div", { class: "q", html: markdown(b.question, { noCites: true }) }));
      if (it.kind === "mcq") {
        view.append(h("div", { class: "options" }, b.options.map((o, i) => h("div", { class: "opt" + (i === b.answer ? " right" : "") },
          h("span", { class: "l" }, "ABCD"[i]), h("span", {}, o)))));
        if (b.explanation) view.append(h("div", { class: "small" }, h("b", {}, "Explanation: "), b.explanation));
      }
      if (it.kind === "short") view.append(h("div", { class: "small" }, h("b", {}, "Model answer: "), h("span", { html: markdown(b.model_answer, { noCites: true }) })));
      if (it.kind === "code") {
        view.append(h("details", { class: "more", open: true }, h("summary", {}, "Starter code, reference solution and tests"),
          h("div", { html: markdown("Starter:\n```python\n" + b.starter + "```\nReference solution:\n```python\n" + b.solution + "```\nTests:\n```python\n" + b.tests.join("\n") + "\n```") })));
        if (b.hint) view.append(h("div", { class: "small" }, h("b", {}, "Hint for students: "), b.hint));
      }
      view.append(h("div", { class: "small", style: "margin-top:8px" }, h("b", {}, it.kind === "code" ? `Marking: tests ${b.test_marks} marks, explanation:` : "Marking scheme:"),
        h("ul", { style: "margin:4px 0" }, b.scheme.map(p => h("li", {}, `[${p.marks}] ${p.point}`)))));
      enhance(view);
    };
    showView();
    card.append(view);
    const msg = h("div");
    const act = async (status, bodyEdit) => {
      try {
        await api(`/api/instructor/items/${it.id}`, { body: { status, body: bodyEdit } });
        this.list(listBox);
      } catch (e) { msg.replaceChildren(h("div", { class: "error small" }, e.message)); }
    };
    const edit = () => {
      const f = {};
      const ed = h("div", { style: "display:grid;gap:8px" });
      f.question = h("textarea", { class: "inp", rows: 3 }); f.question.value = b.question;
      ed.append(h("label", { class: "field" }, "Question", f.question));
      if (it.kind === "mcq") {
        f.options = b.options.map((o, i) => { const x = h("input", { class: "inp" }); x.value = o; ed.append(h("label", { class: "field" }, `Option ${"ABCD"[i]}`, x)); return x; });
        f.answer = h("select", {}, [0, 1, 2, 3].map(i => h("option", { value: i, selected: i === b.answer }, "ABCD"[i])));
        f.explanation = h("textarea", { class: "inp", rows: 2 }); f.explanation.value = b.explanation || "";
        ed.append(h("label", { class: "field" }, "Correct option", f.answer), h("label", { class: "field" }, "Explanation", f.explanation));
      }
      if (it.kind === "short") { f.model = h("textarea", { class: "inp", rows: 4 }); f.model.value = b.model_answer; ed.append(h("label", { class: "field" }, "Model answer", f.model)); }
      if (it.kind === "code") {
        f.solution = codeEditor(b.solution); f.tests = codeEditor(b.tests.join("\n")); f.tests.style.minHeight = "120px";
        ed.append(h("label", { class: "field" }, "Reference solution", f.solution), h("label", { class: "field" }, "Tests (one assert per line)", f.tests));
      }
      f.scheme = h("textarea", { class: "inp", rows: 4 }); f.scheme.value = this.schemeText(b.scheme);
      ed.append(h("label", { class: "field" }, it.kind === "code" ? "Explanation marking points ([marks] point per line)" : "Marking scheme ([marks] point per line)", f.scheme));
      const saveBtn = h("button", { class: "btn primary" }, "Save");
      saveBtn.onclick = async () => {
        try {
          const nb = { question: f.question.value.trim(), scheme: this.parseScheme(f.scheme.value) };
          if (it.kind === "mcq") Object.assign(nb, { options: f.options.map(x => x.value.trim()), answer: +f.answer.value, explanation: f.explanation.value.trim() });
          if (it.kind === "short") nb.model_answer = f.model.value.trim();
          if (it.kind === "code") Object.assign(nb, { solution: f.solution.value, tests: f.tests.value.split("\n").map(l => l.trim()).filter(Boolean) });
          saveBtn.disabled = true;
          await act(null, nb);
        } catch (e) { saveBtn.disabled = false; msg.replaceChildren(h("div", { class: "error small" }, e.message)); }
      };
      view.replaceChildren(ed, h("div", { class: "row", style: "margin-top:8px" }, saveBtn, h("button", { class: "btn", onclick: () => { showView(); } }, "Cancel")));
    };
    card.append(h("div", { class: "row", style: "margin-top:12px" },
      it.status !== "approved" ? h("button", { class: "btn primary", onclick: () => act("approved") }, "Approve") : null,
      it.status !== "rejected" ? h("button", { class: "btn bad", onclick: () => act("rejected") }, "Reject") : null,
      it.status !== "pending" ? h("button", { class: "btn", onclick: () => act("pending") }, "Back to review") : null,
      h("button", { class: "btn", onclick: edit }, "Edit"), msg));
    return card;
  },

  selectionBar() {
    const title = h("input", { class: "inp grow", placeholder: "Title, e.g. Week 3 quiz: n-grams" });
    let kind = "quiz";
    const count = h("span", { class: "muted small" });
    const msg = h("span");
    const btn = h("button", { class: "btn primary" }, "Create assessment");
    btn.onclick = async () => {
      try {
        const a = await api("/api/instructor/assessments", { body: { title: title.value, kind, item_ids: [...this.selected] } });
        this.selected.clear();
        msg.replaceChildren(h("span", { class: "badge ok" }, `Created "${a.title}". Publish it on the Assessments tab.`));
        this.updateBar();
      } catch (e) { msg.replaceChildren(h("span", { class: "error small" }, e.message)); }
    };
    this.bar = { count, btn };
    const bar = h("div", { class: "selbar" }, count, title, seg({ quiz: "Quiz", exam: "Exam" }, kind, v => kind = v), btn, msg);
    this.updateBar();
    return bar;
  },
  updateBar() {
    if (!this.bar) return;
    this.bar.count.textContent = `${this.selected.size} selected`;
    this.bar.btn.disabled = !this.selected.size;
  },

  /* ---------- assessments ---------- */
  async assessments() {
    const b = this.body();
    b.replaceChildren(spinner("Loading…"));
    try {
      const list = await api("/api/instructor/assessments");
      b.replaceChildren(h("div", { class: "card" }, h("h3", {}, "Assessments"),
        list.length ? h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" },
          h("tr", {}, h("th", {}, "Title"), h("th", {}, "Type"), h("th", {}, "Questions"), h("th", {}, "Submissions"), h("th", {}, "Created"), h("th", {}, "")),
          list.map(a => h("tr", {}, h("td", {}, a.title), h("td", {}, a.kind === "exam" ? "Exam" : "Quiz"), h("td", {}, a.item_ids.length), h("td", {}, a.submissions),
            h("td", { class: "muted" }, fmtDate(a.created)),
            h("td", {}, h("button", { class: "btn " + (a.published ? "" : "primary"), onclick: async () => {
              await api(`/api/instructor/assessments/${a.id}`, { body: { published: !a.published } }); this.assessments(); } },
              a.published ? "Unpublish" : "Publish")))))) :
          h("div", { class: "muted" }, "No assessments yet. Approve questions in the Question bank, tick them, and create one.")));
    } catch (e) { this.fail(e); }
  },

  /* ---------- submissions ---------- */
  async submissions() {
    const b = this.body();
    b.replaceChildren(spinner("Loading…"));
    try {
      const [assess] = await Promise.all([api("/api/instructor/assessments")]);
      const pick = h("select", {}, h("option", { value: "" }, "All assessments"), assess.map(a => h("option", { value: a.id, selected: String(a.id) === String(this.aid || "") }, a.title)));
      const table = h("div");
      pick.onchange = () => { this.aid = pick.value || null; load(); };
      const load = async () => {
        table.replaceChildren(spinner("Loading…"));
        const subs = (await api("/api/instructor/submissions" + (this.aid ? "?assessment_id=" + this.aid : ""))).filter(s => s.assessment_id != null || !this.aid);
        if (!subs.length) { table.replaceChildren(h("div", { class: "muted" }, "No submissions yet.")); return; }
        const t = h("table", { class: "tbl" }, h("tr", {}, h("th", {}, "When"), h("th", {}, "Student"), h("th", {}, "Question"), h("th", {}, "Auto mark"),
          h("th", {}, "Your mark"), h("th", {}, "Flags")));
        for (const s of subs) {
          const row = h("tr", { class: "click" }, h("td", { class: "muted" }, fmtDate(s.created)), h("td", {}, s.student,
            s.assessment_id == null ? h("span", { class: "badge", style: "margin-left:6px" }, "practice") : null),
            h("td", {}, (s.item?.question || "").slice(0, 80) + "…"), h("td", {}, `${+s.auto_score.toFixed(2)}/${s.max_score}`),
            h("td", {}, s.instructor_score != null ? h("span", { class: "badge ok" }, `${s.instructor_score}/${s.max_score}`) : h("span", { class: "muted" }, "—")),
            h("td", {}, (s.flags || []).length ? h("span", { class: "badge fail" }, `⚑ ${(s.flags || []).length}`) : ""));
          const detail = h("tr", { hidden: true }, h("td", { colspan: 6 }, this.subDetail(s, () => load())));
          row.onclick = () => detail.hidden = !detail.hidden;
          t.append(row, detail);
        }
        table.replaceChildren(h("div", { class: "tbl-wrap" }, t));
      };
      b.replaceChildren(h("div", { class: "card" }, h("div", { class: "row", style: "margin-bottom:10px" }, h("h3", { style: "margin:0" }, "Submissions"), h("span", { class: "spacer" }), pick), table));
      load();
    } catch (e) { this.fail(e); }
  },

  subDetail(s, reload) {
    const box = h("div", { style: "padding:6px 4px 12px" });
    const a = s.answer || {};
    const kind = s.item?.kind;
    box.append(h("div", { class: "small", html: "<b>Question:</b> " + markdown(s.item?.question || "", { noCites: true }) }));
    if (kind === "mcq") box.append(h("div", { class: "small" }, h("b", {}, "Chose: "), a.choice != null ? "ABCD"[a.choice] : "nothing"));
    if (a.text) box.append(h("div", { class: "small" }, h("b", {}, "Answer:"), h("div", { class: "feedback", style: "font-size:14.5px" }, a.text)));
    if (a.code != null) box.append(h("div", { class: "small", html: "<b>Code:</b>" + markdown("```python\n" + a.code + "\n```") }));
    if (a.explanation) box.append(h("div", { class: "small" }, h("b", {}, "Explanation: "), a.explanation));
    enhance(box);
    if (s.flags?.length) box.append(h("div", { class: "notes" }, s.flags.map(f => h("div", { class: "note fail" }, "⚑ " + f.text))));
    box.append(renderResult(kind, s.auto_score, s.max_score, s.result || {}));
    const mark = h("input", { class: "inp", type: "number", min: 0, max: s.max_score, step: 0.5, style: "width:110px", value: s.instructor_score ?? s.auto_score });
    const comment = h("input", { class: "inp grow", placeholder: "Comment for the student (optional)", value: s.instructor_comment || "" });
    const msg = h("span");
    box.append(h("div", { class: "row", style: "margin-top:10px" }, h("span", { class: "small" }, "Your mark"), mark, h("span", { class: "muted small" }, `/ ${s.max_score}`), comment,
      h("button", { class: "btn primary", onclick: async () => {
        try { await api(`/api/instructor/submissions/${s.id}`, { body: { score: +mark.value, comment: comment.value } }); msg.replaceChildren(h("span", { class: "badge ok" }, "Saved")); setTimeout(reload, 600); }
        catch (e) { msg.replaceChildren(h("span", { class: "error small" }, e.message)); }
      } }, "Save review"), msg),
      h("div", { class: "muted small" }, "Saving your mark (even when it equals the automatic one) counts towards the grading-consistency check in Insights."));
    return box;
  },

  /* ---------- insights ---------- */
  async insights() {
    const b = this.body();
    b.replaceChildren(spinner("Loading…"));
    try {
      const d = await api("/api/instructor/analytics");
      const c = d.consistency, hp = d.helpfulness;
      const stat = (v, l) => h("div", { class: "stat" }, h("div", { class: "v" }, v), h("div", { class: "l" }, l));
      const download = (path, name) => h("button", { class: "btn", onclick: async () => {
        const blob = await api(path); const url = URL.createObjectURL(blob);
        const a = h("a", { href: url, download: name }); document.body.append(a); a.click(); a.remove(); URL.revokeObjectURL(url);
      } }, "Download " + name);
      const table = (heads, rows) => rows.length ? h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" }, h("tr", {}, heads.map(x => h("th", {}, x))),
        rows.map(r => h("tr", {}, r.map(x => h("td", {}, x)))))) : h("div", { class: "muted small" }, "No data yet.");
      b.replaceChildren(
        h("div", { class: "stats" }, stat(d.students, "students"), stat(d.submissions, "submissions"), stat(d.flagged, "flagged submissions"),
          stat(d.bank.approved || 0, "approved questions"), stat(d.bank.pending || 0, "waiting for review")),
        h("div", { class: "card" }, h("h3", {}, "Grading consistency (automatic mark vs your mark)"),
          c.reviewed ? h("div", { class: "stats", style: "margin:0" }, stat(c.reviewed, "submissions you reviewed"), stat(c.exact + "%", "exact agreement"),
            stat(c.within10 + "%", "within 10 percentage points"), stat(c.mean_abs_diff + " pts", "mean difference"),
            c.correlation != null ? stat(c.correlation, "correlation") : null) :
            h("div", { class: "muted small" }, "Review some submissions (Submissions tab) to measure how closely the automatic marks match yours.")),
        h("div", { class: "card" }, h("h3", {}, "Explanation helpfulness (students' ratings of chat answers)"),
          hp.ratings ? h("div", {}, h("div", { class: "stats", style: "margin:0 0 10px" }, stat(hp.ratings, "ratings"), stat(Math.round(100 * hp.up / hp.ratings) + "%", "rated helpful")),
            table(["Model", "Ratings", "Helpful"], Object.entries(hp.by_model).map(([k, v]) => [k, v.n, v.up_pct + "%"])),
            table(["Mode", "Ratings", "Helpful"], Object.entries(hp.by_style).map(([k, v]) => [k, v.n, v.up_pct + "%"])),
            hp.comments.length ? h("div", { style: "margin-top:10px" }, h("b", { class: "small" }, "Comments"), h("ul", { class: "small" },
              hp.comments.map(x => h("li", {}, (x.rating > 0 ? "👍 " : "👎 ") + x.comment, h("span", { class: "muted" }, ` — on "${x.question}"`))))) : null) :
            h("div", { class: "muted small" }, "No ratings yet. Students rate answers with the thumbs under each chat reply.")),
        h("div", { class: "card" }, h("h3", {}, "Concept retention (same student, same topic, later attempts)"),
          table(["Student", "Topic", "Attempts", "First", "Latest", "Change", "Days between"],
            d.retention.map(r => [r.student, r.topic, r.attempts, r.first + "%", r.latest + "%",
              h("span", { class: "badge " + (r.latest >= r.first ? "ok" : "fail") }, (r.latest - r.first >= 0 ? "+" : "") + Math.round(r.latest - r.first)), r.days]))),
        h("div", { class: "card" }, h("h3", {}, "Topics"), table(["Topic", "Attempts", "Average mark"], d.topics.map(t => [t.topic, t.attempts,
          h("div", { class: "row" }, h("div", { class: "bar grow" }, h("i", { style: `width:${t.mean}%` })), t.mean + "%")]))),
        h("div", { class: "card" }, h("h3", {}, "Hardest questions"), table(["#", "Question", "Attempts", "Average"], d.hardest.map(x => [x.item_id, x.question, x.attempts, x.mean + "%"]))),
        h("div", { class: "card" }, h("h3", {}, "Export"), h("div", { class: "muted small", style: "margin-bottom:8px" },
          "The problem bank is the approved questions with solutions, marking schemes, verification results and sources (JSON lines), ready to share or to train on."),
          h("div", { class: "row" }, download("/api/instructor/export/items.jsonl", "shannon-problem-bank.jsonl"), download("/api/instructor/export/submissions.csv", "shannon-submissions.csv"))));
    } catch (e) { this.fail(e); }
  },
};
