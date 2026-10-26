"""Shannon: the NLP tutor platform (chat, practice, assignments, instructor dashboard), served on this PC only.

Run from the project root (or double-click Shannon.bat):
  python scripts/shannon_server.py [--port 8765] [--adapter models/adapters/full1] [--instructor-pin 1234] [--open]

Student API
  GET  /api/status                    ready / error / adapter / whether the instructor area needs a PIN
  POST /api/chat                      {"messages", "style": "hint"|"explain", "base": bool} -> answer, sources,
                                      stats, code_checks (one per ``` block: syntax, security and style notes)
  POST /api/feedback                  {"student", "question", "answer", "rating": 1|-1, "comment", "style", "model"}
  POST /api/practice                  {"student", "topic", "kind": mcq|short|code, "difficulty"} -> a new question
  GET  /api/assessments               published quizzes/exams (questions only, no answers)
  POST /api/submit                    {"student", "item_id", "assessment_id"?, "answer"} -> mark and feedback
  GET  /api/history?student=NAME      that student's submissions
Instructor API (header X-Instructor-Pin when --instructor-pin is set)
  POST /api/instructor/generate       {"topic", "kind", "difficulty", "n"} -> pending items
  GET  /api/instructor/items?status=  question bank;  POST /api/instructor/items/<id>  {"status"?, "body"?}
  GET/POST /api/instructor/assessments;  POST /api/instructor/assessments/<id>  {"published"}
  GET  /api/instructor/submissions?assessment_id=;  POST /api/instructor/submissions/<id>  {"score", "comment"}
  GET  /api/instructor/analytics;  GET /api/instructor/export/items.jsonl | submissions.csv
Data: data/shannon.db (SQLite). Chats stay in the browser. One model on the GPU, so model calls take turns.
Listens on 127.0.0.1 only: nothing is reachable from the network.
"""
import argparse
import contextlib
import csv
import io
import json
import re
import socket
import statistics
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent))
import tutor as T  # noqa: E402
from shannon import codecheck, plagiarism  # noqa: E402
from shannon.generate import generate  # noqa: E402
from shannon.grade import grade  # noqa: E402
from shannon.llm import LLM, LOCK  # noqa: E402
from shannon.store import Store  # noqa: E402

WEB = T.ROOT / "web" / "shannon"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css",
         ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}
# the only prompt change from training: the tutor gets a name, so "who are you?" has an answer
T.SYSTEM = T.SYSTEM.replace("You are an NLP tutor", "You are Shannon, an NLP tutor", 1)

# readable names for sources whose index title is a file name
RENAME = [(r"^ed3book_\w+", "Speech and Language Processing (Jurafsky & Martin)"),
          (r"^harvardnlp/annotated-transformer — AnnotatedTransformer\.ipynb", "The Annotated Transformer"),
          (r"^stanford-cs336/(\S+) — \S+", r"Stanford CS336 \1")]
FENCED = re.compile(r"```([\w+-]*)[^\n]*\n?([\s\S]*?)```")
# requests for code need room: at the hint-mode budget (260 tokens) a function was cut off mid-line
WANTS_CODE = re.compile(r"(?i)\b(code|implement|python|function|program|script|snippet|write .{0,30}\b(class|loop))\b")

state = {"tutor": None, "llm": None, "error": None, "adapter": None, "pin": None}
store = Store(T.ROOT / "data" / "shannon.db")


class Problem(Exception):
    """An error the user should see, with an HTTP status."""
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status = status


def nice_title(s):
    title = s["title"]
    if s.get("doc_id", "").startswith("slp3-2026aug-slides-"):
        title = re.sub(r"^\S+", "SLP3 lecture slides (" + title.split(" — ")[0] + ")", title, count=1)
    for pat, rep in RENAME:
        title = re.sub(pat, rep, title)
    return title


def load(a):
    try:
        targs = argparse.Namespace(style="hint", open_only=a.open_only, k=6, search="hybrid", model=a.model,
                                   adapter=a.adapter, max_new_tokens=None, no_cite_check=False, llm_verify=False,
                                   no_rag=False)
        t = T.Tutor(targs)
        retrieve = t.retrieve

        def keep_hits(q):  # remember the passages, so the page can show what each [n] points to
            t.last_hits = retrieve(q)
            return t.last_hits
        t.retrieve = keep_hits
        state.update(tutor=t, llm=LLM(t), adapter=a.adapter)
        print(f"Shannon is ready: http://127.0.0.1:{a.port}", flush=True)
    except Exception as e:
        traceback.print_exc()
        state["error"] = f"{type(e).__name__}: {e}"


# ---------- chat ----------

def code_checks(answer):
    """Security and best-practice notes for each Python block the tutor wrote (None for other languages)."""
    out = []
    for m in FENCED.finditer(answer):
        lang, code = m.group(1).lower(), m.group(2)
        python = lang in ("python", "py", "python3") or (not lang and re.search(r"^\s*(def |import |from |\w+ = )", code, re.M))
        if not python:
            out.append(None)
            continue
        r = codecheck.review(code)
        out.append({"syntax_ok": r["syntax_ok"], "warnings": [{"line": l, "text": t} for l, t in r["warnings"]]})
    return out


def chat(body):
    t = state["tutor"]
    msgs = [m for m in body["messages"] if m.get("role") in ("user", "assistant")]
    question = msgs[-1]["content"].strip()
    use_base = bool(body.get("base")) and hasattr(t.model, "disable_adapter")
    with LOCK:
        t.a.style = "explain" if body.get("style") == "explain" else "hint"
        t.a.max_new_tokens = 800 if WANTS_CODE.search(question) else None
        t.history = [(m["role"], m["content"]) for m in msgs[:-1]]
        t.last_hits = []
        with t.model.disable_adapter() if use_base else contextlib.nullcontext():
            answer, sources, stats = t.ask(question)
        bodies = T.format_passages(t.last_hits)[2] if sources else []
    for s, text in zip(sources, bodies):
        s["text"] = text.split("\n", 1)[-1]
    for s in sources:
        s["title"] = nice_title(s)
    log = stats.get("citation_check", [])
    return {"answer": answer, "sources": sources, "code_checks": code_checks(answer),
            "stats": {"seconds": stats.get("gen_seconds", 0.0), "tokens": stats.get("new_tokens", 0),
                      "off_topic": stats.get("off_topic", False), "style": t.a.style,
                      "model": "base" if use_base or not state["adapter"] else Path(state["adapter"]).name,
                      "unsourced": sum(c["action"] == "marked-unsourced" for c in log)}}


# ---------- items, practice, submissions ----------

def max_score(item):
    b = item["body"]
    return sum(p["marks"] for p in b["scheme"]) + (b.get("test_marks", 0.0) if item["kind"] == "code" else 0.0)


def student_view(item):
    """What a student may see before answering: no key, model answer, solution, tests or scheme."""
    b = item["body"]
    srcs = {}
    for s in item.get("sources") or []:  # two passages from one section would list the same source twice
        srcs.setdefault((nice_title(s), s.get("loc", "")), {"title": nice_title(s), "loc": s.get("loc", "")})
    v = {"id": item["id"], "kind": item["kind"], "difficulty": item["difficulty"], "topic": item["topic"],
         "question": b["question"], "max_score": max_score(item), "sources": list(srcs.values())}
    if item["kind"] == "mcq":
        v["options"] = b["options"]
    if item["kind"] == "code":
        v.update(starter=b["starter"], hint=b.get("hint", ""), n_tests=len(b["tests"]))
    return v


def name_of(body):
    s = re.sub(r"\s+", " ", str(body.get("student") or "")).strip()[:40]
    if not s:
        raise Problem(400, "Please enter your name first.")
    return s


def practice(body):
    student = name_of(body)
    topic = str(body.get("topic") or "").strip()[:120]
    if len(topic) < 3:
        raise Problem(400, "Type a topic to practise, e.g. 'n-gram smoothing'.")
    kind, level = body.get("kind", "mcq"), body.get("difficulty", "medium")
    prev = [i["body"]["question"] for i in store.items("practice")
            if i["body"].get("owner") == student and i["topic"].lower() == topic.lower()]
    try:
        b, checks, sources = generate(state["llm"], state["tutor"], topic, kind, level, previous=prev)
    except ValueError as e:
        raise Problem(422, str(e))
    b["owner"] = student
    iid = store.add_item(kind, level, topic, "practice", b, checks, sources)
    return student_view(store.item(iid))


def student_result(kind, result, exam):
    """Exams show the mark and feedback only; quizzes and practice show the full breakdown."""
    if not exam:
        return result
    keep = {"feedback"} | ({"tests"} if kind == "code" else set())
    return {k: v for k, v in result.items() if k in keep}


def submit(body):
    student = name_of(body)
    item = store.item(int(body.get("item_id", 0)))
    if not item:
        raise Problem(404, "That question no longer exists.")
    aid = body.get("assessment_id")
    exam = False
    if aid is not None:
        a = store.assessment(int(aid))
        if not a or not a["published"] or item["id"] not in a["item_ids"]:
            raise Problem(403, "This assessment is not open.")
        exam = a["kind"] == "exam"
        if exam and any(s["student"].lower() == student.lower()
                        for s in store.submissions(item_id=item["id"], assessment_id=a["id"])):
            raise Problem(409, "You have already submitted this exam question.")
    elif item["status"] != "practice" or item["body"].get("owner") != student:
        raise Problem(403, "This question is not open for practice.")
    answer = body.get("answer") or {}
    if item["kind"] == "mcq":
        answer = {"choice": int(answer["choice"]) if str(answer.get("choice", "")).lstrip("-").isdigit() else None}
    else:
        answer = {k: str(v)[:20000] for k, v in answer.items() if k in ("text", "code", "explanation")}
    score, mx, result, flags = grade(state["llm"], item, answer)
    if aid is not None:
        peers = [(s["id"], s["student"], s["answer"]) for s in store.submissions(item_id=item["id"])
                 if s["student"].lower() != student.lower() and s["assessment_id"] is not None]
        flags += plagiarism.check(item["kind"], answer, item, peers)
    sid = store.add_submission(student, item["id"], int(aid) if aid is not None else None, answer, score, mx, result, flags)
    return {"id": sid, "score": score, "max_score": mx, "result": student_result(item["kind"], result, exam),
            "exam": exam}


def history(student):
    items = {}
    out = []
    for s in store.submissions(student=student):
        it = items.setdefault(s["item_id"], store.item(s["item_id"]))
        out.append({"id": s["id"], "created": s["created"], "item_id": s["item_id"], "assessment_id": s["assessment_id"],
                    "kind": it["kind"] if it else "?", "topic": it["topic"] if it else "?",
                    "question": it["body"]["question"] if it else "", "score": s["auto_score"],
                    "max_score": s["max_score"], "instructor_score": s["instructor_score"],
                    "instructor_comment": s["instructor_comment"]})
    return out


def open_assessments():
    out = []
    for a in store.assessments(published_only=True):
        items = [store.item(i) for i in a["item_ids"]]
        out.append({"id": a["id"], "title": a["title"], "kind": a["kind"],
                    "items": [student_view(i) for i in items if i]})
    return out


# ---------- instructor ----------

def instructor_generate(body):
    topic = str(body.get("topic") or "").strip()[:120]
    if len(topic) < 3:
        raise Problem(400, "Type a topic.")
    n = max(1, min(5, int(body.get("n", 1))))
    made, errors, prev = [], [], []
    for _ in range(n):
        try:
            b, checks, sources = generate(state["llm"], state["tutor"], topic, body.get("kind", "mcq"),
                                          body.get("difficulty", "medium"), previous=prev)
        except ValueError as e:
            errors.append(str(e))
            continue
        prev.append(b["question"])
        made.append(store.item(store.add_item(body.get("kind", "mcq"), body.get("difficulty", "medium"), topic,
                                              "pending", b, checks, sources)))
    return {"items": made, "errors": errors}


def update_item(iid, body):
    item = store.item(iid)
    if not item or item["status"] == "practice":
        raise Problem(404, "No such item in the bank.")
    status = body.get("status")
    if status not in (None, "pending", "approved", "rejected"):
        raise Problem(400, "Bad status.")
    new = body.get("body")
    if new is not None:  # instructor edits: question, options/answer, model answer, scheme, tests
        b = dict(item["body"])
        for k in ("question", "options", "answer", "explanation", "model_answer", "scheme", "starter", "solution",
                  "tests", "hint"):
            if k in new:
                b[k] = new[k]
        if not b.get("scheme") or any(not isinstance(p.get("marks"), (int, float)) or not p.get("point")
                                      for p in b["scheme"]):
            raise Problem(400, "Every marking-scheme line needs marks and a point, e.g. [2] defines perplexity.")
        if item["kind"] == "code" and ("solution" in new or "tests" in new):
            res = codecheck.run_tests(b["solution"], b["tests"])
            if not all(r["ok"] for r in res):
                bad = next(r for r in res if not r["ok"])
                raise Problem(400, f"The reference solution fails a test: {bad['test']} ({bad['error']})")
        store.update_item(iid, body=b)
    if status:
        store.update_item(iid, status=status)
    return store.item(iid)


def submissions_view(aid):
    items, out = {}, []
    for s in store.submissions(assessment_id=aid):
        it = items.setdefault(s["item_id"], store.item(s["item_id"]))
        out.append({**s, "item": {"id": it["id"], "kind": it["kind"], "topic": it["topic"],
                                  "question": it["body"]["question"]} if it else None})
    return out


def pct(s, key="auto_score"):
    return 100.0 * s[key] / s["max_score"] if s["max_score"] else 0.0


def analytics():
    subs = store.submissions()
    items = {}
    for s in subs:
        items.setdefault(s["item_id"], store.item(s["item_id"]))
    rev = [s for s in subs if s["instructor_score"] is not None]
    cons = {"reviewed": len(rev)}
    if rev:
        diffs = [abs(pct(s) - pct(s, "instructor_score")) for s in rev]
        cons.update(exact=round(100 * sum(d < 0.5 for d in diffs) / len(rev), 1),
                    within10=round(100 * sum(d <= 10 for d in diffs) / len(rev), 1),
                    mean_abs_diff=round(statistics.mean(diffs), 1))
        if len(rev) >= 3:
            try:
                cons["correlation"] = round(statistics.correlation([pct(s) for s in rev],
                                                                   [pct(s, "instructor_score") for s in rev]), 3)
            except statistics.StatisticsError:
                pass
    topics = {}
    for s in subs:
        it = items.get(s["item_id"])
        if it:
            topics.setdefault(it["topic"].lower(), []).append(pct(s))
    by_topic = sorted(({"topic": k, "attempts": len(v), "mean": round(statistics.mean(v), 1)} for k, v in topics.items()),
                      key=lambda r: -r["attempts"])
    # retention: the same student on the same topic later on (first attempt vs latest, and the gap in days)
    runs = {}
    for s in sorted(subs, key=lambda s: s["created"]):
        it = items.get(s["item_id"])
        if it:
            runs.setdefault((s["student"], it["topic"].lower()), []).append(s)
    retention = []
    for (student, topic), ss in runs.items():
        if len(ss) >= 2:
            retention.append({"student": student, "topic": topic, "attempts": len(ss), "first": round(pct(ss[0]), 1),
                              "latest": round(pct(ss[-1]), 1),
                              "days": round((ss[-1]["created"] - ss[0]["created"]) / 86400, 1)})
    fb = store.feedback()
    helpful = {"ratings": len(fb), "up": sum(f["rating"] > 0 for f in fb)}
    for key in ("model", "style"):
        groups = {}
        for f in fb:
            groups.setdefault(f[key] or "?", []).append(f["rating"] > 0)
        helpful["by_" + key] = {k: {"n": len(v), "up_pct": round(100 * sum(v) / len(v), 1)} for k, v in groups.items()}
    helpful["comments"] = [{"rating": f["rating"], "comment": f["comment"], "question": f["question"][:120],
                            "created": f["created"]} for f in fb if f["comment"]][:20]
    bank = {}
    for it in store.items():
        bank[it["status"]] = bank.get(it["status"], 0) + 1
    per_item = {}
    for s in subs:
        per_item.setdefault(s["item_id"], []).append(pct(s))
    hardest = sorted(({"item_id": k, "question": items[k]["body"]["question"][:120], "attempts": len(v),
                       "mean": round(statistics.mean(v), 1)} for k, v in per_item.items() if items.get(k)),
                     key=lambda r: r["mean"])[:8]
    return {"consistency": cons, "topics": by_topic, "retention": retention, "helpfulness": helpful, "bank": bank,
            "hardest": hardest, "submissions": len(subs), "students": len({s["student"].lower() for s in subs}),
            "flagged": sum(bool(s["flags"]) for s in subs)}


def export_items():
    lines = []
    for it in store.items("approved"):
        lines.append(json.dumps({"id": it["id"], "kind": it["kind"], "difficulty": it["difficulty"], "topic": it["topic"],
                                 **it["body"], "verification": it["checks"],
                                 "sources": [{"doc_id": s.get("doc_id"), "title": nice_title(s), "loc": s.get("loc"),
                                              "license_tier": s.get("license_tier")} for s in it["sources"] or []]},
                                ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


def export_submissions():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "time", "student", "assessment_id", "item_id", "auto_score", "max_score", "instructor_score",
                "instructor_comment", "flags"])
    for s in store.submissions():
        w.writerow([s["id"], time.strftime("%Y-%m-%d %H:%M", time.localtime(s["created"])), s["student"],
                    s["assessment_id"], s["item_id"], s["auto_score"], s["max_score"], s["instructor_score"],
                    s["instructor_comment"] or "", "; ".join(f["text"] for f in s["flags"] or [])])
    return buf.getvalue().encode("utf-8-sig")


# ---------- HTTP ----------

class Server(ThreadingHTTPServer):
    # the default (address reuse on) lets a second copy bind the same port on Windows and load a second model
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body, ctype="application/json; charset=utf-8", filename=None):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def instructor(self):
        if state["pin"] and self.headers.get("X-Instructor-Pin") != state["pin"]:
            raise Problem(401, "Instructor PIN required.")

    def ready(self):
        if state["tutor"] is None:
            raise Problem(503, state["error"] or "Shannon is still loading.")

    def handle_api(self, method):
        u = urlparse(self.path)
        p, q = u.path, {k: v[0] for k, v in parse_qs(u.query).items()}
        body = {}
        if method == "POST":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if p == "/api/status":
            return {"ready": state["tutor"] is not None, "error": state["error"], "pin_required": bool(state["pin"]),
                    "adapter": Path(state["adapter"]).name if state["adapter"] else None}
        if p.startswith("/api/instructor/"):
            self.instructor()
            sub = p[len("/api/instructor/"):]
            if sub == "check":
                return {"ok": True}
            if sub == "generate" and method == "POST":
                self.ready()
                return instructor_generate(body)
            if sub == "items" and method == "GET":
                return store.items(q.get("status"))
            if (m := re.fullmatch(r"items/(\d+)", sub)) and method == "POST":
                return update_item(int(m.group(1)), body)
            if sub == "assessments":
                if method == "POST":
                    ids = [int(i) for i in body.get("item_ids", [])]
                    if not ids or any((store.item(i) or {}).get("status") != "approved" for i in ids):
                        raise Problem(400, "Pick at least one approved question.")
                    title = str(body.get("title") or "").strip()[:100] or "Untitled quiz"
                    return store.assessment(store.add_assessment(title, "exam" if body.get("kind") == "exam" else "quiz", ids))
                return [{**a, "submissions": len(store.submissions(assessment_id=a["id"]))} for a in store.assessments()]
            if (m := re.fullmatch(r"assessments/(\d+)", sub)) and method == "POST":
                store.publish(int(m.group(1)), bool(body.get("published")))
                return store.assessment(int(m.group(1)))
            if sub == "submissions" and method == "GET":
                return submissions_view(int(q["assessment_id"]) if q.get("assessment_id") else None)
            if (m := re.fullmatch(r"submissions/(\d+)", sub)) and method == "POST":
                s = store.submission(int(m.group(1)))
                if not s:
                    raise Problem(404, "No such submission.")
                score = body.get("score")
                if score is not None:
                    score = float(score)
                    if not 0 <= score <= s["max_score"]:
                        raise Problem(400, f"The mark must be between 0 and {s['max_score']:g}.")
                store.review(s["id"], score, str(body.get("comment") or "")[:2000])
                return store.submission(s["id"])
            if sub == "analytics":
                return analytics()
            if sub == "export/items.jsonl":
                return ("file", export_items(), "application/x-ndjson", "shannon-problem-bank.jsonl")
            if sub == "export/submissions.csv":
                return ("file", export_submissions(), "text/csv; charset=utf-8", "shannon-submissions.csv")
            raise Problem(404, "Unknown instructor endpoint.")
        if p == "/api/assessments" and method == "GET":
            return open_assessments()
        if p == "/api/history" and method == "GET":
            return history(name_of({"student": q.get("student")}))
        if p == "/api/feedback" and method == "POST":
            r = 1 if int(body.get("rating", 0)) > 0 else -1
            store.add_feedback(str(body.get("student") or "")[:40], str(body.get("question") or "")[:2000],
                               str(body.get("answer") or "")[:8000], r, str(body.get("comment") or "")[:2000],
                               body.get("style"), body.get("model"))
            return {"ok": True}
        if method == "POST":
            self.ready()
            if p == "/api/chat":
                return chat(body)
            if p == "/api/practice":
                return practice(body)
            if p == "/api/submit":
                return submit(body)
        raise Problem(404, "not found")

    def api(self, method):
        try:
            out = self.handle_api(method)
            if isinstance(out, tuple) and out[0] == "file":
                return self.reply(200, out[1], out[2], filename=out[3])
            self.reply(200, out)
        except Problem as e:
            self.reply(e.status, {"error": str(e)})
        except Exception as e:
            traceback.print_exc()
            self.reply(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/api/"):
            return self.api("GET")
        f = (WEB / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
        if WEB.resolve() not in f.parents or not f.is_file():
            return self.reply(404, {"error": "not found"})
        self.reply(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

    def do_POST(self):
        self.api("POST")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default=str(T.ROOT / "models" / "adapters" / "full1"),
                    help="LoRA adapter dir; pass '' for the base model only")
    ap.add_argument("--open-only", action="store_true", help="search only openly licensed material")
    ap.add_argument("--instructor-pin", help="require this PIN for the instructor area")
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    a = ap.parse_args()
    a.adapter = a.adapter or None
    state["pin"] = a.instructor_pin
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        server = Server(("127.0.0.1", a.port), Handler)
    except OSError:  # already running (second double-click on Shannon.bat): just show the page
        print(f"Shannon is already running at http://127.0.0.1:{a.port}")
        if a.open:
            webbrowser.open(f"http://127.0.0.1:{a.port}")
        return
    threading.Thread(target=load, args=(a,), daemon=True).start()
    print(f"Shannon page: http://127.0.0.1:{a.port}  (loading the model, about a minute)", flush=True)
    if a.open:
        webbrowser.open(f"http://127.0.0.1:{a.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
