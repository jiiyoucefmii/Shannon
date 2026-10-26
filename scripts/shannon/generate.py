"""Question generator: multiple choice, short answer and Python coding items at three difficulty levels, written
from retrieved course passages, then verified before anyone sees them.

Verification (stored as item['checks'], shown to the instructor as badges):
  multiple choice  the model answers its own question blind (no key); disagreement -> "answer key disputed"
                   the explanation must be supported by a passage (reranker >= 0.3)
  short answer     every marking-scheme point must be supported by a passage; the model answer is graded against
                   its own scheme and must earn >= 80%
  coding           the reference solution must pass the course constraints, and is run against the tests: failing
                   tests are dropped, >= 3 must remain; the starter code must fail at least one test
An item that fails a hard check is regenerated (up to 3 attempts); soft problems are kept but flagged.
"""
import random
import re

from tutor import format_passages  # noqa: E402

from . import codecheck
from .llm import LOCK, sections

KINDS = ("mcq", "short", "code")
LEVELS = ("easy", "medium", "hard")
GUIDE = {
    "easy": "recall or recognise one definition, fact or formula",
    "medium": "apply a concept to a small concrete example, or compare two related ideas",
    "hard": "multi-step reasoning: a new scenario, an edge case, a trade-off, or a short derivation",
}
CODE_GUIDE = {
    "easy": "one short function (5-10 lines) using a single idea, e.g. counting tokens or a simple formula",
    "medium": "a function of 10-25 lines, e.g. n-gram probabilities with add-one smoothing, edit distance, "
              "cosine similarity, or TF-IDF for a small corpus",
    "hard": "a complete small algorithm, e.g. BPE merge learning, Viterbi decoding, beam search over a toy model, "
            "or perplexity of a smoothed bigram model",
}
HEAD = """You write assessment items for a university NLP course, using ONLY the course passages below.

Course passages:
{passages}

The student will NOT see these passages. The question must be fully self-contained: never refer to a figure, table,
example, "the text" or "the passage"; if the question needs numbers or data, write them into the question itself.

"""
# questions that lean on material the student cannot see (caught after generation, then regenerated)
UNSEEN = re.compile(r"(?i)\b(figure|fig\.|table \d|the (table|text|passage|excerpt|reading|chapter|section|slide)s?\b|"
                    r"according to the|in the (above|following) (text|passage))")
AVOID = "\nDo not repeat or rephrase any of these earlier questions:\n{prev}\n"
PROMPTS = {
    "mcq": HEAD + """Write ONE {level} multiple-choice question about: {topic}
Difficulty "{level}" means: {guide}.
Rules: exactly one correct option; the three wrong options are plausible mistakes a student could make; options of
similar length; no "all/none of the above"; never mention "the passage" or "the text".{avoid}
Reply in exactly this format:
### QUESTION
<the question>
### A
<option>
### B
<option>
### C
<option>
### D
<option>
### ANSWER
<one letter>
### EXPLANATION
<2-3 sentences on why that option is right, citing passages as [n]>""",
    "short": HEAD + """Write ONE {level} short-answer question about: {topic}
Difficulty "{level}" means: {guide}.
Never mention "the passage" or "the text".{avoid}
Reply in exactly this format:
### QUESTION
<the question>
### MODEL ANSWER
<a full-marks answer in 3-6 sentences, citing passages as [n]>
### MARKING SCHEME
- [2] <one thing a full answer must state>
- [1] <another>
(3 to 5 points, 1-3 marks each, each point checkable on its own)""",
    "code": HEAD + """Write ONE {level} Python coding exercise about: {topic}
Difficulty "{level}" means: {guide}.
Rules: the student writes ONE function; only the standard library (math, collections, re, itertools, functools,
heapq, string) may be imported; no input(), no files, no printing. Tests are single-line assert statements that call
the function: at least 6, covering normal cases and edge cases (empty input, ties, one element). Compare floats
with round(x, 4) == ... .{avoid}
Reply in exactly this format:
### QUESTION
<the task: what the function takes and returns, with one worked example>
### STARTER
```python
def function_name(args):
    \"\"\"one-line description\"\"\"
    # your code here
    pass
```
### SOLUTION
```python
<complete working solution with the same signature>
```
### TESTS
```python
assert function_name(...) == ...
```
### REASONING POINTS
- [1] <one idea the student's explanation of their approach should contain>
- [1] <another>
- [1] <another>
### HINT
<one hint that does not give the solution away>""",
}
BLIND = """Course passages:
{passages}

Question: {question}
A) {a}
B) {b}
C) {c}
D) {d}

Which option is correct? Answer with the letter only."""
SCHEME_LINE = re.compile(r"^\s*[-*•]?\s*\[(\d+(?:\.\d+)?)\]\s*(.+)$")


def parse_scheme(text):
    out = []
    for line in text.splitlines():
        m = SCHEME_LINE.match(line)
        point = re.sub(r"\s*\[\d+\](\[\d+\])*", "", m.group(2)).strip() if m else ""
        if m and len(point.split()) >= 2:
            out.append({"point": point, "marks": float(m.group(1))})
    return out


def passages_for(tutor, topic, k=4, rng=random, pool_size=6):
    """k passages for the topic: sampled from the top 6 (at most 2 per document), so repeated practice on one topic
    does not always see the same text. (From the top 10, weak hits like a SQuAD page got in for "edit distance".)"""
    with LOCK:
        hits = tutor.index.search(topic, k=20, mode="hybrid")
    seen, pool = {}, []
    for h in hits:
        d = h[1]["doc_id"]
        if seen.get(d, 0) < 2:
            seen[d] = seen.get(d, 0) + 1
            pool.append(h)
        if len(pool) == pool_size:
            break
    pick = sorted(rng.sample(range(len(pool)), min(k, len(pool))))
    return [pool[i] for i in pick]


def build(kind, sec):
    """Section dict -> item body, or (None, reason)."""
    q = sec.get("QUESTION", "").strip()
    if len(q.split()) < 5:
        return None, "no question"
    if UNSEEN.search(q):
        return None, "the question refers to material the student cannot see"
    if kind == "mcq":
        opts = [sec.get(x, "").strip() for x in "ABCD"]
        key = sec.get("ANSWER", "").strip().upper()[:1]
        if not all(opts) or key not in "ABCD" or not key or len({o.lower() for o in opts}) < 4:
            return None, "options or answer letter missing"
        return {"question": q, "options": opts, "answer": "ABCD".index(key),
                "explanation": sec.get("EXPLANATION", "").strip(),
                "scheme": [{"point": "Chooses the correct option", "marks": 1.0}]}, None
    if kind == "short":
        scheme = parse_scheme(sec.get("MARKING SCHEME", ""))
        model = sec.get("MODEL ANSWER", "").strip()
        if len(scheme) < 2 or len(model.split()) < 15:
            return None, "model answer or marking scheme missing"
        return {"question": q, "model_answer": model, "scheme": scheme}, None
    starter, solution = sec.get("STARTER", ""), sec.get("SOLUTION", "")
    tests = [l.strip() for l in sec.get("TESTS", "").splitlines() if l.strip().startswith("assert ")]
    names = [re.search(r"^def\s+(\w+)", c, re.M) for c in (starter, solution)]
    if not all(names) or names[0].group(1) != names[1].group(1):
        return None, "starter and solution do not define the same function"
    if len(tests) < 3:
        return None, "fewer than 3 tests"
    reasoning = parse_scheme(sec.get("REASONING POINTS", ""))[:3]
    for p in reasoning:  # the explanation is always worth 3 marks, next to 7 for the tests
        p["marks"] = round(3.0 / len(reasoning), 2)
    return {"question": q, "function": names[0].group(1), "starter": starter.strip() + "\n", "solution": solution.strip() + "\n",
            "tests": tests, "hint": sec.get("HINT", "").strip(), "test_marks": 7.0,
            "scheme": reasoning or [{"point": "Explains the approach clearly and correctly", "marks": 3.0}]}, None


def verify(llm, kind, body, text, bodies):
    """(notes, hard_fail). notes: [{'level': ok|warn|fail, 'text'}]."""
    from .grade import grade_points
    notes = []
    if kind == "mcq":
        o = body["options"]
        got = llm.gen([{"role": "user", "content": BLIND.format(passages=text, question=body["question"], a=o[0], b=o[1],
                                                                 c=o[2], d=o[3])}], max_new=3).strip().upper()[:1]
        agree = got == "ABCD"[body["answer"]]
        notes.append({"level": "ok" if agree else "warn",
                      "text": "Answer key confirmed by a blind re-solve" if agree else
                      f"Answer key disputed: a blind re-solve chose {got or '?'}, the key says {'ABCD'[body['answer']]}"})
        if body["explanation"]:
            s = llm.support(re.sub(r"\[\d+\]", "", body["explanation"]), bodies)
            notes.append({"level": "ok" if s >= 0.3 else "warn",
                          "text": "Explanation supported by the course passages" if s >= 0.3 else
                          "Explanation not clearly supported by the course passages"})
        return notes, False
    if kind == "short":
        weak = [p["point"] for p in body["scheme"] if llm.support(p["point"], bodies) < 0.3]
        notes.append({"level": "ok" if not weak else "warn",
                      "text": "Every marking point is supported by the course passages" if not weak else
                      f"{len(weak)} marking point(s) not clearly in the course passages: " + "; ".join(weak)})
        got, total, _ = grade_points(llm, body["question"], re.sub(r"\[\d+\]", "", body["model_answer"]), body["scheme"])
        ok = got >= 0.8 * total
        notes.append({"level": "ok" if ok else "warn",
                      "text": f"Model answer earns {got:g}/{total:g} on its own scheme"
                      + ("" if ok else ": the scheme and model answer disagree")})
        return notes, False
    chk = codecheck.constraints(body["solution"])
    if chk["blocked"]:
        return [{"level": "fail", "text": "Reference solution breaks course constraints: " + chk["blocked"][0][1]}], True
    res = codecheck.run_tests(body["solution"], body["tests"])
    passed = [r["test"] for r in res if r["ok"]]
    dropped = len(res) - len(passed)
    if len(passed) < 3:
        return [{"level": "fail", "text": f"Reference solution passes only {len(passed)}/{len(res)} tests"}], True
    body["tests"] = passed
    notes.append({"level": "ok", "text": f"Reference solution passes all {len(passed)} kept tests"
                  + (f" ({dropped} wrong test(s) removed)" if dropped else "")})
    starter = codecheck.run_tests(body["starter"], passed)
    fails = sum(not r["ok"] for r in starter)
    notes.append({"level": "ok" if fails else "warn",
                  "text": "Starter code fails the tests, as it should" if fails else
                  "Starter code already passes every test: the exercise gives the answer away"})
    if chk["warnings"]:
        notes.append({"level": "warn", "text": "Reference solution style: " + chk["warnings"][0][1]})
    return notes, False


def generate(llm, tutor, topic, kind, level, previous=(), attempts=3, rng=random):
    """-> (body, checks, sources) or raises ValueError with a reason the user can read."""
    if kind not in KINDS or level not in LEVELS:
        raise ValueError("unknown question type or difficulty")
    with LOCK:
        tutor.history = []
        if not tutor.on_topic(topic):
            raise ValueError("That topic is outside this NLP course. Try something like 'n-gram smoothing' or 'attention'.")
    hits = passages_for(tutor, topic, rng=rng)
    if not hits:
        raise ValueError("No course material found for that topic.")
    text, sources, bodies = format_passages(hits)
    avoid = AVOID.format(prev="\n".join(f"- {p}" for p in list(previous)[-6:])) if previous else ""
    guide = (CODE_GUIDE if kind == "code" else GUIDE)[level]
    prompt = PROMPTS[kind].format(passages=text, topic=topic, level=level, guide=guide, avoid=avoid)
    last = "no attempt"
    for n in range(attempts):
        raw = llm.gen([{"role": "user", "content": prompt}], max_new=900 if kind == "code" else 600, sample=n > 0)
        body, why = build(kind, sections(raw))
        if body is None:
            last = why
            continue
        notes, hard = verify(llm, kind, body, text, bodies)
        if hard:
            last = notes[0]["text"]
            continue
        for s, b in zip(sources, bodies):
            s["text"] = b.split("\n", 1)[-1]
        return body, {"ok": all(x["level"] == "ok" for x in notes), "notes": notes, "attempts": n + 1}, sources
    raise ValueError(f"Couldn't write a reliable question after {attempts} tries ({last}). Try again or change the topic.")
