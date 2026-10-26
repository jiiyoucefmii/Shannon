"""Turn the Hugging Face course quizzes (<Question> blocks in the .mdx sources) into tutoring SFT items.

Run from the project root:  python scripts/build_quiz_sft.py
Writes corpus/splits/tutor_sft/quiz.jsonl (not yet split or decontaminated; scripts/merge_tutor_sft.py does that).

Per quiz question:
  mcq-explain       question + lettered options -> correct option(s) with the course's own explanation
  answer-feedback   student picks a wrong option -> the course's explanation of why it is wrong, then a nudge
Source text is Apache-2.0 (huggingface/course, huggingface/agents-course); only manifest-registered files are used.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "corpus" / "splits" / "tutor_sft"
SYS = ("You are a patient NLP tutor. When a student answers, say whether they are right, explain why in plain "
       "language, and if they are wrong, help them reconsider instead of just giving the answer.")
# string literals may contain escaped quotes (explain: "Look at <a href=\"...\">"), so \\. is matched as one unit
OPT = re.compile(r"text:\s*([\"'`])((?:\\.|(?!\1).)*)\1\s*,\s*explain:\s*([\"'`])((?:\\.|(?!\3).)*)\3"
                 r"(\s*,\s*correct:\s*true)?", re.S)
VERDICT = re.compile(r"^(correct|right|exactly|yes|nope|no|wrong|false|true|not quite|incorrect)\b[!.,:]*\s*", re.I)


# Varied wording, chosen deterministically per item: the first pilot memorised one fixed closing sentence.
OPEN = ["Not quite.", "Close, but not this one.", "That's a common pick, but it isn't right.", "Not this time.",
        "Good try, but no.", "I can see why you'd think that, but it's not the answer.", "Hmm, not quite.",
        "That one's tempting, but it's wrong.", "Not exactly.", "That's not it, though it's a reasonable guess."]
CLOSE = ["Which option fits better now?", "Want to look at the options again?", "With that in mind, which would you pick?",
         "Can you see which option matches that?", "Try again: which one fits?", "What would you choose instead?",
         "Does that point you to a different option?", "Which of the others holds up?", "Have another go.",
         "Which option survives that check?"]
ASK = ["Question: {q}\nOptions:\n{listing}\n\nI think the answer is: {t}", "{q}\n{listing}\n\nIs it \"{t}\"?",
       "For this one:\n{q}\n{listing}\nI picked: {t}", "{q}\n{listing}\nMy answer: {t}. Right?",
       "Quiz question: {q}\n{listing}\n\nI'd go with \"{t}\"."]
# course-voice: the source is a course, the tutor is not ("as you saw in this chapter", the hugging-face emoji)
VOICE = [(re.compile(r"\b[Aa]s (?:you )?(?:saw|seen|learned|mentioned) in (?:this|the previous|an earlier) "
                     r"(?:chapter|section|unit|lesson),?\s*"), ""),
         (re.compile(r"\s*\(?\b(?:see|in) (?:this|the previous|the next) (?:chapter|section)\)?"), ""),
         (re.compile("\U0001F917\\s*"), "Hugging Face ")]


def pick(options, key):
    import hashlib
    return options[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(options)]


def devoice(s):
    for rx, rep in VOICE:
        s = rx.sub(rep, s)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s.strip())


def clean(s):
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s)
    s = re.sub(r"<a href=['\"]([^'\"]+)['\"]>(.*?)</a>", r"\2 (\1)", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub("\U0001F917\\s*", "Hugging Face ", s)  # the course's emoji brand, also in question/option text
    return re.sub(r"\s+", " ", s.replace("\\'", "'").replace('\\"', '"')).strip()


def quizzes():
    man = {r["source_url"].split("/blob/", 1)[1].split("/", 1)[1]: r
           for r in map(json.loads, open(ROOT / "corpus" / "manifest.jsonl", encoding="utf-8"))
           if r["id"].startswith(("hfcourse-", "agents-course-"))}
    for rel, r in sorted(man.items()):
        repo = "huggingface__course" if r["id"].startswith("hfcourse-") else "huggingface__agents-course"
        path = ROOT / "corpus" / "raw" / "repos" / repo / rel
        if not path.exists() or path.suffix != ".mdx":
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"<Question\b.*?/>", text, re.S):
            head = list(re.finditer(r"^#{2,4}\s*(?:\d+\.\s*)?(.+?)\s*$", text[:m.start()], re.M))
            if not head:
                continue
            q = clean(re.sub(r"\[\[.*?\]\]", "", head[-1].group(1)))
            opts = [(clean(t), clean(e), bool(c)) for _, t, _, e, c in OPT.findall(m.group(0))]
            if len(opts) < 2 or not any(c for *_, c in opts) or not q:
                continue
            yield r, q, opts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items, n_q = [], 0
    letters = "ABCDEFGH"
    for r, q, opts in quizzes():
        n_q += 1
        base = {"source_id": r["id"], "license_tier": r["license_tier"], "license": r["license"], "question_key": q}
        listing = "\n".join(f"{letters[i]}. {t}" for i, (t, _, _) in enumerate(opts))
        multi = sum(c for *_, c in opts) > 1
        correct = [(letters[i], t, e) for i, (t, e, c) in enumerate(opts) if c]
        ans = ("Answers: " if multi else "Answer: ") + ", ".join(f"{l}. {t}" for l, t, _ in correct)
        # keep explanations that say something beyond the verdict ("Correct!", "That's right!")
        why = " ".join(x for x in (VERDICT.sub("", e) for _, _, e in correct if e) if len(x.split()) >= 6)
        why = devoice(why)
        items.append({**base, "task": "mcq-explain", "messages": [
            {"role": "user", "content": f"{q}\n{listing}" + ("\n(More than one answer may be correct.)" if multi else "")},
            {"role": "assistant", "content": ans + (f"\nWhy: {why}" if why else "")}]})
        for t, e, c in opts:
            if c or not e or len(e.split()) < 4:
                continue  # feedback items need a real explanation of the mistake
            key = f"{q}|{t}"
            items.append({**base, "task": "answer-feedback", "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": pick(ASK, key + "a").format(q=q, listing=listing, t=t)},
                {"role": "assistant", "content": f"{pick(OPEN, key + 'o')} {devoice(VERDICT.sub('', e))} {pick(CLOSE, key + 'c')}"}]})
    with open(OUT / "quiz.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(n_q, "quiz questions ->", len(items), "items", dict(Counter(i["task"] for i in items)))


if __name__ == "__main__":
    main()
