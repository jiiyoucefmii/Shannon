"""Split + decontaminate the NLP tutoring SFT data into corpus/splits/tutor_sft/{train,val}.jsonl.

Run from the project root:  python scripts/merge_tutor_sft.py
Inputs: corpus/splits/tutor_sft/quiz.jsonl and every dialogues-*.jsonl (not the *-rejected files).
  quiz items       split by question (10% val), so one question's MCQ and feedback items stay on the same side
  dialogues        keep the split of their source chunk (docs/train vs docs/val)
Dropped if they share a 13-word sequence with: the held-out benchmark files (same index as build_splits.py) or our
own evaluation sets (eval/tutor_eval.jsonl questions + key points, eval/retrieval_queries.jsonl, eval/tutor_questions.txt).
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from build_splits import bucket, build_eval_index, eval_files, grams, manifest  # noqa: E402

OUT = ROOT / "corpus" / "splits" / "tutor_sft"


def own_eval_grams():
    texts = []
    for r in map(json.loads, open(ROOT / "eval" / "tutor_eval.jsonl", encoding="utf-8")):
        texts += [r["question"], *r["key_points"]]
    for r in map(json.loads, open(ROOT / "eval" / "retrieval_queries.jsonl", encoding="utf-8")):
        texts.append(r["query"])
    texts += [l.strip() for l in open(ROOT / "eval" / "tutor_questions.txt", encoding="utf-8") if not l.startswith("#")]
    probes = ROOT / "eval" / "behaviour_probes.jsonl"
    if probes.exists():
        texts += [r["question"] for r in map(json.loads, open(probes, encoding="utf-8"))]
    g = set()
    for t in texts:
        g |= grams(t)
    return g  # mostly short texts: those under 13 words add nothing, so also check exact question matches


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # inference-format items from scripts/rag_format_sft.py (system prompt with retrieved passages)
    items = []
    for name in ("quiz", "dialogues", "behaviour", "hint"):
        items += [json.loads(l) for l in open(OUT / "rag" / f"{name}.jsonl", encoding="utf-8")]
    print("building held-out index ...")
    ev, bit = build_eval_index(eval_files(manifest()))
    own = own_eval_grams()
    own_q = {r["question"].lower().strip() for r in map(json.loads, open(ROOT / "eval" / "tutor_eval.jsonl", encoding="utf-8"))}
    probes = ROOT / "eval" / "behaviour_probes.jsonl"
    if probes.exists():
        own_q |= {r["question"].lower().strip() for r in map(json.loads, open(probes, encoding="utf-8"))}
    out = {"train": [], "val": []}
    st = Counter()
    for it in items:
        # only the conversation is checked: the system prompt's passages are docs/train chunks, already
        # decontaminated, and textbook passages naturally share phrases with our own key points
        text = "\n".join(m["content"] for m in it["messages"] if m["role"] != "system")
        g = grams(text)
        if any(ev.get(x) for x in g):
            st[f"{it['task']}:dropped-benchmark"] += 1; continue
        # long eval questions: substring match; short ones ("hello", "help") would match everything, so they must
        # equal a student message exactly
        user_msgs = {m["content"].lower().strip() for m in it["messages"] if m["role"] == "user"}
        if g & own or any((q in text.lower()) if len(q.split()) >= 6 else (q in user_msgs) for q in own_q):
            st[f"{it['task']}:dropped-own-eval"] += 1; continue
        split = (("val" if bucket(it["question_key"], 0.10) else "train") if "question_key" in it
                 else ("val" if it.get("source_split") == "val" else "train"))
        out[split].append(it)
        st[f"{it['task']}:{split}"] += 1
    for split, rows in out.items():
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"counts": dict(sorted(st.items())), "tiers": {s: dict(Counter(r["license_tier"] for r in rows))
                                                           for s, rows in out.items()}}
    json.dump(report, open(OUT / "report.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
