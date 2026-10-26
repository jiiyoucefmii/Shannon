"""Hint-first tutoring examples, distilled from the base model's own hint-mode answers.

Run from the project root:  python scripts/build_hint_sft.py [--generated 300] [--seed 2]
Writes corpus/splits/tutor_sft/rag/hint.jsonl (already in the tutor's inference format).

Why (2026-10-22): the answer-based test scores hint mode by "ends with a guiding question, <= 120 words". The base
model does that 97% of the time; both fine-tuned pilots dropped to 41-50%, and the training data had no example of
exactly that behaviour (synthetic dialogues end with a question only 39% of the time). So the base model answers
NLP questions in hint mode, with the real tutor prompt, and only answers passing every check are kept.

Questions: the 107 HF quiz questions asked openly (no options), plus questions the model writes from other
open docs/train passages. Passages come only from open/noncommercial docs/train chunks, as in rag_format_sft.py.
Kept only if: ends with a question, <= 100 words, >= 1 citation, <= 1 unsourced mark, no talk about "the passage".
The merge step (merge_tutor_sft.py) then removes anything overlapping the evaluation sets.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from generate_dialogues import META, pick_chunks  # noqa: E402
from search import Index  # noqa: E402
from tutor import CITE_ONE, STYLES, SYSTEM, Tutor, format_passages  # noqa: E402

SFT = ROOT / "corpus" / "splits" / "tutor_sft"
RANK = {"open": 0, "noncommercial": 1}
QGEN = """Here is an excerpt from NLP course material.

{passage}

Write ONE question a student taking an NLP course might ask, which this excerpt helps answer. Ask about the concept
in your own words, as a student would (no "according to the text"). Output only the question."""


def gen(tutor, messages, max_new, sample=True):
    import torch
    x = tutor.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt", enable_thinking=False).to(tutor.model.device)
    with torch.no_grad():
        o = tutor.model.generate(**x, max_new_tokens=max_new, do_sample=sample, temperature=0.6, top_p=0.9,
                                 repetition_penalty=1.05)
    return tutor.proc.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--max-questions", type=int, help="smoke test: stop after this many questions")
    ap.add_argument("--out", default="hint")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    targs = argparse.Namespace(style="hint", open_only=True, k=6, search="hybrid", model="Qwen/Qwen3.5-4B",
                               max_new_tokens=None, no_cite_check=False, no_rag=False, llm_verify=False, adapter=None)
    tutor = Tutor(targs)  # base model; its own index is replaced below by the training-safe one
    allowed = {json.loads(l)["chunk_id"] for l in open(ROOT / "corpus/splits/docs/train.jsonl", encoding="utf-8")}
    idx = Index(private=False)

    def passages_for(q):
        seen, hits = {}, []
        for h in idx.search(q, k=40, mode="hybrid"):
            m = h[1]
            if m["chunk_id"] in allowed and seen.get(m["doc_id"], 0) < 2:
                seen[m["doc_id"]] = seen.get(m["doc_id"], 0) + 1
                hits.append(h)
                if len(hits) == 6:
                    break
        return hits

    questions = []
    quiz = [json.loads(l) for l in open(SFT / "quiz.jsonl", encoding="utf-8")]
    questions += sorted({r["question_key"] for r in quiz})
    for ch in pick_chunks(a.generated, a.seed):
        q = gen(tutor, [{"role": "user", "content": QGEN.format(passage=f"{ch['title']}\n\n{ch['text']}")}], 60)
        q = q.strip().strip('"').split("\n")[0]
        if q.endswith("?") and 5 <= len(q.split()) <= 30 and not META.search(q):
            questions.append(q)
    questions = questions[: a.max_questions] if a.max_questions else questions
    print(len(questions), "questions", flush=True)

    st = Counter()
    out = SFT / "rag" / f"{a.out}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, q in enumerate(questions, 1):
            hits = passages_for(q)
            text, sources, bodies = format_passages(hits)
            system = SYSTEM.format(style=STYLES["hint"], passages=text)
            raw = gen(tutor, [{"role": "system", "content": system}, {"role": "user", "content": q}], 260)
            ans, log = tutor.checker.check(raw, bodies)
            last = re.split(r"(?<=[.!?])\s+", CITE_ONE.sub("", ans).strip())[-1] if ans.strip() else ""
            checks = {"ends-with-question": last.rstrip("*_ ").endswith("?"), "short": len(ans.split()) <= 100,
                      "cited": bool(CITE_ONE.search(ans)), "few-unsourced": ans.count("not from the course material") <= 1,
                      "no-meta": not META.search(ans)}
            failed = [k for k, v in checks.items() if not v]
            st["kept" if not failed else "dropped:" + failed[0]] += 1
            if not failed:
                tier = max([s["license_tier"] for s in sources] + ["open"], key=lambda t: RANK.get(t, 2))
                f.write(json.dumps({"task": "hint-first", "question_key": q, "style": "hint", "license_tier": tier,
                                    "license": "derived (generated by Qwen3.5-4B from the cited passages)",
                                    "source_id": "distilled-hint-20261022", "passage_docs": [s["doc_id"] for s in sources],
                                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": q},
                                                 {"role": "assistant", "content": ans}]}, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(i, dict(st), flush=True)
    print("done", dict(st))


if __name__ == "__main__":
    main()
