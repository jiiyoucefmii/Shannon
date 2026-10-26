"""Dump every factual sentence of a RAG eval run with its best-supporting passage and reranker score.

Run from the project root:  python eval/citation_pairs.py eval/tutor_eval_runs/<stamp>-rag.jsonl
Writes eval/citation_pairs.jsonl. Label a sample of these by hand (eval/citation_labels.jsonl) to choose
the checker thresholds: a pair is "supported" if the passage itself states the sentence's claim.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))
from calibrate_citations import retrieve  # noqa: E402
from search import Index  # noqa: E402
from tutor import CHAT, CITE_ONE, CitationChecker, format_passages, sentences  # noqa: E402

MARK = "*(not from the course material)*"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    idx, chk = Index(private=True), CitationChecker()
    out = []
    for r in map(json.loads, open(sys.argv[1], encoding="utf-8")):
        if not r["sources"]:
            continue
        _, sources, bodies = format_passages(retrieve(idx, r["question"]))
        if [s["doc_id"] for s in sources] != r["sources"]:
            print("retrieval changed, skipping:", r["question"][:60]); continue
        ans = r["answer"]
        for s, e in sentences(ans):
            sent = ans[s:e]
            claim = CITE_ONE.sub("", sent).replace(MARK, "").strip()
            if len(claim.split()) < 6 or claim.rstrip("*_ ").endswith("?") or CHAT.search(claim):
                continue
            scores = chk.ce.predict([(claim, p) for p in bodies], batch_size=8)
            best = int(scores.argmax())
            out.append({"question": r["question"], "claim": claim, "state": "marked" if MARK in sent else
                        ("cited" if CITE_ONE.search(sent) else "uncited"),
                        "cited": [int(n) for n in CITE_ONE.findall(sent)], "best": best + 1,
                        "best_score": round(float(scores[best]), 3), "passage": bodies[best][:700]})
    with open(ROOT / "eval" / "citation_pairs.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    import collections
    print(len(out), "claims;", collections.Counter(o["state"] for o in out))
    bands = collections.Counter(min(int(o["best_score"] * 10), 9) / 10 for o in out)
    print("best-score bands:", dict(sorted(bands.items())))


if __name__ == "__main__":
    main()
