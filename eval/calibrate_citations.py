"""Print the support score of every citation in a tutor run log, for choosing the checker threshold.

Run from the project root:  python eval/calibrate_citations.py eval/tutor_runs/<run>.jsonl
Re-retrieves each question's passages (hybrid search is deterministic), so needs no LLM.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search import Index  # noqa: E402
from tutor import CITE_ONE, CitationChecker, format_passages, sentences  # noqa: E402


def retrieve(index, q, k=6):
    seen, out = {}, []
    for h in index.search(q, k=k * 2, mode="hybrid"):
        d = h[1]["doc_id"]
        if seen.get(d, 0) < 2:
            seen[d] = seen.get(d, 0) + 1
            out.append(h)
        if len(out) == k:
            break
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    idx, chk = Index(private=True), CitationChecker()
    for r in map(json.loads, open(sys.argv[1], encoding="utf-8")):
        _, sources, bodies = format_passages(retrieve(idx, r["question"]))
        assert [s["doc_id"] for s in sources] == [s["doc_id"] for s in r["sources"]], "retrieval changed since run"
        print("\n=== Q:", r["question"])
        for s, e in sentences(r["answer"]):
            sent = r["answer"][s:e]
            cited = [int(n) for n in CITE_ONE.findall(sent)]
            if not cited:
                continue
            claim = CITE_ONE.sub("", sent).strip()
            scores = chk.ce.predict([(claim, p) for p in bodies], batch_size=8)
            best = int(scores.argmax()) + 1
            for n in cited:
                print(f"  [{n}] score={scores[n - 1]:.3f}  best=[{best}] {scores[best - 1]:.3f}  | {claim[:110]}")
                print(f"        cited: {bodies[n - 1][:110]!r}")


if __name__ == "__main__":
    main()
