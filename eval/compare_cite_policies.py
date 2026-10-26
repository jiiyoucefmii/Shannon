"""Apply several citation-checker policies to the SAME raw tutor answers and compare what each does.

Run from the project root:  python eval/compare_cite_policies.py eval/tutor_eval_runs/<stamp>-rag.jsonl
Needs rows with `raw_answer` (saved by run_tutor_eval.py since 2026-10-19). Loads the tutor LLM once for the
hybrid policy's verifier. Re-retrieves passages (deterministic) and checks they match the run.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))
from calibrate_citations import retrieve  # noqa: E402
from tutor import CitationChecker, Tutor, format_passages  # noqa: E402

POLICIES = {  # name -> (lo, hi, use_llm)
    "reranker>=0.5": (0.5, 0.5, False),
    "reranker>=0.3": (0.3, 0.3, False),
    "hybrid 0.3/LLM/0.7": (0.3, 0.7, True),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    targs = argparse.Namespace(style="explain", open_only=False, k=6, search="hybrid", model="Qwen/Qwen3.5-4B",
                               max_new_tokens=None, no_cite_check=True, no_rag=False)
    tutor = Tutor(targs)
    base = CitationChecker()  # load the cross-encoder once and share it
    checkers = {}
    for name, (lo, hi, llm) in POLICIES.items():
        c = CitationChecker.__new__(CitationChecker)
        c.ce, c.lo, c.hi, c.verify = base.ce, lo, hi, (tutor.verify_claim if llm else None)
        checkers[name] = c
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    stats = {n: Counter() for n in POLICIES}
    out = []
    for r in rows:
        if not r["sources"]:
            continue
        _, sources, bodies = format_passages(retrieve(tutor.index, r["question"]))
        if [s["doc_id"] for s in sources] != r["sources"]:
            print("retrieval changed, skipping:", r["question"][:60]); continue
        rec = {"question": r["question"]}
        for name, chk in checkers.items():
            ans, log = chk.check(r["raw_answer"], bodies)
            for c in log:
                stats[name][c["action"].split("->")[0]] += 1
            stats[name]["answers"] += 1
            rec[name] = ans
        out.append(rec)
    with open(Path(sys.argv[1]).with_name(Path(sys.argv[1]).stem + "-policies.jsonl"), "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    keys = ["kept", "moved", "removed", "removed-bare", "added", "marked-unsourced"]
    print(f"\n{'policy':22}" + "".join(f"{k:>18}" for k in keys))
    for name, st in stats.items():
        print(f"{name:22}" + "".join(f"{st[k]:>18}" for k in keys))


if __name__ == "__main__":
    main()
