"""Smoke test for the bge-m3 index: does a known source appear in the top k?

Run from the project root:  python scripts/retrieval_test.py [-k 5]
Searches open + restricted (--private) since most papers are restricted.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search import Index  # noqa: E402

QUERIES = Path(__file__).resolve().parents[1] / "eval" / "retrieval_queries.jsonl"

# query -> any of these doc_ids counts as a hit
CASES = [
    ("scaled dot-product attention divide by square root of d_k", {"arxiv-1706.03762", "slp3-2026aug-textbook-ed3book_aug26"}),
    ("Kneser-Ney smoothing continuation probability", {"slp3-2026aug-textbook-ed3book_aug26"}),
    ("skip-gram negative sampling objective", {"paper-mikolov-et-al-2013-neurips-word2vec-negative-sampling", "arxiv-1301.3781"}),
    ("low-rank adaptation of weight matrices for fine-tuning", {"arxiv-2106.09685"}),
    ("byte pair encoding for rare words in machine translation", {"arxiv-1508.07909"}),
    ("training a reward model from human preference comparisons", {"arxiv-2203.02155"}),
    ("retrieval-augmented generation with a dense passage retriever", {"arxiv-2005.11401"}),
    ("layer normalization statistics computed over hidden units", {"arxiv-1607.06450"}),
    ("rotary position embedding multiplies queries and keys by rotation matrix", {"arxiv-2104.09864"}),
    ("grouped-query attention share key value heads", {"arxiv-2305.13245"}),
    ("masked language modeling next sentence prediction pretraining", {"arxiv-1810.04805"}),
    ("GloVe global co-occurrence matrix factorization", {"paper-glove"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["dense", "hybrid", "rerank"])
    ap.add_argument("--queries", default=str(QUERIES), help="jsonl of {topic, query, relevant}; 'builtin' = the 12 CASES")
    a = ap.parse_args()
    if a.queries == "builtin":
        cases = [("-", q, sorted(w)) for q, w in CASES]
    else:
        cases = [(r["topic"], r["query"], r["relevant"]) for r in map(json.loads, open(a.queries, encoding="utf-8"))]
    idx = Index(private=True)
    summary, by_topic = {}, {}
    for mode in a.modes:
        print(f"=== {mode}")
        hits, rr, h1, srr, t0 = 0, 0.0, 0, 0.0, time.time()
        for topic, q, want in cases:
            res = idx.search(q, k=a.k, mode=mode)
            ranks = [i for i, (_, m, _) in enumerate(res, 1) if m["doc_id"] in want]
            ok = bool(ranks)
            hits += ok
            h1 += ok and ranks[0] == 1
            rr += 1 / ranks[0] if ok else 0
            # strict: only the first-listed (canonical) source counts
            strict = [i for i, (_, m, _) in enumerate(res, 1) if m["doc_id"] == want[0]]
            srr += 1 / strict[0] if strict else 0
            t = by_topic.setdefault(topic, {}).setdefault(mode, [0.0, 0])
            t[0] += 1 / ranks[0] if ok else 0
            t[1] += 1
            print(f"{'HIT ' if ok else 'MISS'} r={ranks[0] if ok else '-'}  {q[:55]:55}  top1: {res[0][1]['doc_id']}")
        summary[mode] = (h1, hits, rr / len(cases), srr / len(cases), (time.time() - t0) / len(cases))
    n = len(cases)
    print(f"\n{'mode':8} hit@1   hit@{a.k}   MRR  strictMRR  s/query   ({n} queries; s/query includes model loading)")
    for mode, (h1, hk, mrr, smrr, sec) in summary.items():
        print(f"{mode:8} {h1:2}/{n}  {hk:2}/{n}  {mrr:.2f}   {smrr:.2f}      {sec:.2f}")
    if len(by_topic) > 1:
        print("\nMRR by topic:              " + "  ".join(f"{m:>6}" for m in a.modes))
        for topic, d in sorted(by_topic.items()):
            print(f"  {topic:26} " + "  ".join(f"{d[m][0] / d[m][1]:6.2f}" for m in a.modes) + f"   (n={d[a.modes[0]][1]})")


if __name__ == "__main__":
    main()
