"""Put the tutoring SFT items into the exact prompt format the tutor uses at inference.

Run from the project root:  python scripts/rag_format_sft.py
Inputs (corpus/splits/tutor_sft/): quiz.jsonl, dialogues-*.jsonl (kept, not *-rejected), behaviour.jsonl
Output: corpus/splits/tutor_sft/rag/{quiz,dialogues,behaviour}.jsonl

Why: the first pilot trained on a short system prompt with no passages, while the tutor prompts with six numbered
passages and citation rules; on that unfamiliar input the adapter copied a passage ("Welcome to our Open R1
course"). Here every item gets tutor.py's SYSTEM prompt with real retrieved passages, and its tutor turns get the
citations tutor.py's checker would add, so training and inference see the same thing.

- passages come ONLY from open/noncommercial chunks in the decontaminated docs/train split (never restricted text,
  never a validation document); at most 2 per document, 6 in total, hybrid search, as tutor.py does
- style: explain for mcq-explain, hint for everything else
- behaviour items (refusal, greeting, not-covered) keep their reply unchanged: they must not cite anything;
  their passages come from `retrieve_for` (for not-covered: an unrelated query, so the passages cannot answer)
- licence tier = the most restrictive of the item and the passages it now contains
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from search import Index  # noqa: E402
from tutor import STYLES, SYSTEM, CitationChecker, format_passages  # noqa: E402

SFT = ROOT / "corpus" / "splits" / "tutor_sft"
RANK = {"open": 0, "noncommercial": 1}
NO_CITE = {"refusal", "greeting", "not-covered"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    allowed = {json.loads(l)["chunk_id"] for l in open(ROOT / "corpus/splits/docs/train.jsonl", encoding="utf-8")}
    idx = Index(private=False)  # open + noncommercial index only
    chk = CitationChecker()
    out_dir = SFT / "rag"
    out_dir.mkdir(exist_ok=True)

    def passages_for(query):
        seen, hits = {}, []
        for h in idx.search(query, k=40, mode="hybrid"):
            m = h[1]
            if m["chunk_id"] not in allowed or seen.get(m["doc_id"], 0) >= 2:
                continue
            seen[m["doc_id"]] = seen.get(m["doc_id"], 0) + 1
            hits.append(h)
            if len(hits) == 6:
                break
        return hits

    inputs = {"quiz": [SFT / "quiz.jsonl"], "behaviour": [SFT / "behaviour.jsonl"],
              "dialogues": [Path(f) for f in sorted(glob.glob(str(SFT / "dialogues-*.jsonl"))) if not f.endswith("-rejected.jsonl")]}
    for name, files in inputs.items():
        st = Counter()
        rows = [json.loads(l) for f in files for l in open(f, encoding="utf-8")]
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as fo:
            for r in rows:
                msgs = [m for m in r["messages"] if m["role"] != "system"]
                first_user = next(m["content"] for m in msgs if m["role"] == "user")
                hits = passages_for(r.get("retrieve_for") or first_user)
                text, sources, bodies = format_passages(hits)
                style = "explain" if r["task"] == "mcq-explain" else "hint"
                new = [{"role": "system", "content": SYSTEM.format(style=STYLES[style], passages=text)}]
                for m in msgs:
                    if m["role"] == "assistant" and r["task"] not in NO_CITE and bodies:
                        cited, log = chk.check(m["content"], bodies)
                        for c in log:
                            st[c["action"].split("->")[0]] += 1
                        m = {**m, "content": cited}
                    new.append(m)
                tiers = [r["license_tier"]] + [s["license_tier"] for s in sources]
                tier = max(tiers, key=lambda t: RANK.get(t, 2))
                fo.write(json.dumps({**r, "messages": new, "style": style, "license_tier": tier,
                                     "passage_docs": [s["doc_id"] for s in sources]}, ensure_ascii=False) + "\n")
                st["items"] += 1
                st[f"tier:{tier}"] += 1
        print(name, dict(st), flush=True)


if __name__ == "__main__":
    main()
