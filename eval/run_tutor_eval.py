"""Score the RAG tutor on eval/tutor_eval.jsonl: key-point coverage, citations, length, speed.

Run from the project root:  python eval/run_tutor_eval.py [--limit N] [--modes rag norag]

  rag     scripts/tutor.py as-is (explain style, hybrid retrieval, citation check)
  norag   same model and prompt with no passages: shows what retrieval adds

Key-point coverage is judged by the same local model with a narrow yes/no question per key point
("does the answer state this?"). A 4B judge is noisy and may favour its own wording, so treat absolute numbers
as rough; the rag-vs-norag difference is the more trustworthy signal. Results: eval/tutor_eval_runs/.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tutor import CITE_ONE, Tutor  # noqa: E402

JUDGE = """Student question: {q}

Tutor answer:
{a}

Key point: {k}

Does the tutor answer state or clearly imply this key point? If the key point says what the answer should NOT do,
answer yes only if the answer avoids it. Answer only yes or no."""


def judge(tutor, q, answer, kp):
    import torch
    clean = CITE_ONE.sub("", answer).replace("*(not from the course material)*", "")
    msg = [{"role": "user", "content": JUDGE.format(q=q, a=clean, k=kp)}]
    inputs = tutor.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True,
                                            return_tensors="pt", enable_thinking=False).to(tutor.model.device)
    with torch.no_grad():
        out = tutor.model.generate(**inputs, max_new_tokens=3, do_sample=False)
    return tutor.proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower().startswith("yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--modes", nargs="+", default=["rag", "norag"])
    ap.add_argument("--resume", metavar="STAMP", help="continue the run with this timestamp prefix")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    items = [json.loads(l) for l in open(ROOT / "eval" / "tutor_eval.jsonl", encoding="utf-8")][: a.limit]
    targs = argparse.Namespace(style="explain", open_only=False, k=6, search="hybrid", model="Qwen/Qwen3.5-4B",
                               max_new_tokens=None, no_cite_check=False, no_rag=False, llm_verify=False)
    tutor = Tutor(targs)
    out_dir = ROOT / "eval" / "tutor_eval_runs"
    out_dir.mkdir(exist_ok=True)
    stamp = a.resume or time.strftime("%Y%m%d-%H%M%S")
    summary = {}
    import torch
    for mode in a.modes:
        targs.no_rag = mode == "norag"
        path = out_dir / f"{stamp}-{mode}.jsonl"
        # rows are appended as they finish, so a crash loses at most one question; --resume STAMP continues
        rows = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
        done = {r["question"] for r in rows}
        for i, it in enumerate(items, 1):
            if it["question"] in done:
                continue
            torch.cuda.empty_cache()
            tutor.history = []
            answer, sources, stats = tutor.ask(it["question"])
            hits = [judge(tutor, it["question"], answer, k) for k in it["key_points"]]
            log = stats.get("citation_check", [])
            rows.append({**it, "mode": mode, "answer": answer, "raw_answer": stats.get("raw_answer", answer), "kp_hits": hits, "coverage": sum(hits) / len(hits),
                         "n_citations": len(CITE_ONE.findall(answer)),
                         "n_unsourced": answer.count("not from the course material"),
                         "cite_actions": [c["action"] for c in log], "words": len(answer.split()),
                         "seconds": stats["gen_seconds"], "hit_token_limit": stats.get("hit_token_limit"),
                         "sources": [s["doc_id"] for s in sources]})
            print(f"[{mode} {i}/{len(items)}] cov {sum(hits)}/{len(hits)}  cites {rows[-1]['n_citations']}  "
                  f"{rows[-1]['words']}w  {stats['gen_seconds']}s  {it['question'][:60]}", flush=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")
        n = len(rows)
        by_src = {}
        for src in ("tutoreval", "author"):
            rs = [r for r in rows if r["source"] == src]
            if rs:
                by_src[src] = round(sum(r["coverage"] for r in rs) / len(rs), 3)
        summary[mode] = {
            "questions": n,
            "key_point_coverage": round(sum(r["coverage"] for r in rows) / n, 3),
            "coverage_by_source": by_src,
            "answers_with_citation": sum(r["n_citations"] > 0 for r in rows),
            "mean_citations": round(sum(r["n_citations"] for r in rows) / n, 2),
            "sentences_marked_unsourced": sum(r["n_unsourced"] for r in rows),
            "citations_removed_or_moved": sum(a.startswith(("removed", "moved")) for r in rows for a in r["cite_actions"]),
            "mean_words": round(sum(r["words"] for r in rows) / n),
            "hit_token_limit": sum(bool(r["hit_token_limit"]) for r in rows),
            "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 1),
        }
    json.dump(summary, open(out_dir / f"{stamp}-summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
