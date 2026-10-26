"""Answer-based evaluation of a tutor model (base or base + LoRA adapter). Loss is not enough: the first pilot
cut tutoring loss by two-thirds while some answers got worse.

Run from the project root:
  python eval/run_adapter_eval.py --name base
  python eval/run_adapter_eval.py --name pilot2 --adapter models/adapters/pilot2
Results: eval/adapter_eval/<name>.jsonl (every answer) + <name>-summary.json; compare with --compare a b c.

Measures (all with retrieval, as the tutor runs):
  key-point coverage   40 questions of eval/tutor_eval.jsonl in explain mode; the GRADER is always the base
                       model (adapter switched off), so an adapter cannot grade itself
  hint-first rate      the 32 NLP questions in hint mode: answer ends with a guiding question, <= 120 words
  behaviour            eval/behaviour_probes.jsonl (held out from training): off-topic -> refused; greeting ->
                       no citations, no course voice, asks what to study; not covered -> says so, cites nothing
  copying              answers containing an 8-word phrase that recurs in >= 3 training targets
  plus mean words, unsourced marks per answer, seconds per answer
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))
from tutor import CITE_ONE, Tutor  # noqa: E402

OUT = ROOT / "eval" / "adapter_eval"
MARK = "not from the course material"
REFUSED = re.compile(r"outside (this|the) (nlp )?course|off-?topic|can'?t (help|answer) (with )?(that|this)|"
                     r"not something i can help|only set up to help|isn'?t about nlp|i'?ll pass", re.I)
VOICE = re.compile(r"welcome to|our (course|class)|open r1|this (chapter|course|unit)|in the next (section|chapter)", re.I)
NOT_COV = re.compile(r"(don'?t|doesn'?t|do not|does not) (cover|have|mention|include)|not (in|covered|something)|"
                     r"won'?t guess|rather not (guess|make)|can'?t find|no (source|information)|isn'?t in|not able to", re.I)


def words8(t):
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + 8]) for i in range(len(w) - 7)}


def template_phrases():
    seen = Counter()
    for l in open(ROOT / "corpus/splits/tutor_sft/train.jsonl", encoding="utf-8"):
        r = json.loads(l)
        for m in r["messages"]:
            if m["role"] == "assistant":
                for g in words8(CITE_ONE.sub("", m["content"])):
                    seen[g] += 1
    return {g for g, c in seen.items() if c >= 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--adapter")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--compare", nargs="+", help="print a table of these names' summaries and exit")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.compare:
        rows = {n: json.load(open(OUT / f"{n}-summary.json", encoding="utf-8")) for n in a.compare}
        keys = list(next(iter(rows.values())).keys())
        print(f"{'measure':34}" + "".join(f"{n:>14}" for n in rows))
        for k in keys:
            print(f"{k:34}" + "".join(f"{str(rows[n].get(k)):>14}" for n in rows))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    from run_tutor_eval import judge
    targs = argparse.Namespace(style="explain", open_only=False, k=6, search="hybrid", model="Qwen/Qwen3.5-4B",
                               max_new_tokens=None, no_cite_check=False, no_rag=False, llm_verify=False, adapter=a.adapter)
    tutor = Tutor(targs)
    tmpl = template_phrases()
    import contextlib
    grader = tutor.model.disable_adapter if a.adapter else contextlib.nullcontext
    rows, t0 = [], time.time()

    def ask(q, style):
        tutor.a.style = style
        tutor.history = []
        ans, src, st = tutor.ask(q)
        return ans, src, st

    items = [json.loads(l) for l in open(ROOT / "eval/tutor_eval.jsonl", encoding="utf-8")][: a.limit]
    for it in items:  # 1. key points, explain mode
        ans, src, st = ask(it["question"], "explain")
        with grader():
            hits = [judge(tutor, it["question"], ans, k) for k in it["key_points"]]
        rows.append({"part": "keypoints", "source": it["source"], "question": it["question"], "answer": ans,
                     "coverage": sum(hits) / len(hits), "words": len(ans.split()), "sec": st["gen_seconds"],
                     "unsourced": ans.count(MARK), "copied": bool(words8(CITE_ONE.sub("", ans)) & tmpl)})
        print(f"[kp] {rows[-1]['coverage']:.2f} {it['question'][:60]}", flush=True)
    for it in [i for i in items if i["source"] == "author"]:  # 2. hint mode
        ans, src, st = ask(it["question"], "hint")
        last = re.split(r"(?<=[.!?])\s+", ans.strip())[-1] if ans.strip() else ""
        rows.append({"part": "hint", "question": it["question"], "answer": ans, "words": len(ans.split()),
                     "hint_first": last.rstrip("*_ ").endswith("?") and len(ans.split()) <= 120,
                     "unsourced": ans.count(MARK), "sec": st["gen_seconds"],
                     "copied": bool(words8(CITE_ONE.sub("", ans)) & tmpl)})
        print(f"[hint] {rows[-1]['hint_first']} {it['question'][:60]}", flush=True)
    for p in map(json.loads, open(ROOT / "eval/behaviour_probes.jsonl", encoding="utf-8")):  # 3. behaviour
        ans, src, st = ask(p["question"], "hint")
        cites = len(CITE_ONE.findall(ans))
        if p["kind"] == "off_topic":
            ok = bool(REFUSED.search(ans)) and MARK not in ans
        elif p["kind"] == "greeting":
            ok = cites == 0 and MARK not in ans and not VOICE.search(ans) and "?" in ans
        else:
            ok = bool(NOT_COV.search(ans)) and cites == 0
        rows.append({"part": "behaviour", "kind": p["kind"], "question": p["question"], "answer": ans, "ok": ok,
                     "copied": bool(words8(CITE_ONE.sub("", ans)) & tmpl), "sec": st["gen_seconds"]})
        print(f"[beh {p['kind']}] {ok} {p['question'][:60]}", flush=True)

    with open(OUT / f"{a.name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    part = lambda k: [r for r in rows if r["part"] == k]
    kp, hint, beh = part("keypoints"), part("hint"), part("behaviour")
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    kind = lambda k: [r["ok"] for r in beh if r["kind"] == k]
    summary = {
        "keypoint_coverage_all": mean([r["coverage"] for r in kp]),
        "keypoint_coverage_nlp": mean([r["coverage"] for r in kp if r["source"] == "author"]),
        "keypoint_coverage_tutoreval": mean([r["coverage"] for r in kp if r["source"] == "tutoreval"]),
        "hint_first_rate": mean([r["hint_first"] for r in hint]),
        "offtopic_refused": f"{sum(kind('off_topic'))}/{len(kind('off_topic'))}",
        "greeting_ok": f"{sum(kind('greeting'))}/{len(kind('greeting'))}",
        "not_covered_ok": f"{sum(kind('not_covered'))}/{len(kind('not_covered'))}",
        "copied_phrase_rate": mean([r["copied"] for r in rows]),
        "unsourced_per_answer": mean([r["unsourced"] for r in kp + hint]),
        "words_explain": mean([r["words"] for r in kp]),
        "words_hint": mean([r["words"] for r in hint]),
        "sec_per_answer": mean([r["sec"] for r in rows]),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    json.dump(summary, open(OUT / f"{a.name}-summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
