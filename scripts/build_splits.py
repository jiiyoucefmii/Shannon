"""Build fine-tuning splits in corpus/splits/ from open + noncommercial data only.

Run from the project root:  python scripts/build_splits.py [--include-competitive-code] [--val-frac 0.05]

Outputs
  corpus/splits/docs/{train,val}.jsonl   document chunks (continued pretraining / grounding), split by DOCUMENT
  corpus/splits/sft/{train,val}.jsonl    chat-format task data (tutoring dialogues, answer feedback, worked solutions)
  corpus/splits/eval_heldout.json        which benchmark/test files are held out (never trained on)
  corpus/splits/report.json              counts, licence tiers, decontamination drops

Rules
  - Only manifest records whose `use` contains "finetune" are eligible; restricted material never enters.
  - Datasets contribute only their `finetune_splits` (+ their own validation split as val). Test splits never.
  - Every training/val item is dropped if it shares a 13-word sequence with any held-out eval item.
  - Every row keeps `license_tier`, so a commercial run can filter out "noncommercial".
"""
import argparse
import glob
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "corpus"
OUT = C / "splits"
NGRAM = 13
VAL_NAMES = ("validation", "valid", "val")

SYS_TUTOR = ("You are a patient tutor. Diagnose the student's misunderstanding, ask guiding questions, "
             "and give hints before answers.")
SYS_GRADER = "You grade a student's short answer against the reference answer and give constructive feedback."


# ---------------------------------------------------------------- helpers
def manifest():
    return {r["id"]: r for r in map(json.loads, open(C / "manifest.jsonl", encoding="utf-8"))}


def ds_dir(rec):
    owner_name = rec["source_url"].rstrip("/").split("/datasets/")[-1].replace("/", "__")
    for base in ("raw", "quarantine"):
        d = C / base / "datasets" / owner_name
        if d.exists():
            return d
    return None


def split_of(path):
    """Split name from an HF file path: data/train-0000.parquet, main/test-..., test.jsonl, test-ua-00001.parquet."""
    name = Path(path).name
    return re.split(r"-\d{5}|\.(?:parquet|jsonl|json)$", name)[0]


def data_files(d):
    fs = [f for f in glob.glob(str(d / "**" / "*"), recursive=True)
          if f.endswith((".parquet", ".jsonl")) and ".cache" not in f]
    return sorted(fs)


def rows(f):
    if f.endswith(".parquet"):
        yield from pq.read_table(f).to_pylist()
    else:
        for line in open(f, encoding="utf-8"):
            if line.strip():
                yield json.loads(line)


def bucket(key, frac):
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10_000 < frac * 10_000


WORD = re.compile(r"\w+")


def grams(text):
    w = WORD.findall(text.lower())
    return {hash(tuple(w[i:i + NGRAM])) for i in range(len(w) - NGRAM + 1)}


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from strings(v)


# ---------------------------------------------------------------- held-out eval set
def eval_files(m):
    """Benchmarks: every file. Fine-tunable datasets: every split that is neither trained on nor validation."""
    held = []
    for r in m.values():
        if r["resource_type"] != "dataset":
            continue
        d = ds_dir(r)
        if d is None or r["id"] == "ds-codesearchnet":  # code-search corpus, not a test set; huge
            continue
        ft = set(r.get("finetune_splits") or [])
        for f in data_files(d):
            s = split_of(f)
            if ft and (s in ft or s in VAL_NAMES):
                continue
            if not ft and s.startswith(("train", "auxiliary_train")):
                continue  # a training split we simply don't use is not test data
            held.append((r["id"], f))
    return held


def build_eval_index(held):
    """gram -> bitmask of the datasets whose held-out files contain it.
    Lets a dataset be checked against everyone else's test data but not its own author-defined splits
    (SAF test_unseen_answers / SciEntsBank test-ua deliberately reuse the training questions)."""
    bit = {d: 1 << i for i, d in enumerate(sorted({d for d, _ in held}))}
    idx = defaultdict(int)
    for d, f in held:
        seen = set()
        for row in rows(f):
            for s in strings(row):
                if len(s) > 40 and s not in seen:
                    seen.add(s)
                    for g in grams(s):
                        idx[g] |= bit[d]
    return idx, bit


# ---------------------------------------------------------------- SFT converters (one per dataset)
def msgs(*pairs):
    return [{"role": r, "content": c.strip()} for r, c in pairs if c and c.strip()]


def conv_mathdial(row):
    turns = [t for t in row["conversation"].split("|EOM|") if t.strip()]
    out = [("system", f"{SYS_TUTOR}\n\nProblem: {row['question']}"),
           ("user", f"Here is my solution:\n{row['student_incorrect_solution']}")]
    for t in turns:
        who, _, text = t.partition(":")
        role = "assistant" if who.strip().lower() == "teacher" else "user"
        text = re.sub(r"^\s*\([a-z]+\)", "", text).strip()  # drop teacher-move labels like (probing)
        if out and out[-1][0] == role:
            out[-1] = (role, out[-1][1] + "\n" + text)
        else:
            out.append((role, text))
    if out[-1][0] != "assistant":
        out = out[:-1]
    return msgs(*out), "tutoring-dialogue"


def conv_saf(row):
    user = (f"Question: {row['question']}\nReference answer: {row['reference_answer']}\n"
            f"Student answer: {row['provided_answer']}")
    return msgs(("system", SYS_GRADER), ("user", user),
                ("assistant", f"Score: {row['score']}\nFeedback: {row['answer_feedback']}")), "answer-feedback"


def conv_scientsbank(row):
    user = (f"Question: {row['question']}\nReference answer: {row['reference_answer']}\n"
            f"Student answer: {row['student_answer']}\nLabel the student answer.")
    return msgs(("system", SYS_GRADER), ("user", user), ("assistant", str(row["label"]))), "answer-labeling"


def conv_sciq(row):
    opts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
    random.Random(row["question"]).shuffle(opts)
    letters = "ABCD"
    q = row["question"] + "\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))
    a = f"Answer: {letters[opts.index(row['correct_answer'])]}. {row['correct_answer']}"
    if row.get("support"):
        a += f"\nExplanation: {row['support']}"
    return msgs(("user", q), ("assistant", a)), "mcq-explain"


def conv_gsm8k(row):
    return msgs(("user", row["question"]), ("assistant", row["answer"])), "worked-solution"


def conv_math(row):
    return msgs(("user", row["problem"]), ("assistant", row["solution"])), "worked-solution"


def conv_mbpp(row):
    tests = "\n".join(row.get("test_list") or [])
    return msgs(("user", f"{row['text']}\nYour code should pass these tests:\n{tests}"),
                ("assistant", f"```python\n{row['code']}\n```")), "code"


def conv_apps(row):
    sols = json.loads(row["solutions"]) if row.get("solutions") else []
    if not sols:
        return None, None
    return msgs(("user", row["question"]), ("assistant", f"```python\n{sols[0]}\n```")), "code"


def conv_code_contests(row):
    sols = (row.get("solutions") or {})
    py = [s for s, lang in zip(sols.get("solution", []), sols.get("language", [])) if lang in (1, 3)]  # PYTHON(2/3)
    if not py:
        return None, None
    return msgs(("user", row["description"]), ("assistant", f"```python\n{py[0]}\n```")), "code"


# dataset id -> (converter, file filter)
CONVERTERS = {
    "ds-mathdial": (conv_mathdial, None),
    "ds-saf-en": (conv_saf, None),
    "ds-saf-de": (conv_saf, None),
    "ds-scientsbank": (conv_scientsbank, None),
    "ds-sciq": (conv_sciq, None),
    "ds-gsm8k": (conv_gsm8k, "socratic"),  # socratic variant: same problems, step-by-step sub-questions
    "ds-math": (conv_math, None),
    "ds-mbpp": (conv_mbpp, "full"),
}
COMPETITIVE = {"ds-apps": (conv_apps, None), "ds-code-contests": (conv_code_contests, None)}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--include-competitive-code", action="store_true", help="also APPS + CodeContests (large)")
    a = ap.parse_args()
    m = manifest()
    conv = dict(CONVERTERS, **(COMPETITIVE if a.include_competitive_code else {}))
    OUT.mkdir(parents=True, exist_ok=True)

    held = eval_files(m)
    print(f"building eval index from {len(held)} held-out files ...")
    ev, bit = build_eval_index(held)
    print(f"  {len(ev):,} {NGRAM}-grams")
    report = {"ngram": NGRAM, "val_frac": a.val_frac, "sft": {}, "docs": {}, "skipped": [], "contamination_sources": {}}

    def contaminated(text, self_id=None):
        mask = ~bit.get(self_id, 0)
        hits = 0
        for g in grams(text):
            hits |= ev.get(g, 0) & mask
        if hits:
            src = [d for d, b in bit.items() if hits & b]
            key = self_id or "docs"
            for d in src:
                report["contamination_sources"].setdefault(key, Counter())[d] += 1
        return bool(hits)

    # ---- SFT
    sft = {"train": [], "val": []}
    for did, r in m.items():
        if r["resource_type"] != "dataset" or "finetune" not in r["use"] or not r.get("finetune_splits"):
            continue
        if did not in conv:
            report["skipped"].append({"id": did, "why": "no converter (competitive code is opt-in)" if did in COMPETITIVE
                                      else "no converter"})
            continue
        d = ds_dir(r)
        fn, sub = conv[did]
        files = [f for f in data_files(d) if (sub is None or f"{sub}" in Path(f).parts or f"\\{sub}\\" in f)]
        use_splits = set(r["finetune_splits"])
        has_val = any(split_of(f) in VAL_NAMES for f in files)
        st = Counter()
        if not files:
            report["skipped"].append({"id": did, "why": "no data files on disk"}); continue
        for f in files:
            s = split_of(f)
            if s in use_splits:
                target = None  # decided per row
            elif s in VAL_NAMES:
                target = "val"
            else:
                continue
            for k, row in enumerate(rows(f)):
                messages, task = fn(row)
                if not messages:
                    st["empty"] += 1; continue
                text = "\n".join(x["content"] for x in messages)
                if contaminated(text, did):
                    st["dropped_contaminated"] += 1; continue
                key = f"{did}:{Path(f).name}:{k}"
                split = target or ("val" if not has_val and bucket(key, a.val_frac) else "train")
                sft[split].append({"messages": messages, "task": task, "source_id": did, "source_split": s,
                                   "license_tier": r["license_tier"], "license": r["license"]})
                st[split] += 1
        report["sft"][did] = dict(st)
        print(f"  sft {did:16} {dict(st)}")

    # ---- document chunks (open + noncommercial only, eligible for finetune)
    docs = {"train": [], "val": []}
    BLOCK = 50
    n_chunks = Counter(json.loads(l)["doc_id"] for l in open(C / "chunks" / "chunks.jsonl", encoding="utf-8"))
    for line in open(C / "chunks" / "chunks.jsonl", encoding="utf-8"):
        ch = json.loads(line)
        if "finetune" not in (ch.get("use") or []) or ch.get("suspect_math"):
            continue
        st = report["docs"].setdefault(ch["doc_id"], Counter())
        if contaminated(ch["text"]):
            st["dropped_contaminated"] += 1; continue
        # split by document; documents over BLOCK chunks (the textbook) split by contiguous 50-chunk blocks so one
        # book cannot land wholly in val, while neighbouring (overlapping) chunks stay on the same side
        k = int(ch["chunk_id"].rsplit("::", 1)[1])
        unit = ch["doc_id"] if n_chunks[ch["doc_id"]] <= BLOCK else f"{ch['doc_id']}#{k // BLOCK}"
        split = "val" if bucket(unit, a.val_frac) else "train"
        docs[split].append({"text": ch["text"], "doc_id": ch["doc_id"], "chunk_id": ch["chunk_id"], "title": ch["title"],
                            "topics": ch["topics"], "license_tier": ch["license_tier"]})
        st[split] += 1

    for kind, data in (("sft", sft), ("docs", docs)):
        (OUT / kind).mkdir(exist_ok=True)
        for split, items in data.items():
            random.Random(0).shuffle(items)
            with open(OUT / kind / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for it in items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

    json.dump([{"dataset": d, "file": str(Path(f).relative_to(ROOT)).replace("\\", "/")} for d, f in held],
              open(OUT / "eval_heldout.json", "w", encoding="utf-8"), indent=1)
    tiers = lambda items: dict(Counter(i["license_tier"] for i in items))
    report["totals"] = {f"{k}/{s}": {"n": len(v[s]), "tiers": tiers(v[s])} for k, v in (("sft", sft), ("docs", docs)) for s in v}
    report["docs_contaminated"] = {d: c["dropped_contaminated"] for d, c in report["docs"].items() if c.get("dropped_contaminated")}
    report["docs"] = {"n_docs": len(report["docs"]),
                      "val_docs": sorted({c["doc_id"] for c in docs["val"]})}
    json.dump(report, open(OUT / "report.json", "w", encoding="utf-8"), indent=1, default=dict)
    print(json.dumps(report["totals"], indent=1))
    print("docs with contaminated chunks dropped:", report["docs_contaminated"] or "none")
    print("skipped:", report["skipped"])


if __name__ == "__main__":
    main()
