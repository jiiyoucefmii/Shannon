"""Build the fine-tuning mix in corpus/splits/mix/{train,val}.jsonl from the existing splits.

Run from the project root:  python scripts/finetune/build_mix.py [--tutor-upsample 3] [--general 2000] [--seed 0]

  NLP tutoring (corpus/splits/tutor_sft)   all rows, repeated --tutor-upsample times (it is small and is the target style)
  general SFT (corpus/splits/sft)          a fixed-size sample per source, weighted toward tutoring-like tasks
Validation keeps the sets apart (`set` field) so the pilot can report tutoring and general loss separately.
Rows keep their licence tier; --open-only drops noncommercial rows for a model you will share commercially.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "corpus" / "splits"
# share of the general sample per source: tutoring dialogues and answer feedback first, then worked solutions
# 2026-10-22: fewer worked solutions (median 68 words, never end with a question) after both pilots lost the
# hint-first habit; more tutoring dialogue (MathDial)
WEIGHTS = {"ds-mathdial": 0.40, "ds-saf-en": 0.15, "ds-scientsbank": 0.15, "ds-saf-de": 0.05, "ds-gsm8k": 0.07,
           "ds-math": 0.04, "ds-sciq": 0.09, "ds-mbpp": 0.05}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tutor-upsample", type=int, default=3)
    ap.add_argument("--general", type=int, default=1200, help="general SFT conversations in train")
    ap.add_argument("--general-val", type=int, default=120)
    ap.add_argument("--open-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = SPLITS / "mix"
    out.mkdir(parents=True, exist_ok=True)
    keep = (lambda r: r["license_tier"] == "open") if a.open_only else (lambda r: True)

    def sample_general(rows, n):
        by = defaultdict(list)
        for r in rows:
            if keep(r):
                by[r["source_id"]].append(r)
        picked = []
        for src, w in WEIGHTS.items():
            pool = by.get(src, [])
            rng.shuffle(pool)
            picked += pool[: round(n * w)]
        return picked

    tutor_tr = [r for r in load(SPLITS / "tutor_sft" / "train.jsonl") if keep(r)]
    tutor_va = [r for r in load(SPLITS / "tutor_sft" / "val.jsonl") if keep(r)]
    gen_tr = sample_general(load(SPLITS / "sft" / "train.jsonl"), a.general)
    gen_va = sample_general(load(SPLITS / "sft" / "val.jsonl"), a.general_val)

    train = [{**r, "set": "tutor"} for r in tutor_tr] * a.tutor_upsample + [{**r, "set": "general"} for r in gen_tr]
    val = [{**r, "set": "tutor"} for r in tutor_va] + [{**r, "set": "general"} for r in gen_va]
    rng.shuffle(train)
    for name, rows in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rows), dict(Counter(r["set"] for r in rows)), dict(Counter(r["license_tier"] for r in rows)))
    print("general sources (train):", dict(Counter(r["source_id"] for r in gen_tr)))


if __name__ == "__main__":
    main()
