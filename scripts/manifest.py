"""Manifest helpers for the NLP tutor corpus.

Usage:
    python scripts/manifest.py validate corpus/manifest.jsonl
    python scripts/manifest.py summary  corpus/manifest.jsonl
"""
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TOPICS = {
    "preprocessing-tokenization", "ngram-lm", "embeddings", "rnn-lstm", "attention",
    "transformer", "positional-encoding", "architectures", "pretraining-finetuning",
    "mlm-autoregressive", "decoding", "mt-seq2seq", "classification-labeling", "ner-ie",
    "evaluation-metrics", "llm-instruction-peft", "rag", "hallucination-robustness",
    "responsible-security", "advanced",
}
RESOURCE_TYPES = {
    "slides", "notes", "notebook", "assignment", "solution", "exam", "paper", "dataset",
    "model_card", "repo", "transcript", "docs", "syllabus", "reading-list", "textbook",
}
ACCESS = {"verified_accessible", "verified_restricted", "unverified", "unavailable"}
TIERS = {"open", "noncommercial", "restricted"}
USES = {"rag", "private-rag", "finetune", "eval", "redistribute-derived", "reference"}
REQUIRED = [
    "id", "title", "source_url", "retrieved_at", "sha256", "resource_type", "format",
    "topics", "license", "license_evidence_url", "license_tier", "access", "use", "parser",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_record(r):
    errs = []
    for k in REQUIRED:
        if k not in r:
            errs.append(f"missing {k}")
    if r.get("resource_type") not in RESOURCE_TYPES:
        errs.append(f"bad resource_type {r.get('resource_type')}")
    if r.get("access") not in ACCESS:
        errs.append(f"bad access {r.get('access')}")
    if r.get("license_tier") not in TIERS:
        errs.append(f"bad license_tier {r.get('license_tier')}")
    bad_topics = set(r.get("topics") or []) - TOPICS
    if bad_topics:
        errs.append(f"unknown topics {sorted(bad_topics)}")
    bad_uses = set(r.get("use") or []) - USES
    if bad_uses:
        errs.append(f"unknown use {sorted(bad_uses)}")
    if r.get("license_tier") == "restricted" and "finetune" in (r.get("use") or []):
        errs.append("restricted item marked for finetune")
    if r.get("is_benchmark_test") and "finetune" in (r.get("use") or []):
        errs.append("benchmark test split marked for finetune")
    if r.get("license_tier") != "restricted" and not r.get("license_evidence_url"):
        errs.append("non-restricted tier without license_evidence_url")
    return errs


def append(path, record):
    errs = validate_record(record)
    if errs:
        raise ValueError(f"{record.get('id')}: {errs}")
    existing = {r["id"] for r in load(path)}
    if record["id"] in existing:
        return False
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return True


def main():
    cmd, path = sys.argv[1], sys.argv[2]
    recs = load(path)
    if cmd == "validate":
        ids = Counter(r.get("id") for r in recs)
        bad = 0
        for r in recs:
            errs = validate_record(r)
            if ids[r.get("id")] > 1:
                errs.append("duplicate id")
            if errs:
                bad += 1
                print(r.get("id"), errs)
        print(f"{len(recs)} records, {bad} with errors")
        sys.exit(1 if bad else 0)
    if cmd == "summary":
        print("records:", len(recs))
        for field in ["resource_type", "license_tier", "access"]:
            print(field, dict(Counter(r.get(field) for r in recs)))
        print("topics", dict(Counter(t for r in recs for t in r.get("topics", []))))


if __name__ == "__main__":
    main()
