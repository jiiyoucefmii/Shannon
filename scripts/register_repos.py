"""Register cloned repos in the manifest; move unlicensed ones to quarantine/.

Run from the project root:  python scripts/register_repos.py
"""
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from manifest import append, now_utc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "corpus" / "manifest.jsonl"
RAW = ROOT / "corpus" / "raw" / "repos"
QUAR = ROOT / "corpus" / "quarantine" / "repos"

OPEN = ["rag", "finetune", "eval", "redistribute-derived"]
PRIVATE = ["private-rag", "eval"]

REPOS = [
    # dir, title, license, tier, topics, institution/course, version
    ("stanford-cs336__spring2025-lectures", "CS336 lecture code + slide PDFs", "MIT", "open",
     ["transformer", "architectures", "pretraining-finetuning", "llm-instruction-peft", "advanced"],
     "Stanford", "CS336", "Spring 2025"),
    ("stanford-cs336__assignment1-basics", "CS336 A1: tokenizer, transformer LM, training loop", "MIT", "open",
     ["preprocessing-tokenization", "transformer", "attention", "positional-encoding", "mlm-autoregressive"],
     "Stanford", "CS336", "Spring 2025"),
    ("stanford-cs336__assignment2-systems", "CS336 A2: systems (FlashAttention, DDP)", "MIT", "open",
     ["attention", "advanced"], "Stanford", "CS336", "Spring 2025"),
    ("stanford-cs336__assignment3-scaling", "CS336 A3: scaling laws", "MIT", "open",
     ["pretraining-finetuning", "advanced"], "Stanford", "CS336", "Spring 2025"),
    ("stanford-cs336__assignment4-data", "CS336 A4: pretraining data filtering", "MIT", "open",
     ["preprocessing-tokenization", "pretraining-finetuning"], "Stanford", "CS336", "Spring 2025"),
    ("stanford-cs336__assignment5-alignment", "CS336 A5: SFT, RLHF/GRPO alignment", "unknown", "restricted",
     ["llm-instruction-peft", "responsible-security"], "Stanford", "CS336", "Spring 2025"),
    ("cmu-l3__anlp-spring2026-code", "CMU 11-711 ANLP lecture code (23 lectures)", "unknown", "restricted",
     ["embeddings", "ngram-lm", "rnn-lstm", "transformer", "pretraining-finetuning", "decoding", "rag",
      "evaluation-metrics", "llm-instruction-peft", "advanced"], "CMU", "11-711 ANLP", "Spring 2026"),
    ("huggingface__course", "Hugging Face LLM/NLP course (English chapters)", "Apache-2.0", "open",
     ["preprocessing-tokenization", "pretraining-finetuning", "classification-labeling", "ner-ie",
      "mt-seq2seq", "llm-instruction-peft"], "Hugging Face", "HF Course", "main"),
    ("harvardnlp__annotated-transformer", "The Annotated Transformer", "MIT", "open",
     ["transformer", "attention", "positional-encoding", "mt-seq2seq"], "Harvard NLP", None, None),
    ("karpathy__nanoGPT", "nanoGPT", "MIT", "open",
     ["transformer", "mlm-autoregressive", "pretraining-finetuning"], None, None, None),
    ("karpathy__minbpe", "minbpe (BPE tokenizer exercises)", "MIT", "open",
     ["preprocessing-tokenization"], None, None, None),
    ("jacobeisenstein__gt-nlp-class", "Eisenstein NLP notes + Georgia Tech class", "unknown", "restricted",
     ["ngram-lm", "classification-labeling", "ner-ie", "embeddings", "rnn-lstm", "mt-seq2seq"],
     "Georgia Tech", "CS 4650/7650", None),
    # text CC BY-SA 4.0 (LICENSE), sample code modified MIT (LICENSE-SAMPLECODE); added 2026-10-18 for ngram/decoding gaps
    ("d2l-ai__d2l-en", "Dive into Deep Learning (d2l.ai) book source", "CC-BY-SA-4.0", "open",
     ["ngram-lm", "rnn-lstm", "mt-seq2seq", "decoding", "attention", "transformer", "embeddings", "pretraining-finetuning"],
     None, None, None),
]


def git(d, *args):
    return subprocess.check_output(["git", "-C", str(d), *args], text=True).strip()


def tree_hash(d):
    """sha256 over the committed tree (stable, ignores .git)."""
    data = subprocess.check_output(["git", "-C", str(d), "archive", "--format=tar", "HEAD"])
    return hashlib.sha256(data).hexdigest()


def main():
    QUAR.mkdir(parents=True, exist_ok=True)
    added = quarantined = skipped = 0
    for dname, title, lic, tier, topics, inst, course, version in REPOS:
        src = RAW / dname
        if not src.exists():
            src = QUAR / dname
        if not src.exists():
            print("missing", dname)
            continue
        owner, repo = dname.split("__", 1)
        url = f"https://github.com/{owner}/{repo}"
        sha = git(src, "rev-parse", "HEAD")
        lic_file = next((p.name for p in src.iterdir() if p.name.lower().startswith(("license", "licence"))), None)
        rec = {
            "id": f"repo-{repo.lower()}",
            "title": title,
            "source_url": url,
            "retrieved_at": now_utc(),
            "sha256": tree_hash(src),
            "resource_type": "repo",
            "format": "git",
            "topics": topics,
            "license": lic,
            "license_evidence_url": f"{url}/blob/{sha}/{lic_file}" if lic_file else None,
            "license_tier": tier,
            "access": "verified_accessible",
            "use": OPEN if tier == "open" else PRIVATE,
            "parser": None,
            "commit_sha": sha,
        }
        for k, v in (("institution", inst), ("course", course), ("version", version)):
            if v:
                rec[k] = v
        if tier == "restricted" and src.parent == RAW:
            shutil.move(str(src), QUAR / dname)
            quarantined += 1
        if append(MANIFEST, rec):
            added += 1
        else:
            skipped += 1
    print(f"added {added}, quarantined {quarantined}, skipped (already registered) {skipped}")


if __name__ == "__main__":
    main()
