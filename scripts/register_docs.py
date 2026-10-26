"""Register individual documents that live inside already-cloned repos, so the extractor parses them.

Run from the project root:  python scripts/register_docs.py
Re-running is safe: ids already in the manifest are skipped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from manifest import append, now_utc, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "corpus" / "raw" / "repos"
QUAR = ROOT / "corpus" / "quarantine" / "repos"
OPEN = ["rag", "finetune", "eval", "redistribute-derived"]
PRIVATE = ["private-rag", "eval"]

D2L = "d2l-ai__d2l-en"
D2L_SHA = "23d7a5aecceee57d1292c56e90cce307f183bb0a"
# d2l chapter dir -> topics (only the NLP-relevant chapters)
D2L_CHAPTERS = {
    "chapter_recurrent-neural-networks": ["ngram-lm", "rnn-lstm", "preprocessing-tokenization", "evaluation-metrics"],
    "chapter_recurrent-modern": ["rnn-lstm", "mt-seq2seq", "decoding"],
    "chapter_attention-mechanisms-and-transformers": ["attention", "transformer", "positional-encoding"],
    "chapter_natural-language-processing-pretraining": ["embeddings", "mlm-autoregressive", "pretraining-finetuning"],
    "chapter_natural-language-processing-applications": ["classification-labeling", "pretraining-finetuning"],
}

EISENSTEIN = "jacobeisenstein__gt-nlp-class"

# Teaching files inside the open (MIT / Apache-2.0) repos cloned by clone_repos.sh, added 2026-10-19.
# (repo dir, github slug, licence, [(glob, resource_type, topics)]). AGENTS.md / CLAUDE.md / CHANGELOG.md are
# agent instructions or changelogs, not teaching content, and are never registered.
HF_TOPICS = {  # HF LLM course chapter -> topics; chapters 0,4,8,9,10 (setup, Hub, asking for help, Gradio,
    # Argilla) are tool walk-throughs, not NLP concepts, and are skipped
    1: ["transformer", "architectures", "mlm-autoregressive", "pretraining-finetuning"],
    2: ["preprocessing-tokenization", "transformer"],
    3: ["pretraining-finetuning", "classification-labeling"],
    5: ["preprocessing-tokenization"],
    6: ["preprocessing-tokenization"],
    7: ["classification-labeling", "ner-ie", "mt-seq2seq", "mlm-autoregressive", "rag", "evaluation-metrics"],
    11: ["llm-instruction-peft"],
    12: ["llm-instruction-peft", "advanced"],
}
REPO_DOCS = [
    ("harvardnlp__annotated-transformer", "harvardnlp/annotated-transformer", "MIT",
     [("AnnotatedTransformer.ipynb", "notebook", ["transformer", "attention", "positional-encoding", "architectures"])]),
    ("stanford-cs336__spring2025-lectures", "stanford-cs336/spring2025-lectures", "MIT",
     [("lecture_[0-9][0-9].py", "notes", ["transformer", "architectures", "pretraining-finetuning", "advanced"]),
      ("nonexecutable/*.pdf", "slides", ["transformer", "architectures", "pretraining-finetuning", "llm-instruction-peft",
                                         "advanced"])]),
    ("stanford-cs336__assignment1-basics", "stanford-cs336/assignment1-basics", "MIT",
     [("cs336_assignment1_basics.pdf", "assignment", ["preprocessing-tokenization", "transformer", "attention",
                                                      "positional-encoding"])]),
    ("stanford-cs336__assignment2-systems", "stanford-cs336/assignment2-systems", "MIT",
     [("cs336_assignment2_systems.pdf", "assignment", ["attention", "advanced"])]),
    ("stanford-cs336__assignment3-scaling", "stanford-cs336/assignment3-scaling", "MIT",
     [("cs336_assignment3_scaling.pdf", "assignment", ["pretraining-finetuning", "advanced"])]),
    ("stanford-cs336__assignment4-data", "stanford-cs336/assignment4-data", "MIT",
     [("cs336_assignment4_data.pdf", "assignment", ["preprocessing-tokenization", "pretraining-finetuning"])]),
    ("karpathy__minbpe", "karpathy/minbpe", "MIT",
     [("README.md", "docs", ["preprocessing-tokenization"]), ("lecture.md", "notes", ["preprocessing-tokenization"]),
      ("exercise.md", "assignment", ["preprocessing-tokenization"]),
      ("minbpe/[bgr]*.py", "docs", ["preprocessing-tokenization"])]),
    ("karpathy__nanoGPT", "karpathy/nanoGPT", "MIT",
     [("README.md", "docs", ["transformer", "pretraining-finetuning"]),
      ("model.py", "docs", ["transformer", "attention", "architectures"]),
      ("*.ipynb", "notebook", ["pretraining-finetuning", "transformer"])]),
]


def repo_sha(d):
    import subprocess
    return subprocess.check_output(["git", "-C", str(d), "rev-parse", "HEAD"], text=True).strip()


def repo_doc_records():
    """Manifest records for teaching files inside the open repos (HF course + REPO_DOCS)."""
    import re
    recs = []
    hf = RAW / "huggingface__course"
    if hf.exists():
        sha = repo_sha(hf)
        for f in sorted((hf / "chapters" / "en").glob("chapter*/*.mdx"), key=lambda p: (int(p.parent.name[7:]), p.stem)):
            ch = int(f.parent.name[7:])
            if ch not in HF_TOPICS:
                continue
            m = re.search(r"^# (.+?)(\[\[.*\]\])?\s*$", f.read_text(encoding="utf-8"), re.M)
            rel = f.relative_to(hf).as_posix()
            recs.append(base_rec(f"hfcourse-ch{ch}-{f.stem}", f"Hugging Face LLM Course ch.{ch}.{f.stem} — "
                                 f"{m.group(1).strip() if m else f.stem}", "huggingface/course", sha, rel, f,
                                 "notes", "mdx", HF_TOPICS[ch], "Apache-2.0", "Hugging Face", "HF LLM Course"))
    for dname, slug, lic, specs in REPO_DOCS:
        d = RAW / dname
        if not d.exists():
            continue
        sha = repo_sha(d)
        short = dname.split("__", 1)[1].lower()
        for pattern, rtype, topics in specs:
            for f in sorted(d.glob(pattern)):
                rel = f.relative_to(d).as_posix()
                stem = re.sub(r"[^a-z0-9]+", "-", f.stem.lower()).strip("-")
                course = "CS336" if "cs336" in dname else None
                recs.append(base_rec(f"{short}-{stem}", f"{slug} — {rel}", slug, sha, rel, f, rtype,
                                     f.suffix.lstrip(".") or "txt", topics, lic,
                                     "Stanford" if course else None, course))
    return recs


def base_rec(rid, title, slug, sha, rel, path, rtype, fmt, topics, lic, inst, course):
    rec = {
        "id": rid, "title": title, "source_url": f"https://github.com/{slug}/blob/{sha}/{rel}",
        "retrieved_at": now_utc(), "sha256": sha256_file(path), "resource_type": rtype, "format": fmt,
        "topics": topics, "license": lic, "license_evidence_url": f"https://github.com/{slug}/blob/{sha}/LICENSE",
        "license_tier": "open", "access": "verified_accessible", "use": OPEN, "parser": None, "commit_sha": sha,
    }
    if inst:
        rec["institution"] = inst
    if course:
        rec["course"] = course
    return rec


def main():
    added = skipped = 0
    for chap, topics in D2L_CHAPTERS.items():
        for f in sorted((RAW / D2L / chap).glob("*.md")):
            if f.name == "index.md":
                continue
            rel = f"{chap}/{f.name}"
            rec = {
                "id": f"d2l-{f.stem}",
                "title": f"Dive into Deep Learning — {f.stem.replace('-', ' ')}",
                "source_url": f"https://github.com/d2l-ai/d2l-en/blob/{D2L_SHA}/{rel}",
                "retrieved_at": now_utc(), "sha256": sha256_file(f),
                "resource_type": "notes", "format": "md", "topics": topics,
                "license": "CC-BY-SA-4.0",
                "license_evidence_url": f"https://github.com/d2l-ai/d2l-en/blob/{D2L_SHA}/LICENSE",
                "license_tier": "open", "access": "verified_accessible", "use": OPEN, "parser": None,
                "commit_sha": D2L_SHA, "version": "d2l-en master",
            }
            if append(ROOT / "corpus" / "manifest.jsonl", rec):
                added += 1
            else:
                skipped += 1
    # Eisenstein notes: repo has no licence -> restricted, private RAG only (repo already in quarantine)
    pdf = QUAR / EISENSTEIN / "notes" / "eisenstein-nlp-notes.pdf"
    if pdf.is_file():
        import subprocess
        sha = subprocess.check_output(["git", "-C", str(QUAR / EISENSTEIN), "rev-parse", "HEAD"], text=True).strip()
        rec = {
            "id": "eisenstein-nlp-notes", "title": "Eisenstein, Natural Language Processing (notes)",
            "source_url": f"https://github.com/jacobeisenstein/gt-nlp-class/blob/{sha}/notes/eisenstein-nlp-notes.pdf",
            "retrieved_at": now_utc(), "sha256": sha256_file(pdf),
            "resource_type": "notes", "format": "pdf",
            "topics": ["ngram-lm", "classification-labeling", "ner-ie", "embeddings", "rnn-lstm", "mt-seq2seq"],
            "license": "unknown", "license_evidence_url": None, "license_tier": "restricted",
            "access": "verified_accessible", "use": PRIVATE, "parser": None, "commit_sha": sha,
            "institution": "Georgia Tech", "course": "CS 4650/7650",
        }
        if append(ROOT / "corpus" / "manifest.jsonl", rec):
            added += 1
        else:
            skipped += 1
    for rec in repo_doc_records():
        if append(ROOT / "corpus" / "manifest.jsonl", rec):
            added += 1
        else:
            skipped += 1
    print(f"added {added}, skipped (already registered) {skipped}")


if __name__ == "__main__":
    main()
