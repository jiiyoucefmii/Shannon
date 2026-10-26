"""Download everything in seeds/ into corpus/ and register it in corpus/manifest.jsonl.

Run on your own machine (needs normal internet access):

    pip install -r requirements.txt
    python scripts/download_all.py --contact you@example.com            # all stages
    python scripts/download_all.py --contact you@example.com --only courses papers
    python scripts/download_all.py --contact you@example.com --dry-run  # list what would be fetched

Stages: datasets, models, papers, courses, captions (captions is opt-in: --only captions).
Re-running is safe: files already registered (same id) or with a known sha256 are skipped.
"""
import argparse
import json
import re
import shutil
import sys
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from manifest import append, load, now_utc, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / "seeds"
CORPUS = ROOT / "corpus"
RAW = CORPUS / "raw"
QUAR = CORPUS / "quarantine"
MANIFEST = CORPUS / "manifest.jsonl"
LOGS = ROOT / "logs"

USE_BY_TIER = {
    "open": ["rag", "finetune", "eval", "redistribute-derived"],
    "noncommercial": ["rag", "finetune", "eval"],
    "restricted": ["private-rag", "eval"],
}

# Filename/title keyword -> topic, for tagging course files. Edit freely; it is only a first pass.
TOPIC_KEYWORDS = [
    (r"token|bpe|subword|words", "preprocessing-tokenization"),
    (r"n-?gram|\blm\b|lm_|lm\.|language.?model|kneser", "ngram-lm"),
    (r"wordvec|vector|embed|glove|word2vec|represent|ppmi", "embeddings"),
    (r"rnn|lstm|recurrent", "rnn-lstm"),
    (r"attention", "attention"),
    (r"transformer", "transformer"),
    (r"pretrain|posttrain|finetun|fine-tun", "pretraining-finetuning"),
    (r"mlm|masked|bert", "mlm-autoregressive"),
    (r"decod|sampling|beam", "decoding"),
    (r"translation|seq2seq|\bmt\b", "mt-seq2seq"),
    (r"logreg|classif|naive|\bnb\d|sentiment", "classification-labeling"),
    (r"posner|\bner\b|named.?entit|sequence.?label|extraction", "ner-ie"),
    (r"evaluat|metric|bleu|rouge|perplex", "evaluation-metrics"),
    (r"peft|lora|instruct|prompting|icl|alignment|rlhf|\brl\b", "llm-instruction-peft"),
    (r"\brag\b|-rag|retriev|\bir_|agents", "rag"),
    (r"hallucin|factual|robust", "hallucination-robustness"),
    (r"impact|ethic|safety|security", "responsible-security"),
    (r"reasoning|moe|quantiz|longcontext|scaling|multimodal|inference|parallel", "advanced"),
]


def tag_topics(text):
    t = text.lower()
    topics = [topic for pat, topic in TOPIC_KEYWORDS if re.search(pat, t)]
    return sorted(set(topics)) or ["advanced"]


class Fetcher:
    """Polite HTTP: robots.txt per host, per-host delay, retries with backoff, on-disk skip."""

    def __init__(self, contact, delay=1.0, dry_run=False):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = f"NLP-Tutor-Research-Crawler/0.1 (contact: {contact})"
        self.delay = delay
        self.dry = dry_run
        self.last = {}
        self.robots = {}
        LOGS.mkdir(exist_ok=True)
        self.log = open(LOGS / f"download-{time.strftime('%Y%m%d-%H%M%S')}.jsonl", "a")

    def allowed(self, url):
        host = urlparse(url).scheme + "://" + urlparse(url).netloc
        if host not in self.robots:
            rp = urllib.robotparser.RobotFileParser(host + "/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None
            self.robots[host] = rp
        rp = self.robots[host]
        return True if rp is None else rp.can_fetch(self.s.headers["User-Agent"], url)

    def _wait(self, url, delay):
        host = urlparse(url).netloc
        dt = time.time() - self.last.get(host, 0)
        if dt < delay:
            time.sleep(delay - dt)
        self.last[host] = time.time()

    def get(self, url, dest=None, delay=None):
        if not self.allowed(url):
            self._log(url, "blocked-by-robots")
            return None
        if dest is not None:
            # already on disk (raw/ or moved to quarantine/): don't fetch again, and never re-create a
            # restricted file in raw/ that register() moved to quarantine/
            for have in (dest, QUAR / dest.relative_to(RAW) if dest.is_relative_to(RAW) else None):
                if have is not None and have.is_file():
                    self._log(url, "skip-on-disk")
                    return have
        if self.dry:
            self._log(url, "dry-run")
            print("  would fetch", url)
            return None
        for attempt in range(3):
            self._wait(url, delay or self.delay)
            try:
                r = self.s.get(url, timeout=60, stream=dest is not None)
            except requests.RequestException as e:
                self._log(url, f"error:{e.__class__.__name__}")
                time.sleep(2 ** attempt * 5)
                continue
            if r.status_code in (429,) or r.status_code >= 500:
                wait = int(r.headers.get("Retry-After", 2 ** attempt * 5))
                self._log(url, f"retry:{r.status_code}")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                self._log(url, f"http:{r.status_code}")
                return None
            if dest is None:
                self._log(url, "ok")
                return r
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            except requests.RequestException as e:  # connection reset mid-stream: retry, don't crash the run
                self._log(url, f"error-midstream:{e.__class__.__name__}")
                tmp.unlink(missing_ok=True)
                time.sleep(2 ** attempt * 5)
                continue
            tmp.replace(dest)
            self._log(url, "saved", bytes=dest.stat().st_size)
            return dest
        return None

    def _log(self, url, decision, **kw):
        self.log.write(json.dumps({"url": url, "decision": decision, "at": now_utc(), **kw}) + "\n")
        self.log.flush()


def known(records):
    return {r["id"] for r in records}, {r["sha256"] for r in records if r.get("sha256")}


def register(rec, path, tier, report):
    """Hash the file, move restricted items to quarantine/, append to the manifest."""
    ids, hashes = known(load(MANIFEST))
    if rec["id"] in ids:
        report["skipped"] += 1
        return
    rec["sha256"] = sha256_file(path) if path.is_file() else rec.get("sha256")
    if rec["sha256"] in hashes:
        report["duplicates"] += 1
        return
    rec.setdefault("retrieved_at", now_utc())
    rec["license_tier"] = tier
    rec["use"] = list(USE_BY_TIER[tier])
    if rec.get("resource_type") == "dataset" and not rec.get("finetune_splits"):
        # benchmarks / eval-only datasets never go into fine-tuning data
        rec["use"] = [u for u in rec["use"] if u not in ("finetune", "redistribute-derived")]
    if tier == "restricted" and str(path).startswith(str(RAW)):
        target = QUAR / path.relative_to(RAW)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), target)
        report["quarantined"] += 1
    try:
        append(MANIFEST, rec)
        report["added"] += 1
    except ValueError as e:
        report["failed"].append(str(e))


def new_report():
    return {"added": 0, "quarantined": 0, "skipped": 0, "duplicates": 0, "failed": []}


# ---------------------------------------------------------------- datasets
def stage_datasets(args, fetch):
    from huggingface_hub import snapshot_download

    rep = new_report()
    seeds = yaml.safe_load((SEEDS / "datasets.yaml").read_text())
    for d in seeds["datasets"]:
        if not isinstance(d, dict) or "hf_id" not in d:
            continue
        if args.priority and d.get("priority") not in args.priority:
            continue
        if args.skip and d.get("id") in args.skip:
            continue
        dest = RAW / "datasets" / d["hf_id"].replace("/", "__")
        print("dataset", d["hf_id"])
        if args.dry_run:
            continue
        try:
            snapshot_download(repo_id=d["hf_id"], repo_type="dataset", local_dir=dest,
                              allow_patterns=["*.parquet", "*.json", "*.jsonl", "*.csv", "*.arrow", "README.md", "*.txt", "*.zip", "*.gz"])
        except Exception as e:
            rep["failed"].append(f"{d['hf_id']}: {e}")
            continue
        card = dest / "README.md"
        rec = {
            "id": d["id"], "title": d["title"],
            "source_url": f"https://huggingface.co/datasets/{d['hf_id']}",
            "resource_type": "dataset", "format": "hf-dataset", "topics": d["topics"],
            "license": d["license"], "license_evidence_url": f"https://huggingface.co/datasets/{d['hf_id']}",
            "access": "verified_accessible", "parser": None,
            "sha256": sha256_file(card) if card.exists() else None,
            "role": d.get("role"), "finetune_splits": d.get("finetune_splits", []),
            "is_benchmark_test": bool(d.get("is_benchmark")) and not d.get("finetune_splits"),
            "priority": d.get("priority"),
        }
        register(rec, dest, d["license_tier"], rep)
    return rep


# ---------------------------------------------------------------- models (cards only)
def stage_models(args, fetch):
    from huggingface_hub import hf_hub_download

    rep = new_report()
    seeds = yaml.safe_load((SEEDS / "models.yaml").read_text())
    ids = []
    for group in seeds.values():
        items = group if isinstance(group, list) else [m for v in group.values() for m in v]
        ids += [m["hf_id"] for m in items]
    for hf_id in ids:
        dest = RAW / "models" / hf_id.replace("/", "__")
        print("model card", hf_id)
        if args.dry_run:
            continue
        got = []
        for fname in ("README.md", "config.json", "generation_config.json", "tokenizer_config.json"):
            try:
                got.append(Path(hf_hub_download(hf_id, fname, local_dir=dest)))
            except Exception:
                pass
        if not got:
            rep["failed"].append(f"{hf_id}: nothing downloaded (gated? run `huggingface-cli login`)")
            continue
        rec = {
            "id": "model-" + hf_id.split("/")[1].lower(), "title": hf_id,
            "source_url": f"https://huggingface.co/{hf_id}", "resource_type": "model_card",
            "format": "md+json", "topics": ["llm-instruction-peft"],
            "license": "see card", "license_evidence_url": f"https://huggingface.co/{hf_id}",
            "access": "verified_accessible", "parser": None,
        }
        register(rec, got[0], "open", rep)  # model cards are reference docs; weights' license is separate
    return rep


# ---------------------------------------------------------------- papers
ARXIV_LICENSES = [
    (r"creativecommons\.org/(licenses/by/|licenses/by-sa/|publicdomain/zero)", "open"),
    (r"creativecommons\.org/licenses/by-nc", "noncommercial"),
    (r"arxiv\.org/licenses/nonexclusive-distrib", "restricted"),
]


def arxiv_license(fetch, aid):
    r = fetch.get(f"https://arxiv.org/abs/{aid}", delay=3.0)
    if r is None:
        return None, "restricted"
    m = re.search(r'href="(https?://(?:creativecommons\.org|arxiv\.org/licenses)[^"]+)"', r.text)
    if not m:
        return None, "restricted"
    url = m.group(1)
    tier = next((t for pat, t in ARXIV_LICENSES if re.search(pat, url)), "restricted")
    return url, tier


def stage_papers(args, fetch):
    rep = new_report()
    seeds = yaml.safe_load((SEEDS / "papers.yaml").read_text())
    for section, items in seeds.items():
        for p in items:
            if "id" not in p:   # non-arXiv readings
                url = p["url"]
                if "github.com" in url or "seas.harvard.edu" in url:
                    continue
                name = unquote(Path(urlparse(url).path).name) or "index.html"
                dest = RAW / "papers" / "other" / name
                print("paper (other)", p["title"])
                if fetch.get(url, dest) is None:
                    continue
                rec = {"id": "paper-" + re.sub(r"[^a-z0-9]+", "-", p["title"].lower()).strip("-")[:60],
                       "title": p["title"], "source_url": url, "resource_type": "paper",
                       "format": dest.suffix.lstrip(".") or "html", "topics": p["topics"],
                       "license": "unknown", "license_evidence_url": None, "access": "verified_accessible",
                       "parser": None}
                register(rec, dest, "restricted", rep)
                continue
            aid = p["id"]
            folder = RAW / "papers" / "arxiv" / aid
            print("arXiv", aid, p["title"])
            lic_url, tier = arxiv_license(fetch, aid)
            pdf = fetch.get(f"https://arxiv.org/pdf/{aid}", folder / f"{aid}.pdf", delay=3.0)
            fetch.get(f"https://arxiv.org/e-print/{aid}", folder / f"{aid}-source.tar.gz", delay=3.0)
            if pdf is None:
                if not args.dry_run:
                    rep["failed"].append(f"arXiv {aid}: pdf not downloaded")
                continue
            rec = {"id": f"arxiv-{aid}", "title": p["title"], "source_url": f"https://arxiv.org/abs/{aid}",
                   "resource_type": "paper", "format": "pdf+tex", "topics": p["topics"],
                   "license": lic_url or "unknown", "license_evidence_url": f"https://arxiv.org/abs/{aid}" if lic_url else None,
                   "access": "verified_accessible", "parser": None, "section": section}
            # arXiv's default non-exclusive licence => restricted (RAG/eval, no redistribution)
            register(rec, pdf, tier, rep)
    return rep


# ---------------------------------------------------------------- courses
TYPE_MAP = {"slides": "slides", "notes": "notes", "assignment": "assignment", "syllabus": "syllabus",
            "textbook": "textbook"}


def stage_courses(args, fetch):
    rep = new_report()
    seeds = yaml.safe_load((SEEDS / "courses.yaml").read_text())
    for c in seeds:
        files = c.get("files") or {}
        inst = re.sub(r"[^a-z0-9]+", "-", c["institution"].lower()).strip("-")
        for kind, urls in files.items():
            if kind not in TYPE_MAP or not isinstance(urls, list):
                continue
            for i, url in enumerate(urls, 1):
                name = unquote(Path(urlparse(url).path).name) or "index"
                if not Path(name).suffix:
                    name += ".html"
                name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).lower()
                dest = RAW / "courses" / inst / c["id"] / kind / name
                print(c["id"], kind, name)
                if dest.exists():
                    path = dest
                elif fetch.get(url, dest) is None:
                    if not args.dry_run:
                        rep["failed"].append(f"{url}: not downloaded (see log)")
                    continue
                else:
                    path = dest
                rec = {"id": f"{c['id']}-{kind}-{Path(name).stem}"[:120], "title": Path(name).stem,
                       "source_url": url, "resource_type": TYPE_MAP[kind], "format": Path(name).suffix.lstrip("."),
                       "topics": tag_topics(name), "license": c.get("license", "unknown"),
                       "license_evidence_url": c.get("license_evidence_url"), "access": "verified_accessible",
                       "parser": None, "institution": c["institution"], "course": c["course"],
                       "version": c.get("version"), "has_solutions": False}
                register(rec, path, c.get("license_tier", "restricted"), rep)
    return rep


# ---------------------------------------------------------------- captions (opt-in)
def stage_captions(args, fetch):
    rep = new_report()
    seeds = yaml.safe_load((SEEDS / "courses.yaml").read_text())
    for c in seeds:
        if not c.get("video"):
            continue
        out = RAW / "transcripts" / c["id"]
        out.mkdir(parents=True, exist_ok=True)
        cmd = ["yt-dlp", "--skip-download", "--write-subs", "--write-auto-subs", "--sub-langs", "en.*",
               "--sub-format", "vtt", "-o", str(out / "%(playlist_index)03d-%(title)s.%(ext)s"), c["video"]]
        print(" ".join(cmd))
        if not args.dry_run:
            import subprocess
            subprocess.run(cmd, check=False)
        # Captions stay in raw/transcripts; register them after review (licence = course licence).
    return rep


STAGES = {"datasets": stage_datasets, "models": stage_models, "papers": stage_papers,
          "courses": stage_courses, "captions": stage_captions}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", required=True, help="email put in the crawler User-Agent")
    ap.add_argument("--only", nargs="+", choices=STAGES, help="run only these stages")
    ap.add_argument("--priority", nargs="+", help="datasets only: e.g. P1 P2")
    ap.add_argument("--skip", nargs="+", help="datasets only: seed ids to leave out, e.g. ds-natural-questions")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stages = args.only or ["datasets", "models", "papers", "courses"]
    fetch = Fetcher(args.contact, dry_run=args.dry_run)
    for name in stages:
        print(f"\n=== {name} ===")
        rep = STAGES[name](args, fetch)
        print(f"--- {name}: added {rep['added']}, quarantined {rep['quarantined']}, "
              f"skipped {rep['skipped']}, duplicates {rep['duplicates']}, failed {len(rep['failed'])}")
        for f in rep["failed"]:
            print("   FAILED", f)


if __name__ == "__main__":
    main()
