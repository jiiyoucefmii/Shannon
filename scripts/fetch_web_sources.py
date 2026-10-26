"""Fetch and register the open web sources in seeds/web.yaml (Wikipedia pages + sparse repo checkouts).

Run from the project root:  python scripts/fetch_web_sources.py --contact YOUR_EMAIL [--dry-run]
Re-running is safe: files on disk are not re-fetched and registered ids are skipped.
"""
import argparse
import fnmatch
import re
import subprocess
import sys
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import quote

import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from manifest import append, now_utc, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "corpus" / "raw"
MANIFEST = ROOT / "corpus" / "manifest.jsonl"
OPEN = ["rag", "finetune", "eval", "redistribute-derived"]


def rec(rid, title, url, path, rtype, fmt, topics, lic, evidence, **extra):
    r = {"id": rid, "title": title, "source_url": url, "retrieved_at": now_utc(), "sha256": sha256_file(path),
         "resource_type": rtype, "format": fmt, "topics": topics, "license": lic, "license_evidence_url": evidence,
         "license_tier": "open", "access": "verified_accessible", "use": OPEN, "parser": None}
    r.update(extra)
    return r


def wikipedia(items, contact, dry):
    ua = f"NLP-Tutor-Research-Crawler/0.1 (contact: {contact})"
    # fetch robots.txt with our own User-Agent: RobotFileParser.read() uses Python's default UA, which Wikipedia
    # answers with 403, and a 403 makes the parser treat the whole site as disallowed
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(requests.get("https://en.wikipedia.org/robots.txt", headers={"User-Agent": ua}, timeout=30).text.splitlines())
    out = RAW / "web" / "wikipedia"
    out.mkdir(parents=True, exist_ok=True)
    added = skipped = 0
    for it in items:
        url = "https://en.wikipedia.org/wiki/" + quote(it["title"], safe="'")
        if not rp.can_fetch(ua, url):
            print("  robots disallows", url); continue
        dest = out / (re.sub(r"[^a-z0-9]+", "-", it["title"].lower()).strip("-") + ".html")
        if dry:
            print("  would fetch", url); continue
        if not dest.exists():
            r = requests.get(url, headers={"User-Agent": ua}, timeout=60)
            time.sleep(1.0)
            if r.status_code != 200:
                print("  http", r.status_code, url); continue
            dest.write_bytes(r.content)
        rev = re.search(rb'"wgCurRevisionId":(\d+)', dest.read_bytes())
        oldid = rev.group(1).decode() if rev else None
        title = it["title"].replace("_", " ")
        r = rec(f"wikipedia-{dest.stem}", f"Wikipedia — {title}", url, dest, "docs", "html", it["topics"],
                "CC-BY-SA-4.0", "https://en.wikipedia.org/wiki/Wikipedia:Text_of_the_Creative_Commons_"
                "Attribution-ShareAlike_4.0_International_License", **({"version": f"oldid {oldid}"} if oldid else {}))
        if append(MANIFEST, r):
            added += 1
        else:
            skipped += 1
    print(f"wikipedia: added {added}, skipped {skipped}")


def repos(items, dry):
    base = RAW / "repos"
    added = skipped = 0
    for it in items:
        slug = it["repo"]
        d = base / slug.replace("/", "__")
        dirs = sorted({p.split("*")[0].rsplit("/", 1)[0] for p in it["paths"]})
        if dry:
            print("  would sparse-clone", slug, dirs); continue
        if not d.exists():
            subprocess.check_call(["git", "-c", "core.longpaths=true", "clone", "-q", "--depth", "1",
                                   "--filter=blob:none", "--sparse", f"https://github.com/{slug}", str(d)])
            subprocess.check_call(["git", "-C", str(d), "sparse-checkout", "set", "--no-cone", it["licence_file"],
                                   *it["paths"]])
        sha = subprocess.check_output(["git", "-C", str(d), "rev-parse", "HEAD"], text=True).strip()
        files = subprocess.check_output(["git", "-C", str(d), "ls-files"], text=True).split()
        for rel in sorted(files):
            # gitignore "a/**/b" (sparse-checkout) = fnmatch "a/*b": fnmatch's * already crosses "/"
            if not any(fnmatch.fnmatch(rel, p.replace("**/", "")) for p in it["paths"]):
                continue
            f = d / rel
            short = slug.split("/")[1].lower()
            stem = re.sub(r"[^a-z0-9]+", "-", rel.rsplit(".", 1)[0].lower()).strip("-")
            r = rec(f"{short}-{stem}", f"{slug} — {rel}", f"https://github.com/{slug}/blob/{sha}/{rel}", f,
                    it["type"], f.suffix.lstrip("."), it["topics"], it["licence"],
                    f"https://github.com/{slug}/blob/{sha}/{it['licence_file']}", commit_sha=sha)
            if append(MANIFEST, r):
                added += 1
            else:
                skipped += 1
    print(f"repos: added {added}, skipped {skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contact", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    seeds = yaml.safe_load((ROOT / "seeds" / "web.yaml").read_text(encoding="utf-8"))
    wikipedia(seeds["wikipedia"], a.contact, a.dry_run)
    repos(seeds["repos"], a.dry_run)


if __name__ == "__main__":
    main()
