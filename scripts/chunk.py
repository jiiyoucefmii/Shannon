"""Split corpus/parsed/*.md into retrieval chunks in corpus/chunks/.

Run from the project root:  python scripts/chunk.py [--target 350] [--max 600] [--overlap 50]

- Splits on page/slide/cell/section markers first, then packs units up to --target words.
- Never cuts inside a $$...$$ block or a ``` code fence.
- Every chunk carries the document's licence tier, use, topics and its page/slide range,
  so retrieval and training can filter on them.
- Restricted documents go to corpus/chunks/quarantine.jsonl; everything else to corpus/chunks/chunks.jsonl.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "corpus" / "parsed"
OUT = ROOT / "corpus" / "chunks"

MARKER = re.compile(r"<!-- (page|slide|cell): (\d+)[^>]*-->")
HEADING = re.compile(r"^(#{1,3}) (.+)$", re.M)


def blocks(text):
    """Paragraph-level blocks; $$ math and ``` fences are kept whole."""
    out, buf, fence, math = [], [], False, False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
        elif not fence and s.count("$$") % 2 == 1:
            math = not math
        buf.append(line)
        if not s and not fence and not math:
            if "".join(buf).strip():
                out.append("\n".join(buf).strip())
            buf = []
    if "".join(buf).strip():
        out.append("\n".join(buf).strip())
    return out


def split_big(b, target):
    """Cut an oversize block at line boundaries; code fences are closed and reopened per piece.
    Lines are never cut, so a single $$...$$ line stays whole."""
    lines = b.split("\n")
    fence = lines[0].strip() if lines[0].strip().startswith("```") else None
    if fence:
        lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
    pieces, cur = [], []
    for ln in lines:
        if cur and words("\n".join(cur)) + words(ln) > target and (fence or not in_math(cur)):
            pieces.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        pieces.append(cur)
    if fence:
        return [fence + "\n" + "\n".join(p) + "\n```" for p in pieces]
    return ["\n".join(p) for p in pieces]


def in_math(lines):
    return "\n".join(lines).count("$$") % 2 == 1


def units(md):
    """Yield (kind, number, heading, text) per page/slide/cell."""
    parts = MARKER.split(md)
    head = parts[0]
    if len(parts) == 1:  # no markers (html, zip members): one unit, sections split later by size
        yield None, None, head
        return
    if head.strip():
        yield None, None, head
    for i in range(1, len(parts), 3):
        yield parts[i], int(parts[i + 1]), parts[i + 2]


def words(t):
    return len(t.split())


def chunk_doc(md, meta, target, maxw, overlap):
    chunks, cur, cur_loc, heading = [], [], [], meta["title"]

    def flush():
        if not cur:
            return
        text = "\n\n".join(cur).strip()
        if words(text) < 8:  # skip near-empty slide fragments
            return
        chunks.append((text, list(cur_loc), heading))

    for kind, num, body in units(md):
        bs = [p for b in blocks(body) for p in (split_big(b, target) if words(b) > maxw else [b])]
        for b in bs:
            h = HEADING.match(b)
            if h and len(h.group(1)) > 1:  # "## Section" / slide title: new heading context
                heading = h.group(2).strip()
            if cur and words("\n\n".join(cur)) + words(b) > target:
                flush()
                # overlap = the last whole block if it is short, so math/code are never cut mid-span
                tail = cur[-1] if overlap and words(cur[-1]) <= overlap * 2 and not MARKER.search(cur[-1]) else None
                cur, cur_loc = ([tail] if tail and words(b) < maxw else []), cur_loc[-1:]
            cur.append(b)
            if kind and (not cur_loc or cur_loc[-1] != (kind, num)):
                cur_loc.append((kind, num))
    flush()

    recs = []
    for k, (text, loc, head) in enumerate(chunks):
        kinds = {x[0] for x in loc}
        nums = [x[1] for x in loc]
        rec = {
            "chunk_id": f"{meta['id']}::{k:04d}",
            "doc_id": meta["id"],
            "title": meta["title"],
            "heading": head,
            "loc_kind": next(iter(kinds)) if len(kinds) == 1 else None,
            "loc_start": min(nums) if nums else None,
            "loc_end": max(nums) if nums else None,
            "text": text,
            "n_words": words(text),
            "has_math": "$" in text,
            "has_code": "```" in text,
            # a display equation this long is formula-model repetition, not real math
            "suspect_math": any(words(m) > 300 for m in re.findall(r"\$\$(.+?)\$\$", text, re.S)),
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        for f in ("resource_type", "course", "institution", "topics", "license_tier", "use"):
            rec[f] = meta.get(f)
        rec["doc_status"] = meta["quality"]["status"]
        recs.append(rec)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=350, help="target words per chunk")
    ap.add_argument("--max", type=int, default=600, help="soft max words (one oversize block may exceed it)")
    ap.add_argument("--overlap", type=int, default=50, help="words carried over between chunks")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"open": open(OUT / "chunks.jsonl", "w", encoding="utf-8"),
             "restricted": open(OUT / "quarantine.jsonl", "w", encoding="utf-8")}
    stats, seen = {}, set()
    for jf in sorted(PARSED.rglob("*.json")):
        if "_reports" in jf.parts:
            continue
        meta = json.loads(jf.read_text(encoding="utf-8"))
        md = jf.with_suffix(".md").read_text(encoding="utf-8")
        md = md.split("\n", 1)[1] if md.startswith("# ") else md  # title goes into metadata
        dest = "restricted" if meta["license_tier"] == "restricted" else "open"
        n = 0
        for r in chunk_doc(md, meta, a.target, a.max, a.overlap):
            if r["text_sha256"] in seen:  # identical text across docs (e.g. repeated slides)
                continue
            seen.add(r["text_sha256"])
            files[dest].write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
        s = stats.setdefault(meta["license_tier"], [0, 0])
        s[0] += 1
        s[1] += n
    for f in files.values():
        f.close()
    for tier, (d, c) in sorted(stats.items()):
        print(f"{tier:13} {d:4} docs -> {c:6} chunks")


if __name__ == "__main__":
    main()
