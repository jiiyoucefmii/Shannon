import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); C = root / "corpus"
recs = [json.loads(l) for l in open(C / "manifest.jsonl", encoding="utf-8")]
by_sha = {}
for r in recs:
    by_sha.setdefault(r.get("sha256"), []).append(r["id"])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
inv = []
for base in ("raw", "quarantine"):
    for p in (C / base).rglob("*"):
        if not p.is_file(): continue
        rel = p.relative_to(C).as_posix()
        if "/datasets/" in "/" + rel or "/models/" in "/" + rel or "/.cache/" in rel: continue
        s = sha(p)
        inv.append({"path": rel, "sha256": s, "ids": by_sha.get(s, [])})
json.dump(inv, open(sys.argv[2], "w"), indent=1)
matched = {i for x in inv for i in x["ids"]}
doc_types = {"paper", "slides", "notes", "assignment", "syllabus", "textbook"}
print("files", len(inv), "unmatched files", sum(not x["ids"] for x in inv))
print("doc records without file:", [r["id"] for r in recs if r["resource_type"] in doc_types and r["id"] not in matched])
from collections import defaultdict
d = defaultdict(list)
for x in inv:
    for i in x["ids"]: d[i].append(x["path"])
dups = {k: v for k, v in d.items() if len(v) > 1}
print("records with >1 copy:", len(dups)); [print(" ", k, v) for k, v in list(dups.items())[:6]]
print("unmatched sample:", [x["path"] for x in inv if not x["ids"]][:15])
