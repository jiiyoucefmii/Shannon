"""Append the PDF text layer to pages where Docling kept <70% of the source words."""
import glob, json, re, sys
from importlib.metadata import version
import pymupdf as fitz
root = sys.argv[1]; tot = {}
for jf in glob.glob(f"{root}/corpus/parsed/**/*.json", recursive=True):
    if "_reports" in jf: continue
    j = json.load(open(jf, encoding="utf-8"))
    if not j["source_path"].endswith(".pdf") or not j["parser"].startswith("docling") or j.get("textlayer_pages") is not None: continue
    mdf = jf[:-5] + ".md"; md = open(mdf, encoding="utf-8").read()
    d = fitz.open(f"{root}/corpus/{j['source_path']}")
    chunks = re.split(r"(<!-- page: \d+ -->)", md)  # [head, marker1, body1, marker2, body2, ...]
    patched = []
    for k in range(1, len(chunks), 2):
        p = int(re.search(r"\d+", chunks[k]).group()) - 1
        src = d[p].get_text().strip()
        a = set(re.sub(r"\W+", " ", src).lower().split()); b = set(re.sub(r"\W+", " ", chunks[k + 1]).lower().split())
        if len(a) > 15 and len(a & b) / len(a) < 0.7:
            chunks[k + 1] = chunks[k + 1].rstrip() + f"\n\n<!-- text-layer: pymupdf (layout parser dropped text on this page) -->\n\n{src}\n\n"
            patched.append(p + 1)
    open(mdf, "w", encoding="utf-8").write("".join(chunks))
    j["textlayer_pages"] = patched; j["parser"] += f"+textlayer:pymupdf=={version('pymupdf')}"
    json.dump(j, open(jf, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    tot[j["id"]] = (len(patched), j["n_pages"])
s = sum(v[0] for v in tot.values()); n = sum(v[1] for v in tot.values())
print(f"patched {s} of {n} pages in {sum(1 for v in tot.values() if v[0])} of {len(tot)} docs")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1][0] / max(1, kv[1][1]))[:8]: print(" ", k, v)
