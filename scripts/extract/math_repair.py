"""Repair Docling formula-model output in corpus/parsed PDF docs. Never invents math.

- span fails LaTeX parse      -> keep it, append <!-- equation: low-confidence -->
- span > 300 words (runaway)  -> replace with <!-- equation: formula-model output discarded (repetition) -->
- any page with such a span   -> append that page's PDF text layer (real equation text, unicode) unless already there
Re-runnable: spans already marked are skipped.
usage: python -I math_repair.py PROJECT_ROOT
"""
import glob, json, re, sys
from importlib.metadata import version
import pymupdf as fitz
from pylatexenc.latexwalker import LatexWalker

SPAN = re.compile(r"\$\$(.+?)\$\$", re.S)
LOW = "<!-- equation: low-confidence -->"
DROP = "<!-- equation: formula-model output discarded (repetition) -->"
TL = "<!-- text-layer: pymupdf"


def bad(s):
    try:
        LatexWalker(s, tolerant_parsing=False).get_latex_nodes()
        return False
    except Exception:
        return True


def repair_page(body, c):
    out, pos, hit = [], 0, False
    for m in SPAN.finditer(body):
        out.append(body[pos:m.start()]); pos = m.end()
        s, after = m.group(1), body[m.end():m.end() + 40]
        if LOW in after or DROP in after:
            out.append(m.group(0))
        elif len(s.split()) > 300:
            c["drop"] += 1; hit = True; out.append(DROP)
        elif bad(s):
            c["low"] += 1; hit = True; out.append(m.group(0) + " " + LOW)
        else:
            out.append(m.group(0))
    out.append(body[pos:])
    return "".join(out), hit


def main(root):
    tot = {}
    for jf in glob.glob(f"{root}/corpus/parsed/**/*.json", recursive=True):
        if "_reports" in jf:
            continue
        j = json.load(open(jf, encoding="utf-8"))
        if not j["source_path"].endswith(".pdf") or "formula" not in j["parser"]:
            continue
        mdf = jf[:-5] + ".md"
        chunks = re.split(r"(<!-- page: \d+ -->)", open(mdf, encoding="utf-8").read())
        d = fitz.open(f"{root}/corpus/{j['source_path']}")
        c = {"low": 0, "drop": 0, "tl": 0}
        for k in range(1, len(chunks), 2):
            body, hit = repair_page(chunks[k + 1], c)
            if hit and TL not in body:
                p = int(re.search(r"\d+", chunks[k]).group()) - 1
                body = body.rstrip() + f"\n\n{TL} (equations on this page are low-confidence) -->\n\n{d[p].get_text().strip()}\n\n"
                c["tl"] += 1
            chunks[k + 1] = body
        if c["low"] or c["drop"]:
            open(mdf, "w", encoding="utf-8").write("".join(chunks))
            j["math_repair"] = {"low_confidence": c["low"], "discarded_repetition": c["drop"], "textlayer_pages_added": c["tl"],
                                "tool": f"pylatexenc=={version('pylatexenc')}+pymupdf=={version('pymupdf')}"}
            json.dump(j, open(jf, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            tot[j["id"]] = (c["low"], c["drop"], c["tl"], j["n_equations"])
    print(f"{len(tot)} docs repaired; low-confidence {sum(v[0] for v in tot.values())}, "
          f"discarded {sum(v[1] for v in tot.values())}, text-layer pages added {sum(v[2] for v in tot.values())}")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1][0] - kv[1][1])[:8]:
        print(f"  {k}: low={v[0]} dropped={v[1]} pages+={v[2]} of {v[3]} eq")


if __name__ == "__main__":
    main(sys.argv[1])
