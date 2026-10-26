import re, sys
from pylatexenc.latexwalker import LatexWalker
MATH_RE = re.compile(r"\$\$(.+?)\$\$|(?<![\$])\$([^$\n]+?)\$(?!\$)", re.S)
t = open(sys.argv[1], encoding="utf-8").read()
for a, b in MATH_RE.findall(t):
    s = a or b
    try: LatexWalker(s, tolerant_parsing=False).get_latex_nodes()
    except Exception as e: print(repr(s[:200])); print("  ->", str(e)[:150])
