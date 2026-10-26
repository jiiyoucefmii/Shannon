"""Anti-plagiarism: how much of a submission's n-grams appear in another student's submission for the same item,
in the reference solution / model answer, or in the course passages the item was written from.

Code is compared after normalisation (names -> ID, numbers -> N, strings -> S, comments dropped), so renaming
variables or reformatting does not hide copying. Score = containment |A & B| / |A| of the submission's n-grams.
"""
import io
import keyword
import re
import tokenize

CODE_N, TEXT_N = 6, 5
CODE_MIN, TEXT_MIN = 25, 25  # shorter answers are too similar by nature to judge
PEER, REFERENCE, SOURCE = 0.75, 0.85, 0.6


def code_tokens(src):
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type == tokenize.NAME:
                out.append(t.string if keyword.iskeyword(t.string) else "ID")
            elif t.type == tokenize.NUMBER:
                out.append("N")
            elif t.type == tokenize.STRING:
                out.append("S")
            elif t.type == tokenize.OP:
                out.append(t.string)
            elif t.type in (tokenize.INDENT, tokenize.DEDENT):
                out.append(tokenize.tok_name[t.type])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        out = re.findall(r"\w+|[^\w\s]", src)
    return out


def text_tokens(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def grams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def containment(a, b):
    return len(a & b) / len(a) if a else 0.0


def check(kind, answer, item, peers):
    """peers: [(submission_id, student, answer)] for the same item by other students. -> flags"""
    if kind == "mcq":
        return []
    flags = []
    if kind == "code":
        toks = code_tokens(answer.get("code") or "")
        if len(toks) < CODE_MIN:
            return []
        mine, n = grams(toks, CODE_N), CODE_N
        get = lambda a: grams(code_tokens(a.get("code") or ""), n)
        reference = grams(code_tokens(item["body"].get("solution", "")), n)
    else:
        toks = text_tokens(answer.get("text") or "")
        if len(toks) < TEXT_MIN:
            return []
        mine, n = grams(toks, TEXT_N), TEXT_N
        get = lambda a: grams(text_tokens(a.get("text") or ""), n)
        reference = grams(text_tokens(item["body"].get("model_answer", "")), n)
    for sid, student, other in peers:
        s = containment(mine, get(other))
        if s >= PEER:
            flags.append({"type": "peer", "submission_id": sid, "student": student, "similarity": round(s, 2),
                          "text": f"{round(100 * s)}% overlap with {student}'s submission"})
    s = containment(mine, reference)
    if s >= REFERENCE:
        flags.append({"type": "reference", "similarity": round(s, 2),
                      "text": f"{round(100 * s)}% overlap with the reference " +
                              ("solution" if kind == "code" else "model answer")})
    if kind == "short":
        for src in item.get("sources") or []:
            s = containment(mine, grams(text_tokens(src.get("text", "")), n))
            if s >= SOURCE:
                flags.append({"type": "source", "similarity": round(s, 2),
                              "text": f"{round(100 * s)}% copied word for word from the course material ({src['title'][:60]})"})
                break
    return flags
