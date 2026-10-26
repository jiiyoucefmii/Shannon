"""Generate grounded NLP tutoring dialogues from open/noncommercial document chunks with the local LLM.

Run from the project root:  python scripts/generate_dialogues.py --n 30 [--seed 0] [--out NAME]
Sources: corpus/splits/docs/{train,val}.jsonl (already open/noncommercial-only and decontaminated); a dialogue
inherits the split and licence of its source chunk. Output: corpus/splits/tutor_sft/dialogues-<NAME>.jsonl
(kept) and dialogues-<NAME>-rejected.jsonl (with the reason), appended as they are made, so a run can be resumed.

Quality gates (all must pass):
  format     valid JSON, 4-8 turns, student first, alternating, tutor turns <= 120 words
  pedagogy   the first tutor turn asks a question or gives a hint (contains "?"), no meta-talk about "the passage"
  grounding  >= 80% of the tutor's factual sentences are supported by the source chunk (bge-reranker >= 0.3,
             the threshold calibrated for the citation checker)
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from tutor import CHAT, CitationChecker, SUPPORT_LO, Tutor, sentences  # noqa: E402

SPLITS = ROOT / "corpus" / "splits"
SYS = ("You are a patient NLP tutor. Find out what the student already understands, give hints and guiding "
       "questions before full answers, and check their understanding at the end.")
PROMPT = """Below is an excerpt from NLP course material.

{passage}

Write a short tutoring conversation between a STUDENT and a TUTOR about one concept from this excerpt.
- The student starts with a realistic question, or a common misconception about the concept.
- The tutor does not lecture right away: first a guiding question or hint, then explains step by step.
- Every fact the tutor states must come from the excerpt. Do not mention "the excerpt", "the text" or "the passage".
- 4 to 6 turns in total, alternating, starting with the student. Each tutor turn under 100 words.
- If the excerpt is mostly code, references or a table of contents, output exactly {{"skip": true}}.

Output only JSON: {{"concept": "...", "turns": [{{"role": "student", "text": "..."}}, {{"role": "tutor", "text": "..."}}, ...]}}"""
# meta-talk about the source ("The text mentions that ..."): common forms are rewritten, the rest rejected.
# "the text" alone is normal NLP vocabulary ("tokenize the text"), so only source-referring uses count.
SRC = r"(?:the|this) (?:excerpt|passage|text|material|document|course material|reading)(?: describing [^,]+)?"
META_LEAD = re.compile(r"\b(?:according to|as (?:stated|mentioned|described|explained|noted) in) " + SRC + r",?\s*", re.I)
# only neutral reporting verbs are stripped ("X states that Y" -> "Y"); "warns against", "recommends", "asks us to"
# carry meaning, so stripping them would change the claim: those dialogues are rejected instead
META_VERB = re.compile(r"\b" + SRC + r" (?:also |specifically |explicitly |clearly )?(?:says|states|mentions|notes|"
                       r"explains|highlights|concludes|shows|describes|points out|emphasi[sz]es) that,?\s*", re.I)
SRC_WORD = r"(?:the|this) (?:text|material|document|reading)"
META = re.compile(r"\b(?:the|this) (?:excerpt|passage)\b|\b(?:in|from|according to|as (?:stated|mentioned|described) in) "
                  + SRC_WORD + r"\b|\b" + SRC_WORD + r" (?:also |specifically |explicitly )?(?:says|states|mentions|"
                  r"notes|explains|highlights|recommends|concludes|warns|shows|describes|points|emphasi|suggests|asks|"
                  r"discusses|covers)|\b(?:does|did) " + SRC_WORD + r" (?:say|mention|state)|\bspeaker notes\b", re.I)
PAPER_SHARE = 0.2  # paper chunks give narrow "what did Figure 3 show" dialogues; prefer textbook/notes/course text


def demeta(s):
    s = META_LEAD.sub("", s)
    s = META_VERB.sub("", s)
    # re-capitalise sentence starts the rewrite exposed ("Exactly. features can be" -> "Exactly. Features can be")
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s.strip())


REWRITE = """Rewrite this tutor message so it states the facts directly, as a teacher who knows the subject.
Remove every reference to "the text", "the excerpt", "the passage", "the material" or "the reading".
Keep the meaning, the facts and any question exactly; do not add new information. Output only the rewritten message.

Message: {t}"""


def fix_meta(tutor, d):
    """LLM rewrite of tutor turns that still refer to the source after the regex pass. Grounding is checked
    afterwards, so a rewrite that changes a fact still gets the dialogue rejected."""
    import torch
    n = 0
    for t in d.get("turns") or []:
        if t.get("role") != "tutor" or not META.search(t.get("text", "")):
            continue
        msg = [{"role": "user", "content": REWRITE.format(t=t["text"])}]
        x = tutor.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", enable_thinking=False).to(tutor.model.device)
        with torch.no_grad():
            o = tutor.model.generate(**x, max_new_tokens=200, do_sample=False)
        t["text"] = demeta(tutor.proc.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        n += 1
    return n


def pick_chunks(n, seed, exclude=()):
    rtype = {r["id"]: r["resource_type"] for r in map(json.loads, open(ROOT / "corpus" / "manifest.jsonl", encoding="utf-8"))}
    rows = []
    for split in ("train", "val"):
        for line in open(SPLITS / "docs" / f"{split}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            t = r["text"]
            words = len(t.split())
            code = t.count("```") >= 2 and len(re.findall(r"```.*?```", t, re.S)[0] if "```" in t else "") > len(t) / 2
            if words < 120 or code or "(correct)" in t or t.count("|") > 40:
                continue
            # bibliographies / reference lists: the model ignores "skip references" and invents content from titles
            heading = (r.get("title", "") + " " + (r.get("heading") or "")).lower()
            if len(re.findall(r"\b(?:19|20)\d\d[a-z]?\.", t)) >= 4 or re.search(r"bibliograph|references|reading list", heading):
                continue
            if r["chunk_id"] in exclude:
                continue
            r["split"] = split
            r["is_paper"] = rtype.get(r["doc_id"]) == "paper"
            rows.append(r)
    # spread over topics: round-robin across each chunk's first topic, so big sources don't dominate
    rng = random.Random(seed)
    by_topic = {}
    for r in rows:
        by_topic.setdefault((r.get("topics") or ["none"])[0], []).append(r)
    for v in by_topic.values():
        rng.shuffle(v)
    picked, keys, papers = [], sorted(by_topic), 0
    while len(picked) < n and any(by_topic.values()):
        for k in keys:
            while by_topic[k] and len(picked) < n:
                r = by_topic[k].pop()
                if r["is_paper"] and papers >= PAPER_SHARE * n:
                    continue  # paper quota used up: try the next chunk of this topic
                papers += r["is_paper"]
                picked.append(r)
                break
    return picked


def parse(raw):
    """First JSON object in the output (the model sometimes adds a stray "}" or text after it)."""
    i = raw.find("{")
    if i < 0:
        return None
    try:
        d, _ = json.JSONDecoder().raw_decode(raw[i:])
    except json.JSONDecodeError:
        return None
    if isinstance(d, dict):
        for t in d.get("turns") or []:
            if isinstance(t, dict) and isinstance(t.get("text"), str):
                t["text"] = demeta(t["text"])
    return d if isinstance(d, dict) else None


def check(d, passage, ce):
    if not d or d.get("skip"):
        return "skipped-by-model" if d and d.get("skip") else "bad-json", None
    turns = d.get("turns") or []
    if not 4 <= len(turns) <= 8:
        return "turn-count", None
    roles = [t.get("role") for t in turns]
    if roles[0] != "student" or any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)) \
            or set(roles) - {"student", "tutor"}:
        return "roles", None
    tutor = [t["text"].strip() for t in turns if t["role"] == "tutor"]
    if any(len(t.split()) > 120 for t in tutor):
        return "tutor-too-long", None
    if any(META.search(t["text"]) for t in turns if t["role"] == "tutor"):  # students may refer to their reading
        return "meta-talk", None
    if "?" not in tutor[0]:
        return "no-guiding-question", None
    claims = []
    for t in tutor:
        for s, e in sentences(t):
            c = t[s:e].strip()
            if len(c.split()) >= 6 and not c.rstrip("*_ ").endswith("?") and not CHAT.search(c):
                claims.append(c)
    if not claims:
        return "no-claims", None
    scores = ce.predict([(c, passage) for c in claims], batch_size=8)
    support = sum(float(s) >= SUPPORT_LO for s in scores) / len(claims)
    if support < 0.8:
        return f"ungrounded ({support:.2f})", support
    return "ok", support


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="source chunks to try")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pilot")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    targs = argparse.Namespace(style="hint", open_only=True, k=6, search="hybrid", model="Qwen/Qwen3.5-4B",
                               max_new_tokens=None, no_cite_check=True, no_rag=True, llm_verify=False)
    tutor = Tutor(targs)
    ce = CitationChecker().ce
    out = SPLITS / "tutor_sft"
    out.mkdir(parents=True, exist_ok=True)
    keep_p, rej_p = out / f"dialogues-{a.out}.jsonl", out / f"dialogues-{a.out}-rejected.jsonl"
    done = set()  # chunks already tried in ANY dialogues file (this run or earlier ones): never reuse a source
    for p in out.glob("dialogues-*.jsonl"):
        done |= {json.loads(l)["chunk_id"] for l in open(p, encoding="utf-8")}
    import torch
    stats = Counter()
    for i, ch in enumerate(pick_chunks(a.n, a.seed, exclude=done), 1):
        if ch["chunk_id"] in done:
            continue
        passage = f"{ch['title']}\n\n{ch['text']}"
        msg = [{"role": "user", "content": PROMPT.format(passage=passage)}]
        inputs = tutor.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                return_tensors="pt", enable_thinking=False).to(tutor.model.device)
        t0 = time.time()
        with torch.no_grad():
            o = tutor.model.generate(**inputs, max_new_tokens=700, do_sample=True, temperature=0.7, top_p=0.9)
        raw = tutor.proc.decode(o[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        d = parse(raw)
        rewrites = fix_meta(tutor, d) if d and not d.get("skip") else 0
        verdict, support = check(d, ch["text"], ce)
        stats[verdict.split(" ")[0]] += 1
        base = {"chunk_id": ch["chunk_id"], "doc_id": ch["doc_id"], "source_split": ch["split"],
                "license_tier": ch["license_tier"], "topics": ch.get("topics"), "generator": "Qwen/Qwen3.5-4B",
                "support": None if support is None else round(support, 2), "meta_rewrites": rewrites}
        if verdict == "ok":
            messages = [{"role": "system", "content": SYS}] + [
                {"role": "user" if t["role"] == "student" else "assistant", "content": t["text"].strip()}
                for t in d["turns"]]
            if messages[-1]["role"] != "assistant":
                messages = messages[:-1]
            rec = {**base, "task": "tutoring-dialogue-synthetic", "concept": d.get("concept"), "messages": messages}
            with open(keep_p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            with open(rej_p, "a", encoding="utf-8") as f:
                f.write(json.dumps({**base, "reason": verdict, "raw": raw[:3000]}, ensure_ascii=False) + "\n")
        torch.cuda.empty_cache()
        print(f"[{i}/{a.n}] {verdict:22} {time.time() - t0:5.1f}s  {ch['doc_id'][:50]}", flush=True)
    print(dict(stats))


if __name__ == "__main__":
    main()
