"""RAG tutor prototype: hybrid retrieval over the corpus + a local LLM with a tutor prompt.

  python scripts/tutor.py                                   # interactive chat (type 'quit' to exit)
  python scripts/tutor.py --ask "why divide by sqrt(d_k)?"  # one question
  python scripts/tutor.py --file eval/tutor_questions.txt   # batch; answers logged to eval/tutor_runs/
  options: --style hint|explain   --open-only (skip restricted material)   --k 6   --model Qwen/Qwen3.5-4B

Default model: Qwen3.5-4B, 4-bit NF4 (~3 GB VRAM) so it fits an 8 GB GPU next to bge-m3.
Restricted (quarantined) material is searched by default: this is private RAG on your own machine,
which is what its licence tier allows. Use --open-only for anything you will share.
"""
import argparse
import re
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from search import Index  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "eval" / "tutor_runs"

STYLES = {
    "hint": ("Teach Socratically. Give at most two short sentences of help (one key idea or hint), then end with ONE "
             "guiding question. Do not give the full answer unless the student asks for it directly or has already "
             "tried. Hard limit: 120 words."),
    "explain": ("Give a clear explanation at the level of an undergraduate NLP course: intuition first, then the "
                "precise definition or equation, then a tiny example only if it helps. Hard limit: 250 words."),
}
# asked the other way round ("clearly unrelated?") so the gate only refuses obvious off-topic messages:
# the "is it about NLP?" version rejected real questions like "can we swap sigmoids for step functions?"
TOPIC_GATE = ("Previous student question: {context}\nNew student message: {q}\n\n"
              "Topics of this course: NLP, machine learning, neural networks, deep learning, AI, statistics, math, "
              "programming and computer science, plus greetings or requests for study help.\n"
              "Is the new message CLEARLY UNRELATED to all of these topics (for example cooking, sports, travel, "
              "geography trivia)? Answer only yes or no.")
OFF_TOPIC = ("That's outside this NLP course, so I can't help with it here. "
             "Ask me anything about NLP or machine learning!")

# token budget per style: a little above the word limit so answers end on their own, not mid-sentence
MAX_NEW = {"hint": 260, "explain": 520}

SYSTEM = """You are an NLP tutor for a university course.
{style}

Citations: every sentence that states a fact must end with the [n] of the passage that states that fact
(one citation, the best one). Base each fact on what the passage actually says; if no passage states it,
leave it out or mark it "(not from the course material)". Questions you ask the student need no citation.
Never put a citation on a line by itself.
Off-topic questions (not about NLP or machine learning): reply in one sentence that this is outside the course,
and do not cite anything. If the passages do not cover an NLP question, say so and mark anything you add
"(not from the course material)". Never invent equations or paper results.
Write math in LaTeX: $...$ inline, $$...$$ for display.

Course passages:
{passages}"""


CITE = re.compile(r"\s*\[(\d+)\](?:\[(\d+)\])*")
# Support decision: a passage supports a claim if the reranker scores it >= SUPPORT_LO. Chosen from 68 hand-labelled
# claim/passage pairs (eval/citation_labels.jsonl): 0.3 -> precision 0.82 / recall 0.99, 0.5 -> 0.86 / 0.91.
# On the same 40 eval answers (eval/compare_cite_policies.py, run 20261019-144914) 0.3 marked 37 sentences
# unsourced vs 61 at 0.5. An LLM verifier for the 0.3-0.7 band (--llm-verify) was more precise on labels but marked
# 66, because the 4B model says "no" to ~40% of true claims; kept as an option, off by default.
SUPPORT_LO = SUPPORT_HI = 0.3
LLM_BAND = (0.3, 0.7)
VERIFY = """Passage:
{p}

Claim: {c}

Does the passage itself state this claim (not just discuss the same topic)? Answer only yes or no."""
# conversational sentences (greetings, offers, the tutor talking about itself) are not claims to source
CHAT = re.compile(r"^(hello|hi|hey|yes|sure|great|good question|let'?s|okay|ok)\b|\bI (can|will|'ll|am|'m)\b|"
                  r"\b(we|let us) (can|will|'ll) (start|begin|look|go)|"
                  # verdicts on the student's answer are judgements, not facts to source
                  r"^(not quite|not exactly|not this time|close\b|good try|hmm|exactly|correct|precisely|right\b|no\b|"
                  r"that'?s (a common pick|not it|right|correct|it|a reasonable guess)|that one'?s tempting)", re.I)
CITE_ONE = re.compile(r"\[(\d+)\]")


def sentences(text):
    """Split into (start, end) sentence spans; display math blocks are never split."""
    spans, start = [], 0
    # no split before "[": a trailing "[2]" after "?" belongs to the sentence before it
    for m in re.finditer(r"\$\$.*?\$\$|(?<=[.!?])\s+(?=[A-Z*($-])|\n+", text, re.S):
        if m.group().startswith("$$"):
            continue
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


class CitationChecker:
    """Checks each [n] against the sentence it supports. Support = bge-reranker-v2-m3 score, with the uncertain
    middle band decided by `verify(passage, claim) -> bool` (the tutor's LLM). Unsupported citations are moved to
    the best retrieved passage if that one is supported, otherwise removed. A factual sentence left without a
    citation (uncited by the model, or its citation removed) gets the best passage if supported, otherwise it is
    marked "(not from the course material)" so the student knows it is unsourced."""

    def __init__(self, threshold=None, verify=None, lo=None, hi=None):
        import torch
        from sentence_transformers import CrossEncoder
        self.verify = verify  # None: middle band falls back to score >= 0.5 (reranker only)
        self.lo = SUPPORT_LO if lo is None else lo  # lo == hi gives a plain reranker threshold
        self.hi = SUPPORT_HI if hi is None else hi
        self.ce = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda", max_length=512,
                               model_kwargs={"torch_dtype": torch.float16})

    def supported(self, claim, n, scores, passages):
        """(is_supported, how) for passage n (1-based)."""
        if not 0 < n <= len(passages):
            return False, "invalid"
        s = float(scores[n - 1])
        if s < self.lo:
            return False, f"score {s:.2f}"
        if s >= self.hi:
            return True, f"score {s:.2f}"
        if self.verify is None:
            return s >= 0.5, f"score {s:.2f}"
        ok = self.verify(passages[n - 1], claim)
        return ok, f"score {s:.2f}, llm {'yes' if ok else 'no'}"

    def check(self, answer, passages):
        """passages: list of passage texts, index 0 = [1]. Returns (new_answer, log)."""
        # code blocks are not claims: without this, every code line was treated as a sentence and got citations or
        # "(not from the course material)" written into the code (2026-10-25). An unclosed block (token limit) too.
        blocks = []

        def hold(m):
            blocks.append(m.group(0))
            return f"\n\u0000{len(blocks) - 1}\u0000\n"
        answer = re.sub(r"```[\s\S]*?(?:```|$)", hold, answer)
        out, log = self._check(answer, passages)
        out = re.sub(r"\n*\u0000(\d+)\u0000\n*", lambda m: "\n\n" + blocks[int(m.group(1))] + "\n\n", out)
        return out.strip(), log

    def _check(self, answer, passages):
        # the model sometimes writes "...large. [3] Next sentence": move the citation inside its own sentence
        answer = re.sub(r"([.!?])[ \t]*((?:\[\d+\])+)", r" \2\1", answer)
        log, out, pos = [], [], 0
        for s, e in sentences(answer):
            sent = answer[s:e]
            cited = [int(n) for n in CITE_ONE.findall(sent)]
            out.append(answer[pos:s]); pos = e
            if not cited:
                plain = sent.strip()
                if (len(plain.split()) >= 6 and not plain.rstrip("*_ ").endswith("?")
                        and "not from the course" not in plain and not CHAT.search(plain)):
                    scores = self.ce.predict([(plain, p) for p in passages], batch_size=8)
                    out.append(self.attach(sent, scores, log, plain, passages)); continue
                out.append(sent); continue
            claim = CITE_ONE.sub("", sent).strip()
            if len(claim.split()) < 3:  # a bare "[3]" line: nothing to support
                log.append({"sentence": sent.strip(), "action": "removed-bare", "cited": cited})
                continue
            scores = self.ce.predict([(claim, p) for p in passages], batch_size=8)
            keep = []
            best = int(max(range(len(passages)), key=lambda i: scores[i])) + 1
            best_ok = None  # computed at most once per sentence
            for n in cited:
                ok, how = self.supported(claim, n, scores, passages)
                if ok:
                    keep.append(n); log.append({"sentence": claim, "cited": n, "how": how, "action": "kept"})
                    continue
                if best != n and best not in keep:
                    if best_ok is None:
                        best_ok = self.supported(claim, best, scores, passages)
                    if best_ok[0]:
                        keep.append(best)
                        log.append({"sentence": claim, "cited": n, "how": how, "action": f"moved->{best}",
                                    "new_how": best_ok[1]})
                        continue
                log.append({"sentence": claim, "cited": n, "how": how, "action": "removed"})
            new = CITE_ONE.sub("", sent).rstrip()
            new = re.sub(r"\s+([.,;:!?])", r"\1", new)
            if keep:  # put the surviving citations before the sentence's final punctuation
                new = with_tag(new, "".join(f"[{n}]" for n in sorted(set(keep))))
            elif not new.rstrip("*_ ").endswith("?"):  # all its citations failed: treat as an uncited claim
                new = self.attach(new, scores, log, claim, passages)
            out.append(new)
        out.append(answer[pos:])
        return "".join(out).strip(), log

    def attach(self, sent, scores, log, claim, passages=None):
        best = int(max(range(len(scores)), key=lambda i: scores[i])) + 1
        ok, how = self.supported(claim, best, scores, passages)
        if ok:
            log.append({"sentence": claim, "cited": None, "action": f"added->{best}", "how": how})
            return with_tag(sent, f"[{best}]")
        log.append({"sentence": claim, "cited": None, "action": "marked-unsourced", "how": how})
        return with_tag(sent, "*(not from the course material)*")


def with_tag(sent, tag):
    """Insert tag before the sentence's final punctuation, keeping trailing whitespace."""
    body, tail = sent.rstrip(), sent[len(sent.rstrip()):]
    m = re.search(r"([.!?:]*)$", body)
    return body[:m.start()] + " " + tag + m.group(1) + tail


def load_model(name, adapter=None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForImageTextToText.from_pretrained(name, quantization_config=q, device_map="cuda:0",
                                                        dtype=torch.bfloat16)
    # the checkpoint has no generation_config.json, so generate() stopped only at <|endoftext|>; in multi-turn chats
    # the model ended its turn with <|im_end|> and went on to write the student's next message (2026-10-25)
    tok = proc.tokenizer
    model.generation_config.eos_token_id = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    if adapter:  # a LoRA adapter from scripts/finetune/train_qlora.py (models/adapters/<name>)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return proc, model


# 150 words per passage (was 220, 2026-10-21): with 6 passages the prompt was ~2,250 tokens, which made the
# inference-format training examples too long for QLoRA on 8 GB (7.8 GB peak, memory spill, 10-min steps).
# Training data (rag_format_sft.py) uses this same function, so training and inference stay identical.
def format_passages(hits, max_words=150):
    lines, sources, bodies = [], [], []
    for i, (score, m, text) in enumerate(hits, 1):
        loc = ""
        if m.get("loc_kind"):
            loc = f", {m['loc_kind']} {m['loc_start']}" + (f"-{m['loc_end']}" if m["loc_end"] != m["loc_start"] else "")
        head = m["title"] if m["heading"] in (None, m["title"]) else f"{m['title']} — {m['heading']}"
        words = " ".join(text.split()).split(" ")
        body = " ".join(words[:max_words]) + (" …" if len(words) > max_words else "")
        lines.append(f"[{i}] {head}{loc}\n{body}")
        bodies.append(f"{head}\n{body}")  # exactly what the model saw, for the citation check
        sources.append({"n": i, "doc_id": m["doc_id"], "title": head, "loc": loc.lstrip(", "),
                        "license_tier": m["license_tier"], "score": round(float(score), 4)})
    return "\n\n".join(lines), sources, bodies


class Tutor:
    def __init__(self, a):
        self.a = a
        self.index = Index(private=not a.open_only)
        self.proc, self.model = load_model(a.model, getattr(a, "adapter", None))
        self.history = []  # (role, content) of the conversation, without system prompt
        if a.no_cite_check:
            self.checker = None
        elif getattr(a, "llm_verify", False):
            self.checker = CitationChecker(verify=self.verify_claim, lo=LLM_BAND[0], hi=LLM_BAND[1])
        else:
            self.checker = CitationChecker()

    def retrieve(self, question):
        # follow-ups like "why?" need the previous question for context
        prev = [c for r, c in self.history if r == "user"][-1:]
        query = " ".join(prev + [question])
        hits = self.index.search(query, k=self.a.k * 2, mode=self.a.search)
        seen, out = {}, []
        for h in hits:  # at most 2 passages per document, so one long source cannot crowd out the rest
            d = h[1]["doc_id"]
            if seen.get(d, 0) < 2:
                seen[d] = seen.get(d, 0) + 1
                out.append(h)
            if len(out) == self.a.k:
                break
        return out

    def verify_claim(self, passage, claim):
        """LLM yes/no: does the passage state the claim? Used only for the reranker's uncertain middle band."""
        import torch
        msg = [{"role": "user", "content": VERIFY.format(p=passage, c=claim)}]
        inputs = self.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=3, do_sample=False)
        return self.proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower().startswith("yes")

    def on_topic(self, question):
        """One-token yes/no gate. Retrieval scores can't tell off-topic questions apart (an unrelated passage that
        mentions a capital city scores like a real hit), so the model judges topicality before any retrieval."""
        import torch
        prev = [c for r, c in self.history if r == "user"][-1:]  # "why?" after an NLP question is on topic
        msg = [{"role": "user", "content": TOPIC_GATE.format(context=prev[0] if prev else "(none)", q=question)}]
        inputs = self.proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=3, do_sample=False)
        unrelated = self.proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
        return not unrelated.startswith("yes")

    def ask(self, question):
        import torch
        if not self.on_topic(question):
            answer = OFF_TOPIC
            self.history += [("user", question), ("assistant", answer)]
            return answer, [], {"gen_seconds": 0.0, "prompt_tokens": 0, "new_tokens": 0, "off_topic": True,
                                "raw_answer": answer, "citation_check": []}
        hits = [] if getattr(self.a, "no_rag", False) else self.retrieve(question)
        passages, sources, bodies = format_passages(hits)
        if not hits:  # baseline: same prompt, no passages, nothing to cite or check
            passages = "(no passages: answer from your own knowledge, without citations)"
        messages = [{"role": "system", "content": SYSTEM.format(style=STYLES[self.a.style], passages=passages)}]
        messages += [{"role": r, "content": c} for r, c in self.history[-6:]]
        messages.append({"role": "user", "content": question})
        inputs = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False).to(self.model.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.a.max_new_tokens or MAX_NEW[self.a.style],
                                      do_sample=True, temperature=0.6, top_p=0.9, repetition_penalty=1.05)
        raw = self.proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        n_new = int(out.shape[1] - inputs["input_ids"].shape[1])
        answer, cite_log = (raw, []) if self.checker is None or not bodies else self.checker.check(raw, bodies)
        self.history += [("user", question), ("assistant", answer)]
        return answer, sources, {"gen_seconds": round(time.time() - t0, 1),
                                 "prompt_tokens": int(inputs["input_ids"].shape[1]), "new_tokens": n_new,
                                 "hit_token_limit": n_new >= (self.a.max_new_tokens or MAX_NEW[self.a.style]),
                                 "raw_answer": raw, "citation_check": cite_log}


def show(answer, sources, stats):
    print("\n" + answer + "\n")
    print("Sources:")
    for s in sources:
        tag = "" if s["license_tier"] == "open" else f"  [{s['license_tier']}]"
        print(f"  [{s['n']}] {s['title'][:90]}" + (f" ({s['loc']})" if s["loc"] else "") + tag)
    log = stats.get("citation_check", [])
    changed = [c for c in log if c["action"] != "kept"]
    count = lambda p: sum(c["action"].startswith(p) for c in log)
    print(f"  ({stats['new_tokens']} tokens in {stats['gen_seconds']}s; citations kept {count('kept')}, "
          f"moved {count('moved')}, removed {count('removed')}, added {count('added')}, "
          f"marked unsourced {count('marked')}"
          + ("; HIT TOKEN LIMIT" if stats.get("hit_token_limit") else "") + ")")
    for c in changed:
        print(f"    - {c['action']}" + (f" (was [{c['cited']}])" if c.get("cited") else "")
              + f", {c.get('new_how', c.get('how', '-'))}: {c['sentence'][:80]}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask")
    ap.add_argument("--file", help="one question per line; each is a fresh conversation")
    ap.add_argument("--style", choices=list(STYLES), default="hint")
    ap.add_argument("--open-only", action="store_true")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--search", choices=["dense", "hybrid", "rerank"], default="hybrid")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", help="LoRA adapter dir, e.g. models/adapters/pilot")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: 260 (hint) / 520 (explain)")
    ap.add_argument("--no-cite-check", action="store_true")
    ap.add_argument("--llm-verify", action="store_true",
                    help="ask the LLM about claims scoring 0.3-0.7 (fewer wrong citations, more sentences marked)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tutor = Tutor(a)

    if a.file:
        RUNS.mkdir(parents=True, exist_ok=True)
        log = RUNS / f"run-{time.strftime('%Y%m%d-%H%M%S')}-{a.style}.jsonl"
        qs = [q.strip() for q in open(a.file, encoding="utf-8") if q.strip() and not q.startswith("#")]
        with open(log, "w", encoding="utf-8") as f:
            for q in qs:
                tutor.history = []
                answer, sources, stats = tutor.ask(q)
                print(f"\n=== Q: {q}")
                show(answer, sources, stats)
                f.write(json.dumps({"question": q, "style": a.style, "model": a.model, "answer": answer,
                                    "sources": sources, **stats}, ensure_ascii=False) + "\n")
        print("logged to", log)
    elif a.ask:
        show(*tutor.ask(a.ask))
    else:
        print(f"NLP tutor ({a.style} mode). Ask a question; 'quit' to exit, 'new' for a fresh topic.")
        while True:
            try:
                q = input("you> ").strip()
            except EOFError:
                break
            if q.lower() in ("quit", "exit"):
                break
            if q.lower() == "new":
                tutor.history = []
                continue
            if q:
                show(*tutor.ask(q))


if __name__ == "__main__":
    main()
