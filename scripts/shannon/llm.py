"""Shared access to the tutor's model on the GPU, plus the plain-text section format the generator uses.

One GPU, one model: every model or search call goes through LOCK, so chat, generation and grading take turns.
Generation and grading use the BASE model (adapter switched off), as the evaluation grader does: the fine-tuned
adapter was trained for the tutoring voice, not for writing exam items or judging answers.
"""
import contextlib
import re
import threading

LOCK = threading.RLock()
SECTION = re.compile(r"^#{2,4}\s*([A-Z][A-Z ]*?)\s*:?\s*$", re.M)
FENCE = re.compile(r"^\s*```[\w+-]*\s*\n?|\n?\s*```\s*$")


class LLM:
    def __init__(self, tutor):
        self.t = tutor

    def gen(self, messages, max_new=400, sample=False, adapter=False):
        import torch
        t = self.t
        with LOCK:
            off = not adapter and hasattr(t.model, "disable_adapter")
            with t.model.disable_adapter() if off else contextlib.nullcontext():
                x = t.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                               return_tensors="pt", enable_thinking=False).to(t.model.device)
                kw = {"do_sample": True, "temperature": 0.8, "top_p": 0.95} if sample else {"do_sample": False}
                with torch.no_grad():
                    o = t.model.generate(**x, max_new_tokens=max_new, repetition_penalty=1.05, **kw)
        return t.proc.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def yes(self, prompt):
        return self.gen([{"role": "user", "content": prompt}], max_new=3).lower().startswith("yes")

    def support(self, claim, bodies):
        """Best reranker score of the claim against the passages (the citation checker's model)."""
        if not bodies:
            return 0.0
        with LOCK:
            return float(max(self.t.checker.ce.predict([(claim, b) for b in bodies], batch_size=8)))


def sections(text):
    """'### NAME' headed sections -> {NAME: body}; code fences inside a section are removed."""
    marks = list(SECTION.finditer(text))
    out = {}
    for m, nxt in zip(marks, marks[1:] + [None]):
        body = text[m.end(): nxt.start() if nxt else len(text)].strip()
        out[m.group(1).strip()] = FENCE.sub("", body).strip("\n")
    return out
