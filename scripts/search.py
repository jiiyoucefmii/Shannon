"""Query the bge-m3 index built by scripts/embed.py.

  python scripts/search.py "how does multi-head attention work"            # open + noncommercial only
  python scripts/search.py "kneser-ney smoothing" --private -k 8           # also search restricted material
  python scripts/search.py "lora" --topic llm-instruction-peft --type slides

Restricted chunks are only searched with --private (private RAG / eval use only).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "corpus" / "index"
CHUNKS = ROOT / "corpus" / "chunks"
RERANKER = "BAAI/bge-reranker-v2-m3"
RRF_K = 60  # standard reciprocal-rank-fusion constant


def tokens(t):
    return re.findall(r"\w+", t.lower())


class Index:
    def __init__(self, private=False):
        names = ["chunks"] + (["quarantine"] if private else [])
        self.emb = np.concatenate([np.load(INDEX / f"{n}.npy").astype(np.float32) for n in names])
        self.meta, self.src = [], []
        for n in names:
            for i, l in enumerate(open(INDEX / f"{n}.meta.jsonl", encoding="utf-8")):
                self.meta.append(json.loads(l))
                self.src.append((n, i))
        self.model = self.reranker = self.bm25 = None
        self._text = {}

    def encode(self, q):
        if self.model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            info = json.loads((INDEX / "info.json").read_text())
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            # fp16 on GPU: ~1 GB less VRAM, which the tutor needs next to its LLM on an 8 GB card
            self.model = SentenceTransformer(info["model"], device=dev,
                                             model_kwargs={"torch_dtype": torch.float16 if dev == "cuda" else torch.float32})
        return self.model.encode([q], normalize_embeddings=True)[0]

    def text(self, j):
        name, i = self.src[j]
        if name not in self._text:
            self._text[name] = [json.loads(l)["text"] for l in open(CHUNKS / f"{name}.jsonl", encoding="utf-8")]
        return self._text[name][i]

    def passage(self, j):
        m = self.meta[j]
        head = m["title"] if m["heading"] in (None, m["title"]) else f"{m['title']} — {m['heading']}"
        return f"{head}\n\n{self.text(j)}"

    def bm25_scores(self, q):
        if self.bm25 is None:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([tokens(self.passage(j)) for j in range(len(self.meta))])
        return self.bm25.get_scores(tokens(q))

    def rerank(self, q, cands):
        if self.reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.reranker = CrossEncoder(RERANKER, device=dev, max_length=1024,
                                         model_kwargs={"torch_dtype": torch.float16 if dev == "cuda" else torch.float32})
        scores = self.reranker.predict([(q, self.passage(j)) for j in cands], batch_size=8)
        return sorted(zip(cands, map(float, scores)), key=lambda x: -x[1])

    def search(self, q, k=5, topic=None, rtype=None, skip_suspect=True, mode="dense", pool=30):
        """mode: dense | hybrid (dense + BM25, reciprocal-rank fusion) | rerank (hybrid pool -> bge-reranker)."""
        mask = np.ones(len(self.meta), bool)
        for j, m in enumerate(self.meta):
            if (topic and topic not in (m.get("topics") or [])) or (rtype and m["resource_type"] != rtype) \
                    or (skip_suspect and m.get("suspect_math")):
                mask[j] = False
        dense = self.emb @ self.encode(q)
        dense[~mask] = -np.inf
        if mode == "dense":
            top = [j for j in np.argsort(-dense)[:k] if mask[j]]
            return [(float(dense[j]), self.meta[j], self.text(j)) for j in top]
        lex = self.bm25_scores(q)
        lex[~mask] = -np.inf
        fused = {}
        for ranking in (np.argsort(-dense)[:pool], np.argsort(-lex)[:pool]):
            for r, j in enumerate(ranking, 1):
                if mask[j]:
                    fused[j] = fused.get(j, 0.0) + 1 / (RRF_K + r)
        order = sorted(fused, key=lambda j: -fused[j])
        if mode == "hybrid":
            return [(fused[j], self.meta[j], self.text(j)) for j in order[:k]]
        return [(s, self.meta[j], self.text(j)) for j, s in self.rerank(q, order[:pool])[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--private", action="store_true", help="include restricted (quarantine) chunks")
    ap.add_argument("--topic")
    ap.add_argument("--type", dest="rtype")
    ap.add_argument("--mode", choices=["dense", "hybrid", "rerank"], default="hybrid")
    a = ap.parse_args()
    for score, m, text in Index(a.private).search(a.query, a.k, a.topic, a.rtype, mode=a.mode):
        loc = f"{m['loc_kind']} {m['loc_start']}" + (f"-{m['loc_end']}" if m["loc_end"] != m["loc_start"] else "") if m["loc_kind"] else ""
        print(f"{score:.3f}  [{m['license_tier']}] {m['title']} — {m['heading']}  ({loc})")
        print("       " + " ".join(text.split())[:220] + "\n")


if __name__ == "__main__":
    main()
