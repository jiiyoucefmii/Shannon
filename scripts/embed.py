"""Embed corpus/chunks/*.jsonl with BAAI/bge-m3 (dense) into corpus/index/.

Run from the project root:  python scripts/embed.py [--batch 16] [--max-len 1024]

Writes, per chunk file (chunks, quarantine):
  corpus/index/<name>.npy        float16, L2-normalised, one row per chunk
  corpus/index/<name>.meta.jsonl chunk metadata without the text (row order = .npy order)
  corpus/index/info.json         model name/revision, dims, counts
Restricted chunks stay in their own index so they can be left out of anything shared.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "corpus" / "chunks"
INDEX = ROOT / "corpus" / "index"
MODEL = "BAAI/bge-m3"


def passage(r):
    # title and heading give a short chunk its context; bge-m3 needs no instruction prefix
    head = r["title"] if r["heading"] in (None, r["title"]) else f"{r['title']} — {r['heading']}"
    return f"{head}\n\n{r['text']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    a = ap.parse_args()
    INDEX.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=dev, model_kwargs={"torch_dtype": torch.float16 if dev == "cuda" else torch.float32})
    model.max_seq_length = a.max_len
    info = {"model": MODEL, "device": dev, "max_seq_length": a.max_len, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "indexes": {}}
    for name in ("chunks", "quarantine"):
        rows = [json.loads(l) for l in open(CHUNKS / f"{name}.jsonl", encoding="utf-8")]
        # longest first keeps padding per batch small
        order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["text"]))
        t0 = time.time()
        emb = model.encode([passage(rows[i]) for i in order], batch_size=a.batch, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
        out = np.zeros_like(emb)
        out[order] = emb
        np.save(INDEX / f"{name}.npy", out.astype(np.float16))
        with open(INDEX / f"{name}.meta.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k != "text"}, ensure_ascii=False) + "\n")
        info["indexes"][name] = {"n": len(rows), "dim": int(out.shape[1]), "seconds": round(time.time() - t0)}
        print(name, out.shape, f"{time.time() - t0:.0f}s")
    (INDEX / "info.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
