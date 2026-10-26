"""Concept coverage audit of the chunk corpus (keyword-based, CPU only).

Run from the project root:  python eval/coverage_audit.py
For each concept of a standard NLP curriculum: how many chunks / distinct documents mention it, split into
trainable (open + noncommercial) and restricted. Writes eval/coverage_audit.json and prints the gaps.
Keyword matching over-counts passing mentions, so "covered" means "mentioned in enough places to be
explained", not "explained well"; weak spots should be checked by reading.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# area -> concept -> regex (case-insensitive)
CONCEPTS = {
    "foundations": {
        "regular expressions": r"regular expression|\bregex",
        "tokenization basics": r"\btokeni[sz]ation\b",
        "BPE": r"byte[- ]pair encoding|\bBPE\b",
        "WordPiece / SentencePiece / unigram LM tokenizer": r"wordpiece|sentencepiece|unigram (language model )?tokeni",
        "text normalization / stemming / lemmatization": r"lemmati[sz]|stemming|porter stemmer|case folding",
        "minimum edit distance": r"edit distance|levenshtein",
    },
    "language modelling": {
        "n-gram LM": r"\bn-?gram",
        "smoothing (Laplace/add-k)": r"laplace smoothing|add-one|add-k",
        "interpolation / backoff": r"interpolat\w+.{0,40}(n-?gram|lambda|smooth)|back-?off",
        "Kneser-Ney": r"kneser",
        "perplexity": r"perplexity",
        "neural LM (feed-forward)": r"neural (probabilistic )?language model|feed-?forward language model",
    },
    "classification": {
        "naive Bayes": r"na[iï]ve bayes",
        "logistic regression": r"logistic regression",
        "softmax / cross-entropy": r"cross-?entropy",
        "regularization": r"regulari[sz]ation|weight decay|\bL2\b",
        "precision/recall/F1": r"\bF1\b|f-?measure|precision and recall",
        "sentiment analysis": r"sentiment",
        "statistical significance / bootstrap": r"bootstrap|statistical significance|paired test",
    },
    "representations": {
        "TF-IDF": r"tf-?idf",
        "PMI / PPMI": r"\bP?PMI\b|pointwise mutual",
        "word2vec / skip-gram": r"word2vec|skip-?gram",
        "GloVe": r"\bglove\b",
        "fastText / subword embeddings": r"fasttext",
        "embedding bias": r"(gender|social) bias.{0,60}embedding|embedding.{0,60}bias",
        "contextual embeddings (ELMo)": r"\belmo\b|contextual(ized)? embedding",
    },
    "neural basics": {
        "backpropagation": r"backprop",
        "computation graph / autograd": r"computation(al)? graph|autograd|automatic differentiation",
        "activation functions": r"\brelu\b|\btanh\b|activation function",
        "dropout": r"dropout",
        "optimizers (SGD/Adam)": r"\badam\b|stochastic gradient descent",
        "weight initialization": r"initiali[sz]ation.{0,60}(weight|xavier|glorot|he )|xavier|glorot",
        "batch / layer normalization": r"layer ?norm|batch ?norm",
    },
    "sequence models": {
        "RNN": r"recurrent neural network|\bRNN",
        "LSTM": r"\bLSTM|long short-term memory",
        "GRU": r"\bGRU\b|gated recurrent",
        "vanishing/exploding gradients": r"vanishing gradient|exploding gradient",
        "bidirectional RNN": r"bidirectional (rnn|lstm|recurrent)",
        "CNN for text": r"convolutional neural network.{0,80}(text|sentence|nlp)|textcnn|1d convolution",
    },
    "sequence labelling & structure": {
        "POS tagging": r"part[- ]of[- ]speech",
        "HMM / Viterbi": r"hidden markov|viterbi",
        "CRF": r"conditional random field|\bCRF\b",
        "NER / BIO": r"named entity|\bBIO\b",
        "constituency parsing / CKY": r"constituency|\bCKY\b|context-free grammar",
        "dependency parsing": r"dependency pars",
        "coreference resolution": r"coreference",
        "semantic role labelling": r"semantic role",
        "relation / event extraction": r"relation extraction|event extraction",
        "word sense disambiguation": r"word sense",
    },
    "seq2seq & attention": {
        "encoder-decoder": r"encoder-?decoder|seq2seq|sequence-to-sequence",
        "attention (Bahdanau/Luong)": r"bahdanau|luong|additive attention",
        "machine translation": r"machine translation",
        "BLEU / MT evaluation": r"\bBLEU\b|chrF|\bCOMET\b",
        "beam search": r"beam search",
    },
    "transformers": {
        "self-attention": r"self-?attention",
        "multi-head attention": r"multi-?head",
        "positional encoding": r"positional (encoding|embedding)",
        "RoPE": r"rotary|\brope\b",
        "causal masking": r"causal (mask|attention|self-attention)|masked self-attention",
        "KV cache": r"kv[- ]cache|key-?value cache",
        "mixture of experts": r"mixture[- ]of[- ]experts|\bMoE\b",
        "efficient attention (Flash/sparse/linear)": r"flashattention|flash attention|sparse attention|linear attention",
    },
    "pretraining & LLMs": {
        "BERT / masked LM": r"\bBERT\b|masked language model",
        "GPT / autoregressive LM": r"\bGPT\b|autoregressive",
        "T5 / encoder-decoder pretraining": r"\bT5\b|span corruption",
        "scaling laws": r"scaling law",
        "pretraining data / filtering": r"(pretraining|pre-training) data|data filtering|deduplicat",
        "in-context learning / few-shot": r"in-context learning|few-shot",
        "prompting / chain-of-thought": r"chain[- ]of[- ]thought|prompt engineering",
        "instruction tuning": r"instruction[- ]tun",
        "RLHF / PPO": r"\bRLHF\b|\bPPO\b|human feedback",
        "DPO / preference optimization": r"\bDPO\b|direct preference",
        "LoRA / PEFT": r"\bLoRA\b|parameter[- ]efficient|adapter",
        "quantization": r"quantiz",
        "decoding: sampling / temperature / top-p": r"top-?p|nucleus sampling|temperature sampling|top-?k sampling",
        "speculative decoding": r"speculative decoding",
        "reasoning models / test-time compute": r"test-time compute|inference[- ]time scaling|reasoning model",
    },
    "applications & evaluation": {
        "question answering": r"question answering",
        "summarization": r"summari[sz]ation",
        "information retrieval / BM25": r"\bBM25\b|information retrieval|inverted index",
        "dense retrieval / RAG": r"dense retrieval|retrieval-augmented|\bRAG\b",
        "dialogue / chatbots": r"dialogue system|chatbot|conversational agent",
        "agents / tool use": r"tool use|\bagents?\b.{0,40}(llm|language model)|function calling",
        "LLM evaluation / benchmarks": r"benchmark|leaderboard|LLM-as-a-judge",
        "hallucination / factuality": r"hallucinat|factuality",
        "multilinguality": r"multilingual|low-resource language",
        "speech (ASR/TTS)": r"speech recognition|\bASR\b|text-to-speech",
        "multimodal (vision-language)": r"vision-language|multimodal|\bCLIP\b",
    },
    "responsible NLP": {
        "bias & fairness": r"\bbias(es)?\b.{0,40}(fair|gender|social|stereotyp)|fairness",
        "safety / jailbreaks / prompt injection": r"jailbreak|prompt injection|red[- ]teaming",
        "privacy / memorization": r"memori[sz]ation|privacy",
        "ethics / societal impact": r"ethic|societal impact",
    },
}


def main():
    chunks = []
    for name, tier_hint in (("chunks", None), ("quarantine", "restricted")):
        for line in open(ROOT / "corpus" / "chunks" / f"{name}.jsonl", encoding="utf-8"):
            c = json.loads(line)
            chunks.append((c["doc_id"], c["license_tier"], (c["title"] or "") + " " + (c["heading"] or "") + " " + c["text"]))
    report, gaps = {}, []
    for area, concepts in CONCEPTS.items():
        for concept, pat in concepts.items():
            rx = re.compile(pat, re.I)
            n = defaultdict(int)
            docs = defaultdict(set)
            for doc, tier, text in chunks:
                if rx.search(text):
                    k = "restricted" if tier == "restricted" else "trainable"
                    n[k] += 1
                    docs[k].add(doc)
            row = {"area": area, "chunks_trainable": n["trainable"], "chunks_restricted": n["restricted"],
                   "docs_trainable": len(docs["trainable"]), "docs_restricted": len(docs["restricted"]),
                   "example_docs": sorted(docs["trainable"] | docs["restricted"])[:6]}
            total = n["trainable"] + n["restricted"]
            row["status"] = ("GAP" if total < 5 else "thin" if total < 15 else
                             "restricted-only" if n["trainable"] < 3 else "ok")
            report[concept] = row
    json.dump(report, open(ROOT / "eval" / "coverage_audit.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for area in CONCEPTS:
        print(f"\n## {area}")
        for concept, r in report.items():
            if r["area"] == area:
                flag = "" if r["status"] == "ok" else f"  <-- {r['status']}"
                print(f"  {concept:45} train {r['chunks_trainable']:4} ({r['docs_trainable']:2} docs)  "
                      f"restricted {r['chunks_restricted']:4} ({r['docs_restricted']:2} docs){flag}")
    from collections import Counter
    print("\nstatus counts:", dict(Counter(r["status"] for r in report.values())))


if __name__ == "__main__":
    main()
