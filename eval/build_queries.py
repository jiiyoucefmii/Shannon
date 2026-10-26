"""Build eval/retrieval_queries.jsonl: student-style questions, each with every doc that is a correct source.

Run from the project root:  python eval/build_queries.py
Edit QUERIES to extend the set; unknown doc ids are reported.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = "slp3-2026aug-textbook-ed3book_aug26"
C = "cs224n-2026-slides-cs224n-2026-"
M = "cmu-anlp-2026s-slides-anlp-s2026-"
S = "slp3-2026aug-slides-"
N = "cs224n-2026-notes-"
ATT = ["arxiv-1706.03762", T, S + "transformer_aug26", C + "lecture05-transformers",
       N + "cs224n-self-attention-transformers-2023_draft", M + "05-attention-transformers"]
RNN = [T, S + "rnnjan25", C + "lecture04-rnnlm", M + "04-rnns", N + "cs224n-2019-notes05-lm_rnn"]

QUERIES = [
    ("preprocessing-tokenization", "how does byte pair encoding decide which symbols to merge",
     [T, "arxiv-1508.07909", S + "tokens_jan26", C + "lecture14-guest-julie-tokenization-multilinguality"]),
    ("preprocessing-tokenization", "why do subword tokenizers handle rare and unseen words better than word-level vocabularies",
     ["arxiv-1508.07909", T, S + "tokens_jan26", C + "lecture14-guest-julie-tokenization-multilinguality"]),
    ("preprocessing-tokenization", "why are some languages more expensive to process with an English-centric tokenizer",
     [C + "lecture14-guest-julie-tokenization-multilinguality", T]),
    ("preprocessing-tokenization", "what is minimum edit distance and how is it computed with dynamic programming",
     [T, S + "med24"]),
    ("ngram-lm", "what is the Markov assumption in an n-gram language model",
     [T, S + "lm_jan25", M + "03-lm", N + "cs224n-2019-notes05-lm_rnn"]),
    ("ngram-lm", "how does add-one Laplace smoothing change bigram probabilities", [T, S + "lm_jan25"]),
    ("ngram-lm", "what does Kneser-Ney continuation probability capture", [T, S + "lm_jan25"]),
    ("evaluation-metrics", "how is perplexity computed and why is lower better",
     [T, S + "lm_jan25", M + "03-lm", C + "lecture04-rnnlm", N + "cs224n-2019-notes05-lm_rnn"]),
    ("embeddings", "what is the distributional hypothesis behind word vectors",
     [T, S + "vector25aug", C + "lecture02-wordvecs", N + "cs224n_winter2023_lecture1_notes_draft", M + "02-representations"]),
    ("embeddings", "skip-gram with negative sampling training objective",
     ["paper-mikolov-et-al-2013-neurips-word2vec-negative-sampling", "arxiv-1301.3781", T, S + "vector25aug",
      C + "lecture02-wordvecs", N + "cs224n_winter2023_lecture1_notes_draft"]),
    ("embeddings", "how does GloVe use global co-occurrence counts",
     ["paper-glove", N + "cs224n-2019-notes02-wordvecs2", C + "lecture02-wordvecs"]),
    ("embeddings", "what is PPMI and why use it instead of raw counts", [T, S + "vector25aug"]),
    ("embeddings", "cosine similarity between word vectors",
     [T, S + "vector25aug", C + "lecture02-wordvecs", N + "cs224n_winter2023_lecture1_notes_draft"]),
    ("classification-labeling", "naive Bayes classifier for sentiment with bag of words", [T, S + "nb24aug"]),
    ("classification-labeling", "logistic regression cross-entropy loss and its gradient", [T, S + "logreg25aug"]),
    ("classification-labeling", "why use softmax for multinomial logistic regression", [T, S + "logreg25aug", S + "nn25aug"]),
    ("ner-ie", "how is named entity recognition framed as BIO sequence labeling", [T, S + "8_posner_intro_may_6_2021"]),
    ("ner-ie", "what is part-of-speech tagging", [T, S + "8_posner_intro_may_6_2021"]),
    ("rnn-lstm", "why do vanilla RNNs suffer from vanishing and exploding gradients", ["arxiv-1211.5063"] + RNN),
    ("rnn-lstm", "how does gradient clipping help train recurrent networks",
     ["arxiv-1211.5063", C + "lecture04-rnnlm", N + "cs224n-2019-notes05-lm_rnn", T]),
    ("rnn-lstm", "what do the input forget and output gates in an LSTM do", RNN),
    ("mt-seq2seq", "encoder-decoder sequence to sequence model for translation",
     ["arxiv-1409.3215", "arxiv-1406.1078", T, S + "rnnjan25"]),
    ("attention", "how did Bahdanau attention fix the fixed-length bottleneck in seq2seq",
     ["arxiv-1409.0473", T, C + "lecture05-transformers", M + "05-attention-transformers"]),
    ("attention", "why scale dot-product attention by the square root of the key dimension", ATT),
    ("attention", "what are queries keys and values in self-attention", ATT),
    ("attention", "why use multiple attention heads", ATT),
    ("attention", "causal masking so the decoder cannot look at future tokens", ATT),
    ("attention", "grouped-query attention shares key and value heads to shrink the KV cache",
     ["arxiv-2305.13245", M + "22-longcontext", M + "20-scaling-parallelism"]),
    ("attention", "FlashAttention tiling to reduce memory reads and writes",
     ["arxiv-2205.14135", M + "22-longcontext", M + "20-scaling-parallelism"]),
    ("positional-encoding", "sinusoidal positional encodings in the transformer",
     ["arxiv-1706.03762", T, C + "lecture05-transformers", N + "cs224n-self-attention-transformers-2023_draft",
      M + "05-attention-transformers"]),
    ("positional-encoding", "rotary position embeddings rotate query and key vectors",
     ["arxiv-2104.09864", M + "05-attention-transformers", M + "22-longcontext", T]),
    ("transformer", "pre-layer-norm versus post-layer-norm transformer training stability",
     ["arxiv-2002.04745", "arxiv-1607.06450", M + "05-attention-transformers"]),
    ("transformer", "RMSNorm versus LayerNorm", ["arxiv-1910.07467", "arxiv-1607.06450", M + "05-attention-transformers"]),
    ("transformer", "residual connections and the feed-forward sublayer in a transformer block", ATT),
    ("mlm-autoregressive", "masked language modeling in BERT",
     ["arxiv-1810.04805", T, S + "mlmjan25", C + "lecture07-pretraining", M + "06-pretraining"]),
    ("mlm-autoregressive", "difference between encoder-only and decoder-only pretrained models",
     [T, C + "lecture07-pretraining", M + "06-pretraining", S + "mlmjan25"]),
    ("pretraining-finetuning", "scaling laws relating loss to parameters data and compute",
     ["arxiv-2001.08361", "arxiv-2203.15556", M + "20-scaling-parallelism", C + "lecture07-pretraining"]),
    ("pretraining-finetuning", "compute-optimal number of training tokens per parameter",
     ["arxiv-2203.15556", "arxiv-2001.08361", M + "20-scaling-parallelism"]),
    ("llm-instruction-peft", "in-context few-shot learning without gradient updates",
     ["arxiv-2005.14165", T, M + "07-pretraining2-prompting", C + "lecture07-pretraining"]),
    ("llm-instruction-peft", "instruction tuning improves zero-shot performance",
     ["arxiv-2109.01652", "arxiv-2210.11416", T, C + "lecture08-posttraining", M + "08-finetuning"]),
    ("llm-instruction-peft", "LoRA low-rank update matrices",
     ["arxiv-2106.09685", "arxiv-2305.14314", C + "lecture09-peft", M + "08-finetuning"]),
    ("llm-instruction-peft", "adapters versus full fine-tuning parameter efficiency",
     ["arxiv-1902.00751", "arxiv-2106.09685", C + "lecture09-peft", M + "08-finetuning"]),
    ("llm-instruction-peft", "how RLHF trains a reward model from human comparisons",
     ["arxiv-2203.02155", T, C + "lecture08-posttraining", M + "17-rl-llms", M + "16-rl"]),
    ("llm-instruction-peft", "DPO optimizes preferences without a separate reward model",
     ["arxiv-2305.18290", T, C + "lecture08-posttraining", M + "17-rl-llms"]),
    ("llm-instruction-peft", "chain-of-thought prompting for multi-step reasoning",
     ["arxiv-2201.11903", "arxiv-2203.11171", C + "lecture12-reasoning-part1", M + "07-pretraining2-prompting",
      M + "23-inference-scaling", T]),
    ("decoding", "greedy decoding versus beam search versus top-p sampling",
     ["arxiv-2406.16838", T, M + "09-decoding", C + "lecture08-posttraining"]),
    ("decoding", "speculative decoding with a small draft model",
     ["arxiv-2211.17192", M + "09-decoding", M + "23-inference-scaling", "arxiv-2406.16838"]),
    ("rag", "retrieval-augmented generation combines a retriever with a generator",
     ["arxiv-2005.11401", "arxiv-2310.11511", T, M + "10-rag", C + "lecture10-rag-agents"]),
    ("rag", "dense retrieval versus sparse BM25 retrieval",
     [T, M + "10-rag", S + "ir_nov25", C + "lecture10-rag-agents", "arxiv-2005.11401"]),
    ("rag", "when should a model rely on retrieval instead of parametric memory",
     ["arxiv-2212.10511", "arxiv-2310.11511", M + "10-rag"]),
    ("advanced", "int8 quantization of large model weights and outlier features",
     ["arxiv-2208.07339", M + "19-quantization", "arxiv-2305.14314"]),
    ("advanced", "mixture of experts routing tokens to experts", [M + "21-moe"]),
    ("evaluation-metrics", "what does MMLU measure",
     ["arxiv-2009.03300", "arxiv-2406.01574", M + "13-evaluation", C + "lecture11-evaluation"]),
    ("evaluation-metrics", "using an LLM as a judge to evaluate open-ended answers",
     ["arxiv-2306.05685", M + "13-evaluation", C + "lecture11-evaluation"]),
    ("evaluation-metrics", "pass@k metric for code generation", ["arxiv-2107.03374", "arxiv-2108.07732", M + "13-evaluation"]),
]


# Sources added 2026-10-18 (gap fill). Appended, never prepended, so each query's canonical
# first-listed source (used by the strict metric) is unchanged.
E = "eisenstein-nlp-notes"
EXTRA = {
    "how does byte pair encoding decide which symbols to merge": ["d2l-subword-embedding"],
    "what is the Markov assumption in an n-gram language model": ["d2l-language-model", E],
    "how does add-one Laplace smoothing change bigram probabilities": ["d2l-language-model", E, "arxiv-cmp-lg/9606011", "arxiv-cs/0108005"],
    "what does Kneser-Ney continuation probability capture": [E, "arxiv-cmp-lg/9606011", "arxiv-cs/0108005"],
    "how is perplexity computed and why is lower better": ["d2l-language-model", E],
    "what is the distributional hypothesis behind word vectors": [E],
    "skip-gram with negative sampling training objective": ["d2l-word2vec", "d2l-approx-training", E],
    "how does GloVe use global co-occurrence counts": ["d2l-glove", E],
    "what is PPMI and why use it instead of raw counts": [E],
    "cosine similarity between word vectors": ["d2l-similarity-analogy"],
    "naive Bayes classifier for sentiment with bag of words": [E],
    "logistic regression cross-entropy loss and its gradient": [E],
    "how is named entity recognition framed as BIO sequence labeling": [E],
    "what is part-of-speech tagging": [E],
    "why do vanilla RNNs suffer from vanishing and exploding gradients": ["d2l-bptt", "d2l-rnn", E],
    "how does gradient clipping help train recurrent networks": ["d2l-rnn-scratch", "d2l-bptt"],
    "what do the input forget and output gates in an LSTM do": ["d2l-lstm", E],
    "encoder-decoder sequence to sequence model for translation": ["d2l-seq2seq", "d2l-encoder-decoder", E, "arxiv-1702.01806"],
    "how did Bahdanau attention fix the fixed-length bottleneck in seq2seq": ["d2l-bahdanau-attention", E],
    "why scale dot-product attention by the square root of the key dimension": ["d2l-attention-scoring-functions"],
    "what are queries keys and values in self-attention": ["d2l-queries-keys-values", "d2l-attention-scoring-functions", "d2l-self-attention-and-positional-encoding"],
    "why use multiple attention heads": ["d2l-multihead-attention"],
    "causal masking so the decoder cannot look at future tokens": ["d2l-transformer"],
    "sinusoidal positional encodings in the transformer": ["d2l-self-attention-and-positional-encoding"],
    "residual connections and the feed-forward sublayer in a transformer block": ["d2l-transformer"],
    "masked language modeling in BERT": ["d2l-bert", "d2l-bert-pretraining", "d2l-bert-dataset"],
    "difference between encoder-only and decoder-only pretrained models": ["d2l-large-pretraining-transformers"],
    "greedy decoding versus beam search versus top-p sampling":
        ["d2l-beam-search", "arxiv-1904.09751", "arxiv-1805.04833", "arxiv-2010.02650", "arxiv-1702.01806"],
}


# Open-repo documents added 2026-10-19 (HF LLM course, Annotated Transformer, CS336, minbpe, nanoGPT).
# Same rule: appended only where the document genuinely answers the query.
EXTRA_REPOS = {
    "how does byte pair encoding decide which symbols to merge": ["hfcourse-ch6-5", "minbpe-readme", "minbpe-lecture",
                                                                  "minbpe-basic", "assignment1-basics-cs336-assignment1-basics"],
    "why do subword tokenizers handle rare and unseen words better than word-level vocabularies":
        ["hfcourse-ch2-4", "hfcourse-ch6-1", "hfcourse-ch6-5", "hfcourse-ch6-6", "hfcourse-ch6-7"],
    "why are some languages more expensive to process with an English-centric tokenizer": ["minbpe-lecture"],
    "how is named entity recognition framed as BIO sequence labeling": ["hfcourse-ch7-2"],
    "encoder-decoder sequence to sequence model for translation": ["hfcourse-ch7-4", "hfcourse-ch1-6"],
    "why scale dot-product attention by the square root of the key dimension":
        ["annotated-transformer-annotatedtransformer", "assignment1-basics-cs336-assignment1-basics"],
    "what are queries keys and values in self-attention": ["annotated-transformer-annotatedtransformer"],
    "why use multiple attention heads": ["annotated-transformer-annotatedtransformer"],
    "causal masking so the decoder cannot look at future tokens":
        ["annotated-transformer-annotatedtransformer", "nanogpt-model", "assignment1-basics-cs336-assignment1-basics"],
    "sinusoidal positional encodings in the transformer": ["annotated-transformer-annotatedtransformer"],
    "rotary position embeddings rotate query and key vectors": ["assignment1-basics-cs336-assignment1-basics",
                                                                "spring2025-lectures-2025-lecture-3-architecture"],
    "pre-layer-norm versus post-layer-norm transformer training stability":
        ["spring2025-lectures-2025-lecture-3-architecture", "assignment1-basics-cs336-assignment1-basics"],
    "RMSNorm versus LayerNorm": ["spring2025-lectures-2025-lecture-3-architecture",
                                 "assignment1-basics-cs336-assignment1-basics"],
    "residual connections and the feed-forward sublayer in a transformer block":
        ["annotated-transformer-annotatedtransformer", "assignment1-basics-cs336-assignment1-basics"],
    "masked language modeling in BERT": ["hfcourse-ch7-3", "hfcourse-ch1-6"],
    "difference between encoder-only and decoder-only pretrained models": ["hfcourse-ch1-6", "hfcourse-ch1-5"],
    "scaling laws relating loss to parameters data and compute": ["spring2025-lectures-2025-lecture-9-scaling-laws-basics",
                                                                  "spring2025-lectures-2025-lecture-11-scaling-details",
                                                                  "nanogpt-scaling-laws"],
    "compute-optimal number of training tokens per parameter": ["spring2025-lectures-2025-lecture-9-scaling-laws-basics",
                                                                "nanogpt-scaling-laws"],
    "LoRA low-rank update matrices": ["hfcourse-ch11-4"],
    "how RLHF trains a reward model from human comparisons": ["spring2025-lectures-2025-lecture-15-rlhf-alignment"],
    "DPO optimizes preferences without a separate reward model": ["spring2025-lectures-2025-lecture-15-rlhf-alignment"],
    "greedy decoding versus beam search versus top-p sampling": ["hfcourse-ch1-8"],
    "grouped-query attention shares key and value heads to shrink the KV cache":
        ["spring2025-lectures-2025-lecture-3-architecture"],
    "FlashAttention tiling to reduce memory reads and writes": ["assignment2-systems-cs336-assignment2-systems"],
    "mixture of experts routing tokens to experts": ["spring2025-lectures-2025-lecture-4-moes"],
}


# Web sources added 2026-10-19 (Wikipedia, HF generation docs, OWASP, agents course), same rule.
EXTRA_WEB = {
    "greedy decoding versus beam search versus top-p sampling":
        ["transformers-docs-source-en-generation-strategies", "wikipedia-beam-search", "wikipedia-top-p-sampling"],
    "speculative decoding with a small draft model": ["transformers-docs-source-en-generation-strategies"],
    "what does Kneser-Ney continuation probability capture": ["wikipedia-kneser-ney-smoothing"],
    "how does add-one Laplace smoothing change bigram probabilities": ["wikipedia-additive-smoothing"],
    "what is PPMI and why use it instead of raw counts": ["wikipedia-pointwise-mutual-information"],
    "what is minimum edit distance and how is it computed with dynamic programming": [],
}


def main():
    for q, docs in list(EXTRA_REPOS.items()) + list(EXTRA_WEB.items()):
        EXTRA[q] = EXTRA.get(q, []) + docs
    for i, (topic, q, rel) in enumerate(QUERIES):
        QUERIES[i] = (topic, q, rel + [d for d in EXTRA.get(q, []) if d not in rel])
    unused = set(EXTRA) - {q for _, q, _ in QUERIES}
    assert not unused, f"EXTRA keys match no query: {unused}"
    ids = set()
    for fn in ("chunks", "quarantine"):
        ids |= {json.loads(l)["doc_id"] for l in open(ROOT / "corpus" / "index" / f"{fn}.meta.jsonl", encoding="utf-8")}
    bad = sorted({d for _, _, rel in QUERIES for d in rel if d not in ids})
    with open(ROOT / "eval" / "retrieval_queries.jsonl", "w", encoding="utf-8") as f:
        for topic, q, rel in QUERIES:
            f.write(json.dumps({"topic": topic, "query": q, "relevant": rel}) + "\n")
    print(len(QUERIES), "queries;", "unknown ids:", bad or "none")


if __name__ == "__main__":
    main()
