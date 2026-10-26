"""Build eval/tutor_eval.jsonl: questions + key points for scoring tutor answers.

Run from the project root:  python eval/build_tutor_eval.py

Two sources, kept apart by `source`:
  tutoreval   closed-book TutorEval items from the Nielsen neural-networks book (official key points;
              the other TutorEval items need the chapter text or are not NLP/ML). TutorEval is restricted:
              eval use only, never training.
  author      NLP questions with key points written for this project on 2026-10-19 (not from any dataset);
              the key points are standard textbook facts, phrased as checkable statements.
"""
import json
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TE = ROOT / "corpus/quarantine/datasets/princeton-nlp__TutorEval/data/train-00000-of-00001-d787727b4981fd44.parquet"

AUTHOR = [  # (topic, question, [key points])
    ("preprocessing-tokenization", "How does byte-pair encoding build its vocabulary?",
     ["starts from characters (or bytes) as the initial vocabulary",
      "repeatedly merges the most frequent adjacent pair of symbols",
      "stops after a fixed number of merges / target vocabulary size"]),
    ("preprocessing-tokenization", "Why do modern language models use subword tokens instead of whole words?",
     ["subwords handle rare and unseen words (no out-of-vocabulary problem)",
      "keeps the vocabulary size manageable"]),
    ("preprocessing-tokenization", "What is minimum edit distance?",
     ["the minimum number of insertions, deletions and substitutions to turn one string into another",
      "computed with dynamic programming"]),
    ("ngram-lm", "What is the Markov assumption in an n-gram language model?",
     ["the probability of a word depends only on the previous n-1 words",
      "not on the whole history"]),
    ("ngram-lm", "Why do n-gram models need smoothing?",
     ["unseen n-grams would get zero probability",
      "zero probabilities make the probability of a whole test sentence zero / perplexity undefined",
      "smoothing moves some probability mass to unseen events"]),
    ("ngram-lm", "How is add-one (Laplace) smoothing computed for bigrams?",
     ["add 1 to every bigram count",
      "add the vocabulary size V to the denominator (the history count)"]),
    ("ngram-lm", "What is the idea behind Kneser-Ney smoothing?",
     ["subtracts a fixed discount from observed counts (absolute discounting)",
      "lower-order distribution uses continuation counts: how many different contexts a word appears in",
      "e.g. 'Francisco' is frequent but follows few words, so it gets low continuation probability"]),
    ("evaluation-metrics", "What is perplexity and why is lower better?",
     ["perplexity is the inverse probability of the test set normalised by the number of words (exponentiated cross-entropy)",
      "lower perplexity means the model assigns higher probability to the test data"]),
    ("embeddings", "What is the distributional hypothesis?",
     ["words that occur in similar contexts tend to have similar meanings"]),
    ("embeddings", "How does skip-gram with negative sampling train word vectors?",
     ["predicts context words from a target word",
      "treats real (word, context) pairs as positive examples and randomly sampled words as negatives",
      "trains a logistic/sigmoid classifier on the dot product of the vectors"]),
    ("embeddings", "Why use cosine similarity to compare word vectors?",
     ["cosine measures the angle between vectors",
      "it normalises away vector length, which raw dot product is sensitive to (e.g. frequency)"]),
    ("embeddings", "What is PPMI?",
     ["positive pointwise mutual information",
      "PMI compares how often two words co-occur with how often they would by chance",
      "negative values are replaced by zero"]),
    ("classification-labeling", "How does a naive Bayes text classifier make its prediction?",
     ["chooses the class maximising prior times likelihood",
      "assumes words are conditionally independent given the class (bag of words)"]),
    ("classification-labeling", "What loss does logistic regression minimise?",
     ["cross-entropy loss (negative log-likelihood)",
      "of the sigmoid (or softmax) output probability of the correct class"]),
    ("ner-ie", "How is named entity recognition turned into a sequence labeling task?",
     ["each token gets a tag",
      "BIO tagging: B = beginning of an entity, I = inside, O = outside"]),
    ("rnn-lstm", "Why do vanilla RNNs struggle with long-range dependencies?",
     ["gradients are multiplied through many time steps",
      "so they vanish (or explode)",
      "the network can't learn to use information from far back"]),
    ("rnn-lstm", "What does the forget gate in an LSTM do?",
     ["decides how much of the previous cell state to keep",
      "uses a sigmoid output between 0 and 1 multiplied elementwise with the cell state"]),
    ("rnn-lstm", "What is gradient clipping and when do you use it?",
     ["rescales the gradient when its norm exceeds a threshold",
      "used against exploding gradients, e.g. in RNN training"]),
    ("mt-seq2seq", "How does an encoder-decoder model translate a sentence?",
     ["the encoder reads the source sentence into representations",
      "the decoder generates the target sentence one token at a time conditioned on the encoder output"]),
    ("attention", "What problem did attention solve in sequence-to-sequence models?",
     ["the encoder had to compress the whole source into one fixed-length vector (bottleneck)",
      "attention lets the decoder look at all encoder states, weighting them per output step"]),
    ("attention", "What are queries, keys and values in self-attention?",
     ["each token's vector is projected into a query, a key and a value with learned matrices",
      "query-key dot products give attention weights (via softmax)",
      "the output is the weighted sum of the values"]),
    ("attention", "Why is dot-product attention scaled by the square root of d_k?",
     ["dot products grow in magnitude with the dimension d_k",
      "large values push softmax into saturated regions with tiny gradients",
      "dividing by sqrt(d_k) keeps the variance of the scores stable"]),
    ("attention", "What is multi-head attention?",
     ["several attention heads run in parallel with different learned projections",
      "each head can attend to different relationships",
      "head outputs are concatenated and projected"]),
    ("attention", "What is causal (masked) self-attention?",
     ["each position may only attend to itself and earlier positions",
      "implemented by masking future positions (e.g. setting scores to minus infinity before softmax)",
      "needed for autoregressive language modelling/decoding"]),
    ("positional-encoding", "Why do transformers need positional encodings?",
     ["self-attention by itself is permutation-invariant / has no notion of word order",
      "positional information is added to (or injected into) the token representations"]),
    ("transformer", "What is the role of residual connections and layer normalization in a transformer block?",
     ["residual connections add the sublayer input to its output, helping gradient flow in deep networks",
      "layer normalization normalises activations to stabilise training"]),
    ("mlm-autoregressive", "How is BERT pretrained?",
     ["masked language modelling: some input tokens (about 15%) are masked and predicted",
      "uses bidirectional context",
      "original BERT also used next sentence prediction"]),
    ("decoding", "What is the difference between top-k and top-p (nucleus) sampling?",
     ["top-k samples from the k most probable tokens",
      "top-p samples from the smallest set of tokens whose cumulative probability exceeds p",
      "top-p's candidate set size adapts to the shape of the distribution"]),
    ("decoding", "How does beam search work?",
     ["keeps the k (beam width) best partial sequences at each step",
      "expands each and keeps the top k by total (log) probability",
      "better than greedy decoding at finding high-probability sequences, but not guaranteed optimal"]),
    ("llm-instruction-peft", "How does LoRA reduce the number of trainable parameters?",
     ["freezes the pretrained weights",
      "learns the weight update as a product of two low-rank matrices (B A)",
      "only the low-rank matrices are trained"]),
    ("llm-instruction-peft", "What is RLHF?",
     ["humans compare/rank model outputs",
      "a reward model is trained on those preferences",
      "the language model is optimised against the reward model with RL (e.g. PPO)"]),
    ("rag", "What is retrieval-augmented generation?",
     ["retrieve relevant documents for the query",
      "condition the language model's generation on the retrieved text",
      "helps with knowledge the model lacks or that changes, and reduces hallucination"]),
]


def main():
    rows = []
    for r in pq.read_table(TE).to_pylist():
        if "Nielsen" in r["path_to_chapter"] and r["closed_book"]:
            kps = [k.strip(" -") for k in r["key_points"].split("\n") if k.strip(" -")]
            if len(kps) == 1:  # some key points are one line separated by " - "
                kps = [k.strip() for k in r["key_points"].strip(" -").split(" - ") if k.strip()]
            rows.append({"source": "tutoreval", "topic": "neural-networks", "question": r["question"],
                         "key_points": kps, "difficulty": r["difficulty"]})
    for topic, q, kps in AUTHOR:
        rows.append({"source": "author", "topic": topic, "question": q, "key_points": kps})
    with open(ROOT / "eval" / "tutor_eval.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), "questions:", sum(r["source"] == "tutoreval" for r in rows), "TutorEval,",
          sum(r["source"] == "author" for r in rows), "author;", sum(len(r["key_points"]) for r in rows), "key points")


if __name__ == "__main__":
    main()
