# Shannon: an NLP tutor that runs on one laptop

Shannon answers students' NLP questions from a library of course material, writes and marks quizzes and Python
exercises, checks for plagiarism, and gives instructors a dashboard. It runs entirely on a single 8 GB GPU
(tested on an RTX 5050 laptop) with Qwen3.5-4B in 4-bit, fine-tuned with QLoRA to tutor: hint first, cite the
course passages, refuse off-topic questions, admit gaps.

**This repository holds the code only.** The course library (textbooks, slides, papers), the trained adapter
and student data are not included; see *What is not in this repository* below.

## How it works

1. A question arrives (with the last 6 chat messages and the Hint / Explain choice).
2. Topic check: the model is asked whether the message is clearly unrelated to the course; if so it declines.
3. Hybrid search over the library: BM25 keyword search + bge-m3 embeddings, merged; best 6 passages of 150 words.
4. The prompt: tutor rules + the 6 numbered passages + the chat.
5. Qwen3.5-4B (4-bit NF4) + the LoRA adapter writes the answer, citing [1]-[6].
6. A reranker (bge-reranker-v2-m3) checks every citation; unsupported sentences are tagged
   *(not from the course material)*.
7. Python code in the answer is checked for syntax, unsafe calls, undefined names and style.
8. The page renders math (KaTeX) and code, and makes every citation open its passage.

Practice questions are generated from library passages and verified before use (blind re-solve of
multiple-choice keys, passage support for marking points, reference solutions run against their tests).
Written answers are marked point by point against the marking scheme; code is run against tests in a
separate, time-limited Python process. See [SHANNON.md](SHANNON.md) for the user guide.

## Layout

| Path | What |
| --- | --- |
| `Shannon.bat`, `scripts/shannon_server.py` | Start the platform (http://127.0.0.1:8765) |
| `scripts/shannon/` | Question generation, marking, code checks, plagiarism, storage |
| `web/shannon/` | The web pages: chat, practice, assignments, progress, instructor dashboard |
| `scripts/tutor.py`, `scripts/search.py` | The RAG tutor and hybrid search |
| `seeds/`, `scripts/download_all.py`, `scripts/manifest.py`, `scripts/fetch_web_sources.py` | What the library contains, with licences, and how to download it |
| `scripts/extract/`, `scripts/chunk.py`, `scripts/embed.py` | Turning documents into searchable passages |
| `scripts/build_*_sft.py`, `scripts/rag_format_sft.py`, `scripts/merge_tutor_sft.py`, `scripts/build_splits.py` | Building and decontaminating the training data |
| `scripts/finetune/` | Training mix and QLoRA trainer (checkpoints every 50 steps, `--resume`) |
| `eval/` | Test scripts, our own test questions, and result summaries |
| [SHANNON-TESTS.md](SHANNON-TESTS.md) | 80 manual test cases for the platform |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Then rebuild the library and the adapter, which are not in the repository. Run each step from the project root;
every script's first lines document its options.

1. **Library**: `scripts/clone_repos.sh` + `scripts/register_repos.py` (course repositories),
   `scripts/download_all.py --contact YOUR_EMAIL` (datasets, papers, course files in `seeds/`),
   `scripts/fetch_web_sources.py --contact YOUR_EMAIL` (Wikipedia and web pages), `scripts/register_docs.py`.
2. **Extraction**: `scripts/extract/inventory.py`, then `scripts/extract/extract.py PROJECT_ROOT INVENTORY_JSON`
   (documents → Markdown with page and slide numbers).
3. **Search index**: `scripts/chunk.py`, then `scripts/embed.py`.
4. **Training data**: `scripts/build_splits.py` (decontaminated splits), then the tutoring data:
   `build_quiz_sft.py`, `generate_dialogues.py`, `build_behaviour_sft.py`, `build_hint_sft.py`,
   `rag_format_sft.py`, `merge_tutor_sft.py`.
5. **Fine-tuning**: `scripts/finetune/build_mix.py`, then
   `scripts/finetune/train_qlora.py --name full1 --epochs 1 --accum 8 --val-max 200` (about 3 hours on 8 GB).
6. **Testing**: `eval/run_adapter_eval.py --name full1 --adapter models/adapters/full1`.

Start Shannon with `Shannon.bat`, or `python scripts/shannon_server.py --open`
(add `--instructor-pin 1234` to protect the instructor area, `--adapter ''` to run the base model).

## Results (adapter full1 vs the base model)

| Measure | Base | full1 |
| --- | --- | --- |
| Key points covered, NLP questions | 94.8% | 93.8% |
| Hint mode ends with a guiding question | 96.9% | 100% |
| Off-topic questions refused | 3/10 | 5/10 |
| "Not in the course" handled well | 1/6 | 5/6 |

With the library, answers cover 96.4% of key points on NLP questions, against 88.5% without it.
Search ranks the right document first almost always (MRR 0.96 on 55 queries).

## What is not in this repository

- **`corpus/`**: the course library. Much of it is copyrighted or licensed for teaching only (e.g. the
  Jurafsky & Martin textbook and slides, Eisenstein's notes). `seeds/` lists every source with its licence;
  `scripts/download_all.py` fetches what may be fetched.
- **`models/`**: the fine-tuned adapter. Its training data is partly non-commercial.
- **`data/`**: Shannon's database of questions, submissions and ratings.
- **Some test files**: the 8 TutorEval questions (restricted) and saved answer logs that quote course passages.
  Result summaries are included.

## Licence

The code is under the [MIT licence](LICENSE). It does not cover the course material, datasets or models the
code downloads or uses: each keeps its own licence, listed in `seeds/`.

## Limits

The model is small (4B parameters): generated answer keys and automatic marks can be wrong, so every generated
question needs instructor approval and marks should be spot-checked. The code runner is a guard for course
exercises, not a security sandbox. One GPU answers one request at a time.
