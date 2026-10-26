#!/usr/bin/env bash
# Re-create the 12 cloned repos on your machine, then register them.
# Run from the project root:  bash scripts/clone_repos.sh
set -euo pipefail
mkdir -p corpus/raw/repos && cd corpus/raw/repos
for r in stanford-cs336/spring2025-lectures stanford-cs336/assignment1-basics stanford-cs336/assignment2-systems \
         stanford-cs336/assignment3-scaling stanford-cs336/assignment4-data stanford-cs336/assignment5-alignment \
         cmu-l3/anlp-spring2026-code harvardnlp/annotated-transformer karpathy/nanoGPT karpathy/minbpe \
         jacobeisenstein/gt-nlp-class d2l-ai/d2l-en; do
  d=${r/\//__}
  [ -d "$d" ] || [ -d "../../quarantine/repos/$d" ] || git -c core.longpaths=true clone -q --depth 1 "https://github.com/$r" "$d"
done
# Hugging Face course: English chapters only
if [ ! -d huggingface__course ]; then
  git -c core.longpaths=true clone -q --depth 1 --filter=blob:none --sparse https://github.com/huggingface/course huggingface__course
  git -c core.longpaths=true -C huggingface__course sparse-checkout set chapters/en
fi
cd ../../..
python scripts/register_repos.py
