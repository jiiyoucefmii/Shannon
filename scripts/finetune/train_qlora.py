"""QLoRA fine-tuning of Qwen3.5-4B on corpus/splits/mix, sized for an 8 GB GPU.

Run from the project root:
  python scripts/finetune/train_qlora.py --name pilot --max-steps 40          # short pilot
  python scripts/finetune/train_qlora.py --name run1 --epochs 1               # full pass
  python scripts/finetune/train_qlora.py --name run1 --epochs 1 --resume      # continue after a stop
Checkpoints: every --save-every steps to models/adapters/<name>/checkpoint (LoRA weights, optimizer, scheduler,
RNG, log); --resume picks up from there and refuses if the data or settings changed.

Design
  - base model frozen in 4-bit NF4 (bitsandbytes), LoRA adapters on the language model only
    (full-attention q/k/v/o, linear-attention in_proj_qkv/in_proj_z/out_proj, MLP gate/up/down; vision untouched)
  - one example per assistant turn, built exactly as the tutor prompts at inference: the conversation so far +
    the chat template's generation prompt (which adds an empty <think></think>), then that turn as the target.
    At most --turns-per-conv turns per conversation, so long MathDial dialogues do not dominate.
  - loss only on the assistant tokens; logits are computed only at those positions (the 248k-token vocabulary
    would otherwise need ~1 GB per 1k-token sequence)
  - batch 1 + gradient accumulation, gradient checkpointing, paged 8-bit AdamW
  - validation loss with the adapter off (= base model) and on, per set (tutor / general)
Output: models/adapters/<name>/ (adapter weights, config, train_log.json)
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGETS = (r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
           r"|mlp\.(gate|up|down)_proj)")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def expand(rows, tok, proc, max_len, per_conv, rng):
    """(input_ids, labels, set) per chosen assistant turn; labels -100 outside the target turn."""
    out, too_long = [], 0
    for r in rows:
        msgs = r["messages"]
        turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        if len(turns) > per_conv:  # spread the picks over the conversation, always keep the last turn
            turns = sorted(rng.sample(turns[:-1], per_conv - 1)) + [turns[-1]]
        for i in turns:
            prefix = proc.apply_chat_template(msgs[:i], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
            t_ids = tok(msgs[i]["content"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(t_ids) > max_len:
                too_long += 1
                continue
            out.append((p_ids + t_ids, [-100] * len(p_ids) + t_ids, r.get("set", "?")))
    return out, too_long


def turn_loss(model, ids, labels, device):
    """Mean NLL over the target tokens, with logits only at those positions."""
    import torch
    import torch.nn.functional as F
    base = model.get_base_model()
    x = torch.tensor([ids], device=device)
    y = torch.tensor([labels], device=device)
    h = base.model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
    sel = y[:, 1:] != -100
    logits = base.lm_head(h[:, :-1][sel]).float()
    return F.cross_entropy(logits, y[:, 1:][sel]), int(sel.sum())


def evaluate(model, data, device, adapter_on):
    import torch
    model.eval()
    tot = {}
    with torch.no_grad():
        ctx = model.disable_adapter() if not adapter_on else _null()
        with ctx:
            for ids, labels, s in data:
                loss, n = turn_loss(model, ids, labels, device)
                a = tot.setdefault(s, [0.0, 0])
                a[0] += float(loss) * n
                a[1] += n
    model.train()
    return {s: round(v[0] / v[1], 4) for s, v in tot.items()}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps; 0 = --epochs over the data")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    # 3072: the inference-format tutoring items carry ~2,250 tokens of passages (median item 2,401 tokens);
    # at 1024 every one of them was silently dropped (2026-10-20). Peak at 3072: 6.8 GB.
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--turns-per-conv", type=int, default=3)
    ap.add_argument("--val-max", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--longest-first", action="store_true", help="memory test: train on the longest examples first")
    ap.add_argument("--save-every", type=int, default=50, help="checkpoint every N optimizer steps (0 = never)")
    ap.add_argument("--resume", action="store_true", help="continue from models/adapters/<name>/checkpoint")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    train, long_tr = expand(load(ROOT / "corpus/splits/mix/train.jsonl"), tok, proc, a.max_len, a.turns_per_conv, rng)
    val, long_va = expand(load(ROOT / "corpus/splits/mix/val.jsonl"), tok, proc, a.max_len, a.turns_per_conv, rng)
    if long_tr > 0.10 * (len(train) + long_tr):
        sys.exit(f"STOP: {long_tr} of {len(train) + long_tr} examples exceed --max-len {a.max_len}; "
                 "raise --max-len or shorten the prompts (the run would silently train on the wrong data)")
    rng.shuffle(train)
    rng.shuffle(val)
    if a.longest_first:  # memory test: the worst case first (a random sample missed it on 2026-10-20)
        train.sort(key=lambda ex: -len(ex[0]))
    val = val[: a.val_max]
    n_tok = sum(sum(l != -100 for l in ex[1]) for ex in train)
    print(f"train examples {len(train)} ({long_tr} over {a.max_len} tokens dropped), target tokens {n_tok:,}; "
          f"val examples {len(val)}", flush=True)

    steps = a.max_steps or math.ceil(len(train) * a.epochs / a.accum)
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(a.model, quantization_config=q, device_map="cuda:0",
                                                        dtype=torch.bfloat16)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, bias="none"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable LoRA params {trainable / 1e6:.1f} M; optimizer steps {steps} x accum {a.accum}", flush=True)
    device = next(model.parameters()).device

    out = ROOT / "models" / "adapters" / a.name
    ckpt = out / "checkpoint"
    # same data + order <=> same fingerprint; a resume on different data would silently mix two runs
    fp = hashlib.sha256(json.dumps([len(train), steps, a.accum, [ex[0][:64] for ex in train[:50]]]).encode()).hexdigest()
    opt = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    t0 = time.time()
    start, log, i = 0, [], 0
    if a.resume and (ckpt / "state.json").exists():
        st = json.load(open(ckpt / "state.json", encoding="utf-8"))
        if st["fingerprint"] != fp:
            sys.exit("STOP: checkpoint was made with different data or settings; delete it or drop --resume")
        missing = model.load_state_dict(torch.load(ckpt / "lora.pt"), strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        opt.load_state_dict(torch.load(ckpt / "optimizer.pt", weights_only=False))
        sched.load_state_dict(torch.load(ckpt / "scheduler.pt", weights_only=False))
        rng_state = torch.load(ckpt / "rng.pt", weights_only=False)
        torch.set_rng_state(rng_state["cpu"]); torch.cuda.set_rng_state(rng_state["cuda"])
        start, i, log, base_val = st["step"], st["i"], st["log"], st["base_val"]
        print(f"resumed from step {start}/{steps}; val loss, base model: {base_val}", flush=True)
    else:
        if a.resume:
            print("no checkpoint found, starting from step 0", flush=True)
        base_val = evaluate(model, val, device, adapter_on=False)
        print(f"val loss, base model: {base_val}  ({time.time() - t0:.0f}s)", flush=True)

    def save_checkpoint(step_done):
        # write to a temp dir and swap it in, so a stop mid-save never leaves a half-written checkpoint
        tmp, old = out / "checkpoint.tmp", out / "checkpoint.old"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, tmp / "lora.pt")
        torch.save(opt.state_dict(), tmp / "optimizer.pt")
        torch.save(sched.state_dict(), tmp / "scheduler.pt")
        torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}, tmp / "rng.pt")
        json.dump({"step": step_done, "i": i, "log": log, "base_val": base_val, "fingerprint": fp},
                  open(tmp / "state.json", "w", encoding="utf-8"))
        shutil.rmtree(old, ignore_errors=True)
        if ckpt.exists():
            os.replace(ckpt, old)
        os.replace(tmp, ckpt)
        shutil.rmtree(old, ignore_errors=True)
        print(f"checkpoint saved at step {step_done}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(start, steps):
        ts, tot_loss, tot_n = time.time(), 0.0, 0
        for _ in range(a.accum):
            ids, labels, _s = train[i % len(train)]
            i += 1
            loss, n = turn_loss(model, ids, labels, device)
            (loss / a.accum).backward()
            tot_loss += float(loss) * n
            tot_n += n
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        # hand cached blocks back after every step: on an 8 GB card the allocator's cache grew from the 6.9 GB
        # peak to 7.85 GB in use, spilled into system RAM, and steps slowed 14 s -> 77 s (2026-10-22)
        torch.cuda.empty_cache()
        rec = {"step": step + 1, "loss": round(tot_loss / tot_n, 4), "lr": sched.get_last_lr()[0],
               "sec": round(time.time() - ts, 1), "tokens": tot_n,
               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if a.save_every and (step + 1) % a.save_every == 0 and step + 1 < steps:
            save_checkpoint(step + 1)

    tuned_val = evaluate(model, val, device, adapter_on=True)
    print(f"val loss, base: {base_val}  ->  adapter: {tuned_val}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    json.dump({"args": vars(a), "train_examples": len(train), "dropped_too_long": long_tr, "target_tokens": n_tok,
               "trainable_params": trainable, "val_base": base_val, "val_adapter": tuned_val, "log": log,
               "minutes": round((time.time() - t0) / 60, 1)},
              open(out / "train_log.json", "w", encoding="utf-8"), indent=1)
    shutil.rmtree(ckpt, ignore_errors=True)  # the finished adapter supersedes it
    print("saved", out)


if __name__ == "__main__":
    main()
