"""Evaluate a trained LoRA adapter on a held-out math benchmark.

Loads the base model + PEFT adapter via HF transformers, generates
completions over the chosen test set, and reports accuracy as scored by
`trl.rewards.accuracy_reward` (the same reward GRPO trains against).

Usage:
    python experiments/eval_lora.py \
        --test_set math500 \
        --lora_path runs/math_capo_20260522_090736

    python experiments/eval_lora.py \
        --test_set aime2024 \
        --lora_path runs/math_capo_20260522_090736 \
        --sampling passk --n 8 --temperature 0.7 --top_p 0.95 --seed 0
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common  # noqa: E402

from trl.rewards import accuracy_reward  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test_set", required=True, choices=sorted(_common._LOADERS))
    p.add_argument("--lora_path", required=True, help="Path to the saved PEFT adapter directory.")
    p.add_argument("--sampling", default="greedy", choices=["greedy", "sampled", "passk"])
    p.add_argument("--n", type=int, default=1, help="Samples per prompt (passk only).")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--num_samples", type=int, default=0, help="0 = full split.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base_model", default=None, help="Override base model id; default reads adapter_config.json.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output", default=None, help="Optional JSONL output of per-prompt results.")
    return p.parse_args()


def load_dataset(test_set: str):
    loader, _train_split, eval_split = _common._LOADERS[test_set]
    return loader(eval_split)


def generation_kwargs(args: argparse.Namespace) -> tuple[dict, int]:
    if args.sampling == "greedy":
        return dict(do_sample=False, num_return_sequences=1), 1
    if args.sampling == "sampled":
        return (
            dict(do_sample=True, temperature=args.temperature, top_p=args.top_p, num_return_sequences=1),
            1,
        )
    return (
        dict(do_sample=True, temperature=args.temperature, top_p=args.top_p, num_return_sequences=args.n),
        args.n,
    )


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    lora_path = Path(args.lora_path).resolve()
    adapter_cfg = json.loads((lora_path / "adapter_config.json").read_text())
    base_model_id = args.base_model or adapter_cfg["base_model_name_or_path"]

    ds = load_dataset(args.test_set)
    if args.num_samples > 0:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    tokenizer = AutoTokenizer.from_pretrained(str(lora_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, str(lora_path))
    model.eval()

    prompts = [
        tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        for ex in ds
    ]
    solutions = [ex["solution"] for ex in ds]

    gen_kwargs, n_per_prompt = generation_kwargs(args)
    gen_kwargs.update(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    all_completions: list[list[str]] = []  # one inner list per prompt, len n_per_prompt
    for start in tqdm(range(0, len(prompts), args.batch_size)):
        batch = prompts[start : start + args.batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        prompt_len = enc["input_ids"].shape[1]
        gen_only = out[:, prompt_len:]
        decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        # decoded is [batch_size * n_per_prompt]; rows for prompt i are
        # decoded[i*n : (i+1)*n] because HF lays out num_return_sequences per input.
        for i in range(len(batch)):
            all_completions.append(decoded[i * n_per_prompt : (i + 1) * n_per_prompt])
        print(f"  generated {start + len(batch)}/{len(prompts)}", flush=True)

    completions_flat: list[list[dict]] = []
    solutions_flat: list[str] = []
    for sol, samples in zip(solutions, all_completions, strict=True):
        for text in samples:
            completions_flat.append([{"role": "assistant", "content": text}])
            solutions_flat.append(sol)

    rewards = accuracy_reward(completions_flat, solutions_flat)

    per_prompt: list[list[float | None]] = [
        rewards[i * n_per_prompt : (i + 1) * n_per_prompt] for i in range(len(ds))
    ]

    pass1_per_prompt: list[float] = []
    passk_per_prompt: list[float] = []
    n_unparseable = 0
    for row in per_prompt:
        scored = [r for r in row if r is not None]
        if not scored:
            n_unparseable += 1
            continue
        pass1_per_prompt.append(sum(scored) / len(scored))
        passk_per_prompt.append(1.0 if any(s > 0 for s in scored) else 0.0)

    n_eff = len(pass1_per_prompt)
    pass1 = sum(pass1_per_prompt) / n_eff if n_eff else float("nan")
    passk = sum(passk_per_prompt) / n_eff if n_eff else float("nan")
    sem = math.sqrt(pass1 * (1 - pass1) / n_eff) if n_eff else float("nan")

    print()
    print(f"test_set       : {args.test_set}")
    print(f"lora_path      : {lora_path}")
    print(f"base_model     : {base_model_id}")
    print(f"sampling       : {args.sampling} (n={n_per_prompt}, T={args.temperature}, top_p={args.top_p}, seed={args.seed})")
    print(f"num_prompts    : {len(ds)}  (scored: {n_eff}, unparseable gold: {n_unparseable})")
    print(f"pass@1 (mean)  : {pass1:.4f}  (sem {sem:.4f})")
    if n_per_prompt > 1:
        print(f"pass@{n_per_prompt:<8} : {passk:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for i, ex in enumerate(ds):
                f.write(json.dumps({
                    "idx": i,
                    "prompt": ex["prompt"],
                    "solution": ex["solution"],
                    "completions": all_completions[i],
                    "rewards": per_prompt[i],
                }, ensure_ascii=False) + "\n")
        print(f"wrote per-prompt results -> {out_path}")


if __name__ == "__main__":
    main()
