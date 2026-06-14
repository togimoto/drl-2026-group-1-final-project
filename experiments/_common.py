"""Shared helpers for GRPO pilot scripts.

Each script in this folder imports from here so the per-experiment files stay
tutorial-short and only express what differs between runs.

Supports three tasks via the `TASK` env var:
- `math`: EleutherAI/hendrycks_math (~7.5k train / 5k test, concatenated across all
  7 subjects). Filter by difficulty via `MATH_LEVELS` (e.g. `MATH_LEVELS=3,4,5`).
- `math500` (default): HuggingFaceH4/MATH-500 (500 problems).
- `minervamath`: math-ai/minervamath (272 problems).

Set `EVAL_TASK` to use a different dataset for eval than for training
(e.g. `TASK=math EVAL_TASK=math500`). Defaults to `TASK`.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import random
import re
from datetime import datetime

from datasets import concatenate_datasets, load_dataset

from trl import GRPOConfig
from trl.rewards import accuracy_reward


MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "300"))
USE_VLLM = os.environ.get("USE_VLLM", "0") == "1"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "caps")
# Eval is off by default; set `DO_EVAL=1` to enable periodic test-set eval.
DO_EVAL = os.environ.get("DO_EVAL", "0") == "1"
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "25"))
# 0 = use the full eval split. Default 256 keeps eval fast (~1 min/eval).
EVAL_SAMPLES = int(os.environ.get("EVAL_SAMPLES", "256"))
TASK = os.environ.get("TASK", "math").lower()
# Defaults to TASK; set explicitly to eval on a different dataset than you trained on
# (e.g. `TASK=math500 EVAL_TASK=minervamath`).
EVAL_TASK = os.environ.get("EVAL_TASK", TASK).lower()
# Comma-separated MATH difficulty levels to keep (e.g. "3,4,5"). Empty = all levels.
# Only applies when `TASK=math`.
MATH_LEVELS = os.environ.get("MATH_LEVELS", "")
# Generic GRPOConfig override hook. Set GRPO_OVERRIDES to a JSON object to tweak
# any GRPOConfig field from the launch command without editing this file.
GRPO_OVERRIDES = json.loads(os.environ.get("GRPO_OVERRIDES", "{}"))
# When set (to any value), draw a random seed at config-build time.
RANDOM_SEED = "RANDOM_SEED" in os.environ
# When `SAVE_FINAL_MODEL=1`, scripts call `trainer.save_model()` after `.train()`
# so the final policy weights land under `runs/<run_name>/`. Off by default to
# avoid filling disk during smoke runs / sweeps. Named `SAVE_FINAL_MODEL` (not
# `SAVE_MODEL`) to disambiguate from mid-run checkpointing — `save_strategy` in
# `base_config` is `"no"`, so this only controls the post-`.train()` dump.
SAVE_FINAL_MODEL = os.environ.get("SAVE_FINAL_MODEL", "0") == "1"
# LoRA is off by default; set `USE_LORA=1` to wrap the policy in a PEFT LoRA adapter.
USE_LORA = os.environ.get("USE_LORA", "0") == "1"
LORA_R = int(os.environ.get("LORA_R", "1"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_TARGET_MODULES = os.environ.get("LORA_TARGET_MODULES", "all-linear")

# Hendrycks MATH ships 7 subject configs; we concatenate them into a single dataset.
_MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
# Matches `\boxed{...}` with one level of brace nesting (covers `\frac{a}{b}` etc.).
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def _extract_boxed(solution: str) -> str:
    # Re-wrap the extracted answer in `\boxed{...}` so `math_verify.parse` uses its
    # boxed extractor; on a bare LaTeX expression like `288\sqrt{3}` it would
    # otherwise fall back to expression-extraction and pick up only the leading
    # integer (`288`).
    matches = _BOXED_RE.findall(solution)
    return rf"\boxed{{{matches[-1]}}}" if matches else solution


def _format_math(sample):
    return {
        "prompt": [{"role": "user", "content": sample["problem"]}],
        "solution": _extract_boxed(sample["solution"]),
    }


def _load_math(split):
    parts = [
        load_dataset("EleutherAI/hendrycks_math", subj, split=split, trust_remote_code=True)
        for subj in _MATH_SUBJECTS
    ]
    ds = concatenate_datasets(parts)
    if MATH_LEVELS:
        wanted = {f"Level {lv.strip()}" for lv in MATH_LEVELS.split(",") if lv.strip()}
        ds = ds.filter(lambda s: s["level"] in wanted)
    return ds.map(_format_math, remove_columns=ds.column_names)

def _format_math500(sample):
    return {
        "prompt": [{"role": "user", "content": sample["problem"]}],
        "solution": sample["answer"],
    }


def _load_math500(split):
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    return ds.map(_format_math500, remove_columns=ds.column_names)


def _format_minervamath(sample):
    # MinervaMath answers are bare LaTeX (e.g. `\frac{...}{...}`) with no math
    # delimiters, which math_verify.parse() can't extract (returns []). Wrap in
    # `$...$` so the LaTeX extractor fires — takes unparseable golds 53 -> 0.
    return {
        "prompt": [{"role": "user", "content": sample["question"]}],
        "solution": f"${sample['answer']}$",
    }


def _load_minervamath(split):
    ds = load_dataset("math-ai/minervamath", split=split)
    # `load_from_cache_file=False`: the `$...$` wrap in `_format_minervamath` is a
    # gold-answer transform whose output is fed straight to `math_verify.parse`. A
    # stale map cache from before that wrap landed silently served bare answers
    # (53/272 unparseable golds). The dataset is only 272 rows, so re-mapping every
    # load is free and removes the stale-cache hazard.
    return ds.map(_format_minervamath, remove_columns=ds.column_names, load_from_cache_file=False)


# MATH-500 and MinervaMath each ship a single split (500 and 272 rows), so
# train and eval point at the same data; pair with `EVAL_SAMPLES` (default 256)
# for a quick held-out-style probe.
_LOADERS = {
    "math": (_load_math, "train", "test"),
    "math500": (_load_math500, "test", "test"),
    "minervamath": (_load_minervamath, "test", "test"),
}


def load_train():
    if TASK not in _LOADERS:
        raise ValueError(f"Unknown TASK={TASK!r}; expected one of {list(_LOADERS)}.")
    loader, train_split, _ = _LOADERS[TASK]
    return loader(train_split)


def load_eval():
    """Eval reward == split accuracy (accuracy_reward returns 0/1).

    Returns `None` when `DO_EVAL` is unset/false so callers can pass the result
    straight through as `eval_dataset=` without branching.
    """
    if not DO_EVAL:
        return None
    if EVAL_TASK not in _LOADERS:
        raise ValueError(f"Unknown EVAL_TASK={EVAL_TASK!r}; expected one of {list(_LOADERS)}.")
    loader, _, eval_split = _LOADERS[EVAL_TASK]
    ds = loader(eval_split)
    if EVAL_SAMPLES > 0:
        ds = ds.select(range(min(EVAL_SAMPLES, len(ds))))
    return ds


def lora_config():
    """Return a `peft.LoraConfig` when `USE_LORA=1`, else `None`.

    Pass the result straight through as `GRPOTrainer(..., peft_config=lora_config())`.
    Defaults mirror the TRL GRPO LoRA example (`r=1`, `lora_alpha=32`,
    `target_modules="all-linear"`); override via `LORA_R`, `LORA_ALPHA`,
    `LORA_TARGET_MODULES` env vars.
    """
    if not USE_LORA:
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
    )


def _init_wandb_with_extras(run_name: str) -> None:
    """Pre-initialize wandb so env-var knobs land in `wandb.config` alongside the
    GRPOConfig fields that HF's WandbCallback auto-logs. HF skips its own
    `wandb.init` if a run already exists, then unions `args.to_dict()` on top.
    No-op if wandb isn't installed; use `WANDB_MODE=disabled` to silence wandb."""
    # Under `accelerate launch`, this script runs once per GPU. Only rank 0 should
    # init wandb; otherwise we'd get one stray run per non-main rank.
    if int(os.environ.get("RANK", "0")) != 0:
        return
    try:
        import wandb
    except ImportError:
        return
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
        name=run_name,
        config={
            "task": TASK,
            "eval_task": EVAL_TASK,
            "model_id": MODEL_ID,
            "do_eval_env": DO_EVAL,
            "eval_samples": EVAL_SAMPLES,
            "eval_steps_env": EVAL_STEPS,
            "max_steps_env": MAX_STEPS,
            "use_vllm_env": USE_VLLM,
            "random_seed_env": RANDOM_SEED,
            "save_final_model_env": SAVE_FINAL_MODEL,
            "grpo_overrides": GRPO_OVERRIDES,
            "use_lora_env": USE_LORA,
            "lora_r": LORA_R if USE_LORA else None,
            "lora_alpha": LORA_ALPHA if USE_LORA else None,
            "lora_target_modules": LORA_TARGET_MODULES if USE_LORA else None,
        },
    )


def base_config(run_name: str, **overrides) -> GRPOConfig:
    """GRPOConfig defaults tuned for a 2-4hr run on a single H100 with Qwen2.5-0.5B-Instruct."""
    run_name = f"{TASK}_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _init_wandb_with_extras(run_name)
    cfg = dict(
        output_dir=f"runs/{run_name}",
        run_name=run_name,
        save_strategy="no",

        # we match the batch size in https://huggingface.co/docs/trl/en/grpo_trainer,
        # which uses `per_device_train_batch_size=8`, num_generations=8, `gradient_accumulation_steps=1`, and 8 GPUs. We have 1 GPU, so we set `gradient_accumulation_steps=8` to keep the same effective batch size.
        # Note that too small a batch size is detrimental. See https://cameronrwolfe.substack.com/p/grpo-tricks ctrl-f "batch size". Batch size may need tuning and affects training reward greatly.
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        num_generations=8,

        # max_completion_length=1024,
        max_steps=MAX_STEPS,

        # 1e-4 + LoRA r=1, alpha=32 picked per "LoRA Without Regret"; default GRPO lr is 1e-6.
        learning_rate=1e-4,

        bf16=True,
        gradient_checkpointing=False,
        use_vllm=USE_VLLM,
        vllm_gpu_memory_utilization=0.3,
        log_completions=True,
        logging_steps=5,
        report_to="wandb",
    )
    if DO_EVAL:
        # Test-set eval. `num_generations_eval=1` => pass@1 sampled accuracy.
        cfg.update(
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            num_generations_eval=1,
            per_device_eval_batch_size=16,
        )
    cfg.update(overrides)
    cfg.update(GRPO_OVERRIDES)
    if RANDOM_SEED:
        cfg["seed"] = random.randint(0, 2**31 - 1)
    return GRPOConfig(**cfg)


__all__ = [
    "MODEL_ID",
    "SAVE_FINAL_MODEL",
    "TASK",
    "WANDB_PROJECT",
    "load_train",
    "load_eval",
    "base_config",
    "accuracy_reward",
    "lora_config",
]
