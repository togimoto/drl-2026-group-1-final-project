#!/bin/bash

export MASTER_PORT=$(shuf -i 49152-65535 -n 1)  # pick random port; fix for torch.distributed.DistNetworkError EADDRINUSE when multiple experiments are running on the same node

source .venv/bin/activate
export MODEL_ID='Qwen/Qwen2.5-Math-1.5B'
export GRPO_OVERRIDES='{"max_completion_length": 1024, "num_train_epochs": 2, "temperature": 0.7, "top_p": 0.7, "num_generations": 8, "per_device_train_batch_size": 16, "gradient_accumulation_steps": 4, "gradient_checkpointing": true, "vllm_gpu_memory_utilization": 0.3, "vllm_enable_sleep_mode": true, "learning_rate": 1e-5, "logging_steps": 1, "log_completions": false, "eval_on_start": true}'
export MAX_STEPS='300'
export USE_LORA='1'
export USE_VLLM='1'
export TASK='math'
export MATH_LEVELS='3,4,5'
export DO_EVAL='1'
export EVAL_STEPS='30'
export EVAL_SAMPLES='0'
export EVAL_TASK='math500'
export SAVE_FINAL_MODEL='1'
python experiments/train_lens_positive.py
