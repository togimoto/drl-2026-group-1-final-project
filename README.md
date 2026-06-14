# DRL 2026 Group 1 Final Project

CAPS: Calibrated Advantage with Positive Selection for Reliable GRPO Training

## Setup

### Environment

Run the following commands to set up the environment:

```bash
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -e trl_package/ torch==2.8.0 math_verify wandb python-dotenv peft vllm "transformers>=4.56.0,<4.58.0" --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
```

### WandB

This project requires a Weights and Biases (WandB) key. Copy `.env.example` to `.env` and add your WandB key to the `WANDB_API_KEY` variable.

### TRL

We rely on a local modified version of the TRL library located at `trl_package/`. TRL is licensed under the Apache-2.0 License. See `trl_package/LICENSE` for the license.

## Experiments

To run each experiment and each ablation, run the following commands:

### GRPO

```bash
bash scripts/grpo.sh
```

### NegCal

```bash
bash scripts/negcal.sh
```

### PosCal

```bash
bash scripts/poscal.sh
```

### PosFilter

```bash
bash scripts/posfilter.sh
```

### NegCal + PosCal

```bash
bash scripts/negcal_poscal.sh
```

### NegCal + PosFilter

```bash
bash scripts/negcal_posfilter.sh
```

### PosCal + PosFilter

```bash
bash scripts/poscal_posfilter.sh
```

### CAPS (Ours)

```bash
bash scripts/caps.sh
```

### Minerva Evaluation

```bash
python experiments/eval_lora.py minervamath --lora_path <path to trained lora adapter> 
```