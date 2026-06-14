"""Ablation: LENS applied only to positive rewards.

`lens_apply_to_negatives=False` disables the standard LENS penalty on r=0
rollouts; `lens_apply_to_positives=True` keeps the r=1 confidence discount.
Isolates the contribution of the all-positive-group rescue from the
all-negative-group rescue.
"""

import os

from trl import GRPOTrainer

from _common import (
    MODEL_ID,
    SAVE_FINAL_MODEL,
    WANDB_PROJECT,
    accuracy_reward,
    base_config,
    load_eval,
    load_train,
    lora_config,
)


def main():
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    trainer = GRPOTrainer(
        model=MODEL_ID,
        args=base_config(
            "lens_positives_only",
            use_lens=True,
            lens_apply_to_positives=True,
            lens_apply_to_negatives=False,
        ),
        train_dataset=load_train(),
        eval_dataset=load_eval(),
        reward_funcs=accuracy_reward,
        peft_config=lora_config(),
    )
    trainer.train()
    if SAVE_FINAL_MODEL:
        trainer.save_model()


if __name__ == "__main__":
    main()
