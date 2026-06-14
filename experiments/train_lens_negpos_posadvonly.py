"""Negative LENS + positive LENS, paired with positive_advantage_only.

Both sides of LENS are active under the always-on positive mask
(`lens_apply_to_negatives_in_warmup=True` keeps the negative-LENS term from
being gated off by the mask): all-negative groups are rescued by the
confidence-calibrated negative penalty and all-positive groups by the positive
discount, while mixed groups follow standard LENS.
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
            "lens_negpos_posadvonly",
            use_lens=True,
            lens_apply_to_positives=True,
            positive_advantage_only=True,
            lens_apply_to_negatives_in_warmup=True,
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
