"""LENS applied to 1-reward generations.

r=0 rollouts are masked out by `positive_advantage_only`. r=1 rollouts get a
confidence-calibrated discount: r̃ = 1 - (1/G) * π̄_old / (D - π̄_old). This
rescues all-positive groups (zero-gradient under vanilla GRPO) by pushing the
policy toward less-confident correct rollouts.
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
            "lens_positive",
            use_lens=True,
            lens_apply_to_positives=True,
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
