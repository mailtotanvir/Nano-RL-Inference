#!/usr/bin/env python3
"""Run GRPO from the SFT adapter using deterministic exact-match arithmetic reward."""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from mini_stack.campaign_settings import settings
from mini_stack.rewards.exact_match import exact_match_reward

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = Path("data/processed/arithmetic_v1/train.jsonl")


def reward(completions: list[str], answer: list[str], **_: object) -> list[float]:
    return [
        exact_match_reward(completion, expected)
        for completion, expected in zip(completions, answer, strict=True)
    ]


def main() -> None:
    run = settings("grpo", "artifacts/model/grpo")
    sft_adapter = Path(os.environ["MINI_STACK_SFT_ADAPTER"])
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    records = [
        {"prompt": f"{row['prompt']} Reply with only the answer.", "answer": row["answer"]}
        for row in rows
    ]
    dataset = Dataset.from_list(records)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto")
    model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=str(run.output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_generations=4,
            max_completion_length=16,
            learning_rate=1e-5,
            seed=run.seed,
            logging_steps=5,
            save_strategy="epoch",
            report_to="none",
            bf16=True,
        ),
    )
    trainer.train()
    trainer.save_model(str(run.output_dir))


if __name__ == "__main__":
    main()
