#!/usr/bin/env python3
"""Pinned-TRL 0.15 RLOO baseline with a deterministic arithmetic reward callable."""

from __future__ import annotations

import copy
import json
import os
import re
from fractions import Fraction
from pathlib import Path

from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from trl import RLOOConfig, RLOOTrainer

from mini_stack.campaign_settings import settings
from mini_stack.rewards.exact_match import exact_match_reward

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = Path("data/processed/arithmetic_v1/train.jsonl")
QUESTION = re.compile(r"What is (-?\d+) ([+\-*/]) (-?\d+)\?")


def arithmetic_reward(texts: list[str]) -> list[float]:
    rewards: list[float] = []
    for text in texts:
        match = QUESTION.search(text)
        if match is None:
            rewards.append(0.0)
            continue
        left, operator, right = match.groups()
        operands = Fraction(left), Fraction(right)
        expected = {
            "+": operands[0] + operands[1],
            "-": operands[0] - operands[1],
            "*": operands[0] * operands[1],
            "/": operands[0] / operands[1],
        }[operator]
        rewards.append(exact_match_reward(text, str(expected)))
    return rewards


def main() -> None:
    run = settings("rloo", "artifacts/model/rloo")
    sft_adapter = Path(os.environ["MINI_STACK_SFT_ADAPTER"])
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{row['prompt']} Reply with only the answer."}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    dataset = Dataset.from_dict({"input_ids": inputs})
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto")
    policy = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)
    reference = copy.deepcopy(policy)
    reference.requires_grad_(False)
    config = RLOOConfig(
        output_dir=str(run.output_dir),
        total_episodes=128,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_mini_batches=1,
        rloo_k=4,
        response_length=16,
        num_ppo_epochs=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        bf16=True,
        seed=run.seed,
        kl_coef=0.05,
    )
    trainer = RLOOTrainer(
        config=config,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=reference,
        reward_model=arithmetic_reward,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(run.output_dir))


if __name__ == "__main__":
    main()
