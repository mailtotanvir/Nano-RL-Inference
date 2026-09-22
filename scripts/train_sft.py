#!/usr/bin/env python3
"""Run a reproducible LoRA SFT baseline on frozen arithmetic data."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from mini_stack.campaign_settings import settings

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = Path("data/processed/arithmetic_v1/train.jsonl")


def main() -> None:
    run = settings("sft", "artifacts/model/sft")
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    records = []
    for row in rows:
        text = (
            f"<|im_start|>user\n{row['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n{row['answer']}<|im_end|>"
        )
        records.append({"text": text})
    dataset = Dataset.from_list(records)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        ),
        args=TrainingArguments(
            output_dir=str(run.output_dir), num_train_epochs=1, per_device_train_batch_size=4,
            gradient_accumulation_steps=4, learning_rate=2e-4, logging_steps=10,
            seed=run.seed,
            save_strategy="epoch", report_to="none", bf16=True,
        ),
    )
    trainer.train()
    trainer.save_model(str(run.output_dir))
    tokenizer.save_pretrained(run.output_dir)


if __name__ == "__main__":
    main()
