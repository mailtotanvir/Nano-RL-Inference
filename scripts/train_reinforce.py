#!/usr/bin/env python3
"""Minimal LoRA REINFORCE baseline with deterministic exact-match rewards."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_stack.campaign_settings import settings
from mini_stack.rewards.exact_match import exact_match_reward

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = Path("data/processed/arithmetic_v1/train.jsonl")
STEPS = int(os.environ.get("MINI_STACK_STEPS", "64"))


def main() -> None:
    run = settings("reinforce", "artifacts/model/reinforce")
    sft_adapter = Path(os.environ["MINI_STACK_SFT_ADAPTER"])
    torch.manual_seed(run.seed)
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    for step in range(STEPS):
        row = rows[step % len(rows)]
        messages = [{"role": "user", "content": f"{row['prompt']} Reply with only the answer."}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).cuda()
        generated = model.generate(prompt_ids, max_new_tokens=16, do_sample=True, temperature=0.8)
        completion_ids = generated[:, prompt_ids.shape[1] :]
        completion = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
        reward = exact_match_reward(completion, row["answer"])
        if reward == 0:
            continue
        logits = model(generated).logits[:, :-1]
        targets = generated[:, 1:]
        token_log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = token_log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        completion_log_prob = token_log_probs[:, prompt_ids.shape[1] - 1 :].mean()
        loss = -reward * completion_log_prob
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 8 == 0:
            print(json.dumps({"step": step, "reward": reward, "loss": float(loss.detach())}))
    run.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(run.output_dir)
    tokenizer.save_pretrained(run.output_dir)


if __name__ == "__main__":
    main()
