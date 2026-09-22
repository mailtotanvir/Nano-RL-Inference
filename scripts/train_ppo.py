#!/usr/bin/env python3
"""Small clipped-ratio PPO-style baseline from SFT with exact-match reward and frozen reference."""

from __future__ import annotations

import copy
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
BETA = 0.05


def completion_log_prob(
    model: PeftModel, generated: torch.Tensor, prompt_length: int
) -> torch.Tensor:
    logits = model(generated).logits[:, :-1]
    targets = generated[:, 1:]
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs[:, prompt_length - 1 :].mean()


def main() -> None:
    run = settings("ppo", "artifacts/model/ppo")
    sft_adapter = Path(os.environ["MINI_STACK_SFT_ADAPTER"])
    torch.manual_seed(run.seed)
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    policy = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True).cuda()
    reference = copy.deepcopy(policy).eval()
    reference.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    for step in range(STEPS):
        row = rows[step % len(rows)]
        messages = [{"role": "user", "content": f"{row['prompt']} Reply with only the answer."}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).cuda()
        generated = policy.generate(prompt_ids, max_new_tokens=16, do_sample=True, temperature=0.8)
        completion = tokenizer.decode(generated[0][prompt_ids.shape[1] :], skip_special_tokens=True)
        reward = exact_match_reward(completion, row["answer"])
        if reward == 0:
            continue
        with torch.no_grad():
            old_log_prob = completion_log_prob(policy, generated, prompt_ids.shape[1])
            reference_log_prob = completion_log_prob(reference, generated, prompt_ids.shape[1])
        new_log_prob = completion_log_prob(policy, generated, prompt_ids.shape[1])
        ratio = torch.exp(new_log_prob - old_log_prob)
        unclipped = ratio * reward
        clipped = torch.clamp(ratio, 0.8, 1.2) * reward
        surrogate = torch.minimum(unclipped, clipped)
        kl = new_log_prob - reference_log_prob
        loss = -surrogate + BETA * kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 8 == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "reward": reward,
                        "loss": float(loss.detach()),
                        "kl": float(kl.detach()),
                    }
                )
            )
    run.output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(run.output_dir)
    tokenizer.save_pretrained(run.output_dir)


if __name__ == "__main__":
    main()
