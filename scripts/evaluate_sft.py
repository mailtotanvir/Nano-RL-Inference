#!/usr/bin/env python3
"""Evaluate the SFT adapter with deterministic exact-match arithmetic scoring."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_stack.rewards.exact_match import exact_match_reward

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = Path(os.environ.get("ADAPTER_PATH", "artifacts/model/sft"))
DATA = Path(os.environ.get("EVAL_DATA", "data/processed/arithmetic_v1/test.jsonl"))
OUT = Path(os.environ.get("EVAL_OUT", "artifacts/metrics/sft-test.json"))


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="cuda")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()]
    correct = 0
    results = []
    for row in rows:
        messages = [{"role": "user", "content": f"{row['prompt']} Reply with only the answer."}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(inputs, max_new_tokens=16, do_sample=False)
        completion = tokenizer.decode(output[0][inputs.shape[1] :], skip_special_tokens=True)
        reward = exact_match_reward(completion, row["answer"])
        correct += int(reward)
        results.append(
            {
                "prompt": row["prompt"],
                "answer": row["answer"],
                "completion": completion,
                "reward": reward,
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(ADAPTER),
        "split": "test",
        "examples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("examples", "correct", "accuracy")}))


if __name__ == "__main__":
    main()
