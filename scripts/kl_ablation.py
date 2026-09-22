#!/usr/bin/env python3
"""Compute k1 and k3 KL estimates for SFT policy against frozen base reference."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_stack.rl.kl import kl_k1, kl_k3

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = Path("artifacts/model/sft")
DATA = Path("data/processed/arithmetic_v1/dev.jsonl")
OUT = Path("artifacts/metrics/kl-k1-k3.json")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER)
    reference = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto").cuda().eval()
    policy_base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto")
    policy = PeftModel.from_pretrained(policy_base, ADAPTER).cuda().eval()
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines()[:32]]
    estimates = []
    with torch.inference_mode():
        for row in rows:
            ids = tokenizer(row["prompt"], return_tensors="pt").input_ids.cuda()
            policy_logits = policy(ids).logits[0, -1]
            reference_logits = reference(ids).logits[0, -1]
            policy_log_probs = torch.log_softmax(policy_logits, dim=-1).cpu().tolist()
            reference_log_probs = torch.log_softmax(reference_logits, dim=-1).cpu().tolist()
            estimates.append(
                {
                    "k1": kl_k1(policy_log_probs, reference_log_probs),
                    "k3": kl_k3(policy_log_probs, reference_log_probs),
                }
            )
    k1_mean = sum(item["k1"] for item in estimates) / len(estimates)
    k3_mean = sum(item["k3"] for item in estimates) / len(estimates)
    payload = {
        "examples": len(estimates),
        "k1_mean": k1_mean,
        "k3_mean": k3_mean,
        "per_example": estimates,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("examples", "k1_mean", "k3_mean")}))


if __name__ == "__main__":
    main()
