# Publication-Evidence Campaign Results

## Experimental envelope

- Base: `Qwen/Qwen2.5-0.5B-Instruct`
- Training split: 512 arithmetic examples from addition and subtraction templates
- Held-out split: 128 division-template examples
- Scoring: exact numeric match after conservative normalization
- Seeds: 13, 42, 97
- Hardware: on-demand GCP `g2-standard-8`, one NVIDIA L4
- Campaign: 21 completed train cells, each followed by held-out evaluation

## Three-seed held-out results

| Method | Seed 13 | Seed 42 | Seed 97 | Mean accuracy |
|---|---:|---:|---:|---:|
| SFT | 4/128 | 4/128 | 4/128 | 0.03125 |
| REINFORCE | 4/128 | 4/128 | 4/128 | 0.03125 |
| PPO | 4/128 | 4/128 | 4/128 | 0.03125 |
| RLOO | 4/128 | 4/128 | 4/128 | 0.03125 |
| GRPO | 4/128 | 4/128 | 4/128 | 0.03125 |
| GRPO-k1 | 4/128 | 4/128 | 4/128 | 0.03125 |
| GRPO-k3 | 4/128 | 4/128 | 4/128 | 0.03125 |

The repeated result is more useful than a single favorable point estimate: within this constrained setup, no tested update rule changed the held-out outcome.

## Execution timing

| Method | Mean wall-clock time |
|---|---:|
| SFT | 35.06 s |
| REINFORCE | 25.46 s |
| PPO | 30.16 s |
| RLOO | 55.99 s |
| GRPO | 436.42 s |
| GRPO-k1 | 432.64 s |
| GRPO-k3 | 438.42 s |

## Estimator measurement

The earlier distribution-level estimator run over 32 development prompts measured k1 mean 0.06868447 and k3 mean 0.06909129. The repeated GRPO-k1 and GRPO-k3 campaign cells were operationally successful, but the current trainer does not yet alter its loss formula by the estimator label. They therefore provide matched execution timing, not a causal estimator comparison.

## Honest conclusion

The publication claim is not that GRPO, PPO, RLOO, or REINFORCE improved arithmetic reasoning. The claim is that a constrained, reproducible system made a stable failure visible: reward-bearing rollouts and optimizer-specific behavior did not transfer across the held-out template family.

## Evidence

- `artifacts/metrics/*-seed*-test.json`
- `artifacts/metrics/publication-campaign.jsonl`
- `artifacts/metrics/kl-k1-k3.json`
- `artifacts/benchmarks/vllm-full.json`
- `artifacts/benchmarks/sglang-full.json`
- `reports/definition-of-done.md`
