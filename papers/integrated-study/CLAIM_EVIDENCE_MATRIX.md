# Claim to Evidence Matrix

This matrix is the publication boundary for the integrated technical report. A claim is publishable only when its artifact exists and the wording stays inside its scope.

| Claim ID | Claim | Evidence path | Scope boundary | Publication status |
|---|---|---|---|---|
| C1 | The three-seed, 21-cell campaign completed every configured method and evaluation cell. | `artifacts/metrics/publication-campaign.jsonl` | Successful lifecycle completion, not an algorithmic comparison by itself. | Publishable |
| C2 | SFT, REINFORCE, PPO, RLOO, GRPO, and GRPO k1/k3 labels each scored 4/128 at seeds 13, 42, and 97. | `artifacts/metrics/*-seed*-test.json`, `reports/initial-training-results.md` | One model family, deterministic arithmetic, template-disjoint division test. | Publishable |
| C3 | One completed GRPO cell reached mean rollout reward 0.79167 and KL 0.37643 while scoring 4/128 held out. | Campaign log and `reports/initial-training-results.md` | Internal reward signal does not establish transfer. | Publishable with limitation |
| C4 | k1 and k3 next-token KL estimates were 0.06868447 and 0.06909129 across 32 development prompts. | `artifacts/metrics/kl-k1-k3.json` | Estimator measurement, not a matched loss-formula ablation. | Publishable with limitation |
| C5 | vLLM and SGLang had workload-dependent latency and aggregate output-rate tradeoffs on one L4. | `artifacts/benchmarks/vllm-full.json`, `artifacts/benchmarks/sglang-full.json` | One merged adapter, ten measured requests per cell, non-equivalent output lengths. | Publishable with limitation |
| C6 | The local OpenAI-compatible provider implements routing, authentication, rate limits, streaming, usage, and metrics. | `src/mini_stack/provider/`, provider contract tests | Local contract verification, not a public production deployment. | Publishable |
| C7 | A live remote OpenAI-compatible provider request returned a completion and recorded health and usage. | Runtime-only acceptance record, summarized in the paper and blog | The endpoint and credentials are intentionally not published. | Publishable with limitation |
| C8 | Metric-complete evidence was copied to laptop and OCI with a matching SHA-256 archive hash. | `reports/initial-training-results.md`, local evidence custody record | Archive integrity, not an independently reproduced environment. | Publishable |

## Claims intentionally excluded

- No method is claimed to outperform SFT on template-disjoint arithmetic.
- No engine is claimed to be universally faster.
- No result is claimed to establish general RL algorithm equivalence.
- No provider is claimed to be a public production deployment.
- GRPO k1 and k3 labels are not claimed as a causal estimator ablation because the trainer loss did not differ by label.
