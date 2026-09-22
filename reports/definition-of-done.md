# Definition of Done Evidence Matrix

| Requirement | Status | Evidence |
|---|---|---|
| 0.5B SFT-trained | Pass | `artifacts/model/sft/`, `artifacts/metrics/sft-test.json` |
| GRPO plus classical baseline | Pass | GRPO, REINFORCE, PPO, RLOO adapters and extracted artifacts |
| Deterministic reward | Pass | `src/mini_stack/rewards/exact_match.py`, frozen manifest |
| KL-regularized algorithm | Pass with limitation | PPO and GRPO KL logs plus k1/k3 estimator measurement; matched training ablation remains pending |
| Reproducible metrics | Partial | executed JSONL ledger exists, but the original runs lack several required metrics |
| Model served through vLLM/SGLang | Pass | engine smoke and full benchmark JSONs |
| Custom OpenAI-compatible provider | Pass locally | provider contract suite and live remote-backend acceptance |
| Multiple registered models/routing | Pass locally | registry and routing tests |
| Streaming | Pass locally | provider streaming contract test |
| Usage and latency | Pass | provider usage tests and engine benchmark JSONs |
| Entire stack on GCP | Partial | training and engine workloads ran on GCP; integrated end-to-end provider campaign remains pending |
| Two-copy artifact verification | Pass | laptop and OCI evidence archive hashes match |

Publication must accurately retain every `Partial` status. No paper or blog may describe the full Definition of Done as complete.
