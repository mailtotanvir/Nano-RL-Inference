# Publication Readiness Decision

## Publishable now

The repository, integrated technical report, local blog draft, benchmark artifacts, provider acceptance evidence, and current LoRA adapters can be prepared for local review now.

The paper must be labeled an engineering note or constrained replication. It must not claim a robust multi-seed algorithm comparison because the 21-cell campaign recorded successful execution and wall-clock time but did not evaluate every seed-specific adapter or emit complete per-run metric payloads.

## Required campaign before stronger model claims

One additional continuous GPU campaign is sufficient. It must evaluate every seed-specific saved adapter on train, development, and held-out splits immediately after its run, then append required JSONL metrics. No new algorithm family is required.

The expected active GPU time is approximately 3 to 6 hours: about 2 hours for the 21 observed training cells plus evaluation, model reloads, provider evidence, checkpoints, and artifact verification. Reserve the previously approved 12-hour VM maximum and 15-hour watchdog to avoid an incomplete campaign.

## Hugging Face decision

A Hugging Face release is optional for the paper and Zenodo record, but appropriate if the project is presented as an inference artifact. Do not upload a full merged base-model copy unless licensing and provenance are reviewed. The current safe artifact is the selected LoRA adapter plus tokenizer and model card.

Recommended model repository: `mailtotanvir/nano-rl-inference-arithmetic-sft-adapter`.

The selected release candidate is SFT, not GRPO, because both reached 5/128 held-out accuracy while SFT is simpler and faster. The card must state that it is a research adapter for deterministic arithmetic and does not improve template-disjoint accuracy over the tested alternatives.

## Required model-card contents

- base model and adapter format
- exact train/test family split
- all evaluated method results, not only the selected adapter
- intended use and explicit non-use
- L4 and cost disclosure
- paper, repository, blog, and DOI placeholders
- artifact hashes

## Staged publication order

1. Complete the metric-complete re-test campaign.
2. Revise paper and blog from the updated evidence.
3. Build local review bundle: README, PDF, blog preview, claim matrix, model card, sanitize report.
4. Get user approval.
5. Push sanitized GitHub repository and enable Pages.
6. Upload selected adapter to Hugging Face after separate approval.
7. Create the GitHub release and one Zenodo concept DOI after separate approval.
