# Method and evaluation protocol

## Temporal split

For each user sequence `[i1, ..., in]`, use the first `n - 2` interactions for training, the penultimate interaction as the validation target and the last interaction as the test target. Validation sees the training prefix. Test history also includes the observed validation interaction.

Training examples predict the next item within the training prefix. Validation targets are excluded from gradient updates. Test targets are excluded from fitting and checkpoint selection. SASRec encodes the latest 50 items with right padding and reads the final non-padding position. Preference memory retains the full observed history.

Each user contributes one validation and one test event. SASRec scores the catalog from the latest 50 observed items. Exclude padding and every item in the full prefix before selecting the 20 highest raw scores. Break ties by ascending item ID. At validation time the prefix ends before the validation event. At test time it includes that event. Only past interactions determine eligibility. If a user repeats a previously observed item, that target counts as a retrieval miss.

Freeze the selected item IDs and raw scores together. Shuffle their positions using the run seed (20261101) and the event ID. Every method receives the same displayed order. The agent reviews eligible options in that order and uses base scores to identify each option's immediate competitor. Validation NDCG for SASRec uses its score ranking. Each prediction requires 20 eligible items.

## SASRec and checkpoint selection

| Parameter | Setting |
| --- | --- |
| Item and position embedding dimension | 64 |
| Causal self-attention blocks | 2 |
| Heads per block | 4 |
| Dropout | 0.1 |
| Maximum encoder length | 50 |
| Loss | Full-catalog softmax cross-entropy, excluding padding |
| Optimizer | AdamW |
| Learning rate / weight decay | 0.0015 / 0.00001 |
| Batch size / gradient norm clip | 512 / 1.0 |
| Minimum training length | 20 epochs |
| Validation interval | 5 epochs |
| Selection metric | Validation NDCG@10 using the current model's top 20 outside the observed prefix |
| Early stopping | 5 consecutive evaluations without a strict improvement |

The minimum training length determines when training can stop. The best checkpoint may come from any validation epoch, including one before epoch 20. Restore that checkpoint and freeze it for testing.

## History and preference evidence

The baseline combines SASRec scores with category, title-token and rare-attribute matches over the full observed history. Each component is standardized within the candidate set. Scoring weights and uncertainty gating follow the earlier protocol.

The auxiliary Qwen prompt summarizes the titles and categories of the latest 25 observed items. It accepts up to 2,048 input tokens and generates up to 96 tokens with greedy decoding. The parser can fall back to tokens from the observed history. The summary supplies one feature for baseline scoring. Cumulative matching and preference memory continue to use the full pre-prediction history.

Preference memory uses observed items, timestamps and ratings. Category and attribute boundaries are fitted on training prefixes. The agent ranks the candidates before the test interaction enters memory. It has no access to the target's identity or rating. Title tokens serve as proxies for attributes.

The full method enables adaptive granularity, local revision and recovery. Each distinct candidate and hypothesis pair can supply one approval. Repeating the same pair adds no evidence. The agent reviews up to eight candidates and six hypotheses per batch and continues with any remaining candidates or hypotheses. The approval cites the candidate directly above it in the baseline ordering. The executor compares it with the neighbor directly above its current position.

The minimum confidence is medium. Each qualifying claim contributes a bonus of `0.5 * effective claim weight`. Process candidates by baseline score plus their highest qualifying bonus. Within each candidate, try claims in order of increasing bonus. For each claim, compare the candidate's baseline score plus that claim's bonus with its current neighbor's adjusted score. Break ties by ascending item ID. When the comparison succeeds, swap the two items once. Continue from the new rank with the next claim. The first position is the natural boundary. A displaced candidate can also move down repeatedly. The execution trace records the hypothesis ID, both items and the ranks for each swap. `max_displacement` reports the largest observed net displacement.

## Inference and metrics

Qwen uses `Qwen3-4B-Instruct-2507` with greedy vLLM inference, temperature 0 and seed 20261101. Plans and decisions must conform to their JSON schemas. The input limit is 12,000 tokens. Planning attempts allow 256, 512 and 1,024 output tokens. Decision attempts allow 2,048, 4,096 and 8,192. Retry only a truncated or invalid response. Maximum model length is 20,192 tokens. Run logs record budgets, responses and retries.

Report HR and NDCG at cutoffs 1, 3, 5 and 10 on a 0 to 1 scale. The denominator includes every event. If the target is absent from the retrieved 20, its saved rank is JSON `null` and its HR and NDCG contributions are zero at every cutoff. If it is retrieved, use its one-based rank after reranking. A full Beauty test requires one event for each of the 22,363 users.

The history prompts, scoring weights and several mechanisms were developed in earlier experiments. The release carries that development history into this protocol. Account for it when describing the evaluation and its limitations in a paper.
