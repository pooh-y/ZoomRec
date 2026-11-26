# ZoomRec

Run the following commands from the project directory on a Linux machine with a compatible NVIDIA GPU.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python prepare_data.py --download
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir ../models/Qwen3-4B-Instruct-2507

# Train SASRec and run one evaluation shard. Discard the outputs on exit.
python launch.py --gpu 0 --workers 1 --checkpoint fresh --pilot --no-save

# To keep outputs and resume, use the same run directory for both commands.
export ZOOMREC_RUN_DIR="../zoomrec_runs/loo_unseen_top20"
python launch.py --gpu 0 --workers 1 --checkpoint fresh --pilot
python launch.py --gpu 0 --workers 4 --checkpoint fresh
```

## Evaluation protocol

- Sort each user's interactions chronologically. Hold out the last for testing and the penultimate for validation. Use the earlier interactions for training.
- SASRec encodes the latest 50 items. Preference matching and evidence retrieval use the full observed history. At test time this includes the validation interaction, with model parameters frozen.
- For each validation or test event, exclude every item in the user's observed prefix and retrieve the 20 highest-scoring remaining items. Break SASRec score ties by ascending item ID. Shuffle the item-score pairs once per event and share that order across methods. A target outside the retrieved set contributes zero at every cutoff and stays in the denominator.
- SASRec uses two causal self-attention blocks, four heads per block, a hidden dimension of 64 and dropout of 0.1.
- Train with full-catalog softmax cross-entropy and AdamW. Use a learning rate of `1.5e-3`, weight decay of `1e-5`, batch size 512 and gradient norm clipping at 1.0.
- Evaluate validation NDCG@10 every five epochs. After at least 20 epochs, stop when five consecutive evaluations have failed to improve the best score. Restore the best checkpoint.

The protocol describes the data boundaries, the role of history summaries and the interpretation of metrics. [Method and evaluation protocol](docs/PROTOCOL.md)

To evaluate ZoomRec and all four Section 5.3 ablations together, set
`ZOOMREC_ABLATION_REQUEST=configs/paper_ablations.json` before running the launcher. [Running the ablations](docs/RUNNING.md)

## Project files

```text
ZoomRec/
├── README.md
├── requirements.txt
├── .gitignore / .gitattributes
├── configs/data_sources.json       # Data sources and reference hashes
├── docs/                           # Setup, data, execution and validation
├── prepare_data.py                 # Rebuild the catalog and user sequences
├── data_split.py                   # Leave-one-out validation and test queries
├── sasrec_candidates.py / ranking_metrics.py  # Unseen-item retrieval and end-to-end metrics
├── baseline_source.py              # SASRec training and history-based scoring
├── training_protocol.py            # Checkpoint selection and early stopping
├── prepare_loo.py                  # Prepare splits, checkpoints and baseline ranks
├── history_prompts.py              # History summary prompts and parser
├── history_qwen.py / history_states.py
├── ledger.py / preference_graph.py # Preference evidence over the full history
├── recommendation_agent.py         # Retrieve evidence, verify claims and rerank
├── qwen_vllm.py / generation_protocol.py
├── full_evaluation.py / launch.py  # Sharded inference, resumption and aggregation
├── audit_inputs.py / audit_ablation_outputs.py
├── run_paths.py / run_ledger_gate.py / status.py / write_report.py
├── ablation_request.json / selection.json
└── test_*.py                       # Protocol, training, evidence and cache tests
```

[Environment](docs/ENVIRONMENT.md) · [Data preparation](docs/DATA.md) · [Running and resuming](docs/RUNNING.md) · [Method and evaluation](docs/PROTOCOL.md) · [Publishing to GitHub](docs/GITHUB.md) · [Validation](docs/VALIDATION.md)
