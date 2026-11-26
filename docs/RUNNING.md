# Running, resuming and inspecting outputs

Run commands from the project directory after setting up the environment and preparing the data. [Environment](ENVIRONMENT.md) · [Data preparation](DATA.md)

## Run one shard, then resume

```bash
export ZOOMREC_RUN_DIR="../zoomrec_runs/loo_unseen_top20"
python launch.py --gpu 0 --workers 1 --checkpoint fresh --pilot
python launch.py --gpu 0 --workers 4 --checkpoint fresh
```

When `ZOOMREC_RUN_DIR` is unset, the launcher cleans up its temporary RAM directory on exit. Pass `--no-save` to request this behavior explicitly. To resume after a pilot, set the run directory before the first invocation.

The launcher prepares the leave-one-out splits, trains SASRec, starts a local Qwen server, generates history summaries and scores the baseline. Workers retrieve evidence and rank candidates by shard. A full run aggregates and audits the results. The launcher uses the active Python environment.

The default shard size is 256 users, which gives 88 shards for the full dataset. Each `--pilot` invocation processes at most one unfinished shard. Full results require all 22,363 test events.

## Run the Section 5.3 ablations

The default request selects the full method. The paper configuration evaluates it alongside all four ablations on the same candidates, frozen checkpoint and baseline scores.

```bash
export ZOOMREC_ABLATION_REQUEST=configs/paper_ablations.json
export ZOOMREC_RUN_DIR=../zoomrec_runs/paper_ablations
python launch.py --gpu 0 --workers 4 --checkpoint fresh
```

Use the same two environment settings for a resumed run or a separate audit. The results table and each shard identify the methods by their configuration names. [Ablation definitions](PROTOCOL.md)

To run in the background, keep the same run directory setting:

```bash
python launch.py --gpu 0 --workers 4 --checkpoint fresh --detach
python status.py
tail -f ../zoomrec_runs/loo_unseen_top20/supervisor.log
```

In the foreground, Ctrl+C stops the launcher and it releases the processes it started. Running the same configuration again in the same directory skips completed shards. Inference cache entries are checked against the prompt and settings. Recovery backs up an incomplete final line. A corrupt complete line raises an error.

## Train from scratch

```bash
export ZOOMREC_RUN_DIR="../zoomrec_runs/from_scratch"
python launch.py --gpu 0 --checkpoint fresh --pilot
python launch.py --gpu 0 --workers 4 --checkpoint fresh
```

To train and prepare the candidate pools without starting Qwen:

```bash
export ZOOMREC_RUN_DIR="../zoomrec_runs/train_only"
export ZOOMREC_CHECKPOINT=fresh
python prepare_loo.py --device cuda:0 --checkpoint fresh
```

Validate every five epochs. Training lasts at least 20 epochs and has no fixed maximum. A strictly better validation score replaces the best checkpoint. Evaluation restores the best weights across all validations. An interrupted training run restarts from scratch because only the best model parameters are saved.

## Main outputs

```text
../zoomrec_runs/loo_unseen_top20/
├── model/
│   ├── sasrec_corrected_seed_20261101.pt
│   ├── train_config.json
│   └── loo_training.json
├── inputs/
│   ├── freeze/{train,valid,test}.jsonl
│   ├── events.jsonl
│   ├── states.jsonl / states_manifest.json
│   ├── history_calls.jsonl
│   └── shards/0000/cases.jsonl ...
├── boundary_model.json / selection.json
├── manifest.json
├── shards/0000/full/
│   ├── predictions.jsonl / metrics.json
│   ├── actions.jsonl / ranking_executions.jsonl
│   └── plan_calls.jsonl / decision_calls.jsonl
├── results.json / metrics.csv       # After full aggregation
├── FULL_METHOD_REPORT.md            # Full-method report
└── verification.json                # After the independent full audit
```

A shard with no Qwen requests for a stage has no call log for that stage. The run directory holds the checkpoints and logs. New manifests record file paths relative to the ZoomRec project directory. Sibling paths start with `../`.

## Tests and audits

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m unittest discover -v
python test_full_protocol.py
python test_ledger_gate.py

# After input preparation
python audit_inputs.py
# After a pilot
python audit_ablation_outputs.py --allow-partial
# After a complete run
python audit_ablation_outputs.py
```

`python full_evaluation.py aggregate` requires every shard to be complete. After changing code, splits, checkpoint mode or settings, select a new `ZOOMREC_RUN_DIR`. Manifest checks prevent reuse of incompatible scores.

## Reuse existing history summaries

A new run generates summaries with Qwen. If you already have summaries and their matching event file, you can import them after model preparation. This example uses a sibling directory named `previous_run`.

```bash
python history_states.py \
  --import-cache ../previous_run/inputs/states.jsonl \
  --import-events ../previous_run/inputs/events.jsonl
```

The importer checks the user, event index, full history, target, candidates and original parser output. It records the source file hashes. The import covers history summaries only. Baseline scoring, agent decisions and checkpoint preparation follow the current protocol.
