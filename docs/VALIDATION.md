# Validation

Run the following checks from the project directory in a Linux environment with the project dependencies installed.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -B -m unittest discover -v
python -B test_full_protocol.py
python -B test_ledger_gate.py
```

Candidate tests check exclusion of the full observed prefix and padding, ties at the twentieth position, reproducible shuffling, item-score alignment and zero credit for missing targets. The ranking tests check strict comparisons in both item-ID orders, repeated promotions, duplicate claims, confidence filtering and the independent audit. The ablation tests distinguish destination reopening from claim-state recovery and check rule-only promotions with zero agent calls.

With the dataset and Qwen weights prepared, run one shard to check GPU execution. The launcher trains SASRec, runs inference, audits the shard and removes temporary run files on exit.

```bash
python -B launch.py --gpu 0 --checkpoint fresh --workers 1 --pilot --no-save
```

To retain a full run for inspection, configure a persistent directory. [Running and resuming](RUNNING.md)

The release manifest lists the distributed files and their SHA-256 digests. [Release manifest](release_manifest.json)
