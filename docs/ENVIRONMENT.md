# Environment

## Reference environment

The project was tested on Linux x86_64 with Python 3.11.16, an NVIDIA A100 80 GB GPU, PyTorch 2.6.0 with CUDA 12.4, vLLM 0.8.5, Transformers 4.51.3 and NumPy 2.2.6. [Recorded package versions](reference_environment.json)

The launcher uses Linux process groups and `fcntl` file locks. Run the complete pipeline on Linux with a vLLM-compatible NVIDIA GPU. The release checks used an existing server installation. CPU execution covers unit tests and small SASRec checks.

## Install dependencies

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The requirements pin the tested versions. Use fresh run caches after changing vLLM, PyTorch or Transformers.

## Download Qwen

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir ../models/Qwen3-4B-Instruct-2507
```

The model directory must contain `config.json`, the tokenizer files, the weight shards and `model.safetensors.index.json`. The launcher loads the model offline from the sibling model directory. [Qwen model repository](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

## Paths and settings

Relative directory settings are resolved from the ZoomRec project directory. Use `../` to reference a sibling directory. The launcher and its workers use the same base even when their working directories differ.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `ZOOMREC_DATA_ROOT` | `../zoomrec_data/` | Data root containing `data/raw/` and `freeze/` |
| `ZOOMREC_RUN_DIR` | Temporary RAM directory for the launcher | Splits, summaries, checkpoints, logs and predictions |
| `ZOOMREC_QWEN_MODEL` | `../models/Qwen3-4B-Instruct-2507/` | Local Qwen weights and tokenizer |
| `ZOOMREC_QWEN_PORT` | `24215` | Port for the inference server started by the launcher |
| `ZOOMREC_CHECKPOINT` | `fresh` | Train SASRec and select the best validation checkpoint |
| `ZOOMREC_QWEN_URL` | Local server on the configured port, endpoint `/v1/completions` | Service address for manually invoked inference scripts |

For example, with the model and data in directories next to ZoomRec:

```bash
export ZOOMREC_QWEN_MODEL="../models/Qwen3-4B-Instruct-2507"
export ZOOMREC_DATA_ROOT="../zoomrec_data"
export ZOOMREC_RUN_DIR="../zoomrec_runs/experiment_01"
python prepare_data.py --download
python launch.py --gpu 0 --checkpoint fresh --pilot
```

Export the variables in the shell. The scripts read the process environment. The launcher sets the Qwen URL for its subprocesses and stops its inference server on exit.

The default `--gpu-memory-utilization` is 0.48. If vLLM reports insufficient memory or KV cache capacity, inspect `inference_server.log` and `nvidia-smi`. Select an available GPU or adjust the memory fraction for the run.
