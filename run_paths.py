"""Portable defaults. All writable experiment files stay under RUN_DIR."""
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent


def configured_path(variable, default):
    """Interpret relative settings from the project, including in worker subprocesses."""
    value = Path(os.environ.get(variable, default)).expanduser()
    # Preserve configured symlink names in shared manifests.
    return Path(os.path.abspath(CODE_DIR / value))


def project_reference(path):
    """Serialize a file reference relative to the project, including external siblings."""
    return Path(os.path.relpath(CODE_DIR / path, CODE_DIR)).as_posix()


def project_hashes(hashes):
    """Read relative manifests and absolute keys from older training records."""
    relative = {}
    for path, digest in hashes.items():
        reference = project_reference(path)
        if reference in relative and relative[reference] != digest:
            raise ValueError(f'Conflicting digests for {reference}')
        relative[reference] = digest
    return relative


DATA_ROOT = configured_path('ZOOMREC_DATA_ROOT', '../zoomrec_data')
ABLATION_REQUEST = configured_path('ZOOMREC_ABLATION_REQUEST', 'ablation_request.json')
HERE = configured_path('ZOOMREC_RUN_DIR', '../zoomrec_runs/loo_unseen_top20')
SOURCE = HERE / 'inputs'
MODEL_PATH = configured_path('ZOOMREC_QWEN_MODEL', '../models/Qwen3-4B-Instruct-2507')
QWEN_PORT = int(os.environ.get('ZOOMREC_QWEN_PORT', '24215'))
QWEN_URL = os.environ.get('ZOOMREC_QWEN_URL', f'http://127.0.0.1:{QWEN_PORT}/v1/completions')
CHECKPOINT_MODE = os.environ.get('ZOOMREC_CHECKPOINT', 'fresh')
