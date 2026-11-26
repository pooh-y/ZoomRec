"""Train SASRec, freeze the selected model and build chronological Top-20 candidate pools."""
import argparse
import math
import shutil

import torch

import baseline_source as baseline
from candidate_audit import verify_phase
from data_split import PROTOCOL, audit_events, make_events, make_queries, training_rows
from full_evaluation import check_hashes, dump, jsonl, read, sha, validate_cases, write_jsonl
from ledger import clean_cards, fit_boundaries
from run_paths import CHECKPOINT_MODE, CODE_DIR, DATA_ROOT, HERE, SOURCE, project_hashes, project_reference
from sasrec_candidates import POLICY
from training_protocol import training_policy, validate_training_record

MODEL_DIR = HERE / 'model'
CHECKPOINT = MODEL_DIR / 'sasrec_corrected_seed_20261101.pt'
TRAINING = MODEL_DIR / 'loo_training.json'
RECORD = MODEL_DIR / 'train_config.json'
CANDIDATE_AUDIT = HERE / 'candidate_audit.json'


def prepare_training(sequences, cards, data_files, device, min_epochs, batch):
    freeze = SOURCE / 'freeze'
    policy = training_policy(min_epochs, batch)
    train_path = freeze / 'train.jsonl'
    valid_path = freeze / 'valid_queries.jsonl'
    inputs = [*data_files, *(CODE_DIR / name for name in
              ('baseline_source.py', 'training_protocol.py', 'data_split.py', 'sasrec_candidates.py', 'ranking_metrics.py'))]
    config = dict(protocol=PROTOCOL, candidate_policy=POLICY, training_policy=policy,
                  checkpoint_mode='fresh', seed=baseline.SEED, encoder_max_length=50,
                  input_hashes={project_reference(p): sha(p) for p in inputs}, item_map_sha256=sha(data_files[2]),
                  validation_used_for_gradients=False, validation_used_for_checkpoint_selection=True,
                  test_used_for_fitting=False, test_used_for_checkpoint_selection=False)
    cached = read(TRAINING) if TRAINING.is_file() else None
    if cached is not None:
        # Older training records used absolute keys in this mapping.
        cached['config']['input_hashes'] = project_hashes(cached['config']['input_hashes'])
        if any(cached['config'].get(k) != v for k, v in config.items()):
            raise ValueError('Saved SASRec training uses different inputs or candidates. Use a new ZOOMREC_RUN_DIR.')
    elif RECORD.is_file() and read(RECORD).get('policy') != policy:
        raise ValueError('Incompatible partial training run. Use a new ZOOMREC_RUN_DIR.')

    freeze.mkdir(parents=True, exist_ok=True)
    for path in data_files[:3]:
        local = freeze / path.name
        if not local.exists():
            shutil.copyfile(path, local)
        if sha(local) != config['input_hashes'][project_reference(path)]:
            raise ValueError(f'Frozen data mismatch: {project_reference(local)}. Use a new ZOOMREC_RUN_DIR.')

    if cached is None:
        write_jsonl(train_path, training_rows(sequences))
        write_jsonl(valid_path, make_queries(sequences, cards, 'valid'))
    config.update(train_sha256=sha(train_path), valid_queries_sha256=sha(valid_path))
    if cached is None:
        print(f'Training SASRec from scratch on {device} using current-model Top-20 validation', flush=True)
        model, _ = baseline.train_model(SOURCE, device, min_epochs, batch, output_dir=MODEL_DIR)
        del model
    elif cached['config'] != config:
        raise ValueError('Training examples or validation queries changed. Use a new ZOOMREC_RUN_DIR.')

    record = read(RECORD)
    selected = validate_training_record(record)
    digest = sha(CHECKPOINT)
    if (record['policy'] != policy or record['checkpoint_sha256'] != digest
            or any(record[k] != config[k] for k in ('train_sha256', 'valid_queries_sha256', 'item_map_sha256'))):
        raise ValueError('Training record does not match the selected checkpoint and inputs.')
    result = dict(config=config, selection=selected, checkpoint_sha256=digest)
    if cached is None:
        dump(TRAINING, result)
    elif cached != result:
        raise ValueError('Saved training manifest does not match the selected checkpoint.')
    return result


def prepare_model(device='cpu', min_epochs=20, batch=512, checkpoint_mode=None):
    """Write audited validation/test pools before history summaries or ranking start."""
    requested = checkpoint_mode or CHECKPOINT_MODE
    if requested == 'bundled':
        print('The bundled checkpoint uses an earlier validation protocol. '
              'Selecting Top-20 weights in the run directory.', flush=True)
    elif requested != 'fresh':
        raise ValueError('Unknown checkpoint mode')
    policy = training_policy(min_epochs, batch)
    if CANDIDATE_AUDIT.is_file():
        cached = read(CANDIDATE_AUDIT)
        if (cached['training_policy'] != policy or cached['candidate_policy'] != POLICY
                or cached['protocol'] != PROTOCOL):
            raise ValueError('Incompatible candidate cache. Use a new ZOOMREC_RUN_DIR.')
        check_hashes(cached['source_sha256'])
        return cached
    if (HERE / 'manifest.json').exists() or (SOURCE / 'manifest.json').exists():
        raise ValueError('Existing evaluation lacks its Top-20 candidate audit. Use a new ZOOMREC_RUN_DIR.')

    torch.set_num_threads(4)
    data_files = [DATA_ROOT / 'freeze' / name for name in
                  ('sequences.json', 'cards.json', 'item_map.json', 'categories.json')]
    if any(not p.is_file() for p in data_files):
        raise FileNotFoundError('Prepare the Beauty data with python prepare_data.py --download first.')
    sequences = read(data_files[0])
    cards = {int(k): v for k, v in read(data_files[1]).items()}
    if set(cards) != set(read(data_files[2]).values()):
        raise ValueError('Catalog and item mapping differ')
    training = prepare_training(sequences, cards, data_files, device, min_epochs, batch)
    model = baseline.SASRec(len(cards)).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval().requires_grad_(False)
    freeze = SOURCE / 'freeze'
    audits = {}
    for phase in ('valid', 'test'):
        print(f'Mining and independently checking SASRec Top-20 for {phase}', flush=True)
        events, counts = make_events(sequences, cards, phase, model=model, device=device, batch=batch)
        audit_events(events, sequences, cards, phase)
        audit = verify_phase(model, events, sequences, len(cards), device, batch)
        path = freeze / (phase + '.jsonl')
        write_jsonl(path, events)
        audits[phase] = dict(**audit, events_sha256=sha(path))
    del model
    if not math.isclose(audits['valid']['sasrec_validation_or_test_ndcg10'],
                        training['selection']['best_validation_ndcg10'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Restored validation ranking differs from the selected checkpoint')
    if sha(CHECKPOINT) != training['checkpoint_sha256']:
        raise ValueError('Candidate evaluation modified the checkpoint')
    write_jsonl(SOURCE / 'events.jsonl', events)
    categories = {int(k): v for k, v in read(data_files[3]).items()}
    boundary = fit_boundaries(clean_cards(cards, categories), categories, sequences, set())
    assert boundary['audit']['training_events'] == counts['train_interactions']
    boundary['audit']['protocol'] = PROTOCOL
    dump(HERE / 'boundary_model.json', boundary)
    dump(HERE / 'selection.json', read(CODE_DIR / 'selection.json'))
    paths = [*data_files, CODE_DIR / 'selection.json', *sorted(CODE_DIR.glob('*.py')),
             HERE / 'boundary_model.json', HERE / 'selection.json', CHECKPOINT, TRAINING, RECORD,
             SOURCE / 'events.jsonl', *(freeze / name for name in ('train.jsonl', 'valid_queries.jsonl',
                 'valid.jsonl', 'test.jsonl', 'sequences.json', 'cards.json', 'item_map.json'))]
    result = dict(protocol=PROTOCOL, candidate_policy=POLICY, training_policy=policy,
                  model_selection=training['selection'], checkpoint_sha256=training['checkpoint_sha256'],
                  counts=dict(**counts, items=len(cards)), source_sha256={project_reference(p): sha(p) for p in paths}, **audits)
    dump(CANDIDATE_AUDIT, result)
    return result


def prepare_inputs(device='cpu', min_epochs=20, batch=512, chunk_size=256):
    from history_states import validate_states
    from run_ledger_gate import load_baseline

    if chunk_size < 1:
        raise ValueError('Chunk size must be positive')
    manifest_path = SOURCE / 'manifest.json'
    if manifest_path.exists():
        saved = read(manifest_path)
        if (saved.get('candidate_policy') != POLICY or saved['training_policy'] != training_policy(min_epochs, batch)
                or saved['chunk_size'] != chunk_size):
            raise ValueError('Input manifest uses a different protocol. Use a new ZOOMREC_RUN_DIR.')
        check_hashes(saved['source_sha256'])
        return saved
    setup = prepare_model(device, min_epochs, batch)
    events = jsonl(SOURCE / 'events.jsonl')
    states, state_manifest = validate_states(events)
    cards = {int(k): v for k, v in read(SOURCE / 'freeze/cards.json').items()}
    _, cases = load_baseline(cards, events, states_override=states, checkpoint=CHECKPOINT, device=device)
    validate_cases(cases, events)
    if sha(CHECKPOINT) != setup['checkpoint_sha256']:
        raise ValueError('Baseline scoring modified the checkpoint')
    paths = [CANDIDATE_AUDIT, SOURCE / 'states.jsonl', SOURCE / 'states_manifest.json']
    for start in range(0, len(cases), chunk_size):
        path = SOURCE / 'shards' / f'{start // chunk_size:04d}' / 'cases.jsonl'
        write_jsonl(path, cases[start:start + chunk_size])
        paths.append(path)
    hashes = project_hashes(setup['source_sha256'])
    hashes.update({project_reference(p): sha(p) for p in paths})
    training_config = read(TRAINING)['config']
    training_config['input_hashes'] = project_hashes(training_config['input_hashes'])
    manifest = dict(protocol=PROTOCOL, candidate_policy=POLICY, training_policy=setup['training_policy'],
        model_selection=setup['model_selection'], counts=setup['counts'],
        n_shards=math.ceil(len(cases) / chunk_size), chunk_size=chunk_size, source_sha256=hashes,
        provenance=dict(training=training_config, model_selection=setup['model_selection'],
            boundary=read(HERE / 'boundary_model.json')['audit'], summary=state_manifest,
            model_sha256_before_scoring=setup['checkpoint_sha256'], model_sha256_after_scoring=sha(CHECKPOINT),
            candidate_eligibility=POLICY['eligibility'], validation_in_test_history=True,
            preference_memory='complete observed history'))
    dump(manifest_path, manifest)
    return manifest


if __name__ == '__main__':
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', choices=('fresh', 'bundled'), default=CHECKPOINT_MODE)
    parser.add_argument('--min-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=512)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = prepare_model(args.device, args.min_epochs, args.batch_size, args.checkpoint)
    print(dict(counts=result['counts'], selection=result['model_selection']))

