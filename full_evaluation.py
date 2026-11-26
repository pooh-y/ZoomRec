"""Evaluate Beauty on the shuffled SASRec top 20 outside each observed prefix.

Only history-derived baseline scores are reused. Agent approvals are independent.
"""
from __future__ import annotations
import argparse, ast, copy, csv, gzip, hashlib, json, math, os, time
from dataclasses import asdict
from pathlib import Path
from ledger import clean_cards, review_event
from preference_graph import ScopedPreferenceMemory, ALL_VARIANTS
from recommendation_agent import (AGENT_CONFIG, bounded_rank, decision_prompt,
    evidence_pages, execute_plan, make_context, plan_prompt, rule_approvals, validate_decision)
from generation_protocol import generation_settings, MAX_MODEL_LEN
from data_split import PROTOCOL, cuts, make_events, audit_events
from run_paths import ABLATION_REQUEST, CODE_DIR, HERE, DATA_ROOT, SOURCE, project_hashes, project_reference
from sasrec_candidates import POLICY as CANDIDATE_POLICY
from ranking_metrics import METRICS, TOP_K, metric_table, target_rank
from training_protocol import VERSION as TRAINING_VERSION, training_policy, validate_training_record

SEED = 20261101

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


_request = read(ABLATION_REQUEST)
_names = _request['requested_variants']
if not _names or len(_names) != len(set(_names)) or set(_names) - set(ALL_VARIANTS):
    raise ValueError('requested_variants must contain distinct supported method names')
if _request['run_full_method'] != ('full' in _names):
    raise ValueError('run_full_method disagrees with requested_variants')
VARIANTS = {name: ALL_VARIANTS[name] for name in _names}

def jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def check_hashes(hashes):
    hashes = project_hashes(hashes)
    if project_reference(DATA_ROOT / 'freeze/sequences.json') not in hashes:
        raise ValueError('The prepared data directory changed. Use a new ZOOMREC_RUN_DIR.')
    for path, digest in hashes.items():
        if sha(CODE_DIR / path) != digest:
            raise ValueError(f'Inputs or implementation changed: {path}. Use a new ZOOMREC_RUN_DIR.')


def verify_source():
    files = [*sorted(CODE_DIR.glob('*.py')), ABLATION_REQUEST,
             CODE_DIR / 'selection.json', CODE_DIR / 'configs/data_sources.json']
    return dict(source_directory='.', portable=True, files_checked=len(files),
                sha256={project_reference(p): sha(p) for p in files})


def validate_cases(cases, events):
    assert len(cases) == len(events)
    for case, event in zip(cases, events):
        assert all(case[k] == event[k] for k in ('event_id', 'uid', 'index', 'history', 'target', 'candidates', 'sasrec_scores'))
        assert len(case['baseline']) == TOP_K and all(math.isfinite(s) for s in case['baseline'])

def verify_manifest(manifest):
    assert manifest['protocol'] == PROTOCOL
    if manifest.get('training_policy', {}).get('version') != TRAINING_VERSION:
        raise ValueError('Old fixed-epoch model: regenerate this run with validation-based checkpoint selection')
    if manifest.get('candidate_policy') != CANDIDATE_POLICY:
        raise ValueError('Candidate policy differs; use a fresh Top-20 run directory')
    selection = validate_training_record(read(HERE / 'model/train_config.json'))
    assert manifest['model_selection'] == selection
    assert manifest['counts']['test_events'] == manifest['counts']['valid_events'] == manifest['counts']['users']
    assert manifest['variants'] == {k: asdict(v) for k, v in VARIANTS.items()}
    assert manifest['metrics'] == METRICS and manifest['cold_category_limit'] is False
    assert manifest['inference'] == inference_settings()
    check_hashes(manifest['source_sha256'])

def inference_settings():
    return dict(plan=generation_settings(AGENT_CONFIG['plan_tokens']),
        decision=generation_settings(AGENT_CONFIG['decision_tokens']), max_model_len=MAX_MODEL_LEN)


def prepare(device='cpu', min_epochs=20, batch=512, chunk_size=256):
    import fcntl
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(device, min_epochs, batch, chunk_size)


def _prepare(device, min_epochs, batch, chunk_size):
    policy = training_policy(min_epochs, batch)
    if (HERE / 'manifest.json').exists():
        manifest = read(HERE / 'manifest.json'); verify_manifest(manifest)
        assert manifest['training_policy'] == policy, 'Requested training policy differs from saved run'
        assert manifest['chunk_size'] == chunk_size, 'Chunk size changed; use a new run directory'
        assert manifest['input_provenance']['training']['checkpoint_mode'] == 'fresh'
        return manifest
    request = read(ABLATION_REQUEST)
    assert request['confirmed_decisions']['restrict_to_cold_categories'] is False
    source_check = verify_source()
    from prepare_loo import prepare_inputs
    inputs = prepare_inputs(device=device, min_epochs=min_epochs, batch=batch, chunk_size=chunk_size)
    seqs = read(DATA_ROOT / 'freeze/sequences.json')
    cards = {int(k): v for k, v in read(DATA_ROOT / 'freeze/cards.json').items()}
    events = jsonl(SOURCE / 'events.jsonl'); audit = audit_events(events, seqs, set(cards))
    paths = [CODE_DIR / path for path in inputs['source_sha256']]
    for shard_id in range(inputs['n_shards']):
        case_path = SOURCE / 'shards' / f'{shard_id:04d}' / 'cases.jsonl'
        cases = jsonl(case_path)
        batch_events = events[shard_id * inputs['chunk_size']:(shard_id + 1) * inputs['chunk_size']]
        validate_cases(cases, batch_events)
    paths += [SOURCE / 'manifest.json', DATA_ROOT / 'data/raw/reviews_Beauty_5.json.gz',
              ABLATION_REQUEST, *sorted(CODE_DIR.glob('*.py'))]
    manifest = dict(status='prepared', protocol=PROTOCOL, training_policy=policy,
        model_selection=inputs['model_selection'], candidate_policy=inputs['candidate_policy'], seed=SEED, candidate_count=TOP_K, counts=inputs['counts'],
        audit=audit, n_shards=inputs['n_shards'], chunk_size=inputs['chunk_size'],
        selection=read(HERE / 'selection.json')['selected'], variants={k: asdict(v) for k, v in VARIANTS.items()},
        metrics=METRICS, cold_category_limit=False, source_preservation=source_check,
        inference=inference_settings(),
        source_sha256={project_reference(p): sha(p) for p in paths},
        baseline='SASRec trained from scratch, selected by validation NDCG@10 with early stopping; frozen best checkpoint + history preference scores + evidence-constrained reranking.',
        input_provenance=inputs['provenance'],
        attributes='Filtered title-token proxies; no semantic attribute ground truth.',
        design=('Full method with adaptive granularity, local revision and recovery enabled.'
                if list(VARIANTS) == ['full'] else
                'Selected methods evaluated on the same frozen inputs and ranking settings.'),
        scope_revision='Narrow only contradicted destination scope; reopen after two subsequent distinct supportive items.',
        granularity='Adaptive: supported attribute else independently supported category. Fixed: category then attribute.',
        rating_interpretation='Low item ratings are weak scoped counterevidence, not causal attribute labels.')
    dump(HERE / 'manifest.json', manifest); return manifest

class Reviews:
    def __init__(self, path, cards):
        self.asins = {c['asin']: i for i, c in cards.items()}; self.rows = []
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            for line in handle:
                try: row = json.loads(line)
                except json.JSONDecodeError: row = ast.literal_eval(line)
                self.rows.append(row)

    def event(self, uid, index, item, sequence):
        row_id = sequence['row_ids'][index]; timestamp = sequence['times'][index]
        # Retain original exact identity check and 0/1-based row-ID fallback.
        for candidate in (row_id, row_id - 1):
            if 0 <= candidate < len(self.rows):
                r = self.rows[candidate]
                if (r['reviewerID'] == uid and self.asins.get(r['asin']) == item and r['unixReviewTime'] == timestamp):
                    return review_event(uid, index, item, timestamp, r.get('overall'))
        raise ValueError(f'cannot match actual review rating: {uid}:{index}')

def run_variant(qwen, cases, cards, boundary, sequences, reviews, variant, folder, setting):
    folder = Path(folder)
    mechanisms = ALL_VARIANTS[variant]
    prepared, plan_requests = [], []
    for case in cases:
        m = ScopedPreferenceMemory(boundary, variant)
        seq = sequences[case['uid']]
        for j, item in enumerate(case['history']):
            m.observe(reviews.event(case['uid'], j, item, seq))
        ctx = make_context(case['history'], case['candidates'], case['baseline'], case['gate'], m, cards,
                           include_first=not mechanisms.agent_review)
        pages = {}
        review_pages = evidence_pages(ctx) if mechanisms.agent_review else ()
        for page_index, page in enumerate(review_pages):
            call_id = case['event_id'] if page_index == 0 else f"{case['event_id']}/page/{page_index + 1}"
            pages[call_id] = page
            plan_requests.append(dict(id=call_id, prompt=plan_prompt(page)))
        prepared.append((case, m, ctx, pages))

    def generate(requests, stage):
        if not requests:
            return {}
        calls = qwen.generate(requests, folder / f'{stage}_calls.jsonl', AGENT_CONFIG[f'{stage}_tokens'])
        return {row['id']: row['response'] for row in calls}

    plans = generate(plan_requests, 'plan')
    observations, decision_requests = {}, []
    for _, _, _, pages in prepared:
        for call_id, page in pages.items():
            obs = execute_plan(plans[call_id], page, cards)
            observations[call_id] = obs
            if obs['retrieved'] and obs['verified']:
                decision_requests.append(dict(id=call_id, prompt=decision_prompt(page, obs)))
    decisions = generate(decision_requests, 'decision')
    actions, rows, executions = [], [], []
    for case, m, ctx, pages in prepared:
        approved, errors, retrieved, plan_errors, verified = [], [], {}, [], {}
        decision_ids = []
        for call_id, page in pages.items():
            obs = observations[call_id]
            retrieved.update((row['id'], row) for row in obs['retrieved'])
            plan_errors.extend(obs['errors'])
            for item in obs['verified']:
                combined = verified.setdefault(item['item'], dict(item, matching_hypotheses=[]))
                combined['matching_hypotheses'].extend(
                    hid for hid in item['matching_hypotheses'] if hid not in combined['matching_hypotheses'])
            if call_id in decisions:
                result = validate_decision(decisions[call_id], page, obs)
                approved.extend(result['approved'])
                errors.extend(result['errors'])
                decision_ids.append(call_id)
        if not mechanisms.agent_review:
            approved = rule_approvals(ctx)
        obs = dict(retrieved=list(retrieved.values()), verified=list(verified.values()), errors=plan_errors)
        prior = m.trace()
        context = copy.deepcopy(ctx['summary'])
        order, execution = bounded_rank(case['candidates'], case['baseline'], case['gate'], approved, setting,
                                        agent_review=mechanisms.agent_review,
                                        strongest_only=variant == 'no_adaptive_granularity')
        row = {k: v for k, v in case.items() if k not in ('baseline', 'target_category', 'sasrec_rank',
            'sasrec_top10', 'baseline_top10', 'baseline_rank')}
        row.update(method=variant, rank=target_rank(order, case['target']), top10=order[:10],
            n_approved=len({a['item'] for a in approved}),
            n_approved_claims=len(approved), ranking_changed=bool(execution['crossings']))
        rows.append(row); executions.append(dict(event_id=case['event_id'], method=variant, **execution))
        # Hidden outcome enters memory only after this event's ranking decisions.
        m.observe(reviews.event(case['uid'], case['index'], case['target'], sequences[case['uid']]))
        actions.append(dict(event_id=case['event_id'], method=variant, context=context, observations=obs,
            approved=approved, decision_errors=errors, plan_called=bool(pages),
            decision_called=bool(decision_ids), plan_call_ids=list(pages), decision_call_ids=decision_ids, pre_observation=prior, post_observation=m.trace()))
    write_jsonl(folder / 'actions.jsonl', actions); write_jsonl(folder / 'predictions.jsonl', rows)
    write_jsonl(folder / 'ranking_executions.jsonl', executions); dump(folder / 'metrics.json', metric_table(rows))
    return rows

def worker(worker_id, workers, limit=0):
    import fcntl
    from qwen_vllm import LocalQwen
    started = time.time(); manifest = read(HERE / 'manifest.json'); verify_manifest(manifest)
    seqs = read(DATA_ROOT / 'freeze/sequences.json')
    cards0 = {int(k): v for k, v in read(DATA_ROOT / 'freeze/cards.json').items()}
    boundary = read(HERE / 'boundary_model.json')
    cards = clean_cards(cards0, {int(k): v for k, v in boundary['categories'].items()})
    reviews = Reviews(DATA_ROOT / 'data/raw/reviews_Beauty_5.json.gz', cards0)
    events = jsonl(SOURCE / 'events.jsonl'); qwen = LocalQwen(); completed = []
    status_path = HERE / f'worker_{worker_id}_status.json'
    order = list(range(worker_id, manifest['n_shards'], workers)) + [j for j in range(manifest['n_shards']) if j % workers != worker_id]
    try:
        for shard_id in order:
            folder = HERE / 'shards' / f'{shard_id:04d}'; folder.mkdir(parents=True, exist_ok=True)
            with (folder / 'worker.lock').open('a') as lock:
                try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: continue
                if (folder / 'done.json').exists(): continue
                cases = jsonl(SOURCE / 'shards' / f'{shard_id:04d}' / 'cases.jsonl')
                validate_cases(cases, events[shard_id * manifest['chunk_size']:(shard_id + 1) * manifest['chunk_size']])
                tables = {}
                for variant in VARIANTS:
                    dump(status_path, dict(state='running', worker=worker_id, shard=shard_id, variant=variant,
                        completed_shards=completed, elapsed=time.time() - started))
                    rows = run_variant(qwen, cases, cards, boundary, seqs, reviews, variant, folder / variant, manifest['selection'])
                    tables[variant] = metric_table(rows)
                dump(folder / 'done.json', dict(shard=shard_id, n_events=len(cases), metrics=tables))
                completed.append(shard_id)
                print(json.dumps(dict(shard_done=shard_id, worker=worker_id, events=len(cases), elapsed=time.time() - started)), flush=True)
                if limit and len(completed) >= limit: break
        dump(status_path, dict(state='complete', worker=worker_id, completed_shards=completed, elapsed=time.time() - started))
    except Exception as exc:
        dump(status_path, dict(state='failed', error=repr(exc), worker=worker_id)); raise

def aggregate():
    manifest = read(HERE / 'manifest.json'); verify_manifest(manifest); events = jsonl(SOURCE / 'events.jsonl')
    folders = [HERE / 'shards' / f'{j:04d}' for j in range(manifest['n_shards'])]
    missing = [p.name for p in folders if not (p / 'done.json').exists()]
    if missing: raise RuntimeError(f'{len(missing)} unfinished shards; refusing partial final metrics')
    tables, mechanism = {}, {}
    for variant in VARIANTS:
        rows = [r for p in folders for r in jsonl(p / variant / 'predictions.jsonl')]
        assert len(rows) == len(events)
        for event, row in zip(events, rows):
            assert row['method'] == variant
            assert all(event[k] == row[k] for k in ('event_id', 'uid', 'index', 'history', 'target', 'candidates', 'sasrec_scores'))
            assert len(row['top10']) == len(set(row['top10'])) == 10 and set(row['top10']) <= set(row['candidates'])
            rank = row['rank']
            assert (rank is not None) == (row['target'] in row['candidates'])
            assert (row['target'] in row['top10']) == (rank is not None and rank <= 10)
            if rank is not None and rank <= 10:
                assert row['top10'][rank - 1] == row['target']
        tables[variant] = metric_table(rows)
        mechanism[variant] = dict(changed_rankings=sum(r['ranking_changed'] for r in rows),
            approved_items=sum(r['n_approved'] for r in rows),
            approved_claims=sum(r['n_approved_claims'] for r in rows))
    result = dict(status='complete', seed=SEED, n_users=manifest['counts']['users'], n_events=len(events), metrics=tables,
        manifest=manifest, source_preservation=verify_source())
    dump(HERE / 'results.json', result); dump(HERE / 'mechanism_audit.json', mechanism)
    with (HERE / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['method', *METRICS]); writer.writeheader()
        writer.writerows(dict(method=v, **tables[v]) for v in VARIANTS)
    from write_report import write_report
    write_report(result)
    print(json.dumps(dict(status='complete', metrics=tables), ensure_ascii=False, indent=2)); return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'worker', 'aggregate', 'status'])
    parser.add_argument('--worker', type=int, default=0); parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--device', default='cpu', help='SASRec preparation device, e.g. cuda:2 or cpu')
    parser.add_argument('--min-epochs', '--epochs', dest='min_epochs', type=int, default=20,
                        help='Minimum training duration (at least 20); not a fixed epoch budget')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--chunk-size', type=int, default=256)
    parser.add_argument('--limit', type=int, default=0); args = parser.parse_args()
    if args.command == 'prepare': print(json.dumps(prepare(args.device, args.min_epochs, args.batch_size, args.chunk_size), ensure_ascii=False, indent=2))
    elif args.command == 'worker': worker(args.worker, args.workers, args.limit)
    elif args.command == 'aggregate': aggregate()
    else:
        done = list((HERE / 'shards').glob('*/done.json'))
        print(json.dumps(dict(completed_shards=len(done), completed_events=sum(read(p)['n_events'] for p in done),
            workers=[read(p) for p in HERE.glob('worker_*_status.json')]), ensure_ascii=False))

if __name__ == '__main__': main()
