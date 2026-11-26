"""Generate reproducible pre-test summaries; raw data and caches are not shipped."""
import argparse
import json
from pathlib import Path

from run_paths import CODE_DIR, DATA_ROOT, HERE, SOURCE
from history_prompts import prompt_for, parse_state


def validate_states(events):
    from full_evaluation import jsonl, read, sha
    path = SOURCE / 'states.jsonl'
    meta = read(SOURCE / 'states_manifest.json')
    assert meta['complete'], 'History summaries are incomplete; run history_states.py'
    assert meta['events_sha256'] == sha(SOURCE / 'events.jsonl')
    assert meta['cards_sha256'] == sha(DATA_ROOT / 'freeze/cards.json')
    assert meta['prompt_parser_sha256'] == sha(CODE_DIR / 'history_prompts.py')
    assert meta['states_sha256'] == sha(path)
    rows = jsonl(path); states = {r['event_id']: r['state'] for r in rows}
    assert len(states) == len(rows) == len(events) == meta['n_events']
    assert set(states) == {e['event_id'] for e in events}
    for state in states.values():
        assert isinstance(state['stable'], list) and isinstance(state['transient'], list)
        assert isinstance(state['raw'], str)
    return states, meta


def generate(import_cache=None, import_events=None, limit=0):
    from full_evaluation import dump, jsonl, read, sha, write_jsonl
    events = jsonl(SOURCE / 'events.jsonl')
    cards = {int(k): v for k, v in read(DATA_ROOT / 'freeze/cards.json').items()}
    if (SOURCE / 'states_manifest.json').exists():
        previous = read(SOURCE / 'states_manifest.json')
        if previous['complete']:
            states, _ = validate_states(events)
            print(f'Validated {len(states)} cached history summaries', flush=True)
            return
    chosen = events[:limit] if limit else events
    if import_cache:
        if not import_events: raise ValueError('--import-events is required with --import-cache')
        old_events = {e['event_id']: e for e in jsonl(import_events)}
        rows = jsonl(import_cache); states = {r['event_id']: r['state'] for r in rows}
        assert len(states) == len(rows), 'Duplicate imported history state'
        for event in chosen:
            old = old_events[event['event_id']]
            assert all(old[k] == event[k] for k in ('uid', 'index', 'history', 'target', 'candidates'))
            state = states[event['event_id']]
            assert state == parse_state(state['raw'], event, cards), 'Imported state/parser mismatch'
        origin = dict(mode='imported exact-history summaries', states_sha256=sha(import_cache),
                      events_sha256=sha(import_events))
    else:
        from history_qwen import HistoryQwen
        requests = [dict(id=e['event_id'], prompt=prompt_for(e, cards)) for e in chosen]
        responses = HistoryQwen().generate(requests, SOURCE / 'history_calls.jsonl', 96, max_length=2048)
        states = {e['event_id']: parse_state(r['response'], e, cards) for e, r in zip(chosen, responses)}
        origin = dict(mode='generated', calls_sha256=sha(SOURCE / 'history_calls.jsonl'),
                      max_new_tokens=96, max_input_tokens=2048, temperature=0., seed=20261101)
    path = SOURCE / 'states.jsonl'
    write_jsonl(path, (dict(event_id=e['event_id'], state=states[e['event_id']]) for e in chosen))
    meta = dict(complete=len(chosen) == len(events), n_events=len(chosen),
        events_sha256=sha(SOURCE / 'events.jsonl'), cards_sha256=sha(DATA_ROOT / 'freeze/cards.json'),
        prompt_parser_sha256=sha(CODE_DIR / 'history_prompts.py'), states_sha256=sha(path), origin=origin)
    dump(SOURCE / 'states_manifest.json', meta)
    if meta['complete']: validate_states(events)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    import fcntl
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-cache', type=Path)
    parser.add_argument('--import-events', type=Path)
    parser.add_argument('--limit', type=int, default=0, help='Debug only; partial states cannot enter formal evaluation')
    args = parser.parse_args()
    if args.limit < 0: parser.error('limit must be nonnegative')
    SOURCE.mkdir(parents=True, exist_ok=True)
    with (SOURCE / 'states.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generate(args.import_cache, args.import_events, args.limit)
