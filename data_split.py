"""Chronological leave-one-out with full pre-event histories."""
import math

from ranking_metrics import TOP_K

SEED = 20261101
PROTOCOL = 'leave-one-out-v1'


def cuts(n):
    if n < 3:
        raise ValueError('Leave-one-out requires at least three interactions per user')
    return n - 2, n - 1


def training_rows(sequences):
    for uid, sequence in sorted(sequences.items()):
        items = sequence['items']
        train_stop, _ = cuts(len(items))
        for index in range(1, train_stop):
            yield [uid, items[:index], items[index]]


def make_queries(sequences, catalog, phase):
    """Keep the observed prefix and held-out label separate."""
    if phase not in ('valid', 'test'):
        raise ValueError('phase must be valid or test')
    ids = set(catalog)
    if ids != set(range(1, len(ids) + 1)):
        raise ValueError('Expected contiguous positive catalog IDs')
    if len(ids) < TOP_K:
        raise ValueError(f'The catalog must contain at least {TOP_K} real items')
    rows = []
    for uid, sequence in sorted(sequences.items()):
        items = sequence['items']
        validation, test = cuts(len(items))
        index = validation if phase == 'valid' else test
        if len(items) != len(sequence['times']) or len(items) != len(sequence['row_ids']):
            raise ValueError(f'Incomplete sequence for {uid}')
        if sequence['times'] != sorted(sequence['times']) or not set(items) <= ids:
            raise ValueError(f'Invalid sequence for {uid}')
        rows.append(dict(event_id=f'{phase}:{uid}:{index}', uid=uid, index=index, phase=phase,
            history=items[:index], target=items[index]))
    return rows


def make_events(sequences, catalog, phase='test', *, model, device='cpu', batch=512):
    """Freeze one shuffled unseen-item pool with its raw scores for each prediction."""
    from sasrec_candidates import materialize_events

    queries = make_queries(sequences, catalog, phase)
    events = materialize_events(model, queries, device, batch)
    users = len(sequences)
    interactions = sum(len(row['items']) for row in sequences.values())
    counts = dict(users=users, interactions=interactions, train_interactions=interactions - 2 * users,
                  training_transitions=interactions - 3 * users, valid_events=users, test_events=users)
    return events, counts


def audit_events(events, sequences, catalog, phase='test'):
    if phase not in ('valid', 'test'):
        raise ValueError('phase must be valid or test')
    expected = {(uid, cuts(len(s['items']))[int(phase == 'test')]) for uid, s in sequences.items()}
    actual = {(e['uid'], e['index']) for e in events}
    assert actual == expected and len(events) == len(actual)
    assert len({e['event_id'] for e in events}) == len(events)
    catalog = set(catalog)
    for event in events:
        sequence = sequences[event['uid']]; index = event['index']; candidates = event['candidates']
        assert event['phase'] == phase
        assert event['event_id'] == f"{phase}:{event['uid']}:{index}"
        assert event['history'] == sequence['items'][:index]
        assert event['target'] == sequence['items'][index]
        assert len(candidates) == len(set(candidates)) == TOP_K
        assert set(candidates) <= catalog and 0 not in candidates
        assert set(candidates).isdisjoint(event['history'])
        assert len(event['sasrec_scores']) == len(candidates)
        assert all(math.isfinite(score) for score in event['sasrec_scores'])
    return dict(protocol=PROTOCOL, n_events=len(events), n_users=len(sequences),
        history='complete observed pre-event prefix; test includes validation interaction',
        encoder_max_length=50, memory_history='complete observed pre-event prefix',
        candidates='raw SASRec top 20 outside the observed prefix, shuffled with aligned scores',
        coverage=f'exactly one {phase} event per user', model_updates_during_test=False)
