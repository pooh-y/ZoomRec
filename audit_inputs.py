"""Audit actual leave-one-out files and their training/inference boundaries."""
from collections import Counter
from data_split import PROTOCOL, audit_events
from sasrec_candidates import POLICY
from full_evaluation import read, jsonl, dump, sha, verify_manifest
from run_paths import DATA_ROOT, HERE, SOURCE
from training_protocol import validate_training_record


def main():
    manifest = read(HERE / 'manifest.json'); verify_manifest(manifest)
    sequences = read(DATA_ROOT / 'freeze/sequences.json')
    cards = {int(k) for k in read(DATA_ROOT / 'freeze/cards.json')}
    freeze = SOURCE / 'freeze'
    valid, test = jsonl(freeze / 'valid.jsonl'), jsonl(freeze / 'test.jsonl')
    audit_events(valid, sequences, cards, 'valid')
    audit_events(test, sequences, cards, 'test')
    transitions = Counter(); observed_positions = set()
    for uid, history, target in jsonl(freeze / 'train.jsonl'):
        items = sequences[uid]['items']; index = len(history)
        assert 1 <= index < len(items) - 2
        assert history == items[:index] and target == items[index]
        assert (uid, index) not in observed_positions
        observed_positions.add((uid, index)); transitions[uid] += 1
    assert all(transitions[u] == len(s['items']) - 3 for u, s in sequences.items())
    assert all(v['uid'] == t['uid'] and t['history'] == v['history'] + [v['target']]
               for v, t in zip(valid, test))
    training = read(HERE / 'model/loo_training.json')
    assert training['config']['protocol'] == PROTOCOL
    assert training['config']['train_sha256'] == sha(freeze / 'train.jsonl')
    assert training['config']['valid_queries_sha256'] == sha(freeze / 'valid_queries.jsonl')
    audit = read(HERE / 'candidate_audit.json')
    assert audit['candidate_policy'] == manifest['candidate_policy'] == POLICY
    assert audit['checkpoint_sha256'] == training['checkpoint_sha256']
    for phase, events in [('valid', valid), ('test', test)]:
        assert audit[phase]['exact_top20_matches'] == audit[phase]['verified_queries'] == len(events)
        retrieved = sum(row['target'] in row['candidates'] for row in events)
        assert audit[phase]['retrieved_targets'] == retrieved
        assert audit[phase]['missed_targets'] == len(events) - retrieved
        assert audit[phase]['events_sha256'] == sha(freeze / (phase + '.jsonl'))
    assert training['checkpoint_sha256'] == sha(HERE / 'model/sasrec_corrected_seed_20261101.pt')
    record = read(HERE / 'model/train_config.json')
    selection = validate_training_record(record)
    assert selection == training['selection'] == manifest['model_selection']
    assert record['train_sha256'] == training['config']['train_sha256']
    assert record['valid_queries_sha256'] == training['config']['valid_queries_sha256']
    assert record['checkpoint_sha256'] == training['checkpoint_sha256']
    assert record['policy'] == training['config']['training_policy'] == manifest['training_policy']
    out = dict(status='passed', protocol=PROTOCOL, counts=manifest['counts'],
        training_transitions=sum(transitions.values()),
        test_history_includes_validation=True, encoder_max_length=50,
        preference_memory='complete observed prefix', model_checkpoint_unchanged=True,
        training_policy=record['policy'], model_selection=selection,
        checks=['exactly one validation and one test target per user',
                'all training targets precede validation', 'full pre-event histories',
                'top 20 outside the full observed prefix, shuffled with the original SASRec scores',
                'model was trained on the new split and checkpoint remains unchanged',
                'validation every 5 epochs, minimum 20 epochs, patience of 5 evaluations',
                'saved checkpoint has the highest historical validation NDCG@10',
                'validation targets excluded from gradient updates; no test-based selection'])
    dump(HERE / 'input_audit.json', out)
    print(out)


if __name__ == '__main__': main()
