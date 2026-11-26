"""Write the requested method rows and eight ranking metrics."""
from full_evaluation import HERE, METRICS, VARIANTS, read


LABELS = {
    'full': 'ZoomRec',
    'no_adaptive_granularity': 'w/o Adaptive Granularity',
    'no_local_reopening': 'w/o Local Reopening',
    'no_local_revision_and_state_recovery': 'w/o Local Revision and State Recovery',
    'no_agent_review': 'w/o Agent Review',
    'no_revision': 'No local revision (claim content stays fixed)',
    'no_recovery': 'w/o Local Reopening',
    'no_both': 'All three mechanisms disabled',
}


def write_report(result):
    assert result['status'] == 'complete' and set(result['metrics']) == set(VARIANTS)
    is_full = list(VARIANTS) == ['full']
    selection = result['manifest']['model_selection']
    lines = [
        '# Full-method results' if is_full else '# Ablation results',
        '',
        f"Amazon Beauty leave-one-out evaluation covered {result['n_users']:,} users and "
        f"{result['n_events']:,} test events. SASRec retrieves the 20 highest-scoring items "
        "outside the full observed history, breaking ties by ascending item ID. "
        "All methods receive the same shuffled candidates and their aligned raw scores. "
        "A target outside the retrieved list contributes zero at every HR and NDCG cutoff. "
        "All test events are included in the denominator.",
        '',
        "The last interaction is held out for testing and the penultimate interaction for "
        "validation. Earlier interactions form the training set. The encoder reads the latest "
        "50 items, while preference evidence uses the full observed history. Test history "
        "includes the validation interaction. Model parameters stay frozen during testing.",
        '',
        f"SASRec trained from scratch for {selection['epochs_completed']} epochs. Validation "
        "ran every five epochs, with early stopping after five consecutive evaluations "
        "without improvement once the minimum training length was reached. The frozen checkpoint "
        f"from epoch {selection['best_epoch']} had the best validation NDCG@10.",
        '',
        '| Method | ' + ' | '.join(METRICS) + ' |',
        '| ' + ' | '.join(['---'] * (len(METRICS) + 1)) + ' |',
    ]
    for variant in VARIANTS:
        row = result['metrics'][variant]
        assert list(row) == METRICS
        lines.append('| ' + LABELS[variant] + ' | ' + ' | '.join(f'{row[m]:.6f}' for m in METRICS) + ' |')
    lines += [
        '',
        'Metrics range from 0 to 1. All methods use the same frozen history summaries, '
        'baseline rankings and candidate sets. The w/o Agent Review variant uses rule-qualified matches directly.',
        '',
        'The full method and all four ablations allow interventions regardless of the number of '
        'earlier visits to the target category. The test set was excluded from parameter selection.',
        '',
        'This exploratory evaluation uses title tokens as attribute proxies. HR and NDCG measure '
        'ranking quality. Assessing the meaning of the selected granularity and the effects of '
        'local revisions requires separate evidence.',
        '',
        'The manifest records the selected methods, implementation, input hashes and inference settings.',
        '',
    ]
    filename = 'FULL_METHOD_REPORT.md' if is_full else 'ABLATION_REPORT.md'
    (HERE / filename).write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    write_report(read(HERE / 'results.json'))
