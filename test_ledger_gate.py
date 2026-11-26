"""Small mechanism tests; these do not establish recommendation quality."""
import copy
import json
import random

from ledger import PreferenceEvidenceTrajectory, review_event
from recommendation_agent import (
    bounded_rank, calibration_grid, decision_prompt, execute_plan, make_context,
    plan_prompt, validate_decision,
)


def run():
    model = dict(
        categories={'1': 'skin', '2': 'skin', '3': 'hair', '4': 'hair', '5': 'hair'},
        attributes={'1': ['travel'], '2': ['travel'], '3': ['travel'], '4': [], '5': ['travel']},
        boundaries=[dict(source='skin', destination='hair', attribute='travel', weight=.8)],
    )
    cards = {i: dict(title=f'item {i} travel' if i != 4 else 'item 4',
                     tokens=['travel'] if i != 4 else [],
                     categories=['skin' if i < 3 else 'hair']) for i in range(1, 6)}
    day = 86400
    m = PreferenceEvidenceTrajectory(model)
    m.observe(review_event('u', 0, 1, day, 5))
    m.observe(review_event('u', 1, 2, 10 * day, 5))
    key = ('skin', 'hair', 'travel')
    initial = m.assess(key)
    assert initial['state'] == 'pending' and initial['effective_weight'] > 0
    assert initial['source']['distinct_positive'] == 2 and initial['source']['episodes'] == 2
    m.observe(review_event('u', 2, 4, 11 * day, 5))
    assert m.assess(key)['destination']['distinct_positive'] == 0
    assert m.snapshot()['hypotheses'][key]['uninformative_target_events'] == 1
    m.observe(review_event('u', 3, 3, 12 * day, 1))
    assert m.assess(key)['destination']['negative'] > 0
    assert m.assess(key)['destination']['distinct_positive'] == 0
    withdrawn = PreferenceEvidenceTrajectory(model)
    for j, (item, rating) in enumerate([(1, 5), (2, 5), (3, 1), (5, 1)]):
        withdrawn.observe(review_event('w', j, item, (j * 10 + 1) * day, rating))
    assert withdrawn.assess(key)['state'] == 'withdrawn'
    assert withdrawn.assess(key)['effective_weight'] == 0
    withdrawn.revoke('w:3', 'invalid review')
    assert withdrawn.assess(key)['state'] == 'pending'
    duplicate_item = PreferenceEvidenceTrajectory(model)
    duplicate_item.observe(review_event('d', 0, 1, day, 5))
    duplicate_item.observe(review_event('d', 1, 1, 10 * day, 5))
    assert key not in duplicate_item.hypotheses
    before = m.snapshot()['version']
    m.revoke('u:1', 'corrected review')
    assert key not in m.hypotheses
    m.restore('u:1', 'correction reverted')
    assert key in m.hypotheses and m.snapshot()['version'] != before
    replay = PreferenceEvidenceTrajectory.replay(model, copy.deepcopy(m.operations))
    assert replay.trace() == m.trace()
    count = len(m.events)
    m.observe(review_event('u', 3, 3, 12 * day, 1))
    assert len(m.events) == count

    base = [.2, .3, .1]
    context = make_context([1, 2], [3, 4, 5], base, .8,
                           PreferenceEvidenceTrajectory.replay(model, m.operations[:2]), cards)
    reverse = make_context([1, 2], [5, 4, 3], base[::-1], .8,
                           PreferenceEvidenceTrajectory.replay(model, m.operations[:2]), cards)
    assert [row['item'] for row in context['summary']['options']] == [3, 5]
    assert [row['item'] for row in reverse['summary']['options']] == [5, 3]
    assert context['order'] == reverse['order']
    assert '"target"' not in plan_prompt(context)
    plan = json.dumps(dict(retrieve_history=['H1'], verify_items=[3]))
    observations = execute_plan(plan, context, cards)
    assert observations['retrieved'] and observations['verified']
    partial = execute_plan('{"retrieve_history":["H1"],"verify_items":[3,999]}', context, cards)
    assert partial['errors'] and partial['retrieved'] and partial['verified']
    assert [x['item'] for x in partial['verified']] == [3]
    reverse_observations = execute_plan(plan, reverse, cards)
    assert observations == reverse_observations
    assert '"target"' not in decision_prompt(reverse, reverse_observations)
    competitor = next(x for x in context['summary']['options'] if x['item'] == 3)['competitor']['item']
    decision = json.dumps(dict(transfers=[dict(item=3, hypothesis='H1',
        competitor=competitor, confidence='high', reason='observed twice')]))
    approved = validate_decision(decision, context, observations)
    assert len(approved['approved']) == 1 and not approved['errors']
    assert validate_decision(decision, reverse, reverse_observations) == approved
    assert not validate_decision(decision, context,
        dict(retrieved=[], verified=[], errors=[]))['approved']
    assert execute_plan('{"retrieve_history":["bad"],"verify_items":[3]}',
                        context, cards)['errors']

    rng = random.Random(20261104)
    for _ in range(25):
        candidates = list(range(1, 101))
        scores = [rng.uniform(-1, 1) for _ in candidates]
        approvals = [dict(item=i, ledger_weight=rng.random(), confidence='high')
                     for i in rng.sample(candidates, 8)]
        base_order = sorted(candidates, key=lambda i: (-scores[i - 1], i))
        positions = {i: j for j, i in enumerate(base_order)}
        for setting in calibration_grid():
            order, trace = bounded_rank(candidates, scores, .8, approvals, setting)
            reversed_order, _ = bounded_rank(candidates[::-1], scores[::-1], .8,
                                              approvals, setting)
            assert order == reversed_order and sorted(order) == candidates
            assert max(abs(j - positions[i]) for j, i in enumerate(order)) == trace['max_displacement']
            assert len(trace['crossings']) <= len(approvals)
            assert all(c['from_rank'] == c['to_rank'] + 1 for c in trace['crossings'])
    return dict(replay=True, revocation=True, counterevidence_withdrawal=True,
                repeated_item_not_independent_support=True, missing_attribute_neutral=True,
                low_rating_weak_counterevidence=True, duplicate_event_idempotent=True,
                tool_evidence_required=True, partial_invalid_requests_filtered=True,
                hidden_target_absent=True,
                candidate_display_order_preserved=True, score_ranking_permutation_invariant=True,
                bounded_ranking_cases=25 * len(calibration_grid()))


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
