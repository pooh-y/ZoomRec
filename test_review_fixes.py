"""Regression checks for the four issues reproduced in the server review."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from itertools import permutations

from audit_ablation_outputs import reconstruct_order
from full_evaluation import run_variant
from ledger import review_event
from preference_graph import ScopedPreferenceMemory, VARIANTS
from recommendation_agent import make_context, execute_plan, validate_decision, bounded_rank, evidence_pages
from test_ablation_mechanisms import MODEL, CARDS, KEY, OTHER, DAY, initial, observe


class ActionOptionsTests(unittest.TestCase):
    def test_all_supported_candidates_after_rank_one_are_reachable(self):
        m = initial('no_revision')
        candidates = [3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
        baseline = list(reversed(range(len(candidates))))
        ctx = make_context(m.history, candidates, baseline, .8, m, CARDS)
        options = ctx['summary']['options']
        self.assertEqual([x['item'] for x in options], candidates[1:])
        pages = list(evidence_pages(ctx))
        self.assertEqual([len(p['summary']['options']) for p in pages], [8, 1])
        self.assertEqual([x['item'] for p in pages for x in p['summary']['options']], candidates[1:])
        self.assertEqual(ctx['summary']['baseline_top5'][0]['item'], candidates[0])
        for option in options:
            self.assertGreater(option['rank'], 1)
            self.assertEqual(option['competitor']['item'], candidates[option['rank'] - 2])

    def test_twelve_hypotheses_share_a_page_with_separate_tool_limits(self):
        ids = [f'H{i}' for i in range(1, 13)]
        context = dict(hypotheses=dict.fromkeys(ids),
                       summary=dict(options=[dict(item=1, hypotheses=ids)],
                                    hypotheses=[dict(id=hid) for hid in ids]))
        page, = evidence_pages(context)
        self.assertEqual([h['id'] for h in page['summary']['hypotheses']], ids)
        for requested, items in ((ids[:7], [1]), ([], list(range(1, 10)))):
            with self.subTest(requested=requested, items=items):
                plan = json.dumps(dict(retrieve_history=requested, verify_items=items))
                self.assertEqual(execute_plan(plan, page, {}),
                                 dict(retrieved=[], verified=[], errors=['tool request budget exceeded']))

    def test_only_rank_one_supported_skips_both_agent_calls(self):
        class NoCalls:
            def generate(self, *args, **kwargs):
                raise AssertionError('No executable option should require inference')

        class Reviews:
            def event(self, uid, index, item, sequence):
                return review_event(uid, index, item, (1 + 10 * index) * DAY, 5)

        case = dict(event_id='test:u:2', uid='u', index=2, history=[1, 2], target=3,
                    candidates=[3, 7, 8], baseline=[.3, .2, .1], gate=.8)
        with tempfile.TemporaryDirectory() as tmp:
            rows = run_variant(NoCalls(), [case], CARDS, MODEL, {'u': {}}, Reviews(),
                               'no_revision', Path(tmp),
                               dict(strength=.5, min_confidence='medium'))
            action = json.loads((Path(tmp) / 'actions.jsonl').read_text().strip())
        self.assertFalse(action['plan_called'])
        self.assertFalse(action['decision_called'])
        self.assertEqual(action['context']['options'], [])
        self.assertEqual(rows[0]['rank'], 1)


class EvidenceOwnershipTests(unittest.TestCase):
    def memory_with_two_recent_negatives(self):
        m = ScopedPreferenceMemory(MODEL, 'no_revision')
        for item, rating, day in [(3, 5, 1), (4, 5, 10), (5, 1, 20), (6, 1, 30)]:
            observe(m, item, rating, day)
        return m

    def retrieve_category(self, m):
        ctx = make_context(m.history, [9, 7], [.4, .2], .8, m, CARDS)
        hid = next(hid for hid, h in ctx['hypotheses'].items()
                   if tuple(h['key']) == ('@category', 'hair', '*'))
        obs = execute_plan(json.dumps(dict(retrieve_history=[hid], verify_items=[7])), ctx, CARDS)
        return obs['retrieved'][0]['source_evidence']

    def test_one_category_evidence_record_per_observed_event(self):
        m = self.memory_with_two_recent_negatives()
        claim = m.hypotheses[('@category', 'hair', '*')]
        expected = [e['event_id'] for e in m.events]
        self.assertEqual([e['event_id'] for e in claim['source_evidence']], expected)
        self.assertEqual([e['event_id'] for e in m._category['hair']], expected)

    def test_retrieval_preserves_both_recent_counterexamples(self):
        evidence = self.retrieve_category(self.memory_with_two_recent_negatives())
        self.assertEqual([e['event_id'] for e in evidence], ['u:0', 'u:1', 'u:2', 'u:3'])

    def test_duplicate_legacy_records_do_not_displace_recent_counterexamples(self):
        m = self.memory_with_two_recent_negatives()
        h = m.hypotheses[('@category', 'hair', '*')]
        unique = {e['event_id']: e for e in h['source_evidence']}
        h['source_evidence'] = [copy.deepcopy(unique[eid]) for eid in
                               ['u:0', 'u:1', 'u:1', 'u:2', 'u:2', 'u:3', 'u:3']]
        self.assertEqual([e['event_id'] for e in self.retrieve_category(m)],
                         ['u:0', 'u:1', 'u:2', 'u:3'])

    def test_existing_attribute_claims_receive_new_source_evidence_once(self):
        m = initial('no_revision')
        observe(m, 1, rating=1, day=30)
        m.observe(copy.deepcopy(m.events[-1]))
        for key in (KEY, OTHER, ('@category', 'skin', '*')):
            self.assertEqual([e['event_id'] for e in m.hypotheses[key]['source_evidence']],
                             ['u:0', 'u:1', 'u:2'])
        self.assertGreater(m.assess(KEY)['source']['negative'], 0)


class ActivationTimeTests(unittest.TestCase):
    def tied_memory(self, variant, target_first):
        m = ScopedPreferenceMemory(MODEL, variant)
        observe(m, 1, day=1)
        order = [(3, 1), (2, 5)] if target_first else [(2, 5), (3, 1)]
        for item, rating in order:
            observe(m, item, rating=rating, day=10)
        return m

    def test_same_timestamp_counterevidence_has_same_effect_in_either_order(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                before = self.tied_memory(variant, True)
                after = self.tied_memory(variant, False)
                a, b = before.assess(KEY), after.assess(KEY)
                self.assertAlmostEqual(a['destination']['negative'], .35)
                self.assertEqual(a['destination'], b['destination'])
                self.assertEqual(a['state'], b['state'])
                self.assertAlmostEqual(a['effective_weight'], b['effective_weight'])
                self.assertEqual(before.hypotheses[KEY]['content'], after.hypotheses[KEY]['content'])
                self.assertEqual(len(before.hypotheses[KEY]['target_evidence']), 1)

    def test_older_evidence_excluded_and_future_evidence_added_only_when_observed(self):
        m = ScopedPreferenceMemory(MODEL, 'no_revision')
        observe(m, 1, day=1)
        observe(m, 3, rating=1, day=9)
        observe(m, 2, day=10)
        self.assertEqual(m.hypotheses[KEY]['target_evidence'], [])
        observe(m, 5, day=11)
        self.assertEqual([e['event_id'] for e in m.hypotheses[KEY]['target_evidence']], ['u:3'])

    def test_backfilled_neutral_event_and_later_matching_event_are_each_counted_once(self):
        m = ScopedPreferenceMemory(MODEL, 'no_revision')
        observe(m, 1, day=1)
        observe(m, 7, rating=1, day=10)
        observe(m, 2, day=10)
        self.assertEqual(m.hypotheses[KEY]['uninformative_target_events'], 1)
        self.assertEqual(m.assess(KEY)['destination']['negative'], 0)
        observe(m, 3, rating=1, day=11)
        evidence = m.hypotheses[KEY]['target_evidence']
        self.assertEqual([e['event_id'] for e in evidence], ['u:1', 'u:3'])
        self.assertEqual([e['match'] for e in evidence], [False, True])

    def test_backfilled_withdrawal_stays_permanent_when_recovery_disabled(self):
        for variant in ('no_revision', 'no_local_revision_and_state_recovery'):
            with self.subTest(variant=variant):
                m = ScopedPreferenceMemory(MODEL, variant)
                observe(m, 1, day=1)
                observe(m, 3, rating=1, day=10)
                observe(m, 4, rating=1, day=10)
                observe(m, 2, day=10)
                self.assertEqual(m.assess(KEY)['state'], 'withdrawn')
                observe(m, 5, day=20)
                observe(m, 6, day=30)
                if variant == 'no_local_revision_and_state_recovery':
                    self.assertEqual(m.assess(KEY)['state'], 'withdrawn')
                    self.assertEqual(m.assess(KEY)['effective_weight'], 0)
                else:
                    self.assertGreater(m.assess(KEY)['effective_weight'], 0)


class MultipleClaimTests(unittest.TestCase):
    def setUp(self):
        m = initial('no_adaptive_granularity')
        observe(m, 7)
        observe(m, 8)
        self.ctx = make_context(m.history, [9, 3], [.4, .2], .8, m, CARDS)
        option = next(x for x in self.ctx['summary']['options'] if x['item'] == 3)
        self.hids = option['hypotheses']
        self.obs = execute_plan(json.dumps(dict(retrieve_history=self.hids, verify_items=[3])),
                                self.ctx, CARDS)
        self.transfers = [dict(item=3, hypothesis=hid, competitor=9, confidence='high',
                               reason='independently supported claim') for hid in self.hids]
        weights = {h['id']: h['assessment']['effective_weight'] for h in self.obs['retrieved']}
        self.strong = max(self.hids, key=weights.get)
        self.weak = min(self.hids, key=weights.get)
        self.weights = weights
        self.setting = dict(strength=.5, min_confidence='medium')

    def validate(self, transfers):
        return validate_decision(json.dumps(dict(transfers=transfers)), self.ctx, self.obs)

    def test_both_independent_claims_survive_and_rank_is_order_invariant(self):
        expected = None
        for transfers in permutations(self.transfers):
            result = self.validate(transfers)
            self.assertEqual(result['errors'], [])
            self.assertEqual(len(result['approved']), 2)
            order, execution = bounded_rank([9, 3], [.4, .2], .8, result['approved'], self.setting)
            self.assertEqual(order, [3, 9])
            self.assertAlmostEqual(execution['bonuses'][3], .5 * self.weights[self.strong])
            if expected is None:
                expected = result['approved']
            self.assertEqual(result['approved'], expected)

    def test_low_confidence_claim_cannot_block_another_valid_claim(self):
        transfers = copy.deepcopy(self.transfers)
        for row in transfers:
            row['confidence'] = 'low' if row['hypothesis'] == self.strong else 'high'
        for ordered in permutations(transfers):
            result = self.validate(ordered)
            self.assertEqual(result['errors'], [])
            self.assertEqual([a['hypothesis'] for a in result['approved']], [self.weak])

    def test_duplicate_same_claim_keeps_high_confidence_without_adding_weight(self):
        first = dict(self.transfers[0], confidence='medium', reason='medium rationale')
        second = dict(first, confidence='high', reason='high rationale')
        for ordered in permutations([first, second]):
            result = self.validate(ordered)
            self.assertEqual(result['errors'], [])
            self.assertEqual(len(result['approved']), 1)
            self.assertEqual(result['approved'][0]['confidence'], 'high')

    def test_rejected_claim_does_not_reserve_the_item(self):
        valid = self.transfers[0]
        invalid = dict(valid, competitor=999)
        for ordered in permutations([invalid, valid]):
            result = self.validate(ordered)
            self.assertEqual(len(result['approved']), 1)
            self.assertEqual(result['errors'], ['incorrect first competitor'])

    def test_confidence_filter_happens_before_per_item_maximum(self):
        transfers = copy.deepcopy(self.transfers)
        for row in transfers:
            row['confidence'] = 'medium' if row['hypothesis'] == self.strong else 'high'
        result = self.validate(transfers)
        setting = dict(self.setting, min_confidence='high')
        order, execution = bounded_rank([9, 3], [.4, .2], .8, result['approved'], setting)
        self.assertEqual(order, [9, 3])
        self.assertAlmostEqual(execution['bonuses'][3], .5 * self.weights[self.weak])

    def test_claims_accumulate_one_step_each_and_retries_are_idempotent(self):
        candidates = list(range(1, 21))
        baseline = [0.] * len(candidates)
        for count in (0, 1, 2, 5, 19, 22):
            approvals = [dict(item=20, hypothesis=f'H{i}', ledger_weight=1., confidence='high')
                         for i in range(count)]
            # Duplicate and reverse every approval to simulate a retried response.
            order, trace = bounded_rank(candidates, baseline, .8, approvals + approvals[::-1], self.setting)
            steps = min(count, 19)
            self.assertEqual(order.index(20) + 1, 20 - steps)
            self.assertEqual(trace['max_displacement'], steps)
            self.assertEqual([(c['from_rank'], c['to_rank']) for c in trace['crossings']],
                             [(20 - j, 19 - j) for j in range(steps)])

    def test_displacements_from_other_items_also_accumulate(self):
        approvals = [dict(item=i, hypothesis='H1', ledger_weight=1., confidence='high')
                     for i in (2, 3, 4)]
        order, trace = bounded_rank([1, 2, 3, 4], [0.] * 4, .8, approvals, self.setting)
        self.assertEqual(order, [2, 3, 4, 1])
        self.assertEqual(trace['max_displacement'], 3)

    def test_executor_takes_maximum_for_repeated_item_in_any_order(self):
        approvals = [dict(item=3, ledger_weight=.8, confidence='high'),
                     dict(item=3, ledger_weight=.2, confidence='high')]
        for ordered in permutations(approvals):
            order, execution = bounded_rank([9, 3], [.4, .2], .8, ordered, self.setting)
            self.assertEqual(order, [3, 9])
            self.assertAlmostEqual(execution['bonuses'][3], .4)

    def test_saved_outputs_count_items_separately_and_pass_rank_audit(self):
        class Reviews:
            def event(self, uid, index, item, sequence):
                return review_event(uid, index, item, (1 + 10 * index) * DAY, 5)

        class Qwen:
            def __init__(self, reverse):
                self.reverse = reverse
                self.visible = []

            def generate(self, requests, path, *args):
                rows = []
                for request in requests:
                    prompt = request['prompt']
                    if '\nINPUT:\n' in prompt:
                        ctx = json.loads(prompt.split('\nINPUT:\n')[1])
                        option = next(x for x in ctx['options'] if x['item'] == 3)
                        self.visible.append(option['hypotheses'])
                        result = dict(retrieve_history=option['hypotheses'][-6:], verify_items=[3])
                    else:
                        summary, tools = prompt.split('\nCONTEXT:\n')[1].split('\nTOOL_RESULTS:\n')
                        option = next(x for x in json.loads(summary)['options'] if x['item'] == 3)
                        transfers = [dict(item=3, hypothesis=h['id'], competitor=option['competitor']['item'],
                                          confidence='high', reason='independent evidence')
                                     for h in json.loads(tools)['retrieved']]
                        result = dict(transfers=transfers[::-1] if self.reverse else transfers)
                    rows.append(dict(**request, response=json.dumps(result)))
                return rows

        # The planner chooses six claims from each of the two batches for this candidate.
        attributes = [f'attribute{i}' for i in range(19)]
        model = dict(categories={str(i): 'skin' if i < 3 else 'hair' for i in range(1, 23)},
                     attributes={str(i): attributes if i <= 3 else [] for i in range(1, 23)},
                     boundaries=[dict(source='skin', destination='hair', attribute=a, weight=.8)
                                 for a in attributes])
        cards = {int(i): dict(title=f'item {i}', tokens=values, categories=[model['categories'][i]])
                 for i, values in model['attributes'].items()}
        candidates = list(range(4, 23)) + [3]
        case = dict(event_id='test:u:2', uid='u', index=2, history=[1, 2],
                    target=3, candidates=candidates, baseline=[.001] * 19 + [0.], gate=.8)
        for reverse in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                qwen = Qwen(reverse)
                rows = run_variant(qwen, [case], cards, model, {'u': {}}, Reviews(),
                                   'full', folder, self.setting)
                action = json.loads((folder / 'actions.jsonl').read_text().strip())
                execution = json.loads((folder / 'ranking_executions.jsonl').read_text().strip())
            self.assertEqual(rows[0]['n_approved'], 1)
            self.assertEqual([len(page) for page in qwen.visible], [12, 7])
            self.assertEqual(len({hid for page in qwen.visible for hid in page}), 19)
            selected = {hid for page in qwen.visible for hid in page[-6:]}
            self.assertEqual({row['hypothesis'] for row in action['approved']}, selected)
            self.assertEqual(rows[0]['n_approved_claims'], 12)
            self.assertEqual(len(action['plan_call_ids']), 2)
            self.assertEqual(len(action['decision_call_ids']), 2)
            self.assertEqual(rows[0]['rank'], 8)
            order = reconstruct_order(case, action, rows[0], execution, self.setting)
            self.assertEqual(order[:10], rows[0]['top10'])
            execution['crossings'][1]['hypothesis'] = execution['crossings'][0]['hypothesis']
            with self.assertRaises(AssertionError):
                reconstruct_order(case, action, rows[0], execution, self.setting)


if __name__ == '__main__':
    unittest.main(verbosity=2)
