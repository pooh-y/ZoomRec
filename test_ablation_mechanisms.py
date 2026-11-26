"""Controlled mechanism checks, not recommendation-quality measurements."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from full_evaluation import METRICS, metric_table, run_variant
from ledger import review_event
from audit_ablation_outputs import check_mechanisms, reconstruct_order
from preference_graph import ALL_VARIANTS, ScopedPreferenceMemory, VARIANTS
from recommendation_agent import bounded_rank, make_context, plan_prompt, execute_plan, validate_decision

DAY = 86400
MODEL = dict(categories={str(i): ('skin' if i < 3 else 'hair' if i < 9 else 'nails')
    for i in range(1, 21)}, attributes={str(i): ([] if i in (7, 8) else ['travel']) for i in range(1, 21)},
    boundaries=[dict(source='skin', destination=dst, attribute='travel', weight=.8)
                for dst in ('hair', 'nails')])
CARDS = {i: dict(title=f'item {i} travel', tokens=[] if i in (7, 8) else ['travel'],
                categories=[MODEL['categories'][str(i)]]) for i in range(1, 21)}
KEY = ('skin', 'hair', 'travel')
OTHER = ('skin', 'nails', 'travel')

def observe(m, item, rating=5, day=None):
    j = len(m.events)
    m.observe(review_event('u', j, item, (day if day is not None else 1 + 10*j)*DAY, rating))

def initial(variant):
    m = ScopedPreferenceMemory(MODEL, variant)
    observe(m, 1); observe(m, 2)
    return m

class MechanismTests(unittest.TestCase):
    def test_exact_four_configurations(self):
        disabled = {name: {k for k, v in vars(flags).items() if not v} for name, flags in VARIANTS.items()}
        self.assertEqual(disabled, {
            'no_adaptive_granularity': {'adaptive_granularity'},
            'no_local_reopening': {'local_reopening'},
            'no_local_revision_and_state_recovery': {'local_revision', 'local_reopening', 'recovery'},
            'no_agent_review': {'agent_review'},
        })
        self.assertEqual(ALL_VARIANTS['no_recovery'], VARIANTS['no_local_reopening'])
        config = Path(__file__).resolve().parent / 'configs' / 'paper_ablations.json'
        self.assertEqual(json.loads(config.read_text())['requested_variants'], ['full', *VARIANTS])

    def test_local_revision_and_no_revision(self):
        for v in VARIANTS:
            m = initial(v); before = copy.deepcopy(m.hypotheses[KEY]['content'])
            unrelated = copy.deepcopy(m.hypotheses[OTHER]['content'])
            observe(m, 3, 1)
            self.assertEqual(m.hypotheses[OTHER]['content'], unrelated)
            self.assertNotEqual(m.assess(KEY)['state'], 'withdrawn')
            if VARIANTS[v].local_revision:
                self.assertNotEqual(m.hypotheses[KEY]['content'], before)
                self.assertNotIn('hair', m.hypotheses[KEY]['content']['applicable_categories'])
            else:
                self.assertEqual(m.hypotheses[KEY]['content'], before)
                self.assertFalse(m.revisions)

    def test_permanent_preference_withdrawal_survives_replay(self):
        for v in VARIANTS:
            m = initial(v); observe(m, 3, 1); observe(m, 4, 1)
            self.assertEqual(m.assess(KEY)['state'], 'withdrawn')
            observe(m, 5); observe(m, 6)
            replay = ScopedPreferenceMemory(MODEL, v)
            for event in m.events: replay.observe(event)
            self.assertEqual(replay.trace(), m.trace())
            if VARIANTS[v].recovery:
                self.assertNotEqual(m.assess(KEY)['state'], 'withdrawn')
                self.assertEqual(m.assess(KEY)['effective_weight'] > 0, VARIANTS[v].local_reopening)
            else:
                self.assertEqual(m.assess(KEY)['state'], 'withdrawn')
                self.assertEqual(m.assess(KEY)['effective_weight'], 0)
                self.assertIn(KEY, m.permanently_withdrawn)
            # Target category now has four visits; there is no cold-only gate.
            self.assertEqual(m.visits['hair'], 4)
            if VARIANTS[v].recovery: self.assertGreater(m.score([5])[0][0], 0)

    def test_local_reopening_is_independent_of_state_recovery(self):
        for variant in ('full', 'no_local_reopening', 'no_recovery'):
            m = initial(variant)
            observe(m, 3, 1)
            self.assertNotEqual(m.assess(KEY)['state'], 'withdrawn')
            self.assertFalse(m.assess(KEY)['content_allows_destination'])
            observe(m, 5); observe(m, 5)
            self.assertFalse(m.assess(KEY)['content_allows_destination'])
            observe(m, 6)
            self.assertEqual(m.assess(KEY)['state'], 'confirmed')
            self.assertEqual(m.assess(KEY)['content_allows_destination'], variant == 'full')
            self.assertEqual(m.assess(KEY)['effective_weight'] > 0, variant == 'full')
            self.assertGreater(m.assess(OTHER)['effective_weight'], 0.)
            self.assertGreater(m.assess(KEY)['destination']['negative'], 0.)

    def test_no_inferred_ancestor_and_missing_token_is_neutral(self):
        m = initial('no_recovery')
        self.assertNotIn(('@category','hair','*'), m.hypotheses)
        observe(m, 7)
        self.assertEqual(m.assess(KEY)['destination']['negative'], 0)
        self.assertEqual(m.assess(KEY)['destination']['distinct_positive'], 0)
        self.assertEqual(m.hypotheses[KEY]['uninformative_target_events'], 1)
        self.assertEqual(m.assess(('@category','hair','*'))['effective_weight'], 0)
        observe(m, 8)
        self.assertGreater(m.assess(('@category','hair','*'))['effective_weight'], 0)

    def test_fixed_order_has_no_evidence_level_selector(self):
        for v in VARIANTS:
            m = ScopedPreferenceMemory(MODEL, v)
            observe(m,1,day=1); observe(m,2,day=2)
            # Two source items within one episode do not authorize adaptive detail.
            scores, _ = m.score([3])
            self.assertEqual(scores[0] > 0, not VARIANTS[v].adaptive_granularity)
        m = initial('no_adaptive_granularity'); observe(m,7); observe(m,8)
        levels = [m.hypotheses[tuple(r['key'])]['level'] for r in m.score([5])[1][0]['attributes'].values()]
        self.assertEqual(levels[0], 'category'); self.assertIn('attribute', levels)

    def test_prompt_preserves_display_order_and_score_based_competitors(self):
        m = initial('no_revision')
        c1 = make_context([1,2], [3,7,9], [.2,.3,.1], .8, m, CARDS)
        c2 = make_context([1,2], [9,7,3], [.1,.3,.2], .8, m, CARDS)
        options1, options2 = (ctx['summary']['options'] for ctx in (c1, c2))
        self.assertEqual([row['item'] for row in options1], [3, 9])
        self.assertEqual([row['item'] for row in options2], [9, 3])
        for options in (options1, options2):
            self.assertEqual({row['item']: row['competitor']['item'] for row in options}, {3: 7, 9: 3})
        self.assertNotIn('"target"', plan_prompt(c1))

    def test_duplicate_item_and_event_do_not_create_support(self):
        m = ScopedPreferenceMemory(MODEL,'no_both')
        observe(m,1); m.observe(copy.deepcopy(m.events[0])); observe(m,1)
        self.assertEqual(len(m.events), 2); self.assertNotIn(KEY,m.hypotheses)

    def test_eight_metrics_known_ranks(self):
        table = metric_table([dict(rank=r) for r in (1,2,4,11)])
        self.assertEqual(list(table), METRICS)
        self.assertEqual(table['HR@1'], .25); self.assertEqual(table['HR@3'], .5)
        self.assertEqual(table['HR@10'], .75); self.assertEqual(table['NDCG@1'], .25)
        self.assertAlmostEqual(table['NDCG@5'], (1+1/math.log2(3)+1/math.log2(5))/4)

    def test_real_validation_accepts_independent_category_and_attribute_claims(self):
        m=initial('no_revision'); observe(m,7); observe(m,8)
        ctx=make_context(m.history,[9,3,7],[.4,.2,.1],.8,m,CARDS)
        for item in (3,7):
            option=next(x for x in ctx['summary']['options'] if x['item']==item)
            hid=option['hypotheses'][0]
            obs=execute_plan(json.dumps(dict(retrieve_history=[hid],verify_items=[item])),ctx,CARDS)
            decision=dict(transfers=[dict(item=item,hypothesis=hid,confidence='high',
                competitor=option['competitor']['item'],reason='two independent supportive items')])
            approved=validate_decision(json.dumps(decision),ctx,obs)
            self.assertEqual(approved['errors'],[])
            self.assertEqual(len(approved['approved']),1)

    def test_only_reviewed_variants_call_the_agent(self):
        class Qwen:
            def __init__(self): self.calls = []
            def generate(self, requests, path, *args, **kwargs):
                self.calls.append(str(path))
                result=[]
                for request in requests:
                    prompt=request['prompt']
                    if '\nINPUT:\n' in prompt:
                        ctx=json.loads(prompt.split('\nINPUT:\n')[1])
                        raw=dict(retrieve_history=[h['id'] for h in ctx['hypotheses'][:2]],
                            verify_items=[r['item'] for r in ctx['options'][:2]])
                    else: raw=dict(transfers=[])
                    result.append(dict(**request,response=json.dumps(raw)))
                return result
        class Reviews:
            def event(self,uid,index,item,sequence): return review_event(uid,index,item,(1+10*index)*DAY,5)
        candidates = [7, 3] + [i for i in range(1, 21) if i not in (7, 3)]
        case=dict(event_id='test:u:2',uid='u',index=2,history=[1,2],target=3,
            candidates=candidates,baseline=[1., .999] + [-float(i) for i in range(18)],gate=.8)
        qwen=Qwen()
        with tempfile.TemporaryDirectory() as tmp:
            for v in VARIANTS:
                setting = dict(strength=.5, min_confidence='medium')
                predictions = run_variant(qwen,[case],CARDS,MODEL,{'u':{}},Reviews(),v,Path(tmp)/v,setting)
                action=json.loads((Path(tmp)/v/'actions.jsonl').read_text().splitlines()[0])
                execution=json.loads((Path(tmp)/v/'ranking_executions.jsonl').read_text().splitlines()[0])
                check_mechanisms(action, v)
                order=reconstruct_order(case, action, predictions[0], execution, setting)
                self.assertEqual(order[:10], predictions[0]['top10'])
                if v == 'no_agent_review':
                    self.assertEqual(order[0], 3)
                    self.assertTrue(action['approved'])
                    self.assertFalse(action['decision_called'])
                    self.assertFalse(any('confidence' in row for row in action['approved']))
                self.assertEqual(action['pre_observation']['events'],2)
                self.assertEqual(action['post_observation']['events'],3)
                self.assertEqual(action['pre_observation']['variant'],v)
                self.assertEqual(action['context']['history_length'],2)
            for v in VARIANTS:
                self.assertEqual(str(Path(tmp)/v/'plan_calls.jsonl') in qwen.calls, VARIANTS[v].agent_review)

    def test_strict_swap_and_auditor_reject_ties_for_both_id_orders(self):
        setting = dict(strength=.5, min_confidence='medium')
        for neighbor, item in [(2, 1), (1, 2)]:
            candidates = [neighbor, item, *range(3, 21)]
            scores = [1., .75] + [-float(i) for i in range(18)]
            claim = dict(item=item, hypothesis='H1', ledger_weight=.5, confidence='high')
            case = dict(candidates=candidates, baseline=scores)
            action = dict(method='full', approved=[claim])
            prediction = dict(n_approved=1, n_approved_claims=1, ranking_changed=False)
            order, trace = bounded_rank(candidates, scores, .8, [claim], setting)
            self.assertEqual(order, candidates)
            self.assertEqual(reconstruct_order(case, action, prediction, trace, setting), candidates)
            fake = dict(promoted=item, displaced=neighbor, hypothesis='H1', from_rank=2,
                        to_rank=1, original_score_gap=.25, bonus=.25)
            with self.assertRaises(AssertionError):
                reconstruct_order(case, action, dict(prediction, ranking_changed=True),
                                  dict(trace, crossings=[fake], max_displacement=1), setting)
            scores[0] = math.nextafter(1., 0.)
            self.assertEqual(bounded_rank(candidates, scores, .8, [claim], setting)[0][0], item)

    def test_nonadaptive_variant_uses_only_the_strongest_verified_match(self):
        claims = [dict(item=3, hypothesis=hid, ledger_weight=weight, confidence='high')
                  for hid, weight in [('H1', .5), ('H2', 1.)]]
        args = ([1, 2, 3], [.375, .25, .125], .8, claims, dict(strength=.5, min_confidence='medium'))
        self.assertEqual(bounded_rank(*args)[0], [3, 1, 2])
        order, trace = bounded_rank(*args, strongest_only=True)
        self.assertEqual(order, [1, 3, 2])
        self.assertEqual([row['hypothesis'] for row in trace['crossings']], ['H2'])

    def test_rule_only_bonus_applies_to_the_first_candidate_reference_score(self):
        class Reviews:
            def event(self, uid, index, item, sequence):
                return review_event(uid, index, item, (1 + 10 * index) * DAY, 5)

        candidates = [3, 9] + [i for i in range(1, 21) if i not in (3, 9)]
        case = dict(event_id='test:u:2', uid='u', index=2, history=[1, 2], target=9,
                    candidates=candidates, baseline=[1., .999] + [-float(i) for i in range(18)], gate=.8)
        setting = dict(strength=.5, min_confidence='medium')
        with tempfile.TemporaryDirectory() as folder:
            predictions = run_variant(None, [case], CARDS, MODEL, {'u': {}}, Reviews(),
                                      'no_agent_review', folder, setting)
            action = json.loads((Path(folder) / 'actions.jsonl').read_text())
            execution = json.loads((Path(folder) / 'ranking_executions.jsonl').read_text())
        self.assertEqual(predictions[0]['top10'][:2], [3, 9])
        self.assertGreater(execution['bonuses']['3'], 0.)
        self.assertEqual(execution['bonuses']['3'], execution['bonuses']['9'])
        check_mechanisms(action, 'no_agent_review')
        self.assertEqual(reconstruct_order(case, action, predictions[0], execution, setting)[:2], [3, 9])

if __name__ == '__main__': unittest.main(verbosity=2)
