"""Controlled mechanism checks, not recommendation-quality measurements."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from full_evaluation import METRICS, metric_table, run_variant
from ledger import review_event
from preference_graph import ScopedPreferenceMemory, VARIANTS
from recommendation_agent import make_context, plan_prompt, execute_plan, validate_decision

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
        self.assertEqual(set(VARIANTS), {'no_adaptive_granularity','no_revision','no_recovery','no_both'})
        self.assertFalse(any(vars(VARIANTS['no_both']).values()))

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
                self.assertGreater(m.assess(KEY)['effective_weight'], 0)
            else:
                self.assertEqual(m.assess(KEY)['state'], 'withdrawn')
                self.assertEqual(m.assess(KEY)['effective_weight'], 0)
                self.assertIn(KEY, m.permanently_withdrawn)
            # Target category now has four visits; there is no cold-only gate.
            self.assertEqual(m.visits['hair'], 4)
            if VARIANTS[v].recovery: self.assertGreater(m.score([5])[0][0], 0)

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

    def test_four_variants_execute_independent_agent_calls(self):
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
        case=dict(event_id='test:u:2',uid='u',index=2,history=[1,2],target=3,
            candidates=list(range(1,21)),baseline=list(reversed(range(20))),gate=.8)
        qwen=Qwen()
        with tempfile.TemporaryDirectory() as tmp:
            for v in VARIANTS:
                run_variant(qwen,[case],CARDS,MODEL,{'u':{}},Reviews(),v,Path(tmp)/v,
                    dict(strength=.5,min_confidence='medium'))
                action=json.loads((Path(tmp)/v/'actions.jsonl').read_text().splitlines()[0])
                self.assertEqual(action['pre_observation']['events'],2)
                self.assertEqual(action['post_observation']['events'],3)
                self.assertEqual(action['pre_observation']['variant'],v)
                self.assertEqual(action['context']['history_length'],2)
            for v in VARIANTS:
                self.assertTrue(any(str(Path(tmp)/v/'plan_calls.jsonl')==p for p in qwen.calls))

if __name__ == '__main__': unittest.main(verbosity=2)
