"""Verify that the matched full method exercises all three mechanisms."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from audit_ablation_outputs import check_mechanisms, check_trace, reconstruct_order
from full_evaluation import run_variant
from ledger import review_event
from preference_graph import ALL_VARIANTS, VARIANTS, ScopedPreferenceMemory
from test_ablation_mechanisms import MODEL, CARDS, KEY, DAY, initial, observe


class FullMethodTests(unittest.TestCase):
    def test_full_is_enabled_without_changing_four_ablation_definitions(self):
        self.assertTrue(all(vars(ALL_VARIANTS['full']).values()))
        self.assertNotIn('full', VARIANTS)
        self.assertEqual(len(VARIANTS), 4)

    def test_full_adaptive_gate_requires_independent_time_periods(self):
        memory = ScopedPreferenceMemory(MODEL, 'full')
        observe(memory, 1, day=1)
        observe(memory, 2, day=2)
        self.assertEqual(memory.score([3])[0][0], 0)
        self.assertGreater(initial('full').score([3])[0][0], 0)

    def test_full_revises_withdraws_and_recovers_then_replays_identically(self):
        memory = initial('full')
        original = copy.deepcopy(memory.hypotheses[KEY]['content'])
        observe(memory, 3, rating=1)
        self.assertNotEqual(memory.hypotheses[KEY]['content'], original)
        self.assertNotIn('hair', memory.hypotheses[KEY]['content']['applicable_categories'])
        observe(memory, 4, rating=1)
        self.assertEqual(memory.assess(KEY)['state'], 'withdrawn')
        observe(memory, 5)
        observe(memory, 6)
        self.assertNotEqual(memory.assess(KEY)['state'], 'withdrawn')
        self.assertIn('hair', memory.hypotheses[KEY]['content']['applicable_categories'])
        self.assertGreater(memory.assess(KEY)['effective_weight'], 0)
        self.assertTrue(memory.revisions)
        self.assertFalse(memory.permanently_withdrawn)
        replay = ScopedPreferenceMemory(MODEL, 'full')
        for event in memory.events:
            replay.observe(event)
        self.assertEqual(replay.trace(), memory.trace())

    def test_full_executes_agent_and_saved_trace_passes_audit(self):
        class Qwen:
            def __init__(self):
                self.calls = []

            def generate(self, requests, path, *args, **kwargs):
                self.calls.append(Path(path).name)
                result = []
                for request in requests:
                    if '\nINPUT:\n' in request['prompt']:
                        ctx = json.loads(request['prompt'].split('\nINPUT:\n')[1])
                        raw = dict(retrieve_history=[h['id'] for h in ctx['hypotheses'][:2]],
                                   verify_items=[r['item'] for r in ctx['options'][:2]])
                    else:
                        raw = dict(transfers=[])
                    result.append(dict(**request, response=json.dumps(raw)))
                return result

        class Reviews:
            def event(self, uid, index, item, sequence):
                return review_event(uid, index, item, (1 + 10 * index) * DAY, 5)

        case = dict(event_id='test:u:2', uid='u', index=2, history=[1, 2], target=3,
                    candidates=list(range(1, 21)), baseline=list(reversed(range(20))), gate=.8)
        setting = dict(strength=.5, min_confidence='medium')
        qwen = Qwen()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            predictions = run_variant(qwen, [case], CARDS, MODEL, {'u': {}}, Reviews(),
                                      'full', folder, setting)
            action = json.loads((folder / 'actions.jsonl').read_text().strip())
            execution = json.loads((folder / 'ranking_executions.jsonl').read_text().strip())
        self.assertEqual(qwen.calls, ['plan_calls.jsonl', 'decision_calls.jsonl'])
        self.assertEqual(predictions[0]['method'], 'full')
        check_trace(action, case)
        check_mechanisms(action, 'full')
        reconstruct_order(case, action, predictions[0], execution, setting)


if __name__ == '__main__':
    unittest.main()
