"""Regression checks for the protocol described in the manuscript."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from data_split import audit_events, cuts, make_events, training_rows
from baseline_source import SASRec, right_pad
from full_evaluation import run_variant, sha
from ledger import fit_boundaries, review_event


def sequence(n):
    return dict(items=list(range(1, n + 1)), times=list(range(n)), row_ids=list(range(n)))


class LeaveOneOutTests(unittest.TestCase):
    def test_training_validation_and_test_positions(self):
        seqs = {str(n): sequence(n) for n in (3, 5, 6, 9, 10, 71)}
        model = SASRec(119).eval().requires_grad_(False)
        valid, counts = make_events(seqs, range(1, 120), 'valid', model=model)
        test, _ = make_events(seqs, range(1, 120), 'test', model=model)
        self.assertEqual(counts['valid_events'], len(seqs))
        self.assertEqual(counts['test_events'], len(seqs))
        audit_events(valid, seqs, range(1, 120), 'valid')
        audit_events(test, seqs, range(1, 120), 'test')
        for v, t in zip(valid, test):
            items = seqs[v['uid']]['items']
            self.assertEqual(v['target'], items[-2])
            self.assertEqual(t['target'], items[-1])
            self.assertEqual(t['history'], v['history'] + [v['target']])
        rows = list(training_rows(seqs))
        self.assertEqual(len(rows), sum(n - 3 for n in (3, 5, 6, 9, 10, 71)))
        for uid, history, target in rows:
            self.assertEqual(history + [target], seqs[uid]['items'][:len(history) + 1])
            self.assertLess(len(history), len(seqs[uid]['items']) - 2)
        with self.assertRaises(ValueError): cuts(2)

    def test_shuffled_candidates_keep_their_unseen_top20_scores(self):
        seqs = {str(n): sequence(n) for n in range(5, 35)}
        catalog = range(1, 100)
        model = SASRec(len(catalog)).eval().requires_grad_(False)
        events, _ = make_events(seqs, catalog, model=model)
        again, _ = make_events(dict(reversed(list(seqs.items()))), reversed(catalog), model=model)
        self.assertEqual(events, again)
        histories = torch.tensor([right_pad(e['history']) for e in events])
        with torch.inference_mode():
            scores = model.logits(histories).tolist()
        shuffled = False
        for event, logits in zip(events, scores):
            eligible = set(catalog).difference(event['history'])
            expected = sorted(eligible, key=lambda item: (-logits[item], item))[:20]
            self.assertCountEqual(event['candidates'], expected)
            for item, score in zip(event['candidates'], event['sasrec_scores']):
                self.assertAlmostEqual(score, logits[item], places=4)
            shuffled |= event['candidates'] != expected
        self.assertTrue(shuffled)
        broken = copy.deepcopy(events); broken[0]['history'] = broken[0]['history'][:-1]
        with self.assertRaises(AssertionError): audit_events(broken, seqs, range(1, 100))
        with self.assertRaises(ValueError):
            make_events({'u': sequence(5)}, range(1, 20), model=model)

    def test_catalog_boundary_tied_scores_seen_items_and_retrieval_misses(self):
        from audit_ablation_outputs import metric_table as independent_metrics
        from candidate_audit import verify_phase
        from ranking_metrics import metric_table, target_rank

        class TiedModel(torch.nn.Module):
            def logits(self, seq):
                scores = torch.zeros((len(seq), 24), device=seq.device)
                scores[:, 0] = 100.  # Padding would otherwise win.
                return scores

        model = TiedModel().eval().requires_grad_(False)
        sequences = {str(index): dict(items=[1, 2, target], times=[0, 1, 2], row_ids=[0, 1, 2])
                     for index, target in enumerate((1, 3, 22, 23))}
        events, _ = make_events(sequences, range(1, 24), model=model, batch=2)
        for event in events:
            self.assertCountEqual(event['candidates'], range(3, 23))
        self.assertEqual(events, make_events(sequences, range(1, 24), model=model, batch=1)[0])
        # All logits tie. SASRec's metric uses ascending item ID, independent of presentation.
        ranks = [target_rank(sorted(e['candidates']), e['target']) for e in events]
        self.assertEqual(ranks, [None, 1, 20, None])
        table = metric_table([dict(rank=r) for r in ranks])
        self.assertEqual(table, independent_metrics(ranks))
        self.assertTrue(all(v == 0.25 for v in table.values()))
        self.assertTrue(all(v == 0 for v in metric_table([dict(rank=None)]).values()))
        audit = verify_phase(model, events, sequences, 23, 'cpu')
        self.assertEqual((audit['retrieved_targets'], audit['missed_targets']), (2, 2))
        self.assertEqual(audit['sasrec_validation_or_test_ndcg10'], table['NDCG@10'])
        self.assertEqual(audit['target_position_histogram'], {1: 1, 20: 1})
        broken = copy.deepcopy(events)
        broken[-1]['candidates'][-1] = broken[-1]['target']
        with self.assertRaises(ValueError):
            verify_phase(model, broken, sequences, 23, 'cpu')
        broken = copy.deepcopy(events)
        broken[0]['sasrec_scores'][0] += 1
        with self.assertRaisesRegex(ValueError, 'scores differ'):
            verify_phase(model, broken, sequences, 23, 'cpu')
        with self.assertRaisesRegex(ValueError, 'Fewer than 20 unseen'):
            make_events({'u': sequence(5)}, range(1, 24), model=model)

    def test_exclusion_sees_beyond_encoder_window_and_never_uses_future_labels(self):
        class TiedModel(torch.nn.Module):
            def logits(self, seq):
                return torch.zeros((len(seq), 81))

        model = TiedModel().eval().requires_grad_(False)
        history = [1] + [2] * 52
        seqs = {'u': dict(items=history + [4, 5], times=list(range(55)), row_ids=list(range(55)))}
        def event(phase):
            return make_events(seqs, range(1, 81), phase, model=model)[0][0]
        valid, test = event('valid'), event('test')
        self.assertCountEqual(valid['candidates'], range(3, 23))
        self.assertIn(5, valid['candidates'])  # Future test item is eligible in validation.
        self.assertCountEqual(test['candidates'], [3] + list(range(5, 24)))
        seqs['u']['items'][-1] = 70
        changed = event('test')
        self.assertEqual(changed['candidates'], test['candidates'])
        self.assertEqual(changed['sasrec_scores'], test['sasrec_scores'])

    def test_encoder_truncates_but_memory_observes_complete_history(self):
        items = list(range(1, 73)); history = items[:-1]
        self.assertEqual(right_pad(history), list(range(22, 72)))
        self.assertEqual(right_pad([1, 2])[:4], [1, 2, 0, 0])
        cards = {i: dict(title=f'item {i}', categories=['skin'], tokens=[]) for i in range(1, 100)}
        boundary = dict(categories={str(i): 'skin' for i in cards},
                        attributes={str(i): [] for i in cards}, boundaries=[])
        class Reviews:
            def event(self, uid, index, item, seq):
                return review_event(uid, index, item, index + 1, 5)
        class EmptyPlans:
            def generate(self, requests, *args, **kwargs):
                return [dict(id=r['id'], response='{"retrieve_history":[],"verify_items":[]}')
                        for r in requests]
        case = dict(event_id='test:u:71', uid='u', index=71, phase='test', history=history,
            target=72, candidates=list(range(72, 92)), baseline=list(range(20)), gate=.5)
        with tempfile.TemporaryDirectory() as tmp:
            run_variant(EmptyPlans(), [case], cards, boundary, {'u': {}}, Reviews(), 'full', tmp,
                dict(strength=.5, min_confidence='medium'))
            action = json.loads((Path(tmp) / 'actions.jsonl').read_text())
            self.assertEqual(action['pre_observation']['events'], 71)
            self.assertEqual(action['context']['history_length'], 71)
            self.assertEqual(action['post_observation']['events'], 72)

    def test_population_prior_does_not_fit_heldout_interactions(self):
        cards = {i: dict(tokens=[f'attribute{i}']) for i in range(1, 8)}
        categories = {i: 'skin' for i in cards}
        first = {'u': sequence(5)}
        changed = copy.deepcopy(first); changed['u']['items'][-2:] = [6, 7]
        fitted = fit_boundaries(cards, categories, first, set())
        self.assertEqual(fitted['audit']['training_events'], 3)
        self.assertEqual(fitted, fit_boundaries(cards, categories, changed, set()))

    def test_test_scoring_does_not_update_model_or_drop_full_history(self):
        from run_ledger_gate import load_baseline
        cards = {i: dict(categories=['skin'], tokens=[]) for i in range(1, 101)}
        model = SASRec(len(cards)).eval().requires_grad_(False)
        events, _ = make_events({'u': sequence(71)}, cards, model=model)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            torch.save(model.state_dict(), path)
            before = sha(path)
            with patch.object(torch.optim.AdamW, 'step', side_effect=AssertionError('Testing must not train')):
                _, cases = load_baseline(cards, events, states_override={events[0]['event_id']: {}},
                                         checkpoint=path, device='cpu')
            self.assertEqual(sha(path), before)
            self.assertEqual(len(cases[0]['history']), 70)
            self.assertEqual(cases[0]['history'][-1], 70)
            self.assertEqual(len(cases[0]['baseline']), 20)


if __name__ == '__main__': unittest.main()
