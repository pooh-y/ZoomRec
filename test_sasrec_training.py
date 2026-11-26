"""Guard checkpoint selection, validation isolation, and the manuscript's training rule."""
import contextlib
import copy
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import baseline_source as baseline
from full_evaluation import verify_manifest
from training_protocol import EarlyStopping, training_policy, validate_training_record


class TrainingProtocolTests(unittest.TestCase):
    def test_ties_count_and_strict_improvement_resets_patience(self):
        stop = EarlyStopping()
        scores = [.3, .3, .2, .4, .4, .39, .38, .37, .36]
        history = [stop.update(5 * (i + 1), score) for i, score in enumerate(scores)]
        self.assertEqual(history[1]['non_improving_evaluations'], 1)
        self.assertEqual(history[3]['non_improving_evaluations'], 0)
        self.assertFalse(any(row['stop'] for row in history[:-1]))
        self.assertTrue(history[-1]['stop'])
        self.assertEqual(stop.best_epoch, 20)
        self.assertEqual(stop.bad_evaluations, 5)

    def test_minimum_duration_is_respected(self):
        stop = EarlyStopping(min_epochs=40)
        rows = [stop.update(ep, .5) for ep in range(5, 41, 5)]
        self.assertFalse(rows[5]['stop'])  # patience reached at 30, minimum not yet reached
        self.assertTrue(rows[-1]['stop'])
        with self.assertRaises(ValueError): training_policy(19)

    def test_cadence_and_nonfinite_metrics_are_rejected(self):
        stop = EarlyStopping()
        for epoch, score in [(1, .5), (10, .5), (5, math.nan), (5, math.inf), (5, -1)]:
            with self.subTest(epoch=epoch, score=score), self.assertRaises(ValueError):
                stop.update(epoch, score)
        self.assertTrue(stop.update(5, .5)['improved'])

    def test_old_four_epoch_manifest_cannot_be_reused(self):
        with self.assertRaisesRegex(ValueError, 'Old fixed-epoch model'):
            verify_manifest(dict(protocol='leave-one-out-v1'))

    def test_sasrec_architecture_and_causal_attention(self):
        model = baseline.SASRec(30).eval()
        self.assertEqual(model.item.embedding_dim, 64)
        self.assertEqual(model.pos.num_embeddings, 50)
        self.assertEqual(len(model.encoder.layers), 2)
        for layer in model.encoder.layers:
            self.assertEqual(layer.self_attn.num_heads, 4)
            self.assertEqual(layer.self_attn.dropout, .1)
            self.assertEqual(layer.dropout.p, .1)
        states = []
        handle = model.encoder.register_forward_hook(lambda module, args, out: states.append(out.detach().clone()))
        with torch.inference_mode():
            model.encode(torch.tensor([[1, 2, 3, 4]]))
            model.encode(torch.tensor([[1, 2, 9, 10]]))
        handle.remove()
        torch.testing.assert_close(states[0][:, :2], states[1][:, :2], rtol=0, atol=0)

    def test_validation_ndcg_candidate_order_and_ties(self):
        class TiedModel(torch.nn.Module):
            def logits(self, seq):
                assert not self.training and torch.is_inference_mode_enabled()
                return torch.zeros((len(seq), 22))
        events = [dict(uid=str(i), phase='valid', history=[2], target=target) for i, target in enumerate((2, 3, 21))]
        tensors = baseline.validation_tensors(events, 'cpu')
        score = baseline.validation_ndcg10(TiedModel(), tensors, batch=1)
        self.assertAlmostEqual(score, 1 / math.log2(3) / 3)
        events[0]['phase'] = 'test'
        with self.assertRaisesRegex(ValueError, 'validation queries'):
            baseline.validation_tensors(events, 'cpu')

    def test_training_restores_best_weights_and_never_reads_test(self):
        # A real optimizer loop with a controlled validation curve makes the last
        # epoch worse than epoch 15. Regressions that save the final weights fail.
        scores = iter([.6, .5, .8, .8, .7, .6, .5, .4, .8])
        observed_states = []
        def validate(model, tensors, batch):
            observed_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return next(scores)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); freeze = root / 'freeze'; freeze.mkdir()
            baseline.dump(freeze / 'item_map.json', {str(i): i for i in range(1, 25)})
            baseline.write_jsonl(freeze / 'train.jsonl', [['u', [1], 2], ['u', [1, 2], 3]])
            baseline.write_jsonl(freeze / 'valid_queries.jsonl', [dict(uid='u', phase='valid',
                history=[1, 2, 3], target=4)])
            (freeze / 'valid.jsonl').write_text('PRECOMPUTED CANDIDATES MUST NEVER BE OPENED')
            (freeze / 'test.jsonl').write_text('TEST DATA MUST NEVER BE OPENED')
            output = root / 'model'
            real_ce = torch.nn.functional.cross_entropy
            real_adam = torch.optim.AdamW
            real_clip = torch.nn.utils.clip_grad_norm_
            with patch.object(baseline, 'validation_ndcg10', side_effect=validate), \
                 patch.object(baseline, 'read_jsonl', wraps=baseline.read_jsonl) as reader, \
                 patch.object(torch.nn.functional, 'cross_entropy', wraps=real_ce) as loss, \
                 patch.object(torch.optim, 'AdamW', wraps=real_adam) as optimizer, \
                 patch.object(torch.nn.utils, 'clip_grad_norm_', wraps=real_clip) as clip, \
                 contextlib.redirect_stdout(io.StringIO()):
                model, logs = baseline.train_model(root, 'cpu', output_dir=output)
            self.assertEqual([call.args[0].name for call in reader.call_args_list], ['train.jsonl', 'valid_queries.jsonl'])
            self.assertEqual(optimizer.call_args.kwargs, dict(lr=.0015, weight_decay=1e-5))
            self.assertTrue(all(call.args[0].shape[1] == 24 for call in loss.call_args_list))
            self.assertTrue(all(set(call.args[1].tolist()) == {1, 2} for call in loss.call_args_list))
            self.assertTrue(all(call.args[1] == 1.0 for call in clip.call_args_list))
            self.assertEqual(len(logs), 40)
            self.assertFalse(model.training)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            best = observed_states[2]  # epoch 15
            self.assertTrue(any(not torch.equal(best[k], observed_states[-2][k]) for k in best))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, best[key], rtol=0, atol=0)
            saved = torch.load(output / 'sasrec_corrected_seed_20261101.pt', weights_only=True)
            for key in best: torch.testing.assert_close(saved[key], best[key], rtol=0, atol=0)
            record = json.loads((output / 'train_config.json').read_text())
            self.assertEqual(validate_training_record(record)['best_epoch'], 15)
            self.assertEqual(record['epochs_completed'], 40)
            self.assertEqual(record['policy']['batch_size'], 512)
            for field, value in [('best_epoch', 40), ('parameters_frozen', False), ('status', 'training')]:
                broken = copy.deepcopy(record); broken[field] = value
                with self.subTest(field=field), self.assertRaises(AssertionError): validate_training_record(broken)


if __name__ == '__main__': unittest.main()
