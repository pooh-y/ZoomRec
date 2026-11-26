"""Check retry boundaries, strict valid caches and independent output audit."""
import copy
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from audit_ablation_outputs import check_calls
from generation_protocol import (SCHEMAS, TOKEN_LIMITS, MAX_MODEL_LEN,
    generation_settings, response_error, check_completed_call)
from recommendation_agent import AGENT_CONFIG
from test_run_reliability import qwen, cached_row, encode
import test_run_reliability as reliability

PLAN = json.dumps(dict(retrieve_history=[], verify_items=[], reason='no evidence'))
DECISION = json.dumps(dict(transfers=[], reason='no supported change'))


def response(*choices):
    return io.BytesIO(encode(dict(choices=[dict(index=i, text=text, finish_reason=finish,
        logprobs=dict(tokens=['token'] * tokens))
        for i, (text, finish, tokens) in enumerate(choices)])))


class GenerationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'plan_calls.jsonl'
        self.client = reliability.CacheResumeTests.client(self)

    def test_server_context_covers_largest_retry_without_truncating_evidence(self):
        self.assertEqual(AGENT_CONFIG['decision_tokens'], 2048)
        self.assertEqual(MAX_MODEL_LEN, AGENT_CONFIG['max_context_tokens'] + 8192)
        self.assertEqual(TOKEN_LIMITS['decision'], (2048, 4096, 8192))

    def test_only_truncated_member_of_batch_is_retried_and_good_cache_is_preserved(self):
        requests = [dict(id='ok', prompt='first'), dict(id='retry', prompt='second')]
        with patch.object(qwen.urllib.request, 'urlopen', side_effect=[
            response((PLAN, 'stop', 30), ('{"retrieve_history":', 'length', 256)),
            response((PLAN, 'stop', 40))]) as http:
            rows = self.client.generate(requests, self.path, 256)
        sent = [json.loads(call.args[0].data) for call in http.call_args_list]
        self.assertEqual([x['max_tokens'] for x in sent], [256, 512])
        self.assertEqual([len(x['prompt']) for x in sent], [2, 1])
        self.assertEqual(sent[0]['guided_json'], SCHEMAS['plan'])
        self.assertEqual(sent[0]['guided_decoding_backend'], 'xgrammar')
        self.assertEqual([len(row['attempts']) for row in rows], [1, 2])
        self.assertEqual([row['id'] for row in rows], ['ok', 'retry'])
        for row in rows:
            check_completed_call(row, generation_settings(256))
        with patch.object(qwen.urllib.request, 'urlopen') as resumed:
            self.assertEqual(self.client.generate(requests, self.path, 256), rows)
            resumed.assert_not_called()

    def test_valid_looking_json_with_length_finish_is_still_retried(self):
        with patch.object(qwen.urllib.request, 'urlopen', side_effect=[
            response((DECISION, 'length', 2048)), response((DECISION, 'stop', 25))]) as http:
            rows = self.client.generate([dict(id='d', prompt='decision')], self.path, 2048)
        self.assertEqual([json.loads(c.args[0].data)['max_tokens'] for c in http.call_args_list], [2048, 4096])
        self.assertEqual(rows[0]['attempt_token_limit'], 4096)
        self.assertFalse(rows[0]['hit_output_limit'])
        self.assertTrue(rows[0]['attempts'][0]['hit_output_limit'])

    def test_malformed_or_wrong_schema_retries_then_fails_without_empty_fallback(self):
        invalid = ['{"transfers":', '{"transfers":"wrong type","reason":"x"}', '{}']
        with patch.object(qwen.urllib.request, 'urlopen', side_effect=[
            response((raw, 'stop', 10)) for raw in invalid]) as http:
            with self.assertRaisesRegex(RuntimeError, 'No fallback decisions cached'):
                self.client.generate([dict(id='bad', prompt='decision')], self.path, 2048)
        self.assertEqual(http.call_count, 3)
        self.assertEqual(qwen.load_cached_rows(self.path), [])
        diagnostic = json.loads(self.path.with_name('plan_calls.failures.jsonl').read_text())
        self.assertEqual([a['response'] for a in diagnostic['attempts']], invalid)
        self.assertTrue(all(a['schema_error'] for a in diagnostic['attempts']))

    def test_successful_rows_survive_another_request_exhausting_all_retries(self):
        with patch.object(qwen.urllib.request, 'urlopen', side_effect=[
            response((PLAN, 'stop', 10), ('{', 'length', 256)),
            response(('{', 'length', 512)), response(('{', 'length', 1024))]):
            with self.assertRaises(RuntimeError):
                self.client.generate([dict(id='good', prompt='p'), dict(id='bad', prompt='q')], self.path, 256)
        self.assertEqual([r['id'] for r in qwen.load_cached_rows(self.path)], ['good'])

    def test_bad_cached_response_cannot_bypass_generation_validation(self):
        bad = cached_row()
        bad['response'] = '{}'
        self.path.write_bytes(encode(bad) + b'\n')
        with patch.object(qwen.urllib.request, 'urlopen') as http, self.assertRaises(AssertionError):
            self.client.generate([dict(id='completed', prompt='saved')], self.path, 256)
        http.assert_not_called()

    def test_audit_rejects_old_protocol_and_incomplete_accepted_attempt(self):
        action = dict(event_id='completed', plan_called=True, decision_called=False)
        good = cached_row()
        self.path.write_bytes(encode(good) + b'\n')
        self.assertEqual(check_calls(self.path.parent, [action])['plan_calls'], 1)
        for broken in ('old_policy', 'incomplete'):
            row = copy.deepcopy(good)
            if broken == 'old_policy':
                row['generation'].pop('output_protocol')
            else:
                row['attempts'][-1]['finish_reason'] = 'length'
                row['attempts'][-1]['hit_output_limit'] = True
            self.path.write_bytes(encode(row) + b'\n')
            with self.subTest(broken=broken), self.assertRaises(AssertionError):
                check_calls(self.path.parent, [action])

    def test_schema_changes_format_not_semantic_recommendation_checks(self):
        # Structurally valid invented IDs remain the semantic validator's job.
        raw = json.dumps(dict(transfers=[dict(item=999999, hypothesis='unknown',
            competitor=888888, confidence='high', reason='untrusted claim')], reason='test'))
        self.assertIsNone(response_error(raw, 'decision'))
        self.assertIsNotNone(response_error('```json\n' + raw + '\n```', 'decision'))

    def test_api_validation_error_preserves_server_explanation(self):
        error = urllib.error.HTTPError('http://localhost/v1/completions', 400,
            'Bad Request', {}, io.BytesIO(b'{"error":"backend mismatch"}'))
        with patch.object(qwen.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'vLLM HTTP 400.*backend mismatch'):
                self.client.generate([dict(id='bad', prompt='p')], self.path, 256)
        self.assertFalse(self.path.exists())


if __name__ == '__main__':
    unittest.main()
