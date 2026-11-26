"""Exercise zero-call audits and interrupted generation-cache resume."""
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import full_evaluation
import run_paths
from audit_ablation_outputs import check_calls
from full_evaluation import run_variant
from ledger import review_event
from test_ablation_mechanisms import MODEL, CARDS, DAY
from generation_protocol import generation_settings

# Cache and HTTP behavior can be tested without downloading or loading a model.
spec = importlib.util.spec_from_file_location('_qwen_reliability',
    Path(__file__).with_name('qwen_vllm.py'))
qwen = importlib.util.module_from_spec(spec)
stub = types.ModuleType('transformers')
stub.AutoTokenizer = object
with patch.dict(sys.modules, {'transformers': stub}):
    spec.loader.exec_module(qwen)


def settings(limit=256):
    return generation_settings(limit)


def cached_row(event_id='completed'):
    response = json.dumps(dict(retrieve_history=[], verify_items=[], reason=''))
    attempt = dict(response=response, output_tokens=1, finish_reason='stop',
        max_new_tokens=256, hit_output_limit=False, schema_error=None)
    return dict(id=event_id, prompt='saved', response=response, generation=settings(),
        input_tokens=2, output_tokens=1, finish_reason='stop', hit_output_limit=False,
        attempt_token_limit=256, attempts=[attempt])


def encode(row):
    return json.dumps(row, ensure_ascii=False).encode('utf-8')


class PortablePathTests(unittest.TestCase):
    def test_relative_settings_use_the_project_as_their_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / 'project'
            with patch.object(run_paths, 'CODE_DIR', project), patch.dict(
                    os.environ, ZOOMREC_DATA_ROOT='../data'):
                data = run_paths.configured_path('ZOOMREC_DATA_ROOT', 'data')
                self.assertEqual(data, project.parent / 'data')
                self.assertEqual(run_paths.project_reference(data / 'items.json'), '../data/items.json')

    def test_hash_audit_reads_relative_and_legacy_absolute_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / 'project'
            project.mkdir()
            data = project.parent / 'data'
            sequences = data / 'freeze/sequences.json'
            sequences.parent.mkdir(parents=True)
            sequences.write_bytes(b'{}')
            digest = full_evaluation.sha(sequences)
            with patch.object(run_paths, 'CODE_DIR', project), patch.multiple(
                    full_evaluation, CODE_DIR=project, DATA_ROOT=data):
                for key in ('../data/freeze/sequences.json', str(sequences)):
                    full_evaluation.check_hashes({key: digest})
                    with self.assertRaises(ValueError):
                        full_evaluation.check_hashes({key: 'stale'})
                with self.assertRaises(ValueError):
                    full_evaluation.check_hashes({'../data/freeze/sequences.json': digest,
                                                  str(sequences): 'conflicting'})


class NoCallsAuditTests(unittest.TestCase):
    def test_real_no_call_variant_output_passes_audit_without_logs(self):
        class NoCalls:
            def generate(self, *args, **kwargs):
                raise AssertionError('Unexpected inference')

        class Reviews:
            def event(self, uid, index, item, sequence):
                return review_event(uid, index, item, (1 + 10 * index) * DAY, 5)

        case = dict(event_id='test:u:2', uid='u', index=2, history=[1, 2], target=3,
                    candidates=[3, 7, 8], baseline=[.3, .2, .1], gate=.8)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            run_variant(NoCalls(), [case], CARDS, MODEL, {'u': {}}, Reviews(),
                'no_revision', folder,
                dict(strength=.5, min_confidence='medium'))
            actions = [json.loads(line) for line in
                       (folder / 'actions.jsonl').read_text().splitlines()]
            self.assertFalse((folder / 'plan_calls.jsonl').exists())
            self.assertFalse((folder / 'decision_calls.jsonl').exists())
            self.assertEqual(check_calls(folder, actions),
                             dict(plan_calls=0, decision_calls=0))

    def test_required_missing_log_still_fails_for_either_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flag in ('plan_called', 'decision_called'):
                action = dict(event_id='e', plan_called=False, decision_called=False)
                action[flag] = True
                with self.subTest(stage=flag), self.assertRaises(FileNotFoundError):
                    check_calls(Path(tmp), [action])

    def test_unexpected_cached_call_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'plan_calls.jsonl').write_bytes(encode(cached_row()) + b'\n')
            with self.assertRaises(AssertionError):
                check_calls(folder, [dict(event_id='e', plan_called=False,
                                          decision_called=False)])

    def test_plan_only_output_requires_only_plan_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'plan_calls.jsonl').write_bytes(encode(cached_row('e')) + b'\n')
            counts = check_calls(folder, [dict(event_id='e', plan_called=True,
                                               decision_called=False)])
            self.assertEqual(counts, dict(plan_calls=1, decision_calls=0))


class CacheResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'plan_calls.jsonl'
        self.row = cached_row()
        self.prefix = encode(self.row) + b'\n'

    def backups(self):
        return list(self.path.parent.glob('*.incomplete-*.bin'))

    def test_missing_cache_is_empty_without_creating_file(self):
        self.assertEqual(qwen.load_cached_rows(self.path), [])
        self.assertFalse(self.path.exists())

    def test_valid_unicode_and_crlf_are_preserved_byte_for_byte(self):
        row = dict(self.row, response='Price: 5 \N{EURO SIGN}')
        raw = encode(row) + b'\r\n\n'
        self.path.write_bytes(raw)
        self.assertEqual(qwen.load_cached_rows(self.path), [row])
        self.assertEqual(self.path.read_bytes(), raw)
        self.assertFalse(self.backups())

    def test_torn_json_tail_is_archived_and_prefix_survives_repeat_resume(self):
        tail = b'{"id":"interrupted"'
        self.path.write_bytes(self.prefix + tail)
        for _ in range(2):
            self.assertEqual(qwen.load_cached_rows(self.path), [self.row])
        self.assertEqual(self.path.read_bytes(), self.prefix)
        self.assertEqual(len(self.backups()), 1)
        self.assertEqual(self.backups()[0].read_bytes(), tail)

    def test_partial_utf8_codepoint_is_recoverable(self):
        tail = b'{"response":"' + '\N{EURO SIGN}'.encode('utf-8')[:2]
        self.path.write_bytes(self.prefix + tail)
        self.assertEqual(qwen.load_cached_rows(self.path), [self.row])
        self.assertEqual(self.path.read_bytes(), self.prefix)
        self.assertEqual(self.backups()[0].read_bytes(), tail)

    def test_interruption_in_first_record_leaves_empty_cache(self):
        self.path.write_bytes(b'{"id":')
        self.assertEqual(qwen.load_cached_rows(self.path), [])
        self.assertEqual(self.path.read_bytes(), b'')
        self.assertEqual(self.backups()[0].read_bytes(), b'{"id":')

    def test_complete_corrupt_line_fails_without_modification(self):
        for raw in (self.prefix + b'{broken}\n',
                    self.prefix + b'{broken}\n' + encode(cached_row('last')),
                    self.prefix + b'\xff\n'):
            self.path.write_bytes(raw)
            with self.subTest(raw=raw), self.assertRaises((json.JSONDecodeError, UnicodeDecodeError)):
                qwen.load_cached_rows(self.path)
            self.assertEqual(self.path.read_bytes(), raw)
            self.assertFalse(self.backups())

    def client(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]['content']

            def __call__(self, texts, **kwargs):
                return dict(input_ids=[[1, 2] for _ in texts])

        client = qwen.LocalQwen.__new__(qwen.LocalQwen)
        client.tokenizer = Tokenizer()
        return client

    def resume_one_pending_request(self):
        requests = [dict(id='completed', prompt='saved'), dict(id='pending', prompt='new')]
        response = dict(choices=[dict(index=0, text=self.row['response'], finish_reason='stop',
                                      logprobs=dict(tokens=['json']))])
        with patch.object(qwen.urllib.request, 'urlopen',
                          return_value=io.BytesIO(encode(response))) as http:
            result = self.client().generate(requests, self.path, 256)
        http.assert_called_once()
        sent = json.loads(http.call_args.args[0].data)
        self.assertEqual(len(sent['prompt']), 1)
        self.assertEqual(result[0], self.row)
        self.assertEqual([row['id'] for row in result], ['completed', 'pending'])
        self.assertEqual(qwen.load_cached_rows(self.path), result)
        self.assertTrue(self.path.read_bytes().startswith(self.prefix))

    def test_resume_generates_only_interrupted_request(self):
        self.path.write_bytes(self.prefix + b'{"id":"pending"')
        self.resume_one_pending_request()
        self.assertEqual(len(self.backups()), 1)

    def test_complete_last_record_without_newline_is_reused_before_append(self):
        self.path.write_bytes(encode(self.row))
        self.resume_one_pending_request()
        self.assertFalse(self.backups())

    def test_duplicate_ids_and_prompt_or_setting_mismatch_still_fail(self):
        invalid = [self.prefix * 2,
            encode(dict(self.row, prompt='changed')) + b'\n',
            encode(dict(self.row, generation=dict(settings(), seed=123))) + b'\n']
        for raw in invalid:
            self.path.write_bytes(raw)
            with self.subTest(raw=raw), patch.object(qwen.urllib.request, 'urlopen') as http:
                with self.assertRaises(AssertionError):
                    self.client().generate([dict(id='completed', prompt='saved')], self.path, 256)
                http.assert_not_called()
            self.assertEqual(self.path.read_bytes(), raw)

    def test_all_completed_requests_require_no_http_call(self):
        self.path.write_bytes(self.prefix)
        with patch.object(qwen.urllib.request, 'urlopen') as http:
            result = self.client().generate([dict(id='completed', prompt='saved')], self.path, 256)
            http.assert_not_called()
        self.assertEqual(result, [self.row])


if __name__ == '__main__':
    unittest.main()
