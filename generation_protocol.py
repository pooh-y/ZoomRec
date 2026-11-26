"""One frozen structured-output and bounded-retry policy for all five methods."""
import hashlib
import json

from jsonschema import Draft202012Validator
from recommendation_agent import AGENT_CONFIG

PROTOCOL_VERSION = 'structured-json-v1'
GUIDED_BACKEND = 'xgrammar'
TOKEN_LIMITS = {'plan': (256, 512, 1024), 'decision': (2048, 4096, 8192)}
MAX_MODEL_LEN = AGENT_CONFIG['max_context_tokens'] + max(TOKEN_LIMITS['decision'])

PLAN_SCHEMA = dict(type='object', additionalProperties=False,
    properties=dict(retrieve_history=dict(type='array', items=dict(type='string')),
                    verify_items=dict(type='array', items=dict(type='integer')),
                    reason=dict(type='string')),
    required=['retrieve_history', 'verify_items', 'reason'])
TRANSFER_SCHEMA = dict(type='object', additionalProperties=False,
    properties=dict(item=dict(type='integer'), hypothesis=dict(type='string'),
        competitor=dict(type='integer'), confidence=dict(type='string', enum=['low', 'medium', 'high']),
        reason=dict(type='string')),
    required=['item', 'hypothesis', 'competitor', 'confidence', 'reason'])
DECISION_SCHEMA = dict(type='object', additionalProperties=False,
    properties=dict(transfers=dict(type='array', items=TRANSFER_SCHEMA), reason=dict(type='string')),
    required=['transfers', 'reason'])
SCHEMAS = dict(plan=PLAN_SCHEMA, decision=DECISION_SCHEMA)
VALIDATORS = {name: Draft202012Validator(schema) for name, schema in SCHEMAS.items()}


def stage_for_limit(max_new_tokens):
    for stage, limits in TOKEN_LIMITS.items():
        if max_new_tokens == limits[0]:
            return stage
    raise ValueError(f'Unsupported initial output budget: {max_new_tokens}')


def generation_settings(max_new_tokens, max_length=None):
    stage = stage_for_limit(max_new_tokens)
    schema = json.dumps(SCHEMAS[stage], sort_keys=True, separators=(',', ':')).encode()
    return dict(engine='vllm-0.8.5', max_new_tokens=max_new_tokens, max_length=max_length,
        do_sample=False, temperature=0., seed=20261101, request_batch_size=256,
        output_protocol=PROTOCOL_VERSION, guided_decoding_backend=GUIDED_BACKEND,
        output_schema_sha256=hashlib.sha256(schema).hexdigest(),
        attempt_token_limits=list(TOKEN_LIMITS[stage]))


def response_error(raw, stage):
    try:
        value = json.loads(raw)
    except (ValueError, TypeError) as exc:
        return f'invalid JSON: {exc}'
    error = next(VALIDATORS[stage].iter_errors(value), None)
    return None if error is None else f'{error.json_path}: {error.message}'


def check_completed_call(row, settings):
    """A final cache entry must prove that its accepted attempt was complete."""
    assert row['generation'] == settings, 'inference policy changed'
    stage = stage_for_limit(settings['max_new_tokens'])
    attempts = row['attempts']
    assert 1 <= len(attempts) <= len(TOKEN_LIMITS[stage])
    for index, attempt in enumerate(attempts):
        limit = TOKEN_LIMITS[stage][index]
        assert attempt['max_new_tokens'] == limit
        assert 0 <= attempt['output_tokens'] <= limit
        error = response_error(attempt['response'], stage)
        assert attempt['schema_error'] == error
        assert attempt['hit_output_limit'] == (attempt['finish_reason'] == 'length')
        if index < len(attempts) - 1:
            assert attempt['hit_output_limit'] or error is not None
        else:
            assert not attempt['hit_output_limit'] and error is None
            assert attempt['finish_reason'] == 'stop' and attempt['output_tokens'] > 0
    accepted = attempts[-1]
    for key in ('response', 'output_tokens', 'finish_reason', 'hit_output_limit'):
        assert row[key] == accepted[key]
    assert row['attempt_token_limit'] == accepted['max_new_tokens']
    assert row['input_tokens'] <= AGENT_CONFIG['max_context_tokens']
