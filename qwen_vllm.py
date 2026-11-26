"""Greedy local Qwen generation through one batched vLLM server.

Uses the original tokenizer/chat template and exact original input token limits.
This engine's outputs are kept separately from the earlier HF throughput pilot.
"""
import hashlib,json,os,time,urllib.error,urllib.request
from pathlib import Path
from transformers import AutoTokenizer
from recommendation_agent import AGENT_CONFIG
from generation_protocol import (SCHEMAS, GUIDED_BACKEND, TOKEN_LIMITS,
    stage_for_limit, generation_settings, response_error, check_completed_call)
from run_paths import MODEL_PATH, QWEN_URL, project_reference

URL=QWEN_URL


def load_cached_rows(path):
    """Recover only an unfinished final append; complete corrupt lines are fatal.

    The caller holds the shard lock. Archive discarded bytes before truncation,
    retaining every complete response so only the missing requests are generated.
    """
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.split(b'\n')
    rows = []
    offset = 0
    for index, line in enumerate(lines):
        terminated = index < len(lines) - 1
        if line.strip():
            try:
                row = json.loads(line.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if terminated:
                    raise
                backup = path.with_name(path.name + '.incomplete-' +
                    hashlib.sha256(line).hexdigest() + '.bin')
                with path.open('r+b') as handle:
                    if handle.read() != raw:
                        raise RuntimeError(f'Cache changed during recovery: {project_reference(path.absolute())}')
                    try:
                        with backup.open('xb') as saved:
                            saved.write(line)
                            saved.flush()
                            os.fsync(saved.fileno())
                    except FileExistsError:
                        if backup.read_bytes() != line:
                            raise RuntimeError(f'Cache recovery backup mismatch: {backup}')
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                print(json.dumps(dict(cache_recovered=project_reference(path.absolute()), kept_rows=len(rows),
                    discarded_tail_bytes=len(line), tail_backup=project_reference(backup.absolute()))), flush=True)
                return rows
            rows.append(row)
        offset += len(line) + int(terminated)
    # A complete JSON object without a final newline is valid, but the next
    # append needs a separator. Do not discard or regenerate this response.
    if raw and not raw.endswith(b'\n'):
        with path.open('r+b') as handle:
            if handle.read() != raw:
                raise RuntimeError(f'Cache changed during recovery: {project_reference(path.absolute())}')
            handle.write(b'\n')
            handle.flush()
            os.fsync(handle.fileno())
    return rows


class LocalQwen:
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
        self.tokenizer.padding_side='left'

    def generate(self,requests,path,max_new_tokens,max_length=None):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        existing=load_cached_rows(path)
        by_id={r['id']:r for r in existing}; assert len(by_id)==len(existing)
        request_ids={r['id'] for r in requests}
        assert len(request_ids)==len(requests) and set(by_id)<=request_ids
        settings=generation_settings(max_new_tokens,max_length)
        stage=stage_for_limit(max_new_tokens)
        for request in requests:
            if request['id'] in by_id:
                row=by_id[request['id']]
                assert row['prompt']==request['prompt'] and row['generation']==settings
                check_completed_call(row,settings)
        pending=[r for r in requests if r['id'] not in by_id]
        for start in range(0,len(pending),256):
            batch=pending[start:start+256]
            texts=[self.tokenizer.apply_chat_template([dict(role='user',content=r['prompt'])],
                    tokenize=False,add_generation_prompt=True) for r in batch]
            kwargs=dict(padding=False)
            if max_length is not None: kwargs.update(truncation=True,max_length=max_length)
            ids=self.tokenizer(texts,**kwargs)['input_ids']
            if max_length is None: assert max(map(len,ids))<=AGENT_CONFIG['max_context_tokens']
            ids_by_request={r['id']:tokens for r,tokens in zip(batch,ids)}
            attempts={r['id']:[] for r in batch}
            retry=batch
            for limit in TOKEN_LIMITS[stage]:
                data=dict(model='beauty-qwen',prompt=[ids_by_request[r['id']] for r in retry],
                    max_tokens=limit,temperature=0.,seed=20261101,logprobs=0,
                    skip_special_tokens=True,stop_token_ids=[151645,151643],
                    guided_json=SCHEMAS[stage],guided_decoding_backend=GUIDED_BACKEND)
                req=urllib.request.Request(URL,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
                t=time.perf_counter()
                try:
                    with urllib.request.urlopen(req,timeout=1800) as response: result=json.load(response)
                except urllib.error.HTTPError as exc:
                    detail=exc.read().decode('utf-8',errors='replace')[:4000]
                    raise RuntimeError(f'vLLM HTTP {exc.code}: {detail}') from exc
                duration=time.perf_counter()-t
                choices=sorted(result['choices'],key=lambda x:x['index'])
                assert len(choices)==len(retry) and [x['index'] for x in choices]==list(range(len(retry)))
                next_retry=[]
                with path.open('a',encoding='utf-8') as handle:
                    for request,choice in zip(retry,choices):
                        raw=choice['text'].strip()
                        attempt=dict(response=raw,output_tokens=len(choice['logprobs']['tokens']),
                            finish_reason=choice['finish_reason'],max_new_tokens=limit,
                            hit_output_limit=choice['finish_reason']=='length',
                            schema_error=response_error(raw,stage),batch_seconds=duration,
                            allocated_seconds=duration/len(retry),server_batch_usage=result.get('usage'))
                        attempts[request['id']].append(attempt)
                        if attempt['hit_output_limit'] or attempt['schema_error'] is not None:
                            next_retry.append(request)
                            continue
                        row=dict(**request,generation=settings,response=raw,
                            input_tokens=len(ids_by_request[request['id']]),
                            output_tokens=attempt['output_tokens'],finish_reason=attempt['finish_reason'],
                            batch_seconds=duration,allocated_seconds=sum(a['allocated_seconds'] for a in attempts[request['id']]),
                            hit_output_limit=False,attempt_token_limit=limit,
                            attempts=attempts[request['id']],inference_url=URL)
                        check_completed_call(row,settings)
                        handle.write(json.dumps(row,ensure_ascii=False)+'\n'); by_id[row['id']]=row
                print(json.dumps(dict(qwen_file=path.name,complete=len(by_id),total=len(requests),
                    attempt_token_limit=limit,retrying=len(next_retry),batch_seconds=round(duration,2))),flush=True)
                retry=next_retry
                if not retry:
                    break
            if retry:
                failure_path=path.with_name(path.stem+'.failures.jsonl')
                with failure_path.open('a',encoding='utf-8') as handle:
                    for request in retry:
                        handle.write(json.dumps(dict(**request,generation=settings,
                            attempts=attempts[request['id']]),ensure_ascii=False)+'\n')
                raise RuntimeError(f'{len(retry)} {stage} responses remained incomplete/invalid after '
                    f'{len(TOKEN_LIMITS[stage])} attempts; see {project_reference(failure_path.absolute())}. No fallback decisions cached.')
        return [by_id[r['id']] for r in requests]
