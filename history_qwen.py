"""Greedy local Qwen generation through one batched vLLM server.

Uses the original tokenizer/chat template and exact original input token limits.
This engine's outputs are kept separately from the earlier HF throughput pilot.
"""
import json,os,time,urllib.request
from pathlib import Path
from transformers import AutoTokenizer
from recommendation_agent import AGENT_CONFIG
from run_paths import MODEL_PATH, QWEN_URL
from qwen_vllm import load_cached_rows
URL=QWEN_URL

class HistoryQwen:
    def __init__(self):
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
        self.tokenizer.padding_side='left'

    def generate(self,requests,path,max_new_tokens,max_length=None):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        existing=load_cached_rows(path)
        by_id={r['id']:r for r in existing}; assert len(by_id)==len(existing)
        settings=dict(engine='vllm-0.8.5',max_new_tokens=max_new_tokens,max_length=max_length,
                      do_sample=False,temperature=0.,seed=20261101,request_batch_size=256)
        for request in requests:
            if request['id'] in by_id:
                row=by_id[request['id']]
                assert row['prompt']==request['prompt'] and row['generation']==settings
        pending=[r for r in requests if r['id'] not in by_id]
        for start in range(0,len(pending),256):
            batch=pending[start:start+256]
            texts=[self.tokenizer.apply_chat_template([dict(role='user',content=r['prompt'])],
                    tokenize=False,add_generation_prompt=True) for r in batch]
            kwargs=dict(padding=False)
            if max_length is not None: kwargs.update(truncation=True,max_length=max_length)
            ids=self.tokenizer(texts,**kwargs)['input_ids']
            if max_length is None: assert max(map(len,ids))<=AGENT_CONFIG['max_context_tokens']
            data=dict(model='beauty-qwen',prompt=ids,max_tokens=max_new_tokens,
                      temperature=0.,seed=20261101,logprobs=0,skip_special_tokens=True,
                      stop_token_ids=[151645,151643])
            req=urllib.request.Request(URL,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
            t=time.perf_counter()
            with urllib.request.urlopen(req,timeout=1800) as response: result=json.load(response)
            duration=time.perf_counter()-t
            choices=sorted(result['choices'],key=lambda x:x['index'])
            assert len(choices)==len(batch) and [x['index'] for x in choices]==list(range(len(batch)))
            with path.open('a',encoding='utf-8') as f:
                for j,(request,choice) in enumerate(zip(batch,choices)):
                    length=len(choice['logprobs']['tokens'])
                    row=dict(**request,generation=settings,response=choice['text'].strip(),
                         input_tokens=len(ids[j]),output_tokens=length,finish_reason=choice['finish_reason'],
                         batch_seconds=duration,allocated_seconds=duration/len(batch),
                         hit_output_limit=choice['finish_reason']=='length',
                         server_batch_usage=result.get('usage'),inference_url=URL)
                    f.write(json.dumps(row,ensure_ascii=False)+'\n'); by_id[row['id']]=row
            print(json.dumps(dict(qwen_file=path.name,complete=len(by_id),total=len(requests),batch_seconds=round(duration,2))),flush=True)
        return [by_id[r['id']] for r in requests]
