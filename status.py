import json
from pathlib import Path
from run_paths import HERE as p
read=lambda f:json.loads(f.read_text())
m=read(p/'manifest.json'); completed=[read(f) for f in (p/'shards').glob('*/done.json')]
status={}
for f in p.glob('worker_*_status.json'):
    s=read(f); s['n_completed_by_worker']=len(s.pop('completed_shards',[])); status[f.stem]=s
counts={name:sum(sum(1 for line in f.open() if line.strip()) for f in (p/'shards').glob('*/*/'+name+'.jsonl'))
        for name in ['qwen_state_calls','plan_calls','decision_calls']}
print(json.dumps(dict(completed_events=sum(s['n_events'] for s in completed),total_events=m['counts']['test_events'],
    completed_shards=len(completed),total_shards=m['n_shards'],calls=counts,workers=status,
    final_results=(p/'results.json').exists()),indent=2))
