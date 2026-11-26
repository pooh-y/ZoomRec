"""Test the evaluator's candidate and chronological information boundaries."""
import copy
from baseline_source import SASRec
from data_split import make_events, audit_events, cuts

def main():
    sequences={u:dict(items=list(range(1,n+1)),times=list(range(n)),row_ids=list(range(n)))
               for u,n in [('short',5),('long',23)]}
    model = SASRec(59).eval().requires_grad_(False)
    events,counts=make_events(sequences,range(1,60),model=model)
    audit_events(events,sequences,range(1,60))
    again,_=make_events(dict(reversed(list(sequences.items()))),range(1,60),model=model)
    assert events==again
    assert counts['test_events']==sum(len(s['items'])-cuts(len(s['items']))[1] for s in sequences.values())
    bad=copy.deepcopy(events); e=bad[0]
    e['candidates'][0] = 0
    try: audit_events(bad,sequences,range(1,60))
    except AssertionError: pass
    else: raise AssertionError('Padding must never enter the retrieved list')
    bad=copy.deepcopy(events); bad[0]['history'].append(bad[0]['target'])
    try: audit_events(bad,sequences,range(1,60))
    except AssertionError: pass
    else: raise AssertionError('Target must not leak into history')
    print('PASS: full coverage, chronological history, deterministic candidates, padding excluded')

if __name__=='__main__': main()
