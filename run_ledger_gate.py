"""Frozen SASRec plus the unchanged observed-history preference scorer."""
import gc
import numpy as np
import torch
from run_paths import CODE_DIR

def load_baseline(cards,events,states_override=None,checkpoint=None,device='cuda:0'):
    if states_override is None or checkpoint is None:
        raise ValueError('Explicit history states and a validated checkpoint are required')
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_baseline',CODE_DIR/'baseline_source.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    states=states_override; features,cats,tokens,tidf,cidf=mod.make_features(events,cards,len(cards),np.ones(len(cards)+1),states)
    model=mod.SASRec(len(cards)).to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    model.eval(); model.requires_grad_(False)
    cases=[]
    for start in range(0,len(events),128):
        batch=events[start:start+128]
        with torch.inference_mode(): raw=model.logits(torch.tensor([mod.right_pad(e['history']) for e in batch],device=device)).float().cpu().numpy()
        for j,e in enumerate(batch):
            c=e['candidates']; z=lambda v:(v-v.mean())/(v.std()+1e-6)
            checkpoint_scores = raw[j, np.asarray(c)]
            retained = np.asarray(e.get('sasrec_scores', checkpoint_scores), dtype=checkpoint_scores.dtype)
            if retained.shape != checkpoint_scores.shape or not np.allclose(
                    retained, checkpoint_scores, rtol=1e-4, atol=1e-5):
                raise ValueError('Retained candidate scores disagree with the frozen SASRec checkpoint')
            b0=z(retained); f=features[start+j]
            m2=z(np.asarray([mod.score_memory(f,i,cats,tokens,2) for i in c])); m3=z(np.asarray([mod.pairwise_score(f,i,cats,tokens,tidf,cidf) for i in c]))
            sorted_base=np.sort(b0)[::-1]; uncertainty=float(np.clip(1-(sorted_base[0]-sorted_base[1])/1.5,.20,.85))
            scores=b0+.48*m2+(.22+.58*uncertainty)*m3; br,bt=mod.rank(scores,c,e['target'])
            target_cat=(cards[e['target']].get('categories') or [None])[0]
            sr,st=mod.rank(retained,c,e['target'])
            cases.append(dict(**e,baseline=scores.tolist(),gate=uncertainty,baseline_rank=br,baseline_top10=bt,sasrec_rank=sr,sasrec_top10=st,target_category=target_cat))
    del model; gc.collect(); torch.cuda.empty_cache(); return mod,cases
