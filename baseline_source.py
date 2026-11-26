"""ZoomRec SASRec training and observed-history preference scores."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, shutil, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ranking_metrics import metric_table, target_rank
from training_protocol import EarlyStopping, training_policy, validate_training_record

SEED = 20261101


def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def dump(p, x):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(p, rows):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False, allow_nan=False)+"\n")

def right_pad(h, max_len=50):
    h=list(h[-max_len:])
    return h+[0]*(max_len-len(h))

class SASRec(nn.Module):
    """A finite, right-padded SASRec with explicit last-valid readout."""
    def __init__(self, n_items, max_len=50, hidden=64, layers=2, heads=4):
        super().__init__(); self.n_items=n_items; self.max_len=max_len
        self.item=nn.Embedding(n_items+1, hidden, padding_idx=0)
        self.pos=nn.Embedding(max_len, hidden)
        block=nn.TransformerEncoderLayer(hidden, heads, hidden*4, 0.1,
                                         batch_first=True, norm_first=True, activation="gelu")
        self.encoder=nn.TransformerEncoder(block, layers)
        self.norm=nn.LayerNorm(hidden)
    def encode(self, seq):
        b,l=seq.shape; p=torch.arange(l,device=seq.device).unsqueeze(0).expand(b,-1)
        pad=seq.eq(0); x=self.item(seq)+self.pos(p)
        causal=torch.triu(torch.ones(l,l,device=seq.device,dtype=torch.bool),1)
        x=self.encoder(x,mask=causal,src_key_padding_mask=pad)
        n=(~pad).sum(1).clamp_min(1)-1
        z=self.norm(x[torch.arange(b,device=seq.device),n])
        if not torch.isfinite(z).all(): raise FloatingPointError("nonfinite SASRec representation")
        return z
    def logits(self,seq):
        z=self.encode(seq); out=z @ self.item.weight.T
        if not torch.isfinite(out).all(): raise FloatingPointError("nonfinite SASRec logits")
        return out

def metrics(rows):
    o = metric_table(rows)
    o.update(n_events=len(rows), n_users=len({x["uid"] for x in rows}))
    return o

def rank(scores, cand, target):
    c=np.asarray(cand,dtype=np.int64); s=np.asarray(scores,dtype=np.float64)
    # candidate order is never used as a signal; item id only resolves ties.
    order=np.lexsort((c,-s)); ranked=c[order]
    return target_rank(ranked.tolist(), target), ranked[:10].tolist()

def load_state(root, events):
    q={x["event_id"]:x["state"] for x in read_jsonl(root/"qwen/states.jsonl")}
    missing=[e["event_id"] for e in events if e["event_id"] not in q]
    if missing: raise RuntimeError(f"missing Qwen states: {len(missing)}")
    return q

def token_set(card):
    return set(str(x).lower() for x in card.get("tokens",[])) | set(str(x).lower() for x in card.get("categories",[]))

def make_features(events, cards, n_items, train_pop, states):
    """Evidence account from observed history only, with item-content bridge."""
    # Build a compact category vocabulary and item memberships.
    cat_vocab=sorted({c for card in cards.values() for c in card.get("categories",[]) if c!="beauty"})
    cat_id={c:i for i,c in enumerate(cat_vocab)}
    item_cats={int(i):set(c for c in card.get("categories",[]) if c!="beauty") for i,card in cards.items()}
    item_tokens={int(i):token_set(card) for i,card in cards.items()}
    token_df=Counter(t for ts in item_tokens.values() for t in ts)
    n_catalog=max(1,len(item_tokens))
    token_idf={t:math.log1p((n_catalog+1)/(df+1)) for t,df in token_df.items()}
    cat_df=Counter(c for cs in item_cats.values() for c in cs)
    cat_idf={c:math.log1p((n_catalog+1)/(df+1)) for c,df in cat_df.items()}
    ev=[]
    for e in events:
        hist=e["history"]; st=states[e["event_id"]]
        stable=Counter(str(x).lower() for x in st.get("stable",[]))
        transient=Counter(str(x).lower() for x in st.get("transient",[]))
        # observed evidence account: recurrence, recency and category conflict.
        stable_obs=Counter(); recent_obs=Counter(); token_obs=Counter()
        for j,item in enumerate(hist):
            card=cards[str(item)] if str(item) in cards else cards[item]
            age=len(hist)-1-j; w=math.exp(-age/5.0)
            for c in item_cats.get(item,set()):
                stable_obs[c]+=1.0; recent_obs[c]+=w
            for t in item_tokens.get(item,set()): token_obs[t]+=w
        # Qwen terms are evidence hints, never a free-standing target signal.
        for k,v in stable.items(): stable_obs[k]+=0.55*min(3,v)
        for k,v in transient.items(): recent_obs[k]+=0.65*min(3,v)
        # A topic with repeated support is stable; a recent isolated topic is transient.
        conflict={c for c,n in stable_obs.items() if n>=2 and recent_obs[c]<0.7}
        ev.append({"stable":stable_obs,"recent":recent_obs,"tokens":token_obs,
                   "conflict":conflict,"state":st})
    return ev, item_cats, item_tokens, token_idf, cat_idf

def score_memory(feat, item, item_cats, item_tokens, round_no):
    cats=item_cats.get(int(item),set()); toks=item_tokens.get(int(item),set())
    stable=feat["stable"]; recent=feat["recent"]; tok=feat["tokens"]
    if not cats and not toks: return 0.0
    # Round 1: evidence-anchored dual timescale with reversible pause state.
    base=sum(0.19*stable[c]+0.38*recent[c] for c in cats)
    base+=sum(0.055*tok[t] for t in toks)
    base-=sum(0.20*stable[c] for c in cats & feat["conflict"])
    if round_no==1: return base
    # Round 2: competing durable-shift / brief-exploration / return hypotheses.
    durable=sum(stable[c]/(1+stable[c]) for c in cats)
    explore=sum(recent[c]/(1+recent[c]) for c in cats)
    return base + 0.36*durable + 0.48*explore + 0.24*max(0.0, explore-durable)
    
def pairwise_score(feat, item, item_cats, item_tokens, token_idf, cat_idf):
    """Round 3 rare-facet retrieval for candidate-pair adjudication."""
    cats=item_cats.get(int(item),set()); toks=item_tokens.get(int(item),set())
    stable=feat["stable"]; recent=feat["recent"]; tok=feat["tokens"]
    # Downweight generic catalog vocabulary (e.g. broad department words) and
    # retrieve the more discriminative title/category facets in the history.
    facet=sum((0.55+math.log1p(tok[t]))*token_idf.get(t,1.0) for t in toks if tok[t]>0)
    cscore=sum((0.35*stable[c]+0.45*recent[c])*cat_idf.get(c,1.0) for c in cats)
    support=sum(stable[c] for c in cats); rec=sum(recent[c] for c in cats)
    evidence=facet/math.sqrt(max(1,len(toks))) + cscore
    confidence=(support+rec)/(1+support+rec)
    return confidence*evidence - 0.28*sum(cat_idf.get(c,1.0) for c in cats & feat["conflict"])

def validation_tensors(events, device):
    if not events or any(row['phase'] != 'valid' for row in events):
        raise ValueError('Checkpoint selection requires validation queries')
    if len({row['uid'] for row in events}) != len(events):
        raise ValueError('Expected one validation query per user')
    histories = [row['history'] for row in events]
    encoder_input = torch.tensor([right_pad(history) for history in histories], device=device)
    targets = torch.tensor([row['target'] for row in events], device=device)
    return encoder_input, targets, histories


def validation_ndcg10(model, tensors, batch=512):
    """Score the raw unseen-item ranking, including all retrieval misses."""
    from sasrec_candidates import top_candidates
    x, targets, histories = tensors
    if len(x) == 0:
        raise ValueError('Validation set is empty')
    model.eval()
    total = 0.
    discounts = 1 / torch.log2(torch.arange(2, 12, device=x.device, dtype=torch.float64))
    with torch.inference_mode():
        for start in range(0, len(x), batch):
            sl = slice(start, start + batch)
            top10 = top_candidates(model.logits(x[sl]), histories[sl])[:, :10]
            matches = top10.eq(targets[sl, None])
            total += (matches * discounts).sum().item()
    return total / len(x)


def train_model(root, device, min_epochs=20, batch=512, output_dir=None):
    policy = training_policy(min_epochs, batch)
    root = Path(root)
    data=read_jsonl(root/"freeze/train.jsonl"); n_items=len(json.loads((root/"freeze/item_map.json").read_text()))
    if not data: raise ValueError('Training set is empty')
    # This function never opens the test set. Validation is used only for selection.
    validation = validation_tensors(read_jsonl(root/'freeze/valid_queries.jsonl'), device)
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    x=torch.tensor([right_pad(r[1]) for r in data],dtype=torch.long,device=device)
    y=torch.tensor([r[2] for r in data],dtype=torch.long,device=device)
    model=SASRec(n_items).to(device); opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=1e-5)
    out=Path(output_dir) if output_dir is not None else root/'model'
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'sasrec_corrected_seed_20261101.pt'
    stopper=EarlyStopping(min_epochs); logs=[]; validation_history=[]; ep=0
    record=dict(status='training', policy=policy, train_sha256=sha(root/'freeze/train.jsonl'),
        valid_queries_sha256=sha(root/'freeze/valid_queries.jsonl'),
        item_map_sha256=sha(root/'freeze/item_map.json'),
        training_examples=len(data), validation_events=len(validation[0]),
        padding='right', readout='last nonzero', torch=torch.__version__, cuda=torch.version.cuda,
        initialized_from='random seed; no pretrained checkpoint loaded',
        logs=logs, validation_history=validation_history)
    while True:
        ep += 1
        model.train(); order=torch.randperm(len(x),device=device); total=0.
        for start in range(0,len(x),batch):
            ix=order[start:start+batch]; logits=model.logits(x[ix])
            # All real catalog items participate in the denominator; padding is not an item.
            loss=nn.functional.cross_entropy(logits[:,1:],y[ix]-1)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite SASRec loss")
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=float(loss.detach())*len(ix)
        logs.append({"epoch":ep,"mean_loss":total/len(x)})
        print(json.dumps(dict(training_epoch=ep, mean_loss=total/len(x))), flush=True)
        stop=False
        if ep % policy['evaluation_interval'] == 0:
            ndcg10=validation_ndcg10(model,validation,batch)
            result=stopper.update(ep,ndcg10); validation_history.append(result)
            if result['improved']:
                tmp=checkpoint.with_suffix('.pt.tmp'); torch.save(model.state_dict(),tmp); tmp.replace(checkpoint)
            stop=result['stop']
            print(json.dumps(dict(validation=result, best_epoch=stopper.best_epoch,
                                  best_validation_ndcg10=stopper.best_score)),flush=True)
        record.update(epochs_completed=ep, best_epoch=stopper.best_epoch,
            best_validation_ndcg10=stopper.best_score if stopper.best_epoch else None,
            non_improving_evaluations=stopper.bad_evaluations)
        tmp=out/'train_config.json.tmp'; dump(tmp,record); tmp.replace(out/'train_config.json')
        if stop: break
    # The checkpoint can come from any validation epoch, including one before epoch 20.
    # The minimum applies to training duration, not to checkpoint eligibility.
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
    model.eval(); model.requires_grad_(False)
    restored=validation_ndcg10(model,validation,batch)
    record.update(status='complete',stop_reason='patience_exhausted',
        checkpoint_sha256=sha(checkpoint),checkpoint_restored=True,parameters_frozen=True,
        restored_checkpoint_validation_ndcg10=restored)
    validate_training_record(record)
    tmp=out/'train_config.json.tmp'; dump(tmp,record); tmp.replace(out/'train_config.json')
    print(json.dumps(dict(training_complete=True,best_epoch=stopper.best_epoch,
        epochs_completed=ep,best_validation_ndcg10=stopper.best_score)),flush=True)
    return model,logs
