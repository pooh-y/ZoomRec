"""Event-sourced preference hypotheses and explicit, non-probabilistic transfer gates.

Observed reviews have UNKNOWN exposure provenance. Low ratings are weak item-level
counterevidence, never a claim that a particular attribute caused dissatisfaction.
Missing title attributes contribute neither positive nor negative evidence.
"""
from __future__ import annotations
import copy
import hashlib
import html
import json
import math
import re
from collections import Counter, defaultdict
from data_split import cuts

CONFIG = dict(source_distinct=2, cold_visits=3, episode_days=7, half_life_days=180,
              low_rating_weight=.35, min_gate=.05, min_attribute_items=10,
              max_attribute_fraction=.20, prior_strength=20., min_boundary_users=5,
              min_boundary_hits=2)
STOP = set(('a an and are as at be by for from in is it of on or that the this to with '
 'your you our its their new product products oz ounce ounces fl fluid ml liter litre '
 'gram grams lb lbs ct count pack set pcs piece pieces size each one two three four five '
 'six seven eight nine ten amp quot reg eacute egrave trade nbsp best better great '
 'amazing premium quality professional pro original ultimate ultra deluxe perfect '
 'women woman men man girl girls beauty cosmetic cosmetics make up kit inch inches '
 'bottle bottles tube tubes box package packaging free sale shipping buy includes '
 'item items brand genuine authentic assorted collection value combo bonus all').split())
PHRASES = {
 'oil_free':r'\boil[\s-]+free\b',
 'fragrance_free':r'\b(?:fragrance[\s-]+free|unscented)\b',
 'sulfate_free':r'\bsulfate[\s-]+free\b',
 'paraben_free':r'\bparaben[\s-]+free\b',
 'sensitive_skin':r'\bsensitive\s+skin\b',
 'travel_size':r'\btravel[\s-]+(?:size|sized)\b',
 'anti_aging':r'\banti[\s-]+aging\b',
}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                     separators=(',',':')).encode()).hexdigest()

def clean_cards(cards, categories):
    category_words={w for c in categories.values() if c for w in re.findall('[a-z]+',c)}
    out={}
    for item,card in cards.items():
        title=html.unescape(html.unescape(card['title']))
        text=title.lower(); phrases=[]
        for name,pattern in PHRASES.items():
            if re.search(pattern,text):
                phrases.append(name); text=re.sub(pattern,' ',text)
        words=[w for w in re.findall('[a-z]+',text)
               if len(w)>=3 and w not in STOP and w not in category_words]
        out[item]={**card,'title':title,'raw_title':card['title'],
                   'tokens':sorted(set(words+phrases))}
    return out

def fit_boundaries(cards,categories,sequences,excluded_users):
    """Population association is a proposal prior, not individual causal evidence."""
    prefixes={u:s['items'][:cuts(len(s['items']))[0]]
              for u,s in sequences.items() if u not in excluded_users}
    train_items={i for seq in prefixes.values() for i in seq}
    df=Counter(t for i in train_items for t in cards[i]['tokens']); token_cats=defaultdict(set)
    for i in train_items:
        if categories[i]:
            for a in cards[i]['tokens']: token_cats[a].add(categories[i])
    vocab={a for a,n in df.items() if CONFIG['min_attribute_items']<=n<=len(train_items)*CONFIG['max_attribute_fraction']
           and len(token_cats[a])>=2}
    attrs={i:set(c['tokens'])&vocab for i,c in cards.items()}
    totals=Counter(); hits=Counter(); counts=defaultdict(lambda:[0,0,set()])
    for uid,seq in sorted(prefixes.items()):
        source=defaultdict(set); visits=Counter()
        for item in seq:
            cat=categories[item]
            if not cat: continue
            if visits[cat]<CONFIG['cold_visits']:
                totals[cat]+=1
                for a in attrs[item]: hits[cat,a]+=1
                for (src,a),items in source.items():
                    if src!=cat and len(items)>=CONFIG['source_distinct']:
                        q=counts[src,cat,a]; q[0]+=1; q[1]+=a in attrs[item]; q[2].add(uid)
            visits[cat]+=1
            for a in attrs[item]: source[cat,a].add(item)
    boundaries=[]
    for (src,dst,a),(n,k,users) in sorted(counts.items()):
        p0=(hits[dst,a]+1)/(totals[dst]+2)
        p=(k+CONFIG['prior_strength']*p0)/(n+CONFIG['prior_strength'])
        if len(users)<CONFIG['min_boundary_users'] or k<CONFIG['min_boundary_hits'] or p<=p0: continue
        weight=min(1.,math.log(p/p0)/math.log(4))*n/(n+CONFIG['prior_strength'])
        boundaries.append(dict(source=src,destination=dst,attribute=a,weight=weight,
            population_users=len(users),opportunities=n,cooccurrences=k,background=p0,posterior=p))
    return dict(categories={str(k):v for k,v in categories.items()},
                attributes={str(k):sorted(v) for k,v in attrs.items()},boundaries=boundaries,
                config=CONFIG,audit=dict(training_users=len(prefixes),training_events=sum(map(len,prefixes.values())),
                attributes=len(vocab),boundaries=len(boundaries),excluded_users=len(excluded_users)))

def review_event(uid,index,item,timestamp,rating):
    return dict(event_id=f'{uid}:{index}',index=index,item=int(item),timestamp=int(timestamp),
        behavior='review',rating=None if rating is None else float(rating),origin='observed_review_log',
        exposure_source='unknown',recommendation_id=None,transfer_hypothesis_id=None,
        parent_evidence_id=None)

def evidence_direction(event):
    if event.get('origin') in ('simulated','inferred','agent_generated'):
        return 'unsupported_synthetic',0.
    rating=event.get('rating')
    if rating is not None and rating<=2: return 'weak_item_counterevidence',CONFIG['low_rating_weight']
    if rating is not None and rating>=4: return 'support',1.
    return 'weak_support',.25 if rating is not None else .5

class PreferenceEvidenceTrajectory:
    def __init__(self,model):
        self.model=model
        self.categories={int(i):c for i,c in model['categories'].items()}
        self.attrs={int(i):set(a) for i,a in model['attributes'].items()}
        self.events=[]; self.operations=[]; self.revoked=set(); self._ids={}; self._snapshot=None; self._assessment_cache={}

    def observe(self,event):
        e=copy.deepcopy(event); eid=e['event_id']
        if eid in self._ids:
            if self._ids[eid]!=e: raise ValueError('same event ID with different content')
            return
        if self.events and e['timestamp']<self.events[-1]['timestamp']:
            raise ValueError('events must be chronological')
        if e['index']!=len(self.events): raise ValueError('noncontiguous observation index')
        self.events.append(e); self._ids[eid]=e
        self.operations.append(dict(operation='observe',event_id=eid,event=e))
        self._snapshot=None; self._assessment_cache={}

    def revoke(self,event_id,reason):
        if event_id not in self._ids: raise KeyError(event_id)
        if not reason: raise ValueError('revocation requires reason')
        self.revoked.add(event_id)
        self.operations.append(dict(operation='revoke',event_id=event_id,reason=reason))
        self._snapshot=None; self._assessment_cache={}

    def restore(self,event_id,reason):
        if event_id not in self.revoked: raise ValueError('event is not revoked')
        self.revoked.remove(event_id)
        self.operations.append(dict(operation='restore',event_id=event_id,reason=reason))
        self._snapshot=None; self._assessment_cache={}

    @classmethod
    def replay(cls,model,operations):
        obj=cls(model)
        for op in operations:
            if op['operation']=='observe': obj.observe(op['event'])
            else: getattr(obj,op['operation'])(op['event_id'],op['reason'])
        return obj

    @property
    def history(self): return [e['item'] for e in self.events]

    def snapshot(self):
        if self._snapshot is not None: return self._snapshot
        now=self.events[-1]['timestamp'] if self.events else 0
        visits=Counter(); active=[]; roots=set()
        for e in self.events:
            if e['event_id'] in self.revoked: continue
            cat=self.categories[e['item']]
            if cat: visits[cat]+=1
            root=e.get('parent_evidence_id') or e['event_id']
            if root in roots: continue
            roots.add(root)
            direction,weight=evidence_direction(e)
            active.append((cat,e,direction,weight))
        hypotheses={}
        for b in self.model['boundaries']:
            key=(b['source'],b['destination'],b['attribute'])
            src=[]; seen_items=set(); activation=None
            for cat,e,direction,weight in active:
                if cat!=b['source'] or b['attribute'] not in self.attrs[e['item']]: continue
                if e['item'] in seen_items: continue
                seen_items.add(e['item'])
                if direction in ('support','weak_support','weak_item_counterevidence'):
                    src.append(dict(**e,direction=direction,evidence_weight=weight))
                positive_items={x['item'] for x in src if x['direction'] in ('support','weak_support')}
                if len(positive_items)>=CONFIG['source_distinct'] and activation is None:
                    activation=e['timestamp']
            if activation is None: continue
            dst=[]
            for cat,e,direction,weight in active:
                if cat!=b['destination'] or e['timestamp']<activation: continue
                matches=b['attribute'] in self.attrs[e['item']]
                d=direction if matches else 'uninformative'
                dst.append(dict(**e,direction=d,evidence_weight=weight if matches else 0.,match=matches))
            hypotheses[key]=dict(**b,key=list(key),hypothesis_id=digest(key)[:16],source_evidence=src,
                target_evidence=dst,as_of=now,
                uninformative_target_events=sum(not e['match'] for e in dst))
        version=digest(self.operations)
        self._snapshot=dict(version=version,as_of=now,visits=visits,hypotheses=hypotheses,
                            events=len(self.events),revoked=sorted(self.revoked))
        return self._snapshot

    @property
    def visits(self): return self.snapshot()['visits']

    @property
    def hypotheses(self): return self.snapshot()['hypotheses']

    def _summary(self,evidence,now,use_time=True,use_counter=True):
        positive={}; negative={}
        for e in evidence:
            if e['direction'] not in ('support','weak_support','weak_item_counterevidence'):
                continue
            days=max(0,(now-e['timestamp'])/86400)
            decay=2**(-days/CONFIG['half_life_days']) if use_time else 1.
            if e['direction']=='weak_item_counterevidence':
                if use_counter: negative[e['item']]=max(negative.get(e['item'],0),e['evidence_weight']*decay)
            else: positive[e['item']]=max(positive.get(e['item'],0),e['evidence_weight']*decay)
        times=sorted({e['timestamp'] for e in evidence if e['direction'] in ('support','weak_support')})
        episodes=0; last=None
        for t in times:
            if last is None or t-last>=CONFIG['episode_days']*86400:
                episodes+=1; last=t
        span=(times[-1]-times[0])/86400 if times else 0.
        recency=(now-times[-1])/86400 if times else None
        temporal='unknown' if not times else 'dormant' if recency>365 else 'stable' if episodes>=2 else 'transient'
        return dict(positive=sum(positive.values()),negative=sum(negative.values()),
                    distinct_positive=len(positive),episodes=episodes,span_days=span,
                    recency_days=recency,temporal_state=temporal)

    def assess(self,key,variant='full'):
        ck=(tuple(key),variant,self.snapshot()['version'])
        if ck in self._assessment_cache: return self._assessment_cache[ck]
        h=self.hypotheses[key]
        use_time=variant not in ('no_time','no_ledger')
        use_counter=variant not in ('no_counter','no_ledger')
        src=self._summary(h['source_evidence'],h['as_of'],use_time,use_counter)
        dst=self._summary(h['target_evidence'],h['as_of'],use_time,use_counter)
        if src['distinct_positive']<CONFIG['source_distinct']: state='insufficient'
        elif use_counter and ((src['negative']>=.6 and src['negative']>=src['positive']) or
                               (dst['negative']>=.6 and dst['negative']>=dst['positive'])): state='withdrawn'
        elif dst['distinct_positive']>=2 and (dst['episodes']>=2 or not use_time): state='confirmed'
        else: state='pending'
        # These are interpretable heuristic evidence factors, NOT posterior probabilities.
        source_reliability=min(1.,src['distinct_positive']/3)*(1+src['positive'])/(2+src['positive']+2*src['negative'])
        temporal=.35 if use_time and src['episodes']<2 else 1.
        if use_time and src['recency_days'] is not None:
            temporal*=2**(-src['recency_days']/365)
        destination=.5+.5*dst['positive']/(1+dst['positive'])
        destination/=1+2*dst['negative']
        gate=source_reliability*temporal*destination
        allowed=state not in ('withdrawn','insufficient')
        if variant=='no_ledger':
            # Counts-only gate: no temporal or signed evidence state.
            gate=min(1.,src['distinct_positive']/3)*(.5+.5*dst['distinct_positive']/(1+dst['distinct_positive']))
        if variant=='no_gate': gate=1.
        if variant=='static':
            allowed=len({e['item'] for e in h['source_evidence']})>=2; gate=1.
        effective=h['weight']*gate if allowed and gate>=CONFIG['min_gate'] else 0.
        result=dict(state=state,source=src,destination=dst,source_reliability=source_reliability,
            temporal_factor=temporal,destination_factor=destination,gate=gate,
            population_prior=h['weight'],effective_weight=effective,
            ledger_version=self.snapshot()['version'])
        self._assessment_cache[ck]=result
        return result

    def score(self,candidates,variant='full'):
        scores=[]; contributions=[]
        assessments={k:self.assess(k,variant) for k in self.hypotheses}
        for item in candidates:
            cat=self.categories[item]; per_attribute={}
            if cat and self.visits[cat]<CONFIG['cold_visits']:
                for key,h in self.hypotheses.items():
                    if h['destination']!=cat or h['attribute'] not in self.attrs[item]: continue
                    a=assessments[key]; weight=a['effective_weight']
                    if not weight: continue
                    old=per_attribute.get(h['attribute'])
                    if old is None or weight>old['weight']:
                        per_attribute[h['attribute']]=dict(key=list(key),weight=weight,assessment=a)
            value=max([v['weight'] for v in per_attribute.values()] or [0.])
            scores.append(value); contributions.append(dict(item=item,score=value,attributes=per_attribute))
        return scores,contributions

    def trace(self):
        s=self.snapshot()
        return dict(version=s['version'],as_of=s['as_of'],visits=dict(s['visits']),
             revoked=s['revoked'],operations=copy.deepcopy(self.operations),
             hypotheses=[dict(**h,assessment=self.assess(k)) for k,h in self.hypotheses.items()])


# Keep existing experiment entry points and saved workflows readable.
PreferenceEvidenceTrace = PreferenceEvidenceTrajectory
HypothesisLedger = PreferenceEvidenceTrajectory
