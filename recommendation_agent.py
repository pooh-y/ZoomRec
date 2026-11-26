"""Evidence retrieval and adjacent, evidence-gated rank updates."""
from __future__ import annotations
import json, math

AGENT_CONFIG=dict(seed=20261101, max_hypotheses=12, max_retrievals=6,
                  max_verifications=8,batch_size=8,
                  plan_tokens=256,decision_tokens=2048,max_context_tokens=12000,do_sample=False)
PLAN_INSTRUCTION='''You are a recommendation agent deciding whether a broad category-level interest should influence fine-grained item ranking. The next purchase and category are UNKNOWN. Training-population co-occurrence proposes a hypothesis, and matching title attributes identify candidate items; neither proves this user's preference transfers. Request only useful, already-observed evidence and candidate metadata. Return ONLY JSON: {"retrieve_history":["H1"],"verify_items":[123],"reason":"brief rationale"}. Empty lists are allowed. Do not invent IDs.'''
DECISION_INSTRUCTION='''You are a recommendation agent deciding whether an evidence-backed transfer may affect the current ranking. The hidden target/category is unavailable. Approve only when the retrieved source evidence, the preference evidence trajectory state and the verified candidate support the same user-specific hypothesis. Missing target evidence is uncertainty, not dislike; weak item counterevidence may reduce a gate. Shared spelling alone is insufficient. Each approval must cite one retrieved hypothesis, a verified candidate and its immediate baseline competitor. Return ONLY JSON: {"transfers":[{"item":123,"hypothesis":"H1","competitor":456,"confidence":"medium","reason":"brief rationale"}],"reason":"brief overall rationale"}.'''
CLAIM_INSTRUCTION=''' Claims are scoped statements. A category-level claim (attribute="*") authorizes only the independently supported category and requires no matching title word. An attribute-level claim also requires the verified title-token proxy in its destination category. Respect the claim content and the granularity policy. Title tokens and low item ratings do not establish semantic or causal attribute truth. If policy is fixed_category_then_attribute, inspect category claims first then attribute claims; retain both only if separately supported. If evidence_adaptive, consider only the most-specific claim with its own support. Never infer an ancestor claim from its child.'''
ACTION_INSTRUCTION=''' Only items listed in options are eligible for promotion. baseline_top5 is background context, not an additional action list. The first-ranked item cannot be promoted. For each transfer, copy the exact immediate competitor supplied with that option; never invent or substitute a competitor. Empty transfers are allowed.'''
MERGE_INSTRUCTION=''' Multiple independently supported hypotheses may cite the same item. Return one record per item-hypothesis pair; each record must independently satisfy the evidence checks. Each independent approval can trigger one swap with the current neighbor above the item if its score clears that neighbor. Further approvals can move it up again. The cited competitor identifies the initial baseline comparison. A low-confidence or rejected claim does not invalidate a different supported claim.'''

def compact(x): return json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def parse_object(raw):
    if not isinstance(raw,str): raise ValueError('response is not text')
    a,b=raw.find('{'),raw.rfind('}')
    if a<0 or b<a: raise ValueError('missing JSON object')
    x=json.loads(raw[a:b+1])
    if not isinstance(x,dict): raise ValueError('response is not an object')
    return x

def make_context(history,candidates,baseline,gate,memory,cards, *, include_first=False):
    assert len(set(candidates))==len(candidates)==len(baseline)
    base={int(i):float(s) for i,s in zip(candidates,baseline)}
    order=sorted(base,key=lambda i:(-base[i],i)); ranks={i:j+1 for j,i in enumerate(order)}
    # Present eligible options in the frozen shuffle. Ranks and neighbors still use the base scores.
    _,contrib=memory.score(candidates)
    options = [row for row in contrib if row['score'] > 0 and (include_first or ranks[row['item']] > 1)]
    fixed=hasattr(memory,'mechanisms') and not memory.mechanisms.adaptive_granularity
    def claim_order(a):
        key=tuple(a['key'])
        stage=0 if memory.hypotheses[key].get('level')=='category' else 1
        return ((stage,) if fixed else ())+(-a['weight'],key)
    keys=[]
    for x in options:
        for a in sorted(x['attributes'].values(),key=claim_order):
            key=tuple(a['key'])
            if key not in keys: keys.append(key)
    ids={k:f'H{j+1}' for j,k in enumerate(keys)}; hypotheses={ids[k]:memory.hypotheses[k] for k in keys}
    def card(i): return dict(item=i,title=cards[i]['title'][:180],category=memory.categories[i],rank=ranks[i],score=round(base[i],5))
    rows=[]; weights={}
    for x in options:
        i=x['item']; avail=[]
        for a in sorted(x['attributes'].values(),key=claim_order):
            k=tuple(a['key'])
            if k in ids: avail.append(ids[k]); weights[(i,k)]=float(a['weight'])
        if avail:
            competitor = card(order[ranks[i]-2]) if ranks[i] > 1 else None
            rows.append(dict(**card(i),hypotheses=avail,competitor=competitor,prior_category_visits=memory.visits[memory.categories[i]]))
    by_index={e['index']:e for e in memory.events}; recent=[]
    for j,i in list(enumerate(history))[-5:]:
        e=by_index[j]; recent.append(dict(index=j,item=i,title=cards[i]['title'][:160],category=memory.categories[i],timestamp=e['timestamp'],rating=e['rating'],behavior=e['behavior']))
    hs=[]
    for k,h in hypotheses.items():
        h['_key']=tuple(h['key']); a=memory.assess(h['_key']); h['_ledger_assessment']=a
        hs.append(dict(id=k,source=h['source'],destination=h['destination'],attribute=h['attribute'],level=h.get('level','attribute'),content=h.get('content'),parents=h.get('parents',[]),status=a['state'],distinct_source_items=a['source']['distinct_positive'],source_episodes=a['source']['episodes'],temporal_state=a['source']['temporal_state'],effective_transfer_weight=round(a['effective_weight'],5),population_prior=round(h['weight'],5),population_users=h.get('population_users',0),destination_support=round(a['destination']['positive'],4),destination_counterevidence=round(a['destination']['negative'],4),uninformative_target_events=h['uninformative_target_events']))
    summary=dict(history_length=len(history),recent_history=recent,baseline_top5=[card(i) for i in order[:5]],sasrec_uncertainty=round(float(gate),4),options=rows,hypotheses=hs)
    if hasattr(memory,'mechanisms'):
        summary['granularity_policy']='evidence_adaptive' if memory.mechanisms.adaptive_granularity else 'fixed_category_then_attribute'
    return dict(summary=summary,hypotheses=hypotheses,weights=weights,baseline=base,order=order,history=list(history),gate=float(gate),memory=memory)

def evidence_pages(context):
    """Group eligible items and hypotheses for review before the planner selects evidence."""
    summary = context['summary']
    options = summary['options']
    item_budget = AGENT_CONFIG['max_verifications']
    claim_budget = AGENT_CONFIG['max_hypotheses']
    for first in range(0, len(options), item_budget):
        group = options[first:first + item_budget]
        ids = list(dict.fromkeys(hid for row in group for hid in row['hypotheses']))
        for start in range(0, len(ids), claim_budget):
            page_ids = ids[start:start + claim_budget]
            hypotheses = {hid: context['hypotheses'][hid] for hid in page_ids}
            rows = []
            for option in group:
                matching = [hid for hid in option['hypotheses'] if hid in hypotheses]
                if matching:
                    rows.append(dict(option, hypotheses=matching))
            yield dict(context, hypotheses=hypotheses, summary=dict(summary, options=rows,
                       hypotheses=[row for row in summary['hypotheses'] if row['id'] in hypotheses]))


def plan_prompt(context):
    limits = (f" Select at most {AGENT_CONFIG['max_retrievals']} of the listed hypotheses for history retrieval"
              f" and at most {AGENT_CONFIG['max_verifications']} items from options for verification.")
    return PLAN_INSTRUCTION + CLAIM_INSTRUCTION + ACTION_INSTRUCTION + limits + '\nINPUT:\n' + compact(context['summary'])

def execute_plan(raw,context,cards):
    errors=[]; hs=[]; items=[]
    try:
        p=parse_object(raw); hs=p['retrieve_history']; items=p['verify_items']
        if not isinstance(hs,list) or not isinstance(items,list): raise ValueError('tool requests must be lists')
        if len(hs)>AGENT_CONFIG['max_retrievals'] or len(items)>AGENT_CONFIG['max_verifications']: raise ValueError('tool request budget exceeded')
    except (KeyError,ValueError,TypeError) as e: errors.append(str(e)); hs=[]; items=[]
    valid_hs=[]; valid_items=[]
    for hid in hs:
        if isinstance(hid,str) and hid in context['hypotheses']:
            if hid not in valid_hs: valid_hs.append(hid)
        else: errors.append('unknown hypothesis ID')
    for item in items:
        if type(item) is int and item in context['baseline']:
            if item not in valid_items: valid_items.append(item)
        else: errors.append('unknown candidate item ID')
    retrieved=[]; verified=[]
    for hid in valid_hs:
        h=context['hypotheses'][hid]; a=h['_ledger_assessment']
        def ev(e):
            i=e['item']; assert context['history'][e['index']]==i
            return dict(**e,title=cards[i]['title'][:180],categories=cards[i]['categories'],attribute_in_title_tokens=h['attribute'] in cards[i]['tokens'])
        variants={} if hasattr(context['memory'],'mechanism_variant') else {v:context['memory'].assess(h['_key'],v) for v in ('no_time','no_counter','no_ledger','no_gate','static')}
        # Deduplicate before slicing: duplicate records must not hide a recent negative.
        source=sorted({e['event_id']:e for e in h['source_evidence']}.values(),
                      key=lambda e:e['index'])
        # Include two independent supportive items, even after recent negatives.
        positive={}
        for e in reversed(source):
            if e['direction'] in ('support','weak_support') and e['item'] not in positive:
                positive[e['item']]=e
            if len(positive)==2: break
        evidence={e['event_id']:e for e in [*positive.values(),*source[-2:]]}
        source_evidence=sorted(evidence.values(),key=lambda e:e['index'])
        retrieved.append(dict(id=hid,key=list(h['_key']),source=h['source'],destination=h['destination'],attribute=h['attribute'],level=h.get('level','attribute'),content=h.get('content'),assessment=a,variant_assessments=variants,source_evidence=[ev(e) for e in source_evidence],target_evidence=[ev(e) for e in h['target_evidence'][-4:]],uninformative_target_events=h['uninformative_target_events']))
    for i in valid_items:
        matches=[hid for hid,h in context['hypotheses'].items() if (i,h['_key']) in context['weights'] and (context['memory'].matches(h['_key'],i) if hasattr(context['memory'],'matches') else h['attribute'] in cards[i]['tokens'])]
        verified.append(dict(item=i,title=cards[i]['title'],categories=cards[i]['categories'],tokens=cards[i]['tokens'],matching_hypotheses=matches))
    return dict(retrieved=retrieved,verified=verified,errors=errors)

def decision_prompt(context,observations): return DECISION_INSTRUCTION+CLAIM_INSTRUCTION+ACTION_INSTRUCTION+MERGE_INSTRUCTION+'\nCONTEXT:\n'+compact(context['summary'])+'\nTOOL_RESULTS:\n'+compact(observations)

def validate_decision(raw,context,observations):
    try:
        t=parse_object(raw)['transfers']
        if not isinstance(t,list): raise ValueError('transfers must be a list')
    except (ValueError,KeyError,TypeError) as e: return dict(approved=[],errors=[str(e)])
    hs={h['id']:h for h in observations['retrieved']}; verified={x['item']:x for x in observations['verified']}; options={x['item']:x for x in context['summary']['options']}; by_claim={}; errors=[]
    for x in t:
        try:
            if not isinstance(x,dict): raise ValueError('invalid transfer record')
            i,hid,conf,competitor=x['item'],x['hypothesis'],x['confidence'],x['competitor']
            if type(i) is not int or i not in options: raise ValueError('invalid item')
            if hid not in hs or i not in verified: raise ValueError('evidence not fetched')
            h=hs[hid]; key=tuple(h['key'])
            if hid not in verified[i]['matching_hypotheses'] or (i,key) not in context['weights']: raise ValueError('attribute verification failed')
            a=h['assessment']
            if a['state'] in ('withdrawn','insufficient') or a['effective_weight']<=0: raise ValueError('ledger gate closed')
            if len({e['item'] for e in h['source_evidence'] if e['direction'] in ('support','weak_support')})<2: raise ValueError('insufficient source evidence')
            opp=options[i]['competitor']
            if opp is None or competitor!=opp['item']: raise ValueError('incorrect first competitor')
            if conf not in ('low','medium','high'): raise ValueError('invalid confidence')
            reason=x.get('reason','')
            if not isinstance(reason,str) or not reason.strip(): raise ValueError('missing rationale')
            if conf=='low': continue
            approval=dict(item=i,hypothesis=hid,competitor=competitor,confidence=conf,reason=reason,weight=context['weights'][(i,key)],ledger_weight=a['effective_weight'],ledger_state=a['state'],ledger_version=a['ledger_version'],variant_ledger_weights={v:q['effective_weight'] for v,q in h['variant_assessments'].items()})
            pair=(i,hid)
            old=by_claim.get(pair)
            # Repeated copies of one claim add no evidence. Resolve repeated
            # confidence/rationale fields deterministically, independent of order.
            preference=lambda row:(row['confidence']!='high',row['reason'])
            if old is None or preference(approval)<preference(old): by_claim[pair]=approval
        except (KeyError,ValueError,TypeError) as e: errors.append(str(e))
    # Each distinct approval supplies one potential adjacent swap.
    return dict(approved=[by_claim[pair] for pair in sorted(by_claim)],errors=errors)

def rule_approvals(context):
    """Use every eligible match from the pre-prediction memory without an agent call."""
    matches = []
    for option in context['summary']['options']:
        item = option['item']
        for hid in option['hypotheses']:
            key = tuple(context['hypotheses'][hid]['key'])
            matches.append(dict(item=item, hypothesis=hid, ledger_weight=context['weights'][item, key],
                                source='rule'))
    return matches


def bounded_rank(candidates, baseline, model_uncertainty, approved, setting, *,
                 agent_review=True, strongest_only=False):
    """Try a strict adjacent promotion for each admitted claim."""
    base = {int(item): float(score) for item, score in zip(candidates, baseline)}
    assert len(base) == len(candidates) == len(baseline)
    assert all(math.isfinite(score) for score in base.values())
    order = sorted(base, key=lambda item: (-base[item], item))
    original = {item: pos for pos, item in enumerate(order)}
    strength = float(setting['strength'])
    claims = {}
    for row in approved:
        item = row['item']
        if item not in base:
            raise ValueError('approval outside candidates')
        if agent_review:
            confidence = row['confidence']
            if confidence != 'high' and not (confidence == setting['min_confidence'] == 'medium'):
                continue
        elif row.get('source') != 'rule':
            raise ValueError('Expected a rule-qualified preference match')
        weight = row['ledger_weight'] if 'ledger_weight' in row else row['weight']
        bonus = strength * float(weight)
        if bonus > 0:
            # Legacy direct callers can supply a single unnamed claim per item.
            hypothesis = row.get('hypothesis', '')
            by_id = claims.setdefault(item, {})
            by_id[hypothesis] = max(bonus, by_id.get(hypothesis, 0.))

    bonuses = {item: max(by_id.values()) for item, by_id in claims.items()}
    adjusted = {item: base[item] + bonuses.get(item, 0.) for item in order}
    crossings = []
    for item in sorted(claims, key=lambda item: (-adjusted[item], item)):
        pos = order.index(item)
        # Try weaker claims first so stronger ones can clear a later, larger gap.
        attempts = sorted(claims[item].items(), key=lambda pair: (pair[1], pair[0]))
        if strongest_only:
            attempts = attempts[-1:]
        for hypothesis, bonus in attempts:
            if pos == 0:
                break
            neighbor = order[pos - 1]
            if base[item] + bonus <= adjusted[neighbor]:
                continue
            crossings.append(dict(promoted=item, displaced=neighbor, hypothesis=hypothesis,
                                  from_rank=pos + 1, to_rank=pos,
                                  original_score_gap=base[neighbor] - base[item], bonus=bonus))
            order[pos - 1], order[pos] = item, neighbor
            pos -= 1
    displacement = max((abs(pos - original[item]) for pos, item in enumerate(order)), default=0)
    return order, dict(crossings=crossings, bonuses=bonuses, max_displacement=displacement,
                       model_uncertainty=model_uncertainty)


def calibration_grid():
    return [dict(strength=0., min_confidence='high')] + [
        dict(strength=strength, min_confidence=confidence)
        for strength in (.25, .5, 1.) for confidence in ('high', 'medium')]
