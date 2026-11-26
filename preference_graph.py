"""History-only scoped preference claims for the full method and ablations.

Category claims have independent observed support. Attribute claims depend on
source-category evidence and a frozen population proposal, never on child-to-
parent inference. Title tokens are proxies, not semantic attribute labels.
"""
from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import asdict, dataclass

from ledger import CONFIG, PreferenceEvidenceTrajectory, digest, evidence_direction


@dataclass(frozen=True)
class Mechanisms:
    adaptive_granularity: bool
    local_revision: bool
    recovery: bool


VARIANTS = {
    'no_adaptive_granularity': Mechanisms(False, True, True),
    'no_revision': Mechanisms(True, False, True),
    'no_recovery': Mechanisms(True, True, False),
    'no_both': Mechanisms(False, False, False),
}
ALL_VARIANTS = {'full': Mechanisms(True, True, True), **VARIANTS}


class ScopedPreferenceMemory:
    """Replay every observed prefix, including withdrawals before test time.

    Local editing narrows only a contradicted attribute claim's destination
    scope to its source category. Two distinct subsequent supportive destination
    items can reopen that scope. Item-level low ratings are weak evidence, not
    proof of attribute causality. Withdrawal and recovery are separate from edits.
    """

    def __init__(self, model, variant):
        self.mechanism_variant = variant
        self.mechanisms = ALL_VARIANTS[variant]
        self.model = model
        self.categories = {int(i): c for i, c in model['categories'].items()}
        self.attrs = {int(i): set(a) for i, a in model['attributes'].items()}
        self.events = []
        self.operations = []
        self._ids = {}
        self._category = defaultdict(list)
        self._source = defaultdict(list)
        self._by_source = defaultdict(list)
        self._by_destination = defaultdict(list)
        for boundary in model['boundaries']:
            key = (boundary['source'], boundary['destination'], boundary['attribute'])
            self._by_source[(key[0], key[2])].append((key, boundary))
        self.hypotheses = {}
        self._states = {}
        self._assessments = {}
        self._version = None
        self._visits = defaultdict(int)
        self.permanently_withdrawn = set()
        self.ever_withdrawn = set()
        self.transitions = []
        self.revisions = []

    @property
    def history(self):
        return [e['item'] for e in self.events]

    @property
    def visits(self):
        return self._visits

    @property
    def version(self):
        if self._version is None:
            self._version = digest(dict(variant=self.mechanism_variant,
                                        operations=self.operations))
        return self._version

    def _new_claim(self, key, boundary, evidence, level, timestamp):
        source, destination, attribute = key
        parents = [] if level == 'category' else [
            ['@category', source, '*'], ['@category', destination, '*']]
        # Match the original ledger's timestamp >= activation boundary, using
        # only events that have already been observed. Tied timestamps can
        # precede the activating event in the sequence without being older.
        target_evidence = [] if level == 'category' else [
            self._target_record(e, attribute) for e in self._category[destination]
            if e['timestamp'] >= timestamp]
        content = dict(level=level, attribute=None if level == 'category' else attribute,
                       applicable_categories=sorted({destination} if level == 'category'
                                                    else {source, destination}))
        h = dict(**boundary, key=list(key), hypothesis_id=digest(key)[:16],
                 level=level, parents=parents, source_evidence=copy.deepcopy(evidence),
                 target_evidence=target_evidence, activation=timestamp, as_of=timestamp,
                 uninformative_target_events=sum(not e['match'] for e in target_evidence), content=content,
                 original_content=copy.deepcopy(content), last_counter_index=None,
                 scope_reopened_through=None)
        self.hypotheses[key] = h
        if level == 'attribute':
            self._by_destination[destination].append(key)
        return h

    def _target_record(self, evidence, attribute):
        matches = attribute in self.attrs[evidence['item']]
        return dict(evidence, match=matches,
                    direction=evidence['direction'] if matches else 'uninformative',
                    evidence_weight=evidence['evidence_weight'] if matches else 0.)

    def observe(self, event):
        e = copy.deepcopy(event)
        if e['event_id'] in self._ids:
            if self._ids[e['event_id']] != e:
                raise ValueError('same event ID with different content')
            return
        if e['index'] != len(self.events):
            raise ValueError('noncontiguous observation index')
        if self.events and e['timestamp'] < self.events[-1]['timestamp']:
            raise ValueError('events must be chronological')
        if e.get('origin') != 'observed_review_log' or e.get('parent_evidence_id'):
            raise ValueError('only independent observed review events are accepted')
        self.events.append(e)
        self._ids[e['event_id']] = e
        self.operations.append(dict(operation='observe', event=e))
        self._version = None
        self._assessments = {}
        category = self.categories[e['item']]
        if category:
            self._visits[category] += 1
        direction, weight = evidence_direction(e)
        evidence = dict(**e, direction=direction, evidence_weight=weight)
        if category:
            self._category[category].append(evidence)
            ck = ('@category', category, '*')
            if ck not in self.hypotheses:
                self._new_claim(ck, dict(source=category, destination=category,
                                        attribute='*', weight=1., population_users=0),
                                self._category[category], 'category', e['timestamp'])
            else:
                self.hypotheses[ck]['source_evidence'].append(evidence)
            # Existing destination claims see this event, including neutral ones.
            for key in self._by_destination[category]:
                h = self.hypotheses[key]
                target = self._target_record(evidence, h['attribute'])
                h['target_evidence'].append(target)
                if not target['match']:
                    h['uninformative_target_events'] += 1
            for attribute in sorted(self.attrs[e['item']]):
                sk = (category, attribute)
                self._source[sk].append(evidence)
                positives = {x['item'] for x in self._source[sk]
                             if x['direction'] in ('support', 'weak_support')}
                if len(positives) < CONFIG['source_distinct']:
                    continue
                for key, boundary in self._by_source[sk]:
                    if key not in self.hypotheses:
                        self._new_claim(key, boundary, self._source[sk], 'attribute',
                                        e['timestamp'])
                    else:
                        # Claims own their lists; subsequent evidence must be
                        # added explicitly rather than through a shared alias.
                        self.hypotheses[key]['source_evidence'].append(evidence)
        # Process transitions at every prefix, never only when queried/scored.
        for key, h in self.hypotheses.items():
            h['as_of'] = e['timestamp']
            self._update_scope(h, e)
            assessment = self._raw_assessment(h)
            raw_state = assessment['state']
            previous = self._states.get(key, 'insufficient')
            if raw_state == 'withdrawn':
                self.ever_withdrawn.add(key)
                if not self.mechanisms.recovery:
                    self.permanently_withdrawn.add(key)
            state = 'withdrawn' if key in self.permanently_withdrawn else raw_state
            if previous != state:
                operation = ('recover' if previous == 'withdrawn' and
                             state in ('pending', 'confirmed') else 'state_change')
                self.transitions.append(dict(key=list(key), event_id=e['event_id'],
                    operation=operation, before=previous, after=state))
            self._states[key] = state

    def _update_scope(self, h, event):
        if not self.mechanisms.local_revision or h['level'] != 'attribute':
            return
        matching = [x for x in h['target_evidence']
                    if x['direction'] == 'weak_item_counterevidence']
        last = max((x['index'] for x in matching), default=None)
        after = [x for x in h['target_evidence'] if last is not None and
                 x['index'] > last and x['direction'] in ('support', 'weak_support')]
        reopened = len({x['item'] for x in after}) >= CONFIG['source_distinct']
        categories = {h['source'], h['destination']}
        if last is not None and not reopened:
            categories = {h['source']}
        before = copy.deepcopy(h['content'])
        h['content']['applicable_categories'] = sorted(categories)
        h['last_counter_index'] = last
        h['scope_reopened_through'] = ([x['event_id'] for x in after] if reopened else None)
        if before != h['content']:
            self.revisions.append(dict(key=h['key'], event_id=event['event_id'],
                before=before, after=copy.deepcopy(h['content']),
                evidence=[x['event_id'] for x in (after if reopened else matching[-1:])],
                reason='independent destination support' if reopened else
                       'weak local item counterevidence narrows destination scope'))

    def _raw_assessment(self, h):
        summary = PreferenceEvidenceTrajectory._summary
        src = summary(self, h['source_evidence'], h['as_of'])
        dst = summary(self, h['target_evidence'], h['as_of'])
        if src['distinct_positive'] < CONFIG['source_distinct']:
            state = 'insufficient'
        elif ((src['negative'] >= .6 and src['negative'] >= src['positive']) or
              (dst['negative'] >= .6 and dst['negative'] >= dst['positive'])):
            state = 'withdrawn'
        elif h['level'] == 'category' or (dst['distinct_positive'] >= 2 and dst['episodes'] >= 2):
            state = 'confirmed'
        else:
            state = 'pending'
        reliability = min(1., src['distinct_positive'] / 3) * (
            (1 + src['positive']) / (2 + src['positive'] + 2 * src['negative']))
        temporal = .35 if src['episodes'] < 2 else 1.
        if src['recency_days'] is not None:
            temporal *= 2 ** (-src['recency_days'] / 365)
        destination = 1. if h['level'] == 'category' else (
            (.5 + .5 * dst['positive'] / (1 + dst['positive'])) / (1 + 2 * dst['negative']))
        gate = reliability * temporal * destination
        allowed = state not in ('withdrawn', 'insufficient')
        effective = h['weight'] * gate if allowed and gate >= CONFIG['min_gate'] else 0.
        return dict(state=state, source=src, destination=dst, gate=gate,
                    population_prior=h['weight'], effective_weight=effective,
                    source_reliability=reliability, temporal_factor=temporal,
                    destination_factor=destination)

    def assess(self, key, variant=None):
        key = tuple(key)
        if variant is not None and variant != self.mechanism_variant:
            raise ValueError('a variant must assess its own memory')
        if key not in self._assessments:
            h = self.hypotheses[key]
            a = self._raw_assessment(h)
            a['state'] = self._states[key]
            a['permanently_withdrawn'] = key in self.permanently_withdrawn
            a['content_allows_destination'] = h['destination'] in h['content']['applicable_categories']
            if a['state'] == 'withdrawn' or not a['content_allows_destination']:
                a['effective_weight'] = 0.
            a['ledger_version'] = self.version
            a['fine_evidence_supported'] = (
                a['source']['distinct_positive'] >= CONFIG['source_distinct'] and
                a['source']['episodes'] >= 2)
            self._assessments[key] = a
        return self._assessments[key]

    def matches(self, key, item):
        h = self.hypotheses[tuple(key)]
        return (self.categories[item] == h['destination'] and
                (h['level'] == 'category' or h['attribute'] in self.attrs[item]) and
                h['destination'] in h['content']['applicable_categories'])

    def score(self, candidates):
        """Adaptive: most specific supported matching claim per category.

        Fixed: category stage then attribute stage, independent of evidence's
        preferred level. Both stages still obey withdrawal and validity gates.
        """
        values, contributions = [], []
        for item in candidates:
            coarse, fine = [], []
            for key, h in self.hypotheses.items():
                if not self.matches(key, item):
                    continue
                a = self.assess(key)
                if a['effective_weight'] <= 0:
                    continue
                row = dict(key=list(key), weight=a['effective_weight'], assessment=a)
                (coarse if h['level'] == 'category' else fine).append(row)
            if self.mechanisms.adaptive_granularity:
                fine = [x for x in fine if x['assessment']['fine_evidence_supported']]
                selected = fine if fine else coarse
            else:
                selected = coarse + fine
            attributes = {str(j): row for j, row in enumerate(selected)}
            value = max((x['weight'] for x in selected), default=0.)
            values.append(value)
            contributions.append(dict(item=item, score=value, attributes=attributes))
        return values, contributions

    def trace(self):
        return dict(version=self.version, as_of=self.events[-1]['timestamp'] if self.events else 0,
                    variant=self.mechanism_variant, mechanisms=asdict(self.mechanisms),
                    visits=dict(self.visits), events=len(self.events),
                    permanently_withdrawn=[list(k) for k in sorted(self.permanently_withdrawn)],
                    transitions=copy.deepcopy(self.transitions), revisions=copy.deepcopy(self.revisions),
                    hypotheses=[dict(key=list(k), level=h['level'], parents=h['parents'],
                        content=copy.deepcopy(h['content']),
                        original_content=copy.deepcopy(h['original_content']),
                        assessment=self.assess(k)) for k, h in sorted(self.hypotheses.items())])
