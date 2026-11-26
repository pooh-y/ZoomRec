"""Independently verify each final candidate set by sorting all eligible catalog IDs."""
import math
from collections import Counter

import numpy as np
import torch

from baseline_source import right_pad
from ranking_metrics import TOP_K
from sasrec_candidates import presentation_order


@torch.inference_mode()
def verify_phase(model, events, sequences, n_items, device, batch=512):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Candidate audit requires the frozen selected model')
    catalog = np.arange(1, n_items + 1, dtype=np.int64)
    ranks, position_counts = [], Counter()
    for start in range(0, len(events), batch):
        rows = events[start:start + batch]
        # Reconstruct the model input directly from the original sequences.
        histories = [right_pad(sequences[e['uid']]['items'][:e['index']]) for e in rows]
        full_scores = model.logits(torch.tensor(histories, device=device)).cpu().numpy()
        for row, scores in zip(rows, full_scores):
            items = sequences[row['uid']]['items']
            target = row['target']
            if row['history'] != items[:row['index']] or target != items[row['index']]:
                raise ValueError(f'History or target mismatch: {row["event_id"]}')
            eligible = np.ones(n_items + 1, dtype=bool)
            eligible[row['history']] = False
            unseen = catalog[eligible[1:]]
            indices = np.lexsort((unseen, -scores[unseen]))
            expected = unseen[indices[:TOP_K]].tolist()
            shown = [expected[i] for i in presentation_order(row['event_id'], len(expected))]
            if len(expected) != TOP_K or row['candidates'] != shown:
                raise ValueError(f'Candidates differ from the shuffled unseen top 20: {row["event_id"]}')
            retained = row['sasrec_scores']
            if len(retained) != TOP_K or not np.allclose(retained, scores[shown], rtol=1e-5, atol=1e-6):
                raise ValueError(f'Raw SASRec scores differ from the displayed items: {row["event_id"]}')
            rank = next((pos for pos, item in enumerate(expected, 1) if item == target), None)
            ranks.append(rank)
            if rank is not None:
                position_counts[rank] += 1
    return dict(verified_queries=len(events), exact_top20_matches=len(ranks),
                retrieved_targets=sum(position_counts.values()), missed_targets=ranks.count(None),
                reference='unseen catalog IDs, numpy.lexsort by descending raw score then ascending item ID',
                sasrec_validation_or_test_ndcg10=math.fsum(
                    1 / math.log2(r + 1) for r in ranks if r is not None and r <= 10) / len(ranks),
                target_position_histogram=dict(sorted(position_counts.items())))
