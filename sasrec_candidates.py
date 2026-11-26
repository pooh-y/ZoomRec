"""Retrieve unseen items by SASRec score and shuffle their presentation per event."""
import random

import torch

from data_split import SEED
from ranking_metrics import TOP_K

POLICY = dict(version='sasrec-unseen-top20-shuffled-v1', candidate_count=TOP_K,
              eligibility='real catalog items outside the full observed prefix at prediction time',
              selection='descending raw SASRec logit, then ascending item ID',
              presentation='shuffle item-score pairs using the run seed and event ID', seed=SEED,
              validation='current epoch SASRec in eval mode at every validation',
              test='frozen best validation checkpoint SASRec',
              miss='absent target has null rank and contributes zero to HR and NDCG at every cutoff')


def top_candidates(logits, histories):
    """Rank unseen items. The exclusion prefix can be longer than the encoder input."""
    if not torch.isfinite(logits[:, 1:]).all():
        raise ValueError('Nonfinite SASRec catalog scores')
    width = max(map(len, histories))
    blocked = [history + [0] * (width - len(history)) for history in histories]
    blocked = torch.tensor(blocked, dtype=torch.long, device=logits.device)
    scores = logits.clone()
    scores[:, 0] = -torch.inf
    scores.scatter_(1, blocked, -torch.inf)
    # Columns are item IDs, so a stable sort breaks ties by ascending ID.
    items = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :TOP_K]
    if items.shape[1] != TOP_K or not torch.isfinite(scores.gather(1, items)).all():
        raise ValueError(f'Fewer than {TOP_K} unseen catalog items for an event')
    return items


def presentation_order(event_id, count):
    positions = list(range(count))
    random.Random(f'{SEED}:{event_id}').shuffle(positions)
    return positions


@torch.inference_mode()
def materialize_events(model, queries, device, batch=512):
    from baseline_source import right_pad

    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Materialization requires the frozen selected checkpoint in eval mode')
    events = []
    for start in range(0, len(queries), batch):
        rows = queries[start:start + batch]
        histories = [row['history'] for row in rows]
        encoder_input = torch.tensor([right_pad(history) for history in histories], device=device)
        logits = model.logits(encoder_input)
        items = top_candidates(logits, histories)
        raw_scores = logits.gather(1, items).tolist()
        for row, ids, scores in zip(rows, items.tolist(), raw_scores):
            order = presentation_order(row['event_id'], len(ids))
            events.append(dict(row, candidates=[ids[i] for i in order],
                               sasrec_scores=[scores[i] for i in order]))
    return events
