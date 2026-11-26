"""End-to-end metrics for a single relevant target per event."""
import math
from numbers import Integral

TOP_K = 20
CUTOFFS = (1, 3, 5, 10)
METRICS = [f'{name}@{k}' for name in ('HR', 'NDCG') for k in CUTOFFS]


def target_rank(order, target):
    """One-based position in this ranking, or None for a retrieval miss."""
    try:
        return order.index(target) + 1
    except ValueError:
        return None


def metric_table(rows):
    if not rows:
        raise ValueError('no completed events')
    ranks = [row['rank'] for row in rows if row['rank'] is not None]
    if any(isinstance(r, bool) or not isinstance(r, Integral) or not 1 <= r <= TOP_K for r in ranks):
        raise ValueError('invalid candidate rank')
    total = len(rows)
    values = {f'HR@{k}': sum(r <= k for r in ranks) / total for k in CUTOFFS}
    values.update({f'NDCG@{k}': math.fsum(1 / math.log2(r + 1) for r in ranks if r <= k) / total
                   for k in CUTOFFS})
    return values
