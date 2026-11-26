"""ZoomRec's validation-only SASRec checkpoint selection and stopping rule."""
import math

VERSION = 'sasrec-unseen-top20-ndcg10-v1'


def training_policy(min_epochs=20, batch_size=512):
    if min_epochs < 20:
        raise ValueError('SASRec must train for at least 20 epochs')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    return dict(version=VERSION, min_epochs=min_epochs, evaluation_interval=5,
        patience_evaluations=5, selection_metric='NDCG@10', selection_phase='valid',
        improvement='strictly greater than historical best', max_epochs=None,
        seed=20261101, max_len=50, hidden=64, layers=2, heads=4, dropout=.1,
        loss='full-item softmax cross entropy (padding item excluded)',
        optimizer='AdamW', lr=.0015, weight_decay=1e-5,
        batch_size=batch_size, gradient_clip_norm=1.0,
        validation_candidates='current checkpoint top 20 outside the observed prefix, ties by item ID, misses score zero',
        test_used_for_selection=False)


class EarlyStopping:
    def __init__(self, min_epochs=20):
        self.policy = training_policy(min_epochs)
        self.best_score = -math.inf
        self.best_epoch = 0
        self.bad_evaluations = 0
        self.last_epoch = 0

    def update(self, epoch, ndcg10):
        if epoch != self.last_epoch + self.policy['evaluation_interval']:
            raise ValueError('Validation must run at every five epochs, starting at epoch 5')
        if not math.isfinite(ndcg10) or not 0 <= ndcg10 <= 1:
            raise ValueError('Validation NDCG@10 must be finite and in [0, 1]')
        self.last_epoch = epoch
        improved = ndcg10 > self.best_score
        if improved:
            self.best_score, self.best_epoch, self.bad_evaluations = ndcg10, epoch, 0
        else:
            self.bad_evaluations += 1
        stop = epoch >= self.policy['min_epochs'] and self.bad_evaluations >= self.policy['patience_evaluations']
        return dict(epoch=epoch, ndcg10=ndcg10, improved=improved,
                    non_improving_evaluations=self.bad_evaluations, stop=stop)


def validate_training_record(record):
    """Replay validation decisions independently; reject stale or incomplete runs."""
    policy = record['policy']
    assert policy == training_policy(policy['min_epochs'], policy['batch_size'])
    assert record['status'] == 'complete' and record['stop_reason'] == 'patience_exhausted'
    stopper = EarlyStopping(policy['min_epochs'])
    history = record['validation_history']
    assert history
    for index, row in enumerate(history):
        expected = stopper.update(row['epoch'], row['ndcg10'])
        assert row == expected
        assert row['stop'] == (index == len(history) - 1)
    assert record['epochs_completed'] == history[-1]['epoch'] >= policy['min_epochs']
    assert record['best_epoch'] == stopper.best_epoch
    assert record['best_validation_ndcg10'] == stopper.best_score
    assert record['non_improving_evaluations'] == stopper.bad_evaluations
    assert [row['epoch'] for row in record['logs']] == list(range(1, record['epochs_completed'] + 1))
    assert all(math.isfinite(row['mean_loss']) for row in record['logs'])
    assert math.isclose(record['restored_checkpoint_validation_ndcg10'], stopper.best_score,
                        rel_tol=0, abs_tol=1e-12)
    assert record['checkpoint_restored'] and record['parameters_frozen']
    return dict(best_epoch=stopper.best_epoch, best_validation_ndcg10=stopper.best_score,
                epochs_completed=record['epochs_completed'], validations=len(history))
