import numpy
from sklearn.metrics import average_precision_score, recall_score, f1_score
import torch
import numpy as np

# def precision_recall_curve(y_true, pred_scores):
#     precisions = []
#     recalls = []
#
#     # for threshold in thresholds:
#     y_pred = torch.where(pred_scores > 0.6,  torch.tensor(1), torch.tensor(0))
#     y_pred = y_pred.cpu().detach().numpy()
#     y_true = y_true.cpu().detach().numpy()
#     precision = average_precision_score(y_true=y_true, y_score=y_pred, average='macro')
#     # recall = recall_score(y_true=y_true, y_pred=y_pred, average='samples')
#     f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
#
#
#     return precision, f1


def check_inputs(targs, preds):
    '''
    Helper function for input validation.
    '''

    assert (np.shape(preds) == np.shape(targs))
    assert type(preds) is np.ndarray
    assert type(targs) is np.ndarray
    assert (np.max(preds) <= 1.0) and (np.min(preds) >= 0.0)
    assert (np.max(targs) <= 1.0) and (np.min(targs) >= 0.0)
    assert (len(np.unique(targs)) <= 2)


def compute_avg_precision(targs, preds):
    '''
    Compute average precision.

    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    '''

    check_inputs(targs, preds)

    if np.all(targs == 0):
        # If a class has zero true positives, we define average precision to be zero.
        metric_value = 0.0
    else:
        metric_value = average_precision_score(targs, preds)

    return metric_value


def compute_precision_at_k(targs, preds, k):
    '''
    Compute precision@k.

    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    k: Number of predictions to consider.
    '''

    check_inputs(targs, preds)

    classes_rel = np.flatnonzero(targs == 1)
    if len(classes_rel) == 0:
        return 0.0

    top_k_pred = np.argsort(preds)[::-1][:k]

    metric_value = float(len(np.intersect1d(top_k_pred, classes_rel))) / k

    return metric_value


def compute_recall_at_k(targs, preds, k):
    '''
    Compute recall@k.

    Parameters
    targs: Binary targets.
    preds: Predicted probability scores.
    k: Number of predictions to consider.
    '''

    check_inputs(targs, preds)

    classes_rel = np.flatnonzero(targs == 1)
    if len(classes_rel) == 0:
        return 0.0

    top_k_pred = np.argsort(preds)[::-1][:k]

    metric_value = float(len(np.intersect1d(top_k_pred, classes_rel))) / len(classes_rel)

    return metric_value