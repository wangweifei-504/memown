import numpy as np
import pandas as pd
import torch

from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics.ranking import _binary_clf_curve
from tqdm import tqdm


def range_lift_with_delay(array: np.ndarray, label: np.ndarray, delay=None, inplace=False) -> np.ndarray:
    """
    :param delay: maximum acceptable delay
    :param array:
    :param label:
    :param inplace:
    :return: new_array
    """
    assert np.shape(array) == np.shape(label)
    if delay is None:
        delay = len(array)
    splits = np.where(label[1:] != label[:-1])[0] + 1
    is_anomaly = label[0] == 1
    new_array = np.copy(array) if not inplace else array
    pos = 0
    for sp in splits:
        if is_anomaly:
            ptr = min(pos + delay + 1, sp)
            new_array[pos: ptr] = np.max(new_array[pos: ptr])
            new_array[ptr: sp] = np.maximum(new_array[ptr: sp], new_array[pos])
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:
        ptr = min(pos + delay + 1, sp)
        new_array[pos: sp] = np.max(new_array[pos: ptr])
    return new_array


def adjust_predicts(score, label, data_category):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):

    Returns:
        np.ndarray: predict labels
    """
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    new_array = np.copy(score)
    label = np.asarray(label)
    # predict = score > threshold
    # predict = predict.astype(int)
    actual = label == 1
    # print("label::::::::::::::")
    # print(label)
    # print("score::::::::::::::")
    # print(score)
    anomaly_count = 0
    max_score = 0.0
    for i in tqdm(range(len(score))):
        if actual[i]:
            max_score = new_array[i]
            anomaly_count += 1
            for j in range(i - 1, -1, -1):
                if not actual[j]:
                    new_array[j + 1:i + 1] = max_score
                    break
                else:
                    if new_array[j] > max_score:
                        max_score = new_array[j]
    return new_array


def calculate_performance(y_true, y_prob, delay=7, verbose=True, data_category="MSL"):
    # print()
    # print(y_true)
    roc_ori = roc_auc_score(y_true, y_prob)
    mAP_ori = average_precision_score(y_true, y_prob)

    print("begin adjust::::::::::::::::::::::")
    # y_prob = range_lift_with_delay(y_prob, y_true, delay=delay)

    precisions_ori, recalls_ori, thresholds_ori = precision_recall_curve(y_true, y_prob)
    f1_ori = (2 * precisions_ori * recalls_ori) / (precisions_ori + recalls_ori)
    f1_ind_ori = np.nanargmax(f1_ori)

    y_prob = adjust_predicts(y_prob, y_true, data_category)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_ind = np.nanargmax(f1_scores)

    roc = roc_auc_score(y_true, y_prob)
    mAP = average_precision_score(y_true, y_prob)

    performance = {'PR_ORI': precisions_ori[f1_ind_ori], 'REC_ORI': [recalls_ori[f1_ind_ori]],
                   'F1_ORI': [f1_ori[f1_ind_ori]],
                   'PR': [precisions[f1_ind]], 'REC': [recalls[f1_ind]], 'F1': [f1_scores[f1_ind]],
                   'ROC_ORI': [roc_ori], 'mAP_ORI': [mAP_ori], 'ROC': [roc], 'mAP': [mAP]}
    performance = pd.DataFrame(performance)

    if verbose:
        print(performance)

    return performance


def smd_calculate_performance(y_true, y_prob, verbose=True, data_category='machine-1-1'):
    fps_ori, tps_ori, thresholds_ori = _binary_clf_curve(y_true, y_prob)
    total_pos_ori = tps_ori[-1]
    last_ind_ori = tps_ori.searchsorted(tps_ori[-1])
    sl_ori = slice(last_ind_ori, None, -1)
    fps_ori = fps_ori[sl_ori]
    tps_ori = tps_ori[sl_ori]
    precisions_ori = tps_ori / (fps_ori + tps_ori)
    recalls_ori = tps_ori / total_pos_ori
    f1_scores_ori = (2 * precisions_ori * recalls_ori) / (precisions_ori + recalls_ori)
    f1_ind_ori = np.nanargmax(f1_scores_ori)
    fp_ori = fps_ori[f1_ind_ori]
    tp_ori = tps_ori[f1_ind_ori]
    fn_ori = total_pos_ori - tps_ori[f1_ind_ori]

    y_prob_adjust = adjust_predicts(y_prob, y_true, data_category)
    fps, tps, thresholds = _binary_clf_curve(y_true, y_prob_adjust)
    total_pos = tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    fps = fps[sl]
    tps = tps[sl]
    precisions = tps / (fps + tps)
    recalls = tps / total_pos
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    f1_ind = np.nanargmax(f1_scores)
    # print("f1max::::::::::::::::::::::::::::::::::::::::::")
    # print(f1_scores[np.nanargmax(f1_scores)])
    fp = fps[f1_ind]
    tp = tps[f1_ind]
    fn = total_pos - tps[f1_ind]
    # print("metric:::::::::::::::::::::::::::::")
    # pr = tp/(tp+fp)
    # reca = tp/(tp+fn)
    # f1 = (2 * pr * reca) / (pr + reca)
    # print(pr)
    # print(reca)
    # print(f1)
    performance = {'FP_ORI': fp_ori, 'TP_ORI': tp_ori, 'FN_ORI': fn_ori, 'F1_ORI': f1_scores_ori[f1_ind_ori], 'FP': fp,
                   'TP': tp, 'FN': fn, 'F1': f1_scores[f1_ind]}
    performance = pd.DataFrame(performance, index=[0])

    if verbose:
        print(performance)

    return performance
