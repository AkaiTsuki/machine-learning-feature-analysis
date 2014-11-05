__author__ = 'jiachiliu'

import numpy as np


def auc(x, y):
    """
    Compute the area under curve
    :param x: x coordinate
    :param y: y coordinate
    :return: the area under curve
    """
    sort = np.lexsort((y, x))
    x = x[sort]
    y = y[sort]
    return np.trapz(y, x)


def roc(y_true, y_score, positive_label=1, negative_label=0):
    """
    Given the true label and predict score, return points coordinate represented by true positive and false positive
    :param y_true: actual label
    :param y_score: predicted score(log odds, probabilities, weighted hypothesis)
    :return: a list of point
    """
    positive = 1.0 * len(y_true[y_true == positive_label])
    negative = 1.0 * len(y_true[y_true == negative_label])
    sorted_y_score_indices = y_score.argsort()[::-1]
    sorted_y_true = y_true[sorted_y_score_indices]
    tp = 0
    fp = 0
    points = [[tp, fp]]
    for actual_label in sorted_y_true:
        if actual_label == positive_label:
            tp += 1
        else:
            fp += 1
        points.append([tp / positive, fp / negative])
    return np.array(points)

