__author__ = 'jiachiliu'

import math
import numpy as np
import matplotlib.pyplot as plt



def mse(pred, act):
    mse = 0.0
    E = abs(pred - act)
    for i in range(len(E)):
        mse += E[i] ** 2
    return mse / len(E)


def rmse(pred, act):
    return math.sqrt(mse(pred, act))


def mae(pred, act):
    mae = 0.0
    E = abs(pred - act)
    for i in range(len(E)):
        mae += abs(E[i])
    return mae / len(E)


def confusion_matrix_analysis(cm):
    true_negative = cm[0, 0]
    false_positive = cm[0, 1]
    false_negative = cm[1, 0]
    true_positive = cm[1, 1]

    total = true_negative + true_positive + false_negative + false_positive
    neg = true_negative + false_positive
    pos = false_negative + true_positive
    error_rate = 1.0 * (false_positive + false_negative) / total
    accuracy = 1.0 * (true_negative + true_positive) / total
    if neg == 0:
        fpr = 0
    else:
        fpr = 1.0 * false_positive / neg
    if pos == 0:
        tpr = 0
    else:
        tpr = 1.0 * true_positive / pos

    return error_rate, accuracy, fpr, tpr


def confusion_matrix(actual, predict, positive=1, negative=0):
    true_negative = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0

    for i in range(len(actual)):
        act_label = actual[i]
        pred_label = predict[i]
        if act_label == positive and pred_label == negative:
            false_negative += 1
        elif act_label == negative and pred_label == positive:
            false_positive += 1
        elif act_label == positive and pred_label == positive:
            true_positive += 1
        else:
            true_negative += 1

    return np.array([[true_negative, false_positive], [false_negative, true_positive]])


def roc(actual, predict, title, threshold=0.5):
    rc = ROC(actual, predict, threshold, title)
    rc.plot()
    print "AUC: %s" % rc.auc()


class ROC:
    def __init__(self, actual, predict, threshold, title):
        predict = np.array(predict)
        sorted_indices = predict.argsort()[::-1]
        predict = predict[sorted_indices]
        actual = actual[sorted_indices]
        self.title = title
        self.actual = actual
        self.predict = predict
        self.predict_label = map(lambda v: 1 if v >= threshold else 0, predict)
        self.pos = len(actual[actual == 1])
        self.neg = len(actual[actual == 0])
        self.points = []

    def create_roc_records(self):
        roc_records = []

        for i in range(len(self.actual)):
            y = self.actual[i]
            p = self.predict[i]
            roc_records.append((p, y))

        return roc_records

    def auc(self):
        x = self.points[:, 1]
        y = self.points[:, 0]
        sort = np.lexsort((y, x))
        x = x[sort]
        y = y[sort]
        return np.trapz(y, x)

    def plot(self):
        recs = self.create_roc_records()
        self.plot_roc_data(recs)
        x = self.points[:, 1]
        y = self.points[:, 0]
        plt.title(self.title)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.scatter(x, y)
        plt.show()

    def plot_roc_data(self, recs):
        roc_data = []
        for sep in range(1, len(recs)):
            tpr, fpr = self.plot_roc_point(recs, sep)
            roc_data.append([1.0 * tpr / self.pos, 1.0 * fpr / self.neg])
        self.points = np.array(roc_data)
        return self.points

    @staticmethod
    def plot_roc_point(recs, sep):
        pred_pos = recs[:sep]
        tpr = 0
        fpr = 0
        for rec in pred_pos:
            if rec[1] == 1:
                tpr += 1
            else:
                fpr += 1
        return tpr, fpr


    @staticmethod
    def print_roc_record(recs):
        print 'predict, actual'
        for p, y in recs:
            print '%s %s' % (p, y)