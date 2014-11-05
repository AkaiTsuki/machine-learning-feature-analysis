__author__ = 'jiachiliu'

import numpy as np
from ranking import auc, roc
from validation import confusion_matrix, confusion_matrix_analysis


class Cache:
    def __init__(self, to_add, to_remove, threshold):
        self.to_add = to_add
        self.to_remove = to_remove
        self.threshold = threshold


class OptimalLearner:
    def __init__(self):
        self.cache = {}

    def fit(self, train, target, dist):
        m, n = train.shape
        max_error = -1.0
        weighted_error = None
        selected_feature = -1
        selected_threshold = None
        for f in range(n):
            error_on_feature = -1
            weighted_err_on_feature = -1
            threshold = None
            if f in self.cache:
                error_on_feature, weighted_err_on_feature, threshold = self.find_best_error_on_cache(self.cache[f],
                                                                                                     dist)
            else:
                error_on_feature, weighted_err_on_feature, threshold = self.find_best_error_on_feature(train[:, f],
                                                                                                       target,
                                                                                                       dist, f)
            if error_on_feature > max_error:
                max_error = error_on_feature
                weighted_error = weighted_err_on_feature
                selected_feature = f
                selected_threshold = threshold
        return selected_feature, selected_threshold, weighted_error

    def find_best_error_on_cache(self, cache, d):
        best_weighted_error = -1.0
        best_abs_error = -1.0
        best_threshold = None

        current_weighted_error = 0.0
        for i in range(len(cache)):
            to_add = cache[i].to_add
            to_remove = cache[i].to_remove
            current_threshold = cache[i].threshold

            current_weighted_error += d[to_add].sum() - d[to_remove].sum()

            current_abs_error = self.abs_error(current_weighted_error)
            if current_abs_error > best_abs_error:
                best_abs_error = current_abs_error
                best_weighted_error = current_weighted_error
                best_threshold = current_threshold

        return best_abs_error, best_weighted_error, best_threshold

    def find_best_error_on_feature(self, f, t, d, f_index):
        """
        Find the best threshold with maximum abs error
        :param f: a list of feature values
        :param t: actual labels
        :param d: distribution of each data point
        :param f_index: index of the current feature
        :return: the maximum abs error, and corresponding weighted error and threshold
        """
        m = len(t)
        args = f.argsort()
        # Compute the errors with all positive label, in other words threshold is less than the min value of the feature
        p = np.ones(m)
        mismatch = np.where(p != t)[0]
        current_weighted_error = d[mismatch].sum()
        current_abs_error = self.abs_error(current_weighted_error)
        current_threshold = f[args[0]] - 0.5
        self.cache[f_index] = [Cache(mismatch, [], current_threshold)]

        best_weighted_error = current_weighted_error
        best_abs_error = current_abs_error
        best_threshold = current_threshold

        start = 0
        for i in range(1, m + 1):
            if i == m or f[args[i]] != f[args[i - 1]]:
                to_change = map(lambda v: args[v], range(start, i))
                delta_err, to_add, to_remove = self.update(p, t, d, to_change)
                current_weighted_error += delta_err
                current_abs_error = self.abs_error(current_weighted_error)
                current_threshold = (f[args[i - 1]] + f[args[i]]) / 2 if i != m else f[args[-1]] + 0.5
                self.cache[f_index].append(Cache(to_add, to_remove, current_threshold))
                if current_abs_error > best_abs_error:
                    best_abs_error = current_abs_error
                    best_weighted_error = current_weighted_error
                    best_threshold = current_threshold
                start = i

        return best_abs_error, best_weighted_error, best_threshold

    @staticmethod
    def update(p, t, d, to_change):
        delta = 0.0
        to_add = []
        to_remove = []
        for i in to_change:
            p[i] = -1.0
            if p[i] == t[i]:
                delta -= d[i]
                to_remove.append(i)
            else:
                delta += d[i]
                to_add.append(i)
        return delta, to_add, to_remove

    @staticmethod
    def abs_error(weighted_err):
        return abs(0.5 - weighted_err)


class AdaBoost:
    def __init__(self):
        pass

    @staticmethod
    def hypothesis(f, t, data):
        return np.array([1.0 if d[f] > t else -1.0 for d in data])

    @staticmethod
    def sign(vals, negative=-1.0, positive=1.0):
        """
        map every value in given value to +1 or -1. If the value is negative or zero map to -1
        otherwise map to +1
        :param vals: a list of value
        :return: a list of -1 or +1 based on the value
        """
        return np.array([negative if v <= 0 else positive for v in vals])

    def boost(self, train, train_target, test, test_target, T=100):
        m, n = train.shape

        weights = np.array([1.0 / m] * m)
        learner = OptimalLearner()
        round = 0

        train_predicts = np.zeros(m)
        test_predicts = np.zeros(len(test))

        while round < T:
            f, t, weighted_err = learner.fit(train, train_target, weights)
            confidence = 0.5 * np.log((1.0 - weighted_err) / weighted_err)

            predicts = self.hypothesis(f, t, train)
            train_predicts += confidence * predicts
            train_predicts_signed = self.sign(train_predicts)
            train_cm = confusion_matrix(train_predicts_signed, train_target, 1.0, -1.0)
            train_err, train_acc, train_fpr, train_tpr = confusion_matrix_analysis(train_cm)

            test_predicts += confidence * self.hypothesis(f, t, test)
            test_predicts_signed = self.sign(test_predicts)
            test_cm = confusion_matrix(test_predicts_signed, test_target, 1.0, -1.0)
            test_err, test_acc, test_fpr, test_tpr = confusion_matrix_analysis(test_cm)

            roc_points = roc(test_target, test_predicts, 1.0, -1.0)
            test_auc = auc(roc_points[:, 1], roc_points[:, 0])

            print "round %s, feature: %s, threshold: %s, round_error: %s, train error: %s, test error: %s, auc: %s" % (
                round, f, t, weighted_err, train_err, test_err, test_auc)

            tmp = np.sqrt((weighted_err / (1.0 - weighted_err)) ** (train_target * predicts))
            weights = (weights * tmp) / (2.0 * np.sqrt(weighted_err * (1 - weighted_err)))

            round += 1
