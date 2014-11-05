__author__ = 'jiachiliu'

import numpy as np
import operator


class MarginAnalysis:
    def __init__(self, weak_learners):
        self.weak_learners = weak_learners
        self.feature_learners = {}

    def analyze(self, train, train_target):
        self.split_learner_on_feature()
        k, n = train.shape
        contributions = {}

        margin_all = 0.0
        for d in range(k):
            margin_all += self.margin(train[d], train_target[d])

        for f in self.feature_learners.keys():
            mf = 0.0
            for d in range(k):
                mf += self.margin_on_feature(train[d], train_target[d], f)
            contributions[f] = mf / margin_all
            print "contributions of %s: %s" % (f, contributions[f])

        # print contributions
        sorted_contributions = sorted(contributions, key=contributions.get, reverse=True)
        print sorted_contributions
        # print "Feature Contribution Rank: %s" % [f for f, v in sorted_contributions]

    def margin(self, x, label):
        m = 0.0
        for i in range(len(self.weak_learners)):
            f, t, alpha = self.weak_learners[i]
            m += label * alpha * self.h(x, f, t)
        return m

    def margin_on_feature(self, x, label, f):
        m = 0.0
        for i in range(len(self.feature_learners[f])):
            t, alpha = self.feature_learners[f][i]
            m += label * alpha * self.h(x, f, t)
        return m

    def split_learner_on_feature(self):
        for f, t, alpha in self.weak_learners:
            if f not in self.feature_learners:
                self.feature_learners[f] = []
            self.feature_learners[f].append((t, alpha))

    @staticmethod
    def h(x, f, t):
        return -1.0 if x[f] <= t else 1.0


