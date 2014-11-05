__author__ = 'jiachiliu'

import numpy as np
from numpy.linalg import det
from numpy.linalg import inv


class GDA:
    def __init__(self):
        self.cov = None
        self.mean = {}
        self.priors = {}
        self.labels = None
        self.N = 0

    def fit(self, train, target):
        self.cov = np.cov(train, rowvar=0)
        self.labels = np.unique(target)
        self.N = len(train)
        for l in self.labels:
            tuples = train[target == l]
            self.mean[l] = np.array([tuples[:, f].mean() for f in range(tuples.shape[1])])
            self.priors[l] = 1.0 * len(target[target == l]) / self.N

        return self

    def predict(self, test):
        predicts = []

        for t in test:
            res = []
            for l in self.labels:
                likelihood = -0.5 * (t - self.mean[l]).T.dot(inv(self.cov)).dot(t - self.mean[l])
                res.append(likelihood + np.log(self.priors[l]))
            predicts.append(res[1] - res[0])
        return predicts

    @staticmethod
    def predict_class(predicts):
        # print predicts
        return np.array(map(lambda p: 1.0 if p > 0 else 0.0, predicts))


class GaussianNaiveBayes:
    def __init__(self):
        # the mean vector for all features
        self.overall_mean = None
        # the variance vector for all features
        self.overall_var = None
        # class conditional mean
        self.conditional_mean = {}
        # class conditional var
        self.conditional_var = {}
        # all labels
        self.labels = None
        self.priors = {}

    def setup(self, train, target):
        self.overall_mean = self.get_mean_vector(train)
        self.overall_var = self.get_var_vector(train)
        self.labels = np.unique(target)

    @staticmethod
    def get_mean_vector(data):
        return np.array([data[:, f].mean() for f in range(data.shape[1])])

    @staticmethod
    def get_var_vector(data):
        return np.array([data[:, f].var() for f in range(data.shape[1])])

    def fit(self, train, target):
        self.setup(train, target)
        n = len(target)
        p = 1.0 * n / (n + 2)
        for l in self.labels:
            tuples = train[target == l]
            self.priors[l] = 1.0 * len(tuples) / n
            self.conditional_mean[l] = self.get_mean_vector(tuples)
            self.conditional_var[l] = p * self.get_var_vector(tuples) + (1 - p) * self.overall_var
        return self

    def predict(self, test):
        predicts = []
        for t in test:
            res = []
            for l in self.labels:
                log_liklihood = 0.0
                for f in range(test.shape[1]):
                    g = self.gaussian_on_ln(t[f], self.get_class_conditional_mean(l, f),
                                            self.get_class_conditional_var(l, f))
                    log_liklihood += g
                res.append(log_liklihood + np.log(self.priors[l]))
            predicts.append(res[1] - res[0])
        return predicts

    def predict_class(self, predicts):
        # print predicts
        return np.array(map(lambda p: 1.0 if p > 0 else 0.0, predicts))

    @staticmethod
    def gaussian(f, m, v):
        v2 = ((f - m) * (f - m)) / (2.0 * v)
        v1 = np.exp(-v2)
        v3 = v1 / np.sqrt(2.0 * v * np.pi)
        return v3

    @staticmethod
    def gaussian_on_ln(f, m, v):
        v1 = np.log(1.0 / np.sqrt(2.0 * v * np.pi))
        v2 = -((f - m) ** 2) / (2.0 * v)
        return v1 + v2

    def get_class_conditional_mean(self, label, feature):
        return self.conditional_mean[label][feature]

    def get_class_conditional_var(self, label, feature):
        return self.conditional_var[label][feature]


class HistogramNaiveBayes:
    def __init__(self):
        self.overall_mean = None
        self.spam_mean = None
        self.non_spam_mean = None
        self.priors = {}
        self.bins = []
        self.likelihoods = {}
        self.labels = None

    def fit(self, train, target):
        self.setup_bins(train, target)
        self.labels = np.unique(target)

        for l in self.labels:
            tuples = train[target == l]
            self.calculate_likelihoods(tuples, l)
        return self

    def calculate_likelihoods(self, data, label):
        label_count = len(data)
        self.likelihoods[label] = []
        possible_value_count = len(self.bins[0]) - 1
        for f in range(data.shape[1]):
            feature_values = data[:, f]
            bin = self.bins[f]
            bin_count = [1] * possible_value_count
            for val in feature_values:
                bin_count[self.get_bin_index(val, bin)] += 1

            bin_likelihoods = []
            for c in bin_count:
                bin_likelihoods.append(1.0 * c / (label_count + possible_value_count))
            self.likelihoods[label].append(bin_likelihoods)

    @staticmethod
    def get_bin_index(val, bin):
        for i in range(len(bin) - 1):
            if i == 0 and bin[i] == val:
                return i
            if bin[i] < val <= bin[i + 1]:
                return i
        return -1

    def setup_bins(self, train, target):
        self.overall_mean = self.get_mean_vector(train)
        spams = train[target == 1]
        non_spams = train[target == 0]
        self.priors[1] = 1.0 * len(spams) / len(train)
        self.priors[0] = 1.0 * len(non_spams) / len(train)

        self.spam_mean = self.get_mean_vector(spams)
        self.non_spam_mean = self.get_mean_vector(non_spams)

        min_val_vector = [train[:, f].min() for f in range(train.shape[1])]
        max_val_vector = [train[:, f].max() for f in range(train.shape[1])]

        for f in range(train.shape[1]):
            min_value = min_val_vector[f]
            max_value = max_val_vector[f]
            spam_mean_value = self.spam_mean[f]
            non_spam_mean_value = self.non_spam_mean[f]
            mean_value = self.overall_mean[f]
            bin = [min_value, max_value, spam_mean_value, non_spam_mean_value, mean_value]
            bin = sorted(bin)
            self.bins.append(bin)

    def predict(self, test):
        predicts = []

        for t in test:
            res = []
            for l in self.labels:
                likelihoods = self.likelihoods[l]
                likelihood = 1.0
                for f in range(test.shape[1]):
                    likelihood *= likelihoods[f][self.get_bin_index(t[f], self.bins[f])]
                res.append(likelihood * self.priors[l])
            predicts.append(np.log(res[1] / res[0]))

        return predicts

    def predict_class(self, predicts):
        return np.array(map(lambda p: 1.0 if p > 0 else 0.0, predicts))

    @staticmethod
    def get_mean_vector(data):
        return np.array([data[:, f].mean() for f in range(data.shape[1])])


class BernoulliNaiveBayes(HistogramNaiveBayes):
    def __init__(self):
        HistogramNaiveBayes.__init__(self)

    def setup_bins(self, train, target):
        self.overall_mean = self.get_mean_vector(train)
        spams = train[target == 1]
        non_spams = train[target == 0]
        self.priors[1] = 1.0 * len(spams) / len(train)
        self.priors[0] = 1.0 * len(non_spams) / len(train)

        for f in range(train.shape[1]):
            min_value = float('-inf')
            max_value = float('inf')
            mean_value = self.overall_mean[f]
            bin = [min_value, mean_value, max_value]
            self.bins.append(bin)


class NBinsHistogramNaiveBayes(HistogramNaiveBayes):
    def __init__(self, N):
        HistogramNaiveBayes.__init__(self)
        self.N = N

    def setup_bins1(self, train, target):
        spams = train[target == 1]
        non_spams = train[target == 0]
        self.priors[1] = 1.0 * len(spams) / len(train)
        self.priors[0] = 1.0 * len(non_spams) / len(train)

        min_val_vector = [train[:, f].min() for f in range(train.shape[1])]
        max_val_vector = [train[:, f].max() for f in range(train.shape[1])]

        for f in range(train.shape[1]):
            min_value = min_val_vector[f]
            max_value = max_val_vector[f]
            gap = 1.0 * (max_value - min_value) / self.N
            bin = [min_value]

            for i in range(1, self.N):
                bin.append(min_value + i * gap)

            bin.append(max_value)
            self.bins.append(bin)

    def setup_bins(self, train, target):
        spams = train[target == 1]
        non_spams = train[target == 0]
        self.priors[1] = 1.0 * len(spams) / len(train)
        self.priors[0] = 1.0 * len(non_spams) / len(train)

        overall_means = np.array([train[:, f].mean() for f in range(train.shape[1])])
        overall_stds = np.array([train[:, f].std() for f in range(train.shape[1])])
        min_val_vector = [train[:, f].min() for f in range(train.shape[1])]
        max_val_vector = [train[:, f].max() for f in range(train.shape[1])]

        for f in range(train.shape[1]):
            mean = overall_means[f]
            std = overall_stds[f]
            bin = [min_val_vector[f]]

            loop = (self.N/2) if self.N % 2 == 1 else self.N/2 - 1
            lb = 0.5 if self.N % 2 == 1 else 1

            if self.N % 2 == 0:
                bin.append(mean)

            for i in range(loop):
                bin.append(mean + lb*std*(i+1))
                bin.append(mean - lb*std*(i+1))

            bin.append(max_val_vector[f])
            bin = sorted(bin)
            self.bins.append(bin)
