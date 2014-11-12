__author__ = 'jiachiliu'

from numpy.linalg import inv
import numpy as np
from validation import mse
import sys


class LinearRegression(object):
    """docstring for LinearRegression"""

    def __init__(self):
        self.weights = None

    def fit(self, train, target):
        self.weights = inv(train.T.dot(train)).dot(train.T).dot(target)
        return self

    def predict(self, test):
        return test.dot(self.weights)


class GradientDescendingRegression(LinearRegression):
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, train, target, alpha=0.00001, max_loop=1500):
        m, n = train.shape
        self.weights = np.ones(n)
        for k in range(max_loop):
            predict = self.predict(train)
            error = predict - target
            self.weights -= alpha * train.T.dot(error)

        return self


class StochasticGradientDescendingRegression(LinearRegression):
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, train, target, alpha=0.0001, max_loop=130, converge=0.001):
        m, n = train.shape
        self.weights = np.zeros(n)
        for k in range(max_loop):
            prev_error = mse(self.predict(train), target)
            self.print_progress(k, prev_error)
            for t in range(m):
                data_point = train[t]
                error = self.predict(data_point) - target[t]
                self.weights -= alpha * error * data_point
                print self.weights
            if abs(prev_error - mse(self.predict(train), target)) <= converge:
                break
        return self

    @staticmethod
    def print_progress(k, cost):
        print "Iteration: %s, error: %s" % (k + 1, cost)


class LogisticGradientDescendingRegression(StochasticGradientDescendingRegression):
    def __init__(self):
        LinearRegression.__init__(self)

    def fit(self, train, target, alpha=0.0001, max_loop=130, converge=0.001):
        m, n = train.shape
        self.weights = np.zeros(n)
        for k in range(max_loop):
            prev_error = mse(self.predict(train), target)
            self.print_progress(k, prev_error)
            for t in xrange(m):
                data_point = train[t]
                predict = self.predict(data_point)
                error = predict - target[t]
                self.weights -= alpha * error * predict * (1.0 - predict) * data_point
            if abs(prev_error - mse(self.predict(train), target)) <= converge:
                break
        return self

    @staticmethod
    def sigmoid(vals):
        return 1.0 / (1 + np.exp(-vals))

    def predict(self, test):
        return self.sigmoid(test.dot(self.weights))

    @staticmethod
    def convert_to_binary(vals, threshold=0.5):
        return map(lambda v: 1 if v >= threshold else 0, vals)


class BatchLogisticRegression(LogisticGradientDescendingRegression):

    def fit(self, train, target, alpha=0.0001, max_loop=1300, converge=0.0001, beta=10):
        m, n = train.shape
        self.weights = np.ones(n)
        for k in range(max_loop):
            prev_error = mse(self.predict(train), target)
            self.print_progress(k, prev_error)
            predict = self.predict(train)
            error = predict - target
            self.weights -= alpha * (train.T.dot(error) + beta * self.weights)

        return self

class RidgedLogisticRegression(LogisticGradientDescendingRegression):
    def __init__(self):
        LogisticGradientDescendingRegression.__init__(self)

    def fit(self, train, target, alpha=0.0001, max_loop=1000, converge=0.0001, beta=0.001):
        m, n = train.shape
        self.weights = np.zeros(n)
        for k in range(max_loop):
            prev_error = mse(self.predict(train), target)
            self.print_progress(k, prev_error)
            for t in xrange(m):
                data_point = train[t]
                predict = self.predict(data_point)
                error = predict - target[t]
                self.weights -= alpha * ((error * data_point) + beta * self.weights)
            if abs(prev_error - mse(self.predict(train), target)) <= converge:
                break
        return self


class Perceptron:
    def __init__(self):
        self.weights = None

    def predict(self, test):
        return test.dot(self.weights)

    def fit(self, train, target):
        m, n = train.shape
        self.weights = np.zeros(n)
        train, target = self.flip(train, target)
        k = 0
        while not self.all_positive(train):
            count = 0
            for features, label in zip(train, target):
                if self.predict(features) <= 0:
                    count += 1
                    self.weights += features
            print 'Iteration %d, total mistakes: %d' % (k + 1, count)
            k += 1
        print 'Iteration %d, total mistakes: %d' % (k + 1, self.total_error(self.predict(train)))

    @staticmethod
    def total_error(predict):
        count = 0
        for p in predict:
            if p <= 0:
                count += 1
        return count

    def all_positive(self, train):
        for features in train:
            if self.predict(features) <= 0:
                return False
        return True

    @staticmethod
    def flip(train, target):
        new_train = []
        new_target = []
        for features, label in zip(train, target):
            if label == -1:
                new_train.append(-features)
                new_target.append(-label)
            else:
                new_train.append(features)
                new_target.append(label)
        return np.array(new_train), np.array(new_target)