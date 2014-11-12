__author__ = 'jiachiliu'

import numpy as np
import random
from nulearn.AdaBoost import *


class ECOC:
    def __init__(self, num_of_labels, num_of_selected):
        self.num_of_labels = num_of_labels
        self.exhausted_code = self.generate_exhaustive_code(num_of_labels)
        self.selected_code = self.random_select(num_of_selected)
        self.functions = []

    def train(self, train, train_target, test, test_target, T=100, percentage=1):
        k, n = self.selected_code.shape

        train = train[:int(len(train) * percentage)]
        train_target = train_target[:int(len(train_target) * percentage)]
        for f in range(n):
            print "Run Adaboost on function %f" % f
            codes = self.selected_code[:, f]
            labels = self.convert_to_binary(train_target, codes)
            test_labels = self.convert_to_binary(test_target, codes)
            adaboost = AdaBoost()
            adaboost.boost(train, labels, test, test_labels, T)
            self.functions.append(adaboost)

    def test(self, test):
        predicts = np.zeros((len(test), len(self.functions)))
        final_labels = []

        for f in range(len(self.functions)):
            func = self.functions[f]
            predicts[:, f] = func.sign(func.predict(test), negative=0.0)

        for p in predicts:
            final_labels.append(self.get_label(p))

        return np.array(final_labels)

    def get_label(self, predict):
        label = None
        d = float('inf')
        for c in range(len(self.selected_code)):
            dist = self.distance(predict, self.selected_code[c])
            if dist < d:
                d = dist
                label = c
        return label

    def distance(self, p, c):
        d = 0
        for x, y in zip(p, c):
            if x != y:
                d += 1
        return d

    def convert_to_binary(self, target, codes):
        labels = []
        for t in target:
            if codes[t] == 1.0:
                labels.append(1.0)
            else:
                labels.append(-1.0)
        return np.array(labels)

    @staticmethod
    def generate_exhaustive_code(num_of_labels):
        """
        Generate exhaustive code based on given number of classes
        :param num_of_labels: number of class labels
        :return: a 2d-array that represent the exhaustive code
        """
        k = num_of_labels
        n = 2 ** (k - 1) - 1
        code = np.zeros((k, n))

        # first row is all 1
        for col in (range(n)):
            code[0][col] = 1.0

        for row in range(1, k):
            interval = 2 ** (k - row - 1)
            count = 0
            flag = False
            for col in range(n):
                code[row][col] = 1.0 if flag else 0.0
                count += 1
                if count == interval:
                    flag = not flag
                    count = 0
        return code

    def random_select(self, size):
        """
        Random select a subset of exhaustive code
        :param size: number of columns
        :return: subset of code
        """
        m, n = self.exhausted_code.shape
        selected = random.sample(range(n), size)
        return self.exhausted_code[:, selected]