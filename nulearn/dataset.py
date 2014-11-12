__author__ = 'jiachiliu'

import numpy as np
import random


class CsvReader:
    """
    CsvReader will read data from csv file
    """

    def __init__(self, path):
        self.path = path

    def read(self, delimiter, converter):
        f = open(self.path)
        lines = f.readlines()
        return self.parse_lines(lines, delimiter, converter)

    @staticmethod
    def parse_lines(lines, delimiter, converter):
        data = []
        for line in lines:
            if line.strip():
                row = [s.strip() for s in line.strip().split(delimiter) if s.strip()]
                data.append(row)

        return np.array(data, converter)


def load_spambase():
    reader = CsvReader('data/spambase.data')
    data = reader.read(',', float)
    total_col = data.shape[1]
    return data[:, :total_col - 1], data[:, total_col - 1]


def load_polluted_spambase():
    reader = CsvReader('data/spam_polluted/train_feature.txt')
    train = reader.read(' ', float)

    reader = CsvReader('data/spam_polluted/train_label.txt')
    train_target = reader.read(' ', float).flatten()

    reader = CsvReader('data/spam_polluted/test_feature.txt')
    test = reader.read(' ', float)

    reader = CsvReader('data/spam_polluted/test_label.txt')
    test_target = reader.read(' ', float).flatten()

    return train, train_target, test, test_target


def load_20p_missing_spambase():
    reader = CsvReader('data/20_percent_missing_train.txt')
    train = reader.read(',', float)

    reader = CsvReader('data/20_percent_missing_test.txt')
    test = reader.read(',', float)

    total_col = train.shape[1]
    return train[:, :total_col - 1], train[:, total_col - 1], test[:, :total_col - 1], test[:, total_col - 1]


def sample_digital_dataset(train, train_target, test, test_target):
    sample = None
    sample_target = None
    for i in range(10):
        match = train_target == i
        match_train = train[match]
        match_train_target = train_target[match]
        k = int(len(match_train) * 0.2)
        if sample is None:
            sample = match_train[: k]
            sample_target = match_train_target[: k]
        else:
            sample = np.vstack((sample, match_train[: k]))
            sample_target = np.append(sample_target, match_train_target[: k])
    return sample, sample_target, test, test_target


def load_digital_dataset():
    reader = CsvReader('data/digital_train_features.txt')
    train = reader.read(',', float)

    reader = CsvReader('data/digital_train_target.txt')
    train_target = reader.read(',', float).flatten()

    reader = CsvReader('data/digital_test_features.txt')
    test = reader.read(',', float)

    reader = CsvReader('data/digital_test_target.txt')
    test_target = reader.read(',', float).flatten()

    return sample_digital_dataset(train, train_target, test, test_target)