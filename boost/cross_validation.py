__author__ = 'jiachiliu'

import numpy as np

def train_test_split(train, target, test_size):
    return train[test_size:], train[:test_size], target[test_size:], target[:test_size]

def train_test_shuffle_split(data, target, test_size):
    indices = np.random.permutation(data.shape[0])
    training_idx, test_idx = indices[test_size + 1:], indices[:test_size + 1]
    return data[training_idx, :], data[test_idx, :], target[training_idx, :], target[test_idx, :]


def shuffle(train, target):
    indices = np.random.permutation(train.shape[0])
    return train[indices], target[indices]


def k_fold_cross_validation(length, n):
    block_size = length / n

    for i in range(n):
        start = i * block_size
        end = start + block_size
        yield (start, end)
