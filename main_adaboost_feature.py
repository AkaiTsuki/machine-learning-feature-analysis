__author__ = 'jiachiliu'

from nulearn.cross_validation import train_test_shuffle_split
from nulearn.dataset import load_spambase, load_polluted_spambase
from nulearn.AdaBoost import AdaBoost
import numpy as np
import timeit


def spambase(T=100):
    train, target = load_spambase()
    target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, target))

    train, test, train_target, test_target = train_test_shuffle_split(train, target, len(train) / 10)
    boost = AdaBoost()
    start = timeit.default_timer()
    boost.boost(train, train_target, test, test_target, T)
    stop = timeit.default_timer()
    print "Total Run Time: %s secs" % (stop - start)


def polluted_spambase(T=100):
    train, train_target, test, test_target = load_polluted_spambase()
    train_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, train_target))
    test_target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, test_target))
    print "Train data: %s, Train Label: %s" % (train.shape, train_target.shape)
    print "Test data: %s, Test Label: %s" % (test.shape, test_target.shape)
    boost = AdaBoost()
    start = timeit.default_timer()
    boost.boost(train, train_target, test, test_target, T)
    stop = timeit.default_timer()
    print "Total Run Time: %s secs" % (stop - start)


if __name__ == '__main__':
    # spambase(300)
    polluted_spambase(300)