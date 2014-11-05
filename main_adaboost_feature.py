__author__ = 'jiachiliu'

from boost.cross_validation import train_test_shuffle_split
from boost.dataset import load_spambase
from boost.AdaBoost import AdaBoost
import numpy as np
import timeit


if __name__ == '__main__':

    train = np.array([
        [1.0],
        [0.0],
        [0.0],
        [2.0],
        [0.0],
        [5.0],
        [4.0],
        [1.0]
    ])

    target = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    train, target = load_spambase()
    target = np.array(map(lambda v: -1.0 if v == 0 else 1.0, target))

    train, test, train_target, test_target = train_test_shuffle_split(train, target, len(train) / 10)
    boost = AdaBoost()
    start = timeit.default_timer()
    boost.boost(train, train_target, test, test_target)
    stop = timeit.default_timer()
    print "Total Run Time: %s secs" % (stop - start)
