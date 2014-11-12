__author__ = 'jiachiliu'

import os, struct
from pylab import *
from array import array as pyarray
import numpy as np
from numpy import append, array, int8, uint8, zeros
from nulearn.Haar import Haar
from nulearn.ECOC import ECOC
import timeit
from nulearn.dataset import load_digital_dataset


def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def write_feature_to_file(path, features):
    np.savetxt(path, features, delimiter=',', fmt='%d')


def extract_feature_to_file():
    image = np.array(
        [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]])

    images, labels = load_mnist(dataset="training", path="data")
    haar = Haar()
    features = haar.extract_features(images, inners=100)
    write_feature_to_file('data/digital_train_features.txt', features)
    write_feature_to_file('data/digital_train_target.txt', labels)

    images, labels = load_mnist(dataset="testing", path="data")
    features = haar.extract_features(images, inners=100)
    write_feature_to_file('data/digital_test_features.txt', features)
    write_feature_to_file('data/digital_test_target.txt', labels)


def extract_test_feature_to_file():
    images, labels = load_mnist(dataset="testing", path="data")
    haar = Haar()
    features = haar.extract_features(images, inners=100)
    write_feature_to_file('data/digital_test_features.txt', features)
    write_feature_to_file('data/digital_test_target.txt', labels)


def run_ecoc():
    start = timeit.default_timer()
    ecoc = ECOC(10, 20)
    train, train_target, test, test_target = load_digital_dataset()
    print "Train: ", train.shape
    print "Train Target: ", train_target.shape
    print "Test: ", test.shape
    print "Test Target: ", test_target.shape
    ecoc.train(train, train_target, test, test_target, T=50, percentage=1)
    labels = ecoc.test(test)

    err = 0
    for pred, act in zip(labels, test_target):
        if pred != act:
            err += 1
    print "Total error: %s" % (1.0 * err / len(test_target))
    stop = timeit.default_timer()
    print stop - start

if __name__ == '__main__':
    run_ecoc()