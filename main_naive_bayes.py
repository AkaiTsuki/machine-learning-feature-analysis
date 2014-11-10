__author__ = 'jiachiliu'

import numpy as np
from nulearn.bayes import GaussianNaiveBayes
import timeit
from nulearn.validation import *
from nulearn.ranking import *
from nulearn.dataset import load_polluted_spambase
from sklearn.decomposition import PCA
from sklearn.lda import LDA


def naive_bayes_no_pca():
    train, train_target, test, test_target = load_polluted_spambase()

    print "Train data: %s, Train Label: %s" % (train.shape, train_target.shape)
    print "Test data: %s, Test Label: %s" % (test.shape, test_target.shape)

    start = timeit.default_timer()

    cf = GaussianNaiveBayes()
    cf.fit(train, train_target)
    raw_predicts = cf.predict(test)
    predict_class = cf.predict_class(raw_predicts)

    cm = confusion_matrix(test_target, predict_class)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

    stop = timeit.default_timer()
    print "Total Run Time: %s secs" % (stop - start)


def naive_bayes_with_pca():
    train, train_target, test, test_target = load_polluted_spambase()

    print "Train data: %s, Train Label: %s" % (train.shape, train_target.shape)
    print "Test data: %s, Test Label: %s" % (test.shape, test_target.shape)

    start = timeit.default_timer()

    pca = PCA(n_components=100)
    train = pca.fit_transform(train)
    test = pca.transform(test)
    print pca
    print "Train data: %s, Train Label: %s" % (train.shape, train_target.shape)
    print "Test data: %s, Test Label: %s" % (test.shape, test_target.shape)

    cf = GaussianNaiveBayes()
    cf.fit(train, train_target)
    raw_predicts = cf.predict(test)
    predict_class = cf.predict_class(raw_predicts)

    cm = confusion_matrix(test_target, predict_class)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

    stop = timeit.default_timer()
    print "Total Run Time: %s secs" % (stop - start)


def naive_bayes_with_lda():
    train, train_target, test, test_target = load_polluted_spambase()

    print "Train data: %s, Train Label: %s" % (train.shape, train_target.shape)
    print "Test data: %s, Test Label: %s" % (test.shape, test_target.shape)

    start = timeit.default_timer()

    lda = LDA(n_components=100)
    train = lda.fit_transform(train, train_target)
    test = lda.transform(test)

    print lda
    print "Train data: %s, Train Label: %s" % (train.shape, train_target.shape)
    print "Test data: %s, Test Label: %s" % (test.shape, test_target.shape)

    cf = GaussianNaiveBayes()
    cf.fit(train, train_target)
    raw_predicts = cf.predict(test)
    predict_class = cf.predict_class(raw_predicts)

    cm = confusion_matrix(test_target, predict_class)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

    stop = timeit.default_timer()
    print "Total Run Time: %s secs" % (stop - start)

if __name__ == '__main__':
    naive_bayes_with_pca()
    # naive_bayes_with_lda()