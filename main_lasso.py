__author__ = 'jiachiliu'

from sklearn import linear_model
from nulearn.dataset import load_polluted_spambase
import timeit
from nulearn.validation import *
from nulearn.linear_model import LogisticGradientDescendingRegression, RidgedLogisticRegression, BatchLogisticRegression
from nulearn.preprocessing import normalize, append_new_column


def ski_lasso(alpha, max_iter):
    train, train_target, test, test_target = load_polluted_spambase()

    print "Train data: %s, Train Label: %s" % (train.shape, train_target.shape)
    print "Test data: %s, Test Label: %s" % (test.shape, test_target.shape)

    start = timeit.default_timer()

    clf = linear_model.Lasso(alpha=alpha, max_iter=max_iter)
    print clf
    clf.fit(train, train_target)
    predicts = clf.predict(test)
    predict_class = map(lambda v: 0 if v <= 0.42 else 1, predicts)
    cm = confusion_matrix(test_target, predict_class)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

    stop = timeit.default_timer()
    print "Total Run Time: %s secs" % (stop - start)


def logistic_regression(beta):
    train, train_target, test, test_target = load_polluted_spambase()

    print "Train data: %s, Train Label: %s" % (train.shape, train_target.shape)
    print "Test data: %s, Test Label: %s" % (test.shape, test_target.shape)

    scaler = normalize(train)
    scaler.scale_test(test)
    train = append_new_column(train, 1.0, 0)
    test = append_new_column(test, 1.0, 0)

    cf = BatchLogisticRegression()
    cf.fit(train, train_target, beta=beta, converge=0, max_loop=1000)
    predict_values = cf.predict(test)
    predict_classes = cf.convert_to_binary(predict_values)
    cm = confusion_matrix(test_target, predict_classes)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

def ridged_logistic_regression():
    train, train_target, test, test_target = load_polluted_spambase()

    print "Train data: %s, Train Label: %s" % (train.shape, train_target.shape)
    print "Test data: %s, Test Label: %s" % (test.shape, test_target.shape)

    scaler = normalize(train)
    scaler.scale_test(test)
    train = append_new_column(train, 1.0, 0)
    test = append_new_column(test, 1.0, 0)

    cf = RidgedLogisticRegression()
    cf.fit(train, train_target)
    predict_values = cf.predict(test)
    predict_classes = cf.convert_to_binary(predict_values)
    cm = confusion_matrix(test_target, predict_classes)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)

if __name__ == '__main__':
    ski_lasso(0.005, 10000)
    logistic_regression(0)
    logistic_regression(100)