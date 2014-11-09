__author__ = 'jiachiliu'

from boost.dataset import load_20p_missing_spambase
from boost.bayes import BernoulliNaiveBayes
from boost.validation import *

if __name__ == '__main__':
    train, train_target, test, test_target = load_20p_missing_spambase()
    cf = BernoulliNaiveBayes()
    cf.fit(train, train_target)
    predicts = cf.predict(test)
    predict_class = cf.predict_class(predicts)
    cm = confusion_matrix(test_target, predict_class)
    print "confusion matrix: TN: %s, FP: %s, FN: %s, TP: %s" % (cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1])
    er, acc, fpr, tpr = confusion_matrix_analysis(cm)
    print 'Error rate: %f, accuracy: %f, FPR: %f, TPR: %f' % (er, acc, fpr, tpr)