# -*- coding: UTF-8 -*-
from sklearn import datasets
from sklearn import svm
# iris = datasets.load_iris()
digits = datasets.load_digits()
# print digits.data
# print digits.target
# print digits.images[0]
# print "\n"
# print iris.data
# print iris.target
clf = svm.SVC(gamma=0.001,C=100.)
clf.fit(digits.data[:-1],digits.target[:-1])
print "实际标签:"
print digits.target[-1:]
print "预测标签:"
print clf.predict(digits.data[-1:])