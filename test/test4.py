#_*_coding:UTF-8_*_
import numpy as np
from sklearn import datasets
from sklearn.neighbors.classification import KNeighborsClassifier
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print np.unique(iris_y)

np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[indices[:-10]]]
iris_y_train = iris_y[indices[indices[:-10]]]
iris_X_test = iris_X[indices[indices[:-10]]]
iris_y_test = iris_y[indices[indices[:-10]]]
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
print "预测："
print knn.predict(iris_X_test)
print "实际："
print iris_y_test
i = knn.predict(iris_X_test)== iris_y_test
k = 0
for j in i:
    if j==False:
        k += 1
print len(i)
print "预测错误数："
print k

diabetes = datasets.load_diabetes()#糖尿病数据集
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)
print np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)
print regr.score(diabetes_X_test, diabetes_y_test)

X = np.c_[ .5, 1].T
y = [.5, 1]
test = np.c_[ 0, 2].T
regr = linear_model.LinearRegression()
import pylab as pl
pl.figure()
np.random.seed(0)
for _ in range(6):
    this_X = .1*np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    pl.plot(test, regr.predict(test))
    pl.scatter(this_X, y, s=3)
    
regr = linear_model.Ridge(alpha=.1)
pl.figure()
np.random.seed(0)
for _ in range(6):
    this_X = .1*np.random.normal(size=(2, 1)) + X
    regr.fit(this_X, y)
    pl.plot(test, regr.predict(test))
    pl.scatter(this_X, y, s=3)
    
# alphas = np.logspace(-4, -1, 6)
# from __future__ import print_function
# print([regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train,).score(diabetes_X_test, diabetes_y_test) for alpha in alphas])      

# regr = linear_model.Lasso()
# scores = [regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]
# best_alpha = alphas[scores.index(max(scores))]
# regr.alpha = best_alpha
# regr.fit(diabetes_X_train, diabetes_y_train)
# print(regr.coef_)

logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(iris_X_train, iris_y_train)
