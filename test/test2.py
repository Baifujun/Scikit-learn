#_*_coding:UTF-8_*_
from sklearn import  svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)
print "模型持久性：第一种方法"
print "实际标签："
print y[0:1]
print "预测标签："
# print clf.predict(X[0:1])
import pickle
s = pickle.dumps(clf)#存于内存中，随后可以调用
clf2 = pickle.loads(s)
print clf2.predict(X[0:1])

print "模型持久性：第二种方法"
from sklearn.externals import joblib
joblib.dump(clf, 'test.pk1')#存于磁盘中，通常用在大数据集中
clf3 = joblib.load('test.pk1')
print "实际标签："
print y[0:1]
print "预测标签："
print clf3.predict(X[0:1])