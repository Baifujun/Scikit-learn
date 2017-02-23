#_*_coding:UTF-8_*_
from sklearn import datasets
iris = datasets.load_iris()
data = iris.data
print data.shape

digits = datasets.load_digits()
print digits.images.shape
import pylab as pl
# pl.imshow(digits.images[-1], cmap=pl.cm.gray_r)
data1 = digits.images.reshape((digits.images.shape[0], -1))