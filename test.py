import re
import numpy as np
from sklearn.svm import SVC

# a = np.matrix('1 2 3 4')
# print a

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf = SVC()
clf.fit(X, y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
a = clf.predict([[-0.8, -1],[1,2],[9,8]])
print a
b = np.array(a)
print b[1]
print type(b[1])
c = str(b[1])
print c,type(c)

