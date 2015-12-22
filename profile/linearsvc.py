import timeit

from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score


setup = """
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file("data.txt")
svc = LinearSVC()
"""

time = timeit.timeit('svc.fit(X, y)', setup=setup, number=1)
print("Time: {}".format(time))

X, y = load_svmlight_file("data.txt")
svc = LinearSVC()
svc.fit(X, y)
results = svc.predict(X)
accuracy = accuracy_score(y, results)
print("Accuracy: {}".format(accuracy))
