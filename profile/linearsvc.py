import cProfile
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score


X, y = load_svmlight_file("data.txt")

svc = LinearSVC()

cProfile.runctx('svc.fit(X, y)', {'svc': svc, 'X': X, 'y': y}, {})

svc.fit(X, y)
results = svc.predict(X)
accuracy = accuracy_score(y, results)
print("Accuracy: {}".format(accuracy))
