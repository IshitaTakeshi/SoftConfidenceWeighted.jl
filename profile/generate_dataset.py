from sklearn.datasets import make_classification, dump_svmlight_file

X, y = make_classification(n_samples=20000, n_features=2000)
y[y==0] = -1  # replace 0s with 1s since scw doesn't allow 0s as labels

dump_svmlight_file(X, y, "data.txt", zero_based=False)
