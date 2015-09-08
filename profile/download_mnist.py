from os.path import exists

import numpy as np
from sklearn.datasets import fetch_mldata, dump_svmlight_file


def download_mnist(training_ratio=0.8, data_home="."):
    mnist = fetch_mldata('MNIST original', data_home=data_home)

    X, y = shuffle(mnist.data, mnist.target)

    splitter = int(len(X)*training_ratio)
    training = X[:splitter], y[:splitter]
    test = X[splitter:], y[splitter:]
    return training, test


def shuffle(X, y):
    assert(len(X) == len(y))
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]


def extract01(X, y):
    positive = X[y == 1]
    negative = X[y == 0]
    samples = np.concatenate((positive, negative))

    positive = np.ones(np.count_nonzero(y == 1))
    negative = -np.ones(np.count_nonzero(y == 0))
    labels = np.concatenate((positive, negative))

    return shuffle(samples, labels)


training_file = "mnist"
test_file = "mnist.t"

# for binary classification
training_file_binary = "mnist.binary"
test_file_binary = "mnist.binary.t"

if exists(training_file_binary) and exists(test_file_binary):
    exit(0)

training, test = download_mnist()

X, y = training
dump_svmlight_file(X, y, training_file)

X, y = test
dump_svmlight_file(X, y, test_file)

X, y = training
X, y = extract01(X, y)
dump_svmlight_file(X, y, training_file_binary)

X, y = test
X, y = extract01(X, y)
dump_svmlight_file(X, y, test_file_binary)
