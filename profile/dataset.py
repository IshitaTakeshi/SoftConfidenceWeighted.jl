import sys
import argparse

from sklearn.datasets import make_classification, dump_svmlight_file


def generate(filename, n_samples, n_features):
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features)
    y[y==0] = -1  # replace 0s with 1s since scw doesn't allow 0s as labels

    dump_svmlight_file(X, y, "data.txt", zero_based=False)


if __name__ == '__main__':
    n_samples = 20000
    n_features = 2000

    parser = argparse.ArgumentParser()
    parser.add_argument('--nsamples', help='the number of samples',
                        type=int, default=20000, dest='n_samples')
    parser.add_argument('--nfeatures', help='data dimension',
                        type=int, default=2000, dest='n_features')
    parser.add_argument('--filename', help='output filename',
                        type=str, default="data.txt", dest='filename')
    args = parser.parse_args()

    generate(args.filename, args.n_samples, args.n_features)
