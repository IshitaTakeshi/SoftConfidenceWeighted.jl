# SoftConfidenceWeighted.jl
This is an online supervised learning algorithm which utilizes the four salient properties:

* Large margin training
* Confidence weighting
* Capability to handle non-separable data
* Adaptive margin

The paper is [here](http://arxiv.org/pdf/1206.4612v1.pdf).

## Usage
SCW has 2 formulations of its algorithm which are SCW-I and SCW-II.  
You can choose which to use by the parameter of `init`.  

### Note
1. This package performs only binary classification, not multiclass classification.
2. Training labels must be 1 or -1. No other labels allowed.


### Training from matrix
Feature vectors are given as the columns of the matrix X.

```jl
using SoftConfidenceWeighted
using SVMLightLoader

ndim = 64
X_train, y_train = load_svmlight_file("data/svmlight/digits.train.txt", ndim)
X_test, y_test = load_svmlight_file("data/svmlight/digits.test.txt", ndim)

# C and ETA are hyperparameters.

model = init(C = 1, ETA = 1, type_ = SCW1)
model = fit!(model, X_train, y_train)
y_pred = predict(model, X_test)
assert(all(y_pred .== y_test))
```

### Training from file
The input files must be in the svmlight format.

```jl
model = init(C = 1, ETA = 1, type_ = SCW1)
model = fit!(model, "data/svmlight/digits.train.txt", ndim)
y_pred = predict(model, "data/svmlight/digits.test.txt")
assert(all(y_pred .== y_test))
```

See test/example.jl for more details.
