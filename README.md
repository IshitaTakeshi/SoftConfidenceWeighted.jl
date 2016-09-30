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

# C and ETA are hyperparameters.
# X is a data matrix which each column represents a data vector.
# y is corresponding labels.

model = init(C = 1, ETA = 1, type_ = SCW1)
model = fit!(model, X_train, y_train)
y_pred = predict(model, X_test)
```
