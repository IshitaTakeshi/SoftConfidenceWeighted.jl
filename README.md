# SoftConfidenceWeighted.jl
This is an online supervised learning algorithm which utilizes the four salient properties:

* Large margin training
* Confidence weighting
* Capability to handle non-separable data
* Adaptive margin

The paper is [here](http://icml.cc/2012/papers/86.pdf).

## Usage
SCW has 2 formulations of its algorithm which are SCW-I and SCW-II.  
You can choose which to use by the parameter of `init`.  

### Training from matrix
Feature vectors are given as the columns of the matrix X.

```
#C and ETA are hyperparameters.
model = init(C, ETA, type_)
model = fit(model, X, y)
results = predict(model, X)
```

### Training from file
The input files must be in the svmlight format.

```
model = init(C, ETA, SCW1)
model = fit(model, training_file, ndim)
results = predict(model, test_file)
```
