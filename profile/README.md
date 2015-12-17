## Profiling and Comparison

### Generate dataset
First, generate dataset by running `generate_dataset.py`

```
$python3 generate_dataset.py
```

The number of samples and dimensions can be changed by editing the line:

```
X, y = make_classification(n_samples=20000, n_features=2000)
```

### Profiling
Run

```
$julia profile.jl
```

to show the execution time and the accuracy.
More details will be written to `trace.fit.txt`

### Comparison
To compare the performance of sklearn.svm.LinearSVC and SoftConfidenceWeighted.jl, run

```
$python3 dataset.py  #generate dataset
$python3 linearsvc.py
$julia profile.jl
```

Only the fitting part of each algorithm is examined since the process of computation in the prediction part of these algorithms are same: taking dot product.
