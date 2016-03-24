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

model = init(C = 1, ETA = 1, type_ = SCW1)
model = fit!(model, "data/svmlight/digits.train.txt", ndim)
y_pred = predict(model, "data/svmlight/digits.test.txt")
assert(all(y_pred .== y_test))
