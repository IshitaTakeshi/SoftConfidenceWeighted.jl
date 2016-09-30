import Base: size, convert

import SoftConfidenceWeighted: init, fit!, predict, SCW1, SCW2


function split_dataset(X, y, training_ratio=0.8)
    assert(0.0 <= training_ratio <= 1.0)

    split_point = convert(Int64, size(X, 2)*training_ratio)
    training = X[:, 1:split_point-1], y[1:split_point-1]
    test = X[:, split_point:end], y[split_point:end]
    return training, test
end


function calc_accuracy(y_pred, y_true)
    n_correct = 0
    for (a, b) in zip(y_pred, y_true)
        if a == b
            n_correct += 1
        end
    end

    return n_correct / length(y_pred)
end


function test_batch(X, y, type_; training_ratio = 0.8, C = 1.0, ETA = 1.0)
    model = init(C = C, ETA = ETA, type_ = type_)

    training, test = split_dataset(X, y, training_ratio)

    X, labels = training
    model = fit!(model, X, labels)

    X, y_true = test
    y_pred = predict(model, X)

    accuracy = calc_accuracy(y_pred, y_true)
    assert(accuracy == 1.0)

    println("BATCH")
    println("\ttype: $type_")
    println("\taccuracy: $accuracy")
    println("")
end


function test_online(X, y, type_; training_ratio=0.8, C=1.0, ETA=1.0)
    model = init(C = C, ETA = ETA, type_ = type_)

    training, test = split_dataset(X, y, training_ratio)

    X, labels = training
    for i in 1:size(X, 2)
        model = fit!(model, view(X, :, i), labels[i])
    end

    X, y_true = test

    y_pred = Int64[]
    for i in 1:size(X, 2)
        r = predict(model, view(X, :, i))
        append!(y_pred, r)
    end

    accuracy = calc_accuracy(y_pred, y_true)
    assert(accuracy == 1.0)

    println("ONLINE")
    println("\ttype: $type_")
    println("\taccuracy: $accuracy")
    println("")
end


X = readdlm("data/julia_array/digitsX.txt")
y = readdlm("data/julia_array/digitsy.txt")

println("TEST DIGITS\n")

# Dense matrix
test_batch(X, y, SCW1, training_ratio=0.8)
test_batch(X, y, SCW2, training_ratio=0.8)

test_online(X, y, SCW1, training_ratio=0.8)
test_online(X, y, SCW2, training_ratio=0.8)

X = sparse(X)

# Sparse matrix
test_batch(X, y, SCW1, training_ratio=0.8)
test_batch(X, y, SCW2, training_ratio=0.8)

test_online(X, y, SCW1, training_ratio=0.8)
test_online(X, y, SCW2, training_ratio=0.8)
