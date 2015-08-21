import Base: size, convert

import SoftConfidenceWeighted: init, fit, predict, SCW1, SCW2


function split_dataset(X, y, training_ratio=0.7)
    assert(0.0 <= training_ratio <= 1.0)

    split_point = convert(Int64, size(X, 1)*training_ratio)
    training = (X[1:split_point-1, :], y[1:split_point-1])
    test = (X[split_point:end, :], y[split_point:end])
    return (training, test)
end


function calc_accuracy(results, answers)
    n_correct_answers = 0
    for (result, answer) in zip(results, answers)
        if result == answer
            n_correct_answers += 1
        end
    end

    return n_correct_answers / length(results)
end


function test_batch(X, y, type_; training_ratio=0.7, C=1.0, ETA=1.0)
    model = init(C, ETA, type_)

    (training, test) = split_dataset(X, y, training_ratio)

    (samples, labels) = training
    model = fit(model, samples, labels)

    (samples, answers) = test
    results = predict(model, samples)

    accuracy = calc_accuracy(results, answers)
    assert(accuracy == 1.0)

    println("BATCH")
    println("\ttype: $type_")
    println("\taccuracy: $accuracy")
    println("")
end


function test_online(X, y, type_; training_ratio=0.7, C=1.0, ETA=1.0)
    model = init(C, ETA, type_)

    (training, test) = split_dataset(X, y, training_ratio)

    (samples, labels) = training
    for i in 1:size(samples, 1)
        s = reshape(samples[i, :], 1, size(samples, 2))
        model = fit(model, s, [labels[i]])
    end

    (samples, answers) = test

    results = []
    for i in 1:size(samples, 1)
        s = reshape(samples[i, :], 1, size(samples, 2))
        r = predict(model, s)
        append!(results, r)
    end

    accuracy = calc_accuracy(results, answers)
    assert(accuracy == 1.0)

    println("ONLINE")
    println("\ttype: $type_")
    println("\taccuracy: $accuracy")
    println("")
end


X = readdlm("../data/digitsX.txt")
y = readdlm("../data/digitsy.txt")

println("TEST DIGITS\n")

test_batch(X, y, SCW1, training_ratio=0.8)
test_batch(X, y, SCW2, training_ratio=0.8)

test_online(X, y, SCW1, training_ratio=0.8)
test_online(X, y, SCW2, training_ratio=0.8)
