import SVMLightLoader: load_svmlight_file
import SoftConfidenceWeighted: init, fit!, predict


# The code is from
# http://thirld.com/blog/2015/05/30/julia-profiling-cheat-sheet/
function benchmark(func, args...; filename="trace.def")
    # Any setup code goes here.

    # Run once, to force compilation.
    println("======================= First run:")
    @time func(args...)

    # Run a second time, with profiling.
    println("======================= Second run:")

    Profile.init()
    Profile.clear()
    Profile.clear_malloc_data()

    @profile @time func(args...)

    # Write profile results to profile.bin.
    f = open(filename, "w")
    Profile.print(f)
    close(f)
end


function calc_accuracy(results, answers)
    n_correct_answers = 0
    for (result, answer) in zip(results, answers)
        if result == answer
            n_correct_answers += 1
        end
    end
    accuracy = n_correct_answers / length(y)
end


X, y = load_svmlight_file("data.txt")

model = init(10.0, 10.0)

benchmark(fit!, model, X, y, filename="trace.fit.txt")

model = init(10.0, 10.0)
model = fit!(model, X, y)
results = predict(model, X)

accuracy = calc_accuracy(results, y)
println("Accuracy: $accuracy")
