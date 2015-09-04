import SVMLightLoader: load_svmlight_file
import SoftConfidenceWeighted: init, fit, predict


# The code is from below:
# http://thirld.com/blog/2015/05/30/julia-profiling-cheat-sheet/
function benchmark(func, args...; filename="trace.def")
    # Any setup code goes here.

    # Run once, to force compilation.
    println("======================= First run:")
    @time func(args...)

    # Run a second time, with profiling.
    println("\n\n======================= Second run:")

    Profile.init()
    Profile.clear()
    Profile.clear_malloc_data()

    @profile @time func(args...)

    # Write profile results to profile.bin.
    f = open(filename, "w")
    Profile.print(f)
    close(f)
end


url = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a"
filename = basename(url)

if !isfile(filename)
    import HTTPClient: get
    contents = get(url)
    f = open(filename, "w")
    write(f, contents.body.data)
    close(f)
end

X, y = load_svmlight_file(filename)
model = init(1.0, 1.0)
benchmark(fit, model, X, y, filename="trace.fit.txt")
benchmark(predict, model, X, filename="trace.predict.txt")
